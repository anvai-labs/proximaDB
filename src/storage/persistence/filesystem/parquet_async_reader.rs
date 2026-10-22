// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! Parquet [`AsyncFileReader`] over the [`FileSystem`] port (ADR-097 P3.1, PR 3).
//!
//! The Layer A table provider streams spill segments row group by row group.
//! Every segment is reached through `Arc<dyn FileSystem>` — the same handle the
//! spill store writes with — so the reader must be too: an `object_store`
//! reader would need a second, separately-configured handle onto the same
//! bucket, and would bypass the decorators (`CountingFileSystem`'s physical-GET
//! meter, fault injection, encryption) that the port carries.
//!
//! # What it costs
//!
//! * **Metadata**: one `metadata()` (HEAD) for the object size unless the caller
//!   already knows it ([`FileSystemParquetReader::with_file_size`]), then ONE
//!   suffix `read_range` of [`DEFAULT_FOOTER_PREFETCH`] bytes that, for a
//!   spill-sized footer, holds the footer, the page index and the Bloom-filter
//!   offsets together. A footer larger than the prefetch costs one more read.
//! * **Data**: the column-chunk ranges the decoder asks for, issued through
//!   [`FileSystem::read_ranges`], so the backend's range-coalescing policy and
//!   bounded parallelism apply exactly as on every other ranged read.
//!
//! # Fail closed on a short read
//!
//! `read_range` may legally clamp at EOF and return fewer bytes than asked. A
//! Parquet decoder handed a short column chunk reports a decode error at best
//! and misreads pages at worst, so a short buffer is turned into an error here,
//! naming the object, rather than passed on (mandate #1).

use std::ops::Range;
use std::sync::Arc;

use bytes::Bytes;
use futures::FutureExt;
use futures::future::BoxFuture;
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::errors::{ParquetError, Result as ParquetResult};
use parquet::file::metadata::{PageIndexPolicy, ParquetMetaData, ParquetMetaDataReader};

use super::FileSystem;

/// Bytes read from the end of the object when loading metadata. Covers the
/// footer and page index of a warehouse-layout spill segment in a single GET.
pub const DEFAULT_FOOTER_PREFETCH: usize = 64 * 1024;

/// A Parquet [`AsyncFileReader`] whose every byte comes from a [`FileSystem`].
#[derive(Debug, Clone)]
pub struct FileSystemParquetReader {
    filesystem: Arc<dyn FileSystem>,
    path: String,
    file_size: Option<u64>,
    footer_prefetch: usize,
    page_index: PageIndexPolicy,
    metadata: Option<Arc<ParquetMetaData>>,
}

impl FileSystemParquetReader {
    pub fn new(filesystem: Arc<dyn FileSystem>, path: impl Into<String>) -> Self {
        Self {
            filesystem,
            path: path.into(),
            file_size: None,
            footer_prefetch: DEFAULT_FOOTER_PREFETCH,
            page_index: PageIndexPolicy::Skip,
            metadata: None,
        }
    }

    /// The object's size, when the caller already has it — saves the HEAD.
    pub fn with_file_size(mut self, size: u64) -> Self {
        self.file_size = Some(size);
        self
    }

    /// Suffix bytes fetched in the first metadata read (`0` is treated as 8, the
    /// smallest read that still holds the footer length and magic).
    pub fn with_footer_prefetch(mut self, bytes: usize) -> Self {
        self.footer_prefetch = bytes.max(8);
        self
    }

    /// Load the column and offset indexes with the footer (`Optional`: used when
    /// present). Page pruning needs them; a plain scan does not.
    pub fn with_page_index(mut self, policy: PageIndexPolicy) -> Self {
        self.page_index = policy;
        self
    }

    /// Serve metadata already decoded for this object (a pruning pass that read
    /// the footer hands it to the scan) instead of reading it again.
    pub fn with_metadata(mut self, metadata: Arc<ParquetMetaData>) -> Self {
        self.metadata = Some(metadata);
        self
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    async fn file_size(&mut self) -> ParquetResult<u64> {
        if let Some(size) = self.file_size {
            return Ok(size);
        }
        let size = self
            .filesystem
            .metadata(&self.path)
            .await
            .map_err(|e| fs_error(&self.path, "metadata", e))?
            .size;
        self.file_size = Some(size);
        Ok(size)
    }

    /// Decode this object's footer (and page index, per the policy), caching it
    /// on the reader. The cached value is what [`AsyncFileReader::get_metadata`]
    /// serves, so a builder opened after this call issues no further metadata I/O.
    pub async fn load_metadata(
        &mut self,
        options: Option<&ArrowReaderOptions>,
    ) -> ParquetResult<Arc<ParquetMetaData>> {
        if let Some(metadata) = &self.metadata {
            return Ok(metadata.clone());
        }
        let size = self.file_size().await?;
        let (column_index, offset_index) = match options {
            Some(o)
                if o.column_index_policy() != PageIndexPolicy::Skip
                    || o.offset_index_policy() != PageIndexPolicy::Skip =>
            {
                (o.column_index_policy(), o.offset_index_policy())
            }
            _ => (self.page_index, self.page_index),
        };
        let metadata = ParquetMetaDataReader::new()
            .with_metadata_options(options.map(|o| o.metadata_options().clone()))
            .with_column_index_policy(column_index)
            .with_offset_index_policy(offset_index)
            .with_prefetch_hint(Some(self.footer_prefetch))
            .load_and_finish(&mut *self, size)
            .await?;
        let metadata = Arc::new(metadata);
        self.metadata = Some(metadata.clone());
        Ok(metadata)
    }
}

fn fs_error(path: &str, op: &str, error: impl std::fmt::Display) -> ParquetError {
    ParquetError::General(format!("filesystem {op} of '{path}' failed: {error}"))
}

/// The bytes for `range`, or an error if the backend returned a short buffer.
fn exact(path: &str, range: &Range<u64>, bytes: Vec<u8>) -> ParquetResult<Bytes> {
    let wanted = range.end.saturating_sub(range.start);
    if bytes.len() as u64 != wanted {
        return Err(ParquetError::General(format!(
            "short read of '{path}' at {}..{}: got {} of {wanted} bytes",
            range.start,
            range.end,
            bytes.len()
        )));
    }
    Ok(Bytes::from(bytes))
}

impl AsyncFileReader for FileSystemParquetReader {
    fn get_bytes(&mut self, range: Range<u64>) -> BoxFuture<'_, ParquetResult<Bytes>> {
        async move {
            let bytes = self
                .filesystem
                .read_range(&self.path, range.start, range.end - range.start)
                .await
                .map_err(|e| fs_error(&self.path, "read_range", e))?;
            exact(&self.path, &range, bytes)
        }
        .boxed()
    }

    fn get_byte_ranges(
        &mut self,
        ranges: Vec<Range<u64>>,
    ) -> BoxFuture<'_, ParquetResult<Vec<Bytes>>> {
        async move {
            let buffers = self
                .filesystem
                .read_ranges(&self.path, ranges.clone())
                .await
                .map_err(|e| fs_error(&self.path, "read_ranges", e))?;
            if buffers.len() != ranges.len() {
                return Err(ParquetError::General(format!(
                    "read_ranges of '{}' returned {} buffers for {} ranges",
                    self.path,
                    buffers.len(),
                    ranges.len()
                )));
            }
            ranges
                .iter()
                .zip(buffers)
                .map(|(range, bytes)| exact(&self.path, range, bytes))
                .collect()
        }
        .boxed()
    }

    fn get_metadata<'a>(
        &'a mut self,
        options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, ParquetResult<Arc<ParquetMetaData>>> {
        self.load_metadata(options).boxed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use futures::TryStreamExt;
    use parquet::arrow::ParquetRecordBatchStreamBuilder;
    use parquet::file::properties::WriterProperties;
    use proximadb_storage_filesystem_types::counting::{CountingFileSystem, GetCounters};
    use std::sync::atomic::Ordering;

    const ROWS: i64 = 10_000;
    const ROW_GROUP: usize = 1_000;

    fn sample_parquet() -> (Vec<u8>, RecordBatch) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from_iter_values(0..ROWS)),
                Arc::new(StringArray::from_iter_values(
                    (0..ROWS).map(|i| format!("row-{i}")),
                )),
            ],
        )
        .expect("batch");
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(ROW_GROUP))
            .build();
        let bytes = proximadb_storage_common::proxima_parquet::record_batches_to_parquet_bytes(
            std::slice::from_ref(&batch),
            schema,
            Some(props),
        )
        .expect("encode");
        (bytes, batch)
    }

    /// A real local filesystem behind the physical-GET meter, holding one
    /// multi-row-group Parquet object.
    async fn fixture() -> (
        Arc<dyn FileSystem>,
        Arc<GetCounters>,
        String,
        RecordBatch,
        u64,
    ) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("seg.parquet").to_string_lossy().to_string();
        std::mem::forget(dir);
        let local: Arc<dyn FileSystem> = Arc::new(
            LocalFileSystem::new(LocalConfig::default())
                .await
                .expect("local filesystem"),
        );
        let (bytes, batch) = sample_parquet();
        local.write(&path, &bytes, None).await.expect("write");
        let counters = Arc::new(GetCounters::default());
        let fs: Arc<dyn FileSystem> = Arc::new(CountingFileSystem::new(local, counters.clone()));
        (fs, counters, path, batch, bytes.len() as u64)
    }

    #[tokio::test]
    async fn streams_every_row_group_back_identically() {
        let (fs, _, path, expected, _) = fixture().await;
        let builder = ParquetRecordBatchStreamBuilder::new(FileSystemParquetReader::new(fs, path))
            .await
            .expect("open");
        assert_eq!(
            builder.metadata().num_row_groups(),
            (ROWS as usize).div_ceil(ROW_GROUP)
        );
        let batches: Vec<RecordBatch> = builder
            .build()
            .expect("stream")
            .try_collect()
            .await
            .expect("decode");
        let got = arrow::compute::concat_batches(&expected.schema(), &batches).expect("concat");
        assert_eq!(got, expected);
    }

    /// With the size known, metadata is ONE suffix read — no HEAD, no second
    /// footer fetch — and a reader seeded with that metadata reads none at all.
    #[tokio::test]
    async fn metadata_is_one_suffix_read_and_reusable() {
        let (fs, counters, path, _, size) = fixture().await;
        let mut reader =
            FileSystemParquetReader::new(fs.clone(), path.clone()).with_file_size(size);
        let metadata = reader.load_metadata(None).await.expect("metadata");
        assert_eq!(
            counters.total_gets(),
            1,
            "footer must be one prefetched GET"
        );

        counters.reset();
        let seeded = FileSystemParquetReader::new(fs, path)
            .with_file_size(size)
            .with_metadata(metadata);
        let builder = ParquetRecordBatchStreamBuilder::new(seeded)
            .await
            .expect("open with seeded metadata");
        assert_eq!(
            counters.total_gets(),
            0,
            "seeded metadata must not be re-read"
        );
        let rows: usize = builder
            .with_row_groups(vec![3])
            .build()
            .expect("stream")
            .try_collect::<Vec<_>>()
            .await
            .expect("decode")
            .iter()
            .map(RecordBatch::num_rows)
            .sum();
        assert_eq!(rows, ROW_GROUP, "only the selected row group is decoded");
        assert!(
            counters.bytes_read.load(Ordering::Relaxed) < size / 4,
            "one row group must not read the whole object"
        );
    }

    /// Without a size the reader asks the filesystem once, then reads the footer.
    #[tokio::test]
    async fn an_unknown_size_costs_one_metadata_call() {
        let (fs, counters, path, _, _) = fixture().await;
        let mut reader = FileSystemParquetReader::new(fs, path);
        reader.load_metadata(None).await.expect("metadata");
        assert_eq!(
            counters.total_gets(),
            1,
            "the HEAD is not a GET; the footer is"
        );
    }

    /// A footer larger than the prefetch still loads (one extra read).
    #[tokio::test]
    async fn a_footer_larger_than_the_prefetch_still_loads() {
        let (fs, _, path, _, size) = fixture().await;
        let mut reader = FileSystemParquetReader::new(fs, path)
            .with_file_size(size)
            .with_footer_prefetch(0);
        let metadata = reader.load_metadata(None).await.expect("metadata");
        assert_eq!(metadata.file_metadata().num_rows(), ROWS);
    }

    /// A size claiming more bytes than the object holds makes the backend clamp
    /// at EOF; the short buffer must surface as an error, never as decoded data.
    #[tokio::test]
    async fn a_short_read_fails_closed() {
        let (fs, _, path, _, size) = fixture().await;
        let mut reader = FileSystemParquetReader::new(fs, path.clone());
        let err = reader
            .get_bytes(size - 4..size + 64)
            .await
            .expect_err("a clamped read must not be served");
        assert!(err.to_string().contains("short read"), "{err}");
        assert!(err.to_string().contains(&path), "{err}");
    }

    #[tokio::test]
    async fn a_missing_object_names_the_path() {
        let (fs, _, path, _, _) = fixture().await;
        let missing = format!("{path}.absent");
        let err = FileSystemParquetReader::new(fs, missing.clone())
            .load_metadata(None)
            .await
            .expect_err("missing object");
        assert!(err.to_string().contains(&missing), "{err}");
    }
}
