//! # multipart — bounded-memory, multi-file Parquet publication (P2.4)
//!
//! [`MultipartParquetWriter`] streams Arrow batches into a sequence of Parquet data
//! files (`{dir}/{stem}-{nnnnn}.parquet`) written with the warehouse
//! [`ParquetWritePolicy`]. Each file is an [`AsyncArrowWriter`] over an
//! `object_store` [`BufWriter`], so the bytes leave the process as a multipart upload
//! while the next row group encodes: resident memory is one row group per open file,
//! not the table.
//!
//! A part closes when it reaches [`ParquetWritePolicy::target_file_rows`] (exact: a
//! batch crossing the cap is split) or [`ParquetWritePolicy::target_file_bytes`]
//! (checked between batches). Nothing here publishes: [`finish`] returns the written
//! parts for the caller to commit as one snapshot, and [`abort`] removes them.
//!
//! [`finish`]: MultipartParquetWriter::finish
//! [`abort`]: MultipartParquetWriter::abort

use std::pin::Pin;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{Schema, SchemaRef};
use bytes::Bytes;
use futures::FutureExt;
use futures::future::BoxFuture;
use object_store::buffered::BufWriter;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt};
use parquet::arrow::AsyncArrowWriter;
use parquet::arrow::async_writer::AsyncFileWriter;
use proximadb_kernel::error::StorageError;

use crate::object_store_bridge::ObjectStoreBridge;
use crate::parquet_write_policy::{ParquetLayoutHints, ParquetWritePolicy};

/// Stamp Iceberg field ids (`PARQUET:field_id`) on the top-level fields: field id
/// `i + 1` for column `i`, the assignment the warehouse publisher's Iceberg schema
/// uses. Iceberg readers resolve Parquet columns by field id.
pub fn iceberg_field_id_schema(schema: &Schema) -> SchemaRef {
    const FIELD_ID: &str = "PARQUET:field_id";
    let fields: Vec<arrow_schema::Field> = schema
        .fields()
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let mut md = f.metadata().clone();
            md.insert(FIELD_ID.to_string(), (i + 1).to_string());
            f.as_ref().clone().with_metadata(md)
        })
        .collect();
    Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()))
}

/// One data file a [`MultipartParquetWriter`] completed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrittenPart {
    /// Bridge-relative key (what `publish_iceberg_files` takes).
    pub path: Path,
    pub rows: u64,
    pub bytes: u64,
}

struct OpenPart {
    key: Path,
    writer: AsyncArrowWriter<PartUpload>,
    rows: usize,
}

/// A part's upload. Once `complete` reaches `shutdown`, the `BufWriter` has either
/// finished or failed with its upload already gone (a failed multipart completion
/// aborts itself); `BufWriter::abort` would then panic, so [`abort`](Self::abort)
/// becomes a no-op.
struct PartUpload {
    inner: BufWriter,
    shutting_down: bool,
}

impl PartUpload {
    fn new(inner: BufWriter) -> Self {
        Self {
            inner,
            shutting_down: false,
        }
    }

    async fn abort(&mut self) -> object_store::Result<()> {
        if self.shutting_down {
            return Ok(());
        }
        self.inner.abort().await
    }
}

impl AsyncFileWriter for PartUpload {
    fn write(&mut self, bs: Bytes) -> BoxFuture<'_, parquet::errors::Result<()>> {
        AsyncFileWriter::write(&mut self.inner, bs)
    }

    fn complete(&mut self) -> BoxFuture<'_, parquet::errors::Result<()>> {
        async move {
            std::future::poll_fn(|cx| {
                tokio::io::AsyncWrite::poll_flush(Pin::new(&mut self.inner), cx)
            })
            .await?;
            self.shutting_down = true;
            std::future::poll_fn(|cx| {
                tokio::io::AsyncWrite::poll_shutdown(Pin::new(&mut self.inner), cx)
            })
            .await?;
            Ok(())
        }
        .boxed()
    }
}

/// Streams batches into size-capped Parquet parts under one directory. See the
/// module docs. Call [`finish`](Self::finish) or [`abort`](Self::abort); dropping
/// the writer with a part open leaves that part's multipart upload unfinished (no
/// object becomes visible, but the store may keep the uploaded chunks until its
/// lifecycle rule reclaims them).
pub struct MultipartParquetWriter {
    store: Arc<dyn ObjectStore>,
    /// `dir` as a bridge-relative key (what [`WrittenPart::path`] reports) …
    dir: String,
    /// … and as the concrete key in `store`.
    store_dir: String,
    stem: String,
    schema: SchemaRef,
    policy: ParquetWritePolicy,
    hints: ParquetLayoutHints,
    open: Option<OpenPart>,
    done: Vec<WrittenPart>,
    /// Rows waiting for a part to open: a part's Bloom filters are sized from
    /// its first row group, so up to one row group is buffered before opening.
    pending: Vec<RecordBatch>,
    pending_rows: usize,
}

fn write_err(context: &str, e: impl std::fmt::Display) -> StorageError {
    StorageError::Serialization(format!("iceberg-engine: multipart parquet {context}: {e}"))
}

impl MultipartParquetWriter {
    /// A writer of `{dir}/{stem}-{nnnnn}.parquet` parts holding `schema` rows. The
    /// columns are stamped with Iceberg field ids (`i + 1`), as every warehouse file
    /// is. `stem` should be unique per publication generation so a retry or a
    /// concurrent publisher never writes over a published part.
    pub fn new(
        bridge: &dyn ObjectStoreBridge,
        dir: impl Into<String>,
        stem: impl Into<String>,
        schema: SchemaRef,
        policy: ParquetWritePolicy,
        hints: ParquetLayoutHints,
    ) -> Self {
        let mut dir = dir.into();
        while dir.ends_with('/') {
            dir.pop();
        }
        let store_dir = bridge
            .object_store_path(&Path::from(dir.as_str()))
            .to_string();
        Self {
            store: bridge.inner_store(),
            dir,
            store_dir,
            stem: stem.into(),
            schema: iceberg_field_id_schema(&schema),
            policy,
            hints,
            open: None,
            done: Vec::new(),
            pending: Vec::new(),
            pending_rows: 0,
        }
    }

    /// The parts completed so far (the open part, if any, is not included).
    pub fn completed_parts(&self) -> &[WrittenPart] {
        &self.done
    }

    /// Append `batch`, rolling to a new part whenever the current one reaches its
    /// target. On `Err` the writer is unusable: call [`abort`](Self::abort).
    pub async fn write(&mut self, batch: &RecordBatch) -> Result<(), StorageError> {
        let batch = batch
            .clone()
            .with_schema(self.schema.clone())
            .map_err(|e| write_err("batch schema", e))?;
        if batch.num_rows() == 0 {
            return Ok(());
        }
        if self.open.is_some() {
            return self.write_rows(batch, true).await;
        }
        self.pending_rows += batch.num_rows();
        self.pending.push(batch);
        if self.pending_rows >= self.sample_rows() {
            self.flush_pending(true).await?;
        }
        Ok(())
    }

    /// Complete the open part and return every part written, in order. A writer that
    /// saw no rows writes one empty part, so a publication always has a file that
    /// carries the schema.
    pub async fn finish(mut self) -> Result<Vec<WrittenPart>, StorageError> {
        let closed = async {
            self.flush_pending(false).await?;
            if self.open.is_none() && self.done.is_empty() {
                let empty = RecordBatch::new_empty(self.schema.clone());
                self.open = Some(self.open_part(&empty)?);
            }
            self.close_part().await
        }
        .await;
        if let Err(e) = closed {
            // Do not leave the finished siblings of a part that failed to close.
            let _ = self.abort().await;
            return Err(e);
        }
        Ok(std::mem::take(&mut self.done))
    }

    /// Rows a part is sized from: its first row group (or the whole part, if
    /// smaller).
    fn sample_rows(&self) -> usize {
        self.policy
            .row_group_rows
            .min(self.policy.target_file_rows)
            .max(1)
    }

    async fn flush_pending(&mut self, buffer_tail: bool) -> Result<(), StorageError> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let batch = arrow_select::concat::concat_batches(&self.schema, &self.pending)
            .map_err(|e| write_err("buffer", e))?;
        self.pending.clear();
        self.pending_rows = 0;
        self.write_rows(batch, buffer_tail).await
    }

    /// Write `batch` into parts, opening and closing them at the targets. With
    /// `buffer_tail`, rows left over after a part closes are buffered (to size the
    /// next part) unless they already fill a sample.
    async fn write_rows(
        &mut self,
        batch: RecordBatch,
        buffer_tail: bool,
    ) -> Result<(), StorageError> {
        let target_rows = self.policy.target_file_rows.max(1);
        let mut offset = 0;
        while offset < batch.num_rows() {
            let remaining = batch.num_rows() - offset;
            if self.open.is_none() {
                if buffer_tail && remaining < self.sample_rows() {
                    self.pending_rows += remaining;
                    self.pending.push(batch.slice(offset, remaining));
                    return Ok(());
                }
                let sample = batch.slice(offset, remaining.min(target_rows));
                self.open = Some(self.open_part(&sample)?);
            }
            let part = self.open.as_mut().expect("opened above");
            let take = remaining.min(target_rows - part.rows);
            part.writer
                .write(&batch.slice(offset, take))
                .await
                .map_err(|e| write_err("write", e))?;
            part.rows += take;
            offset += take;
            let bytes = part.writer.bytes_written() + part.writer.in_progress_size();
            if part.rows >= target_rows || bytes >= self.policy.target_file_bytes {
                self.close_part().await?;
            }
        }
        Ok(())
    }

    /// Abandon the publication: abort the open part's multipart upload and delete
    /// the completed parts. Every step is attempted; the first error is returned.
    pub async fn abort(mut self) -> Result<(), StorageError> {
        let mut first_err = None;
        if let Some(part) = self.open.take() {
            let mut upload = part.writer.into_inner();
            if let Err(e) = upload.abort().await {
                first_err.get_or_insert(write_err("abort upload", e));
            }
        }
        for part in std::mem::take(&mut self.done) {
            if let Err(e) = self.store.delete(&self.store_path(&part.path)).await {
                first_err.get_or_insert(write_err("delete part", e));
            }
        }
        first_err.map_or(Ok(()), Err)
    }

    /// The concrete store key of a part key under `dir`.
    fn store_path(&self, key: &Path) -> Path {
        let name = key.filename().unwrap_or_default();
        if self.store_dir.is_empty() {
            Path::from(name)
        } else {
            Path::from(format!("{}/{name}", self.store_dir))
        }
    }

    fn open_part(&self, sample: &RecordBatch) -> Result<OpenPart, StorageError> {
        let key = Path::from(format!(
            "{}/{}-{:05}.parquet",
            self.dir,
            self.stem,
            self.done.len()
        ));
        let upload = PartUpload::new(BufWriter::new(self.store.clone(), self.store_path(&key)));
        let props = self.policy.writer_properties(sample, &self.hints);
        let writer = AsyncArrowWriter::try_new(upload, self.schema.clone(), Some(props))
            .map_err(|e| write_err("open part", e))?;
        Ok(OpenPart {
            key,
            writer,
            rows: 0,
        })
    }

    async fn close_part(&mut self) -> Result<(), StorageError> {
        let Some(mut part) = self.open.take() else {
            return Ok(());
        };
        match part.writer.finish().await {
            Ok(_) => {
                self.done.push(WrittenPart {
                    path: part.key,
                    rows: part.rows as u64,
                    bytes: part.writer.bytes_written() as u64,
                });
                Ok(())
            }
            Err(e) => {
                let _ = part.writer.into_inner().abort().await;
                Err(write_err("finish part", e))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, Int64Array, StringArray};
    use arrow_schema::{DataType, Field};
    use futures::StreamExt;
    use futures::stream::BoxStream;
    use object_store::memory::InMemory;
    use parquet::arrow::ParquetRecordBatchStreamBuilder;
    use parquet::arrow::async_reader::ParquetObjectReader;
    use proximadb_records::ProximaRecord;

    /// A bridge whose keys live under `base/` in an in-memory store, so the
    /// writer's key → store-path mapping is exercised.
    struct TestBridge {
        store: Arc<dyn ObjectStore>,
    }

    #[async_trait::async_trait]
    impl ObjectStoreBridge for TestBridge {
        fn inner_store(&self) -> Arc<dyn ObjectStore> {
            self.store.clone()
        }
        fn object_store_path(&self, key: &Path) -> Path {
            Path::from(format!("base/{key}"))
        }
        async fn read_parquet_batches(
            &self,
            _: &Path,
            _: SchemaRef,
            _: usize,
            _: Option<&str>,
        ) -> Result<BoxStream<'static, Result<RecordBatch, StorageError>>, StorageError> {
            unimplemented!("not used")
        }
        async fn write_records_to_parquet(
            &self,
            _: &Path,
            _: &[ProximaRecord],
            _: Option<&str>,
        ) -> Result<(), StorageError> {
            unimplemented!("not used")
        }
        async fn fetch_vector_segment(
            &self,
            _: &Path,
            _: Option<&str>,
        ) -> Result<Vec<u8>, StorageError> {
            unimplemented!("not used")
        }
        async fn persist_vector_segment(
            &self,
            _: &Path,
            _: &[u8],
            _: Option<&str>,
        ) -> Result<(), StorageError> {
            unimplemented!("not used")
        }
    }

    fn bridge() -> TestBridge {
        TestBridge {
            store: Arc::new(InMemory::new()),
        }
    }

    async fn footer(b: &TestBridge, key: &Path) -> Arc<parquet::file::metadata::ParquetMetaData> {
        #[allow(deprecated)]
        let reader = ParquetObjectReader::new(b.store.clone(), b.object_store_path(key));
        ParquetRecordBatchStreamBuilder::new(reader)
            .await
            .unwrap()
            .metadata()
            .clone()
    }

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    fn batch(range: std::ops::Range<i64>) -> RecordBatch {
        let ids: Vec<i64> = range.collect();
        let names: Vec<String> = ids.iter().map(|i| format!("n{i}")).collect();
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(names)),
            ],
        )
        .unwrap()
    }

    fn policy(target_file_rows: usize, target_file_bytes: usize) -> ParquetWritePolicy {
        ParquetWritePolicy {
            row_group_rows: 64,
            target_file_rows,
            target_file_bytes,
            ..ParquetWritePolicy::default()
        }
    }

    async fn read_ids(b: &TestBridge, path: &Path) -> Vec<i64> {
        #[allow(deprecated)]
        let reader = ParquetObjectReader::new(b.store.clone(), b.object_store_path(path));
        let mut stream = ParquetRecordBatchStreamBuilder::new(reader)
            .await
            .unwrap()
            .build()
            .unwrap();
        let mut out = Vec::new();
        while let Some(batch) = stream.next().await {
            let batch = batch.unwrap();
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            out.extend(ids.values().iter().copied());
        }
        out
    }

    /// The row target is exact: batches that straddle it are split, every part but
    /// the last is full, and the parts together hold every row in order.
    #[tokio::test]
    async fn parts_roll_at_the_row_target_and_keep_every_row() {
        let b = bridge();
        let mut w = MultipartParquetWriter::new(
            &b,
            "t/data/",
            "g7",
            schema(),
            policy(100, usize::MAX),
            ParquetLayoutHints::default(),
        );
        for start in (0..250).step_by(30) {
            w.write(&batch(start..(start + 30).min(250))).await.unwrap();
        }
        let parts = w.finish().await.unwrap();
        let rows: Vec<u64> = parts.iter().map(|p| p.rows).collect();
        assert_eq!(rows, vec![100, 100, 50]);
        assert_eq!(parts[0].path.as_ref(), "t/data/g7-00000.parquet");
        assert_eq!(parts[2].path.as_ref(), "t/data/g7-00002.parquet");

        let mut all = Vec::new();
        for part in &parts {
            assert!(part.bytes > 0);
            assert_eq!(
                b.store
                    .head(&b.object_store_path(&part.path))
                    .await
                    .unwrap()
                    .size,
                part.bytes,
                "reported size is the object's size, under the bridge's base"
            );
            all.extend(read_ids(&b, &part.path).await);
        }
        assert_eq!(all, (0..250).collect::<Vec<_>>());
    }

    /// The byte target closes a part between batches; the field ids are stamped.
    #[tokio::test]
    async fn parts_roll_at_the_byte_target() {
        let b = bridge();
        let mut w = MultipartParquetWriter::new(
            &b,
            "t/data",
            "g1",
            schema(),
            ParquetWritePolicy {
                row_group_rows: 8,
                ..policy(usize::MAX, 1)
            },
            ParquetLayoutHints::default(),
        );
        w.write(&batch(0..10)).await.unwrap();
        w.write(&batch(10..20)).await.unwrap();
        assert_eq!(w.completed_parts().len(), 2, "each batch closed a part");
        let parts = w.finish().await.unwrap();
        assert_eq!(parts.len(), 2);

        let meta = footer(&b, &parts[0].path).await;
        let ids: Vec<i32> = meta
            .file_metadata()
            .schema_descr()
            .root_schema()
            .get_fields()
            .iter()
            .map(|f| f.get_basic_info().id())
            .collect();
        assert_eq!(ids, vec![1, 2]);
    }

    /// Small batches are buffered until a row group's worth arrives, so the
    /// part's Bloom filters are sized from a full row group, not the first batch.
    #[tokio::test]
    async fn bloom_filters_are_sized_from_a_full_row_group() {
        let b = bridge();
        let mut w = MultipartParquetWriter::new(
            &b,
            "t/data",
            "g4",
            schema(),
            ParquetWritePolicy {
                row_group_rows: 4096,
                bloom: proximadb_bloom_policy_for_test(),
                ..policy(usize::MAX, usize::MAX)
            },
            ParquetLayoutHints::default(),
        );
        for start in (0..4096).step_by(512) {
            w.write(&batch(start..start + 512)).await.unwrap();
        }
        assert!(w.completed_parts().is_empty());
        let parts = w.finish().await.unwrap();
        let meta = footer(&b, &parts[0].path).await;
        let rg = meta.row_group(0);
        let id_col = rg.column(0);
        let bloom_len = id_col
            .bloom_filter_length()
            .expect("the id column carries a bloom filter");
        // 4096 distinct ids at 1% FPP need ~4.9 KB; a filter sized from one
        // 512-row batch would be ~600 bytes.
        assert!(
            bloom_len > 4000,
            "bloom sized for the row group: {bloom_len}"
        );
    }

    fn proximadb_bloom_policy_for_test() -> crate::parquet_write_policy::BloomPolicy {
        crate::parquet_write_policy::BloomPolicy {
            min_rows: 1,
            ..ParquetWritePolicy::default().bloom
        }
    }

    /// No rows still yields one (empty) part carrying the schema.
    #[tokio::test]
    async fn an_empty_publication_writes_one_empty_part() {
        let b = bridge();
        let w = MultipartParquetWriter::new(
            &b,
            "t/data",
            "g0",
            schema(),
            policy(100, usize::MAX),
            ParquetLayoutHints::default(),
        );
        let parts = w.finish().await.unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].rows, 0);
        assert!(read_ids(&b, &parts[0].path).await.is_empty());
    }

    /// Abort removes the completed parts and leaves no visible open part.
    #[tokio::test]
    async fn abort_removes_every_part() {
        let b = bridge();
        let mut w = MultipartParquetWriter::new(
            &b,
            "t/data",
            "g2",
            schema(),
            policy(100, usize::MAX),
            ParquetLayoutHints::default(),
        );
        w.write(&batch(0..150)).await.unwrap();
        assert_eq!(w.completed_parts().len(), 1);
        w.abort().await.unwrap();
        let left: Vec<_> = b.store.list(None).collect().await;
        assert!(left.is_empty(), "no part survives an abort: {left:?}");
    }

    /// Fails the PUT of any object whose key ends with `suffix`.
    #[derive(Debug)]
    struct FailingPut {
        inner: InMemory,
        suffix: &'static str,
    }

    impl std::fmt::Display for FailingPut {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("FailingPut")
        }
    }

    #[async_trait::async_trait]
    impl ObjectStore for FailingPut {
        async fn put_opts(
            &self,
            path: &Path,
            payload: object_store::PutPayload,
            opts: object_store::PutOptions,
        ) -> object_store::Result<object_store::PutResult> {
            if path.as_ref().ends_with(self.suffix) {
                return Err(object_store::Error::Generic {
                    store: "FailingPut",
                    source: "injected PUT failure".into(),
                });
            }
            self.inner.put_opts(path, payload, opts).await
        }
        async fn put_multipart_opts(
            &self,
            path: &Path,
            opts: object_store::PutMultipartOptions,
        ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
            self.inner.put_multipart_opts(path, opts).await
        }
        async fn get_opts(
            &self,
            path: &Path,
            opts: object_store::GetOptions,
        ) -> object_store::Result<object_store::GetResult> {
            self.inner.get_opts(path, opts).await
        }
        fn delete_stream(
            &self,
            paths: BoxStream<'static, object_store::Result<Path>>,
        ) -> BoxStream<'static, object_store::Result<Path>> {
            self.inner.delete_stream(paths)
        }
        fn list(
            &self,
            prefix: Option<&Path>,
        ) -> BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
            self.inner.list(prefix)
        }
        async fn list_with_delimiter(
            &self,
            prefix: Option<&Path>,
        ) -> object_store::Result<object_store::ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }
        async fn copy_opts(
            &self,
            from: &Path,
            to: &Path,
            opts: object_store::CopyOptions,
        ) -> object_store::Result<()> {
            self.inner.copy_opts(from, to, opts).await
        }
    }

    /// A part whose final PUT fails is an error, not a panic (the upload is past
    /// the point where `BufWriter::abort` is allowed), and its finished sibling is
    /// deleted.
    #[tokio::test]
    async fn a_failed_part_close_is_an_error_and_removes_the_siblings() {
        let b = TestBridge {
            store: Arc::new(FailingPut {
                inner: InMemory::new(),
                suffix: "-00001.parquet",
            }),
        };
        let mut w = MultipartParquetWriter::new(
            &b,
            "t/data",
            "g3",
            schema(),
            policy(100, usize::MAX),
            ParquetLayoutHints::default(),
        );
        w.write(&batch(0..150)).await.unwrap();
        assert_eq!(w.completed_parts().len(), 1);
        let err = w.finish().await.unwrap_err().to_string();
        assert!(err.contains("finish part"), "{err}");
        let left: Vec<_> = b.store.list(None).collect().await;
        assert!(
            left.is_empty(),
            "no part survives a failed finish: {left:?}"
        );
    }

    /// A batch whose columns do not fit the writer's schema is an error, not a
    /// silently mis-typed file.
    #[tokio::test]
    async fn a_mismatched_batch_is_rejected() {
        let b = bridge();
        let mut w = MultipartParquetWriter::new(
            &b,
            "t/data",
            "g3",
            schema(),
            policy(100, usize::MAX),
            ParquetLayoutHints::default(),
        );
        let other = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("x", DataType::Utf8, true)])),
            vec![Arc::new(StringArray::from(vec!["a"])) as Arc<dyn Array>],
        )
        .unwrap();
        assert!(w.write(&other).await.is_err());
        w.abort().await.unwrap();
    }
}
