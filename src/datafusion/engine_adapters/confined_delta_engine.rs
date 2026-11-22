//! Delta Kernel I/O confined to one operator-authorized local table.

use std::num::NonZeroUsize;
use std::sync::{Arc, LazyLock};

use bytes::Bytes;
use delta_kernel::schema::SchemaRef;
use delta_kernel::{
    CancellationTokenRef, Engine, EngineData, EvaluationHandler, FileDataReadResultIterator,
    FileMeta, FileSize, FileSlice, JsonHandler, KernelError, KernelResult,
    KernelResultIteratorStatic, ParquetFooter, ParquetHandler, PredicateRef, StorageHandler,
};
use delta_kernel_default_engine::{DefaultEngine, executor::tokio::TokioBackgroundExecutor};
use object_store::path::Path;
use url::Url;

use super::confined_file_store::ConfinedFileStore;

// This URL is never used to construct a remote store. The engine receives
// an already-open directory capability and the handlers check every URL
// that can otherwise trigger the default engine's presigned-URL I/O.
pub(super) const DELTA_ROOT: &str = "s3://proxima-local-delta/";

const MAX_STORAGE_READ_BYTES: u64 = 16 * 1024 * 1024;

pub(super) fn within_delta_root(url: &Url) -> KernelResult<Path> {
    if !url.as_str().starts_with(DELTA_ROOT) || url.query().is_some() || url.fragment().is_some() {
        return Err(KernelError::unsupported(
            "Delta reference is outside the pinned table",
        ));
    }
    Path::from_url_path(url.path()).map_err(KernelError::generic_err)
}

pub(super) struct ConfinedDeltaEngine {
    inner: DefaultEngine<TokioBackgroundExecutor>,
}

impl ConfinedDeltaEngine {
    pub(super) fn new(store: Arc<ConfinedFileStore>) -> Self {
        // Reuse the I/O executor across bound tables instead of starting a
        // background runtime for each table in a join.
        static EXECUTOR: LazyLock<Arc<TokioBackgroundExecutor>> =
            LazyLock::new(|| Arc::new(TokioBackgroundExecutor::new()));
        let inner = DefaultEngine::builder(store.clone())
            .with_task_executor(EXECUTOR.clone())
            .with_batch_size(NonZeroUsize::new(8192).unwrap_or(NonZeroUsize::MIN))
            .with_buffer_size(NonZeroUsize::MIN)
            .build();
        Self { inner }
    }
}

impl Engine for ConfinedDeltaEngine {
    fn evaluation_handler(&self) -> Arc<dyn EvaluationHandler> {
        self.inner.evaluation_handler()
    }

    fn storage_handler(&self) -> Arc<dyn StorageHandler> {
        Arc::new(ConfinedStorage {
            inner: self.inner.storage_handler(),
        })
    }

    fn json_handler(&self) -> Arc<dyn JsonHandler> {
        // The JSON reader uses only the injected store.
        self.inner.json_handler()
    }

    fn parquet_handler(&self) -> Arc<dyn ParquetHandler> {
        Arc::new(ConfinedParquet(self.inner.parquet_handler()))
    }
}

struct ConfinedParquet(Arc<dyn ParquetHandler>);

impl ParquetHandler for ConfinedParquet {
    fn read_parquet_files(
        &self,
        files: &[FileMeta],
        schema: SchemaRef,
        predicate: Option<PredicateRef>,
    ) -> KernelResult<FileDataReadResultIterator> {
        self.read_parquet_files_with_cancellation(files, schema, predicate, None)
    }

    fn read_parquet_files_with_cancellation(
        &self,
        files: &[FileMeta],
        schema: SchemaRef,
        predicate: Option<PredicateRef>,
        token: Option<CancellationTokenRef>,
    ) -> KernelResult<FileDataReadResultIterator> {
        for file in files {
            within_delta_root(&file.location)?;
        }
        self.0
            .read_parquet_files_with_cancellation(files, schema, predicate, token)
    }

    fn read_parquet_footer(&self, file: &FileMeta) -> KernelResult<ParquetFooter> {
        within_delta_root(&file.location)?;
        self.0.read_parquet_footer(file)
    }

    fn read_parquet_footer_with_cancellation(
        &self,
        file: &FileMeta,
        token: Option<CancellationTokenRef>,
    ) -> KernelResult<ParquetFooter> {
        within_delta_root(&file.location)?;
        self.0.read_parquet_footer_with_cancellation(file, token)
    }

    fn write_parquet_file(
        &self,
        _: Url,
        _: KernelResultIteratorStatic<Box<dyn EngineData>>,
    ) -> KernelResult<FileSize> {
        Err(read_only())
    }
}

struct ConfinedStorage {
    inner: Arc<dyn StorageHandler>,
}

impl StorageHandler for ConfinedStorage {
    fn list_from(&self, path: &Url) -> KernelResult<KernelResultIteratorStatic<FileMeta>> {
        self.list_from_with_cancellation(path, None)
    }

    fn list_from_with_cancellation(
        &self,
        path: &Url,
        token: Option<CancellationTokenRef>,
    ) -> KernelResult<KernelResultIteratorStatic<FileMeta>> {
        within_delta_root(path)?;
        self.inner.list_from_with_cancellation(path, token)
    }

    fn read_files(&self, files: Vec<FileSlice>) -> KernelResult<KernelResultIteratorStatic<Bytes>> {
        self.read_files_with_cancellation(files, None)
    }

    fn read_files_with_cancellation(
        &self,
        files: Vec<FileSlice>,
        token: Option<CancellationTokenRef>,
    ) -> KernelResult<KernelResultIteratorStatic<Bytes>> {
        // Kernel calls the storage handler for small metadata and DV blobs.
        // Commit and Parquet readers stream through their dedicated handlers.
        let inner = self.inner.clone();
        Ok(Box::new(files.into_iter().flat_map(move |(url, range)| {
            // Keep a missing optional object (such as _last_checkpoint) as
            // an item error so Kernel can distinguish it from a failed reader.
            let prepared = (|| {
                if token.as_ref().is_some_and(|token| token.is_cancelled()) {
                    return Err(KernelError::Cancelled);
                }
                within_delta_root(&url)?;
                let range = match range {
                    Some(range) => range,
                    None => 0..inner.head(&url)?.size,
                };
                if range.end.saturating_sub(range.start) > MAX_STORAGE_READ_BYTES {
                    return Err(KernelError::unsupported(
                        "Delta metadata or deletion vector exceeds 16 MiB",
                    ));
                }
                inner.read_files_with_cancellation(vec![(url, Some(range))], token.clone())
            })();
            prepared.unwrap_or_else(|error| Box::new(std::iter::once(Err(error))))
        })))
    }

    fn head(&self, path: &Url) -> KernelResult<FileMeta> {
        within_delta_root(path)?;
        self.inner.head(path)
    }

    fn copy_atomic(&self, _: &Url, _: &Url) -> KernelResult<()> {
        Err(read_only())
    }

    fn put(&self, _: &Url, _: Bytes, _: bool) -> KernelResult<()> {
        Err(read_only())
    }

    fn delete(&self, _: &Url) -> KernelResult<()> {
        Err(read_only())
    }
}

fn read_only() -> KernelError {
    KernelError::unsupported("pinned Delta is read-only")
}
