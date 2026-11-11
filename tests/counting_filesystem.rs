//! TD-096 S2 / S1.5: unit test for the `CountingFileSystem` GET-count wrapper.
//!
//! Wraps a minimal fake `FileSystem` and asserts the read-operation counters
//! (full / range / batched-range) and bytes increment correctly, that the
//! default-impl chaining does NOT double-count at the wrapper, and that
//! `reset()` clears them.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use async_trait::async_trait;
use proximadb_storage_filesystem_types::counting::{CountingFileSystem, GetCounters};
use proximadb_storage_filesystem_types::{
    DirEntry, FileOptions, FileSystem, FilesystemError, FilesystemFile, FsFileMetadata, FsResult,
    RangeCoalescePolicy,
};

/// Minimal fake filesystem: `read` returns a fixed buffer; everything else
/// no-ops or errors (the wrapper only needs the read surface to count).
#[derive(Debug)]
struct FakeFs {
    /// Coalesce adjacent ranges into one physical GET (object-store-like).
    coalesce: bool,
}

#[async_trait]
impl FileSystem for FakeFs {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn filesystem_type(&self) -> &'static str {
        "fake"
    }
    // Coalesce adjacent ranges, like the object-store backends. The wrapper
    // counts PHYSICAL GETs after the inner plan ("count what the wire will
    // see"), so the batched read below is one GET, not one per range. A
    // wrapper counting logical ranges would report 2 and fail this test.
    fn range_coalesce_policy(&self) -> Option<RangeCoalescePolicy> {
        self.coalesce.then_some(RangeCoalescePolicy {
            max_gap_bytes: 0,
            max_merged_bytes: 1 << 20,
        })
    }
    async fn read(&self, _path: &str) -> FsResult<Vec<u8>> {
        Ok(b"hello world".to_vec())
    }
    async fn write(
        &self,
        _path: &str,
        _data: &[u8],
        _options: Option<FileOptions>,
    ) -> FsResult<()> {
        Ok(())
    }
    async fn append(&self, _path: &str, _data: &[u8]) -> FsResult<()> {
        Ok(())
    }
    async fn delete(&self, _path: &str) -> FsResult<()> {
        Ok(())
    }
    async fn exists(&self, _path: &str) -> FsResult<bool> {
        Ok(true)
    }
    async fn metadata(&self, _path: &str) -> FsResult<FsFileMetadata> {
        Err(FilesystemError::InvalidOperation("fake".to_string()))
    }
    async fn list(&self, _path: &str) -> FsResult<Vec<DirEntry>> {
        Ok(Vec::new())
    }
    async fn create_dir(&self, _path: &str) -> FsResult<()> {
        Ok(())
    }
    async fn create_dir_all(&self, _path: &str) -> FsResult<()> {
        Ok(())
    }
    async fn copy(&self, _from: &str, _to: &str) -> FsResult<()> {
        Ok(())
    }
    async fn move_file(&self, _from: &str, _to: &str) -> FsResult<()> {
        Ok(())
    }
    async fn sync(&self) -> FsResult<()> {
        Ok(())
    }
    async fn open_file(&self, _path: &str, _create: bool) -> FsResult<Box<dyn FilesystemFile>> {
        Err(FilesystemError::InvalidOperation("fake".to_string()))
    }
}

#[tokio::test]
async fn counting_filesystem_counts_reads_without_double_counting() {
    let counters = Arc::new(GetCounters::default());
    let fs = CountingFileSystem::new(
        Arc::new(FakeFs { coalesce: true }) as Arc<dyn FileSystem>,
        counters.clone(),
    );

    // One full read, one ranged read, one batched read of two adjacent
    // ranges (coalesced by the fake's plan into one physical GET).
    let buf = fs.read("a").await.expect("read");
    let _ = fs.read_range("a", 0, 5).await.expect("read_range");
    let _ = fs
        .read_ranges("a", vec![0..3, 3..6])
        .await
        .expect("read_ranges");

    assert_eq!(counters.full_reads.load(Ordering::Relaxed), 1);
    assert_eq!(counters.range_reads.load(Ordering::Relaxed), 1);
    assert_eq!(counters.batched_range_reads.load(Ordering::Relaxed), 1);
    assert_eq!(counters.total_gets(), 3);

    // full read returns the whole buffer; range/batched return slices — bytes
    // are the sum of what each read actually returned (no double counting: the
    // inner default `read_range`/`read_ranges` call `inner.read`, not the
    // wrapper's `read`).
    let range_buf = fs.read_range("a", 0, 4).await.expect("read_range again");
    assert_eq!(range_buf.len(), 4);
    assert!(counters.bytes_read.load(Ordering::Relaxed) >= buf.len() as u64);
    // Exact: 11 (full) + 5 (range) + 6 (batched) + 4 (range again). A wrapper
    // that double-counted any read would exceed this.
    assert_eq!(counters.bytes_read.load(Ordering::Relaxed), 11 + 5 + 6 + 4);

    // reset clears everything.
    counters.reset();
    assert_eq!(counters.total_gets(), 0);
    assert_eq!(counters.bytes_read.load(Ordering::Relaxed), 0);
}

/// Without coalescing (the default plan), two ranges are two physical GETs:
/// the wrapper counts what the wire sees, in both directions.
#[tokio::test]
async fn counting_filesystem_counts_each_physical_get_without_coalescing() {
    let counters = Arc::new(GetCounters::default());
    let fs = CountingFileSystem::new(
        Arc::new(FakeFs { coalesce: false }) as Arc<dyn FileSystem>,
        counters.clone(),
    );
    let _ = fs
        .read_ranges("a", vec![0..3, 3..6])
        .await
        .expect("read_ranges");
    assert_eq!(counters.batched_range_reads.load(Ordering::Relaxed), 2);
    assert_eq!(counters.total_gets(), 2);
    assert_eq!(counters.bytes_read.load(Ordering::Relaxed), 6);
}
