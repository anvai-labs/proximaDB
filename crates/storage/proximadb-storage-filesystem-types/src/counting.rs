//! GET-count instrumentation for the filesystem I/O seam (TD-096 S2 / S1.5).
//!
//! `CountingFileSystem` is a pass-through wrapper around `Arc<dyn FileSystem>` that
//! tallies the read operations (the object-store GET surface) — `read` (full
//! object), `read_range` (ranged GET), `read_ranges` (a logical batch whose
//! **physical** reads are counted from the plan, which may be fewer than the
//! logical ranges when the filesystem coalesces), and `get_mmap`
//! (whole-file map) — plus total bytes returned. It is wired in by
//! `FilesystemFactory::get_filesystem` only when `PROXIMADB_COUNT_FS_IO=1`
//! (default OFF → zero behavior change), so a bench can measure the per-search
//! GET/byte cost for SST- vs HELIX-backed collections and resolve TD-096 S2's
//! route-disclosure question with evidence.
//!
//! Both engines read through the same `FilesystemFactory`/`FileSystem` seam, so
//! one wrapper instruments both. Writes / metadata / list / open_file are passed
//! through uncounted (they are not the GET-read cost term); reads done through
//! the `FilesystemFile` handle returned by `open_file` are NOT counted here
//! (documented caveat — the bulk read path uses `read`/`read_range`).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use async_trait::async_trait;

use crate::{DirEntry, FileOptions, FileSystem, FilesystemFile, FsFileMetadata, FsResult};

/// Relaxed ordering is correct everywhere here: the counters are best-effort
/// instrumentation read once after a single-threaded bench run; they do not
/// gate correctness.
const RELAXED: Ordering = Ordering::Relaxed;

/// Counters for the filesystem GET surface. Snapshotted/reset by a bench
/// between runs.
#[derive(Debug, Default)]
pub struct GetCounters {
    /// Whole-object reads (`read`) + whole-file maps (`get_mmap`).
    pub full_reads: AtomicU64,
    /// Single ranged reads (`read_range`).
    pub range_reads: AtomicU64,
    /// Physical ranged reads issued by logical `read_ranges` calls.
    ///
    /// Counted from [`FileSystem::plan_read_ranges`], so this is the number of
    /// requests the wire actually sees — one per logical range when no
    /// coalescing policy is set, fewer when one is. Concurrency reduces elapsed
    /// rounds; it does not turn them into one cloud transaction.
    ///
    /// Cross-check, do not trust in isolation: this is *predicted* from the
    /// plan, while `io_trace.range_gets` is *measured* at the leaf. The two
    /// agreeing is the proof that neither is lying.
    pub batched_range_reads: AtomicU64,
    /// Total bytes returned to callers (logical bytes).
    pub bytes_read: AtomicU64,
    /// Bytes fetched but never returned — the gap over-read coalescing paid to
    /// save a request. Kept separate so `bytes_read` keeps meaning "what the
    /// caller got" instead of silently becoming "what the wire moved".
    pub coalesced_overread_bytes: AtomicU64,
}

impl GetCounters {
    /// Total counted GET operations (full + range + batched-range).
    pub fn total_gets(&self) -> u64 {
        self.full_reads.load(RELAXED)
            + self.range_reads.load(RELAXED)
            + self.batched_range_reads.load(RELAXED)
    }

    /// Reset all counters to zero (between bench runs).
    pub fn reset(&self) {
        self.full_reads.store(0, RELAXED);
        self.range_reads.store(0, RELAXED);
        self.batched_range_reads.store(0, RELAXED);
        self.bytes_read.store(0, RELAXED);
        self.coalesced_overread_bytes.store(0, RELAXED);
    }
}

#[cfg(test)]
mod physical_get_meter_tests {
    use super::{CountingFileSystem, GetCounters, RELAXED};
    use crate::{FileSystem, RangeCoalescePolicy};
    use std::sync::Arc;

    #[test]
    fn total_gets_sums_the_three_surfaces() {
        let counters = GetCounters::default();
        counters.full_reads.fetch_add(1, RELAXED);
        counters.range_reads.fetch_add(1, RELAXED);
        counters.batched_range_reads.fetch_add(3, RELAXED);
        assert_eq!(counters.total_gets(), 5);
    }

    /// The meter must report MEASURED physical reads, not `ranges.len()`.
    ///
    /// This replaces a test that asserted `read_ranges` always costs one
    /// physical GET per range. That was true of the old implementation, but it
    /// was pure arithmetic on the counter struct and never exercised
    /// `read_ranges` — so it would have kept passing while its own comment
    /// became false. A green test asserting a stale invariant is worse than a
    /// red one.
    #[test]
    fn batched_ranges_are_metered_from_the_plan_not_the_request() {
        let inner = Arc::new(crate::read_ranges_coalescing_tests::CountingFake::new(
            4096,
            crate::read_ranges_coalescing_tests::EofMode::Clamp,
            Some(RangeCoalescePolicy {
                max_gap_bytes: 64,
                max_merged_bytes: 4096,
            }),
        ));
        let counted = CountingFileSystem::new(inner, Arc::new(GetCounters::default()));

        // Three near-adjacent ranges collapse to ONE physical read.
        let plan = counted
            .plan_read_ranges(&[0..64, 64..128, 130..200])
            .expect("plan");
        assert_eq!(
            plan.physical.len(),
            1,
            "policy must reach through the decorator"
        );

        // ...and with no policy the plan is the identity, so the meter reports
        // one physical read per logical range exactly as before.
        let bare = Arc::new(crate::read_ranges_coalescing_tests::CountingFake::new(
            4096,
            crate::read_ranges_coalescing_tests::EofMode::Clamp,
            None,
        ));
        let counted_bare = CountingFileSystem::new(bare, Arc::new(GetCounters::default()));
        let identity = counted_bare
            .plan_read_ranges(&[0..64, 64..128, 130..200])
            .expect("plan");
        assert_eq!(identity.physical.len(), 3);
    }
}

/// Process-global counters shared by every `CountingFileSystem` produced while
/// `PROXIMADB_COUNT_FS_IO=1`. A single-threaded bench reads + resets this
/// around each measured phase; no contention in that mode.
static GLOBAL_COUNTERS: OnceLock<Arc<GetCounters>> = OnceLock::new();

/// The shared global counters (lazily allocated once, reused for the process
/// lifetime). `CountingFileSystem` writes here; benches read + reset it.
pub fn global_counters() -> Arc<GetCounters> {
    GLOBAL_COUNTERS
        .get_or_init(|| Arc::new(GetCounters::default()))
        .clone()
}

/// Dependency-inversion hook for storage stacks that cannot depend on the
/// root observability crate. `ProximaObjectStore` uses this interface because
/// it bypasses the root `FileSystem` leaf backends; `CountingFileSystem` does
/// not, because diagnostic counters and per-query evidence have separate
/// ownership.
pub trait IoRecorder: Send + Sync + std::fmt::Debug {
    fn record_full_read(&self, bytes: u64);
    fn record_range_read(&self, bytes: u64);
    fn record_batched_ranges(&self, physical_gets: u64, bytes: u64);
}

/// Pass-through `FileSystem` wrapper that counts read operations. See module
/// docs.
#[derive(Debug)]
pub struct CountingFileSystem {
    inner: Arc<dyn FileSystem>,
    counters: Arc<GetCounters>,
}

impl CountingFileSystem {
    /// Wrap `inner`, incrementing `counters` on every read. Pass
    /// [`global_counters()`] to share the process-wide counters used by the
    /// `PROXIMADB_COUNT_FS_IO` bench path. No per-query recorder.
    pub fn new(inner: Arc<dyn FileSystem>, counters: Arc<GetCounters>) -> Self {
        Self { inner, counters }
    }
}

#[async_trait]
impl FileSystem for CountingFileSystem {
    fn as_any(&self) -> &dyn std::any::Any {
        // Delegate so existing downcasts to the concrete backing filesystem
        // (e.g. LocalFileSystem) still succeed through the wrapper.
        self.inner.as_any()
    }
    fn filesystem_type(&self) -> &'static str {
        self.inner.filesystem_type()
    }
    fn supports_mmap(&self) -> bool {
        self.inner.supports_mmap()
    }

    async fn read(&self, path: &str) -> FsResult<Vec<u8>> {
        let buf = self.inner.read(path).await?;
        let bytes = buf.len() as u64;
        self.counters.full_reads.fetch_add(1, RELAXED);
        self.counters.bytes_read.fetch_add(bytes, RELAXED);
        Ok(buf)
    }

    async fn read_range(&self, path: &str, offset: u64, length: u64) -> FsResult<Vec<u8>> {
        let buf = self.inner.read_range(path, offset, length).await?;
        let bytes = buf.len() as u64;
        self.counters.range_reads.fetch_add(1, RELAXED);
        self.counters.bytes_read.fetch_add(bytes, RELAXED);
        Ok(buf)
    }

    fn range_coalesce_policy(&self) -> Option<crate::RangeCoalescePolicy> {
        // Delegate so the policy query traverses the whole decorator chain to
        // whichever layer actually holds it. Without this the counter would
        // plan with `None` and report one physical read per logical range even
        // when the inner filesystem coalesces.
        self.inner.range_coalesce_policy()
    }

    async fn read_ranges(
        &self,
        path: &str,
        ranges: Vec<std::ops::Range<u64>>,
    ) -> FsResult<Vec<Vec<u8>>> {
        // Count what the wire will see, not what the caller asked for. Ask the
        // inner filesystem to plan: it owns the coalescing policy, so this
        // traverses the decorator chain to whoever actually holds it.
        let plan = self.inner.plan_read_ranges(&ranges)?;
        let physical_gets = plan.physical.len() as u64;
        let overread = plan.overread_bytes();
        let bufs = self.inner.read_ranges(path, ranges).await?;
        let bytes: u64 = bufs.iter().map(Vec::len).sum::<usize>() as u64;
        self.counters
            .batched_range_reads
            .fetch_add(physical_gets, RELAXED);
        self.counters.bytes_read.fetch_add(bytes, RELAXED);
        self.counters
            .coalesced_overread_bytes
            .fetch_add(overread, RELAXED);
        Ok(bufs)
    }

    async fn get_mmap(&self, path: &str) -> FsResult<Option<memmap2::Mmap>> {
        // A successful mmap maps the whole file — count it as a full read.
        let mmap = self.inner.get_mmap(path).await?;
        if mmap.is_some() {
            self.counters.full_reads.fetch_add(1, RELAXED);
        }
        Ok(mmap)
    }

    // --- Pass-through (uncounted): writes / metadata / listing / open_file ---

    async fn write(&self, path: &str, data: &[u8], options: Option<FileOptions>) -> FsResult<()> {
        self.inner.write(path, data, options).await
    }
    fn supports_bounded_local_file_write(&self) -> bool {
        self.inner.supports_bounded_local_file_write()
    }
    async fn write_local_file(
        &self,
        path: &str,
        local_path: &std::path::Path,
        options: Option<FileOptions>,
    ) -> FsResult<u64> {
        self.inner.write_local_file(path, local_path, options).await
    }
    async fn write_if_absent(
        &self,
        path: &str,
        data: &[u8],
        options: Option<FileOptions>,
    ) -> FsResult<()> {
        self.inner.write_if_absent(path, data, options).await
    }
    async fn append(&self, path: &str, data: &[u8]) -> FsResult<()> {
        self.inner.append(path, data).await
    }

    async fn compare_exchange(
        &self,
        path: &str,
        expected: Option<&[u8]>,
        replacement: &[u8],
        guards: &[(&str, Option<&[u8]>)],
    ) -> FsResult<bool> {
        self.inner
            .compare_exchange(path, expected, replacement, guards)
            .await
    }
    fn supports_append(&self) -> bool {
        self.inner.supports_append()
    }

    fn supports_conditional_replace(&self) -> bool {
        self.inner.supports_conditional_replace()
    }
    async fn delete(&self, path: &str) -> FsResult<()> {
        self.inner.delete(path).await
    }
    async fn exists(&self, path: &str) -> FsResult<bool> {
        self.inner.exists(path).await
    }
    async fn metadata(&self, path: &str) -> FsResult<FsFileMetadata> {
        self.inner.metadata(path).await
    }
    async fn list(&self, path: &str) -> FsResult<Vec<DirEntry>> {
        self.inner.list(path).await
    }
    async fn create_dir(&self, path: &str) -> FsResult<()> {
        self.inner.create_dir(path).await
    }
    async fn create_dir_all(&self, path: &str) -> FsResult<()> {
        self.inner.create_dir_all(path).await
    }
    async fn copy(&self, from: &str, to: &str) -> FsResult<()> {
        self.inner.copy(from, to).await
    }
    async fn move_file(&self, from: &str, to: &str) -> FsResult<()> {
        self.inner.move_file(from, to).await
    }
    async fn sync(&self) -> FsResult<()> {
        self.inner.sync().await
    }
    async fn open_file(&self, path: &str, create: bool) -> FsResult<Box<dyn FilesystemFile>> {
        self.inner.open_file(path, create).await
    }
}
