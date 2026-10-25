//! TD-PAXDIR-1 D6 test support: a fixed per-request latency and a sequential
//! wave count (design §4.1, "a test-only `LatencyFileSystem` wrapper").
//!
//! `LatencyFileSystem` sleeps `PROXIMADB_TEST_FS_LATENCY_MS` before every
//! read, ranged read, LIST, HEAD and existence check, and records each
//! request's `(task, start, end)`. Requests a task issues concurrently (one
//! `read_ranges_prefetch` wave) overlap; requests it issues one after another
//! do not, so a task's **waves** are its maximal groups of overlapping
//! requests. [`critical_path_waves`] adds the waves of the query's own task
//! (LIST, directory) to those of its slowest spawned per-segment scan, which
//! is exact for the search fan-out — it waits for every scan — and, unlike
//! one global overlap count, never merges two scans' waves because their
//! tasks started a few microseconds apart.
//!
//! Compiled only for tests and installed by the filesystem factory only when
//! the env gate is set.

use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use proximadb_storage_filesystem_types::{DirEntry, FileSystem, FsFileMetadata, FsResult};

/// One recorded request: issuing task (`None` = not a spawned task), its
/// interval, and a label (`op path [offset+len]`) for failure messages.
type Request = (Option<tokio::task::Id>, Instant, Instant, String);

fn requests() -> &'static Mutex<Vec<Request>> {
    static REQUESTS: OnceLock<Mutex<Vec<Request>>> = OnceLock::new();
    REQUESTS.get_or_init(Default::default)
}

fn delay() -> Option<Duration> {
    static DELAY: OnceLock<Option<Duration>> = OnceLock::new();
    *DELAY.get_or_init(|| {
        std::env::var("PROXIMADB_TEST_FS_LATENCY_MS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|ms| *ms > 0)
            .map(Duration::from_millis)
    })
}

/// Wrap `fs` when `PROXIMADB_TEST_FS_LATENCY_MS` is set; otherwise return it.
pub(crate) fn maybe_wrap(fs: Arc<dyn FileSystem>) -> Arc<dyn FileSystem> {
    match delay() {
        Some(delay) => Arc::new(LatencyFileSystem { inner: fs, delay }),
        None => fs,
    }
}

/// Forget every recorded request.
pub(crate) fn reset() {
    requests()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
}

/// Requests recorded since the last [`reset`].
pub(crate) fn request_count() -> usize {
    requests()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .len()
}

/// Sequential waves since the last [`reset`]: the root task's waves plus the
/// most of any spawned task (see the module docs).
pub(crate) fn critical_path_waves() -> usize {
    let recorded = requests()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    let mut by_task: std::collections::HashMap<Option<tokio::task::Id>, Vec<(Instant, Instant)>> =
        Default::default();
    for (task, start, end, _) in recorded {
        by_task.entry(task).or_default().push((start, end));
    }
    let root = by_task.remove(&None).map_or(0, |r| waves(r));
    root + by_task.into_values().map(waves).max().unwrap_or(0)
}

/// Every recorded request as `task +start_ms..end_ms label`, in start order
/// (for diagnosing a wave count).
pub(crate) fn describe() -> String {
    let mut recorded = requests()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    recorded.sort_by_key(|r| r.1);
    let Some(t0) = recorded.first().map(|r| r.1) else {
        return String::new();
    };
    recorded
        .iter()
        .map(|(task, start, end, label)| {
            format!(
                "{task:?} +{}..{}ms {label}\n",
                (*start - t0).as_millis(),
                (*end - t0).as_millis()
            )
        })
        .collect()
}

/// Maximal groups of overlapping intervals.
fn waves(mut intervals: Vec<(Instant, Instant)>) -> usize {
    intervals.sort_by_key(|(start, _)| *start);
    let mut count = 0;
    let mut group_end: Option<Instant> = None;
    for (start, end) in intervals {
        match group_end {
            Some(e) if start < e => group_end = Some(e.max(end)),
            _ => {
                count += 1;
                group_end = Some(end);
            }
        }
    }
    count
}

#[derive(Debug)]
struct LatencyFileSystem {
    inner: Arc<dyn FileSystem>,
    delay: Duration,
}

impl LatencyFileSystem {
    async fn timed<T>(&self, label: String, op: impl std::future::Future<Output = T>) -> T {
        let start = Instant::now();
        tokio::time::sleep(self.delay).await;
        let out = op.await;
        requests()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((tokio::task::try_id(), start, Instant::now(), label));
        out
    }
}

proximadb_storage_filesystem_types::forward_filesystem_impl! {
    impl FileSystem for LatencyFileSystem => inner;
    forward [
        as_any, filesystem_type, range_coalesce_policy, write, compare_exchange,
        supports_bounded_local_file_write, write_local_file, write_if_absent, append,
        supports_append, supports_conditional_replace, delete, create_dir, create_dir_all,
        copy, move_file, sync, sync_file, sync_data, open_file,
    ];
    {
        fn supports_mmap(&self) -> bool {
            false // every read must pay the injected latency
        }

        // Latency-bearing requests. `read_ranges` / `read_ranges_prefetch` keep
        // the trait defaults, which issue `self.read_range` per physical range.
        async fn read(&self, path: &str) -> FsResult<Vec<u8>> {
            self.timed(format!("read {path}"), self.inner.read(path))
                .await
        }
        async fn read_range(&self, path: &str, offset: u64, length: u64) -> FsResult<Vec<u8>> {
            let label = format!("range {path} [{offset}+{length}]");
            self.timed(label, self.inner.read_range(path, offset, length))
                .await
        }
        async fn list(&self, path: &str) -> FsResult<Vec<DirEntry>> {
            self.timed(format!("list {path}"), self.inner.list(path))
                .await
        }
        async fn metadata(&self, path: &str) -> FsResult<FsFileMetadata> {
            self.timed(format!("head {path}"), self.inner.metadata(path))
                .await
        }
        async fn exists(&self, path: &str) -> FsResult<bool> {
            self.timed(format!("exists {path}"), self.inner.exists(path))
                .await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::waves;
    use std::time::{Duration, Instant};

    #[test]
    fn overlapping_requests_form_one_wave_and_sequential_ones_do_not() {
        let t = Instant::now();
        let at = |ms: u64| t + Duration::from_millis(ms);
        assert_eq!(waves(Vec::new()), 0);
        // Three parallel reads, then two sequential ones.
        let intervals = vec![
            (at(0), at(10)),
            (at(1), at(11)),
            (at(2), at(12)),
            (at(12), at(22)),
            (at(30), at(40)),
        ];
        assert_eq!(waves(intervals), 3);
    }
}
