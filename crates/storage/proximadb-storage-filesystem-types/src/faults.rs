//! TD-COMPACT-13 TDD support: deterministic delete-fault injection.
//!
//! `FaultInjectingFileSystem` delegates everything to its inner filesystem
//! except `delete`, which can be armed — via
//! `PROXIMADB_TEST_FS_DELETE_FAIL_FIRST` — to fail the FIRST delete of each
//! unique path once (`=1`) or every delete (`=always`) with a transient
//! `FilesystemError::Network`. This gives the compaction-retirement tests a
//! deterministic reproduction of a transient object-store delete failure
//! without a proxy. Unset (the default), the wrapper is byte-identical to its
//! inner filesystem.
//!
//! `#[doc(hidden)]` + test-only env arming: never set these variables in
//! production. Nextest's process-per-test isolation makes the env gate
//! safe for parallel test binaries.

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use crate::{
    DirEntry, FileOptions, FileSystem, FilesystemError, FilesystemFile, FsFileMetadata, FsResult,
    RangeCoalescePolicy,
};
use async_trait::async_trait;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FaultMode {
    /// Fail the first delete of each unique path, then pass through.
    FirstOnly,
    /// Fail every delete.
    Always,
}

fn fault_mode() -> Option<FaultMode> {
    static MODE: OnceLock<Option<FaultMode>> = OnceLock::new();
    *MODE.get_or_init(
        || match std::env::var("PROXIMADB_TEST_FS_DELETE_FAIL_FIRST") {
            Ok(v) if v.trim() == "1" => Some(FaultMode::FirstOnly),
            Ok(v) if v.trim().eq_ignore_ascii_case("always") => Some(FaultMode::Always),
            _ => None,
        },
    )
}

fn already_failed() -> &'static Mutex<HashSet<String>> {
    static FAILED: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    FAILED.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Reset the injection state (test isolation helper — nextest runs each test
/// in its own process, so this is only needed for in-process multi-phase
/// tests).
pub fn reset_for_tests() {
    if let Ok(mut failed) = already_failed().lock() {
        failed.clear();
    }
}

#[derive(Debug)]
pub struct FaultInjectingFileSystem {
    inner: std::sync::Arc<dyn FileSystem>,
}

impl FaultInjectingFileSystem {
    pub fn new(inner: std::sync::Arc<dyn FileSystem>) -> Self {
        Self { inner }
    }

    fn delete_fault(&self, path: &str) -> Option<FilesystemError> {
        // Scope to compaction-retirement deletes: the atomic flush publish
        // (staging→final move) also deletes a `.pax` under the `__flush`
        // staging area, and faulting that fails the flush at commit instead of
        // the compaction at retirement. Metadata/WAL rotations are excluded by
        // the `.pax` filter.
        if !path.ends_with(".pax") || path.contains("__flush") || path.contains("__compact") {
            return None;
        }
        match fault_mode()? {
            FaultMode::Always => Some(FilesystemError::Network(format!(
                "fault-injected delete failure (always): {path}"
            ))),
            FaultMode::FirstOnly => {
                let mut failed = already_failed().lock().ok()?;
                if failed.insert(path.to_string()) {
                    Some(FilesystemError::Network(format!(
                        "fault-injected first delete failure: {path}"
                    )))
                } else {
                    None
                }
            }
        }
    }
}

#[async_trait]
impl FileSystem for FaultInjectingFileSystem {
    fn as_any(&self) -> &dyn std::any::Any {
        self.inner.as_any()
    }
    fn filesystem_type(&self) -> &'static str {
        self.inner.filesystem_type()
    }
    fn supports_mmap(&self) -> bool {
        self.inner.supports_mmap()
    }
    fn range_coalesce_policy(&self) -> Option<RangeCoalescePolicy> {
        self.inner.range_coalesce_policy()
    }

    async fn read(&self, path: &str) -> FsResult<Vec<u8>> {
        self.inner.read(path).await
    }
    async fn read_range(&self, path: &str, offset: u64, length: u64) -> FsResult<Vec<u8>> {
        self.inner.read_range(path, offset, length).await
    }
    async fn read_ranges(
        &self,
        path: &str,
        ranges: Vec<std::ops::Range<u64>>,
    ) -> FsResult<Vec<Vec<u8>>> {
        self.inner.read_ranges(path, ranges).await
    }
    async fn get_mmap(&self, path: &str) -> FsResult<Option<memmap2::Mmap>> {
        self.inner.get_mmap(path).await
    }

    async fn delete(&self, path: &str) -> FsResult<()> {
        if let Some(error) = self.delete_fault(path) {
            return Err(error);
        }
        self.inner.delete(path).await
    }

    // --- Pass-through: writes / metadata / listing ---

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fault_mode_unset_is_inert() {
        // In a process where the env was never set, mode is None (passthrough).
        // nextest isolates processes, so a test binary that sets the env would
        // see its own mode; this binary asserts only the unset default.
        if std::env::var_os("PROXIMADB_TEST_FS_DELETE_FAIL_FIRST").is_none() {
            assert_eq!(fault_mode(), None);
        }
    }
}
