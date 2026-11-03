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

use crate::{FileSystem, FilesystemError, FsResult};

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

/// `PROXIMADB_TEST_FS_READ_FAIL_AFTER_FIRST=1`: the first read (`read` or
/// `read_range`) of each unique `.pax` path passes, every later read of that
/// path fails with a transient `FilesystemError::Network`. Reproduces "the
/// compaction read its records, then a later probe read of the same input hit
/// a transient I/O error" (TD-READFID-2 review). Keyed with `file://`
/// stripped so URL and resolved-path forms of one file share a count.
///
/// Read live (no OnceLock): under `cargo test --lib` every test shares one
/// process, so a cached first read would pin the gate before the arming test
/// runs — and would stay armed after it clears the env.
fn read_fault_armed() -> bool {
    std::env::var("PROXIMADB_TEST_FS_READ_FAIL_AFTER_FIRST")
        .map(|v| v.trim() == "1")
        .unwrap_or(false)
}

/// True when the read-fault env gate is set, so the factory installs this
/// wrapper.
pub fn read_fault_requested() -> bool {
    std::env::var_os("PROXIMADB_TEST_FS_READ_FAIL_AFTER_FIRST").is_some()
}

/// `PROXIMADB_TEST_FS_READ_FAIL_SCOPE`: optional directory prefix. When set,
/// only `.pax` paths under it can be faulted (and only they are tracked) —
/// the arming test points this at its own temp dir so that, in the
/// shared-process `cargo test --lib` world, the wrapper the factory installs
/// everywhere stays inert for every concurrent test's files. Unset (the
/// default): any `.pax` path can be faulted, exactly as under nextest.
fn read_fault_scope_covers(key: &str) -> bool {
    let Some(scope) = std::env::var_os("PROXIMADB_TEST_FS_READ_FAIL_SCOPE") else {
        return true;
    };
    let scope = scope.to_string_lossy();
    let scope = scope.strip_prefix("file://").unwrap_or(&scope);
    let scope = scope.trim_end_matches('/');
    key == scope || key.starts_with(&format!("{scope}/"))
}

fn already_read() -> &'static Mutex<HashSet<String>> {
    static READ: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    READ.get_or_init(|| Mutex::new(HashSet::new()))
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
    if let Ok(mut read) = already_read().lock() {
        read.clear();
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

    fn read_fault(&self, path: &str) -> Option<FilesystemError> {
        if !read_fault_armed()
            || !path.ends_with(".pax")
            || path.contains("__flush")
            || path.contains("__compact")
        {
            return None;
        }
        let key = path.strip_prefix("file://").unwrap_or(path).to_string();
        if !read_fault_scope_covers(&key) {
            return None;
        }
        let mut seen = already_read().lock().ok()?;
        if seen.insert(key) {
            None
        } else {
            Some(FilesystemError::Network(format!(
                "fault-injected transient read failure: {path}"
            )))
        }
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

crate::forward_filesystem_impl! {
    impl FileSystem for FaultInjectingFileSystem => inner;
    forward [
        as_any, filesystem_type, supports_mmap, range_coalesce_policy, read_ranges,
        get_mmap, write, supports_bounded_local_file_write, write_local_file,
        write_if_absent, append, supports_append, supports_conditional_replace, exists,
        metadata, list, create_dir, create_dir_all, copy, move_file, sync, sync_file,
        sync_data, open_file,
    ];
    {

        async fn read(&self, path: &str) -> FsResult<Vec<u8>> {
            if let Some(error) = self.read_fault(path) {
                return Err(error);
            }
            self.inner.read(path).await
        }
        async fn read_range(&self, path: &str, offset: u64, length: u64) -> FsResult<Vec<u8>> {
            if let Some(error) = self.read_fault(path) {
                return Err(error);
            }
            self.inner.read_range(path, offset, length).await
        }

        async fn delete(&self, path: &str) -> FsResult<()> {
            if let Some(error) = self.delete_fault(path) {
                return Err(error);
            }
            self.inner.delete(path).await
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
