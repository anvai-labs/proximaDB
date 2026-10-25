//! Test support for `FileSystem` decorators, shared across crates (it is not
//! `#[cfg(test)]` so other crates' tests can use it; it is hidden from docs).

use std::sync::Mutex;

use async_trait::async_trait;

use crate::{
    DirEntry, FileOptions, FileSystem, FilesystemError, FilesystemFile, FsFileMetadata, FsResult,
};

/// Records the durability barriers it receives; everything else fails, so
/// a test can only pass through the two methods under test.
#[derive(Debug, Default)]
pub struct SyncRecorder {
    synced: Mutex<Vec<String>>,
}

impl SyncRecorder {
    /// The barriers received so far, in order (`"file <path>"` / `"data <path>"`).
    pub fn synced(&self) -> Vec<String> {
        self.synced
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

fn unsupported<T>(what: &str) -> FsResult<T> {
    Err(FilesystemError::InvalidOperation(format!(
        "SyncRecorder: {what}"
    )))
}

#[async_trait]
impl FileSystem for SyncRecorder {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn filesystem_type(&self) -> &'static str {
        "sync-recorder"
    }
    async fn sync_file(&self, path: &str) -> FsResult<()> {
        self.synced
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(format!("file {path}"));
        Ok(())
    }
    async fn sync_data(&self, path: &str) -> FsResult<()> {
        self.synced
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(format!("data {path}"));
        Ok(())
    }
    async fn read(&self, _: &str) -> FsResult<Vec<u8>> {
        unsupported("read")
    }
    async fn write(&self, _: &str, _: &[u8], _: Option<FileOptions>) -> FsResult<()> {
        unsupported("write")
    }
    async fn append(&self, _: &str, _: &[u8]) -> FsResult<()> {
        unsupported("append")
    }
    async fn delete(&self, _: &str) -> FsResult<()> {
        unsupported("delete")
    }
    async fn exists(&self, _: &str) -> FsResult<bool> {
        unsupported("exists")
    }
    async fn metadata(&self, _: &str) -> FsResult<FsFileMetadata> {
        unsupported("metadata")
    }
    async fn list(&self, _: &str) -> FsResult<Vec<DirEntry>> {
        unsupported("list")
    }
    async fn create_dir(&self, _: &str) -> FsResult<()> {
        unsupported("create_dir")
    }
    async fn create_dir_all(&self, _: &str) -> FsResult<()> {
        unsupported("create_dir_all")
    }
    async fn copy(&self, _: &str, _: &str) -> FsResult<()> {
        unsupported("copy")
    }
    async fn move_file(&self, _: &str, _: &str) -> FsResult<()> {
        unsupported("move_file")
    }
    async fn sync(&self) -> FsResult<()> {
        unsupported("sync")
    }
    async fn open_file(&self, _: &str, _: bool) -> FsResult<Box<dyn FilesystemFile>> {
        unsupported("open_file")
    }
}
