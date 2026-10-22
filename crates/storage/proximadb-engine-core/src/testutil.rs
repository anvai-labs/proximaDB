//! Shared test doubles for engine-core unit tests (compiled under `cfg(test)`
//! only). Single home of the no-op `FilesystemPort` that metadata/cache/
//! utilities tests inject when they exercise construction, serialization,
//! metrics recording, and pure calculations — never a real filesystem read.

use async_trait::async_trait;
use proximadb_storage_filesystem_types::{DirEntry, FileOptions, FileSystem, FsResult};
use std::sync::Arc;

/// Build a no-op `FilesystemPort` for tests.
///
/// `get_filesystem` returns `NotFound` (matching an empty cache / no reads),
/// and every mutation reports success without touching storage.
pub(crate) fn noop_filesystem_port() -> Arc<dyn proximadb_storage_ports::FilesystemPort> {
    Arc::new(NoopFilesystemPort)
}

struct NoopFilesystemPort;

#[async_trait]
impl proximadb_storage_ports::FilesystemPort for NoopFilesystemPort {
    fn get_filesystem(&self, _url: &str) -> FsResult<Arc<dyn FileSystem>> {
        Err(proximadb_storage_filesystem_types::FilesystemError::NotFound("test stub".to_string()))
    }
    async fn create_dir_all(&self, _url: &str) -> FsResult<()> {
        Ok(())
    }
    async fn write(&self, _url: &str, _data: &[u8], _options: Option<FileOptions>) -> FsResult<()> {
        Ok(())
    }
    async fn move_atomic(&self, _from: &str, _to: &str) -> FsResult<()> {
        Ok(())
    }
    async fn delete(&self, _url: &str) -> FsResult<()> {
        Ok(())
    }
    async fn read(&self, _url: &str) -> FsResult<Vec<u8>> {
        Ok(Vec::new())
    }
    async fn list(&self, _url: &str) -> FsResult<Vec<DirEntry>> {
        Ok(Vec::new())
    }
}
