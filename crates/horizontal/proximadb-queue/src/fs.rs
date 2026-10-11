//! Narrow filesystem abstraction the queue uses for its disk + object tiers.
//!
//! We deliberately do NOT depend on ProximaDB's `FilesystemFactory` here -
//! that lives in the main `proximadb` crate which depends on us, so taking
//! a hard dep back would be circular. Instead the queue defines this small
//! trait, ships a `LocalFs` (tokio::fs) impl, and lets the main crate
//! provide an adapter that wraps `FilesystemFactory`'s output for
//! production deployments (object stores etc.) when Phase 2D lands.
//!
//! Methods are async (the disk tier runs on tokio); paths are `&Path` so
//! callers can choose `PathBuf` ownership freely.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;

use crate::error::QueueError;

pub type Result<T> = std::result::Result<T, QueueError>;

/// Operations the queue needs from a backing filesystem.
#[async_trait]
pub trait QueueFs: Send + Sync + std::fmt::Debug {
    /// True only when this adapter implements the conditional publication contract.
    fn supports_conditional_replace(&self) -> bool {
        false
    }

    async fn create_dir_all(&self, path: &Path) -> Result<()>;

    /// Append bytes to the file, creating it if absent. NOT durable on
    /// its own - call `fsync` to durably persist.
    async fn append(&self, path: &Path, data: &[u8]) -> Result<()>;

    /// Force the file's bytes (and its containing directory entry) to
    /// stable storage. On `LocalFs` this is `tokio::fs::File::sync_all`.
    /// On object stores, this is a no-op (durability is at write time).
    async fn fsync(&self, path: &Path) -> Result<()>;

    /// Missing paths must return `QueueError::NotFound`, never empty bytes or a
    /// generic persistence error. Other failures retain their failure semantics.
    async fn read(&self, path: &Path) -> Result<Vec<u8>>;

    /// Exact byte-conditioned durable replacement. Absence differs from empty.
    /// False means mismatch/contention; errors may follow publication and must
    /// not be turned into successful acquisition. Unsupported adapters fail closed.
    /// Read-only guards compare sibling files under the same serialization boundary
    /// as target publication. No atomic multi-file replacement is implied.
    async fn compare_exchange(
        &self,
        _path: &Path,
        _expected: Option<&[u8]>,
        _replacement: &[u8],
        _guards: &[(&Path, Option<&[u8]>)],
    ) -> Result<bool> {
        Err(QueueError::Persistence(
            "backend does not support conditional publication".into(),
        ))
    }

    /// List children of `dir`, including directories. Returned paths must use
    /// the same coordinates as `dir`: absolute for local absolute roots,
    /// root-relative for an explicitly anchored archive adapter. Never strip
    /// the caller's root from an absolute directory. Order is unspecified.
    async fn list(&self, dir: &Path) -> Result<Vec<PathBuf>>;

    /// Rename `from` to `to`. On local FS this is atomic when both paths
    /// are on the same mount.
    async fn rename(&self, from: &Path, to: &Path) -> Result<()>;

    async fn delete(&self, path: &Path) -> Result<()>;

    async fn metadata(&self, path: &Path) -> Result<Metadata>;
}

#[derive(Debug, Clone, Copy)]
pub struct Metadata {
    pub size_bytes: u64,
    pub is_directory: bool,
}

/// Real-filesystem implementation. Production default.
#[derive(Debug)]
pub struct LocalFs;

impl LocalFs {
    pub fn new_arc() -> Arc<dyn QueueFs> {
        Arc::new(Self)
    }
}

#[async_trait]
impl QueueFs for LocalFs {
    fn supports_conditional_replace(&self) -> bool {
        cfg!(unix)
    }

    async fn create_dir_all(&self, path: &Path) -> Result<()> {
        tokio::fs::create_dir_all(path)
            .await
            .map_err(|e| QueueError::Persistence(format!("create_dir_all {path:?}: {e}")))
    }

    async fn append(&self, path: &Path, data: &[u8]) -> Result<()> {
        use tokio::io::AsyncWriteExt;
        let mut f = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await
            .map_err(|e| QueueError::Persistence(format!("open {path:?}: {e}")))?;
        f.write_all(data)
            .await
            .map_err(|e| QueueError::Persistence(format!("write {path:?}: {e}")))?;
        Ok(())
    }

    async fn fsync(&self, path: &Path) -> Result<()> {
        let f = tokio::fs::OpenOptions::new()
            .read(true)
            .open(path)
            .await
            .map_err(|e| QueueError::Persistence(format!("open-for-fsync {path:?}: {e}")))?;
        f.sync_all()
            .await
            .map_err(|e| QueueError::Persistence(format!("fsync {path:?}: {e}")))?;
        Ok(())
    }

    async fn read(&self, path: &Path) -> Result<Vec<u8>> {
        tokio::fs::read(path).await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                QueueError::NotFound(format!("read {path:?}: {e}"))
            } else {
                QueueError::Persistence(format!("read {path:?}: {e}"))
            }
        })
    }

    async fn compare_exchange(
        &self,
        path: &Path,
        expected: Option<&[u8]>,
        replacement: &[u8],
        guards: &[(&Path, Option<&[u8]>)],
    ) -> Result<bool> {
        let path = path.to_path_buf();
        let expected = expected.map(<[u8]>::to_vec);
        let replacement = replacement.to_vec();
        let guards: Vec<_> = guards
            .iter()
            .map(|(path, expected)| (path.to_path_buf(), expected.map(<[u8]>::to_vec)))
            .collect();
        tokio::task::spawn_blocking(move || {
            let guards: Vec<_> = guards
                .iter()
                .map(|(path, expected)| (path.as_path(), expected.as_deref()))
                .collect();
            proximadb_runtime_common::file_lock::FileLockManager::compare_exchange_file(
                &path,
                expected.as_deref(),
                &replacement,
                &guards,
            )
            .map_err(|e| {
                QueueError::Persistence(format!(
                    "conditional publication {path:?} (outcome may be indeterminate): {e}"
                ))
            })
        })
        .await
        .map_err(|e| QueueError::Persistence(format!("conditional publication task: {e}")))?
    }

    async fn list(&self, dir: &Path) -> Result<Vec<PathBuf>> {
        let mut entries = tokio::fs::read_dir(dir).await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                QueueError::NotFound(format!("read_dir {dir:?}: {e}"))
            } else {
                QueueError::Persistence(format!("read_dir {dir:?}: {e}"))
            }
        })?;
        let mut out = Vec::new();
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| QueueError::Persistence(format!("read_dir-next {dir:?}: {e}")))?
        {
            out.push(entry.path());
        }
        Ok(out)
    }

    async fn rename(&self, from: &Path, to: &Path) -> Result<()> {
        tokio::fs::rename(from, to)
            .await
            .map_err(|e| QueueError::Persistence(format!("rename {from:?} -> {to:?}: {e}")))
    }

    async fn delete(&self, path: &Path) -> Result<()> {
        tokio::fs::remove_file(path)
            .await
            .map_err(|e| QueueError::Persistence(format!("delete {path:?}: {e}")))
    }

    async fn metadata(&self, path: &Path) -> Result<Metadata> {
        let meta = tokio::fs::metadata(path)
            .await
            .map_err(|e| QueueError::Persistence(format!("metadata {path:?}: {e}")))?;
        Ok(Metadata {
            size_bytes: meta.len(),
            is_directory: meta.is_dir(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[tokio::test]
    async fn local_fs_guarded_publication_rejects_changed_sibling() {
        let fs = LocalFs;
        let dir = tempfile::tempdir().unwrap();
        let lease = dir.path().join("lease.meta");
        let offset = dir.path().join("offset.meta");
        assert!(
            fs.compare_exchange(&lease, None, b"owner", &[])
                .await
                .unwrap()
        );
        assert!(
            !fs.compare_exchange(&offset, None, b"progress", &[(&lease, Some(b"stale"))])
                .await
                .unwrap()
        );
        assert!(matches!(
            fs.read(&offset).await,
            Err(QueueError::NotFound(_))
        ));
        assert!(
            fs.compare_exchange(&offset, None, b"progress", &[(&lease, Some(b"owner"))])
                .await
                .unwrap()
        );
        assert_eq!(fs.read(&offset).await.unwrap(), b"progress");
    }

    #[cfg(unix)]
    #[test]
    fn cancelled_guarded_publication_remains_conditioned_after_blocking_queue_delay() {
        use proximadb_runtime_common::file_lock::FileLockManager;
        use std::future::Future;
        use std::task::Poll;
        use std::time::Duration;

        for takeover in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let lease = dir.path().join("lease.meta");
            let offset = dir.path().join("offset.meta");
            std::fs::write(&lease, b"owner").unwrap();
            let runtime = tokio::runtime::Builder::new_current_thread()
                .max_blocking_threads(1)
                .build()
                .unwrap();
            runtime.block_on(async {
                let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
                let (resume_tx, resume_rx) = std::sync::mpsc::sync_channel(1);
                let blocker = tokio::task::spawn_blocking(move || {
                    started_tx.send(()).unwrap();
                    resume_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                });
                started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                let guards = [(lease.as_path(), Some(b"owner".as_slice()))];
                let mut publication =
                    Box::pin(LocalFs.compare_exchange(&offset, None, b"progress", &guards));
                // Poll exactly once: the sole blocking worker is held, so the
                // complete conditional operation is queued but cannot yet run.
                std::future::poll_fn(|cx| {
                    assert!(publication.as_mut().poll(cx).is_pending());
                    Poll::Ready(())
                })
                .await;
                drop(publication);
                if takeover {
                    assert!(
                        FileLockManager::compare_exchange_file(
                            &lease,
                            Some(b"owner"),
                            b"successor",
                            &[],
                        )
                        .unwrap()
                    );
                }
                resume_tx.send(()).unwrap();
                blocker.await.unwrap();
            });
            // Dropping the runtime joins its already-started blocking tasks,
            // including the detached publication whose caller was cancelled.
            drop(runtime);
            if takeover {
                assert!(
                    !offset.exists(),
                    "cancelled stale publication must be fenced"
                );
            } else {
                assert_eq!(
                    std::fs::read(&offset).unwrap(),
                    b"progress",
                    "cancellation alone is not proof that a queued write was rolled back"
                );
            }
        }
    }

    #[tokio::test]
    async fn local_fs_round_trips_append_read_metadata_list_rename_and_delete() {
        let fs = LocalFs;
        let dir = tempfile::tempdir().unwrap();
        let partition_dir = dir.path().join("topic").join("0");
        let segment = partition_dir.join("0000000000.qseg");

        fs.create_dir_all(&partition_dir).await.unwrap();
        fs.append(&segment, b"hello").await.unwrap();
        fs.append(&segment, b" world").await.unwrap();
        fs.fsync(&segment).await.unwrap();

        assert_eq!(fs.read(&segment).await.unwrap(), b"hello world");
        assert_eq!(fs.metadata(&segment).await.unwrap().size_bytes, 11);
        assert!(!fs.metadata(&segment).await.unwrap().is_directory);
        assert!(fs.metadata(&partition_dir).await.unwrap().is_directory);

        let listed = fs.list(&partition_dir).await.unwrap();
        assert!(listed.iter().any(|path| path == &segment));

        let renamed = partition_dir.join("0000000001.qseg");
        fs.rename(&segment, &renamed).await.unwrap();
        assert_eq!(fs.read(&renamed).await.unwrap(), b"hello world");

        fs.delete(&renamed).await.unwrap();
        let error = fs.metadata(&renamed).await.unwrap_err();
        assert!(error.to_string().contains("metadata"));
    }

    #[tokio::test]
    async fn local_fs_surfaces_context_on_missing_paths() {
        let fs = LocalFs;
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("missing.qseg");

        let error = fs.read(&missing).await.unwrap_err();

        assert!(error.to_string().contains("read"));
        assert!(error.to_string().contains("missing.qseg"));
    }
}
