/*
 * Copyright 2026 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! Bridge between the queue's narrow `QueueFs` trait and the main
//! crate's `FilesystemFactory`. Lives here (not in `proximadb-queue`)
//! because the factory pulls in cache + transaction orchestration that
//! the queue can't depend on without going circular.
//!
//! Wiring:
//!   1. `database.rs` already constructs a `FilesystemFactory` for the
//!      storage engine layer (it knows how to resolve `file://`,
//!      `adls://`, `s3://`, `gcs://`, `hdfs://`).
//!   2. At queue-init time, `database.rs` constructs a
//!      [`FactoryQueueFs`] anchored at the queue root URL.
//!   3. `QueueClient::open_with_fs(config, Some(adapter))` injects it.
//!   4. Queue's disk tier + object_tier uploader call through the
//!      adapter, which translates `&Path` → URL by joining with the
//!      configured root URL prefix.
//!
//! ## Path-to-URL translation
//!
//! The queue uses `PathBuf` internally (inherited from its
//! `LocalFs`-only origins). Explicitly relative paths are interpreted under
//! the configured root. For a local queue, `QueueClient` instead supplies the
//! absolute path decoded from its `file://` root; those paths are accepted only
//! when they are lexically confined to that exact root. So if
//! `root_url = "adls://acct.dfs.core.windows.net/queue"` and the queue
//! asks for `path = "embed-ingest/0/0000000000.qseg"`, the adapter calls
//! `factory.get_filesystem("adls://.../queue/embed-ingest/0/0000000000.qseg")`.
//! Bare relative roots are refused: adding `file://` would bypass a backend's
//! configured root directory and may abandon its history. Explicit relative
//! file URLs retain their original working-directory semantics.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use proximadb_queue::error::QueueError;
use proximadb_queue::fs::{Metadata, QueueFs, Result as QueueResult};

use crate::storage::persistence::filesystem::FilesystemFactory;
use proximadb_storage_filesystem_types::FilesystemError;

/// `QueueFs` impl backed by `FilesystemFactory`. Resolves URLs for
/// any scheme the factory knows (`file`, `adls`, `s3`, `gcs`, `hdfs`).
#[derive(Debug)]
pub struct FactoryQueueFs {
    factory: Arc<FilesystemFactory>,
    /// Root URL the queue's relative paths are joined under. Includes
    /// scheme + authority + base path, no trailing slash.
    root_url: String,
    /// Path coordinate used by QueueClient for local roots. This lets the
    /// adapter distinguish an already-rooted queue path from an explicit
    /// relative adapter call without granting absolute-path access elsewhere.
    local_root: Option<PathBuf>,
}

impl FactoryQueueFs {
    fn require_root_capabilities(
        root_url: &str,
        filesystem_type: &str,
        supports_append: bool,
        supports_conditional_replace: bool,
    ) -> QueueResult<()> {
        if !supports_append {
            return Err(QueueError::Persistence(format!(
                "queue root {root_url} uses backend {filesystem_type} without durable append support"
            )));
        }
        if !supports_conditional_replace {
            return Err(QueueError::Persistence(format!(
                "queue root {root_url} uses backend {filesystem_type} without conditional ownership publication support"
            )));
        }
        Ok(())
    }

    // Returns `Arc<dyn QueueFs>` directly because every caller stores the
    // adapter behind a trait object; exposing the concrete type would force
    // every call site to add a redundant `.as_queue_fs()` cast.
    #[allow(clippy::new_ret_no_self)]
    pub fn new(
        factory: Arc<FilesystemFactory>,
        root_url: impl Into<String>,
    ) -> QueueResult<Arc<dyn QueueFs>> {
        Ok(Arc::new(Self::build(factory, root_url)?))
    }

    fn build(factory: Arc<FilesystemFactory>, root_url: impl Into<String>) -> QueueResult<Self> {
        let (root_url, local_root) = Self::normalize_root(root_url.into())?;
        let fs = factory.get_filesystem(&root_url).map_err(Self::map_err)?;
        Self::require_root_capabilities(
            &root_url,
            fs.filesystem_type(),
            fs.supports_append(),
            fs.supports_conditional_replace(),
        )?;
        Ok(Self {
            factory,
            root_url,
            local_root,
        })
    }

    fn normalize_root(mut root_url: String) -> QueueResult<(String, Option<PathBuf>)> {
        if root_url.is_empty() {
            return Err(QueueError::Persistence(
                "queue filesystem root cannot be empty".into(),
            ));
        }
        if !root_url.contains("://") {
            if !Path::new(&root_url).is_absolute() {
                return Err(QueueError::Persistence(format!(
                    "bare relative queue root {root_url:?} is backend-dependent; locate existing history using the original backend root_dir and working directory, then perform explicit offline migration before selecting an absolute root or file:// URL; merely adding a scheme is not a migration"
                )));
            }
            root_url = format!("file://{root_url}");
        }
        // Retain the third slash in the filesystem root `file:///`.
        while root_url.ends_with('/') && !root_url.ends_with(":///") {
            root_url.pop();
        }
        let local_root = root_url.strip_prefix("file://").map(PathBuf::from);
        if let Some(path) = &local_root {
            Self::reject_parent_components(path, "queue filesystem root")?;
            if path.as_os_str().is_empty() {
                return Err(QueueError::Persistence(
                    "queue filesystem root cannot be empty".into(),
                ));
            }
        }
        Ok((root_url, local_root))
    }

    fn reject_parent_components(path: &Path, context: &str) -> QueueResult<()> {
        if path
            .components()
            .any(|component| component == std::path::Component::ParentDir)
        {
            return Err(QueueError::Persistence(format!(
                "{context} contains parent traversal: {path:?}"
            )));
        }
        Ok(())
    }

    // Lexical normalization only: never resolve symlinks or discard ParentDir.
    fn without_current_dir(path: &Path) -> PathBuf {
        path.components()
            .filter(|component| *component != std::path::Component::CurDir)
            .collect()
    }

    /// Return the root-relative suffix plus whether the input used the queue's
    /// root-qualified coordinate system.
    fn relative_path(&self, path: &Path) -> QueueResult<(PathBuf, bool)> {
        Self::reject_parent_components(path, "queue filesystem path")?;
        if let Some(root) = &self.local_root
            && let Ok(relative) = path.strip_prefix(root)
        {
            return Ok((Self::without_current_dir(relative), true));
        }
        if path.is_absolute() {
            return Err(QueueError::Persistence(format!(
                "absolute queue filesystem path is outside configured root {}: {path:?}",
                self.root_url
            )));
        }
        Ok((Self::without_current_dir(path), false))
    }

    fn root_child_prefix(&self) -> String {
        if self.root_url.ends_with('/') {
            self.root_url.clone()
        } else {
            format!("{}/", self.root_url)
        }
    }

    /// Reproduce the old adapter's double-prefix mapping without sending it
    /// through `url_for`, which intentionally uses the corrected coordinates.
    fn legacy_directory_url(&self, path: &Path) -> QueueResult<Option<String>> {
        let Some(root) = &self.local_root else {
            return Ok(None);
        };
        let (relative, root_qualified) = self.relative_path(path)?;
        if !root_qualified {
            // Explicit root-relative operations did not change their mapping.
            return Ok(None);
        }
        let path_text = path
            .to_str()
            .ok_or_else(|| QueueError::Persistence("queue filesystem path is not UTF-8".into()))?;
        let suffix = path_text.trim_start_matches('/');
        let legacy = root.join(suffix);
        let canonical = root.join(relative);
        // `/` and dot-only relative roots never changed physical layouts.
        let significant =
            |component: &std::path::Component<'_>| *component != std::path::Component::CurDir;
        if canonical
            .components()
            .filter(significant)
            .eq(legacy.components().filter(significant))
        {
            return Ok(None);
        }
        Ok(Some(format!("{}{suffix}", self.root_child_prefix())))
    }

    /// QueueClient calls create_dir_all before constructing any topic or writer.
    /// Its root may be a descendant of this adapter's root, so check the actual
    /// requested directory. This is admission, not fencing of old writers.
    async fn reject_legacy_layout(&self, path: &Path) -> QueueResult<()> {
        let Some(legacy_url) = self.legacy_directory_url(path)? else {
            return Ok(());
        };
        let fs = self
            .factory
            .get_filesystem(&legacy_url)
            .map_err(Self::map_err)?;
        match fs.list(&legacy_url).await {
            Ok(entries) if entries.is_empty() => Ok(()),
            Ok(_) => Err(QueueError::Persistence(format!(
                "legacy queue layout at {legacy_url}; refusing to initialize queue directory {path:?}: stop all queue writers, back up both layouts, and perform explicit offline migration before retrying; do not overwrite or discard either history"
            ))),
            Err(FilesystemError::NotFound(_)) => Ok(()),
            Err(FilesystemError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                Ok(())
            }
            Err(error) => Err(QueueError::Persistence(format!(
                "cannot inspect legacy queue layout at {legacy_url}; refusing queue startup: {error}"
            ))),
        }
    }

    /// Translate a queue-supplied `&Path` to a URL the factory accepts.
    fn url_for(&self, path: &Path) -> QueueResult<String> {
        let (relative, _) = self.relative_path(path)?;
        let suffix = relative.to_str().ok_or_else(|| {
            QueueError::Persistence(format!("queue filesystem path is not UTF-8: {path:?}"))
        })?;
        if suffix.is_empty() {
            Ok(self.root_url.clone())
        } else {
            Ok(format!("{}{suffix}", self.root_child_prefix()))
        }
    }

    fn relative_from_listed_url(&self, entry_url: &str) -> QueueResult<PathBuf> {
        let outside_root = || {
            QueueError::Persistence(format!(
                "queue filesystem LIST returned entry outside configured root {}: {entry_url}",
                self.root_url
            ))
        };
        let relative = if let Some(root) = &self.local_root {
            let entry = Path::new(
                entry_url
                    .strip_prefix("file://")
                    .ok_or_else(&outside_root)?,
            );
            Self::reject_parent_components(entry, "queue filesystem LIST entry")?;
            // Some local backends preserve an additional `./` in LIST URLs.
            // Compare path components, not raw URL spelling, without relaxing
            // traversal or root confinement.
            let entry = Self::without_current_dir(entry);
            let root = Self::without_current_dir(root);
            entry
                .strip_prefix(&root)
                .map_err(|_| outside_root())?
                .to_path_buf()
        } else {
            let relative = if entry_url == self.root_url {
                ""
            } else {
                let prefix = self.root_child_prefix();
                entry_url.strip_prefix(&prefix).ok_or_else(outside_root)?
            };
            PathBuf::from(relative)
        };
        if relative.is_absolute() {
            return Err(QueueError::Persistence(format!(
                "queue filesystem LIST returned absolute child suffix: {entry_url}"
            )));
        }
        Self::reject_parent_components(&relative, "queue filesystem LIST entry")?;
        Ok(relative)
    }

    fn listed_path(&self, requested_dir: &Path, entry_url: &str) -> QueueResult<PathBuf> {
        let (requested_relative, _) = self.relative_path(requested_dir)?;
        let relative = self.relative_from_listed_url(entry_url)?;
        if relative.parent() != Some(requested_relative.as_path()) {
            return Err(QueueError::Persistence(format!(
                "queue filesystem LIST returned non-child entry for {requested_dir:?}: {entry_url}"
            )));
        }
        let name = relative.file_name().ok_or_else(|| {
            QueueError::Persistence(format!(
                "queue filesystem LIST returned entry without a child name: {entry_url}"
            ))
        })?;
        // Preserve the caller's coordinates, including a leading `./`. Queue
        // identity checks strip this exact parent from each listed child.
        Ok(requested_dir.join(name))
    }

    fn map_err(e: impl std::fmt::Display) -> QueueError {
        QueueError::Persistence(e.to_string())
    }
}

#[async_trait]
impl QueueFs for FactoryQueueFs {
    fn supports_conditional_replace(&self) -> bool {
        // Construction validates the root adapter before it can accept data.
        true
    }

    async fn create_dir_all(&self, path: &Path) -> QueueResult<()> {
        self.reject_legacy_layout(path).await?;
        let url = self.url_for(path)?;
        let fs = self.factory.get_filesystem(&url).map_err(Self::map_err)?;
        fs.create_dir_all(&url).await.map_err(Self::map_err)
    }

    async fn append(&self, path: &Path, data: &[u8]) -> QueueResult<()> {
        let url = self.url_for(path)?;
        let fs = self.factory.get_filesystem(&url).map_err(Self::map_err)?;
        fs.append(&url, data).await.map_err(Self::map_err)
    }

    async fn fsync(&self, path: &Path) -> QueueResult<()> {
        let url = self.url_for(path)?;
        let fs = self.factory.get_filesystem(&url).map_err(Self::map_err)?;
        // Object-store backends no-op sync_file (PUTs are durable on
        // success); local filesystem invokes File::sync_all.
        fs.sync_file(&url).await.map_err(Self::map_err)
    }

    async fn read(&self, path: &Path) -> QueueResult<Vec<u8>> {
        let url = self.url_for(path)?;
        let fs = self.factory.get_filesystem(&url).map_err(Self::map_err)?;
        fs.read(&url).await.map_err(|error| match error {
            FilesystemError::NotFound(_) => QueueError::NotFound(format!("read {url}: {error}")),
            FilesystemError::Io(ref e) if e.kind() == std::io::ErrorKind::NotFound => {
                QueueError::NotFound(format!("read {url}: {error}"))
            }
            other => Self::map_err(other),
        })
    }

    async fn compare_exchange(
        &self,
        path: &Path,
        expected: Option<&[u8]>,
        replacement: &[u8],
        guards: &[(&Path, Option<&[u8]>)],
    ) -> QueueResult<bool> {
        let url = self.url_for(path)?;
        let fs = self.factory.get_filesystem(&url).map_err(Self::map_err)?;
        let guard_urls: Vec<_> = guards
            .iter()
            .map(|(path, expected)| self.url_for(path).map(|url| (url, *expected)))
            .collect::<QueueResult<Vec<_>>>()?;
        let guards: Vec<_> = guard_urls
            .iter()
            .map(|(url, expected)| (url.as_str(), *expected))
            .collect();
        fs.compare_exchange(&url, expected, replacement, &guards)
            .await
            .map_err(Self::map_err)
    }

    async fn list(&self, dir: &Path) -> QueueResult<Vec<PathBuf>> {
        let url = self.url_for(dir)?;
        let fs = self.factory.get_filesystem(&url).map_err(Self::map_err)?;
        let entries = fs.list(&url).await.map_err(|error| match error {
            FilesystemError::NotFound(_) => QueueError::NotFound(format!("list {url}: {error}")),
            FilesystemError::Io(ref e) if e.kind() == std::io::ErrorKind::NotFound => {
                QueueError::NotFound(format!("list {url}: {error}"))
            }
            other => Self::map_err(other),
        })?;
        entries
            .into_iter()
            .map(|entry| self.listed_path(dir, &entry.url))
            .collect()
    }

    async fn rename(&self, from: &Path, to: &Path) -> QueueResult<()> {
        let from_url = self.url_for(from)?;
        let to_url = self.url_for(to)?;
        let fs = self
            .factory
            .get_filesystem(&from_url)
            .map_err(Self::map_err)?;
        fs.move_file(&from_url, &to_url)
            .await
            .map_err(Self::map_err)
    }

    async fn delete(&self, path: &Path) -> QueueResult<()> {
        let url = self.url_for(path)?;
        let fs = self.factory.get_filesystem(&url).map_err(Self::map_err)?;
        fs.delete(&url).await.map_err(Self::map_err)
    }

    async fn metadata(&self, path: &Path) -> QueueResult<Metadata> {
        let url = self.url_for(path)?;
        let fs = self.factory.get_filesystem(&url).map_err(Self::map_err)?;
        let m = fs.metadata(&url).await.map_err(Self::map_err)?;
        Ok(Metadata {
            size_bytes: m.size,
            is_directory: m.is_directory,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_queue::{Message, QueueClient, QueueConfig, TopicConfig};
    use std::collections::HashMap;
    use std::time::Duration;

    #[cfg(unix)]
    #[tokio::test]
    async fn factory_local_conditional_publication_round_trips_and_preserves_missing() {
        let dir = tempfile::tempdir().unwrap();
        let factory = Arc::new(FilesystemFactory::create_default().await.unwrap());
        let adapter =
            FactoryQueueFs::new(factory, format!("file://{}", dir.path().display())).unwrap();
        let path = Path::new("lease.meta");
        assert!(adapter.supports_conditional_replace());
        assert!(matches!(
            adapter.read(path).await,
            Err(QueueError::NotFound(_))
        ));
        assert!(
            adapter
                .compare_exchange(path, None, b"first", &[])
                .await
                .unwrap()
        );
        assert!(
            !adapter
                .compare_exchange(path, None, b"bad", &[])
                .await
                .unwrap()
        );
        assert!(
            adapter
                .compare_exchange(path, Some(b"first"), b"second", &[])
                .await
                .unwrap()
        );
        assert_eq!(adapter.read(path).await.unwrap(), b"second");
        assert_eq!(std::fs::read(dir.path().join(path)).unwrap(), b"second");
        let offset = Path::new("offset.meta");
        assert!(
            !adapter
                .compare_exchange(offset, None, b"progress", &[(path, Some(b"first"))])
                .await
                .unwrap()
        );
        assert!(
            adapter
                .compare_exchange(offset, None, b"progress", &[(path, Some(b"second"))])
                .await
                .unwrap()
        );
        assert_eq!(adapter.read(offset).await.unwrap(), b"progress");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn actual_adapter_preserves_relative_and_root_qualified_coordinates() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("queue");
        let root_url = format!("file://{}", root.display());
        let factory = Arc::new(
            FilesystemFactory::create_default()
                .await
                .expect("filesystem factory"),
        );
        let adapter =
            FactoryQueueFs::build(factory.clone(), root_url.clone()).expect("queue adapter");

        assert_eq!(
            adapter
                .url_for(Path::new("embed-ingest/0/0000000000.qseg"))
                .expect("relative mapping"),
            format!("{root_url}/embed-ingest/0/0000000000.qseg")
        );
        assert_eq!(
            adapter
                .url_for(&root.join("embed-ingest/0/0000000000.qseg"))
                .expect("root-qualified mapping"),
            format!("{root_url}/embed-ingest/0/0000000000.qseg")
        );
        assert_eq!(adapter.url_for(&root).expect("root mapping"), root_url);
        let filesystem_root =
            FactoryQueueFs::build(factory, "file:///").expect("filesystem-root adapter");
        assert_eq!(
            filesystem_root
                .url_for(Path::new("/tmp/queue-entry"))
                .expect("filesystem-root mapping"),
            "file:///tmp/queue-entry"
        );
        assert_eq!(
            filesystem_root
                .listed_path(Path::new("/tmp"), "file:///tmp/queue-entry")
                .expect("filesystem-root LIST mapping"),
            PathBuf::from("/tmp/queue-entry")
        );

        let traversal = adapter
            .url_for(Path::new("embed-ingest/../foreign"))
            .expect_err("parent traversal must fail closed");
        assert!(traversal.to_string().contains("parent traversal"));
        let outside = adapter
            .url_for(&dir.path().join("foreign"))
            .expect_err("foreign absolute path must fail closed");
        assert!(outside.to_string().contains("outside configured root"));
        let foreign_list = adapter
            .listed_path(&root, "file:///foreign/entry")
            .expect_err("foreign LIST result must fail closed");
        assert!(foreign_list.to_string().contains("outside configured root"));
        let wrong_directory = adapter
            .listed_path(&root.join("relative"), &format!("{root_url}/other/entry"))
            .expect_err("LIST result outside requested directory must fail closed");
        assert!(wrong_directory.to_string().contains("non-child entry"));

        adapter
            .create_dir_all(Path::new("relative"))
            .await
            .expect("create relative directory");
        adapter
            .append(Path::new("relative/child"), b"child")
            .await
            .expect("write relative child");
        assert_eq!(
            adapter
                .list(Path::new("relative"))
                .await
                .expect("relative list"),
            vec![PathBuf::from("relative/child")]
        );
        assert_eq!(
            adapter
                .list(Path::new("./relative"))
                .await
                .expect("dotted relative list"),
            vec![PathBuf::from("./relative/child")]
        );
        let listed_traversal = adapter
            .listed_path(&root, &format!("{root_url}/./../foreign/entry"))
            .expect_err("component normalization must not admit parent traversal");
        assert!(listed_traversal.to_string().contains("parent traversal"));
        assert_eq!(
            adapter
                .list(&root.join("relative"))
                .await
                .expect("absolute list"),
            vec![root.join("relative/child")]
        );
    }

    #[test]
    fn root_url_normalization_strips_trailing_slashes() {
        assert_eq!(
            FactoryQueueFs::normalize_root("adls://acct.dfs.core.windows.net/queue/".to_string())
                .expect("normalize")
                .0,
            "adls://acct.dfs.core.windows.net/queue"
        );
        assert_eq!(
            FactoryQueueFs::normalize_root("s3://bucket/queue///".to_string())
                .expect("normalize")
                .0,
            "s3://bucket/queue"
        );
        assert_eq!(
            FactoryQueueFs::normalize_root("file:///".to_string())
                .expect("normalize filesystem root")
                .0,
            "file:///"
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn queue_client_factory_fs_preserves_identity_group_progress_and_restart() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("queue");
        let root_url = format!("file://{}", root.display());
        let factory = Arc::new(
            FilesystemFactory::create_default()
                .await
                .expect("filesystem factory"),
        );
        let mut topics = HashMap::new();
        topics.insert(
            "events".to_string(),
            TopicConfig {
                partition_count: 1,
                lease_duration: Duration::from_secs(30),
                ..Default::default()
            },
        );
        let config = QueueConfig {
            root: root_url.clone(),
            topics,
            ..Default::default()
        };
        let adapter = FactoryQueueFs::new(factory.clone(), root_url.clone())
            .expect("queue filesystem adapter");
        let client = QueueClient::open_with_fs(config.clone(), Some(adapter.clone()))
            .await
            .expect("queue open through factory adapter");
        let first = client
            .producer()
            .send(Message::new("events", "tenant-a", b"first".to_vec()))
            .await
            .expect("send first");
        client
            .producer()
            .send(Message::new("events", "tenant-a", b"second".to_vec()))
            .await
            .expect("send second");

        assert!(root.join("events/0/0000000000.qseg").is_file());
        let duplicated_root = root.join(
            root.strip_prefix(Path::new("/"))
                .expect("temporary root is absolute"),
        );
        assert!(
            !duplicated_root.exists(),
            "adapter must not prefix an already-rooted queue path twice"
        );

        let consumer = client.consumer("podA");
        consumer
            .subscribe("events", &[0])
            .await
            .expect("exact group spelling accepted");
        let alias = client.consumer("poda");
        let alias_error = alias
            .subscribe("events", &[0])
            .await
            .expect_err("case alias must be rejected");
        assert!(
            alias_error
                .to_string()
                .contains("aliases existing identity")
        );

        let batch = consumer
            .poll(1, Duration::ZERO)
            .await
            .expect("poll first message");
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].message_id, first.message_id);
        consumer
            .ack(&[batch[0].message_id.clone()])
            .await
            .expect("ack first message");
        let groups = proximadb_queue::offset_store::read_all_groups(&adapter, &root, "events", 0)
            .await
            .expect("discover acknowledged group");
        assert_eq!(groups, vec![("podA".to_string(), Some(first.offset))]);

        consumer.shutdown().await.expect("consumer shutdown");
        drop(alias);
        client.shutdown().await.expect("queue shutdown");
        drop(client);

        let restarted_adapter =
            FactoryQueueFs::new(factory, root_url).expect("restart queue filesystem adapter");
        let restarted = QueueClient::open_with_fs(config, Some(restarted_adapter.clone()))
            .await
            .expect("restart queue through factory adapter");
        let restarted_consumer = restarted.consumer("podA");
        restarted_consumer
            .subscribe("events", &[0])
            .await
            .expect("subscribe after restart");
        let resumed = restarted_consumer
            .poll(1, Duration::ZERO)
            .await
            .expect("poll after restart");
        assert_eq!(resumed.len(), 1);
        assert_eq!(resumed[0].payload, b"second");
        let groups =
            proximadb_queue::offset_store::read_all_groups(&restarted_adapter, &root, "events", 0)
                .await
                .expect("discover progress after restart");
        assert_eq!(groups, vec![("podA".to_string(), Some(first.offset))]);
        restarted_consumer
            .shutdown()
            .await
            .expect("restarted consumer shutdown");
        restarted
            .shutdown()
            .await
            .expect("restarted queue shutdown");
    }

    #[test]
    fn constructor_capability_check_rejects_non_append_backend() {
        let err =
            FactoryQueueFs::require_root_capabilities("s3://bucket/queue", "s3", false, false)
                .expect_err("object-store queue roots must fail before accepting data");
        assert!(err.to_string().contains("without durable append support"));
        FactoryQueueFs::require_root_capabilities("file:///queue", "local", true, true).unwrap();
        for backend in ["hdfs", "encrypted-local"] {
            let err =
                FactoryQueueFs::require_root_capabilities("test://queue", backend, true, false)
                    .unwrap_err();
            assert!(
                err.to_string()
                    .contains("without conditional ownership publication support")
            );
        }
    }

    #[cfg(unix)]
    async fn assert_legacy_root_rejected(mixed: bool, progress_only: bool, relative: bool) {
        let dir = if relative {
            tempfile::tempdir_in(".").expect("relative tempdir")
        } else {
            tempfile::tempdir().expect("tempdir")
        };
        let root = if relative {
            // TempDir may return an absolute path even for tempdir_in(".").
            PathBuf::from(".")
                .join(dir.path().file_name().expect("fixture name"))
                .join("queue")
        } else {
            dir.path().join("queue")
        };
        assert_eq!(root.is_relative(), relative);
        // Reproduce the old adapter's literal URL concatenation, independently
        // of the new mapper. The normal frame encoder still writes the fixture.
        let legacy_root = PathBuf::from(format!(
            "{}/{}",
            root.display(),
            root.to_str()
                .expect("UTF-8 fixture")
                .trim_start_matches('/')
        ));
        let config_for = |path: &Path| QueueConfig {
            root: format!("file://{}", path.display()),
            topics: HashMap::from([(
                "events".into(),
                TopicConfig {
                    partition_count: 1,
                    ..Default::default()
                },
            )]),
            ..Default::default()
        };
        let legacy_file = if progress_only {
            let file = legacy_root.join("events/0/podA/offset.meta");
            std::fs::create_dir_all(file.parent().expect("parent")).expect("legacy group");
            std::fs::write(&file, br#"{"group":"podA","committed_offset":42}"#)
                .expect("legacy progress");
            file
        } else {
            let old = QueueClient::open(config_for(&legacy_root))
                .await
                .expect("legacy fixture");
            old.producer()
                .send(Message::new(
                    "events",
                    "tenant-a",
                    b"legacy-unacked".to_vec(),
                ))
                .await
                .expect("persist legacy frame");
            old.shutdown().await.expect("stop fixture queue");
            drop(old);
            legacy_root.join("events/0/0000000000.qseg")
        };
        let legacy_bytes = std::fs::read(&legacy_file).expect("legacy bytes");
        let canonical_file = root.join("events/0/0000000000.qseg");
        let canonical_bytes = if mixed {
            let current = QueueClient::open(config_for(&root))
                .await
                .expect("canonical fixture");
            current
                .producer()
                .send(Message::new(
                    "events",
                    "tenant-a",
                    b"canonical-unacked".to_vec(),
                ))
                .await
                .expect("persist canonical frame");
            current.shutdown().await.expect("stop canonical queue");
            drop(current);
            Some(std::fs::read(&canonical_file).expect("canonical bytes"))
        } else {
            None
        };
        let factory = Arc::new(FilesystemFactory::create_default().await.expect("factory"));
        let adapter =
            FactoryQueueFs::new(factory, format!("file://{}/", root.display())).expect("adapter");
        let error = match QueueClient::open_with_fs(config_for(&root), Some(adapter)).await {
            Err(error) => error,
            Ok(queue) => {
                queue
                    .shutdown()
                    .await
                    .expect("stop unexpectedly opened queue");
                panic!("upgrade must reject legacy queue layout before creating canonical state");
            }
        };
        assert!(error.to_string().contains("legacy queue layout"), "{error}");
        assert!(error.to_string().contains("migration"), "{error}");
        assert_eq!(
            std::fs::read(legacy_file).expect("preserved legacy bytes"),
            legacy_bytes
        );
        match canonical_bytes {
            Some(bytes) => assert_eq!(
                std::fs::read(canonical_file).expect("preserved canonical bytes"),
                bytes
            ),
            None => assert!(
                !root.join("events").exists(),
                "no fresh topic state on rejection"
            ),
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn upgrade_rejects_legacy_queue_frames_before_initializing() {
        assert_legacy_root_rejected(false, false, false).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn upgrade_rejects_mixed_queue_layouts_without_overwriting_either() {
        assert_legacy_root_rejected(true, false, false).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn upgrade_rejects_legacy_progress_after_frames_were_reaped() {
        assert_legacy_root_rejected(false, true, false).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn upgrade_rejects_relative_legacy_queue_root() {
        assert_legacy_root_rejected(false, false, true).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn upgrade_admission_allows_empty_legacy_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("queue");
        let legacy = root.join(root.strip_prefix("/").expect("absolute fixture"));
        std::fs::create_dir_all(&legacy).expect("empty legacy directory");
        let factory = Arc::new(FilesystemFactory::create_default().await.expect("factory"));
        // Bare roots use the same admission check as file URLs.
        let adapter =
            FactoryQueueFs::new(factory, root.to_str().expect("UTF-8 fixture")).expect("adapter");
        let queue = QueueClient::open_with_fs(
            QueueConfig {
                root: root.display().to_string(),
                topics: HashMap::from([(
                    "events".into(),
                    TopicConfig {
                        partition_count: 1,
                        ..Default::default()
                    },
                )]),
                ..Default::default()
            },
            Some(adapter),
        )
        .await
        .expect("empty legacy directory is safe");
        queue.shutdown().await.expect("shutdown");
        assert!(root.join("events/0/0000000000.qseg").is_file());
        assert_eq!(
            std::fs::read_dir(legacy)
                .expect("legacy directory preserved")
                .count(),
            0
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn upgrade_admission_rejects_invalid_and_unknown_legacy_state() {
        let dir = tempfile::tempdir().expect("tempdir");
        let factory = Arc::new(FilesystemFactory::create_default().await.expect("factory"));
        for name in ["non-directory", "unknown-topic"] {
            let root = dir.path().join(name);
            let legacy = root.join(root.strip_prefix("/").expect("absolute fixture"));
            std::fs::create_dir_all(legacy.parent().expect("legacy parent")).expect("parent");
            if name == "non-directory" {
                std::fs::write(&legacy, b"not a directory").expect("invalid legacy root");
            } else {
                // Unknown/dynamically created topics must not be missed just
                // because they are absent from the new queue configuration.
                std::fs::create_dir_all(legacy.join("unconfigured-topic")).expect("legacy topic");
            }
            let adapter =
                FactoryQueueFs::new(factory.clone(), format!("file://{}", root.display()))
                    .expect("adapter");
            let error = adapter
                .create_dir_all(&root)
                .await
                .expect_err("must fail before creating state");
            assert!(error.to_string().contains("legacy queue layout"), "{error}");
            if name == "non-directory" {
                assert!(error.to_string().contains("cannot inspect"), "{error}");
                assert_eq!(
                    std::fs::read(&legacy).expect("preserved invalid root"),
                    b"not a directory"
                );
            } else {
                assert!(legacy.join("unconfigured-topic").is_dir());
            }
            assert!(!root.join("events").exists());
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn upgrade_mapping_skips_only_unchanged_layouts() {
        let factory = Arc::new(FilesystemFactory::create_default().await.expect("factory"));
        for root in ["file:///", "/", "file://./", "file://./."] {
            let adapter = FactoryQueueFs::build(factory.clone(), root).expect("adapter");
            assert_eq!(
                adapter
                    .legacy_directory_url(adapter.local_root.as_deref().expect("local root"))
                    .expect("mapping"),
                None,
                "{root}"
            );
        }
        for (root, expected) in [
            ("file:///var/lib/q/", "file:///var/lib/q/var/lib/q"),
            ("/var/./lib/q", "file:///var/./lib/q/var/./lib/q"),
            ("file://./queue", "file://./queue/./queue"),
            ("file://queue", "file://queue/queue"),
        ] {
            let adapter = FactoryQueueFs::build(factory.clone(), root).expect("adapter");
            assert_eq!(
                adapter
                    .legacy_directory_url(adapter.local_root.as_deref().expect("local root"))
                    .expect("mapping")
                    .as_deref(),
                Some(expected)
            );
        }
        for root in ["queue", "./queue", ".", "./."] {
            let error = FactoryQueueFs::build(factory.clone(), root)
                .expect_err("bare relative roots can have a different backend anchor");
            assert!(error.to_string().contains("migration"), "{error}");
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn upgrade_rejects_bare_relative_root_with_custom_backend_root() {
        use crate::storage::persistence::filesystem::{FilesystemConfig, local::LocalConfig};

        let backend = tempfile::tempdir().expect("backend root");
        let cwd_fixture = tempfile::tempdir_in(".").expect("relative fixture");
        let root = PathBuf::from(".")
            .join(cwd_fixture.path().file_name().expect("fixture name"))
            .join("queue");
        assert!(root.is_relative(), "exercise backend-relative semantics");
        let legacy = backend.path().join(&root).join(&root);
        let topics = HashMap::from([(
            "events".into(),
            TopicConfig {
                partition_count: 1,
                ..Default::default()
            },
        )]);
        let old = QueueClient::open(QueueConfig {
            root: legacy.display().to_string(),
            topics: topics.clone(),
            ..Default::default()
        })
        .await
        .expect("legacy backend-root fixture");
        old.producer()
            .send(Message::new(
                "events",
                "tenant-a",
                b"backend-root-history".to_vec(),
            ))
            .await
            .expect("legacy send");
        old.shutdown().await.expect("legacy shutdown");
        drop(old);
        let legacy_file = legacy.join("events/0/0000000000.qseg");
        let before = std::fs::read(&legacy_file).expect("legacy bytes");
        let factory = Arc::new(
            FilesystemFactory::create(FilesystemConfig {
                local: Some(LocalConfig {
                    root_dir: Some(backend.path().to_path_buf()),
                    ..Default::default()
                }),
                ..Default::default()
            })
            .await
            .expect("custom-root factory"),
        );
        let root_text = root.to_str().expect("UTF-8 fixture");
        let error = match FactoryQueueFs::new(factory, root_text) {
            Err(error) => error,
            Ok(adapter) => match QueueClient::open_with_fs(
                QueueConfig {
                    root: root_text.into(),
                    topics,
                    ..Default::default()
                },
                Some(adapter),
            )
            .await
            {
                Err(error) => error,
                Ok(queue) => {
                    queue
                        .shutdown()
                        .await
                        .expect("stop incorrectly admitted queue");
                    panic!("bare relative root must not abandon custom-backend history");
                }
            },
        };
        assert!(error.to_string().contains("migration"), "{error}");
        assert_eq!(
            std::fs::read(legacy_file).expect("preserved backend history"),
            before
        );
        assert!(!root.exists(), "no queue state may be created under CWD");
        assert!(
            !backend.path().join(&root).join("events").exists(),
            "no fresh backend queue state"
        );
    }
}
