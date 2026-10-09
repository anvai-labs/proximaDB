//! Test fixture: an in-memory `QueueFs` impl with controllable behavior:
//! - inject slow fsync, failing fsync, failing append, etc. Used by the
//!   disk_tier tests to verify Strict-mode semantics without depending on
//!   real disk I/O timing.
//!
//! Lives under `tests/` (not `src/`) because it's only useful to test
//! consumers; the production code path always uses `LocalFs`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::Mutex;

use proximadb_queue::error::QueueError;
use proximadb_queue::fs::{Metadata, QueueFs, Result};

#[derive(Debug, Default, Clone, Copy)]
pub struct FakeFsConfig {
    pub disable_conditional_replace: bool,
    pub omit_directory_entries: bool,
    pub fsync_delay: Duration,
    pub fsync_failure_rate: f32, // 0.0 = never fail, 1.0 = always
    pub append_failure_rate: f32,
}

#[tokio::test]
async fn queue_open_rejects_backend_without_conditional_publication() {
    let fs = FakeFs::with_config(FakeFsConfig {
        disable_conditional_replace: true,
        ..Default::default()
    });
    let result =
        proximadb_queue::QueueClient::open_with_fs(Default::default(), Some(fs.clone())).await;
    assert!(
        result.is_err(),
        "unsupported root must fail at open, not first subscription"
    );
    assert_eq!(fs.append_calls(), 0);
}

#[tokio::test]
async fn fake_fs_enforces_sibling_guards_and_reports_directory_metadata() {
    let fs = FakeFs::new();
    let dir = Path::new("queue/group");
    let lease = dir.join("lease.meta");
    let offset = dir.join("offset.meta");
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
    assert!(fs.metadata(dir).await.unwrap().is_directory);
    assert!(!fs.metadata(&offset).await.unwrap().is_directory);
    assert!(matches!(
        fs.metadata(Path::new("missing")).await,
        Err(QueueError::NotFound(_))
    ));
}

#[tokio::test]
async fn fake_fs_lists_empty_directories_and_rejects_directory_cas() {
    let fs = FakeFs::new();
    let root = Path::new("queue");
    let directory = root.join("ExactGroup");
    fs.create_dir_all(&directory).await.unwrap();
    assert!(fs.metadata(&directory).await.unwrap().is_directory);
    assert_eq!(fs.list(root).await.unwrap(), vec![directory.clone()]);
    assert!(fs.list(&directory).await.unwrap().is_empty());
    assert!(
        fs.compare_exchange(&directory, None, b"not a file", &[])
            .await
            .is_err()
    );
    assert!(
        fs.compare_exchange(
            &root.join("offset.meta"),
            None,
            b"progress",
            &[(&directory, None)]
        )
        .await
        .is_err()
    );
    assert!(fs.metadata(&directory).await.unwrap().is_directory);
}

#[tokio::test]
async fn queue_open_rejects_adapter_omitting_created_directories() {
    let root = tempfile::tempdir().unwrap();
    let config = proximadb_queue::QueueConfig {
        root: format!("file://{}", root.path().display()),
        topics: [(
            "ExactTopic".into(),
            proximadb_queue::TopicConfig {
                partition_count: 1,
                ..Default::default()
            },
        )]
        .into(),
        ..Default::default()
    };
    let incomplete = FakeFs::with_config(FakeFsConfig {
        omit_directory_entries: true,
        ..Default::default()
    });
    let result =
        proximadb_queue::QueueClient::open_with_fs(config.clone(), Some(incomplete.clone())).await;
    assert!(
        matches!(result, Err(QueueError::Persistence(_))),
        "a backend claiming conditional writes must still prove exact directory spelling"
    );
    assert_eq!(
        incomplete.append_calls(),
        0,
        "reject before opening a queue segment"
    );
    let complete = FakeFs::new();
    let client = proximadb_queue::QueueClient::open_with_fs(config, Some(complete))
        .await
        .unwrap();
    client.shutdown().await.unwrap();
}

#[derive(Debug)]
pub struct FakeFs {
    /// None is a directory; Some(bytes) is a regular file. One lock protects
    /// namespace changes and conditional replacement together.
    state: Mutex<HashMap<PathBuf, Option<Vec<u8>>>>,
    pub fsync_call_count: Arc<AtomicUsize>,
    pub append_call_count: Arc<AtomicUsize>,
    config: FakeFsConfig,
}

impl FakeFs {
    fn create_directories(
        state: &mut HashMap<PathBuf, Option<Vec<u8>>>,
        path: &Path,
    ) -> Result<()> {
        if path
            .ancestors()
            .any(|ancestor| state.get(ancestor).is_some_and(Option::is_some))
        {
            return Err(QueueError::Persistence(format!(
                "directory parent is a file: {path:?}"
            )));
        }
        for ancestor in path.ancestors() {
            state.entry(ancestor.to_path_buf()).or_insert(None);
        }
        Ok(())
    }

    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(HashMap::new()),
            fsync_call_count: Arc::new(AtomicUsize::new(0)),
            append_call_count: Arc::new(AtomicUsize::new(0)),
            config: FakeFsConfig::default(),
        })
    }

    pub fn with_config(config: FakeFsConfig) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(HashMap::new()),
            fsync_call_count: Arc::new(AtomicUsize::new(0)),
            append_call_count: Arc::new(AtomicUsize::new(0)),
            config,
        })
    }

    pub fn fsync_calls(&self) -> usize {
        self.fsync_call_count.load(Ordering::Relaxed)
    }

    #[allow(dead_code)]
    pub fn append_calls(&self) -> usize {
        self.append_call_count.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl QueueFs for FakeFs {
    fn supports_conditional_replace(&self) -> bool {
        !self.config.disable_conditional_replace
    }

    async fn compare_exchange(
        &self,
        path: &Path,
        expected: Option<&[u8]>,
        replacement: &[u8],
        guards: &[(&Path, Option<&[u8]>)],
    ) -> Result<bool> {
        if !self.supports_conditional_replace() {
            return Err(QueueError::Persistence(
                "conditional replace disabled".into(),
            ));
        }
        for candidate in std::iter::once(path).chain(guards.iter().map(|(path, _)| *path)) {
            if candidate.parent() != path.parent()
                || candidate.file_name().is_none()
                || candidate
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| {
                        name.eq_ignore_ascii_case("access.lock")
                            || name.eq_ignore_ascii_case("leader.lock")
                    })
            {
                return Err(QueueError::Persistence(
                    "invalid conditional file or guard path".into(),
                ));
            }
        }
        let mut state = self.state.lock().await;
        if let Some(parent) = path.parent() {
            Self::create_directories(&mut state, parent)?;
        }
        for candidate in std::iter::once(path).chain(guards.iter().map(|(path, _)| *path)) {
            if state.get(candidate).is_some_and(Option::is_none) {
                return Err(QueueError::Persistence(
                    "conditional target and guards must be files, not directories".into(),
                ));
            }
        }
        if state.get(path).and_then(|bytes| bytes.as_deref()) != expected
            || guards.iter().any(|(path, expected)| {
                state.get(*path).and_then(|bytes| bytes.as_deref()) != *expected
            })
        {
            return Ok(false);
        }
        state.insert(path.to_path_buf(), Some(replacement.to_vec()));
        Ok(true)
    }

    async fn create_dir_all(&self, path: &Path) -> Result<()> {
        let mut state = self.state.lock().await;
        Self::create_directories(&mut state, path)
    }

    async fn append(&self, path: &Path, data: &[u8]) -> Result<()> {
        self.append_call_count.fetch_add(1, Ordering::Relaxed);
        if self.config.append_failure_rate >= 1.0 {
            return Err(QueueError::Persistence("fake append failure".into()));
        }
        let mut s = self.state.lock().await;
        if let Some(parent) = path.parent() {
            Self::create_directories(&mut s, parent)?;
        }
        s.entry(path.to_path_buf())
            .or_insert_with(|| Some(Vec::new()))
            .as_mut()
            .ok_or_else(|| {
                QueueError::Persistence(format!("append target is a directory: {path:?}"))
            })?
            .extend_from_slice(data);
        Ok(())
    }

    async fn fsync(&self, _path: &Path) -> Result<()> {
        self.fsync_call_count.fetch_add(1, Ordering::Relaxed);
        if self.config.fsync_delay > Duration::ZERO {
            tokio::time::sleep(self.config.fsync_delay).await;
        }
        if self.config.fsync_failure_rate >= 1.0 {
            return Err(QueueError::Persistence("fake fsync failure".into()));
        }
        Ok(())
    }

    async fn read(&self, path: &Path) -> Result<Vec<u8>> {
        let s = self.state.lock().await;
        match s.get(path) {
            Some(Some(bytes)) => Ok(bytes.clone()),
            Some(None) => Err(QueueError::Persistence(format!(
                "read target is a directory: {path:?}"
            ))),
            None => Err(QueueError::NotFound(format!("read missing {path:?}"))),
        }
    }

    async fn list(&self, dir: &Path) -> Result<Vec<PathBuf>> {
        let s = self.state.lock().await;
        match s.get(dir) {
            Some(None) => {}
            Some(Some(_)) => {
                return Err(QueueError::Persistence(format!(
                    "list target is a file: {dir:?}"
                )));
            }
            None => return Err(QueueError::NotFound(format!("list missing {dir:?}"))),
        }
        let mut entries: Vec<_> = s
            .iter()
            .filter(|(path, bytes)| {
                path.parent() == Some(dir)
                    && !(self.config.omit_directory_entries && bytes.is_none())
            })
            .map(|(path, _)| path.clone())
            .collect();
        entries.sort();
        Ok(entries)
    }

    async fn rename(&self, from: &Path, to: &Path) -> Result<()> {
        let mut s = self.state.lock().await;
        if s.get(from).is_some_and(Option::is_none) || s.get(to).is_some_and(Option::is_none) {
            return Err(QueueError::Persistence("rename supports files only".into()));
        }
        if let Some(parent) = to.parent() {
            Self::create_directories(&mut s, parent)?;
        }
        let data = s
            .remove(from)
            .ok_or_else(|| QueueError::Persistence(format!("rename missing {from:?}")))?;
        s.insert(to.to_path_buf(), data);
        Ok(())
    }

    async fn delete(&self, path: &Path) -> Result<()> {
        let mut s = self.state.lock().await;
        if s.get(path).is_some_and(Option::is_none) {
            return Err(QueueError::Persistence(format!(
                "delete target is a directory: {path:?}"
            )));
        }
        s.remove(path);
        Ok(())
    }

    async fn metadata(&self, path: &Path) -> Result<Metadata> {
        let s = self.state.lock().await;
        if let Some(Some(data)) = s.get(path) {
            return Ok(Metadata {
                size_bytes: data.len() as u64,
                is_directory: false,
            });
        }
        if s.get(path).is_some_and(Option::is_none) {
            return Ok(Metadata {
                size_bytes: 0,
                is_directory: true,
            });
        }
        Err(QueueError::NotFound(format!("metadata missing {path:?}")))
    }
}
