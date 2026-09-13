//! Recovery must not turn unavailable/corrupt persistence into an empty queue.
//! Fixtures use the existing frame format; adapters delegate real local I/O.

use async_trait::async_trait;
use proximadb_queue::error::QueueError;
use proximadb_queue::fs::{LocalFs, Metadata, QueueFs, Result};
use proximadb_queue::memory_tier::PartitionMemory;
use proximadb_queue::{Message, QueueClient, QueueConfig, TopicConfig};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

#[derive(Debug)]
enum Fault {
    None,
    List {
        path: PathBuf,
        nth: usize,
        missing: bool,
    },
    Read {
        path: PathBuf,
        missing: bool,
    },
}

#[derive(Debug)]
struct CheckedFs {
    inner: Arc<dyn QueueFs>,
    fault: Fault,
    list_calls: AtomicUsize,
    scope: Option<PathBuf>,
}

impl CheckedFs {
    fn new(fault: Fault) -> Arc<Self> {
        Arc::new(Self {
            inner: LocalFs::new_arc(),
            fault,
            list_calls: AtomicUsize::new(0),
            scope: None,
        })
    }
    fn scoped(root: &Path) -> Arc<Self> {
        Arc::new(Self {
            inner: LocalFs::new_arc(),
            fault: Fault::None,
            list_calls: AtomicUsize::new(0),
            scope: Some(root.to_path_buf()),
        })
    }
    fn check_scope(&self, path: &Path) -> Result<()> {
        if self
            .scope
            .as_ref()
            .is_some_and(|root| !path.starts_with(root))
        {
            return Err(QueueError::Persistence(format!(
                "wrong recovery filesystem for {path:?}"
            )));
        }
        Ok(())
    }
    fn injected(missing: bool) -> QueueError {
        if missing {
            QueueError::NotFound("injected missing path".into())
        } else {
            QueueError::Persistence("injected unavailable persistence".into())
        }
    }
}

#[async_trait]
impl QueueFs for CheckedFs {
    fn supports_conditional_replace(&self) -> bool {
        self.inner.supports_conditional_replace()
    }
    async fn create_dir_all(&self, path: &Path) -> Result<()> {
        self.inner.create_dir_all(path).await
    }
    async fn append(&self, path: &Path, bytes: &[u8]) -> Result<()> {
        self.inner.append(path, bytes).await
    }
    async fn fsync(&self, path: &Path) -> Result<()> {
        self.inner.fsync(path).await
    }
    async fn compare_exchange(
        &self,
        path: &Path,
        expected: Option<&[u8]>,
        replacement: &[u8],
        guards: &[(&Path, Option<&[u8]>)],
    ) -> Result<bool> {
        self.inner
            .compare_exchange(path, expected, replacement, guards)
            .await
    }
    async fn read(&self, path: &Path) -> Result<Vec<u8>> {
        self.check_scope(path)?;
        if let Fault::Read {
            path: target,
            missing,
        } = &self.fault
            && path == target
        {
            return Err(Self::injected(*missing));
        }
        self.inner.read(path).await
    }
    async fn list(&self, path: &Path) -> Result<Vec<PathBuf>> {
        self.check_scope(path)?;
        if let Fault::List {
            path: target,
            nth,
            missing,
        } = &self.fault
            && path == target
            && self.list_calls.fetch_add(1, Ordering::SeqCst) + 1 == *nth
        {
            return Err(Self::injected(*missing));
        }
        self.inner.list(path).await
    }
    async fn rename(&self, from: &Path, to: &Path) -> Result<()> {
        self.inner.rename(from, to).await
    }
    async fn delete(&self, path: &Path) -> Result<()> {
        self.inner.delete(path).await
    }
    async fn metadata(&self, path: &Path) -> Result<Metadata> {
        self.inner.metadata(path).await
    }
}

fn config(root: &Path) -> QueueConfig {
    QueueConfig {
        root: format!("file://{}", root.display()),
        topics: [(
            "t".into(),
            TopicConfig {
                partition_count: 1,
                ..Default::default()
            },
        )]
        .into(),
        ..Default::default()
    }
}

fn frame(offset: u64, payload: &[u8]) -> Vec<u8> {
    let mut bytes = (payload.len() as u32).to_be_bytes().to_vec();
    bytes.extend_from_slice(&offset.to_be_bytes());
    bytes.extend_from_slice(payload);
    bytes
}

fn valid_frame(offset: u64) -> Vec<u8> {
    frame(
        offset,
        &bincode::serialize(&Message::new("t", "tenant", vec![42])).unwrap(),
    )
}

fn seed(root: &Path, bytes: &[u8]) -> PathBuf {
    let path = root.join("t/0/0000000000.qseg");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, bytes).unwrap();
    path
}

async fn assert_open_fails(config: QueueConfig, fs: Arc<dyn QueueFs>, reason: &str) {
    match QueueClient::open_with_fs(config, Some(fs)).await {
        Ok(client) => {
            client.shutdown().await.unwrap();
            panic!("startup accepted {reason}");
        }
        Err(QueueError::Persistence(_) | QueueError::NotFound(_)) => {}
        Err(other) => panic!("unexpected failure for {reason}: {other:?}"),
    }
}

#[tokio::test]
async fn recovery_rejects_local_listing_failure_after_writer_discovery() {
    for missing in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        seed(dir.path(), &valid_frame(0));
        let fs = CheckedFs::new(Fault::List {
            path: dir.path().join("t/0"),
            nth: 2,
            missing,
        });
        assert_open_fails(config(dir.path()), fs, "failed recovery LIST").await;
    }
}

#[tokio::test]
async fn recovery_rejects_archive_listing_outage() {
    let dir = tempfile::tempdir().unwrap();
    let archive = tempfile::tempdir().unwrap();
    let mut config = config(dir.path());
    config.object_archive = Some(format!("file://{}", archive.path().display()));
    let fs = CheckedFs::new(Fault::List {
        path: archive.path().join("t/0"),
        nth: 1,
        missing: false,
    });
    assert_open_fails(config, fs, "archive LIST outage").await;
}

#[tokio::test]
async fn recovery_accepts_typed_absent_archive() {
    let dir = tempfile::tempdir().unwrap();
    let archive = tempfile::tempdir().unwrap();
    let mut config = config(dir.path());
    config.object_archive = Some(format!("file://{}", archive.path().display()));
    let fs = CheckedFs::new(Fault::List {
        path: archive.path().join("t/0"),
        nth: 1,
        missing: true,
    });
    let client = QueueClient::open_with_fs(config, Some(fs)).await.unwrap();
    client.shutdown().await.unwrap();
}

#[tokio::test]
async fn recovery_rejects_unreadable_or_disappeared_listed_segment() {
    for missing in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let path = seed(dir.path(), &valid_frame(0));
        let fs = CheckedFs::new(Fault::Read { path, missing });
        assert_open_fails(config(dir.path()), fs, "unreadable listed segment").await;
    }
}

#[tokio::test]
async fn recovery_rejects_malformed_complete_frame() {
    let dir = tempfile::tempdir().unwrap();
    seed(dir.path(), &frame(0, &[255, 0, 0]));
    assert_open_fails(
        config(dir.path()),
        LocalFs::new_arc(),
        "malformed full frame",
    )
    .await;
}

#[tokio::test]
async fn recovery_rejects_trailing_garbage_inside_complete_frame() {
    let dir = tempfile::tempdir().unwrap();
    let mut bytes = bincode::serialize(&Message::new("t", "tenant", vec![42])).unwrap();
    bytes.push(255);
    seed(dir.path(), &frame(0, &bytes));
    assert_open_fails(
        config(dir.path()),
        LocalFs::new_arc(),
        "trailing payload garbage",
    )
    .await;
}

#[tokio::test]
async fn recovery_preserves_valid_prefix_before_truncated_tail() {
    let dir = tempfile::tempdir().unwrap();
    let mut bytes = valid_frame(0);
    bytes.extend_from_slice(&30u32.to_be_bytes());
    bytes.extend_from_slice(&1u64.to_be_bytes());
    bytes.extend_from_slice(b"short");
    seed(dir.path(), &bytes);
    let client = QueueClient::open(config(dir.path())).await.unwrap();
    let consumer = client.consumer("g");
    consumer.subscribe("t", &[0]).await.unwrap();
    let batch = consumer.poll(2, Duration::ZERO).await.unwrap();
    assert_eq!(batch.len(), 1);
    assert_eq!(batch[0].message.payload, vec![42]);
    client.shutdown().await.unwrap();
}

#[tokio::test]
async fn recovery_rejects_maximum_frame_offset_without_panicking() {
    for acknowledged in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        seed(dir.path(), &valid_frame(u64::MAX));
        if acknowledged {
            let group = dir.path().join("t/0/g");
            std::fs::create_dir_all(&group).unwrap();
            std::fs::write(
                group.join("offset.meta"),
                serde_json::to_vec(&serde_json::json!({
                    "group": "g", "committed_offset": u64::MAX,
                }))
                .unwrap(),
            )
            .unwrap();
        }
        assert_open_fails(
            config(dir.path()),
            LocalFs::new_arc(),
            "unrepresentable next offset, including an already-acknowledged frame",
        )
        .await;
    }
}

#[tokio::test]
async fn memory_rejects_maximum_explicit_offset_without_mutating_state() {
    let memory = Arc::new(PartitionMemory::new(0, 4));
    let result = memory
        .enqueue_with_offset(Message::new("t", "tenant", vec![42]), u64::MAX)
        .await;
    assert!(
        result.is_err(),
        "maximum frame offset has no representable successor"
    );
    assert_eq!(memory.next_offset(), 0);
    assert_eq!(memory.depth().await, 0);
}

#[tokio::test]
async fn memory_implicit_offset_exhaustion_does_not_wrap() {
    let memory = Arc::new(PartitionMemory::new(0, 4));
    memory
        .enqueue_with_offset(Message::new("t", "tenant", vec![42]), u64::MAX - 1)
        .await
        .unwrap();
    assert_eq!(memory.next_offset(), u64::MAX);
    let result = memory
        .try_enqueue(Message::new("t", "tenant", vec![43]))
        .await;
    assert!(
        result.is_err(),
        "exhausted offset must not wrap or be assigned"
    );
    assert_eq!(memory.next_offset(), u64::MAX);
    assert_eq!(memory.depth().await, 1);
    assert_eq!(memory.read_from(u64::MAX - 1, 2).await.len(), 1);
}

#[tokio::test]
async fn recovery_uses_selected_archive_filesystem() {
    let dir = tempfile::tempdir().unwrap();
    let archive = tempfile::tempdir().unwrap();
    let original = seed(archive.path(), &valid_frame(0));
    // The empty local active segment uses ID zero; archive ID one is unshadowed.
    std::fs::rename(&original, original.with_file_name("0000000001.qseg")).unwrap();
    let mut config = config(dir.path());
    config.object_archive = Some(format!("file://{}", archive.path().display()));
    let client = QueueClient::open_with_fs_split(
        config,
        Some(CheckedFs::scoped(dir.path())),
        Some(CheckedFs::scoped(archive.path())),
    )
    .await
    .unwrap();
    let consumer = client.consumer("g");
    consumer.subscribe("t", &[0]).await.unwrap();
    assert_eq!(
        consumer.poll(2, Duration::ZERO).await.unwrap().len(),
        1,
        "archive recovery must use the selected archive adapter"
    );
    client.shutdown().await.unwrap();
}

#[tokio::test]
async fn empty_local_bootstrap_segment_does_not_hide_archived_segment_zero() {
    let dir = tempfile::tempdir().unwrap();
    let archive = tempfile::tempdir().unwrap();
    seed(archive.path(), &valid_frame(0));
    let mut config = config(dir.path());
    config.object_archive = Some(format!("file://{}", archive.path().display()));
    let client = QueueClient::open_with_fs_split(
        config,
        Some(CheckedFs::scoped(dir.path())),
        Some(CheckedFs::scoped(archive.path())),
    )
    .await
    .unwrap();
    let consumer = client.consumer("g");
    consumer.subscribe("t", &[0]).await.unwrap();
    let batch = consumer.poll(2, Duration::ZERO).await.unwrap();
    client.shutdown().await.unwrap();
    assert_eq!(
        batch.len(),
        1,
        "empty local bootstrap must not hide archived segment zero"
    );
    assert_eq!(batch[0].message.payload, vec![42]);
}

#[tokio::test]
async fn nonempty_local_segment_is_authoritative_over_its_archive() {
    let dir = tempfile::tempdir().unwrap();
    let archive = tempfile::tempdir().unwrap();
    seed(dir.path(), &valid_frame(0));
    seed(
        archive.path(),
        &frame(
            0,
            &bincode::serialize(&Message::new("t", "tenant", vec![99])).unwrap(),
        ),
    );
    let mut config = config(dir.path());
    config.object_archive = Some(format!("file://{}", archive.path().display()));
    let client = QueueClient::open(config).await.unwrap();
    let consumer = client.consumer("g");
    consumer.subscribe("t", &[0]).await.unwrap();
    let batch = consumer.poll(2, Duration::ZERO).await.unwrap();
    client.shutdown().await.unwrap();
    assert_eq!(batch.len(), 1);
    assert_eq!(batch[0].message.payload, vec![42]);
}

#[tokio::test]
async fn unreadable_local_placeholder_is_not_treated_as_an_archive_fallback() {
    let dir = tempfile::tempdir().unwrap();
    let archive = tempfile::tempdir().unwrap();
    let local = seed(dir.path(), &[]);
    seed(archive.path(), &valid_frame(0));
    let mut config = config(dir.path());
    config.object_archive = Some(format!("file://{}", archive.path().display()));
    assert_open_fails(
        config,
        CheckedFs::new(Fault::Read {
            path: local,
            missing: false,
        }),
        "unreadable local bootstrap even with an archive copy",
    )
    .await;
}

#[tokio::test]
async fn archive_bootstrap_survives_append_and_second_restart() {
    let dir = tempfile::tempdir().unwrap();
    let archive = tempfile::tempdir().unwrap();
    seed(archive.path(), &valid_frame(0));
    let mut config = config(dir.path());
    config.object_archive = Some(format!("file://{}", archive.path().display()));
    let first = QueueClient::open(config.clone()).await.unwrap();
    first
        .producer()
        .send(Message::new("t", "tenant", vec![43]))
        .await
        .unwrap();
    first.shutdown().await.unwrap();
    let second = QueueClient::open(config).await.unwrap();
    let consumer = second.consumer("g");
    consumer.subscribe("t", &[0]).await.unwrap();
    let batch = consumer.poll(3, Duration::ZERO).await.unwrap();
    second.shutdown().await.unwrap();
    assert_eq!(
        batch
            .iter()
            .map(|delivery| delivery.message.payload.clone())
            .collect::<Vec<_>>(),
        vec![vec![42], vec![43]],
        "a new local append must not hide unacknowledged archived data on the next restart"
    );
}
