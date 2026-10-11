//! Consumer progress faults through the QueueFs port. Every successful write
//! delegates to real LocalFs; only scheduling and returned errors are injected.

use async_trait::async_trait;
use proximadb_queue::error::QueueError;
use proximadb_queue::fs::{LocalFs, Metadata, QueueFs, Result};
use proximadb_queue::leases::{LeaseMeta, renew};
use proximadb_queue::{Consumer, Message, QueueClient, QueueConfig, TopicConfig};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{Notify, oneshot};
use tokio::task::JoinHandle;

const PASS: u8 = 0;
const READ_ERROR: u8 = 1;
const APPLIED_ERROR: u8 = 2;
const PAUSE: u8 = 3;
const PAUSE_RELEASE: u8 = 4;
const PAUSE_PROGRESS_READ: u8 = 5;
const BOUND: Duration = Duration::from_secs(5);

#[derive(Debug)]
struct ControlledFs {
    inner: Arc<dyn QueueFs>,
    fault: AtomicU8,
    offset_attempts: AtomicUsize,
    entered: Notify,
    resume: Arc<Notify>,
    publication: Mutex<Option<JoinHandle<Result<bool>>>>,
}

impl ControlledFs {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: LocalFs::new_arc(),
            fault: AtomicU8::new(PASS),
            offset_attempts: AtomicUsize::new(0),
            entered: Notify::new(),
            resume: Arc::new(Notify::new()),
            publication: Mutex::new(None),
        })
    }

    fn arm(&self, fault: u8) {
        assert_eq!(self.fault.swap(fault, Ordering::SeqCst), PASS);
    }

    async fn wait_for_publication(&self) {
        tokio::time::timeout(BOUND, self.entered.notified())
            .await
            .unwrap();
    }

    async fn resume_and_join(&self) -> bool {
        let publication = self.publication.lock().unwrap().take().unwrap();
        self.resume.notify_one();
        tokio::time::timeout(BOUND, publication)
            .await
            .unwrap()
            .unwrap()
            .unwrap()
    }
}

#[async_trait]
impl QueueFs for ControlledFs {
    fn supports_conditional_replace(&self) -> bool {
        self.inner.supports_conditional_replace()
    }

    async fn compare_exchange(
        &self,
        path: &Path,
        expected: Option<&[u8]>,
        replacement: &[u8],
        guards: &[(&Path, Option<&[u8]>)],
    ) -> Result<bool> {
        let fault = if path.file_name().is_some_and(|name| name == "offset.meta") {
            self.offset_attempts.fetch_add(1, Ordering::SeqCst);
            self.fault.swap(PASS, Ordering::SeqCst)
        } else if path.file_name().is_some_and(|name| name == "lease.meta")
            && serde_json::from_slice::<LeaseMeta>(replacement)
                .is_ok_and(|lease| lease.expires_at_unix_nanos == 0)
            && self
                .fault
                .compare_exchange(PAUSE_RELEASE, PASS, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
        {
            PAUSE
        } else {
            PASS
        };
        match fault {
            APPLIED_ERROR => {
                let applied = self
                    .inner
                    .compare_exchange(path, expected, replacement, guards)
                    .await?;
                assert!(
                    applied,
                    "fault must occur after a successful real publication"
                );
                Err(QueueError::Persistence(
                    "injected response loss after offset publication".into(),
                ))
            }
            PAUSE => {
                let inner = self.inner.clone();
                let path = path.to_path_buf();
                let expected = expected.map(<[u8]>::to_vec);
                let replacement = replacement.to_vec();
                let guards: Vec<_> = guards
                    .iter()
                    .map(|(path, expected)| (path.to_path_buf(), expected.map(<[u8]>::to_vec)))
                    .collect();
                let resume = self.resume.clone();
                let (tx, rx) = oneshot::channel();
                let publication = tokio::spawn(async move {
                    tokio::time::timeout(BOUND, resume.notified())
                        .await
                        .unwrap();
                    let guards: Vec<_> = guards
                        .iter()
                        .map(|(path, bytes)| (path.as_path(), bytes.as_deref()))
                        .collect();
                    let result = inner
                        .compare_exchange(&path, expected.as_deref(), &replacement, &guards)
                        .await;
                    let _ = tx.send(result.as_ref().copied().map_err(|e| e.to_string()));
                    result
                });
                *self.publication.lock().unwrap() = Some(publication);
                self.entered.notify_one();
                rx.await
                    .map_err(|e| QueueError::Persistence(e.to_string()))?
                    .map_err(QueueError::Persistence)
            }
            PASS => {
                self.inner
                    .compare_exchange(path, expected, replacement, guards)
                    .await
            }
            fault => panic!("unexpected armed fault {fault} at offset CAS"),
        }
    }

    async fn read(&self, path: &Path) -> Result<Vec<u8>> {
        if path.file_name().is_some_and(|name| name == "offset.meta")
            && self
                .fault
                .compare_exchange(
                    PAUSE_PROGRESS_READ,
                    PASS,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                )
                .is_ok()
        {
            self.entered.notify_one();
            self.resume.notified().await;
        }
        if path.file_name().is_some_and(|name| name == "offset.meta")
            && self
                .fault
                .compare_exchange(READ_ERROR, PASS, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
        {
            return Err(QueueError::Persistence(
                "injected offset read failure before publication".into(),
            ));
        }
        self.inner.read(path).await
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
    async fn list(&self, path: &Path) -> Result<Vec<PathBuf>> {
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

async fn setup() -> (
    tempfile::TempDir,
    Arc<ControlledFs>,
    Arc<QueueClient>,
    Consumer,
) {
    let dir = tempfile::tempdir().unwrap();
    let fs = ControlledFs::new();
    let config = QueueConfig {
        root: format!("file://{}", dir.path().display()),
        topics: [(
            "t".into(),
            TopicConfig {
                partition_count: 1,
                lease_duration: Duration::from_secs(3600),
                ..Default::default()
            },
        )]
        .into(),
        ..Default::default()
    };
    let client = QueueClient::open_with_fs(config, Some(fs.clone()))
        .await
        .unwrap();
    let consumer = client.consumer("g");
    consumer.subscribe("t", &[0]).await.unwrap();
    for value in 0..2 {
        client
            .producer()
            .send(Message::new("t", "tenant", vec![value]))
            .await
            .unwrap();
    }
    (dir, fs, client, consumer)
}

fn committed(dir: &Path) -> Option<u64> {
    match std::fs::read(dir.join("t/0/g/offset.meta")) {
        Ok(bytes) => Some(
            serde_json::from_slice::<serde_json::Value>(&bytes).unwrap()["committed_offset"]
                .as_u64()
                .unwrap(),
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => panic!("unexpected offset read error: {error}"),
    }
}

#[tokio::test]
async fn ack_read_failure_before_publication_preserves_pending_for_retry() {
    let (dir, fs, client, consumer) = setup().await;
    let batch = consumer.poll(2, Duration::ZERO).await.unwrap();
    fs.arm(READ_ERROR);
    assert!(consumer.ack(&[batch[0].message_id.clone()]).await.is_err());
    assert_eq!(committed(dir.path()), None);
    assert_eq!(fs.offset_attempts.load(Ordering::SeqCst), 0);
    consumer.ack(&[batch[0].message_id.clone()]).await.unwrap();
    assert_eq!(committed(dir.path()), Some(0));
    consumer.nack(&[batch[1].message_id.clone()]).await.unwrap();
    assert_eq!(
        consumer.poll(1, Duration::ZERO).await.unwrap()[0].message_id,
        batch[1].message_id
    );
    consumer.shutdown().await.unwrap();
    client.shutdown().await.unwrap();
}

#[tokio::test]
async fn applied_but_error_ack_cannot_be_followed_by_contradictory_nack_retry() {
    let (dir, fs, client, consumer) = setup().await;
    let batch = consumer.poll(2, Duration::ZERO).await.unwrap();
    let ids = [batch[0].message_id.clone()];
    fs.arm(APPLIED_ERROR);
    let ack = consumer.ack(&ids).await;
    assert_eq!(
        committed(dir.path()),
        Some(0),
        "real write completed before response loss"
    );
    // Reconciliation may return ACK success, or uncertainty may make the consumer
    // terminal. Neither permits NACK to promise retry of already-committed work.
    let nack = consumer.nack(&ids).await;
    let polled = consumer.poll(2, Duration::ZERO).await;
    let contradictory_retry = nack.is_ok()
        && polled
            .as_ref()
            .is_ok_and(|batch| batch.iter().any(|delivery| delivery.message_id == ids[0]));
    consumer.shutdown().await.unwrap();
    let recovered = client.consumer("g");
    recovered.subscribe("t", &[0]).await.unwrap();
    let after_recovery = recovered.poll(2, Duration::ZERO).await.unwrap();
    assert_eq!(after_recovery.len(), 1);
    assert_eq!(
        after_recovery[0].message_id, batch[1].message_id,
        "a new consumer must recover from the committed offset after response loss"
    );
    recovered.shutdown().await.unwrap();
    client.shutdown().await.unwrap();
    assert!(
        !contradictory_retry,
        "committed offset cannot coexist with successful NACK/redelivery: ACK={ack:?}, NACK={nack:?}"
    );
}

#[tokio::test]
async fn cancelled_ack_cannot_publish_after_successor_takeover() {
    let (dir, fs, client, consumer) = setup().await;
    let batch = consumer.poll(1, Duration::ZERO).await.unwrap();
    let ids = [batch[0].message_id.clone()];
    fs.arm(PAUSE);
    let old = consumer.clone();
    let ack_ids = ids.clone();
    let ack = tokio::spawn(async move { old.ack(&ack_ids).await });
    fs.wait_for_publication().await;
    ack.abort();
    assert!(ack.await.unwrap_err().is_cancelled());
    assert!(
        consumer.nack(&ids).await.is_err(),
        "cancelled ACK must fence local retry"
    );
    assert!(consumer.poll(1, Duration::ZERO).await.is_err());

    let lease_path = dir.path().join("t/0/g/lease.meta");
    let bytes = fs.inner.read(&lease_path).await.unwrap();
    let mut lease: LeaseMeta = serde_json::from_slice(&bytes).unwrap();
    lease.expires_at_unix_nanos = 0;
    assert!(
        fs.inner
            .compare_exchange(
                &lease_path,
                Some(&bytes),
                &serde_json::to_vec(&lease).unwrap(),
                &[]
            )
            .await
            .unwrap()
    );
    let successor = client.consumer("g");
    successor.subscribe("t", &[0]).await.unwrap();
    assert!(
        !fs.resume_and_join().await,
        "old guarded ACK must reject successor's lease"
    );
    assert_eq!(committed(dir.path()), None);
    let retry = successor.poll(1, Duration::ZERO).await.unwrap();
    assert_eq!(retry[0].message_id, ids[0]);
    successor.ack(&ids).await.unwrap();
    assert_eq!(committed(dir.path()), Some(0));
    consumer.shutdown().await.unwrap();
    successor.shutdown().await.unwrap();
    client.shutdown().await.unwrap();
}

#[tokio::test]
async fn same_holder_renewal_drift_retries_guarded_ack() {
    let (dir, fs, client, consumer) = setup().await;
    let batch = consumer.poll(1, Duration::ZERO).await.unwrap();
    let lease: LeaseMeta = serde_json::from_slice(
        &fs.inner
            .read(&dir.path().join("t/0/g/lease.meta"))
            .await
            .unwrap(),
    )
    .unwrap();
    fs.arm(PAUSE);
    let old = consumer.clone();
    let ids = [batch[0].message_id.clone()];
    let ack = tokio::spawn(async move { old.ack(&ids).await });
    fs.wait_for_publication().await;
    renew(
        &fs.inner,
        dir.path(),
        "t",
        0,
        "g",
        &lease.holder_id,
        Duration::from_secs(7200),
    )
    .await
    .unwrap();
    assert!(
        !fs.resume_and_join().await,
        "the first guard must detect renewed bytes"
    );
    tokio::time::timeout(BOUND, ack)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(fs.offset_attempts.load(Ordering::SeqCst), 2);
    assert_eq!(committed(dir.path()), Some(0));
    consumer.shutdown().await.unwrap();
    client.shutdown().await.unwrap();
}

#[tokio::test]
async fn cancelled_shutdown_retry_still_awaits_the_pending_release() {
    use std::future::Future;
    use std::task::Poll;

    let (dir, fs, client, consumer) = setup().await;
    fs.arm(PAUSE_RELEASE);
    let closing = consumer.clone();
    let shutdown = tokio::spawn(async move { closing.shutdown().await });
    fs.wait_for_publication().await;
    shutdown.abort();
    assert!(shutdown.await.unwrap_err().is_cancelled());

    let mut retry = Box::pin(consumer.shutdown());
    std::future::poll_fn(|cx| {
        assert!(
            retry.as_mut().poll(cx).is_pending(),
            "retry must still await the release task retained after cancellation"
        );
        Poll::Ready(())
    })
    .await;
    let lease_path = dir.path().join("t/0/g/lease.meta");
    let lease: LeaseMeta =
        serde_json::from_slice(&fs.inner.read(&lease_path).await.unwrap()).unwrap();
    assert_ne!(lease.expires_at_unix_nanos, 0, "release is still paused");
    assert!(fs.resume_and_join().await);
    tokio::time::timeout(BOUND, retry).await.unwrap().unwrap();
    let released: LeaseMeta =
        serde_json::from_slice(&fs.inner.read(&lease_path).await.unwrap()).unwrap();
    assert_eq!(released.expires_at_unix_nanos, 0);
    assert_eq!(released.holder_id, lease.holder_id);
    let successor = client.consumer("g");
    successor.subscribe("t", &[0]).await.unwrap();
    successor.shutdown().await.unwrap();
    client.shutdown().await.unwrap();
}

#[tokio::test]
async fn cancelled_subscription_before_registration_is_terminal_and_expiry_reclaimable() {
    let (dir, fs, client, first) = setup().await;
    first.shutdown().await.unwrap();
    let cancelled = client.consumer("g");
    fs.arm(PAUSE_PROGRESS_READ);
    let subscribing = cancelled.clone();
    let task = tokio::spawn(async move { subscribing.subscribe("t", &[0]).await });
    fs.wait_for_publication().await;
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    assert!(cancelled.poll(1, Duration::ZERO).await.is_err());
    assert!(cancelled.subscribe("t", &[0]).await.is_err());
    cancelled.shutdown().await.unwrap();
    // This phase has no registered renewal task to await. The configured TTL,
    // not prompt release, bounds this availability gap. Model expiry exactly.
    assert!(client.consumer("g").subscribe("t", &[0]).await.is_err());
    let path = dir.path().join("t/0/g/lease.meta");
    let mut lease: LeaseMeta = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_ne!(lease.expires_at_unix_nanos, 0);
    lease.expires_at_unix_nanos = 0;
    std::fs::write(&path, serde_json::to_vec(&lease).unwrap()).unwrap();
    let successor = client.consumer("g");
    successor.subscribe("t", &[0]).await.unwrap();
    assert_eq!(successor.poll(2, Duration::ZERO).await.unwrap().len(), 2);
    client.shutdown().await.unwrap();
}
