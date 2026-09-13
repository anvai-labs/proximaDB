//! Safety properties, not assertions that the current lease algorithm is safe.
//! The scheduling adapter delegates all persistence to real LocalFs. It pauses
//! one caller after its first read, modeling a suspended process without sleeps.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use proximadb_queue::error::QueueError;
use proximadb_queue::fs::{LocalFs, Metadata, QueueFs, Result};
use proximadb_queue::leases::{LeaseMeta, try_acquire};
use tokio::sync::Notify;

#[derive(Debug)]
enum FirstRead {
    Pause { observed: Notify, resume: Notify },
    Unavailable,
    AppliedButResponseLost,
}

#[derive(Debug)]
struct ControlledFs {
    inner: Arc<dyn QueueFs>,
    first: AtomicBool,
    behavior: FirstRead,
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
        let applied = self
            .inner
            .compare_exchange(path, expected, replacement, guards)
            .await?;
        if applied && matches!(self.behavior, FirstRead::AppliedButResponseLost) {
            return Err(QueueError::Persistence(
                "injected failure after publication".into(),
            ));
        }
        Ok(applied)
    }
    async fn create_dir_all(&self, path: &Path) -> Result<()> {
        self.inner.create_dir_all(path).await
    }
    async fn append(&self, path: &Path, data: &[u8]) -> Result<()> {
        self.inner.append(path, data).await
    }
    async fn fsync(&self, path: &Path) -> Result<()> {
        self.inner.fsync(path).await
    }
    async fn read(&self, path: &Path) -> Result<Vec<u8>> {
        if self.first.swap(false, Ordering::SeqCst) {
            match &self.behavior {
                FirstRead::Unavailable => {
                    return Err(QueueError::Persistence("injected unavailable read".into()));
                }
                FirstRead::Pause { observed, resume } => {
                    let snapshot = self.inner.read(path).await;
                    observed.notify_one();
                    resume.notified().await;
                    return snapshot;
                }
                FirstRead::AppliedButResponseLost => {}
            }
        }
        self.inner.read(path).await
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

async fn acquire(fs: &Arc<dyn QueueFs>, root: &Path, holder: &str) -> Result<()> {
    try_acquire(
        fs,
        root,
        "topic",
        0,
        "group",
        holder,
        Duration::from_secs(3600),
    )
    .await
}

fn meta_path(root: &Path) -> PathBuf {
    root.join("topic/0/group/lease.meta")
}

#[tokio::test]
async fn lease_invariant_two_live_holders_cannot_both_acquire() {
    let dir = tempfile::tempdir().unwrap();
    let local = LocalFs::new_arc();
    let delayed = Arc::new(ControlledFs {
        inner: local.clone(),
        first: AtomicBool::new(true),
        behavior: FirstRead::Pause {
            observed: Notify::new(),
            resume: Notify::new(),
        },
    });
    let delayed_fs: Arc<dyn QueueFs> = delayed.clone();
    let FirstRead::Pause { observed, resume } = &delayed.behavior else {
        unreachable!("test configured a paused first read");
    };

    // B reads absent; A reads absent, renames, verifies and returns success;
    // only then B resumes from its absent read. Both use production acquisition.
    let (a, b) = tokio::time::timeout(Duration::from_secs(10), async {
        tokio::join!(
            async {
                observed.notified().await;
                let result = acquire(&local, dir.path(), "owner-a").await;
                resume.notify_one();
                result
            },
            acquire(&delayed_fs, dir.path(), "owner-b"),
        )
    })
    .await
    .expect("deterministic schedule must complete");

    assert!(a.is_ok(), "first owner must acquire: {a:?}");
    assert!(
        matches!(b, Err(QueueError::LeaseConflict { .. })),
        "delayed contender must conflict while A is live; A={a:?}, B={b:?}"
    );
    let stored: LeaseMeta =
        serde_json::from_slice(&local.read(&meta_path(dir.path())).await.unwrap()).unwrap();
    assert_eq!(stored.holder_id, "owner-a");
}

#[tokio::test]
async fn lease_invariant_unavailable_read_cannot_replace_live_owner() {
    let dir = tempfile::tempdir().unwrap();
    let local = LocalFs::new_arc();
    acquire(&local, dir.path(), "owner-a").await.unwrap();
    let path = meta_path(dir.path());
    let original = local.read(&path).await.unwrap();
    let failing: Arc<dyn QueueFs> = Arc::new(ControlledFs {
        inner: local.clone(),
        first: AtomicBool::new(true),
        behavior: FirstRead::Unavailable,
    });

    let outcome = acquire(&failing, dir.path(), "owner-b").await;

    assert!(
        matches!(outcome, Err(QueueError::Persistence(_))),
        "unavailable is not absent: {outcome:?}"
    );
    assert_eq!(local.read(&path).await.unwrap(), original);
}

#[tokio::test]
async fn lease_invariant_corrupt_state_cannot_be_claimed_as_free() {
    assert_corrupt_state_is_rejected(b"{truncated lease").await;
}

#[tokio::test]
async fn lease_invariant_empty_state_cannot_be_claimed_as_free() {
    assert_corrupt_state_is_rejected(b"").await;
}

async fn assert_corrupt_state_is_rejected(corrupt: &[u8]) {
    let dir = tempfile::tempdir().unwrap();
    let local = LocalFs::new_arc();
    let path = meta_path(dir.path());
    local.create_dir_all(path.parent().unwrap()).await.unwrap();
    local.append(&path, corrupt).await.unwrap();
    local.fsync(&path).await.unwrap();

    let outcome = acquire(&local, dir.path(), "owner-b").await;

    assert!(
        matches!(outcome, Err(QueueError::Persistence(_))),
        "corrupt is not absent: {outcome:?}"
    );
    assert_eq!(local.read(&path).await.unwrap(), corrupt);
}

#[tokio::test]
async fn lease_applied_but_lost_response_never_acknowledges_acquisition() {
    let dir = tempfile::tempdir().unwrap();
    let local = LocalFs::new_arc();
    let failing: Arc<dyn QueueFs> = Arc::new(ControlledFs {
        inner: local.clone(),
        first: AtomicBool::new(false),
        behavior: FirstRead::AppliedButResponseLost,
    });
    assert!(matches!(
        acquire(&failing, dir.path(), "owner-a").await,
        Err(QueueError::Persistence(_))
    ));
    let stored: LeaseMeta =
        serde_json::from_slice(&local.read(&meta_path(dir.path())).await.unwrap()).unwrap();
    assert_eq!(
        stored.holder_id, "owner-a",
        "failure was injected after durable replacement"
    );
    assert!(matches!(
        acquire(&local, dir.path(), "owner-b").await,
        Err(QueueError::LeaseConflict { .. })
    ));
}

#[tokio::test]
async fn lease_renewal_rejects_missing_expired_superseded_and_corrupt_state() {
    use proximadb_queue::leases::renew;
    let expired = serde_json::to_vec(&LeaseMeta {
        holder_id: "owner-a".into(),
        expires_at_unix_nanos: 0,
    })
    .unwrap();
    let superseded = serde_json::to_vec(&LeaseMeta {
        holder_id: "owner-b".into(),
        expires_at_unix_nanos: u128::MAX,
    })
    .unwrap();
    for source in [
        None,
        Some(expired),
        Some(superseded),
        Some(b"corrupt".to_vec()),
        Some(Vec::new()),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let local = LocalFs::new_arc();
        let path = meta_path(dir.path());
        if let Some(bytes) = &source {
            local.create_dir_all(path.parent().unwrap()).await.unwrap();
            assert!(
                local
                    .compare_exchange(&path, None, bytes, &[])
                    .await
                    .unwrap()
            );
        }
        let result = renew(
            &local,
            dir.path(),
            "topic",
            0,
            "group",
            "owner-a",
            Duration::from_secs(3600),
        )
        .await;
        assert!(
            result.is_err(),
            "renewal must not implicitly reacquire: {source:?}"
        );
        match source {
            Some(bytes) => assert_eq!(local.read(&path).await.unwrap(), bytes),
            None => assert!(matches!(
                local.read(&path).await,
                Err(QueueError::NotFound(_))
            )),
        }
    }
}
