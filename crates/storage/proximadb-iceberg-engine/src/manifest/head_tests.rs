//! Conformance for the opt-in format, separate from unresolved legacy controls.
use super::*;
use futures::stream::BoxStream;
use object_store::memory::InMemory;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;
use tokio::sync::Notify;

const PREFIX: &str = "data/t/ns/versioned";

async fn setup() -> (ManifestCommitter, ProximaObjectStore) {
    let store = ProximaObjectStore::new(Arc::new(InMemory::new()));
    let committer = ManifestCommitter::create_versioned(store.clone(), PREFIX)
        .await
        .unwrap();
    (committer, store)
}

#[tokio::test]
async fn versioned_roundtrip_plain_and_fenced_history() {
    let (c, store) = setup().await;
    let plain = Bytes::from_static(br#"{"checkpoint":"source-cursor"}"#);
    assert_eq!(c.latest_version().await.unwrap(), None);
    assert_eq!(
        c.commit(None, plain.clone()).await.unwrap(),
        CommitOutcome::Committed(0)
    );
    assert_eq!(c.read_fenced(0).await.unwrap(), (0, plain.clone()));
    assert_eq!(
        c.commit_fenced(Some(0), 5, Bytes::from_static(b"next"))
            .await
            .unwrap(),
        CommitOutcome::Committed(1)
    );
    let reopened = ManifestCommitter::open_versioned(store.clone(), PREFIX)
        .await
        .unwrap();
    assert_eq!(reopened.read_manifest(0).await.unwrap(), plain);
    assert_eq!(
        reopened.read_fenced(1).await.unwrap(),
        (5, Bytes::from_static(b"next"))
    );
    let reader = ManifestCommitter::new(store, PREFIX);
    assert_eq!(reader.latest_version().await.unwrap(), Some(1));
    assert_eq!(reader.read_fenced(1).await.unwrap().0, 5);
    assert_eq!(
        reader.commit(Some(1), Bytes::new()).await.unwrap(),
        CommitOutcome::Conflict { latest: Some(1) },
        "a default serving handle routes to the head and preserves its generation fence"
    );
    assert_eq!(
        reader
            .commit_fenced(Some(1), 5, Bytes::new())
            .await
            .unwrap(),
        CommitOutcome::Committed(2)
    );
}

#[tokio::test]
async fn versioned_pruning_cannot_revalidate_old_or_absent_parent() {
    let (c, store) = setup().await;
    super::tests::seed_fenced(&c, 6, 5).await;
    assert_eq!(c.prune_retention(2, Duration::ZERO).await.unwrap(), 4);
    let delayed = ManifestCommitter::open_versioned(store, PREFIX)
        .await
        .unwrap();
    for parent in [None, Some(0), Some(100)] {
        assert_eq!(
            delayed
                .commit_fenced(parent, 5, Bytes::new())
                .await
                .unwrap(),
            CommitOutcome::Conflict { latest: Some(5) }
        );
    }
    assert!(matches!(
        c.read_manifest(1).await,
        Err(StorageError::NotFound(_))
    ));
    assert_eq!(c.read_fenced(5).await.unwrap().0, 5);
    assert_eq!(c.prune_retention(2, Duration::ZERO).await.unwrap(), 0);
}

#[tokio::test]
async fn versioned_lower_generation_and_plain_downgrade_are_rejected() {
    let (c, _) = setup().await;
    super::tests::seed_fenced(&c, 1, 5).await;
    assert_eq!(
        c.commit_fenced(Some(0), 4, Bytes::new()).await.unwrap(),
        CommitOutcome::Conflict { latest: Some(0) }
    );
    assert_eq!(
        c.commit(Some(0), Bytes::new()).await.unwrap(),
        CommitOutcome::Conflict { latest: Some(0) }
    );
}

#[tokio::test]
async fn versioned_retention_ignores_nested_and_noncanonical_snapshot_names() {
    for suffix in [
        "_history/nested/v00000000000000000004.snapshot",
        "_history/v4.snapshot",
    ] {
        let (c, store) = setup().await;
        super::tests::seed_fenced(&c, 6, 5).await;
        let distractor = Path::from(format!("{PREFIX}/{suffix}"));
        let noise = Bytes::from_static(b"not a canonical archive");
        store.put(&distractor, noise.clone()).await.unwrap();

        assert_eq!(
            c.prune_retention(2, Duration::ZERO).await.unwrap(),
            4,
            "{suffix} must not consume a retained-history slot"
        );
        for version in [4, 5] {
            assert_eq!(
                c.read_fenced(version).await.unwrap(),
                (5, Bytes::from_static(b"g")),
                "{suffix} must preserve the tip and predecessor"
            );
        }
        assert_eq!(store.get(&distractor).await.unwrap(), noise);
        assert_eq!(c.prune_retention(2, Duration::ZERO).await.unwrap(), 0);
    }
}

#[tokio::test]
async fn versioned_retention_ignores_distractor_age_for_young_canonical_archive() {
    let inner = Arc::new(InMemory::new());
    let direct = ProximaObjectStore::new(inner.clone());
    let c = ManifestCommitter::create_versioned(direct.clone(), PREFIX)
        .await
        .unwrap();
    super::tests::seed_fenced(&c, 6, 5).await;
    let distractor = Path::from(format!("{PREFIX}/_history/v0.snapshot"));
    direct
        .put(&distractor, Bytes::from_static(b"old unrelated object"))
        .await
        .unwrap();
    let mut scheduled = Arc::try_unwrap(ScheduledStore::new(inner, "never", false, false)).unwrap();
    scheduled.backdated_listing_path = Some(distractor);
    let reader = ManifestCommitter::new(ProximaObjectStore::new(Arc::new(scheduled)), PREFIX);

    assert_eq!(
        reader
            .prune_retention(2, Duration::from_secs(3600))
            .await
            .unwrap(),
        0,
        "an old distractor must not lend its age to a young canonical archive"
    );
    for version in 0..6 {
        assert!(reader.read_fenced(version).await.is_ok());
    }
}

#[tokio::test]
async fn versioned_initialization_never_converts_a_legacy_log() {
    let store = ProximaObjectStore::new(Arc::new(InMemory::new()));
    let legacy = ManifestCommitter::new(store.clone(), PREFIX);
    legacy
        .commit(None, Bytes::from_static(b"old"))
        .await
        .unwrap();
    assert!(
        ManifestCommitter::create_versioned(store, PREFIX)
            .await
            .is_err()
    );
    assert_eq!(
        legacy.read_manifest(0).await.unwrap(),
        Bytes::from_static(b"old")
    );
}

#[tokio::test]
async fn versioned_open_missing_is_not_automatic_initialization() {
    let store = ProximaObjectStore::new(Arc::new(InMemory::new()));
    assert!(
        ManifestCommitter::open_versioned(store.clone(), PREFIX)
            .await
            .is_err()
    );
    assert!(
        store
            .list(Some(&Path::from(PREFIX)))
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn versioned_deleted_head_fails_closed_for_existing_handle() {
    let (c, store) = setup().await;
    super::tests::seed_fenced(&c, 1, 5).await;
    store
        .delete(&Path::from(format!("{PREFIX}/_publication.head")))
        .await
        .unwrap();
    assert!(c.latest_version().await.is_err());
    assert!(c.commit_fenced(None, 5, Bytes::new()).await.is_err());
    assert!(c.prune_retention(2, Duration::ZERO).await.is_err());
}

#[tokio::test]
async fn versioned_corrupt_head_is_not_absence_or_legacy() {
    let (c, store) = setup().await;
    store
        .put(
            &Path::from(format!("{PREFIX}/_publication.head")),
            Bytes::from_static(b"broken"),
        )
        .await
        .unwrap();
    assert!(matches!(
        c.latest_version().await,
        Err(StorageError::Corruption(_))
    ));
    assert!(c.commit(None, Bytes::new()).await.is_err());
    assert!(
        ManifestCommitter::new(store, PREFIX)
            .latest_version()
            .await
            .is_err()
    );
}

#[tokio::test]
async fn versioned_retention_age_overflow_never_means_delete_immediately() {
    let (c, _) = setup().await;
    super::tests::seed_fenced(&c, 6, 5).await;
    assert!(c.prune_retention(2, Duration::MAX).await.is_err());
    assert!(c.read_manifest(0).await.is_ok());
}

/// Native InMemory decides every CAS. The adapter only controls the scheduling
/// boundary or loses a response; it does not implement an alternative algorithm.
#[derive(Debug)]
struct ScheduledStore {
    inner: Arc<InMemory>,
    path: Path,
    after_put: bool,
    conflict_reply: bool,
    fired: AtomicBool,
    fail_before: AtomicBool,
    gets: AtomicUsize,
    puts: AtomicUsize,
    lists: AtomicUsize,
    backdated_listing_path: Option<Path>,
    reached: Notify,
    resume: Notify,
}

impl ScheduledStore {
    fn new(inner: Arc<InMemory>, suffix: &str, after_put: bool, conflict_reply: bool) -> Arc<Self> {
        Arc::new(Self {
            inner,
            path: Path::from(format!("{PREFIX}/{suffix}")),
            after_put,
            conflict_reply,
            fired: AtomicBool::new(false),
            fail_before: AtomicBool::new(false),
            gets: AtomicUsize::new(0),
            puts: AtomicUsize::new(0),
            lists: AtomicUsize::new(0),
            backdated_listing_path: None,
            reached: Notify::new(),
            resume: Notify::new(),
        })
    }
}

impl std::fmt::Display for ScheduledStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ScheduledStore")
    }
}

#[async_trait::async_trait]
impl ObjectStore for ScheduledStore {
    async fn put_opts(
        &self,
        path: &Path,
        payload: PutPayload,
        options: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.puts.fetch_add(1, Ordering::SeqCst);
        if path != &self.path || self.fired.swap(true, Ordering::SeqCst) {
            return self.inner.put_opts(path, payload, options).await;
        }
        if self.after_put {
            self.inner.put_opts(path, payload, options).await?;
            self.reached.notify_one();
            self.resume.notified().await;
            let source = Box::new(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "lost committed response",
            ));
            if self.conflict_reply {
                Err(object_store::Error::Precondition {
                    path: path.to_string(),
                    source,
                })
            } else {
                Err(object_store::Error::Generic {
                    store: "ScheduledStore",
                    source,
                })
            }
        } else {
            self.reached.notify_one();
            self.resume.notified().await;
            if self.fail_before.load(Ordering::SeqCst) {
                return Err(object_store::Error::Generic {
                    store: "ScheduledStore",
                    source: Box::new(std::io::Error::other("injected failure before PUT")),
                });
            }
            self.inner.put_opts(path, payload, options).await
        }
    }
    async fn put_multipart_opts(
        &self,
        p: &Path,
        o: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(p, o).await
    }
    async fn get_opts(&self, p: &Path, o: GetOptions) -> object_store::Result<GetResult> {
        self.gets.fetch_add(1, Ordering::SeqCst);
        self.inner.get_opts(p, o).await
    }
    fn delete_stream(
        &self,
        p: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        self.inner.delete_stream(p)
    }
    fn list(&self, p: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.lists.fetch_add(1, Ordering::SeqCst);
        let backdated_path = self.backdated_listing_path.clone();
        Box::pin(futures::TryStreamExt::map_ok(
            self.inner.list(p),
            move |mut meta| {
                if backdated_path.as_ref() == Some(&meta.location) {
                    meta.last_modified = Utc::now() - chrono::Duration::hours(2);
                }
                meta
            },
        ))
    }
    async fn list_with_delimiter(&self, p: Option<&Path>) -> object_store::Result<ListResult> {
        self.inner.list_with_delimiter(p).await
    }
    async fn copy_opts(&self, a: &Path, b: &Path, o: CopyOptions) -> object_store::Result<()> {
        self.inner.copy_opts(a, b, o).await
    }
}

async fn paused_across_prune(at_archive: bool, successor_generation: u64) {
    let inner = Arc::new(InMemory::new());
    let c = ManifestCommitter::create_versioned(ProximaObjectStore::new(inner.clone()), PREFIX)
        .await
        .unwrap();
    super::tests::seed_fenced(&c, 1, 5).await;
    let suffix = if at_archive {
        "_history/v00000000000000000000.snapshot"
    } else {
        "_publication.head"
    };
    let scheduled = ScheduledStore::new(inner, suffix, false, false);
    let delayed =
        ManifestCommitter::open_versioned(ProximaObjectStore::new(scheduled.clone()), PREFIX)
            .await
            .unwrap();
    let (_, outcome) = tokio::time::timeout(Duration::from_secs(10), async {
        tokio::join!(
            async {
                scheduled.reached.notified().await;
                for v in 1..=5 {
                    assert_eq!(
                        c.commit_fenced(
                            Some(v - 1),
                            successor_generation,
                            Bytes::from_static(b"current")
                        )
                        .await
                        .unwrap(),
                        CommitOutcome::Committed(v)
                    );
                }
                assert_eq!(c.prune_retention(2, Duration::ZERO).await.unwrap(), 4);
                scheduled.resume.notify_one();
            },
            delayed.commit_fenced(Some(0), 5, Bytes::from_static(b"delayed"))
        )
    })
    .await
    .expect("deterministic schedule must complete");
    // The exact target receipt has expired. A native client may have retried an
    // earlier committed request, so the safe public result is explicitly unknown.
    assert!(
        matches!(outcome, Err(StorageError::TransactionCommitFailed(ref e)) if e.contains("indeterminate")),
        "{outcome:?}"
    );
    assert_eq!(c.latest_version().await.unwrap(), Some(5));
    assert_eq!(
        c.read_fenced(5).await.unwrap(),
        (successor_generation, Bytes::from_static(b"current"))
    );
    assert!(matches!(
        c.read_manifest(1).await,
        Err(StorageError::NotFound(_))
    ));
    assert_eq!(
        c.prune_retention(2, Duration::ZERO).await.unwrap(),
        usize::from(at_archive)
    );
}

#[tokio::test]
async fn versioned_paused_cas_cannot_publish_after_prune() {
    paused_across_prune(false, 5).await;
}

#[tokio::test]
async fn versioned_paused_cas_cannot_publish_after_takeover() {
    paused_across_prune(false, 6).await;
}

#[tokio::test]
async fn versioned_delayed_archive_is_not_publication() {
    paused_across_prune(true, 6).await;
}

async fn lost_reply(advance: u64, prune: bool, conflict_reply: bool) {
    let inner = Arc::new(InMemory::new());
    let c = ManifestCommitter::create_versioned(ProximaObjectStore::new(inner.clone()), PREFIX)
        .await
        .unwrap();
    let scheduled = ScheduledStore::new(inner, "_publication.head", true, conflict_reply);
    let delayed =
        ManifestCommitter::open_versioned(ProximaObjectStore::new(scheduled.clone()), PREFIX)
            .await
            .unwrap();
    let (_, result) = tokio::time::timeout(Duration::from_secs(10), async {
        tokio::join!(
            async {
                scheduled.reached.notified().await;
                for v in 1..=advance {
                    c.commit_fenced(Some(v - 1), 5, Bytes::from_static(b"newer"))
                        .await
                        .unwrap();
                }
                if prune {
                    assert_eq!(c.prune_retention(2, Duration::ZERO).await.unwrap(), 1);
                }
                scheduled.resume.notify_one();
            },
            delayed.commit_fenced(None, 5, Bytes::from_static(b"accepted"))
        )
    })
    .await
    .expect("lost response schedule must finish");
    if prune {
        assert!(
            matches!(result, Err(StorageError::TransactionCommitFailed(ref e)) if e.contains("indeterminate"))
        );
    } else {
        assert_eq!(result.unwrap(), CommitOutcome::Committed(0));
        assert_eq!(
            c.read_fenced(0).await.unwrap().1,
            Bytes::from_static(b"accepted")
        );
    }
    assert_eq!(c.latest_version().await.unwrap(), Some(advance));
}

#[tokio::test]
async fn versioned_lost_reply_reconciles_current_receipt() {
    lost_reply(0, false, false).await;
}

#[tokio::test]
async fn versioned_lost_reply_reconciles_archived_receipt() {
    lost_reply(2, false, false).await;
}

#[tokio::test]
async fn versioned_conditional_error_can_follow_committed_attempt() {
    lost_reply(1, false, true).await;
}

#[tokio::test]
async fn versioned_pruned_receipt_is_indeterminate_not_conflict() {
    lost_reply(2, true, false).await;
}

#[tokio::test]
async fn versioned_same_revision_has_exactly_one_winner() {
    let inner = Arc::new(InMemory::new());
    let c = ManifestCommitter::create_versioned(ProximaObjectStore::new(inner.clone()), PREFIX)
        .await
        .unwrap();
    let a = ScheduledStore::new(inner.clone(), "_publication.head", false, false);
    let b = ScheduledStore::new(inner, "_publication.head", false, false);
    let ca = ManifestCommitter::open_versioned(ProximaObjectStore::new(a.clone()), PREFIX)
        .await
        .unwrap();
    let cb = ManifestCommitter::open_versioned(ProximaObjectStore::new(b.clone()), PREFIX)
        .await
        .unwrap();
    let (_, ra, rb) = tokio::time::timeout(Duration::from_secs(10), async {
        tokio::join!(
            async {
                tokio::join!(a.reached.notified(), b.reached.notified());
                a.resume.notify_one();
                b.resume.notify_one();
            },
            ca.commit(None, Bytes::from_static(b"a")),
            cb.commit(None, Bytes::from_static(b"b"))
        )
    })
    .await
    .expect("two CAS operations must finish");
    let outcomes = [ra.unwrap(), rb.unwrap()];
    assert_eq!(
        outcomes
            .iter()
            .filter(|o| **o == CommitOutcome::Committed(0))
            .count(),
        1
    );
    assert_eq!(
        outcomes
            .iter()
            .filter(|o| **o == CommitOutcome::Conflict { latest: Some(0) })
            .count(),
        1
    );
    assert_eq!(c.latest_version().await.unwrap(), Some(0));
}

#[tokio::test]
async fn versioned_envelope_rejects_truncation_unknown_version_and_damage() {
    let (c, store) = setup().await;
    c.commit(None, Bytes::from_static(b"payload"))
        .await
        .unwrap();
    let path = Path::from(format!("{PREFIX}/_publication.head"));
    let valid = store.get(&path).await.unwrap();
    for len in 0..valid.len() {
        store.put(&path, valid.slice(..len)).await.unwrap();
        assert!(
            matches!(c.latest_version().await, Err(StorageError::Corruption(_))),
            "length {len}"
        );
    }
    for offset in [7, 8, valid.len() - 1] {
        let mut damaged = valid.to_vec();
        damaged[offset] ^= 1;
        store.put(&path, damaged.into()).await.unwrap();
        assert!(matches!(
            c.latest_version().await,
            Err(StorageError::Corruption(_))
        ));
    }
}

#[tokio::test]
async fn versioned_marker_prevents_default_fallback_after_head_deletion() {
    let (_, store) = setup().await;
    store
        .delete(&Path::from(format!("{PREFIX}/_publication.head")))
        .await
        .unwrap();
    let legacy = ManifestCommitter::new(store.clone(), PREFIX);
    assert!(legacy.latest_version().await.is_err());
    assert!(legacy.commit(None, Bytes::new()).await.is_err());
    assert!(
        ManifestCommitter::create_versioned(store, PREFIX)
            .await
            .is_err()
    );
}

#[tokio::test]
async fn versioned_legacy_dispatch_probes_format_only_once_per_operation() {
    let counted = ScheduledStore::new(Arc::new(InMemory::new()), "never", false, false);
    let c = ManifestCommitter::new(ProximaObjectStore::new(counted.clone()), PREFIX)
        .with_legacy_encoding(LegacyEncoding::GenerationPrefixed);
    c.commit_fenced(None, 5, Bytes::new()).await.unwrap();
    counted.gets.store(0, Ordering::SeqCst);
    counted.puts.store(0, Ordering::SeqCst);
    counted.lists.store(0, Ordering::SeqCst);
    c.commit_fenced(Some(0), 5, Bytes::new()).await.unwrap();
    assert_eq!(
        counted.gets.load(Ordering::SeqCst),
        3,
        "two format probes plus predecessor"
    );
    assert_eq!(counted.puts.load(Ordering::SeqCst), 1);
    assert_eq!(counted.lists.load(Ordering::SeqCst), 1);
    counted.gets.store(0, Ordering::SeqCst);
    c.read_fenced(1).await.unwrap();
    assert_eq!(
        counted.gets.load(Ordering::SeqCst),
        3,
        "no nested format probing"
    );
}

#[tokio::test]
async fn versioned_head_lookup_does_not_list_history() {
    let inner = Arc::new(InMemory::new());
    let c = ManifestCommitter::create_versioned(ProximaObjectStore::new(inner.clone()), PREFIX)
        .await
        .unwrap();
    super::tests::seed_fenced(&c, 10, 5).await;
    let counted = ScheduledStore::new(inner, "never", false, false);
    let opened =
        ManifestCommitter::open_versioned(ProximaObjectStore::new(counted.clone()), PREFIX)
            .await
            .unwrap();
    counted.gets.store(0, Ordering::SeqCst);
    assert_eq!(opened.latest_version().await.unwrap(), Some(9));
    assert_eq!(counted.gets.load(Ordering::SeqCst), 1);
    assert_eq!(counted.lists.load(Ordering::SeqCst), 0);
    counted.gets.store(0, Ordering::SeqCst);
    opened
        .commit_fenced(Some(9), 5, Bytes::new())
        .await
        .unwrap();
    assert_eq!(counted.gets.load(Ordering::SeqCst), 1);
    assert_eq!(counted.puts.load(Ordering::SeqCst), 2);
    assert_eq!(counted.lists.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn versioned_partial_provisioning_stays_closed() {
    let scheduled =
        ScheduledStore::new(Arc::new(InMemory::new()), "_publication.head", false, false);
    scheduled.fail_before.store(true, Ordering::SeqCst);
    scheduled.resume.notify_one();
    let store = ProximaObjectStore::new(scheduled);
    assert!(
        ManifestCommitter::create_versioned(store.clone(), PREFIX)
            .await
            .is_err()
    );
    assert!(
        store
            .get(&Path::from(format!("{PREFIX}/_publication.format")))
            .await
            .is_ok()
    );
    assert!(
        ManifestCommitter::open_versioned(store.clone(), PREFIX)
            .await
            .is_err()
    );
    assert!(
        ManifestCommitter::create_versioned(store.clone(), PREFIX)
            .await
            .is_err()
    );
    assert!(
        ManifestCommitter::new(store, PREFIX)
            .commit(None, Bytes::new())
            .await
            .is_err()
    );
}

#[tokio::test]
async fn versioned_local_without_conditional_update_fails_closed() {
    let dir = tempfile::tempdir().unwrap();
    let store = ProximaObjectStore::from_url(&format!("file://{}", dir.path().display())).unwrap();
    let c = ManifestCommitter::create_versioned(store.clone(), PREFIX)
        .await
        .unwrap();
    let path = Path::from(format!("{PREFIX}/_publication.head"));
    let initial = store.get(&path).await.unwrap();
    assert!(
        c.commit(None, Bytes::from_static(b"unsupported"))
            .await
            .is_err()
    );
    assert_eq!(store.get(&path).await.unwrap(), initial);
    assert_eq!(c.latest_version().await.unwrap(), None);
}

#[tokio::test]
async fn versioned_migration_partial_writes_preserve_sources_and_fail_closed() {
    for suffix in [
        "_publication.format",
        "_publication.migration",
        "_history/v00000000000000000000.snapshot",
        "_history/v00000000000000000001.snapshot",
        "_publication.head",
    ] {
        let inner = Arc::new(InMemory::new());
        let direct = ProximaObjectStore::new(inner.clone());
        let source = ManifestCommitter::new(direct.clone(), PREFIX)
            .with_legacy_encoding(LegacyEncoding::GenerationPrefixed);
        super::tests::seed_fenced(&source, 3, 5).await;
        let retained: Vec<_> =
            futures::future::join_all((0..3).map(|version| source.read_manifest(version)))
                .await
                .into_iter()
                .map(Result::unwrap)
                .collect();
        let scheduled = ScheduledStore::new(inner, suffix, false, false);
        scheduled.fail_before.store(true, Ordering::SeqCst);
        scheduled.resume.notify_one();
        let migrator = ManifestCommitter::new(ProximaObjectStore::new(scheduled), PREFIX)
            .with_legacy_encoding(LegacyEncoding::GenerationPrefixed);
        assert!(migrator.migrate_versioned(2).await.is_err(), "{suffix}");
        for (version, bytes) in retained.into_iter().enumerate() {
            assert_eq!(
                direct
                    .get(&source.manifest_path(version as u64))
                    .await
                    .unwrap(),
                bytes,
                "source version {version}, failed at {suffix}"
            );
        }
        assert!(matches!(
            direct
                .get(&Path::from(format!("{PREFIX}/_publication.head")))
                .await,
            Err(StorageError::NotFound(_))
        ));
        let reopened = ManifestCommitter::new(direct.clone(), PREFIX)
            .with_legacy_encoding(LegacyEncoding::GenerationPrefixed);
        if suffix == "_publication.format" {
            assert_eq!(reopened.latest_version().await.unwrap(), Some(2));
            assert_eq!(
                reopened
                    .commit_fenced(Some(2), 5, Bytes::new())
                    .await
                    .unwrap(),
                CommitOutcome::Committed(3)
            );
        } else {
            assert!(reopened.latest_version().await.is_err());
            assert!(
                reopened
                    .commit_fenced(Some(2), 5, Bytes::new())
                    .await
                    .is_err()
            );
            assert!(reopened.prune_retention(2, Duration::ZERO).await.is_err());
            assert!(reopened.migrate_versioned(2).await.is_err());
            assert!(
                ManifestCommitter::open_versioned(direct.clone(), PREFIX)
                    .await
                    .is_err()
            );
            assert!(
                matches!(
                    direct
                        .get(&Path::from(format!("{PREFIX}/_publication.head")))
                        .await,
                    Err(StorageError::NotFound(_))
                ),
                "retry must never reconstruct missing authority after {suffix}"
            );
        }
    }
}

#[tokio::test]
async fn versioned_migration_lost_head_response_reopens_without_reset() {
    let inner = Arc::new(InMemory::new());
    let direct = ProximaObjectStore::new(inner.clone());
    let source = ManifestCommitter::new(direct.clone(), PREFIX)
        .with_legacy_encoding(LegacyEncoding::GenerationPrefixed);
    super::tests::seed_fenced(&source, 2, 5).await;
    let scheduled = ScheduledStore::new(inner, "_publication.head", true, false);
    scheduled.resume.notify_one();
    let migrator = ManifestCommitter::new(ProximaObjectStore::new(scheduled), PREFIX)
        .with_legacy_encoding(LegacyEncoding::GenerationPrefixed);
    assert!(migrator.migrate_versioned(1).await.is_err());
    let reopened = ManifestCommitter::new(direct.clone(), PREFIX);
    assert_eq!(reopened.latest_version().await.unwrap(), Some(1));
    assert_eq!(
        reopened
            .commit_fenced(Some(1), 6, Bytes::from_static(b"after lost response"))
            .await
            .unwrap(),
        CommitOutcome::Committed(2)
    );
    let head_path = Path::from(format!("{PREFIX}/_publication.head"));
    let advanced = direct.get(&head_path).await.unwrap();
    let reconciled = source.migrate_versioned(1).await.unwrap();
    assert_eq!(reconciled.latest_version().await.unwrap(), Some(2));
    assert_eq!(direct.get(&head_path).await.unwrap(), advanced);
    direct.delete(&head_path).await.unwrap();
    assert!(
        ManifestCommitter::new(direct.clone(), PREFIX)
            .with_legacy_encoding(LegacyEncoding::GenerationPrefixed)
            .migrate_versioned(1)
            .await
            .is_err()
    );
    assert!(matches!(
        direct.get(&head_path).await,
        Err(StorageError::NotFound(_))
    ));
}
