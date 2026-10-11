//! Conditional publication prerequisites through the existing store wrapper.
//! These tests certify primitive behavior, not lease ownership or a GC protocol.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use bytes::Bytes;
use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::{Error, PutMode, PutOptions, UpdateVersion};
use proximadb_object_store::ProximaObjectStore;

/// A backend fault adapter, not a mock conditional-write algorithm. Native
/// in-memory preconditions still decide every write.
#[derive(Debug, Default)]
struct FaultStore {
    inner: InMemory,
    lose_put_response: AtomicBool,
    conflict_after_put: AtomicBool,
    replace_after_get: AtomicBool,
    fail_get: AtomicBool,
    fail_body: AtomicBool,
    puts: AtomicUsize,
}

impl std::fmt::Display for FaultStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("FaultStore")
    }
}

fn unavailable() -> Error {
    Error::Generic {
        store: "FaultStore",
        source: Box::new(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "injected timeout",
        )),
    }
}

#[async_trait::async_trait]
impl object_store::ObjectStore for FaultStore {
    async fn put_opts(
        &self,
        path: &Path,
        payload: object_store::PutPayload,
        options: PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
        self.puts.fetch_add(1, Ordering::SeqCst);
        let receipt = self.inner.put_opts(path, payload, options).await?;
        if self.lose_put_response.swap(false, Ordering::SeqCst) {
            return Err(unavailable());
        }
        if self.conflict_after_put.swap(false, Ordering::SeqCst) {
            return Err(Error::Precondition {
                path: path.to_string(),
                source: Box::new(std::io::Error::other(
                    "backend retry after ambiguous response",
                )),
            });
        }
        Ok(receipt)
    }
    async fn put_multipart_opts(
        &self,
        path: &Path,
        options: object_store::PutMultipartOptions,
    ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
        self.inner.put_multipart_opts(path, options).await
    }
    async fn get_opts(
        &self,
        path: &Path,
        options: object_store::GetOptions,
    ) -> object_store::Result<object_store::GetResult> {
        if self.fail_get.load(Ordering::SeqCst) {
            return Err(unavailable());
        }
        let mut snapshot = self.inner.get_opts(path, options).await?;
        if self.fail_body.load(Ordering::SeqCst) {
            use futures::StreamExt;
            snapshot.payload = object_store::GetResultPayload::Stream(
                futures::stream::once(async { Err(unavailable()) }).boxed(),
            );
        }
        if self.replace_after_get.swap(false, Ordering::SeqCst) {
            self.inner
                .put_opts(
                    path,
                    Bytes::from_static(b"successor").into(),
                    PutOptions::default(),
                )
                .await?;
        }
        Ok(snapshot)
    }
    fn delete_stream(
        &self,
        paths: futures::stream::BoxStream<'static, object_store::Result<Path>>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<Path>> {
        self.inner.delete_stream(paths)
    }
    fn list(
        &self,
        prefix: Option<&Path>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        self.inner.list(prefix)
    }
    async fn list_with_delimiter(
        &self,
        prefix: Option<&Path>,
    ) -> object_store::Result<object_store::ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }
    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        options: object_store::CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
}

#[tokio::test]
async fn conditional_publication_lost_response_adds_no_wrapper_retry_or_conflict_conversion() {
    let backend = Arc::new(FaultStore::default());
    let store = ProximaObjectStore::new(backend.clone());
    let path = Path::from("data/t/n/head");
    let initial = store
        .put_opts(
            &path,
            Bytes::from_static(b"initial"),
            PutMode::Create.into(),
        )
        .await
        .unwrap();
    let expected = UpdateVersion::from(initial);
    backend.lose_put_response.store(true, Ordering::SeqCst);

    let outcome = store
        .put_opts(
            &path,
            Bytes::from_static(b"operation-2"),
            PutMode::Update(expected.clone()).into(),
        )
        .await;

    assert!(matches!(outcome, Err(Error::Generic { .. })), "{outcome:?}");
    assert_eq!(
        backend.puts.load(Ordering::SeqCst),
        2,
        "no wrapper-added retry"
    );
    let (body, meta) = store.get_with_meta(&path).await.unwrap();
    assert_eq!(
        body,
        Bytes::from_static(b"operation-2"),
        "write committed despite lost response"
    );
    assert_ne!(revision(&meta), expected);
    let retry = store
        .put_opts(
            &path,
            Bytes::from_static(b"retry"),
            PutMode::Update(expected).into(),
        )
        .await;
    assert!(matches!(retry, Err(Error::Precondition { .. })));
    assert_eq!(store.get(&path).await.unwrap(), body);
}

#[tokio::test]
async fn conditional_publication_body_and_validator_come_from_one_snapshot() {
    let backend = Arc::new(FaultStore::default());
    let store = ProximaObjectStore::new(backend.clone());
    let path = Path::from("data/t/n/head");
    let initial = store
        .put_opts(
            &path,
            Bytes::from_static(b"initial"),
            PutMode::Create.into(),
        )
        .await
        .unwrap();
    backend.replace_after_get.store(true, Ordering::SeqCst);

    let (body, meta) = store.get_with_meta(&path).await.unwrap();

    assert_eq!(body, Bytes::from_static(b"initial"));
    assert_eq!(revision(&meta), UpdateVersion::from(initial));
    assert_eq!(
        store.get(&path).await.unwrap(),
        Bytes::from_static(b"successor")
    );
    let stale = store
        .put_opts(
            &path,
            Bytes::from_static(b"stale"),
            PutMode::Update(revision(&meta)).into(),
        )
        .await;
    assert!(matches!(stale, Err(Error::Precondition { .. })));
}

#[tokio::test]
async fn conditional_publication_read_outage_is_not_absence() {
    let backend = Arc::new(FaultStore::default());
    let store = ProximaObjectStore::new(backend.clone());
    backend.fail_get.store(true, Ordering::SeqCst);
    let outcome = store.get_with_meta(&Path::from("data/t/n/head")).await;
    assert!(matches!(outcome, Err(Error::Generic { .. })), "{outcome:?}");
    assert_eq!(backend.puts.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn conditional_publication_body_failure_is_not_an_empty_snapshot() {
    let backend = Arc::new(FaultStore::default());
    let store = ProximaObjectStore::new(backend.clone());
    let path = Path::from("data/t/n/head");
    store.put(&path, Bytes::from_static(b"body")).await.unwrap();
    backend.fail_body.store(true, Ordering::SeqCst);
    assert!(matches!(
        store.get_with_meta(&path).await,
        Err(Error::Generic { .. })
    ));
}

/// Model a backend that committed, lost a response, retried conditionally and
/// surfaced a conflict. This is not cloud transport certification; it ensures
/// the wrapper does not hide the outcome that callers must reconcile.
#[tokio::test]
async fn conditional_publication_backend_conflict_can_follow_a_committed_attempt() {
    let backend = Arc::new(FaultStore::default());
    let store = ProximaObjectStore::new(backend.clone());
    let path = Path::from("data/t/n/head");
    let initial = store
        .put_opts(
            &path,
            Bytes::from_static(b"initial"),
            PutMode::Create.into(),
        )
        .await
        .unwrap();
    backend.conflict_after_put.store(true, Ordering::SeqCst);
    let outcome = store
        .put_opts(
            &path,
            Bytes::from_static(b"operation-2"),
            PutMode::Update(initial.into()).into(),
        )
        .await;
    assert!(matches!(outcome, Err(Error::Precondition { .. })));
    assert_eq!(backend.puts.load(Ordering::SeqCst), 2);
    assert_eq!(
        store.get(&path).await.unwrap(),
        Bytes::from_static(b"operation-2")
    );
}

fn memory_store() -> ProximaObjectStore {
    ProximaObjectStore::new(Arc::new(InMemory::new()))
}

fn revision(meta: &object_store::ObjectMeta) -> UpdateVersion {
    UpdateVersion {
        e_tag: meta.e_tag.clone(),
        version: meta.version.clone(),
    }
}

#[tokio::test]
async fn conditional_publication_same_revision_has_exactly_one_winner() {
    let store = memory_store();
    let other = store.clone();
    let path = Path::from("data/tenant/ns/resource/head");
    let created = store
        .put_opts(
            &path,
            Bytes::from_static(b"initial"),
            PutMode::Create.into(),
        )
        .await
        .unwrap();
    let (bytes, meta) = store.get_with_meta(&path).await.unwrap();
    assert_eq!(bytes, Bytes::from_static(b"initial"));
    assert_eq!(revision(&meta), UpdateVersion::from(created));
    let expected = revision(&meta);

    let (a, b) = tokio::join!(
        store.put_opts(
            &path,
            Bytes::from_static(b"a"),
            PutMode::Update(expected.clone()).into()
        ),
        other.put_opts(
            &path,
            Bytes::from_static(b"b"),
            PutMode::Update(expected).into()
        ),
    );
    assert_eq!(usize::from(a.is_ok()) + usize::from(b.is_ok()), 1);
    let (winner, rejected) = if a.is_ok() { (b"a", b) } else { (b"b", a) };
    assert!(
        matches!(rejected, Err(Error::Precondition { .. })),
        "{rejected:?}"
    );
    assert_eq!(store.get(&path).await.unwrap().as_ref(), winner);
}

#[tokio::test]
async fn conditional_publication_missing_and_existing_remain_distinct() {
    let store = memory_store();
    let path = Path::from("data/t/n/head");
    assert!(matches!(
        store.get_with_meta(&path).await,
        Err(Error::NotFound { .. })
    ));
    store
        .put_opts(&path, Bytes::from_static(b"first"), PutMode::Create.into())
        .await
        .unwrap();
    let duplicate = store
        .put_opts(&path, Bytes::from_static(b"second"), PutMode::Create.into())
        .await;
    assert!(matches!(duplicate, Err(Error::AlreadyExists { .. })));
    assert_eq!(
        store.get(&path).await.unwrap(),
        Bytes::from_static(b"first")
    );
}

#[tokio::test]
async fn conditional_publication_memory_validator_stays_invalid_after_content_returns() {
    let store = memory_store();
    let path = Path::from("data/t/n/head");
    let initial = store
        .put_opts(&path, Bytes::from_static(b"a"), PutMode::Create.into())
        .await
        .unwrap();
    let original = UpdateVersion::from(initial);
    let changed = store
        .put_opts(
            &path,
            Bytes::from_static(b"b"),
            PutMode::Update(original.clone()).into(),
        )
        .await
        .unwrap();
    store
        .put_opts(
            &path,
            Bytes::from_static(b"a"),
            PutMode::Update(changed.into()).into(),
        )
        .await
        .unwrap();

    let delayed = store
        .put_opts(
            &path,
            Bytes::from_static(b"stale"),
            PutMode::Update(original).into(),
        )
        .await;
    assert!(matches!(delayed, Err(Error::Precondition { .. })));
    assert_eq!(store.get(&path).await.unwrap(), Bytes::from_static(b"a"));
}

#[tokio::test]
async fn conditional_publication_requires_a_nonempty_update_validator() {
    let store = memory_store();
    let path = Path::from("data/t/n/head");
    store
        .put(&path, Bytes::from_static(b"original"))
        .await
        .unwrap();
    for expected in [
        UpdateVersion {
            e_tag: None,
            version: None,
        },
        UpdateVersion {
            e_tag: Some(String::new()),
            version: None,
        },
        UpdateVersion {
            e_tag: None,
            version: Some(String::new()),
        },
    ] {
        let outcome = store
            .put_opts(
                &path,
                Bytes::from_static(b"unconditional"),
                PutMode::Update(expected).into(),
            )
            .await;
        let Err(Error::Generic { source, .. }) = outcome else {
            panic!("invalid input must be distinguished from contention: {outcome:?}");
        };
        assert_eq!(
            source.downcast_ref::<std::io::Error>().unwrap().kind(),
            std::io::ErrorKind::InvalidInput
        );
        assert_eq!(
            store.get(&path).await.unwrap(),
            Bytes::from_static(b"original")
        );
    }
}

#[tokio::test]
async fn conditional_publication_local_update_is_explicitly_unsupported() {
    let dir = tempfile::tempdir().unwrap();
    let store = ProximaObjectStore::from_url(&format!("file://{}", dir.path().display())).unwrap();
    let path = Path::from("data/t/n/head");
    store
        .put_opts(
            &path,
            Bytes::from_static(b"initial"),
            PutMode::Create.into(),
        )
        .await
        .unwrap();
    let (_, meta) = store.get_with_meta(&path).await.unwrap();

    let outcome = store
        .put_opts(
            &path,
            Bytes::from_static(b"replacement"),
            PutMode::Update(revision(&meta)).into(),
        )
        .await;
    assert!(
        matches!(outcome, Err(Error::NotImplemented { .. })),
        "{outcome:?}"
    );
    assert_eq!(
        store.get(&path).await.unwrap(),
        Bytes::from_static(b"initial")
    );
}

#[tokio::test]
async fn conditional_publication_honors_nonempty_base_and_read_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let store = ProximaObjectStore::from_url(&format!("file://{}", dir.path().display())).unwrap();
    let path = Path::from("nested/head");
    let created = store
        .put_opts(
            &path,
            Bytes::from_static(b"body"),
            PutOptions::from(PutMode::Create),
        )
        .await
        .unwrap();
    let (bytes, meta) = store.get_with_meta(&path).await.unwrap();

    assert_eq!(bytes, Bytes::from_static(b"body"));
    assert_eq!(meta.size, bytes.len() as u64);
    assert_eq!(revision(&meta), UpdateVersion::from(created));
    assert_eq!(
        std::fs::read(dir.path().join("nested/head")).unwrap(),
        b"body"
    );
}
