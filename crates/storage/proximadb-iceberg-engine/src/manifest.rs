//! # manifest — Iceberg-style atomic manifest commits over object storage
//!
//! The warehouse base tier needs a way to publish a new table snapshot **atomically**
//! over decoupled object storage, where there is no transaction manager — only
//! per-object operations. This module supplies the optimistic-concurrency commit
//! primitive. Legacy logs use [`ProximaObjectStore::put_if_absent`] (create-only
//! put); legacy pruning is disabled and writes reject presently gapped history.
//! Legacy writers/pruners must be externally excluded at deployment and migration:
//! continuity cannot establish whether a deleted slot was historically refilled.
//! The experimental, explicit-opt-in versioned format instead uses conditional
//! head replacement; see [`ManifestCommitter::create_versioned`].
//!
//! - Snapshots are immutable, **monotonically-versioned** manifest objects named
//!   `{prefix}/v{version}.manifest` (zero-padded so a lexical `list` is in numeric order).
//! - To publish snapshot `parent + 1`, a committer **claims the successor slot** with a
//!   create-only put. If the slot already exists, another committer won the race: the
//!   loser gets a [`CommitOutcome::Conflict`] carrying the latest version to rebase onto
//!   and retry — exactly the Iceberg compare-and-swap commit protocol, minus a catalog.
//!
//! The manifest **bytes are opaque** to the committer: the caller serializes whatever
//! snapshot it needs (typically the set of data-file paths from
//! [`ObjectStoreBridge::list_objects`](crate::IcebergObjectStoreBridge), plus row
//! counts / column stats). This crate owns the *atomicity*, not the manifest schema.

use bytes::Bytes;
use chrono::Utc;
use object_store::path::Path;
use proximadb_kernel::error::StorageError;
use proximadb_object_store::ProximaObjectStore;
pub use proximadb_storage_common::object_store_bridge::CommitOutcome;

#[path = "manifest/head.rs"]
mod head;

#[cfg(test)]
#[path = "manifest/head_tests.rs"]
mod head_tests;

#[cfg(test)]
#[path = "manifest/legacy_policy_tests.rs"]
mod legacy_policy_tests;

/// Floor for [`ManifestCommitter::prune_retention`]'s `keep_k`. Below this the log
/// would collapse to little more than the tip, eliminating the concurrency window in
/// which a reader may still hold a just-read parent and leaving a single point of
/// failure for the generation fence.
const MIN_PRUNE_KEEP_K: usize = 2;

/// Required interpretation of unmarked historical bytes. Never infer this from
/// payload contents; current lease/catalog callers declare GenerationPrefixed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LegacyEncoding {
    Plain,
    GenerationPrefixed,
}

/// Atomic, optimistic-concurrency manifest committer over a [`ProximaObjectStore`].
///
/// One committer serves one manifest log (selected by `prefix`); construct another for
/// a different table/log.
pub struct ManifestCommitter {
    store: ProximaObjectStore,
    prefix: String,
    // Pinned on explicit provisioning/open or detection of a persisted authority.
    // Format creation/migration remains explicit, never default-on.
    authority: std::sync::OnceLock<[u8; 16]>,
    legacy_encoding: LegacyEncoding,
}

impl ManifestCommitter {
    /// Create a committer writing manifests under `prefix` (e.g. `"data/<tenant>/<ns>/_manifests"`).
    pub fn new(store: ProximaObjectStore, prefix: impl Into<String>) -> Self {
        let mut prefix = prefix.into();
        // Normalize so `{prefix}/v..` never produces a double slash or a leading one.
        while prefix.ends_with('/') {
            prefix.pop();
        }
        Self {
            store,
            prefix,
            authority: std::sync::OnceLock::new(),
            legacy_encoding: LegacyEncoding::Plain,
        }
    }

    /// Declare the encoding of an existing legacy log. This changes no persisted
    /// bytes and does not migrate or convert a log. Plain is the default.
    pub fn with_legacy_encoding(mut self, encoding: LegacyEncoding) -> Self {
        self.legacy_encoding = encoding;
        self
    }

    fn decode_legacy(&self, bytes: &Bytes) -> Result<(u64, Bytes), StorageError> {
        match self.legacy_encoding {
            LegacyEncoding::Plain => Ok((0, bytes.clone())),
            LegacyEncoding::GenerationPrefixed if bytes.len() >= 8 => Ok(decode_fenced(bytes)),
            LegacyEncoding::GenerationPrefixed => Err(StorageError::Corruption(
                "manifest: truncated declared generation header".into(),
            )),
        }
    }

    /// Object path of the manifest for `version`. Zero-padded to 20 digits (covers all
    /// of `u64`) so the lexical order returned by `list` matches numeric version order.
    fn manifest_path(&self, version: u64) -> Path {
        Path::from(format!("{}/v{version:020}.manifest", self.prefix))
    }

    /// Parse the version out of a manifest object key's final segment, ignoring any
    /// other objects that happen to live under the prefix.
    fn parse_version(name: &str) -> Option<u64> {
        name.strip_prefix('v')?
            .strip_suffix(".manifest")?
            .parse::<u64>()
            .ok()
    }

    /// The highest committed version, or `None` if the log is empty.
    pub async fn latest_version(&self) -> Result<Option<u64>, StorageError> {
        if let Some(head) = self.load_head().await? {
            return Ok(head.record.version());
        }
        self.latest_legacy_version().await
    }

    async fn latest_legacy_version(&self) -> Result<Option<u64>, StorageError> {
        Ok(self.legacy_versions().await?.last().copied())
    }

    async fn legacy_versions(&self) -> Result<Vec<u64>, StorageError> {
        let prefix = Path::from(self.prefix.as_str());
        let metas = self.store.list(Some(&prefix)).await?;
        let mut versions: Vec<_> = metas
            .into_iter()
            .filter_map(|m| {
                let version = Self::parse_version(m.location.filename()?)?;
                (m.location == self.store.full_path(&self.manifest_path(version)))
                    .then_some(version)
            })
            .collect();
        versions.sort_unstable();
        versions.dedup();
        Ok(versions)
    }

    /// Read the raw manifest bytes for a specific `version`.
    pub async fn read_manifest(&self, version: u64) -> Result<Bytes, StorageError> {
        if let Some(head) = self.load_head().await? {
            return Ok(self.read_versioned(head, version).await?.1);
        }
        self.read_legacy_manifest(version).await
    }

    async fn read_legacy_manifest(&self, version: u64) -> Result<Bytes, StorageError> {
        self.store.get(&self.manifest_path(version)).await
    }

    /// Atomically publish `manifest` as the successor of `parent` (`None` ⇒ the first
    /// commit, version `0`).
    ///
    /// Returns [`CommitOutcome::Committed`] with the new version on success, or
    /// [`CommitOutcome::Conflict`] (carrying the current latest version) if another
    /// committer already claimed the target slot. Real I/O errors propagate as `Err`.
    pub async fn commit(
        &self,
        parent: Option<u64>,
        manifest: Bytes,
    ) -> Result<CommitOutcome, StorageError> {
        if let Some(head) = self.load_head().await? {
            return self.commit_versioned(head, parent, 0, manifest).await;
        }
        self.commit_legacy(parent, None, manifest).await
    }

    async fn commit_legacy(
        &self,
        parent: Option<u64>,
        generation: Option<u64>,
        payload: Bytes,
    ) -> Result<CommitOutcome, StorageError> {
        let versions = self.legacy_versions().await?;
        let latest = versions.last().copied();
        if parent != latest {
            return Ok(CommitOutcome::Conflict { latest });
        }
        if versions
            .iter()
            .enumerate()
            .any(|(index, version)| u64::try_from(index).ok() != Some(*version))
        {
            return Err(StorageError::TransactionCommitFailed(
                "manifest: legacy history has gaps; migrate to versioned authority before writing"
                    .into(),
            ));
        }
        if generation.is_some() != (self.legacy_encoding == LegacyEncoding::GenerationPrefixed) {
            return Err(StorageError::Serialization(
                "manifest: write operation disagrees with declared legacy encoding".into(),
            ));
        }
        if let (Some(latest), Some(generation)) = (latest, generation) {
            // This is the SAME tip whose parent and continuity were validated.
            let (existing, _) = self.decode_legacy(&self.read_legacy_manifest(latest).await?)?;
            if generation < existing {
                return Ok(CommitOutcome::Conflict {
                    latest: Some(latest),
                });
            }
        }
        let manifest = match generation {
            Some(g) => encode_fenced(g, &payload),
            None => payload,
        };
        let target = match parent {
            Some(p) => p.checked_add(1).ok_or_else(|| {
                StorageError::Serialization("manifest: version counter overflow".into())
            })?,
            None => 0,
        };
        match self
            .store
            .put_if_absent(&self.manifest_path(target), manifest)
            .await
        {
            Ok(()) => Ok(CommitOutcome::Committed(target)),
            Err(StorageError::AlreadyExists(_)) => Ok(CommitOutcome::Conflict {
                latest: self.latest_legacy_version().await?,
            }),
            Err(other) => Err(other),
        }
    }

    /// Commit `payload` as a **generation-fenced** successor of `parent` (TD-117/TD-119).
    ///
    /// On top of the version CAS in [`commit`](Self::commit), this fences a *stale
    /// writer*: a `generation` strictly lower than the generation embedded in the latest
    /// committed manifest is rejected with [`CommitOutcome::Conflict`] **before** the slot
    /// is claimed. This is the object-store analog of Neon's `index_part.json` +
    /// generation single-writer guarantee — a resurrected/forked writer carrying an old
    /// generation cannot clobber a branch that a newer writer has taken over.
    ///
    /// Legacy logs store an unmarked 8-byte generation header; arbitrary plain
    /// payloads of eight or more bytes cannot be distinguished from that format.
    /// They do NOT transparently upgrade. The opt-in head format stores generation
    /// separately and checks generation/parent at the conditional update boundary.
    /// Legacy prechecks do not prevent a delayed writer from reusing a pruned slot.
    pub async fn commit_fenced(
        &self,
        parent: Option<u64>,
        generation: u64,
        payload: Bytes,
    ) -> Result<CommitOutcome, StorageError> {
        if let Some(head) = self.load_head().await? {
            return self
                .commit_versioned(head, parent, generation, payload)
                .await;
        }
        self.commit_legacy(parent, Some(generation), payload).await
    }

    /// Read a generation-fenced manifest, returning `(generation, payload)`.
    /// Legacy interpretation is explicitly selected by `with_legacy_encoding`:
    /// Plain returns generation zero without stripping bytes; GenerationPrefixed
    /// requires at least eight header bytes. No content-based codec guessing.
    pub async fn read_fenced(&self, version: u64) -> Result<(u64, Bytes), StorageError> {
        if let Some(head) = self.load_head().await? {
            return self.read_versioned(head, version).await;
        }
        self.decode_legacy(&self.read_legacy_manifest(version).await?)
    }

    /// Best-effort retention of versioned history. Legacy logs always return an
    /// error without deleting anything, including empty logs; migrate explicitly
    /// before pruning. This trades metadata growth for preventing slot reuse.
    ///
    /// # Safety
    ///
    /// The versioned head and marker are never deleted. Historical reads are not
    /// pinned. An archived version becomes eligible when
    /// it is **both**:
    ///
    /// - ranked `keep_k` or more behind the tip (rank is the position in the sorted
    ///   version list, **not** `tip - v` arithmetic, so a log with gaps from a prior
    ///   partial prune is still ranked correctly), **and**
    /// - at least `min_age` old (a grace window that protects the recent burst and the
    ///   tombstone a release publishes at `generation + 1`).
    ///
    /// `keep_k` is clamped to [`MIN_PRUNE_KEEP_K`]. A future-dated `last_modified`
    /// (cloud clock skew) is treated as not-yet-of-age and never reaped early. Deletes
    /// are best-effort: a transient error — including [`StorageError::NotFound`] for an
    /// object a concurrent pass already removed — is logged and the pass continues. A
    /// crash mid-pass leaves the log half-pruned; the next pass recomputes from a fresh
    /// `list`. Returns the number of objects deleted.
    pub async fn prune_retention(
        &self,
        keep_k: usize,
        min_age: std::time::Duration,
    ) -> Result<usize, StorageError> {
        if let Some(head) = self.load_head().await? {
            return self.prune_versioned(head, keep_k, min_age).await;
        }
        Err(StorageError::TransactionCommitFailed(
            "manifest: legacy retention is disabled; migrate before pruning".into(),
        ))
    }
}

/// Prepend the 8-byte big-endian `generation` header to `payload`.
fn encode_fenced(generation: u64, payload: &[u8]) -> Bytes {
    let mut buf = Vec::with_capacity(8 + payload.len());
    buf.extend_from_slice(&generation.to_be_bytes());
    buf.extend_from_slice(payload);
    Bytes::from(buf)
}

/// Split a fenced manifest into `(generation, payload)`. Bytes too short to carry a
/// header decode as generation `0` with the whole buffer as payload (back-compat).
fn decode_fenced(bytes: &Bytes) -> (u64, Bytes) {
    if bytes.len() < 8 {
        return (0, bytes.clone());
    }
    let mut header = [0u8; 8];
    header.copy_from_slice(&bytes[..8]);
    (u64::from_be_bytes(header), bytes.slice(8..))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream::BoxStream;
    use object_store::ObjectStoreExt;
    use object_store::memory::InMemory;
    use object_store::{
        CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
        PutMultipartOptions, PutOptions, PutPayload, PutResult,
    };
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::Notify;

    /// All operations delegate to the same real in-memory object store. Only
    /// this client's PUT is suspended, after the committer has validated state.
    #[derive(Debug)]
    struct PausedPutStore {
        inner: Arc<InMemory>,
        reached_put: Notify,
        resume_put: Notify,
    }

    impl std::fmt::Display for PausedPutStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("PausedPutStore")
        }
    }

    #[async_trait::async_trait]
    impl ObjectStore for PausedPutStore {
        async fn put_opts(
            &self,
            path: &Path,
            payload: PutPayload,
            options: PutOptions,
        ) -> object_store::Result<PutResult> {
            if path.filename() == Some("_publication.head") {
                self.reached_put.notify_one();
                self.resume_put.notified().await;
            }
            self.inner.put_opts(path, payload, options).await
        }
        async fn put_multipart_opts(
            &self,
            path: &Path,
            options: PutMultipartOptions,
        ) -> object_store::Result<Box<dyn MultipartUpload>> {
            self.inner.put_multipart_opts(path, options).await
        }
        async fn get_opts(
            &self,
            path: &Path,
            options: GetOptions,
        ) -> object_store::Result<GetResult> {
            self.inner.get_opts(path, options).await
        }
        fn delete_stream(
            &self,
            paths: BoxStream<'static, object_store::Result<Path>>,
        ) -> BoxStream<'static, object_store::Result<Path>> {
            self.inner.delete_stream(paths)
        }
        fn list(
            &self,
            prefix: Option<&Path>,
        ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
            self.inner.list(prefix)
        }
        async fn list_with_delimiter(
            &self,
            prefix: Option<&Path>,
        ) -> object_store::Result<ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }
        async fn copy_opts(
            &self,
            from: &Path,
            to: &Path,
            options: CopyOptions,
        ) -> object_store::Result<()> {
            self.inner.copy_opts(from, to, options).await
        }
    }

    fn committer() -> ManifestCommitter {
        ManifestCommitter::new(
            ProximaObjectStore::new(Arc::new(InMemory::new())),
            "data/t/ns/_manifests",
        )
    }

    /// An empty log has no latest version; the first commit lands at v0 and is readable.
    #[tokio::test]
    async fn first_commit_is_version_zero() {
        let c = committer();
        assert_eq!(c.latest_version().await.unwrap(), None);

        let out = c
            .commit(None, Bytes::from_static(b"snapshot-0"))
            .await
            .unwrap();
        assert_eq!(out, CommitOutcome::Committed(0));
        assert_eq!(c.latest_version().await.unwrap(), Some(0));
        assert_eq!(
            c.read_manifest(0).await.unwrap(),
            Bytes::from_static(b"snapshot-0")
        );
    }

    /// Sequential commits advance the version monotonically and each is independently readable.
    #[tokio::test]
    async fn sequential_commits_advance_version() {
        let c = committer();
        assert_eq!(
            c.commit(None, Bytes::from_static(b"s0")).await.unwrap(),
            CommitOutcome::Committed(0)
        );
        assert_eq!(
            c.commit(Some(0), Bytes::from_static(b"s1")).await.unwrap(),
            CommitOutcome::Committed(1)
        );
        assert_eq!(
            c.commit(Some(1), Bytes::from_static(b"s2")).await.unwrap(),
            CommitOutcome::Committed(2)
        );
        assert_eq!(c.latest_version().await.unwrap(), Some(2));
        assert_eq!(c.read_manifest(1).await.unwrap(), Bytes::from_static(b"s1"));
    }

    /// Two committers racing from the SAME parent: one wins the slot, the other is told
    /// it conflicted and given the latest version to rebase onto. The winner's bytes
    /// are never clobbered.
    #[tokio::test]
    async fn concurrent_commit_from_same_parent_conflicts() {
        let c = committer();
        c.commit(None, Bytes::from_static(b"s0")).await.unwrap();

        // Both attempt to publish v1 as the successor of v0.
        let winner = c
            .commit(Some(0), Bytes::from_static(b"winner"))
            .await
            .unwrap();
        let loser = c
            .commit(Some(0), Bytes::from_static(b"loser"))
            .await
            .unwrap();

        assert_eq!(winner, CommitOutcome::Committed(1));
        assert_eq!(loser, CommitOutcome::Conflict { latest: Some(1) });
        // The slot holds the winner's manifest, untouched by the loser.
        assert_eq!(
            c.read_manifest(1).await.unwrap(),
            Bytes::from_static(b"winner")
        );
    }

    /// A trailing slash in the prefix must not change where manifests land.
    #[tokio::test]
    async fn prefix_trailing_slash_is_normalized() {
        let c = ManifestCommitter::new(
            ProximaObjectStore::new(Arc::new(InMemory::new())),
            "a/b/_manifests/",
        );
        assert_eq!(
            c.commit(None, Bytes::from_static(b"x")).await.unwrap(),
            CommitOutcome::Committed(0)
        );
        assert_eq!(
            c.manifest_path(0).as_ref(),
            "a/b/_manifests/v00000000000000000000.manifest"
        );
        assert_eq!(c.latest_version().await.unwrap(), Some(0));
    }

    /// Non-manifest objects sharing the prefix are ignored by version discovery.
    #[tokio::test]
    async fn unrelated_objects_under_prefix_are_ignored() {
        let store = ProximaObjectStore::new(Arc::new(InMemory::new()));
        store
            .put(
                &Path::from("data/t/ns/_manifests/README.txt"),
                Bytes::from_static(b"hi"),
            )
            .await
            .unwrap();
        let c = ManifestCommitter::new(store, "data/t/ns/_manifests");
        assert_eq!(
            c.latest_version().await.unwrap(),
            None,
            "non-manifest objects don't count"
        );
        assert_eq!(
            c.commit(None, Bytes::from_static(b"s0")).await.unwrap(),
            CommitOutcome::Committed(0)
        );
        assert_eq!(c.latest_version().await.unwrap(), Some(0));
    }

    /// A stale writer (generation lower than the latest committed) is fenced before it
    /// can claim a slot; a current/newer generation commits and round-trips.
    #[tokio::test]
    async fn fenced_commit_rejects_stale_generation() {
        let c = committer().with_legacy_encoding(LegacyEncoding::GenerationPrefixed);
        // Generation 5 takes ownership of the log at v0.
        assert_eq!(
            c.commit_fenced(None, 5, Bytes::from_static(b"g5"))
                .await
                .unwrap(),
            CommitOutcome::Committed(0)
        );
        let (generation, payload) = c.read_fenced(0).await.unwrap();
        assert_eq!(generation, 5);
        assert_eq!(payload, Bytes::from_static(b"g5"));

        // A resurrected writer with an older generation is fenced; no slot is claimed.
        assert_eq!(
            c.commit_fenced(Some(0), 3, Bytes::from_static(b"stale"))
                .await
                .unwrap(),
            CommitOutcome::Conflict { latest: Some(0) }
        );
        assert_eq!(c.latest_version().await.unwrap(), Some(0));

        // The current generation (>= latest) advances the log normally.
        assert_eq!(
            c.commit_fenced(Some(0), 5, Bytes::from_static(b"g5-next"))
                .await
                .unwrap(),
            CommitOutcome::Committed(1)
        );
        assert_eq!(c.read_fenced(1).await.unwrap().0, 5);
    }

    /// A plain (unfenced) manifest decodes as generation 0 for forward compatibility.
    #[tokio::test]
    async fn plain_commit_decodes_as_generation_zero() {
        let c = committer();
        c.commit(None, Bytes::from_static(b"plain")).await.unwrap();
        assert_eq!(c.read_fenced(0).await.unwrap().0, 0);
    }

    // ---- prune_retention ----

    async fn assert_paused_publication_conflicts(successor_generation: u64) {
        let (current, backend) = committer_with_store().await;
        seed_fenced(&current, 1, 5).await;
        let paused_store = Arc::new(PausedPutStore {
            inner: backend,
            reached_put: Notify::new(),
            resume_put: Notify::new(),
        });
        let delayed = ManifestCommitter::new(
            ProximaObjectStore::new(paused_store.clone()),
            "data/t/ns/_manifests",
        );

        let (_, outcome) = tokio::time::timeout(Duration::from_secs(10), async {
            tokio::join!(
                async {
                    paused_store.reached_put.notified().await;
                    for version in 1..=5 {
                        assert_eq!(
                            current
                                .commit_fenced(
                                    Some(version - 1),
                                    successor_generation,
                                    Bytes::from_static(b"current")
                                )
                                .await
                                .unwrap(),
                            CommitOutcome::Committed(version)
                        );
                    }
                    assert_eq!(current.prune_retention(2, Duration::ZERO).await.unwrap(), 4);
                    paused_store.resume_put.notify_one();
                },
                delayed.commit_fenced(Some(0), 5, Bytes::from_static(b"delayed")),
            )
        })
        .await
        .expect("paused publication schedule must finish");

        assert_eq!(current.latest_version().await.unwrap(), Some(5));
        // Its exact receipt was pruned. A conditional error may follow a
        // client's successful retry, so the protocol must report uncertainty
        // while proving that this delayed request cannot alter authority.
        assert!(
            matches!(outcome, Err(StorageError::TransactionCommitFailed(ref e)) if e.contains("indeterminate")),
            "{outcome:?}"
        );
        assert_eq!(
            current.read_fenced(5).await.unwrap(),
            (successor_generation, Bytes::from_static(b"current"))
        );
        assert!(matches!(
            current.read_manifest(1).await,
            Err(StorageError::NotFound(_))
        ));
    }

    /// A preflight latest-parent check cannot fix a PUT suspended across GC.
    #[tokio::test]
    async fn lease_invariant_prune_during_validated_put_cannot_reuse_slot() {
        assert_paused_publication_conflicts(5).await;
    }

    /// A generation check before PUT cannot fence a writer suspended across
    /// both a successor's generation change and subsequent retention.
    #[tokio::test]
    async fn lease_invariant_takeover_during_validated_put_fences_old_generation() {
        assert_paused_publication_conflicts(6).await;
    }

    /// A deleted successor is not an available CAS slot: the caller's parent
    /// remains stale even when its generation still matches the current owner.
    #[tokio::test]
    async fn lease_invariant_pruning_does_not_revalidate_stale_parent() {
        let (current, backend) = committer_with_store().await;
        let delayed =
            ManifestCommitter::new(ProximaObjectStore::new(backend), "data/t/ns/_manifests");
        let tip = seed_fenced(&current, 6, 5).await;
        assert_eq!(current.prune_retention(2, Duration::ZERO).await.unwrap(), 4);
        assert!(matches!(
            current.read_manifest(1).await,
            Err(StorageError::NotFound(_))
        ));

        let outcome = delayed
            .commit_fenced(Some(0), 5, Bytes::from_static(b"stale checkpoint"))
            .await
            .unwrap();

        assert_eq!(current.latest_version().await.unwrap(), Some(tip));
        assert_eq!(outcome, CommitOutcome::Conflict { latest: Some(tip) });
        assert!(matches!(
            current.read_manifest(1).await,
            Err(StorageError::NotFound(_))
        ));
    }

    #[tokio::test]
    async fn lease_invariant_pruning_does_not_revalidate_empty_parent() {
        let (c, _) = committer_with_store().await;
        let tip = seed_fenced(&c, 6, 5).await;
        c.prune_retention(2, Duration::ZERO).await.unwrap();

        let outcome = c
            .commit(None, Bytes::from_static(b"late initial commit"))
            .await
            .unwrap();

        assert_eq!(outcome, CommitOutcome::Conflict { latest: Some(tip) });
        assert_eq!(c.latest_version().await.unwrap(), Some(tip));
    }

    #[tokio::test]
    async fn lease_invariant_nonexistent_parent_cannot_advance_head() {
        let c = committer();
        c.commit(None, Bytes::from_static(b"initial"))
            .await
            .unwrap();

        let outcome = c
            .commit(Some(100), Bytes::from_static(b"not based on head"))
            .await
            .unwrap();

        assert_eq!(outcome, CommitOutcome::Conflict { latest: Some(0) });
        assert_eq!(c.latest_version().await.unwrap(), Some(0));
    }

    /// The documented plain-to-fenced compatibility must hold beyond the
    /// existing five-byte fixture; ordinary JSON is longer than eight bytes.
    #[tokio::test]
    async fn lease_invariant_plain_payload_is_not_a_generation_header() {
        let c = committer();
        let payload = Bytes::from_static(br#"{"checkpoint":"source-cursor"}"#);
        c.commit(None, payload.clone()).await.unwrap();

        assert_eq!(c.read_fenced(0).await.unwrap(), (0, payload));
    }

    async fn committer_with_store() -> (ManifestCommitter, Arc<InMemory>) {
        let backend = Arc::new(InMemory::new());
        let c = ManifestCommitter::create_versioned(
            ProximaObjectStore::new(backend.clone()),
            "data/t/ns/_manifests",
        )
        .await
        .unwrap();
        (c, backend)
    }

    /// Seed `n` fenced commits at a fixed generation; returns the final version.
    pub(super) async fn seed_fenced(c: &ManifestCommitter, n: u64, generation: u64) -> u64 {
        let mut parent: Option<u64> = None;
        let mut last = 0;
        for v in 0..n {
            assert_eq!(
                c.commit_fenced(parent, generation, Bytes::from_static(b"g"))
                    .await
                    .unwrap(),
                CommitOutcome::Committed(v)
            );
            parent = Some(v);
            last = v;
        }
        last
    }

    /// Pruning deletes only the stale tail; the tip is always retained and readable.
    #[tokio::test]
    async fn prune_never_deletes_tip() {
        let (c, _backend) = committer_with_store().await;
        let tip = seed_fenced(&c, 50, 5).await;
        assert_eq!(tip, 49);

        let deleted = c.prune_retention(10, Duration::ZERO).await.unwrap();
        assert_eq!(deleted, 40, "keep newest 10 of 50");
        assert_eq!(c.latest_version().await.unwrap(), Some(49));
        let (tip_gen, _bytes) = c.read_fenced(49).await.unwrap();
        assert_eq!(tip_gen, 5, "tip still decodes after prune");
    }

    /// After pruning, the owner can still commit a fenced successor — the tip's
    /// generation header survives. This is the load-bearing fence test.
    #[tokio::test]
    async fn prune_preserves_fenced_commit_after_prune() {
        let (c, _backend) = committer_with_store().await;
        let tip = seed_fenced(&c, 50, 5).await;
        assert_eq!(c.prune_retention(5, Duration::ZERO).await.unwrap(), 45);

        let out = c
            .commit_fenced(Some(tip), 5, Bytes::from_static(b"next"))
            .await
            .unwrap();
        assert_eq!(out, CommitOutcome::Committed(50));
    }

    /// A stale writer (generation below the tip's) is still fenced after pruning.
    #[tokio::test]
    async fn prune_then_stale_writer_still_fenced() {
        let (c, _backend) = committer_with_store().await;
        let tip = seed_fenced(&c, 50, 5).await;
        c.prune_retention(5, Duration::ZERO).await.unwrap();

        let out = c
            .commit_fenced(Some(tip), 4, Bytes::from_static(b"stale"))
            .await
            .unwrap();
        // Fenced at the generation check *before* any slot is claimed — the tip (still
        // 49 after pruning) is returned, and no v50 is created.
        assert_eq!(out, CommitOutcome::Conflict { latest: Some(tip) });
        assert_eq!(c.latest_version().await.unwrap(), Some(tip));
    }

    /// `min_age` grants recent history a grace window: a large `min_age` reaps nothing,
    /// then dropping it to zero reaps the stale tail while the keep window stays intact.
    #[tokio::test]
    async fn prune_respects_min_age_grace_window() {
        let (c, _backend) = committer_with_store().await;
        seed_fenced(&c, 50, 5).await;

        // Everything was written moments ago → all within the grace window.
        assert_eq!(
            c.prune_retention(2, Duration::from_secs(3600))
                .await
                .unwrap(),
            0,
            "min_age=1h keeps the recent burst"
        );
        // Drop the age gate; keep only the newest 2.
        assert_eq!(
            c.prune_retention(2, Duration::ZERO).await.unwrap(),
            48,
            "keep newest 2 of 50"
        );
        assert_eq!(c.latest_version().await.unwrap(), Some(49));
        assert!(c.read_manifest(48).await.is_ok(), "predecessor retained");
    }

    /// Rank is the position in the sorted version list, so a log with gaps (from a
    /// prior partial prune) is still ranked correctly — not by `tip - v` arithmetic.
    #[tokio::test]
    async fn prune_rank_robust_to_gaps() {
        let (c, backend) = committer_with_store().await;
        seed_fenced(&c, 50, 5).await;
        // Simulate a prior partial prune that already removed v25.
        let gap_path = Path::from("data/t/ns/_manifests/_history/v00000000000000000025.snapshot");
        backend.delete(&gap_path).await.unwrap();
        assert!(c.read_manifest(25).await.is_err());

        // 49 remaining; keep newest 5 by rank (v45..v49) → delete 44.
        let deleted = c.prune_retention(5, Duration::ZERO).await.unwrap();
        assert_eq!(deleted, 44);
        for v in 45..50u64 {
            assert!(c.read_manifest(v).await.is_ok(), "v{v} retained");
        }
        assert_eq!(c.latest_version().await.unwrap(), Some(49));
    }

    /// Pruning concurrently with commits never breaks the fence: the commit outcome is
    /// always well-formed and the tip stays readable.
    #[tokio::test]
    async fn prune_concurrent_with_commit_fenced_is_safe() {
        let (c, _backend) = committer_with_store().await;
        let mut tip = seed_fenced(&c, 40, 5).await;

        for _ in 0..20 {
            let (pruned, committed) = tokio::join!(
                c.prune_retention(5, Duration::ZERO),
                c.commit_fenced(Some(tip), 5, Bytes::from_static(b"r")),
            );
            let pruned = pruned.unwrap();
            let committed = committed.unwrap();
            assert!(
                matches!(committed, CommitOutcome::Committed(_)),
                "commit must not error under concurrent prune (got {committed:?}, pruned {pruned})"
            );
            if let CommitOutcome::Committed(v) = committed {
                tip = v;
            }
        }
        assert_eq!(c.latest_version().await.unwrap(), Some(tip));
        let (tip_gen, _) = c.read_fenced(tip).await.unwrap();
        assert_eq!(tip_gen, 5);
    }

    /// An empty log is a no-op (never errors, deletes nothing).
    #[tokio::test]
    async fn prune_empty_log_is_noop() {
        let (c, _backend) = committer_with_store().await;
        assert_eq!(
            c.prune_retention(10, Duration::from_secs(60))
                .await
                .unwrap(),
            0
        );
        assert_eq!(c.latest_version().await.unwrap(), None);
    }

    /// `keep_k` below the floor is clamped, not honored — the log never collapses past
    /// the tip-plus-predecessor window.
    #[tokio::test]
    async fn prune_clamps_keep_k_below_minimum() {
        let (c, _backend) = committer_with_store().await;
        seed_fenced(&c, 10, 5).await;
        // keep_k=0 → clamped to MIN_PRUNE_KEEP_K (2): delete 8, keep newest 2.
        let deleted = c.prune_retention(0, Duration::ZERO).await.unwrap();
        assert_eq!(deleted, 8);
        assert_eq!(c.latest_version().await.unwrap(), Some(9));
        assert!(c.read_manifest(8).await.is_ok(), "predecessor retained");
        assert!(c.read_manifest(7).await.is_err(), "v7 reaped");
    }
}
