//! Pluggable durable home for the system catalog's snapshot blob (Phase 5c of
//! the two-tier operator/account system-catalog redesign).
//!
//! The catalog's bounded-restart machinery (Phase 3) periodically writes a
//! point-in-time snapshot of the in-RAM authority, then compacts the WAL up to
//! that watermark. This trait abstracts **only** the snapshot blob's durable
//! read/write so the snapshot can live on the local filesystem (the default,
//! behaviour-identical to before) or on object storage
//! (`s3://`/`gs://`/`az://`/`memory://`) under the tenant/operator `DrPathBuilder`
//! prefix — without touching the crash-consistency ordering or the local WAL.
//!
//! Object-store-*native* WAL durability (per-DDL delta objects) is deferred to
//! Phase 6, where that delta-object log doubles as the multi-pod CAS substrate;
//! building it here would mean building it twice. Until then an object-store
//! deployment keeps its per-DDL WAL on the local working volume and replicates
//! the snapshot to object storage.

use anyhow::{Context, Result};
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

use proximadb_iceberg_engine::manifest::{CommitOutcome, ManifestCommitter};
use proximadb_object_store::ProximaObjectStore;

/// A snapshot blob read back from the durable store, with the pointer metadata
/// the Phase 6b cross-pod coherence protocol compares against.
#[derive(Debug, Clone)]
pub struct SnapshotRead {
    /// Monotonic commit **version** of the snapshot pointer — it advances on
    /// *every* publish, including same-owner re-checkpoints. This is the cheap
    /// sinval staleness key a follower tails: a strictly higher version means
    /// the published snapshot changed and the in-RAM cache must reload. `0` for
    /// the unversioned local single-pod store.
    pub version: u64,
    /// Fencing **generation** the snapshot was committed under — it changes only
    /// when a newer pod *takes the catalog over* (Phase 6a). A generation above
    /// this pod's own write generation means it has been superseded. `0` for the
    /// unfenced local store.
    pub generation: u64,
    /// The snapshot payload.
    pub bytes: Vec<u8>,
}

/// Outcome of publishing a snapshot blob.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotWrite {
    /// The blob landed at this monotonic pointer `version`.
    Published { version: u64 },
    /// **Fenced** — a newer writer (higher generation) already owns the pointer,
    /// so this stale writer must step down rather than overwrite it.
    Fenced,
}

/// Durable, atomically-replaceable home for the catalog snapshot blob.
///
/// `write_atomic` must guarantee a reader concurrent with the write observes
/// either the previous blob or the complete new one — never a torn write. This
/// is the load-bearing property the Phase-3 crash-consistency ordering relies
/// on (snapshot made durable *before* the WAL is compacted).
///
/// Writes carry a monotonic **fencing generation** (Phase 6a). A local store
/// ignores it (single-pod, no contention). An object-store store routes the
/// write through a generation-fenced commit so a *stale* writer — one whose
/// generation is lower than a newer pod that has taken the catalog over — is
/// rejected instead of clobbering the newer pod's snapshot.
#[async_trait]
pub trait CatalogSnapshotStore: Send + Sync {
    /// Whether this snapshot is the only durable copy that survives loss of the
    /// catalog WAL's local volume. Object-store deployments keep a local WAL as
    /// a write-ahead staging log, but that WAL is not durable across VM/pod
    /// replacement. They therefore must publish a snapshot for every committed
    /// DDL before acknowledging it. Local stores return `false`: their WAL is
    /// already on the durable filesystem and snapshots remain a bounded-restart
    /// optimization.
    fn requires_snapshot_on_commit(&self) -> bool {
        false
    }

    /// Read the current snapshot blob with its pointer version + fencing
    /// generation, or `None` if none has been written yet.
    async fn read(&self) -> Result<Option<SnapshotRead>>;

    /// Durably + atomically publish the snapshot blob stamped with `generation`,
    /// returning the [`SnapshotWrite`] outcome (the new pointer version, or
    /// `Fenced` when a newer generation already owns the pointer). Local stores
    /// never fence and always [`Published`](SnapshotWrite::Published).
    async fn write_atomic(&self, generation: u64, bytes: &[u8]) -> Result<SnapshotWrite>;

    /// Cheap probe of the latest published pointer version **without** fetching
    /// the (potentially large) snapshot payload — the Phase 6b sinval staleness
    /// check polls this, and only fetches via [`read`](Self::read) on a version
    /// advance. `None` when nothing has been published yet, or when the store
    /// has no version concept (the local single-pod store, which followers never
    /// tail). Default: `None`.
    async fn latest_version(&self) -> Result<Option<u64>> {
        Ok(None)
    }

    /// Best-effort retention prune of superseded snapshot manifests. Default no-op —
    /// the local single-pod store keeps one rotating blob. The object-store
    /// implementation prunes only explicitly migrated versioned history; legacy
    /// pruning errors without deleting source objects. See
    /// [`ManifestCommitter::prune_retention`]. Returns the count pruned.
    async fn prune_retention(
        &self,
        _keep_k: usize,
        _min_age: std::time::Duration,
    ) -> Result<usize> {
        Ok(0)
    }

    /// Human-readable location, for logs and error context.
    fn describe(&self) -> String;
}

/// Local-filesystem snapshot store: temp file → fsync → atomic rename →
/// best-effort parent-dir fsync. The default; behaviour-identical to the
/// pre-5c inline `write_atomic`.
pub struct LocalSnapshotStore {
    path: PathBuf,
}

impl LocalSnapshotStore {
    /// Snapshot blob at `path` (conventionally the WAL path with a `.snapshot`
    /// extension).
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// The blob path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[async_trait]
impl CatalogSnapshotStore for LocalSnapshotStore {
    async fn read(&self) -> Result<Option<SnapshotRead>> {
        if tokio::fs::try_exists(&self.path).await.unwrap_or(false) {
            let bytes = tokio::fs::read(&self.path)
                .await
                .with_context(|| format!("reading catalog snapshot {}", self.path.display()))?;
            // Local single-pod store is unversioned + unfenced — (0, 0).
            Ok(Some(SnapshotRead {
                version: 0,
                generation: 0,
                bytes,
            }))
        } else {
            Ok(None)
        }
    }

    async fn write_atomic(&self, _generation: u64, bytes: &[u8]) -> Result<SnapshotWrite> {
        let path = &self.path;
        let tmp = path.with_extension("snapshot-tmp");
        {
            let mut file = tokio::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&tmp)
                .await
                .with_context(|| format!("opening snapshot temp {}", tmp.display()))?;
            file.write_all(bytes)
                .await
                .with_context(|| format!("writing snapshot temp {}", tmp.display()))?;
            file.flush().await?;
            file.sync_data()
                .await
                .with_context(|| format!("fsync snapshot temp {}", tmp.display()))?;
        }
        tokio::fs::rename(&tmp, path)
            .await
            .with_context(|| format!("atomically replacing snapshot {}", path.display()))?;
        if let Some(dir) = path.parent()
            && let Ok(handle) = std::fs::File::open(dir)
        {
            let _ = handle.sync_all();
        }
        // The local store is unversioned (single-pod, last-writer-wins).
        Ok(SnapshotWrite::Published { version: 0 })
    }

    fn describe(&self) -> String {
        format!("file:{}", self.path.display())
    }
}

/// Object-store snapshot store with **generation-fencing** (Phase 6a).
///
/// Persists the catalog snapshot as a generation-fenced versioned manifest via
/// [`ManifestCommitter`] (the same `commit_fenced` primitive the warehouse
/// manifest log uses). A write carries the writer's monotonic generation; a
/// *stale* writer — one whose generation is below the generation a newer pod
/// already committed — is rejected (`Conflict`) instead of clobbering the newer
/// pod's snapshot. This is the multi-pod single-writer guarantee (the Neon
/// `index_part.json` + generation model). The per-DDL WAL stays local; only the
/// snapshot pointer is fenced here.
///
/// Works for `memory://` (deterministic tests) and, behind the
/// `proximadb-object-store` crate's cloud features, `s3://`/`gs://`/`az://`.
pub struct ObjectStoreSnapshotStore {
    committer: ManifestCommitter,
    label: String,
}

impl ObjectStoreSnapshotStore {
    /// Build from an object-store base URL (e.g. `s3://bucket/prefix`,
    /// `memory:///`) and the relative prefix that holds the catalog snapshot's
    /// fenced manifest log (e.g. `_operator/catalog/_manifests/`).
    pub fn from_url(base_url: &str, manifests_prefix: impl Into<String>) -> Result<Self> {
        let store = ProximaObjectStore::from_url(base_url)
            .with_context(|| format!("opening object store at {base_url}"))?;
        Ok(Self::new(store, base_url, manifests_prefix))
    }

    /// Build from an already-open store, its base URL (for labelling), and the
    /// fenced-manifest-log prefix. Used by tests and by callers that share one
    /// backing store across catalog opens.
    pub fn new(
        store: ProximaObjectStore,
        base_url: &str,
        manifests_prefix: impl Into<String>,
    ) -> Self {
        let manifests_prefix = manifests_prefix.into();
        let label = format!("{base_url}::{manifests_prefix}");
        Self {
            committer: ManifestCommitter::new(store, manifests_prefix).with_legacy_encoding(
                proximadb_iceberg_engine::manifest::LegacyEncoding::GenerationPrefixed,
            ),
            label,
        }
    }
}

#[async_trait]
impl CatalogSnapshotStore for ObjectStoreSnapshotStore {
    fn requires_snapshot_on_commit(&self) -> bool {
        true
    }

    async fn read(&self) -> Result<Option<SnapshotRead>> {
        match self
            .committer
            .latest_version()
            .await
            .with_context(|| format!("reading catalog snapshot log {}", self.label))?
        {
            Some(version) => {
                let (generation, payload) =
                    self.committer.read_fenced(version).await.with_context(|| {
                        format!("reading catalog snapshot {}@{version}", self.label)
                    })?;
                Ok(Some(SnapshotRead {
                    version,
                    generation,
                    bytes: payload.to_vec(),
                }))
            }
            // No manifest yet → empty-catalog boot; the caller replays the WAL.
            None => Ok(None),
        }
    }

    async fn write_atomic(&self, generation: u64, bytes: &[u8]) -> Result<SnapshotWrite> {
        let parent = self
            .committer
            .latest_version()
            .await
            .with_context(|| format!("reading catalog snapshot head {}", self.label))?;
        match self
            .committer
            .commit_fenced(parent, generation, bytes::Bytes::copy_from_slice(bytes))
            .await
            .with_context(|| format!("committing catalog snapshot {}", self.label))?
        {
            CommitOutcome::Committed(version) => {
                // TD-MANIFEST-1: catalog snapshots are large full-catalog blobs and
                // were never pruned, so the manifest log grows one-per-checkpoint
                // forever. Prune the stale tail opportunistically after each publish
                // (checkpoints are infrequent, so this is cheap and avoids a separate
                // background task). `keep_k` is small — followers tail the tip, never
                // history. Best-effort: a prune failure must not fail the publish.
                if catalog_snapshot_retention_enabled() {
                    let min_age = std::time::Duration::from_secs(catalog_snapshot_min_age_secs());
                    if let Err(e) = self
                        .prune_retention(catalog_snapshot_keep_k(), min_age)
                        .await
                    {
                        tracing::warn!(
                            error = %e,
                            "catalog snapshot prune failed (best-effort); publish unaffected"
                        );
                    }
                }
                Ok(SnapshotWrite::Published { version })
            }
            // Fenced (stale generation) or lost the version CAS race — either way
            // this writer did not publish; it must re-read and step down/retry.
            CommitOutcome::Conflict { .. } => Ok(SnapshotWrite::Fenced),
        }
    }

    async fn latest_version(&self) -> Result<Option<u64>> {
        self.committer
            .latest_version()
            .await
            .with_context(|| format!("probing catalog snapshot version {}", self.label))
    }

    async fn prune_retention(&self, keep_k: usize, min_age: std::time::Duration) -> Result<usize> {
        self.committer
            .prune_retention(keep_k, min_age)
            .await
            .with_context(|| format!("pruning catalog snapshot log {}", self.label))
    }

    fn describe(&self) -> String {
        self.label.clone()
    }
}

// ---- TD-MANIFEST-1: catalog snapshot manifest retention knobs (parsed per write) ----
// Catalog snapshots are large full-catalog blobs, so keep only a tiny history (the
// tip + a couple of predecessors); followers tail the tip, never old snapshots.

/// Whether catalog snapshot manifest pruning runs (default ON). Set
/// `PROXIMADB_CATALOG_SNAPSHOT_RETENTION=0` to disable.
fn catalog_snapshot_retention_enabled() -> bool {
    std::env::var("PROXIMADB_CATALOG_SNAPSHOT_RETENTION")
        .ok()
        .and_then(|v| v.parse::<bool>().ok())
        .unwrap_or(true)
}

/// Number of newest catalog snapshots to keep (the committer clamps this to ≥2).
fn catalog_snapshot_keep_k() -> usize {
    std::env::var("PROXIMADB_CATALOG_SNAPSHOT_KEEP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(3)
}

/// Minimum age (seconds) a superseded snapshot must reach before it is pruned.
fn catalog_snapshot_min_age_secs() -> u64 {
    std::env::var("PROXIMADB_CATALOG_SNAPSHOT_MIN_AGE_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(900)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(opt: Option<SnapshotRead>) -> Option<Vec<u8>> {
        opt.map(|r| r.bytes)
    }

    fn published(w: SnapshotWrite) -> bool {
        matches!(w, SnapshotWrite::Published { .. })
    }

    #[tokio::test]
    async fn catalog_serving_handle_continues_after_explicit_manifest_migration() {
        let backing =
            ProximaObjectStore::new(std::sync::Arc::new(object_store::memory::InMemory::new()));
        let prefix = "catalog/migration";
        let serving = ObjectStoreSnapshotStore::new(backing.clone(), "memory:///", prefix);
        assert!(published(serving.write_atomic(5, b"before").await.unwrap()));
        assert!(
            serving
                .prune_retention(2, std::time::Duration::ZERO)
                .await
                .is_err()
        );
        // All writers and pruners are quiescent during this explicit cutover.
        ManifestCommitter::new(backing.clone(), prefix)
            .with_legacy_encoding(
                proximadb_iceberg_engine::manifest::LegacyEncoding::GenerationPrefixed,
            )
            .migrate_versioned(0)
            .await
            .unwrap();
        assert_eq!(serving.read().await.unwrap().unwrap().bytes, b"before");
        assert!(published(serving.write_atomic(6, b"after").await.unwrap()));
        assert!(!published(serving.write_atomic(5, b"stale").await.unwrap()));
        let reopened = ObjectStoreSnapshotStore::new(backing, "memory:///", prefix);
        let current = reopened.read().await.unwrap().unwrap();
        assert_eq!((current.version, current.generation), (1, 6));
        assert_eq!(current.bytes, b"after");
    }

    #[tokio::test]
    async fn local_store_round_trips_and_reports_absence() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("cat.snapshot");
        let store = LocalSnapshotStore::new(&path);

        // Absent before any write; local store exposes no version concept.
        assert!(store.read().await.unwrap().is_none());
        assert!(store.latest_version().await.unwrap().is_none());

        // Local store is unfenced: it ignores the generation and never fences.
        assert!(published(
            store.write_atomic(0, b"hello-catalog").await.unwrap()
        ));
        let read = store.read().await.unwrap().unwrap();
        assert_eq!(read.version, 0);
        assert_eq!(read.generation, 0);
        assert_eq!(read.bytes, b"hello-catalog");

        // Atomic replace — even an "older" generation still writes locally.
        assert!(published(store.write_atomic(0, b"v2").await.unwrap()));
        assert_eq!(
            body(store.read().await.unwrap()).as_deref(),
            Some(&b"v2"[..])
        );
    }

    #[tokio::test]
    async fn object_store_round_trips_and_reports_absence() {
        // `memory://` exercises the real ManifestCommitter commit/read path
        // deterministically (no cloud creds).
        let store =
            ObjectStoreSnapshotStore::from_url("memory:///", "_operator/catalog/_manifests/")
                .expect("memory store");

        assert!(store.read().await.unwrap().is_none());
        assert!(store.latest_version().await.unwrap().is_none());

        assert_eq!(
            store.write_atomic(1, b"snap-bytes").await.unwrap(),
            SnapshotWrite::Published { version: 0 }
        );
        let read = store.read().await.unwrap().unwrap();
        assert_eq!(read.version, 0);
        assert_eq!(read.generation, 1);
        assert_eq!(read.bytes, b"snap-bytes");
        assert_eq!(store.latest_version().await.unwrap(), Some(0));

        // A same-or-higher generation succeeds, advances the version, and supersedes.
        assert_eq!(
            store.write_atomic(1, b"snap-bytes-2").await.unwrap(),
            SnapshotWrite::Published { version: 1 }
        );
        assert_eq!(
            body(store.read().await.unwrap()).as_deref(),
            Some(&b"snap-bytes-2"[..])
        );
        assert_eq!(store.latest_version().await.unwrap(), Some(1));
    }

    /// Phase 6a multi-instance fence: two stores over ONE shared backing object
    /// store (two pods). After a newer-generation writer takes over, the older
    /// (stale) writer's commit is fenced — it cannot clobber the newer snapshot.
    #[tokio::test]
    async fn fenced_commit_rejects_stale_writer() {
        use object_store::memory::InMemory;
        use std::sync::Arc;

        // One backing store shared by both "pods" (a fresh `from_url("memory://")`
        // would be a different, empty store — we need the same instance).
        let backing: Arc<dyn object_store::ObjectStore> = Arc::new(InMemory::new());
        let prefix = "_operator/catalog/_manifests/";

        let pod_a = ObjectStoreSnapshotStore::new(
            ProximaObjectStore::new(backing.clone()),
            "memory:///",
            prefix,
        );
        let pod_b = ObjectStoreSnapshotStore::new(
            ProximaObjectStore::new(backing.clone()),
            "memory:///",
            prefix,
        );

        // Pod A owns generation 1 and commits.
        assert!(published(pod_a.write_atomic(1, b"from-A").await.unwrap()));

        // Pod B takes over: it reads the current generation (1) and claims the
        // next one (2), then commits successfully (2 >= 1).
        let read = pod_b.read().await.unwrap().unwrap();
        assert_eq!(read.generation, 1);
        assert!(published(pod_b.write_atomic(2, b"from-B").await.unwrap()));

        // Pod A is now stale (generation 1 < 2). Its next commit is FENCED.
        assert_eq!(
            pod_a.write_atomic(1, b"from-A-again").await.unwrap(),
            SnapshotWrite::Fenced,
            "stale generation-1 writer must be fenced once generation 2 owns the log"
        );

        // The newer pod's snapshot is intact — A did not clobber it.
        let read = pod_b.read().await.unwrap().unwrap();
        assert_eq!(read.generation, 2);
        assert_eq!(read.bytes, b"from-B");
    }
}
