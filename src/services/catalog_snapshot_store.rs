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

/// TD-SPINE-2: outcome of committing a publish's account shards.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShardCommit {
    /// Every shard landed; carries each account's NEW manifest-log version
    /// for the root index the caller is about to commit.
    Committed(std::collections::HashMap<String, u64>),
    /// At least one shard log was claimed or fenced by a newer writer
    /// (higher generation). The caller must NOT publish a root referencing
    /// versions it does not own — step down and reload instead. Shards that
    /// did land are harmless: the root still names their predecessors.
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

    // ── TD-SPINE-2 (ADR-098 D1.3): per-account sharded snapshot publish/read ──
    //
    // The whole-image `read`/`write_atomic` surface above stays exactly as it
    // is (it is the local-store path AND the kill-switch escape path). The
    // sharded surface below splits one whole-image blob into a small ROOT
    // manifest log (global registries + the `account → shard log version`
    // index) plus one SHARD manifest log per account, each a sibling of the
    // whole-image log, committed through the same generation-fenced
    // [`ManifestCommitter`]. Defaults make every non-object store (local fs,
    // injected test stores) report "unsupported", which the catalog reads as
    // "keep the whole-image path".

    /// Whether this store implements the sharded publish/read surface.
    fn supports_sharded_publish(&self) -> bool {
        false
    }

    /// Read the current ROOT record (pointer version + fencing generation +
    /// envelope-framed body), or `None` when no root has been published yet
    /// (fresh deployment, or a pre-TD-SPINE-2 deployment still on the
    /// whole-image log — mixed-read).
    async fn read_root(&self) -> Result<Option<SnapshotRead>> {
        Ok(None)
    }

    /// Cheap probe of the latest published ROOT pointer version (the sharded
    /// sinval key followers tail) without fetching any payload.
    async fn latest_root_version(&self) -> Result<Option<u64>> {
        Ok(None)
    }

    /// Commit the given account shards at `generation`. Each shard is its own
    /// manifest log (CAS on the successor slot, fenced on generation), so a
    /// stale writer is rejected before any root is moved. Returns the new
    /// per-account versions, or [`ShardCommit::Fenced`] when any shard log
    /// was claimed/fenced by a newer writer (the caller must step down, not
    /// publish a root referencing versions it does not own).
    async fn commit_shards(
        &self,
        _generation: u64,
        _shards: &[(String, Vec<u8>)],
    ) -> Result<ShardCommit> {
        anyhow::bail!("store does not support sharded catalog snapshots")
    }

    /// Commit the ROOT record at `generation` (carrying the caller's
    /// post-shard-commit index) as the successor of `expected_parent` — the
    /// root version the caller CHECKED (`None` = the caller saw no root).
    /// TD-SPINE-2 R4: the parent is never re-read here; a root committed by
    /// another owner after the caller's check moves the tip past
    /// `expected_parent`, and the CAS refuses. Fenced/conflicting outcomes
    /// surface as [`SnapshotWrite::Fenced`], exactly like [`Self::write_atomic`].
    async fn commit_root(
        &self,
        _generation: u64,
        _expected_parent: Option<u64>,
        _root_bytes: &[u8],
    ) -> Result<SnapshotWrite> {
        anyhow::bail!("store does not support sharded catalog snapshots")
    }

    /// Read one account's current shard record (its manifest log's tip), or
    /// `None` when that account has never published a shard.
    async fn read_shard(&self, _account: &str) -> Result<Option<SnapshotRead>> {
        anyhow::bail!("store does not support sharded catalog snapshots")
    }

    /// Read exactly the shard version named by a published catalog root.
    async fn read_shard_version(&self, _account: &str, _version: u64) -> Result<SnapshotRead> {
        anyhow::bail!("store does not support sharded catalog snapshots")
    }

    /// TD-PATH-2 4a: read the durable identity-high-water sidecar, if this
    /// store keeps one and it has been written. The sidecar is a sibling of
    /// the snapshot pointer, deliberately OUTSIDE every path a snapshot
    /// publisher renames or prunes, so an older (pre-high-water) binary
    /// checkpointing this store cannot destroy the dropped-id floor.
    /// Default: no sidecar support (`None`).
    async fn read_identity_high_water(&self) -> Result<Option<Vec<u8>>> {
        Ok(None)
    }

    /// TD-PATH-2 4a: durably write the identity-high-water sidecar. Callers
    /// merge the current image before writing so the sidecar never lowers.
    /// Default: unsupported — and LOUD about it, so a store implementation
    /// that forgets to override this is observable in the logs rather than
    /// silently leaving the snapshot field as the only floor record.
    async fn write_identity_high_water(&self, _bytes: &[u8]) -> Result<()> {
        tracing::warn!(
            store = %self.describe(),
            "CatalogSnapshotStore implementation does not support the identity \
             high-water sidecar; dropping the write (the in-snapshot field \
             remains the only durable floor record)"
        );
        Ok(())
    }

    /// Human-readable location, for logs and error context.
    fn describe(&self) -> String;
}

/// Local-filesystem snapshot store: temp file → fsync → atomic rename →
/// best-effort parent-dir fsync. The default; behaviour-identical to the
/// pre-5c inline `write_atomic`.
pub struct LocalSnapshotStore {
    path: PathBuf,
    /// TD-PATH-2 4a: the identity-high-water sidecar beside the snapshot blob
    /// (`<stem>.hw` for the conventional `<stem>.snapshot` blob). A
    /// pre-high-water binary writes only `path` and `path`'s `-tmp` sibling,
    /// so this file survives its checkpoint.
    hw_path: PathBuf,
}

impl LocalSnapshotStore {
    /// Snapshot blob at `path` (conventionally the WAL path with a `.snapshot`
    /// extension).
    pub fn new(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let hw_path = path.with_extension("hw");
        Self { path, hw_path }
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

    async fn read_identity_high_water(&self) -> Result<Option<Vec<u8>>> {
        match tokio::fs::try_exists(&self.hw_path).await {
            Ok(true) => {
                let bytes = tokio::fs::read(&self.hw_path).await.with_context(|| {
                    format!(
                        "reading identity high-water sidecar {}",
                        self.hw_path.display()
                    )
                })?;
                Ok(Some(bytes))
            }
            Ok(false) => Ok(None),
            // Review R2 NIT-2: an FS error is NOT proven absence. Treating it
            // as absent under-counts the floor (losing DROPPED-id memory until
            // the next checkpoint rewrite repairs the file), so say so loudly
            // instead of folding silence into `None`.
            Err(error) => {
                tracing::warn!(
                    path = %self.hw_path.display(),
                    error = %error,
                    "probing identity high-water sidecar existence failed; treating \
                     it as absent (dropped-id memory may under-count until the \
                     next checkpoint rewrite)"
                );
                Ok(None)
            }
        }
    }

    async fn write_identity_high_water(&self, bytes: &[u8]) -> Result<()> {
        // Same atomic pattern as `write_atomic`: tmp → fsync → rename →
        // best-effort parent-dir fsync. The tmp name shares no path with the
        // snapshot's own `-tmp` sibling, so the two writers never collide. A
        // failure on any arm removes the tmp file (review R2 NIT-3): `.hw-tmp`
        // orphans are harmless today — extension-filtered out of every list —
        // but leaving none is tidier than relying on the filter.
        let tmp = self.hw_path.with_extension("hw-tmp");
        let written = async {
            {
                let mut file = tokio::fs::OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(&tmp)
                    .await
                    .with_context(|| format!("opening sidecar temp {}", tmp.display()))?;
                file.write_all(bytes)
                    .await
                    .with_context(|| format!("writing sidecar temp {}", tmp.display()))?;
                file.flush().await?;
                file.sync_data()
                    .await
                    .with_context(|| format!("fsync sidecar temp {}", tmp.display()))?;
            }
            tokio::fs::rename(&tmp, &self.hw_path)
                .await
                .with_context(|| {
                    format!(
                        "atomically replacing identity high-water sidecar {}",
                        self.hw_path.display()
                    )
                })?;
            Ok(())
        }
        .await;
        if let Err(error) = written {
            let _ = tokio::fs::remove_file(&tmp).await;
            return Err(error);
        }
        if let Some(dir) = self.hw_path.parent()
            && let Ok(handle) = std::fs::File::open(dir)
        {
            let _ = handle.sync_all();
        }
        Ok(())
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
    /// TD-SPINE-2: the ROOT manifest log — a sibling of the whole-image log
    /// (`_operator/catalog/_root/` in the canonical wiring, derived via
    /// `DrPathBuilder` so the two can never drift; pinned by test).
    root_committer: ManifestCommitter,
    /// TD-PATH-2 4a: the same backing store the committer resolves relative
    /// paths against, used for the sidecar object beside the manifest log.
    sidecar_store: ProximaObjectStore,
    sidecar_key: String,
    label: String,
}

/// TD-PATH-2 4a: the sidecar key is a SIBLING of the manifest-log prefix —
/// `_operator/catalog/_manifests/` → `_operator/catalog/identity_high_water`.
/// Sibling, never inside the prefix: the manifest log's retention prune
/// deletes only its own canonical `{prefix}/_history/` archives, and an older
/// (pre-high-water) binary's publish/prune never names a key outside its
/// manifest log — so the sidecar survives every publisher that does not know
/// it exists.
fn sibling_sidecar_key(manifests_prefix: &str) -> String {
    let trimmed = manifests_prefix.trim_end_matches('/');
    match trimmed.rfind('/') {
        Some(idx) => format!("{}/identity_high_water", &trimmed[..idx]),
        None => "identity_high_water".to_string(),
    }
}

/// TD-SPINE-2: read a generation-fenced manifest log's TIP as a
/// [`SnapshotRead`] — the shared body of `read_root` and `read_shard`.
/// `None` when the log is still empty (a fresh account, or a pre-TD-SPINE-2
/// deployment with no root yet).
async fn read_log_tip(
    committer: &ManifestCommitter,
    what: &str,
    label: &str,
) -> Result<Option<SnapshotRead>> {
    match committer
        .latest_version()
        .await
        .with_context(|| format!("reading catalog snapshot {what} log {label}"))?
    {
        Some(version) => {
            let (generation, payload) = committer
                .read_fenced(version)
                .await
                .with_context(|| format!("reading catalog snapshot {what} {label}@{version}"))?;
            Ok(Some(SnapshotRead {
                version,
                generation,
                bytes: payload.to_vec(),
            }))
        }
        None => Ok(None),
    }
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
        let sidecar_key = sibling_sidecar_key(&manifests_prefix);
        // TD-SPINE-2: the root log is a DrPathBuilder-derived sibling of the
        // whole-image log (canonical equivalence pinned by test).
        let root_prefix = crate::storage::trait_components::path_resolver::DrPathBuilder::system_catalog_root_manifests_subprefix_from(&manifests_prefix);
        Self {
            committer: ManifestCommitter::new(store.clone(), manifests_prefix.clone())
                .with_legacy_encoding(
                    proximadb_iceberg_engine::manifest::LegacyEncoding::GenerationPrefixed,
                ),
            root_committer: ManifestCommitter::new(store.clone(), root_prefix)
                .with_legacy_encoding(
                    proximadb_iceberg_engine::manifest::LegacyEncoding::GenerationPrefixed,
                ),
            sidecar_store: store,
            sidecar_key,
            label,
        }
    }

    /// TD-SPINE-2 R1: one account's shard manifest log, at
    /// [`catalog_shard_manifests_prefix`]. Fresh per call: a committer is a
    /// thin (store-clone + prefix) handle, and accounts are unbounded.
    fn shard_committer(&self, account: &str) -> Result<ManifestCommitter> {
        let prefix = catalog_shard_manifests_prefix(account)?;
        Ok(
            ManifestCommitter::new(self.sidecar_store.clone(), prefix).with_legacy_encoding(
                proximadb_iceberg_engine::manifest::LegacyEncoding::GenerationPrefixed,
            ),
        )
    }
}

/// TD-SPINE-2 R1 (round-1 LOW-1/MED-1): the account ids among `accounts`
/// whose shard cannot be placed injectively — not a valid tenant segment, or
/// equal to another account up to ASCII case (one directory on a
/// case-insensitive store). Sorted, for a stable error message.
pub fn catalog_shard_placement_violations(
    accounts: &std::collections::HashSet<String>,
) -> Vec<String> {
    let mut by_fold: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for account in accounts {
        *by_fold.entry(account.to_ascii_lowercase()).or_default() += 1;
    }
    let mut violations: Vec<String> = accounts
        .iter()
        .filter(|account| {
            catalog_shard_manifests_prefix(account).is_err()
                || by_fold
                    .get(&account.to_ascii_lowercase())
                    .is_some_and(|count| *count > 1)
        })
        .cloned()
        .collect();
    violations.sort();
    violations
}

/// TD-SPINE-2 R1 (ADR-098 D1.3): the object-store prefix of one account's
/// catalog shard manifest log.
///
/// * A tenant's shard lives under that tenant's `DrPathBuilder` prefix:
///   `data/{tenant_id}/_syscat/shard/` — catalog isolation is structural
///   (CLAUDE.md co-design mandate 3), not a naming convention inside a shared
///   operator prefix.
/// * The untenanted shard ([`DEFAULT_SHARD_KEY`](crate::services::system_catalog_state::DEFAULT_SHARD_KEY),
///   namespaces created with no tenant) has no tenant prefix to live under;
///   it is operator-scoped beside the root:
///   `_operator/catalog/_untenanted_shard/`.
///
/// Injective: `_default` is the only key mapped outside `data/`, and
/// `SystemCatalog` refuses it as a namespace tenant id; every other key must
/// be a valid `DrPathBuilder` tenant segment and maps to itself. A key that
/// is not a valid segment fails closed with an actionable error — it is
/// never escaped or re-encoded into a prefix another tenant could own.
pub fn catalog_shard_manifests_prefix(account: &str) -> Result<String> {
    use crate::storage::trait_components::path_resolver::DrPathBuilder;
    if account == crate::services::system_catalog_state::DEFAULT_SHARD_KEY {
        return Ok(DrPathBuilder::system_catalog_untenanted_shard_manifests_subprefix());
    }
    DrPathBuilder::system_catalog_tenant_shard_manifests_subprefix(account).map_err(|e| {
        anyhow::anyhow!(
            "catalog shard for account '{account}' cannot be placed under a tenant \
             prefix ({e}); the account id must be a valid DrPathBuilder tenant segment"
        )
    })
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

    fn supports_sharded_publish(&self) -> bool {
        true
    }

    async fn read_root(&self) -> Result<Option<SnapshotRead>> {
        read_log_tip(&self.root_committer, "root", &self.label).await
    }

    async fn latest_root_version(&self) -> Result<Option<u64>> {
        self.root_committer
            .latest_version()
            .await
            .with_context(|| format!("probing catalog snapshot root version {}", self.label))
    }

    async fn commit_shards(
        &self,
        generation: u64,
        shards: &[(String, Vec<u8>)],
    ) -> Result<ShardCommit> {
        // Resolve every placement before the first PUT, so an unplaceable
        // account never leaves a half-written publish behind.
        let committers = shards
            .iter()
            .map(|(account, _)| self.shard_committer(account))
            .collect::<Result<Vec<_>>>()?;
        let mut versions = std::collections::HashMap::with_capacity(shards.len());
        for ((account, bytes), committer) in shards.iter().zip(committers) {
            let parent = committer.latest_version().await.with_context(|| {
                format!("reading catalog shard '{account}' head at {}", self.label)
            })?;
            match committer
                .commit_fenced(parent, generation, bytes::Bytes::from(bytes.clone()))
                .await
                .with_context(|| {
                    format!("committing catalog shard '{account}' at {}", self.label)
                })? {
                CommitOutcome::Committed(version) => {
                    versions.insert(account.clone(), version);
                }
                // Fenced on the shard (stale generation) or lost the successor
                // slot to a same-or-newer writer: either way this writer does
                // not own that shard's tip, so it must not publish a root
                // naming it. Shards that already landed are harmless — the
                // root still references their predecessors.
                CommitOutcome::Conflict { .. } => return Ok(ShardCommit::Fenced),
            }
        }
        Ok(ShardCommit::Committed(versions))
    }

    async fn commit_root(
        &self,
        generation: u64,
        expected_parent: Option<u64>,
        root_bytes: &[u8],
    ) -> Result<SnapshotWrite> {
        // TD-SPINE-2 R4: CAS against the CALLER's checked parent, never a
        // fresh `latest_version()` read — a root another owner committed
        // after the caller's coverage check must refuse this commit
        // (`commit_fenced` returns `Conflict` when `parent != latest`).
        match self
            .root_committer
            .commit_fenced(
                expected_parent,
                generation,
                bytes::Bytes::copy_from_slice(root_bytes),
            )
            .await
            .with_context(|| format!("committing catalog snapshot root {}", self.label))?
        {
            CommitOutcome::Committed(version) => Ok(SnapshotWrite::Published { version }),
            CommitOutcome::Conflict { .. } => Ok(SnapshotWrite::Fenced),
        }
    }

    async fn read_shard(&self, account: &str) -> Result<Option<SnapshotRead>> {
        read_log_tip(&self.shard_committer(account)?, account, &self.label).await
    }

    async fn read_shard_version(&self, account: &str, version: u64) -> Result<SnapshotRead> {
        let prefix = catalog_shard_manifests_prefix(account)?;
        let (generation, bytes) = match self.shard_committer(account)?.read_fenced(version).await {
            Ok(read) => read,
            // TD-SPINE-2 R1: fail closed, loudly. The root names this exact
            // version, so its absence is never "an empty account". The likely
            // cause after R1 is a store written before the relocation (shards
            // under the retired `_operator/catalog/_shards/`), which this
            // reader deliberately does not consult (pre-GA clean break).
            Err(proximadb_kernel::error::StorageError::NotFound(_)) => anyhow::bail!(
                "catalog root names shard v{version} of account '{account}', but it is \
                 missing at '{prefix}' ({}). Shards live under the tenant prefix since \
                 TD-SPINE-2 R1; a store written before R1 (shards under \
                 '_operator/catalog/_shards/') is not read. Re-seed the catalog from the \
                 legacy whole-image log or restore the object store; refusing to serve \
                 the account without its catalog rows",
                self.label
            ),
            Err(e) => {
                return Err(e).with_context(|| {
                    format!(
                        "reading catalog shard '{account}' at {}@{version} ({prefix})",
                        self.label
                    )
                });
            }
        };
        Ok(SnapshotRead {
            version,
            generation,
            bytes: bytes.to_vec(),
        })
    }

    async fn read_identity_high_water(&self) -> Result<Option<Vec<u8>>> {
        let path = object_store::path::Path::from(self.sidecar_key.as_str());
        match self.sidecar_store.get(&path).await {
            Ok(bytes) => Ok(Some(bytes.to_vec())),
            Err(proximadb_kernel::error::StorageError::NotFound(_)) => Ok(None),
            Err(e) => Err(e).with_context(|| {
                format!(
                    "reading identity high-water sidecar {} at {}",
                    self.sidecar_key, self.label
                )
            }),
        }
    }

    async fn write_identity_high_water(&self, bytes: &[u8]) -> Result<()> {
        // TD-PATH-2 review NIT-4: this PUT is deliberately UNFENCED
        // (last-writer-wins, unlike the generation-fenced snapshot commit).
        // Fencing it would let a stale owner's rejected write strand a HIGHER
        // floor; accepting the race is safe because every writer merges the
        // on-disk image first (fetch-max) and the snapshot field independently
        // carries the floor — so a stale owner can only transiently lower the
        // on-DISK sidecar, never the recovered floor.
        let path = object_store::path::Path::from(self.sidecar_key.as_str());
        self.sidecar_store
            .put(&path, bytes::Bytes::copy_from_slice(bytes))
            .await
            .with_context(|| {
                format!(
                    "writing identity high-water sidecar {} at {}",
                    self.sidecar_key, self.label
                )
            })
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
    use crate::storage::trait_components::path_resolver::DrPathBuilder;

    fn body(opt: Option<SnapshotRead>) -> Option<Vec<u8>> {
        opt.map(|r| r.bytes)
    }

    fn published(w: SnapshotWrite) -> bool {
        matches!(w, SnapshotWrite::Published { .. })
    }

    /// TD-PATH-2 4a: the sidecar key is a SIBLING of the manifest-log prefix —
    /// never inside it — so no snapshot publisher or retention prune that only
    /// names the manifest log can touch the sidecar.
    #[test]
    fn sidecar_key_is_a_sibling_of_the_manifests_prefix() {
        assert_eq!(
            sibling_sidecar_key("_operator/catalog/_manifests/"),
            "_operator/catalog/identity_high_water"
        );
        assert_eq!(
            sibling_sidecar_key("_operator/catalog/_manifests"),
            "_operator/catalog/identity_high_water"
        );
        // A single-segment prefix degrades to a root sibling, still outside
        // the prefix.
        assert_eq!(sibling_sidecar_key("_manifests/"), "identity_high_water");
    }

    /// TD-PATH-2 4a: an older binary's snapshot publish must not disturb the
    /// sidecar — locally, `write_atomic` renames only its own blob + tmp, so
    /// the `catalog.hw` sibling survives any number of publishes.
    #[tokio::test]
    async fn local_sidecar_survives_snapshot_publishes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = LocalSnapshotStore::new(dir.path().join("catalog.snapshot"));

        // New binary: sidecar + snapshot coexist.
        assert!(store.write_identity_high_water(b"hw-v1").await.is_ok());
        assert!(published(store.write_atomic(0, b"snap").await.unwrap()));
        assert_eq!(
            store.read_identity_high_water().await.unwrap().as_deref(),
            Some(&b"hw-v1"[..])
        );

        // Old-binary simulation: snapshot publishes (and its tmp rename)
        // repeat; the sidecar bytes are untouched.
        assert!(published(
            store.write_atomic(0, b"snap-old-1").await.unwrap()
        ));
        assert!(published(
            store.write_atomic(0, b"snap-old-2").await.unwrap()
        ));
        assert_eq!(
            store.read_identity_high_water().await.unwrap().as_deref(),
            Some(&b"hw-v1"[..])
        );
        assert_eq!(
            body(store.read().await.unwrap()).as_deref(),
            Some(&b"snap-old-2"[..])
        );
    }

    /// TD-PATH-2 4a: the object-store sidecar round-trips at its sibling key
    /// and is absent before the first write (mixed-read no-op).
    #[tokio::test]
    async fn object_store_sidecar_round_trips_beside_the_manifest_log() {
        let backing =
            ProximaObjectStore::new(std::sync::Arc::new(object_store::memory::InMemory::new()));
        let store =
            ObjectStoreSnapshotStore::new(backing, "memory:///", "_operator/catalog/_manifests/");

        assert!(store.read_identity_high_water().await.unwrap().is_none());
        assert!(store.write_identity_high_water(b"floor").await.is_ok());
        assert_eq!(
            store.read_identity_high_water().await.unwrap().as_deref(),
            Some(&b"floor"[..])
        );

        // Snapshot publishing (manifest log commits) does not disturb it.
        assert!(published(store.write_atomic(1, b"snap").await.unwrap()));
        assert_eq!(
            store.read_identity_high_water().await.unwrap().as_deref(),
            Some(&b"floor"[..])
        );
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

    // ── TD-SPINE-2: sharded surface (root + per-account shard logs) ─────────

    /// The sharded surface round-trips over the real commit protocol: shards
    /// commit through their own CAS logs, the root through its, `read_root`
    /// / `read_shard` return the tips, and a STALE generation is fenced on
    /// BOTH surfaces (a superseded pod can neither move a shard nor the
    /// root). The legacy whole-image log is untouched by the sharded path.
    #[tokio::test]
    async fn sharded_surface_round_trips_and_fences_stale_writers() -> Result<()> {
        let backing =
            ProximaObjectStore::new(std::sync::Arc::new(object_store::memory::InMemory::new()));
        let store =
            ObjectStoreSnapshotStore::new(backing, "memory:///", "_operator/catalog/_manifests/");
        assert!(store.supports_sharded_publish());

        // Empty: no root, no shard.
        assert!(store.read_root().await?.is_none());
        assert!(store.read_shard("acme").await?.is_none());

        // A newer pod (generation 2) owns the logs and publishes.
        let committed = match store
            .commit_shards(
                2,
                &[
                    ("acme".to_string(), b"acme-shard-v1".to_vec()),
                    ("globex".to_string(), b"globex-shard-v1".to_vec()),
                ],
            )
            .await?
        {
            ShardCommit::Committed(versions) => versions,
            ShardCommit::Fenced => panic!("first shard commit must land"),
        };
        assert_eq!(committed["acme"], 0);
        assert_eq!(committed["globex"], 0);
        assert_eq!(
            store.commit_root(2, None, b"root-v1").await?,
            SnapshotWrite::Published { version: 0 }
        );
        let root = store.read_root().await?.expect("root");
        assert_eq!((root.version, root.generation), (0, 2));
        assert_eq!(root.bytes, b"root-v1");
        let acme = store.read_shard("acme").await?.expect("acme shard");
        assert_eq!(acme.version, 0);
        assert_eq!(acme.generation, 2);
        assert_eq!(acme.bytes, b"acme-shard-v1");

        // A stale writer (generation 1) is fenced on BOTH surfaces, and the
        // stale shard commit is rejected BEFORE any root moves.
        assert_eq!(
            store
                .commit_shards(1, &[("acme".to_string(), b"stale".to_vec())])
                .await?,
            ShardCommit::Fenced
        );
        assert_eq!(
            store.commit_root(1, Some(0), b"stale-root").await?,
            SnapshotWrite::Fenced
        );
        assert_eq!(
            store.read_shard("acme").await?.unwrap().bytes,
            b"acme-shard-v1"
        );
        assert_eq!(store.read_root().await?.unwrap().bytes, b"root-v1");

        // The legacy whole-image log is untouched by the sharded path.
        assert!(store.read().await?.is_none());
        Ok(())
    }

    /// TD-SPINE-2 R1: each tenant's shard log lives under THAT tenant's
    /// DrPathBuilder prefix (`data/{tenant}/_syscat/shard/`), the untenanted
    /// shard beside the root under `_operator/catalog/`, the root at
    /// `_operator/catalog/_root/` — and nothing under the retired
    /// `_operator/catalog/_shards/`, nor inside the legacy `_manifests/` log.
    #[tokio::test]
    async fn shard_objects_land_under_their_tenant_prefix_root_stays_operator_scoped() -> Result<()>
    {
        use futures::TryStreamExt;
        use object_store::ObjectStore;

        let inner = std::sync::Arc::new(object_store::memory::InMemory::new());
        let backing = ProximaObjectStore::new(inner.clone());
        let store = ObjectStoreSnapshotStore::new(
            backing,
            "memory:///",
            DrPathBuilder::system_catalog_manifests_subprefix(),
        );
        store
            .commit_shards(
                1,
                &[
                    ("tenant-a".to_string(), b"a".to_vec()),
                    ("tenant-b".to_string(), b"b".to_vec()),
                    (
                        crate::services::system_catalog_state::DEFAULT_SHARD_KEY.to_string(),
                        b"d".to_vec(),
                    ),
                ],
            )
            .await?;
        store.commit_root(1, None, b"r").await?;

        // Flat (recursive) list — `list_with_delimiter` would stop at the
        // top-level common prefixes.
        let mut keys: Vec<String> = inner
            .list(None)
            .map_ok(|meta| meta.location.to_string())
            .try_collect()
            .await?;
        keys.sort();
        assert_eq!(
            keys,
            vec![
                "_operator/catalog/_root/v00000000000000000000.manifest".to_string(),
                "_operator/catalog/_untenanted_shard/v00000000000000000000.manifest".to_string(),
                "data/tenant-a/_syscat/shard/v00000000000000000000.manifest".to_string(),
                "data/tenant-b/_syscat/shard/v00000000000000000000.manifest".to_string(),
            ]
        );
        assert!(
            keys.iter()
                .all(|k| !k.starts_with("_operator/catalog/_shards/"))
        );
        assert_eq!(
            catalog_shard_manifests_prefix("tenant-a")?,
            DrPathBuilder::build_tenant_system(None, "tenant-a")?.system_catalog_shard_subprefix()
        );
        // The shard reads back from the relocated log.
        assert_eq!(store.read_shard_version("tenant-a", 0).await?.bytes, b"a");
        Ok(())
    }

    /// TD-SPINE-2 R1: an account id that is not a valid tenant path segment
    /// fails closed on commit (no object is written anywhere) instead of
    /// being escaped into a prefix another tenant could own.
    #[tokio::test]
    async fn shard_commit_refuses_an_account_that_is_not_a_tenant_segment() -> Result<()> {
        use futures::TryStreamExt;
        use object_store::ObjectStore;

        let inner = std::sync::Arc::new(object_store::memory::InMemory::new());
        let store = ObjectStoreSnapshotStore::new(
            ProximaObjectStore::new(inner.clone()),
            "memory:///",
            DrPathBuilder::system_catalog_manifests_subprefix(),
        );
        let error = store
            .commit_shards(
                1,
                &[
                    // A valid account FIRST: it must not be written either
                    // (every placement resolves before the first PUT).
                    ("tenant-a".to_string(), b"a".to_vec()),
                    ("tenant one/x".to_string(), b"s".to_vec()),
                ],
            )
            .await
            .expect_err("a separator-bearing account must be refused");
        assert!(
            format!("{error:#}").contains("cannot be placed under a tenant prefix"),
            "{error:#}"
        );
        let keys: Vec<_> = inner.list(None).try_collect().await?;
        assert!(keys.is_empty(), "nothing may be written: {keys:?}");
        Ok(())
    }

    /// TD-SPINE-2 R1: a root that names a shard version missing at the
    /// relocated prefix fails closed with an actionable error — even when a
    /// pre-R1 copy of that shard still sits under the retired
    /// `_operator/catalog/_shards/{hex}/` layout (no dual read).
    #[tokio::test]
    async fn missing_relocated_shard_fails_closed_and_ignores_the_retired_layout() -> Result<()> {
        use object_store::ObjectStore;
        use object_store::ObjectStoreExt as _;

        let inner = std::sync::Arc::new(object_store::memory::InMemory::new());
        // A pre-R1 shard object for `acme` at the retired location
        // (`hex("acme")` = 61636d65), generation-prefixed like the real log.
        let mut legacy = 1u64.to_be_bytes().to_vec();
        legacy.extend_from_slice(b"acme-shard");
        inner
            .put(
                &object_store::path::Path::from(
                    "_operator/catalog/_shards/61636d65/v00000000000000000000.manifest",
                ),
                legacy.into(),
            )
            .await?;
        let store = ObjectStoreSnapshotStore::new(
            ProximaObjectStore::new(inner.clone() as std::sync::Arc<dyn ObjectStore>),
            "memory:///",
            DrPathBuilder::system_catalog_manifests_subprefix(),
        );
        let error = store
            .read_shard_version("acme", 0)
            .await
            .expect_err("a root-named shard missing at the new location must fail closed");
        let text = format!("{error:#}");
        assert!(text.contains("data/acme/_syscat/shard/"), "{text}");
        assert!(text.contains("TD-SPINE-2 R1"), "{text}");
        assert!(text.contains("Re-seed"), "{text}");
        assert!(store.read_shard("acme").await?.is_none());
        Ok(())
    }

    /// Round-1 MED-1/LOW-1: placement violations name every unplaceable id
    /// (invalid segment) and every case-fold collision, and nothing else.
    #[test]
    fn placement_violations_list_invalid_and_case_colliding_ids() {
        let accounts: std::collections::HashSet<String> =
            ["tenantA", "tenanta", "bad tenant", "ok", "_default"]
                .into_iter()
                .map(str::to_string)
                .collect();
        assert_eq!(
            catalog_shard_placement_violations(&accounts),
            vec!["bad tenant", "tenantA", "tenanta"]
        );
    }

    /// TD-SPINE-2 R4: the root CAS is against the CALLER's expected parent.
    /// Another owner's root that landed after the caller checked version 0
    /// makes the commit refuse (`Fenced`), even at an equal generation, and
    /// the other owner's root stays the tip.
    #[tokio::test]
    async fn root_commit_refuses_a_stale_expected_parent() -> Result<()> {
        let store = ObjectStoreSnapshotStore::new(
            ProximaObjectStore::new(std::sync::Arc::new(object_store::memory::InMemory::new())),
            "memory:///",
            DrPathBuilder::system_catalog_manifests_subprefix(),
        );
        assert!(published(store.commit_root(3, None, b"seed").await?));
        // The caller checked v0; another equal-generation owner lands v1.
        assert!(published(
            store.commit_root(3, Some(0), b"other-owner").await?
        ));
        assert_eq!(
            store.commit_root(3, Some(0), b"checked-at-v0").await?,
            SnapshotWrite::Fenced
        );
        let tip = store.read_root().await?.expect("root");
        assert_eq!(
            (tip.version, tip.bytes.as_slice()),
            (1, &b"other-owner"[..])
        );
        Ok(())
    }
}
