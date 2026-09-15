//! Opt-in publication format. The head owns its payload; archives are history,
//! never publication candidates or authority. See the lease consolidation HLD.
use super::*;
use bincode::Options;
use object_store::{ObjectMeta, PutMode, PutOptions, UpdateVersion};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const MAGIC: &[u8; 8] = b"PMHEAD\0\x01";
const MAX_RECORD_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Serialize, Deserialize)]
pub(super) struct Record {
    authority: [u8; 16],
    snapshot: Option<Snapshot>,
}

#[derive(Serialize, Deserialize)]
struct Snapshot {
    version: u64,
    generation: u64,
    operation: [u8; 16],
    payload: Vec<u8>,
}

#[derive(Serialize, Deserialize)]
struct MigrationPlan {
    authority: [u8; 16],
    source_prefix: String,
    encoding: LegacyEncoding,
    expected_tip: u64,
    snapshots: Vec<(u64, [u8; 32])>,
}

impl Record {
    pub(super) fn version(&self) -> Option<u64> {
        self.snapshot.as_ref().map(|s| s.version)
    }
}

pub(super) struct LoadedHead {
    pub(super) record: Record,
    bytes: Bytes,
    meta: ObjectMeta,
}

fn codec() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(MAX_RECORD_BYTES)
        .reject_trailing_bytes()
}

fn encode(record: &Record) -> Result<Bytes, StorageError> {
    let body = codec()
        .serialize(record)
        .map_err(|e| StorageError::Serialization(e.to_string()))?;
    let mut bytes = Vec::with_capacity(MAGIC.len() + 32 + body.len());
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&Sha256::digest(&body));
    bytes.extend_from_slice(&body);
    Ok(bytes.into())
}

fn decode(bytes: &Bytes) -> Result<Record, StorageError> {
    if bytes.len() < 40 || bytes.len() as u64 > MAX_RECORD_BYTES + 40 || !bytes.starts_with(MAGIC) {
        return Err(StorageError::Corruption(
            "manifest: invalid/unsupported head envelope".into(),
        ));
    }
    let body = &bytes[40..];
    if Sha256::digest(body)[..] != bytes[8..40] {
        return Err(StorageError::Corruption(
            "manifest: head checksum mismatch".into(),
        ));
    }
    codec()
        .deserialize(body)
        .map_err(|e| StorageError::Corruption(format!("manifest head: {e}")))
}

fn io_error(error: object_store::Error) -> StorageError {
    match error {
        object_store::Error::NotFound { .. } => StorageError::NotFound(error.to_string()),
        object_store::Error::AlreadyExists { .. } => StorageError::AlreadyExists(error.to_string()),
        other => StorageError::DiskIO(std::io::Error::other(other)),
    }
}

impl ManifestCommitter {
    fn pin_authority(&self, authority: [u8; 16]) -> Result<(), StorageError> {
        if self.authority.get_or_init(|| authority) != &authority {
            return Err(StorageError::Corruption(
                "manifest: authority incarnation changed".into(),
            ));
        }
        Ok(())
    }

    fn head_path(&self) -> Path {
        Path::from(format!("{}/_publication.head", self.prefix))
    }

    fn format_path(&self) -> Path {
        Path::from(format!("{}/_publication.format", self.prefix))
    }

    fn archive_path(&self, version: u64) -> Path {
        Path::from(format!("{}/_history/v{version:020}.snapshot", self.prefix))
    }

    /// Provision a NEW, empty log in the experimental versioned-head format.
    ///
    /// Explicit opt-in, not a migration API. The operator must exclusively provision
    /// this prefix and exclude legacy writers (including suspended writers) before
    /// calling. Rejects any existing objects; does not delete or replace authority.
    /// If both marker and head exist after a lost response, `open_versioned` can
    /// reopen them. A marker-only failure intentionally stays closed and requires
    /// operator repair: it is indistinguishable from deletion of a formerly live
    /// head. Never retry by clearing the prefix. Backend conditional
    /// Update support is required for commits; unsupported backends fail closed.
    pub async fn create_versioned(
        store: ProximaObjectStore,
        prefix: impl Into<String>,
    ) -> Result<Self, StorageError> {
        let c = Self::new(store, prefix);
        if c.prefix.is_empty()
            || !c
                .store
                .list(Some(&Path::from(c.prefix.as_str())))
                .await?
                .is_empty()
        {
            return Err(StorageError::AlreadyExists(
                "manifest: versioned provisioning requires a nonempty, unused prefix".into(),
            ));
        }
        let authority = *uuid::Uuid::new_v4().as_bytes();
        let bytes = encode(&Record {
            authority,
            snapshot: None,
        })?;
        // The immutable marker prevents updated legacy readers/writers from
        // interpreting a missing head as an empty legacy log after partial failure.
        c.store
            .put_if_absent(&c.format_path(), bytes.clone())
            .await?;
        c.store.put_if_absent(&c.head_path(), bytes).await?;
        c.pin_authority(authority)?;
        Ok(c)
    }

    /// Open an existing versioned authority. Never initializes missing state or
    /// upgrades a legacy log. Pins the authority incarnation for this handle.
    pub async fn open_versioned(
        store: ProximaObjectStore,
        prefix: impl Into<String>,
    ) -> Result<Self, StorageError> {
        let c = Self::new(store, prefix);
        let marker = decode(&c.store.get(&c.format_path()).await?)?;
        if marker.snapshot.is_some() {
            return Err(StorageError::Corruption(
                "manifest: invalid format marker".into(),
            ));
        }
        c.pin_authority(marker.authority)?;
        c.load_head().await?;
        Ok(c)
    }

    /// Explicitly migrate a legacy log at a caller-verified tip, preserving source
    /// bytes and retained snapshots. The caller MUST exclude all old writers AND
    /// pruners, including suspended operations, before invoking this method.
    /// The source encoding must be declared; this method never guesses it.
    ///
    /// No source is deleted. Marker-only/partial failure stays closed and requires
    /// operator reconciliation; never clear a marker to retry. Repeating a completed
    /// migration with the same source tip opens its current head without resetting it.
    pub async fn migrate_versioned(self, expected_tip: u64) -> Result<Self, StorageError> {
        let plan_path = Path::from(format!("{}/_publication.migration", self.prefix));
        let source_prefix = self
            .store
            .full_path(&Path::from(self.prefix.as_str()))
            .to_string();
        if let Some(head) = self.load_head().await? {
            let plan: MigrationPlan = serde_json::from_slice(&self.store.get(&plan_path).await?)
                .map_err(|e| StorageError::Corruption(format!("manifest migration plan: {e}")))?;
            if plan.authority != head.record.authority
                || plan.source_prefix != source_prefix
                || plan.encoding != self.legacy_encoding
                || plan.expected_tip != expected_tip
            {
                return Err(StorageError::Corruption(
                    "manifest: completed migration identity mismatch".into(),
                ));
            }
            return Ok(self);
        }
        let versions = self.legacy_versions().await?;
        if versions.last().copied() != Some(expected_tip) {
            return Err(StorageError::TransactionCommitFailed(
                "manifest: migration expected tip does not match source".into(),
            ));
        }
        let authority = *uuid::Uuid::new_v4().as_bytes();
        let mut snapshots = Vec::with_capacity(versions.len());
        for version in &versions {
            let source = self.read_legacy_manifest(*version).await?;
            // Validate declared encoding and size before disabling source writes.
            encode(&self.migration_record(authority, *version, &source)?)?;
            snapshots.push((*version, Sha256::digest(&source).into()));
        }
        let plan = MigrationPlan {
            authority,
            source_prefix,
            encoding: self.legacy_encoding,
            expected_tip,
            snapshots,
        };
        let plan_bytes =
            serde_json::to_vec(&plan).map_err(|e| StorageError::Serialization(e.to_string()))?;
        self.store
            .put_if_absent(
                &self.format_path(),
                encode(&Record {
                    authority,
                    snapshot: None,
                })?,
            )
            .await?;
        self.store
            .put_if_absent(&plan_path, plan_bytes.into())
            .await?;
        let mut tip_bytes = None;
        for (version, checksum) in &plan.snapshots {
            let source = self.read_legacy_manifest(*version).await?;
            if Sha256::digest(&source)[..] != checksum[..] {
                return Err(StorageError::Corruption(
                    "manifest: source changed during migration".into(),
                ));
            }
            let bytes = encode(&self.migration_record(authority, *version, &source)?)?;
            if *version == expected_tip {
                tip_bytes = Some(bytes);
            } else {
                self.store
                    .put_if_absent(&self.archive_path(*version), bytes)
                    .await?;
            }
        }
        if self.legacy_versions().await? != versions {
            return Err(StorageError::TransactionCommitFailed("manifest: source inventory changed during migration; verify writer/pruner exclusion".into()));
        }
        let tip_source = self.read_legacy_manifest(expected_tip).await?;
        let (_, tip_checksum) = plan.snapshots.last().ok_or_else(|| {
            StorageError::Corruption("manifest: empty migration inventory".into())
        })?;
        if Sha256::digest(&tip_source)[..] != tip_checksum[..] {
            return Err(StorageError::Corruption(
                "manifest: source tip changed during migration".into(),
            ));
        }
        let tip_bytes = tip_bytes
            .ok_or_else(|| StorageError::Corruption("manifest: missing migration tip".into()))?;
        self.store
            .put_if_absent(&self.head_path(), tip_bytes)
            .await?;
        self.pin_authority(authority)?;
        Ok(self)
    }

    fn migration_record(
        &self,
        authority: [u8; 16],
        version: u64,
        source: &Bytes,
    ) -> Result<Record, StorageError> {
        let (generation, payload) = self.decode_legacy(source)?;
        let mut hash = Sha256::new();
        hash.update(authority);
        hash.update(version.to_be_bytes());
        let mut operation = [0; 16];
        operation.copy_from_slice(&hash.finalize()[..16]);
        Ok(Record {
            authority,
            snapshot: Some(Snapshot {
                version,
                generation,
                operation,
                payload: payload.to_vec(),
            }),
        })
    }

    pub(super) async fn load_head(&self) -> Result<Option<LoadedHead>, StorageError> {
        match self.store.get_with_meta(&self.head_path()).await {
            Ok((bytes, meta)) => {
                let record = decode(&bytes)?;
                if self.authority.get().is_none() {
                    let marker = decode(&self.store.get(&self.format_path()).await?)?;
                    if marker.snapshot.is_some() || marker.authority != record.authority {
                        return Err(StorageError::Corruption(
                            "manifest: marker/head authority mismatch".into(),
                        ));
                    }
                }
                self.pin_authority(record.authority)?;
                Ok(Some(LoadedHead {
                    record,
                    bytes,
                    meta,
                }))
            }
            Err(object_store::Error::NotFound { .. }) if self.authority.get().is_none() => {
                match self.store.get_with_meta(&self.format_path()).await {
                    Err(object_store::Error::NotFound { .. }) => Ok(None),
                    Ok(_) => Err(StorageError::Corruption(
                        "manifest: format marker exists but head is missing".into(),
                    )),
                    Err(e) => Err(io_error(e)),
                }
            }
            Err(e) => Err(io_error(e)),
        }
    }

    pub(super) async fn read_versioned(
        &self,
        head: LoadedHead,
        version: u64,
    ) -> Result<(u64, Bytes), StorageError> {
        let record = if head.record.version() == Some(version) {
            head.record
        } else {
            if head.record.version().is_none_or(|v| version > v) {
                return Err(StorageError::NotFound(format!(
                    "manifest version {version}"
                )));
            }
            let record = decode(&self.store.get(&self.archive_path(version)).await?)?;
            if record.authority != head.record.authority || record.version() != Some(version) {
                return Err(StorageError::Corruption(
                    "manifest: archive identity mismatch".into(),
                ));
            }
            record
        };
        let snapshot = record
            .snapshot
            .ok_or_else(|| StorageError::Corruption("manifest: empty snapshot".into()))?;
        Ok((snapshot.generation, snapshot.payload.into()))
    }

    pub(super) async fn commit_versioned(
        &self,
        head: LoadedHead,
        parent: Option<u64>,
        generation: u64,
        payload: Bytes,
    ) -> Result<CommitOutcome, StorageError> {
        let latest = head.record.version();
        if parent != latest
            || head
                .record
                .snapshot
                .as_ref()
                .is_some_and(|s| generation < s.generation)
        {
            return Ok(CommitOutcome::Conflict { latest });
        }
        let target = match parent {
            Some(p) => p.checked_add(1).ok_or_else(|| {
                StorageError::Serialization("manifest: version counter overflow".into())
            })?,
            None => 0,
        };
        let operation = *uuid::Uuid::new_v4().as_bytes();
        let candidate = Record {
            authority: head.record.authority,
            snapshot: Some(Snapshot {
                version: target,
                generation,
                operation,
                payload: payload.to_vec(),
            }),
        };
        let bytes = encode(&candidate)?;
        // Archive only a proven committed predecessor, never a candidate. Upload
        // before head replacement so every successful successor preserves history.
        // A delayed archival PUT may recreate pruned HISTORY, but cannot publish.
        if let Some(version) = latest {
            let path = self.archive_path(version);
            match self.store.put_if_absent(&path, head.bytes.clone()).await {
                Ok(()) => {}
                Err(StorageError::AlreadyExists(_)) => {
                    if self.store.get(&path).await? != head.bytes {
                        return Err(StorageError::Corruption(
                            "manifest: conflicting immutable archive".into(),
                        ));
                    }
                }
                Err(e) => return Err(e),
            }
        }
        let options = PutOptions {
            mode: PutMode::Update(UpdateVersion {
                e_tag: head.meta.e_tag,
                version: head.meta.version,
            }),
            ..Default::default()
        };
        match self.store.put_opts(&self.head_path(), bytes, options).await {
            Ok(_) => Ok(CommitOutcome::Committed(target)),
            Err(error) => {
                // Native clients may retry after an earlier attempt committed. Even
                // Precondition is not necessarily proof that this operation lost.
                self.resolve_publication(target, operation, error).await
            }
        }
    }

    async fn resolve_publication(
        &self,
        target: u64,
        operation: [u8; 16],
        error: object_store::Error,
    ) -> Result<CommitOutcome, StorageError> {
        let unknown = || {
            StorageError::TransactionCommitFailed(format!(
                "manifest: indeterminate publication at version {target}: {error}"
            ))
        };
        let head = self
            .load_head()
            .await
            .map_err(|_| unknown())?
            .ok_or_else(unknown)?;
        let latest = head.record.version();
        let record = if latest == Some(target) {
            head.record
        } else if latest.is_some_and(|v| v > target) {
            let bytes = self
                .store
                .get(&self.archive_path(target))
                .await
                .map_err(|_| unknown())?;
            let record = decode(&bytes).map_err(|_| unknown())?;
            if record.authority != head.record.authority || record.version() != Some(target) {
                return Err(unknown());
            }
            record
        } else {
            // No positive receipt. Preserve an unsupported/backend error instead of
            // turning it into either successful publication or harmless contention.
            return Err(io_error(error));
        };
        match record.snapshot {
            Some(s) if s.operation == operation => Ok(CommitOutcome::Committed(target)),
            Some(_) => Ok(CommitOutcome::Conflict { latest }),
            None => Err(unknown()),
        }
    }

    pub(super) async fn prune_versioned(
        &self,
        head: LoadedHead,
        keep_k: usize,
        min_age: std::time::Duration,
    ) -> Result<usize, StorageError> {
        let min_age = chrono::Duration::from_std(min_age).map_err(|_| {
            StorageError::Serialization("manifest: retention age out of range".into())
        })?;
        let Some(tip) = head.record.version() else {
            return Ok(0);
        };
        let history = Path::from(format!("{}/_history", self.prefix));
        let mut entries: Vec<_> = self
            .store
            .list(Some(&history))
            .await?
            .into_iter()
            .filter_map(|m| {
                let v = m
                    .location
                    .filename()?
                    .strip_prefix('v')?
                    .strip_suffix(".snapshot")?
                    .parse::<u64>()
                    .ok()?;
                // Rank and age only canonical archives. Nested or noncanonical
                // names must not contribute retention slots or lend their mtime
                // to a different object addressed by archive_path below.
                (v < tip && m.location == self.store.full_path(&self.archive_path(v)))
                    .then_some((v, m.last_modified))
            })
            .collect();
        entries.sort_unstable_by_key(|(v, _)| *v);
        let eligible = entries
            .len()
            .saturating_sub(keep_k.max(MIN_PRUNE_KEEP_K) - 1);
        let now = Utc::now();
        let mut deleted = 0;
        for (version, modified) in entries.into_iter().take(eligible) {
            if now.signed_duration_since(modified) < min_age {
                continue;
            }
            match self.store.delete(&self.archive_path(version)).await {
                Ok(()) => deleted += 1,
                Err(StorageError::NotFound(_)) => {}
                Err(e) => {
                    tracing::warn!(target: "proximadb::manifest::prune", version, error = %e, "manifest archive delete failed")
                }
            }
        }
        Ok(deleted)
    }
}
