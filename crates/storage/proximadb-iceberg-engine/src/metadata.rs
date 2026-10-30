//! # metadata — atomic, versioned Iceberg `metadata.json` persistence
//!
//! Companion to [`crate::manifest`]. Where `ManifestCommitter` owns the *data* manifest
//! log (`v{N}.manifest`), this owns the **table metadata log**: immutable, monotonically
//! versioned `v{N}.metadata.json` objects that carry the Iceberg `TableMetadata`
//! (schema, snapshots, `refs`, snapshot/metadata logs).
//!
//! Iceberg's commit model is: write a new `metadata.json` whose snapshot history extends
//! the previous one, then atomically swap the "current metadata" pointer. Over decoupled
//! object storage with no transaction manager, the swap is a create-only put
//! ([`ProximaObjectStore::put_if_absent`]) on the successor version slot — the same
//! optimistic-concurrency protocol [`ManifestCommitter`](crate::manifest::ManifestCommitter)
//! uses. The bytes are opaque here: the caller serializes the metadata document; this
//! crate owns only the *atomicity* and *versioning*.

use bytes::Bytes;
use object_store::path::Path;
use proximadb_kernel::error::StorageError;
use proximadb_object_store::ProximaObjectStore;
pub use proximadb_storage_common::object_store_bridge::CommitOutcome;

/// Atomic, optimistic-concurrency committer for a table's `metadata.json` log.
///
/// One committer serves one table's metadata log (selected by `prefix`, typically
/// `data/<tenant>/<ns>/<table>/_metadata`).
pub struct MetadataCommitter {
    store: ProximaObjectStore,
    prefix: String,
}

impl MetadataCommitter {
    /// Create a committer writing metadata under `prefix`
    /// (e.g. `"data/<tenant>/<ns>/<table>/_metadata"`).
    pub fn new(store: ProximaObjectStore, prefix: impl Into<String>) -> Self {
        let mut prefix = prefix.into();
        // Normalize so `{prefix}/v..` never produces a double slash or a leading one.
        while prefix.ends_with('/') {
            prefix.pop();
        }
        Self { store, prefix }
    }

    /// Object path of the metadata document for `version`. Zero-padded to 20 digits so the
    /// lexical order returned by `list` matches numeric version order.
    fn metadata_path(&self, version: u64) -> Path {
        Path::from(format!("{}/v{version:020}.metadata.json", self.prefix))
    }

    /// Parse the version out of a metadata object key's final segment, ignoring any other
    /// objects that happen to live under the prefix.
    fn parse_version(name: &str) -> Option<u64> {
        name.strip_prefix('v')?
            .strip_suffix(".metadata.json")?
            .parse::<u64>()
            .ok()
    }

    /// The highest committed metadata version, or `None` if the log is empty.
    pub async fn latest_version(&self) -> Result<Option<u64>, StorageError> {
        let prefix = Path::from(self.prefix.as_str());
        let metas = self.store.list(Some(&prefix)).await?;
        Ok(metas
            .iter()
            .filter_map(|m| m.location.filename().and_then(Self::parse_version))
            .max())
    }

    /// Read the raw metadata-document bytes for a specific `version`.
    pub async fn read_metadata(&self, version: u64) -> Result<Bytes, StorageError> {
        self.store.get(&self.metadata_path(version)).await
    }

    /// The version a commit on top of `parent` lands at (`None` ⇒ `0`).
    pub fn successor(parent: Option<u64>) -> Result<u64, StorageError> {
        match parent {
            Some(p) => p.checked_add(1).ok_or_else(|| {
                StorageError::Serialization("metadata: version counter overflow".into())
            }),
            None => Ok(0),
        }
    }

    /// Atomically publish `metadata` as the successor of `parent` (`None` ⇒ the first
    /// commit, version `0`).
    ///
    /// Returns [`CommitOutcome::Committed`] with the new version on success, or
    /// [`CommitOutcome::Conflict`] (carrying the current latest version) if another
    /// committer already claimed the target slot. Real I/O errors propagate as `Err`.
    pub async fn commit(
        &self,
        parent: Option<u64>,
        metadata: Bytes,
    ) -> Result<CommitOutcome, StorageError> {
        let target = Self::successor(parent)?;
        match self
            .store
            .put_if_absent(&self.metadata_path(target), metadata)
            .await
        {
            Ok(()) => Ok(CommitOutcome::Committed(target)),
            Err(StorageError::AlreadyExists(_)) => Ok(CommitOutcome::Conflict {
                latest: self.latest_version().await?,
            }),
            Err(other) => Err(other),
        }
    }

    /// The object path (as a string) of the metadata document for `version`. Callers use
    /// this to set Iceberg's `metadata-location`.
    pub fn metadata_location(&self, version: u64) -> String {
        self.metadata_path(version).to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;
    use std::sync::Arc;

    fn committer() -> MetadataCommitter {
        MetadataCommitter::new(
            ProximaObjectStore::new(Arc::new(InMemory::new())),
            "data/t/ns/tbl/_metadata",
        )
    }

    /// Empty log → no version; first commit lands at v0 and is readable back.
    #[tokio::test]
    async fn first_commit_is_version_zero() {
        let c = committer();
        assert_eq!(c.latest_version().await.unwrap(), None);

        let out = c
            .commit(None, Bytes::from_static(b"{\"format-version\":2}"))
            .await
            .unwrap();
        assert_eq!(out, CommitOutcome::Committed(0));
        assert_eq!(c.latest_version().await.unwrap(), Some(0));
        assert_eq!(
            c.read_metadata(0).await.unwrap(),
            Bytes::from_static(b"{\"format-version\":2}")
        );
        assert!(
            c.metadata_location(0)
                .ends_with("/_metadata/v00000000000000000000.metadata.json")
        );
    }

    /// Sequential commits advance the version monotonically; each is independently readable.
    #[tokio::test]
    async fn sequential_commits_advance_version() {
        let c = committer();
        assert_eq!(
            c.commit(None, Bytes::from_static(b"m0")).await.unwrap(),
            CommitOutcome::Committed(0)
        );
        assert_eq!(
            c.commit(Some(0), Bytes::from_static(b"m1")).await.unwrap(),
            CommitOutcome::Committed(1)
        );
        assert_eq!(c.latest_version().await.unwrap(), Some(1));
        assert_eq!(c.read_metadata(1).await.unwrap(), Bytes::from_static(b"m1"));
    }

    /// Two committers racing from the same parent: one wins the slot, the other is told it
    /// conflicted and given the latest version to rebase onto. Winner bytes are preserved.
    #[tokio::test]
    async fn concurrent_commit_from_same_parent_conflicts() {
        let c = committer();
        c.commit(None, Bytes::from_static(b"m0")).await.unwrap();

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
        assert_eq!(
            c.read_metadata(1).await.unwrap(),
            Bytes::from_static(b"winner")
        );
    }

    /// A trailing slash in the prefix must not change where metadata lands.
    #[tokio::test]
    async fn prefix_trailing_slash_is_normalized() {
        let c = MetadataCommitter::new(
            ProximaObjectStore::new(Arc::new(InMemory::new())),
            "a/b/_metadata/",
        );
        assert_eq!(
            c.commit(None, Bytes::from_static(b"x")).await.unwrap(),
            CommitOutcome::Committed(0)
        );
        assert_eq!(
            c.metadata_location(0),
            "a/b/_metadata/v00000000000000000000.metadata.json"
        );
    }
}
