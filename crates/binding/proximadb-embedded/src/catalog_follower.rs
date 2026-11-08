//! Published catalog metadata for short-lived embedded applications.
//!
//! This is a catalog follower, not a cloud query engine. It shares the existing
//! SystemCatalog snapshot format and refresh protocol without creating writer
//! services, claiming a catalog generation, or replaying a local owner's WAL.
//! Collection data queries still require a read-only engine lifecycle and an
//! injected read view; this module deliberately exposes only metadata reads.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use proximadb::catalog::{Catalog, CatalogNamespace, CatalogTableSchema, TableIdentifier};
use proximadb::services::catalog_snapshot_store::{CatalogSnapshotStore, ObjectStoreSnapshotStore};
use proximadb::services::system_catalog::{ReloadOutcome, SystemCatalog};
use proximadb::storage::trait_components::path_resolver::DrPathBuilder;

use crate::{AccessMode, FileLockSet};

/// Explicit authoritative catalog location and disposable local working volume.
#[derive(Debug, Clone)]
pub struct EmbeddedCatalogFollowerConfig {
    /// Exact object-store root used by the catalog owner (for example s3://bucket/prefix).
    /// Enable this crate's matching `aws`, `azure`, or `gcp` Cargo feature, or
    /// `cloud-full` for all three. These provider features are off by default.
    /// Credentials are supplied through the existing object-store backend.
    pub metadata_uri: String,
    /// Absolute local filesystem directory for this process's temporary catalog
    /// appender. This is scratch space, not authoritative storage or a data cache.
    /// Concurrent followers must use distinct local work directories.
    pub local_work_dir: PathBuf,
}

impl EmbeddedCatalogFollowerConfig {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.metadata_uri.contains("://") && !self.metadata_uri.starts_with("file://"),
            "catalog follower requires an explicit object-store metadata URI"
        );
        ensure!(
            self.local_work_dir.is_absolute()
                && !self.local_work_dir.to_string_lossy().contains("://"),
            "catalog follower local_work_dir must be an absolute local filesystem path"
        );
        Ok(())
    }
}

/// A read-only follower of already-published catalog metadata.
///
/// The private catalog is never exposed as a write-capable trait object. No
/// namespace is created on open. Missing or corrupt snapshots fail opening;
/// callers cannot mistake an unpublished catalog for an empty database.
/// Refresh is explicit; callers control how much metadata staleness they accept.
/// The host application remains responsible for authorization of metadata access.
pub struct EmbeddedCatalogFollower {
    // Field order ensures the appender closes before its scratch directory is
    // removed, and local ownership is released only after scratch cleanup.
    catalog: SystemCatalog,
    metadata_uri: String,
    _scratch: tempfile::TempDir,
    _ownership: FileLockSet,
}

impl EmbeddedCatalogFollower {
    /// Reconstruct metadata from the owner's published object-store snapshot.
    /// This neither claims write ownership nor initializes a vector WAL/engine.
    ///
    /// Build with this crate's `aws` feature for S3, `azure` for Azure Blob,
    /// `gcp` for Google Cloud Storage, or `cloud-full` for all providers.
    /// Credentials and provider configuration must also be available at runtime;
    /// enabling a feature alone does not grant storage access. Opening performs
    /// remote reads and requires an already-published catalog snapshot.
    pub async fn open(config: EmbeddedCatalogFollowerConfig) -> Result<Self> {
        config.validate()?;
        let store = Arc::new(ObjectStoreSnapshotStore::from_url(
            &config.metadata_uri,
            DrPathBuilder::system_catalog_manifests_subprefix(),
        )?);
        Self::open_with_store(config, store).await
    }

    async fn open_with_store(
        config: EmbeddedCatalogFollowerConfig,
        store: Arc<dyn CatalogSnapshotStore>,
    ) -> Result<Self> {
        config.validate()?;
        let ownership = FileLockSet::acquire(
            std::slice::from_ref(&config.local_work_dir),
            AccessMode::Exclusive,
        )
        .context("acquiring catalog follower local work directory")?;
        // A follower's only authority is the published snapshot. Never reuse a
        // WAL at a caller-controlled filename: it could contain an owner's
        // uncommitted/local-only tail or belong to a different catalog root.
        let scratch = tempfile::Builder::new()
            .prefix("catalog-follower-")
            .tempdir_in(&config.local_work_dir)
            .context("creating disposable catalog follower work directory")?;
        let catalog =
            SystemCatalog::open_follower("default", scratch.path().join("catalog.wal"), store)
                .await
                .context("opening published catalog follower")?;
        ensure!(
            catalog.loaded_version().is_some(),
            "catalog follower requires an existing published snapshot"
        );
        Ok(Self {
            catalog,
            metadata_uri: config.metadata_uri,
            _scratch: scratch,
            _ownership: ownership,
        })
    }

    /// Exact authoritative metadata URI configured by the caller.
    pub fn metadata_uri(&self) -> &str {
        &self.metadata_uri
    }

    /// This context always has the follower role and never permits DDL.
    pub fn is_read_only(&self) -> bool {
        self.catalog.is_read_only()
    }

    /// Catalog snapshot version currently loaded in memory, not a data-file
    /// visibility watermark or a receipt for any vector writes.
    pub fn loaded_version(&self) -> Option<u64> {
        self.catalog.loaded_version()
    }

    /// Check the published snapshot and reload metadata if its version advanced.
    pub async fn refresh(&self) -> Result<ReloadOutcome> {
        self.catalog.reload_if_stale().await
    }

    pub async fn list_namespaces(
        &self,
        parent: Option<&[String]>,
    ) -> Result<Vec<CatalogNamespace>> {
        self.catalog.list_namespaces(parent).await
    }

    pub async fn list_tables(&self, namespace: &[String]) -> Result<Vec<TableIdentifier>> {
        self.catalog.list_tables(namespace).await
    }

    pub async fn get_table(&self, identifier: &TableIdentifier) -> Result<CatalogTableSchema> {
        self.catalog.get_table(identifier).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};

    use proximadb::catalog::CatalogColumn;
    use proximadb::services::catalog_snapshot_store::{SnapshotRead, SnapshotWrite};
    use proximadb_data_model::ProximaType;

    /// The production snapshot port with versioned, credential-free storage.
    #[derive(Default)]
    struct MemorySnapshots {
        snapshot: Mutex<Option<SnapshotRead>>,
        writes: AtomicU64,
    }

    #[async_trait::async_trait]
    impl CatalogSnapshotStore for MemorySnapshots {
        fn requires_snapshot_on_commit(&self) -> bool {
            true
        }

        async fn read(&self) -> Result<Option<SnapshotRead>> {
            Ok(self.snapshot.lock().expect("snapshot lock").clone())
        }

        async fn write_atomic(&self, generation: u64, bytes: &[u8]) -> Result<SnapshotWrite> {
            let mut snapshot = self.snapshot.lock().expect("snapshot lock");
            if snapshot
                .as_ref()
                .is_some_and(|prior| prior.generation > generation)
            {
                return Ok(SnapshotWrite::Fenced);
            }
            let version = self.writes.fetch_add(1, Ordering::SeqCst) + 1;
            *snapshot = Some(SnapshotRead {
                version,
                generation,
                bytes: bytes.to_vec(),
            });
            Ok(SnapshotWrite::Published { version })
        }

        async fn latest_version(&self) -> Result<Option<u64>> {
            Ok(self
                .snapshot
                .lock()
                .expect("snapshot lock")
                .as_ref()
                .map(|snapshot| snapshot.version))
        }

        fn describe(&self) -> String {
            "memory://catalog-test".to_string()
        }
    }

    fn config(work: PathBuf) -> EmbeddedCatalogFollowerConfig {
        EmbeddedCatalogFollowerConfig {
            metadata_uri: "s3://catalog-bucket/exact//root".to_string(),
            local_work_dir: work,
        }
    }

    fn schema(name: &str) -> CatalogTableSchema {
        CatalogTableSchema::new(name)
            .with_column(CatalogColumn::new(1, "id", ProximaType::Int64).nullable(false))
            .with_primary_key(vec!["id".to_string()])
    }

    #[tokio::test]
    async fn follower_reconstructs_refreshes_and_never_claims_or_publishes() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let store = Arc::new(MemorySnapshots::default());
        let owner = SystemCatalog::open_with_snapshot_store(
            "default",
            temp.path().join("owner.wal"),
            store.clone(),
        )
        .await?;
        let namespace = vec!["published".to_string()];
        let table = TableIdentifier::new(namespace.clone(), "records");
        owner.create_namespace(&namespace, HashMap::new()).await?;
        owner.create_table(&table, schema("records")).await?;
        let initial_writes = store.writes.load(Ordering::SeqCst);
        let cfg = config(temp.path().join("follower-work"));
        let follower = EmbeddedCatalogFollower::open_with_store(cfg.clone(), store.clone()).await?;
        assert!(follower.is_read_only());
        assert_eq!(follower.metadata_uri(), cfg.metadata_uri);
        assert_eq!(follower.catalog.generation(), owner.generation());
        assert_eq!(follower.get_table(&table).await?.name, "records");
        assert_eq!(follower.list_tables(&namespace).await?.len(), 1);
        assert_eq!(follower.list_namespaces(None).await?.len(), 1);
        assert!(
            follower
                .catalog
                .create_namespace(&["forbidden".to_string()], HashMap::new())
                .await
                .is_err()
        );
        assert_eq!(store.writes.load(Ordering::SeqCst), initial_writes);

        let second = TableIdentifier::new(namespace.clone(), "later");
        owner.create_table(&second, schema("later")).await?;
        assert_eq!(
            follower.list_tables(&namespace).await?.len(),
            1,
            "refresh must be caller-controlled"
        );
        assert!(matches!(
            follower.refresh().await?,
            ReloadOutcome::Reloaded { .. }
        ));
        assert_eq!(follower.list_tables(&namespace).await?.len(), 2);
        // A new owner may take over while this read context remains open.
        // Follow its newly published metadata without claiming ownership.
        let next_owner = SystemCatalog::open_with_snapshot_store(
            "default",
            temp.path().join("next-owner.wal"),
            store.clone(),
        )
        .await?;
        assert!(next_owner.generation() > owner.generation());
        let third = TableIdentifier::new(namespace.clone(), "after_takeover");
        next_owner
            .create_table(&third, schema("after_takeover"))
            .await?;
        let writes_before_refresh = store.writes.load(Ordering::SeqCst);
        assert!(matches!(
            follower.refresh().await?,
            ReloadOutcome::Reloaded { .. }
        ));
        assert_eq!(follower.get_table(&third).await?.name, "after_takeover");
        assert_eq!(follower.loaded_version(), store.latest_version().await?);
        assert!(follower.is_read_only());
        assert_eq!(store.writes.load(Ordering::SeqCst), writes_before_refresh);
        let writes_after_owner = store.writes.load(Ordering::SeqCst);
        let scratch = follower._scratch.path().to_path_buf();
        assert!(
            EmbeddedCatalogFollower::open_with_store(cfg.clone(), store.clone())
                .await
                .is_err(),
            "live work root cannot be shared"
        );
        drop(follower);
        assert!(!scratch.exists(), "scratch WAL is disposable");
        drop(owner);
        drop(next_owner);

        // No prior reader work directory or live owner is needed to reconstruct
        // the same published catalog. A stale owner WAL is never replayed.
        std::fs::write(
            cfg.local_work_dir.join("catalog.wal"),
            b"unrelated owner tail",
        )?;
        let restarted = EmbeddedCatalogFollower::open_with_store(cfg, store.clone()).await?;
        assert_eq!(restarted.get_table(&second).await?.name, "later");
        let fresh = EmbeddedCatalogFollower::open_with_store(
            config(temp.path().join("fresh-work")),
            store.clone(),
        )
        .await?;
        assert_eq!(fresh.get_table(&table).await?.name, "records");
        assert_eq!(store.writes.load(Ordering::SeqCst), writes_after_owner);
        Ok(())
    }

    #[tokio::test]
    async fn follower_rejects_missing_corrupt_and_ambiguous_configuration() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let store = Arc::new(MemorySnapshots::default());
        for uri in ["relative/catalog", "file:///local/catalog"] {
            let mut cfg = config(temp.path().join("must-not-exist"));
            cfg.metadata_uri = uri.to_string();
            assert!(
                EmbeddedCatalogFollower::open_with_store(cfg, store.clone())
                    .await
                    .is_err()
            );
            assert!(!temp.path().join("must-not-exist").exists());
        }
        assert!(
            EmbeddedCatalogFollower::open_with_store(
                config(PathBuf::from("relative-work")),
                store.clone()
            )
            .await
            .is_err()
        );
        let cfg = config(temp.path().join("work"));
        let missing = EmbeddedCatalogFollower::open_with_store(cfg.clone(), store.clone())
            .await
            .err()
            .expect("unpublished catalog rejected");
        assert!(missing.to_string().contains("existing published snapshot"));
        store.write_atomic(7, b"corrupt snapshot").await?;
        assert!(
            EmbeddedCatalogFollower::open_with_store(cfg.clone(), store.clone())
                .await
                .is_err()
        );
        assert_eq!(
            store.writes.load(Ordering::SeqCst),
            1,
            "read failures cannot publish a replacement"
        );
        assert!(
            FileLockSet::acquire([cfg.local_work_dir], AccessMode::Exclusive).is_ok(),
            "failed opens release local ownership"
        );
        Ok(())
    }
}
