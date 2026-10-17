// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! Object-store writer-epoch log (ADR-095 D2, TD-PGOLTP-2).
//!
//! Each boot appends a create-only successor slot
//! `{control_url}/_control/{deployment_id}/writer/v{epoch}.manifest`
//! through [`ManifestCommitter::commit_fenced`] — the same generation-prefixed,
//! put-if-absent pattern as the partition-lease writer incarnations. Create-only
//! is the one conditional write that is atomic on every backend we target
//! (`file://`, `memory://`, S3/MinIO `If-None-Match: *`, Azure, GCS), unlike an
//! If-Match update of a single `writer.json`, which `file://` cannot do.
//!
//! The epoch IS the slot version, so it can never be reused. A lost-response
//! retry can surface our own successful commit as a conflict; it is reconciled
//! by reading the slot back and comparing the claim nonce.

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use bytes::Bytes;
use proximadb_iceberg_engine::manifest::{CommitOutcome, LegacyEncoding, ManifestCommitter};
use proximadb_object_store::ProximaObjectStore;
pub use proximadb_storage_ports::{EpochGrant, EpochStore, WriterClaim};

fn writer_epoch_gauge() -> prometheus::IntGauge {
    prometheus::register_int_gauge!(
        "proximadb_writer_epoch",
        "Writer epoch this process won for its durable volume (0 = no durable volume configured)"
    )
    .unwrap_or_else(|_| {
        prometheus::IntGauge::new("proximadb_writer_epoch_fallback", "fallback")
            .unwrap_or_else(|_| unreachable!("valid gauge metric descriptor"))
    })
}

lazy_static::lazy_static! {
    static ref WRITER_EPOCH: prometheus::IntGauge = writer_epoch_gauge();
}

/// Publish the epoch this process owns.
pub fn set_writer_epoch_gauge(epoch: u64) {
    WRITER_EPOCH.set(i64::try_from(epoch).unwrap_or(i64::MAX));
}

/// Writer-epoch log over a [`ProximaObjectStore`].
pub struct ObjectStoreEpochStore {
    committer: ManifestCommitter,
}

impl ObjectStoreEpochStore {
    /// `control_url` is an object-store URL (`file:///abs/dir`, `s3://bucket/p`,
    /// …) that every host which may own the volume can reach.
    pub fn from_url(control_url: &str, deployment_id: &str) -> Result<Self> {
        if deployment_id.is_empty() || deployment_id.contains('/') {
            bail!("deployment id '{deployment_id}' must be non-empty and contain no '/'");
        }
        let store = ProximaObjectStore::from_url(control_url)
            .map_err(|e| anyhow!("opening writer-epoch control store {control_url}: {e}"))?;
        Ok(Self::new(store, deployment_id))
    }

    pub fn new(store: ProximaObjectStore, deployment_id: &str) -> Self {
        let committer = ManifestCommitter::new(store, format!("_control/{deployment_id}/writer"))
            // `commit_fenced` requires the generation-prefixed encoding; the
            // default (`Plain`) rejects every fenced write.
            .with_legacy_encoding(LegacyEncoding::GenerationPrefixed);
        Self { committer }
    }

    async fn read_grant(&self, version: u64) -> Result<EpochGrant> {
        let (_generation, bytes) = self
            .committer
            .read_fenced(version)
            .await
            .map_err(|e| anyhow!("reading writer epoch {version}: {e}"))?;
        serde_json::from_slice(&bytes).with_context(|| format!("decoding writer epoch {version}"))
    }
}

#[async_trait]
impl EpochStore for ObjectStoreEpochStore {
    async fn acquire_next(&self, claim: &WriterClaim) -> Result<EpochGrant> {
        let latest = self
            .committer
            .latest_version()
            .await
            .map_err(|e| anyhow!("reading the writer-epoch log: {e}"))?;
        // Slot versions are dense from 0; epochs are 1-based (0 = "never owned").
        let (parent, epoch) = match latest {
            None => (None, 1),
            Some(version) => (Some(version), version + 2),
        };
        let grant = EpochGrant {
            epoch,
            claim: claim.clone(),
        };
        let payload = serde_json::to_vec(&grant).context("encoding writer epoch claim")?;
        let outcome = self
            .committer
            .commit_fenced(parent, epoch, Bytes::from(payload))
            .await;
        let slot = epoch - 1;
        match outcome {
            Ok(CommitOutcome::Committed(_)) => Ok(grant),
            // Ambiguous: a retried create-only PUT reports our own landed write
            // as a conflict (or the response was lost entirely). The slot's
            // contents decide.
            Ok(CommitOutcome::Conflict { .. }) | Err(_) => match self.read_grant(slot).await {
                Ok(existing) if existing.claim.nonce == claim.nonce => Ok(existing),
                Ok(existing) => bail!(
                    "lost the writer-epoch race: epoch {} was claimed by {}:{} (another process now owns this volume)",
                    existing.epoch,
                    existing.claim.host,
                    existing.claim.pid
                ),
                Err(read_err) => match outcome {
                    Err(commit_err) => Err(anyhow!(
                        "claiming writer epoch {epoch} failed: {commit_err}; slot unreadable: {read_err:#}"
                    )),
                    Ok(_) => Err(read_err.context("writer-epoch conflict with an unreadable slot")),
                },
            },
        }
    }

    async fn current(&self) -> Result<Option<EpochGrant>> {
        match self
            .committer
            .latest_version()
            .await
            .map_err(|e| anyhow!("reading the writer-epoch log: {e}"))?
        {
            Some(version) => Ok(Some(self.read_grant(version).await?)),
            None => Ok(None),
        }
    }
}

/// A fresh, globally unique identity for a durable volume (ADR-095 review F4).
pub fn new_volume_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// A claim for this process, with a fresh nonce.
pub fn claim_for_this_process() -> WriterClaim {
    let host = std::env::var("HOSTNAME")
        .ok()
        .filter(|h| !h.is_empty())
        .or_else(|| {
            std::fs::read_to_string("/etc/hostname")
                .ok()
                .map(|h| h.trim().to_string())
        })
        .unwrap_or_else(|| "unknown-host".to_string());
    WriterClaim {
        host,
        pid: std::process::id(),
        nonce: uuid::Uuid::new_v4().to_string(),
        claimed_at_ms: chrono::Utc::now().timestamp_millis(),
        volume_id: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn claim(tag: &str) -> WriterClaim {
        WriterClaim {
            host: tag.to_string(),
            pid: 1,
            nonce: format!("nonce-{tag}"),
            claimed_at_ms: 0,
            volume_id: None,
        }
    }

    fn memory_store() -> ProximaObjectStore {
        ProximaObjectStore::new(Arc::new(object_store::memory::InMemory::new()))
    }

    #[tokio::test]
    async fn epochs_are_monotonic_and_never_reused() -> Result<()> {
        let store = ObjectStoreEpochStore::new(memory_store(), "dep");
        assert!(store.current().await?.is_none());
        let a = store.acquire_next(&claim("a")).await?;
        let b = store.acquire_next(&claim("b")).await?;
        let c = store.acquire_next(&claim("c")).await?;
        assert_eq!((a.epoch, b.epoch, c.epoch), (1, 2, 3));
        let current = store.current().await?.ok_or_else(|| anyhow!("current"))?;
        assert_eq!(current.epoch, 3);
        assert_eq!(current.claim.host, "c");
        Ok(())
    }

    #[tokio::test]
    async fn deployments_are_isolated() -> Result<()> {
        let shared = memory_store();
        let one = ObjectStoreEpochStore::new(shared.clone(), "one");
        let two = ObjectStoreEpochStore::new(shared, "two");
        one.acquire_next(&claim("a")).await?;
        one.acquire_next(&claim("b")).await?;
        assert_eq!(two.acquire_next(&claim("c")).await?.epoch, 1);
        Ok(())
    }

    // Two processes booting at once: each epoch is won by exactly one of them;
    // the loser errors and must not serve.
    #[tokio::test]
    async fn concurrent_claims_never_share_an_epoch() -> Result<()> {
        let shared = memory_store();
        let mut tasks = Vec::new();
        for i in 0..8 {
            let store = ObjectStoreEpochStore::new(shared.clone(), "dep");
            tasks.push(tokio::spawn(async move {
                store.acquire_next(&claim(&format!("p{i}"))).await
            }));
        }
        let mut won = Vec::new();
        for task in tasks {
            if let Ok(grant) = task.await? {
                won.push(grant.epoch);
            }
        }
        let mut deduped = won.clone();
        deduped.sort_unstable();
        deduped.dedup();
        assert!(!won.is_empty(), "at least one claimant wins");
        assert_eq!(deduped.len(), won.len(), "no epoch granted twice: {won:?}");
        Ok(())
    }

    #[tokio::test]
    async fn file_backed_log_survives_reopen() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let url = format!("file://{}", dir.path().display());
        ObjectStoreEpochStore::from_url(&url, "dep")?
            .acquire_next(&claim("a"))
            .await?;
        let reopened = ObjectStoreEpochStore::from_url(&url, "dep")?;
        assert_eq!(reopened.acquire_next(&claim("b")).await?.epoch, 2);
        Ok(())
    }

    /// Runs against a real S3-compatible server when `AWS_ENDPOINT` and
    /// `PROXIMADB_TEST_EPOCH_S3_URL` (e.g. `s3://bucket/epoch-test`) are set —
    /// the MinIO smoke uses this.
    #[tokio::test]
    #[ignore = "needs an S3-compatible endpoint (MinIO smoke)"]
    async fn s3_compatible_epoch_log() -> Result<()> {
        let Ok(base) = std::env::var("PROXIMADB_TEST_EPOCH_S3_URL") else {
            return Ok(());
        };
        let deployment = format!("t{}", uuid::Uuid::new_v4().simple());
        let store = ObjectStoreEpochStore::from_url(&base, &deployment)?;
        let a = store.acquire_next(&claim("a")).await?;
        let b = store.acquire_next(&claim("b")).await?;
        assert_eq!((a.epoch, b.epoch), (1, 2));
        Ok(())
    }
}
