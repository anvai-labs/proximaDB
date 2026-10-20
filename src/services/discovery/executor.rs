//! `DiscoveryJobExecutor` — background worker that drives the CS/CD loop.
//!
//! For each claimed job: pin a snapshot, mark the discovery projection
//! `Updating`, run the refinement pass, then atomically `commit_publish`
//! (`Fresh`) — or `abort_publish` (`RebuildRequired`) on failure. The spawn
//! pattern mirrors `start_axis_consumer` / the embedding drainer: a tokio task
//! with a `watch` shutdown channel.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::watch;
use tracing::{error, info, warn};

use super::job::{DiscoveryJob, DiscoveryJobResult, DiscoveryJobStatus};
use super::passes::PassContext;
use super::registry::DiscoveryRegistry;
use crate::services::snapshot::{SnapshotPin, SnapshotPublishCoordinator};
use proximadb_kernel::time::now_unix_ms;

/// Default poll interval for the background loop.
pub const DEFAULT_POLL_INTERVAL: Duration = Duration::from_secs(2);

pub struct DiscoveryJobExecutor {
    registry: Arc<DiscoveryRegistry>,
    coordinator: Arc<SnapshotPublishCoordinator>,
    /// v2 canonical read/write path used by refinement passes. `None` => every
    /// pass is an identity pass (walking skeleton / lightweight tests).
    vector_ops: Option<Arc<crate::services::VectorOperationsService>>,
}

impl DiscoveryJobExecutor {
    pub fn new(
        registry: Arc<DiscoveryRegistry>,
        coordinator: Arc<SnapshotPublishCoordinator>,
    ) -> Self {
        Self {
            registry,
            coordinator,
            vector_ops: None,
        }
    }

    /// Attach the v2 vector-operations service so refinement passes (dedup)
    /// can read and rewrite records via the canonical path.
    pub fn with_vector_ops(
        mut self,
        vector_ops: Arc<crate::services::VectorOperationsService>,
    ) -> Self {
        self.vector_ops = Some(vector_ops);
        self
    }

    /// Claim and process one scheduled job, if any. Returns the processed job id.
    pub async fn run_once(&self) -> Option<String> {
        let job = self.registry.claim_next_scheduled()?;
        let job_id = job.job_id.clone();
        if let Err(err) = self.execute(job).await {
            error!("discovery job {job_id} failed: {err:#}");
        }
        Some(job_id)
    }

    async fn execute(&self, mut job: DiscoveryJob) -> anyhow::Result<()> {
        // 1. Pin a read-only snapshot of the collection's canonical state.
        let pin = self.coordinator.pin(&job.collection_id).await?;
        job.snapshot_from_lsn = pin.from_lsn;
        job.snapshot_to_lsn = pin.to_lsn;
        job.checkpoint_id = pin.checkpoint_id;
        // Capture the per-collection write high-water-mark as the drift baseline.
        // `pin.to_lsn` is dominated by the global allocator (other collections'
        // writes), so it can't serve as a per-collection baseline; resolve the
        // user-facing name to the internal id the WAL keys entries under and read
        // *this* collection's manifest high-water-mark. No vector_ops (walking
        // skeleton / lightweight tests) => baseline stays 0.
        if let Some(vops) = self.vector_ops.as_ref() {
            let object_id = vops
                .resolve_collection_object_id(&job.collection_id)
                .await?;
            job.collection_write_lsn = self
                .coordinator
                .collection_write_lsn(&object_id.to_string())
                .await;
        }
        self.registry.upsert(job.clone());

        // 2. Mark the discovery projection Updating (serving still reads prior).
        if let Err(err) = self.coordinator.begin_publish(&pin).await {
            self.fail(&mut job, format!("begin_publish: {err:#}"));
            return Err(err);
        }

        // 3. Run the refinement pass against the pinned snapshot.
        let result = match self.run_pass(&job, &pin).await {
            Ok(result) => result,
            Err(err) => {
                let _ = self.coordinator.abort_publish(&pin).await;
                self.fail(&mut job, format!("pass: {err:#}"));
                return Err(err);
            }
        };

        // 4. Atomic serving switch: commit the republished snapshot (Fresh).
        if let Err(err) = self.coordinator.commit_publish(&pin).await {
            let _ = self.coordinator.abort_publish(&pin).await;
            self.fail(&mut job, format!("commit_publish: {err:#}"));
            return Err(err);
        }

        // 5. Fold the result into the job record and mark Complete.
        job.status = DiscoveryJobStatus::Complete;
        job.completed_at_ms = Some(now_unix_ms());
        job.input_record_count = result.input_record_count;
        job.refined_record_count = result.refined_record_count;
        job.removed_count = result.removed_count;
        job.quality_metrics = result.quality_metrics;
        self.registry.upsert(job);
        Ok(())
    }

    /// Run the refinement pass for a job's kind against the pinned snapshot.
    ///
    /// Orchestration only: build the [`PassContext`] (pinned snapshot +
    /// capability handles) and dispatch by kind via [`super::passes::run`]. The
    /// pass itself owns the refinement logic; a not-yet-implemented kind or a
    /// missing capability resolves to an identity pass and the caller
    /// atomically republishes the pinned snapshot unchanged.
    async fn run_pass(
        &self,
        job: &DiscoveryJob,
        pin: &SnapshotPin,
    ) -> anyhow::Result<DiscoveryJobResult> {
        let ctx = PassContext::new(pin.clone()).with_vector_ops(self.vector_ops.clone());
        super::passes::run(job.kind, &ctx).await
    }

    fn fail(&self, job: &mut DiscoveryJob, msg: String) {
        warn!("discovery job {} aborted: {}", job.job_id, msg);
        job.status = DiscoveryJobStatus::Failed;
        job.error = Some(msg);
        job.completed_at_ms = Some(now_unix_ms());
        self.registry.upsert(job.clone());
    }
}

/// Spawn the background executor loop. Drains all currently-scheduled jobs each
/// tick, until the shutdown signal flips to `true` (or the sender is dropped).
pub fn spawn_discovery_executor(
    executor: Arc<DiscoveryJobExecutor>,
    mut shutdown: watch::Receiver<bool>,
    interval: Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        info!("DiscoveryJobExecutor started (poll interval = {interval:?})");
        loop {
            tokio::select! {
                changed = shutdown.changed() => {
                    if changed.is_err() || *shutdown.borrow() {
                        break;
                    }
                }
                _ = tokio::time::sleep(interval) => {
                    // TD-LIFECYCLE-1: re-check shutdown between drained jobs so
                    // a queued backlog cannot extend teardown.
                    while executor.run_once().await.is_some() {
                        if *shutdown.borrow() {
                            break;
                        }
                    }
                }
            }
        }
        info!("DiscoveryJobExecutor stopped");
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::CatalogManager;
    use crate::services::discovery::job::DiscoveryJobKind;
    use proximadb_catalog::{CatalogColumn, CatalogTableSchema, TableIdentifier};
    use proximadb_data_model::ProximaType;

    async fn wired(
        collection: &str,
    ) -> (
        Arc<DiscoveryJobExecutor>,
        Arc<SnapshotPublishCoordinator>,
        Arc<DiscoveryRegistry>,
    ) {
        let tmp = std::env::temp_dir().join(format!(
            "proximadb_discovery_exec_{}_{}",
            collection,
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&tmp).unwrap();

        let manager = Arc::new(CatalogManager::new());
        manager
            .create_native_catalog("default", &format!("file://{}", tmp.display()))
            .await
            .unwrap();
        manager.set_default_catalog("default").await.unwrap();
        let catalog = manager.default_catalog().await.unwrap();
        let identifier = TableIdentifier::new(vec!["default".to_string()], collection.to_string());
        catalog
            .create_namespace(&identifier.namespace, std::collections::HashMap::new())
            .await
            .unwrap();
        catalog
            .create_table(
                &identifier,
                CatalogTableSchema::new(collection.to_string()).with_column(CatalogColumn::new(
                    0,
                    "oid",
                    ProximaType::String,
                )),
            )
            .await
            .unwrap();

        let coordinator = Arc::new(SnapshotPublishCoordinator::new(manager));
        let registry = Arc::new(DiscoveryRegistry::new());
        let executor = Arc::new(DiscoveryJobExecutor::new(
            registry.clone(),
            coordinator.clone(),
        ));
        (executor, coordinator, registry)
    }

    #[tokio::test]
    async fn identity_job_completes_and_publishes_fresh_snapshot() {
        let (executor, coordinator, registry) = wired("c_exec").await;
        let job = registry.schedule(DiscoveryJob::new("c_exec", DiscoveryJobKind::Dedup));

        let processed = executor.run_once().await;
        assert_eq!(processed.as_deref(), Some(job.job_id.as_str()));

        let done = registry.get(&job.job_id).unwrap();
        assert_eq!(done.status, DiscoveryJobStatus::Complete);
        assert!(done.completed_at_ms.is_some());

        // Atomic republish landed: discovery projection is Fresh with lineage.
        let proj = coordinator
            .active_projection("c_exec")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            proj.freshness_state,
            proximadb_catalog::ProjectionFreshnessState::Fresh
        );
        assert!(proj.source_range.is_some());
    }

    #[tokio::test]
    async fn job_against_uncataloged_collection_is_marked_failed() {
        let (executor, _coordinator, registry) = wired("c_real").await;
        let job = registry.schedule(DiscoveryJob::new("c_missing", DiscoveryJobKind::Dedup));
        executor.run_once().await;
        let done = registry.get(&job.job_id).unwrap();
        assert_eq!(done.status, DiscoveryJobStatus::Failed);
        assert!(done.error.is_some());
    }
}
