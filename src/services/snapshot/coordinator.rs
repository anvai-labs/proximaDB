//! `SnapshotPin` + `SnapshotPublishCoordinator`.

use std::sync::Arc;

use anyhow::{Context, Result};
use proximadb_catalog::{CatalogProjection, CatalogProjectionKind, ProjectionFreshnessState};

use crate::catalog::CatalogManager;

/// Canonical name of the per-collection projection that tracks the active
/// discovery-republished snapshot. One per collection.
pub const DISCOVERY_ACTIVE_PROJECTION: &str = "discovery_active";

/// A read-only, immutable view of a collection's canonical WAL position,
/// captured before offline refinement so discovery operates against a stable
/// snapshot while serving continues on live data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotPin {
    /// Collection (logical name) this pin belongs to.
    pub collection_id: String,
    /// Lowest global LSN included in the pinned range.
    pub from_lsn: u64,
    /// Highest global LSN included in the pinned range.
    pub to_lsn: u64,
    /// Latest manifest checkpoint id at pin time (0 if none / manifest absent).
    pub checkpoint_id: u64,
}

impl SnapshotPin {
    /// Lineage `source_range` string recorded on the republished projection.
    pub fn source_range(&self) -> String {
        format!("wal:{}..{}", self.from_lsn, self.to_lsn)
    }

    /// Lineage `last_included_position` string recorded on the republished projection.
    pub fn position(&self) -> String {
        format!("checkpoint:{}", self.checkpoint_id)
    }
}

/// Pins canonical snapshots and atomically republishes refined snapshots by
/// driving the collection's discovery projection through the catalog freshness
/// state machine. Reused by F1 (discovery) and F5 (external collection).
pub struct SnapshotPublishCoordinator {
    catalog_manager: Arc<CatalogManager>,
}

impl SnapshotPublishCoordinator {
    pub fn new(catalog_manager: Arc<CatalogManager>) -> Self {
        Self { catalog_manager }
    }

    /// Read-only pin of the collection's current canonical WAL position.
    ///
    /// Resilient: if the global manifest is not initialized (e.g. a minimal
    /// embedded harness) the pin degrades to a zero range rather than failing,
    /// so the publish path stays testable without a live manifest.
    pub async fn pin(&self, collection_id: &str) -> Result<SnapshotPin> {
        let (from_lsn, to_lsn, checkpoint_id) =
            match crate::storage::persistence::write_ahead_log::manifest::get_service() {
                Some(svc) => {
                    let entries = svc.get_collection_entries(collection_id).await;
                    let from_lsn = entries.iter().map(|e| e.global_lsn).min().unwrap_or(0);
                    let max_entry = entries.iter().map(|e| e.global_lsn).max().unwrap_or(0);
                    let current = svc.lsn_allocator().current().await;
                    let to_lsn = max_entry.max(current);
                    let checkpoint_id = svc
                        .get_latest_checkpoint()
                        .await
                        .map(|c| c.checkpoint_id)
                        .unwrap_or(0);
                    (from_lsn, to_lsn, checkpoint_id)
                }
                None => (0, 0, 0),
            };

        Ok(SnapshotPin {
            collection_id: collection_id.to_string(),
            from_lsn,
            to_lsn,
            checkpoint_id,
        })
    }

    /// Per-collection write high-water-mark: the highest global LSN among *this
    /// collection's* manifest entries — write progress for THIS collection
    /// alone, deliberately **not** maxed with the global LSN allocator (which is
    /// what [`pin`](Self::pin)'s `to_lsn` does for snapshot completeness).
    ///
    /// Used as a stable *cutoff position* captured at recluster time: the drift
    /// watcher records this on the completed job, then later counts the
    /// collection's writes *past* it via [`collection_writes_since`]. A quiet
    /// collection no longer drifts on other collections' writes because its HWM
    /// only advances when *it* flushes a new entry.
    ///
    /// `collection_id` must be the canonical internal id the WAL keys entries
    /// under (resolve the user-facing name first via
    /// `VectorOperationsService::resolve_collection_id`). Degrades to 0 without a
    /// live manifest, so it stays testable in minimal harnesses.
    ///
    /// [`collection_writes_since`]: Self::collection_writes_since
    pub async fn collection_write_lsn(&self, collection_id: &str) -> u64 {
        match crate::storage::persistence::write_ahead_log::manifest::get_service() {
            Some(svc) => svc
                .get_collection_entries(collection_id)
                .await
                .iter()
                .map(|e| e.global_lsn)
                .max()
                .unwrap_or(0),
            None => 0,
        }
    }

    /// Per-collection write *volume* since a cutoff: the number of *records*
    /// written to *this* collection across its WAL manifest entries with
    /// `global_lsn` strictly greater than `baseline_lsn` (summing each batch's
    /// `vector_count`).
    ///
    /// This is the drift watcher's magnitude signal, deliberately a **record
    /// count for this collection** rather than a global-LSN delta. The LSN
    /// allocator is global, so a delta of `global_lsn` values is inflated by
    /// *other* collections' writes between this collection's flushes — a
    /// low-traffic tenant in a busy system would over-recluster. A per-collection
    /// record count is immune to that and intuitively calibratable ("recluster
    /// after N new records").
    ///
    /// `collection_id` must be the canonical internal id (see
    /// [`collection_write_lsn`]). Degrades to 0 without a live manifest.
    /// `vector_count` is populated on the live flush path (the disk manager
    /// records each batch's `vector_records.len()`); older manifest entries
    /// written before that wiring carry 0 and simply contribute nothing — a
    /// conservative undercount that resolves as the baseline advances.
    ///
    /// [`collection_write_lsn`]: Self::collection_write_lsn
    pub async fn collection_writes_since(&self, collection_id: &str, baseline_lsn: u64) -> u64 {
        match crate::storage::persistence::write_ahead_log::manifest::get_service() {
            Some(svc) => svc
                .get_collection_entries(collection_id)
                .await
                .iter()
                .filter(|e| e.global_lsn > baseline_lsn)
                .map(|e| e.vector_count)
                .sum(),
            None => 0,
        }
    }

    /// Begin a republish: mark the discovery projection `Updating`. Serving
    /// keeps reading the prior (Fresh) state until `commit_publish`.
    pub async fn begin_publish(&self, pin: &SnapshotPin) -> Result<()> {
        self.upsert_projection(&pin.collection_id, |p| {
            p.freshness_state = ProjectionFreshnessState::Updating;
        })
        .await
    }

    /// Atomically commit the republish: mark the discovery projection `Fresh`
    /// and record snapshot lineage. This is the atomic serving switch.
    pub async fn commit_publish(&self, pin: &SnapshotPin) -> Result<()> {
        let source_range = pin.source_range();
        let position = pin.position();
        self.upsert_projection(&pin.collection_id, move |p| {
            p.freshness_state = ProjectionFreshnessState::Fresh;
            p.source_range = Some(source_range);
            p.last_included_position = Some(position);
        })
        .await
    }

    /// Abort an in-flight republish: mark the discovery projection
    /// `RebuildRequired` so planners avoid routing through a half-built result.
    pub async fn abort_publish(&self, pin: &SnapshotPin) -> Result<()> {
        self.upsert_projection(&pin.collection_id, |p| {
            p.freshness_state = ProjectionFreshnessState::RebuildRequired;
        })
        .await
    }

    /// Read the current discovery projection (for EXPLAIN / route-health).
    pub async fn active_projection(
        &self,
        collection_id: &str,
    ) -> Result<Option<CatalogProjection>> {
        let (catalog, identifier) = self.catalog_manager.resolve_table(collection_id).await?;
        if !catalog.table_exists(&identifier).await? {
            return Ok(None);
        }
        let schema = catalog.get_table(&identifier).await?;
        Ok(schema
            .projections
            .into_iter()
            .find(|p| p.name == DISCOVERY_ACTIVE_PROJECTION))
    }

    /// Find-or-create the discovery projection, apply `mutate`, and persist via
    /// `Catalog::replace_table` (in place, no drop window — TD-CATREPLACE-1).
    async fn upsert_projection<F>(&self, collection_id: &str, mutate: F) -> Result<()>
    where
        F: FnOnce(&mut CatalogProjection),
    {
        let (catalog, identifier) = self
            .catalog_manager
            .resolve_table(collection_id)
            .await
            .with_context(|| format!("resolve catalog table for collection '{collection_id}'"))?;

        if !catalog.table_exists(&identifier).await? {
            anyhow::bail!(
                "collection '{collection_id}' is not registered in the catalog; \
                 cannot publish discovery snapshot"
            );
        }

        let mut schema = catalog.get_table(&identifier).await?;
        match schema
            .projections
            .iter_mut()
            .find(|p| p.name == DISCOVERY_ACTIVE_PROJECTION)
        {
            Some(proj) => mutate(proj),
            None => {
                let mut proj = CatalogProjection::rebuildable(
                    DISCOVERY_ACTIVE_PROJECTION,
                    CatalogProjectionKind::VectorAnn,
                    collection_id,
                );
                mutate(&mut proj);
                schema.projections.push(proj);
            }
        }

        // TD-CATREPLACE-1: replace in place — no drop window (the drop+create
        // pattern this retires).
        catalog.replace_table(&identifier, schema).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_catalog::{CatalogColumn, CatalogTableSchema, TableIdentifier};
    use proximadb_data_model::ProximaType;

    async fn coordinator_with_table(collection: &str) -> SnapshotPublishCoordinator {
        let tmp = std::env::temp_dir().join(format!(
            "proximadb_snapshot_test_{}_{}",
            collection,
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&tmp);
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

        SnapshotPublishCoordinator::new(manager)
    }

    #[tokio::test]
    async fn pin_degrades_without_manifest() {
        let coord = coordinator_with_table("c_pin").await;
        let pin = coord.pin("c_pin").await.unwrap();
        assert_eq!(pin.collection_id, "c_pin");
        // The collection was just created with no writes, so it contributes no
        // WAL entries → from_lsn pins at 0.
        assert_eq!(pin.from_lsn, 0);
        // `to_lsn`/`checkpoint_id` track the GLOBAL WAL manifest service, which
        // is a process singleton other tests in this binary may have advanced
        // (the LSN allocator is shared). Assert the range is well-formed and
        // anchored at from_lsn rather than a fixed `wal:0..0`, which is only
        // true when this test runs in isolation.
        assert!(pin.to_lsn >= pin.from_lsn, "range must be well-formed");
        assert!(pin.source_range().starts_with("wal:0.."));
    }

    #[tokio::test]
    async fn begin_then_commit_drives_freshness_and_lineage() {
        let coord = coordinator_with_table("c_pub").await;
        let pin = SnapshotPin {
            collection_id: "c_pub".to_string(),
            from_lsn: 3,
            to_lsn: 9,
            checkpoint_id: 2,
        };

        coord.begin_publish(&pin).await.unwrap();
        let proj = coord.active_projection("c_pub").await.unwrap().unwrap();
        assert_eq!(proj.freshness_state, ProjectionFreshnessState::Updating);

        coord.commit_publish(&pin).await.unwrap();
        let proj = coord.active_projection("c_pub").await.unwrap().unwrap();
        assert_eq!(proj.freshness_state, ProjectionFreshnessState::Fresh);
        assert_eq!(proj.source_range.as_deref(), Some("wal:3..9"));
        assert_eq!(proj.last_included_position.as_deref(), Some("checkpoint:2"));
    }

    #[tokio::test]
    async fn abort_marks_rebuild_required() {
        let coord = coordinator_with_table("c_abort").await;
        let pin = coord.pin("c_abort").await.unwrap();
        coord.begin_publish(&pin).await.unwrap();
        coord.abort_publish(&pin).await.unwrap();
        let proj = coord.active_projection("c_abort").await.unwrap().unwrap();
        assert_eq!(
            proj.freshness_state,
            ProjectionFreshnessState::RebuildRequired
        );
    }

    #[tokio::test]
    async fn publish_to_uncataloged_collection_errors() {
        let coord = coordinator_with_table("c_real").await;
        let pin = coord.pin("c_missing").await.unwrap();
        assert!(coord.begin_publish(&pin).await.is_err());
    }
}
