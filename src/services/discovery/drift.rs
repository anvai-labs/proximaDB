//! `DriftWatcher` — the first *live* producer for the F1 trigger arm (T1.9).
//!
//! The three named signal sources (RecallProbeGate, freshness SLA, AutoML) are
//! not yet driven by live data (audited 2026-05-30 — none have production
//! callers). Write volume, however, IS live. This watcher periodically sums the
//! *records* written to *this collection* (WAL manifest entries' `vector_count`)
//! past the high-water mark captured at its last completed recluster and, once
//! that count crosses a threshold, emits a `WorkloadDrift` signal — closing the
//! flywheel on real data:
//!
//! ```text
//! writes to THIS collection accumulate
//!   -> collection_writes_since(@last_recluster) >= THRESHOLD records
//!   -> DiscoveryService::on_signal(coll, WorkloadDrift)
//!   -> executor runs Recluster -> atomic republish
//!   -> baseline (high-water mark) advances -> drift resets
//! ```
//!
//! Per-collection precision (2026-05-30): the magnitude is a **record count for
//! this collection**, not a global-LSN delta. The LSN allocator is global, so a
//! delta of `global_lsn` values is inflated by *other* collections' writes
//! between this collection's flushes — a low-traffic tenant in a busy system
//! would over-recluster. A per-collection record count is immune to that. The
//! high-water mark (an LSN cutoff) only advances when *this* collection flushes,
//! so a quiet collection never drifts. Considered by name (discovery keys
//! jobs/pins by name); records counted by internal id (the WAL manifest keys
//! entries by uuid) — the catalog list supplies the name->id mapping.
//!
//! Coalescing (in `DiscoveryTrigger`) prevents a sustained drift from flooding
//! the registry; a completed recluster advances the baseline. AutoML will later
//! subsume this as the cost-aware scheduler.
//!
//! Scope: watches collections that already have discovery history (the registry
//! keys off existing jobs). Recluster is currently non-mutating (it computes
//! cluster-quality metrics + republishes freshness), so an on-by-default watcher
//! is safe even if it fires more often than ideal.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::watch;
use tracing::{debug, info, warn};

use super::job::DiscoveryJob;
use super::service::DiscoveryService;
use super::trigger::TriggerSignal;
use crate::services::snapshot::SnapshotPublishCoordinator;

/// Default number of *records* written to *this collection* since the last
/// recluster before drift triggers a new one. A per-collection record count (not
/// a global-LSN delta), so it isn't inflated by other collections' writes.
/// Conservative starting heuristic; tune per workload (AutoML will subsume).
pub const DEFAULT_DRIFT_THRESHOLD_WRITES: u64 = 10_000;
/// Default interval between drift sweeps.
pub const DEFAULT_DRIFT_INTERVAL: Duration = Duration::from_secs(60);

/// Env override for [`DEFAULT_DRIFT_THRESHOLD_WRITES`] (used at watcher
/// construction in `SharedServices`). Lets operators tune drift sensitivity, and
/// lets the e2e drive the loop fast without a code change.
pub const DRIFT_THRESHOLD_ENV: &str = "PROXIMADB_DRIFT_THRESHOLD_WRITES";
/// Env override (whole seconds) for [`DEFAULT_DRIFT_INTERVAL`].
pub const DRIFT_INTERVAL_ENV: &str = "PROXIMADB_DRIFT_INTERVAL_SECS";

/// Resolve the drift threshold from `PROXIMADB_DRIFT_THRESHOLD_WRITES`, else default.
pub fn threshold_writes_from_env() -> u64 {
    parse_threshold(std::env::var(DRIFT_THRESHOLD_ENV).ok())
}

/// Resolve the drift interval from `PROXIMADB_DRIFT_INTERVAL_SECS`, else default.
pub fn interval_from_env() -> Duration {
    parse_interval(std::env::var(DRIFT_INTERVAL_ENV).ok())
}

/// Pure parse of the threshold override (testable without touching the env).
fn parse_threshold(raw: Option<String>) -> u64 {
    raw.and_then(|v| v.trim().parse().ok())
        .unwrap_or(DEFAULT_DRIFT_THRESHOLD_WRITES)
}

/// Pure parse of the interval override. Non-positive / unparseable => default.
fn parse_interval(raw: Option<String>) -> Duration {
    raw.and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|s| *s > 0)
        .map(Duration::from_secs)
        .unwrap_or(DEFAULT_DRIFT_INTERVAL)
}

/// True when `writes_since` (the number of records written to this collection
/// past its recluster baseline) crosses `threshold`. A zero count (never
/// reclustered + no writes, or manifest absent) never signals.
pub fn drift_exceeds(writes_since: u64, threshold: u64) -> bool {
    writes_since >= threshold && threshold > 0
}

/// Background watcher that turns write-volume drift into recluster signals.
pub struct DriftWatcher {
    service: Arc<DiscoveryService>,
    coordinator: Arc<SnapshotPublishCoordinator>,
    threshold_writes: u64,
    /// Source of all collection names, swept even before a collection has any
    /// discovery history (closes the bootstrap gap). `None` => history-only.
    /// Collections with no / too-few indexed vectors simply no-op in the pass.
    /// Names (not ids) — the discovery pipeline keys jobs/pins by name.
    collection_source: Option<Arc<dyn proximadb_runtime::CollectionPort>>,
}

impl DriftWatcher {
    pub fn new(
        service: Arc<DiscoveryService>,
        coordinator: Arc<SnapshotPublishCoordinator>,
        threshold_writes: u64,
    ) -> Self {
        Self {
            service,
            coordinator,
            threshold_writes,
            collection_source: None,
        }
    }

    /// Attach a collection source so the watcher sweeps every collection by name
    /// — not just those with prior discovery history. This is what makes the
    /// loop autonomous for brand-new collections (no operator seed needed).
    pub fn with_collection_source(
        mut self,
        source: Arc<dyn proximadb_runtime::CollectionPort>,
    ) -> Self {
        self.collection_source = Some(source);
        self
    }

    /// One sweep. Considers the union of (a) collections with discovery history
    /// and (b) every collection in the catalog (so a never-reclustered
    /// collection is picked up automatically). Each is considered by **name**
    /// (the discovery pipeline keys jobs/pins by name) but its write volume is
    /// read by **internal id** (the WAL manifest keys entries by object id) — hence
    /// the (name, id) pairing. Returns the number of signals emitted (coalesced
    /// considerations don't count).
    pub async fn tick(&self) -> usize {
        let by_name = self.sweep_targets().await;
        let mut emitted = 0;
        for (name, internal_id) in by_name {
            if self.consider(&name, &internal_id).await.is_some() {
                emitted += 1;
            }
        }
        if emitted > 0 {
            debug!("DriftWatcher: emitted {emitted} recluster signal(s)");
        }
        emitted
    }

    /// The `name -> internal id` pairs one sweep considers. The id is used
    /// solely for the per-collection write-volume lookup; the catalog list is
    /// the authoritative source of the name->id mapping (proto `Collection`
    /// carries both).
    ///
    /// The watcher is a background *system* caller sweeping every tenant's
    /// collections, so it must use the explicit internal listing. The
    /// request-scoped `list_collections(None)` is tenant-denied (no tenant =>
    /// empty listing, since 076592ba1); calling it here left only the
    /// history-only fallback below (name-as-id), whose manifest lookup is
    /// always 0 — so the watcher never signalled.
    async fn sweep_targets(&self) -> std::collections::HashMap<String, String> {
        let mut by_name: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        // The internal listing spans ALL tenants, but the discovery pipeline
        // (baseline, signal, job, executor pin) is keyed by collection NAME and
        // is not tenant-aware. A name owned by more than one collection (two
        // tenants' `orders`) cannot be attributed safely: one tenant's writes
        // would trigger a recluster pinned to the other's collection. Such names
        // are skipped (warned), never guessed (TD-DISCOVERY-TENANT-1).
        let mut ambiguous: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        if let Some(source) = &self.collection_source {
            match source.list_collections_internal().await {
                Ok(cols) => {
                    for c in cols {
                        let Some(name) = c.config.as_ref().map(|cfg| cfg.name.clone()) else {
                            continue;
                        };
                        match by_name.get(&name) {
                            Some(existing) if *existing != c.id => {
                                ambiguous.insert(name);
                            }
                            Some(_) => {}
                            None => {
                                by_name.insert(name, c.id);
                            }
                        }
                    }
                }
                Err(err) => debug!("DriftWatcher: collection listing failed: {err:#}"),
            }
        }
        for name in &ambiguous {
            by_name.remove(name);
        }
        if !ambiguous.is_empty() {
            warn!(
                "DriftWatcher: skipping {} collection name(s) owned by more than one tenant \
                 (discovery is name-keyed; TD-DISCOVERY-TENANT-1): {:?}",
                ambiguous.len(),
                ambiguous
            );
        }
        // History-only collections may not be in the catalog list; fall back to
        // name-as-id (the manifest lookup then returns 0 — no spurious signal).
        // An ambiguous name is never re-added under a guessed id.
        for c in self.service.registry().collections() {
            if !ambiguous.contains(&c) {
                by_name.entry(c.clone()).or_insert_with(|| c.clone());
            }
        }
        by_name
    }

    /// Consider one collection: sum the records written to *this* collection
    /// since its recluster baseline (manifest entries keyed by `internal_id`,
    /// baseline keyed by `name`), signal if that count crosses the threshold. A
    /// missing manifest / unknown id yields 0 writes (no spurious signal).
    async fn consider(&self, name: &str, internal_id: &str) -> Option<DiscoveryJob> {
        // Baseline is the per-collection write high-water-mark (an LSN cutoff)
        // captured at the last recluster; `None` => never reclustered => count
        // all of this collection's writes as drift.
        let baseline = self.service.last_reclustered_lsn(name).unwrap_or(0);
        let writes_since = self
            .coordinator
            .collection_writes_since(internal_id, baseline)
            .await;
        self.consider_with_writes(name, writes_since)
    }

    /// Testable decision core: decide using an explicit per-collection write
    /// count (no manifest needed). `collection_id` is the user-facing **name**
    /// (`on_signal` keys by name). Returns the enqueued job, or `None` if within
    /// threshold or coalesced against an in-flight recluster.
    fn consider_with_writes(&self, collection_id: &str, writes_since: u64) -> Option<DiscoveryJob> {
        if drift_exceeds(writes_since, self.threshold_writes) {
            self.service
                .on_signal(collection_id, TriggerSignal::WorkloadDrift)
        } else {
            None
        }
    }
}

/// Spawn the background drift-watch loop (mirrors `spawn_discovery_executor`).
pub fn spawn_drift_watcher(
    watcher: Arc<DriftWatcher>,
    mut shutdown: watch::Receiver<bool>,
    interval: Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        info!(
            "DriftWatcher started (interval = {interval:?}, threshold = {} record writes)",
            watcher.threshold_writes
        );
        loop {
            tokio::select! {
                changed = shutdown.changed() => {
                    if changed.is_err() || *shutdown.borrow() {
                        break;
                    }
                }
                _ = tokio::time::sleep(interval) => {
                    watcher.tick().await;
                }
            }
        }
        info!("DriftWatcher stopped");
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::CatalogManager;
    use crate::services::discovery::DiscoveryRegistry;

    #[test]
    fn env_override_parsing() {
        assert_eq!(super::parse_threshold(Some("1".to_string())), 1);
        assert_eq!(super::parse_threshold(Some("  42 ".to_string())), 42);
        assert_eq!(
            super::parse_threshold(None),
            super::DEFAULT_DRIFT_THRESHOLD_WRITES
        );
        assert_eq!(
            super::parse_threshold(Some("nope".to_string())),
            super::DEFAULT_DRIFT_THRESHOLD_WRITES
        );
        assert_eq!(
            super::parse_interval(Some("2".to_string())),
            Duration::from_secs(2)
        );
        // Zero / garbage fall back to the default (never a 0s busy-loop).
        assert_eq!(
            super::parse_interval(Some("0".to_string())),
            super::DEFAULT_DRIFT_INTERVAL
        );
        assert_eq!(super::parse_interval(None), super::DEFAULT_DRIFT_INTERVAL);
    }

    #[test]
    fn drift_threshold_math() {
        assert!(!drift_exceeds(5, 10)); // 5 writes < threshold 10
        assert!(drift_exceeds(10, 10)); // exactly at threshold
        assert!(drift_exceeds(64, 10)); // well past
        assert!(!drift_exceeds(0, 10)); // no writes => never
        assert!(!drift_exceeds(100, 0)); // threshold 0 disables (never a 0-write loop)
    }

    async fn watcher() -> (Arc<DriftWatcher>, Arc<DiscoveryRegistry>) {
        // consider_with_writes never touches the manifest, so an empty in-memory
        // catalog suffices for the pure threshold/coalescing core.
        let coordinator = Arc::new(SnapshotPublishCoordinator::new(Arc::new(
            CatalogManager::new(),
        )));
        let registry = Arc::new(DiscoveryRegistry::new());
        let service = Arc::new(DiscoveryService::new(registry.clone(), coordinator.clone()));
        // Threshold = 10 record writes.
        (
            Arc::new(DriftWatcher::new(service, coordinator, 10)),
            registry,
        )
    }

    #[tokio::test]
    async fn signals_when_writes_exceed_threshold() {
        let (w, _r) = watcher().await;
        // 20 record writes >= threshold 10 -> signal.
        assert!(w.consider_with_writes("c1", 20).is_some());
    }

    #[tokio::test]
    async fn within_threshold_does_not_signal() {
        let (w, _r) = watcher().await;
        // 5 record writes < threshold 10 -> no signal.
        assert!(w.consider_with_writes("c1", 5).is_none());
    }

    #[tokio::test]
    async fn degrades_to_no_signal_without_manifest() {
        // `consider` reads the manifest (absent here) -> 0 writes -> no signal,
        // even for a never-reclustered collection. Proves the safe degradation.
        let (w, _r) = watcher().await;
        assert!(w.consider("c1", "c1-id").await.is_none());
    }

    #[tokio::test]
    async fn sustained_drift_is_coalesced_while_recluster_in_flight() {
        let (w, _r) = watcher().await;
        assert!(
            w.consider_with_writes("c1", 50).is_some(),
            "first drift enqueues"
        );
        assert!(
            w.consider_with_writes("c1", 60).is_none(),
            "still drifted but a recluster is in flight -> coalesced"
        );
    }

    /// The re-arm half of the loop the e2e (`discovery_drift_watcher_e2e`)
    /// pins: once the in-flight recluster completes (baseline advances), new
    /// drift must enqueue a SECOND recluster — and the completed job's
    /// `collection_write_lsn` must become the baseline the next sweep reads.
    /// Negative control: while it is still Running, the same drift coalesces.
    #[tokio::test]
    async fn drift_rearms_after_recluster_completes() {
        let (w, r) = watcher().await;
        let mut first = w
            .consider_with_writes("c1", 50)
            .expect("first drift enqueues");
        assert_eq!(w.service.last_reclustered_lsn("c1"), None);

        first.status = super::super::job::DiscoveryJobStatus::Running;
        r.upsert(first.clone());
        assert!(
            w.consider_with_writes("c1", 50).is_none(),
            "Running recluster -> coalesced (negative control)"
        );

        first.status = super::super::job::DiscoveryJobStatus::Complete;
        first.collection_write_lsn = 42;
        r.upsert(first);
        assert_eq!(
            w.service.last_reclustered_lsn("c1"),
            Some(42),
            "completed recluster's HWM becomes the drift baseline"
        );
        assert!(
            w.consider_with_writes("c1", 0).is_none(),
            "no writes past the new baseline -> no signal"
        );
        assert!(
            w.consider_with_writes("c1", 10).is_some(),
            "threshold-many writes past the baseline after completion -> a second recluster"
        );
    }

    /// A port that scopes like `CollectionService`: the request listing denies
    /// a missing tenant (empty), only the internal listing sees everything.
    struct TenantScopedPort {
        /// (internal id, collection name) across ALL tenants.
        collections: Vec<(&'static str, &'static str)>,
    }

    impl TenantScopedPort {
        fn orders() -> Self {
            Self {
                collections: vec![("7", "orders")],
            }
        }
    }

    #[async_trait::async_trait]
    impl proximadb_runtime::CollectionPort for TenantScopedPort {
        async fn get_collection(
            &self,
            _identifier: &str,
            _tenant_id: Option<&str>,
        ) -> anyhow::Result<Option<proximadb_proto::v1::Collection>> {
            Ok(None)
        }
        async fn create_collection(
            &self,
            _config: proximadb_proto::v1::CollectionConfig,
            _tenant_id: Option<&str>,
        ) -> anyhow::Result<proximadb_proto::v1::Collection> {
            anyhow::bail!("unused")
        }
        async fn update_collection(
            &self,
            _id: &str,
            _config: proximadb_proto::v1::CollectionConfig,
            _tenant_id: Option<&str>,
        ) -> anyhow::Result<proximadb_proto::v1::Collection> {
            anyhow::bail!("unused")
        }
        async fn delete_collection(
            &self,
            _id: &str,
            _tenant_id: Option<&str>,
        ) -> anyhow::Result<bool> {
            Ok(false)
        }
        async fn list_collections(
            &self,
            tenant_id: Option<&str>,
        ) -> anyhow::Result<Vec<proximadb_proto::v1::Collection>> {
            // No tenant => denied (empty), as in CollectionService.
            match tenant_id {
                None => Ok(Vec::new()),
                Some(_) => self.list_collections_internal().await,
            }
        }
        async fn list_collections_internal(
            &self,
        ) -> anyhow::Result<Vec<proximadb_proto::v1::Collection>> {
            Ok(self
                .collections
                .iter()
                .map(|(id, name)| proximadb_proto::v1::Collection {
                    id: id.to_string(),
                    config: Some(proximadb_proto::v1::CollectionConfig {
                        name: name.to_string(),
                        ..Default::default()
                    }),
                    ..Default::default()
                })
                .collect())
        }
        async fn resolve_collection_id(&self, identifier: &str) -> anyhow::Result<Option<String>> {
            Ok(Some(identifier.to_string()))
        }
    }

    /// The watcher is a system caller: it must pair each catalog collection's
    /// name with its internal id via the INTERNAL listing. Regression for the
    /// tenant-denied `list_collections(None)` call, which yielded no catalog
    /// collections, so the manifest was looked up by name and drift was
    /// always 0 (the `discovery_drift_watcher_e2e` failure).
    #[tokio::test]
    async fn sweep_targets_pair_catalog_names_with_internal_ids() {
        let coordinator = Arc::new(SnapshotPublishCoordinator::new(Arc::new(
            CatalogManager::new(),
        )));
        let service = Arc::new(DiscoveryService::new(
            Arc::new(DiscoveryRegistry::new()),
            coordinator.clone(),
        ));
        let w = DriftWatcher::new(service, coordinator, 10)
            .with_collection_source(Arc::new(TenantScopedPort::orders()));
        let targets = w.sweep_targets().await;
        assert_eq!(
            targets.get("orders").map(String::as_str),
            Some("7"),
            "catalog name must map to the internal id the WAL manifest keys by"
        );

        // Negative control: without a collection source, only history-only
        // collections (name-as-id) are swept — `orders` is absent.
        let (bare, _r) = watcher().await;
        assert!(bare.sweep_targets().await.is_empty());
    }

    /// Two tenants own a collection named `orders` (internal ids 7 and 9). The
    /// name-keyed discovery pipeline cannot attribute drift to either safely,
    /// so the name is skipped, never resolved to whichever id listed first; a
    /// uniquely named collection is still swept (#2511 review).
    #[tokio::test]
    async fn sweep_targets_skip_names_shared_across_tenants() {
        let coordinator = Arc::new(SnapshotPublishCoordinator::new(Arc::new(
            CatalogManager::new(),
        )));
        let service = Arc::new(DiscoveryService::new(
            Arc::new(DiscoveryRegistry::new()),
            coordinator.clone(),
        ));
        let port = TenantScopedPort {
            collections: vec![("7", "orders"), ("9", "orders"), ("11", "invoices")],
        };
        let w = DriftWatcher::new(service, coordinator, 10).with_collection_source(Arc::new(port));
        let targets = w.sweep_targets().await;
        assert!(
            !targets.contains_key("orders"),
            "an ambiguous cross-tenant name must not be swept: {targets:?}"
        );
        assert_eq!(targets.get("invoices").map(String::as_str), Some("11"));
    }
}
