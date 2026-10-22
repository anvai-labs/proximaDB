//! P2.2c (TD-SEAM-2 / TD-SEAM-12): the canonical-WAL truncator — the first
//! caller of [`FramedTableWalAppender::truncate_below`].
//!
//! Truncation is safe only below the minimum of what every WAL consumer still
//! needs. Each consumer is a [`WalFloorSource`]: `Ok(Some(lsn))` — entries at
//! or above `lsn` must stay; `Ok(None)` — no constraint; `Err` — refuse. One
//! refusal stops the whole run; there is no partial answer.
//!
//! Order matters twice:
//!
//! * The WAL head `W` is read FIRST, and the floor never exceeds `W + 1`, so an
//!   entry appended while the floor is being computed is never a candidate.
//! * Sources that only READ run before the catalog checkpoints, which APPEND:
//!   a run that some reader refuses costs no WAL growth.
//!
//! On by default (`PROXIMADB_CANONICAL_WAL_TRUNCATE_INTERVAL_SECS`, `0`/`off`
//! disables it). Truncation only ever removes SEALED segments, so the server
//! schedules it only with segment rotation on
//! (`PROXIMADB_CANONICAL_WAL_SEGMENT_BYTES`), and only after a complete WAL
//! recovery.
//!
//! A source that APPENDS ([`WalFloorSource::appends`]) is asked only once the
//! read-only sources' floor would truncate something, so an idle server's runs
//! add nothing to the WAL.

use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;

use crate::services::canonical_wal::{BranchFenceRefused, FramedTableWalAppender, WalTruncation};
use crate::services::record_store::DirectWalTableRecordStore;
use crate::services::{FunctionStore, RankProfileStore};

/// Env gate: seconds between truncation runs. Default ON
/// ([`DEFAULT_TRUNCATE_INTERVAL_SECS`](crate::services::storage_format::DEFAULT_TRUNCATE_INTERVAL_SECS));
/// `0` or `off` disables it. See [`crate::services::storage_format`].
pub const TRUNCATE_INTERVAL_ENV: &str = "PROXIMADB_CANONICAL_WAL_TRUNCATE_INTERVAL_SECS";

/// How many of the newest WAL sequences are always kept for change-feed
/// clients, whose cursors the server cannot see. A client further behind gets
/// `410 Gone` and re-syncs from a snapshot.
pub const RETAIN_ENTRIES_ENV: &str = "PROXIMADB_CANONICAL_WAL_RETAIN_ENTRIES";

/// Default for [`RETAIN_ENTRIES_ENV`].
pub const DEFAULT_RETAIN_ENTRIES: u64 = 1_000_000;

/// The configured run interval, or `None` when the truncator is off.
pub fn configured_interval() -> Option<std::time::Duration> {
    crate::services::storage_format::default_on_gate(
        TRUNCATE_INTERVAL_ENV,
        crate::services::storage_format::DEFAULT_TRUNCATE_INTERVAL_SECS,
    )
    .map(std::time::Duration::from_secs)
}

/// The configured change-feed retention ([`RETAIN_ENTRIES_ENV`]).
pub fn configured_retain_entries() -> u64 {
    std::env::var(RETAIN_ENTRIES_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_RETAIN_ENTRIES)
}

/// One consumer's answer to "which WAL entries do you still need?".
#[async_trait]
pub trait WalFloorSource: Send + Sync {
    /// Short name, used in logs and refusal reasons.
    fn name(&self) -> &str;
    /// `Ok(Some(lsn))`: entries `>= lsn` must stay. `Ok(None)`: no constraint.
    /// `Err`: do not truncate. `head` is the WAL's last sequence, read before
    /// any source is asked.
    async fn floor(&self, head: u64) -> Result<Option<u64>>;
    /// `true` for a source whose `floor` appends to the WAL (a checkpoint).
    /// Such sources run last, and only when the run can truncate something.
    fn appends(&self) -> bool {
        false
    }
}

/// Outcome of one run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TruncationRun {
    /// A source refused; nothing was deleted.
    Refused { source: String, reason: String },
    /// The floor did not reach past what is already truncated.
    NothingToDo { floor: u64 },
    /// Segments were deleted.
    Truncated { floor: u64, report: WalTruncation },
}

/// Compose the floor from `sources` (in order) and truncate below it.
pub async fn run_once(
    wal: &FramedTableWalAppender,
    sources: &[Arc<dyn WalFloorSource>],
) -> Result<TruncationRun> {
    let head = wal.last_sequence();
    let mut floor = head.saturating_add(1);
    let (readers, appenders): (Vec<_>, Vec<_>) =
        sources.iter().partition(|source| !source.appends());
    for (pass, sources) in [readers, appenders].into_iter().enumerate() {
        // Nothing to drop on the readers' answer alone: stop before any
        // appending source grows the WAL for nothing.
        if pass == 1 && floor <= wal.retained_from().max(1) {
            return Ok(TruncationRun::NothingToDo { floor });
        }
        for source in sources {
            match source.floor(head).await {
                Ok(Some(source_floor)) => floor = floor.min(source_floor),
                Ok(None) => {}
                Err(error) => {
                    return Ok(TruncationRun::Refused {
                        source: source.name().to_string(),
                        reason: format!("{error:#}"),
                    });
                }
            }
        }
    }
    if floor <= wal.retained_from().max(1) {
        return Ok(TruncationRun::NothingToDo { floor });
    }
    let report = match wal.truncate_below(floor).await {
        Ok(report) => report,
        Err(error) => {
            if let Some(refused) = error.downcast_ref::<BranchFenceRefused>() {
                return Ok(TruncationRun::Refused {
                    source: "branch-fence".to_string(),
                    reason: refused.0.clone(),
                });
            }
            return Err(error);
        }
    };
    if report.segments == 0 {
        return Ok(TruncationRun::NothingToDo { floor });
    }
    Ok(TruncationRun::Truncated { floor, report })
}

/// Refuse while the WAL holds branch-tagged records (TD-SEAM-12 hazard 2):
/// branch merge reads base records that may predate any floor, and no durable
/// merge-base snapshot exists yet.
pub struct BranchEntriesGuard(pub Arc<FramedTableWalAppender>);

#[async_trait]
impl WalFloorSource for BranchEntriesGuard {
    fn name(&self) -> &str {
        "branch-merge"
    }
    async fn floor(&self, _head: u64) -> Result<Option<u64>> {
        if self.0.has_branch_entries() {
            anyhow::bail!(
                "the WAL holds branch-tagged records; branch merge needs base records that may \
                 predate any floor until a durable merge-base snapshot exists"
            );
        }
        Ok(None)
    }
}

/// Keep the newest `retain` sequences for change-feed clients.
pub struct ChangeFeedRetention(pub u64);

#[async_trait]
impl WalFloorSource for ChangeFeedRetention {
    fn name(&self) -> &str {
        "change-feed-retention"
    }
    async fn floor(&self, head: u64) -> Result<Option<u64>> {
        Ok(Some(head.saturating_add(1).saturating_sub(self.0)))
    }
}

/// The relational record tier: rows not yet durable in a segment, and writes
/// appended but not yet applied. A memtable partition refuses (its rows exist
/// only in memory), so this source refuses unless every partition spills.
pub struct RecordTierFloor(pub Arc<DirectWalTableRecordStore>);

#[async_trait]
impl WalFloorSource for RecordTierFloor {
    fn name(&self) -> &str {
        "record-tier"
    }
    async fn floor(&self, _head: u64) -> Result<Option<u64>> {
        let inflight = self.0.inflight_floor();
        let unflushed = self.0.min_unflushed_lsn().await?;
        Ok(match (inflight, unflushed) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        })
    }
}

/// Published OLAP snapshots (ADR-025): the delta merge reads every entry after
/// a table's `snapshot_lsn`, so those must stay. Found by walking every
/// registered catalog; a catalog that cannot be walked refuses (it may hold a
/// snapshot the floor would otherwise pass).
pub struct PublishedSnapshotFloor(pub Arc<crate::catalog::CatalogManager>);

#[async_trait]
impl WalFloorSource for PublishedSnapshotFloor {
    fn name(&self) -> &str {
        "published-snapshots"
    }
    async fn floor(&self, _head: u64) -> Result<Option<u64>> {
        let schemas = self
            .0
            .table_schemas(|_| true)
            .await
            .context("walking the catalogs for published snapshot LSNs")?;
        Ok(schemas
            .iter()
            .flat_map(|schema| &schema.storage_layouts)
            .filter_map(|layout| layout.properties.get("snapshot_lsn")?.parse::<u64>().ok())
            .map(|lsn| lsn.saturating_add(1))
            .min())
    }
}

/// Rank profiles: re-append the live set, needing nothing below it.
pub struct RankProfileCheckpoint(pub Arc<dyn RankProfileStore>);

#[async_trait]
impl WalFloorSource for RankProfileCheckpoint {
    fn name(&self) -> &str {
        "rank-profiles"
    }
    fn appends(&self) -> bool {
        true
    }
    async fn floor(&self, _head: u64) -> Result<Option<u64>> {
        self.0.checkpoint().await
    }
}

/// SQL functions: re-append the live set, needing nothing below it.
pub struct FunctionCheckpoint(pub Arc<dyn FunctionStore>);

#[async_trait]
impl WalFloorSource for FunctionCheckpoint {
    fn name(&self) -> &str {
        "sql-functions"
    }
    fn appends(&self) -> bool {
        true
    }
    async fn floor(&self, _head: u64) -> Result<Option<u64>> {
        self.0.checkpoint().await
    }
}

/// Run [`run_once`] every `interval` until the process-wide shutdown fires.
/// Every outcome is logged; a refusal or an error is not fatal — the next run
/// re-evaluates from scratch.
pub fn spawn(
    wal: Arc<FramedTableWalAppender>,
    sources: Vec<Arc<dyn WalFloorSource>>,
    interval: std::time::Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        ticker.tick().await; // the first tick is immediate; let the server settle
        loop {
            ticker.tick().await;
            match run_once(&wal, &sources).await {
                Ok(TruncationRun::Truncated { floor, report }) => tracing::info!(
                    floor,
                    segments = report.segments,
                    truncated_through = report.truncated_through,
                    "canonical WAL truncated"
                ),
                Ok(TruncationRun::NothingToDo { floor }) => {
                    tracing::debug!(floor, "canonical WAL truncation: nothing below the floor")
                }
                Ok(TruncationRun::Refused { source, reason }) => {
                    tracing::info!(source, reason, "canonical WAL truncation refused this run")
                }
                Err(error) => tracing::warn!(
                    error = %format!("{error:#}"),
                    "canonical WAL truncation failed this run"
                ),
            }
        }
    })
}
