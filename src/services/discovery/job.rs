//! `DiscoveryJob` control-plane types (Phase 8 F1).

use std::collections::HashMap;

use proximadb_kernel::time::now_unix_ms;

use serde::{Deserialize, Serialize};

/// Kind of offline refinement a discovery job performs over a pinned snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiscoveryJobKind {
    /// Remove near-duplicate records (F1 keystone, S3).
    Dedup,
    /// Recompute IVF clusters / index (later phase — stub).
    Recluster,
    /// Re-embed records with a newer model (later phase — stub).
    ReEmbed,
    /// Scan for quality / drift issues (later phase — stub).
    QualityScan,
    /// Analyze agent execution trajectories (later phase — stub).
    TrajectoryAnalysis,
}

/// Lifecycle state of a discovery job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DiscoveryJobStatus {
    /// Created, waiting for the executor to claim it.
    #[default]
    Scheduled,
    /// Claimed and running (snapshot pinned, refinement in progress).
    Running,
    /// Refined snapshot atomically republished.
    Complete,
    /// Aborted; the discovery projection is `RebuildRequired`.
    Failed,
}

/// A cataloged discovery job: pins a snapshot, refines it, and republishes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryJob {
    /// Stable job identifier (`dj_<uuid>`), never reused.
    pub job_id: String,
    /// Target collection (logical name).
    pub collection_id: String,
    /// Refinement kind.
    pub kind: DiscoveryJobKind,
    /// Lifecycle state.
    pub status: DiscoveryJobStatus,
    /// Creation time (epoch ms).
    pub created_at_ms: i64,
    /// When the executor claimed the job (epoch ms).
    pub started_at_ms: Option<i64>,
    /// When the job reached a terminal state (epoch ms).
    pub completed_at_ms: Option<i64>,
    /// Pinned snapshot lower bound (global LSN).
    pub snapshot_from_lsn: u64,
    /// Pinned snapshot upper bound (global LSN).
    pub snapshot_to_lsn: u64,
    /// Per-collection write high-water-mark at pin time: the highest global LSN
    /// among *this collection's* WAL manifest entries (not the global
    /// allocator). This is the drift baseline — write volume to *this*
    /// collection past it measures per-collection staleness, so a quiet
    /// collection does not drift on other collections' writes. `#[serde(default)]`
    /// keeps older persisted jobs (without the field) deserializable as 0.
    #[serde(default)]
    pub collection_write_lsn: u64,
    /// Manifest checkpoint id at pin time.
    pub checkpoint_id: u64,
    /// Records considered (input to the refinement pass).
    pub input_record_count: u64,
    /// Records retained after refinement.
    pub refined_record_count: u64,
    /// Records removed by refinement (e.g. duplicates).
    pub removed_count: u64,
    /// Failure detail when `status == Failed`.
    pub error: Option<String>,
    /// Refinement quality metrics (pass-specific).
    pub quality_metrics: HashMap<String, f64>,
}

impl DiscoveryJob {
    /// Create a freshly `Scheduled` job for a collection.
    pub fn new(collection_id: impl Into<String>, kind: DiscoveryJobKind) -> Self {
        Self {
            job_id: format!("dj_{}", uuid::Uuid::new_v4().simple()),
            collection_id: collection_id.into(),
            kind,
            status: DiscoveryJobStatus::Scheduled,
            created_at_ms: now_unix_ms(),
            started_at_ms: None,
            completed_at_ms: None,
            snapshot_from_lsn: 0,
            snapshot_to_lsn: 0,
            collection_write_lsn: 0,
            checkpoint_id: 0,
            input_record_count: 0,
            refined_record_count: 0,
            removed_count: 0,
            error: None,
            quality_metrics: HashMap::new(),
        }
    }

    /// Whether the job has reached a terminal (`Complete`/`Failed`) state.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status,
            DiscoveryJobStatus::Complete | DiscoveryJobStatus::Failed
        )
    }
}

/// Outcome of a refinement pass, folded back into the `DiscoveryJob` record.
#[derive(Debug, Clone, Default)]
pub struct DiscoveryJobResult {
    /// Records considered.
    pub input_record_count: u64,
    /// Records retained.
    pub refined_record_count: u64,
    /// Records removed.
    pub removed_count: u64,
    /// Pass-specific quality metrics.
    pub quality_metrics: HashMap<String, f64>,
}
