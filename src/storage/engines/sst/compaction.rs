/*
 * Copyright 2025 Vijaykumar Singh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! SST File Compaction for LSM Tree
//!
//! Implements level-based compaction strategy to prevent unbounded growth
//! of SST files. Uses background workers to merge files when thresholds are exceeded.

use super::block_format::BlockFormat;
use super::compactor_impl::{SstCompactor, ZeroCopyCompactionStats};
use super::retirement_ledger;
use crate::core::search::mvcc_resolution::MvccResolver;
use crate::core::{SstConfig, String}; // OPTIMIZED: VectorRecord imported above
use crate::proto::proximadb_v1::VectorRecord; // OPTIMIZED: Added VectorRecord import
use crate::storage::Result;
// Removed ZeroCopyIOSystem - using UnifiedCachingFilesystem instead
use crate::storage::engines::core::formats::arrow_block::{ArrowBlockConfig, ArrowBlockWriter};
use crate::storage::engines::sst::readers::sst_query_engine::UnifiedSstableReader;
use crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem;
use crate::storage::persistence::filesystem::{FilesystemError, FilesystemFactory, FsResult};
use crate::storage::transaction_coordinator::{
    StagingConfig, TransactionCoordinator, TransactionStageType,
};
use proximadb_records::ProximaRecord;
// Quantization now handled by unified compute module
// Import unified level-based compaction framework
use crate::storage::common::compaction_utils::{
    CompactionTaskBuilder, StorageEngineType as CompactionEngineType,
};
use crate::storage::common::*;

/// Temporary compatibility structure for level-based compaction task
#[derive(Debug, Clone)]
pub struct LevelBasedSstCompactionTask {
    pub level: u32,
    pub input_files: Vec<String>,
    pub input_bytes: u64,
    pub target_level: u32,
    pub extension: String,
}
use chrono::Utc;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::{Mutex, Notify, OnceCell, RwLock};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

/// Backwards-compat alias for [`SstCompactionTask`].
pub type CompactionTask = SstCompactionTask;

/// TD-PRECISE-GLOBAL: process-global `CanonicalPrecisionResolver`, set ONCE at
/// boot (database.rs) and consulted by EVERY `Compaction` as the fallback when
/// its per-instance resolver is unset. This closes the "wrong instance" wiring
/// defect: the SST engine creates a fresh `Compaction` per collection (each with
/// its own empty resolver), and the boot-time `storage_engine
/// .compaction_manager().set_precision_resolver(..)` targeted the StorageEngine's
/// idle instance — so fp16/bf16/int8 collections silently degraded to fp32 at
/// compaction. A global (same pattern as `GLOBAL_SST_AXIS_MANAGER` /
/// `GLOBAL_WARM_TIER_CACHES` in core.rs) reaches every per-collection Compaction
/// without per-instance wiring. Per-instance wiring (`set_precision_resolver` /
/// `with_precision_resolver`) still takes precedence for tests/overrides.
static GLOBAL_PRECISION_RESOLVER: std::sync::OnceLock<
    Arc<proximadb_catalog::canonical_precision::CanonicalPrecisionResolver>,
> = std::sync::OnceLock::new();

/// Set the process-global precision resolver (first writer wins; idempotent
/// thereafter — mirrors the OnceLock registry pattern). Called once at boot.
pub fn set_global_precision_resolver(
    resolver: Arc<proximadb_catalog::canonical_precision::CanonicalPrecisionResolver>,
) {
    let _ = GLOBAL_PRECISION_RESOLVER.set(resolver);
}

/// The process-global precision resolver, if boot armed one.
pub fn get_global_precision_resolver()
-> Option<&'static Arc<proximadb_catalog::canonical_precision::CanonicalPrecisionResolver>> {
    GLOBAL_PRECISION_RESOLVER.get()
}

/// Stage-boundary process-RSS checkpoints for the unified compaction pipeline
/// (memory-holder triage: the PAX writer's own peak buffer is ~65 MB while the
/// process grows by tens of GB at N=120k, so the holder is an intermediate
/// stage — pin it by measuring, not assuming).
///
/// Zero work unless `PROXIMADB_TRACE_COMPACTION_MEM` is set (read once per
/// process). When enabled, each pipeline stage boundary emits ONE line to both
/// `tracing::info!` and stderr:
///
/// `[COMPACT mem] stage=<name> records=<n> rss_mb=<resident MB> delta_mb=<since previous stage>`
///
/// plus a one-time canonical-record deep-size sample after input collection:
///
/// `[COMPACT mem] sample pr_bytes=<..> emb_count=<..>`
struct CompactionMemTrace {
    /// `Some` only when tracing is enabled (holds the sysinfo handle so the
    /// disabled path allocates nothing and refreshes nothing).
    sys: Option<Box<sysinfo::System>>,
    pid: sysinfo::Pid,
    last_rss_mb: i64,
}

impl CompactionMemTrace {
    fn enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| {
            std::env::var("PROXIMADB_TRACE_COMPACTION_MEM")
                .map(|v| !v.is_empty() && v != "0")
                .unwrap_or(false)
        })
    }

    /// Construct the tracer; captures the baseline RSS when enabled so the
    /// first stage's `delta_mb` is relative to compaction entry.
    fn new() -> Self {
        let pid = sysinfo::Pid::from_u32(std::process::id());
        let mut tracer = Self {
            sys: Self::enabled().then(|| Box::new(sysinfo::System::new())),
            pid,
            last_rss_mb: 0,
        };
        tracer.last_rss_mb = tracer.current_rss_mb().unwrap_or(0);
        tracer
    }

    /// Refresh ONLY the current process (cheap at stage granularity) and
    /// return its resident set in MB. `None` when tracing is disabled.
    fn current_rss_mb(&mut self) -> Option<i64> {
        let sys = self.sys.as_mut()?;
        sys.refresh_process_specifics(self.pid, sysinfo::ProcessRefreshKind::new().with_memory());
        sys.process(self.pid)
            .map(|p| (p.memory() / (1024 * 1024)) as i64)
    }

    /// Emit one stage-boundary checkpoint line. No-op when disabled.
    fn stage(&mut self, stage: &str, records: usize) {
        let Some(rss_mb) = self.current_rss_mb() else {
            return;
        };
        let delta_mb = rss_mb - self.last_rss_mb;
        self.last_rss_mb = rss_mb;
        let line = format!(
            "[COMPACT mem] stage={stage} records={records} rss_mb={rss_mb} delta_mb={delta_mb}"
        );
        tracing::info!("{line}");
        eprintln!("{line}");
    }

    /// One-time serialized deep-size sample of the canonical record. The
    /// compaction pipeline intentionally has no legacy `VectorRecord` copy.
    fn sample_record(&self, pr: &ProximaRecord) {
        if self.sys.is_none() {
            return;
        }
        let pr_bytes = bincode::serialize(pr).map(|b| b.len()).unwrap_or(0);
        let emb_count = pr.embeddings.len();
        let line = format!("[COMPACT mem] sample pr_bytes={pr_bytes} emb_count={emb_count}");
        tracing::info!("{line}");
        eprintln!("{line}");
    }
}

/// Compaction task to be processed by background workers
#[derive(Debug, Clone)]
pub struct SstCompactionTask {
    /// Globally unique catalog object identity used to re-evaluate
    /// higher-level work after this morsel publishes. Paths and collection
    /// aliases are not authoritative identity sources.
    pub collection_object_id: crate::core::stable_id::CollectionObjectId,
    /// Composite identity (ADR-0083 rev2) for observability — which tenant/ns/coll
    /// this task compacts. The compaction active-map is path-keyed (not u64-keyed),
    /// so this is diagnostic, not a key.
    pub collection_identity: crate::core::stable_id::CollectionIdentity,
    pub level: u8,
    pub input_files: Vec<PathBuf>,
    /// Immutable input bytes used by process-wide weighted memory admission.
    pub input_bytes: u64,
    pub output_file: PathBuf,
    pub priority: CompactionPriority,
    /// Block size in KB for compacted output (uses server default if None)
    pub block_size_kb: Option<u32>,
    /// Compression configuration (uses server default if None)
    pub compression_config: Option<crate::proto::proximadb_v1::CompressionConfig>,
    /// Target embedding precision to enforce before flushing the rewritten
    /// block. When `None`, canonical records keep the precision they carried
    /// into compaction. When `Some(target)`, every embedding is coerced before
    /// either PAX or Arrow writes it.
    ///
    /// The compaction scheduler populates this from
    /// `CanonicalPrecisionResolver::resolve` for the output collection;
    /// callers constructing tasks manually leave it `None`.
    pub precision_hint: Option<proximadb_records::EmbeddingScalarType>,
    /// TD-PAXRG-1: the collection's row-group Region D write switch, resolved
    /// by the task producer from the `pax_rg_layout` tag (precedence tag >
    /// env > default). `None` = follow the env gate
    /// (`PROXIMADB_PAX_WRITE_RG_LAYOUT`, default ON post-flip). The resolved
    /// value is honored at BOTH compaction executors (spill + atomic/in-place)
    /// so an opted-out collection's rewrite output stays intent-matched.
    pub rg_layout: Option<bool>,
}

fn training_follow_up_threshold(
    training_chain_active: bool,
    source_level: u8,
    output_file: &Path,
) -> Option<usize> {
    let writes_trained_pax = source_level == 0
        && output_file
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("pax"));
    (training_chain_active || writes_trained_pax).then_some(1)
}

fn retain_training_guard_for_follow_up(
    training_chain_active: bool,
    scheduled_follow_up_level: Option<u8>,
) -> bool {
    training_chain_active && scheduled_follow_up_level.is_some()
}

/// Priority levels for compaction tasks
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum CompactionPriority {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3, // When storage is nearly full
}

/// Backwards-compat alias for [`SstCompactionStats`].
pub type CompactionStats = SstCompactionStats;

/// Statistics for compaction operations
#[derive(Debug, Clone, Default)]
pub struct SstCompactionStats {
    pub total_compactions: u64,
    pub bytes_written: u64,
    pub bytes_read: u64,
    pub files_merged: u64,
    pub avg_compaction_time_ms: u64,
    pub last_compaction_time: Option<chrono::DateTime<chrono::Utc>>,
    pub expired_records_deleted: u64,
    pub tombstones_removed: u64,
}

/// Enhanced compaction statistics with vector tracking for AXIS integration
#[derive(Debug, Clone, Default)]
pub struct EnhancedSstCompactionStats {
    /// Basic compaction statistics
    pub base_stats: SstCompactionStats,

    /// Vector IDs that were deleted (expired or tombstoned)
    pub deleted_vector_ids: Vec<String>,

    /// Vectors that were merged/updated
    pub merged_vectors: Vec<VectorRecord>,

    /// Whether a full index rebuild is recommended
    pub recommend_full_rebuild: bool,
}

/// Whether a caller needs the legacy AXIS update payload in enhanced stats.
/// Background compaction consumes only `base_stats`, so enabling tracking
/// there would retain a second full corpus until the write completes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MergedVectorTracking {
    Disabled,
    Enabled,
}

struct CompactionRunningMetricGuard;

impl CompactionRunningMetricGuard {
    fn begin() -> Self {
        crate::metrics::operational_metrics::COMPACTION_RUNNING.inc();
        Self
    }
}

impl Drop for CompactionRunningMetricGuard {
    fn drop(&mut self) {
        crate::metrics::operational_metrics::COMPACTION_RUNNING.dec();
    }
}

struct PreparedCompactionRecords {
    records: Vec<ProximaRecord>,
    deleted_vector_ids: Vec<String>,
    merged_vectors: Vec<VectorRecord>,
    expired_records_count: u64,
    tombstones_removed_count: u64,
}

/// Manages background compaction of SST files
pub struct Compaction {
    config: SstConfig,
    task_queue: Arc<Mutex<VecDeque<SstCompactionTask>>>,
    task_notify: Arc<Notify>,
    /// Background worker join handles. Interior-mutable so `start_workers` can
    /// run via the `self: &Arc<Self>` receiver (workers reuse this persistent
    /// `Compaction` — no per-task `temp_manager` construction).
    worker_handles: std::sync::Mutex<Vec<JoinHandle<()>>>,
    shutdown_signal: Arc<AtomicBool>,
    stats: Arc<RwLock<SstCompactionStats>>,
    active_compactions: Arc<RwLock<HashMap<String, SstCompactionTask>>>,
    #[allow(dead_code)]
    atomic_coordinator: Option<Arc<TransactionCoordinator>>,
    unified_reader: Arc<UnifiedSstableReader>,
    sst_compactor: Option<Arc<SstCompactor>>,
    filesystem_factory: Arc<FilesystemFactory>,
    /// Process-owned local scratch namespace. Present only when spill is
    /// enabled; every task receives a deterministic child with a manifest.
    spill_scratch_owner: Option<crate::storage::engines::sst::compaction_spill::SpillScratchOwner>,
    /// New compaction orchestrator
    #[allow(dead_code)]
    compaction_orchestrator: Option<Arc<CompactionOrchestrator>>,
    /// Set-once resolver — initialized after bootstrap once the catalog
    /// handle becomes available (the storage engine constructs Compaction
    /// before SharedServices does, so the resolver can't be injected at
    /// construction time). When initialized, `check_compaction_needed`
    /// populates the produced `SstCompactionTask.precision_hint` from
    /// `resolver.resolve(table_id)` so the writer reconstitution sites
    /// coerce records to the collection's canonical precision before
    /// flushing. When unset, `precision_hint` stays `None` (records keep
    /// whatever precision the VectorRecord intermediate hands them —
    /// always fp32 today).
    precision_resolver:
        Arc<OnceCell<Arc<proximadb_catalog::canonical_precision::CanonicalPrecisionResolver>>>,
    // manifest: Option<Arc<super::SstManifest>>, // Removed - using directory discovery
    /// TD-COMPACT-8 / TD-COMPACT-6 (ADR-076 D1): per-collection
    /// `training_in_flight` guard, shared (same `Arc`) with the `SstEngine`
    /// that owns this `Compaction`. The flush path inserts the collection's
    /// `storage_url` before enqueuing a training compaction; the background
    /// worker removes it after the compaction completes (Ok or Err). This
    /// closes the async gap the inline path did not have: while a training
    /// pass is queued or running, `should_trigger_compaction` skips re-arming
    /// (eliminating the redundant re-training loop). Instances constructed
    /// without `set_training_in_flight` (e.g. the worker's `temp_manager`)
    /// carry a disconnected empty set — correct, since they never arm the guard.
    training_in_flight: Arc<std::sync::Mutex<std::collections::HashSet<String>>>,
}

impl std::fmt::Debug for Compaction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Compaction")
            .field("config", &self.config)
            .field("task_queue", &"<task_queue>")
            .field("task_notify", &"<task_notify>")
            .field(
                "worker_handles",
                &self.worker_handles.lock().map(|h| h.len()).unwrap_or(0),
            )
            .field("shutdown_signal", &self.shutdown_signal)
            .field("stats", &"<stats>")
            .field("active_compactions", &"<active_compactions>")
            .field("atomic_coordinator", &self.atomic_coordinator.is_some())
            .field("unified_reader", &"<unified_reader>")
            .field("sst_compactor", &self.sst_compactor.is_some())
            .field("filesystem_factory", &"<filesystem_factory>")
            .field(
                "spill_scratch_owner",
                &self.spill_scratch_owner.as_ref().map(|owner| owner.path()),
            )
            .finish()
    }
}

/// Decide a compaction output segment's format from its INPUT segments —
/// format-preserving compaction (M1, ADR-049). M1-3: PAX is the only vector
/// write format (the legacy `.sst` streaming write is retired), so `.pax` inputs
/// compact to `.pax` AND legacy `.sst` inputs compact FORWARD to `.pax` (compaction
/// is the migration on-ramp — old segments become PAX at the next merge). Arrow
/// inputs stay Arrow. Mixed inputs consolidate to the richest format present
/// (PAX > Arrow). The flush-side kill-switch / opt-out now select the RawF32
/// quant at flush, so compaction simply follows the resulting PAX inputs.
fn compaction_output_format<P: AsRef<std::path::Path>>(input_files: &[P]) -> BlockFormat {
    if input_files
        .iter()
        .any(|f| f.as_ref().extension().is_some_and(|e| e == "pax"))
    {
        return BlockFormat::PaxBlock;
    }
    if input_files
        .iter()
        .any(|f| f.as_ref().extension().is_some_and(|e| e == "arrow"))
    {
        return BlockFormat::ArrowBlock;
    }
    // M1-3: legacy `.sst` / no-extension inputs compact FORWARD to PAX (the
    // streaming write path is retired; compaction no longer re-emits `.sst`).
    BlockFormat::PaxBlock
}

fn atomic_coordinator_staging(task: &SstCompactionTask) -> Result<StagingConfig> {
    let parent = task.output_file.parent().ok_or_else(|| {
        crate::core::StorageError::SstEngine(
            "local-spill compaction output has no parent".to_string(),
        )
    })?;
    Ok(StagingConfig {
        base_url: parent.to_string_lossy().into_owned(),
        collection_id: None,
        operation_type: TransactionStageType::Compaction,
        skip_uuid_subdir: true,
        ..Default::default()
    })
}

impl Compaction {
    /// Filesystem factory as the extracted-port view (TD-DECOMP-82).
    fn filesystem_port(&self) -> std::sync::Arc<dyn proximadb_storage_ports::FilesystemPort> {
        self.filesystem_factory.clone()
    }

    /// Extract collection ID from file paths
    #[allow(dead_code)]
    fn extract_collection_id_from_paths(&self, paths: &[PathBuf]) -> Result<String> {
        if paths.is_empty() {
            return Ok("unknown".to_string());
        }

        // Extract collection ID from path like: /path/to/collection_id/data/level0/file.sst
        if let Some(path) = paths.first()
            && let Some(_parent) = path.parent()
            && let Some(_parent_parent) = _parent.parent()
            && let Some(parent_parent_parent) = _parent_parent.parent()
            && let Some(_collection_id) = parent_parent_parent.file_name()
        {
            return Ok(_collection_id.to_string_lossy().to_string());
        }

        Ok("unknown".to_string())
    }

    /// Create a new compaction manager
    pub async fn new(config: SstConfig) -> Result<Self> {
        Self::with_atomic_coordinator(config, None).await
    }

    /// Create a new compaction manager with atomic coordinator
    pub async fn with_atomic_coordinator(
        config: SstConfig,
        atomic_coordinator: Option<Arc<TransactionCoordinator>>,
    ) -> Result<Self> {
        let spill_scratch_owner = if let Some(compaction_config) = config.compaction_config.as_ref()
            && compaction_config.spill_enabled
        {
            let spill_directory = compaction_config
                .spill_directory
                .as_deref()
                .filter(|path| !path.trim().is_empty())
                .ok_or_else(|| {
                    crate::core::StorageError::SstEngine(
                        "compaction spill is enabled without a local spill_directory".to_string(),
                    )
                })?;
            let owner = crate::storage::engines::sst::compaction_spill::SpillScratchOwner::open(
                Path::new(spill_directory),
            )
            .map_err(|error| {
                crate::core::StorageError::SstEngine(format!(
                    "initialize compaction spill ownership at {spill_directory}: {error}"
                ))
            })?;
            info!(
                path = %owner.path().display(),
                reclaimed_owner_directories = owner.reclaimed_owner_directories(),
                "Compaction local-spill scratch owner initialized"
            );
            Some(owner)
        } else {
            None
        };

        // Create unified reader with filesystem factory
        let filesystem_factory = Arc::new(
            crate::storage::persistence::filesystem::FilesystemFactory::create(
                crate::storage::persistence::filesystem::FilesystemConfig::default(),
            )
            .await
            .map_err(|e| crate::core::StorageError::SstEngine(e.to_string()))?,
        );
        // Create unified caching filesystem for compaction
        let base_fs = filesystem_factory.get_filesystem("file://").map_err(|e| {
            crate::core::StorageError::SstEngine(format!("Failed to get base filesystem: {}", e))
        })?;
        let unified_fs = Arc::new(UnifiedCachingFilesystem::new(
            base_fs,
            "compaction".to_string(), // collection_id
            "sst_compaction".to_string(),
        ));
        let unified_reader = Arc::new(UnifiedSstableReader::new(
            filesystem_factory.clone(),
            unified_fs,
            String::from("compaction"),
        ));

        // Initialize zero-copy compactor with proper block size from config
        let sst_compactor = if let Some(ref _coord) = atomic_coordinator {
            // Create MVCC resolver for the compactor
            let mvcc_resolver = Arc::new(MvccResolver::new());
            debug!(
                "🔍 COMPACTION_MANAGER: Creating SstCompactor with block_size_kb: {}",
                config.block_size_kb
            );
            Some(Arc::new(SstCompactor::with_block_size(
                filesystem_factory.clone(),
                Some(mvcc_resolver),
                config.block_size_kb,
            )))
        } else {
            // No atomic coordinator, create compactor without MVCC resolver
            debug!(
                "🔍 COMPACTION_MANAGER: Creating SstCompactor with block_size_kb: {}",
                config.block_size_kb
            );
            Some(Arc::new(SstCompactor::with_block_size(
                filesystem_factory.clone(),
                None,
                config.block_size_kb,
            )))
        };

        // Initialize new compaction orchestrator with SST-specific configuration
        let compaction_config = CompactionConfig {
            level0_threshold: config.compaction_threshold as usize,
            level_threshold: (config.compaction_threshold * 2) as usize,
            max_level: config.level_count as u32,
            max_concurrent_per_collection: 1,
            global_max_concurrent: 4,
            operation_timeout: std::time::Duration::from_secs(3600),
            queue_aware_compaction: true,
            max_queue_wait: std::time::Duration::from_secs(60),
            urgency_threshold: 0.8,
        };

        let orchestrator = Some(Arc::new(CompactionOrchestrator::new(
            filesystem_factory.clone(),
            compaction_config,
        )));

        Ok(Self {
            config,
            task_queue: Arc::new(Mutex::new(VecDeque::new())),
            task_notify: Arc::new(Notify::new()),
            worker_handles: std::sync::Mutex::new(Vec::new()),
            shutdown_signal: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(RwLock::new(SstCompactionStats::default())),
            active_compactions: Arc::new(RwLock::new(HashMap::new())),
            atomic_coordinator,
            unified_reader,
            sst_compactor,
            filesystem_factory,
            spill_scratch_owner,
            compaction_orchestrator: orchestrator,
            precision_resolver: Arc::new(OnceCell::new()),
            // TD-COMPACT-6 D1: disconnected empty set until
            // `set_training_in_flight` links this instance to the owning
            // `SstEngine`'s guard (done in core.rs before `start_workers`).
            training_in_flight: Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
        })
    }

    /// TD-COMPACT-6 (ADR-076 D1): link this `Compaction` to the owning
    /// `SstEngine`'s `training_in_flight` guard by sharing the same `Arc`.
    /// Called once, in core.rs, after construction and before
    /// `start_workers` (so the workers clone the linked handle). Instances
    /// that never call this (e.g. the worker's own `temp_manager`) keep a
    /// private empty set — harmless, since they never arm the guard.
    pub fn set_training_in_flight(
        &mut self,
        guard: Arc<std::sync::Mutex<std::collections::HashSet<String>>>,
    ) {
        self.training_in_flight = guard;
    }

    /// TD-COMPACT-8: is a training compaction in-flight for this collection?
    /// Read by `SstEngine::should_trigger_compaction` (via
    /// `compaction_manager()`) to skip re-arming the TD-COMPACT-5 training
    /// arm while a training pass is queued or running.
    pub fn training_in_flight_for(&self, storage_url: &str) -> bool {
        self.training_in_flight
            .lock()
            .is_ok_and(|guard| guard.contains(storage_url))
    }

    /// TD-COMPACT-8: mark a training compaction in-flight for a collection
    /// (insert before enqueuing). Called by the flush path.
    pub fn mark_training_in_flight(&self, storage_url: &str) {
        if let Ok(mut guard) = self.training_in_flight.lock() {
            guard.insert(storage_url.to_string());
        }
    }

    /// Attach a `CanonicalPrecisionResolver` so produced `SstCompactionTask`s
    /// carry `precision_hint = Some(target)` for fp16 / non-fp32 collections.
    /// Without this, all tasks ship with `precision_hint: None` and
    /// compaction writes records at whatever precision the VectorRecord
    /// intermediate produced (today: always fp32).
    ///
    /// Maps `collection_id` strings to `TableIdentifier` using the same
    /// convention as `CollectionManager::collection_table_identifier`:
    /// `TableIdentifier::parse(id)`, with unqualified names defaulting
    /// to the `"default"` namespace.
    ///
    /// Builder form — consumes self. For post-construction injection
    /// (the bootstrap path, where Compaction is wrapped in `Arc` before
    /// the catalog handle becomes available), use
    /// [`Self::set_precision_resolver`].
    pub fn with_precision_resolver(
        self,
        resolver: Arc<proximadb_catalog::canonical_precision::CanonicalPrecisionResolver>,
    ) -> Self {
        // Best-effort init; the resolver is the first one set in this
        // construction path so a duplicate-set is a programming error.
        let _ = self.precision_resolver.set(resolver);
        self
    }

    /// Post-construction setter for the canonical-precision resolver.
    /// Works through `Arc<Compaction>`, so bootstrap code that builds
    /// the catalog handle after the storage engine has already wrapped
    /// Compaction in `Arc` can still wire the resolver.
    ///
    /// Returns `Err` if the resolver was already set (typically: a
    /// bootstrap-ordering bug — there should only be one
    /// `with_precision_resolver` or `set_precision_resolver` call per
    /// Compaction instance).
    pub fn set_precision_resolver(
        &self,
        resolver: Arc<proximadb_catalog::canonical_precision::CanonicalPrecisionResolver>,
    ) -> std::result::Result<
        (),
        tokio::sync::SetError<
            Arc<proximadb_catalog::canonical_precision::CanonicalPrecisionResolver>,
        >,
    > {
        self.precision_resolver.set(resolver)
    }

    /// Existing catalog-name adapter for callers that have not yet migrated
    /// canonical precision into the resolved collection scope. New compaction
    /// tasks receive the precision at flush admission and do not carry this
    /// string representation through the worker queue.
    fn collection_to_table_identifier(collection: &str) -> proximadb_catalog::TableIdentifier {
        let parsed = proximadb_catalog::TableIdentifier::parse(collection);
        if parsed.namespace.is_empty() {
            proximadb_catalog::TableIdentifier::new(vec!["default".to_string()], parsed.name)
        } else {
            parsed
        }
    }

    /// Resolve canonical precision through the pre-existing catalog-name
    /// adapter. This remains for non-flush callers while the v1 metadata DTO is
    /// retired; admitted flush/compaction work carries a native precision hint.
    pub async fn resolve_precision_hint(
        &self,
        collection: &str,
    ) -> Option<proximadb_records::EmbeddingScalarType> {
        let resolver: Arc<proximadb_catalog::canonical_precision::CanonicalPrecisionResolver> =
            if let Some(resolver) = self.precision_resolver.get() {
                Arc::clone(resolver)
            } else {
                Arc::clone(get_global_precision_resolver()?)
            };
        let table_id = Self::collection_to_table_identifier(collection);
        match resolver.resolve(&table_id).await {
            Ok(precision) => Some(precision),
            Err(error) => {
                warn!(
                    collection,
                    %error,
                    "compaction: precision resolver failed; falling back to fp32"
                );
                None
            }
        }
    }

    // Quantization sorting now handled by unified compute module
    pub async fn with_quantization_sorting(mut self) -> Result<Self> {
        // PQ-based sorting is now integrated directly into the compactor
        // using the unified quantization engine
        if let Some(ref mut _compactor) = self.sst_compactor {
            // Create unified quantization engine for PQ-based sorting
            let distance_compute =
                Arc::new(proximadb_distance_kernel::engine::UnifiedDistanceCompute::default());
            let codebook_store = Arc::new(
                crate::compute::quantization::quantization_engine::InMemoryCodebookStore::new(),
            );
            let unified_engine = Arc::new(
                crate::compute::quantization::quantization_engine::UnifiedQuantizationEngine::new(
                    distance_compute,
                    codebook_store,
                ),
            );
            // Create distance compute engine
            let distance_compute = Arc::new(
                proximadb_distance_kernel::engine::UnifiedDistanceCompute::new(
                    proximadb_distance_kernel::engine::DistanceMetric::Cosine,
                ),
            );
            // Create quantization config
            let quant_config =
                crate::compute::quantization::storage_engine::StorageQuantizationConfig::default();
            let quantization_engine = Arc::new(
                crate::compute::quantization::storage_engine::StorageQuantizationEngine::new(
                    unified_engine,
                    distance_compute,
                    quant_config,
                ),
            );

            let new_compactor = Arc::new(
                SstCompactor::with_block_size(
                    self.filesystem_factory.clone(),
                    self.atomic_coordinator
                        .as_ref()
                        .map(|_coord| Arc::new(MvccResolver::new())),
                    self.config.block_size_kb,
                )
                .with_pq_sorting(quantization_engine),
            );

            self.sst_compactor = Some(new_compactor);

            info!(
                "🎯 Compaction configured with PQ-based similarity sorting for better compression"
            );
        } else {
            warn!("⚠️ Cannot enable PQ sorting: no SST compactor available");
        }

        Ok(self)
    }

    /// Start background compaction workers. Takes `self: &Arc<Self>` so each
    /// worker can hold an `Arc<Compaction>` clone and call `perform_compaction`
    /// on THIS persistent instance — reusing its warmed reader/compactor/factory
    /// instead of constructing a cold `temp_manager` per task (which hung on the
    /// read path). Callers must wrap the `Compaction` in `Arc` before starting.
    pub async fn start_workers(self: &Arc<Self>, worker_count: usize) -> Result<()> {
        info!("Starting {} compaction workers", worker_count);

        for worker_id in 0..worker_count {
            let task_queue = Arc::clone(&self.task_queue);
            let task_notify = Arc::clone(&self.task_notify);
            let shutdown_signal = Arc::clone(&self.shutdown_signal);
            let stats = Arc::clone(&self.stats);
            let active_compactions = Arc::clone(&self.active_compactions);
            // TD-COMPACT-6 D1: each worker reuses this persistent Compaction
            // (shared Arc) — no per-task temp_manager construction.
            let compaction = Arc::clone(self);
            let training_in_flight = Arc::clone(&self.training_in_flight);

            let handle = tokio::spawn(async move {
                Self::worker_loop(
                    worker_id,
                    task_queue,
                    task_notify,
                    shutdown_signal,
                    stats,
                    active_compactions,
                    compaction,
                    training_in_flight,
                )
                .await;
            });

            if let Ok(mut handles) = self.worker_handles.lock() {
                handles.push(handle);
            }
        }

        Ok(())
    }

    /// Stop all compaction workers gracefully
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping compaction manager");

        self.shutdown_signal.store(true, Ordering::SeqCst);
        self.task_notify.notify_waiters();

        // Wait for all workers to finish (take the handles out before awaiting
        // so the lock is not held across the await).
        let handles: Vec<JoinHandle<()>> = if let Ok(mut h) = self.worker_handles.lock() {
            std::mem::take(&mut *h)
        } else {
            Vec::new()
        };
        for handle in handles {
            if let Err(e) = handle.await {
                warn!("Compaction worker failed to shutdown cleanly: {}", e);
            }
        }

        // Complete any remaining compactions
        let remaining_tasks = {
            let queue = self.task_queue.lock().await;
            queue.len()
        };

        if remaining_tasks > 0 {
            warn!(
                "Compaction manager stopped with {} pending tasks",
                remaining_tasks
            );
        }

        info!("Compaction manager stopped successfully");
        Ok(())
    }

    /// Schedule a compaction task
    pub async fn schedule_compaction(&self, task: SstCompactionTask) -> Result<bool> {
        debug!(
            "Scheduling compaction for level {} with {} input files",
            task.level,
            task.input_files.len()
        );

        // The output file path is the unique key for active compactions —
        // prevents two compactions writing to the same output file.
        let compaction_key = task.output_file.to_string_lossy().to_string();

        // TD-COMPACT-6 (ADR-076 D1) RACE FIX: do the dedup check AND the
        // active-compaction insert under ONE write lock, at enqueue time
        // (not worker time). The previous code took a read lock for the
        // check and deferred the insert to the worker (~:830) in a separate
        // critical section, so two rapid `schedule_compaction` calls for the
        // same output could both pass the read check before either worker
        // inserted — double-enqueue to the same output. Marking active here,
        // atomically with the dedup, closes that TOCTOU. The worker only
        // removes the entry after completion.
        {
            let mut active = self.active_compactions.write().await;
            if active.contains_key(&compaction_key) {
                debug!(
                    "Skipping compaction - already active for output file {}",
                    compaction_key
                );
                return Ok(false);
            }
            if active.values().any(|active_task| {
                active_task
                    .input_files
                    .iter()
                    .any(|input| task.input_files.contains(input))
            }) {
                debug!(
                    collection_object_id = task.collection_object_id,
                    "Skipping compaction - an input file is already owned by another morsel"
                );
                return Ok(false);
            }
            active.insert(compaction_key, task.clone());
        }

        let mut queue = self.task_queue.lock().await;

        // Insert task in priority order
        let insert_pos = queue
            .iter()
            .position(|existing_task| existing_task.priority < task.priority)
            .unwrap_or(queue.len()); // Insert at end if no lower priority task found

        queue.insert(insert_pos, task);
        crate::metrics::operational_metrics::COMPACTION_QUEUE_DEPTH.inc();
        drop(queue);
        self.task_notify.notify_one();

        debug!(
            "Compaction task queued (position: {}, queue size: {})",
            insert_pos,
            self.task_queue.lock().await.len()
        );

        Ok(true)
    }

    /// TD-COMPACT-6 (ADR-076 D1): the production flush-path compaction trigger.
    /// Builds the task if a collection has due compaction (L0 ≥ threshold or an
    /// untrained large L0), then **enqueues** it to the background worker pool —
    /// flush returns immediately (~1s after the L0 write, not ~35s blocked on
    /// the re-cluster). This is the producer half of the producer/consumer
    /// rate-control loop; the consumer half is the L0 admission watermarks
    /// (TD-COMPACT-7) at the flush boundary.
    ///
    /// Returns whether a task was enqueued (`false` = nothing was due).
    /// Dedup is `schedule_compaction`'s active-compaction guard (output-file
    /// keyed); the per-collection `training_in_flight` guard (set by the flush
    /// caller) prevents `should_trigger_compaction` from re-arming the training
    /// arm while a pass is queued/running. The worker clears
    /// `training_in_flight` after completion (collection dir derived from
    /// `task.output_file.parent()`).
    ///
    /// The caller treats errors as best-effort — an enqueue failure must never
    /// fail the flush that armed it. Tests that need the merge complete before
    /// asserting use `await_compaction_quiescence`.
    pub async fn enqueue_due_compaction(
        &self,
        collection_object_id: crate::core::stable_id::CollectionObjectId,
        collection_identity: crate::core::stable_id::CollectionIdentity,
        collection_dir: &Path,
        l0_threshold: usize,
        precision_hint: Option<proximadb_records::EmbeddingScalarType>,
        rg_layout: Option<bool>,
    ) -> Result<bool> {
        let Some(mut task) = self
            .check_compaction_needed(
                collection_object_id,
                collection_dir,
                Some(l0_threshold),
                precision_hint,
            )
            .await?
        else {
            return Ok(false);
        };
        task.rg_layout = rg_layout;
        task.collection_identity = collection_identity;
        // TD-COMPACT-6 D1: the worker clears the shared training_in_flight guard
        // for this collection on completion. It derives the collection dir from
        // `task.output_file.parent()` (the output is always generated under the
        // collection dir — see `generate_output_file_path`), so no separate
        // field is needed on the task.
        self.schedule_compaction(task).await
    }

    /// Check if compaction is needed for the given collection and level.
    ///
    /// `l0_threshold_override` (TD-WLP-7 / TD-WLP-2): when `Some`, the L0
    /// file-count threshold the unified builder gates on is replaced with the
    /// per-collection value resolved from the `l0_threshold:` tag, so this stays
    /// consistent with the flush-path `should_trigger_compaction` decision.
    /// `None` uses the deployment `CompactionConfig` default.
    pub async fn check_compaction_needed(
        &self,
        collection_object_id: crate::core::stable_id::CollectionObjectId,
        collection_dir: &Path,
        l0_threshold_override: Option<usize>,
        precision_hint: Option<proximadb_records::EmbeddingScalarType>,
    ) -> Result<Option<SstCompactionTask>> {
        // TD-PAXRG-1: rg resolution happens at the task PRODUCERS (they hold
        // the collection tags); check_compaction_needed stays config-free and
        // defaults the field to env-follow at the write sites.

        debug!(
            "🔍 SST COMPACTION: Delegating to unified framework for collection: {}",
            collection_object_id
        );

        let collection_path = collection_dir.to_string_lossy();
        // The common cross-engine discovery adapter is still string-based.
        // Keep that representation local to the adapter call; the scheduler,
        // task queue, and catalog resolver retain the authoritative u64.
        let collection_id_text = collection_object_id.to_string();

        // Use SST-specific config if available, otherwise use defaults
        let mut compaction_config = self
            .config
            .compaction_config
            .as_ref()
            .cloned()
            .unwrap_or_else(crate::core::config::CompactionConfig::default);

        // TD-WLP-7: honor the per-collection L0 threshold (from the
        // `l0_threshold:` tag) so this agrees with the flush-path
        // `should_trigger_compaction` gate that armed the run.
        if let Some(threshold) = l0_threshold_override {
            compaction_config.l0_file_threshold = threshold;
        }

        // Use unified compaction task builder with configuration.
        // (Was `SstCompactionTaskBuilder` before the
        // disambiguate-rename pass; companion call-site fix-up.)
        // TD-WLP-7: discover `.pax` segments — PAX is the sole vector write
        // format since ADR-049 M1-3 (flush + compaction both emit `.pax`), so
        // the prior `"sst"` extension found nothing and compaction never ran.
        // This matches the `.pax`-aware `discover_sstable_files` gate that arms
        // the run.
        let task_info = CompactionTaskBuilder::check_and_build_compaction_task(
            &collection_id_text,
            &collection_path,
            "pax",
            CompactionEngineType::SST,
            &compaction_config,
            self.filesystem_factory.clone(),
        )
        .await
        .map_err(|e| crate::core::StorageError::SstEngine(e.to_string()))?;

        let unified_task = task_info.map(|info| LevelBasedSstCompactionTask {
            level: info.source_level,
            input_files: info.input_files,
            input_bytes: info.input_bytes,
            target_level: info.target_level,
            extension: info.extension,
        });

        if let Some(task) = unified_task {
            debug!(
                "🔄 SST COMPACTION: Converting unified task to SST-specific format for collection {} level {}",
                collection_object_id, task.level
            );

            // Convert unified task to SST SstCompactionTask
            let input_files: Vec<PathBuf> =
                task.input_files.into_iter().map(PathBuf::from).collect();
            // M1-3 (ADR-049): compaction output format follows the INPUT segments'
            // format — `.pax`/`.sst` → `.pax` (PAX is the only vector write format;
            // legacy inputs compact forward to PAX), `.arrow` → `.arrow` — so the
            // output filename extension matches the writer chosen at the write site.
            let output_block_format = compaction_output_format(&input_files);

            let output_file = self.generate_output_file_path(
                collection_object_id,
                collection_dir,
                task.target_level as u8,
                output_block_format,
            );

            let priority = if input_files.len() >= (self.config.compaction_threshold * 2) as usize {
                CompactionPriority::High
            } else {
                CompactionPriority::Medium
            };

            return Ok(Some(SstCompactionTask {
                collection_object_id,
                collection_identity: crate::core::stable_id::CollectionIdentity::default(),
                level: task.level as u8,
                input_files,
                input_bytes: task.input_bytes,
                output_file,
                priority,
                block_size_kb: None,      // Use server default
                compression_config: None, // Use server default
                precision_hint,
                rg_layout: None,
            }));
        }

        debug!(
            "📋 COMPACTION: Unified framework reports no compaction needed for collection: {}",
            collection_object_id
        );
        Ok(None)
    }

    /// Get compaction statistics
    pub async fn stats(&self) -> SstCompactionStats {
        self.stats.read().await.clone()
    }

    /// Number of compaction tasks waiting in the queue (not yet picked up by a
    /// worker). Observability + test seam for the TD-COMPACT-6 admission checks.
    pub async fn pending_task_count(&self) -> usize {
        self.task_queue.lock().await.len()
    }

    /// Number of compactions currently in-flight (a worker is processing them).
    /// Populated at enqueue time (the race-fix insertion site).
    pub async fn active_compaction_count(&self) -> usize {
        self.active_compactions.read().await.len()
    }

    /// TD-COMPACT-6 (ADR-076 D1): quiescence barrier for tests + the async
    /// flush path. Returns once the task queue is empty AND no compaction is
    /// active — i.e. every enqueued compaction has been processed by a worker.
    /// The flush path itself never blocks on the worker (it returns ~immediately
    /// after `enqueue_due_compaction`); callers that need the merge complete
    /// before asserting on the post-compaction layout call this. Polls with a
    /// short backoff rather than spinning; `timeout` bounds the wait (returns
    /// `false` on timeout so the caller can fail loudly rather than hang).
    pub async fn await_compaction_quiescence(&self, timeout: std::time::Duration) -> bool {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let queue_empty = self.task_queue.lock().await.is_empty();
            let active_empty = self.active_compactions.read().await.is_empty();
            if queue_empty && active_empty {
                return true;
            }
            if tokio::time::Instant::now() >= deadline {
                return false;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    }

    /// Worker loop for processing compaction tasks. Each worker holds an
    /// `Arc<Compaction>` clone of the persistent instance and calls
    /// `perform_compaction` on it — reusing the warmed reader/compactor/factory
    /// rather than constructing a cold `temp_manager` per task.
    async fn worker_loop(
        worker_id: usize,
        task_queue: Arc<Mutex<VecDeque<SstCompactionTask>>>,
        task_notify: Arc<Notify>,
        shutdown_signal: Arc<AtomicBool>,
        stats: Arc<RwLock<SstCompactionStats>>,
        active_compactions: Arc<RwLock<HashMap<String, SstCompactionTask>>>,
        compaction: Arc<Compaction>,
        // TD-COMPACT-6 D1: shared training_in_flight guard. The worker clears
        // the originating collection's entry once the task is processed (Ok or
        // Err), re-arming `should_trigger_compaction` for that collection.
        training_in_flight: Arc<std::sync::Mutex<std::collections::HashSet<String>>>,
    ) {
        debug!("Compaction worker {} started", worker_id);

        loop {
            if shutdown_signal.load(Ordering::SeqCst) {
                break;
            }

            // Register for notification before checking the queue to avoid lost wakeups.
            let notified = task_notify.notified();

            // Get next task from queue
            let task = {
                let mut queue = task_queue.lock().await;
                queue.pop_front()
            };

            if let Some(task) = task {
                crate::metrics::operational_metrics::COMPACTION_QUEUE_DEPTH.dec();
                debug!(
                    "Worker {} processing compaction for level {} with {} files -> {}",
                    worker_id,
                    task.level,
                    task.input_files.len(),
                    task.output_file.display()
                );

                // TD-COMPACT-6 D1: the task was marked active at enqueue time
                // (schedule_compaction) — no insert here. We only remove below
                // once the task is processed.
                let compaction_key = task.output_file.to_string_lossy().to_string();
                let training_guard_active = task.output_file.parent().is_some_and(|dir| {
                    let key = dir.to_string_lossy();
                    training_in_flight
                        .lock()
                        .is_ok_and(|guard| guard.contains(key.as_ref()))
                });

                let start_time = std::time::Instant::now();

                let memory_config = compaction
                    .config
                    .compaction_config
                    .as_ref()
                    .cloned()
                    .unwrap_or_default();
                let admission_started = std::time::Instant::now();
                let Some(resource_reservation) = acquire_compaction_resources(
                    task.input_bytes,
                    &memory_config,
                    shutdown_signal.as_ref(),
                )
                .await
                else {
                    Self::release_task_state(
                        &active_compactions,
                        &training_in_flight,
                        &compaction_key,
                        task.output_file.parent(),
                        true,
                    )
                    .await;
                    break;
                };
                let admission_wait = admission_started.elapsed();
                if admission_wait >= std::time::Duration::from_millis(1) {
                    crate::metrics::operational_metrics::COMPACTION_MEMORY_ADMISSION_WAITS_TOTAL
                        .inc();
                }
                crate::metrics::operational_metrics::COMPACTION_MEMORY_RESERVED_BYTES
                    .set(i64::try_from(reserved_compaction_memory_bytes()).unwrap_or(i64::MAX));
                crate::metrics::operational_metrics::COMPACTION_SCRATCH_RESERVED_BYTES
                    .set(i64::try_from(reserved_compaction_scratch_bytes()).unwrap_or(i64::MAX));
                let resource_plan = resource_reservation.plan();
                if resource_plan.mode == CompactionExecutionMode::LocalSpill {
                    crate::metrics::operational_metrics::COMPACTION_SPILL_TOTAL.inc();
                }
                info!(
                    input_bytes = task.input_bytes,
                    mode = ?resource_plan.mode,
                    projected_memory_bytes = resource_plan.memory_bytes,
                    projected_scratch_bytes = resource_plan.scratch_bytes,
                    admission_wait_ms = admission_wait.as_millis(),
                    "Compaction resource reservation admitted"
                );
                let _running_metric = CompactionRunningMetricGuard::begin();

                // Reuse the persistent Compaction instance (shared Arc). The
                // former code built a fresh `Compaction` per task via
                // `with_atomic_coordinator`; that cold instance's read path
                // hung on the worker. `perform_compaction` only reads through
                // the Arc-shared reader/compactor/factory, so concurrent
                // invocation across workers + flush is safe.
                let succeeded = match compaction
                    .perform_compaction_with_plan(&task, resource_plan)
                    .await
                {
                    Ok(compaction_stats) => {
                        info!(
                            "Compaction completed for level {} in {}ms: {} files merged -> {}",
                            task.level,
                            start_time.elapsed().as_millis(),
                            compaction_stats.files_merged,
                            task.output_file.display()
                        );

                        // Update statistics
                        {
                            let mut stats_guard = stats.write().await;
                            stats_guard.total_compactions += 1;
                            stats_guard.bytes_written += compaction_stats.bytes_written;
                            stats_guard.bytes_read += compaction_stats.bytes_read;
                            stats_guard.files_merged += compaction_stats.files_merged;
                            stats_guard.last_compaction_time = Some(Utc::now());

                            // Update average compaction time
                            let elapsed_ms = start_time.elapsed().as_millis() as u64;
                            if stats_guard.total_compactions == 1 {
                                stats_guard.avg_compaction_time_ms = elapsed_ms;
                            } else {
                                stats_guard.avg_compaction_time_ms =
                                    (stats_guard.avg_compaction_time_ms + elapsed_ms) / 2;
                            }
                        }
                        true
                    }
                    Err(e) => {
                        error!(
                            "Compaction failed for level {} -> {}: {}",
                            task.level,
                            task.output_file.display(),
                            e
                        );
                        false
                    }
                };
                drop(resource_reservation);
                // TD-COMPACT-13: debounced reconciliation sweep (runs on both
                // success and failure paths so a failed retirement is re-driven
                // instead of rotting).
                compaction.sweep_pending_retirements_if_due().await;
                crate::metrics::operational_metrics::COMPACTION_MEMORY_RESERVED_BYTES
                    .set(i64::try_from(reserved_compaction_memory_bytes()).unwrap_or(i64::MAX));
                crate::metrics::operational_metrics::COMPACTION_SCRATCH_RESERVED_BYTES
                    .set(i64::try_from(reserved_compaction_scratch_bytes()).unwrap_or(i64::MAX));

                // A completed morsel changes the level geometry. Re-evaluate
                // it immediately so higher-level compaction does not depend on
                // a future flush that may never arrive after ingest settles.
                // Schedule before releasing the current active marker, keeping
                // the quiescence barrier closed across the hand-off.
                let mut scheduled_follow_up_level = None;
                let follow_up_threshold = training_follow_up_threshold(
                    training_guard_active,
                    task.level,
                    &task.output_file,
                );
                if succeeded
                    && !shutdown_signal.load(Ordering::SeqCst)
                    && let Some(collection_dir) = task.output_file.parent()
                {
                    match compaction
                        .check_compaction_needed(
                            task.collection_object_id,
                            collection_dir,
                            follow_up_threshold,
                            task.precision_hint,
                        )
                        .await
                    {
                        Ok(Some(mut follow_up)) => {
                            // TD-PAXRG-1: the follow-up carries the ORIGINAL
                            // task's rg resolution — the collection's intent
                            // must not drift between compaction rounds.
                            follow_up.rg_layout = task.rg_layout;
                            let follow_up_level = follow_up.level;
                            match compaction.schedule_compaction(follow_up).await {
                                Ok(true) => {
                                    scheduled_follow_up_level = Some(follow_up_level);
                                    info!(
                                        collection_object_id = task.collection_object_id,
                                        layout_promotion_chain = follow_up_threshold.is_some(),
                                        "Compaction follow-up morsel enqueued after level {}",
                                        task.level
                                    );
                                }
                                Ok(false) => {
                                    // Another active task owns this work. Keep
                                    // the training guard when that owner is an
                                    // L0 morsel so it inherits threshold one.
                                    scheduled_follow_up_level = Some(follow_up_level);
                                    debug!(
                                        collection_object_id = task.collection_object_id,
                                        "Compaction follow-up already owned by another morsel"
                                    );
                                }
                                Err(error) => warn!(
                                    collection_object_id = task.collection_object_id,
                                    "Compaction follow-up enqueue failed: {error}"
                                ),
                            }
                        }
                        Ok(None) => {}
                        Err(error) => warn!(
                            collection_object_id = task.collection_object_id,
                            "Compaction follow-up discovery failed: {error}"
                        ),
                    }
                }

                // TD-COMPACT-6 D1: release the enqueue-time active marker and the
                // per-collection training guard so the flush path can re-arm.
                // Retain the training guard across the complete follow-up
                // chain, including higher-level spill. A flush can publish a
                // new L0 while that longer task is running; the terminal
                // threshold-one rescan must observe and train that late tail.
                let retain_training_guard = retain_training_guard_for_follow_up(
                    training_guard_active,
                    scheduled_follow_up_level,
                );
                Self::release_task_state(
                    &active_compactions,
                    &training_in_flight,
                    &compaction_key,
                    task.output_file.parent(),
                    !retain_training_guard,
                )
                .await;
            } else {
                // TD-COMPACT-13: while idle, keep the reconciler draining —
                // obligations recorded by earlier tasks (or a previous boot's
                // sidecar) must converge even when no new compaction ever
                // arrives. The sweep is debounced (30s) and bounded; the
                // notified arm wakes immediately on a real task.
                let sweep_tick = tokio::time::sleep(std::time::Duration::from_secs(30));
                tokio::select! {
                    _ = notified => {}
                    _ = sweep_tick => {
                        compaction.sweep_pending_retirements_if_due().await;
                    }
                }
            }
        }

        debug!("Compaction worker {} stopped", worker_id);
    }

    /// TD-COMPACT-6 (ADR-076 D1): drop the per-task bookkeeping once a worker
    /// finishes a task (Ok, Err, or manager-construction failure). Removes the
    /// output-file key from `active_compactions` (re-allowing compaction to the
    /// same output). It normally clears the shared `training_in_flight` guard;
    /// a successfully handed-off L0 training morsel retains it so the bounded
    /// chain cannot lose its threshold-one reason. Shared by every worker exit
    /// path so the guard can never leak.
    async fn release_task_state(
        active_compactions: &Arc<RwLock<HashMap<String, SstCompactionTask>>>,
        training_in_flight: &Arc<std::sync::Mutex<std::collections::HashSet<String>>>,
        compaction_key: &str,
        collection_dir: Option<&Path>,
        clear_training_guard: bool,
    ) {
        {
            let mut active = active_compactions.write().await;
            active.remove(compaction_key);
        }
        if clear_training_guard && let Some(dir) = collection_dir {
            let key = dir.to_string_lossy();
            if let Ok(mut guard) = training_in_flight.lock() {
                guard.remove(key.as_ref());
            }
        }
    }

    /// Perform the actual compaction operation on a task, using this
    /// persistent instance's config + atomic coordinator. Called by the
    /// background workers (which hold an `Arc<Compaction>` clone).
    async fn perform_compaction(&self, task: &SstCompactionTask) -> Result<SstCompactionStats> {
        let config = self
            .config
            .compaction_config
            .as_ref()
            .cloned()
            .unwrap_or_default();
        let plan = CompactionResourcePlan {
            mode: CompactionExecutionMode::InMemory,
            memory_bytes: projected_compaction_memory_bytes(task.input_bytes, &config),
            scratch_bytes: 0,
        };
        self.perform_compaction_with_plan(task, plan).await
    }

    async fn perform_compaction_with_plan(
        &self,
        task: &SstCompactionTask,
        plan: CompactionResourcePlan,
    ) -> Result<SstCompactionStats> {
        if plan.mode == CompactionExecutionMode::LocalSpill {
            return Ok(self
                .perform_local_spill_compaction(task, plan)
                .await?
                .base_stats);
        }
        let enhanced_stats = self
            .perform_unified_canonical_compaction(
                task,
                &self.config,
                self.atomic_coordinator.clone(),
                None,
                MergedVectorTracking::Disabled,
            )
            .await?;
        Ok(enhanced_stats.base_stats)
    }

    /// Retire immutable inputs only after their replacement is published (or
    /// after MVCC proves the replacement is empty). Cache and deletion-vector
    /// invalidation share the same boundary so no path can leave dead segment
    /// state resident merely because it used the spill executor.
    /// TD-COMPACT-13: ONE delete attempt per input (warm-cache purge + DV
    /// removal on success). Returns the inputs that failed this attempt.
    async fn delete_inputs_attempt(&self, inputs: &[PathBuf]) -> Vec<PathBuf> {
        let warm_caches = crate::storage::engines::sst::core::get_warm_tier_caches();
        #[cfg(feature = "cold-deletion-vectors")]
        let dv_store =
            crate::storage::engines::sst::deletion_vector_store::DeletionVectorStore::new(
                self.filesystem_factory.clone(),
            );
        let mut failed = Vec::new();
        for input_file in inputs {
            let input_path = input_file.to_string_lossy();
            match retire_compaction_input(&self.filesystem_factory, &input_path).await {
                Err(error) => {
                    warn!(file = %input_path, %error, "input retirement attempt failed");
                    failed.push(input_file.clone());
                }
                Ok(()) => {
                    purge_retired_segment_cache_entries(&input_path, warm_caches.as_ref()).await;
                    #[cfg(feature = "cold-deletion-vectors")]
                    if let Err(error) = dv_store.remove(&input_path).await {
                        warn!("compaction DV retire: failed to remove {input_path}.dv: {error:?}");
                    }
                }
            }
        }
        failed
    }

    /// TD-COMPACT-13: bounded-retry retirement. Residual failures are recorded
    /// as a durable obligation (sidecar + pending set + gauge) so the
    /// reconciler can drain them across restarts; the superseded paths are
    /// filtered from query discovery immediately. Returns the still-failed
    /// inputs (empty on full success); `Err` only when the obligation itself
    /// cannot be recorded.
    async fn retire_inputs_with_retry(
        &self,
        output: &Path,
        inputs: &[PathBuf],
    ) -> Result<Vec<PathBuf>> {
        let mut failed = self.delete_inputs_attempt(inputs).await;
        for attempt in 1..3u32 {
            if failed.is_empty() {
                break;
            }
            crate::metrics::operational_metrics::PAX_RETIREMENT_RETRIES_TOTAL
                .inc_by(failed.len() as u64);
            tokio::time::sleep(std::time::Duration::from_millis(if attempt == 1 {
                100
            } else {
                300
            }))
            .await;
            failed = self.delete_inputs_attempt(&failed).await;
        }
        if failed.is_empty() {
            return Ok(Vec::new());
        }
        let collection_dir = output.parent().unwrap_or(Path::new(""));
        retirement_ledger::register_collection_dir(collection_dir);
        let failed_urls: Vec<String> = failed
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        if let Err(error) =
            retirement_ledger::record_pending(collection_dir, output, &failed_urls).await
        {
            error!(
                output = %output.display(),
                %error,
                "retirement obligation UNRECORDABLE — failing the compaction loudly"
            );
            return Err(error.into());
        }
        error!(
            output = %output.display(),
            pending = failed.len(),
            "compaction inputs pending retirement — obligation recorded; superseded \
             files are filtered from query discovery until the reconciler drains them"
        );
        Ok(failed)
    }

    /// TD-COMPACT-13 reconciler sweep (debounced 30s): drain recorded
    /// obligations for every known collection dir.
    async fn sweep_pending_retirements_if_due(&self) {
        if !retirement_ledger::sweep_due() {
            return;
        }
        for dir in retirement_ledger::known_collection_dirs() {
            let _ = self.drain_pending_for_dir(Path::new(&dir)).await;
        }
    }

    /// Drain one collection dir's recorded obligations (reconciler pass).
    async fn drain_pending_for_dir(&self, collection_data_dir: &Path) -> Result<()> {
        // Hold the ledger lock across this whole load-modify-save sequence.
        let _ledger_guard = retirement_ledger::ledger_lock().await;
        let mut ledger = retirement_ledger::load_ledger(collection_data_dir);
        if ledger.pending.is_empty() {
            return Ok(());
        }
        for entry in &mut ledger.pending {
            let inputs: Vec<PathBuf> = entry.inputs.iter().map(PathBuf::from).collect();
            let failed = self.delete_inputs_attempt(&inputs).await;
            entry.attempts += 1;
            if failed.is_empty() {
                retirement_ledger::mark_drained(&entry.inputs);
            }
            entry.inputs = failed
                .into_iter()
                .map(|p| p.to_string_lossy().into_owned())
                .collect();
        }
        ledger.pending.retain(|entry| !entry.inputs.is_empty());
        retirement_ledger::save_after_drain(collection_data_dir, &ledger).map_err(|error| {
            crate::core::StorageError::SstEngine(format!(
                "save retire-pending ledger for {}: {error:#}",
                collection_data_dir.display()
            ))
        })
    }

    async fn retire_task_inputs(&self, task: &SstCompactionTask) -> Result<()> {
        // TD-COMPACT-13: shared bounded-retry path; residual failures are
        // recorded as durable obligations and the task succeeds (the output is
        // published and correct — failing here only killed the follow-up chain
        // that re-drives retirement, the exact unrecoverable-defect mechanism).
        let failed = self
            .retire_inputs_with_retry(&task.output_file, &task.input_files)
            .await?;
        if !failed.is_empty() {
            warn!(
                output = %task.output_file.display(),
                pending = failed.len(),
                "local-spill inputs pending retirement (obligation recorded)"
            );
        }
        Ok(())
    }

    /// Execute the bounded construction selected by dual-resource admission.
    /// Inputs remain authoritative until the final PAX has been uploaded and
    /// atomically published; every scratch owner is RAII-cleaned on error.
    async fn perform_local_spill_compaction(
        &self,
        task: &SstCompactionTask,
        plan: CompactionResourcePlan,
    ) -> Result<EnhancedSstCompactionStats> {
        use crate::storage::engines::sst::compaction_spill::{
            ExternalClusterOutput, ExternalMvccOutput, SpillTaskScratch, cluster_external_mvcc,
            resolve_external_mvcc_from_segments,
        };

        enum SpillOrdering {
            Ivf(ExternalClusterOutput),
            Mvcc(ExternalMvccOutput),
        }

        if compaction_output_format(&task.input_files) != BlockFormat::PaxBlock
            || task
                .input_files
                .iter()
                .any(|path| path.extension().and_then(|value| value.to_str()) != Some("pax"))
        {
            return Err(crate::core::StorageError::SstEngine(
                "local-spill compaction currently requires PAX-only inputs".to_string(),
            ));
        }
        let scratch_owner = self.spill_scratch_owner.as_ref().ok_or_else(|| {
            crate::core::StorageError::SstEngine(
                "local-spill execution has no process-owned scratch namespace".to_string(),
            )
        })?;
        let start_time = std::time::Instant::now();
        let mut memtrace = CompactionMemTrace::new();
        let cache_on_write = std::env::var("PROXIMADB_CACHE_ON_WRITE")
            .ok()
            .and_then(|value| crate::core::config::CacheOnWritePolicy::parse(&value))
            .unwrap_or(self.config.cache_on_write);
        let run_buffer_bytes = (plan.memory_bytes / 4).clamp(1024 * 1024, 64 * 1024 * 1024);
        let merge_fan_in = 32usize;
        let input_paths = task
            .input_files
            .iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        let output_path = task.output_file.to_string_lossy().into_owned();
        let direct_remote_publication =
            crate::storage::engines::sst::staged_write::StagedSegmentWrite::is_remote_target(
                &output_path,
            );
        if direct_remote_publication {
            let output_filesystem = self
                .filesystem_factory
                .get_filesystem(&output_path)
                .map_err(|error| {
                    crate::core::StorageError::SstEngine(format!(
                        "resolve local-spill publication backend: {error}"
                    ))
                })?;
            if !output_filesystem.supports_bounded_local_file_write() {
                return Err(crate::core::StorageError::SstEngine(format!(
                    "{} backend does not guarantee bounded local-file publication for {output_path}",
                    output_filesystem.filesystem_type()
                )));
            }
        }
        let mut task_scratch = SpillTaskScratch::begin(
            scratch_owner.path(),
            task.collection_object_id,
            task.level,
            &input_paths,
            task.input_bytes,
            &output_path,
            plan.memory_bytes,
            plan.scratch_bytes,
        )
        .map_err(|error| {
            crate::core::StorageError::SstEngine(format!(
                "initialize local-spill task scratch: {error}"
            ))
        })?;
        let scratch_root = task_scratch.path().to_path_buf();

        #[cfg(feature = "cold-deletion-vectors")]
        let dv_store =
            crate::storage::engines::sst::deletion_vector_store::DeletionVectorStore::new(
                self.filesystem_factory.clone(),
            );
        #[cfg(feature = "cold-deletion-vectors")]
        let deleted_positions = {
            let mut all = Vec::with_capacity(input_paths.len());
            for path in &input_paths {
                let _ = dv_store.load(path).await;
                let positions = if let Some(vector) = dv_store.get(path).await {
                    vector.deleted_positions().into_iter().collect()
                } else {
                    std::collections::HashSet::new()
                };
                all.push(positions);
            }
            all
        };
        #[cfg(not(feature = "cold-deletion-vectors"))]
        let deleted_positions: Vec<std::collections::HashSet<u32>> = Vec::new();

        let (winners, input_stats) = resolve_external_mvcc_from_segments(
            self.filesystem_factory.as_ref(),
            &input_paths,
            &scratch_root,
            run_buffer_bytes,
            merge_fan_in,
            Self::current_time_ns(),
            &[],
            &[],
            None,
            &deleted_positions,
        )
        .await
        .map_err(|error| {
            crate::core::StorageError::SstEngine(format!(
                "local-spill ranged MVCC pass failed: {error}"
            ))
        })?;
        let mvcc_stats = *winners.stats();
        memtrace.stage(
            "spill_mvcc_resolved",
            usize::try_from(mvcc_stats.output_records).unwrap_or(usize::MAX),
        );
        task_scratch
            .update_phase("mvcc-resolved", Some(&input_stats.source_sizes))
            .map_err(|error| {
                crate::core::StorageError::SstEngine(format!(
                    "update local-spill MVCC manifest: {error}"
                ))
            })?;
        if mvcc_stats.output_records == 0 {
            task_scratch
                .update_phase("empty-retirement", None)
                .map_err(|error| {
                    crate::core::StorageError::SstEngine(format!(
                        "update empty local-spill manifest: {error}"
                    ))
                })?;
            self.retire_task_inputs(task).await?;
            {
                use crate::metrics::operational_metrics as metrics;
                metrics::COMPACTIONS_TOTAL.inc();
                metrics::COMPACTION_BYTES_READ_TOTAL.inc_by(task.input_bytes);
                metrics::COMPACTION_SECONDS.observe(start_time.elapsed().as_secs_f64());
            }
            return Ok(EnhancedSstCompactionStats {
                base_stats: SstCompactionStats {
                    total_compactions: 1,
                    bytes_written: 0,
                    bytes_read: task.input_bytes,
                    files_merged: task.input_files.len() as u64,
                    avg_compaction_time_ms: start_time.elapsed().as_millis() as u64,
                    last_compaction_time: Some(Utc::now()),
                    expired_records_deleted: 0,
                    tombstones_removed: 0,
                },
                deleted_vector_ids: Vec::new(),
                merged_vectors: Vec::new(),
                recommend_full_rebuild: false,
            });
        }

        let use_ivf = crate::storage::engines::sst::block_cluster::block_cluster_enabled()
            && crate::storage::engines::sst::block_cluster::ivf_probe_enabled()
            && mvcc_stats.output_records >= 64;
        let ordering = if use_ivf {
            task_scratch
                .update_phase("ivf-ordering", None)
                .map_err(|error| {
                    crate::core::StorageError::SstEngine(format!(
                        "update local-spill IVF manifest: {error}"
                    ))
                })?;
            SpillOrdering::Ivf(
                cluster_external_mvcc(
                    winners,
                    &scratch_root,
                    run_buffer_bytes,
                    merge_fan_in,
                    plan.memory_bytes,
                    task.output_file.to_str(),
                )
                .map_err(|error| {
                    crate::core::StorageError::SstEngine(format!(
                        "local-spill IVF ordering failed: {error}"
                    ))
                })?,
            )
        } else {
            SpillOrdering::Mvcc(winners)
        };
        memtrace.stage(
            "spill_ordering_ready",
            usize::try_from(mvcc_stats.output_records).unwrap_or(usize::MAX),
        );
        task_scratch
            .update_phase("pax-construction", None)
            .map_err(|error| {
                crate::core::StorageError::SstEngine(format!(
                    "update local-spill PAX manifest: {error}"
                ))
            })?;
        // The spill writer already owns a complete local PAX. Object-store
        // multipart uploads keep parts invisible until their final block-list
        // commit, so uploading that file once to its unique final key is the
        // atomic publication boundary. A remote `__compact` upload followed by
        // COPY adds another full-object operation; the Azure fallback also
        // materializes the entire multi-GiB object in process. Local files keep
        // the coordinator's staging + atomic rename path.
        let staging_config = atomic_coordinator_staging(task)?;
        let atomic_operation = if let Some(coordinator) = &self.atomic_coordinator {
            Some(
                if direct_remote_publication {
                    coordinator
                        .begin_zero_copy_managed_operation(&staging_config)
                        .await
                } else {
                    coordinator.begin_atomic_operation(&staging_config).await
                }
                .map_err(|error| {
                    crate::core::StorageError::SstEngine(format!(
                        "begin local-spill atomic publication: {error}"
                    ))
                })?,
            )
        } else {
            None
        };
        let publication_target = if direct_remote_publication {
            output_path.clone()
        } else if let Some(operation) = &atomic_operation {
            let filename = task
                .output_file
                .file_name()
                .and_then(|value| value.to_str())
                .ok_or_else(|| {
                    crate::core::StorageError::SstEngine(
                        "local-spill output has no UTF-8 filename".to_string(),
                    )
                })?;
            format!(
                "{}/{}",
                operation.staging_url.trim_end_matches('/'),
                filename
            )
        } else {
            output_path.clone()
        };

        let write_result: Result<(
            u64,
            crate::storage::engines::sst::staged_write::StagedSegmentWrite,
        )> = async {
            let staged = crate::storage::engines::sst::staged_write::StagedSegmentWrite::begin_in(
                &publication_target,
                &scratch_root,
            )
            .await
            .map_err(|error| {
                crate::core::StorageError::SstEngine(format!(
                    "prepare local-spill publication: {error}"
                ))
            })?;
            let target_block = task
                .block_size_kb
                .and_then(|kilobytes| usize::try_from(kilobytes).ok())
                .and_then(|kilobytes| kilobytes.checked_mul(1024));
            let collection_id = task.collection_identity.collection_id.to_string();
            let model = match &ordering {
                SpillOrdering::Ivf(output) => Some(output.model().clone()),
                SpillOrdering::Mvcc(_) => None,
            };
            // TD-PAXRG-1: the exact tier follows the INTERSECTION rule — the
            // spill executor must not silently drop it when every input is
            // exact (same resolution the non-spill arms perform).
            let f32_tier = crate::storage::engines::sst::segment_format::pax_inputs_have_f32_tier(
                &task.input_files,
            );
            let mut writer =
                crate::storage::engines::sst::segment_format::pax_spill_compaction_writer(
                    Path::new(staged.local_path()),
                    &scratch_root,
                    &collection_id,
                    1,
                    proximadb_block_format::VectorQuant::Sq8,
                    f32_tier,
                    crate::storage::engines::sst::segment_format::rg_layout_enabled(),
                    target_block,
                    usize::try_from(mvcc_stats.output_records).map_err(|_| {
                        crate::core::StorageError::SstEngine(
                            "local-spill output row count exceeds usize".to_string(),
                        )
                    })?,
                    model,
                )
                .map_err(|error| {
                    crate::core::StorageError::SstEngine(format!(
                        "create disk-backed PAX writer: {error}"
                    ))
                })?;
            let mut append = |mut record: ProximaRecord| {
                if let Some(target) = task.precision_hint {
                    for embedding in &mut record.embeddings {
                        embedding.coerce_to_precision(target);
                    }
                }
                writer.add_record(&record).map_err(|error| {
                    crate::storage::engines::sst::error::SstError::Compaction(format!(
                        "append local-spill PAX row: {error}"
                    ))
                })
            };
            match &ordering {
                SpillOrdering::Ivf(output) => output.for_each_record(&mut append),
                SpillOrdering::Mvcc(output) => {
                    output.for_each_record(|item| append(item.into_record()))
                }
            }
            .map_err(|error| {
                crate::core::StorageError::SstEngine(format!(
                    "stream local-spill rows into PAX: {error}"
                ))
            })?;
            writer.finish().map_err(|error| {
                crate::core::StorageError::SstEngine(format!(
                    "finish disk-backed PAX segment: {error}"
                ))
            })?;
            task_scratch
                .update_phase("uploading", None)
                .map_err(|error| {
                    crate::core::StorageError::SstEngine(format!(
                        "update local-spill upload manifest: {error}"
                    ))
                })?;
            let bytes = staged
                .upload_bounded_retaining_local(&self.filesystem_port())
                .await
                .map_err(|error| {
                    crate::core::StorageError::SstEngine(format!(
                        "upload local-spill PAX segment: {error}"
                    ))
                })?;
            Ok((bytes, staged))
        }
        .await;

        let (bytes_written, staged) = match write_result {
            Ok(output) => output,
            Err(error) => {
                if let (Some(coordinator), Some(operation)) =
                    (&self.atomic_coordinator, &atomic_operation)
                {
                    let _ = coordinator
                        .abort_atomic_operation(
                            &operation.operation_id,
                            "local-spill construction or upload failed",
                        )
                        .await;
                }
                return Err(error);
            }
        };
        memtrace.stage(
            "spill_pax_uploaded",
            usize::try_from(mvcc_stats.output_records).unwrap_or(usize::MAX),
        );
        if let (Some(coordinator), Some(operation)) = (&self.atomic_coordinator, &atomic_operation)
            && let Err(error) = coordinator
                .finalize_atomic_operation(&operation.operation_id)
                .await
        {
            let _ = coordinator
                .abort_atomic_operation(
                    &operation.operation_id,
                    "local-spill atomic publication failed",
                )
                .await;
            return Err(crate::core::StorageError::SstEngine(format!(
                "publish local-spill PAX segment: {error}"
            )));
        }
        if let Err(error) = task_scratch.update_phase("published", None) {
            warn!(%error, "Local-spill publication committed but manifest update failed");
        }

        // D6 for the disk-backed writer: promote from the retained local PAX
        // only after the final path is visible. Cloud scratch still exists at
        // `staged.local_path()`; a local atomic rename moves that file to the
        // task output, so select whichever path remains present.
        if cache_on_write.includes_invariants()
            && let Some((invariants, survivor)) =
                crate::storage::engines::sst::core::get_warm_tier_caches()
        {
            let retained_path = Path::new(staged.local_path());
            let final_local = output_path.strip_prefix("file://").unwrap_or(&output_path);
            let cache_source = if retained_path.exists() {
                retained_path
            } else {
                Path::new(final_local)
            };
            match
                crate::storage::engines::sst::segment_format::install_pax_cache_seed_from_local_file(
                    &output_path,
                    cache_source,
                    cache_on_write.includes_survivors(),
                    Some(&invariants),
                    Some(&survivor),
                )
                .await
            {
                Ok(seeded_to_disk) => info!(
                    output = %output_path,
                    source = %cache_source.display(),
                    seeded_to_disk,
                    include_survivors = cache_on_write.includes_survivors(),
                    "post-publication local-spill cache promotion completed"
                ),
                Err(error) => warn!(
                    output = %output_path,
                    source = %cache_source.display(),
                    %error,
                    "post-publication local-spill cache promotion failed"
                ),
            }
        }
        drop(staged);

        self.retire_task_inputs(task).await.map_err(|error| {
            crate::core::StorageError::SstEngine(format!(
                "local-spill compaction published {} but input retirement failed: {error}",
                task.output_file.display()
            ))
        })?;
        if let Err(error) = task_scratch.update_phase("retired", None) {
            warn!(%error, "Local-spill retirement committed but manifest update failed");
        }
        memtrace.stage(
            "spill_inputs_retired",
            usize::try_from(mvcc_stats.output_records).unwrap_or(usize::MAX),
        );

        let elapsed = start_time.elapsed();
        let cluster_stats = match &ordering {
            SpillOrdering::Ivf(output) => Some(*output.stats()),
            SpillOrdering::Mvcc(_) => None,
        };
        info!(
            input_files = task.input_files.len(),
            input_bytes = task.input_bytes,
            output_bytes = bytes_written,
            input_records = mvcc_stats.input_records,
            output_records = mvcc_stats.output_records,
            mvcc_scratch_bytes = mvcc_stats.scratch_bytes_written,
            input_range_reads = input_stats.block_range_reads + input_stats.sq8_range_reads,
            largest_input_range_bytes = input_stats.largest_range_bytes,
            ivf_scratch_bytes = cluster_stats.map_or(0, |stats| stats.scratch_bytes_written),
            elapsed_ms = elapsed.as_millis(),
            "Local-spill compaction completed"
        );
        {
            use crate::metrics::operational_metrics as metrics;
            metrics::COMPACTIONS_TOTAL.inc();
            metrics::COMPACTION_BYTES_READ_TOTAL.inc_by(task.input_bytes);
            metrics::COMPACTION_BYTES_WRITTEN_TOTAL.inc_by(bytes_written);
            metrics::COMPACTION_SECONDS.observe(elapsed.as_secs_f64());
        }
        Ok(EnhancedSstCompactionStats {
            base_stats: SstCompactionStats {
                total_compactions: 1,
                bytes_written,
                bytes_read: task.input_bytes,
                files_merged: task.input_files.len() as u64,
                avg_compaction_time_ms: elapsed.as_millis() as u64,
                last_compaction_time: Some(Utc::now()),
                expired_records_deleted: 0,
                tombstones_removed: 0,
            },
            deleted_vector_ids: Vec::new(),
            merged_vectors: Vec::new(),
            recommend_full_rebuild: false,
        })
    }

    /// Unified canonical-record compaction path.
    pub async fn perform_compaction_enhanced(
        &self,
        task: &SstCompactionTask,
        _config: &SstConfig,
        atomic_coordinator: Option<Arc<TransactionCoordinator>>,
        // M1-3 (ADR-049): compression only applied to the retired ProximaBlocks
        // streaming writer; PAX/Arrow compaction don't take it. Kept on the
        // signature for API stability and still logged below.
        compression_config: Option<crate::proto::proximadb_v1::CompressionConfig>,
    ) -> Result<EnhancedSstCompactionStats> {
        debug!(
            "🚀 UNIFIED COMPACTION: Single optimized path for {} files at level {}",
            task.input_files.len(),
            task.level
        );

        // Single canonical path: no legacy DTO conversion between reader,
        // MVCC resolution, and writer.
        self.perform_unified_canonical_compaction(
            task,
            _config,
            atomic_coordinator,
            compression_config,
            MergedVectorTracking::Enabled,
        )
        .await
    }

    /// Convert zero-copy stats to enhanced stats format
    #[allow(dead_code)]
    fn convert_zero_copy_stats_to_enhanced(
        &self,
        stats: ZeroCopyCompactionStats,
    ) -> EnhancedSstCompactionStats {
        // For zero-copy compaction, we don't have VectorRecords but we DO have counts
        // Create placeholder VectorRecords just for counting purposes
        let merged_vectors = (0..stats.records_written)
            .map(|_| VectorRecord::default())
            .collect();

        EnhancedSstCompactionStats {
            base_stats: SstCompactionStats {
                total_compactions: 1,
                bytes_written: stats.bytes_written,
                bytes_read: stats.bytes_read,
                files_merged: stats.files_compacted as u64,
                avg_compaction_time_ms: stats.compaction_time_ms,
                last_compaction_time: Some(chrono::Utc::now()),
                expired_records_deleted: stats.records_deleted,
                tombstones_removed: stats.tombstoned_ids.len() as u64,
            },
            deleted_vector_ids: stats.deleted_vector_ids,
            merged_vectors, // Use placeholder records for accurate counting
            recommend_full_rebuild: stats.recommend_index_rebuild,
        }
    }

    /// TD-DELVEC-1 WI-6: collect the oids deleted (DV bit set) across the input
    /// segments, so the compaction merge can physically drop them. Reads each
    /// input's DV (`{input}.dv`, disk-authoritative) + the segment's OID resolver
    /// footer region (position → oid). The canonical merge loses each record's
    /// source segment/position, so this is computed up front from the inputs.
    /// Feature-gated.
    #[cfg(feature = "cold-deletion-vectors")]
    pub(crate) async fn build_deleted_oids(
        filesystem: &crate::storage::persistence::filesystem::FilesystemFactory,
        dv_store: &crate::storage::engines::sst::deletion_vector_store::DeletionVectorStore,
        input_files: &[std::path::PathBuf],
    ) -> std::collections::HashSet<String> {
        use crate::storage::engines::sst::oid_resolve::read_segment_resolver;
        let mut deleted = std::collections::HashSet::new();
        for input in input_files {
            let path = input.to_string_lossy().into_owned();
            // Absent `.dv` → no deletes for this segment (load is a no-op).
            let _ = dv_store.load(&path).await;
            let Some(vdv) = dv_store.get(&path).await else {
                continue;
            };
            let positions = vdv.deleted_positions();
            if positions.is_empty() {
                continue;
            }
            let Ok(Some(resolver)) = read_segment_resolver(filesystem, &path).await else {
                continue;
            };
            for pos in positions {
                if let Some(oid) = resolver.oid_at(pos) {
                    deleted.insert(oid.to_string());
                }
            }
        }
        deleted
    }

    /// Canonical compaction path: `ProximaRecord` ownership flows from the
    /// mixed-format reader through MVCC into the PAX/Arrow writer.
    async fn perform_unified_canonical_compaction(
        &self,
        task: &SstCompactionTask,
        config: &SstConfig,
        atomic_coordinator: Option<Arc<TransactionCoordinator>>,
        compression_config: Option<crate::proto::proximadb_v1::CompressionConfig>,
        merged_vector_tracking: MergedVectorTracking,
    ) -> Result<EnhancedSstCompactionStats> {
        debug!(
            "🚀 UNIFIED COMPACTION: canonical record path with compression: {:?}",
            compression_config
                .as_ref()
                .map(|c| format!("algorithm={}, level={:?}", c.algorithm, c.level))
        );
        let start_time = std::time::Instant::now();

        // Stage-boundary RSS checkpoints (env-gated, PROXIMADB_TRACE_COMPACTION_MEM).
        let mut memtrace = CompactionMemTrace::new();

        // Canonical ownership from reader through writer. The former legacy
        // VectorRecord pivot cloned vectors/properties, dropped record-envelope
        // fields, and then re-expanded the same corpus for MVCC and writing.
        let mut all_records: Vec<ProximaRecord> = Vec::new();
        let mut bytes_read = 0u64;

        debug!(
            "🚀 UNIFIED: Merging {} input files for level {} (canonical record path)",
            task.input_files.len(),
            task.level
        );

        for input_file in &task.input_files {
            let input_path = input_file.to_string_lossy();

            match self.read_all_records_from_file_unified(&input_path).await {
                Ok(records) => {
                    // Per-file checkpoint: catches reader-side residue (decoded
                    // blocks, whole-file byte copies) accumulating across files.
                    memtrace.stage("input_file_read", records.len());
                    info!(
                        "✅ Extracted {} canonical records from {}",
                        records.len(),
                        input_path
                    );
                    // TD-COMPACT-1 S2: resolve metadata from the URL's backend.
                    // A task may contain `az://`, `s3://`, or `file://` inputs;
                    // pinning this call to the local filesystem made cloud
                    // throughput report zero even though the read succeeded.
                    match self.filesystem_factory.metadata(&input_path).await {
                        Ok(metadata) => bytes_read += metadata.size,
                        Err(error) => warn!(
                            "compaction throughput: cannot size input {input_path}: {error} \
                             (bytes_read will under-report)"
                        ),
                    }

                    if records.is_empty() {
                        warn!("No records extracted from SST file: {}", input_path);
                    }

                    all_records.extend(records);
                }
                Err(e) => {
                    // TD-COMPACT-13: FAIL CLOSED. A partial merge that publishes
                    // and then retires ALL inputs would delete the only copy of
                    // the unread rows. Failing leaves every input intact (and,
                    // post- ledger, loud). Poison-input lockout is accepted as
                    // strictly better than silent loss (TD-COMPACT-13
                    // follow-up).
                    return Err(crate::core::StorageError::SstEngine(format!(
                        "input unreadable; refusing to publish a partial merge of {}                          (failed input: {}): {e}",
                        task.output_file.display(),
                        input_path
                    )));
                }
            }
        }

        memtrace.stage("all_records_extended", all_records.len());
        info!(
            "✅ Collected {} canonical records from {} input files",
            all_records.len(),
            task.input_files.len()
        );

        if let Some(record) = all_records.first() {
            memtrace.sample_record(record);
        }

        // TD-DELVEC-1 WI-6: materialize deletion-vector OIDs before source
        // identity is lost by MVCC resolution. Remove every version of a
        // deleted OID from the owned canonical corpus so neither the writer nor
        // optional AXIS merged-vector tracking can observe a deleted row. Keep
        // this store alive through input retirement below so the corresponding
        // `.dv` objects are removed only after the replacement is published.
        #[cfg(feature = "cold-deletion-vectors")]
        let dv_store =
            crate::storage::engines::sst::deletion_vector_store::DeletionVectorStore::new(
                self.filesystem_factory.clone(),
            );
        #[cfg(feature = "cold-deletion-vectors")]
        let deleted_oids =
            Self::build_deleted_oids(&self.filesystem_factory, &dv_store, &task.input_files).await;
        #[cfg(feature = "cold-deletion-vectors")]
        all_records.retain(|record| !deleted_oids.contains(&record.oid));

        let mut prepared = Self::prepare_canonical_records(
            all_records,
            Self::current_time_ns(),
            merged_vector_tracking,
        );
        memtrace.stage("mvcc_resolved_in_place", prepared.records.len());
        info!(
            "🔍 UNIFIED COMPACTION: MVCC resolution retained {} canonical records",
            prepared.records.len()
        );

        let expired_records_count = prepared.expired_records_count;
        let tombstones_removed_count = prepared.tombstones_removed_count;
        let deleted_vector_ids = std::mem::take(&mut prepared.deleted_vector_ids);
        #[cfg(feature = "cold-deletion-vectors")]
        let deleted_vector_ids = {
            let mut deleted_vector_ids = deleted_vector_ids;
            deleted_vector_ids.extend(deleted_oids);
            deleted_vector_ids.sort_unstable();
            deleted_vector_ids.dedup();
            deleted_vector_ids
        };
        let merged_vectors = std::mem::take(&mut prepared.merged_vectors);
        let mut records = prepared.records;

        // Log cleanup statistics
        if expired_records_count > 0 || tombstones_removed_count > 0 {
            info!(
                "🧹 LSM COMPACTION CLEANUP: {} expired records deleted, {} old tombstones removed",
                expired_records_count, tombstones_removed_count
            );
        }

        // Precision is a writer-boundary policy and applies uniformly to PAX
        // and Arrow. Canonical input already preserves non-fp32 payloads; the
        // hint only coerces when the collection policy explicitly requests it.
        if let Some(target) = task.precision_hint {
            for record in &mut records {
                for cell in &mut record.embeddings {
                    cell.coerce_to_precision(target);
                }
            }
        }

        let records_len = records.len();

        // Handle empty records case - return early without writing SSTable
        if records.is_empty() {
            info!(
                "📋 SST COMPACTION: No records to compact after merging. Returning without writing SSTable."
            );
            return Ok(EnhancedSstCompactionStats {
                base_stats: SstCompactionStats {
                    total_compactions: 1,
                    bytes_written: 0,
                    bytes_read,
                    files_merged: task.input_files.len() as u64,
                    avg_compaction_time_ms: start_time.elapsed().as_millis() as u64,
                    last_compaction_time: Some(chrono::Utc::now()),
                    expired_records_deleted: expired_records_count,
                    tombstones_removed: tombstones_removed_count,
                },
                merged_vectors,
                deleted_vector_ids,
                recommend_full_rebuild: false,
            });
        }

        memtrace.stage("pre_write", records.len());

        // M1-3 (ADR-049): the legacy ProximaBlocks write arms (which needed a
        // block size + a writer-local filesystem factory + compression config)
        // are retired — compaction re-emits PAX (`write_pax_segment_compacted`) or
        // Arrow (`ArrowBlockWriter`), neither of which takes those.

        let cache_on_write = std::env::var("PROXIMADB_CACHE_ON_WRITE")
            .ok()
            .and_then(|value| crate::core::config::CacheOnWritePolicy::parse(&value))
            .unwrap_or(config.cache_on_write);
        let mut pax_cache_seed: Option<proximadb_storage_common::pax_block::PaxCacheSeed> = None;

        let bytes_written = if let Some(ref coordinator) = atomic_coordinator {
            // Use atomic operations for compaction
            info!("🔒 LSM COMPACTION: Using atomic operations for compaction_info");

            // Create staging configuration
            // Don't include collection_id in StagingConfig - the base_url already points to the final location
            let staging_config = StagingConfig {
                base_url: task
                    .output_file
                    .parent()
                    .ok_or_else(|| {
                        crate::core::StorageError::SstEngine("Invalid output file path".to_string())
                    })?
                    .to_string_lossy()
                    .to_string(),
                collection_id: None, // Don't add /collections/{id} structure
                operation_type: TransactionStageType::Compaction,
                skip_uuid_subdir: true, // Use simple __compact directory without UUID subdirectory
                ..Default::default()
            };

            // Begin atomic operation
            let atomic_op = coordinator
                .begin_atomic_operation(&staging_config)
                .await
                .map_err(|e| {
                    crate::core::StorageError::SstEngine(format!(
                        "Failed to begin atomic operation: {}",
                        e
                    ))
                })?;

            debug!(
                "Started atomic operation {} for compaction_info",
                atomic_op.operation_id
            );

            // Get the staging filename
            let staging_filename = task
                .output_file
                .file_name()
                .ok_or_else(|| {
                    crate::core::StorageError::SstEngine("Invalid output filename".to_string())
                })?
                .to_string_lossy()
                .to_string();

            // Write the merged segment via a staged write (TD-OBJSTORE-4
            // defect-6 class): the writers below are LOCAL-file writers, and a
            // CLOUD staging URL string-stripped into a "path" writes the output
            // to a literal local `az:...` directory — the promote then stages
            // nothing, compaction false-succeeds, and the INPUT segments (the
            // only copies) get deleted. Stage locally, upload on finalize.
            let staging_file_url = format!(
                "{}/{}",
                atomic_op.staging_url.trim_end_matches('/'),
                staging_filename
            );
            let staged = crate::storage::engines::sst::staged_write::StagedSegmentWrite::begin(
                &staging_file_url,
            )
            .await
            .map_err(|e| {
                crate::core::StorageError::SstEngine(format!("staged compaction write: {e}"))
            })?;
            let staging_file_path = PathBuf::from(staged.local_path());
            debug!("Writing to staging path: {}", staging_file_path.display());

            // Check block format and use appropriate writer
            let block_format = compaction_output_format(&task.input_files);
            debug!("🔍 COMPACTION: Using block format: {:?}", block_format);

            match block_format {
                // M1-3 (ADR-049): ProximaBlocks is folded into PaxBlock — the
                // legacy `.sst` streaming write is retired, so compaction always
                // re-emits PAX for non-Arrow output (unreachable here anyway, since
                // `compaction_output_format` never returns ProximaBlocks).
                BlockFormat::PaxBlock | BlockFormat::ProximaBlocks => {
                    // P3 Phase E: compact to a `.pax` segment, re-encoding the merged
                    // records with RaBitQ. Mixed-read-safe — source segments may be
                    // legacy `.sst` or `.pax` (canonical records already merged them); the
                    // rewrite emits the configured PAX format. The output filename is
                    // format-aware (`.pax`) via `generate_output_file_path`, so
                    // discovery + the cascade dispatch route it correctly.
                    let collection_id = self
                        .extract_collection_id_from_paths(&task.input_files)
                        .unwrap_or_else(|_| "default".to_string());
                    if let Some(parent) = staging_file_path.parent() {
                        std::fs::create_dir_all(parent)
                            .map_err(crate::core::StorageError::DiskIO)?;
                    }
                    let f32_tier =
                        crate::storage::engines::sst::segment_format::pax_inputs_have_f32_tier(
                            &task.input_files,
                        );
                    // Preserve the source segments' tier-2 rerank quant (SQ8/FP16/f32)
                    // through compaction — see `pax_inputs_rerank_quant`.
                    let rerank_quant =
                        crate::storage::engines::sst::segment_format::pax_inputs_rerank_quant(
                            &task.input_files,
                        );
                    // TD-COMPACT-2 root cause: this argument is the collection's
                    // embedding-MODALITY count, not the record count. Passing
                    // `records.len()` made the block writer allocate one embedding
                    // column buffer PER RECORD (105k columns for a 105k-record
                    // compaction → ~40 GB of untracked per-block row buffers and
                    // ~42 s/block serializing all-null stripes). Derive it from the
                    // merged records (compaction has no collection config in hand).
                    let embedding_count = records
                        .iter()
                        .map(|r| r.embeddings.len())
                        .max()
                        .unwrap_or(1)
                        .max(1);
                    // TD-FPRUNE-1 A1: preserve the shredded filterable-tag columns
                    // across compaction. Rebuild the P-Shred spec from an input
                    // segment's SELF-DESCRIBING footer `shred_field_map` (no catalog
                    // / schema in hand at compaction). The merged records keep their
                    // `props` (ADR-055), so re-shred with the same spec reproduces
                    // the columns. Empty (legacy/non-shredded inputs) ⇒ byte-identical.
                    let shred_spec = Self::extract_shred_spec_from_inputs(
                        &self.filesystem_port(),
                        &task.input_files,
                    )
                    .await;
                    if cache_on_write.includes_invariants() {
                        let write = crate::storage::engines::sst::segment_format::
                            write_pax_segment_compacted_with_cache_seed_shredded(
                                &staging_file_path,
                                &records,
                                &collection_id,
                                embedding_count,
                                proximadb_block_format::VectorQuant::RaBitQ,
                                rerank_quant,
                                f32_tier,
                                None,
                                cache_on_write.includes_survivors(),
                                &shred_spec,
                             task.output_file.to_str())
                            .map_err(|e| {
                                crate::core::StorageError::SstEngine(format!(
                                    "PAX compaction write failed: {e}"
                                ))
                            })?;
                        pax_cache_seed = write.cache_seed;
                    } else {
                        crate::storage::engines::sst::segment_format::write_pax_segment_compacted_shredded(
                            &staging_file_path,
                            &records,
                            &collection_id,
                            embedding_count,
                            proximadb_block_format::VectorQuant::RaBitQ,
                            rerank_quant,
                            f32_tier,
                            None,
                            &shred_spec,
                         task.output_file.to_str())
                        .map_err(|e| {
                            crate::core::StorageError::SstEngine(format!(
                                "PAX compaction write failed: {e}"
                            ))
                        })?;
                    }
                    info!(
                        "✅ COMPACTION: Wrote {} records to PAX segment",
                        records.len()
                    );
                }
                BlockFormat::ArrowBlock => {
                    // Use ArrowBlockWriter for Arrow IPC format
                    debug!("🔍 COMPACTION: Using ArrowBlockWriter for Arrow format");

                    // Infer dimension from first record
                    let dimension = records
                        .first()
                        .and_then(|record| record.embeddings.first())
                        .map_or(128, |embedding| embedding.dim);

                    // Ensure parent directory exists
                    if let Some(parent) = staging_file_path.parent() {
                        std::fs::create_dir_all(parent)
                            .map_err(crate::core::StorageError::DiskIO)?;
                    }

                    let config = ArrowBlockConfig::new(dimension);
                    let mut writer = ArrowBlockWriter::new(&staging_file_path, config)
                        .map_err(|e| crate::core::StorageError::SstEngine(e.to_string()))?;

                    writer
                        .write_block(&records)
                        .map_err(|e| crate::core::StorageError::SstEngine(e.to_string()))?;
                    writer
                        .finalize()
                        .map_err(|e| crate::core::StorageError::SstEngine(e.to_string()))?;

                    info!(
                        "✅ COMPACTION: Wrote {} records to Arrow block",
                        records.len()
                    );
                }
            }

            // Promote the locally staged merged segment to the (possibly remote)
            // staging URL so the atomic commit has a real file to move.
            // finalize returns the byte count — do NOT probe the scratch path
            // afterwards (finalize removes it; the old post-finalize local
            // metadata probe made cloud compaction fail at 100%).
            let written_bytes = staged
                .finalize(&self.filesystem_port())
                .await
                .map_err(|e| {
                    crate::core::StorageError::SstEngine(format!(
                        "upload staged compaction segment: {e}"
                    ))
                })?;

            // Finalize atomic operation - this moves the file from staging to final location
            coordinator
                .finalize_atomic_operation(&atomic_op.operation_id)
                .await
                .map_err(|e| {
                    crate::core::StorageError::SstEngine(format!(
                        "Failed to finalize atomic operation: {}",
                        e
                    ))
                })?;

            info!(
                "✅ LSM COMPACTION: Atomic operation {} completed successfully",
                atomic_op.operation_id
            );
            debug!("File should be at final location: {}", atomic_op.final_url);

            written_bytes
        } else {
            // Fallback to direct write (non-atomic)
            debug!("Writing directly to: {}", task.output_file.display());

            // Check block format and use appropriate writer
            let block_format = compaction_output_format(&task.input_files);
            debug!(
                "🔍 COMPACTION (non-atomic): Using block format: {:?}",
                block_format
            );

            match block_format {
                // M1-3 (ADR-049): ProximaBlocks is folded into PaxBlock — see the
                // atomic arm above (legacy `.sst` streaming write retired).
                BlockFormat::PaxBlock | BlockFormat::ProximaBlocks => {
                    // P3 Phase E: compact to a `.pax` segment (re-encode RaBitQ).
                    // See the atomic arm above; this is the non-atomic direct-write path.
                    let collection_id = self
                        .extract_collection_id_from_paths(&task.input_files)
                        .unwrap_or_else(|_| "default".to_string());
                    if let Some(parent) = task.output_file.parent() {
                        std::fs::create_dir_all(parent)
                            .map_err(crate::core::StorageError::DiskIO)?;
                    }
                    let f32_tier =
                        crate::storage::engines::sst::segment_format::pax_inputs_have_f32_tier(
                            &task.input_files,
                        );
                    // Preserve the source segments' tier-2 rerank quant (SQ8/FP16/f32)
                    // through compaction — see `pax_inputs_rerank_quant`.
                    let rerank_quant =
                        crate::storage::engines::sst::segment_format::pax_inputs_rerank_quant(
                            &task.input_files,
                        );
                    // TD-COMPACT-2 root cause: embedding-MODALITY count, not the
                    // record count — see the atomic-path call above.
                    let embedding_count = records
                        .iter()
                        .map(|r| r.embeddings.len())
                        .max()
                        .unwrap_or(1)
                        .max(1);
                    // TD-FPRUNE-1 A1: preserve shredded filterable-tag columns across
                    // compaction (see the atomic arm above) — rebuild the spec from an
                    // input segment's self-describing footer.
                    let shred_spec = Self::extract_shred_spec_from_inputs(
                        &self.filesystem_port(),
                        &task.input_files,
                    )
                    .await;
                    if cache_on_write.includes_invariants() {
                        let write = crate::storage::engines::sst::segment_format::
                            write_pax_segment_compacted_with_cache_seed_shredded(
                                &task.output_file,
                                &records,
                                &collection_id,
                                embedding_count,
                                proximadb_block_format::VectorQuant::RaBitQ,
                                rerank_quant,
                                f32_tier,
                                None,
                                cache_on_write.includes_survivors(),
                                &shred_spec,
                             task.output_file.to_str())
                            .map_err(|e| {
                                crate::core::StorageError::SstEngine(format!(
                                    "PAX compaction write failed: {e}"
                                ))
                            })?;
                        pax_cache_seed = write.cache_seed;
                    } else {
                        crate::storage::engines::sst::segment_format::write_pax_segment_compacted_shredded(
                            &task.output_file,
                            &records,
                            &collection_id,
                            embedding_count,
                            proximadb_block_format::VectorQuant::RaBitQ,
                            rerank_quant,
                            f32_tier,
                            None,
                            &shred_spec,
                         task.output_file.to_str())
                        .map_err(|e| {
                            crate::core::StorageError::SstEngine(format!(
                                "PAX compaction write failed: {e}"
                            ))
                        })?;
                    }
                    info!(
                        "✅ COMPACTION (non-atomic): Wrote {} records to PAX segment",
                        records.len()
                    );
                }
                BlockFormat::ArrowBlock => {
                    // Use ArrowBlockWriter for Arrow IPC format
                    debug!("🔍 COMPACTION (non-atomic): Using ArrowBlockWriter for Arrow format");

                    // Infer dimension from first record
                    let dimension = records
                        .first()
                        .and_then(|record| record.embeddings.first())
                        .map_or(128, |embedding| embedding.dim);

                    // Ensure parent directory exists
                    if let Some(parent) = task.output_file.parent() {
                        std::fs::create_dir_all(parent)
                            .map_err(crate::core::StorageError::DiskIO)?;
                    }

                    let config = ArrowBlockConfig::new(dimension);
                    let mut writer = ArrowBlockWriter::new(&task.output_file, config)
                        .map_err(|e| crate::core::StorageError::SstEngine(e.to_string()))?;

                    writer
                        .write_block(&records)
                        .map_err(|e| crate::core::StorageError::SstEngine(e.to_string()))?;
                    writer
                        .finalize()
                        .map_err(|e| crate::core::StorageError::SstEngine(e.to_string()))?;

                    info!(
                        "✅ COMPACTION (non-atomic): Wrote {} records to Arrow block",
                        records.len()
                    );
                }
            }

            // TD-COMPACT-1 S2: size the output through its URL-selected
            // backend, and never fail the already-durable compaction over a
            // metric probe.
            let output_path = task.output_file.to_string_lossy();
            match self.filesystem_factory.metadata(&output_path).await {
                Ok(metadata) => metadata.size,
                Err(error) => {
                    warn!(
                        "compaction throughput: cannot size output {output_path}: {error} \
                         (bytes_written will report 0)"
                    );
                    0
                }
            }
        };
        memtrace.stage("write_done", records_len);

        debug!(
            "Wrote {} bytes to output file {}",
            bytes_written,
            task.output_file.display()
        );

        // NOTE: Manifest removed - using directory-based discovery instead
        // Files are automatically discovered by scanning collection directories
        debug!(
            "Compaction completed - files will be discovered automatically by directory scanning"
        );

        // D6: populate only after the output is published at its final path.
        // A staged/failed write must never become visible through the warm tier.
        let warm_caches = crate::storage::engines::sst::core::get_warm_tier_caches();
        if let (Some(seed), Some((invariants, survivor))) = (pax_cache_seed, &warm_caches) {
            crate::storage::engines::sst::segment_format::install_pax_cache_seed(
                &task.output_file.to_string_lossy(),
                seed,
                Some(invariants),
                Some(survivor),
            )
            .await;
        }

        // Remove input files after successful compaction using plugin filesystem
        // TD-CACHE-2 S2d: evict the warm-tier cache entries for each deleted
        // input file — correctness is structural (UUID-unique outputs, old
        // paths never queried again), but without eviction the dead entries
        // squat in the invariants/survivor budgets until recency ages them out.
        // TD-COMPACT-13: shared bounded-retry retirement; residual failures are
        // recorded as durable obligations (sidecar + pending set + gauge) and
        // the task SUCCEEDS — the output is published and correct, and failing
        // here only killed the follow-up chain that re-drives retirement (the
        // exact mechanism that made the campaign defect unrecoverable).
        let retirement_pending = self
            .retire_inputs_with_retry(&task.output_file, &task.input_files)
            .await?;
        if !retirement_pending.is_empty() {
            warn!(
                output = %task.output_file.display(),
                pending = retirement_pending.len(),
                "unified-compaction inputs pending retirement (obligation recorded; \
                 discovery filters them until drained)"
            );
        }

        // DETAILED COMPACTION PERFORMANCE ANALYSIS
        let total_time = start_time.elapsed();
        let input_files_count = task.input_files.len();
        let compression_ratio = if bytes_read > 0 {
            bytes_written as f64 / bytes_read as f64
        } else {
            1.0
        };
        let read_throughput_mb_sec =
            (bytes_read as f64 / 1024.0 / 1024.0) / total_time.as_secs_f64();
        let write_throughput_mb_sec =
            (bytes_written as f64 / 1024.0 / 1024.0) / total_time.as_secs_f64();

        tracing::info!(
            "🗜️ [LSM COMPACTION] Level {} complete: {} files → 1 file in {:?}",
            task.level,
            input_files_count,
            total_time
        );

        // Print absolute bytes alongside the rate: a long stall (e.g. the
        // TD-COMPACT-2 swap storm) makes a real ~28 MB compaction print as
        // "0.0MB/s" at one decimal, which reads as "metric broken" (the
        // original TD-COMPACT-1 S2 misdiagnosis).
        tracing::info!(
            "⚡ [LSM COMPACTION PERFORMANCE] Read: {:.1}MB ({:.2}MB/s), Write: {:.1}MB ({:.2}MB/s), Compression: {:.1}x",
            bytes_read as f64 / 1024.0 / 1024.0,
            read_throughput_mb_sec,
            bytes_written as f64 / 1024.0 / 1024.0,
            write_throughput_mb_sec,
            compression_ratio
        );

        // TD-COMPACT-1 S2: event-sourced compaction counters (default
        // prometheus registry → /metrics/prometheus).
        {
            use crate::metrics::operational_metrics as om;
            om::COMPACTIONS_TOTAL.inc();
            om::COMPACTION_BYTES_READ_TOTAL.inc_by(bytes_read);
            om::COMPACTION_BYTES_WRITTEN_TOTAL.inc_by(bytes_written);
            om::COMPACTION_SECONDS.observe(total_time.as_secs_f64());
        }

        // COMPACTION PERFORMANCE WARNINGS (compaction can be slower than flush)
        if total_time.as_millis() > 5000 {
            // >5s is very slow for compaction
            tracing::warn!(
                "⚠️ SLOW LSM COMPACTION: {}ms for {} files. Consider:",
                total_time.as_millis(),
                input_files_count
            );
            tracing::warn!("   • Moving compaction to dedicated background process");
            tracing::warn!("   • Using faster storage for compaction temp files");
            tracing::warn!("   • Reducing compaction scope/frequency");
        }

        if read_throughput_mb_sec < 50.0 {
            // <50MB/s read is slow
            tracing::warn!(
                "⚠️ LSM COMPACTION READ WARNING: {:.1}MB/s below target 50MB/s",
                read_throughput_mb_sec
            );
        }

        if write_throughput_mb_sec < 30.0 {
            // <30MB/s write is slow
            tracing::warn!(
                "⚠️ LSM COMPACTION WRITE WARNING: {:.1}MB/s below target 30MB/s",
                write_throughput_mb_sec
            );
        }

        // DESIGN INSIGHT: Comparison with flush performance
        if total_time.as_millis() > 1000 {
            tracing::info!(
                "💡 DESIGN INSIGHT: LSM compaction ({}ms) much slower than VIPER flush target (<200ms) - async compaction recommended",
                total_time.as_millis()
            );
        }

        debug!(
            "🗜️ LSM compaction stats: {}MB read, {}MB written, {:.1}x compression, {} records merged, {} expired deleted, {} tombstones removed",
            bytes_read / 1024 / 1024,
            bytes_written / 1024 / 1024,
            compression_ratio,
            records_len,
            expired_records_count,
            tombstones_removed_count
        );

        Ok(EnhancedSstCompactionStats {
            base_stats: SstCompactionStats {
                total_compactions: 1,
                bytes_written,
                bytes_read,
                files_merged: task.input_files.len() as u64,
                avg_compaction_time_ms: start_time.elapsed().as_millis() as u64,
                last_compaction_time: Some(Utc::now()),
                expired_records_deleted: expired_records_count,
                tombstones_removed: tombstones_removed_count,
            },
            deleted_vector_ids,
            merged_vectors,
            recommend_full_rebuild: false,
        })
    }

    /// Get SST files organized by level using unified framework
    #[allow(dead_code)]
    async fn get_sst_files_by_level(
        &self,
        collection_dir: &Path,
    ) -> Result<HashMap<u8, Vec<PathBuf>>> {
        debug!(
            "🔍 COMPACTION: Using unified framework to discover SST files in: {}",
            collection_dir.display()
        );

        let _collection_path = collection_dir.to_string_lossy();

        // Use new orchestrator if available, otherwise fall back to direct file discovery
        let unified_files = if let Some(ref orchestrator) = self.compaction_orchestrator {
            orchestrator
                .registry
                .discover_files(&orchestrator.filesystem, &_collection_path, "sst")
                .await
                .map_err(|e| crate::core::StorageError::SstEngine(e.to_string()))?
        } else {
            // Fallback to empty result if no orchestrator
            HashMap::new()
        };

        // Convert from unified format to legacy format for compatibility
        let mut files_by_level = HashMap::new();
        for (level, metadata_list) in unified_files {
            let paths: Vec<PathBuf> = metadata_list
                .into_iter()
                .map(|metadata| PathBuf::from(&metadata.path))
                .collect();

            if !paths.is_empty() {
                files_by_level.insert(level as u8, paths);
            }
        }

        debug!(
            "📊 COMPACTION: Found SST files by level using unified framework: {:?}",
            files_by_level
                .iter()
                .map(|(level, files)| (*level, files.len()))
                .collect::<Vec<_>>()
        );

        Ok(files_by_level)
    }

    /// Generate output file path for compacted file
    /// Uses unified FilenameCodec for consistency with compaction framework
    /// Respects the configured block_format setting for file extension
    fn generate_output_file_path(
        &self,
        _collection_object_id: crate::core::stable_id::CollectionObjectId,
        collection_dir: &Path,
        level: u8,
        block_format: BlockFormat,
    ) -> PathBuf {
        use crate::storage::common::compaction_orchestrator::FilenameCodec;

        let codec = FilenameCodec::new();
        let extension = match block_format {
            BlockFormat::ArrowBlock => "arrow",
            BlockFormat::ProximaBlocks => "sst",
            BlockFormat::PaxBlock => "pax",
        };
        let filename = codec.generate(level as u32, extension);
        collection_dir.join(filename)
    }

    /// TD-FPRUNE-1 A1: rebuild the P-Shred `(name, col-id)` spec from the input
    /// segments' SELF-DESCRIBING footer `shred_field_map` so compaction preserves
    /// the shredded filterable-tag user-columns (footer filter-pruning survives
    /// L1/L2). Compaction has no catalog/schema in hand — the segment footer is
    /// the authority. Reads input footers in order and takes the FIRST non-empty
    /// map (all L0 flushes of one collection share the same filterable schema; a
    /// full union across inputs is a robustness follow-up for cross-flush schema
    /// evolution). Any read/parse failure ⇒ empty spec (byte-identical, no shred),
    /// never a compaction failure.
    async fn extract_shred_spec_from_inputs(
        factory: &Arc<dyn proximadb_storage_ports::FilesystemPort>,
        input_files: &[PathBuf],
    ) -> Vec<(String, i32)> {
        use proximadb_storage_common::segment_layout::SegmentFooterIndex;
        for input in input_files {
            let url = input.to_string_lossy();
            let Ok(bytes) =
                crate::storage::engines::sst::staged_write::read_object_bytes(factory, &url).await
            else {
                continue;
            };
            if let Ok(Some(footer)) = SegmentFooterIndex::locate_in_segment(&bytes)
                && !footer.shred_field_map.is_empty()
            {
                return footer.shred_field_map;
            }
        }
        Vec::new()
    }

    /// 🚀 NEW: Read all records from SSTable using unified reader with compaction optimizations
    async fn read_all_records_from_file_unified(
        &self,
        file_path: &str,
    ) -> Result<Vec<ProximaRecord>> {
        info!(
            "🔥 COMPACTION UNIFIED: Reading {} with optimized strategy",
            file_path
        );

        // Mixed-read: decode a PAX segment via the canonical inverse
        // (`read_segment_records` — handles the legacy-block, coalesced v1
        // AND two-level v3 layouts, including the Region-B vector overlay).
        // The unified SSTable reader below predates the coalesced layout
        // ("Invalid SSTable magic marker"), which made the ARMED compaction of
        // coalesced `.pax` L0s a silent no-op: 0 records collected, the
        // empty-merge early return reported success, inputs stayed unmerged —
        // caught by the TD-RDSTRAT-8 production-trigger route proof. Non-PAX
        // inputs fall through to the unified reader unchanged.
        //
        // Bytes come through the sanctioned URL→bytes boundary (#1082
        // `read_object_bytes`), so a cloud-hosted `.pax` L0 reads through the
        // FileSystem — never `tokio::fs::read` on a raw scheme path (the
        // URL-string-strip class #1082 removed; would be silent data loss here
        // since false-success compaction deletes the input segments).
        if file_path.ends_with(proximadb_storage_common::pax_block::PAX_SEGMENT_EXT)
            && let Ok(bytes) = crate::storage::engines::sst::staged_write::read_object_bytes(
                &self.filesystem_port(),
                file_path,
            )
            .await
            && crate::storage::engines::sst::segment_format::SegmentFormat::detect(&bytes)
                == crate::storage::engines::sst::segment_format::SegmentFormat::Pax
        {
            let records = crate::storage::engines::sst::segment_format::read_segment_records(
                &bytes,
                &[],
                &[],
                None,
            )
            .map_err(|e| {
                crate::core::StorageError::SstEngine(format!(
                    "PAX compaction read failed for {file_path}: {e}"
                ))
            })?;
            info!(
                "✅ COMPACTION UNIFIED: Read {} records from PAX segment {}",
                records.len(),
                file_path
            );
            return Ok(records);
        }

        // Use compaction-optimized reading strategy
        match self
            .unified_reader
            .read_all_records_for_compaction(&[file_path.to_string()])
            .await
        {
            Ok(records) => {
                info!(
                    "✅ COMPACTION UNIFIED: Successfully read {} records from {}",
                    records.len(),
                    file_path
                );

                // Debug: print sample records
                for (i, record) in records.iter().take(3).enumerate() {
                    debug!(
                        "  UNIFIED Record {}: id={:?}, vector_len={}, metadata_len={}",
                        i,
                        record.oid,
                        record
                            .embeddings
                            .first()
                            .map_or(0, |embedding| embedding.values.len()),
                        record.props.len()
                    );
                }

                Ok(records)
            }
            Err(e) => {
                warn!("❌ COMPACTION UNIFIED: Failed to read {}: {}", file_path, e);
                Err(crate::core::StorageError::SstEngine(format!(
                    "Unified reader failed for {}: {}",
                    file_path, e
                )))
            }
        }
    }

    fn prepare_canonical_records(
        records: Vec<ProximaRecord>,
        current_time_ns: i64,
        merged_vector_tracking: MergedVectorTracking,
    ) -> PreparedCompactionRecords {
        const TOMBSTONE_RETENTION_NS: i64 = 60 * 60 * 1_000_000_000;

        let mut expired_records_count = 0u64;
        let mut tombstones_removed_count = 0u64;
        let mut deleted_vector_ids = Vec::new();
        for record in &records {
            let is_expired = record
                .valid_to_ns
                .is_some_and(|valid_to_ns| valid_to_ns < current_time_ns);
            if !is_expired {
                continue;
            }

            deleted_vector_ids.push(record.oid.clone());
            if record.embeddings.is_empty() {
                let age_ns = current_time_ns.saturating_sub(record.created_at_ns);
                if age_ns >= TOMBSTONE_RETENTION_NS {
                    tombstones_removed_count += 1;
                }
            } else {
                expired_records_count += 1;
            }
        }
        deleted_vector_ids.sort_unstable();
        deleted_vector_ids.dedup();

        let resolver = MvccResolver::with_timestamp_ns(current_time_ns);
        let records = resolver.resolve_sorted_batch(records);
        let merged_vectors = match merged_vector_tracking {
            MergedVectorTracking::Disabled => Vec::new(),
            MergedVectorTracking::Enabled => records
                .iter()
                .filter(|record| !record.embeddings.is_empty())
                .map(VectorRecord::from)
                .collect(),
        };

        PreparedCompactionRecords {
            records,
            deleted_vector_ids,
            merged_vectors,
            expired_records_count,
            tombstones_removed_count,
        }
    }

    fn current_time_ns() -> i64 {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        nanos.min(i64::MAX as u128) as i64
    }

    fn epoch_millis(timestamp: i64) -> i64 {
        let abs = timestamp.unsigned_abs();
        if timestamp == 0 {
            0
        } else if abs < 10_000_000_000 {
            timestamp * 1000
        } else if abs > 10_000_000_000_000_000 {
            timestamp / 1_000_000
        } else if abs > 10_000_000_000_000 {
            timestamp / 1000
        } else {
            timestamp
        }
    }
}

type WarmTierCachePair = (
    Arc<crate::storage::engines::sst::segment_format::SegmentInvariantsCache>,
    Arc<crate::storage::engines::sst::survivor_range_cache::SurvivorRangeCache>,
);

/// Retire one immutable compaction input through the backend selected by its
/// URL. A missing input is an idempotent success: another already-running
/// morsel may have completed retirement after this task took its discovery
/// snapshot. Every other backend error is surfaced so the worker cannot report
/// a false-success or schedule a follow-up over still-live inputs.
async fn retire_compaction_input(factory: &FilesystemFactory, path: &str) -> FsResult<()> {
    match factory.delete(path).await {
        Ok(()) | Err(FilesystemError::NotFound(_)) => Ok(()),
        Err(error) => Err(error),
    }
}

/// Purge immutable bytes for one retired compaction input.
///
/// The compaction loop invokes this only after the backing filesystem confirms
/// deletion. A failed delete therefore keeps acceleration bytes for a
/// still-discoverable segment.
async fn purge_retired_segment_cache_entries(
    path: &str,
    warm_caches: Option<&WarmTierCachePair>,
) -> usize {
    // TD-DELVEC-1 C2: invalidate the OID resolver cache entry for this retired
    // segment — BEFORE the warm_caches guard, so it fires even when the warm-tier
    // pair is None (independent cache). Miss-safe (no-op if the resolver was never
    // loaded for this path).
    #[cfg(feature = "cold-deletion-vectors")]
    if let Some(cache) = crate::storage::engines::sst::core::get_global_oid_resolver_cache() {
        cache.invalidate(path);
    }
    let Some((invariants, survivor)) = warm_caches else {
        return 0;
    };
    invariants.invalidate_all(path).await;
    survivor.purge_path(path).await
}

impl Drop for Compaction {
    fn drop(&mut self) {
        self.shutdown_signal.store(true, Ordering::SeqCst);

        // Abort remaining worker handles
        if let Ok(handles) = self.worker_handles.lock() {
            for handle in handles.iter() {
                handle.abort();
            }
        }
    }
}

// Test module for vector tracking during compaction
// Deferred: Missing test file - compaction_vector_tracking_tests.rs
// #[cfg(test)]
// #[path = "compaction_vector_tracking_tests.rs"]
// mod vector_tracking_tests;

#[cfg(test)]
#[cfg_attr(test, path = "compaction_tests.rs")]
mod tests;

#[cfg(test)]
mod compaction_format_tests {
    use super::BlockFormat;
    use super::compaction_output_format;

    /// M1-3 (ADR-049): compaction output format follows the INPUT segments' format
    /// — PAX is the only vector write format, so `.pax` inputs compact to `.pax`
    /// AND legacy `.sst` inputs compact FORWARD to `.pax` (compaction is the
    /// migration on-ramp). Arrow inputs stay Arrow. Mixed `.sst`+`.pax` → PAX.
    #[test]
    fn compaction_output_format_preserves_input_format() {
        // PAX input → PAX.
        assert_eq!(
            compaction_output_format(&["/c/x.pax".to_string()]),
            BlockFormat::PaxBlock
        );
        // Legacy `.sst` input → compact FORWARD to PAX (streaming retired).
        assert_eq!(
            compaction_output_format(&["/c/x.sst".to_string()]),
            BlockFormat::PaxBlock
        );
        // Mixed `.sst` + `.pax` → consolidate to PAX.
        assert_eq!(
            compaction_output_format(&["/c/a.sst".to_string(), "/c/b.pax".to_string()]),
            BlockFormat::PaxBlock
        );
        // Arrow input → Arrow.
        assert_eq!(
            compaction_output_format(&["/c/x.arrow".to_string()]),
            BlockFormat::ArrowBlock
        );
        // No recognized inputs (no extension) → PAX (the default vector format).
        assert_eq!(
            compaction_output_format(&[] as &[String]),
            BlockFormat::PaxBlock
        );
    }
}

#[cfg(test)]
mod compaction_cache_retirement_tests {
    use super::{purge_retired_segment_cache_entries, retire_compaction_input};
    use crate::storage::engines::sst::segment_format::{SegmentInvariants, SegmentInvariantsCache};
    use crate::storage::engines::sst::survivor_range_cache::SurvivorRangeCache;
    use crate::storage::persistence::filesystem::{FilesystemError, FilesystemFactory};
    use proximadb_cache::PersistentByteStore;
    use std::sync::Arc;

    /// TD-DELVEC-1 C2: compaction retire invalidates the OID resolver cache entry
    /// for the retired segment, even when the warm-tier pair is None (the resolver
    /// cache is independent of the warm caches).
    #[cfg(feature = "cold-deletion-vectors")]
    #[tokio::test]
    async fn retire_invalidates_oid_resolver_cache_even_when_warm_caches_none() {
        use crate::storage::engines::sst::core::set_global_oid_resolver_cache_for_tests;
        use crate::storage::engines::sst::oid_resolver_cache::OidResolverCache;
        use proximadb_storage_common::oid_position_resolver::OidPositionResolver;

        let path = "c2://test/retired-segment.pax";
        let cache = Arc::new(OidResolverCache::new(1024 * 1024));
        set_global_oid_resolver_cache_for_tests(cache.clone());
        cache.put(
            path.to_string(),
            Arc::new(OidPositionResolver::from_stream_order(vec![
                "oid-a".to_string(),
            ])),
        );
        assert!(cache.get(path).is_some(), "precondition: entry present");

        // None warm_caches deliberately — proves the invalidate fires before the
        // warm_caches guard (the resolver cache is independent).
        let _ = purge_retired_segment_cache_entries(path, None).await;

        assert!(
            cache.get(path).is_none(),
            "retire must invalidate the entry"
        );
        assert!(cache.is_empty(), "cache should be empty after retire");
    }

    #[tokio::test]
    async fn input_retirement_routes_by_url_scheme_and_never_falls_back_to_local() {
        let factory = FilesystemFactory::create_default()
            .await
            .expect("create filesystem factory");
        let error =
            retire_compaction_input(&factory, "retirement-spy://bucket/collection/L0_input.pax")
                .await
                .expect_err("unknown object-store scheme must not be treated as a local path");

        assert!(
            matches!(error, FilesystemError::UnsupportedScheme(ref scheme)
                if scheme == "retirement-spy"),
            "unexpected routing error: {error}"
        );
    }

    #[tokio::test]
    async fn input_retirement_deletes_local_file_and_is_idempotent() {
        let directory = tempfile::tempdir().expect("retirement tempdir");
        let input = directory.path().join("L0_input.pax");
        std::fs::write(&input, b"segment").expect("write retirement fixture");
        let input_url = format!("file://{}", input.display());
        let factory = FilesystemFactory::create_default()
            .await
            .expect("create filesystem factory");

        retire_compaction_input(&factory, &input_url)
            .await
            .expect("retire existing input");
        assert!(!input.exists(), "input must be deleted");

        retire_compaction_input(&factory, &input_url)
            .await
            .expect("retiring an absent immutable input is idempotent");
    }

    #[tokio::test]
    async fn retired_input_purges_dram_and_both_persistent_namespaces() {
        let directory = tempfile::tempdir().expect("cache tempdir");
        let store = Arc::new(
            PersistentByteStore::open(directory.path(), 1 << 20).expect("persistent cache store"),
        );
        let invariants = Arc::new(SegmentInvariantsCache::with_l2(1 << 20, store.clone()));
        let survivor = Arc::new(SurvivorRangeCache::with_resolver_and_l2(
            1 << 20,
            None,
            Some(store.clone()),
        ));
        let path = "file:///tenant-a/collection-a/L1_retired.pax";
        invariants
            .put_with_l2(
                path.to_string(),
                Arc::new(SegmentInvariants {
                    header_bytes: vec![1, 2, 3],
                    region_bytes: Some(Arc::from(&b"region-a"[..])),
                    footer_bytes: vec![4, 5],
                    a0_bytes: None,
                    rabitq_header_bytes: None,
                }),
            )
            .await;
        survivor
            .seed_parent_region(path, 1_000, Arc::from(&b"sq8-region"[..]))
            .await
            .expect("seed survivor parent");

        let pair = (invariants.clone(), survivor);
        purge_retired_segment_cache_entries(path, Some(&pair)).await;

        assert!(invariants.get(path).is_none(), "invariant DRAM entry");
        assert!(
            store
                .get(&format!("invariants/control/{path}"))
                .await
                .expect("read invariant L2")
                .is_none(),
            "invariant persistent entry"
        );
        assert!(
            store
                .get(&format!(
                    "survivor-parent/{path}:{}:{}",
                    1_000,
                    b"sq8-region".len()
                ))
                .await
                .expect("read survivor L2")
                .is_none(),
            "survivor persistent parent"
        );
    }

    #[tokio::test]
    async fn collection_prefix_retirement_preserves_similarly_named_sibling() {
        let directory = tempfile::tempdir().expect("cache tempdir");
        let store = Arc::new(
            PersistentByteStore::open(directory.path(), 1 << 20).expect("persistent cache store"),
        );
        let invariants = SegmentInvariantsCache::with_l2(1 << 20, store.clone());
        let survivor = SurvivorRangeCache::with_resolver_and_l2(1 << 20, None, Some(store.clone()));
        let retired_root = "file:///tenant-a/collection-12";
        let retired_path = "file:///tenant-a/collection-12/L1.pax";
        let sibling_path = "file:///tenant-a/collection-123/L1.pax";

        for path in [retired_path, sibling_path] {
            invariants
                .put_with_l2(
                    path.to_string(),
                    Arc::new(SegmentInvariants {
                        header_bytes: vec![1],
                        region_bytes: Some(Arc::from(&b"region"[..])),
                        footer_bytes: vec![2],
                        a0_bytes: None,
                        rabitq_header_bytes: None,
                    }),
                )
                .await;
            survivor
                .seed_parent_region(path, 100, Arc::from(&b"sq8"[..]))
                .await
                .expect("seed survivor parent");
        }

        assert!(invariants.invalidate_prefix_all(retired_root).await > 0);
        assert!(survivor.purge_prefix(retired_root).await > 0);
        assert!(invariants.get(retired_path).is_none());
        assert!(
            invariants.get(sibling_path).is_some(),
            "separator-aware prefix retirement must preserve collection-123"
        );
        assert!(
            store
                .get(&format!("survivor-parent/{retired_path}:100:3"))
                .await
                .expect("retired persistent lookup")
                .is_none()
        );
        assert!(
            store
                .get(&format!("survivor-parent/{sibling_path}:100:3"))
                .await
                .expect("sibling persistent lookup")
                .is_some()
        );
    }
}
