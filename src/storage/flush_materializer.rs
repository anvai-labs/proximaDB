//! Shared per-collection flush materialization.
//!
//! The server shutdown flush (`StorageEngine::flush_memtable_to_storage`) and the
//! embedded flush (`EmbeddedDb::flush`) both have to turn a collection's unflushed
//! WAL batches into a materialized storage segment. Historically only the embedded
//! path worked; the server path relied on a never-populated `sst_storages` registry
//! plus a metadata-less coordinator and silently materialized nothing (TD-163). This
//! module is the single, shared materialization core both call, so the proven
//! recipe lives in exactly one place:
//!
//! ```text
//! resolve engine from catalog metadata → StorageFormatFactory::create_from_proto_async
//! → build collection_config + FlushParameters → engine.flush()
//! → clear flushed batches → delete the now-redundant WAL files.
//! ```
//!
//! It also applies the **A6 storage-write fence** (default-OFF) before the write,
//! so a pod displaced by a lease takeover cannot publish stale data on its way out
//! (CLAUDE.md #16). The caller supplies the resolved per-collection metadata (the
//! server reads the catalog; embedded uses its collection port) and, optionally, a
//! fallback engine to reuse.

use std::sync::{Arc, OnceLock, Weak};

use anyhow::{Context, Result};
use dashmap::DashMap;
use tokio::sync::{Mutex, OnceCell, Semaphore};

#[cfg(feature = "axis")]
use crate::index::AxisManager;

/// Flush-path AXIS projection reap target threaded through the flush entry points.
/// `Option<&AxisManager>` when AXIS is compiled in (the reap runs); `Option<&()>`
/// otherwise — the reap is cfg-gated out, so this is always `None` off. Keeping one
/// alias keeps the entry-point signatures identical across builds so callers don't
/// branch on the feature.
#[cfg(feature = "axis")]
pub(crate) type AxisFlushArg<'a> = Option<&'a AxisManager>;
#[cfg(not(feature = "axis"))]
pub(crate) type AxisFlushArg<'a> = Option<&'a ()>;
use crate::proto::proximadb_v1::{
    Collection, CollectionConfig, StorageAssignment, StorageEngine as ProtoStorageEngine,
};
use crate::storage::memtable::specialized::wal_behavior::WALBehaviorWrapper;
use crate::storage::traits::{FlushParameters, UnifiedStorageFormat};
use crate::storage::write_fence::{
    FenceDecision, StorageWriteFence, evaluate_fence, write_fencing_enabled,
};

/// Resolved materialization inputs for one collection. The caller fills these from
/// its own metadata source (catalog for the server, collection port for embedded).
pub struct CollectionFlushPlan {
    /// Globally unique catalog object identity used for admission, scheduling,
    /// WAL ownership, and cache ownership. This is resolved once at the catalog
    /// boundary and remains native in the in-memory plan.
    pub collection_object_id: crate::core::stable_id::CollectionObjectId,
    /// Mutable display name used only to populate the transitional engine
    /// configuration DTO and human-readable diagnostics.
    pub collection_name: String,
    /// Base storage location (`file://…` / `s3://…`) for this collection.
    pub base_location: String,
    /// Immutable L2 addressing composite used only for typed object-store path
    /// composition. Admission and WAL ownership remain keyed by the L1
    /// [`CollectionObjectId`](crate::core::stable_id::CollectionObjectId).
    pub collection_identity: Option<crate::core::stable_id::CollectionIdentity>,
    /// Proto `StorageEngine` discriminant for the collection's configured engine.
    pub engine_type: i32,
    /// Vector dimension (required by VIPER/NOVA flush).
    pub dimension: u32,
    /// Owning tenant, when known — the A6 fence input. `None` ⇒ fence fails open.
    pub tenant_id: Option<String>,
    /// TD-FPRUNE-1 M1: the collection's declared filterable columns, carried from
    /// the catalog `meta` so the flush can shred them into typed user-columns
    /// (footer min/max + bloom + self-describing field-map). The transitional
    /// `collection_config` DTO the materializer rebuilds is otherwise minimal
    /// (name/engine/dimension only) and drops these — so footer filter-pruning
    /// stayed dark end-to-end. Empty ⇒ no shredding (byte-identical).
    pub filterable_columns: Vec<crate::proto::proximadb_v1::FilterableColumnSpec>,
    /// TD-FPRUNE-1 M1: whether this is a ProximaRecord (document) collection —
    /// the gate for shredding (mirrors `collection_mapping`'s precondition).
    pub enable_proxima_record: Option<bool>,
    /// TD-READFID-1: the collection's declared canonical embedding precision
    /// (proto `EmbeddingPrecision` discriminant). The flush resolves the exact-f32
    /// tier from it; without it the transitional DTO below dropped the declaration
    /// and an `fp32` collection flushed only lossy RaBitQ/SQ8 columns.
    pub canonical_embedding_precision: Option<i32>,
}

/// Build a [`CollectionFlushPlan`] from a collection's catalog metadata. The single
/// recipe shared by the inline size-trigger (`spawn_inline_flush`), the periodic
/// `AutoFlushDriver`, and the shutdown flush — only *when* to fire differs across the
/// three triggers, not *what* they materialize.
pub fn flush_plan_from_collection_meta(
    meta: &crate::proto::proximadb_v1::Collection,
) -> Result<CollectionFlushPlan> {
    let config = meta.config.as_ref();
    let assignment = meta.storage_assignment.as_ref();
    let engine_type = assignment
        .map(|a| a.engine)
        .or_else(|| config.and_then(|c| c.storage_engine))
        .unwrap_or(crate::proto::proximadb_v1::StorageEngine::Sst as i32);
    let dimension = config.map(|c| c.dimension).unwrap_or(0);
    let base_location = assignment
        .map(|a| a.base_location.clone())
        .unwrap_or_default();
    let collection_identity = match assignment.map(|assignment| {
        (
            assignment.typed_account_id,
            assignment.typed_namespace_id,
            assignment.typed_collection_id,
        )
    }) {
        None | Some((None, None, None)) => None,
        Some((Some(_), Some(_), Some(_))) => Some(
            crate::storage::trait_components::path_resolver::typed_identity_from_storage_assignment(
                assignment,
            )
            .context("catalog storage assignment has an out-of-range typed identity")?,
        ),
        Some(_) => {
            anyhow::bail!(
                "catalog storage assignment has an incomplete typed account/namespace/collection identity"
            )
        }
    };
    let tenant_id = proximadb_tenant::tenant_id_of(meta);
    let collection_object_id = meta.id.parse().with_context(|| {
        format!(
            "catalog collection '{}' has a non-numeric object identity",
            meta.id
        )
    })?;
    Ok(CollectionFlushPlan {
        collection_object_id,
        collection_name: config
            .map(|config| config.name.clone())
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| collection_object_id.to_string()),
        base_location,
        collection_identity,
        engine_type,
        dimension,
        tenant_id,
        // TD-FPRUNE-1 M1: carry the declared filterable columns from the full
        // catalog meta so the flush can shred them (the reconstructed DTO below
        // would otherwise drop them).
        filterable_columns: config
            .map(|c| c.filterable_columns.clone())
            .unwrap_or_default(),
        enable_proxima_record: config.and_then(|c| c.enable_proxima_record),
        canonical_embedding_precision: config.and_then(|c| c.canonical_embedding_precision),
    })
}

/// What a single collection's materialization produced.
#[derive(Debug)]
pub struct CollectionFlushOutcome {
    /// Records submitted to the engine (post tombstone-filter) — the count used
    /// for collection stats / optimizer row estimates.
    pub vectors_submitted: u64,
    /// Records the engine reported written (`FlushResult::entries_flushed`) — the
    /// count used for flush summaries.
    pub entries_flushed: u64,
    pub bytes: u64,
}

/// Process-wide owner of flush engines, per-collection serialization, and
/// bounded cross-collection admission (ADR-081).
struct FlushExecutionCoordinator {
    collection_gates: DashMap<crate::core::stable_id::CollectionIdentity, Weak<Mutex<()>>>,
    engines: DashMap<i32, Arc<OnceCell<Arc<dyn UnifiedStorageFormat>>>>,
    permits: Arc<Semaphore>,
}

impl FlushExecutionCoordinator {
    fn new(max_parallel_flushes: usize) -> Self {
        Self {
            collection_gates: DashMap::new(),
            engines: DashMap::new(),
            permits: Arc::new(Semaphore::new(max_parallel_flushes.max(1))),
        }
    }

    fn collection_gate(&self, plan: &CollectionFlushPlan) -> Arc<Mutex<()>> {
        let key = plan.collection_identity.unwrap_or_default();
        if self.collection_gates.len() > 4_096 {
            self.collection_gates
                .retain(|_, gate| gate.strong_count() > 0);
        }
        match self.collection_gates.entry(key) {
            dashmap::mapref::entry::Entry::Occupied(mut entry) => {
                if let Some(gate) = entry.get().upgrade() {
                    gate
                } else {
                    let gate = Arc::new(Mutex::new(()));
                    entry.insert(Arc::downgrade(&gate));
                    gate
                }
            }
            dashmap::mapref::entry::Entry::Vacant(entry) => {
                let gate = Arc::new(Mutex::new(()));
                entry.insert(Arc::downgrade(&gate));
                gate
            }
        }
    }

    async fn engine_for(
        &self,
        engine_type: ProtoStorageEngine,
        fallback_engine: Option<Arc<dyn UnifiedStorageFormat>>,
    ) -> Result<Arc<dyn UnifiedStorageFormat>> {
        let key = engine_type as i32;
        let cell = self
            .engines
            .entry(key)
            .or_insert_with(|| Arc::new(OnceCell::new()))
            .clone();
        let engine = cell
            .get_or_try_init(|| async move {
                // Route to the collection's DECLARED engine via the factory.
                // The fallback engine is a create-FAILURE fallback only: the
                // previous `if let Some(engine) = fallback_engine` returned it
                // unconditionally, so any caller supplying one (embedded mode
                // passes the unified SST engine) flushed EVERY collection
                // through SST regardless of its declared engine — engine/read
                // mismatch: a HELIX collection flushed to SST files was then
                // searched by HelixEngine, which found zero .helix SSTables
                // and returned empty results (the embedded 1536-d failure).
                match crate::storage::engines::factory::StorageFormatFactory::create_from_proto_async(
                    engine_type,
                )
                .await
                {
                    Ok(engine) => Ok(engine),
                    Err(create_err) => {
                        if let Some(engine) = fallback_engine {
                            tracing::warn!(
                                engine = engine_type.as_str_name(),
                                error = %create_err,
                                "canonical flush engine construction failed;                                  falling back to the unified engine"
                            );
                            return Ok(engine);
                        }
                        Err(create_err).with_context(|| {
                            format!(
                                "failed to construct canonical {} flush engine",
                                engine_type.as_str_name()
                            )
                        })
                    }
                }
            })
            .await?;
        Ok(engine.clone())
    }
}

static FLUSH_EXECUTION_COORDINATOR: OnceLock<FlushExecutionCoordinator> = OnceLock::new();

pub(crate) fn default_parallel_flushes() -> usize {
    crate::core::config::SstConfig::default()
        .background_thread_count
        .max(1) as usize
}

fn flush_execution_coordinator() -> &'static FlushExecutionCoordinator {
    FLUSH_EXECUTION_COORDINATOR
        .get_or_init(|| FlushExecutionCoordinator::new(default_parallel_flushes()))
}

/// Configure the process-wide flush concurrency before the first materializer
/// runs. Production calls this from `StorageEngine::start` using the existing
/// SST background worker setting; tests may rely on the bounded default.
pub fn configure_flush_admission(max_parallel_flushes: usize) {
    if FLUSH_EXECUTION_COORDINATOR
        .set(FlushExecutionCoordinator::new(max_parallel_flushes))
        .is_err()
    {
        tracing::debug!(
            requested = max_parallel_flushes,
            "flush admission coordinator already initialized; retaining its original limit"
        );
    }
}

/// Seed the canonical engine instance for a storage format.
///
/// The composition root uses this for SST so flush and compaction share the
/// configured worker pool. Other formats are constructed lazily once by the
/// coordinator until their composition roots provide an instance.
pub async fn register_flush_engine(
    engine_type: ProtoStorageEngine,
    engine: Arc<dyn UnifiedStorageFormat>,
) {
    flush_execution_coordinator()
        .register_engine(engine_type, engine)
        .await;
}

impl FlushExecutionCoordinator {
    /// Seed the canonical engine instance for a storage format on THIS
    /// coordinator (the free function does the same for the process-wide
    /// one). Composition roots use it to share configured instances across
    /// flush and compaction; tests use it to inject recording engines — the
    /// flush-time `fallback_engine` is a create-FAILURE fallback, not an
    /// injection point.
    pub async fn register_engine(
        &self,
        engine_type: ProtoStorageEngine,
        engine: Arc<dyn UnifiedStorageFormat>,
    ) {
        let cell = self
            .engines
            .entry(engine_type as i32)
            .or_insert_with(|| Arc::new(OnceCell::new()))
            .clone();
        let requested = engine.clone();
        let resolved = cell.get_or_init(|| async move { engine }).await;
        if !Arc::ptr_eq(resolved, &requested) {
            tracing::warn!(
                engine = engine_type.as_str_name(),
                "flush engine was resolved before registration; retaining the existing canonical instance"
            );
        }
    }
}

struct FlushAdmissionTicket;

impl FlushAdmissionTicket {
    fn new(wait: std::time::Duration) -> Self {
        crate::metrics::wal_flush_metrics::record_admission_wait(wait.as_secs_f64());
        crate::metrics::wal_flush_metrics::inc_admitted_in_flight();
        Self
    }
}

impl Drop for FlushAdmissionTicket {
    fn drop(&mut self) {
        crate::metrics::wal_flush_metrics::dec_admitted_in_flight();
    }
}

/// Cancellation-safe metrics ownership for an exact source claim.
///
/// The WAL claim itself owns correctness. This companion keeps the observable
/// gauge/counter consistent when a task is aborted at any await point.
struct FlushClaimTicket {
    collection_id: String,
    batch_count: u64,
    completed: bool,
}

impl FlushClaimTicket {
    fn new(collection_id: &str, batch_count: usize) -> Self {
        crate::metrics::wal_flush_metrics::set_claimed_batches(collection_id, batch_count as i64);
        Self {
            collection_id: collection_id.to_string(),
            batch_count: batch_count as u64,
            completed: false,
        }
    }

    fn complete(&mut self) {
        crate::metrics::wal_flush_metrics::set_claimed_batches(&self.collection_id, 0);
        self.completed = true;
    }
}

impl Drop for FlushClaimTicket {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        crate::metrics::wal_flush_metrics::record_claim_rollback(
            &self.collection_id,
            self.batch_count,
        );
        crate::metrics::wal_flush_metrics::set_claimed_batches(&self.collection_id, 0);
    }
}

/// Materialize one collection's unflushed WAL batches to its configured storage
/// engine. Returns `Ok(None)` when there is nothing to flush, `Err` when the A6
/// fence rejects the write or the storage flush fails.
///
/// A6 fence (default-OFF; `PROXIMADB_WRITE_FENCING=1`) is checked first — before
/// any record read or storage write — so a fenced-out pod is rejected at the
/// boundary. Fail-open when the gate is off, no fence is wired, or the tenant is
/// unknown.
///
/// `free_wal` controls whether the source WAL is reclaimed after the segment is
/// written. Post-TD-165 (cold-read recall fixed) **both callers pass `true`**: clear
/// the flushed batches and delete the WAL files (no 2× WAL+segment overhead), so the
/// materialized segment — not a WAL replay — is the durable restart-recall source.
/// The flag is retained (rather than inlined) as an explicit safety valve: passing
/// `false` keeps the WAL so recovery replays it into the FP32 memtable (exact recall
/// that bypasses the cold SST path) — the escape hatch if a cold-read regression
/// ever resurfaces. No caller uses `false` today.
///
/// `axis_index_manager` is the AxisManager whose in-memory `collection_vectors`
/// projection must be reaped after the segment is durable (TD-FLUSH-1). Inserted
/// vectors live in two places pre-flush — the WAL batches AND this projection — and
/// before TD-FLUSH-1 only the WAL was cleared, so the projection grew unbounded
/// (OOM) and warm GETs served from the stale copy (masked cold-read bugs). Pass
/// `None` only when no AxisManager owns this collection (recovery / tests); the
/// projection is then left as-is (caller's responsibility). Failure to reap is
/// logged and non-fatal — the durable segment is the source of truth post-flush.
pub async fn materialize_collection(
    write_buffer: &Arc<WALBehaviorWrapper>,
    plan: &CollectionFlushPlan,
    fence: Option<&Arc<dyn StorageWriteFence>>,
    fallback_engine: Option<Arc<dyn UnifiedStorageFormat>>,
    free_wal: bool,
    axis_index_manager: AxisFlushArg<'_>,
) -> Result<Option<CollectionFlushOutcome>> {
    materialize_collection_with_coordinator_mode(
        flush_execution_coordinator(),
        write_buffer,
        plan,
        fence,
        fallback_engine,
        free_wal,
        axis_index_manager,
        CollectionAdmission::Wait,
    )
    .await
}

/// Inline-trigger variant that collapses a burst when this collection already
/// has a materialization in progress. The same canonical coordinator owns the
/// decision; no trigger-local semaphore or scheduling registry is involved.
pub async fn materialize_collection_if_idle(
    write_buffer: &Arc<WALBehaviorWrapper>,
    plan: &CollectionFlushPlan,
    fence: Option<&Arc<dyn StorageWriteFence>>,
    fallback_engine: Option<Arc<dyn UnifiedStorageFormat>>,
    free_wal: bool,
    axis_index_manager: AxisFlushArg<'_>,
) -> Result<Option<CollectionFlushOutcome>> {
    materialize_collection_with_coordinator_mode(
        flush_execution_coordinator(),
        write_buffer,
        plan,
        fence,
        fallback_engine,
        free_wal,
        axis_index_manager,
        CollectionAdmission::SkipIfBusy,
    )
    .await
}

#[derive(Clone, Copy)]
enum CollectionAdmission {
    Wait,
    SkipIfBusy,
}

#[cfg(test)]
async fn materialize_collection_with_coordinator(
    coordinator: &FlushExecutionCoordinator,
    write_buffer: &Arc<WALBehaviorWrapper>,
    plan: &CollectionFlushPlan,
    fence: Option<&Arc<dyn StorageWriteFence>>,
    fallback_engine: Option<Arc<dyn UnifiedStorageFormat>>,
    free_wal: bool,
    axis_index_manager: AxisFlushArg<'_>,
) -> Result<Option<CollectionFlushOutcome>> {
    materialize_collection_with_coordinator_mode(
        coordinator,
        write_buffer,
        plan,
        fence,
        fallback_engine,
        free_wal,
        axis_index_manager,
        CollectionAdmission::Wait,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn materialize_collection_with_coordinator_mode(
    coordinator: &FlushExecutionCoordinator,
    write_buffer: &Arc<WALBehaviorWrapper>,
    plan: &CollectionFlushPlan,
    fence: Option<&Arc<dyn StorageWriteFence>>,
    fallback_engine: Option<Arc<dyn UnifiedStorageFormat>>,
    free_wal: bool,
    axis_index_manager: AxisFlushArg<'_>,
    collection_admission: CollectionAdmission,
) -> Result<Option<CollectionFlushOutcome>> {
    let collection_id_text = plan.collection_object_id.to_string();

    // A6 storage-write fence — reject a displaced pod before touching storage.
    if write_fencing_enabled() {
        let now_ms = chrono::Utc::now().timestamp_millis();
        if evaluate_fence(
            true,
            fence,
            plan.tenant_id.as_deref(),
            &collection_id_text,
            now_ms,
        )
        .await
            == FenceDecision::Fenced
        {
            tracing::warn!(
                target: "proximadb.fence",
                tenant_id = plan.tenant_id.as_deref().unwrap_or("<unknown>"),
                collection_object_id = plan.collection_object_id,
                "🛡️ A6 fence: this pod is fenced out; rejecting stale-pod flush before storage write"
            );
            return Err(anyhow::anyhow!(
                "A6 storage-write fence: pod is fenced out of collection '{}' (tenant '{}') — a live lease is held by another pod",
                plan.collection_object_id,
                plan.tenant_id.as_deref().unwrap_or("<unknown>")
            ));
        }
    }

    // ADR-081 D1: serialize only this collection. Contenders wait without
    // consuming a global worker permit, so a hot collection cannot starve
    // independent collections.
    let collection_gate = coordinator.collection_gate(plan);
    let _collection_guard = match collection_admission {
        CollectionAdmission::Wait => collection_gate.lock_owned().await,
        CollectionAdmission::SkipIfBusy => match collection_gate.try_lock_owned() {
            Ok(guard) => guard,
            Err(_) => {
                tracing::trace!(
                    collection_object_id = plan.collection_object_id,
                    "inline flush collapsed into the collection's admitted materialization"
                );
                return Ok(None);
            }
        },
    };
    if write_buffer
        .unflushed_batch_count(&collection_id_text)
        .await
        == 0
    {
        return Ok(None);
    }

    let permit_wait = std::time::Instant::now();
    let _permit = coordinator
        .permits
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| anyhow::anyhow!("flush admission coordinator is closed"))?;
    let _admission_ticket = FlushAdmissionTicket::new(permit_wait.elapsed());

    // Resolve one canonical engine per configured storage format. An
    // unrecognized engine id or construction failure is fail-closed; silently
    // substituting SST would write bytes in the wrong format.
    let proto_engine = ProtoStorageEngine::try_from(plan.engine_type).map_err(|_| {
        anyhow::anyhow!(
            "flush: collection '{}' declares an unrecognized storage engine id {} — \
             refusing to substitute a default engine",
            plan.collection_object_id,
            plan.engine_type
        )
    })?;
    let engine = coordinator
        .engine_for(proto_engine, fallback_engine)
        .await?;

    // Collection config carries engine + dimension + the on-disk layout path so the
    // flush writes into the same directory recovery reads from.
    let (typed_account_id, typed_namespace_id, typed_collection_id) = match plan.collection_identity
    {
        Some(identity) => (
            Some(identity.account_id),
            Some(u32::from(identity.namespace_id)),
            Some(identity.collection_id),
        ),
        None => (None, None, None),
    };
    let collection_config = Collection {
        id: collection_id_text.clone(),
        storage_assignment: Some(StorageAssignment {
            base_location: plan.base_location.clone(),
            engine: plan.engine_type,
            typed_account_id,
            typed_namespace_id,
            typed_collection_id,
            ..Default::default()
        }),
        config: Some(CollectionConfig {
            name: plan.collection_name.clone(),
            storage_engine: Some(plan.engine_type),
            dimension: plan.dimension,
            // TD-FPRUNE-1 M1: propagate the filterable columns + proxima-record
            // flag so `resolve_flush_shred_spec` can shred declared tags into
            // typed user-columns (footer filter-pruning). These were dropped by
            // this transitional DTO, keeping footer-pruning dark end-to-end.
            filterable_columns: plan.filterable_columns.clone(),
            enable_proxima_record: plan.enable_proxima_record,
            canonical_embedding_precision: plan.canonical_embedding_precision,
            ..Default::default()
        }),
        ..Default::default()
    };

    // ADR-081 D3: engine admission occurs before a source claim and before
    // cloning any ProximaRecord. The preflight payload is metadata-only.
    let preflight_params = FlushParameters {
        collection_id: Some(plan.collection_object_id.to_string()),
        force: true,
        synchronous: true,
        collection_config: Some(collection_config.clone()),
        ..Default::default()
    };
    if let Err(error) = engine.preflight_flush(&preflight_params).await {
        crate::metrics::wal_flush_metrics::record_admission(&collection_id_text, false);
        return Err(error);
    }
    crate::metrics::wal_flush_metrics::record_admission(&collection_id_text, true);

    // Exact source ownership is acquired only after engine admission. The
    // claim's Drop releases every batch on error or future cancellation.
    let Some(mut claim) = write_buffer
        .claim_segment_materializable_batches(&collection_id_text)
        .await?
    else {
        tracing::debug!(
            collection_object_id = plan.collection_object_id,
            "flush: no segment-materializable WAL batches; retaining deferred records in WAL"
        );
        return Ok(None);
    };
    let mut claim_ticket = FlushClaimTicket::new(&collection_id_text, claim.batch_ids().len());

    // Vector-proportional work starts here, inside the admitted job. The claim
    // contains only vector-only batches, so exact batch retirement cannot drop
    // a tombstone that the current segment writer cannot represent.
    let vector_records: Vec<proximadb_records::ProximaRecord> = claim
        .batches()
        .iter()
        .flat_map(|batch| batch.vector_records.iter().cloned())
        .collect();
    let vector_count = vector_records.len() as u64;
    let flush_params = FlushParameters {
        collection_id: Some(plan.collection_object_id.to_string()),
        force: true,
        synchronous: true,
        vector_records,
        batch_ids: claim.batches().iter().map(|batch| batch.batch_id).collect(),
        collection_config: Some(collection_config),
        ..Default::default()
    };

    // The single trait `flush()` funnel (validation + post-processing + do_flush).
    let result = match engine.flush(flush_params).await {
        Ok(result) => result,
        Err(error) => return Err(error),
    };
    if !result.success {
        anyhow::bail!(
            "{} flush returned an unsuccessful publication result for '{}'; exact WAL claim retained for retry",
            engine.engine_name(),
            plan.collection_object_id
        );
    }
    let bytes = result.bytes_written.unwrap_or(0);
    let entries_flushed = result.entries_flushed.unwrap_or(0);

    // Per-tenant object-store WRITE metering (co-design KIU/KSU write side). This is the single
    // tenant-aware flush funnel through which every engine's memtable→object-store segment write
    // passes, so metering here — rather than at each per-engine PAX/segment write call site —
    // keeps the emission DRY and engine-neutral (SST/HELIX/NOVA/VIPER all funnel through
    // `engine.flush()` above). It mirrors the read side's `record_object_store_op(.., "fetch_pax")`.
    //
    // OSS boundary: this emits NEUTRAL usage telemetry only (byte + op counts attributed by
    // tenant) — no pricing, no $/unit weights, no cloud-cost constants. Applying policy weights to
    // these dimensions is the commercial anvaiops control plane's concern, never OSS code.
    //
    // Tier "hot" is the write-time default: a fresh flush writes hot, and an unset
    // `storage_class` ⇒ the account/backend default. A zero-byte engine
    // publication records nothing.
    if bytes > 0 {
        let tenant = plan.tenant_id.as_deref();
        crate::metrics::consumption_metrics::record_object_store_op(tenant, "flush_write");
        crate::metrics::consumption_metrics::record_object_store_write_bytes_by_tier(
            tenant.unwrap_or(""),
            "hot",
            bytes,
        );
    }

    // WAL cleanup after successful publication retires only the exact claim.
    // Batches appended while the segment encoded remain unflushed.
    if free_wal {
        let batch_id_strings = claim.batch_ids().to_vec();
        if let Err(error) = write_buffer.complete_flush_claim(&mut claim).await {
            return Err(error).with_context(|| {
                format!(
                    "flush segment published but exact WAL retirement failed for '{}'",
                    plan.collection_object_id
                )
            });
        }
        claim_ticket.complete();
        if !batch_id_strings.is_empty()
            && let Err(e) = crate::storage::persistence::write_ahead_log::manifest::mark_flushed_and_delete_files(
                &batch_id_strings,
            )
            .await
        {
            tracing::warn!(
                "flush: failed to delete WAL files for '{}': {}",
                plan.collection_object_id,
                e
            );
        }
    } else {
        tracing::debug!(
            "flush: kept WAL for '{}' (free_wal=false) — recovery replays it for exact recall",
            plan.collection_object_id
        );
    }

    // TD-FLUSH-1: reap the AxisManager's in-memory `collection_vectors` projection
    // for this collection now that the durable segment is written. The WAL cleanup
    // above only clears the memtable batches — inserted vectors ALSO live in the
    // AxisManager projection (used for warm reads + cold-index hydration), and
    // leaving it resident caused unbounded memory growth (OOM) and masked cold-read
    // bugs (warm GETs served from the stale copy). Non-fatal: the segment is the
    // source of truth post-flush, so a reap failure is logged, not fatal.
    // Do not clear AXIS records belonging to batches that arrived while this
    // flush encoded. Once the WAL delta reaches empty, the whole projection is
    // safe to reap.
    let collection_wal_empty = write_buffer
        .get_unflushed_batches(&collection_id_text)
        .await
        .map(|batches| batches.is_empty())
        .unwrap_or(false);
    #[cfg(feature = "axis")]
    if collection_wal_empty
        && let Some(axis) = axis_index_manager
        && let Err(e) = axis.clear_collection_vectors(&collection_id_text).await
    {
        tracing::warn!(
            "flush: failed to clear in-memory collection_vectors for '{}': {}",
            plan.collection_object_id,
            e
        );
    }
    // When AXIS is compiled out there is no in-memory projection to reap; the
    // durable segment written above is the sole source of truth.
    #[cfg(not(feature = "axis"))]
    let _ = (collection_wal_empty, axis_index_manager);

    Ok(Some(CollectionFlushOutcome {
        vectors_submitted: vector_count,
        entries_flushed,
        bytes,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::memtable::core::MemtableConfig;
    use crate::storage::memtable::specialized::wal_behavior::WALVectorBatch;
    use crate::storage::persistence::write_ahead_log::BatchId;
    use crate::storage::traits::{
        CompactionParameters, CompactionResult, FlushResult, StorageEngineStrategy,
        StorageQueryContext,
    };
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use tokio::sync::{Barrier, Notify};

    struct ActiveFlush<'a>(&'a AtomicUsize);

    impl Drop for ActiveFlush<'_> {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }

    struct RecordingEngine {
        preflight_allowed: AtomicBool,
        fail_flush: AtomicBool,
        successful_result: AtomicBool,
        preflight_calls: AtomicUsize,
        flush_calls: AtomicUsize,
        active_flushes: AtomicUsize,
        max_active_flushes: AtomicUsize,
        rendezvous: Option<Arc<Barrier>>,
        wait_for_release: bool,
        entered_flush: Notify,
        release_flush: Notify,
        last_storage_assignment: Mutex<Option<StorageAssignment>>,
        last_precision: Mutex<Option<i32>>,
    }

    impl RecordingEngine {
        fn new() -> Self {
            Self {
                preflight_allowed: AtomicBool::new(true),
                fail_flush: AtomicBool::new(false),
                successful_result: AtomicBool::new(true),
                preflight_calls: AtomicUsize::new(0),
                flush_calls: AtomicUsize::new(0),
                active_flushes: AtomicUsize::new(0),
                max_active_flushes: AtomicUsize::new(0),
                rendezvous: None,
                wait_for_release: false,
                entered_flush: Notify::new(),
                release_flush: Notify::new(),
                last_storage_assignment: Mutex::new(None),
                last_precision: Mutex::new(None),
            }
        }

        fn with_rendezvous(parties: usize) -> Self {
            Self {
                rendezvous: Some(Arc::new(Barrier::new(parties))),
                ..Self::new()
            }
        }

        fn blocking() -> Self {
            Self {
                wait_for_release: true,
                ..Self::new()
            }
        }
    }

    #[async_trait::async_trait]
    impl UnifiedStorageFormat for RecordingEngine {
        fn engine_name(&self) -> &'static str {
            "recording"
        }

        fn engine_version(&self) -> &'static str {
            "test"
        }

        fn strategy(&self) -> StorageEngineStrategy {
            StorageEngineStrategy::Sst
        }

        async fn preflight_flush(&self, _params: &FlushParameters) -> Result<()> {
            self.preflight_calls.fetch_add(1, Ordering::SeqCst);
            if self.preflight_allowed.load(Ordering::SeqCst) {
                Ok(())
            } else {
                anyhow::bail!("test admission rejection")
            }
        }

        async fn do_flush(&self, params: &FlushParameters) -> Result<FlushResult> {
            self.flush_calls.fetch_add(1, Ordering::SeqCst);
            *self
                .last_storage_assignment
                .lock()
                .expect("test storage-assignment lock") = params
                .collection_config
                .as_ref()
                .and_then(|collection| collection.storage_assignment.clone());
            *self.last_precision.lock().expect("test precision lock") = params
                .collection_config
                .as_ref()
                .and_then(|collection| collection.config.as_ref())
                .and_then(|config| config.canonical_embedding_precision);
            let active = self.active_flushes.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_active_flushes.fetch_max(active, Ordering::SeqCst);
            let _active_guard = ActiveFlush(&self.active_flushes);

            if let Some(rendezvous) = &self.rendezvous {
                rendezvous.wait().await;
            }
            if self.wait_for_release {
                self.entered_flush.notify_one();
                self.release_flush.notified().await;
            }
            if self.fail_flush.load(Ordering::SeqCst) {
                anyhow::bail!("test flush failure")
            }

            Ok(FlushResult {
                success: self.successful_result.load(Ordering::SeqCst),
                entries_flushed: Some(params.vector_records.len() as u64),
                bytes_written: Some(params.vector_records.len() as u64 * 32),
                ..Default::default()
            })
        }

        async fn do_compact(&self, _params: &CompactionParameters) -> Result<CompactionResult> {
            Ok(CompactionResult::default())
        }

        async fn collect_engine_metrics(&self) -> Result<HashMap<String, serde_json::Value>> {
            Ok(HashMap::new())
        }

        async fn vector_by_id(
            &self,
            _collection_id: &str,
            _base_path: &str,
            _vector_id: &str,
        ) -> Result<Option<proximadb_records::ProximaRecord>> {
            Ok(None)
        }

        async fn search_vectors_unified(
            &self,
            _ctx: &StorageQueryContext,
        ) -> Result<Vec<proximadb_search_types::results::OptimizedSearchRecord>> {
            Ok(Vec::new())
        }
    }

    fn plan(collection_object_id: u64) -> CollectionFlushPlan {
        CollectionFlushPlan {
            collection_object_id,
            collection_name: format!("collection-{collection_object_id}"),
            base_location: "file:///tmp/adr081-tests".to_string(),
            // Distinct composite per plan (ADR-0083 rev2: the flush gate keys on this).
            collection_identity: Some(crate::core::stable_id::CollectionIdentity {
                account_id: 1,
                namespace_id: 1,
                collection_id: collection_object_id as u32,
            }),
            engine_type: ProtoStorageEngine::Sst as i32,
            dimension: 2,
            tenant_id: Some("test-tenant".to_string()),
            filterable_columns: Vec::new(),
            enable_proxima_record: None,
            canonical_embedding_precision: None,
        }
    }

    fn record(oid: &str) -> proximadb_records::ProximaRecord {
        proximadb_records::ProximaRecord {
            oid: oid.to_string(),
            embeddings: vec![proximadb_records::EmbeddingCell {
                model_id: "test".to_string(),
                modality: "dense_vector".to_string(),
                dim: 2,
                values: proximadb_records::EmbeddingValues::Fp32(vec![1.0, 2.0]),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn tombstone(oid: &str) -> proximadb_records::ProximaRecord {
        proximadb_records::ProximaRecord {
            oid: oid.to_string(),
            valid_to_ns: Some(0),
            origin: Some("delete".to_string()),
            ..Default::default()
        }
    }

    async fn add_records_batch(
        write_buffer: &WALBehaviorWrapper,
        collection_object_id: u64,
        records: Vec<proximadb_records::ProximaRecord>,
    ) {
        write_buffer
            .add_vector_batch(
                &collection_object_id.to_string(),
                WALVectorBatch {
                    batch_id: BatchId::new(),
                    total_size_bytes: WALVectorBatch::estimate_records_size(&records),
                    vector_records: Arc::new(records),
                    timestamp: std::time::SystemTime::now(),
                    is_flushed: false,
                    metadata_bloom_filter: None,
                },
            )
            .await
            .expect("test batch must enter the WAL");
    }

    async fn add_batch(write_buffer: &WALBehaviorWrapper, collection_object_id: u64, oid: &str) {
        add_records_batch(write_buffer, collection_object_id, vec![record(oid)]).await;
    }

    #[tokio::test]
    async fn adr081_rejection_happens_before_batch_claim_or_flush() {
        let coordinator = FlushExecutionCoordinator::new(1);
        let write_buffer = Arc::new(WALBehaviorWrapper::new(MemtableConfig::default()));
        add_batch(&write_buffer, 101, "v0").await;
        let engine = Arc::new(RecordingEngine::new());
        engine.preflight_allowed.store(false, Ordering::SeqCst);

        coordinator
            .register_engine(ProtoStorageEngine::Sst, engine.clone())
            .await;
        coordinator
            .register_engine(ProtoStorageEngine::Sst, engine.clone())
            .await;
        let error = materialize_collection_with_coordinator(
            &coordinator,
            &write_buffer,
            &plan(101),
            None,
            None,
            true,
            None,
        )
        .await
        .expect_err("preflight must reject the job");

        assert!(error.to_string().contains("admission rejection"));
        assert_eq!(engine.preflight_calls.load(Ordering::SeqCst), 1);
        assert_eq!(engine.flush_calls.load(Ordering::SeqCst), 0);
        assert_eq!(write_buffer.claimed_flush_batch_count("101").unwrap(), 0);
        assert_eq!(
            write_buffer
                .get_unflushed_batches("101")
                .await
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn adr081_zero_byte_source_batch_is_not_mistaken_for_empty() {
        let coordinator = FlushExecutionCoordinator::new(1);
        let write_buffer = Arc::new(WALBehaviorWrapper::new(MemtableConfig::default()));
        let records = vec![record("legacy-zero")];
        write_buffer
            .add_vector_batch(
                "102",
                WALVectorBatch {
                    batch_id: BatchId::new(),
                    vector_records: Arc::new(records),
                    timestamp: std::time::SystemTime::now(),
                    total_size_bytes: 0,
                    is_flushed: false,
                    metadata_bloom_filter: None,
                },
            )
            .await
            .expect("legacy zero-byte batch must enter the WAL");
        let engine = Arc::new(RecordingEngine::new());

        coordinator
            .register_engine(ProtoStorageEngine::Sst, engine.clone())
            .await;
        coordinator
            .register_engine(ProtoStorageEngine::Sst, engine.clone())
            .await;
        let outcome = materialize_collection_with_coordinator(
            &coordinator,
            &write_buffer,
            &plan(102),
            None,
            None,
            true,
            None,
        )
        .await
        .expect("zero-byte source still contains a record");

        assert!(outcome.is_some());
        assert_eq!(engine.flush_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn tombstone_only_flush_is_retained_without_calling_segment_writer() {
        let coordinator = FlushExecutionCoordinator::new(1);
        let write_buffer = Arc::new(WALBehaviorWrapper::new(MemtableConfig::default()));
        add_records_batch(&write_buffer, 112, vec![tombstone("dead")]).await;
        let engine = Arc::new(RecordingEngine::new());

        coordinator
            .register_engine(ProtoStorageEngine::Sst, engine.clone())
            .await;
        coordinator
            .register_engine(ProtoStorageEngine::Sst, engine.clone())
            .await;
        let outcome = materialize_collection_with_coordinator(
            &coordinator,
            &write_buffer,
            &plan(112),
            None,
            None,
            true,
            None,
        )
        .await
        .expect("tombstone-only source remains durable in WAL");

        assert!(outcome.is_none());
        assert_eq!(engine.preflight_calls.load(Ordering::SeqCst), 1);
        assert_eq!(engine.flush_calls.load(Ordering::SeqCst), 0);
        assert_eq!(write_buffer.claimed_flush_batch_count("112").unwrap(), 0);
        assert_eq!(
            write_buffer
                .get_unflushed_batches("112")
                .await
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn vector_flush_retires_only_vector_batches_and_preserves_tombstones() {
        let coordinator = FlushExecutionCoordinator::new(1);
        let write_buffer = Arc::new(WALBehaviorWrapper::new(MemtableConfig::default()));
        add_records_batch(&write_buffer, 113, vec![record("live")]).await;
        add_records_batch(&write_buffer, 113, vec![tombstone("dead")]).await;
        let engine = Arc::new(RecordingEngine::new());

        coordinator
            .register_engine(ProtoStorageEngine::Sst, engine.clone())
            .await;
        coordinator
            .register_engine(ProtoStorageEngine::Sst, engine.clone())
            .await;
        let outcome = materialize_collection_with_coordinator(
            &coordinator,
            &write_buffer,
            &plan(113),
            None,
            None,
            true,
            None,
        )
        .await
        .expect("vector-only source batch must publish")
        .expect("one vector batch must produce a segment");

        assert_eq!(outcome.vectors_submitted, 1);
        assert_eq!(engine.flush_calls.load(Ordering::SeqCst), 1);
        let remaining = write_buffer.get_unflushed_batches("113").await.unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].vector_records[0].oid, "dead");
    }

    #[tokio::test]
    async fn mixed_vector_tombstone_batch_fails_closed_before_segment_write() {
        let coordinator = FlushExecutionCoordinator::new(1);
        let write_buffer = Arc::new(WALBehaviorWrapper::new(MemtableConfig::default()));
        add_records_batch(&write_buffer, 114, vec![record("live"), tombstone("dead")]).await;
        let engine = Arc::new(RecordingEngine::new());

        coordinator
            .register_engine(ProtoStorageEngine::Sst, engine.clone())
            .await;
        coordinator
            .register_engine(ProtoStorageEngine::Sst, engine.clone())
            .await;
        let error = materialize_collection_with_coordinator(
            &coordinator,
            &write_buffer,
            &plan(114),
            None,
            None,
            true,
            None,
        )
        .await
        .expect_err("mixed batch must not be partially published");

        assert!(error.to_string().contains("mixed vector/tombstone"));
        assert_eq!(engine.preflight_calls.load(Ordering::SeqCst), 1);
        assert_eq!(engine.flush_calls.load(Ordering::SeqCst), 0);
        assert_eq!(write_buffer.claimed_flush_batch_count("114").unwrap(), 0);
        assert_eq!(
            write_buffer
                .get_unflushed_batches("114")
                .await
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn typed_storage_identity_reaches_the_engine_flush_boundary() {
        let coordinator = FlushExecutionCoordinator::new(1);
        let write_buffer = Arc::new(WALBehaviorWrapper::new(MemtableConfig::default()));
        add_batch(&write_buffer, 111, "v0").await;
        let engine = Arc::new(RecordingEngine::new());
        let mut typed_plan = plan(111);
        typed_plan.collection_identity = Some(crate::core::stable_id::CollectionIdentity {
            account_id: 7,
            namespace_id: 11,
            collection_id: 13,
        });

        coordinator
            .register_engine(ProtoStorageEngine::Sst, engine.clone())
            .await;
        materialize_collection_with_coordinator(
            &coordinator,
            &write_buffer,
            &typed_plan,
            None,
            None,
            true,
            None,
        )
        .await
        .expect("typed flush must succeed")
        .expect("typed flush must publish its batch");

        let assignment = engine
            .last_storage_assignment
            .lock()
            .expect("test storage-assignment lock")
            .clone()
            .expect("flush parameters must carry a storage assignment");
        assert_eq!(assignment.typed_account_id, Some(7));
        assert_eq!(assignment.typed_namespace_id, Some(11));
        assert_eq!(assignment.typed_collection_id, Some(13));
    }

    /// TD-READFID-1: the declared canonical precision must survive the catalog
    /// meta -> plan -> transitional engine DTO hop; the SST flush resolves the
    /// exact-f32 tier from it. It used to be dropped here, so an `fp32`
    /// collection flushed lossy-only no matter what the resolver did.
    #[tokio::test]
    async fn canonical_precision_reaches_the_engine_flush_boundary() {
        use crate::proto::proximadb_v1::EmbeddingPrecision;
        let meta = Collection {
            id: "115".to_string(),
            config: Some(CollectionConfig {
                name: "fp32-collection".to_string(),
                dimension: 2,
                storage_engine: Some(ProtoStorageEngine::Sst as i32),
                canonical_embedding_precision: Some(EmbeddingPrecision::Fp32 as i32),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut fp32_plan = flush_plan_from_collection_meta(&meta).expect("valid flush plan");
        assert_eq!(
            fp32_plan.canonical_embedding_precision,
            Some(EmbeddingPrecision::Fp32 as i32)
        );
        fp32_plan.base_location = "file:///tmp/readfid1-tests".to_string();

        let coordinator = FlushExecutionCoordinator::new(1);
        let write_buffer = Arc::new(WALBehaviorWrapper::new(MemtableConfig::default()));
        add_batch(&write_buffer, 115, "v0").await;
        let engine = Arc::new(RecordingEngine::new());
        coordinator
            .register_engine(ProtoStorageEngine::Sst, engine.clone())
            .await;
        materialize_collection_with_coordinator(
            &coordinator,
            &write_buffer,
            &fp32_plan,
            None,
            None,
            true,
            None,
        )
        .await
        .expect("flush must succeed")
        .expect("flush must publish its batch");

        assert_eq!(
            *engine.last_precision.lock().expect("test precision lock"),
            Some(EmbeddingPrecision::Fp32 as i32),
            "the engine flush DTO must carry the declared canonical precision"
        );
    }

    #[test]
    fn flush_plan_preserves_catalog_storage_identity() {
        let mut collection = Collection {
            id: "111".to_string(),
            config: Some(CollectionConfig {
                name: "typed-collection".to_string(),
                dimension: 2,
                storage_engine: Some(ProtoStorageEngine::Sst as i32),
                ..Default::default()
            }),
            storage_assignment: Some(StorageAssignment {
                base_location: "file:///typed".to_string(),
                engine: ProtoStorageEngine::Sst as i32,
                typed_account_id: Some(7),
                typed_namespace_id: Some(11),
                typed_collection_id: Some(13),
                ..Default::default()
            }),
            ..Default::default()
        };

        let plan = flush_plan_from_collection_meta(&collection).expect("valid flush plan");
        assert_eq!(
            plan.collection_identity,
            Some(crate::core::stable_id::CollectionIdentity {
                account_id: 7,
                namespace_id: 11,
                collection_id: 13,
            })
        );

        collection
            .storage_assignment
            .as_mut()
            .expect("storage assignment")
            .typed_namespace_id = None;
        assert!(
            flush_plan_from_collection_meta(&collection)
                .err()
                .expect("partial typed identity must fail closed")
                .to_string()
                .contains("incomplete typed account/namespace/collection identity")
        );

        collection
            .storage_assignment
            .as_mut()
            .expect("storage assignment")
            .typed_namespace_id = Some(u32::from(u16::MAX) + 1);
        assert!(
            flush_plan_from_collection_meta(&collection)
                .err()
                .expect("out-of-range typed identity must fail closed")
                .to_string()
                .contains("out-of-range typed identity")
        );
    }

    #[tokio::test]
    async fn adr081_same_collection_contenders_publish_one_source_set_once() {
        let coordinator = FlushExecutionCoordinator::new(2);
        let write_buffer = Arc::new(WALBehaviorWrapper::new(MemtableConfig::default()));
        add_batch(&write_buffer, 103, "v0").await;
        let engine = Arc::new(RecordingEngine::new());
        let plan = plan(103);

        coordinator
            .register_engine(ProtoStorageEngine::Sst, engine.clone())
            .await;

        let (first, second) = tokio::join!(
            materialize_collection_with_coordinator(
                &coordinator,
                &write_buffer,
                &plan,
                None,
                None,
                true,
                None,
            ),
            materialize_collection_with_coordinator(
                &coordinator,
                &write_buffer,
                &plan,
                None,
                None,
                true,
                None,
            )
        );

        let published = [first.unwrap(), second.unwrap()]
            .into_iter()
            .filter(Option::is_some)
            .count();
        assert_eq!(published, 1);
        assert_eq!(engine.flush_calls.load(Ordering::SeqCst), 1);
        assert!(
            write_buffer
                .get_unflushed_batches("103")
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn adr081_different_collections_flush_in_parallel_within_global_bound() {
        let coordinator = FlushExecutionCoordinator::new(2);
        let write_buffer = Arc::new(WALBehaviorWrapper::new(MemtableConfig::default()));
        add_batch(&write_buffer, 104, "left-v0").await;
        add_batch(&write_buffer, 105, "right-v0").await;
        let engine = Arc::new(RecordingEngine::with_rendezvous(2));
        let left = plan(104);
        let right = plan(105);

        coordinator
            .register_engine(ProtoStorageEngine::Sst, engine.clone())
            .await;

        let (left_result, right_result) =
            tokio::time::timeout(std::time::Duration::from_secs(2), async {
                tokio::join!(
                    materialize_collection_with_coordinator(
                        &coordinator,
                        &write_buffer,
                        &left,
                        None,
                        None,
                        false,
                        None,
                    ),
                    materialize_collection_with_coordinator(
                        &coordinator,
                        &write_buffer,
                        &right,
                        None,
                        None,
                        false,
                        None,
                    )
                )
            })
            .await
            .expect("independent collections must both reach the engine");
        left_result.expect("left flush must succeed");
        right_result.expect("right flush must succeed");

        assert_eq!(engine.flush_calls.load(Ordering::SeqCst), 2);
        assert_eq!(engine.max_active_flushes.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn adr081_failed_flush_releases_claim_for_retry() {
        let coordinator = FlushExecutionCoordinator::new(1);
        let write_buffer = Arc::new(WALBehaviorWrapper::new(MemtableConfig::default()));
        add_batch(&write_buffer, 106, "v0").await;
        let engine = Arc::new(RecordingEngine::new());
        engine.fail_flush.store(true, Ordering::SeqCst);

        coordinator
            .register_engine(ProtoStorageEngine::Sst, engine.clone())
            .await;
        materialize_collection_with_coordinator(
            &coordinator,
            &write_buffer,
            &plan(106),
            None,
            None,
            true,
            None,
        )
        .await
        .expect_err("first publication must fail");
        assert_eq!(write_buffer.claimed_flush_batch_count("106").unwrap(), 0);

        engine.fail_flush.store(false, Ordering::SeqCst);
        let retry = materialize_collection_with_coordinator(
            &coordinator,
            &write_buffer,
            &plan(106),
            None,
            None,
            true,
            None,
        )
        .await
        .expect("released source must be retryable");
        assert!(retry.is_some());
        assert_eq!(engine.flush_calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn adr081_unsuccessful_result_does_not_retire_source() {
        let coordinator = FlushExecutionCoordinator::new(1);
        let write_buffer = Arc::new(WALBehaviorWrapper::new(MemtableConfig::default()));
        add_batch(&write_buffer, 107, "v0").await;
        let engine = Arc::new(RecordingEngine::new());
        engine.successful_result.store(false, Ordering::SeqCst);

        coordinator
            .register_engine(ProtoStorageEngine::Sst, engine.clone())
            .await;
        materialize_collection_with_coordinator(
            &coordinator,
            &write_buffer,
            &plan(107),
            None,
            None,
            true,
            None,
        )
        .await
        .expect_err("an unsuccessful result is not a durable publication");

        assert_eq!(write_buffer.claimed_flush_batch_count("107").unwrap(), 0);
        assert_eq!(
            write_buffer
                .get_unflushed_batches("107")
                .await
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn adr081_cancellation_releases_claim_for_retry() {
        let coordinator = Arc::new(FlushExecutionCoordinator::new(1));
        let write_buffer = Arc::new(WALBehaviorWrapper::new(MemtableConfig::default()));
        add_batch(&write_buffer, 108, "v0").await;
        let engine = Arc::new(RecordingEngine::blocking());
        let task_coordinator = coordinator.clone();
        let task_buffer = write_buffer.clone();
        let task_engine = engine.clone();
        coordinator
            .register_engine(ProtoStorageEngine::Sst, engine.clone())
            .await;

        let task = tokio::spawn(async move {
            materialize_collection_with_coordinator(
                &task_coordinator,
                &task_buffer,
                &plan(108),
                None,
                None,
                true,
                None,
            )
            .await
        });
        tokio::time::timeout(
            std::time::Duration::from_secs(2),
            engine.entered_flush.notified(),
        )
        .await
        .expect("flush must enter the engine");
        assert_eq!(write_buffer.claimed_flush_batch_count("108").unwrap(), 1);

        task.abort();
        let join_error = task.await.expect_err("aborted flush must be cancelled");
        assert!(join_error.is_cancelled());
        assert_eq!(write_buffer.claimed_flush_batch_count("108").unwrap(), 0);
        assert_eq!(
            write_buffer
                .get_unflushed_batches("108")
                .await
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn adr081_batch_appended_during_flush_remains_for_next_epoch() {
        let coordinator = Arc::new(FlushExecutionCoordinator::new(1));
        let write_buffer = Arc::new(WALBehaviorWrapper::new(MemtableConfig::default()));
        add_batch(&write_buffer, 109, "old").await;
        let engine = Arc::new(RecordingEngine::blocking());
        let task_coordinator = coordinator.clone();
        let task_buffer = write_buffer.clone();
        let task_engine = engine.clone();
        coordinator
            .register_engine(ProtoStorageEngine::Sst, engine.clone())
            .await;

        let task = tokio::spawn(async move {
            materialize_collection_with_coordinator(
                &task_coordinator,
                &task_buffer,
                &plan(109),
                None,
                None,
                true,
                None,
            )
            .await
        });
        tokio::time::timeout(
            std::time::Duration::from_secs(2),
            engine.entered_flush.notified(),
        )
        .await
        .expect("first source epoch must enter the engine");
        add_batch(&write_buffer, 109, "new").await;
        engine.release_flush.notify_one();
        task.await
            .expect("flush task must join")
            .expect("flush must succeed");

        let remaining = write_buffer
            .get_unflushed_batches("109")
            .await
            .expect("new source epoch must remain readable");
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].vector_records[0].oid, "new");
    }

    #[tokio::test]
    async fn adr081_engine_instance_is_canonical_per_format() {
        let coordinator = FlushExecutionCoordinator::new(2);
        let write_buffer = Arc::new(WALBehaviorWrapper::new(MemtableConfig::default()));
        add_batch(&write_buffer, 110, "v0").await;
        add_batch(&write_buffer, 111, "v1").await;
        let canonical = Arc::new(RecordingEngine::new());
        let discarded = Arc::new(RecordingEngine::new());

        coordinator
            .register_engine(ProtoStorageEngine::Sst, canonical.clone())
            .await;
        materialize_collection_with_coordinator(
            &coordinator,
            &write_buffer,
            &plan(110),
            None,
            None,
            true,
            None,
        )
        .await
        .unwrap();
        materialize_collection_with_coordinator(
            &coordinator,
            &write_buffer,
            &plan(111),
            None,
            None,
            true,
            None,
        )
        .await
        .unwrap();

        assert_eq!(canonical.flush_calls.load(Ordering::SeqCst), 2);
        assert_eq!(discarded.flush_calls.load(Ordering::SeqCst), 0);
    }
}
