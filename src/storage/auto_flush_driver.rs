//! ADR-069 / TD-WAL-1 — the live auto-flush DRIVER.
//!
//! The foundation PR (#1117) landed the flush *decision* (`flush_policy`) and the
//! *observability* (`wal_flush_metrics`) but deliberately shipped no trigger,
//! because the live canonical WAL path has no flush driver. This is that driver.
//!
//! It is a policy-gated, timer-driven clone of
//! [`crate::storage::engine::StorageEngine::flush_memtable_to_storage`]: each tick
//! it enumerates collections with unflushed data in the GLOBAL write buffer (the
//! same singleton REST/gRPC ingest writes to), evaluates the [`FlushPolicy`] per
//! collection (size / time / capacity envelope), and for those that are due calls
//! the proven [`materialize_collection`] recipe with `free_wal=true` — writing an
//! L0 SST and freeing the covered WAL. Every decision + flush is metered through
//! [`wal_flush_metrics`], so `/metrics/prometheus` shows exactly what fired.
//!
//! Durability is unaffected: the WAL fsync on write (SyncMode=PerBatch) is the
//! durability primitive; this driver only controls WHEN the memtable is
//! materialized to SST (RTO / WAL reclamation / object-store coalescing, plus the
//! volume-loss RPO bound). `free_wal=true` is the same setting the shutdown path
//! uses (de-risked by TD-165 cold-read recall); the live insert→flush→search
//! round-trip is the acceptance check.
//!
//! Spawned once from `StorageEngine::start()` when a periodic trigger is armed
//! (`policy.needs_scheduler()` — time or capacity budget set). The default config arms
//! the 300s time floor (ADR-069 D2 RPO safety net), so the driver spawns by default;
//! a config setting both `flush_interval_secs = 0` and `wal_max_bytes = 0` opts out.
//!
//! Operational notes (verified on a local smoke, 2026-11-08):
//! * The keys live in `[storage.wal_config]` (WriteBufferUserConfig) —
//!   `flush_interval_s` (no trailing `secs`) is a DIFFERENT knob: the
//!   io_trace ETL sink's segment-sealing timer ([observability.io_trace_sink]).
//! * The unflushed-window clock is lazily initialized on first sight of a
//!   collection (`window_age_secs`), and `list_collections_with_unflushed_data`
//!   does not report a collection recovered from WAL at startup until its
//!   first post-recovery write — so a restart-recovered memtable drains only
//!   after that first write plus `time_floor_secs`, not at boot.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::RwLock;

#[cfg(feature = "axis")]
use crate::index::AxisManager;
use crate::metrics::wal_flush_metrics;
use crate::storage::flush_materializer::materialize_collection;
use crate::storage::persistence::write_ahead_log::flush_policy::FlushPolicy;
use crate::storage::persistence::write_ahead_log::{
    get_global_write_buffer_behavior, list_collections_from_catalog,
};
use crate::storage::write_fence::StorageWriteFence;

/// Background driver that materializes unflushed memtable data to SST on the
/// ADR-069 flush policy.
pub struct AutoFlushDriver {
    policy: FlushPolicy,
    #[cfg(feature = "axis")]
    axis_index_manager: Option<Arc<AxisManager>>,
    /// A6 storage-write fence (default-OFF). Captured at spawn time; on the live
    /// server it is injected into the storage engine post-construction, so this
    /// may be `None` here — acceptable at MVP (single-pod, fence default-OFF).
    /// Hardening (spawn after fence injection) is a follow-up.
    storage_write_fence: Option<Arc<dyn StorageWriteFence>>,
    /// Per-collection start of the current unflushed window (last flush, or first
    /// observation). Feeds the time trigger's RPO decision; one clock, read by the
    /// decision and reset on a successful flush.
    flush_clock: Arc<RwLock<HashMap<String, Instant>>>,
}

impl AutoFlushDriver {
    /// Spawn the driver's background loop iff a periodic trigger is armed. The default
    /// config arms the 300s time floor so this spawns by default; setting both
    /// `flush_interval_secs = 0` and `wal_max_bytes = 0` makes this a no-op.
    pub fn spawn(
        policy: FlushPolicy,
        #[cfg(feature = "axis")] axis_index_manager: Option<Arc<AxisManager>>,
        storage_write_fence: Option<Arc<dyn StorageWriteFence>>,
    ) {
        if !policy.needs_scheduler() {
            return;
        }
        let tick = policy.scheduler_tick_secs();
        let driver = Self {
            policy,
            #[cfg(feature = "axis")]
            axis_index_manager,
            storage_write_fence,
            flush_clock: Arc::new(RwLock::new(HashMap::new())),
        };
        tracing::info!(
            "🕒 ADR-069 auto-flush driver armed: tick={}s time_floor={}s capacity_budget={}B",
            tick,
            driver.policy.time_floor_secs,
            driver.policy.capacity_budget_bytes
        );
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(std::time::Duration::from_secs(tick));
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                ticker.tick().await;
                driver.tick().await;
            }
        });
    }

    fn now_unixtime() -> i64 {
        chrono::Utc::now().timestamp()
    }

    /// Age (seconds) of the current unflushed window; lazily initialized to *now*
    /// on first sight so a brand-new collection never time-flushes immediately.
    async fn window_age_secs(&self, collection_id: &str) -> u64 {
        let mut clock = self.flush_clock.write().await;
        let base = clock
            .entry(collection_id.to_string())
            .or_insert_with(Instant::now);
        base.elapsed().as_secs()
    }

    /// Reset the unflushed-window clock after a flush (the RPO window restarts).
    async fn note_flushed(&self, collection_id: &str) {
        let mut clock = self.flush_clock.write().await;
        clock.insert(collection_id.to_string(), Instant::now());
    }

    /// One evaluation+flush pass over all collections with unflushed data.
    async fn tick(&self) {
        let Some(write_buffer) = get_global_write_buffer_behavior() else {
            return;
        };
        let collections = write_buffer.list_collections_with_unflushed_data().await;
        if collections.is_empty() {
            return;
        }
        let (budget, high, critical) = self.policy.budget_gauges();
        // Catalog is the metadata authority (engine / dimension / path / tenant) —
        // the same source flush_memtable_to_storage resolves each plan from.
        let catalog = list_collections_from_catalog().await;
        let mut pending = tokio::task::JoinSet::new();

        for collection_id in &collections {
            let mem_bytes = write_buffer.unflushed_bytes(collection_id).await;
            // Observation is emitted at the decision boundary: what /metrics shows
            // is exactly what the policy acted on.
            wal_flush_metrics::set_wal_size(collection_id, mem_bytes);
            wal_flush_metrics::set_budget(collection_id, budget, high, critical);

            let age = self.window_age_secs(collection_id).await;
            // TD-FLUSH-3 S1: the driver honors the predicted-segment floor too —
            // without this, its ~30s tick size-flushed sub-floor segments and
            // silently overrode the floor the inline path enforced (measured:
            // 4 segments instead of 2 on the 1M verification). Time (RPO) and
            // capacity verdicts are unaffected.
            let predicted_bytes = write_buffer.unflushed_predicted_bytes(collection_id).await;
            let decision = self
                .policy
                .evaluate_with_predicted(mem_bytes, age, predicted_bytes);
            let Some(reason) = decision.reason else {
                continue;
            };
            if decision.backpressure {
                wal_flush_metrics::inc_backpressure(collection_id);
            }

            // TD-FLUSH-8: the catalog is the identity authority — a WAL id MUST
            // be catalog-resolvable. The bulk list (now `["default"]`-fallback
            // aware) is the fast path; on a miss, fall back to the per-id
            // resolver, which additionally has the `resolve_table` fast-path
            // the write path trusts. Only when BOTH miss is the invariant
            // violated — escalate loudly (metric + error) instead of spinning
            // silently at WARN: a permanently unflushable collection is a
            // durability incident (its unflushed window grows without bound).
            let resolved_fallback = match catalog
                .iter()
                .find(|meta| &meta.collection.id == collection_id)
            {
                Some(meta) => Some(meta.clone()),
                None => {
                    crate::storage::persistence::write_ahead_log::resolve_collection_meta_from_catalog(
                        collection_id,
                    )
                    .await
                }
            };
            let Some(meta) = resolved_fallback.as_ref() else {
                wal_flush_metrics::inc_catalog_unresolved(collection_id);
                tracing::error!(
                    collection_id,
                    "🚨 ADR-069/TD-FLUSH-8 auto-flush: WAL collection id is NOT \
                     catalog-resolvable (identity-invariant violation) — flush \
                     skipped; unflushed window keeps growing"
                );
                continue;
            };
            let plan = match crate::storage::flush_materializer::flush_plan_from_collection_meta(
                &meta.collection,
                // Read from the catalog: LOADED (empty = the collection set none).
                Some(meta.properties.clone()),
            ) {
                Ok(plan) => plan,
                Err(error) => {
                    tracing::warn!(
                        collection_id,
                        "ADR-069 auto-flush: catalog identity resolution failed: {error}"
                    );
                    continue;
                }
            };

            // ADR-081 D4: submit independent collections concurrently. The
            // process-wide materializer permit pool bounds the actual work, and
            // its per-collection gate collapses overlap with inline/shutdown
            // triggers.
            let task_write_buffer = write_buffer.clone();
            #[cfg(feature = "axis")]
            let task_axis = self.axis_index_manager.clone();
            let task_fence = self.storage_write_fence.clone();
            let reason_label = reason.as_str().to_string();
            pending.spawn(async move {
                let start = Instant::now();
                #[cfg(feature = "axis")]
                let axis_arg: Option<&AxisManager> = task_axis.as_deref();
                #[cfg(not(feature = "axis"))]
                let axis_arg: Option<&()> = None;
                let result = materialize_collection(
                    &task_write_buffer,
                    &plan,
                    task_fence.as_ref(),
                    None,
                    true,
                    axis_arg,
                )
                .await;
                (
                    plan.collection_object_id.to_string(),
                    reason_label,
                    start.elapsed().as_secs_f64(),
                    result,
                )
            });
        }

        let mut any_flush_succeeded = false;
        while let Some(joined) = pending.join_next().await {
            let (collection_id, reason, dur, result) = match joined {
                Ok(value) => value,
                Err(error) => {
                    tracing::error!("ADR-081 auto-flush task failed to join: {}", error);
                    continue;
                }
            };
            match result {
                Ok(Some(outcome)) => {
                    any_flush_succeeded = true;
                    let remaining = write_buffer.unflushed_bytes(&collection_id).await;
                    wal_flush_metrics::record_successful_flush(
                        &collection_id,
                        &reason,
                        outcome.bytes,
                        outcome.entries_flushed,
                        dur,
                        Self::now_unixtime(),
                        remaining,
                    );
                    self.note_flushed(&collection_id).await;
                    tracing::info!(
                        "✅ ADR-081 auto-flush [{}] '{}': {} vectors, {} bytes in {:.3}s",
                        reason,
                        collection_id,
                        outcome.entries_flushed,
                        outcome.bytes,
                        dur
                    );
                }
                Ok(None) => {
                    // Raced to empty; reset the window so we don't re-decide every tick.
                    self.note_flushed(&collection_id).await;
                    let remaining = write_buffer.unflushed_bytes(&collection_id).await;
                    wal_flush_metrics::set_wal_size(&collection_id, remaining);
                }
                Err(e) => {
                    wal_flush_metrics::record_flush(&collection_id, &reason, false, 0, 0, dur, 0);
                    // TD-FLUSH-6: `{:#}` prints the full anyhow source chain, not
                    // just the outer "Failed to commit atomic flush operation"
                    // wrapper — the proximate object-store cause lives 1-2 .context
                    // layers down and was previously dropped.
                    tracing::warn!(
                        "❌ ADR-081 auto-flush [{}] '{}' failed: {:#}",
                        reason,
                        collection_id,
                        e
                    );
                }
            }
        }

        if any_flush_succeeded {
            Self::reap_manifest_segments().await;
        }
    }

    /// Checkpoint the global WAL manifest and reap fully-flushed segments.
    ///
    /// The manifest writes one `manifest_*.jsonl` file per append AND per
    /// status update, and its reaper previously had NO production caller —
    /// so the `data/wal` directory grew monotonically for the life of the
    /// data dir (~50 MB of dead manifest files after a Victor repo-scale
    /// ingest). Running checkpoint+cleanup once per tick that completed at
    /// least one flush bounds the directory to roughly one flush-interval
    /// of files. Best-effort: reaping must never fail the flush tick.
    pub(crate) async fn reap_manifest_segments() {
        use crate::storage::persistence::write_ahead_log::manifest as manifest_singleton;
        match manifest_singleton::create_checkpoint().await {
            Ok(_) => match manifest_singleton::cleanup_checkpointed().await {
                Ok(removed) if removed > 0 => {
                    tracing::info!(
                        "🧹 manifest reaper: removed {} checkpointed entries",
                        removed
                    );
                }
                Ok(_) => {}
                Err(e) => tracing::warn!("manifest cleanup failed (non-fatal): {:#}", e),
            },
            Err(e) => tracing::warn!("manifest checkpoint failed (non-fatal): {:#}", e),
        }
    }
}
