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

//! SST Engine Flush Module
//!
//! Contains flush operations and coordination logic for the SST engine.
//! This module is responsible for:
//! - Main flush operation implementation
//! - Multi-batch sorting and optimization
//! - Atomic flush operations with staging
//! - Vector sorting for SSTable encoding
//! - Flush result coordination

pub mod coordinator;
pub mod operations;
pub mod optimizer;

use anyhow::{Context, Result};
use chrono::Utc;
use std::collections::HashMap;
use tracing::{debug, info, warn};

use crate::storage::common::compaction_orchestrator::FilenameCodec;
use crate::storage::engines::core::formats::arrow_block::{ArrowBlockConfig, ArrowBlockWriter};
use crate::storage::engines::sst::block_format::BlockFormat;
use crate::storage::engines::sst::utils::SortingStats;
use crate::storage::engines::sst::{SstEngine, SstError};
use crate::storage::traits::{FlushParameters, FlushResult};
use crate::storage::transaction_coordinator::{StagingConfig, TransactionStageType};
use proximadb_records::ProximaRecord;

pub use coordinator::FlushCoordinator;
pub use operations::FlushOperations;
pub use optimizer::FlushOptimizer;

impl SstEngine {
    /// Register just-flushed vectors into the per-collection AXIS ANN index (TD-112).
    ///
    /// Without this, flushed/compacted vectors are never indexed, so post-flush
    /// vector search falls back to a brute-force segment scan
    /// (`sst/search` `fallback_to_direct_search`) and recall degrades as data
    /// ages out of the WAL memtable. Best-effort: the segments are already
    /// durable, so an indexing error is logged, not propagated. The per-collection
    /// `IndexUpdateMode` governs whether this blocks flush completion
    /// (Synchronous) or runs in the background. There is no double-index risk —
    /// the live write path does not populate AXIS, so flush is the first
    /// indexing point.
    /// ADR-078: the guard + call now live in one place shared by every engine
    /// (`storage::common::axis_flush_hook`); SST passes the handle it already
    /// holds, and the shared hook falls back to the boot-registered global.
    async fn index_flushed_into_axis(&self, params: &FlushParameters, files_created: Vec<String>) {
        crate::storage::common::axis_flush_hook::index_flushed_into_axis(
            self.axis_manager(),
            params,
            files_created,
        )
        .await
    }

    /// Main flush operation for SST engine
    ///
    /// This method implements the core flush logic for the SST engine:
    /// 1. Extract storage configuration from parameters
    /// 2. Sort vectors for optimal SSTable encoding
    /// 3. Write vectors to SSTable using atomic operations
    /// 4. Update metadata and trigger compaction if needed
    pub async fn flush_implementation(&self, params: &FlushParameters) -> Result<FlushResult> {
        // Check if quantization is enabled in collection config
        let quantization_enabled = params
            .collection_config
            .as_ref()
            .and_then(|c| c.config.as_ref())
            .and_then(|cfg| cfg.quantization.as_ref())
            .map(|q| q.enabled);

        if quantization_enabled.flatten().unwrap_or(false) {
            debug!("🔄 SST FLUSH: Quantization enabled, processing with quantization support");
            // Quantization will be handled internally during the flush process
            // The flush_with_quantization method has been removed - quantization is now internalized
        }

        let start_time = std::time::Instant::now();

        info!(
            "🔄 SST FLUSH: Starting standard flush operation for {} batches, {} vectors",
            params.batch_ids.len(),
            params.vector_records.len()
        );

        // Validate input parameters
        if params.vector_records.is_empty() {
            debug!("🔄 SST FLUSH: No vectors to flush, returning early");
            return Ok(FlushResult::default());
        }

        // Extract storage URL from parameters
        let collection_storage_url = Self::get_collection_storage_url_from_params(params)?;
        let collection_id = params
            .collection_id
            .as_ref()
            .ok_or_else(|| SstError::InvalidArgument("Collection ID is required".to_string()))?;

        debug!(
            "🔍 SST FLUSH: Using storage URL: {} for collection: {}",
            collection_storage_url, collection_id
        );

        // ADR-081 D3: admission is the first vector-path boundary. Direct
        // `do_flush` callers reach this check here; the WAL materializer also
        // invokes the same preflight before it claims/clones source records.
        self.preflight_flush_implementation(params).await?;

        // Sort canonical records for optimal SSTable encoding.
        #[cfg(test)]
        self.flush_sort_invocations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (sorted_vectors, sort_stats) = self
            .sort_vectors_for_sstable_encoding(params.vector_records.clone())
            .await?;

        // Convert the result to expected format
        let sorted_tuples: Vec<(String, ProximaRecord)> = sorted_vectors
            .into_iter()
            .map(|record| (record.oid.clone(), record))
            .collect();

        info!(
            "✅ SST FLUSH: Sorted {} vectors (estimated compression improvement: {:.1}%)",
            sort_stats.records_sorted,
            sort_stats.compression_estimate * 100.0
        );

        // Generate SSTable filename with appropriate extension based on block format
        let codec = FilenameCodec::new();
        // M1-3 (ADR-049): PAX is the ONLY vector write format — the legacy v1
        // streaming path (`write_sorted_proxima_records` → ProximaBlocks `.sst`) is
        // retired from flush. A non-Arrow collection ALWAYS writes a `.pax`
        // segment; the global kill-switch `PROXIMADB_PAX_QUANT_DISABLE`
        // and the per-collection `pax_vector_format:off` tag no longer force
        // legacy `.sst` — they select the recall-exact RawF32 quant instead (see
        // `resolve_pax_vector_quant`), so such a segment is exact-scan searchable
        // (`search_pax_file_exact`) rather than RaBitQ-dequantized. ArrowBlock
        // stays a distinct configured format (`config.block_format = "ArrowBlock"`,
        // exercised by `arrowblock_full_lifecycle_test`) — orthogonal to the
        // PAX/streaming retirement. Legacy `.sst` segments still READ back
        // (mixed-format-safe via magic-byte `is_pax_segment` + the extension /
        // `read_segment_records` read path).
        let (block_format, file_extension) =
            match BlockFormat::parse_block_format(&self.config().block_format) {
                BlockFormat::ArrowBlock => (BlockFormat::ArrowBlock, "arrow"),
                // ProximaBlocks config (the default) now writes PAX — streaming is
                // retired, so the legacy format selection is gone.
                BlockFormat::ProximaBlocks | BlockFormat::PaxBlock => {
                    (BlockFormat::PaxBlock, "pax")
                }
            };
        let sst_filename = params
            .hints
            .get("recovery_materialization_id")
            .and_then(|value| value.as_str())
            .map(|id| {
                let safe_id: String = id
                    .chars()
                    .filter(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_'))
                    .collect();
                format!("L0_recovery_{safe_id}.{file_extension}")
            })
            .unwrap_or_else(|| codec.generate(0, file_extension)); // Level 0 for flush
        debug!(
            "🔧 SST: Creating {} file: {} for collection: {}",
            if block_format == BlockFormat::ArrowBlock {
                "Arrow"
            } else {
                "SSTable"
            },
            sst_filename,
            collection_id
        );

        // Perform atomic flush operation
        let flush_result = self
            .perform_atomic_flush(
                sorted_tuples,
                &collection_storage_url,
                &sst_filename,
                params,
                block_format,
            )
            .await?;

        // Train/update PCA model for Z-Order spatial encoding.
        // TD-IVF-1: skip during recovery flushes (collection_config absent) — the segments
        // already have the persisted PCA/IVF model in the A0 region, and re-training loads
        // the entire dataset into RAM (5.7 GB for 1M vectors, 100% CPU for minutes).
        // The cascade reader loads centroids lazily from A0 at search time.
        let needs_flush_pca = crate::storage::common::axis_flush_hook::axis_needed(params);
        if params.vector_records.len() >= 100 && needs_flush_pca {
            // Only train with enough samples
            match self
                .train_and_cache_pca_model(
                    collection_id,
                    &collection_storage_url,
                    &params.vector_records,
                )
                .await
            {
                Ok(()) => {
                    // Also update the global cache for search access
                    if let Some(model) = self
                        .get_pca_model(collection_id, &collection_storage_url)
                        .await
                    {
                        super::core::set_collection_pca_model(collection_id, model);
                    }
                }
                Err(e) => {
                    // Log but don't fail flush - PCA is an optimization
                    tracing::warn!(
                        "[SST] Failed to train PCA model during flush: {}. Z-Order pruning may be less effective.",
                        e
                    );
                }
            }
        }

        let duration = start_time.elapsed();
        info!(
            "🏁 SST FLUSH: Completed in {:.2}ms - {} vectors, {} bytes",
            duration.as_millis(),
            flush_result.entries_flushed.unwrap_or(0),
            flush_result.bytes_written.unwrap_or(0)
        );

        Ok(flush_result)
    }

    /// Get collection storage URL from flush parameters
    fn get_collection_storage_url_from_params(params: &FlushParameters) -> Result<String> {
        debug!("🔍 SST FLUSH: Determining storage URL");
        info!(
            "   - Has collection_config: {}",
            params.collection_config.is_some()
        );

        // Extract storage location from collection config in parameters
        if let Some(ref collection) = params.collection_config {
            info!(
                "   - Has storage_assignment: {}",
                collection.storage_assignment.is_some()
            );
            if let Some(ref assignment) = collection.storage_assignment {
                info!("   - Base location: {}", assignment.base_location);
                info!("   - Collection ID: {:?}", params.collection_id);
                let collection_id = params.collection_id.as_deref().ok_or_else(|| {
                    SstError::InvalidArgument("Collection ID is required".to_string())
                })?;
                let identity =
                    crate::storage::trait_components::path_resolver::typed_path_identity(
                     crate::storage::trait_components::path_resolver::typed_identity_from_storage_assignment(
                         Some(assignment),
                     ),
                 );
                let storage_url =
                    crate::storage::trait_components::path_resolver::collection_data_path_typed(
                        &assignment.base_location,
                        collection_id,
                        identity,
                    );
                debug!(
                    "🔍 SST FLUSH: Using storage URL from params: {}",
                    storage_url
                );
                return Ok(storage_url);
            }
        }

        Err(SstError::InvalidArgument(
            "No storage assignment found in collection config".to_string(),
        )
        .into())
    }

    // Removed duplicate sort_vectors_for_sstable_encoding method - using the one from utils.rs

    /// ADR-081 D3: bounded metadata-only admission for an SST flush.
    ///
    /// This method must remain independent of `vector_records`: the WAL
    /// materializer calls it before claiming or cloning record payloads.
    pub(crate) async fn preflight_flush_implementation(
        &self,
        params: &FlushParameters,
    ) -> Result<()> {
        let recovery_materialization = params
            .hints
            .get("recovery_materialization_id")
            .and_then(|value| value.as_str());

        // Recovery must complete even while the normal live L0 backlog is at
        // STOP, otherwise boot can deadlock before compaction workers run.
        if self.compaction_manager().is_none() || recovery_materialization.is_some() {
            return Ok(());
        }

        let cfg = self.config();
        let stop_trigger = std::env::var("PROXIMADB_L0_STOP_TRIGGER")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(cfg.l0_stop_trigger);
        let slowdown_trigger = std::env::var("PROXIMADB_L0_SLOWDOWN_TRIGGER")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(cfg.l0_slowdown_trigger);
        if stop_trigger == 0 && slowdown_trigger == 0 {
            return Ok(());
        }

        let storage_url = Self::get_collection_storage_url_from_params(params)?;
        let files = self
            .discover_sstable_files(&storage_url)
            .await
            .map_err(|error| {
                SstError::Flush(format!(
                    "ADR-081: cannot evaluate L0 admission for '{}': {}",
                    storage_url, error
                ))
            })?;
        // ADR-081 corrects TD-COMPACT-7's all-level count. The admission
        // backlog is L0 only; stable L1/L2 objects must not stall new flushes.
        let l0_count = files
            .iter()
            .filter(|path| {
                path.url
                    .rsplit('/')
                    .next()
                    .is_some_and(|name| name.starts_with("L0_"))
            })
            .count() as u32;

        if stop_trigger > 0 && l0_count >= stop_trigger {
            tracing::warn!(
                collection = %storage_url,
                l0_count,
                stop_trigger,
                "ADR-081: L0 admission STOP before record materialization"
            );
            return Err(SstError::Flush(format!(
                "ADR-081: L0 admission stop — {l0_count} L0 files >= stop_trigger \
                 {stop_trigger}; flush rejected before record materialization"
            ))
            .into());
        }

        if slowdown_trigger > 0 && l0_count >= slowdown_trigger {
            tracing::debug!(
                collection = %storage_url,
                l0_count,
                slowdown_trigger,
                "ADR-081: L0 admission SLOWDOWN before record materialization"
            );
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
        Ok(())
    }

    /// Perform atomic flush operation with staging
    async fn perform_atomic_flush(
        &self,
        sorted_vectors: Vec<(String, ProximaRecord)>,
        storage_url: &str,
        filename: &str,
        params: &FlushParameters,
        block_format: BlockFormat,
    ) -> Result<FlushResult> {
        let recovery_materialization = params
            .hints
            .get("recovery_materialization_id")
            .and_then(|value| value.as_str());

        // Begin atomic operation
        let staging_config = StagingConfig {
            base_url: storage_url.to_string(),
            collection_id: None, // Already included in base_url
            operation_type: TransactionStageType::Flush,
            custom_staging_dir: None,
            auto_cleanup: true,
            max_orphaned_age_hours: 24,
            skip_uuid_subdir: true,
        };

        tracing::debug!(storage_url = %storage_url, filename = %filename, "Starting flush operation");

        let atomic_op = if recovery_materialization.is_some() {
            // Recovery publishes directly to the deterministic final object via
            // write_if_absent. Track it as zero-copy managed so abort only removes
            // coordinator metadata and never tries to delete an object-store
            // staging prefix that was intentionally never created.
            self.atomic_coordinator()
                .begin_zero_copy_managed_operation(&staging_config)
                .await
                .context("Failed to begin recovery publication")?
        } else {
            self.atomic_coordinator()
                .begin_atomic_operation(&staging_config)
                .await
                .context("Failed to begin atomic flush operation")?
        };

        tracing::debug!(staging_url = %atomic_op.staging_url, final_url = %atomic_op.final_url, "Atomic operation initialized");

        // PAX/Arrow writers materialize through a local path. During recovery the
        // resulting bytes are published directly to the deterministic object name
        // with write_if_absent, so using the object-store transaction staging URL
        // here would create a local `adls:/...` path and then try to read an object
        // that was never uploaded. Keep a scoped local directory alive until the
        // conditional publication completes. Normal flushes retain their existing
        // transaction-coordinator staging path.
        let recovery_staging = if recovery_materialization.is_some() {
            Some(tempfile::tempdir().context("creating local recovery staging directory")?)
        } else {
            None
        };
        let staging_url = if let Some(directory) = recovery_staging.as_ref() {
            format!("file://{}", directory.path().join(filename).display())
        } else {
            format!("{}/{}", atomic_op.staging_url, filename)
        };
        tracing::debug!(staging_path = %staging_url, "Full staging path constructed");

        // TD-OBJSTORE-4 defect 6 (normal-flush arm): the segment writers below are
        // LOCAL-file writers. Stage through `StagedSegmentWrite` so a CLOUD staging
        // URL gets a real local scratch file + an upload on finalize — instead of a
        // literal local `az:...` directory, a promote that stages nothing, and a
        // false-success that lets WAL deletion destroy the only durable copy. The
        // recovery path's `file://{tempdir}` and plain local bases take the direct
        // zero-copy arm (finalize is a no-op).
        let staged =
            crate::storage::engines::sst::staged_write::StagedSegmentWrite::begin(&staging_url)
                .await
                .context("Failed to prepare staged segment write")?;

        // Count entries for writing
        let entries_written = sorted_vectors.len() as u64;

        // Captured outcome from the underlying writer. M1-3 (ADR-049): the only
        // arm that populated this (legacy ProximaBlocks streaming) is retired, so
        // it is always `None` today — the object-economy directory emission below
        // stays dormant until the PAX writer grows an equivalent outcome
        // (`write_pax_segment_full` returns `PaxSegmentMeta`, not
        // `SstableWriteOutcome`). Kept wired so the directory hook revives
        // unchanged when that lands.
        // TD-RDSTRAT-5 S2: set from the PAX write when block clustering produced
        // per-block centroids, so the object-economy directory emission below fires
        // (write-through of the vector zone-map). Stays `None` when clustering is
        // off (empty centroids) ⇒ no emission — S2 is default-OFF via the S1 flag.
        let mut write_outcome: Option<crate::storage::engines::sst::writer::SstableWriteOutcome> =
            None;
        let cache_on_write = std::env::var("PROXIMADB_CACHE_ON_WRITE")
            .ok()
            .and_then(|value| crate::core::config::CacheOnWritePolicy::parse(&value))
            .unwrap_or(self.config().cache_on_write);
        let mut pax_cache_seed: Option<proximadb_storage_common::pax_block::PaxCacheSeed> = None;

        match block_format {
            BlockFormat::ArrowBlock => {
                // Use ArrowBlockWriter for Arrow IPC format
                // Get dimension from collection config or infer from first vector
                let dimension = params
                    .collection_config
                    .as_ref()
                    .and_then(|c| c.config.as_ref())
                    .map(|cfg| cfg.dimension)
                    .or_else(|| {
                        // Fallback: infer from first vector
                        sorted_vectors
                            .first()
                            .and_then(|(_, rec)| rec.embeddings.first())
                            .map(|embedding| embedding.values.len() as u32)
                    })
                    .unwrap_or(128); // Default dimension if not available

                tracing::debug!(entries_written, dimension, "Writing vectors to Arrow block");

                // Local write path from the staged write (uploaded on finalize
                // when the staging URL is remote).
                let staging_path = staged.local_path();

                // Ensure parent directory exists
                if let Some(parent) = std::path::Path::new(staging_path).parent() {
                    std::fs::create_dir_all(parent)
                        .context("Failed to create staging directory for Arrow block")?;
                }

                let config = ArrowBlockConfig::new(dimension);
                let mut writer = ArrowBlockWriter::new(staging_path, config)
                    .context("Failed to create ArrowBlockWriter")?;

                let records: Vec<ProximaRecord> =
                    sorted_vectors.iter().map(|(_, rec)| rec.clone()).collect();

                writer
                    .write_block(&records)
                    .context("Failed to write block to Arrow file")?;
                writer
                    .finalize()
                    .context("Failed to finalize Arrow block")?;

                tracing::debug!("Arrow block write operation completed");
            }
            // M1-3 (ADR-049): the legacy ProximaBlocks streaming write is retired.
            // The flush gating above normalizes every non-Arrow collection to
            // PaxBlock, so ProximaBlocks is unreachable here — it is folded into
            // the PAX arm so the match stays exhaustive and any future ProximaBlocks
            // request re-encodes as PAX (mixed-read-safe) rather than streaming.
            BlockFormat::PaxBlock | BlockFormat::ProximaBlocks => {
                // P3 Phase B: write a columnar PAX vector segment (flag-gated, default
                // OFF). Mirrors the Arrow arm's local-staging approach — write to the
                // local staging path; the atomic op promotes it to the final (possibly
                // object-store) URL. Reads are mixed-format-safe via
                // `segment_format::read_segment_records` (magic-byte detection), so no
                // manifest/reader change is required for this segment to be read back.
                use crate::storage::engines::sst::segment_format::{
                    write_pax_segment_full_with_cache_seed_and_layout,
                    write_pax_segment_full_with_layout,
                };
                // Local write path from the staged write (uploaded on finalize
                // when the staging URL is remote).
                let staging_path = staged.local_path();
                if let Some(parent) = std::path::Path::new(staging_path).parent() {
                    tokio::fs::create_dir_all(parent)
                        .await
                        .context("Failed to create PAX staging directory")?;
                }
                let embedding_count = sorted_vectors
                    .first()
                    .map(|(_, rec)| rec.embeddings.len().max(1))
                    .unwrap_or(1);
                let records: Vec<_> = sorted_vectors.into_iter().map(|(_, rec)| rec).collect();
                let collection_id = params.collection_id.as_deref().unwrap_or("default");
                // P3 Phase D / TD-155 + Phase F + M1-3: vector-quantization
                // strategy. Precedence (see `resolve_pax_vector_quant`): kill-switch
                // / `pax_vector_format:off` → recall-exact RawF32 (M1-3: the legacy
                // `.sst` escape is gone — RawF32-PAX is exact-scan searchable via
                // `search_pax_file_exact`); else per-collection `pax_vector_quant` >
                // env `PROXIMADB_PAX_VECTOR_QUANT` > default RaBitQ (Phase F: the
                // cascade's stage-1 ranks RaBitQ-coded segments).
                let quant = resolve_pax_vector_quant(params.collection_config.as_ref());
                // TD-156 / ADR-026: configurable PAX block geometry. `None` keeps
                // the writer default; a larger value (e.g. 8-16 MiB for object
                // storage) coalesces rows into fewer blocks, cutting the per-block
                // ranged-GET fragmentation measured by the footer-cache economics
                // harness. Per-collection config is the productionization follow-up.
                let target_block = std::env::var("PROXIMADB_PAX_BLOCK_SIZE")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok());
                // P3 Phase D f32 tier (opt-in): per-collection `pax_f32_tier` tag >
                // env `PROXIMADB_PAX_F32_TIER` > default off. Emits an exact-f32
                // stripe (read lazily) for an exact final rerank + include_vectors.
                let f32_tier = resolve_pax_f32_tier(params.collection_config.as_ref());
                // Extract the tag list here (flush config is the legacy v1 proto
                // type, already referenced by the calls above) so resolve_pax_rerank_quant
                // stays v1-free — TD-123: don't add new v1-proto references.
                let rerank_tags: &[String] = params
                    .collection_config
                    .as_ref()
                    .and_then(|c| c.config.as_ref())
                    .map(|cfg| cfg.tags.as_slice())
                    .unwrap_or(&[]);
                let rerank_quant = resolve_pax_rerank_quant(rerank_tags);
                // TD-PAXRG-1: row-group Region D — per-collection
                // `pax_rg_layout` tag > env `PROXIMADB_PAX_WRITE_RG_LAYOUT` >
                // default OFF. No-ops for non-RaBitQ-coalesced writes.
                let rg_layout = resolve_pax_rg_layout(rerank_tags);
                // TD-FPRUNE-1 M1: shred declared filterable tags into typed
                // user-columns at flush so footer-resident filter pruning fires
                // (empty spec ⇒ byte-identical to the non-shredded write).
                let shred_spec = resolve_flush_shred_spec(params.collection_config.as_ref());
                let meta = if cache_on_write.includes_invariants() {
                    let write = write_pax_segment_full_with_cache_seed_and_layout(
                        std::path::Path::new(staging_path),
                        &records,
                        collection_id,
                        embedding_count,
                        quant,
                        rerank_quant,
                        f32_tier,
                        target_block,
                        rg_layout,
                        cache_on_write.includes_survivors(),
                        &shred_spec,
                        Some(storage_url),
                    )
                    .context("Failed to write PAX vector segment")?;
                    pax_cache_seed = write.cache_seed;
                    write.meta
                } else {
                    write_pax_segment_full_with_layout(
                        std::path::Path::new(staging_path),
                        &records,
                        collection_id,
                        embedding_count,
                        quant,
                        rerank_quant,
                        f32_tier,
                        target_block,
                        rg_layout,
                        &shred_spec,
                        Some(storage_url),
                    )
                    .context("Failed to write PAX vector segment")?
                };
                tracing::debug!(blocks = meta.block_count, "PAX segment write completed");
                // TD-RDSTRAT-5 S2: when block clustering produced per-block centroids,
                // build an outcome from the segment metadata so the directory emission
                // below fires with the real vector zone-map. Empty centroids (clustering
                // off) ⇒ no outcome ⇒ no emission (default-OFF).
                if !meta.block_centroids.is_empty() {
                    write_outcome = Some(pax_write_outcome(&meta));
                }
            }
        }

        // Promote the locally staged bytes to the (possibly remote) staging URL
        // so the atomic commit has a real file to move.
        let _staged_bytes = staged
            .finalize(&self.filesystem_port())
            .await
            .context("Failed to upload staged segment to the staging URL")?;

        // Get actual bytes written from the filesystem
        let fs = self.filesystem().get_filesystem(&staging_url)?;
        let file_metadata = fs.metadata(&staging_url).await.unwrap_or_else(|_| {
            crate::storage::persistence::filesystem::FileMetadata {
                path: staging_url.clone(),
                size: entries_written * 1024, // Estimate if metadata unavailable
                created: None,
                modified: None,
                is_directory: false,
                permissions: None,
                etag: None,
                storage_class: None,
            }
        });
        let bytes_written = file_metadata.size;

        let mut already_materialized = false;
        if recovery_materialization.is_some() {
            // The deterministic segment object is the recovery commit record.
            // Create-only publication closes the crash window between flush and
            // WAL retirement without a second marker object.
            let staged_bytes = fs
                .read(&staging_url)
                .await
                .context("reading staged recovery segment")?;
            let final_url = format!("{}/{}", atomic_op.final_url, filename);
            let final_fs = self.filesystem().get_filesystem(&final_url)?;
            match final_fs
                .write_if_absent(
                    &final_url,
                    &staged_bytes,
                    Some(crate::storage::persistence::filesystem::FileOptions {
                        create_dirs: true,
                        overwrite: false,
                        ..Default::default()
                    }),
                )
                .await
            {
                Ok(()) => {}
                Err(crate::storage::persistence::filesystem::FilesystemError::AlreadyExists(_)) => {
                    let existing = final_fs
                        .read(&final_url)
                        .await
                        .context("verifying existing recovery segment")?;
                    if existing != staged_bytes {
                        anyhow::bail!(
                            "recovery segment collision at {final_url}: deterministic name exists with different bytes"
                        );
                    }
                    already_materialized = true;
                }
                Err(error) => return Err(error).context("publishing recovery segment"),
            }
            if let Err(error) = self
                .atomic_coordinator()
                .finalize_atomic_operation(&atomic_op.operation_id)
                .await
            {
                // TD-FLUSH-6: print the full source chain (`{:#}`) so the inner
                // object-store cause is visible, not just the outer wrapper.
                tracing::warn!(
                    "failed to finalize recovery publication tracking: {:#}",
                    error
                );
            }
        } else {
            tracing::debug!(operation_id = %atomic_op.operation_id, "Committing atomic operation");
            self.atomic_coordinator()
                .finalize_atomic_operation(&atomic_op.operation_id)
                .await
                .context("Failed to commit atomic flush operation")?;
        }
        tracing::debug!(
            final_url = %atomic_op.final_url,
            filename = %filename,
            bytes = bytes_written,
            "SST flush atomic commit done"
        );

        // Vector Object Economy directory emission (Phase 4, option 1-B).
        // Emit at the final URL — staging is irrelevant once the atomic
        // commit succeeded. Skipped entirely when:
        //   * The engine was constructed without `with_directory_cache`.
        //   * The block format doesn't carry index metadata (ArrowBlock).
        //   * The flush carries no collection_id (defensive — validation
        //     already rejected this case, but keep the guard explicit).
        // Conservative defaults: storage_epoch=0, freshness_lsn=0,
        // authority_mode=RebuildableProjection until WAL/manifest sources
        // are plumbed. Emit failures log and continue — directory is a
        // rebuildable projection and must not fail the flush.
        if let (Some(cache), Some(outcome), Some(collection_id)) = (
            self.directory_cache_ref(),
            write_outcome.as_ref(),
            params.collection_id.as_ref(),
        ) {
            use crate::storage::engines::sst::object_economy_directory::SstableWriterDirectoryHooks;
            use proximadb_catalog::CatalogAuthorityMode;

            // Resolve the freshness watermark from the configured source
            // when present. Without a source the engine emits `0`, which
            // forces strong-route readers to always re-scan the WAL delta.
            let freshness_lsn = match self.freshness_lsn_source_ref() {
                Some(source) => source.current_lsn(collection_id.as_str()).await,
                None => 0,
            };

            let hooks = SstableWriterDirectoryHooks {
                cache: cache.clone(),
                collection_id: collection_id.clone(),
                collection_root: storage_url.to_string(),
                storage_epoch: 0,
                authority_mode: CatalogAuthorityMode::RebuildableProjection,
                freshness_lsn,
                level: 0,
            };
            let final_file_url = format!("{}/{}", atomic_op.final_url, filename);
            let fs_handle = self.filesystem().get_filesystem(&final_file_url)?;
            if let Err(err) = hooks
                .emit_after_flush(
                    &*fs_handle,
                    &final_file_url,
                    &final_file_url,
                    outcome.file_size_bytes,
                    outcome.block_index_offset,
                    outcome.block_index_size,
                    &outcome.index_entries,
                )
                .await
            {
                tracing::warn!(
                    "SST Flush: directory emission failed for {} ({}); read-side \
                     route will degrade to embedded-index path until next flush rebuilds",
                    final_file_url,
                    err
                );
            } else {
                tracing::debug!(
                    "SST Flush: emitted object-economy directory entry for {} (collection={})",
                    final_file_url,
                    collection_id
                );
            }
        }

        let final_file_path = format!("{}/{}", atomic_op.final_url, filename);

        if let Some(seed) = pax_cache_seed {
            crate::storage::engines::sst::segment_format::install_pax_cache_seed(
                &final_file_path,
                seed,
                self.segment_invariants_cache.as_deref(),
                self.survivor_cache.as_deref(),
            )
            .await;
        }

        if already_materialized {
            return Ok(FlushResult {
                success: true,
                collections_affected: vec![params.collection_id.clone().unwrap_or_default()],
                entries_flushed: Some(entries_written),
                bytes_written: Some(bytes_written),
                files_created: Some(0),
                file_paths: vec![final_file_path],
                duration_ms: Some(0),
                completed_at: Utc::now(),
                engine_metrics: HashMap::from([
                    (
                        "engine".to_string(),
                        serde_json::Value::String("SST".to_string()),
                    ),
                    (
                        "already_materialized".to_string(),
                        serde_json::Value::Bool(true),
                    ),
                ]),
                compaction_triggered: false,
                compaction_error: None,
                flushed_batch_ids: params.batch_ids.clone(),
            });
        }

        // Check if compaction should be triggered (per-collection tag cascade,
        // TD-WLP-2 — the global env stays the master kill-switch).
        let collection_tags: &[String] = params
            .collection_config
            .as_ref()
            .and_then(|c| c.config.as_ref())
            .map(|cfg| cfg.tags.as_slice())
            .unwrap_or(&[]);
        let suppress_recovery_compaction = params
            .hints
            .get("suppress_compaction_until_wal_retired")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let compaction_due_threshold = if suppress_recovery_compaction {
            None
        } else {
            self.should_trigger_compaction(storage_url, collection_tags)
                .await?
        };

        // TD-WLP-7 (ADR-061 D3): schedule compaction when armed. Before
        // this, the flush only set `compaction_triggered` and nothing scheduled
        // the compactor (the post-flush hook was a historical stub), so the
        // TD-WLP-4 armed-by-default re-cluster never ran and the read-side prune
        // it unlocks stayed dark. The compaction itself goes through the
        // atomic-swap path (ADR-046 LSN-coherent read across the segment swap).
        //
        // It is ENQUEUED, not run inline: the call below is
        // `enqueue_due_compaction`, which returns once the task is on the queue
        // and a worker notified, and the Ok(true) arm logs "enqueued (async)".
        // So the work happens on a background worker, and a caller that needs the
        // compacted product on disk must await `await_compaction_quiescence` --
        // which is the premise the SIFT IVF2 ratchet's barrier rests on, and
        // reading this comment is how a reviewer would check that premise. (Two
        // earlier revisions said "actually EXECUTE" and "Run it inline (awaited)
        // ... synchronous so it is deterministic and spawns no background worker
        // thread"; both were false and both are removed.) The
        // segment this flush wrote is already committed at its final URL above,
        // so the merge reads a consistent L0 set. Best-effort: a compaction
        // failure is recorded on the result but never fails the flush.
        let mut compaction_error: Option<String> = None;
        let mut compaction_ran = false;
        if let Some(due_threshold) = compaction_due_threshold
            && let Some(cid) = params.collection_id.as_deref()
            && let Some(compaction) = self.compaction_manager()
        {
            let collection_dir = std::path::Path::new(storage_url);
            // The gate's effective threshold: the per-collection count value,
            // or 1 when the TD-COMPACT-5 training arm fired (so the executor
            // compacts the few large untrained L0s instead of declining).
            let l0_threshold = due_threshold;
            // TD-COMPACT-8 / TD-COMPACT-6 D1: mark training in-flight BEFORE
            // enqueuing. Only for the training arm (threshold <= 1). On the
            // async path the flag is NOT cleared here — the background worker
            // clears it (via release_task_state) once the compaction completes,
            // covering the entire queue→run window. This is what lets
            // `should_trigger_compaction` skip re-arming across rapid flushes
            // (eliminating the redundant re-training loop). self.training_in_flight
            // is the same Arc the Compaction's workers hold.
            let is_training = due_threshold <= 1;
            if is_training && let Ok(mut guard) = self.training_in_flight.lock() {
                guard.insert(storage_url.to_string());
            }
            // TD-COMPACT-6 D1 (ADR-076): ENQUEUE to the background worker pool
            // instead of executing inline. Flush returns immediately (~1s after
            // the L0 write, not ~35s blocked on the re-cluster training). The
            // producer/consumer rate-control loop is closed by the L0 admission
            // watermarks (TD-COMPACT-7) checked above; the dedup + the
            // training_in_flight guard above prevent redundant work. The worker
            // clears training_in_flight on completion (Ok or Err).
            let precision_hint = params
                .collection_config
                .as_ref()
                .and_then(|collection| collection.config.as_ref())
                .and_then(|config| config.canonical_embedding_precision)
                .and_then(|precision| {
                    use crate::proto::proximadb_v1::EmbeddingPrecision;
                    match EmbeddingPrecision::try_from(precision) {
                        Ok(EmbeddingPrecision::Fp16) => {
                            Some(proximadb_records::EmbeddingScalarType::Fp16)
                        }
                        Ok(EmbeddingPrecision::Bf16) => {
                            Some(proximadb_records::EmbeddingScalarType::Bf16)
                        }
                        Ok(EmbeddingPrecision::Int8) => {
                            Some(proximadb_records::EmbeddingScalarType::Int8Scalar)
                        }
                        Ok(EmbeddingPrecision::Uint8) => {
                            Some(proximadb_records::EmbeddingScalarType::UInt8Scalar)
                        }
                        Ok(EmbeddingPrecision::Unspecified | EmbeddingPrecision::Fp32) | Err(_) => {
                            None
                        }
                    }
                });
            let enqueue_result = match params.get_collection_object_id() {
                Ok(collection_object_id) => {
                    compaction
                        .enqueue_due_compaction(
                            collection_object_id,
                            params.get_collection_identity(),
                            collection_dir,
                            l0_threshold,
                            precision_hint,
                            // TD-PAXRG-1: per-collection `pax_rg_layout` tag >
                            // env > default (resolved at the enqueue seam where
                            // the collection config is in scope).
                            params
                                .collection_config
                                .as_ref()
                                .and_then(|c| c.config.as_ref())
                                .map(|cfg| cfg.tags.as_slice())
                                .and_then(pax_rg_layout_tag)
                                .or_else(|| Some(resolve_pax_rg_layout(&[]))),
                        )
                        .await
                }
                Err(error) => Err(crate::core::StorageError::SstEngine(format!(
                    "compaction admission requires a catalog-resolved collection object id for {cid:?}: {error}"
                ))),
            };
            match enqueue_result {
                Ok(true) => {
                    compaction_ran = true;
                    info!(
                        "✅ SST Flush: re-cluster compaction enqueued (async) for collection {cid}"
                    );
                }
                Ok(false) => {
                    debug!("SST Flush: compaction armed but nothing due for collection {cid}");
                    // Nothing was due, so no worker will run to clear the guard
                    // — clear it here so the training arm can re-arm next flush.
                    if is_training && let Ok(mut guard) = self.training_in_flight.lock() {
                        guard.remove(storage_url);
                    }
                }
                Err(e) => {
                    warn!(
                        "SST Flush: re-cluster compaction enqueue failed for {cid} (best-effort, \
                         flush succeeded): {e}"
                    );
                    compaction_error = Some(e.to_string());
                    // Enqueue failed — no worker will clear the guard; do it here.
                    if is_training && let Ok(mut guard) = self.training_in_flight.lock() {
                        guard.remove(storage_url);
                    }
                }
            }
        }

        // TD-112: index the just-flushed vectors into AXIS so post-flush search is
        // served by the ANN index rather than a brute-force segment scan.
        self.index_flushed_into_axis(params, vec![final_file_path.clone()])
            .await;

        // ADR-037 / TD-174: maintain the resident statistics summary at the flush
        // write boundary (the sibling of the KSU resident-bytes meter) and stamp
        // the freshness watermark. Best-effort and O(records) — observes the
        // records already in hand, never a separate corpus scan (Decision 1).
        if let Some(cid) = params.collection_id.as_deref() {
            // Text columns (if any) drive the document/BM25 corpus block; all
            // other scalar props drive per-field distribution sketches.
            let text_columns: std::collections::HashSet<&str> = params
                .collection_config
                .as_ref()
                .and_then(|c| c.config.as_ref())
                .map(|cfg| cfg.text_columns.iter().map(|s| s.as_str()).collect())
                .unwrap_or_default();
            observe_flush_into_statistics(cid, &params.vector_records, &text_columns);
        }

        // Create flush result with file path for AXIS index building
        Ok(FlushResult {
            success: true,
            collections_affected: vec![params.collection_id.clone().unwrap_or_default()],
            entries_flushed: Some(entries_written),
            bytes_written: Some(bytes_written),
            files_created: Some(1),
            file_paths: vec![final_file_path],
            duration_ms: Some(0), // Will be set by caller
            completed_at: Utc::now(),
            engine_metrics: {
                let mut metrics = HashMap::new();
                metrics.insert(
                    "engine".to_string(),
                    serde_json::Value::String("SST".to_string()),
                );
                metrics.insert(
                    "filename".to_string(),
                    serde_json::Value::String(filename.to_string()),
                );
                // TD-WLP-7: whether the armed re-cluster compaction was
                // ENQUEUED by this flush (distinct from `compaction_triggered`,
                // which only means "armed + over threshold"). It is set from
                // `enqueue_due_compaction`'s Ok(true), so it does NOT mean the
                // compaction finished -- an earlier revision of this comment said
                // "actually executed", which callers relied on. Await
                // `await_compaction_quiescence` if you need the product. The
                // integration gate asserts on this.
                metrics.insert(
                    "compaction_ran".to_string(),
                    serde_json::Value::Bool(compaction_ran),
                );
                metrics
            },
            compaction_triggered: compaction_due_threshold.is_some(),
            compaction_error,
            flushed_batch_ids: params.batch_ids.clone(),
        })
    }

    /// Whether L0→base compaction should be triggered after this flush (TD-114,
    /// per-collection override TD-WLP-2 / ADR-061 D5).
    ///
    /// Default-OFF: absent a per-collection `compaction:on` tag the trigger only
    /// arms when `PROXIMADB_L0_COMPACTION_ENABLED` is truthy, so the untagged
    /// flush path is byte-for-byte unchanged. A tagged collection may arm (at
    /// its own `l0_threshold:N`) while the global gate is unset; an explicitly
    /// falsy global env is the master kill-switch nothing can override. When
    /// armed, it reuses the existing segment discovery to count L0 segments and
    /// arms once the effective threshold is reached. Discovery errors are
    /// treated as "not yet" (best-effort, never fails flush).
    /// Returns the effective L0 threshold to compact with, or `None` when
    /// nothing is due. Two arms:
    /// - COUNT (original): `l0_count >= per-collection threshold` → compact at
    ///   that threshold.
    /// - TRAINING (TD-COMPACT-5): any L0 segment is UNTRAINED (header
    ///   `layout_version != 3 || a0_len == 0` — no A0 coarse directory) and at
    ///   least `PROXIMADB_TRAINING_COMPACTION_MIN_MB` (default 32) on disk →
    ///   compact at threshold 1, so the flush-floor's few-large-segments
    ///   layout still receives its IVF training pass (`
    ///   write_pax_segment_compacted` emits the trained v3/A0 layout; the
    ///   condition self-clears because the output is trained). Measured
    ///   stake: untrained cold queries pay ~180 GETs/104 MB vs 20–51
    ///   GETs/24–30 MB trained (TD-FLUSH-4 adjudication evals).
    async fn should_trigger_compaction(
        &self,
        storage_url: &str,
        tags: &[String],
    ) -> Result<Option<usize>> {
        if !proximadb_storage_common::resolve_compaction_armed(tags) {
            return Ok(None);
        }
        let threshold =
            proximadb_storage_common::resolve_l0_threshold(tags, L0_COMPACTION_THRESHOLD);
        let files = self
            .discover_sstable_files(storage_url)
            .await
            .unwrap_or_default();
        if files.len() >= threshold {
            return Ok(Some(threshold));
        }
        // TD-COMPACT-5 training arm — only meaningful when the A0-emitting
        // trainer is enabled (`PROXIMADB_PAX_WRITE_A0_TRAIN`, default ON).
        // Without it compaction outputs stay layout v1, the untrained condition
        // never self-clears, and the arm would fire on every flush.
        if !crate::storage::engines::sst::block_cluster::ivf_probe_enabled() {
            return Ok(None);
        }
        // TD-COMPACT-8: skip re-arming while a training compaction is in-flight
        // for this collection (prevents the redundant re-training loop — up to
        // N-1 wasted cycles per ingest batch, each costing N GETs + PCA + PUT +
        // DELETEs). TD-COMPACT-6 D1 (async path): the flag is set by the flush
        // caller before `enqueue_due_compaction` and cleared by the background
        // worker once the compaction completes (covers the full queue→run
        // window); the flush caller also clears it on the no-worker paths
        // (nothing-due Ok(false), or enqueue Err).
        if self
            .training_in_flight
            .lock()
            .is_ok_and(|guard| guard.contains(storage_url))
        {
            tracing::debug!(
                "TD-COMPACT-8: training compaction already in-flight for \
                 {storage_url} — skipping arm"
            );
            return Ok(None);
        }
        // Best-effort header sniffing (72 bytes/file); any error means "not
        // yet", never fails the flush.
        let min_bytes = std::env::var("PROXIMADB_TRAINING_COMPACTION_MIN_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(32)
            .saturating_mul(1024 * 1024);
        for path in &files {
            if !path.url.ends_with(".pax") {
                continue;
            }
            if let Ok(true) = self.segment_is_untrained(&path.url, min_bytes).await {
                tracing::info!(
                    segment = %path.url,
                    "TD-COMPACT-5: untrained L0 segment above size floor — arming training compaction"
                );
                return Ok(Some(1));
            }
        }
        Ok(None)
    }

    /// TD-COMPACT-5: is this segment untrained (no A0 coarse directory) and
    /// big enough to be worth a training pass? One stat + one 72-byte read.
    async fn segment_is_untrained(&self, path: &str, min_bytes: u64) -> Result<bool> {
        use proximadb_storage_common::segment_layout::{
            SEG_HEADER_PREFIX_LEN, SegmentHeaderPrefix,
        };
        let fs = self.filesystem().get_filesystem(path)?;
        let meta = fs.metadata(path).await?;
        if meta.size < min_bytes {
            return Ok(false);
        }
        let header_bytes = fs
            .read_range(path, 0, (SEG_HEADER_PREFIX_LEN as u64).min(meta.size))
            .await?;
        Ok(match SegmentHeaderPrefix::parse(&header_bytes) {
            Ok(h) => h.a0_len == 0,
            // Not a coalesced segment (legacy layout) — training does not
            // apply; leave it to the count arm.
            Err(_) => false,
        })
    }
}

/// L0 segment count at which compaction arms. Mirrors
/// `OrchestratorCompactionConfig::level0_threshold` (default 5).
const L0_COMPACTION_THRESHOLD: usize = 5;

/// Statistics from vector sorting operation
// Using SortingStats from utils.rs instead of local SortStats
pub type SortStats = SortingStats;

/// ADR-037 / TD-174: fold the just-flushed *live* records into the resident
/// statistics summary and stamp the flush freshness watermark. Best-effort —
/// it never fails or blocks the flush, and observes records already in hand at
/// the write boundary (no separate scan; Decision 1). The generic
/// `observe_*` API lives in `core::statistics`; the record→value mapping is
/// engine-local (this layer owns `ProximaRecord`).
fn observe_flush_into_statistics(
    collection_id: &str,
    records: &[ProximaRecord],
    text_columns: &std::collections::HashSet<&str>,
) {
    if records.is_empty() {
        return;
    }
    // Real wall-clock "now" for liveness; a None fallback (impossible for current
    // dates) degrades to 0, which filters only tombstones — never panics.
    let now_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
    let registry = crate::core::statistics::statistics_registry();
    registry.update(collection_id, |summary| {
        for rec in records {
            // Invariant #16a: dead records (tombstoned / TTL-expired) must not
            // skew live statistics.
            if rec.is_dead(now_ns) {
                continue;
            }
            // Vector centroid/spread from the embedding payload(s).
            for cell in &rec.embeddings {
                let v = cell.values.to_fp32_owned();
                if !v.is_empty() {
                    summary.observe_vector(&v);
                }
            }
            // Walk the property tree once: text columns feed the document/BM25
            // corpus block (TD-175); every other scalar feeds per-field sketches.
            let mut doc_tokens: Vec<String> = Vec::new();
            for (name, node) in &rec.props {
                let proximadb_records::ProximaTreeNode::Value(pv) = node else {
                    continue;
                };
                if text_columns.contains(name.as_str()) {
                    if let Some(text) = scalar_text(pv) {
                        tokenize_into(text, &mut doc_tokens);
                    }
                } else if let Some((json, ty)) = scalar_proxima_to_json(pv) {
                    summary.observe_field(
                        name,
                        &serde_json::Value::String(ty.to_string()),
                        Some(&json),
                    );
                }
            }
            // One document per record (across all its text columns): term
            // frequency → top_terms + unique-terms HLL; doc length + distinct
            // terms → doc_count / avg_doc_length / document-frequency (idf).
            if !doc_tokens.is_empty() {
                for t in &doc_tokens {
                    summary.observe_term(t);
                }
                let mut distinct = doc_tokens.clone();
                distinct.sort();
                distinct.dedup();
                summary.observe_document(doc_tokens.len() as u64, &distinct);
            }
        }
        // Attest the freshness FACT: this snapshot is as-of the flush, now. A
        // watermark, never an SLA (the SLA is AnvaiOps policy, ADR-0021).
        summary.set_freshness(chrono::Utc::now().to_rfc3339(), "flush", None);
    });
}

/// Map a scalar `ProximaValue` to its (JSON value, canonical `ProximaType`
/// label) for field-level statistics. Non-scalar variants
/// (Array/Map/Struct/Json/Binary/vectors) and non-finite floats return `None` —
/// they carry no orderable field distribution.
fn scalar_proxima_to_json(
    pv: &proximadb_records::ProximaValue,
) -> Option<(serde_json::Value, &'static str)> {
    use proximadb_records::ProximaValue as V;
    use serde_json::Value as J;
    let out = match pv {
        V::Boolean(b) => (J::Bool(*b), "Boolean"),
        V::Int8(n) => (J::from(*n), "Int8"),
        V::Int16(n) => (J::from(*n), "Int16"),
        V::Int32(n) => (J::from(*n), "Int32"),
        V::Int64(n) => (J::from(*n), "Int64"),
        V::UInt8(n) => (J::from(*n), "UInt8"),
        V::UInt16(n) => (J::from(*n), "UInt16"),
        V::UInt32(n) => (J::from(*n), "UInt32"),
        V::UInt64(n) => (J::from(*n), "UInt64"),
        V::Float16(f) | V::Float32(f) => (
            J::Number(serde_json::Number::from_f64(*f as f64)?),
            "Float32",
        ),
        V::Float64(f) => (J::Number(serde_json::Number::from_f64(*f)?), "Float64"),
        V::Decimal(s) => (J::String(s.clone()), "Decimal"),
        V::String(s) | V::Symbol(s) => (J::String(s.clone()), "String"),
        V::Date(d) => (J::from(*d), "Date"),
        // Temporal epochs travel as Int64 so they get numeric min/max + quantiles.
        V::Time(t, _) | V::Timestamp(t, _) | V::TimestampTz(t, _) => (J::from(*t), "Int64"),
        _ => return None,
    };
    Some(out)
}

/// Borrow the textual payload of a scalar value (String/Symbol) for tokenization.
fn scalar_text(pv: &proximadb_records::ProximaValue) -> Option<&str> {
    use proximadb_records::ProximaValue as V;
    match pv {
        V::String(s) | V::Symbol(s) => Some(s.as_str()),
        _ => None,
    }
}

/// Word-level tokenizer for document/BM25 corpus statistics (ADR-037 TD-175):
/// lowercase, split on non-alphanumeric, drop empties and absurdly long tokens.
/// This is the BM25 **word** granularity (idf/df over terms), deliberately
/// distinct from the BERT subword `tokenizers::Tokenizer` used for
/// embeddings/reranking — different purpose, not a duplication.
fn tokenize_into(text: &str, out: &mut Vec<String>) {
    for raw in text.split(|c: char| !c.is_alphanumeric()) {
        if raw.is_empty() || raw.len() > 64 {
            continue;
        }
        out.push(raw.to_lowercase());
    }
}

// ---------------------------------------------------------------------------
// PAX vector-segment quant resolution (M1-3 / ADR-049): flush ALWAYS writes PAX;
// the kill-switch / opt-out below select the recall-exact RawF32 quant, not the
// retired legacy `.sst` streaming format.
// ---------------------------------------------------------------------------

/// Global kill-switch. Any truthy value selects the recall-exact `RawF32` quant
/// for EVERY collection (M1-3: it used to force legacy `.sst`; with the streaming
/// write path retired it now means RawF32-PAX, exact-scan searchable via
/// `search_pax_file_exact`). Follows the `PROXIMADB_DISABLE_*` convention
/// (`PROXIMADB_DISABLE_SYSTEM_CATALOG`, `PROXIMADB_DISABLE_WAL`).
const PAX_VECTOR_SEGMENTS_DISABLE_ENV: &str = "PROXIMADB_PAX_QUANT_DISABLE";

/// Tag key prefix encoding the per-collection PAX-format override on
/// `CollectionConfig.tags` — mirrors the `recall_target:` convention
/// (`services::collection::recall_target`). Stored as a tag rather than a typed
/// field because the proto-regen pipeline is manual (collection_types.proto).
const PAX_VECTOR_FORMAT_TAG_PREFIX: &str = "pax_vector_format:";

/// True when `key` is set to a truthy value (`1`/`true`/`on`/`yes`,
/// case-insensitive). Unset / any other value → false.
fn env_truthy(key: &str) -> bool {
    std::env::var(key)
        .map(|v| matches!(v.as_str(), "1" | "true" | "on" | "yes"))
        .unwrap_or(false)
}

/// Read the per-collection `pax_vector_format:on|off` tag from
/// `CollectionConfig.tags`. M1-3: `Some(true)` is redundant (PAX is always on);
/// `Some(false)` opts the collection to the recall-exact `RawF32` quant (it used
/// to force legacy `.sst`, but the streaming write path is retired — see
/// `resolve_pax_vector_quant`). Unrecognized values and absence → `None` (defer
/// to the default quant). The last matching tag wins, matching
/// `parse_recall_target`'s last-wins semantics.
fn pax_vector_format_tag(config: &crate::proto::proximadb_v1::CollectionConfig) -> Option<bool> {
    let mut latest: Option<bool> = None;
    for tag in &config.tags {
        if let Some(rest) = tag.strip_prefix(PAX_VECTOR_FORMAT_TAG_PREFIX) {
            latest = match rest.trim().to_ascii_lowercase().as_str() {
                "on" | "true" | "1" | "yes" => Some(true),
                "off" | "false" | "0" | "no" => Some(false),
                _ => latest, // unrecognized value: keep prior resolution
            };
        }
    }
    latest
}

/// Traverse `Collection.config` to read the per-collection PAX-format tag.
fn collection_pax_format_tag(
    collection: Option<&crate::proto::proximadb_v1::Collection>,
) -> Option<bool> {
    collection
        .as_ref()
        .and_then(|c| c.config.as_ref())
        .and_then(pax_vector_format_tag)
}

/// Resolve the PAX vector-quantization strategy. M1-3 (ADR-049): the global
/// kill-switch `PROXIMADB_PAX_QUANT_DISABLE` and the per-collection
/// `pax_vector_format:off` tag are no longer FORMAT escapes (flush always writes
/// PAX) — they select the recall-exact `RawF32` quant instead, so such a segment
/// is exact-scan searchable (`search_pax_file_exact`) rather than
/// RaBitQ-dequantized. Otherwise precedence: per-collection `pax_vector_quant`
/// (catalog config) > env `PROXIMADB_PAX_VECTOR_QUANT` > default `RaBitQ` (Phase
/// F: the cascade's stage-1 ranks RaBitQ-coded segments, so the default quant is
/// RaBitQ, not `Auto`). An unrecognized per-collection value falls back to `Auto`
/// (defensive against a typo'd config); only the deployment default (no config,
/// no env) is RaBitQ. The kill-switch / opt-out are mirrored here (not just at
/// the flush call site) so the precedence is unit-testable.
fn resolve_pax_vector_quant(
    collection_config: Option<&crate::proto::proximadb_v1::Collection>,
) -> proximadb_block_format::VectorQuant {
    use proximadb_block_format::VectorQuant;
    // M1-3: the legacy `.sst` escapes now mean RawF32-PAX (recall-exact).
    if env_truthy(PAX_VECTOR_SEGMENTS_DISABLE_ENV) {
        return VectorQuant::RawF32;
    }
    if collection_pax_format_tag(collection_config) == Some(false) {
        return VectorQuant::RawF32;
    }
    const ENV: &str = "PROXIMADB_PAX_VECTOR_QUANT";
    match collection_config
        .as_ref()
        .and_then(|c| c.config.as_ref())
        .and_then(|cfg| cfg.pax_vector_quant.as_deref())
        .map(|s| s.to_ascii_lowercase())
    {
        Some(s) => match s.as_str() {
            "rabitq" => VectorQuant::RaBitQ,
            "sq8" => VectorQuant::Sq8,
            "rawf32" | "raw_f32" | "raw" | "f32" => VectorQuant::RawF32,
            _ => VectorQuant::Auto,
        },
        None => match std::env::var(ENV)
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "rabitq" => VectorQuant::RaBitQ,
            "sq8" => VectorQuant::Sq8,
            "rawf32" | "raw_f32" | "raw" | "f32" => VectorQuant::RawF32,
            _ => VectorQuant::RaBitQ,
        },
    }
}

/// Tag prefix encoding the per-collection f32-tier opt-in on
/// `CollectionConfig.tags` — mirrors `pax_vector_format:`. `on`/`off`.
const PAX_F32_TIER_TAG_PREFIX: &str = "pax_f32_tier:";

/// Read the per-collection `pax_f32_tier:on|off` tag (last wins; an unrecognized
/// value keeps the prior resolution). Mirrors `pax_vector_format_tag`.
fn pax_f32_tier_tag(config: &crate::proto::proximadb_v1::CollectionConfig) -> Option<bool> {
    let mut latest: Option<bool> = None;
    for tag in &config.tags {
        if let Some(rest) = tag.strip_prefix(PAX_F32_TIER_TAG_PREFIX) {
            latest = match rest.trim().to_ascii_lowercase().as_str() {
                "on" | "true" | "1" | "yes" => Some(true),
                "off" | "false" | "0" | "no" => Some(false),
                _ => latest,
            };
        }
    }
    latest
}

/// Resolve whether a flush should ALSO emit the exact-f32 tier (P3 Phase D).
/// Precedence: per-collection `pax_f32_tier` tag > env `PROXIMADB_PAX_F32_TIER` >
/// default OFF. The tier is read lazily (exact final rerank / `include_vectors`),
/// so the only always-paid cost is the storage bytes — not scan/egress.
/// Tag prefix encoding the per-collection row-group Region D layout switch
/// (TD-PAXRG-1). Values: `on`, `off`.
const PAX_RG_LAYOUT_TAG_PREFIX: &str = "pax_rg_layout:";

/// Read the per-collection `pax_rg_layout:on|off` tag from a tag list.
fn pax_rg_layout_tag(tags: &[String]) -> Option<bool> {
    for tag in tags {
        if let Some(rest) = tag.strip_prefix(PAX_RG_LAYOUT_TAG_PREFIX) {
            return match rest.trim().to_ascii_lowercase().as_str() {
                "on" | "true" | "yes" | "1" => Some(true),
                "off" | "false" | "no" | "0" => Some(false),
                _ => None,
            };
        }
    }
    None
}

/// Resolve the row-group Region D write switch. Precedence: per-collection
/// `pax_rg_layout` tag > env `PROXIMADB_PAX_WRITE_RG_LAYOUT` > default OFF.
/// Only effective for RaBitQ-coalesced writes (Regions A/B are the premise).
pub(crate) fn resolve_pax_rg_layout(tags: &[String]) -> bool {
    let per_collection = pax_rg_layout_tag(tags);
    per_collection.unwrap_or_else(crate::storage::engines::sst::segment_format::rg_layout_enabled)
}

fn resolve_pax_f32_tier(
    collection_config: Option<&crate::proto::proximadb_v1::Collection>,
) -> bool {
    let per_collection = collection_config
        .as_ref()
        .and_then(|c| c.config.as_ref())
        .and_then(pax_f32_tier_tag);
    per_collection.unwrap_or_else(|| env_truthy("PROXIMADB_PAX_F32_TIER"))
}

/// TD-FPRUNE-1 (M1): build the P-Shred spec for the SST flush from the in-scope
/// collection config, so declared filterable tags are materialized as typed
/// user-columns (min/max + bloom + the self-describing footer field-map) —
/// making footer-resident filter pruning fire end-to-end for vector collections.
///
/// This reconstructs the SAME `(prop_name, col_id)` pairs the catalog schema
/// assigns (`collection_mapping::catalog_schema_from_collection`): `col_id =
/// 100 + idx` over `config.filterable_columns` (skipping empties), gated on
/// `enable_proxima_record == Some(true)`. It reads the proto config directly
/// (no `CatalogTableSchema` at the flush site) — and the read path resolves via
/// the segment's SELF-DESCRIBING footer field-map, so any consistent id works;
/// matching the catalog assignment is just tidy. Empty spec (non-document
/// collection, or no filterable columns) ⇒ byte-identical, mixed-read-safe.
fn resolve_flush_shred_spec(
    collection_config: Option<&crate::proto::proximadb_v1::Collection>,
) -> Vec<(String, i32)> {
    let Some(config) = collection_config.and_then(|c| c.config.as_ref()) else {
        return Vec::new();
    };
    if config.enable_proxima_record != Some(true) {
        return Vec::new();
    }
    let spec: Vec<(String, i32)> = config
        .filterable_columns
        .iter()
        .enumerate()
        .filter(|(_, col)| !col.name.is_empty())
        .map(|(idx, col)| (col.name.clone(), 100 + idx as i32))
        .collect();
    if !spec.is_empty() {
        tracing::debug!("TD-FPRUNE-1 M1: flush shred spec = {:?}", spec);
    }
    spec
}

/// Tag prefix encoding the per-collection tier-2 rerank quant strategy.
/// Mirrors the `pax_vector_quant:` convention. Values: `sq8`, `fp16`, `f32`.
const PAX_RERANK_QUANT_TAG_PREFIX: &str = "pax_rerank_quant:";

/// Read the per-collection `pax_rerank_quant:sq8|fp16|f32` tag from a tag list.
/// Takes `&[String]` (not the v1 config type) so this does not add a v1-proto
/// reference (TD-123 ratchet) — tag extraction happens at the call site.
fn pax_rerank_quant_tag(tags: &[String]) -> Option<proximadb_block_format::VectorQuant> {
    for tag in tags {
        if let Some(rest) = tag.strip_prefix(PAX_RERANK_QUANT_TAG_PREFIX) {
            return match rest.trim().to_ascii_lowercase().as_str() {
                "sq8" | "sq_8" => Some(proximadb_block_format::VectorQuant::Sq8),
                "fp16" | "f16" => Some(proximadb_block_format::VectorQuant::Fp16),
                "f32" | "raw" | "raw_f32" => Some(proximadb_block_format::VectorQuant::RawF32),
                _ => None,
            };
        }
    }
    None
}

/// Resolve the tier-2 rerank quant strategy. Precedence: per-collection
/// `pax_rerank_quant` tag > env `PROXIMADB_PAX_RERANK_QUANT` > default `Sq8`
/// (the validated tier-2). Only used when tier 1 is RaBitQ. Takes the resolved
/// tag slice (extracted by the caller) to avoid a v1-proto reference (TD-123).
fn resolve_pax_rerank_quant(tags: &[String]) -> proximadb_block_format::VectorQuant {
    const ENV: &str = "PROXIMADB_PAX_RERANK_QUANT";
    let per_collection = pax_rerank_quant_tag(tags);
    per_collection.unwrap_or_else(|| {
        match std::env::var(ENV)
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "fp16" | "f16" => proximadb_block_format::VectorQuant::Fp16,
            "f32" | "raw" | "raw_f32" => proximadb_block_format::VectorQuant::RawF32,
            _ => proximadb_block_format::VectorQuant::Sq8,
        }
    })
}

/// TD-RDSTRAT-5 S2: build an [`SstableWriteOutcome`] from a freshly-written PAX
/// segment's [`SegmentMeta`], so the Vector Object Economy directory emission can
/// fire with real per-block centroids. The PAX writer lives in a lower crate and
/// can't construct `IndexEntry` (defined here), so we assemble the entries at the
/// flush seam from the metadata the writer already returned:
///
/// * per-block **offset** = cumulative sum of `block_stats[..i].block_size_bytes`
///   (blocks are written contiguously from offset 0), **size** = that block's bytes;
/// * per-block **centroid** = `block_centroids[i]` (the S1 vector zone-map);
/// * **block_index_offset** = Σ block sizes (the segment index starts after the
///   last block), **block_index_size** = `size_bytes − Σ − len(SEGMENT_MAGIC)`.
///
/// Only the fields the directory consumes are set; the rest default (PAX carries
/// no key bloom / key ranges — those are the retired ProximaBlocks path's).
fn pax_write_outcome(
    meta: &proximadb_storage_common::pax_block::SegmentMeta,
) -> crate::storage::engines::sst::writer::SstableWriteOutcome {
    use crate::storage::engines::sst::{IndexEntry, VectorFormat};
    let mut entries = Vec::with_capacity(meta.block_stats.len());
    // ADR-062/065: in the coalesced layout the Region D data blocks begin after
    // the header-prefix + RaBitQ region (A) + SQ8 region (B)
    // (`rabitq_off + rabitq_len + sq8_len`); for the legacy layout all are 0, so
    // blocks start at offset 0 (unchanged). Block offsets recorded here are
    // absolute file offsets (the VOE directory / read path).
    let mut offset = meta.rabitq_off + meta.rabitq_len + meta.sq8_len;
    for (block_id, stats) in meta.block_stats.iter().enumerate() {
        let centroid = meta
            .block_centroids
            .get(block_id)
            .cloned()
            .unwrap_or_default();
        let vector_format = if centroid.is_empty() {
            VectorFormat::Variable
        } else {
            VectorFormat::Fixed {
                dimension: centroid.len(),
            }
        };
        // Lever-3: per-block RMS radius (0.0 when the writer didn't compute
        // centroids, or for a Fp32-less block) — 1:1 with block_centroids.
        let block_radius = meta.block_radii.get(block_id).copied().unwrap_or(0.0);
        entries.push(IndexEntry {
            offset,
            size: stats.block_size_bytes,
            block_id: block_id as u32,
            block_centroid: centroid,
            block_radius,
            vector_format,
            ..Default::default()
        });
        offset += stats.block_size_bytes as u64;
    }
    // `offset` is now Σ block sizes = where the segment index begins.
    let block_index_offset = offset;
    let magic = proximadb_storage_common::pax_block::SEGMENT_MAGIC.len() as u64;
    let block_index_size = meta.size_bytes.saturating_sub(block_index_offset + magic) as u32;
    crate::storage::engines::sst::writer::SstableWriteOutcome {
        index_entries: entries,
        block_index_offset,
        block_index_size,
        file_size_bytes: meta.size_bytes,
        record_count: meta.row_count,
    }
}

#[cfg(test)]
mod pax_write_outcome_tests {
    use super::pax_write_outcome;
    use proximadb_block_format::BlockStats;
    use proximadb_storage_common::pax_block::SegmentMeta;

    fn stats(size: u32) -> BlockStats {
        // `from_metas` with no column metas yields a BlockStats carrying just the
        // row/size framing this helper reads (BlockStats has no Default).
        BlockStats::from_metas(2, size, 0, 0, &[])
    }

    /// The outcome carries one IndexEntry per block with cumulative offsets, the
    /// S1 centroids threaded through, and a self-consistent segment-index frame
    /// (Σ sizes + index + 8-byte magic == file size).
    #[test]
    fn builds_entries_with_cumulative_offsets_and_centroids() {
        let meta = SegmentMeta {
            path: std::path::PathBuf::from("seg.pax"),
            size_bytes: 100 + 8 + 12, // Σ(40+60) blocks + 12 index + 8 magic
            block_count: 2,
            row_count: 4,
            block_stats: vec![stats(40), stats(60)],
            block_centroids: vec![vec![1.0, 1.0], vec![2.0, 2.0]],
            block_radii: vec![0.5, 1.5],
            rabitq_off: 0,
            rabitq_len: 0,
            sq8_off: 0,
            sq8_len: 0,
        };
        let out = pax_write_outcome(&meta);
        assert_eq!(out.index_entries.len(), 2);
        assert_eq!(out.index_entries[0].offset, 0);
        assert_eq!(out.index_entries[0].size, 40);
        assert_eq!(out.index_entries[1].offset, 40, "cumulative from block 0");
        assert_eq!(out.index_entries[1].size, 60);
        assert_eq!(out.index_entries[0].block_centroid, vec![1.0, 1.0]);
        assert_eq!(out.index_entries[1].block_centroid, vec![2.0, 2.0]);
        // TD-WLP-3: the per-block RMS radius is carried 1:1 into the entries.
        assert_eq!(out.index_entries[0].block_radius, 0.5);
        assert_eq!(out.index_entries[1].block_radius, 1.5);
        assert_eq!(out.block_index_offset, 100, "Σ block sizes");
        assert_eq!(out.block_index_size, 12, "size − Σ − 8 magic");
        assert_eq!(out.file_size_bytes, 120);
        assert_eq!(out.record_count, 4);
    }
}

#[cfg(test)]
mod statistics_observe_tests {
    use super::{scalar_text, tokenize_into};
    use proximadb_records::ProximaValue;

    #[test]
    fn tokenizer_lowercases_and_splits_on_non_alphanumeric() {
        let mut out = Vec::new();
        tokenize_into("Checkout 500s on payment-submit!", &mut out);
        assert_eq!(out, vec!["checkout", "500s", "on", "payment", "submit"]);
    }

    #[test]
    fn tokenizer_drops_empties_and_overlong_tokens() {
        let mut out = Vec::new();
        let long = "x".repeat(100);
        tokenize_into(&format!("ok   {long} fine"), &mut out);
        assert_eq!(out, vec!["ok", "fine"]);
    }

    #[test]
    fn scalar_text_only_for_string_like() {
        assert_eq!(scalar_text(&ProximaValue::String("hi".into())), Some("hi"));
        assert_eq!(
            scalar_text(&ProximaValue::Symbol("sym".into())),
            Some("sym")
        );
        assert_eq!(scalar_text(&ProximaValue::Int64(7)), None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::engines::sst::SstConfig;
    use crate::storage::persistence::filesystem::FilesystemFactory;
    use proximadb_distance_kernel::engine::UnifiedDistanceCompute;
    use std::sync::Arc;

    // TD-FPRUNE-1 M1: the flush-side shred spec reconstructs the SAME
    // `(name, 100+idx)` pairs the catalog schema assigns, gated on
    // `enable_proxima_record`. This is what makes footer filter-pruning fire for
    // vector collections (the write path previously shredded nothing).
    #[test]
    fn resolve_flush_shred_spec_from_filterable_columns() {
        use crate::proto::proximadb_v1::{Collection, CollectionConfig, FilterableColumnSpec};
        let col = |name: &str| FilterableColumnSpec {
            name: name.to_string(),
            ..Default::default()
        };
        let mk = |enable: Option<bool>, cols: Vec<FilterableColumnSpec>| Collection {
            config: Some(CollectionConfig {
                filterable_columns: cols,
                enable_proxima_record: enable,
                ..Default::default()
            }),
            ..Default::default()
        };

        // enable_proxima_record + two filterable columns → (name, 100+idx).
        let c = mk(Some(true), vec![col("partition"), col("lang")]);
        assert_eq!(
            resolve_flush_shred_spec(Some(&c)),
            vec![("partition".to_string(), 100), ("lang".to_string(), 101)]
        );
        // Empty column names are skipped but still advance the index (mirrors
        // catalog_schema_from_collection's enumerate-then-skip).
        let c = mk(Some(true), vec![col(""), col("lang")]);
        assert_eq!(
            resolve_flush_shred_spec(Some(&c)),
            vec![("lang".to_string(), 101)]
        );
        // Not a document collection ⇒ no shredding (byte-identical write).
        assert!(
            resolve_flush_shred_spec(Some(&mk(Some(false), vec![col("partition")]))).is_empty()
        );
        assert!(resolve_flush_shred_spec(Some(&mk(None, vec![col("partition")]))).is_empty());
        // No config / no collection ⇒ empty.
        assert!(resolve_flush_shred_spec(Some(&Collection::default())).is_empty());
        assert!(resolve_flush_shred_spec(None).is_empty());
    }

    #[tokio::test]
    async fn test_sort_vectors_for_sstable_encoding() {
        let engine = create_test_engine().await;

        let vectors = vec![
            create_test_vector("vector_3", vec![3.0, 4.0]),
            create_test_vector("vector_1", vec![1.0, 2.0]),
            create_test_vector("vector_2", vec![2.0, 3.0]),
        ];

        let (sorted, stats) = engine
            .sort_vectors_for_sstable_encoding(vectors)
            .await
            .unwrap();

        // Convert to tuples and verify sorting
        let sorted_tuples: Vec<(String, ProximaRecord)> = sorted
            .into_iter()
            .map(|record| (record.oid.clone(), record))
            .collect();
        assert_eq!(sorted_tuples[0].0, "vector_1");
        assert_eq!(sorted_tuples[1].0, "vector_2");
        assert_eq!(sorted_tuples[2].0, "vector_3");

        // Verify stats
        assert_eq!(stats.records_sorted, 3);
        assert!(stats.compression_estimate > 0.0);
    }

    async fn create_test_engine() -> SstEngine {
        let config = SstConfig::default();
        create_test_engine_with_config(config).await
    }

    async fn create_test_engine_with_config(config: SstConfig) -> SstEngine {
        let filesystem_config =
            crate::storage::persistence::filesystem::FilesystemConfig::default();
        let filesystem = Arc::new(FilesystemFactory::create(filesystem_config).await.unwrap());
        let distance_compute = Arc::new(UnifiedDistanceCompute::default());

        SstEngine::new_with_config(config, filesystem, distance_compute)
            .await
            .unwrap()
    }

    fn create_test_vector(id: &str, vector: Vec<f32>) -> ProximaRecord {
        ProximaRecord {
            oid: id.to_string(),
            created_at_ns: 12_345_000_000,
            updated_at_ns: 12_345_000_000,
            record_version: 1,
            embeddings: vec![proximadb_records::EmbeddingCell {
                model_id: "test".to_string(),
                modality: "dense_vector".to_string(),
                dim: vector.len() as u32,
                values: proximadb_records::EmbeddingValues::Fp32(vector),
                ..Default::default()
            }],
            ..ProximaRecord::default()
        }
    }

    #[test]
    fn flush_storage_path_matches_search_layout_policy() {
        use crate::proto::proximadb_v1::{Collection, StorageAssignment};
        use crate::storage::trait_components::path_resolver::{
            collection_data_path_typed, typed_identity_from_storage_assignment, typed_path_identity,
        };

        let assignment = StorageAssignment {
            base_location: "file:///base".to_string(),
            typed_account_id: Some(7),
            typed_namespace_id: Some(11),
            typed_collection_id: Some(13),
            ..Default::default()
        };
        let params = FlushParameters {
            collection_id: Some("13".to_string()),
            collection_config: Some(Collection {
                id: "13".to_string(),
                storage_assignment: Some(assignment.clone()),
                ..Default::default()
            }),
            ..Default::default()
        };

        let expected = collection_data_path_typed(
            &assignment.base_location,
            "13",
            typed_path_identity(typed_identity_from_storage_assignment(Some(&assignment))),
        );
        assert_eq!(
            SstEngine::get_collection_storage_url_from_params(&params).unwrap(),
            expected
        );
    }

    #[test]
    fn flush_storage_path_preserves_legacy_collection_layout() {
        use crate::proto::proximadb_v1::{Collection, StorageAssignment};

        let assignment = StorageAssignment {
            base_location: "file:///base".to_string(),
            ..Default::default()
        };
        let params = FlushParameters {
            collection_id: Some("legacy-id".to_string()),
            collection_config: Some(Collection {
                id: "legacy-id".to_string(),
                storage_assignment: Some(assignment.clone()),
                ..Default::default()
            }),
            ..Default::default()
        };

        assert_eq!(
            SstEngine::get_collection_storage_url_from_params(&params).unwrap(),
            proximadb_storage_common::storage_path::StoragePath::collection_data_path(
                &assignment.base_location,
                "legacy-id"
            )
        );
    }

    #[tokio::test]
    async fn adr081_l0_stop_rejects_before_sort_and_ignores_stable_levels() {
        use crate::proto::proximadb_v1::{
            Collection, CollectionConfig, StorageAssignment, StorageEngine,
        };
        use crate::storage::traits::UnifiedStorageFormat;

        let mut config = SstConfig::default();
        config.l0_stop_trigger = 1;
        config.l0_slowdown_trigger = 0;
        config.background_thread_count = 1;
        let engine = create_test_engine_with_config(config).await;
        let temp_dir = tempfile::TempDir::new().unwrap();
        let collection_id = "adr081_preflight";
        let data_dir = proximadb_storage_common::storage_path::StoragePath::collection_data_path(
            temp_dir.path().to_string_lossy().as_ref(),
            collection_id,
        );
        std::fs::create_dir_all(&data_dir).unwrap();
        std::fs::write(
            std::path::Path::new(&data_dir).join("L1_20260729T000000_stable.pax"),
            b"stable",
        )
        .unwrap();

        let collection = Collection {
            id: collection_id.to_string(),
            config: Some(CollectionConfig {
                name: collection_id.to_string(),
                dimension: 2,
                storage_engine: Some(StorageEngine::Sst as i32),
                ..Default::default()
            }),
            storage_assignment: Some(StorageAssignment {
                base_location: temp_dir.path().to_string_lossy().into_owned(),
                engine: StorageEngine::Sst as i32,
                ..Default::default()
            }),
            ..Default::default()
        };
        let params = FlushParameters {
            collection_id: Some(collection_id.to_string()),
            vector_records: vec![create_test_vector("v0", vec![1.0, 0.0])],
            force: true,
            synchronous: true,
            collection_config: Some(collection),
            ..Default::default()
        };

        engine
            .preflight_flush(&params)
            .await
            .expect("a stable L1 segment must not consume the L0 admission budget");
        std::fs::write(
            std::path::Path::new(&data_dir).join("L0_20260729T000001_blocked.pax"),
            b"l0",
        )
        .unwrap();

        let before = engine
            .flush_sort_invocations
            .load(std::sync::atomic::Ordering::Relaxed);
        let error = engine
            .do_flush(&params)
            .await
            .expect_err("L0 stop watermark must reject the flush");
        let after = engine
            .flush_sort_invocations
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            error.to_string().contains("L0 admission stop"),
            "unexpected rejection: {error:#}"
        );
        assert_eq!(
            after, before,
            "admission rejection must occur before vector sorting"
        );
    }

    /// TD-112: the LIVE flush path (`do_flush` -> `flush_implementation`) must
    /// register flushed vectors into the per-collection AXIS index, so post-flush
    /// search is served by the ANN index instead of a brute-force segment scan.
    /// (The `FlushCoordinator` is test-only scaffolding; the hook lives on the
    /// real path exercised here.)
    #[cfg(feature = "axis")]
    #[tokio::test]
    async fn td112_live_flush_indexes_vectors_into_axis() {
        use crate::index::axis::management::manager::AxisManager;
        use crate::proto::proximadb_v1::{
            Collection, CollectionConfig, StorageAssignment, StorageEngine,
        };
        use crate::storage::traits::UnifiedStorageEngine;
        use proximadb_distance_kernel::DistanceMetric;
        use proximadb_index_types::AxisConfig;

        // Attach an AXIS manager (process-global OnceLock; a no-op if a prior test
        // already set one). We read the effective manager back from the engine so
        // the assertion targets whatever the live flush path will use.
        crate::storage::engines::sst::core::set_sst_axis_manager(Arc::new(
            AxisManager::new(AxisConfig::default()).await.unwrap(),
        ));
        let engine = create_test_engine().await;
        let axis = engine
            .axis_manager()
            .expect("an AXIS manager must be attached for index-on-flush");

        let temp_dir = tempfile::TempDir::new().unwrap();
        let collection_id = "td112_live_flush";
        let collection = Collection {
            id: collection_id.to_string(),
            config: Some(CollectionConfig {
                name: collection_id.to_string(),
                dimension: 4,
                distance_metric: Some(DistanceMetric::Cosine as i32),
                storage_engine: Some(StorageEngine::Sst as i32),
                // This test asserts flush indexes into AXIS, so declare an
                // explicit collection index. Format tags are not index intent.
                index_configs: vec![crate::proto::proximadb_v1::IndexConfig::default()],
                ..Default::default()
            }),
            storage_assignment: Some(StorageAssignment {
                base_location: temp_dir.path().to_str().unwrap().to_string(),
                ..Default::default()
            }),
            ..Default::default()
        };

        let records = vec![
            create_test_vector("v0", vec![1.0, 0.0, 0.0, 0.0]),
            create_test_vector("v1", vec![0.0, 1.0, 0.0, 0.0]),
            create_test_vector("v2", vec![0.0, 0.0, 1.0, 0.0]),
        ];

        let params = FlushParameters {
            collection_id: Some(collection_id.to_string()),
            vector_records: records,
            force: true,
            synchronous: true,
            hints: std::collections::HashMap::new(),
            timeout_ms: None,
            trigger_compaction: false,
            batch_ids: vec![],
            collection_config: Some(collection),
            estimated_size: 0,
        };

        let result = engine
            .do_flush(&params)
            .await
            .expect("flush should succeed");
        assert!(result.success, "flush should succeed");

        assert_eq!(
            axis.registered_vector_count(collection_id).await,
            3,
            "the live flush path must index all flushed vectors into AXIS (TD-112)"
        );
    }

    #[tokio::test]
    async fn recovery_materialization_is_idempotent() {
        use crate::proto::proximadb_v1::{
            Collection, CollectionConfig, StorageAssignment, StorageEngine,
        };
        use crate::storage::traits::UnifiedStorageFormat;

        let engine = create_test_engine().await;
        let temp_dir = tempfile::TempDir::new().unwrap();
        let collection_id = "recovery_idempotence";
        let collection = Collection {
            id: collection_id.to_string(),
            config: Some(CollectionConfig {
                name: collection_id.to_string(),
                dimension: 4,
                storage_engine: Some(StorageEngine::Sst as i32),
                ..Default::default()
            }),
            storage_assignment: Some(StorageAssignment {
                base_location: temp_dir.path().to_string_lossy().into_owned(),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut hints = std::collections::HashMap::new();
        hints.insert(
            "recovery_materialization_id".to_string(),
            serde_json::Value::String("00000000000000000001-00000000000000000007-digest".into()),
        );
        hints.insert(
            "recovery_content_digest".to_string(),
            serde_json::Value::String("digest".into()),
        );
        hints.insert(
            "suppress_compaction_until_wal_retired".to_string(),
            serde_json::Value::Bool(true),
        );
        let params = FlushParameters {
            collection_id: Some(collection_id.to_string()),
            vector_records: vec![
                create_test_vector("v0", vec![1.0, 0.0, 0.0, 0.0]),
                create_test_vector("v1", vec![0.0, 1.0, 0.0, 0.0]),
            ],
            force: true,
            synchronous: true,
            hints,
            timeout_ms: None,
            trigger_compaction: false,
            batch_ids: vec![],
            collection_config: Some(collection),
            estimated_size: 0,
        };

        let first = engine
            .do_flush(&params)
            .await
            .expect("first materialization");
        assert_eq!(first.files_created, Some(1));

        let replay = engine.do_flush(&params).await.expect("idempotent replay");
        assert_eq!(replay.files_created, Some(0));
        assert_eq!(
            replay
                .engine_metrics
                .get("already_materialized")
                .and_then(serde_json::Value::as_bool),
            Some(true)
        );
        assert_eq!(replay.file_paths, first.file_paths);
    }

    // ---- Phase C: per-collection PAX opt-out tag + global kill-switch ----

    #[test]
    fn pax_vector_format_tag_parses_on_off_and_aliases() {
        use crate::proto::proximadb_v1::CollectionConfig;
        let cfg = |tags: &[&str]| CollectionConfig {
            tags: tags.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        };
        // canonical on/off
        assert_eq!(
            pax_vector_format_tag(&cfg(&["pax_vector_format:on"])),
            Some(true)
        );
        assert_eq!(
            pax_vector_format_tag(&cfg(&["pax_vector_format:off"])),
            Some(false)
        );
        // accepted aliases (case-insensitive, whitespace-trimmed)
        assert_eq!(
            pax_vector_format_tag(&cfg(&["pax_vector_format: TRUE"])),
            Some(true)
        );
        assert_eq!(
            pax_vector_format_tag(&cfg(&["pax_vector_format:0"])),
            Some(false)
        );
        assert_eq!(
            pax_vector_format_tag(&cfg(&["pax_vector_format:Yes"])),
            Some(true)
        );
        assert_eq!(
            pax_vector_format_tag(&cfg(&["pax_vector_format:no"])),
            Some(false)
        );
        // unrecognized value / unrelated tag / absent → None (defer to global)
        assert_eq!(
            pax_vector_format_tag(&cfg(&["pax_vector_format:maybe"])),
            None
        );
        assert_eq!(pax_vector_format_tag(&cfg(&["recall_target:0.95"])), None);
        assert_eq!(pax_vector_format_tag(&cfg(&[])), None);
        // last matching tag wins
        assert_eq!(
            pax_vector_format_tag(&cfg(&["pax_vector_format:on", "pax_vector_format:off"])),
            Some(false)
        );
    }

    fn collect_exts(dir: &std::path::Path, out: &mut std::collections::HashSet<String>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_exts(&path, out);
            } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                out.insert(ext.to_string());
            }
        }
    }

    /// Flush `records` for a collection carrying `tags` and return the set of
    /// segment-file extensions written under `base_location`.
    async fn flush_segment_exts(
        engine: &SstEngine,
        collection_id: &str,
        base_location: &str,
        tags: Vec<String>,
        records: Vec<ProximaRecord>,
    ) -> std::collections::HashSet<String> {
        use crate::proto::proximadb_v1::{
            Collection, CollectionConfig, StorageAssignment, StorageEngine,
        };
        use crate::storage::traits::UnifiedStorageFormat;
        let collection = Collection {
            id: collection_id.to_string(),
            config: Some(CollectionConfig {
                name: collection_id.to_string(),
                dimension: 4,
                storage_engine: Some(StorageEngine::Sst as i32),
                tags,
                ..Default::default()
            }),
            storage_assignment: Some(StorageAssignment {
                base_location: base_location.to_string(),
                ..Default::default()
            }),
            ..Default::default()
        };
        let params = FlushParameters {
            collection_id: Some(collection_id.to_string()),
            vector_records: records,
            force: true,
            synchronous: true,
            hints: std::collections::HashMap::new(),
            timeout_ms: None,
            trigger_compaction: false,
            batch_ids: vec![],
            collection_config: Some(collection),
            estimated_size: 0,
        };
        let result = engine
            .do_flush(&params)
            .await
            .expect("flush should succeed");
        assert!(result.success, "flush should succeed");

        let mut exts = std::collections::HashSet::new();
        collect_exts(std::path::Path::new(base_location), &mut exts);
        assert!(
            !exts.is_empty(),
            "expected at least one flushed segment under {base_location}"
        );
        exts
    }

    /// Staged adoption: with no PAX env set, a `pax_vector_format:on` tag opts
    /// the collection INTO PAX — the flush emits a `.pax` segment.
    #[tokio::test]
    async fn pax_optin_tag_writes_pax_segment_without_env() {
        // nextest isolates each test in its own process; `set_var`/`remove_var`
        // are `unsafe` (edition 2024).
        unsafe {
            std::env::remove_var(PAX_VECTOR_SEGMENTS_DISABLE_ENV);
        }
        let engine = create_test_engine().await;
        let temp_dir = tempfile::TempDir::new().unwrap();
        let base = temp_dir.path().to_str().unwrap();
        let records = vec![
            create_test_vector("v0", vec![1.0, 0.0, 0.0, 0.0]),
            create_test_vector("v1", vec![0.0, 1.0, 0.0, 0.0]),
        ];
        let exts = flush_segment_exts(
            &engine,
            "pax_optin_tag",
            base,
            vec!["pax_vector_format:on".to_string()],
            records,
        )
        .await;
        assert!(
            exts.contains("pax"),
            "opt-in tag should write a .pax segment (got {exts:?})"
        );
    }

    /// Global kill-switch selects the recall-exact RawF32-PAX quant — M1-3: it
    /// no longer forces legacy `.sst` (the streaming write path is retired); the
    /// flushed segment is still `.pax`, just RawF32-coded so `search_pax_file_exact`
    /// scans it exactly. The (now-redundant) opt-in tag is kept here to prove the
    /// kill-switch still wins the quant decision.
    #[tokio::test]
    async fn pax_kill_switch_writes_rawf32_pax_segment() {
        unsafe {
            std::env::set_var(PAX_VECTOR_SEGMENTS_DISABLE_ENV, "1");
        }
        let engine = create_test_engine().await;
        let temp_dir = tempfile::TempDir::new().unwrap();
        let base = temp_dir.path().to_str().unwrap();
        let records = vec![
            create_test_vector("v0", vec![1.0, 0.0, 0.0, 0.0]),
            create_test_vector("v1", vec![0.0, 1.0, 0.0, 0.0]),
        ];
        let exts = flush_segment_exts(
            &engine,
            "pax_kill_switch",
            base,
            vec!["pax_vector_format:on".to_string()],
            records,
        )
        .await;
        assert!(
            exts.contains("pax") && !exts.contains("sst"),
            "kill-switch must still write a .pax segment (RawF32-PAX), not legacy .sst (got {exts:?})"
        );
        unsafe {
            std::env::remove_var(PAX_VECTOR_SEGMENTS_DISABLE_ENV);
        }
    }

    /// A per-collection `pax_vector_format:off` tag opts the collection to the
    /// recall-exact RawF32-PAX quant — M1-3: it no longer forces legacy `.sst`.
    #[tokio::test]
    async fn pax_optout_tag_writes_rawf32_pax_segment() {
        unsafe {
            std::env::remove_var(PAX_VECTOR_SEGMENTS_DISABLE_ENV);
        }
        let engine = create_test_engine().await;
        let temp_dir = tempfile::TempDir::new().unwrap();
        let base = temp_dir.path().to_str().unwrap();
        let records = vec![
            create_test_vector("v0", vec![1.0, 0.0, 0.0, 0.0]),
            create_test_vector("v1", vec![0.0, 1.0, 0.0, 0.0]),
        ];
        let exts = flush_segment_exts(
            &engine,
            "pax_optout_tag",
            base,
            vec!["pax_vector_format:off".to_string()],
            records,
        )
        .await;
        assert!(
            exts.contains("pax") && !exts.contains("sst"),
            "opt-out tag should write a RawF32-PAX .pax segment, not legacy .sst (got {exts:?})"
        );
    }

    /// Quant resolution. M1-3: kill-switch / `pax_vector_format:off` → RawF32
    /// (recall-exact escape); else per-collection `pax_vector_quant` > env > default
    /// RaBitQ (Phase F: the cascade's stage-1 ranks RaBitQ-coded segments). An
    /// unrecognized per-collection value falls back to `Auto` (defensive), NOT the
    /// RaBitQ default; only the deployment default (no config, no env) is RaBitQ.
    #[test]
    fn resolve_pax_vector_quant_default_and_precedence() {
        use crate::proto::proximadb_v1::{Collection, CollectionConfig};
        use proximadb_block_format::VectorQuant;
        unsafe {
            std::env::remove_var("PROXIMADB_PAX_VECTOR_QUANT");
            std::env::remove_var(PAX_VECTOR_SEGMENTS_DISABLE_ENV);
        }
        let none: Option<&Collection> = None;
        // default → RaBitQ
        assert_eq!(resolve_pax_vector_quant(none), VectorQuant::RaBitQ);
        // env overrides the default
        unsafe {
            std::env::set_var("PROXIMADB_PAX_VECTOR_QUANT", "sq8");
        }
        assert_eq!(resolve_pax_vector_quant(none), VectorQuant::Sq8);
        unsafe {
            std::env::set_var("PROXIMADB_PAX_VECTOR_QUANT", "rawf32");
        }
        assert_eq!(resolve_pax_vector_quant(none), VectorQuant::RawF32);
        unsafe {
            std::env::remove_var("PROXIMADB_PAX_VECTOR_QUANT");
        }
        // per-collection overrides env + default
        let coll_sq8 = Collection {
            config: Some(CollectionConfig {
                pax_vector_quant: Some("sq8".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(resolve_pax_vector_quant(Some(&coll_sq8)), VectorQuant::Sq8);
        // M1-3: per-collection rawf32 config is honored.
        let coll_raw = Collection {
            config: Some(CollectionConfig {
                pax_vector_quant: Some("raw_f32".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(
            resolve_pax_vector_quant(Some(&coll_raw)),
            VectorQuant::RawF32
        );
        let coll_rabitq = Collection {
            config: Some(CollectionConfig {
                pax_vector_quant: Some("rabitq".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(
            resolve_pax_vector_quant(Some(&coll_rabitq)),
            VectorQuant::RaBitQ
        );
        // unrecognized per-collection value → Auto (defensive), NOT the RaBitQ default
        let coll_bad = Collection {
            config: Some(CollectionConfig {
                pax_vector_quant: Some("nonsense".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(resolve_pax_vector_quant(Some(&coll_bad)), VectorQuant::Auto);
        // M1-3: the kill-switch selects RawF32 even when a collection asks for
        // RaBitQ (the recall-exact escape outranks the per-collection quant).
        unsafe {
            std::env::set_var(PAX_VECTOR_SEGMENTS_DISABLE_ENV, "1");
        }
        assert_eq!(
            resolve_pax_vector_quant(Some(&coll_rabitq)),
            VectorQuant::RawF32
        );
        unsafe {
            std::env::remove_var(PAX_VECTOR_SEGMENTS_DISABLE_ENV);
        }
        // M1-3: a per-collection `pax_vector_format:off` tag also selects RawF32.
        let coll_optout = Collection {
            config: Some(CollectionConfig {
                tags: vec!["pax_vector_format:off".into()],
                pax_vector_quant: Some("rabitq".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(
            resolve_pax_vector_quant(Some(&coll_optout)),
            VectorQuant::RawF32
        );
    }

    /// Tier-2 rerank quant default is SQ8 (the validated tier-2). Precedence:
    /// per-collection `pax_rerank_quant` tag > env `PROXIMADB_PAX_RERANK_QUANT` >
    /// default SQ8. Only the 3 valid values (sq8/fp16/f32) are accepted;
    /// anything else falls back to the default.
    #[test]
    fn resolve_pax_rerank_quant_default_and_precedence() {
        use proximadb_block_format::VectorQuant;
        unsafe {
            std::env::remove_var("PROXIMADB_PAX_RERANK_QUANT");
        }
        // no tags → default Sq8
        assert_eq!(resolve_pax_rerank_quant(&[]), VectorQuant::Sq8);
        // env overrides the default
        unsafe {
            std::env::set_var("PROXIMADB_PAX_RERANK_QUANT", "fp16");
        }
        assert_eq!(resolve_pax_rerank_quant(&[]), VectorQuant::Fp16);
        unsafe {
            std::env::set_var("PROXIMADB_PAX_RERANK_QUANT", "f32");
        }
        assert_eq!(resolve_pax_rerank_quant(&[]), VectorQuant::RawF32);
        unsafe {
            std::env::remove_var("PROXIMADB_PAX_RERANK_QUANT");
        }
        // per-collection tag overrides env + default
        let tags_fp16 = ["pax_rerank_quant:fp16".to_string()];
        assert_eq!(resolve_pax_rerank_quant(&tags_fp16), VectorQuant::Fp16);
        let tags_f32 = ["pax_rerank_quant:f32".to_string()];
        assert_eq!(resolve_pax_rerank_quant(&tags_f32), VectorQuant::RawF32);
        let tags_sq8 = ["pax_rerank_quant:sq8".to_string()];
        assert_eq!(resolve_pax_rerank_quant(&tags_sq8), VectorQuant::Sq8);
        // unrecognized tag value → None → default Sq8 (RaBitQ rerank is invalid
        // and rejected at the parser: the tag only accepts sq8/fp16/f32).
        let tags_bad = ["pax_rerank_quant:rabitq".to_string()];
        assert_eq!(resolve_pax_rerank_quant(&tags_bad), VectorQuant::Sq8);
    }

    /// TD-PAXRG-1 (post-flip): the row-group Region D resolver — default ON,
    /// `PROXIMADB_PAX_WRITE_RG_LAYOUT=0` kill-switch, per-collection
    /// `pax_rg_layout:on|off` tag outranks the env, and an unrecognized tag
    /// value falls through to the env.
    #[test]
    fn resolve_pax_rg_layout_default_and_precedence() {
        use crate::storage::engines::sst::segment_format::rg_layout_enabled;
        const ENV: &str = "PROXIMADB_PAX_WRITE_RG_LAYOUT";

        // Default ON (flipped 2026-08-28 on the TD-PAXRG-1 Phase-G evidence).
        unsafe { std::env::remove_var(ENV) };
        assert!(resolve_pax_rg_layout(&[]));

        // Kill-switch.
        unsafe { std::env::set_var(ENV, "0") };
        assert!(!resolve_pax_rg_layout(&[]));

        // Per-collection tag outranks the env — including ON against env OFF.
        let tags_on = ["pax_rg_layout:on".to_string()];
        assert!(resolve_pax_rg_layout(&tags_on));
        let tags_off = ["pax_rg_layout:off".to_string()];
        assert!(!resolve_pax_rg_layout(&tags_off));

        // Unrecognized tag value falls through to the kill-switch env.
        let tags_bad = ["pax_rg_layout:maybe".to_string()];
        assert!(!resolve_pax_rg_layout(&tags_bad));

        // And with the env removed the default-ON holds.
        unsafe { std::env::remove_var(ENV) };
        assert!(resolve_pax_rg_layout(&tags_on));
        assert!(resolve_pax_rg_layout(&[]));
        assert!(rg_layout_enabled());
    }
}
