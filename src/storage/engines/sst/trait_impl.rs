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

//! SST Engine Trait Implementations
//!
//! Contains the main trait implementations for the SST engine,
//! delegating to the appropriate modules for actual functionality.

use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use tracing::{debug, info};

use crate::core::search::results::OptimizedSearchRecord;
use crate::storage::engines::sst::core::SstEngine;
use crate::storage::engines::sst::{BACKGROUND_COMPACTION_HINT, COLLECTION_META_COMPLETE_HINT};
use crate::storage::traits::{
    CompactionParameters, CompactionResult, FlushParameters, FlushResult, StorageFormatStrategy,
    StorageQueryContext, UnifiedStorageFormat,
};

#[async_trait]
impl UnifiedStorageFormat for SstEngine {
    fn engine_name(&self) -> &'static str {
        "sst"
    }

    fn engine_version(&self) -> &'static str {
        crate::version::PROXIMADB_VERSION
    }

    fn strategy(&self) -> StorageFormatStrategy {
        StorageFormatStrategy::Sst
    }

    async fn preflight_flush(&self, params: &FlushParameters) -> Result<()> {
        self.preflight_flush_implementation(params).await
    }

    async fn do_flush(&self, params: &FlushParameters) -> Result<FlushResult> {
        info!("🚀 SST: Starting flush operation");
        // Use the flush module implementation directly
        self.flush_implementation(params).await
    }

    /// TD-DELVEC-1 WI-5 P1: re-mark deletion-vector bits for recovery tombstones
    /// (see `UnifiedStorageFormat::reconcile_deletion_vectors`). The just-flushed
    /// `L0_recovery_*` segments exist on disk, so `resolve_oid_positions` can find
    /// each tombstone's target row; the mark persists to `{segment}.dv` (disk-
    /// authoritative, read by the canonical serving engine). Best-effort — a
    /// resolve/mark failure logs + continues; the tombstone stays coherent.
    #[cfg(feature = "cold-deletion-vectors")]
    async fn reconcile_deletion_vectors(
        &self,
        collection_id: &str,
        tombstones: &[(String, u64)],
    ) -> Result<()> {
        let Some(dv_store) = self.deletion_vector_store.as_ref() else {
            return Ok(());
        };
        for (oid, lsn) in tombstones {
            let hits = match self.resolve_oid_positions(collection_id, oid).await {
                Ok(hits) => hits,
                Err(e) => {
                    tracing::warn!("recovery DV reconcile: resolve failed for {oid}: {e:?}");
                    continue;
                }
            };
            for (seg, pos) in hits {
                if let Err(e) = dv_store.mark_deleted(&seg, pos, *lsn).await {
                    tracing::warn!(
                        "recovery DV reconcile: mark failed for {oid} @ {seg}:{pos} (gen {lsn}): {e:?}"
                    );
                }
            }
        }
        Ok(())
    }

    /// SST's LSM bulk-load override (Phase 2F-b).
    ///
    /// Same shape as NOVA's override — SST's `flush_implementation`
    /// already takes `FlushParameters { vector_records, collection_config, .. }`
    /// and writes a single SSTable file via the existing writer, so
    /// the trait method just builds a synthetic params and delegates.
    /// WAL + memtable are bypassed because we never call into the
    /// per-record insert path.
    ///
    /// SST adds quantization on top of NOVA's path (Binary → INT8 →
    /// FP32 progressive search), so bulk-loaded segments inherit the
    /// same hierarchical bloom filters and progressive quantization
    /// the normal flush produces.
    async fn ingest_sorted_segment(
        &self,
        collection_id: &str,
        base_path: &str,
        records: Vec<proximadb_records::ProximaRecord>,
    ) -> Result<crate::storage::traits::SegmentIngestResult> {
        use crate::storage::engines::core::ops::bulk_ingest::{
            build_bulk_flush_parameters, empty_segment_result, segment_result_from_flush,
        };

        let count = records.len();
        if count == 0 {
            return Ok(empty_segment_result(collection_id));
        }

        let params = build_bulk_flush_parameters(collection_id, base_path, records, "SST")?;
        let flush_result = self.flush_implementation(&params).await?;
        Ok(segment_result_from_flush(
            collection_id,
            count,
            &flush_result,
            "sst",
        ))
    }

    /// Delegate compaction to the compaction module
    async fn do_compact(&self, params: &CompactionParameters) -> Result<CompactionResult> {
        info!("🔄 SST: Starting compaction operation");

        // TD-READFID-2 review: the compaction task fixes the output's base
        // quant, rg layout and arming from the collection's catalog meta (tags +
        // `pax_vector_quant`). A producer that cannot supply that meta must not
        // enqueue a quant-deciding task on a guess — it defers to the
        // meta-aware producers (the flush trigger, the background manager).
        let collection_tags: &[String] = match params
            .collection_config
            .as_ref()
            .and_then(|collection| collection.config.as_ref())
        {
            Some(config)
                if params
                    .hints
                    .get(COLLECTION_META_COMPLETE_HINT)
                    .and_then(|value| value.as_bool())
                    != Some(false) =>
            {
                config.tags.as_slice()
            }
            _ => {
                anyhow::bail!(
                    "SST compaction declined: the caller supplied no catalog collection meta \
                     (tags + pax_vector_quant), so the output encoding cannot be resolved; \
                     deferring to a meta-aware compaction producer"
                );
            }
        };
        let collection_id = params.get_collection_id()?;
        let collection_object_id = params.get_collection_object_id()?;
        let collection_dir = params.get_data_dir()?;
        let compaction = self
            .compaction_manager()
            .ok_or_else(|| anyhow::anyhow!("SST compaction manager is unavailable"))?;
        let configured_l0_threshold = self
            .config()
            .compaction_config
            .as_ref()
            .map(|config| config.l0_file_threshold)
            .unwrap_or(self.config().compaction_threshold as usize);
        // Only a BACKGROUND producer honours the per-collection arming +
        // threshold cascade the flush trigger uses. An explicit operator
        // request keeps its pre-existing semantics: never disarmed by
        // `compaction:off` / `workload_profile:churn`, configured threshold.
        let background = !params.force
            && params
                .hints
                .get(BACKGROUND_COMPACTION_HINT)
                .and_then(|value| value.as_bool())
                == Some(true);
        if background && !proximadb_storage_common::resolve_compaction_armed(collection_tags) {
            return Ok(CompactionResult {
                success: true,
                collections_affected: vec![collection_id],
                completed_at: chrono::Utc::now(),
                engine_metrics: HashMap::from([(
                    "task_enqueued".to_string(),
                    serde_json::json!(false),
                )]),
                ..Default::default()
            });
        }
        let l0_threshold = if params.force {
            1
        } else if background {
            proximadb_storage_common::resolve_l0_threshold(collection_tags, configured_l0_threshold)
        } else {
            configured_l0_threshold
        };
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
                    Ok(EmbeddingPrecision::Unspecified | EmbeddingPrecision::Fp32) | Err(_) => None,
                }
            });
        // TD-PAXRG-1: resolve the row-group switch from the collection's
        // `pax_rg_layout` tag (same precedence the flush seam applies).
        let rg_tags = collection_tags;
        let enqueued = compaction
            .enqueue_due_compaction(
                collection_object_id,
                crate::core::stable_id::CollectionIdentity::default(),
                std::path::Path::new(&collection_dir),
                l0_threshold,
                precision_hint,
                Some(super::flush::resolve_pax_rg_layout(rg_tags)),
                // TD-READFID-2 review: same base-quant resolution as flush.
                Some(super::flush::resolve_pax_vector_quant(
                    params.collection_config.as_ref(),
                )),
            )
            .await?;

        if enqueued && params.synchronous {
            let timeout = std::time::Duration::from_millis(params.timeout_ms.unwrap_or(1_200_000));
            if !compaction.await_compaction_quiescence(timeout).await {
                return Err(anyhow::anyhow!(
                    "SST compaction for collection {collection_object_id} did not quiesce within {}ms",
                    timeout.as_millis()
                ));
            }
        }

        Ok(CompactionResult {
            success: true,
            collections_affected: vec![collection_id],
            entries_processed: None,
            entries_removed: None,
            bytes_read: None,
            bytes_written: None,
            input_files: None,
            output_files: None,
            duration_ms: None,
            completed_at: chrono::Utc::now(),
            engine_metrics: HashMap::from([
                ("task_enqueued".to_string(), serde_json::json!(enqueued)),
                (
                    "synchronous".to_string(),
                    serde_json::json!(params.synchronous),
                ),
            ]),
        })
    }

    /// Get vector by ID
    async fn vector_by_id(
        &self,
        collection_id: &str,
        _base_path: &str,
        vector_id: &str,
    ) -> Result<Option<proximadb_records::ProximaRecord>> {
        debug!(
            "🔍 SST: Looking up vector {} in collection {}",
            vector_id, collection_id
        );

        // Check if vector exists using bloom filters
        let exists = self.contains_vector(collection_id, vector_id).await?;

        if !exists {
            return Ok(None);
        }

        // In a real implementation, this would read from SST files
        // For now, return None as a placeholder
        Ok(None)
    }

    /// Delegate search to the search module
    async fn search_vectors_unified(
        &self,
        ctx: &StorageQueryContext,
    ) -> Result<Vec<OptimizedSearchRecord>> {
        info!("🔍 SST: Starting unified search");

        // Use the modular search implementation
        self.search_vectors_unified(ctx).await
    }

    /// ADR-031 + point-lookup: batch ID-based record retrieval. One
    /// `read_all_records` scan + ID filter (vs N stub `vector_by_id` calls
    /// that each returned `None`). Uses the typed data path when identity is
    /// present (Phase 4d).
    async fn point_lookup(
        &self,
        collection_id: &str,
        base_path: &str,
        ids: &[String],
        identity: Option<crate::core::stable_id::CollectionIdentity>,
    ) -> Result<Vec<proximadb_records::ProximaRecord>> {
        use crate::storage::trait_components::path_resolver::{
            collection_data_path_typed, typed_path_identity,
        };
        use std::collections::HashSet;

        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let data_path =
            collection_data_path_typed(base_path, collection_id, typed_path_identity(identity));
        let all_records = self
            .read_all_records(collection_id, Some(&data_path))
            .await?;

        let id_set: HashSet<&str> = ids.iter().map(|s| s.as_str()).collect();
        Ok(all_records
            .into_iter()
            .filter(|r| id_set.contains(r.oid.as_str()))
            .collect())
    }

    /// Get real collection statistics for cost-based query optimization
    async fn collection_stats(
        &self,
        collection_id: &str,
    ) -> Result<crate::storage::traits::CollectionStats> {
        let storage_url = self.get_collection_storage_url(collection_id).await?;
        let fs = self.filesystem().get_filesystem(&storage_url)?;

        let mut total_bytes: u64 = 0;
        let mut file_count: u64 = 0;

        // Scan SSTable files to estimate vector count and total bytes
        if let Ok(entries) = fs.list(&storage_url).await {
            for entry in &entries {
                if !entry.metadata.is_directory {
                    total_bytes += entry.metadata.size;
                    if entry.url.ends_with(".sst") || entry.url.ends_with(".proximablock") {
                        file_count += 1;
                    }
                }
            }
        }

        // Estimate row count from file sizes:
        // Average vector record ~= 4 bytes/dim * 128 dims + 256 bytes metadata = 768 bytes
        // With compression (~2x), avg ~384 bytes per record on disk
        let avg_record_bytes: u64 = 384;
        let estimated_row_count = if avg_record_bytes > 0 && total_bytes > 0 {
            total_bytes / avg_record_bytes
        } else {
            0
        };

        Ok(crate::storage::traits::CollectionStats {
            row_count: estimated_row_count,
            avg_vector_bytes: if file_count > 0 { 512 } else { 0 },
            engine_strategy: StorageFormatStrategy::Sst,
            has_metadata_index: true, // SST always has bloom filters
            has_hnsw_index: self.axis_manager().is_some(),
            total_bytes,
            dimension: None, // Determined at query time from collection config
            index_type: Some("bloom_filter".to_string()),
        })
    }

    /// Read ALL records of a collection from persisted SST files (Phase 8 F1).
    /// Discovers SST data files at the service-resolved `storage_url` and reads
    /// them via the compaction reader. WAL/memtable records are merged in by the
    /// service layer.
    async fn read_all_records(
        &self,
        collection_id: &str,
        storage_url: Option<&str>,
    ) -> Result<Vec<proximadb_records::ProximaRecord>> {
        let Some(storage_url) = storage_url else {
            return Ok(Vec::new());
        };
        let fs = self.filesystem().get_filesystem(storage_url)?;

        // A listing failure must not read as "this collection is empty". Discarding
        // the error here made an unreadable segment directory indistinguishable from
        // a genuinely empty one, and every caller of this function — scan, and the
        // TD-112 AXIS rebuild — would then report zero rows with no signal at all.
        // That silence is expensive: it is usable as a false durability oracle, which
        // is exactly how it was hit (anvai-labs/proximaDB#1524).
        let mut files = Vec::new();
        match fs.list(storage_url).await {
            Ok(entries) => {
                for entry in &entries {
                    if !entry.metadata.is_directory
                        && (entry.url.ends_with(".sst")
                            || entry.url.ends_with(".proximablock")
                            || entry.url.ends_with(".pax"))
                    {
                        files.push(entry.url.clone());
                    }
                }
                tracing::debug!(
                    "sst.read_all_records: {} listed {} entries, {} segment files",
                    storage_url,
                    entries.len(),
                    files.len()
                );
            }
            Err(e) => {
                tracing::warn!(
                    "sst.read_all_records: cannot list segment directory {} ({}); \
                     reporting no stored records for collection {} — callers will see \
                     an empty result that reflects the listing failure, not the data",
                    storage_url,
                    e,
                    collection_id
                );
            }
        }
        if files.is_empty() {
            return Ok(Vec::new());
        }

        let reader = super::sst_reader::UnifiedSSTReader::for_compaction(
            self.filesystem().clone(),
            collection_id.to_string(),
        )?;
        let mut all = Vec::new();
        for file in &files {
            if file.ends_with(".pax") {
                // M1-1b (ADR-049): PAX segments decode via the mixed-format reader
                // (`read_segment_records`, magic-detected) — the ProximaBlocks
                // compaction reader (`read_batch`) cannot decode `.pax`. This is
                // the path TD-112 AXIS-rebuild-from-SST uses, so it must read
                // `.pax` or the index stays empty after a loss.
                let bytes = fs
                    .read(file)
                    .await
                    .map_err(|e| anyhow::anyhow!("read_all_records: read pax {file}: {e}"))?;
                all.extend(
                    crate::storage::engines::sst::segment_format::read_segment_records(
                        &bytes,
                        &[],
                        &[],
                        None,
                    )?,
                );
            } else {
                all.extend(reader.read_batch(file).await?);
            }
        }
        Ok(all)
    }

    /// Collect engine metrics
    async fn collect_engine_metrics(&self) -> Result<HashMap<String, serde_json::Value>> {
        let mut metrics = HashMap::new();

        metrics.insert(
            "engine".to_string(),
            serde_json::Value::String("sst".to_string()),
        );

        metrics.insert(
            "version".to_string(),
            serde_json::Value::String(self.format_version().to_string()),
        );

        // Add SST-specific metrics
        if let Some(_compaction_manager) = self.compaction_manager() {
            metrics.insert(
                "compaction_enabled".to_string(),
                serde_json::Value::Bool(true),
            );
        }

        // Get performance metrics from the universal optimizer
        // Deferred: Add performance metrics collection when available
        metrics.insert(
            "optimizer_status".to_string(),
            serde_json::Value::String("active".to_string()),
        );

        Ok(metrics)
    }

    /// Engine-level RLS predicate (spec §8 — Phase E proof-of-concept).
    ///
    /// Extracts tenant_id from the query context so the scan iterator can
    /// apply row-level isolation without referencing application-layer middleware.
    fn rls_record_filter(
        &self,
        ctx: &crate::storage::traits::StorageQueryContext,
    ) -> Option<crate::storage::traits::RlsRecordPredicate> {
        crate::storage::traits::rls_predicate_from_ctx(ctx)
    }
}

impl crate::storage::traits::EngineFilesystemAccess for SstEngine {
    fn get_filesystem_factory(
        &self,
    ) -> &crate::storage::persistence::filesystem::FilesystemFactory {
        self.filesystem().as_ref()
    }
}

/// Additional trait implementations for SST engine optimization
#[async_trait]
impl crate::storage::engines::core::ops::UniversallyOptimized for SstEngine {
    fn universal_optimizer(
        &self,
    ) -> &crate::storage::engines::core::ops::UniversalPerformanceOptimizer {
        self.universal_optimizer()
    }

    async fn setup_engine_optimizations(&self) -> Result<()> {
        info!("🔧 SST: Setting up engine-specific optimizations");

        // Enable prefetching for SSTable files based on access patterns
        self.universal_optimizer().prefetch_data(&[]).await?;

        // Setup SSTable-specific cache eviction if needed
        self.universal_optimizer().evict_cache_if_needed().await?;

        info!("✅ SST: Engine-specific optimizations setup complete");
        Ok(())
    }

    async fn collect_performance_metrics(&self) -> Result<HashMap<String, serde_json::Value>> {
        let mut metrics = HashMap::new();

        // SST-specific metrics
        metrics.insert(
            "engine_type".to_string(),
            serde_json::Value::String("SST".to_string()),
        );

        metrics.insert(
            "optimization_strategy".to_string(),
            serde_json::Value::String(format!("{:?}", self.universal_optimizer().get_strategy())),
        );

        // Universal optimizer configuration
        let config = self.universal_optimizer().get_config();
        metrics.insert(
            "cache_size_mb".to_string(),
            serde_json::Value::Number(serde_json::Number::from(config.cache_size_mb)),
        );
        metrics.insert(
            "parallel_operations".to_string(),
            serde_json::Value::Number(serde_json::Number::from(config.parallel_operations)),
        );
        metrics.insert(
            "enable_prefetching".to_string(),
            serde_json::Value::Bool(config.enable_prefetching),
        );

        Ok(metrics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::engines::sst::SstConfig;
    use crate::storage::persistence::filesystem::FilesystemFactory;
    use proximadb_distance_kernel::engine::UnifiedDistanceCompute;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_engine_name() {
        let engine = create_test_engine().await;
        assert_eq!(engine.format_name(), "sst");
    }

    #[tokio::test]
    async fn test_engine_strategy() {
        let engine = create_test_engine().await;
        assert!(matches!(engine.strategy(), StorageFormatStrategy::Sst));
    }

    #[tokio::test]
    async fn test_collect_metrics() {
        let engine = create_test_engine().await;
        let metrics = engine.collect_engine_metrics().await.unwrap();

        assert_eq!(
            metrics["engine"],
            serde_json::Value::String("sst".to_string())
        );
        assert!(metrics.contains_key("version"));
    }

    #[tokio::test]
    async fn explicit_compaction_uses_real_scheduler_and_reports_noop() {
        let engine = create_test_engine().await;
        let directory = tempfile::tempdir().unwrap();
        let data_dir = directory.path().join("7").join("data");
        std::fs::create_dir_all(&data_dir).unwrap();
        // TD-READFID-2 review: SST compaction now requires the catalog meta
        // (it fixes the output's base quant / rg layout / arming); a bare
        // `data_dir` hint without it is declined — see the next test.
        let meta = crate::proto::proximadb_v1::Collection {
            id: "7".to_string(),
            config: Some(crate::proto::proximadb_v1::CollectionConfig::default()),
            storage_assignment: Some(crate::proto::proximadb_v1::StorageAssignment {
                base_location: directory.path().to_string_lossy().into_owned(),
                ..Default::default()
            }),
            ..Default::default()
        };
        let params = CompactionParameters {
            collection_id: Some("7".to_string()),
            synchronous: true,
            collection_config: Some(meta),
            ..Default::default()
        };

        let result = engine.do_compact(&params).await.unwrap();

        assert!(result.success);
        assert_eq!(result.collections_affected, vec!["7"]);
        assert_eq!(
            result.engine_metrics.get("task_enqueued"),
            Some(&serde_json::json!(false))
        );
    }

    /// TD-READFID-2 review: a producer with only a `data_dir` hint (no
    /// catalog meta) is declined instead of compacting on a guessed encoding.
    #[tokio::test]
    async fn meta_less_explicit_compaction_is_declined() {
        let engine = create_test_engine().await;
        let directory = tempfile::tempdir().unwrap();
        let params = CompactionParameters {
            collection_id: Some("7".to_string()),
            synchronous: true,
            hints: HashMap::from([(
                "data_dir".to_string(),
                serde_json::json!(directory.path().to_string_lossy()),
            )]),
            ..Default::default()
        };
        let error = engine
            .do_compact(&params)
            .await
            .expect_err("a meta-less compaction request must be declined");
        assert!(error.to_string().contains("no catalog collection meta"));
    }

    async fn create_test_engine() -> SstEngine {
        let config = SstConfig::default();
        let filesystem_config =
            crate::storage::persistence::filesystem::FilesystemConfig::default();
        let filesystem = Arc::new(FilesystemFactory::create(filesystem_config).await.unwrap());
        let distance_compute = Arc::new(UnifiedDistanceCompute::default());

        SstEngine::new_with_config(config, filesystem, distance_compute)
            .await
            .unwrap()
    }
}
