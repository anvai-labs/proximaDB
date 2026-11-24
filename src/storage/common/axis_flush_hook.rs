// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Single flush → AXIS indexing hook, shared by every storage engine (ADR-078).
//!
//! # Why this exists
//!
//! Flush→AXIS used to be split by engine. SST called
//! `AxisManager::handle_flushed_vectors` **directly** with the in-memory
//! `ProximaRecord`s it already held (zero I/O). HELIX / NOVA instead
//! published a metadata event to the AXIS coordination queue, whose consumer
//! then **re-read those same vectors back out of storage** — N round-trips to
//! recover bytes the flush had in hand. On object storage that is the dominant
//! cost term (`CODESIGN_DIMENSIONAL_ARCHITECTURE_2026_06_19.adoc`), paid to
//! move data that never needed to move.
//!
//! Those three engines took the queue route for a mundane reason: their
//! `axis_manager` field was constructed `None` and **nothing ever set it**, so
//! they had no direct handle to call. This module supplies one, mirroring the
//! `set_sst_axis_manager` global that SST already relied on, and collapses the
//! four call sites onto one guard + one call.
//!
//! # Consequences
//!
//! Passing records **by value** removes the file-lifetime coupling entirely:
//! AXIS no longer depends on the flushed files still existing when indexing
//! runs, which is what the queue's compaction barrier existed to guarantee.
//!
//! Compaction→AXIS is deliberately **not** routed here. It already has a
//! correct path — `compaction_coordinator` →
//! `IndexMaintenance::update_indexes_after_compaction` — which applies
//! tombstone removals, updates merged vectors, and rebuilds static indexes.

use std::sync::{Arc, OnceLock};

use proximadb_index_traits::IndexEngine;
use proximadb_storage_traits::FlushParameters;

/// Process-global AXIS handle for the flush path, registered once at boot.
static FLUSH_AXIS_MANAGER: OnceLock<Arc<dyn IndexEngine>> = OnceLock::new();

/// Register the AXIS manager every engine's flush path will index into.
/// Idempotent: the first registration wins (`OnceLock`), matching
/// `set_sst_axis_manager`.
pub fn set_flush_axis_manager(axis_manager: Arc<dyn IndexEngine>) {
    let _ = FLUSH_AXIS_MANAGER.set(axis_manager);
}

/// The registered AXIS manager, if boot got that far.
pub fn flush_axis_manager() -> Option<Arc<dyn IndexEngine>> {
    FLUSH_AXIS_MANAGER.get().cloned()
}

/// Whether this collection wants an AXIS index at all.
///
/// Co-design: the AXIS build (HNSW/IVF training + RAM) is expensive and was the
/// flush-latency bottleneck (~101s per 21k vectors of pure HNSW build), so
/// collections that never query AXIS must not pay for it. Mirrors the gate the
/// search route applies (`use_axis_indexes`: `index_configs` non-empty).
///
/// Absent config ⇒ **false**. Co-design is the default per ADR-070, so an
/// unknown collection means "no AXIS", not "train AXIS". Recovery flushes land
/// here.
pub fn axis_needed(params: &FlushParameters) -> bool {
    crate::storage::traits::collection_declares_axis_index(
        params
            .collection_config
            .as_ref()
            .and_then(|collection| collection.config.as_ref()),
    )
}

/// Index just-flushed vectors into AXIS (TD-112, ADR-078).
///
/// `explicit` lets an engine pass a handle it holds itself; otherwise the
/// process-global registered at boot is used.
///
/// **Best-effort by design.** The segments are already durable when this runs,
/// so an indexing failure degrades search to a segment scan rather than failing
/// a flush that has, in fact, succeeded. The per-collection `IndexUpdateMode`
/// decides whether this blocks flush completion or runs in the background.
///
/// No double-index risk: the live write path does not populate AXIS, so flush is
/// the first indexing point.
pub async fn index_flushed_into_axis(
    explicit: Option<Arc<dyn IndexEngine>>,
    params: &FlushParameters,
    files_created: Vec<String>,
) {
    if !axis_needed(params) || std::env::var("PROXIMADB_SKIP_AXIS_INDEXING").as_deref() == Ok("1") {
        return;
    }
    let Some(axis_manager) = explicit.or_else(flush_axis_manager) else {
        tracing::debug!("AXIS index-on-flush skipped: no AXIS manager registered");
        return;
    };
    let Some(collection_id) = params.collection_id.as_ref() else {
        return;
    };
    if params.vector_records.is_empty() {
        return;
    }
    if let Err(e) = axis_manager
        .handle_flushed_vectors(collection_id, params.vector_records.clone(), files_created)
        .await
    {
        tracing::warn!(
            "TD-112: AXIS index-on-flush failed for collection {collection_id}: {e} \
             (post-flush search will fall back to a segment scan)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use proximadb_index_traits::{
        IndexHybridQuery, IndexIngest, IndexMaintenance, IndexMetrics, IndexQuery,
        IndexQueryResult, IndexReaderSnapshot,
    };
    use proximadb_proto::proximadb_v1::{Collection, CollectionConfig};
    use std::sync::Mutex;

    /// A recording `IndexEngine` that captures the flush-dispatch the hook makes.
    ///
    /// The hook's job, once its guard passes, is a single
    /// `handle_flushed_vectors` call. This spy asserts that call fires with the
    /// exact records + files the engine passed — the property the guard-only
    /// unit tests above leave open. Every other trait method is a trivial stub;
    /// they exist only because `IndexEngine` composes five roles.
    #[derive(Default)]
    struct RecordingIndexEngine {
        flushes: Mutex<Vec<(String, usize, Vec<String>)>>,
    }

    #[async_trait]
    impl IndexIngest for RecordingIndexEngine {
        async fn handle_flushed_vectors(
            &self,
            collection_id: &str,
            flushed_vectors: Vec<proximadb_records::ProximaRecord>,
            files_created: Vec<String>,
        ) -> anyhow::Result<()> {
            self.flushes.lock().unwrap().push((
                collection_id.to_string(),
                flushed_vectors.len(),
                files_created,
            ));
            Ok(())
        }
    }

    #[async_trait]
    impl IndexQuery for RecordingIndexEngine {
        async fn query(&self, _query: IndexHybridQuery) -> anyhow::Result<IndexQueryResult> {
            Ok(IndexQueryResult { results: vec![] })
        }
    }

    #[async_trait]
    impl IndexMetrics for RecordingIndexEngine {
        async fn registered_vector_count(&self, _collection_id: &str) -> usize {
            0
        }
        async fn has_ivf_index(&self, _collection_id: &str) -> bool {
            false
        }
        async fn has_persisted_ivf_index(&self, _collection_id: &str) -> bool {
            false
        }
        async fn ivf_cold_serving_status(
            &self,
            _collection_id: &str,
        ) -> Option<(String, usize, usize)> {
            None
        }
    }

    #[async_trait]
    impl IndexMaintenance for RecordingIndexEngine {
        async fn rebuild_index(
            &self,
            _collection_id: &str,
            _index_name: &str,
        ) -> anyhow::Result<()> {
            Ok(())
        }
        async fn update_indexes_after_compaction(
            &self,
            _collection_id: &str,
            _deleted_vector_ids: &[String],
            _merged_vectors: &[proximadb_records::ProximaRecord],
        ) -> anyhow::Result<()> {
            Ok(())
        }
        async fn collection_index_stats(
            &self,
            _collection_id: &str,
        ) -> anyhow::Result<Vec<IndexReaderSnapshot>> {
            Ok(vec![])
        }
        async fn analyze_and_optimize(&self, _collection_id: &str) -> anyhow::Result<()> {
            Ok(())
        }
        async fn apply_hnsw_ef_hot_swap(
            &self,
            _collection_id: &str,
            _new_ef_search: u32,
        ) -> anyhow::Result<serde_json::Value> {
            Ok(serde_json::Value::Null)
        }
        async fn apply_ivf_nprobe_hot_swap(
            &self,
            _collection_id: &str,
            _new_nprobe: u32,
        ) -> anyhow::Result<serde_json::Value> {
            Ok(serde_json::Value::Null)
        }
    }

    #[async_trait]
    impl proximadb_index_traits::IndexLifecycle for RecordingIndexEngine {
        async fn drop_collection(&self, _collection_id: &str) -> anyhow::Result<()> {
            Ok(())
        }
        async fn suspend_collection(&self, _collection_id: &str) -> anyhow::Result<()> {
            Ok(())
        }
        async fn resume_collection(&self, _collection_id: &str) -> anyhow::Result<bool> {
            Ok(false)
        }
        async fn is_suspended(&self, _collection_id: &str) -> bool {
            false
        }
    }

    impl proximadb_index_traits::IndexEngine for RecordingIndexEngine {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    fn params_with(config: Option<CollectionConfig>, records: usize) -> FlushParameters {
        FlushParameters {
            collection_id: Some("c1".to_string()),
            collection_config: config.map(|c| Collection {
                config: Some(c),
                ..Default::default()
            }),
            vector_records: (0..records)
                .map(|_| proximadb_records::ProximaRecord::default())
                .collect(),
            ..Default::default()
        }
    }

    /// ADR-070: an unknown collection means "no AXIS", never "train AXIS".
    /// Recovery flushes arrive with no config and must not trigger a build.
    #[test]
    fn absent_config_does_not_request_axis() {
        assert!(!axis_needed(&params_with(None, 10)));
    }

    #[test]
    fn empty_index_configs_do_not_request_axis() {
        let cfg = CollectionConfig {
            index_configs: vec![],
            tags: vec![],
            ..Default::default()
        };
        assert!(!axis_needed(&params_with(Some(cfg), 10)));
    }

    #[test]
    fn format_tag_without_index_config_does_not_request_axis() {
        let cfg = CollectionConfig {
            index_configs: vec![],
            tags: vec!["  PAX_Vector_Format:OFF ".to_string()],
            ..Default::default()
        };
        assert!(!axis_needed(&params_with(Some(cfg), 10)));
    }

    /// No manager registered and none passed ⇒ returns quietly rather than
    /// panicking or failing the (already durable) flush.
    #[tokio::test]
    async fn missing_axis_manager_is_a_no_op() {
        let cfg = CollectionConfig {
            index_configs: vec![proximadb_proto::proximadb_v1::IndexConfig::default()],
            ..Default::default()
        };
        index_flushed_into_axis(None, &params_with(Some(cfg), 5), vec!["f.pax".into()]).await;
    }

    /// ADR-078's load-bearing property: once the guard passes, the hook delivers
    /// the exact records + files the engine held in memory straight to
    /// `handle_flushed_vectors` — no storage re-read. This is the dispatch the
    /// guard-only tests above leave open.
    #[tokio::test]
    async fn convergence_dispatches_records_to_handle_flushed_vectors() {
        let cfg = CollectionConfig {
            index_configs: vec![proximadb_proto::proximadb_v1::IndexConfig::default()],
            ..Default::default()
        };
        let spy = Arc::new(RecordingIndexEngine::default());
        index_flushed_into_axis(
            Some(spy.clone()),
            &params_with(Some(cfg), 3),
            vec!["seg-0.pax".into()],
        )
        .await;
        let captured = spy.flushes.lock().unwrap();
        assert_eq!(captured.len(), 1, "exactly one flush dispatch");
        assert_eq!(captured[0].0, "c1", "collection id threaded through");
        assert_eq!(captured[0].1, 3, "all three records delivered, by value");
        assert_eq!(
            captured[0].2,
            vec!["seg-0.pax".to_string()],
            "files threaded through"
        );
    }

    /// HELIX/NOVA construct `axis_manager: None`, so their engines pass
    /// `None` explicitly and rely on the boot-registered global (the SST-style
    /// `set_flush_axis_manager` registration `SharedServices::new` now also wires
    /// for the shared hook). This pins that fallback path.
    ///
    /// Relies on nextest's process-per-test isolation: the `OnceLock` registration
    /// is first-wins, so a shared `cargo test` process could see a prior test's
    /// registration. The repo standard is nextest (`cargo nxlib`), which isolates
    /// each test in its own process.
    #[tokio::test]
    async fn boot_registered_global_is_used_when_engine_passes_none() {
        let cfg = CollectionConfig {
            index_configs: vec![proximadb_proto::proximadb_v1::IndexConfig::default()],
            ..Default::default()
        };
        let concrete = Arc::new(RecordingIndexEngine::default());
        // The global holds a trait object; keep the concrete Arc to read the
        // shared Mutex it writes (same allocation as the registered clone).
        set_flush_axis_manager(concrete.clone());
        index_flushed_into_axis(None, &params_with(Some(cfg), 2), vec!["v.pax".into()]).await;
        assert_eq!(
            concrete.flushes.lock().unwrap().len(),
            1,
            "the boot-registered global dispatched the flush"
        );
    }
}
