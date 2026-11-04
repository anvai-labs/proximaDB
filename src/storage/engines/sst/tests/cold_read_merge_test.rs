//! TD-DELVEC-1 WI-4 (slice 1) integration test: a cold-resident delete is
//! **invisible to a scan** — merge-on-read applies the deletion vector on the
//! exact `.pax` read path (`search_pax_file_exact`).
//!
//! WI-3b proved the DV *bits* are set (keyed by the delete LSN) + snapshot-correct
//! at the store level. The bits were never *read* on the scan path — this test
//! closes that gap: flush records to a cold `.pax` → baseline search returns all
//! of them → mark DV bits for two positions → a second search **omits the deleted
//! rows and keeps the live one**. In-crate so it can read the `pub(crate)` DV store
//! and the `pub(crate)` file discovery (the latter is used to mark the *exact*
//! segment path the scan will probe, so the DV key matches regardless of the
//! filesystem's URL scheme).
//!
//! **Path coverage:** the query uses `SearchMode::Exact` + a non-cascade metric
//! (`Manhattan`) so the scan routes to `search_pax_file_exact` deterministically —
//! the RaBitQ→SQ8 cascade is gated on `Euclidean|Cosine|DotProduct`, so Manhattan
//! bypasses it independent of the segment's quantization. The snapshot LSN is
//! captured inside `fallback_to_direct_search` (every route to a `.pax` exact read
//! funnels through it); with no freshness source wired in the test engine it
//! falls back to `u64::MAX` (see all deletes), which is exactly what hides the
//! rows.

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use anyhow::Result;
    use tempfile::TempDir;

    use crate::core::search::{SearchMode, SearchParams};
    use crate::proto::proximadb_v1::{
        Collection, CollectionConfig, StorageAssignment, StorageEngine, VectorRecord,
    };
    use crate::storage::engines::sst::{SstConfig, core::SstEngine};
    use crate::storage::persistence::filesystem::{FilesystemConfig, FilesystemFactory};
    use crate::storage::traits::{FlushParameters, StorageQueryContext, UnifiedStorageFormat};
    use proximadb_distance_kernel::DistanceMetric;
    use proximadb_distance_kernel::engine::UnifiedDistanceCompute;

    fn collection(id: &str, temp_dir: &TempDir) -> Collection {
        Collection {
            id: id.to_string(),
            config: Some(CollectionConfig {
                name: id.to_string(),
                dimension: 4,
                storage_engine: Some(StorageEngine::Sst as i32),
                ..Default::default()
            }),
            storage_assignment: Some(StorageAssignment {
                base_location: temp_dir.path().to_str().unwrap().to_string(),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn record(oid: &str, i: usize) -> VectorRecord {
        VectorRecord {
            id: oid.to_string(),
            vector: vec![i as f32, 0.0, 0.0, 0.0],
            metadata: HashMap::new(),
            version: Some(1),
            timestamp: Some(i as i64),
            updated_at: None,
            expires_at: None,
            source: None,
        }
    }

    async fn make_engine(base_path: &str) -> SstEngine {
        let mut fs_config = FilesystemConfig::default();
        fs_config.default_fs = Some(format!("file://{base_path}"));
        let filesystem = Arc::new(FilesystemFactory::create(fs_config).await.unwrap());
        let distance_compute = Arc::new(UnifiedDistanceCompute::default());
        SstEngine::new_with_config(SstConfig::default(), filesystem, distance_compute)
            .await
            .unwrap()
    }

    /// The collection's data directory as the scan sees it (mirrors
    /// `StorageQueryContext::collection_storage_path`), so file discovery matches.
    fn storage_url_for(ctx: &StorageQueryContext) -> String {
        ctx.collection_storage_path()
            .expect("ctx has a storage assignment")
    }

    #[tokio::test]
    async fn cold_delete_is_invisible_on_exact_pax_scan() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let base = temp_dir.path().to_str().unwrap().to_string();
        let collection = collection("cold_read_merge", &temp_dir);
        let engine = make_engine(&base).await;

        // Flush 3 records → an immutable cold `.pax` (with the OID resolver embedded
        // under the feature). r0=[0,0,0,0], r1=[1,0,0,0], r2=[2,0,0,0].
        let records: Vec<VectorRecord> = ["r0", "r1", "r2"]
            .iter()
            .copied()
            .enumerate()
            .map(|(i, o)| record(o, i))
            .collect();
        let flush = FlushParameters {
            collection_id: Some(collection.id.clone()),
            vector_records: records.into_iter().map(Into::into).collect(),
            force: true,
            synchronous: true,
            hints: HashMap::new(),
            timeout_ms: None,
            trigger_compaction: false,
            batch_ids: vec![],
            collection_config: Some(collection.clone()),
            estimated_size: 0,
            collection_properties: std::collections::HashMap::new(),
        };
        let res = engine.do_flush(&flush).await?;
        assert!(res.success, "flush should produce a cold segment");

        // Exact scan + a non-cascade metric → routes to `search_pax_file_exact`
        // (Manhattan bypasses the RaBitQ cascade gate), where the DV merge-on-read
        // lives. top_k large enough to surface every record.
        let search_params = Arc::new(SearchParams {
            vector: Some(vec![0.0, 0.0, 0.0, 0.0]),
            top_k: Some(10),
            distance_metric: Some(DistanceMetric::Manhattan),
            search_mode: SearchMode::Exact,
            ..Default::default()
        });
        let ctx = StorageQueryContext::new(search_params, Arc::new(collection.clone()));

        // Resolve the exact `.pax` path the scan will probe (same `entry.url` the
        // search passes to `search_pax_file_exact`), so the DV key matches.
        let storage_url = storage_url_for(&ctx);
        let files = engine.discover_sstable_files(&storage_url).await?;
        let seg = files
            .iter()
            .find(|f| f.url.ends_with(".pax"))
            .map(|f| f.url.clone())
            .expect("flush should have produced a .pax segment");
        assert!(
            files.iter().filter(|f| f.url.ends_with(".pax")).count() == 1,
            "expected exactly one .pax segment for a deterministic position map, got {}",
            files
                .iter()
                .map(|f| f.url.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );

        // Baseline: with no deletes, the scan returns all three records.
        let baseline = engine.search_vectors_unified(&ctx).await?;
        let baseline_ids: Vec<&str> = baseline.iter().map(|r| r.id.as_str()).collect();
        assert!(
            baseline_ids.contains(&"r0")
                && baseline_ids.contains(&"r1")
                && baseline_ids.contains(&"r2"),
            "baseline scan should return all records, got {baseline_ids:?}"
        );

        // Mark r0 (pos 0) + r2 (pos 2) deleted; leave r1 (pos 1) live. The Vec index
        // IS the row position (`read_segment_records` reconstructs PAX positionally).
        let dv = engine
            .deletion_vector_store
            .as_ref()
            .expect("DV store is armed under cold-deletion-vectors");
        assert!(dv.mark_deleted(&seg, 0, 100).await?, "r0 mark");
        assert!(dv.mark_deleted(&seg, 2, 200).await?, "r2 mark");

        // Merge-on-read: the deleted rows are now invisible; the live row remains.
        let after = engine.search_vectors_unified(&ctx).await?;
        let after_ids: Vec<&str> = after.iter().map(|r| r.id.as_str()).collect();
        assert!(
            !after_ids.contains(&"r0"),
            "deleted r0 must be absent on merge-on-read, got {after_ids:?}"
        );
        assert!(
            !after_ids.contains(&"r2"),
            "deleted r2 must be absent on merge-on-read, got {after_ids:?}"
        );
        assert!(
            after_ids.contains(&"r1"),
            "live r1 must still be present, got {after_ids:?}"
        );
        Ok(())
    }
}
