//! TD-DELVEC-1 WI-4 **slice 2** integration test: a cold-resident delete is
//! invisible on the **RaBitQ ANN cascade path** (`try_pax_cascade`), not only the
//! exact scan (slice 1).
//!
//! Slice 1 (`cold_read_merge_test`) covers filtered/exact reads through
//! `search_pax_file_exact`. The coalesced RaBitQ→SQ8 cascade
//! (`rabitq_search_segment_coalesced`) serves *unfiltered* ANN queries itself and
//! never reached that exact reader, so its hits also need merge-on-read. Slice 2
//! threads each cascade hit's global row position (`CascadeHit::position`) into a
//! DV filter inside `try_pax_cascade`.
//!
//! This test writes a coalesced RaBitQ `.pax` straight into the engine's data dir
//! (via `write_pax_segment`, the same helper the coalesced-cascade unit tests use
//! — `coalesced_rabitq_enabled()` is default-ON, so a `RaBitQ` write is coalesced),
//! maps an oid → its on-disk row position with `read_segment_records` (the same
//! positional space the DV and the cascade's row index `g` key on), marks that
//! position deleted, and asserts an unfiltered Cosine scan (which the cascade
//! serves for a coalesced RaBitQ segment) omits the deleted row.

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;

    use anyhow::Result;
    use tempfile::TempDir;

    use crate::core::search::SearchParams;
    use crate::proto::proximadb_v1::{
        Collection, CollectionConfig, StorageAssignment, StorageEngine,
    };
    use crate::storage::engines::sst::SstConfig;
    use crate::storage::engines::sst::core::SstEngine;
    use crate::storage::engines::sst::segment_format::{read_segment_records, write_pax_segment};
    use crate::storage::persistence::filesystem::{FilesystemConfig, FilesystemFactory};
    use crate::storage::traits::StorageQueryContext;
    use proximadb_block_format::VectorQuant;
    use proximadb_distance_kernel::DistanceMetric;
    use proximadb_distance_kernel::engine::UnifiedDistanceCompute;
    use proximadb_records::{EmbeddingCell, EmbeddingValues, ProximaRecord};

    /// RaBitQ needs a non-trivial dimension to avoid degenerate random
    /// projections; 64 mirrors the coalesced-cascade unit tests.
    const DIM: usize = 64;

    fn collection(id: &str, temp_dir: &TempDir) -> Collection {
        Collection {
            id: id.to_string(),
            config: Some(CollectionConfig {
                name: id.to_string(),
                dimension: DIM as u32,
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

    fn rec(oid: &str, i: usize) -> ProximaRecord {
        let ts = 1_700_000_000_000_000_000 + i as i64;
        let mut r = ProximaRecord {
            oid: oid.into(),
            tenant_id: "t".into(),
            created_at_ns: ts,
            updated_at_ns: ts,
            ..Default::default()
        };
        let vec: Vec<f32> = (0..DIM).map(|d| (i as f32 + d as f32) * 0.1).collect();
        r.embeddings
            .push(EmbeddingCell::new_fp32("", "dense", DIM as u32, vec));
        r
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

    #[tokio::test]
    async fn cold_delete_is_invisible_on_rabitq_cascade() -> Result<()> {
        let n: usize = 8;
        let temp_dir = TempDir::new()?;
        let base = temp_dir.path().to_str().unwrap().to_string();
        let collection = collection("cold_cascade_merge", &temp_dir);
        let engine = make_engine(&base).await;

        let records: Vec<ProximaRecord> = (0..n).map(|i| rec(&format!("r{i}"), i)).collect();

        // Unfiltered Cosine query — the cascade serves coalesced RaBitQ segments
        // for Euclidean/Cosine/DotProduct. top_k = n so every row is surfaced.
        let query: Vec<f32> = (0..DIM).map(|d| d as f32 * 0.1).collect();
        let search_params = Arc::new(SearchParams {
            vector: Some(query),
            top_k: Some(n as u16),
            distance_metric: Some(DistanceMetric::Cosine),
            ..Default::default()
        });
        let ctx = StorageQueryContext::new(search_params, Arc::new(collection.clone()));

        // Write a coalesced RaBitQ `.pax` straight into the collection's data dir.
        let storage_url = ctx
            .collection_storage_path()
            .expect("ctx has a storage assignment");
        tokio::fs::create_dir_all(&storage_url).await?;
        let pax_local = format!("{storage_url}/cascade.pax");
        write_pax_segment(
            Path::new(&pax_local),
            &records,
            &collection.id,
            1,
            VectorQuant::RaBitQ,
            None,
            None, // destination_url
        )?;

        // The exact `.pax` path the scan probes (the engine's discovered URL), so
        // the DV key matches.
        let files = engine.discover_sstable_files(&storage_url).await?;
        let seg = files
            .iter()
            .find(|f| f.url.ends_with(".pax"))
            .map(|f| f.url.clone())
            .expect("the written .pax should be discoverable");

        // Map oid → on-disk row position (read_segment_records reconstructs PAX
        // positionally — the same space the DV and the cascade's row index key on).
        let bytes = std::fs::read(&pax_local)?;
        let on_disk = read_segment_records(&bytes, &[], &[], None)?;
        assert!(!on_disk.is_empty(), "segment should decode to records");
        let target_oid = on_disk[0].oid.clone();
        let target_pos = 0u32;

        // Baseline: the cascade returns the target row (among all n).
        let baseline = engine.search_vectors_unified(&ctx).await?;
        let baseline_ids: Vec<&str> = baseline.iter().map(|r| r.id.as_str()).collect();
        assert!(
            baseline_ids.contains(&target_oid.as_str()),
            "baseline cascade scan should include the target, got {baseline_ids:?}"
        );

        // Mark the target's position deleted.
        let dv = engine
            .deletion_vector_store
            .as_ref()
            .expect("DV store is armed under cold-deletion-vectors");
        assert!(dv.mark_deleted(&seg, target_pos, 100).await?, "target mark");

        // Merge-on-read on the cascade path: the deleted row is now invisible, and
        // exactly one row is removed.
        let after = engine.search_vectors_unified(&ctx).await?;
        let after_ids: Vec<&str> = after.iter().map(|r| r.id.as_str()).collect();
        assert!(
            !after_ids.contains(&target_oid.as_str()),
            "deleted target must be absent on cascade merge-on-read, got {after_ids:?}"
        );
        assert_eq!(
            after.len(),
            baseline.len() - 1,
            "exactly the deleted row should be removed; before={}, after={}",
            baseline.len(),
            after.len()
        );
        Ok(())
    }
}
