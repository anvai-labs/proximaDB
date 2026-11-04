//! TD-DELVEC-1 WI-3b integration test: a cold-resident delete sets a **versioned
//! deletion-vector bit** on a real flushed immutable `.pax` segment — MVCC
//! snapshot-correct + restart-safe.
//!
//! Validates the WI-3b DV-bit substance end-to-end at the engine level: flush
//! records to a cold `.pax` → mark deletion-vector bits (keyed by the delete LSN)
//! → `is_deleted_as_of` is snapshot-correct → the bits survive a restart
//! (disk-authoritative reload). In-crate so it can read the `pub(crate)` DV store.
//!
//! The segment is located via the engine's own `discover_sstable_files` (the
//! `get_collection_storage_url` placeholder was retired by #1352 — it now resolves
//! the collection's real `base_location`).

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use anyhow::Result;
    use tempfile::TempDir;

    use crate::core::search::SearchParams;
    use crate::proto::proximadb_v1::{
        Collection, CollectionConfig, StorageAssignment, StorageEngine, VectorRecord,
    };
    use crate::storage::engines::sst::{SstConfig, core::SstEngine};
    use crate::storage::persistence::filesystem::{FilesystemConfig, FilesystemFactory};
    use crate::storage::traits::{FlushParameters, StorageQueryContext, UnifiedStorageFormat};
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

    #[tokio::test]
    async fn cold_delete_sets_versioned_dv_bit_on_a_flushed_segment() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let base = temp_dir.path().to_str().unwrap().to_string();
        let collection = collection("cold_del_dv", &temp_dir);
        let engine = make_engine(&base).await;

        // Flush 3 records → an immutable cold `.pax` (with the OID resolver embedded,
        // WI-3a-1 `with_oid_resolver(true)` under the feature).
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

        // Locate the flushed segment via the engine's own discovery (#1352 retired
        // the get_collection_storage_url placeholder — it resolves base_location now).
        let storage_url = StorageQueryContext::new(
            Arc::new(SearchParams::default()),
            Arc::new(collection.clone()),
        )
        .collection_storage_path()
        .expect("ctx has a storage assignment");
        let files = engine.discover_sstable_files(&storage_url).await?;
        let seg = files
            .iter()
            .find(|f| f.url.ends_with(".pax"))
            .map(|f| f.url.clone())
            .expect("flush should have produced a .pax segment");

        let dv = engine
            .deletion_vector_store
            .as_ref()
            .expect("DV store is armed under cold-deletion-vectors");

        // Mark positions 0 + 2 deleted at distinct LSNs (position 1 left live) — the
        // delete LSN is the `generation` key for the versioned DV.
        dv.mark_deleted(&seg, 0, 100).await?;
        dv.mark_deleted(&seg, 2, 200).await?;

        // MVCC snapshot-correct: a bit is visible at its gen + any later snapshot,
        // invisible to an earlier snapshot.
        assert!(
            dv.is_deleted_as_of(&seg, 0, 100).await,
            "pos 0 deleted @ 100"
        );
        assert!(
            dv.is_deleted_as_of(&seg, 0, 150).await,
            "pos 0 visible at a later snapshot"
        );
        assert!(
            !dv.is_deleted_as_of(&seg, 0, 99).await,
            "pos 0 invisible before its delete"
        );
        assert!(
            dv.is_deleted_as_of(&seg, 2, 200).await,
            "pos 2 deleted @ 200"
        );
        assert!(
            !dv.is_deleted_as_of(&seg, 2, 150).await,
            "pos 2 invisible before its delete (150 < 200)"
        );
        assert!(
            !dv.is_deleted_as_of(&seg, 1, u64::MAX).await,
            "pos 1 was never deleted"
        );

        // Restart-safety: a fresh engine over the same dir reloads the `.dv` from disk.
        let engine2 = make_engine(&base).await;
        let dv2 = engine2
            .deletion_vector_store
            .as_ref()
            .expect("DV store armed");
        dv2.load(&seg).await?;
        assert!(
            dv2.is_deleted_as_of(&seg, 0, 100).await,
            "pos 0 DV bit survives a restart (disk-authoritative reload)"
        );
        assert!(
            dv2.is_deleted_as_of(&seg, 2, 200).await,
            "pos 2 DV bit survives a restart"
        );
        Ok(())
    }
}
