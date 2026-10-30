//! TD-DELVEC-1 WI-6 test: compaction's `build_deleted_oids` collects the oids
//! deleted (DV bit set) across the input segments — the core of compaction
//! DV-awareness (a DV-deleted row is physically dropped at the merge, else it
//! resurrects in the compacted output).
//!
//! Writes a coalesced RaBitQ `.pax` (OID resolver embedded) → marks one position
//! deleted in a `.dv` → asserts a fresh `build_deleted_oids` pass (as compaction
//! runs it, with a fresh `DeletionVectorStore` loading the `.dv` from disk)
//! collects exactly that oid, not the live ones. In-crate so it can call the
//! `pub(crate)` feature-gated associated fn + the segment helpers.

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use anyhow::Result;
    use tempfile::TempDir;

    use crate::storage::engines::sst::compaction::Compaction;
    use crate::storage::engines::sst::deletion_vector_store::DeletionVectorStore;
    use crate::storage::engines::sst::segment_format::{read_segment_records, write_pax_segment};
    use crate::storage::persistence::filesystem::{FilesystemConfig, FilesystemFactory};
    use proximadb_block_format::VectorQuant;
    use proximadb_records::{EmbeddingCell, EmbeddingValues, ProximaRecord};

    /// RaBitQ needs a non-trivial dimension; 64 mirrors the coalesced-cascade tests.
    const DIM: usize = 64;

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

    #[tokio::test]
    async fn build_deleted_oids_collects_dv_deleted_oids_only() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let base = temp_dir.path().to_str().unwrap().to_string();
        let pax_local = format!("{base}/seg.pax");
        let seg = format!("file://{pax_local}");

        // Write a coalesced RaBitQ segment (OID resolver embedded in the footer).
        let records: Vec<ProximaRecord> = (0..3).map(|i| rec(&format!("r{i}"), i)).collect();
        write_pax_segment(
            Path::new(&pax_local),
            &records,
            "col",
            1,
            VectorQuant::RaBitQ,
            None,
            None, // destination_url
        )?;

        // Find r1's on-disk row position (the space the DV keys on).
        let bytes = std::fs::read(&pax_local)?;
        let on_disk = read_segment_records(&bytes, &[], &[], None)?;
        let r1_pos = on_disk
            .iter()
            .position(|r| r.oid == "r1")
            .expect("r1 in segment") as u32;

        // Filesystem rooted at the temp dir.
        let mut fs_config = FilesystemConfig::default();
        fs_config.default_fs = Some(format!("file://{base}"));
        let fs = Arc::new(FilesystemFactory::create(fs_config).await?);

        // Mark r1 deleted (writes `{seg}.dv` — disk-authoritative).
        let dv_marker = DeletionVectorStore::new(fs.clone());
        assert!(
            dv_marker.mark_deleted(&seg, r1_pos, 100).await?,
            "r1 mark should be newly-set"
        );

        // A FRESH DV store (as compaction uses it) loads `{seg}.dv` from disk.
        let dv_store = DeletionVectorStore::new(fs.clone());
        let deleted = Compaction::build_deleted_oids(&fs, &dv_store, &[PathBuf::from(&seg)]).await;

        assert!(
            deleted.contains("r1"),
            "the DV-deleted oid r1 must be collected: {deleted:?}"
        );
        assert!(
            !deleted.contains("r0") && !deleted.contains("r2"),
            "live oids r0/r2 must not be collected: {deleted:?}"
        );
        Ok(())
    }
}
