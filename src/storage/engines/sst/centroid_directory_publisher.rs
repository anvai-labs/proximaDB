//! TD-PAXDIR-1 D1: publish the collection-global PAX centroid directory.
//!
//! After a compaction has published its output and retired its inputs, the
//! collection's directory (`centroids-<gen>.pxcd`, format in
//! [`proximadb_storage_common::centroid_directory`]) is brought up to date
//! with the live segment set:
//!
//! 1. LIST the data directory, as discovery does: live segments are the
//!    `*.pax` objects not pending retirement (TD-COMPACT-13).
//! 2. Load the newest valid directory as the base; keep its entries for
//!    segments that are still live with the same size.
//! 3. Read the Region A0 of every other live segment (header prefix, then A0:
//!    two ranged reads, off the query path). Segments without a usable A0 are
//!    recorded as examined-but-uncovered, so later publishes skip them.
//! 4. Write generation `g + 1` **create-only**. Losing the race to another
//!    publisher reloads and retries; repeated losses give up, since the next
//!    compaction repairs the directory.
//! 5. Delete generations older than the previous one (readers holding an older
//!    generation stay correct: every entry is validated against the live set).
//!
//! Gate: `PROXIMADB_PAX_WRITE_CENTROID_DIRECTORY` (default OFF). A publish
//! failure is logged and never fails the compaction: the directory is a
//! projection, and without one queries use the per-segment cascade.

use proximadb_storage_common::centroid_directory::{
    LiveSegment, PaxCentroidDirectory, object_name, parse_object_name, plan_reconcile,
};
use proximadb_storage_common::coarse_directory::CoarseDirectory;
use proximadb_storage_common::segment_layout::{SEG_HEADER_PREFIX_LEN, SegmentHeaderPrefix};
use proximadb_storage_filesystem_types::{DirEntry, FileSystem, FilesystemError};

use crate::storage::engines::sst::block_cluster::env_gate_on;
use crate::storage::engines::sst::retirement_ledger;

/// Create-only publish attempts before giving up to a concurrent publisher.
const PUBLISH_ATTEMPTS: usize = 3;

/// WRITE gate: compaction publishes the collection centroid directory.
pub fn centroid_directory_write_enabled() -> bool {
    env_gate_on("PROXIMADB_PAX_WRITE_CENTROID_DIRECTORY")
}

/// Directory objects in a listing, newest generation first.
pub(crate) fn directory_objects_newest_first(entries: &[DirEntry]) -> Vec<(u64, &DirEntry)> {
    let mut objects: Vec<(u64, &DirEntry)> = entries
        .iter()
        .filter(|e| !e.metadata.is_directory)
        .filter_map(|e| parse_object_name(&e.name).map(|generation| (generation, e)))
        .collect();
    objects.sort_by_key(|o| std::cmp::Reverse(o.0));
    objects
}

/// Read and parse a segment's Region A0. `Ok(None)` when the segment has no
/// usable A0 (a non-coalesced layout, an untrained segment, or a corrupt
/// directory); `Err` only for I/O failures.
pub(crate) async fn read_segment_a0(
    fs: &dyn FileSystem,
    url: &str,
    size_bytes: u64,
) -> Result<Option<CoarseDirectory>, FilesystemError> {
    let prefix_len = (SEG_HEADER_PREFIX_LEN as u64).min(size_bytes);
    if prefix_len < SEG_HEADER_PREFIX_LEN as u64 {
        return Ok(None);
    }
    let prefix = fs.read_range(url, 0, prefix_len).await?;
    let Ok(header) = SegmentHeaderPrefix::parse(&prefix) else {
        return Ok(None);
    };
    let in_bounds = header
        .a0_off
        .checked_add(header.a0_len)
        .is_some_and(|end| end <= size_bytes);
    if header.a0_len == 0 || !in_bounds {
        return Ok(None);
    }
    let bytes = fs.read_range(url, header.a0_off, header.a0_len).await?;
    Ok(CoarseDirectory::parse(&bytes).ok())
}

/// Newest directory in `entries` that reads and parses; corrupt or unreadable
/// generations are skipped (an older valid base still saves A0 reads).
async fn load_newest_valid(
    fs: &dyn FileSystem,
    objects: &[(u64, &DirEntry)],
) -> Option<PaxCentroidDirectory> {
    for (generation, entry) in objects {
        match fs.read(&entry.url).await {
            Ok(bytes) => match PaxCentroidDirectory::parse(&bytes) {
                Ok(dir) if dir.generation == *generation => return Some(dir),
                Ok(_) => tracing::warn!(url = %entry.url, "centroid directory generation mismatch"),
                Err(error) => {
                    tracing::warn!(url = %entry.url, %error, "corrupt centroid directory")
                }
            },
            Err(error) => tracing::warn!(url = %entry.url, %error, "unreadable centroid directory"),
        }
    }
    None
}

async fn object_size(fs: &dyn FileSystem, entry: &DirEntry) -> Option<u64> {
    if entry.metadata.size > 0 {
        return Some(entry.metadata.size);
    }
    fs.metadata(&entry.url).await.ok().map(|m| m.size)
}

/// Bring the directory of the collection data directory `dir_url` up to date.
/// Returns the generation written, or `None` when nothing changed (or a
/// concurrent publisher kept winning).
pub async fn publish_centroid_directory(
    fs: &dyn FileSystem,
    dir_url: &str,
) -> anyhow::Result<Option<u64>> {
    let dir_url = dir_url.trim_end_matches('/');
    for _ in 0..PUBLISH_ATTEMPTS {
        let entries = fs.list(dir_url).await?;
        let pending = retirement_ledger::pending_paths();
        let mut live = Vec::new();
        let mut urls = std::collections::HashMap::new();
        for entry in &entries {
            if entry.metadata.is_directory
                || !entry.name.ends_with(".pax")
                || pending.contains(&entry.url)
            {
                continue;
            }
            let Some(size_bytes) = object_size(fs, entry).await else {
                continue; // vanished between LIST and stat: not live
            };
            urls.insert(entry.name.clone(), entry.url.clone());
            live.push(LiveSegment {
                name: entry.name.clone(),
                size_bytes,
            });
        }
        let objects = directory_objects_newest_first(&entries);
        let newest_generation = objects.first().map(|(g, _)| *g);
        let base = load_newest_valid(fs, &objects).await;
        let mut plan = plan_reconcile(base.as_ref(), &live);
        // A base older than the newest object (the newest was corrupt) is
        // republished even if it matches the live set.
        let base_is_newest = base.as_ref().map(|d| d.generation) == newest_generation;
        if !plan.changed && base_is_newest {
            return Ok(None);
        }
        for segment in &plan.to_load {
            let url = &urls[&segment.name];
            match read_segment_a0(fs, url, segment.size_bytes).await {
                Ok(Some(a0)) => {
                    if let Err(error) =
                        plan.builder
                            .add_covered(&segment.name, segment.size_bytes, &a0)
                    {
                        tracing::debug!(%url, %error, "segment not addressable by the directory");
                        plan.builder
                            .add_uncovered(&segment.name, segment.size_bytes);
                    }
                }
                Ok(None) => plan
                    .builder
                    .add_uncovered(&segment.name, segment.size_bytes),
                // Transient read failure: leave the segment out of the stamp
                // entirely, so readers scan it per segment and the next
                // publish retries it.
                Err(error) => tracing::warn!(%url, %error, "segment A0 unreadable; not stamped"),
            }
        }
        let generation = newest_generation.map_or(1, |g| g.saturating_add(1));
        let bytes = plan.builder.build(generation)?.to_bytes()?;
        let url = format!("{dir_url}/{}", object_name(generation));
        match fs.write_if_absent(&url, &bytes, None).await {
            Ok(()) => {
                for (old, entry) in &objects {
                    if old.saturating_add(1) < generation
                        && let Err(error) = fs.delete(&entry.url).await
                    {
                        tracing::debug!(url = %entry.url, %error, "centroid directory GC deferred");
                    }
                }
                return Ok(Some(generation));
            }
            Err(FilesystemError::AlreadyExists(_)) => continue,
            Err(error) => return Err(error.into()),
        }
    }
    tracing::debug!(
        dir_url,
        "centroid directory publish lost every race; next compaction repairs"
    );
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::engines::sst::segment_format::write_pax_segment_compacted;
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
    use proximadb_block_format::VectorQuant;
    use proximadb_records::{EmbeddingCell, ProximaRecord};

    const DIM: usize = 16;

    fn enable_trained_layout() {
        // SAFETY: nextest runs each test in its own process.
        unsafe {
            std::env::set_var("PROXIMADB_PAX_COALESCED_RABITQ", "1");
            std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "1");
            std::env::set_var("PROXIMADB_IVF_K", "4");
        }
    }

    fn records(seed: usize, n: usize) -> Vec<ProximaRecord> {
        (0..n)
            .map(|i| {
                let vector = (0..DIM)
                    .map(|d| ((((i + seed) * 131 + d * 17) % 251) as f32) * 0.01)
                    .collect::<Vec<f32>>();
                let mut r = ProximaRecord {
                    oid: format!("s{seed}-{i}"),
                    tenant_id: "t".into(),
                    created_at_ns: 1000 + i as i64,
                    updated_at_ns: 1000 + i as i64,
                    ..Default::default()
                };
                r.embeddings
                    .push(EmbeddingCell::new_fp32("", "dense", DIM as u32, vector));
                r
            })
            .collect()
    }

    fn write_segment(dir: &std::path::Path, name: &str, seed: usize) {
        write_pax_segment_compacted(
            &dir.join(name),
            &records(seed, 300),
            "col",
            1,
            VectorQuant::RaBitQ,
            VectorQuant::Sq8,
            false,
            Some(16 * 1024),
            None,
        )
        .expect("write trained segment");
    }

    async fn local_fs() -> LocalFileSystem {
        LocalFileSystem::new(LocalConfig::default()).await.unwrap()
    }

    /// Parsable directories, newest first.
    async fn newest(fs: &dyn FileSystem, dir: &str) -> Vec<(u64, PaxCentroidDirectory)> {
        let entries = fs.list(dir).await.unwrap();
        let mut out = Vec::new();
        for (generation, entry) in directory_objects_newest_first(&entries) {
            let bytes = fs.read(&entry.url).await.unwrap();
            if let Ok(dir) = PaxCentroidDirectory::parse(&bytes) {
                out.push((generation, dir));
            }
        }
        out
    }

    #[tokio::test]
    async fn publish_covers_exactly_the_live_trained_segments() {
        enable_trained_layout();
        let tmp = tempfile::tempdir().unwrap();
        write_segment(tmp.path(), "a.pax", 0);
        write_segment(tmp.path(), "b.pax", 7);
        std::fs::write(tmp.path().join("l0.pax"), b"not a coalesced segment").unwrap();
        std::fs::write(tmp.path().join("retire-pending.json"), b"{}").unwrap();
        let fs = local_fs().await;
        let dir_url = tmp.path().to_string_lossy().to_string();

        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            Some(1)
        );
        let published = newest(&fs, &dir_url).await;
        assert_eq!(published.len(), 1);
        let dir = &published[0].1;
        let covered: Vec<&str> = dir
            .covered_segments()
            .map(|(_, s)| s.name.as_str())
            .collect();
        assert_eq!(covered, ["a.pax", "b.pax"]);
        let uncovered: Vec<&str> = dir
            .segments
            .iter()
            .filter(|s| !s.is_covered())
            .map(|s| s.name.as_str())
            .collect();
        assert_eq!(uncovered, ["l0.pax"], "examined but not addressable");
        assert_eq!(dir.dim as usize, DIM);
        for (idx, segment) in dir.covered_segments() {
            assert_eq!(segment.rows_covered, 300);
            assert!(dir.cells.iter().any(|c| c.segment_idx as usize == idx));
        }

        // Every cell's extents equal the segment's own A0.
        for (idx, segment) in dir.covered_segments() {
            let url = tmp.path().join(&segment.name).to_string_lossy().to_string();
            let a0 = read_segment_a0(&fs, &url, segment.size_bytes)
                .await
                .unwrap()
                .unwrap();
            for cell in dir.cells.iter().filter(|c| c.segment_idx as usize == idx) {
                assert_eq!(cell.entry, a0.cells[cell.a0_cell as usize]);
            }
        }

        // Nothing changed: no new generation.
        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn publish_tracks_compaction_and_garbage_collects_old_generations() {
        enable_trained_layout();
        let tmp = tempfile::tempdir().unwrap();
        let fs = local_fs().await;
        let dir_url = tmp.path().to_string_lossy().to_string();
        write_segment(tmp.path(), "a.pax", 0);
        write_segment(tmp.path(), "b.pax", 7);
        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            Some(1)
        );

        // A compaction folds a + b into c.
        write_segment(tmp.path(), "c.pax", 3);
        std::fs::remove_file(tmp.path().join("a.pax")).unwrap();
        std::fs::remove_file(tmp.path().join("b.pax")).unwrap();
        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            Some(2)
        );
        let covered: Vec<String> = newest(&fs, &dir_url).await[0]
            .1
            .covered_segments()
            .map(|(_, s)| s.name.clone())
            .collect();
        assert_eq!(covered, ["c.pax"]);

        write_segment(tmp.path(), "d.pax", 5);
        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            Some(3)
        );
        let generations: Vec<u64> = newest(&fs, &dir_url)
            .await
            .iter()
            .map(|(g, _)| *g)
            .collect();
        assert_eq!(generations, [3, 2], "keeps the newest two generations");
    }

    #[tokio::test]
    async fn a_corrupt_newest_generation_is_superseded_by_a_rebuild() {
        enable_trained_layout();
        let tmp = tempfile::tempdir().unwrap();
        let fs = local_fs().await;
        let dir_url = tmp.path().to_string_lossy().to_string();
        write_segment(tmp.path(), "a.pax", 0);
        std::fs::write(tmp.path().join(object_name(4)), b"garbage").unwrap();
        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            Some(5)
        );
        let dir = &newest(&fs, &dir_url).await[0].1;
        assert_eq!(dir.generation, 5);
        assert_eq!(dir.covered_segments().count(), 1);
    }

    #[tokio::test]
    async fn losing_the_create_only_race_retries_on_the_winner() {
        enable_trained_layout();
        let tmp = tempfile::tempdir().unwrap();
        let fs = local_fs().await;
        let dir_url = tmp.path().to_string_lossy().to_string();
        write_segment(tmp.path(), "a.pax", 0);
        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            Some(1)
        );
        write_segment(tmp.path(), "b.pax", 7);
        // Two publishers race on the same change: exactly one generation is
        // written for it, and the loser finds nothing left to do.
        let (first, second) = tokio::join!(
            publish_centroid_directory(&fs, &dir_url),
            publish_centroid_directory(&fs, &dir_url)
        );
        let mut written: Vec<u64> = [first.unwrap(), second.unwrap()]
            .into_iter()
            .flatten()
            .collect();
        written.sort();
        assert_eq!(written, [2]);
        let dir = &newest(&fs, &dir_url).await[0].1;
        assert_eq!(dir.covered_segments().count(), 2);
    }

    #[test]
    fn write_gate_defaults_off() {
        // SAFETY: nextest runs each test in its own process.
        unsafe { std::env::remove_var("PROXIMADB_PAX_WRITE_CENTROID_DIRECTORY") };
        assert!(!centroid_directory_write_enabled());
        unsafe { std::env::set_var("PROXIMADB_PAX_WRITE_CENTROID_DIRECTORY", "on") };
        assert!(centroid_directory_write_enabled());
    }
}
