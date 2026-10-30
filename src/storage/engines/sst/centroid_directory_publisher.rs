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
//!    TD-PAXDIR-1 D6 (`PROXIMADB_PAX_WRITE_DIRECTORY_DECODE_KEYS`, default
//!    ON since D8): a covered segment also gets a **decode key** (its Region-A header
//!    and footer: two more ranged reads, and its listed ETag), kept only when
//!    the A0 the directory rebuilds from it is byte-identical to the
//!    segment's. Entries carried from the base keep their key, if any.
//! 4. Write generation `g + 1` **create-only**. Losing the race to another
//!    publisher reloads and retries; repeated losses give up, since the next
//!    compaction repairs the directory.
//! 5. Delete generations older than the previous one (readers holding an older
//!    generation stay correct: every entry is validated against the live set).
//!
//! Gate: `PROXIMADB_PAX_WRITE_CENTROID_DIRECTORY` (default OFF). A publish
//! failure is logged and never fails the compaction: the directory is a
//! projection, and without one queries use the per-segment cascade.

use std::collections::HashMap;

use proximadb_storage_common::centroid_directory::{
    DirectorySegment, LiveSegment, PaxCentroidDirectory, SegmentDecodeKey, object_name,
    parse_object_name, plan_reconcile_keeping,
};
use proximadb_storage_common::coarse_directory::CoarseDirectory;
use proximadb_storage_common::segment_layout::{
    SEG_HEADER_PREFIX_LEN, SegmentFooterIndex, SegmentHeaderPrefix,
};
use proximadb_storage_filesystem_types::{DirEntry, FileSystem, FilesystemError};

use crate::storage::engines::sst::block_cluster::env_kill_switch_off;
use crate::storage::engines::sst::retirement_ledger;

/// Create-only publish attempts before giving up to a concurrent publisher.
const PUBLISH_ATTEMPTS: usize = 3;

/// WRITE gate: compaction publishes the collection centroid directory.
pub fn centroid_directory_write_enabled() -> bool {
    env_kill_switch_off("PROXIMADB_PAX_WRITE_CENTROID_DIRECTORY")
}

/// TD-PAXDIR-1 D6 WRITE gate: publishes add decode keys for the segments they
/// load (two more ranged reads each) and write the v2 format. Default ON
/// since D8; off, the directory is the D1 format, byte for byte.
pub fn directory_decode_keys_write_enabled() -> bool {
    env_kill_switch_off("PROXIMADB_PAX_WRITE_DIRECTORY_DECODE_KEYS")
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

/// A segment's Region A0 as the publisher reads it.
pub(crate) struct SegmentA0 {
    pub(crate) a0: CoarseDirectory,
    /// The raw A0 bytes (the decode-key rebuild must reproduce them).
    a0_bytes: Vec<u8>,
    header: SegmentHeaderPrefix,
    prefix: Vec<u8>,
}

/// Read and parse a segment's Region A0. `Ok(None)` when the segment has no
/// usable A0 (a non-coalesced layout, an untrained segment, or a corrupt
/// directory); `Err` only for I/O failures.
pub(crate) async fn read_segment_a0(
    fs: &dyn FileSystem,
    url: &str,
    size_bytes: u64,
) -> Result<Option<CoarseDirectory>, FilesystemError> {
    Ok(read_segment_a0_with_header(fs, url, size_bytes)
        .await?
        .map(|s| s.a0))
}

async fn read_segment_a0_with_header(
    fs: &dyn FileSystem,
    url: &str,
    size_bytes: u64,
) -> Result<Option<SegmentA0>, FilesystemError> {
    let prefix_len = (SEG_HEADER_PREFIX_LEN as u64).min(size_bytes);
    if prefix_len < SEG_HEADER_PREFIX_LEN as u64 {
        return Ok(None);
    }
    let prefix = fs.read_range(url, 0, prefix_len).await?;
    let Ok(header) = SegmentHeaderPrefix::parse(&prefix) else {
        return Ok(None);
    };
    if header.a0_len == 0 || !within(size_bytes, header.a0_off, header.a0_len) {
        return Ok(None);
    }
    let a0_bytes = fs.read_range(url, header.a0_off, header.a0_len).await?;
    Ok(CoarseDirectory::parse(&a0_bytes).ok().map(|a0| SegmentA0 {
        a0,
        a0_bytes,
        header,
        prefix,
    }))
}

fn within(size_bytes: u64, off: u64, len: u64) -> bool {
    off.checked_add(len).is_some_and(|end| end <= size_bytes)
}

/// TD-PAXDIR-1 D6: read the rest of a covered segment's query-path metadata
/// (the Region-A header and the footer) and build its decode key. `None` when
/// a read fails or the parts disagree: the segment stays covered without a
/// key and its queries read their own metadata.
async fn read_decode_key(
    fs: &dyn FileSystem,
    url: &str,
    size_bytes: u64,
    etag: Option<&str>,
    segment: &SegmentA0,
) -> Option<SegmentDecodeKey> {
    let header = &segment.header;
    let hdr_len = proximadb_block_format::region_header_len(segment.a0.model.dim) as u64;
    if !within(size_bytes, header.rabitq_off, hdr_len)
        || !within(size_bytes, header.footer_off, header.footer_len)
    {
        return None;
    }
    let read = async {
        let rabitq_header = fs.read_range(url, header.rabitq_off, hdr_len).await?;
        let footer = fs
            .read_range(url, header.footer_off, header.footer_len)
            .await?;
        Ok::<_, FilesystemError>((rabitq_header, footer))
    };
    let (rabitq_header, footer) = match read.await {
        Ok(parts) => parts,
        Err(error) => {
            tracing::debug!(%url, %error, "segment metadata unreadable; no decode key");
            return None;
        }
    };
    let footer = SegmentFooterIndex::parse(&footer).ok()?;
    SegmentDecodeKey::from_segment(&segment.prefix, &segment.a0, &rabitq_header, &footer, etag)
        .map_err(|error| tracing::debug!(%url, %error, "segment has no decode key"))
        .ok()
}

/// Keys the published directory may keep: none when the write gate is off
/// (so turning it off returns the directory to the D1 v1 format, carried
/// keys included), and otherwise only keys whose recorded ETag still matches
/// the segment's listing (a carried key for a rewritten object is dropped
/// rather than refused by every reader).
fn retain_current_decode_keys(
    dir: &mut PaxCentroidDirectory,
    decode_keys: bool,
    listed: &HashMap<String, (String, Option<String>)>,
) {
    for segment in &mut dir.segments {
        if !(decode_keys && key_matches_listing(segment, listed)) {
            segment.decode_key = None;
        }
    }
}

/// Whether `segment`'s decode key records the ETag its listing reports.
fn key_matches_listing(
    segment: &DirectorySegment,
    listed: &HashMap<String, (String, Option<String>)>,
) -> bool {
    segment.decode_key.as_ref().is_some_and(|key| {
        listed
            .get(&segment.name)
            .is_some_and(|(_, etag)| key.matches_object(etag.as_deref()))
    })
}

/// Drop every freshly added key whose rebuilt A0 is not byte-identical to
/// the segment's own (e.g. two models sharing a `model_id`): a key must never
/// describe a segment differently from its bytes.
fn verify_decode_keys(dir: &mut PaxCentroidDirectory, a0_by_name: &HashMap<String, Vec<u8>>) {
    for idx in 0..dir.segments.len() {
        let Some(expected) = a0_by_name.get(&dir.segments[idx].name) else {
            continue;
        };
        if dir.segments[idx].decode_key.is_none() {
            continue;
        }
        let rebuilt = dir.segment_decode(idx).map(|d| d.a0);
        if rebuilt.as_ref().ok() != Some(expected) {
            tracing::warn!(
                segment = %dir.segments[idx].name,
                "decode key does not reproduce the segment's A0; publishing without it"
            );
            dir.segments[idx].decode_key = None;
        }
    }
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

/// The newest valid directory of the collection data directory `dir_url`
/// (one LIST plus one read), or `None` when there is none or it is unreadable.
pub(crate) async fn load_newest_directory(
    fs: &dyn FileSystem,
    dir_url: &str,
) -> Option<PaxCentroidDirectory> {
    let entries = fs.list(dir_url.trim_end_matches('/')).await.ok()?;
    load_newest_valid(fs, &directory_objects_newest_first(&entries)).await
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
        let mut urls: HashMap<String, (String, Option<String>)> = HashMap::new();
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
            urls.insert(
                entry.name.clone(),
                (entry.url.clone(), entry.metadata.etag.clone()),
            );
            live.push(LiveSegment {
                name: entry.name.clone(),
                size_bytes,
            });
        }
        let objects = directory_objects_newest_first(&entries);
        let newest_generation = objects.first().map(|(g, _)| *g);
        let base = load_newest_valid(fs, &objects).await;
        let decode_keys = directory_decode_keys_write_enabled();
        // D6: a carried entry whose key no longer matches its object (a
        // rewrite in place) is reloaded, cells included.
        let mut plan = plan_reconcile_keeping(base.as_ref(), &live, |segment| {
            segment
                .decode_key
                .as_ref()
                .is_none_or(|_| key_matches_listing(segment, &urls))
        });
        // A base older than the newest object (the newest was corrupt) is
        // republished even if it matches the live set, and so is one whose
        // keys the write gate no longer allows (turning the gate off must
        // return the directory to v1 without waiting for another change).
        let base_is_newest = base.as_ref().map(|d| d.generation) == newest_generation;
        let keys_to_drop = !decode_keys
            && base
                .as_ref()
                .is_some_and(|d| d.segments.iter().any(|s| s.decode_key.is_some()));
        if !plan.changed && base_is_newest && !keys_to_drop {
            return Ok(None);
        }
        let mut a0_by_name = HashMap::new();
        for segment in &plan.to_load {
            let (url, etag) = &urls[&segment.name];
            match read_segment_a0_with_header(fs, url, segment.size_bytes).await {
                Ok(Some(read)) => {
                    if let Err(error) =
                        plan.builder
                            .add_covered(&segment.name, segment.size_bytes, &read.a0)
                    {
                        tracing::debug!(%url, %error, "segment not addressable by the directory");
                        plan.builder
                            .add_uncovered(&segment.name, segment.size_bytes);
                    } else if decode_keys
                        && let Some(key) =
                            read_decode_key(fs, url, segment.size_bytes, etag.as_deref(), &read)
                                .await
                        && plan.builder.set_decode_key(&segment.name, key).is_ok()
                    {
                        a0_by_name.insert(segment.name.clone(), read.a0_bytes);
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
        let mut dir = plan.builder.build(generation)?;
        verify_decode_keys(&mut dir, &a0_by_name);
        retain_current_decode_keys(&mut dir, decode_keys, &urls);
        let bytes = dir.to_bytes()?;
        let url = format!("{dir_url}/{}", object_name(generation));
        match fs.write_if_absent(&url, &bytes, None).await {
            Ok(()) => {
                for (old, entry) in &objects {
                    // TD-TIER-1 SC4-5: g-1 is kept remotely for readers but is
                    // no longer what queries read — drop its resident copy so
                    // it stops charging the grant (review N-b).
                    if old.saturating_add(1) == generation {
                        crate::storage::engines::sst::segment_io::forget_object(&entry.url);
                    }
                    if old.saturating_add(1) < generation {
                        match fs.delete(&entry.url).await {
                            // TD-TIER-1 SC4-5: a collected generation's resident
                            // copy must not keep charging the grant.
                            Ok(()) => {
                                crate::storage::engines::sst::segment_io::forget_object(&entry.url)
                            }
                            Err(error) => tracing::debug!(
                                url = %entry.url,
                                %error,
                                "centroid directory GC deferred"
                            ),
                        }
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
pub(crate) mod tests {
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

    pub(crate) fn write_segment(dir: &std::path::Path, name: &str, seed: usize) {
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

    /// TD-PAXDIR-1 D6: every covered segment is published with a decode key
    /// that rebuilds its header, Region A0, Region-A header and the footer's
    /// ANN fields exactly as the segment stores them.
    #[tokio::test]
    async fn published_decode_keys_reproduce_the_segment_metadata() {
        enable_trained_layout();
        // SAFETY: nextest runs each test in its own process.
        unsafe { std::env::set_var("PROXIMADB_PAX_WRITE_DIRECTORY_DECODE_KEYS", "1") };
        let tmp = tempfile::tempdir().unwrap();
        write_segment(tmp.path(), "a.pax", 0);
        write_segment(tmp.path(), "b.pax", 7);
        let fs = local_fs().await;
        let dir_url = tmp.path().to_string_lossy().to_string();
        publish_centroid_directory(&fs, &dir_url).await.unwrap();
        let dir = &newest(&fs, &dir_url).await[0].1;
        assert_eq!(dir.covered_segments().count(), 2, "both segments covered");
        for (idx, segment) in dir.covered_segments() {
            // Read the segment's own metadata through the FileSystem seam,
            // as ranged reads — the same way the query path would.
            let url = format!("{dir_url}/{}", segment.name);
            let read = |off: u64, len: u64| fs.read_range(&url, off, len);
            let prefix = read(0, SEG_HEADER_PREFIX_LEN as u64).await.unwrap();
            let header = SegmentHeaderPrefix::parse(&prefix).unwrap();
            let decoded = dir
                .segment_decode(idx)
                .expect("covered segments carry a key");
            assert_eq!(decoded.header_prefix, prefix);
            assert_eq!(
                decoded.a0,
                read(header.a0_off, header.a0_len).await.unwrap()
            );
            let hdr_len = proximadb_block_format::region_header_len(DIM as u32) as u64;
            assert_eq!(
                decoded.rabitq_header,
                read(header.rabitq_off, hdr_len).await.unwrap()
            );
            let real = SegmentFooterIndex::parse(
                &read(header.footer_off, header.footer_len).await.unwrap(),
            )
            .unwrap();
            let ann = SegmentFooterIndex::parse(&decoded.ann_footer).unwrap();
            assert_eq!(ann.row_count, real.row_count);
            assert_eq!(
                (ann.sq8_min.to_bits(), ann.sq8_scale.to_bits()),
                (real.sq8_min.to_bits(), real.sq8_scale.to_bits())
            );
            assert_eq!(ann.embed_dim, real.embed_dim);
            assert_eq!(ann.embed_count, real.embed_count);
            assert_eq!(ann.embed_quant_tag, real.embed_quant_tag);
            assert_eq!(ann.has_f32_tier, real.has_f32_tier);
            assert_eq!(
                (ann.rabitq_off, ann.rabitq_len, ann.sq8_off, ann.sq8_len),
                (real.rabitq_off, real.rabitq_len, real.sq8_off, real.sq8_len)
            );
            assert_eq!(
                (ann.a0_off, ann.a0_len, ann.c_off, ann.c_len),
                (real.a0_off, real.a0_len, real.c_off, real.c_len)
            );
            assert_eq!(ann.blocks, real.blocks);
            assert_eq!(ann.block_stats, real.block_stats);
        }
    }

    /// Turning the decode-key write gate off returns the next publish to the
    /// D1 (v1) format, even for keys carried from a keyed base.
    #[tokio::test]
    async fn turning_the_key_gate_off_drops_carried_keys() {
        enable_trained_layout();
        let tmp = tempfile::tempdir().unwrap();
        write_segment(tmp.path(), "a.pax", 0);
        let fs = local_fs().await;
        let dir_url = tmp.path().to_string_lossy().to_string();
        // SAFETY: nextest runs each test in its own process.
        unsafe { std::env::set_var("PROXIMADB_PAX_WRITE_DIRECTORY_DECODE_KEYS", "1") };
        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            Some(1)
        );
        let keyed = fs
            .read(&format!("{dir_url}/{}", object_name(1)))
            .await
            .unwrap();
        assert_eq!(keyed[4], 2, "v2 with a key");

        // No live change: turning the gate off alone republishes as v1,
        // once; a later change keeps v1.
        unsafe { std::env::set_var("PROXIMADB_PAX_WRITE_DIRECTORY_DECODE_KEYS", "0") };
        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            Some(2)
        );
        let v1 = fs
            .read(&format!("{dir_url}/{}", object_name(2)))
            .await
            .unwrap();
        assert_eq!(v1[4], 1, "the gate alone returns the directory to v1");
        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            None,
            "and only once"
        );
        write_segment(tmp.path(), "b.pax", 7);
        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            Some(3)
        );
        let bytes = fs
            .read(&format!("{dir_url}/{}", object_name(3)))
            .await
            .unwrap();
        assert_eq!(bytes[4], 1, "back to v1");
        let dir = PaxCentroidDirectory::parse(&bytes).unwrap();
        assert_eq!(dir.covered_segments().count(), 2);
        assert!(dir.segments.iter().all(|s| s.decode_key.is_none()));
    }

    /// A carried key whose recorded ETag no longer matches the listing means
    /// the object was rewritten in place: the segment is reloaded (cells and
    /// key), not carried. The local backend lists no ETag, so the newest
    /// generation is planted with a key recording one.
    #[tokio::test]
    async fn a_stale_etag_reloads_the_segment() {
        enable_trained_layout();
        // SAFETY: nextest runs each test in its own process.
        unsafe { std::env::set_var("PROXIMADB_PAX_WRITE_DIRECTORY_DECODE_KEYS", "1") };
        let tmp = tempfile::tempdir().unwrap();
        write_segment(tmp.path(), "a.pax", 0);
        let fs = local_fs().await;
        let dir_url = tmp.path().to_string_lossy().to_string();
        publish_centroid_directory(&fs, &dir_url).await.unwrap();
        let mut planted = newest(&fs, &dir_url).await.remove(0).1;
        planted.generation = 2;
        planted.segments[0].decode_key.as_mut().unwrap().object_etag = Some("old".into());
        fs.write_if_absent(
            &format!("{dir_url}/{}", object_name(2)),
            &planted.to_bytes().unwrap(),
            None,
        )
        .await
        .unwrap();

        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            Some(3),
            "a stale key is a change"
        );
        let dir = &newest(&fs, &dir_url).await[0].1;
        let key = dir.segments[0].decode_key.as_ref().expect("reloaded key");
        assert!(key.matches_object(None), "rebuilt from the listed object");
        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            None
        );
    }

    /// With the decode-key write gate off, a publish writes the D1 (v1)
    /// format byte for byte: no keys, no extra segment reads.
    #[tokio::test]
    async fn without_the_key_gate_the_directory_is_the_d1_format() {
        enable_trained_layout();
        // Default ON since D8 — disable via the kill-switch for this arm.
        // SAFETY: nextest runs each test in its own process.
        unsafe { std::env::set_var("PROXIMADB_PAX_WRITE_DIRECTORY_DECODE_KEYS", "0") };
        assert!(!directory_decode_keys_write_enabled());
        let tmp = tempfile::tempdir().unwrap();
        write_segment(tmp.path(), "a.pax", 0);
        let fs = local_fs().await;
        let dir_url = tmp.path().to_string_lossy().to_string();
        publish_centroid_directory(&fs, &dir_url).await.unwrap();
        let bytes = fs
            .read(&format!("{dir_url}/{}", object_name(1)))
            .await
            .unwrap();
        assert_eq!(bytes[4], 1, "v1");
        let dir = PaxCentroidDirectory::parse(&bytes).unwrap();
        assert_eq!(dir.covered_segments().count(), 1);
        assert!(dir.segments.iter().all(|s| s.decode_key.is_none()));
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
    fn write_gate_defaults_on_with_kill_switch() {
        // Default ON since D8; the gate is a kill-switch.
        // SAFETY: nextest runs each test in its own process.
        unsafe { std::env::remove_var("PROXIMADB_PAX_WRITE_CENTROID_DIRECTORY") };
        assert!(centroid_directory_write_enabled());
        unsafe { std::env::set_var("PROXIMADB_PAX_WRITE_CENTROID_DIRECTORY", "0") };
        assert!(!centroid_directory_write_enabled());
    }
}
