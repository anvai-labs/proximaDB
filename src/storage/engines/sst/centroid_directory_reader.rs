//! TD-PAXDIR-1 D2: query-side use of the collection-global centroid directory.
//!
//! Discovery's LIST already returns the directory object
//! (`centroids-<gen>.pxcd`) beside the segments; [`note_listing`] remembers
//! the newest one per collection data directory, so locating it costs no
//! extra request. A query then:
//!
//! 1. loads that generation (one GET the first time, cached per collection
//!    until a newer generation is listed);
//! 2. ranks every usable cell across every segment in RAM and selects the
//!    global `nprobe` nearest ([`PaxCentroidDirectory::plan_probe`]);
//! 3. hands each usable segment its selected A0 cells (the coarse probe uses
//!    them instead of ranking its own A0), and skips usable segments that own
//!    none of the selected cells with **zero reads**.
//!
//! A usable segment is one the directory covers and that is still live with
//! the recorded size. Every other segment — new flushes, untrained or legacy
//! segments, rewritten objects — keeps the per-segment cascade, so a missing,
//! stale or corrupt directory changes cost, never results.
//!
//! Gate: `PROXIMADB_PAX_READ_CENTROID_DIRECTORY` — default ON since D8 (the
//! P5b gate run passed), kill-switch off. Filtered and exact queries never
//! use the directory.
//!
//! TD-PAXDIR-1 D6: with `PROXIMADB_PAX_READ_DIRECTORY_DECODE_KEYS` (default
//! ON since D8) also on, a planned segment whose directory entry carries a decode key
//! rebuilds its header, Region A0, Region-A header and ANN footer from the
//! directory on an invariants-cache miss, so its first read is its probed
//! cells ([`DirectoryChoice::decode_invariants`]).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

use proximadb_storage_common::centroid_directory::{LiveSegment, PaxCentroidDirectory};
use proximadb_storage_filesystem_types::{DirEntry, FileSystem};

use crate::storage::engines::sst::block_cluster::env_kill_switch_off;
use crate::storage::engines::sst::centroid_directory_publisher::directory_objects_newest_first;
use crate::storage::engines::sst::segment_format::{
    SegmentInvariants, centroid_directory_nprobe, pax_rabitq_pool_for_top_k,
    prefix_read_covers_segment,
};

/// READ gate: plan unfiltered ANN queries on the collection centroid directory.
pub fn centroid_directory_read_enabled() -> bool {
    env_kill_switch_off("PROXIMADB_PAX_READ_CENTROID_DIRECTORY")
}

/// TD-PAXDIR-1 D6 READ gate: planned segments with a decode key take their
/// query-path metadata from the directory instead of reading it. Default ON
/// since D8; needs directories published with the write-side gate on.
pub fn directory_decode_keys_read_enabled() -> bool {
    env_kill_switch_off("PROXIMADB_PAX_READ_DIRECTORY_DECODE_KEYS")
}

/// One segment's part of a directory-planned query.
#[derive(Debug)]
pub(crate) struct DirectoryChoice {
    /// The Region A0 cells to probe, in rank order (empty ⇒ skip the segment).
    pub(crate) cells: Arc<[u32]>,
    /// D6: the directory and this segment's index in it, when its entry has a
    /// decode key, the decode-key read gate is on, and the segment is larger
    /// than the scan's cold prefix read.
    decode: Option<(Arc<PaxCentroidDirectory>, usize)>,
    /// D6: top-k row-group blocks the cascade fetched whole (by absolute
    /// offset), handed to this query's rehydration so the rows cost one wave.
    /// Lives as long as the query's choice; rehydration takes what it uses.
    blocks: Mutex<HashMap<u64, Arc<[u8]>>>,
}

impl DirectoryChoice {
    /// D6: the segment's header, A0, Region-A header and ANN footer rebuilt
    /// from the directory, or `None` (no key, gate off, or the rebuild
    /// failed — the scan then reads them as before). The footer holds only the
    /// unfiltered ANN cascade's fields, so the result must never be admitted
    /// to the invariants cache.
    pub(crate) fn decode_invariants(&self) -> Option<SegmentInvariants> {
        let (dir, idx) = self.decode.as_ref()?;
        match dir.segment_decode(*idx) {
            Ok(d) => Some(SegmentInvariants {
                header_bytes: d.header_prefix,
                region_bytes: None,
                footer_bytes: d.ann_footer,
                a0_bytes: Some(Arc::from(d.a0)),
                rabitq_header_bytes: Some(Arc::from(d.rabitq_header)),
            }),
            Err(error) => {
                tracing::debug!(%error, "decode key unusable; reading segment metadata");
                None
            }
        }
    }

    /// D6: keep a top-k block the cascade fetched for this query's
    /// rehydration.
    pub(crate) fn stash_block(&self, offset: u64, bytes: Arc<[u8]>) {
        lock(&self.blocks).insert(offset, bytes);
    }

    /// D6: take a block the cascade stashed (`None`: rehydration reads it).
    pub(crate) fn take_block(&self, offset: u64) -> Option<Arc<[u8]>> {
        lock(&self.blocks).remove(&offset)
    }
}

/// Collection directories whose parsed centroid directory stays cached. Each
/// entry is one collection's directory (a few KB to a few MB); past the bound
/// an arbitrary entry is evicted, which costs that collection one re-read.
const MAX_CACHED_DIRECTORIES: usize = 1024;

/// A listed directory object: its URL and listed size. Generations restart at
/// 1 if every directory object is deleted, so the URL alone cannot identify
/// the bytes; the size makes a stale cache hit far less likely, and every
/// entry is re-validated against the live set anyway.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ListedObject {
    url: String,
    size: u64,
}

/// Newest directory object per collection data directory, as last listed.
fn newest_listed() -> &'static Mutex<HashMap<String, ListedObject>> {
    static MAP: OnceLock<Mutex<HashMap<String, ListedObject>>> = OnceLock::new();
    MAP.get_or_init(Default::default)
}

/// Loaded directory per collection data directory, with the object it was
/// read from.
fn loaded() -> &'static Mutex<HashMap<String, (ListedObject, Arc<PaxCentroidDirectory>)>> {
    static MAP: OnceLock<Mutex<HashMap<String, (ListedObject, Arc<PaxCentroidDirectory>)>>> =
        OnceLock::new();
    MAP.get_or_init(Default::default)
}

/// Lock a cache map. The maps hold plain data with no cross-entry
/// invariants, so a poisoned lock is recovered rather than turning the
/// feature off for the rest of the process.
fn lock<T>(map: &Mutex<T>) -> MutexGuard<'_, T> {
    map.lock().unwrap_or_else(PoisonError::into_inner)
}

fn dir_key(dir_url: &str) -> String {
    dir_url.trim_end_matches('/').to_owned()
}

/// Record the newest directory object in a collection directory listing.
/// Called by discovery with the entries it already listed. A listing with no
/// directory forgets the collection's cached one too.
pub(crate) fn note_listing(dir_url: &str, entries: &[DirEntry]) {
    let newest = directory_objects_newest_first(entries)
        .first()
        .map(|(_, entry)| ListedObject {
            url: entry.url.clone(),
            size: entry.metadata.size,
        });
    let key = dir_key(dir_url);
    match newest {
        Some(object) => {
            lock(newest_listed()).insert(key, object);
        }
        None => {
            lock(newest_listed()).remove(&key);
            lock(loaded()).remove(&key);
        }
    }
}

/// The newest listed directory for `dir_url`, loading it on first use of its
/// generation. `None` when none is listed or it fails to read or parse (the
/// query then runs per segment).
pub(crate) async fn load_directory(
    fs: &dyn FileSystem,
    dir_url: &str,
) -> Option<Arc<PaxCentroidDirectory>> {
    let key = dir_key(dir_url);
    let object = lock(newest_listed()).get(&key).cloned()?;
    if let Some((cached, dir)) = lock(loaded()).get(&key)
        && *cached == object
    {
        return Some(Arc::clone(dir));
    }
    let url = &object.url;
    // TD-TIER-1 SC4-5: a resident directory object costs no GET.
    let bytes =
        match crate::storage::engines::sst::segment_io::ResidentRangeRead::read_resident(fs, url)
            .await
        {
            Ok(bytes) => bytes,
            Err(error) => {
                tracing::debug!(%url, %error, "centroid directory unreadable; per-segment path");
                return None;
            }
        };
    let dir = match PaxCentroidDirectory::parse(&bytes) {
        Ok(dir) => Arc::new(dir),
        Err(error) => {
            tracing::warn!(%url, %error, "corrupt centroid directory; per-segment path");
            return None;
        }
    };
    let mut cache = lock(loaded());
    if cache.len() >= MAX_CACHED_DIRECTORIES
        && !cache.contains_key(&key)
        && let Some(evict) = cache.keys().next().cloned()
    {
        cache.remove(&evict);
    }
    cache.insert(key, (object, Arc::clone(&dir)));
    Some(dir)
}

fn object_name(url: &str) -> &str {
    url.rsplit('/').next().unwrap_or(url)
}

/// Per-segment directory choices for one query, keyed by segment URL: the A0
/// cells to probe (empty ⇒ skip the segment) and, with D6, the decode key.
/// Segments absent from the map keep the per-segment path. `files` are the
/// discovered `(url, size, etag)` triples; a segment of unknown size is never
/// usable (its stamp cannot be checked), and a key is used only for the
/// object whose listed ETag it records.
pub(crate) fn plan_segments<'a>(
    dir: &Arc<PaxCentroidDirectory>,
    files: impl IntoIterator<Item = (&'a str, Option<u64>, Option<&'a str>)>,
    query: &[f32],
    k: usize,
) -> Option<HashMap<String, Arc<DirectoryChoice>>> {
    let mut url_by_name = HashMap::new();
    let mut etag_by_name: HashMap<String, Option<&str>> = HashMap::new();
    let mut live = Vec::new();
    for (url, size, etag) in files {
        let Some(size_bytes) = size else { continue };
        let name = object_name(url);
        url_by_name.insert(name.to_owned(), url.to_owned());
        etag_by_name.insert(name.to_owned(), etag);
        live.push(LiveSegment {
            name: name.to_owned(),
            size_bytes,
        });
    }
    let plan = dir.plan_probe(query, &live, centroid_directory_nprobe, |rows| {
        let rows = usize::try_from(rows).unwrap_or(usize::MAX);
        pax_rabitq_pool_for_top_k(k, rows) as u64
    })?;
    let decode_keys = directory_decode_keys_read_enabled();
    let keyed: HashMap<&str, usize> = dir
        .segments
        .iter()
        .enumerate()
        // D6: a segment the cold prefix read fetches whole keeps that read
        // (one request for every byte, then cached); only larger ones use
        // the key.
        .filter(|(_, s)| {
            decode_keys
                && !prefix_read_covers_segment(s.size_bytes)
                && s.decode_key.as_ref().is_some_and(|key| {
                    etag_by_name
                        .get(&s.name)
                        .is_some_and(|etag| key.matches_object(*etag))
                })
        })
        .map(|(idx, s)| (s.name.as_str(), idx))
        .collect();
    Some(
        plan.cells_by_segment
            .into_iter()
            .filter_map(|(name, cells)| {
                let decode = keyed.get(name.as_str()).map(|&idx| (Arc::clone(dir), idx));
                url_by_name.remove(&name).map(|url| {
                    let cells = Arc::<[u32]>::from(cells);
                    (
                        url,
                        Arc::new(DirectoryChoice {
                            cells,
                            decode,
                            blocks: Mutex::default(),
                        }),
                    )
                })
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::engines::sst::centroid_directory_publisher::publish_centroid_directory;
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};

    /// A key is used only for the object whose listed ETag it recorded: a
    /// rewritten object (another ETag) or a listing without one keeps the
    /// segment on its own metadata reads, while its cells are still planned.
    #[tokio::test]
    async fn a_decode_key_is_used_only_for_the_listed_object_it_recorded() {
        // SAFETY: nextest runs each test in its own process.
        unsafe {
            std::env::set_var("PROXIMADB_PAX_COALESCED_RABITQ", "1");
            std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "1");
            std::env::set_var("PROXIMADB_IVF_K", "4");
            std::env::set_var("PROXIMADB_PAX_WRITE_DIRECTORY_DECODE_KEYS", "1");
            std::env::set_var("PROXIMADB_PAX_READ_DIRECTORY_DECODE_KEYS", "1");
            // Header-only prefix reads, so the small segment is key-eligible.
            std::env::set_var("PROXIMADB_PAX_PREFIX_PREFETCH_BYTES", "0");
        }
        let tmp = tempfile::tempdir().unwrap();
        let fs = LocalFileSystem::new(LocalConfig::default()).await.unwrap();
        let dir_url = tmp.path().to_string_lossy().to_string();
        crate::storage::engines::sst::centroid_directory_publisher::tests::write_segment(
            tmp.path(),
            "a.pax",
            0,
        );
        publish_centroid_directory(&fs, &dir_url).await.unwrap();
        let bytes = fs
            .read(&format!(
                "{dir_url}/{}",
                proximadb_storage_common::centroid_directory::object_name(1)
            ))
            .await
            .unwrap();
        let mut dir = PaxCentroidDirectory::parse(&bytes).unwrap();
        // The local backend lists no ETag; give the key one, as a cloud
        // listing would have recorded.
        dir.segments[0].decode_key.as_mut().unwrap().object_etag = Some("e1".into());
        let dir = Arc::new(dir);
        let url = format!("{dir_url}/a.pax");
        let size = dir.segments[0].size_bytes;
        let query = vec![0.5f32; 16];
        let decoded = |etag: Option<&'static str>| {
            let plan = plan_segments(&dir, [(url.as_str(), Some(size), etag)], &query, 10)
                .expect("the segment is planned");
            let choice = &plan[&url];
            (
                !choice.cells.is_empty(),
                choice.decode_invariants().is_some(),
            )
        };
        assert_eq!(decoded(Some("e1")), (true, true), "same object");
        assert_eq!(decoded(Some("e2")), (true, false), "rewritten object");
        assert_eq!(decoded(None), (true, false), "listing without an ETag");
    }

    #[test]
    fn read_gate_defaults_on_with_kill_switch() {
        // Default ON since D8; the gate is a kill-switch.
        // SAFETY: nextest runs each test in its own process.
        unsafe { std::env::remove_var("PROXIMADB_PAX_READ_CENTROID_DIRECTORY") };
        assert!(centroid_directory_read_enabled());
        unsafe { std::env::set_var("PROXIMADB_PAX_READ_CENTROID_DIRECTORY", "off") };
        assert!(!centroid_directory_read_enabled());
    }

    #[tokio::test]
    async fn the_listed_generation_is_loaded_once_and_replaced_when_a_newer_one_lists() {
        let tmp = tempfile::tempdir().unwrap();
        let fs = LocalFileSystem::new(LocalConfig::default()).await.unwrap();
        let dir_url = tmp.path().to_string_lossy().to_string();
        std::fs::write(tmp.path().join("l0.pax"), b"untrained").unwrap();
        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            Some(1)
        );

        note_listing(&dir_url, &fs.list(&dir_url).await.unwrap());
        let first = load_directory(&fs, &dir_url).await.unwrap();
        assert_eq!(first.generation, 1);
        let again = load_directory(&fs, &dir_url).await.unwrap();
        assert!(Arc::ptr_eq(&first, &again), "served from the cache");

        std::fs::write(tmp.path().join("l1.pax"), b"untrained").unwrap();
        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            Some(2)
        );
        // Not listed yet: the cached generation keeps serving.
        assert_eq!(load_directory(&fs, &dir_url).await.unwrap().generation, 1);
        note_listing(&dir_url, &fs.list(&dir_url).await.unwrap());
        assert_eq!(load_directory(&fs, &dir_url).await.unwrap().generation, 2);

        // A listing without any directory forgets it.
        note_listing(&dir_url, &[]);
        assert!(load_directory(&fs, &dir_url).await.is_none());
    }
}
