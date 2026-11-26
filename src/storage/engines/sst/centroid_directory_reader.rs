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
//!
//! ADR-042 / TD-PAXMETA-1 (filtered queries, [`plan_filtered_segments`]): a
//! FILTERED query on a segment whose v3 entry matches the listing's `(size,
//! ETag)` binding evaluates the entry's per-block prune statistics in RAM. A
//! segment none of whose blocks can match is skipped with zero reads; a
//! surviving segment's Stage F takes its block table, shred field map and
//! surviving blocks from the entry (no header or footer read) and fetches only
//! the surviving blocks, and its cascade and rehydration take their metadata
//! from the decode key exactly as a D6-planned segment does. Gates: both read
//! gates above, and `PROXIMADB_PAX_READ_BLOCK_PRUNE_STATS_DISABLE` (unset ⇒
//! on; truthy ⇒ filtered queries never use the directory, the pre-ADR-042
//! path).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

use proximadb_storage_common::centroid_directory::{LiveSegment, PaxCentroidDirectory};
use proximadb_storage_filesystem_types::{DirEntry, FileSystem};

use crate::storage::engines::sst::centroid_directory_publisher::directory_objects_newest_first;
use crate::storage::engines::sst::segment_format::{
    SegmentInvariants, centroid_directory_nprobe_for, pax_rabitq_pool_for_top_k,
    prefix_read_covers_segment,
};

// READ gate: plan unfiltered ANN queries on the collection centroid directory.
proximadb_config::env_kill_switch! {
    /// Kill-switch gate; default ON (0|false|off|no disables).
    pub fn centroid_directory_read_enabled() { gate: "PROXIMADB_PAX_READ_CENTROID_DIRECTORY", default: true }
}

// TD-PAXDIR-1 D6 READ gate: planned segments with a decode key take their
// query-path metadata from the directory instead of reading it. Default ON
// since D8; needs directories published with the write-side gate on.
proximadb_config::env_kill_switch! {
    /// Kill-switch gate; default ON (0|false|off|no disables).
    pub fn directory_decode_keys_read_enabled() { gate: "PROXIMADB_PAX_READ_DIRECTORY_DECODE_KEYS", default: true }
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
    /// ADR-042: set on a FILTERED query's choice (whose `cells` are empty and
    /// unused — the filtered coarse probe ranks its own A0).
    filtered: Option<FilteredSegmentPlan>,
}

/// ADR-042 / TD-PAXMETA-1: one segment's part of a filtered query planned on
/// the directory's per-block prune statistics.
#[derive(Debug)]
pub(crate) struct FilteredSegmentPlan {
    /// The footer Stage F needs, rebuilt from the decode key (block table,
    /// shred field map, prune stats). Never the segment's own footer.
    pub(crate) footer: proximadb_storage_common::segment_layout::SegmentFooterIndex,
    /// The segment carries Region E, which the directory does not describe.
    pub(crate) has_filter_region: bool,
    /// Blocks the prune stats keep (every block when the entry has none).
    pub(crate) candidate_blocks: Vec<usize>,
}

impl DirectoryChoice {
    /// D6: the segment's header, A0, Region-A header and ANN footer rebuilt
    /// from the directory, or `None` (no key, gate off, or the rebuild
    /// failed — the scan then reads them as before). The footer holds only the
    /// unfiltered ANN cascade's fields, so the result must never be admitted
    /// to the invariants cache.
    pub(crate) fn decode_invariants(&self) -> Option<SegmentInvariants> {
        let (dir, idx) = self.decode.as_ref()?;
        // An uncovered keyed entry (ADR-042 filtered plans only) has no A0:
        // its header and footer come from the key, Region A is read as usual.
        match dir.segment_metadata_decode(*idx) {
            Ok(d) => Some(SegmentInvariants {
                header_bytes: d.header_prefix,
                region_bytes: None,
                footer_bytes: d.ann_footer,
                a0_bytes: (!d.a0.is_empty()).then(|| Arc::from(d.a0)),
                rabitq_header_bytes: (!d.rabitq_header.is_empty())
                    .then(|| Arc::from(d.rabitq_header)),
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

    /// ADR-042: the filtered plan, when this is a filtered query's choice.
    pub(crate) fn filtered(&self) -> Option<&FilteredSegmentPlan> {
        self.filtered.as_ref()
    }

    /// Whether the query scans this segment at all: an unfiltered choice
    /// with no selected cell, or a filtered one with no surviving block, is
    /// skipped with zero reads.
    pub(crate) fn keeps_segment(&self) -> bool {
        match &self.filtered {
            Some(plan) => !plan.candidate_blocks.is_empty(),
            None => !self.cells.is_empty(),
        }
    }
}

/// Outcome of [`plan_filtered_segments`] for the query-scoped I/O trace.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FilteredPlanCounts {
    /// Segments planned on the directory (entry valid and carrying facts).
    pub(crate) segments_planned: u64,
    /// Planned segments skipped because no block can match.
    pub(crate) segments_pruned: u64,
    /// Blocks of planned segments, and those the prune stats dropped.
    pub(crate) blocks_total: u64,
    pub(crate) blocks_pruned: u64,
}

/// ADR-042 / TD-PAXMETA-1: plan a FILTERED query on the directory. Every
/// discovered segment whose entry carries a v3 decode key matching the
/// listing's `(size, ETag)` binding gets a choice: the blocks its prune
/// statistics keep for `filter` (all blocks when the entry has no stats), and
/// — when the segment is larger than the cold prefix read, as for D6 — its
/// decode key for the cascade's metadata. Segments absent from the map keep
/// the per-segment path (stale, missing, v2 or unkeyed entries).
pub(crate) fn plan_filtered_segments<'a>(
    dir: &Arc<PaxCentroidDirectory>,
    files: impl IntoIterator<Item = (&'a str, Option<u64>, Option<&'a str>)>,
    filter: &proximadb_filter_expression::FilterExpression,
) -> (HashMap<String, Arc<DirectoryChoice>>, FilteredPlanCounts) {
    let by_name = dir.segment_indices_by_name();
    let mut counts = FilteredPlanCounts::default();
    let mut out = HashMap::new();
    for (url, size, etag) in files {
        let Some(&idx) = by_name.get(object_name(url)) else {
            continue;
        };
        let entry = &dir.segments[idx];
        if !entry.key_matches_listing(size, etag) {
            continue;
        }
        let Some(key) = entry.decode_key.as_ref() else {
            continue;
        };
        let footer = match key.filtered_footer() {
            Ok(Some(footer)) => footer,
            Ok(None) => continue, // a v2 key: no facts
            Err(error) => {
                tracing::debug!(%error, url, "filter facts unusable; per-segment path");
                continue;
            }
        };
        let has_filter_region = key
            .filter_facts
            .as_ref()
            .is_some_and(|facts| facts.has_filter_region);
        let candidate_blocks: Vec<usize> = (0..footer.blocks.len())
            .filter(|&b| {
                footer
                    .block_prune_stats
                    .as_ref()
                    .is_none_or(|stats| stats.block_may_match(b, filter))
            })
            .collect();
        counts.segments_planned += 1;
        counts.blocks_total += footer.blocks.len() as u64;
        counts.blocks_pruned += (footer.blocks.len() - candidate_blocks.len()) as u64;
        if candidate_blocks.is_empty() {
            counts.segments_pruned += 1;
        }
        let decode = size
            .is_some_and(|size| !prefix_read_covers_segment(size))
            .then(|| (Arc::clone(dir), idx));
        out.insert(
            url.to_owned(),
            Arc::new(DirectoryChoice {
                cells: Arc::from(Vec::new()),
                decode,
                blocks: Mutex::default(),
                filtered: Some(FilteredSegmentPlan {
                    footer,
                    has_filter_region,
                    candidate_blocks,
                }),
            }),
        );
    }
    (out, counts)
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

/// A collection's directory objects as last listed: the newest (the cache
/// identity) and every listed generation, newest first (the fallback order
/// when the newest is corrupt or unreadable).
#[derive(Debug, Clone)]
struct Listed {
    newest: ListedObject,
    generations: Vec<(u64, String)>,
}

/// Directory objects per collection data directory, as last listed.
fn newest_listed() -> &'static Mutex<HashMap<String, Listed>> {
    static MAP: OnceLock<Mutex<HashMap<String, Listed>>> = OnceLock::new();
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
    let objects = directory_objects_newest_first(entries);
    let newest = objects.first().map(|(_, entry)| Listed {
        newest: ListedObject {
            url: entry.url.clone(),
            size: entry.metadata.size,
        },
        generations: objects
            .iter()
            .map(|(generation, entry)| (*generation, entry.url.clone()))
            .collect(),
    });
    let key = dir_key(dir_url);
    match newest {
        Some(listed) => {
            lock(newest_listed()).insert(key, listed);
        }
        None => {
            lock(newest_listed()).remove(&key);
            lock(loaded()).remove(&key);
        }
    }
}

/// The newest VALID listed directory for `dir_url`, loading it on first use
/// of the newest listed generation. A corrupt or unreadable newest generation
/// falls back to the next listed one, as the publisher's base load does
/// (post-#2535 review L3). The fallback is cached against the newest object
/// (so a corrupt one is not re-read per query) only when every newer object
/// was read and found invalid; after a read ERROR it is served uncached, so
/// the next query retries (#2581 review LOW-1). Every entry is still
/// validated against the listing by its consumers. `None` when none is
/// listed or none is valid (the query then runs per segment).
pub(crate) async fn load_directory(
    fs: &dyn FileSystem,
    dir_url: &str,
) -> Option<Arc<PaxCentroidDirectory>> {
    let key = dir_key(dir_url);
    let listed = lock(newest_listed()).get(&key).cloned()?;
    let object = listed.newest;
    if let Some((cached, dir)) = lock(loaded()).get(&key)
        && *cached == object
    {
        return Some(Arc::clone(dir));
    }
    // TD-TIER-1 SC4-5: a resident directory object costs no GET.
    let (dir, skipped_unreadable) =
        crate::storage::engines::sst::centroid_directory_publisher::first_valid_directory_noting_read_errors(
            fs,
            &listed.generations,
            true,
        )
        .await;
    let Some(dir) = dir.map(Arc::new) else {
        tracing::debug!(url = %object.url, "no valid centroid directory; per-segment path");
        return None;
    };
    if skipped_unreadable {
        // #2581 review LOW-1: a newer generation failed to READ (possibly
        // transiently): serve the fallback but do not cache it against the
        // newest object, so the next query retries the newest.
        return Some(dir);
    }
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
    // TD-RDSTRAT-14: the probe policy needs corpus/geometry context. The
    // corpus size comes from plan_probe itself (`usable_rows` — the
    // live AND stamp-matched set it actually plans over), handed to the
    // closure below. Deriving it here from the directory's segment list
    // counted entries whose size no longer matches the live object, which
    // inflated a byte_budget plan by (counted/usable)^0.25 mid-compaction.
    // The rowgroup geometry comes from the lexicographically-smallest live
    // URL (deterministic across restarts even when a collection spans URL
    // schemes mid-migration; its residual nondeterminism is filed as
    // RDSTRAT-14-F14) — resolved LAZILY, inside the closure and only for
    // byte_budget, so the default path pays no registry/env scan per query
    // (#2419 review round 19).
    let first_url = url_by_name.values().min().map(String::as_str).unwrap_or("");
    let plan = dir.plan_probe(
        query,
        &live,
        // `avg_group_bytes` comes from `plan_probe`, which alone knows the
        // usable, stamp-matched set and the real grouping — see its doc.
        |total_cells: usize, usable_rows: u64, avg_group_bytes: u64| {
            centroid_directory_nprobe_for(first_url, total_cells, usable_rows, avg_group_bytes)
        },
        |rows| {
            let rows = usize::try_from(rows).unwrap_or(usize::MAX);
            pax_rabitq_pool_for_top_k(k, rows) as u64
        },
    )?;
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
                            filtered: None,
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
    use serial_test::serial;

    /// A key is used only for the object whose listed ETag it recorded: a
    /// rewritten object (another ETag) or a listing without one keeps the
    /// segment on its own metadata reads, while its cells are still planned.
    #[tokio::test]
    #[serial]
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

    /// Post-#2535 review L3: a corrupt (truncated) newest generation falls
    /// back to the newest VALID listed one, as the publisher's base load
    /// does, instead of disabling the directory; the fallback is cached
    /// against the newest object (no re-read of the bad one per query), and
    /// its entries still bind to the listing's size.
    #[tokio::test]
    #[serial]
    async fn a_corrupt_newest_generation_falls_back_to_the_newest_valid_one() {
        // SAFETY: nextest runs each test in its own process.
        unsafe {
            std::env::set_var("PROXIMADB_PAX_COALESCED_RABITQ", "1");
            std::env::set_var("PROXIMADB_PAX_WRITE_A0_TRAIN", "1");
            std::env::set_var("PROXIMADB_IVF_K", "4");
            std::env::set_var("PROXIMADB_PAX_WRITE_DIRECTORY_DECODE_KEYS", "1");
        }
        let tmp = tempfile::tempdir().unwrap();
        let fs = LocalFileSystem::new(LocalConfig::default()).await.unwrap();
        let dir_url = tmp.path().to_string_lossy().to_string();
        let write =
            crate::storage::engines::sst::centroid_directory_publisher::tests::write_segment;
        write(tmp.path(), "a.pax", 0);
        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            Some(1)
        );
        write(tmp.path(), "b.pax", 7);
        assert_eq!(
            publish_centroid_directory(&fs, &dir_url).await.unwrap(),
            Some(2)
        );
        let newest = tmp
            .path()
            .join(proximadb_storage_common::centroid_directory::object_name(2));
        let bytes = std::fs::read(&newest).unwrap();
        std::fs::write(&newest, &bytes[..bytes.len() / 2]).unwrap();

        note_listing(&dir_url, &fs.list(&dir_url).await.unwrap());
        let dir = load_directory(&fs, &dir_url)
            .await
            .expect("falls back to the valid older generation");
        assert_eq!(dir.generation, 1);
        let by_name = dir.segment_indices_by_name();
        assert!(!by_name.contains_key("b.pax"), "gen 1 predates b.pax");
        let a = &dir.segments[by_name["a.pax"]];
        let size = std::fs::metadata(tmp.path().join("a.pax")).unwrap().len();
        assert!(
            a.key_matches_listing(Some(size), None),
            "a live, unchanged entry"
        );
        assert!(
            !a.key_matches_listing(Some(size + 1), None),
            "entries still bind to the listed size"
        );
        let again = load_directory(&fs, &dir_url).await.unwrap();
        assert!(Arc::ptr_eq(&dir, &again), "the fallback is cached");
    }

    /// Fails the first `fail` whole-object reads of `target` (a transient
    /// object-store error); everything else is the local backend.
    #[derive(Debug)]
    struct FlakyReadFs {
        inner: Arc<LocalFileSystem>,
        target: String,
        fail: std::sync::atomic::AtomicUsize,
    }

    proximadb_storage_filesystem_types::forward_filesystem_impl! {
        impl FileSystem for FlakyReadFs => inner;
        forward [
            as_any, filesystem_type, supports_mmap, write,
            supports_bounded_local_file_write, write_local_file, write_if_absent, append,
            supports_append, supports_conditional_replace, delete, exists, metadata,
            create_dir, create_dir_all, copy, move_file, sync, sync_file, sync_data, open_file,
        ];
        {
            async fn read(&self, path: &str) -> proximadb_storage_filesystem_types::FsResult<Vec<u8>> {
                use std::sync::atomic::Ordering;
                if path.ends_with(&self.target)
                    && self
                        .fail
                        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
                        .is_ok()
                {
                    return Err(proximadb_storage_filesystem_types::FilesystemError::Io(
                        std::io::Error::other("injected transient read error"),
                    ));
                }
                self.inner.read(path).await
            }

            async fn read_range(
                &self,
                path: &str,
                offset: u64,
                length: u64,
            ) -> proximadb_storage_filesystem_types::FsResult<Vec<u8>> {
                self.inner.read_range(path, offset, length).await
            }

            async fn list(
                &self,
                path: &str,
            ) -> proximadb_storage_filesystem_types::FsResult<Vec<DirEntry>> {
                self.inner.list(path).await
            }
        }
    }

    /// #2581 review LOW-1: a newest generation that fails to READ (a
    /// transient error, not corruption) still falls back for this query, but
    /// the fallback is not cached against it: the next query retries the
    /// newest and gets it.
    #[tokio::test]
    #[serial]
    async fn a_transient_read_error_on_the_newest_generation_is_retried() {
        let tmp = tempfile::tempdir().unwrap();
        let local = Arc::new(LocalFileSystem::new(LocalConfig::default()).await.unwrap());
        let dir_url = tmp.path().to_string_lossy().to_string();
        std::fs::write(tmp.path().join("l0.pax"), b"untrained").unwrap();
        assert_eq!(
            publish_centroid_directory(local.as_ref(), &dir_url)
                .await
                .unwrap(),
            Some(1)
        );
        std::fs::write(tmp.path().join("l1.pax"), b"untrained").unwrap();
        assert_eq!(
            publish_centroid_directory(local.as_ref(), &dir_url)
                .await
                .unwrap(),
            Some(2)
        );
        let newest = proximadb_storage_common::centroid_directory::object_name(2);
        let fs = FlakyReadFs {
            inner: local.clone(),
            target: format!("/{newest}"),
            fail: std::sync::atomic::AtomicUsize::new(1),
        };
        let listing = fs.list(&dir_url).await.unwrap();
        // Serve the newest from the store, not a resident copy, so the
        // injected read error is what this query sees.
        for entry in &listing {
            crate::storage::engines::sst::segment_io::forget_object(&entry.url);
        }
        note_listing(&dir_url, &listing);
        assert_eq!(
            load_directory(&fs, &dir_url).await.unwrap().generation,
            1,
            "this query falls back"
        );
        assert_eq!(
            load_directory(&fs, &dir_url).await.unwrap().generation,
            2,
            "the next query retries the newest instead of a cached fallback"
        );
    }
}
