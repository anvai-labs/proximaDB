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
//! Gate: `PROXIMADB_PAX_READ_CENTROID_DIRECTORY` (default OFF). Filtered and
//! exact queries never use the directory.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use proximadb_storage_common::centroid_directory::{LiveSegment, PaxCentroidDirectory};
use proximadb_storage_filesystem_types::{DirEntry, FileSystem};

use crate::storage::engines::sst::block_cluster::env_gate_on;
use crate::storage::engines::sst::centroid_directory_publisher::directory_objects_newest_first;
use crate::storage::engines::sst::segment_format::{
    centroid_directory_nprobe, pax_rabitq_pool_for_top_k,
};

/// READ gate: plan unfiltered ANN queries on the collection centroid directory.
pub fn centroid_directory_read_enabled() -> bool {
    env_gate_on("PROXIMADB_PAX_READ_CENTROID_DIRECTORY")
}

/// Newest directory object URL per collection data directory, as last listed.
fn newest_listed() -> &'static Mutex<HashMap<String, String>> {
    static MAP: OnceLock<Mutex<HashMap<String, String>>> = OnceLock::new();
    MAP.get_or_init(Default::default)
}

/// Loaded directory per collection data directory: `(object url, directory)`.
fn loaded() -> &'static Mutex<HashMap<String, (String, Arc<PaxCentroidDirectory>)>> {
    static MAP: OnceLock<Mutex<HashMap<String, (String, Arc<PaxCentroidDirectory>)>>> =
        OnceLock::new();
    MAP.get_or_init(Default::default)
}

fn dir_key(dir_url: &str) -> String {
    dir_url.trim_end_matches('/').to_owned()
}

/// Record the newest directory object in a collection directory listing.
/// Called by discovery with the entries it already listed.
pub(crate) fn note_listing(dir_url: &str, entries: &[DirEntry]) {
    let newest = directory_objects_newest_first(entries)
        .first()
        .map(|(_, entry)| entry.url.clone());
    let Ok(mut map) = newest_listed().lock() else {
        return;
    };
    match newest {
        Some(url) => {
            map.insert(dir_key(dir_url), url);
        }
        None => {
            map.remove(&dir_key(dir_url));
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
    let url = newest_listed().lock().ok()?.get(&key).cloned()?;
    if let Some((cached_url, dir)) = loaded().lock().ok()?.get(&key)
        && *cached_url == url
    {
        return Some(Arc::clone(dir));
    }
    let bytes = match fs.read(&url).await {
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
    if let Ok(mut map) = loaded().lock() {
        map.insert(key, (url, Arc::clone(&dir)));
    }
    Some(dir)
}

fn object_name(url: &str) -> &str {
    url.rsplit('/').next().unwrap_or(url)
}

/// Per-segment directory choices for one query, keyed by segment URL: the A0
/// cells to probe (empty ⇒ skip the segment). Segments absent from the map
/// keep the per-segment path. `files` are the discovered `(url, size)` pairs;
/// a segment of unknown size is never usable (its stamp cannot be checked).
pub(crate) fn plan_segments<'a>(
    dir: &PaxCentroidDirectory,
    files: impl IntoIterator<Item = (&'a str, Option<u64>)>,
    query: &[f32],
    k: usize,
) -> Option<HashMap<String, Arc<[u32]>>> {
    let mut url_by_name = HashMap::new();
    let mut live = Vec::new();
    for (url, size) in files {
        let Some(size_bytes) = size else { continue };
        let name = object_name(url);
        url_by_name.insert(name.to_owned(), url.to_owned());
        live.push(LiveSegment {
            name: name.to_owned(),
            size_bytes,
        });
    }
    let plan = dir.plan_probe(query, &live, centroid_directory_nprobe, |rows| {
        let rows = usize::try_from(rows).unwrap_or(usize::MAX);
        pax_rabitq_pool_for_top_k(k, rows) as u64
    })?;
    Some(
        plan.cells_by_segment
            .into_iter()
            .filter_map(|(name, cells)| {
                url_by_name
                    .remove(&name)
                    .map(|url| (url, Arc::<[u32]>::from(cells)))
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::engines::sst::centroid_directory_publisher::publish_centroid_directory;
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};

    #[test]
    fn read_gate_defaults_off() {
        // SAFETY: nextest runs each test in its own process.
        unsafe { std::env::remove_var("PROXIMADB_PAX_READ_CENTROID_DIRECTORY") };
        assert!(!centroid_directory_read_enabled());
        unsafe { std::env::set_var("PROXIMADB_PAX_READ_CENTROID_DIRECTORY", "1") };
        assert!(centroid_directory_read_enabled());
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
