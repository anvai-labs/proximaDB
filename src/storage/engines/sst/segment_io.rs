//! TD-TIER-1 SC4-4: whole-segment residency for reservation-granted
//! collections.
//!
//! A PERFORMANCE grant asks for its collection's immutable PAX segments to be
//! served from local disk. This module is the single seam that makes that
//! true:
//!
//! * [`ResidentRangeRead::read_range_resident`] — every SST segment ranged
//!   read goes through it. When the process persistent-L2 store holds the
//!   whole object ([`proximadb_cache::object_key`]), the range is served from
//!   it (checksum-verified) and no object-store request is made; otherwise
//!   the read falls through to the filesystem unchanged. A guard test pins
//!   that the segment readers never call `read_range` directly.
//! * [`warm_object`] — one whole-object GET then a local put, for a path a
//!   grant covers (post-flush seeding and backfill).
//! * [`seed_object_from_local_file`] — the same, from a local file the
//!   writer already holds (compaction's local-spill path; no GET).
//!
//! Inert unless a grant's protection group covers the path: with no group
//! registered nothing is written, and reads pay one map lookup.

use std::path::Path;
use std::sync::Arc;

use proximadb_cache::{L2Class, object_key};
use proximadb_storage_filesystem_types::{FileSystem, FsResult};

/// Ranged reads that prefer a resident whole-segment copy.
#[async_trait::async_trait]
pub trait ResidentRangeRead {
    /// [`FileSystem::read_range`], served from the persistent-L2 whole-object
    /// copy when one is resident and covers the range.
    async fn read_range_resident(&self, path: &str, offset: u64, length: u64) -> FsResult<Vec<u8>>;
}

#[async_trait::async_trait]
impl<T: FileSystem + ?Sized> ResidentRangeRead for T {
    async fn read_range_resident(&self, path: &str, offset: u64, length: u64) -> FsResult<Vec<u8>> {
        if let Some(bytes) = resident_range(path, offset, length).await {
            return Ok(bytes);
        }
        self.read_range(path, offset, length).await
    }
}

/// The range from the resident whole-object copy, if one covers it. Any
/// failure (absent, short, corrupt — the store drops a corrupt entry) is a
/// miss: the caller falls back to the object store.
async fn resident_range(path: &str, offset: u64, length: u64) -> Option<Vec<u8>> {
    let store = proximadb_cache::process_store()?;
    let key = object_key(path);
    let resident_len = store.entry_len(&key)?;
    if offset.checked_add(length)? > resident_len {
        return None;
    }
    match store.get_range(&key, offset, length).await {
        Ok(Some(bytes)) => Some(bytes.to_vec()),
        Ok(None) => None,
        Err(error) => {
            tracing::warn!("TD-TIER-1 SC4: resident read of {path} failed: {error}");
            None
        }
    }
}

/// Make the whole object at `path` resident when a grant covers it and it is
/// not already: one object-store GET, then a local put (pinned by the grant's
/// prefix protection within its budget). Returns whether it was written.
pub async fn warm_object(fs: &dyn FileSystem, path: &str) -> anyhow::Result<bool> {
    let Some(store) = proximadb_cache::process_store() else {
        return Ok(false);
    };
    let key = object_key(path);
    if store.entry_len(&key).is_some() || !store.policy_covers(&key) {
        return Ok(false);
    }
    let bytes = fs
        .read(path)
        .await
        .map_err(|error| anyhow::anyhow!("warm read {path}: {error}"))?;
    store
        .put(key, L2Class::Invariants, Arc::from(bytes))
        .await
        .map_err(|error| anyhow::anyhow!("warm put {path}: {error}"))?;
    Ok(true)
}

/// Make the object published at `path` resident from a local file holding
/// its exact bytes (no object-store request). Returns whether it was written.
pub async fn seed_object_from_local_file(path: &str, local: &Path) -> anyhow::Result<bool> {
    let Some(store) = proximadb_cache::process_store() else {
        return Ok(false);
    };
    let key = object_key(path);
    if store.entry_len(&key).is_some() || !store.policy_covers(&key) {
        return Ok(false);
    }
    let len = std::fs::metadata(local)
        .map_err(|error| anyhow::anyhow!("stat {}: {error}", local.display()))?
        .len();
    store
        .put_file_range(key, L2Class::Invariants, local, 0, len)
        .await
        .map_err(|error| anyhow::anyhow!("seed {path}: {error}"))?;
    Ok(true)
}

/// Drop the resident copy of a retired segment (compaction retirement).
pub fn forget_object(path: &str) {
    if let Some(store) = proximadb_cache::process_store() {
        store.remove(&object_key(path));
    }
}

/// The live segment data files of one collection data directory (one LIST,
/// minus pending retirements — the same rule search discovery applies). A
/// directory that does not exist yet (no flush) has no segments.
pub(crate) async fn live_segments(
    fs: &dyn FileSystem,
    data_url: &str,
) -> anyhow::Result<Vec<crate::storage::engines::sst::search::DiscoveredSstFile>> {
    let entries = match fs.list(data_url).await {
        Ok(entries) => entries,
        Err(error) if error.to_string().contains("No such file or directory") => {
            return Ok(Vec::new());
        }
        Err(error) => anyhow::bail!("list {data_url}: {error}"),
    };
    let pending = crate::storage::engines::sst::retirement_ledger::pending_paths();
    Ok(crate::storage::engines::sst::search::live_segment_files(
        entries, &pending,
    ))
}

/// Outcome of a [`warm_collection`] backfill.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct WarmReport {
    /// Segments made resident by this pass.
    pub warmed: usize,
    /// Segments already resident, or not covered by any grant.
    pub skipped: usize,
    /// Segments whose GET or put failed (logged; retried by the next pass).
    pub failed: usize,
}

/// Backfill: make every live segment under `data_url` resident when a grant
/// covers it (one whole-object GET each, `concurrency` at a time). Runs at
/// grant registration — segments published before the grant existed would
/// otherwise never be resident. Best-effort and idempotent: a resident
/// segment is skipped; a put the group budget cannot pin stays unpinned and
/// evictable, so coverage stays observably partial.
pub async fn warm_collection(
    fs: &dyn FileSystem,
    data_url: &str,
    concurrency: usize,
) -> anyhow::Result<WarmReport> {
    use futures::StreamExt;
    let segments = live_segments(fs, data_url).await?;
    let outcomes: Vec<anyhow::Result<bool>> = futures::stream::iter(segments)
        .map(|segment| async move { warm_object(fs, &segment.url).await })
        .buffer_unordered(concurrency.max(1))
        .collect()
        .await;
    let mut report = WarmReport::default();
    for outcome in outcomes {
        match outcome {
            Ok(true) => report.warmed += 1,
            Ok(false) => report.skipped += 1,
            Err(error) => {
                tracing::warn!("TD-TIER-1 SC4: warm failed under {data_url}: {error}");
                report.failed += 1;
            }
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
    use proximadb_storage_filesystem_types::counting::{CountingFileSystem, GetCounters};

    /// One process store (first registration wins) with a generous budget,
    /// and a grant group covering `covered_dir`. Each test runs in its own
    /// process under nextest; groups are keyed by a per-test id.
    fn arm(
        store_dir: &Path,
        covered_dir: &str,
        group_id: &str,
    ) -> Arc<proximadb_cache::PersistentByteStore> {
        let candidate = Arc::new(
            proximadb_cache::PersistentByteStore::open(store_dir, 64 << 20).expect("store"),
        );
        let _ = proximadb_cache::register_process_store(candidate);
        proximadb_cache::set_process_reserved_budget(32 << 20);
        proximadb_cache::register_protection_group(proximadb_cache::ProtectionGroup {
            id: group_id.to_string(),
            prefixes: vec![format!("{}/", covered_dir.trim_end_matches('/'))],
            budget_bytes: 32 << 20,
        });
        proximadb_cache::process_store().expect("a store is registered")
    }

    async fn counting_local(root: &Path) -> (Arc<CountingFileSystem>, Arc<GetCounters>) {
        let local = LocalFileSystem::new(LocalConfig {
            root_dir: Some(root.to_path_buf()),
            ..Default::default()
        })
        .await
        .expect("local fs");
        let counters = Arc::new(GetCounters::default());
        (
            Arc::new(CountingFileSystem::new(Arc::new(local), counters.clone())),
            counters,
        )
    }

    fn segment(dir: &Path, name: &str, len: usize) -> (String, Vec<u8>) {
        let bytes: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
        let path = dir.join(name);
        std::fs::write(&path, &bytes).expect("write segment");
        (path.to_string_lossy().to_string(), bytes)
    }

    /// The SC4-4 core: a covered segment made resident is then read with ZERO
    /// object-store requests, byte-exact; an uncovered segment is untouched
    /// and still read from the store.
    #[tokio::test]
    async fn covered_segment_reads_with_zero_gets_uncovered_unchanged() {
        let root = tempfile::tempdir().expect("root");
        let store_dir = tempfile::tempdir().expect("store");
        let covered = root.path().join("c1/data");
        let other = root.path().join("c2/data");
        std::fs::create_dir_all(&covered).expect("dirs");
        std::fs::create_dir_all(&other).expect("dirs");
        arm(store_dir.path(), &covered.to_string_lossy(), "zero-gets");
        let (fs, counters) = counting_local(root.path()).await;

        let (hot, hot_bytes) = segment(&covered, "L0_a.pax", 200_000);
        let (cold, cold_bytes) = segment(&other, "L0_b.pax", 1_000);
        assert!(warm_object(fs.as_ref(), &hot).await.expect("warm"));
        assert!(
            !warm_object(fs.as_ref(), &cold).await.expect("warm"),
            "not covered"
        );

        counters.reset();
        let got = fs
            .read_range_resident(&hot, 70_000, 65_536)
            .await
            .expect("read");
        assert_eq!(got, hot_bytes[70_000..135_536]);
        assert_eq!(counters.total_gets(), 0, "a resident segment costs no GET");

        let got = fs.read_range_resident(&cold, 10, 100).await.expect("read");
        assert_eq!(got, cold_bytes[10..110]);
        assert_eq!(
            counters.total_gets(),
            1,
            "an uncovered segment still reads remotely"
        );
    }

    /// Compaction's local-spill path seeds from the file it already holds
    /// (no GET); retirement forgets the resident copy.
    #[tokio::test]
    async fn local_seed_then_forget() {
        let root = tempfile::tempdir().expect("root");
        let store_dir = tempfile::tempdir().expect("store");
        let covered = root.path().join("c1/data");
        std::fs::create_dir_all(&covered).expect("dirs");
        let store = arm(store_dir.path(), &covered.to_string_lossy(), "local-seed");
        let (fs, counters) = counting_local(root.path()).await;
        let (path, bytes) = segment(&covered, "L1_c.pax", 4_096);

        assert!(
            seed_object_from_local_file(&path, Path::new(&path))
                .await
                .expect("seed")
        );
        assert_eq!(
            counters.total_gets(),
            0,
            "seeding from a local file issues no GET"
        );
        assert_eq!(
            store.entry_len(&object_key(&path)),
            Some(bytes.len() as u64)
        );
        assert!(store.is_protected(&object_key(&path)), "the grant pins it");
        assert_eq!(
            fs.read_range_resident(&path, 0, 16).await.expect("read"),
            bytes[..16]
        );
        assert_eq!(counters.total_gets(), 0);

        forget_object(&path);
        assert_eq!(store.entry_len(&object_key(&path)), None);
        let _ = fs.read_range_resident(&path, 0, 16).await.expect("read");
        assert_eq!(
            counters.total_gets(),
            1,
            "after retirement the read is remote again"
        );
    }

    /// Backfill warms every live segment once (non-segment files ignored) and
    /// is idempotent.
    #[tokio::test]
    async fn backfill_warms_live_segments_idempotently() {
        let root = tempfile::tempdir().expect("root");
        let store_dir = tempfile::tempdir().expect("store");
        let covered = root.path().join("c1/data");
        std::fs::create_dir_all(&covered).expect("dirs");
        let store = arm(store_dir.path(), &covered.to_string_lossy(), "backfill");
        let (fs, _counters) = counting_local(root.path()).await;
        let (a, _) = segment(&covered, "L0_a.pax", 2_048);
        let (b, _) = segment(&covered, "L0_b.pax", 2_048);
        std::fs::write(covered.join("notes.txt"), b"not a segment").expect("write");
        let data_url = covered.to_string_lossy().to_string();

        let first = warm_collection(fs.as_ref(), &data_url, 2)
            .await
            .expect("warm");
        assert_eq!((first.warmed, first.failed), (2, 0), "{first:?}");
        let listed: Vec<String> = live_segments(fs.as_ref(), &data_url)
            .await
            .expect("list")
            .into_iter()
            .map(|segment| segment.url)
            .collect();
        for url in &listed {
            assert!(
                store.entry_len(&object_key(url)).is_some(),
                "{url} resident"
            );
        }
        assert!(
            listed.iter().any(|u| u.ends_with("L0_a.pax"))
                && listed.iter().any(|u| u.ends_with("L0_b.pax"))
        );
        let _ = (a, b);
        let second = warm_collection(fs.as_ref(), &data_url, 2)
            .await
            .expect("warm");
        assert_eq!((second.warmed, second.skipped), (0, 2), "{second:?}");
    }

    /// Guard: the SST segment readers never call `read_range` directly — a
    /// direct call would silently bypass residency and cost a GET on a READY
    /// collection.
    #[test]
    fn segment_readers_route_ranged_reads_through_residency() {
        for (name, source) in [
            ("segment_format.rs", include_str!("segment_format.rs")),
            ("oid_resolve.rs", include_str!("oid_resolve.rs")),
            ("search/mod.rs", include_str!("search/mod.rs")),
        ] {
            let direct = source.matches(".read_range(").count();
            assert_eq!(
                direct, 0,
                "{name}: {direct} direct `.read_range(` call(s) — use `read_range_resident`"
            );
        }
    }
}
