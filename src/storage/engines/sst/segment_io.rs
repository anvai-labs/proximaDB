//! TD-TIER-1 SC4-4: whole-segment residency for reservation-granted
//! collections.
//!
//! A PERFORMANCE grant asks for its collection's immutable PAX segments to be
//! served from local disk. This module is the single seam that makes that
//! true:
//!
//! * [`ResidentRangeRead`] — every SST segment read on the query path goes
//!   through it: single ranges ([`ResidentRangeRead::read_range_resident`]),
//!   batched prefetch waves ([`ResidentRangeRead::read_ranges_prefetch_resident`])
//!   and whole-object reads ([`ResidentRangeRead::read_resident`]). When the
//!   process persistent-L2 store holds the whole object
//!   ([`proximadb_cache::object_key`]), bytes come from it (checksum-verified)
//!   and no object-store request is made; otherwise the call falls through to
//!   the filesystem unchanged. A guard test pins that the segment readers
//!   never call the filesystem's read methods directly.
//! * [`warm_object`] — fetch a covered object (bounded chunks staged through a
//!   temp file, never the whole object in RAM) and put it, only when the
//!   grant's budget can pin it (no GET is paid for bytes that would be
//!   evicted again).
//! * [`seed_object_from_local_file`] — the same from a local file the writer
//!   already holds (compaction's local-spill path; no GET).
//! * [`forget_object`] — retirement; also records the path so a backfill that
//!   raced the retirement removes its late copy.
//!
//! Inert unless a grant's protection group exists: with none, reads take one
//! lock-free flag check and nothing is written.

use std::path::Path;
use std::sync::Arc;

use proximadb_cache::{L2Class, PersistentByteStore, object_key};
use proximadb_storage_filesystem_types::{FileSystem, FilesystemError, FsResult};

/// Whole-object fetches are staged in chunks of this size (bounded RAM).
const WARM_CHUNK_BYTES: u64 = 64 << 20;
/// Recently retired object keys remembered for the backfill race.
const RETIRED_MEMORY: usize = 4096;
/// Whole-object fetches in flight across EVERY warm path in the process
/// (post-flush, post-compaction, backfills): bounds RAM (one chunk each),
/// staging disk, and how far concurrent warms can overshoot the advisory
/// budget check.
const WARMS_IN_FLIGHT: usize = 4;

fn warm_permits() -> &'static tokio::sync::Semaphore {
    static PERMITS: std::sync::OnceLock<tokio::sync::Semaphore> = std::sync::OnceLock::new();
    PERMITS.get_or_init(|| tokio::sync::Semaphore::new(WARMS_IN_FLIGHT))
}

/// Segment reads that prefer a resident whole-segment copy.
#[async_trait::async_trait]
pub trait ResidentRangeRead {
    /// [`FileSystem::read_range`], served from the resident copy when one
    /// covers the range.
    async fn read_range_resident(&self, path: &str, offset: u64, length: u64) -> FsResult<Vec<u8>>;

    /// [`FileSystem::read_ranges_prefetch`], served from the resident copy
    /// when it covers every range (one buffer per range, input order).
    async fn read_ranges_prefetch_resident(
        &self,
        path: &str,
        ranges: Vec<std::ops::Range<u64>>,
    ) -> FsResult<Vec<Vec<u8>>>;

    /// [`FileSystem::read`], served from the resident copy when present.
    async fn read_resident(&self, path: &str) -> FsResult<Vec<u8>>;

    /// The object's size: from the resident copy when present (no HEAD),
    /// otherwise one recorded metadata op.
    async fn object_size_resident(&self, path: &str) -> FsResult<u64>;
}

#[async_trait::async_trait]
impl<T: FileSystem + ?Sized> ResidentRangeRead for T {
    async fn read_range_resident(&self, path: &str, offset: u64, length: u64) -> FsResult<Vec<u8>> {
        if let Some(resident) = Resident::lookup(path)
            && let Some(bytes) = resident.range(offset, length).await
        {
            return Ok(bytes);
        }
        self.read_range(path, offset, length).await
    }

    async fn read_ranges_prefetch_resident(
        &self,
        path: &str,
        ranges: Vec<std::ops::Range<u64>>,
    ) -> FsResult<Vec<Vec<u8>>> {
        if let Some(resident) = Resident::lookup(path)
            && ranges
                .iter()
                .all(|range| range.end <= resident.len && range.start <= range.end)
        {
            let mut out = Vec::with_capacity(ranges.len());
            for range in &ranges {
                match resident.range(range.start, range.end - range.start).await {
                    Some(bytes) => out.push(bytes),
                    None => return self.read_ranges_prefetch(path, ranges).await,
                }
            }
            return Ok(out);
        }
        self.read_ranges_prefetch(path, ranges).await
    }

    async fn read_resident(&self, path: &str) -> FsResult<Vec<u8>> {
        if let Some(resident) = Resident::lookup(path)
            && let Some(bytes) = resident.range(0, resident.len).await
        {
            return Ok(bytes);
        }
        self.read(path).await
    }

    async fn object_size_resident(&self, path: &str) -> FsResult<u64> {
        if let Some(resident) = Resident::lookup(path) {
            return Ok(resident.len);
        }
        crate::observability::io_trace::record_metadata_op();
        crate::storage::engines::sst::metrics::record_metadata_op();
        Ok(self.metadata(path).await?.size)
    }
}

/// A resident whole-object copy of one segment.
struct Resident {
    store: Arc<PersistentByteStore>,
    key: String,
    len: u64,
}

impl Resident {
    /// The resident copy of `path`, if any. Lock-free and allocation-free
    /// when no grant exists (the common, inert case).
    fn lookup(path: &str) -> Option<Self> {
        let store = proximadb_cache::process_store()?;
        if !store.policy_active() {
            return None;
        }
        let key = object_key(path);
        let len = store.entry_len(&key)?;
        Some(Self { store, key, len })
    }

    /// `[offset, offset+length)` from the copy. Any failure (short, corrupt —
    /// the store drops a corrupt entry) is a miss: the caller falls back to
    /// the object store.
    async fn range(&self, offset: u64, length: u64) -> Option<Vec<u8>> {
        if offset.checked_add(length)? > self.len {
            return None;
        }
        match self.store.get_range(&self.key, offset, length).await {
            Ok(Some(bytes)) => Some(bytes.to_vec()),
            Ok(None) => None,
            Err(error) => {
                warn_rate_limited(&format!("resident read of {}: {error}", self.key));
                None
            }
        }
    }
}

/// The length of `path`'s resident whole-object copy, if one exists — lets a
/// reader that would stat the object skip the HEAD.
pub fn resident_len(path: &str) -> Option<u64> {
    Resident::lookup(path).map(|resident| resident.len)
}

/// Warn for the first few failures, then demote to debug (a persistently
/// failing disk must not flood the log on every read).
fn warn_rate_limited(message: &str) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static WARNED: AtomicU64 = AtomicU64::new(0);
    if WARNED.fetch_add(1, Ordering::Relaxed) < 16 {
        tracing::warn!("TD-TIER-1 SC4: {message}");
    } else {
        tracing::debug!("TD-TIER-1 SC4: {message}");
    }
}

/// Object keys retired recently (bounded FIFO): a backfill that listed a
/// segment, then lost the race to its retirement, removes its late copy.
fn retired() -> std::sync::MutexGuard<'static, std::collections::VecDeque<String>> {
    static RETIRED: std::sync::OnceLock<std::sync::Mutex<std::collections::VecDeque<String>>> =
        std::sync::OnceLock::new();
    RETIRED
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// After a put: if the object was retired meanwhile, drop the late copy.
fn drop_if_retired(store: &PersistentByteStore, key: &str) -> bool {
    if retired().iter().any(|retired| retired == key) {
        store.remove(key);
        return true;
    }
    false
}

/// The store, when `path` is a grant-covered object not yet resident whose
/// `len` bytes the grant's budget can pin.
fn warm_target(path: &str, len: u64) -> Option<(Arc<PersistentByteStore>, String)> {
    let store = proximadb_cache::process_store()?;
    let key = object_key(path);
    if store.entry_len(&key).is_some() || !store.policy_headroom(&key, len) {
        return None;
    }
    Some((store, key))
}

/// Make the whole object at `path` resident when a grant covers it, it is not
/// already, and the grant's budget can pin it. `len` is the object size when
/// the caller knows it (a listing or the writer); otherwise one stat. The
/// bytes are fetched in bounded chunks into a temp file, then put. Returns
/// whether it was written.
pub async fn warm_object(
    fs: &dyn FileSystem,
    path: &str,
    len: Option<u64>,
) -> anyhow::Result<bool> {
    let Some(store) = proximadb_cache::process_store() else {
        return Ok(false);
    };
    if !store.policy_covers(&object_key(path)) {
        return Ok(false);
    }
    let len = match len {
        Some(len) => len,
        None => {
            fs.metadata(path)
                .await
                .map_err(|error| anyhow::anyhow!("warm stat {path}: {error}"))?
                .size
        }
    };
    // Process-wide limit, then re-judge the budget: a warm that waited may
    // find the room taken by the ones ahead of it.
    let _permit = warm_permits()
        .acquire()
        .await
        .map_err(|error| anyhow::anyhow!("warm permit for {path}: {error}"))?;
    let Some((store, key)) = warm_target(path, len) else {
        return Ok(false);
    };
    // Stage on the store's own filesystem (never a possibly RAM-backed
    // system temp dir), with async writes; removed on every exit (Drop).
    let staging_dir = store
        .staging_dir()
        .map_err(|error| anyhow::anyhow!("warm staging dir for {path}: {error}"))?;
    let staged = tempfile::NamedTempFile::new_in(staging_dir)
        .map_err(|error| anyhow::anyhow!("warm temp file for {path}: {error}"))?;
    {
        use tokio::io::AsyncWriteExt;
        let file = staged
            .reopen()
            .map_err(|error| anyhow::anyhow!("warm stage {path}: {error}"))?;
        let mut out = tokio::fs::File::from_std(file);
        let mut offset = 0u64;
        while offset < len {
            let chunk = WARM_CHUNK_BYTES.min(len - offset);
            let bytes = fs
                .read_range(path, offset, chunk)
                .await
                .map_err(|error| anyhow::anyhow!("warm read {path}@{offset}: {error}"))?;
            if bytes.len() as u64 != chunk {
                anyhow::bail!("warm read {path}@{offset}: short read");
            }
            out.write_all(&bytes)
                .await
                .map_err(|error| anyhow::anyhow!("warm stage {path}: {error}"))?;
            offset += chunk;
        }
        out.flush()
            .await
            .map_err(|error| anyhow::anyhow!("warm stage {path}: {error}"))?;
    }
    store
        .put_file_range(key.clone(), L2Class::Invariants, staged.path(), 0, len)
        .await
        .map_err(|error| anyhow::anyhow!("warm put {path}: {error}"))?;
    Ok(!drop_if_retired(&store, &key))
}

/// Make the object published at `path` resident from a local file holding
/// its exact bytes (no object-store request). Returns whether it was written.
pub async fn seed_object_from_local_file(path: &str, local: &Path) -> anyhow::Result<bool> {
    let len = std::fs::metadata(local)
        .map_err(|error| anyhow::anyhow!("stat {}: {error}", local.display()))?
        .len();
    let Some((store, key)) = warm_target(path, len) else {
        return Ok(false);
    };
    store
        .put_file_range(key.clone(), L2Class::Invariants, local, 0, len)
        .await
        .map_err(|error| anyhow::anyhow!("seed {path}: {error}"))?;
    Ok(!drop_if_retired(&store, &key))
}

/// Retirement: drop the resident copy and remember the key, so a backfill
/// that raced this retirement drops its late copy too.
pub fn forget_object(path: &str) {
    let key = object_key(path);
    {
        let mut recent = retired();
        // A key retired twice (a directory generation dropped as g-1, then
        // collected) is remembered once, so it does not double its share of
        // the bounded set.
        if !recent.iter().any(|retired| retired == &key) {
            if recent.len() >= RETIRED_MEMORY {
                recent.pop_front();
            }
            recent.push_back(key.clone());
        }
    }
    if let Some(store) = proximadb_cache::process_store() {
        store.remove(&key);
    }
}

/// Detached [`warm_object`] for a just-published segment, so a whole-object
/// fetch never adds to flush/compaction latency (verification round).
pub fn spawn_warm_object(fs: Arc<dyn FileSystem>, path: String) {
    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
        return;
    };
    runtime.spawn(async move {
        if let Err(error) = warm_object(fs.as_ref(), &path, None).await {
            warn_rate_limited(&format!("post-publication residency for {path}: {error}"));
        }
    });
}

/// Detached backfill of every residency prefix (one [`warm_collection`] per
/// segment root, a few fetches at a time). Shared by the class-change path
/// and the store-registration path at startup.
pub fn spawn_warm_prefixes(
    factory: Arc<crate::storage::persistence::filesystem::FilesystemFactory>,
    prefixes: Vec<String>,
) {
    const CONCURRENCY: usize = 4;
    if prefixes.is_empty() {
        return;
    }
    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
        tracing::warn!("TD-TIER-1 SC4: residency warm skipped outside a runtime");
        return;
    };
    runtime.spawn(async move {
        for prefix in prefixes {
            let data_url = prefix.trim_end_matches('/').to_string();
            let Ok(fs) = factory.get_filesystem(&data_url) else {
                continue;
            };
            match warm_collection(fs.as_ref(), &data_url, CONCURRENCY).await {
                Ok(report) => tracing::debug!("TD-TIER-1 SC4: warmed {data_url}: {report:?}"),
                Err(error) => warn_rate_limited(&format!("warm of {data_url}: {error}")),
            }
        }
    });
}

/// TD-TIER-1 SC4-5: what a residency proof observed for one grant.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ResidencyProof {
    /// Whether a process persistent store is registered at all.
    pub store_present: bool,
    /// Objects in the committed set: live segments plus the newest centroid
    /// directory object ([`residency_objects`]).
    pub segments: usize,
    /// Of those, live SEGMENTS (READY needs at least one: a set holding only
    /// a directory proves nothing about the data — review N-a).
    pub segment_files: usize,
    /// Of those, resident in full at their listed size AND protected.
    pub covered_segments: usize,
    /// Segment roots examined (0 = the grant names none: nothing can be
    /// proven).
    pub roots: usize,
    /// Sum of the residency objects' listed sizes (segments + the newest
    /// directory; an object with an unknown size adds 0 and is never
    /// covered).
    pub inventory_bytes: u64,
    /// The same set's STORED size — what pins are charged against the grant
    /// and the store (header + key + CRC table + value per object).
    pub inventory_stored_bytes: u64,
    /// The persistent store's capacity (0 without a store).
    pub store_capacity: u64,
    /// Generation `g` of the committed set: a stable hash (FNV-1a 64, masked
    /// to 53 bits so it is exact in every client — int64 schema, JS numbers)
    /// of the sorted canonical (url, size) pairs, so any publication or
    /// retirement changes it.
    pub generation: u64,
}

/// Prove (or refute) that every live segment under `prefixes` is resident in
/// full and protected. One LIST per prefix plus map lookups — a status read,
/// never the query path. A segment whose listing reports no size counts as
/// not covered (the proof never guesses).
pub async fn prove_residency(
    factory: &crate::storage::persistence::filesystem::FilesystemFactory,
    prefixes: &[String],
) -> anyhow::Result<ResidencyProof> {
    let store = proximadb_cache::process_store();
    let mut inventory: Vec<(String, Option<u64>)> = Vec::new();
    for prefix in prefixes {
        let data_url = prefix.trim_end_matches('/');
        let fs = factory.get_filesystem(data_url).map_err(|error| {
            anyhow::anyhow!("residency proof: filesystem for {data_url}: {error}")
        })?;
        inventory.extend(residency_objects(fs.as_ref(), data_url).await?);
    }
    let mut keyed: Vec<(String, Option<u64>)> = inventory
        .iter()
        .map(|(url, size)| (proximadb_cache::canonical_location(url), *size))
        .collect();
    keyed.sort();
    keyed.dedup();
    let mut generation: u64 = 0xcbf2_9ce4_8422_2325;
    for (url, size) in &keyed {
        for byte in url
            .as_bytes()
            .iter()
            .chain(&size.unwrap_or(0).to_le_bytes())
        {
            generation ^= u64::from(*byte);
            generation = generation.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    let mut proof = ResidencyProof {
        store_present: store.is_some(),
        roots: prefixes.len(),
        segments: keyed.len(),
        segment_files: keyed
            .iter()
            .filter(|(url, _)| {
                url.rsplit('/').next().is_none_or(|name| {
                    proximadb_storage_common::centroid_directory::parse_object_name(name).is_none()
                })
            })
            .count(),
        generation: generation & ((1u64 << 53) - 1),
        store_capacity: store.as_ref().map_or(0, |store| store.capacity()),
        ..ResidencyProof::default()
    };
    for (url, size) in &keyed {
        proof.inventory_bytes = proof.inventory_bytes.saturating_add(size.unwrap_or(0));
        proof.inventory_stored_bytes =
            proof
                .inventory_stored_bytes
                .saturating_add(PersistentByteStore::stored_size(
                    &object_key(url),
                    size.unwrap_or(0),
                ));
        let (Some(store), Some(size)) = (&store, size) else {
            continue;
        };
        let key = object_key(url);
        if store.entry_len(&key) == Some(*size) && store.is_protected(&key) {
            proof.covered_segments += 1;
        }
    }
    Ok(proof)
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
        Err(FilesystemError::NotFound(_)) => return Ok(Vec::new()),
        Err(FilesystemError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(Vec::new());
        }
        Err(error) => anyhow::bail!("list {data_url}: {error}"),
    };
    let pending = crate::storage::engines::sst::retirement_ledger::pending_paths();
    Ok(crate::storage::engines::sst::search::live_segment_files(
        entries, &pending,
    ))
}

/// Everything a query on the vector profile reads under one data directory:
/// the live segments plus the collection's NEWEST centroid directory object
/// (read on every cold query when the directory is enabled).
/// Backfill makes these resident and the READY proof requires them all —
/// search discovery itself stays segments-only ([`live_segments`]).
pub(crate) async fn residency_objects(
    fs: &dyn FileSystem,
    data_url: &str,
) -> anyhow::Result<Vec<(String, Option<u64>)>> {
    let entries = match fs.list(data_url).await {
        Ok(entries) => entries,
        Err(FilesystemError::NotFound(_)) => return Ok(Vec::new()),
        Err(FilesystemError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(Vec::new());
        }
        Err(error) => anyhow::bail!("list {data_url}: {error}"),
    };
    // Only the NEWEST directory generation is read by queries; older ones
    // linger until GC and must not be required (or warmed) for READY. The
    // publisher's own selection rule, so the two cannot drift (review N-c).
    let directories: Vec<(String, Option<u64>)> =
        crate::storage::engines::sst::centroid_directory_publisher::directory_objects_newest_first(
            &entries,
        )
        .first()
        .map(|(_, entry)| {
            (
                entry.url.clone(),
                (entry.metadata.size > 0).then_some(entry.metadata.size),
            )
        })
        .into_iter()
        .collect();
    let pending = crate::storage::engines::sst::retirement_ledger::pending_paths();
    let mut objects: Vec<(String, Option<u64>)> =
        crate::storage::engines::sst::search::live_segment_files(entries, &pending)
            .into_iter()
            .map(|segment| (segment.url, segment.size_bytes))
            .collect();
    objects.extend(directories);
    Ok(objects)
}

/// Outcome of a [`warm_collection`] backfill.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct WarmReport {
    /// Segments made resident by this pass.
    pub warmed: usize,
    /// Segments already resident, not covered, or beyond the budget.
    pub skipped: usize,
    /// Segments whose fetch or put failed (logged; retried by the next pass).
    pub failed: usize,
    /// Resident copies of objects the committed set no longer names, dropped
    /// by [`sweep_orphan_objects`].
    pub orphans_dropped: usize,
}

/// Crash reconciliation: drop every resident whole-object copy directly
/// under `data_url` whose object the committed set (`objects`, from the same
/// listing) no longer names. A crash between a segment's delete and its
/// [`forget_object`] — or a retirement in another process — otherwise leaves
/// the copy pinned, charging the grant, until the grant is released.
///
/// Plain removal, NOT [`forget_object`]: an object published after the
/// listing and warmed before this sweep loses its copy too, and must stay
/// re-warmable (the retired set would block that). Losing a copy only costs a
/// re-warm; the proof reports WARMING meanwhile.
async fn sweep_orphan_objects(data_url: &str, objects: &[(String, Option<u64>)]) -> usize {
    let Some(store) = proximadb_cache::process_store() else {
        return 0;
    };
    // Matched like the protection groups (`contains` on canonical forms): a
    // listing may spell the root differently (`file:///abs/...` for a grant
    // prefix of `/abs/...`), so an exact-prefix match would sweep nothing.
    let object_prefix = object_key("");
    let root = proximadb_cache::canonical_location(&format!("{}/", data_url.trim_end_matches('/')));
    let live: std::collections::HashSet<String> =
        objects.iter().map(|(url, _)| object_key(url)).collect();
    store
        .remove_where(|key| {
            let Some(location) = key.strip_prefix(object_prefix.as_str()) else {
                return false;
            };
            location
                .rfind(root.as_str())
                .map(|at| &location[at + root.len()..])
                // Direct children only: the listing is not recursive, so a
                // nested object is not judged by it.
                .is_some_and(|name| !name.is_empty() && !name.contains('/'))
                && !live.contains(key)
        })
        .await
}

/// Backfill: make every live segment under `data_url` resident when a grant
/// covers it (`concurrency` fetches at a time). Needed whenever a grant starts
/// covering a collection that already has segments. Best-effort and
/// idempotent; does nothing (not even the LIST) without a registered store
/// and an active policy.
pub async fn warm_collection(
    fs: &dyn FileSystem,
    data_url: &str,
    concurrency: usize,
) -> anyhow::Result<WarmReport> {
    use futures::StreamExt;
    if !proximadb_cache::process_store().is_some_and(|store| store.policy_active()) {
        return Ok(WarmReport::default());
    }
    let objects = residency_objects(fs, data_url).await?;
    let orphans = sweep_orphan_objects(data_url, &objects).await;
    let outcomes: Vec<anyhow::Result<bool>> = futures::stream::iter(objects)
        .map(|(url, size)| async move { warm_object(fs, &url, size).await })
        .buffer_unordered(concurrency.max(1))
        .collect()
        .await;
    let mut report = WarmReport {
        orphans_dropped: orphans,
        ..WarmReport::default()
    };
    for outcome in outcomes {
        match outcome {
            Ok(true) => report.warmed += 1,
            Ok(false) => report.skipped += 1,
            Err(error) => {
                warn_rate_limited(&format!("warm failed under {data_url}: {error}"));
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

    /// The process store (first registration wins) and a grant group covering
    /// `covered_dir` with `budget` bytes. Holds the crate-wide residency gate,
    /// so shared-process runners cannot interleave tests that touch the
    /// process-global store, budget or groups.
    struct Armed {
        _gate: std::sync::MutexGuard<'static, ()>,
        store: Arc<PersistentByteStore>,
        group_id: String,
    }

    impl Drop for Armed {
        fn drop(&mut self) {
            proximadb_cache::unregister_protection_group(&self.group_id);
            proximadb_cache::set_process_reserved_budget(0);
        }
    }

    fn arm(covered_dir: &Path, group_id: &str, budget: u64) -> Armed {
        let gate = crate::services::collection::reservation::residency_test_gate();
        let store = crate::services::collection::reservation::residency_test_store();
        proximadb_cache::set_process_reserved_budget(32 << 20);
        proximadb_cache::register_protection_group(proximadb_cache::ProtectionGroup {
            id: group_id.to_string(),
            prefixes: vec![format!(
                "{}/",
                covered_dir.to_string_lossy().trim_end_matches('/')
            )],
            budget_bytes: budget,
        });
        Armed {
            _gate: gate,
            store,
            group_id: group_id.to_string(),
        }
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

    fn dirs(root: &Path) -> (std::path::PathBuf, std::path::PathBuf) {
        let covered = root.join("c1/data");
        let other = root.join("c2/data");
        std::fs::create_dir_all(&covered).expect("dirs");
        std::fs::create_dir_all(&other).expect("dirs");
        (covered, other)
    }

    /// The SC4-4 core: a covered segment, once resident, is read — single
    /// range, batched prefetch wave, and whole object — byte-exact with ZERO
    /// object-store requests (review B1: the batched payload path is where a
    /// search spends its GETs). An uncovered segment is untouched.
    #[tokio::test]
    async fn covered_segment_reads_with_zero_gets_uncovered_unchanged() {
        let root = tempfile::tempdir().expect("root");
        let (covered, other) = dirs(root.path());
        let _armed = arm(&covered, "zero-gets", 32 << 20);
        let (fs, counters) = counting_local(root.path()).await;
        let (hot, hot_bytes) = segment(&covered, "L0_a.pax", 200_000);
        let (cold, cold_bytes) = segment(&other, "L0_b.pax", 1_000);
        assert!(warm_object(fs.as_ref(), &hot, None).await.expect("warm"));
        assert!(
            !warm_object(fs.as_ref(), &cold, None).await.expect("warm"),
            "not covered"
        );

        counters.reset();
        let got = fs
            .read_range_resident(&hot, 70_000, 65_536)
            .await
            .expect("read");
        assert_eq!(got, hot_bytes[70_000..135_536]);
        let waves = fs
            .read_ranges_prefetch_resident(&hot, vec![0..10, 150_000..150_100, 199_000..200_000])
            .await
            .expect("prefetch");
        assert_eq!(waves[0], hot_bytes[0..10]);
        assert_eq!(waves[1], hot_bytes[150_000..150_100]);
        assert_eq!(waves[2], hot_bytes[199_000..200_000]);
        assert_eq!(fs.read_resident(&hot).await.expect("whole"), hot_bytes);
        assert_eq!(resident_len(&hot), Some(200_000), "a HEAD can be skipped");
        assert_eq!(fs.object_size_resident(&hot).await.expect("size"), 200_000);
        assert_eq!(counters.total_gets(), 0, "a resident segment costs no GET");

        assert_eq!(
            fs.read_range_resident(&cold, 10, 100).await.expect("read"),
            cold_bytes[10..110]
        );
        assert_eq!(
            counters.total_gets(),
            1,
            "an uncovered segment still reads remotely"
        );
    }

    /// Compaction's local-spill path seeds from its file (no GET); retirement
    /// forgets the copy even when spelled differently (review B2: canonical
    /// keys), and reads are remote again.
    #[tokio::test]
    async fn local_seed_then_forget_under_another_spelling() {
        let root = tempfile::tempdir().expect("root");
        let (covered, _) = dirs(root.path());
        let armed = arm(&covered, "local-seed", 32 << 20);
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
        assert!(
            armed.store.is_protected(&object_key(&path)),
            "the grant pins it"
        );
        assert_eq!(
            fs.read_range_resident(&path, 0, 16).await.expect("read"),
            bytes[..16]
        );
        assert_eq!(counters.total_gets(), 0);

        let respelled = path.replacen("/c1/", "//c1/", 1);
        assert_ne!(respelled, path);
        forget_object(&respelled);
        assert_eq!(
            resident_len(&path),
            None,
            "one canonical key for both spellings"
        );
        let _ = fs.read_range_resident(&path, 0, 16).await.expect("read");
        assert_eq!(
            counters.total_gets(),
            1,
            "after retirement the read is remote again"
        );
    }

    /// Review S3: a warm that loses the race to the segment's retirement (the
    /// forget ran before its put) removes its late copy — no orphan pin.
    #[tokio::test]
    async fn warm_that_races_retirement_leaves_no_copy() {
        let root = tempfile::tempdir().expect("root");
        let (covered, _) = dirs(root.path());
        let armed = arm(&covered, "race", 32 << 20);
        let (fs, _) = counting_local(root.path()).await;
        let (path, _) = segment(&covered, "L0_r.pax", 2_048);
        forget_object(&path); // retirement wins the race
        assert!(!warm_object(fs.as_ref(), &path, None).await.expect("warm"));
        assert_eq!(armed.store.entry_len(&object_key(&path)), None);
        assert_eq!(armed.store.covered_bytes("race"), 0, "nothing stays pinned");
    }

    /// Review S5: an object the grant's budget cannot pin is not fetched at
    /// all — no GET paid for bytes that would only churn the cache.
    #[tokio::test]
    async fn object_beyond_the_budget_costs_no_get() {
        let root = tempfile::tempdir().expect("root");
        let (covered, _) = dirs(root.path());
        let _armed = arm(&covered, "tight", 1_024);
        let (fs, counters) = counting_local(root.path()).await;
        let (path, _) = segment(&covered, "L0_big.pax", 8_192);
        assert!(
            !warm_object(fs.as_ref(), &path, Some(8_192))
                .await
                .expect("warm")
        );
        assert_eq!(counters.total_gets(), 0, "skipped before the fetch");
        assert_eq!(resident_len(&path), None);
    }

    /// Backfill warms every live segment once (non-segment files ignored) and
    /// is idempotent.
    #[tokio::test]
    async fn backfill_warms_live_segments_idempotently() {
        let root = tempfile::tempdir().expect("root");
        let (covered, _) = dirs(root.path());
        let armed = arm(&covered, "backfill", 32 << 20);
        let (fs, _) = counting_local(root.path()).await;
        segment(&covered, "L0_a.pax", 2_048);
        segment(&covered, "L0_b.pax", 2_048);
        std::fs::write(covered.join("notes.txt"), b"not a segment").expect("write");
        let data_url = covered.to_string_lossy().to_string();

        let first = warm_collection(fs.as_ref(), &data_url, 2)
            .await
            .expect("warm");
        assert_eq!((first.warmed, first.failed), (2, 0), "{first:?}");
        for segment in live_segments(fs.as_ref(), &data_url).await.expect("list") {
            assert!(armed.store.entry_len(&object_key(&segment.url)).is_some());
        }
        let second = warm_collection(fs.as_ref(), &data_url, 2)
            .await
            .expect("warm");
        assert_eq!((second.warmed, second.skipped), (0, 2), "{second:?}");
    }

    /// Crash reconciliation: a resident copy whose segment vanished without a
    /// forget (a crash between delete and forget) is dropped by the next
    /// backfill; live copies, nested keys and other roots are untouched.
    #[tokio::test]
    async fn backfill_drops_copies_of_vanished_segments() {
        let root = tempfile::tempdir().expect("root");
        let (covered, other) = dirs(root.path());
        let armed = arm(&covered, "orphans", 32 << 20);
        let (fs, _) = counting_local(root.path()).await;
        let (_, _) = segment(&covered, "L0_live.pax", 2_048);
        let (gone_path, _) = segment(&covered, "L0_gone.pax", 2_048);
        let data_url = covered.to_string_lossy().to_string();
        warm_collection(fs.as_ref(), &data_url, 2)
            .await
            .expect("warm");
        // Copies are keyed by the LISTED url (`file:///...`), which differs
        // from the plain root the grant names — the sweep must match anyway.
        let listed: Vec<String> = live_segments(fs.as_ref(), &data_url)
            .await
            .expect("list")
            .into_iter()
            .map(|segment| segment.url)
            .collect();
        let pick = |name: &str| {
            listed
                .iter()
                .find(|url| url.ends_with(name))
                .cloned()
                .expect("listed")
        };
        let (live, gone) = (pick("L0_live.pax"), pick("L0_gone.pax"));
        assert!(resident_len(&gone).is_some());
        // The crash: the object is deleted, its forget never ran.
        std::fs::remove_file(&gone_path).expect("delete");
        let nested = object_key(&format!("{data_url}/sub/L0_nested.pax"));
        let sibling = object_key(&format!("{data_url}2/L0_sibling.pax"));
        let foreign = object_key(&format!("{}/L0_x.pax", other.to_string_lossy()));
        for key in [&nested, &sibling, &foreign] {
            armed
                .store
                .put(key.clone(), L2Class::Invariants, Arc::from(&b"bytes"[..]))
                .await
                .expect("put");
        }
        let covered_before = armed.store.covered_bytes("orphans");

        let report = warm_collection(fs.as_ref(), &data_url, 2)
            .await
            .expect("warm");
        assert_eq!(report.orphans_dropped, 1, "{report:?}");
        assert_eq!(resident_len(&gone), None, "the orphan is gone");
        assert!(resident_len(&live).is_some(), "the live copy stays");
        for key in [&nested, &sibling, &foreign] {
            assert!(armed.store.entry_len(key).is_some(), "{key} untouched");
        }
        assert!(
            armed.store.covered_bytes("orphans") < covered_before,
            "the grant stops paying for the orphan"
        );
    }

    /// A key retired twice is remembered once (review NIT on SC4-5).
    #[test]
    fn retiring_a_key_twice_records_it_once() {
        let path = format!("mem://retire-twice/{}/L0.pax", std::process::id());
        forget_object(&path);
        forget_object(&path);
        let key = object_key(&path);
        assert_eq!(retired().iter().filter(|k| **k == key).count(), 1);
    }

    /// Guard: the SST segment readers never read through the filesystem
    /// directly — a direct call would silently bypass residency and cost a
    /// GET on a resident collection (review B1: the first guard matched only
    /// `.read_range(` and missed the batched and whole-object reads).
    #[test]
    fn segment_readers_route_reads_through_residency() {
        for (name, source) in [
            ("segment_format.rs", include_str!("segment_format.rs")),
            ("oid_resolve.rs", include_str!("oid_resolve.rs")),
            ("search/mod.rs", include_str!("search/mod.rs")),
            (
                "centroid_directory_reader.rs (non-test)",
                include_str!("centroid_directory_reader.rs")
                    .split("#[cfg(test)]")
                    .next()
                    .unwrap_or_default(),
            ),
        ] {
            let heads =
                source.matches(".metadata(").count() - source.matches(".metadata()").count();
            assert_eq!(
                heads, 0,
                "{name}: {heads} direct `.metadata(` — use `object_size_resident`"
            );
            for direct in [
                ".read_range(",
                ".read_ranges_prefetch(",
                ".read_ranges(",
                "fs.read(",
            ] {
                let count = source.matches(direct).count();
                assert_eq!(
                    count, 0,
                    "{name}: {count} direct `{direct}` — use the resident read"
                );
            }
        }
    }
}
