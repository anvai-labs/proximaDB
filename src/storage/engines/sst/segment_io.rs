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

use futures::StreamExt as _;
use proximadb_cache::{L2Class, PersistentByteStore, object_key};
use proximadb_storage_filesystem_types::read_ranges_plan;
use proximadb_storage_filesystem_types::read_ranges_prefetch_inflight;
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

/// TD-TIER-1 SC6 S3: one pool's aggregate warm bandwidth (bytes/s), shared
/// by every grant in that pool. A token bucket with one second of burst
/// capacity; `rate <= 0` paces nothing.
pub struct WarmBudget {
    rate_bytes_per_s: f64,
    state: std::sync::Mutex<WarmBucketState>,
}

struct WarmBucketState {
    tokens: f64,
    last: std::time::Instant,
}

impl WarmBudget {
    /// A bucket passing `rate_bytes_per_s` in aggregate.
    pub fn new(rate_bytes_per_s: f64) -> Self {
        Self {
            rate_bytes_per_s,
            state: std::sync::Mutex::new(WarmBucketState {
                tokens: rate_bytes_per_s,
                last: std::time::Instant::now(),
            }),
        }
    }

    /// Account `bytes` of warm transfer and sleep for the deficit. Concurrency
    /// is bounded by WARMS_IN_FLIGHT, so lock contention is negligible.
    pub async fn pace(&self, bytes: u64) {
        if self.rate_bytes_per_s <= 0.0 {
            return;
        }
        let deficit = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let now = std::time::Instant::now();
            let refill =
                now.duration_since(state.last).as_secs_f64().min(1.0) * self.rate_bytes_per_s;
            state.tokens = (state.tokens + refill).min(self.rate_bytes_per_s);
            state.last = now;
            state.tokens -= bytes as f64;
            let deficit_tokens = -state.tokens.min(0.0);
            deficit_tokens / self.rate_bytes_per_s
        };
        if deficit > 0.0 {
            tokio::time::sleep(std::time::Duration::from_secs_f64(deficit.min(60.0))).await;
        }
    }
}

static WARM_PREFIX_BUDGETS: std::sync::OnceLock<
    std::sync::RwLock<Vec<(String, std::sync::Arc<WarmBudget>)>>,
> = std::sync::OnceLock::new();

/// Install the prefix → pool-bucket map (the reservation registry refreshes
/// it whenever grants change). Longest matching prefix wins; an empty map
/// paces nothing (today's behavior).
pub fn set_warm_prefix_budgets(budgets: Vec<(String, std::sync::Arc<WarmBudget>)>) {
    let registry = WARM_PREFIX_BUDGETS.get_or_init(|| std::sync::RwLock::new(Vec::new()));
    let mut guard = registry
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *guard = budgets;
}

fn warm_budget_for(path: &str) -> Option<std::sync::Arc<WarmBudget>> {
    let registry = WARM_PREFIX_BUDGETS.get_or_init(|| std::sync::RwLock::new(Vec::new()));
    let guard = registry
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    guard
        .iter()
        .filter(|(prefix, _)| !prefix.is_empty() && path.starts_with(prefix.as_str()))
        .max_by_key(|(prefix, _)| prefix.len())
        .map(|(_, budget)| budget.clone())
}

fn warm_permits() -> &'static tokio::sync::Semaphore {
    static PERMITS: std::sync::OnceLock<tokio::sync::Semaphore> = std::sync::OnceLock::new();
    PERMITS.get_or_init(|| tokio::sync::Semaphore::new(WARMS_IN_FLIGHT))
}

/// TD-TIER-1 SC4 PR-D: whole-object warms in flight, keyed by the CANONICAL
/// store key ([`object_key`] — the same identity the [`Resident`] copy uses).
/// The WARMS_IN_FLIGHT semaphore bounds CONCURRENCY; this set bounds
/// DUPLICATE targets — a post-flush [`spawn_warm_object`] racing a
/// [`warm_collection`] pass over the same object must not pay the GET twice.
fn warm_inflight() -> std::sync::MutexGuard<'static, std::collections::HashSet<String>> {
    static INFLIGHT: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> =
        std::sync::OnceLock::new();
    INFLIGHT
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Claims one canonical key for the guard's lifetime: every exit path —
/// return, `?` error, panic, task cancellation — runs `Drop`, so a failed or
/// aborted warm never wedges the target behind a dead claim.
struct WarmClaim {
    key: String,
}

impl WarmClaim {
    /// `None` when the key is already being warmed elsewhere in the process.
    fn try_claim(key: String) -> Option<Self> {
        let mut inflight = warm_inflight();
        if inflight.contains(&key) {
            return None;
        }
        inflight.insert(key.clone());
        Some(Self { key })
    }
}

impl Drop for WarmClaim {
    fn drop(&mut self) {
        warm_inflight().remove(&self.key);
    }
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
            crate::storage::engines::sst::metrics::record_resident_hit();
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
            // F1 (SC6 measurement): serve the batch with the SAME plan the
            // remote path uses — one coalesced physical plan, bounded-
            // concurrent fetch, identical logical slicing — swapping only the
            // transport for the resident copy. The previous loop served each
            // logical range through its own spawn_blocking hop one at a time,
            // serializing exactly what the cold path issues as parallel waves
            // (granted warm p50 ~215 ms vs cold ~60 ms at zero GETs).
            let inflight = read_ranges_prefetch_inflight(path).max(1);
            let plan = self.plan_read_ranges(&ranges)?;
            let identity = self.range_coalesce_policy().is_none();
            let physical: Vec<std::ops::Range<u64>> = plan.physical.clone();
            let mut fetch = futures::stream::iter(physical.into_iter().map(|range| {
                let resident = resident.clone();
                async move { resident.range(range.start, range.end - range.start).await }
            }))
            .buffered(inflight);
            let mut buffers = Vec::with_capacity(plan.physical.len());
            let mut complete = true;
            while let Some(item) = fetch.next().await {
                match item {
                    Some(bytes) => buffers.push(bytes),
                    None => {
                        // A resident miss (eviction/corruption): fall back to
                        // the remote batch wholesale, dropping in-flight reads.
                        complete = false;
                        break;
                    }
                }
            }
            if complete {
                crate::storage::engines::sst::metrics::record_resident_hit();
                if identity {
                    return Ok(buffers);
                }
                return Ok(plan
                    .mapping
                    .iter()
                    .map(|slice| match slice.physical {
                        Some(idx) => read_ranges_plan::slice_from_physical(&buffers[idx], *slice),
                        None => Vec::new(),
                    })
                    .collect());
            }
            return self.read_ranges_prefetch(path, ranges).await;
        }
        self.read_ranges_prefetch(path, ranges).await
    }

    async fn read_resident(&self, path: &str) -> FsResult<Vec<u8>> {
        if let Some(resident) = Resident::lookup(path)
            && let Some(bytes) = resident.range(0, resident.len).await
        {
            crate::storage::engines::sst::metrics::record_resident_hit();
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
#[derive(Clone)]
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
    // find the room taken by the ones ahead of it. The setup permit is
    // dropped before the chunk loop: each chunk reacquires (the RAM bound is
    // one chunk per permit), and S3 pacing sleeps BETWEEN permits — a
    // budgeted pool must never pin the process-wide permits while it sleeps
    // (review SF1), or a slow pool starves every other pool's warm.
    {
        let _setup_permit = warm_permits()
            .acquire()
            .await
            .map_err(|error| anyhow::anyhow!("warm permit for {path}: {error}"))?;
        if warm_target(path, len).is_none() {
            return Ok(false);
        }
    }
    let Some((store, key)) = warm_target(path, len) else {
        return Ok(false);
    };
    // SC4 PR-D: claim the canonical key for the duration of the fetch — a
    // concurrent warm of the same object (a post-flush spawn racing a
    // backfill pass) returns early here instead of paying a second GET for
    // bytes the claim holder is already fetching. The claim drops on every
    // exit (return, error, panic, cancellation), and the loser's early
    // return is the same outcome as losing the budget re-judge above: the
    // next warm pass retries if the winner does not land the copy.
    let Some(_claim) = WarmClaim::try_claim(key.clone()) else {
        tracing::debug!("TD-TIER-1 SC4: warm of {path} already in flight; skipping");
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
        // TD-TIER-1 SC6 S3: pace this object's chunks against its pool's
        // aggregate warm budget, when one is installed for the path. Pacing
        // happens OUTSIDE the warm permit (review SF1): a budgeted pool must
        // not pin the process-wide permits while it sleeps, or a slow pool
        // starves every other pool's warm.
        let budget = warm_budget_for(path);
        while offset < len {
            let chunk = WARM_CHUNK_BYTES.min(len - offset);
            if let Some(budget) = &budget {
                budget.pace(chunk).await;
            }
            // The chunk buffer lives INSIDE the permit (review SF2): one
            // WARMS_IN_FLIGHT permit ⇒ at most one 64 MiB buffer held per
            // warm across the process — the write included. Pacing stays
            // before the permit, so a budgeted pool still never pins permits
            // while sleeping.
            {
                let _chunk_permit = warm_permits()
                    .acquire()
                    .await
                    .map_err(|error| anyhow::anyhow!("warm permit for {path}: {error}"))?;
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
            };
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

/// Crash reconciliation, step 1 — snapshot BEFORE the listing: the keys of
/// resident whole-object copies directly under `data_url`. Only these are
/// judged by [`sweep_orphan_objects`], so a copy made after the snapshot (a
/// segment published while the LIST is in flight) is never removed for
/// missing from a listing that predates it (review SF1: nothing would
/// re-warm it).
///
/// Keys normalize documented scheme aliases and absolute local spellings,
/// while preserving exact cloud-key bytes. The root is matched ANCHORED at
/// the start of the location, including every existing trailing delimiter.
pub(crate) fn orphan_candidates(store: &PersistentByteStore, data_url: &str) -> Vec<String> {
    let directory = if data_url.ends_with('/') {
        data_url.to_owned()
    } else {
        format!("{data_url}/")
    };
    let root = object_key(&directory);
    store.keys_where(|key| {
        key.strip_prefix(root.as_str())
            // Direct children only: the listing is not recursive, so a
            // nested object is not judged by it.
            .is_some_and(|name| !name.is_empty() && !name.contains('/'))
    })
}

/// Crash reconciliation, step 2: drop each snapshotted copy whose object the
/// committed set (`objects`, from a listing taken AFTER the snapshot) no
/// longer names. A crash between a segment's delete and its
/// [`forget_object`] — or a retirement in another process — otherwise leaves
/// the copy pinned, charging the grant, until the grant is released. Plain
/// removal, NOT [`forget_object`]: only objects the listing proves gone go.
pub(crate) async fn sweep_orphan_objects(
    store: &PersistentByteStore,
    candidates: Vec<String>,
    objects: &[(String, Option<u64>)],
) -> usize {
    let live: std::collections::HashSet<String> =
        objects.iter().map(|(url, _)| object_key(url)).collect();
    let doomed: std::collections::HashSet<String> = candidates
        .into_iter()
        .filter(|key| !live.contains(key))
        .collect();
    if doomed.is_empty() {
        return 0;
    }
    store.remove_where(|key| doomed.contains(key)).await
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
    let candidates = proximadb_cache::process_store()
        .map(|store| orphan_candidates(&store, data_url))
        .unwrap_or_default();
    let objects = residency_objects(fs, data_url).await?;
    let orphans = match proximadb_cache::process_store() {
        Some(store) => sweep_orphan_objects(&store, candidates, &objects).await,
        None => 0,
    };
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

    #[tokio::test]
    async fn orphan_candidates_preserve_exact_cloud_delimiters() {
        let directory = tempfile::tempdir().expect("cache directory");
        let store = PersistentByteStore::open(directory.path(), 1 << 20).expect("cache store");
        let intended = object_key("s3://bucket/x//a.pax");
        let alternate = object_key("s3://bucket/x/a.pax");
        let nested = object_key("s3://bucket/x///nested.pax");
        let sibling = object_key("s3://bucket/x-other/a.pax");
        for key in [&intended, &alternate, &nested, &sibling] {
            store
                .put(key.clone(), L2Class::Invariants, Arc::from(&b"body"[..]))
                .await
                .expect("seed entry");
        }
        let candidates = orphan_candidates(&store, "s3://bucket/x//");
        assert_eq!(
            candidates,
            vec![intended.clone()],
            "only direct children of the exact prefix qualify"
        );
        assert_eq!(
            orphan_candidates(&store, "s3://bucket/x"),
            vec![alternate.clone()]
        );
        assert_eq!(sweep_orphan_objects(&store, candidates, &[]).await, 1);
        assert!(store.entry_len(&intended).is_none());
        for key in [&alternate, &nested, &sibling] {
            assert!(
                store.entry_len(key).is_some(),
                "other prefixes must survive: {key}"
            );
        }
    }

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

    /// A local fs carrying a range-coalesce policy, so the resident batched
    /// path exercises bridging + logical slicing (the plain local fs is
    /// identity-mapped and would not).
    #[derive(Debug)]
    struct CoalescingLocalFs {
        inner: Arc<CountingFileSystem>,
    }
    proximadb_storage_filesystem_types::forward_filesystem_impl! {
        impl FileSystem for CoalescingLocalFs => inner;
        forward [
            as_any, filesystem_type, supports_mmap, write,
            supports_bounded_local_file_write, write_local_file, write_if_absent, append,
            supports_append, supports_conditional_replace, delete, exists, metadata,
            create_dir, create_dir_all, copy, move_file, sync, sync_file, sync_data, open_file,
        ];
        {
            async fn read(&self, path: &str) -> FsResult<Vec<u8>> {
                self.inner.read(path).await
            }

            async fn read_range(&self, path: &str, offset: u64, length: u64) -> FsResult<Vec<u8>> {
                self.inner.read_range(path, offset, length).await
            }

            async fn list(
                &self,
                path: &str,
            ) -> FsResult<Vec<proximadb_storage_filesystem_types::DirEntry>> {
                self.inner.list(path).await
            }

            fn range_coalesce_policy(
                &self,
            ) -> Option<proximadb_storage_filesystem_types::RangeCoalescePolicy> {
                Some(proximadb_storage_filesystem_types::RangeCoalescePolicy {
                    max_gap_bytes: 64 * 1024,
                    max_merged_bytes: 256 * 1024,
                })
            }
        }
    }

    /// TD-TIER-1 SC6 S3: an installed warm budget paces the aggregate —
    /// warming a multi-chunk object under a small rate takes at least
    /// bytes/rate (minus the one-second burst), and no budget paces nothing.
    #[tokio::test]
    async fn warm_budget_bounds_the_aggregate() {
        let root = tempfile::tempdir().expect("root");
        let (covered, _) = dirs(root.path());
        let _armed = arm(&covered, "s3-budget", 64 << 20);
        let (fs, _counters) = counting_local(root.path()).await;
        // 4 MiB object; a 1 MiB/s budget allows ~1 MiB burst, so >= 3 s.
        let (path, _) = segment(&covered, "L0_budget.pax", 4 << 20);
        let rate = 1024.0 * 1024.0;
        let budget = std::sync::Arc::new(WarmBudget::new(rate));
        set_warm_prefix_budgets(vec![(covered.to_string_lossy().to_string(), budget)]);

        let started = std::time::Instant::now();
        assert!(warm_object(fs.as_ref(), &path, None).await.expect("warm"));
        let elapsed = started.elapsed().as_secs_f64();
        let minimum = (4.0 - 1.0) / 1.0_f64 - 0.3;
        assert!(
            elapsed >= minimum,
            "budgeted warm took {elapsed:.2}s, expected >= {minimum:.2}s"
        );
        set_warm_prefix_budgets(Vec::new());
    }

    /// F1 regression (SC6 measurement): the resident batched-prefetch path
    /// serves the SAME coalesced physical plan the remote path uses —
    /// bridging small gaps, slicing logical ranges back out — byte-exactly
    /// and with zero object-store GETs. The previous implementation served
    /// each logical range through one spawn_blocking hop at a time,
    /// serializing what the cold path issues as parallel waves (granted warm
    /// p50 ~215 ms vs cold ~60 ms at zero GETs). A range beyond the resident
    /// copy falls back to the remote batch wholesale.
    #[tokio::test]
    async fn resident_prefetch_is_coalesced_concurrent_and_byte_exact() {
        let root = tempfile::tempdir().expect("root");
        let (covered, _) = dirs(root.path());
        let _armed = arm(&covered, "f1", 32 << 20);
        let (fs, counters) = counting_local(root.path()).await;
        let coalescing = CoalescingLocalFs { inner: fs.clone() };
        let (path, bytes) = segment(&covered, "L0_f1.pax", 300_000);
        assert!(warm_object(fs.as_ref(), &path, None).await.expect("warm"));

        // Small gaps bridge under the policy; a gap beyond max_gap does not.
        let ranges: Vec<std::ops::Range<u64>> = vec![0..1_000, 2_000..3_000, 150_000..151_000];
        let plan = coalescing.plan_read_ranges(&ranges).expect("plan");
        assert!(
            plan.physical.len() < ranges.len(),
            "coalescing active: {} physical for {} logical",
            plan.physical.len(),
            ranges.len()
        );

        let before = counters.total_gets();
        let out = coalescing
            .read_ranges_prefetch_resident(&path, ranges.clone())
            .await
            .expect("resident batch");
        assert_eq!(counters.total_gets(), before, "zero remote GETs");
        for (slice, range) in out.iter().zip(&ranges) {
            let start = range.start as usize;
            let end = range.end as usize;
            assert_eq!(slice, &bytes[start..end], "logical slice {range:?}");
        }

        // A range past the resident copy falls back to the remote batch
        // wholesale — remote GETs return, results stay byte-exact.
        let mut beyond = ranges.clone();
        beyond.push(400_000..401_000);
        let out = coalescing
            .read_ranges_prefetch_resident(&path, beyond.clone())
            .await
            .expect("fallback batch");
        assert!(
            counters.total_gets() > before,
            "the fallback reads remotely"
        );
        for (slice, range) in out.iter().zip(&beyond) {
            let start = std::cmp::min(range.start as usize, bytes.len());
            let end = std::cmp::min(range.end as usize, bytes.len());
            assert_eq!(slice, &bytes[start..end], "fallback slice {range:?}");
        }
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

    /// SC4 PR-D (in-flight warm dedup): two concurrent warms of the SAME
    /// object pay exactly ONE GET — the loser finds the canonical key already
    /// in flight and returns without fetching (the WARMS_IN_FLIGHT semaphore
    /// bounds concurrency; the in-flight set bounds duplicate targets — a
    /// post-flush `spawn_warm_object` racing a `warm_collection` pass is the
    /// production shape of this). A different object still warms, and a
    /// FAILED warm releases its claim so a later warm of the same object is
    /// not wedged behind it.
    #[tokio::test]
    async fn concurrent_warms_of_one_object_fetch_it_once() {
        let root = tempfile::tempdir().expect("root");
        let (covered, _) = dirs(root.path());
        let _armed = arm(&covered, "warm-dedup", 32 << 20);
        let (fs, counters) = counting_local(root.path()).await;
        let (path, _) = segment(&covered, "L0_d.pax", 200_000);
        let (other, _) = segment(&covered, "L0_e.pax", 2_048);
        let (flaky, _) = segment(&covered, "L0_f.pax", 2_048);

        let (first, second) = tokio::join!(
            warm_object(fs.as_ref(), &path, None),
            warm_object(fs.as_ref(), &path, None),
        );
        let wrote = first.expect("warm a") as usize + second.expect("warm b") as usize;
        assert_eq!(wrote, 1, "exactly one of the racing warms writes");
        assert_eq!(counters.total_gets(), 1, "the duplicate warm paid no GET");
        assert_eq!(
            resident_len(&path),
            Some(200_000),
            "the winner's copy lands"
        );

        // A different object is never deduplicated away.
        assert!(
            warm_object(fs.as_ref(), &other, None)
                .await
                .expect("warm other")
        );

        // A failed warm (declared length past the file: short read) releases
        // its in-flight claim — the same object warms cleanly right after
        // instead of returning early behind a dead claim forever.
        assert!(
            warm_object(fs.as_ref(), &flaky, Some(1 << 20))
                .await
                .is_err(),
            "a length past the file fails the fetch"
        );
        assert!(
            warm_object(fs.as_ref(), &flaky, None)
                .await
                .expect("warm flaky"),
            "the claim was released on failure"
        );
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

    /// A filesystem whose LIST publishes and warms a new segment while the
    /// listing is in flight (it returns the listing taken BEFORE the put).
    #[derive(Debug)]
    struct PublishDuringList {
        inner: Arc<CountingFileSystem>,
        store: Arc<PersistentByteStore>,
        fresh_key: String,
    }

    proximadb_storage_filesystem_types::forward_filesystem_impl! {
        impl FileSystem for PublishDuringList => inner;
        forward [
            as_any, filesystem_type, supports_mmap, write,
            supports_bounded_local_file_write, write_local_file, write_if_absent, append,
            supports_append, supports_conditional_replace, delete, exists, metadata,
            create_dir, create_dir_all, copy, move_file, sync, sync_file, sync_data, open_file,
        ];
        {
            async fn read(&self, path: &str) -> proximadb_storage_filesystem_types::FsResult<Vec<u8>> {
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
            ) -> proximadb_storage_filesystem_types::FsResult<Vec<proximadb_storage_filesystem_types::DirEntry>> {
                let listing = self.inner.list(path).await;
                self.store
                    .put(
                        self.fresh_key.clone(),
                        L2Class::Invariants,
                        Arc::from(&b"bytes"[..]),
                    )
                    .await
                    .map_err(proximadb_storage_filesystem_types::FilesystemError::Io)?;
                listing
            }
        }
    }

    /// Review SF1 (end to end): a copy made while the backfill's LIST is in
    /// flight survives the sweep against that older listing; a stale copy
    /// snapshotted before it does not. Red if `warm_collection` snapshots
    /// after listing.
    #[tokio::test]
    async fn backfill_spares_copies_made_during_its_listing() {
        let root = tempfile::tempdir().expect("root");
        let (covered, _) = dirs(root.path());
        let armed = arm(&covered, "sweep-race", 32 << 20);
        let (fs, _) = counting_local(root.path()).await;
        let data_url = covered.to_string_lossy().to_string();
        let stale = object_key(&format!("{data_url}/L0_stale.pax"));
        armed
            .store
            .put(stale.clone(), L2Class::Invariants, Arc::from(&b"bytes"[..]))
            .await
            .expect("put");
        let fresh = object_key(&format!("{data_url}/L0_fresh.pax"));
        let racing = PublishDuringList {
            inner: fs,
            store: armed.store.clone(),
            fresh_key: fresh.clone(),
        };
        let report = warm_collection(&racing, &data_url, 2).await.expect("warm");
        assert_eq!(report.orphans_dropped, 1, "{report:?}");
        assert_eq!(armed.store.entry_len(&stale), None);
        assert!(
            armed.store.entry_len(&fresh).is_some(),
            "the fresh copy stays"
        );
    }

    /// Review (verification) SF1: the writer's plain path and the listing's
    /// `file://` URL name ONE resident copy, and a root is matched anchored.
    #[test]
    fn local_spellings_share_one_key_and_roots_anchor() {
        assert_eq!(
            object_key("/abs/c1/data/L0.pax"),
            object_key("file:///abs/c1/data/L0.pax")
        );
        let root = object_key("/data/c1/");
        assert!(!object_key("/mnt/data/c1/L0.pax").starts_with(root.as_str()));
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
    ///
    /// TD-TIER-1 SC4 (readers outside the vector profile): the pin covers the
    /// non-vector readers too — the point-lookup/scan PAX path
    /// (`trait_impl.rs`), the legacy `.sst` reader (`sst_query_engine.rs`),
    /// and the DataFusion PAX adapter (`pax_adapter.rs`) — plus
    /// `search/operations.rs`, whose point-lookup was silently uncovered (it
    /// delegates to the legacy reader, which the pin above now catches, but a
    /// future direct read there would have bypassed both).
    ///
    /// Matching runs on WHITESPACE-STRIPPED source: the point-lookup site was
    /// originally a line-broken `fs\n    .read(` that a raw-substring guard
    /// cannot see — exactly the evasion this guard exists to prevent.
    ///
    /// Source PREPARATION (review round-1 B1): the previous helper split at
    /// the first literal `#[cfg(test)]`, which truncates at an attribute
    /// mentioned inside a `//` comment (segment_format.rs line 33) or at the
    /// FIRST interleaved test mod — sst_query_engine.rs's routed sites at
    /// ~2105/~4906/~5220/~5448/~5582 all sat AFTER its first test mod and
    /// were therefore never actually pinned. Preparation now (a) strips `//`
    /// line comments first and (b) cuts every real `#[cfg(test)]` ITEM with
    /// a brace-aware scanner, so production code after interleaved test mods
    /// stays pinned and files whose tests live at the end are handled by the
    /// same scanner. Truncation sentinels below fail loudly if a future
    /// preparation regression re-guts the pin.
    ///
    /// Accepted limitations (disclosed on the TD row): the comment strip
    /// eats the rest of a line containing `//` inside a string literal (none
    /// in the pinned files' production code); the brace scanner counts
    /// braces inside string literals (none unbalanced there); and the
    /// FILESYSTEM-OBJECT forms match on the receiver NAME `fs` — a rename
    /// evades them (a future `source preparation` could match on type, not
    /// text).
    #[test]
    fn segment_readers_route_reads_through_residency() {
        fn strip_line_comments(source: &str) -> String {
            source
                .lines()
                .map(|line| match line.find("//") {
                    Some(cut) => &line[..cut],
                    None => line,
                })
                .collect::<Vec<_>>()
                .join("\n")
        }

        /// Index just past the `}` matching the `{` at `open_at` (None when
        /// the braces never balance — cut to end-of-source in that case; the
        /// sentinels below make that visible instead of silent).
        fn balanced_brace_end(source: &str, open_at: usize) -> Option<usize> {
            let mut depth = 0usize;
            for (i, b) in source.bytes().enumerate().skip(open_at) {
                match b {
                    b'{' => depth += 1,
                    b'}' => {
                        depth -= 1;
                        if depth == 0 {
                            return Some(i + 1);
                        }
                    }
                    _ => {}
                }
            }
            None
        }

        /// Every `#[cfg(test)]` ITEM cut — the attribute(s), the item head,
        /// and for braced items the whole balanced body (`mod tests { … }`,
        /// test-only `fn`s, `#[cfg(test)] use …;`). Chained attributes
        /// (`#[cfg(test)] #[path = "…"] mod tests;`) are skipped to the item
        /// head; a `;`-terminated item ends at the semicolon.
        fn without_test_items(source: &str) -> String {
            const ATTR: &str = "#[cfg(test)]";
            let mut prepared = String::with_capacity(source.len());
            let mut cursor = 0usize;
            while let Some(attr_at) = source[cursor..].find(ATTR).map(|i| i + cursor) {
                prepared.push_str(&source[cursor..attr_at]);
                let mut i = attr_at + ATTR.len();
                loop {
                    i += source[i..].len() - source[i..].trim_start().len();
                    if !source[i..].starts_with("#[") {
                        break;
                    }
                    let mut depth = 0isize;
                    while i < source.len() {
                        match source.as_bytes()[i] {
                            b'[' => depth += 1,
                            b']' => {
                                depth -= 1;
                                if depth == 0 {
                                    i += 1;
                                    break;
                                }
                            }
                            _ => {}
                        }
                        i += 1;
                    }
                }
                let item_end = match source[i..].find(['{', ';']) {
                    Some(rel) => {
                        let at = i + rel;
                        if source.as_bytes()[at] == b';' {
                            at + 1
                        } else {
                            balanced_brace_end(source, at).unwrap_or(source.len())
                        }
                    }
                    None => source.len(),
                };
                cursor = item_end;
            }
            prepared.push_str(&source[cursor..]);
            prepared
        }

        fn production_source(source: &str) -> String {
            without_test_items(&strip_line_comments(source))
        }

        fn norm(source: &str) -> String {
            source.replace(['\n', '\t', ' '], "")
        }

        // Truncation sentinels (review round-1 B1): each prepared source
        // must still reach the END of its file's production code — the old
        // first-`#[cfg(test)]`-split pinned 0.5% of segment_format.rs (first
        // hit: a comment), 3.5% of search/mod.rs (a mid-file test-only fn)
        // and only lines 1-1437 of sst_query_engine.rs. Each sentinel names
        // a production identifier from deep inside the previously-truncated
        // region.
        let segment_format = production_source(include_str!("segment_format.rs"));
        assert!(
            segment_format.contains("rabitq_search_segment_coalesced_allowed"),
            "segment_format.rs preparation truncated: its last production fn \
             (rabitq_search_segment_coalesced_allowed, ~line 4911) is gone"
        );
        let search_mod = production_source(include_str!("search/mod.rs"));
        assert!(
            search_mod.contains("morsel_degree"),
            "search/mod.rs preparation truncated: morsel_degree (~line 267, \
             past the first test-only item) is gone"
        );
        let legacy_reader = production_source(include_str!("readers/sst_query_engine.rs"));
        assert!(
            legacy_reader.contains("MemoryPressure"),
            "sst_query_engine.rs preparation truncated: MemoryPressure \
             (~line 6829, past the interleaved test mods) is gone"
        );

        // STRICT pins: files whose every production read routes through the
        // seam — no direct ranged/batched/whole-object read may appear.
        for (name, source) in [
            ("segment_format.rs", segment_format.as_str()),
            (
                "oid_resolve.rs",
                production_source(include_str!("oid_resolve.rs")).as_str(),
            ),
            ("search/mod.rs", search_mod.as_str()),
            (
                "centroid_directory_reader.rs — directory reads",
                production_source(include_str!("centroid_directory_reader.rs")).as_str(),
            ),
            (
                "trait_impl.rs — point-lookup/scan PAX path",
                production_source(include_str!("trait_impl.rs")).as_str(),
            ),
        ] {
            let normalized = norm(source);
            for direct in [
                ".read_range(",
                ".read_ranges_prefetch(",
                ".read_ranges(",
                "fs.read(",
            ] {
                let count = normalized.matches(direct).count();
                assert_eq!(
                    count, 0,
                    "{name}: {count} direct `{direct}` — use the resident read"
                );
            }
            let heads = normalized.matches(".metadata(").count()
                - normalized.matches(".metadata()").count();
            assert_eq!(
                heads, 0,
                "{name}: {heads} direct `.metadata(` — use `object_size_resident`"
            );
        }

        // FILESYSTEM-OBJECT pins: files that read through wrapper handles
        // (`caching_filesystem`) — banned on the `fs.`-receiver call forms
        // plus the metadata probes (review round-1 M3: a
        // `caching_filesystem.metadata` "cache probe" is a real object HEAD
        // on a metadata-cache miss). The legacy reader cannot carry the
        // `fs.read_range(` absence ban — see the presence ratchet below.
        for (name, source, banned) in [
            (
                "readers/sst_query_engine.rs — legacy .sst reader",
                legacy_reader.as_str(),
                &[
                    "fs.read(",
                    "fs.read_ranges(",
                    "fs.metadata(",
                    "caching_filesystem.metadata(",
                ][..],
            ),
            (
                "search/operations.rs — point-lookup coverage hole",
                production_source(include_str!("search/operations.rs")).as_str(),
                &[
                    "fs.read(",
                    "fs.read_range(",
                    "fs.read_ranges(",
                    "fs.metadata(",
                    "caching_filesystem.metadata(",
                ][..],
            ),
        ] {
            let normalized = norm(source);
            for direct in banned {
                let count = normalized.matches(direct).count();
                assert_eq!(
                    count, 0,
                    "{name}: {count} direct `{direct}` — use the resident read"
                );
            }
        }

        // PRESENCE ratchet for the legacy reader: `fs.read_range(` cannot be
        // an ABSENCE pin — the pre-slice SstDirectReader/UnifiedSstableReader
        // paths still hold ~25 direct `fs.read_range(` reads that slice 4
        // deliberately did not route (open item on the TD row). What slice 4
        // DID route is pinned by PRESENCE: revert any of the routed forms
        // below and this fails.
        let legacy_normalized = norm(&legacy_reader);
        for (routed, at_least, what) in [
            (
                "fs.read_range_resident(",
                1,
                "the ModularBlockReader read_range choke point",
            ),
            ("fs.object_size_resident(", 3, "size probes"),
            ("fs.read_resident(", 2, "whole-object reads"),
        ] {
            let count = legacy_normalized.matches(routed).count();
            assert!(
                count >= at_least,
                "sst_query_engine.rs: only {count} `{routed}` ({what}) — \
                 slice 4 routed {at_least}; a routed site regressed"
            );
        }

        // The DataFusion adapter reads through the FACTORY's convenience
        // methods, so it is pinned on the factory call forms (the resident
        // variants end `_resident` and do not match).
        let adapter = norm(include_str!(
            "../../../datafusion/engine_adapters/pax_adapter.rs"
        ));
        for direct in [
            "filesystem_factory.read(",
            "filesystem_factory.read_range(",
            "filesystem_factory.read_ranges(",
            "filesystem_factory.metadata(",
        ] {
            let count = adapter.matches(direct).count();
            assert_eq!(
                count, 0,
                "pax_adapter.rs: {count} direct `{direct}` — use the `_resident` factory form"
            );
        }
    }
}
