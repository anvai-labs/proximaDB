//! Per-collection vector object-economy directory.
//!
//! This directory is the compact metadata object that cold queries read to plan
//! coalesced object-storage reads against SST/HELIX block layouts. It is a
//! rebuildable projection over canonical SST indexes and the catalog — it is
//! never durable authority. Readers must fall back to embedded SST index
//! metadata when it is missing, stale, or corrupt, and surface
//! `route_degraded=true` in EXPLAIN.
//!
//! Shape mirrors `docs/12-design/VECTOR_OBJECT_ECONOMY_ROUTE_HLD_LLD_2026_05_27.adoc`
//! §LLD:
//!
//! * [`VectorObjectEconomyDirectory`] — top-level per-collection object.
//! * [`ObjectEconomyFileEntry`] — one SST file (or equivalent ProximaBlocks
//!   file) within that collection.
//! * [`ObjectEconomyBlockEntry`] — one data block inside a file.
//!
//! Authority and freshness are explicit fields so the planner can reject stale
//! or external-authoritative directories per ADR-020.

use anyhow::{Context, Result, bail};
use dashmap::DashMap;
use proximadb_catalog::CatalogAuthorityMode;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::sync::OnceCell;

use crate::storage::engines::core::formats::proximablocks::spatial_encoding::SpatialCode;
use crate::storage::engines::sst::IndexEntry;

const DIRECTORY_MAGIC: &[u8; 8] = b"VECTOEDR";
const DIRECTORY_VERSION: u16 = 1;
const HEADER_LEN: usize = 8 + 2 + 4 + 4;

/// Per-collection vector object-economy directory persisted as a compact
/// sidecar object so cold queries can plan in a single round trip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorObjectEconomyDirectory {
    pub version: u16,
    pub collection_id: String,
    pub storage_epoch: u64,
    pub authority_mode: CatalogAuthorityMode,
    pub freshness_watermark_lsn: u64,
    pub freshness_watermark_ns: i64,
    pub pca_model_ref: Option<ObjectRef>,
    pub files: Vec<ObjectEconomyFileEntry>,
}

/// One SST/ProximaBlocks file within a collection's object-economy directory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectEconomyFileEntry {
    pub file_id: String,
    pub object_url: String,
    pub level: u8,
    pub min_key: String,
    pub max_key: Option<String>,
    pub record_count: u64,
    pub file_size_bytes: u64,
    pub vector_dimension: Option<u32>,
    pub centroid_encoding: CentroidEncoding,
    pub centroid_fp16: Option<Vec<u16>>,
    pub centroid_fp32: Option<Vec<f32>>,
    pub zorder_min: Option<SpatialCode>,
    pub zorder_max: Option<SpatialCode>,
    pub block_index_offset: u64,
    pub block_index_size: u32,
    pub blocks: Vec<ObjectEconomyBlockEntry>,
}

/// One data block inside an [`ObjectEconomyFileEntry`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectEconomyBlockEntry {
    pub block_id: u32,
    /// Points at the 4-byte block-size prefix in the file.
    pub offset: u64,
    /// Serialized ProximaDataBlock length, excluding the 4-byte size prefix.
    pub serialized_len: u32,
    pub record_count: u32,
    pub centroid_fp16: Option<Vec<u16>>,
    pub centroid_fp32: Option<Vec<f32>>,
    /// TD-RDSTRAT-5 lever-3: block RMS radius (spread). `#[serde(default)]` →
    /// legacy sidecars (no radius) deserialize to `0.0`, so the read-side prune
    /// scores `d(q,c) − k·0 = d(q,c)` (today's centroid-only ranking) — a change
    /// is mixed-read-safe until segments are re-flushed with the field.
    #[serde(default)]
    pub centroid_radius: f32,
    pub zorder_code: Option<SpatialCode>,
    pub metadata_min_values: serde_json::Map<String, serde_json::Value>,
    pub metadata_max_values: serde_json::Map<String, serde_json::Value>,
    pub metadata_null_counts: BTreeMap<String, u32>,
}

/// Reference to an external object (PCA model, manifest, etc.) used by the
/// route. Kept descriptive — durable authority remains in xCatalog + WAL.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObjectRef {
    pub url: String,
    pub size_bytes: Option<u64>,
    pub etag: Option<String>,
}

/// Centroid representation available in the directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CentroidEncoding {
    #[default]
    None,
    Fp32,
    Fp16,
    Mixed,
}

impl VectorObjectEconomyDirectory {
    /// Construct an empty directory for `collection_id` at `storage_epoch`.
    pub fn empty(
        collection_id: impl Into<String>,
        storage_epoch: u64,
        authority_mode: CatalogAuthorityMode,
    ) -> Self {
        Self {
            version: DIRECTORY_VERSION,
            collection_id: collection_id.into(),
            storage_epoch,
            authority_mode,
            freshness_watermark_lsn: 0,
            freshness_watermark_ns: 0,
            pca_model_ref: None,
            files: Vec::new(),
        }
    }

    /// Set the freshness watermark advertised by this directory. The
    /// watermark is advisory — strong-consistency routes must still merge
    /// canonical WAL/memtable deltas committed after this LSN.
    pub fn with_freshness_watermark(mut self, lsn: u64, ns: i64) -> Self {
        self.freshness_watermark_lsn = lsn;
        self.freshness_watermark_ns = ns;
        self
    }

    pub fn with_pca_model_ref(mut self, pca_model_ref: Option<ObjectRef>) -> Self {
        self.pca_model_ref = pca_model_ref;
        self
    }

    pub fn push_file(&mut self, file: ObjectEconomyFileEntry) {
        self.files.push(file);
    }

    /// Insert or replace a file entry, matching by `file_id`. Returns true if
    /// an existing entry was replaced. Used by the writer to update the
    /// directory after a flush or compaction emits a new file.
    pub fn upsert_file(&mut self, file: ObjectEconomyFileEntry) -> bool {
        if let Some(slot) = self
            .files
            .iter_mut()
            .find(|entry| entry.file_id == file.file_id)
        {
            *slot = file;
            true
        } else {
            self.files.push(file);
            false
        }
    }

    /// Remove a file entry by `file_id`. Returns true if an entry was
    /// removed. Used during compaction when source files are replaced by
    /// output files.
    pub fn remove_file(&mut self, file_id: &str) -> bool {
        let before = self.files.len();
        self.files.retain(|entry| entry.file_id != file_id);
        self.files.len() != before
    }

    /// Total number of cataloged blocks across all files.
    pub fn block_count(&self) -> usize {
        self.files.iter().map(|file| file.blocks.len()).sum()
    }

    pub fn validate(&self) -> Result<()> {
        if self.version != DIRECTORY_VERSION {
            bail!(
                "unsupported vector object-economy directory version {}",
                self.version
            );
        }

        if self.collection_id.is_empty() {
            bail!("vector object-economy directory missing collection_id");
        }

        for file in &self.files {
            file.validate()?;
        }

        Ok(())
    }

    pub fn serialize(&self) -> Result<Vec<u8>> {
        self.validate()?;

        let payload = serde_json::to_vec(self)
            .context("serialize vector object-economy directory payload")?;
        let checksum = crc32fast::hash(&payload);
        let mut out = Vec::with_capacity(HEADER_LEN + payload.len());
        out.extend_from_slice(DIRECTORY_MAGIC);
        out.extend_from_slice(&DIRECTORY_VERSION.to_le_bytes());
        out.extend_from_slice(&checksum.to_le_bytes());
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&payload);
        Ok(out)
    }

    pub fn deserialize(data: &[u8]) -> Result<Self> {
        if data.len() < HEADER_LEN {
            bail!("vector object-economy directory too short");
        }

        if &data[..8] != DIRECTORY_MAGIC {
            bail!("invalid vector object-economy directory magic");
        }

        let version = u16::from_le_bytes(data[8..10].try_into()?);
        if version != DIRECTORY_VERSION {
            bail!("unsupported vector object-economy directory version {version}");
        }

        let expected_checksum = u32::from_le_bytes(data[10..14].try_into()?);
        let payload_len = u32::from_le_bytes(data[14..18].try_into()?) as usize;
        let payload_end = HEADER_LEN
            .checked_add(payload_len)
            .context("vector object-economy directory payload length overflow")?;
        if payload_end != data.len() {
            bail!(
                "vector object-economy directory payload length mismatch: header={} actual={}",
                payload_len,
                data.len().saturating_sub(HEADER_LEN)
            );
        }

        let payload = &data[HEADER_LEN..payload_end];
        let actual_checksum = crc32fast::hash(payload);
        if actual_checksum != expected_checksum {
            bail!("vector object-economy directory checksum mismatch");
        }

        let directory: Self = serde_json::from_slice(payload)
            .context("deserialize vector object-economy directory payload")?;
        directory.validate()?;
        Ok(directory)
    }
}

impl ObjectEconomyFileEntry {
    /// Build a file entry from an SST's IndexEntries plus the surrounding
    /// file-level metadata that the writer or compactor already knows.
    pub fn from_index_entries(
        file_id: impl Into<String>,
        object_url: impl Into<String>,
        level: u8,
        file_size_bytes: u64,
        block_index_offset: u64,
        block_index_size: u32,
        entries: &[IndexEntry],
    ) -> Result<Self> {
        let blocks: Vec<ObjectEconomyBlockEntry> = entries
            .iter()
            .map(ObjectEconomyBlockEntry::from_index_entry)
            .collect();
        let vector_dimension = infer_vector_dimension(&blocks)?;
        let centroid_encoding = infer_centroid_encoding(&blocks);
        let (min_key, max_key) = derive_key_range(entries);
        let (zorder_min, zorder_max) = derive_zorder_range(entries);
        let record_count = blocks.iter().map(|block| block.record_count as u64).sum();

        Ok(Self {
            file_id: file_id.into(),
            object_url: object_url.into(),
            level,
            min_key,
            max_key,
            record_count,
            file_size_bytes,
            vector_dimension,
            centroid_encoding,
            centroid_fp16: None,
            centroid_fp32: None,
            zorder_min,
            zorder_max,
            block_index_offset,
            block_index_size,
            blocks,
        })
    }

    pub fn validate(&self) -> Result<()> {
        if self.file_id.is_empty() {
            bail!("object-economy file entry missing file_id");
        }
        if self.object_url.is_empty() {
            bail!("object-economy file entry missing object_url");
        }

        let inferred_dimension = infer_vector_dimension(&self.blocks)?;
        if self.vector_dimension != inferred_dimension {
            bail!(
                "object-economy file entry vector dimension mismatch: header={:?} inferred={:?}",
                self.vector_dimension,
                inferred_dimension
            );
        }

        Ok(())
    }
}

impl ObjectEconomyBlockEntry {
    fn from_index_entry(entry: &IndexEntry) -> Self {
        Self {
            block_id: entry.block_id,
            offset: entry.offset,
            serialized_len: entry.size,
            // Per-block record_count is not directly carried on IndexEntry today;
            // callers that have it should overwrite this field after construction.
            record_count: 0,
            centroid_fp16: entry.block_centroid_fp16.clone(),
            centroid_fp32: if entry.block_centroid.is_empty() {
                None
            } else {
                Some(entry.block_centroid.clone())
            },
            centroid_radius: entry.block_radius,
            zorder_code: entry.zorder_code.clone(),
            metadata_min_values: entry.metadata_min_values.clone().into_iter().collect(),
            metadata_max_values: entry.metadata_max_values.clone().into_iter().collect(),
            metadata_null_counts: entry.metadata_null_counts.clone().into_iter().collect(),
        }
    }
}

/// Stable per-collection sidecar path. Cold queries fetch this object first;
/// it contains all file/block metadata needed to plan coalesced range reads.
pub fn vector_object_economy_directory_path(collection_root: &str, storage_epoch: u64) -> String {
    let trimmed = collection_root.trim_end_matches('/');
    format!("{trimmed}/oedir/v{storage_epoch}.bin")
}

/// Read-modify-write helper for the per-collection directory.
///
/// The writer and compactor use this to update the directory after a flush
/// or compaction adds, replaces, or removes SST files. The contract is:
///
/// * [`load_or_empty`] returns the current directory, or an empty one if the
///   sidecar is missing or corrupt. Corruption is logged but not propagated —
///   readers must always treat the directory as a rebuildable projection.
/// * [`store`] persists the directory atomically. The caller is responsible
///   for ensuring writes are serialized per collection (a per-collection lock
///   in the flush coordinator is the natural place).
///
/// This module deliberately does not bake in a locking strategy. Concurrency
/// belongs to the caller — typically the SST flush coordinator, which already
/// serializes per-collection work.
pub struct VectorObjectEconomyDirectoryStore<'a> {
    fs: &'a dyn crate::storage::persistence::filesystem::FileSystem,
    pub collection_id: String,
    pub collection_root: String,
    pub storage_epoch: u64,
    pub authority_mode: CatalogAuthorityMode,
}

impl<'a> VectorObjectEconomyDirectoryStore<'a> {
    pub fn new(
        fs: &'a dyn crate::storage::persistence::filesystem::FileSystem,
        collection_id: impl Into<String>,
        collection_root: impl Into<String>,
        storage_epoch: u64,
        authority_mode: CatalogAuthorityMode,
    ) -> Self {
        Self {
            fs,
            collection_id: collection_id.into(),
            collection_root: collection_root.into(),
            storage_epoch,
            authority_mode,
        }
    }

    pub fn path(&self) -> String {
        vector_object_economy_directory_path(&self.collection_root, self.storage_epoch)
    }

    /// Load the directory if present, otherwise return an empty one with the
    /// configured `collection_id`, `storage_epoch`, and `authority_mode`.
    /// Corrupt sidecars are treated as missing — the projection is rebuilt.
    pub async fn load_or_empty(&self) -> VectorObjectEconomyDirectory {
        self.load_with_status().await.0
    }

    /// Load the directory and a diagnostic [`DirectoryLoadStatus`] explaining
    /// whether the sidecar was used as-is or rebuilt. Readers use the status
    /// to mark the route as degraded in EXPLAIN so callers can detect when
    /// they fell back to embedded SST index reads.
    pub async fn load_with_status(&self) -> (VectorObjectEconomyDirectory, DirectoryLoadStatus) {
        let path = self.path();
        let empty = || {
            VectorObjectEconomyDirectory::empty(
                self.collection_id.clone(),
                self.storage_epoch,
                self.authority_mode,
            )
        };

        match self.fs.read(&path).await {
            Ok(bytes) => match VectorObjectEconomyDirectory::deserialize(&bytes) {
                Ok(directory) if directory.collection_id == self.collection_id => {
                    (directory, DirectoryLoadStatus::Loaded)
                }
                Ok(directory) => {
                    let status = DirectoryLoadStatus::Mismatch {
                        expected_collection: self.collection_id.clone(),
                        found_collection: directory.collection_id,
                    };
                    tracing::warn!("{}", status.reason(&path));
                    (empty(), status)
                }
                Err(err) => {
                    let status = DirectoryLoadStatus::Corrupt(err.to_string());
                    tracing::warn!("{}", status.reason(&path));
                    (empty(), status)
                }
            },
            Err(_) => (empty(), DirectoryLoadStatus::Missing),
        }
    }

    /// Persist the directory. Validates before writing so a corrupt update
    /// fails fast rather than poisoning the sidecar.
    pub async fn store(&self, directory: &VectorObjectEconomyDirectory) -> Result<()> {
        let bytes = directory.serialize()?;
        let path = self.path();
        // Ensure the `oedir/` parent exists before writing. Object stores have no
        // real directories so this is a no-op there; on a LOCAL filesystem the
        // write ENOENTs without it — which silently degraded the sidecar to
        // "missing", dropping the centroid probe-prune to a full scan
        // (`centroid_pruned_blocks=0`, caught by the SIFT recall gate). Mirrors the
        // `.pax`/Arrow flush write paths, which create the parent dir first.
        if let Some(parent) = path.rfind('/').map(|i| &path[..i])
            && let Err(err) = self.fs.create_dir_all(parent).await
        {
            // Non-fatal: surface as a warn and let the write below report the real
            // failure if the parent is genuinely unwritable.
            tracing::warn!(
                "object-economy directory: create_dir_all({parent}) failed ({err}); \
                 sidecar write may follow"
            );
        }
        self.fs
            .write(&path, &bytes, None)
            .await
            .map_err(|err| anyhow::anyhow!("write vector object-economy directory: {err}"))
    }

    /// One-call read-modify-write for the writer/compactor: load the current
    /// directory, upsert `file_entry` (insert or replace by `file_id`),
    /// advance the freshness watermark, and persist.
    ///
    /// **Concurrency contract:** the caller MUST serialize these calls per
    /// collection (the SST flush coordinator is the natural lock-holder).
    /// Concurrent calls for the same collection race on the read-modify-write
    /// cycle and can lose updates — the directory will eventually be
    /// rebuilt because it is a [`CatalogAuthorityMode::RebuildableProjection`],
    /// but stale entries in between are possible.
    pub async fn upsert_and_persist(
        &self,
        file_entry: ObjectEconomyFileEntry,
        freshness_lsn: u64,
        freshness_ns: i64,
    ) -> Result<()> {
        let mut directory = self.load_or_empty().await;
        directory.upsert_file(file_entry);
        directory = directory.with_freshness_watermark(freshness_lsn, freshness_ns);
        self.store(&directory).await
    }
}

/// Outcome of a per-collection directory load. Readers use this to populate
/// `rejected_route_reasons` in the EXPLAIN payload so degraded routes are
/// visible to callers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DirectoryLoadStatus {
    /// Sidecar was present, valid, and matched the expected collection.
    Loaded,
    /// Sidecar object was not found at the expected path. Common in
    /// pre-flush state; the route falls back to per-SST index reads.
    Missing,
    /// Sidecar existed but failed validation/deserialization. The route
    /// falls back to per-SST index reads and the projection should be
    /// rebuilt out-of-band.
    Corrupt(String),
    /// Sidecar existed and parsed cleanly but described a different
    /// collection. Treated as a missing projection.
    Mismatch {
        expected_collection: String,
        found_collection: String,
    },
}

impl DirectoryLoadStatus {
    /// Returns true when the route had to fall back to embedded SST index
    /// reads. Callers should mark the route as degraded in EXPLAIN.
    pub fn is_degraded(&self) -> bool {
        !matches!(self, DirectoryLoadStatus::Loaded)
    }

    /// Stable short reason string suitable for
    /// `VectorObjectEconomyExplain.rejected_route_reasons`. Includes the
    /// sidecar path for operability.
    pub fn reason(&self, sidecar_path: &str) -> String {
        match self {
            Self::Loaded => format!("object_economy_directory_loaded:{sidecar_path}"),
            Self::Missing => format!("object_economy_directory_missing:{sidecar_path}"),
            Self::Corrupt(err) => {
                format!("object_economy_directory_corrupt:{sidecar_path}:{err}")
            }
            Self::Mismatch {
                expected_collection,
                found_collection,
            } => format!(
                "object_economy_directory_collection_mismatch:{sidecar_path}:\
                 expected={expected_collection} found={found_collection}"
            ),
        }
    }
}

/// One cached directory load — directory bytes plus the diagnostic status
/// from the load that populated this entry. Arc-wrapped so the OnceCell can
/// hand out cheap shared clones to concurrent readers.
#[derive(Debug, Clone)]
pub struct CachedDirectoryEntry {
    pub directory: VectorObjectEconomyDirectory,
    pub status: DirectoryLoadStatus,
}

/// Per-collection OnceCell wrapper. First reader pays the load cost,
/// subsequent readers reuse the cached `Arc<CachedDirectoryEntry>`.
///
/// **Invalidation contract:** invalidation is by handle replacement, not
/// in-place mutation. When the writer's `upsert_and_persist` lands a new
/// directory version, the orchestrator must call
/// [`VectorObjectEconomyDirectoryCache::invalidate`] for that collection so
/// the next reader gets a fresh handle. This is a deliberate trade-off:
/// `tokio::sync::OnceCell` is one-shot and cheaper than a swap-based
/// alternative, and Phase 7 cache-affinity routing will need a richer
/// epoch-keyed structure anyway.
#[derive(Debug, Default)]
pub struct CachedDirectoryHandle {
    cell: OnceCell<Arc<CachedDirectoryEntry>>,
    /// When the cell was last populated (TD-CACHE-10: drives the cross-pod
    /// staleness re-check — a handle older than the recheck TTL is replaced
    /// so the next reader reloads from the shared sidecar).
    loaded_at: std::sync::Mutex<Option<std::time::Instant>>,
}

impl CachedDirectoryHandle {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns true if the underlying [`OnceCell`] has been populated by a
    /// successful `get_or_load`. Useful for tests and observability.
    pub fn is_initialized(&self) -> bool {
        self.cell.initialized()
    }

    /// Age of the populated cell, or `None` if never loaded.
    pub fn age(&self) -> Option<std::time::Duration> {
        self.loaded_at.lock().ok()?.map(|at| at.elapsed())
    }

    /// True when the handle is populated AND older than `ttl` — the
    /// cross-pod staleness condition for the recheck-TTL contract.
    pub fn expired_for(&self, ttl: std::time::Duration) -> bool {
        self.age().is_some_and(|age| age > ttl)
    }

    /// Return the cached directory entry if this handle has already been
    /// populated. This is intentionally non-loading so diagnostics can inspect
    /// cache state without triggering object-storage reads.
    pub fn get_cached(&self) -> Option<Arc<CachedDirectoryEntry>> {
        self.cell.get().cloned()
    }

    /// Load-once: the first caller's `loader` populates the cell; concurrent
    /// callers wait for the same load. The returned `Arc<CachedDirectoryEntry>`
    /// is cheap to clone for downstream consumers (search plan, EXPLAIN).
    ///
    /// `loader` returns the freshly-loaded directory and its
    /// [`DirectoryLoadStatus`] so the cached entry carries the original
    /// degradation reason — even cached `Missing` entries can populate
    /// EXPLAIN until the cache is invalidated by a write.
    pub async fn get_or_load<F, Fut>(&self, loader: F) -> Arc<CachedDirectoryEntry>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = (VectorObjectEconomyDirectory, DirectoryLoadStatus)>,
    {
        let entry = self
            .cell
            .get_or_init(|| async {
                let (directory, status) = loader().await;
                Arc::new(CachedDirectoryEntry { directory, status })
            })
            .await
            .clone();
        if let Ok(mut at) = self.loaded_at.lock() {
            *at = Some(std::time::Instant::now());
        }
        entry
    }
}

/// Per-process directory cache shared across query workers. Indexed by
/// `collection_id` so a single OnceCell load amortizes across every query
/// against that collection until invalidation.
///
/// **TD-CACHE-10 cross-pod staleness bound:** a handle is invalidated by
/// LOCAL writes only — another pod's flush/compaction is invisible to this
/// process. `recheck_ttl` bounds that window: a handle older than the TTL is
/// replaced on `handle_for`, so the next reader reloads from the shared
/// sidecar. `None` = legacy never-recheck (kill value).
#[derive(Debug, Default)]
pub struct VectorObjectEconomyDirectoryCache {
    inner: DashMap<String, Arc<CachedDirectoryHandle>>,
    recheck_ttl: Option<std::time::Duration>,
}

impl VectorObjectEconomyDirectoryCache {
    pub fn new() -> Self {
        Self {
            recheck_ttl: directory_recheck_ttl_from_env(),
            ..Self::default()
        }
    }

    /// Test/builder form: explicit recheck TTL (`None` = legacy never-recheck).
    pub fn with_recheck_ttl(ttl: Option<std::time::Duration>) -> Self {
        Self {
            recheck_ttl: ttl,
            ..Self::default()
        }
    }

    /// Return a handle for `collection_id`, creating a fresh one on first
    /// touch. The returned `Arc` is cheap to clone for callers that want to
    /// hold the handle across awaits.
    ///
    /// With a recheck TTL configured, an expired handle (populated longer ago
    /// than the TTL) is dropped and replaced so the next `get_or_load`
    /// reloads from shared storage — bounding cross-pod staleness without any
    /// transport. Same-pod writer invalidation (`invalidate`) continues to
    /// force an immediate reload regardless of the TTL.
    pub fn handle_for(&self, collection_id: &str) -> Arc<CachedDirectoryHandle> {
        let handle = self
            .inner
            .entry(collection_id.to_string())
            .or_insert_with(|| Arc::new(CachedDirectoryHandle::new()))
            .clone();
        // TD-OEDIRWM-1: a cached `Missing` entry is NEGATIVE-CACHED — never
        // TTL-replaced. Each replacement re-issued the sidecar GET, which on a
        // collection with no sidecar (never flushed, or a non-SST engine that
        // never emits one) is a swallowed 404 repeating after every idle gap
        // longer than the recheck TTL, forever. Pinning `Missing` is
        // correctness-safe by construction: it yields watermark (0, 0), which
        // makes every Strong/BoundedStale query run the delta scan — the most
        // conservative answer. It is cleared exactly when it can become wrong:
        // the flush-side writer (`emit_after_flush`) calls `invalidate` the
        // moment a sidecar is written, so the next read loads the real one.
        // Present sidecars (`Loaded`) and degraded ones (`Corrupt`,
        // `Mismatch`) keep the recheck TTL, which exists for cross-pod
        // staleness of a sidecar that DOES exist.
        let negative_cached = handle
            .get_cached()
            .is_some_and(|entry| matches!(entry.status, DirectoryLoadStatus::Missing));
        if let Some(ttl) = self.recheck_ttl
            && !negative_cached
            && handle.expired_for(ttl)
        {
            // Replace only if still the same expired handle (a concurrent
            // invalidate/replace wins over our expiry).
            if let Some(mut current) = self.inner.get_mut(collection_id)
                && Arc::ptr_eq(&handle, &current)
            {
                *current = Arc::new(CachedDirectoryHandle::new());
                return current.clone();
            }
            return self
                .inner
                .get(collection_id)
                .map(|e| e.clone())
                .unwrap_or(handle);
        }
        handle
    }

    /// Return an existing handle without creating or loading one. Used by
    /// diagnostics that must not perform object-storage I/O.
    pub fn get_handle(&self, collection_id: &str) -> Option<Arc<CachedDirectoryHandle>> {
        self.inner.get(collection_id).map(|entry| entry.clone())
    }

    /// Drop the cached handle for `collection_id`. The next reader will
    /// allocate a new handle and reload via its loader closure. Called by
    /// the writer/compactor after a directory update lands (Phase 4
    /// integration step still pending).
    pub fn invalidate(&self, collection_id: &str) -> bool {
        self.inner.remove(collection_id).is_some()
    }

    /// True when a handle exists for this collection — does not guarantee
    /// the underlying OnceCell has been initialized.
    pub fn has_handle(&self, collection_id: &str) -> bool {
        self.inner.contains_key(collection_id)
    }

    /// The configured recheck TTL (test/diagnostics visibility).
    pub fn recheck_ttl(&self) -> Option<std::time::Duration> {
        self.recheck_ttl
    }

    /// Total number of cached collection handles. Used by metrics and tests.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

/// TD-CACHE-10: the directory-cache recheck TTL — how long a cached
/// collection directory may serve before the next `handle_for` replaces it
/// so the reader reloads from the shared sidecar. Bounds cross-pod staleness
/// (another pod's flush/compaction is otherwise invisible to this process).
///
/// `PROXIMADB_DIRECTORY_CACHE_RECHECK_MS`: unset ⇒ 30 s; `0` ⇒ never
/// re-check (kill value, legacy behavior); garbage ⇒ default.
pub fn directory_recheck_ttl_from_env() -> Option<std::time::Duration> {
    const DEFAULT_MS: u64 = 30_000;
    match std::env::var("PROXIMADB_DIRECTORY_CACHE_RECHECK_MS")
        .ok()
        .as_deref()
        .map(str::trim)
        .map(str::parse::<u64>)
    {
        Some(Ok(0)) => None,
        Some(Ok(ms)) => Some(std::time::Duration::from_millis(ms)),
        _ => Some(std::time::Duration::from_millis(DEFAULT_MS)),
    }
}

/// Source of "current committed LSN" for a collection, used by the
/// engine-level emission path to populate
/// [`SstableWriterDirectoryHooks::freshness_lsn`] with a real WAL cursor
/// instead of the placeholder `0`.
///
/// Implementations:
/// * `WalCursorLsnSource` (in the WAL module) delegates to the global
///   manifest service.
/// * [`StaticLsnSource`] is a test double that returns a fixed value.
///
/// The trait takes `collection_id` so future per-collection cursors can
/// be plumbed without breaking callers. The initial production impl
/// returns the global LSN regardless of collection — that is still a
/// safe over-approximation: a watermark advanced past unrelated writes
/// just means delta-merge does more work, never less correctness.
#[async_trait::async_trait]
pub trait FreshnessLsnSource: Send + Sync {
    async fn current_lsn(&self, collection_id: &str) -> u64;
}

/// Static source returning a fixed LSN. Used by the engine when no real
/// source has been wired (callers explicitly opt in to default `0`)
/// and by unit tests that need a deterministic value.
#[derive(Debug, Clone, Copy)]
pub struct StaticLsnSource(pub u64);

#[async_trait::async_trait]
impl FreshnessLsnSource for StaticLsnSource {
    async fn current_lsn(&self, _collection_id: &str) -> u64 {
        self.0
    }
}

/// Constructor-injected dependencies for emitting object-economy
/// directory updates from the SST writer/compactor.
///
/// Conservative-by-design: every input the emit path needs is supplied
/// up front by the caller (typically the flush coordinator). There is no
/// path-derived `collection_root`, no wall-clock `storage_epoch`, no
/// implicit `authority_mode`. When a `SstableWriter` is constructed
/// without these hooks it skips directory emission entirely — existing
/// call sites are unaffected until they opt in.
///
/// Concurrency contract: the caller MUST serialize `emit_after_flush`
/// calls per `collection_id`. Concurrent writers racing on the same
/// collection's read-modify-write cycle can lose updates. The natural
/// lock-holder is the flush coordinator, which already serializes
/// flushes per collection. This struct does not introduce its own
/// locking — keeping it pure makes it cheap to wrap in a higher-level
/// coordinator without double-locking.
#[derive(Debug, Clone)]
pub struct SstableWriterDirectoryHooks {
    pub cache: Arc<VectorObjectEconomyDirectoryCache>,
    pub collection_id: String,
    pub collection_root: String,
    pub storage_epoch: u64,
    pub authority_mode: CatalogAuthorityMode,
    /// Freshness LSN that this flush advances the directory's watermark
    /// to. When the WAL cursor isn't plumbed yet, callers pass `0` and
    /// readers requesting strong-freshness routes will treat the
    /// projection as fully stale until a real cursor lands.
    pub freshness_lsn: u64,
    /// LSM level of the output file. Stored on the file entry so the
    /// planner can route reads by level (cold L1+ vs warm L0).
    pub level: u8,
}

impl SstableWriterDirectoryHooks {
    /// Persist a new (or replacement) file entry for this flush and
    /// invalidate the read-side cache so the next reader picks up the
    /// change.
    ///
    /// Idempotent at the file_id level — calling twice with the same
    /// `file_id` replaces the entry in place rather than duplicating.
    /// Different `file_id`s accumulate.
    pub async fn emit_after_flush(
        &self,
        fs: &dyn crate::storage::persistence::filesystem::FileSystem,
        file_id: &str,
        object_url: &str,
        file_size_bytes: u64,
        block_index_offset: u64,
        block_index_size: u32,
        index_entries: &[IndexEntry],
    ) -> Result<()> {
        let file_entry = ObjectEconomyFileEntry::from_index_entries(
            file_id,
            object_url,
            self.level,
            file_size_bytes,
            block_index_offset,
            block_index_size,
            index_entries,
        )?;

        let store = VectorObjectEconomyDirectoryStore::new(
            fs,
            self.collection_id.as_str(),
            self.collection_root.as_str(),
            self.storage_epoch,
            self.authority_mode,
        );

        let freshness_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);

        store
            .upsert_and_persist(file_entry, self.freshness_lsn, freshness_ns)
            .await?;

        // Drop the read-side cached handle so the next reader allocates a
        // fresh OnceCell and reloads from the freshly-written sidecar.
        self.cache.invalidate(&self.collection_id);

        Ok(())
    }
}

/// Loader-closure helper for [`CachedDirectoryHandle::get_or_load`].
///
/// Construct a [`VectorObjectEconomyDirectoryStore`] from the supplied
/// references and run `load_with_status`, yielding the
/// `(directory, status)` tuple the cache expects.
///
/// Callers wrap this in a closure when populating the cache. Typical
/// shape from the search path:
///
/// ```ignore
/// let entry = cache
///     .handle_for(collection_id)
///     .get_or_load(|| async {
///         load_directory_for(
///             &*fs,
///             collection_id,
///             &collection_root,
///             storage_epoch,
///             authority_mode,
///         )
///         .await
///     })
///     .await;
/// ```
///
/// Keeping this as a free function (instead of an `into_loader` method on
/// the store) avoids dragging the store's `'a` lifetime through the
/// `OnceCell`-held future — the closure captures the inputs and constructs
/// the store inside its own async block.
pub async fn load_directory_for(
    fs: &dyn crate::storage::persistence::filesystem::FileSystem,
    collection_id: &str,
    collection_root: &str,
    storage_epoch: u64,
    authority_mode: CatalogAuthorityMode,
) -> (VectorObjectEconomyDirectory, DirectoryLoadStatus) {
    let store = VectorObjectEconomyDirectoryStore::new(
        fs,
        collection_id,
        collection_root,
        storage_epoch,
        authority_mode,
    );
    store.load_with_status().await
}

fn infer_vector_dimension(blocks: &[ObjectEconomyBlockEntry]) -> Result<Option<u32>> {
    let mut dimension = None;
    for block in blocks {
        let block_dimension = block
            .centroid_fp32
            .as_ref()
            .filter(|centroid| !centroid.is_empty())
            .map(|centroid| centroid.len() as u32)
            .or_else(|| {
                block
                    .centroid_fp16
                    .as_ref()
                    .filter(|centroid| !centroid.is_empty())
                    .map(|centroid| centroid.len() as u32)
            });

        if let Some(block_dimension) = block_dimension {
            match dimension {
                Some(existing) if existing != block_dimension => {
                    bail!(
                        "mixed centroid dimensions in object-economy file entry: {} vs {}",
                        existing,
                        block_dimension
                    );
                }
                Some(_) => {}
                None => dimension = Some(block_dimension),
            }
        }
    }
    Ok(dimension)
}

fn infer_centroid_encoding(blocks: &[ObjectEconomyBlockEntry]) -> CentroidEncoding {
    let has_fp32 = blocks
        .iter()
        .any(|block| block.centroid_fp32.as_ref().is_some_and(|v| !v.is_empty()));
    let has_fp16 = blocks
        .iter()
        .any(|block| block.centroid_fp16.as_ref().is_some_and(|v| !v.is_empty()));

    match (has_fp32, has_fp16) {
        (false, false) => CentroidEncoding::None,
        (true, false) => CentroidEncoding::Fp32,
        (false, true) => CentroidEncoding::Fp16,
        (true, true) => CentroidEncoding::Mixed,
    }
}

fn derive_key_range(entries: &[IndexEntry]) -> (String, Option<String>) {
    let min_key = entries
        .first()
        .map(|entry| entry.key.clone())
        .unwrap_or_default();
    let max_key = entries
        .last()
        .and_then(|entry| entry.last_key.clone().or(Some(entry.key.clone())));
    (min_key, max_key)
}

fn derive_zorder_range(entries: &[IndexEntry]) -> (Option<SpatialCode>, Option<SpatialCode>) {
    let mut min_code: Option<SpatialCode> = None;
    let mut max_code: Option<SpatialCode> = None;
    for entry in entries {
        if let Some(code) = entry.zorder_code.clone() {
            min_code = Some(match min_code.take() {
                Some(current) if spatial_lt(&code, &current) => code.clone(),
                Some(current) => current,
                None => code.clone(),
            });
            max_code = Some(match max_code.take() {
                Some(current) if spatial_lt(&current, &code) => code,
                Some(current) => current,
                None => code,
            });
        }
    }
    (min_code, max_code)
}

fn spatial_lt(a: &SpatialCode, b: &SpatialCode) -> bool {
    match (a, b) {
        (SpatialCode::Code64(x), SpatialCode::Code64(y)) => x < y,
        (SpatialCode::Code128(x), SpatialCode::Code128(y)) => x < y,
        // Mixed widths are not directly comparable; treat as not-less so the
        // first-seen value wins. The blueprint expects a single curve type per
        // file, so this is a defensive fallback.
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::engines::core::formats::proximablocks::spatial_encoding::SpatialCode;
    use crate::storage::engines::sst::{IndexEntry, VectorFormat};

    fn index_entry(block_id: u32, offset: u64) -> IndexEntry {
        IndexEntry {
            key: format!("k{block_id:04}"),
            last_key: Some(format!("k{block_id:04}_last")),
            offset,
            size: 128,
            block_id,
            block_offset: block_id * 10,
            compressed: true,
            block_centroid: vec![block_id as f32, block_id as f32 + 1.0],
            block_centroid_fp16: Some(vec![0x3c00, 0x4000]),
            metadata_min_values: [("tenant".to_string(), serde_json::json!("a"))].into(),
            metadata_max_values: [("tenant".to_string(), serde_json::json!("z"))].into(),
            metadata_null_counts: [("tenant".to_string(), 0)].into(),
            zorder_code: Some(SpatialCode::Code64(100 + block_id as u64)),
            vector_format: VectorFormat::Fixed { dimension: 2 },
            ..IndexEntry::default()
        }
    }

    fn sample_file(file_id: &str, level: u8, base_block: u32) -> ObjectEconomyFileEntry {
        let entries = vec![
            index_entry(base_block, 100),
            index_entry(base_block + 1, 300),
        ];
        ObjectEconomyFileEntry::from_index_entries(
            file_id,
            format!("s3://bucket/coll/{file_id}.sst"),
            level,
            4096,
            512,
            64,
            &entries,
        )
        .expect("file entry")
    }

    #[test]
    fn directory_roundtrips_with_checksum_and_authority_fields() {
        let mut directory = VectorObjectEconomyDirectory::empty(
            "coll",
            7,
            CatalogAuthorityMode::ProximaAuthoritative,
        )
        .with_freshness_watermark(1_234, 1_700_000_000)
        .with_pca_model_ref(Some(ObjectRef {
            url: "s3://bucket/coll/model/pca_model.bin".to_string(),
            size_bytes: Some(8192),
            etag: Some("etag-1".to_string()),
        }));
        directory.push_file(sample_file("l0_0001", 0, 0));
        directory.push_file(sample_file("l1_0002", 1, 10));

        let bytes = directory.serialize().expect("serialize");
        let decoded = VectorObjectEconomyDirectory::deserialize(&bytes).expect("deserialize");

        assert_eq!(decoded, directory);
        assert_eq!(decoded.version, DIRECTORY_VERSION);
        assert_eq!(decoded.files.len(), 2);
        assert_eq!(decoded.block_count(), 4);
        assert_eq!(
            decoded.authority_mode,
            CatalogAuthorityMode::ProximaAuthoritative
        );
        assert_eq!(decoded.freshness_watermark_lsn, 1_234);
        assert_eq!(decoded.freshness_watermark_ns, 1_700_000_000);
        assert!(decoded.pca_model_ref.is_some());
        assert_eq!(decoded.files[0].centroid_encoding, CentroidEncoding::Mixed);
        assert_eq!(decoded.files[0].vector_dimension, Some(2));
        assert_eq!(decoded.files[0].blocks[1].offset, 300);
        assert_eq!(decoded.files[0].zorder_min, Some(SpatialCode::Code64(100)));
        assert_eq!(decoded.files[0].zorder_max, Some(SpatialCode::Code64(101)));
    }

    #[test]
    fn directory_rejects_corrupt_payload_checksum() {
        let mut directory = VectorObjectEconomyDirectory::empty(
            "coll",
            1,
            CatalogAuthorityMode::ProximaAuthoritative,
        );
        directory.push_file(sample_file("l0_0001", 0, 0));
        let mut bytes = directory.serialize().expect("serialize");
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;

        let err = VectorObjectEconomyDirectory::deserialize(&bytes)
            .expect_err("corrupt payload should fail");

        assert!(err.to_string().contains("checksum mismatch"));
    }

    #[test]
    fn directory_rejects_empty_collection_id() {
        let mut directory =
            VectorObjectEconomyDirectory::empty("", 1, CatalogAuthorityMode::ProximaAuthoritative);
        directory.push_file(sample_file("l0_0001", 0, 0));

        let err = directory
            .serialize()
            .expect_err("empty collection_id should fail");
        assert!(err.to_string().contains("collection_id"));
    }

    #[test]
    fn file_entry_rejects_mixed_centroid_dimensions() {
        let mut entries = vec![index_entry(0, 100), index_entry(1, 300)];
        entries[1].block_centroid.push(3.0);

        let err = ObjectEconomyFileEntry::from_index_entries(
            "l0_0001",
            "file:///tmp/a.sst",
            0,
            1024,
            128,
            32,
            &entries,
        )
        .expect_err("mixed dimensions should fail");

        assert!(err.to_string().contains("mixed centroid dimensions"));
    }

    #[test]
    fn sidecar_path_is_stable_per_collection_and_epoch() {
        assert_eq!(
            vector_object_economy_directory_path("s3://bucket/coll", 7),
            "s3://bucket/coll/oedir/v7.bin"
        );
        assert_eq!(
            vector_object_economy_directory_path("s3://bucket/coll/", 7),
            "s3://bucket/coll/oedir/v7.bin"
        );
    }

    #[test]
    fn file_entry_derives_key_and_zorder_range() {
        let entries = vec![
            index_entry(0, 100),
            index_entry(1, 300),
            index_entry(2, 500),
        ];
        let file = ObjectEconomyFileEntry::from_index_entries(
            "l0_0001",
            "file:///tmp/a.sst",
            0,
            4096,
            512,
            64,
            &entries,
        )
        .expect("file entry");

        assert_eq!(file.min_key, "k0000");
        assert_eq!(file.max_key.as_deref(), Some("k0002_last"));
        assert_eq!(file.zorder_min, Some(SpatialCode::Code64(100)));
        assert_eq!(file.zorder_max, Some(SpatialCode::Code64(102)));
    }

    #[test]
    fn upsert_file_inserts_then_replaces_by_file_id() {
        let mut directory = VectorObjectEconomyDirectory::empty(
            "coll",
            1,
            CatalogAuthorityMode::ProximaAuthoritative,
        );

        let replaced = directory.upsert_file(sample_file("l0_0001", 0, 0));
        assert!(!replaced);
        assert_eq!(directory.files.len(), 1);

        // Replace the same file_id with a different level — should overwrite,
        // not duplicate.
        let replaced = directory.upsert_file(sample_file("l0_0001", 1, 20));
        assert!(replaced);
        assert_eq!(directory.files.len(), 1);
        assert_eq!(directory.files[0].level, 1);
    }

    #[test]
    fn remove_file_drops_matching_entry() {
        let mut directory = VectorObjectEconomyDirectory::empty(
            "coll",
            1,
            CatalogAuthorityMode::ProximaAuthoritative,
        );
        directory.upsert_file(sample_file("l0_0001", 0, 0));
        directory.upsert_file(sample_file("l1_0002", 1, 10));

        assert!(directory.remove_file("l0_0001"));
        assert_eq!(directory.files.len(), 1);
        assert_eq!(directory.files[0].file_id, "l1_0002");

        assert!(!directory.remove_file("nonexistent"));
    }

    // ── Persistence helper tests use an in-memory FileSystem so the suite
    // ── does not depend on the local filesystem driver setup.

    use crate::storage::persistence::filesystem::{
        DirEntry, FileOptions, FileSystem, FilesystemError, FsFileMetadata, FsResult,
    };
    use std::collections::HashMap;
    use std::sync::Mutex;

    #[derive(Debug, Default)]
    struct InMemoryFs {
        files: Mutex<HashMap<String, Vec<u8>>>,
    }

    #[async_trait::async_trait]
    impl FileSystem for InMemoryFs {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        async fn read(&self, path: &str) -> FsResult<Vec<u8>> {
            self.files
                .lock()
                .unwrap()
                .get(path)
                .cloned()
                .ok_or_else(|| FilesystemError::NotFound(path.to_string()))
        }
        async fn write(
            &self,
            path: &str,
            data: &[u8],
            _options: Option<FileOptions>,
        ) -> FsResult<()> {
            self.files
                .lock()
                .unwrap()
                .insert(path.to_string(), data.to_vec());
            Ok(())
        }
        async fn append(&self, _path: &str, _data: &[u8]) -> FsResult<()> {
            unimplemented!("not needed for these tests")
        }
        async fn delete(&self, _path: &str) -> FsResult<()> {
            unimplemented!("not needed for these tests")
        }
        async fn exists(&self, path: &str) -> FsResult<bool> {
            Ok(self.files.lock().unwrap().contains_key(path))
        }
        async fn metadata(&self, _path: &str) -> FsResult<FsFileMetadata> {
            unimplemented!("not needed for these tests")
        }
        async fn list(&self, _path: &str) -> FsResult<Vec<DirEntry>> {
            unimplemented!("not needed for these tests")
        }
        async fn create_dir(&self, _path: &str) -> FsResult<()> {
            Ok(())
        }
        async fn create_dir_all(&self, _path: &str) -> FsResult<()> {
            Ok(())
        }
        async fn copy(&self, _src: &str, _dst: &str) -> FsResult<()> {
            unimplemented!("not needed for these tests")
        }
        async fn move_file(&self, _src: &str, _dst: &str) -> FsResult<()> {
            unimplemented!("not needed for these tests")
        }
        fn filesystem_type(&self) -> &'static str {
            "in-memory-test"
        }
        async fn sync(&self) -> FsResult<()> {
            Ok(())
        }
        async fn open_file(
            &self,
            _path: &str,
            _create: bool,
        ) -> FsResult<Box<dyn crate::storage::persistence::filesystem::FilesystemFile>> {
            unimplemented!("not needed for these tests")
        }
    }

    #[tokio::test]
    async fn store_returns_empty_directory_when_sidecar_absent() {
        let fs = InMemoryFs::default();
        let store = VectorObjectEconomyDirectoryStore::new(
            &fs,
            "coll",
            "s3://bucket/coll",
            5,
            CatalogAuthorityMode::ProximaAuthoritative,
        );

        let directory = store.load_or_empty().await;
        assert_eq!(directory.collection_id, "coll");
        assert_eq!(directory.storage_epoch, 5);
        assert!(directory.files.is_empty());
    }

    #[tokio::test]
    async fn store_then_load_roundtrips_directory_through_filesystem() {
        let fs = InMemoryFs::default();
        let store = VectorObjectEconomyDirectoryStore::new(
            &fs,
            "coll",
            "s3://bucket/coll",
            9,
            CatalogAuthorityMode::ProximaAuthoritative,
        );

        let mut directory = store.load_or_empty().await;
        directory.upsert_file(sample_file("l0_0001", 0, 0));
        directory = directory.with_freshness_watermark(42, 1_700_000_000);
        store.store(&directory).await.expect("store");

        let reloaded = store.load_or_empty().await;
        assert_eq!(reloaded.files.len(), 1);
        assert_eq!(reloaded.freshness_watermark_lsn, 42);
        assert_eq!(reloaded.freshness_watermark_ns, 1_700_000_000);
        assert_eq!(reloaded.files[0].file_id, "l0_0001");
    }

    #[tokio::test]
    async fn store_treats_collection_id_mismatch_as_missing() {
        let fs = InMemoryFs::default();
        // Pre-populate the sidecar with a directory for a DIFFERENT collection.
        let path = vector_object_economy_directory_path("s3://bucket/coll", 3);
        let other = VectorObjectEconomyDirectory::empty(
            "other-collection",
            3,
            CatalogAuthorityMode::ProximaAuthoritative,
        );
        // Skip empty-files validation by adding a file entry.
        let mut other = other;
        other.upsert_file(sample_file("l0_0001", 0, 0));
        let bytes = other.serialize().expect("serialize");
        fs.write(&path, &bytes, None).await.expect("seed sidecar");

        let store = VectorObjectEconomyDirectoryStore::new(
            &fs,
            "coll",
            "s3://bucket/coll",
            3,
            CatalogAuthorityMode::ProximaAuthoritative,
        );
        let directory = store.load_or_empty().await;
        assert_eq!(directory.collection_id, "coll");
        assert!(directory.files.is_empty());
    }

    #[tokio::test]
    async fn load_with_status_reports_missing_when_sidecar_absent() {
        let fs = InMemoryFs::default();
        let store = VectorObjectEconomyDirectoryStore::new(
            &fs,
            "coll",
            "s3://bucket/coll",
            5,
            CatalogAuthorityMode::ProximaAuthoritative,
        );

        let (directory, status) = store.load_with_status().await;
        assert_eq!(directory.collection_id, "coll");
        assert!(directory.files.is_empty());
        assert_eq!(status, DirectoryLoadStatus::Missing);
        assert!(status.is_degraded());
    }

    #[tokio::test]
    async fn load_with_status_reports_loaded_after_store() {
        let fs = InMemoryFs::default();
        let store = VectorObjectEconomyDirectoryStore::new(
            &fs,
            "coll",
            "s3://bucket/coll",
            5,
            CatalogAuthorityMode::ProximaAuthoritative,
        );
        let mut dir = store.load_or_empty().await;
        dir.upsert_file(sample_file("l0_0001", 0, 0));
        store.store(&dir).await.expect("store");

        let (reloaded, status) = store.load_with_status().await;
        assert_eq!(reloaded.files.len(), 1);
        assert_eq!(status, DirectoryLoadStatus::Loaded);
        assert!(!status.is_degraded());
    }

    #[tokio::test]
    async fn load_with_status_reports_corrupt_when_payload_truncated() {
        let fs = InMemoryFs::default();
        let path = vector_object_economy_directory_path("s3://bucket/coll", 7);
        // Write something that fails the magic-bytes check.
        fs.write(&path, b"NOT_A_DIRECTORY_PAYLOAD", None)
            .await
            .expect("seed corrupt sidecar");

        let store = VectorObjectEconomyDirectoryStore::new(
            &fs,
            "coll",
            "s3://bucket/coll",
            7,
            CatalogAuthorityMode::ProximaAuthoritative,
        );
        let (directory, status) = store.load_with_status().await;
        assert!(directory.files.is_empty());
        match status {
            DirectoryLoadStatus::Corrupt(_) => {}
            other => panic!("expected Corrupt, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn load_with_status_reports_mismatch_when_collection_differs() {
        let fs = InMemoryFs::default();
        let path = vector_object_economy_directory_path("s3://bucket/coll", 3);
        let mut other = VectorObjectEconomyDirectory::empty(
            "other-collection",
            3,
            CatalogAuthorityMode::ProximaAuthoritative,
        );
        other.upsert_file(sample_file("l0_0001", 0, 0));
        let bytes = other.serialize().expect("serialize");
        fs.write(&path, &bytes, None).await.expect("seed sidecar");

        let store = VectorObjectEconomyDirectoryStore::new(
            &fs,
            "coll",
            "s3://bucket/coll",
            3,
            CatalogAuthorityMode::ProximaAuthoritative,
        );
        let (directory, status) = store.load_with_status().await;
        assert!(directory.files.is_empty());
        match status {
            DirectoryLoadStatus::Mismatch {
                expected_collection,
                found_collection,
            } => {
                assert_eq!(expected_collection, "coll");
                assert_eq!(found_collection, "other-collection");
            }
            other => panic!("expected Mismatch, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn upsert_and_persist_evolves_directory_across_simulated_flushes() {
        let fs = InMemoryFs::default();
        let store = VectorObjectEconomyDirectoryStore::new(
            &fs,
            "coll",
            "s3://bucket/coll",
            11,
            CatalogAuthorityMode::ProximaAuthoritative,
        );

        // Flush 1: first L0 SST lands. Directory starts empty, ends with 1 file.
        store
            .upsert_and_persist(sample_file("l0_0001", 0, 0), 100, 1_700_000_000_000)
            .await
            .expect("flush 1");

        let after_flush_1 = store.load_or_empty().await;
        assert_eq!(after_flush_1.files.len(), 1);
        assert_eq!(after_flush_1.freshness_watermark_lsn, 100);

        // Flush 2: second L0 SST. Both files now visible; watermark advances.
        store
            .upsert_and_persist(sample_file("l0_0002", 0, 10), 250, 1_700_000_001_000)
            .await
            .expect("flush 2");

        let after_flush_2 = store.load_or_empty().await;
        assert_eq!(after_flush_2.files.len(), 2);
        assert_eq!(after_flush_2.freshness_watermark_lsn, 250);
        assert!(after_flush_2.files.iter().any(|f| f.file_id == "l0_0001"));
        assert!(after_flush_2.files.iter().any(|f| f.file_id == "l0_0002"));

        // Compaction: l0_0001 is rewritten at level 1. Same file_id, new level.
        // upsert replaces the entry rather than duplicating.
        store
            .upsert_and_persist(sample_file("l0_0001", 1, 0), 400, 1_700_000_002_000)
            .await
            .expect("compaction rewrite");

        let after_compaction = store.load_or_empty().await;
        assert_eq!(after_compaction.files.len(), 2);
        assert_eq!(after_compaction.freshness_watermark_lsn, 400);
        let l0_0001 = after_compaction
            .files
            .iter()
            .find(|f| f.file_id == "l0_0001")
            .expect("l0_0001 still present");
        assert_eq!(l0_0001.level, 1, "compaction promoted level");
    }

    #[tokio::test]
    async fn upsert_and_persist_recovers_from_torn_sidecar() {
        // Simulate a previous flush that wrote a corrupt sidecar (e.g. process
        // crash mid-write). The next flush should treat it as missing and
        // rebuild — the directory is a rebuildable projection.
        let fs = InMemoryFs::default();
        let path = vector_object_economy_directory_path("s3://bucket/coll", 3);
        fs.write(&path, b"TORN_PAYLOAD", None)
            .await
            .expect("seed torn sidecar");

        let store = VectorObjectEconomyDirectoryStore::new(
            &fs,
            "coll",
            "s3://bucket/coll",
            3,
            CatalogAuthorityMode::ProximaAuthoritative,
        );
        store
            .upsert_and_persist(sample_file("l0_0001", 0, 0), 50, 0)
            .await
            .expect("recovery flush");

        let (recovered, status) = store.load_with_status().await;
        assert_eq!(recovered.files.len(), 1);
        assert_eq!(status, DirectoryLoadStatus::Loaded);
    }

    // ── CachedDirectoryHandle / cache tests ──────────────────────────────

    #[tokio::test]
    async fn cached_handle_calls_loader_once_and_reuses_entry() {
        use std::sync::atomic::Ordering;

        let handle = CachedDirectoryHandle::new();
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        assert!(!handle.is_initialized());

        // First load.
        let calls_clone = calls.clone();
        let first = handle
            .get_or_load(|| async move {
                calls_clone.fetch_add(1, Ordering::SeqCst);
                (
                    VectorObjectEconomyDirectory::empty(
                        "coll",
                        1,
                        CatalogAuthorityMode::ProximaAuthoritative,
                    ),
                    DirectoryLoadStatus::Missing,
                )
            })
            .await;
        assert!(handle.is_initialized());
        assert_eq!(first.status, DirectoryLoadStatus::Missing);

        // Second load reuses the cached entry; loader does not fire again.
        let calls_clone = calls.clone();
        let second = handle
            .get_or_load(|| async move {
                calls_clone.fetch_add(1, Ordering::SeqCst);
                (
                    VectorObjectEconomyDirectory::empty(
                        "coll",
                        2,
                        CatalogAuthorityMode::ProximaAuthoritative,
                    ),
                    DirectoryLoadStatus::Loaded,
                )
            })
            .await;
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "loader should fire once"
        );
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[tokio::test]
    async fn cached_handle_concurrent_callers_share_single_load() {
        let handle = Arc::new(CachedDirectoryHandle::new());
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let tasks: Vec<_> = (0..8)
            .map(|_| {
                let handle = handle.clone();
                let calls = calls.clone();
                tokio::spawn(async move {
                    handle
                        .get_or_load(|| async move {
                            // Yield to let other tasks queue on the OnceCell.
                            tokio::task::yield_now().await;
                            calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            (
                                VectorObjectEconomyDirectory::empty(
                                    "coll",
                                    1,
                                    CatalogAuthorityMode::ProximaAuthoritative,
                                ),
                                DirectoryLoadStatus::Missing,
                            )
                        })
                        .await
                })
            })
            .collect();

        for task in tasks {
            task.await.expect("task");
        }

        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "concurrent callers must coalesce to one load"
        );
    }

    #[tokio::test]
    async fn directory_cache_isolates_per_collection_handles() {
        let cache = VectorObjectEconomyDirectoryCache::new();
        let h1 = cache.handle_for("coll-a");
        let h2 = cache.handle_for("coll-b");
        let h1_again = cache.handle_for("coll-a");

        assert!(
            !Arc::ptr_eq(&h1, &h2),
            "different collections get distinct handles"
        );
        assert!(
            Arc::ptr_eq(&h1, &h1_again),
            "same collection returns same handle"
        );
        assert_eq!(cache.len(), 2);
        assert!(cache.has_handle("coll-a"));
        assert!(!cache.has_handle("coll-c"));
    }

    #[tokio::test]
    async fn directory_cache_invalidate_forces_fresh_handle_and_reload() {
        let cache = VectorObjectEconomyDirectoryCache::new();
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let load = |epoch: u64, status: DirectoryLoadStatus| {
            let calls = calls.clone();
            async move {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                (
                    VectorObjectEconomyDirectory::empty(
                        "coll",
                        epoch,
                        CatalogAuthorityMode::ProximaAuthoritative,
                    ),
                    status,
                )
            }
        };

        let handle1 = cache.handle_for("coll");
        let entry1 = handle1
            .get_or_load(|| load(1, DirectoryLoadStatus::Missing))
            .await;
        assert_eq!(entry1.directory.storage_epoch, 1);
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);

        // Without invalidation, the cached handle returns the same entry.
        let entry_again = handle1
            .get_or_load(|| load(99, DirectoryLoadStatus::Loaded))
            .await;
        assert!(Arc::ptr_eq(&entry1, &entry_again));
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);

        // Invalidate → next handle_for returns a fresh OnceCell.
        assert!(cache.invalidate("coll"));
        let handle2 = cache.handle_for("coll");
        assert!(!Arc::ptr_eq(&handle1, &handle2));

        let entry2 = handle2
            .get_or_load(|| load(2, DirectoryLoadStatus::Loaded))
            .await;
        assert_eq!(entry2.directory.storage_epoch, 2);
        assert_eq!(entry2.status, DirectoryLoadStatus::Loaded);
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "reload fires after invalidate"
        );
    }

    #[test]
    fn directory_cache_invalidate_missing_collection_is_no_op() {
        let cache = VectorObjectEconomyDirectoryCache::new();
        assert!(!cache.invalidate("never-touched"));
    }

    // ── TD-CACHE-10: cross-pod staleness bound (recheck TTL) ─────────────────
    //
    // The OnceCell handle is invalidated only by LOCAL writes; a different pod
    // flushing/compacting the same collection leaves this pod serving a stale
    // segment list indefinitely. The recheck TTL bounds that window: after
    // `ttl` elapses since the last load, `handle_for` replaces the handle so
    // the next reader reloads from the shared sidecar.

    fn counting_loader(
        calls: &std::sync::Arc<std::sync::atomic::AtomicUsize>,
        epoch: u64,
    ) -> impl FnOnce() -> std::future::Ready<(VectorObjectEconomyDirectory, DirectoryLoadStatus)> + '_
    {
        let calls = calls.clone();
        move || {
            calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            std::future::ready((
                VectorObjectEconomyDirectory::empty(
                    "coll",
                    epoch,
                    CatalogAuthorityMode::ProximaAuthoritative,
                ),
                DirectoryLoadStatus::Loaded,
            ))
        }
    }

    #[tokio::test]
    async fn recheck_ttl_reloads_expired_handle_exactly_once() {
        use std::time::Duration;
        let cache =
            VectorObjectEconomyDirectoryCache::with_recheck_ttl(Some(Duration::from_millis(25)));
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let handle = cache.handle_for("coll");
        let e1 = handle
            .get_or_load(|| {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                std::future::ready((
                    VectorObjectEconomyDirectory::empty(
                        "coll",
                        1,
                        CatalogAuthorityMode::ProximaAuthoritative,
                    ),
                    DirectoryLoadStatus::Loaded,
                ))
            })
            .await;
        assert_eq!(e1.directory.storage_epoch, 1);

        // Within TTL: same handle, no reload.
        let e1_again = cache
            .handle_for("coll")
            .get_or_load(|| {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                std::future::ready((
                    VectorObjectEconomyDirectory::empty(
                        "coll",
                        99,
                        CatalogAuthorityMode::ProximaAuthoritative,
                    ),
                    DirectoryLoadStatus::Loaded,
                ))
            })
            .await;
        assert!(Arc::ptr_eq(&e1, &e1_again));
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);

        // Past TTL: handle_for replaces the handle; next reader reloads.
        tokio::time::sleep(Duration::from_millis(45)).await;
        let handle2 = cache.handle_for("coll");
        assert!(
            !Arc::ptr_eq(&handle, &handle2),
            "expired handle must be replaced"
        );
        let e2 = handle2
            .get_or_load(|| {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                std::future::ready((
                    VectorObjectEconomyDirectory::empty(
                        "coll",
                        2,
                        CatalogAuthorityMode::ProximaAuthoritative,
                    ),
                    DirectoryLoadStatus::Loaded,
                ))
            })
            .await;
        assert_eq!(e2.directory.storage_epoch, 2);
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "exactly one reload after expiry"
        );
    }

    /// TD-OEDIRWM-1: a `Missing` entry is negative-cached. Past the recheck
    /// TTL the handle is KEPT (no second sidecar GET — on a collection with
    /// no sidecar that GET is a 404 that repeated after every idle gap,
    /// forever), and `invalidate` — which the flush-side writer calls when it
    /// emits a sidecar — clears it so the next reader loads the real one.
    #[tokio::test]
    async fn missing_entry_is_negative_cached_until_invalidated() {
        use std::time::Duration;
        let cache =
            VectorObjectEconomyDirectoryCache::with_recheck_ttl(Some(Duration::from_millis(25)));
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let load = |epoch: u64, status: DirectoryLoadStatus| {
            let calls = Arc::clone(&calls);
            move || {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                std::future::ready((
                    VectorObjectEconomyDirectory::empty(
                        "coll",
                        epoch,
                        CatalogAuthorityMode::ProximaAuthoritative,
                    ),
                    status,
                ))
            }
        };

        let handle = cache.handle_for("coll");
        handle
            .get_or_load(load(1, DirectoryLoadStatus::Missing))
            .await;
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);

        // Well past the TTL: a Missing handle must NOT be replaced.
        tokio::time::sleep(Duration::from_millis(60)).await;
        let same = cache.handle_for("coll");
        assert!(
            Arc::ptr_eq(&handle, &same),
            "a Missing entry must survive the recheck TTL (negative cache)"
        );
        same.get_or_load(load(2, DirectoryLoadStatus::Loaded)).await;
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "no second sidecar load while negative-cached"
        );

        // A write invalidates it: the next reader loads the real sidecar.
        assert!(cache.invalidate("coll"));
        let fresh = cache.handle_for("coll");
        let loaded = fresh
            .get_or_load(load(3, DirectoryLoadStatus::Loaded))
            .await;
        assert_eq!(loaded.directory.storage_epoch, 3);
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn recheck_ttl_none_preserves_legacy_never_recheck() {
        use std::time::Duration;
        let cache = VectorObjectEconomyDirectoryCache::with_recheck_ttl(None);
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let handle = cache.handle_for("coll");
        let e1 = handle
            .get_or_load(|| {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                std::future::ready((
                    VectorObjectEconomyDirectory::empty(
                        "coll",
                        1,
                        CatalogAuthorityMode::ProximaAuthoritative,
                    ),
                    DirectoryLoadStatus::Loaded,
                ))
            })
            .await;

        tokio::time::sleep(Duration::from_millis(40)).await;
        let e_again = cache
            .handle_for("coll")
            .get_or_load(|| {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                std::future::ready((
                    VectorObjectEconomyDirectory::empty(
                        "coll",
                        99,
                        CatalogAuthorityMode::ProximaAuthoritative,
                    ),
                    DirectoryLoadStatus::Loaded,
                ))
            })
            .await;
        assert!(Arc::ptr_eq(&e1, &e_again));
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "None TTL = legacy behavior"
        );
    }

    #[tokio::test]
    async fn writer_invalidate_forces_reload_even_with_ttl_set() {
        use std::time::Duration;
        let cache =
            VectorObjectEconomyDirectoryCache::with_recheck_ttl(Some(Duration::from_secs(3600)));
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let handle = cache.handle_for("coll");
        let e1 = handle
            .get_or_load(|| {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                std::future::ready((
                    VectorObjectEconomyDirectory::empty(
                        "coll",
                        1,
                        CatalogAuthorityMode::ProximaAuthoritative,
                    ),
                    DirectoryLoadStatus::Loaded,
                ))
            })
            .await;

        // Writer-side invalidation (same pod) must bypass the TTL entirely.
        assert!(cache.invalidate("coll"));
        let e2 = cache
            .handle_for("coll")
            .get_or_load(|| {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                std::future::ready((
                    VectorObjectEconomyDirectory::empty(
                        "coll",
                        2,
                        CatalogAuthorityMode::ProximaAuthoritative,
                    ),
                    DirectoryLoadStatus::Loaded,
                ))
            })
            .await;
        assert_ne!(e1.directory.storage_epoch, e2.directory.storage_epoch);
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[test]
    fn directory_recheck_ttl_env_parses_contract() {
        use std::time::Duration;
        unsafe { std::env::remove_var("PROXIMADB_DIRECTORY_CACHE_RECHECK_MS") };
        assert_eq!(
            directory_recheck_ttl_from_env(),
            Some(Duration::from_secs(30)),
            "unset ⇒ 30 s default"
        );
        unsafe { std::env::set_var("PROXIMADB_DIRECTORY_CACHE_RECHECK_MS", "5000") };
        assert_eq!(
            directory_recheck_ttl_from_env(),
            Some(Duration::from_millis(5000))
        );
        unsafe { std::env::set_var("PROXIMADB_DIRECTORY_CACHE_RECHECK_MS", "0") };
        assert_eq!(
            directory_recheck_ttl_from_env(),
            None,
            "0 ⇒ never re-check (kill value)"
        );
        unsafe { std::env::set_var("PROXIMADB_DIRECTORY_CACHE_RECHECK_MS", "garbage") };
        assert_eq!(
            directory_recheck_ttl_from_env(),
            Some(Duration::from_secs(30)),
            "garbage ⇒ default"
        );
        unsafe { std::env::remove_var("PROXIMADB_DIRECTORY_CACHE_RECHECK_MS") };
    }

    #[tokio::test]
    async fn load_directory_for_closure_pattern_works_with_cache() {
        // End-to-end: cache + loader closure backed by an in-memory FileSystem
        // is the integration pattern the search path will follow.
        let fs = Arc::new(InMemoryFs::default());
        let cache = VectorObjectEconomyDirectoryCache::new();

        // Seed a real sidecar so the loader returns Loaded, not Missing.
        let store = VectorObjectEconomyDirectoryStore::new(
            &*fs,
            "coll",
            "s3://bucket/coll",
            12,
            CatalogAuthorityMode::ProximaAuthoritative,
        );
        let mut directory = store.load_or_empty().await;
        directory.upsert_file(sample_file("l0_0001", 0, 0));
        store.store(&directory).await.expect("seed sidecar");

        // First cache miss: the loader closure runs once and populates the
        // OnceCell.
        let fs_for_closure = fs.clone();
        let entry = cache
            .handle_for("coll")
            .get_or_load(|| async move {
                load_directory_for(
                    &*fs_for_closure,
                    "coll",
                    "s3://bucket/coll",
                    12,
                    CatalogAuthorityMode::ProximaAuthoritative,
                )
                .await
            })
            .await;

        assert_eq!(entry.status, DirectoryLoadStatus::Loaded);
        assert_eq!(entry.directory.files.len(), 1);
        assert_eq!(entry.directory.storage_epoch, 12);

        // Second read returns the same cached Arc — closure does not fire
        // again. We confirm by passing a panicking loader.
        let entry_again = cache
            .handle_for("coll")
            .get_or_load(|| async { panic!("loader should not fire on cached read") })
            .await;
        assert!(Arc::ptr_eq(&entry, &entry_again));
    }

    #[test]
    fn directory_load_status_reason_strings_are_stable() {
        let path = "s3://bucket/coll/oedir/v3.bin";

        assert_eq!(
            DirectoryLoadStatus::Missing.reason(path),
            "object_economy_directory_missing:s3://bucket/coll/oedir/v3.bin"
        );
        assert!(
            DirectoryLoadStatus::Corrupt("bad magic".into())
                .reason(path)
                .starts_with("object_economy_directory_corrupt:")
        );
        assert!(
            DirectoryLoadStatus::Mismatch {
                expected_collection: "coll".into(),
                found_collection: "other".into(),
            }
            .reason(path)
            .contains("collection_mismatch")
        );
        // Loaded is non-degraded but still has a stable reason so EXPLAIN
        // can opt in to verbose tracing if needed.
        assert!(!DirectoryLoadStatus::Loaded.is_degraded());
    }

    // ── SstableWriterDirectoryHooks tests (TDD) ───────────────────────────
    //
    // The writer-side counterpart of the cache: after a successful SST
    // flush, hooks.emit_after_flush updates the per-collection sidecar
    // and invalidates the read-side cache so subsequent readers see the
    // new file. The hooks are constructor-injected (no path derivation,
    // no fallback magic) — when unset, the writer skips emission entirely.

    fn sample_index_entries(base_block: u32) -> Vec<IndexEntry> {
        vec![
            index_entry(base_block, 100),
            index_entry(base_block + 1, 300),
        ]
    }

    fn hooks_with_fresh_cache(
        collection_id: &str,
        storage_epoch: u64,
    ) -> (
        SstableWriterDirectoryHooks,
        Arc<VectorObjectEconomyDirectoryCache>,
    ) {
        let cache = Arc::new(VectorObjectEconomyDirectoryCache::new());
        let hooks = SstableWriterDirectoryHooks {
            cache: cache.clone(),
            collection_id: collection_id.to_string(),
            collection_root: "s3://bucket/coll".to_string(),
            storage_epoch,
            authority_mode: CatalogAuthorityMode::RebuildableProjection,
            freshness_lsn: 100,
            level: 0,
        };
        (hooks, cache)
    }

    // ── FreshnessLsnSource tests (Slice 5.2) ─────────────────────────────

    #[tokio::test]
    async fn static_lsn_source_returns_configured_value_per_collection() {
        let src = StaticLsnSource(1_234);
        assert_eq!(src.current_lsn("collection-a").await, 1_234);
        assert_eq!(src.current_lsn("collection-b").await, 1_234);
    }

    #[tokio::test]
    async fn freshness_lsn_source_trait_object_dispatch_works() {
        let dyn_src: Arc<dyn FreshnessLsnSource> = Arc::new(StaticLsnSource(99));
        assert_eq!(dyn_src.current_lsn("coll").await, 99);
    }

    #[tokio::test]
    async fn emit_after_flush_persists_directory_entry_for_new_file() {
        let fs = InMemoryFs::default();
        let (hooks, _cache) = hooks_with_fresh_cache("coll", 7);

        hooks
            .emit_after_flush(
                &fs,
                "l0_0001",
                "s3://bucket/coll/data/level0/l0_0001.sst",
                4096,
                512,
                64,
                &sample_index_entries(0),
            )
            .await
            .expect("emit");

        // Confirm a fresh store load picks up the entry.
        let store = VectorObjectEconomyDirectoryStore::new(
            &fs,
            "coll",
            "s3://bucket/coll",
            7,
            CatalogAuthorityMode::RebuildableProjection,
        );
        let dir = store.load_or_empty().await;
        assert_eq!(dir.files.len(), 1);
        assert_eq!(dir.files[0].file_id, "l0_0001");
        assert_eq!(dir.freshness_watermark_lsn, 100);
    }

    #[tokio::test]
    async fn emit_after_flush_is_idempotent_for_same_file_id() {
        let fs = InMemoryFs::default();
        let (hooks, _cache) = hooks_with_fresh_cache("coll", 7);

        // Emit twice for the same file_id (e.g., retry after transient error).
        for _ in 0..2 {
            hooks
                .emit_after_flush(
                    &fs,
                    "l0_0001",
                    "s3://bucket/coll/data/level0/l0_0001.sst",
                    4096,
                    512,
                    64,
                    &sample_index_entries(0),
                )
                .await
                .expect("emit");
        }

        let store = VectorObjectEconomyDirectoryStore::new(
            &fs,
            "coll",
            "s3://bucket/coll",
            7,
            CatalogAuthorityMode::RebuildableProjection,
        );
        let dir = store.load_or_empty().await;
        assert_eq!(dir.files.len(), 1, "same file_id must not duplicate");
    }

    #[tokio::test]
    async fn emit_after_flush_accumulates_distinct_file_ids() {
        let fs = InMemoryFs::default();
        let (hooks, _cache) = hooks_with_fresh_cache("coll", 7);

        hooks
            .emit_after_flush(
                &fs,
                "l0_0001",
                "s3://bucket/coll/data/level0/l0_0001.sst",
                4096,
                512,
                64,
                &sample_index_entries(0),
            )
            .await
            .expect("emit 1");
        hooks
            .emit_after_flush(
                &fs,
                "l0_0002",
                "s3://bucket/coll/data/level0/l0_0002.sst",
                4096,
                512,
                64,
                &sample_index_entries(10),
            )
            .await
            .expect("emit 2");

        let store = VectorObjectEconomyDirectoryStore::new(
            &fs,
            "coll",
            "s3://bucket/coll",
            7,
            CatalogAuthorityMode::RebuildableProjection,
        );
        let dir = store.load_or_empty().await;
        assert_eq!(dir.files.len(), 2);
        let ids: Vec<&str> = dir.files.iter().map(|f| f.file_id.as_str()).collect();
        assert!(ids.contains(&"l0_0001"));
        assert!(ids.contains(&"l0_0002"));
    }

    #[tokio::test]
    async fn emit_after_flush_invalidates_read_side_cache() {
        let fs = InMemoryFs::default();
        let (hooks, cache) = hooks_with_fresh_cache("coll", 7);

        // Warm the read-side cache with an empty directory (sidecar absent).
        let pre_handle = cache.handle_for("coll");
        let pre_entry = pre_handle
            .get_or_load(|| async {
                load_directory_for(
                    &fs,
                    "coll",
                    "s3://bucket/coll",
                    7,
                    CatalogAuthorityMode::RebuildableProjection,
                )
                .await
            })
            .await;
        assert_eq!(pre_entry.status, DirectoryLoadStatus::Missing);

        // Writer-side emit must invalidate the warmed handle.
        hooks
            .emit_after_flush(
                &fs,
                "l0_0001",
                "s3://bucket/coll/data/level0/l0_0001.sst",
                4096,
                512,
                64,
                &sample_index_entries(0),
            )
            .await
            .expect("emit");

        // Fresh handle after invalidation → reload returns Loaded.
        let post_handle = cache.handle_for("coll");
        assert!(
            !Arc::ptr_eq(&pre_handle, &post_handle),
            "invalidate must replace the handle"
        );
        let post_entry = post_handle
            .get_or_load(|| async {
                load_directory_for(
                    &fs,
                    "coll",
                    "s3://bucket/coll",
                    7,
                    CatalogAuthorityMode::RebuildableProjection,
                )
                .await
            })
            .await;
        assert_eq!(post_entry.status, DirectoryLoadStatus::Loaded);
        assert_eq!(post_entry.directory.files.len(), 1);
    }
}
