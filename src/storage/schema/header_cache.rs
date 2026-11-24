//! # ProximaHeaderCache - Smart I/O Layer for Rowgroup/Block Pruning
//!
//! This module provides intelligent metadata caching for sub-millisecond pruning decisions
//! BEFORE issuing any S3/cloud I/O. This is critical for reducing I/O costs and improving
//! query performance.
//!
//! ## Design Philosophy
//!
//! Arrow is for in-memory data exchange, NOT for storage scanning. We need Hadoop-style
//! rowgroup/block pruning for I/O reduction. ProximaHeaderCache caches all file metadata
//! needed to make pruning decisions without reading data.
//!
//! ## I/O Reduction Example
//!
//! ```text
//! Without ProximaHeaderCache:
//!   GET s3://bucket/file.sst  → 100 MB (full file)
//!
//! With ProximaHeaderCache:
//!   GET s3://bucket/file.sst Range: bytes=0-4096     → 4KB (header, cached)
//!   GET s3://bucket/file.sst Range: bytes=1024-2048  → 1KB (rowgroup 3 only)
//!   GET s3://bucket/file.sst Range: bytes=5120-6144  → 1KB (rowgroup 7 only)
//!   Total: 6KB (94% I/O reduction)
//! ```
//!
//! ## Smart I/O Flow
//!
//! ```text
//! 1. Query arrives with predicate
//! 2. Load header from ProximaHeaderCache (or fetch once, cache forever)
//! 3. Apply predicate to cached column_stats → skip rowgroups
//! 4. Apply Hilbert/Z-order/AdaCurve range to spatial_range → skip blocks
//! 5. Only issue S3 Range requests for matching rowgroups
//! 6. Decompress and convert to Arrow RecordBatch in memory
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use proximadb_runtime_common::cache::CacheStats;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, trace};

/// Cached file header with all metadata needed for rowgroup/block pruning.
/// Eliminates repeated S3 reads for file headers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedHeader {
    /// File path (absolute or cloud URL)
    pub path: String,
    /// Schema fingerprint for fast comparison
    pub schema_fingerprint: u64,
    /// Schema version
    pub schema_version: u32,
    /// RowGroup/Block metadata for pruning
    pub rowgroups: Vec<RowGroupMeta>,
    /// Total file size in bytes
    pub file_size: u64,
    /// Last modified timestamp (Unix millis)
    pub last_modified: i64,
    /// Header size in bytes (for cache memory tracking)
    pub header_size_bytes: usize,
    /// Format type (SST, HELIX, etc.)
    pub format_type: String,
    /// Engine-specific metadata
    pub engine_metadata: HashMap<String, String>,
}

impl CachedHeader {
    /// Create a new cached header.
    pub fn new(path: String, schema_fingerprint: u64) -> Self {
        Self {
            path,
            schema_fingerprint,
            schema_version: 0,
            rowgroups: Vec::new(),
            file_size: 0,
            last_modified: 0,
            header_size_bytes: 0,
            format_type: String::new(),
            engine_metadata: HashMap::new(),
        }
    }

    /// Get total row count across all rowgroups.
    pub fn total_rows(&self) -> i64 {
        self.rowgroups.iter().map(|rg| rg.row_count).sum()
    }

    /// Get total data size (excluding header).
    pub fn total_data_size(&self) -> u64 {
        self.rowgroups.iter().map(|rg| rg.length).sum()
    }

    /// Apply scalar predicate to prune rowgroups.
    /// Returns indices of rowgroups that may contain matching data.
    pub fn prune_by_scalar_predicate(
        &self,
        column: &str,
        predicate: &ScalarPredicate,
    ) -> Vec<usize> {
        self.rowgroups
            .iter()
            .enumerate()
            .filter(|(_, rg)| {
                if let Some(bounds) = rg.column_stats.get(column) {
                    predicate.may_match(bounds)
                } else {
                    // No stats for column, can't prune
                    true
                }
            })
            .map(|(i, _)| i)
            .collect()
    }

    /// Apply spatial predicate to prune rowgroups.
    /// Returns indices of rowgroups that may contain matching vectors.
    pub fn prune_by_spatial_range(&self, query_range: &SpatialRange) -> Vec<usize> {
        self.rowgroups
            .iter()
            .enumerate()
            .filter(|(_, rg)| {
                if let Some(ref rg_range) = rg.spatial_range {
                    rg_range.may_overlap(query_range)
                } else {
                    // No spatial range, can't prune
                    true
                }
            })
            .map(|(i, _)| i)
            .collect()
    }

    /// Apply centroid-based pruning for vector search.
    /// Returns indices of rowgroups within distance threshold of query.
    pub fn prune_by_centroid(&self, query: &[f32], max_distance: f32) -> Vec<usize> {
        self.rowgroups
            .iter()
            .enumerate()
            .filter(|(_, rg)| {
                if let Some(ref centroid) = rg.centroid {
                    let distance = Self::l2_distance(query, centroid);
                    distance <= max_distance
                } else {
                    // No centroid, can't prune
                    true
                }
            })
            .map(|(i, _)| i)
            .collect()
    }

    /// Compute L2 distance between two vectors (for centroid pruning).
    fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::MAX;
        }
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Estimate I/O bytes saved by pruning.
    pub fn estimate_io_savings(&self, matching_rowgroups: &[usize]) -> IoSavingsEstimate {
        let total_size: u64 = self.rowgroups.iter().map(|rg| rg.length).sum();
        let matching_size: u64 = matching_rowgroups
            .iter()
            .filter_map(|&i| self.rowgroups.get(i))
            .map(|rg| rg.length)
            .sum();

        IoSavingsEstimate {
            total_bytes: total_size,
            read_bytes: matching_size,
            saved_bytes: total_size.saturating_sub(matching_size),
            savings_ratio: if total_size > 0 {
                (total_size - matching_size) as f64 / total_size as f64
            } else {
                0.0
            },
            rowgroups_total: self.rowgroups.len(),
            rowgroups_read: matching_rowgroups.len(),
        }
    }
}

/// Per-rowgroup metadata for smart pruning BEFORE I/O.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RowGroupMeta {
    /// Rowgroup index
    pub index: usize,
    /// Byte offset in file
    pub offset: u64,
    /// Byte length
    pub length: u64,
    /// Row count
    pub row_count: i64,
    /// Column statistics (min/max/null_count) for scalar pruning
    pub column_stats: HashMap<String, ColumnBounds>,
    /// Bloom filter bytes (serialized, for point lookups)
    pub bloom_filter: Option<Vec<u8>>,
    /// Centroid for vector search pruning (representative point)
    pub centroid: Option<Vec<f32>>,
    /// Spatial range (HELIX Hilbert, SWIFT AdaCurve, RAPTOR Z-order)
    pub spatial_range: Option<SpatialRange>,
    /// Compression codec used for this rowgroup
    pub compression: Option<String>,
    /// Encoding info (quantization type, bits, etc.)
    pub encoding: Option<EncodingInfo>,
}

impl RowGroupMeta {
    /// Create a new rowgroup metadata entry.
    pub fn new(index: usize, offset: u64, length: u64, row_count: i64) -> Self {
        Self {
            index,
            offset,
            length,
            row_count,
            column_stats: HashMap::new(),
            bloom_filter: None,
            centroid: None,
            spatial_range: None,
            compression: None,
            encoding: None,
        }
    }

    /// Add column statistics.
    pub fn with_column_stats(mut self, column: &str, bounds: ColumnBounds) -> Self {
        self.column_stats.insert(column.to_string(), bounds);
        self
    }

    /// Add centroid for vector search.
    pub fn with_centroid(mut self, centroid: Vec<f32>) -> Self {
        self.centroid = Some(centroid);
        self
    }

    /// Add spatial range for spatial pruning.
    pub fn with_spatial_range(mut self, range: SpatialRange) -> Self {
        self.spatial_range = Some(range);
        self
    }
}

/// Column min/max bounds for scalar predicate pruning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnBounds {
    /// Minimum value (serialized)
    pub min: ColumnValue,
    /// Maximum value (serialized)
    pub max: ColumnValue,
    /// Null count
    pub null_count: i64,
    /// Distinct count (approximate)
    pub distinct_count: Option<i64>,
}

impl ColumnBounds {
    /// Create new column bounds.
    pub fn new(min: ColumnValue, max: ColumnValue) -> Self {
        Self {
            min,
            max,
            null_count: 0,
            distinct_count: None,
        }
    }
}

/// Serialized column value for min/max stats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColumnValue {
    Null,
    Bool(bool),
    Int64(i64),
    Float64(f64),
    String(String),
    Binary(Vec<u8>),
    Timestamp(i64), // Unix nanos
}

impl ColumnValue {
    /// Compare two column values (for predicate evaluation).
    pub fn compare(&self, other: &ColumnValue) -> Option<std::cmp::Ordering> {
        use ColumnValue::*;
        match (self, other) {
            (Null, Null) => Some(std::cmp::Ordering::Equal),
            (Null, _) => Some(std::cmp::Ordering::Less),
            (_, Null) => Some(std::cmp::Ordering::Greater),
            (Bool(a), Bool(b)) => Some(a.cmp(b)),
            (Int64(a), Int64(b)) => Some(a.cmp(b)),
            (Float64(a), Float64(b)) => a.partial_cmp(b),
            (String(a), String(b)) => Some(a.cmp(b)),
            (Binary(a), Binary(b)) => Some(a.cmp(b)),
            (Timestamp(a), Timestamp(b)) => Some(a.cmp(b)),
            // Cross-type comparisons: try numeric
            (Int64(a), Float64(b)) => (*a as f64).partial_cmp(b),
            (Float64(a), Int64(b)) => a.partial_cmp(&(*b as f64)),
            _ => None,
        }
    }
}

/// Scalar predicate for rowgroup pruning.
#[derive(Debug, Clone)]
pub enum ScalarPredicate {
    /// column = value
    Eq(ColumnValue),
    /// column != value
    Ne(ColumnValue),
    /// column < value
    Lt(ColumnValue),
    /// column <= value
    Le(ColumnValue),
    /// column > value
    Gt(ColumnValue),
    /// column >= value
    Ge(ColumnValue),
    /// column IN (values)
    In(Vec<ColumnValue>),
    /// column BETWEEN min AND max
    Between(ColumnValue, ColumnValue),
    /// column IS NULL
    IsNull,
    /// column IS NOT NULL
    IsNotNull,
}

impl ScalarPredicate {
    /// Check if rowgroup with given bounds may contain matching data.
    pub fn may_match(&self, bounds: &ColumnBounds) -> bool {
        use ScalarPredicate::*;
        match self {
            Eq(v) => {
                // min <= v <= max
                bounds
                    .min
                    .compare(v)
                    .is_none_or(|o| o != std::cmp::Ordering::Greater)
                    && bounds
                        .max
                        .compare(v)
                        .is_none_or(|o| o != std::cmp::Ordering::Less)
            }
            Ne(_) => {
                // Can only skip if min == max == v
                true
            }
            Lt(v) => {
                // min < v
                bounds
                    .min
                    .compare(v)
                    .is_none_or(|o| o == std::cmp::Ordering::Less)
            }
            Le(v) => {
                // min <= v
                bounds
                    .min
                    .compare(v)
                    .is_none_or(|o| o != std::cmp::Ordering::Greater)
            }
            Gt(v) => {
                // max > v
                bounds
                    .max
                    .compare(v)
                    .is_none_or(|o| o == std::cmp::Ordering::Greater)
            }
            Ge(v) => {
                // max >= v
                bounds
                    .max
                    .compare(v)
                    .is_none_or(|o| o != std::cmp::Ordering::Less)
            }
            In(values) => {
                // Any value in range
                values.iter().any(|v| {
                    bounds
                        .min
                        .compare(v)
                        .is_none_or(|o| o != std::cmp::Ordering::Greater)
                        && bounds
                            .max
                            .compare(v)
                            .is_none_or(|o| o != std::cmp::Ordering::Less)
                })
            }
            Between(min_v, max_v) => {
                // ranges overlap
                bounds
                    .min
                    .compare(max_v)
                    .is_none_or(|o| o != std::cmp::Ordering::Greater)
                    && bounds
                        .max
                        .compare(min_v)
                        .is_none_or(|o| o != std::cmp::Ordering::Less)
            }
            IsNull => bounds.null_count > 0,
            IsNotNull => bounds.null_count < 1, // Conservative
        }
    }
}

/// Spatial range for block/rowgroup pruning.
/// Each engine uses different spatial indexing schemes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpatialRange {
    /// HELIX: Hilbert curve range (preserves locality in high-dimensional space)
    Hilbert {
        min: u64,
        max: u64,
        order: u8, // Hilbert curve order (determines precision)
    },
    /// SWIFT: Adaptive curve code range (learned ordering)
    AdaCurve {
        min: u64,
        max: u64,
        model_version: u32, // Version of the adaptive model
    },
    /// RAPTOR: Z-order curve range (Morton codes)
    ZOrder { min: u64, max: u64 },
    /// NOVA: Zone map bounds (per-dimension min/max)
    ZoneMap {
        bounds: HashMap<u32, (f32, f32)>, // dimension -> (min, max)
    },
    /// SST: Block ID range (sequential blocks)
    BlockRange { start_block: u32, end_block: u32 },
    /// Parquet rowgroup (native Parquet stats, NOVA)
    ParquetRowGroup { row_group_index: usize },
}

impl SpatialRange {
    /// Check if two spatial ranges may overlap.
    pub fn may_overlap(&self, other: &SpatialRange) -> bool {
        match (self, other) {
            // Hilbert: 1D range intersection
            (
                SpatialRange::Hilbert {
                    min: a_min,
                    max: a_max,
                    ..
                },
                SpatialRange::Hilbert {
                    min: b_min,
                    max: b_max,
                    ..
                },
            ) => a_min <= b_max && b_min <= a_max,
            // AdaCurve: 1D range intersection
            (
                SpatialRange::AdaCurve {
                    min: a_min,
                    max: a_max,
                    ..
                },
                SpatialRange::AdaCurve {
                    min: b_min,
                    max: b_max,
                    ..
                },
            ) => a_min <= b_max && b_min <= a_max,
            // Z-order: 1D range intersection
            (
                SpatialRange::ZOrder {
                    min: a_min,
                    max: a_max,
                },
                SpatialRange::ZOrder {
                    min: b_min,
                    max: b_max,
                },
            ) => a_min <= b_max && b_min <= a_max,
            // ZoneMap: per-dimension overlap check
            (
                SpatialRange::ZoneMap { bounds: a_bounds },
                SpatialRange::ZoneMap { bounds: b_bounds },
            ) => {
                // Must overlap in all shared dimensions
                for (dim, (a_min, a_max)) in a_bounds {
                    if let Some((b_min, b_max)) = b_bounds.get(dim)
                        && (a_min > b_max || b_min > a_max)
                    {
                        return false;
                    }
                }
                true
            }
            // BlockRange: simple range intersection
            (
                SpatialRange::BlockRange {
                    start_block: a_start,
                    end_block: a_end,
                },
                SpatialRange::BlockRange {
                    start_block: b_start,
                    end_block: b_end,
                },
            ) => a_start <= b_end && b_start <= a_end,
            // Different types: can't prune, assume overlap
            _ => true,
        }
    }

    /// Create Hilbert range from min/max codes.
    pub fn hilbert(min: u64, max: u64, order: u8) -> Self {
        SpatialRange::Hilbert { min, max, order }
    }

    /// Create Z-order range from min/max Morton codes.
    pub fn z_order(min: u64, max: u64) -> Self {
        SpatialRange::ZOrder { min, max }
    }

    /// Create zone map from dimension bounds.
    pub fn zone_map(bounds: HashMap<u32, (f32, f32)>) -> Self {
        SpatialRange::ZoneMap { bounds }
    }
}

/// Encoding information for a rowgroup (quantization, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodingInfo {
    /// Quantization type (None, Binary, INT8, PQ, etc.)
    pub quantization_type: String,
    /// Bits per component (for quantized data)
    pub bits: Option<u8>,
    /// Subquantizer count (for PQ)
    pub pq_subquantizers: Option<u8>,
    /// Codebook size (for PQ)
    pub pq_codebook_size: Option<u16>,
}

/// I/O savings estimate from pruning.
#[derive(Debug, Clone)]
pub struct IoSavingsEstimate {
    /// Total bytes in file
    pub total_bytes: u64,
    /// Bytes that will be read
    pub read_bytes: u64,
    /// Bytes saved by pruning
    pub saved_bytes: u64,
    /// Savings ratio (0.0 - 1.0)
    pub savings_ratio: f64,
    /// Total rowgroups in file
    pub rowgroups_total: usize,
    /// Rowgroups that will be read
    pub rowgroups_read: usize,
}

impl IoSavingsEstimate {
    /// Format savings as human-readable string.
    pub fn format(&self) -> String {
        format!(
            "I/O: {} of {} ({:.1}% saved, {}/{} rowgroups)",
            Self::format_bytes(self.read_bytes),
            Self::format_bytes(self.total_bytes),
            self.savings_ratio * 100.0,
            self.rowgroups_read,
            self.rowgroups_total
        )
    }

    fn format_bytes(bytes: u64) -> String {
        if bytes >= 1_073_741_824 {
            format!("{:.1}GB", bytes as f64 / 1_073_741_824.0)
        } else if bytes >= 1_048_576 {
            format!("{:.1}MB", bytes as f64 / 1_048_576.0)
        } else if bytes >= 1024 {
            format!("{:.1}KB", bytes as f64 / 1024.0)
        } else {
            format!("{}B", bytes)
        }
    }
}

/// LRU entry wrapper with access tracking.
struct CacheEntry {
    header: CachedHeader,
    last_access: Instant,
    #[allow(dead_code)]
    access_count: u64,
}

/// ProximaHeaderCache - LRU cache for file headers.
///
/// Caches all metadata needed for rowgroup/block pruning decisions.
/// Eliminates repeated S3/cloud header reads for hot files.
pub struct ProximaHeaderCache {
    /// LRU cache: file_path -> CacheEntry
    cache: RwLock<HashMap<String, CacheEntry>>,
    /// Max cache size in bytes
    max_size_bytes: usize,
    /// Current cache size in bytes
    current_size_bytes: RwLock<usize>,
    /// Cache hit counter
    hits: RwLock<u64>,
    /// Cache miss counter
    misses: RwLock<u64>,
    /// TTL for cache entries (None = forever)
    ttl: Option<Duration>,
}

impl ProximaHeaderCache {
    /// Create a new header cache with specified max size.
    pub fn new(max_size_bytes: usize) -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            max_size_bytes,
            current_size_bytes: RwLock::new(0),
            hits: RwLock::new(0),
            misses: RwLock::new(0),
            ttl: None,
        }
    }

    /// Create a header cache with TTL.
    pub fn with_ttl(max_size_bytes: usize, ttl: Duration) -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            max_size_bytes,
            current_size_bytes: RwLock::new(0),
            hits: RwLock::new(0),
            misses: RwLock::new(0),
            ttl: Some(ttl),
        }
    }

    /// Get cached header for a file path.
    pub fn get(&self, path: &str) -> Option<CachedHeader> {
        let now = Instant::now();

        // Check if entry exists and is valid
        {
            let cache = self.cache.read();
            if let Some(entry) = cache.get(path) {
                // Check TTL
                if let Some(ttl) = self.ttl
                    && now.duration_since(entry.last_access) > ttl
                {
                    drop(cache);
                    self.remove(path);
                    *self.misses.write() += 1;
                    return None;
                }

                *self.hits.write() += 1;
                trace!("Header cache hit: {}", path);
                return Some(entry.header.clone());
            }
        }

        *self.misses.write() += 1;
        trace!("Header cache miss: {}", path);
        None
    }

    /// Insert a header into the cache.
    pub fn insert(&self, header: CachedHeader) {
        let path = header.path.clone();
        let size = header.header_size_bytes;

        // Evict if necessary
        self.ensure_capacity(size);

        let entry = CacheEntry {
            header,
            last_access: Instant::now(),
            access_count: 1,
        };

        {
            let mut cache = self.cache.write();
            // Update size tracking
            if let Some(old) = cache.get(&path) {
                *self.current_size_bytes.write() -= old.header.header_size_bytes;
            }
            cache.insert(path.clone(), entry);
            *self.current_size_bytes.write() += size;
        }

        debug!("Cached header for {} ({} bytes)", path, size);
    }

    /// Remove a header from the cache.
    pub fn remove(&self, path: &str) -> Option<CachedHeader> {
        let mut cache = self.cache.write();
        if let Some(entry) = cache.remove(path) {
            *self.current_size_bytes.write() -= entry.header.header_size_bytes;
            debug!("Evicted header for {}", path);
            return Some(entry.header);
        }
        None
    }

    /// Invalidate all cached headers for a directory prefix.
    pub fn invalidate_prefix(&self, prefix: &str) {
        let mut cache = self.cache.write();
        let to_remove: Vec<String> = cache
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();

        let mut size_freed = 0;
        for path in to_remove {
            if let Some(entry) = cache.remove(&path) {
                size_freed += entry.header.header_size_bytes;
            }
        }

        *self.current_size_bytes.write() -= size_freed;
        info!(
            "Invalidated {} bytes of headers with prefix {}",
            size_freed, prefix
        );
    }

    /// Ensure capacity for new entry by evicting LRU entries.
    fn ensure_capacity(&self, needed_bytes: usize) {
        let mut cache = self.cache.write();
        let current = *self.current_size_bytes.read();

        if current + needed_bytes <= self.max_size_bytes {
            return;
        }

        // Sort by last access time (oldest first)
        let mut entries: Vec<_> = cache
            .iter()
            .map(|(k, v)| (k.clone(), v.last_access))
            .collect();
        entries.sort_by_key(|(_, t)| *t);

        let mut freed = 0;
        let target_free = (current + needed_bytes).saturating_sub(self.max_size_bytes);

        for (path, _) in entries {
            if freed >= target_free {
                break;
            }
            if let Some(entry) = cache.remove(&path) {
                freed += entry.header.header_size_bytes;
                debug!("Evicted LRU header: {}", path);
            }
        }

        *self.current_size_bytes.write() -= freed;
    }

    /// Get cache statistics.
    pub fn stats(&self) -> HeaderCacheStats {
        let cache = self.cache.read();
        let hits = *self.hits.read();
        let misses = *self.misses.read();
        let inner = CacheStats {
            hits,
            misses,
            gets: hits + misses,
            size: cache.len(),
            ..CacheStats::default()
        };
        HeaderCacheStats {
            // Canonical CacheStats::hit_ratio() returns 0–100 (percent); divide for ratio.
            hit_ratio: inner.hit_ratio() / 100.0,
            inner,
            size_bytes: *self.current_size_bytes.read(),
            max_size_bytes: self.max_size_bytes,
        }
    }

    /// Clear all cached headers.
    pub fn clear(&self) {
        let mut cache = self.cache.write();
        cache.clear();
        *self.current_size_bytes.write() = 0;
        info!("Cleared header cache");
    }
}

impl Default for ProximaHeaderCache {
    fn default() -> Self {
        // Default: 256MB cache
        Self::new(256 * 1024 * 1024)
    }
}

/// Cache statistics.
///
/// Embeds the canonical `proximadb_runtime_common::cache::CacheStats` for
/// hits/misses/size counters. Engine-specific fields kept on the wrapper
/// track byte-granularity capacity that the canonical (entry-count
/// model) doesn't represent.
#[derive(Debug, Clone, Default)]
pub struct HeaderCacheStats {
    /// Canonical cache counters (hits, misses, evictions, gets, puts, size, ...).
    pub inner: CacheStats,
    /// Current cache size in bytes (byte-granularity; canonical tracks entries).
    pub size_bytes: usize,
    /// Maximum cache size in bytes (byte-granularity capacity).
    pub max_size_bytes: usize,
    /// Hit ratio (0.0 - 1.0). Derived in `stats()` from `inner.hit_ratio()`.
    pub hit_ratio: f64,
}

impl HeaderCacheStats {
    /// Number of cached entries (delegates to canonical `inner.size`).
    pub fn entries(&self) -> usize {
        self.inner.size
    }

    /// Number of cache hits (delegates to canonical `inner.hits`).
    pub fn hits(&self) -> u64 {
        self.inner.hits
    }

    /// Number of cache misses (delegates to canonical `inner.misses`).
    pub fn misses(&self) -> u64 {
        self.inner.misses
    }
}

impl std::fmt::Display for HeaderCacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HeaderCache: {} entries, {:.1}MB/{:.1}MB, {:.1}% hit rate ({} hits, {} misses)",
            self.inner.size,
            self.size_bytes as f64 / 1_048_576.0,
            self.max_size_bytes as f64 / 1_048_576.0,
            self.hit_ratio * 100.0,
            self.inner.hits,
            self.inner.misses
        )
    }
}

// ============================================================================
// Global Header Cache
// ============================================================================

/// Global header cache singleton.
static GLOBAL_HEADER_CACHE: std::sync::OnceLock<Arc<ProximaHeaderCache>> =
    std::sync::OnceLock::new();

/// Get the global header cache.
pub fn global_header_cache() -> Arc<ProximaHeaderCache> {
    GLOBAL_HEADER_CACHE
        .get_or_init(|| Arc::new(ProximaHeaderCache::default()))
        .clone()
}

/// Initialize the global header cache with custom configuration.
pub fn init_global_header_cache(cache: Arc<ProximaHeaderCache>) -> Result<(), String> {
    GLOBAL_HEADER_CACHE
        .set(cache)
        .map_err(|_| "Global header cache already initialized".to_string())
}

// ============================================================================
// Header Loader Trait
// ============================================================================

/// Trait for loading file headers from storage.
/// Implemented by each storage engine to extract metadata.
#[async_trait::async_trait]
pub trait HeaderLoader: Send + Sync {
    /// Load header from file path.
    async fn load_header(&self, path: &str) -> anyhow::Result<CachedHeader>;

    /// Check if file format is supported.
    fn supports_format(&self, format_type: &str) -> bool;
}

/// Cached header loader that uses ProximaHeaderCache.
pub struct CachingHeaderLoader {
    cache: Arc<ProximaHeaderCache>,
    loaders: Vec<Arc<dyn HeaderLoader>>,
}

impl CachingHeaderLoader {
    /// Create a new caching header loader.
    pub fn new(cache: Arc<ProximaHeaderCache>) -> Self {
        Self {
            cache,
            loaders: Vec::new(),
        }
    }

    /// Register a header loader for a format.
    pub fn register_loader(&mut self, loader: Arc<dyn HeaderLoader>) {
        self.loaders.push(loader);
    }

    /// Load header with caching.
    pub async fn load(
        &self,
        path: &str,
        format_hint: Option<&str>,
    ) -> anyhow::Result<CachedHeader> {
        // Check cache first
        if let Some(header) = self.cache.get(path) {
            return Ok(header);
        }

        // Find appropriate loader
        let loader = self
            .loaders
            .iter()
            .find(|l| {
                if let Some(fmt) = format_hint {
                    l.supports_format(fmt)
                } else {
                    true // Try all loaders
                }
            })
            .ok_or_else(|| anyhow::anyhow!("No loader found for format: {:?}", format_hint))?;

        // Load from storage
        let header = loader.load_header(path).await?;

        // Cache it
        self.cache.insert(header.clone());

        Ok(header)
    }
}

// ============================================================================
// Enhanced ProximaHeaderCache with CentroidTree Integration
// ============================================================================

use super::centroid_tree::CentroidTree;
use super::pruning_strategies::{
    PruningResult, ScalarPruner, SpatialPruner, SpatialRangeType, VectorPruner,
};

/// Enhanced header cache entry with CentroidTree for vector pruning.
pub struct EnhancedCachedHeader {
    /// Base cached header.
    pub header: CachedHeader,

    /// CentroidTree for O(log n) vector pruning.
    pub centroid_tree: Option<CentroidTree>,

    /// Whether pruning indexes are built.
    pub indexes_built: bool,
}

impl EnhancedCachedHeader {
    /// Create from base header, optionally building centroid tree.
    pub fn from_header(header: CachedHeader, build_centroid_tree: bool) -> Self {
        let centroid_tree = if build_centroid_tree {
            Self::build_centroid_tree(&header)
        } else {
            None
        };

        Self {
            header,
            centroid_tree,
            indexes_built: build_centroid_tree,
        }
    }

    /// Build CentroidTree from rowgroup centroids.
    fn build_centroid_tree(header: &CachedHeader) -> Option<CentroidTree> {
        let centroids: Vec<Vec<f32>> = header
            .rowgroups
            .iter()
            .filter_map(|rg| rg.centroid.clone())
            .collect();

        if centroids.is_empty() {
            return None;
        }

        match CentroidTree::build(&centroids, 16) {
            Ok(tree) => Some(tree),
            Err(e) => {
                debug!("Failed to build CentroidTree: {:?}", e);
                None
            }
        }
    }

    /// Prune rowgroups by vector distance using CentroidTree if available.
    pub fn prune_by_vector(&self, query: &[f32], max_distance: f32) -> PruningResult {
        match &self.centroid_tree {
            Some(tree) => tree.prune_by_vector(query, max_distance),
            None => {
                // Fall back to linear scan on centroids
                let start = std::time::Instant::now();
                let matching = self.header.prune_by_centroid(query, max_distance);
                let elapsed_ns = start.elapsed().as_nanos() as u64;

                PruningResult::with_indices(
                    matching,
                    self.header.rowgroups.len(),
                    "linear_centroid_scan",
                    elapsed_ns,
                )
            }
        }
    }

    /// Prune by scalar predicate.
    pub fn prune_by_scalar(&self, column: &str, predicate: &ScalarPredicate) -> PruningResult {
        let start = std::time::Instant::now();
        let matching = self.header.prune_by_scalar_predicate(column, predicate);
        let elapsed_ns = start.elapsed().as_nanos() as u64;

        PruningResult::with_indices(
            matching,
            self.header.rowgroups.len(),
            "scalar_bounds",
            elapsed_ns,
        )
    }

    /// Prune by spatial range.
    pub fn prune_by_spatial(&self, range: &SpatialRange) -> PruningResult {
        let start = std::time::Instant::now();
        let matching = self.header.prune_by_spatial_range(range);
        let elapsed_ns = start.elapsed().as_nanos() as u64;

        PruningResult::with_indices(
            matching,
            self.header.rowgroups.len(),
            "spatial_range",
            elapsed_ns,
        )
    }

    /// Get spatial range type based on format.
    pub fn spatial_type(&self) -> SpatialRangeType {
        match self.header.format_type.to_lowercase().as_str() {
            "helix" => SpatialRangeType::Hilbert,
            "raptor" => SpatialRangeType::ZOrder,
            "swift" => SpatialRangeType::AdaCurve,
            "nova" => SpatialRangeType::ZoneMap,
            "sst" => SpatialRangeType::BlockRange,
            "viper" | "parquet" => SpatialRangeType::ParquetRowGroup,
            _ => SpatialRangeType::BlockRange,
        }
    }
}

// ============================================================================
// VectorPruner Implementation for EnhancedCachedHeader
// ============================================================================

impl VectorPruner for EnhancedCachedHeader {
    fn prune_by_vector(&self, query: &[f32], max_distance: f32) -> PruningResult {
        EnhancedCachedHeader::prune_by_vector(self, query, max_distance)
    }

    fn prune_quantized(&self, query: &[f32], max_distance: f32) -> PruningResult {
        match &self.centroid_tree {
            Some(tree) => tree.prune_quantized(query, max_distance),
            None => self.prune_by_vector(query, max_distance),
        }
    }

    fn dimension(&self) -> usize {
        self.header
            .rowgroups
            .first()
            .and_then(|rg| rg.centroid.as_ref())
            .map_or(0, |c| c.len())
    }

    fn num_entries(&self) -> usize {
        self.header.rowgroups.len()
    }
}

// ============================================================================
// ScalarPruner Implementation for EnhancedCachedHeader
// ============================================================================

impl ScalarPruner for EnhancedCachedHeader {
    fn prune_by_predicate(&self, column: &str, predicate: &ScalarPredicate) -> PruningResult {
        self.prune_by_scalar(column, predicate)
    }

    fn num_rowgroups(&self) -> usize {
        self.header.rowgroups.len()
    }

    fn available_columns(&self) -> Vec<String> {
        let mut columns = std::collections::HashSet::new();

        for rg in &self.header.rowgroups {
            for col_name in rg.column_stats.keys() {
                columns.insert(col_name.clone());
            }
        }

        columns.into_iter().collect()
    }
}

// ============================================================================
// SpatialPruner Implementation for EnhancedCachedHeader
// ============================================================================

impl SpatialPruner for EnhancedCachedHeader {
    fn prune_by_spatial_range(&self, range: &SpatialRange) -> PruningResult {
        self.prune_by_spatial(range)
    }

    fn spatial_type(&self) -> SpatialRangeType {
        EnhancedCachedHeader::spatial_type(self)
    }

    fn num_rowgroups(&self) -> usize {
        self.header.rowgroups.len()
    }
}

// ============================================================================
// CachedHeader with Pruner Trait Integration
// ============================================================================

impl CachedHeader {
    /// Create an EnhancedCachedHeader with CentroidTree.
    pub fn with_centroid_tree(self) -> EnhancedCachedHeader {
        EnhancedCachedHeader::from_header(self, true)
    }

    /// Create a simple EnhancedCachedHeader without building indexes.
    pub fn as_enhanced(self) -> EnhancedCachedHeader {
        EnhancedCachedHeader::from_header(self, false)
    }

    /// Build and return a CentroidTree from rowgroup centroids.
    pub fn build_centroid_tree(&self) -> Option<CentroidTree> {
        let centroids: Vec<Vec<f32>> = self
            .rowgroups
            .iter()
            .filter_map(|rg| rg.centroid.clone())
            .collect();

        if centroids.is_empty() {
            return None;
        }

        CentroidTree::build(&centroids, 16).ok()
    }
}

// ============================================================================
// ProximaHeaderCache with Enhanced Caching
// ============================================================================

impl ProximaHeaderCache {
    /// Get enhanced header with CentroidTree built.
    pub fn get_enhanced(&self, path: &str) -> Option<EnhancedCachedHeader> {
        self.get(path).map(|header| header.with_centroid_tree())
    }

    /// Insert and return enhanced header.
    pub fn insert_enhanced(&self, header: CachedHeader) -> EnhancedCachedHeader {
        let enhanced = header.clone().with_centroid_tree();
        self.insert(header);
        enhanced
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic() {
        let cache = ProximaHeaderCache::new(1024 * 1024); // 1MB

        let mut header = CachedHeader::new("/data/test.sst".to_string(), 12345);
        header.header_size_bytes = 100;

        cache.insert(header);

        let retrieved = cache.get("/data/test.sst").unwrap();
        assert_eq!(retrieved.schema_fingerprint, 12345);

        let stats = cache.stats();
        assert_eq!(stats.entries(), 1);
        assert_eq!(stats.hits(), 1);
        assert_eq!(stats.misses(), 0);
    }

    #[test]
    fn test_cache_eviction() {
        let cache = ProximaHeaderCache::new(500); // 500 bytes

        // Insert entries that exceed capacity
        for i in 0..10 {
            let mut header = CachedHeader::new(format!("/data/test_{}.sst", i), i as u64);
            header.header_size_bytes = 100;
            cache.insert(header);
        }

        let stats = cache.stats();
        // Should have evicted some entries to stay under limit
        assert!(stats.size_bytes <= 500);
    }

    #[test]
    fn test_scalar_predicate_pruning() {
        let mut header = CachedHeader::new("/data/test.sst".to_string(), 12345);

        // Add rowgroup with column stats
        let rg = RowGroupMeta::new(0, 0, 1000, 100).with_column_stats(
            "age",
            ColumnBounds::new(ColumnValue::Int64(18), ColumnValue::Int64(65)),
        );
        header.rowgroups.push(rg);

        // Query: age > 70 (should prune this rowgroup)
        let matching =
            header.prune_by_scalar_predicate("age", &ScalarPredicate::Gt(ColumnValue::Int64(70)));
        assert!(matching.is_empty());

        // Query: age > 30 (should NOT prune)
        let matching =
            header.prune_by_scalar_predicate("age", &ScalarPredicate::Gt(ColumnValue::Int64(30)));
        assert_eq!(matching.len(), 1);
    }

    #[test]
    fn test_spatial_range_overlap() {
        let range1 = SpatialRange::hilbert(100, 200, 8);
        let range2 = SpatialRange::hilbert(150, 250, 8);
        let range3 = SpatialRange::hilbert(300, 400, 8);

        assert!(range1.may_overlap(&range2)); // Overlapping
        assert!(!range1.may_overlap(&range3)); // Non-overlapping
    }

    #[test]
    fn test_io_savings_estimate() {
        let mut header = CachedHeader::new("/data/test.sst".to_string(), 12345);

        // Add 5 rowgroups of 1MB each
        for i in 0..5 {
            let rg = RowGroupMeta::new(i, i as u64 * 1_048_576, 1_048_576, 10000);
            header.rowgroups.push(rg);
        }

        // Read only 2 of 5 rowgroups
        let matching = vec![1, 3];
        let savings = header.estimate_io_savings(&matching);

        assert_eq!(savings.total_bytes, 5 * 1_048_576);
        assert_eq!(savings.read_bytes, 2 * 1_048_576);
        assert_eq!(savings.saved_bytes, 3 * 1_048_576);
        assert!((savings.savings_ratio - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_centroid_pruning() {
        let mut header = CachedHeader::new("/data/test.sst".to_string(), 12345);

        // Rowgroup with centroid at [1.0, 1.0, 1.0]
        let rg = RowGroupMeta::new(0, 0, 1000, 100).with_centroid(vec![1.0, 1.0, 1.0]);
        header.rowgroups.push(rg);

        // Query near centroid (should match)
        let matching = header.prune_by_centroid(&[1.1, 1.1, 1.1], 1.0);
        assert_eq!(matching.len(), 1);

        // Query far from centroid (should prune)
        let matching = header.prune_by_centroid(&[10.0, 10.0, 10.0], 1.0);
        assert!(matching.is_empty());
    }
}
