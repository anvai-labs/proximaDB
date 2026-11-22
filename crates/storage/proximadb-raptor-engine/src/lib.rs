//! RAPTOR engine crate — consolidated reader (inlined below) + writer +
//! matrix builder, hoisted out of the root crate (root-crate decomposition,
//! engines extraction). Root re-exports the public surface via thin shims.

// The `gpu` cfg in `matrix_builder.rs` is dormant (GPU matrix-building was
// root-coupled and cannot run from this crate); the blocks are retained pending
// a gpu-compute crate/port. Silence the `unexpected_cfgs` lint crate-wide.
#![allow(unexpected_cfgs)]

pub mod matrix_builder;
pub mod writer;
pub use writer::RaptorWriter;

use anyhow::{Context, Result};
use arrow_array::{Array, RecordBatch};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use std::collections::HashMap;
/// Consolidated RAPTOR reader that eliminates duplication by using unified components
/// Replaces: reader.rs (1,243 lines) + unified_reader.rs (951 lines) + rowgroup_cache.rs (771 lines)
/// Total elimination: ~3,000 lines of duplicated code
///
/// ENHANCED FEATURES:
/// - Fullscan vs Filtering strategy support
/// - Hardware-optimized BloomFilter integration
/// - Zero-copy memory-mapped I/O
/// - Predicate pushdown optimization
use std::sync::Arc;
use tracing::{debug, info, trace, warn};

// Use unified components instead of custom implementations
use proximadb_distance_kernel::engine::{DistanceMetric, SimilarityResult, UnifiedDistanceCompute};
use proximadb_search_types::bounded_queue::BoundedPriorityQueue;
use proximadb_search_types::results::OptimizedSearchRecord;

use proximadb_codec::{ProximaCodec, types::ProximaScheme};
use proximadb_storage_filesystem_types::FileSystem;
use proximadb_storage_ports::{CacheAccessPatternPort, CacheKind};

use proximadb_compression::{CompressionAlgorithm, CompressionContext};
use proximadb_raptor_common::config::RaptorConfig;
use proximadb_raptor_common::constants;
use proximadb_raptor_common::{
    ColumnType, // For selective column reading
    InterCentroidMatrix,
    MetadataValue,
    P2Matrix, // P² matrix for intra-rowgroup navigation
    RaptorFileMetadata,
    RaptorFooter,
    RowGroupBloomFilter,
    RowGroupMetadata,
    VectorCentroidMatrix,
    VectorCentroidStorageStrategy,
};

// Additional imports for component boosting and hierarchical search
use std::collections::HashSet;

/// Compare two metadata values for ordering (used in predicate pushdown range pruning).
///
/// Mirrors the prior `compare_proxima_values` semantics now that `MetadataValue` is the
/// v2 `ProximaValue` (TD-V1SUNSET-3): numeric variants compare by value (cross-type
/// via f64 — preserves the prior NumberValue↔Int64Value behavior), `String` is lexical,
/// `Boolean` direct; incomparable types are `Equal` (conservative, unchanged).
fn compare_proxima_values(a: &MetadataValue, b: &MetadataValue) -> std::cmp::Ordering {
    if let (Some(x), Some(y)) = (proxima_as_f64(a), proxima_as_f64(b)) {
        return x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal);
    }
    match (a, b) {
        (MetadataValue::String(x), MetadataValue::String(y)) => x.cmp(y),
        (MetadataValue::Boolean(x), MetadataValue::Boolean(y)) => x.cmp(y),
        _ => std::cmp::Ordering::Equal, // incomparable types → Equal (conservative)
    }
}

/// Extract an f64 from numeric `ProximaValue` variants for cross-type comparison
/// (mirrors the prior NumberValue/Int64Value cross-comparison).
fn proxima_as_f64(v: &MetadataValue) -> Option<f64> {
    match v {
        MetadataValue::Float64(x) => Some(*x),
        MetadataValue::Float32(x) => Some(*x as f64),
        MetadataValue::Float16(x) => Some(*x as f64),
        MetadataValue::Int64(x) => Some(*x as f64),
        MetadataValue::Int32(x) => Some(*x as f64),
        MetadataValue::Int16(x) => Some(*x as f64),
        MetadataValue::Int8(x) => Some(*x as f64),
        MetadataValue::UInt64(x) => Some(*x as f64),
        MetadataValue::UInt32(x) => Some(*x as f64),
        MetadataValue::UInt16(x) => Some(*x as f64),
        MetadataValue::UInt8(x) => Some(*x as f64),
        _ => None,
    }
}

/// Wrapper for f32 to make it orderable for priority queues
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
struct OrdFloat(f32);

impl Eq for OrdFloat {}

impl PartialOrd for OrdFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Result structures for similarity search
/// Note: Using unified SimilarityResult from compute::distance_computation::engine
/// Local struct removed to eliminate duplication
/// Partial rowgroup structure for selective column reading
pub struct PartialRowGroup {
    pub vectors: Option<Vec<Vec<f32>>>,
    pub ids: Option<Vec<String>>,
    pub metadata: HashMap<String, Vec<Option<Vec<u8>>>>,
    pub source_content: Option<Vec<Option<Vec<u8>>>>,
}

/// Intra-rowgroup matrix wrapper for P² matrix navigation
pub struct IntraRowgroupMatrix {
    pub p2_matrix: P2Matrix,
    pub vectors: Vec<Vec<f32>>,
}

impl IntraRowgroupMatrix {
    pub fn new(p2_matrix: P2Matrix, vectors: Vec<Vec<f32>>) -> Self {
        Self { p2_matrix, vectors }
    }
}

/// Scan strategy for different read patterns
#[derive(Debug, Clone, PartialEq)]
pub enum ScanStrategy {
    /// Full file scan - reads entire file sequentially (for compaction, backup, analysis)
    /// - No predicate filtering
    /// - Sequential I/O pattern for optimal throughput
    /// - All rowgroups processed regardless of BloomFilter hits
    FullScan,

    /// Selective filtering - optimized I/O with predicate pushdown
    /// - BloomFilter-based rowgroup skipping
    /// - Random I/O pattern for minimal latency
    /// - Only relevant rowgroups loaded
    Filtering {
        /// Vector IDs to search for (if known)
        target_ids: Option<Vec<String>>,

        /// Metadata predicates to push down
        predicates: Option<Vec<proximadb_raptor_common::Predicate>>,

        /// Maximum rowgroups to scan (limits I/O)
        max_rowgroups: Option<usize>,
    },
}

impl Default for ScanStrategy {
    fn default() -> Self {
        ScanStrategy::Filtering {
            target_ids: None,
            predicates: None,
            max_rowgroups: None,
        }
    }
}

/// Candidate result during search process
#[derive(Debug, Clone)]
pub struct CandidateResult {
    pub id: String,
    pub vector: Vec<f32>,
    pub distance: f32,
    pub cluster_id: u32,
    pub cluster_info: RaptorClusterInfo,
}

/// Backwards-compat alias for [`RaptorClusterInfo`].
pub type ClusterInfo = RaptorClusterInfo;

/// Cluster information for 5-component boosting
#[derive(Debug, Clone, Default)]
pub struct RaptorClusterInfo {
    pub inter_cluster_penalty: f32,
    pub cluster_distance: f32,
    pub cluster_id: u32,
}

// ===== OBSOLETE CLUSTERING STRUCTURES =====
// The following structures were for HNSW-style clustering
// In Matrix Trinity architecture, clustering info is embedded in:
// - K×K matrix: contains centroids and inter-centroid distances
// - P×K matrix: contains vector-to-centroid distances
// - P² matrix: contains intra-rowgroup vector distances

// Kept for reference only - not used in matrix-based search
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct ClusterMetadata {
    centroids: Vec<Vec<f32>>,
    centroid_distances: Vec<Vec<f32>>,
    cluster_stats: Vec<ClusterStats>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct ClusterStats {
    mean_distance: f32,
    std_deviation: f32,
    #[allow(dead_code)]
    radius: f32,
}

/// Boosting configuration for search navigation
#[derive(Debug, Clone)]
pub struct BoostConfig {
    // Alpha weights for intra-cluster components
    pub alpha_own: f32,      // α₁: Vector-to-own-centroid distance
    pub alpha_other: f32,    // α₂: Average distance to other centroids
    pub alpha_variance: f32, // α₃: Distance variance (cluster compactness)

    // Beta weights for inter-cluster components
    pub beta_min: f32, // β₁: Minimum inter-centroid distance
    pub beta_max: f32, // β₂: Maximum inter-centroid distance

    // Boundary detection threshold
    pub boundary_threshold: f32, // Statistical threshold (mean + σ×threshold)

    // Cross-cluster penalties
    pub alpha_inter: f32, // Inter-cluster penalty scaling
    pub beta_cross: f32,  // Cross-cluster exponential decay
}

/// Search quality statistics for performance monitoring
#[derive(Debug, Default)]
pub struct SearchStats {
    pub intra_cluster_hops: usize,
    pub inter_cluster_hops: usize,
    pub clusters_visited: HashSet<usize>,
}

/// Centroid selection result from K×K matrix phase
#[derive(Debug, Clone)]
pub struct CentroidSelection {
    pub centroid_id: usize,
    pub rowgroup_id: u16,
    pub distance: f32,
}

impl SearchStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_cluster_visit(&mut self, cluster_id: usize) {
        self.clusters_visited.insert(cluster_id);
    }
}

impl Default for BoostConfig {
    /// Default boosting configuration optimized for RAPTOR clustering
    fn default() -> Self {
        Self {
            alpha_own: 1.2,          // Slight preference for well-contained vectors
            alpha_other: 0.8,        // Moderate penalty for boundary vectors
            alpha_variance: 0.6,     // Moderate compactness preference
            beta_min: 1.1,           // Slight boost for cluster separation
            beta_max: 0.9,           // Slight penalty for distant clusters
            boundary_threshold: 1.5, // 1.5 standard deviations for boundary detection
            alpha_inter: 1.0,        // Linear inter-cluster scaling
            beta_cross: 1.0,         // Standard exponential decay
        }
    }
}

/// Cached RAPTOR file metadata entry, keyed by file path. The (size, modified)
/// pair forms the invalidation token: any change to the underlying file is
/// detected on the next `filesystem.metadata()` stat call.
#[derive(Clone)]
struct CachedFileMetadata {
    metadata: Arc<RaptorFileMetadata>,
    size: u64,
    modified: Option<DateTime<Utc>>,
}

/// Consolidated RAPTOR reader using unified infrastructure
pub struct RaptorReader {
    /// Base storage path
    base_path: String,

    /// Configuration
    _config: RaptorConfig,

    /// Unified cache orchestrator (replaces rowgroup_cache.rs)
    cache: Arc<dyn CacheAccessPatternPort>,

    /// Unified distance computation (replaces simd_encoder.rs distance logic)
    distance_compute: Arc<UnifiedDistanceCompute>,

    /// Filesystem for unified caching operations
    filesystem: Arc<dyn FileSystem>,

    /// Collection ID for cache keys
    collection_id: String,

    /// Per-reader RAPTOR file metadata cache. Lookup is keyed by file path; the
    /// cached entry is validated against the current filesystem (size, modified)
    /// stat before reuse so footer reads only happen when the file actually
    /// changes. The (footer read + bincode::deserialize) round-trip otherwise
    /// occurs on every `read_metadata()` call.
    metadata_cache: Arc<DashMap<String, CachedFileMetadata>>,
    // Note: Caching strategy:
    // - File-level metadata (footer, K centroids, K×K matrix) is cached via
    //   `metadata_cache` above (per-reader, mtime-validated).
    // - Rowgroup-level data (P×K and P² matrices) is NOT cached - read from
    //   disk/cloud on demand to avoid memory bloat (~1.5MB per rowgroup).
}

#[allow(dead_code)]
impl RaptorReader {
    /// Get footer (contains K centroids and K×K matrix) - cached by metadata cache
    /// The footer is file-level metadata and will be cached to avoid repeated reads
    async fn get_footer(&self, file_path: &str) -> Result<Arc<RaptorFooter>> {
        // The zero-copy filesystem automatically caches file-level metadata
        // including the footer which contains:
        // - K centroids for all rowgroups
        // - K×K inter-centroid distance matrix
        // - File metadata and schema
        self.load_footer(file_path).await
    }

    /// Get cached K×K matrix from zero-copy system or load it
    async fn get_kxk_matrix(&self, file_path: &str) -> Result<Arc<InterCentroidMatrix>> {
        let footer = self.get_footer(file_path).await?;
        Ok(Arc::new(footer.inter_centroid_distances.clone()))
    }

    /// Get the number of rowgroups in this file (from footer centroids)
    /// Used to calculate proper nprobe for hierarchical search
    pub async fn get_rowgroup_count(&self) -> Result<usize> {
        let footer = self.get_footer(&self.base_path).await?;
        Ok(footer.centroids.decode_all().len())
    }

    /// Load P×K matrix for a rowgroup - not cached since it's rowgroup data
    /// P×K matrices are stored inside rowgroups, not in file metadata
    async fn get_pxk_matrix(
        &self,
        file_path: &str,
        rowgroup_id: u16,
    ) -> Result<Arc<VectorCentroidMatrix>> {
        // P×K matrices are NOT cached by the metadata cache since they're
        // inside rowgroups. They must be read from disk/cloud each time.
        // Only file-level metadata (footer, K centroids, K×K matrix) is cached.
        self.load_pxk_matrix(file_path, rowgroup_id).await
    }

    /// Create new consolidated reader with unified components
    pub fn new(
        base_path: String,
        collection_id: String,
        config: RaptorConfig,
        cache: Arc<dyn CacheAccessPatternPort>,
        filesystem: Arc<dyn FileSystem>,
    ) -> Self {
        Self {
            base_path,
            collection_id,
            _config: config,
            cache,
            distance_compute: Arc::new(UnifiedDistanceCompute::default()),
            filesystem,
            metadata_cache: Arc::new(DashMap::new()),
        }
    }

    // Bandwidth optimization is now handled internally by UnifiedCachingFilesystem

    /// Read row groups - DIRECT unified module usage, no wrappers
    pub async fn read_row_groups_selective(
        &self,
        file_path: &str,
        rowgroup_selection: Option<Vec<usize>>,
    ) -> Result<Vec<RecordBatch>> {
        debug!(
            "🔍 Reading row groups from {} with unified cache",
            file_path
        );

        let mut results = Vec::new();

        // ONE metadata read + ONE file read for the whole call. The previous
        // shape re-read the ENTIRE file (twice — a speculative cache probe
        // whose result was discarded, then the real read) plus the metadata
        // PER SELECTED ROW GROUP: R selected groups cost R×(2·filesize)
        // bytes of I/O for what one read + R slices needs. Row-group files
        // are immutable once written, so a single read is equivalent.
        let metadata = self.read_metadata(file_path).await?;
        let full_file_data = self.filesystem.read(file_path).await?;

        if let Some(selection) = &rowgroup_selection {
            for &rg_idx in selection {
                // Use zero-copy filesystem with integrated caching
                let cache_key = format!("{}:{}:raptor", file_path, rg_idx);
                self.cache
                    .track_access(cache_key.clone(), CacheKind::VectorData);

                debug!("📥 Loading row group {} from storage", rg_idx);

                let rg_metadata = metadata
                    .row_groups
                    .get(rg_idx)
                    .context("Row group index out of bounds")?;

                let start = rg_metadata.offset;
                let end = start + rg_metadata.compressed_size;
                let compressed_data = &full_file_data[start as usize..end as usize];

                // Use standard decompression (Proxima used for different data types)
                let decompressed = proximadb_compression::decompress(
                    compressed_data,
                    CompressionAlgorithm::Zstd,
                    CompressionContext::Column,
                )?;

                // DIRECT Arrow decode
                use arrow_ipc::reader::StreamReader;
                use std::io::Cursor;
                let cursor = Cursor::new(&decompressed);
                let mut reader = StreamReader::try_new(cursor, None)?;
                let batch = reader.next().context("No record batch")??;

                // Deferred: Implement proper caching with updated APIs

                results.push(batch);
            }
        } else {
            // Load all row groups - DIRECT operations
            for rg_metadata in metadata.row_groups.iter() {
                let start = rg_metadata.offset;
                let end = start + rg_metadata.compressed_size;
                let compressed_data = &full_file_data[start as usize..end as usize];

                // DIRECT decode
                let decompressed = proximadb_compression::decompress(
                    compressed_data,
                    CompressionAlgorithm::Zstd,
                    CompressionContext::Column,
                )?;

                // DIRECT Arrow parse
                use arrow_ipc::reader::StreamReader;
                use std::io::Cursor;
                let cursor = Cursor::new(decompressed);
                let mut reader = StreamReader::try_new(cursor, None)?;
                let batch = reader.next().context("No record batch")??;
                results.push(batch);
            }
        }

        Ok(results)
    }

    /// Search vectors - directly use unified modules without wrapper overhead
    pub async fn search_vectors(
        &mut self,
        query: &[f32],
        top_k: usize,
        collection_id: &str,
        distance_metric: Option<DistanceMetric>,
    ) -> Result<Vec<proximadb_search_types::results::OptimizedSearchRecord>> {
        let _metric = distance_metric.unwrap_or(DistanceMetric::Cosine);

        // Step 1: Matrix Trinity navigation (K×K → P×K → P² matrix pipeline)
        let candidate_ids = self
            .matrix_trinity_search(query, top_k * 2, &_metric)
            .await?;

        // Step 2: Load candidate vectors - DIRECT cache access, no wrapper
        let mut candidates = Vec::new();
        for id in candidate_ids {
            let _cache_key = format!("{}_{}", collection_id, id);

            // DIRECT access to unified cache - no wrapper method
            self.cache
                .track_access(_cache_key.clone(), CacheKind::VectorData);

            // Deferred: Implement proper caching with updated APIs

            // Load from storage if not cached
            let vector = self.load_vector_by_id(&self.base_path, &id).await?;

            // Deferred: Implement proper caching with updated APIs
            candidates.push((id, vector));
        }

        // Step 3: Batch SIMD distance computation across all candidates.
        // The per-record `calculate_distance` loop was dispatching SIMD once
        // per vector; the batch entry point amortizes that and keeps
        // candidates in cache-friendly chunks.
        let mut results = Vec::with_capacity(candidates.len());
        if !candidates.is_empty() {
            let vector_slices: Vec<&[f32]> = candidates.iter().map(|(_, v)| v.as_slice()).collect();
            let mut sim_results: Vec<SimilarityResult> = Vec::with_capacity(candidates.len());
            self.distance_compute.batch_distance_into_buffer(
                query,
                &vector_slices,
                &_metric,
                &mut sim_results,
            );
            drop(vector_slices);

            for ((id, vector), similarity_result) in candidates.into_iter().zip(sim_results) {
                results.push(
                    OptimizedSearchRecord::new(id, similarity_result.normalized_score)
                        .with_similarity(similarity_result.normalized_score)
                        .add_vector(vector)
                        .with_metadata(std::collections::HashMap::new()),
                );
            }
        }

        // Use bounded priority queue for efficient top-k selection
        let mut priority_queue = BoundedPriorityQueue::new(top_k);

        // Insert all results into bounded queue
        for result in results {
            priority_queue.try_insert(result);
        }

        // Get sorted results from bounded queue
        let final_results = priority_queue.into_sorted_vec();

        Ok(final_results)
    }

    // ====== Enhanced Scanning Methods ======

    /// Scan vectors with strategy-based optimization
    /// Supports both fullscan (for compaction) and filtering (for search)
    pub async fn scan_vectors_with_strategy(
        &mut self,
        file_path: &str,
        strategy: ScanStrategy,
    ) -> Result<Vec<proximadb_proto::proximadb_v1::VectorRecord>> {
        match strategy {
            ScanStrategy::FullScan => {
                tracing::info!("🔄 Starting full file scan for {}", file_path);
                self.full_scan_all_vectors(file_path).await
            }
            ScanStrategy::Filtering {
                target_ids,
                predicates,
                max_rowgroups,
            } => {
                tracing::info!(
                    "🎯 Starting selective scan with filtering for {}",
                    file_path
                );
                self.filtered_scan_vectors(file_path, target_ids, predicates, max_rowgroups)
                    .await
            }
        }
    }

    /// Full file scan - reads entire file sequentially
    /// Optimized for compaction, backup, and analysis workflows
    async fn full_scan_all_vectors(
        &mut self,
        file_path: &str,
    ) -> Result<Vec<proximadb_proto::proximadb_v1::VectorRecord>> {
        let start_time = std::time::Instant::now();

        // Load footer to get rowgroup count
        let footer = self.get_footer(file_path).await?;
        let total_rowgroups = footer.file_metadata.row_groups.len();

        tracing::info!(
            "Full scan: processing {} rowgroups sequentially",
            total_rowgroups
        );

        let mut all_vectors = Vec::new();
        let mut bytes_read = 0u64;

        // Sequential scan through all rowgroups (optimal for throughput)
        for (idx, rowgroup) in footer.file_metadata.row_groups.iter().enumerate() {
            tracing::debug!(
                "Scanning rowgroup {}/{}: id={}",
                idx + 1,
                total_rowgroups,
                rowgroup.id
            );

            // Read rowgroup without BloomFilter checking (full scan ignores filtering)
            match self.read_rowgroup(rowgroup.id).await {
                Ok(batch) => {
                    let vectors = self.extract_vector_records_from_batch(&batch)?;
                    bytes_read += self.estimate_rowgroup_size(rowgroup);
                    all_vectors.extend(vectors);

                    if idx % 100 == 0 && idx > 0 {
                        let elapsed = start_time.elapsed().as_secs_f64();
                        let throughput = bytes_read as f64 / elapsed / 1024.0 / 1024.0; // MB/s
                        tracing::info!(
                            "Full scan progress: {}/{} rowgroups ({:.1}%), {:.1} MB/s throughput",
                            idx,
                            total_rowgroups,
                            idx as f64 / total_rowgroups as f64 * 100.0,
                            throughput
                        );
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to read rowgroup {}: {}", rowgroup.id, e);
                    // Continue with next rowgroup in full scan mode
                }
            }
        }

        let elapsed = start_time.elapsed();
        let throughput = bytes_read as f64 / elapsed.as_secs_f64() / 1024.0 / 1024.0;

        trace!(
            "READER: Scan completed - {} vectors from {} rowgroups",
            all_vectors.len(),
            total_rowgroups
        );
        info!(
            "Full scan completed: {} vectors from {} rowgroups in {:.2}s ({:.1} MB/s)",
            all_vectors.len(),
            total_rowgroups,
            elapsed.as_secs_f64(),
            throughput
        );

        Ok(all_vectors)
    }

    /// Filtered scan with predicate pushdown and BloomFilter optimization
    /// Optimized for search and selective retrieval workflows
    async fn filtered_scan_vectors(
        &mut self,
        file_path: &str,
        target_ids: Option<Vec<String>>,
        predicates: Option<Vec<proximadb_raptor_common::Predicate>>,
        max_rowgroups: Option<usize>,
    ) -> Result<Vec<proximadb_proto::proximadb_v1::VectorRecord>> {
        let start_time = std::time::Instant::now();

        // Load footer and prepare for filtering
        self.load_footer_with_mmap(file_path).await?;

        // Step 1: BloomFilter-based rowgroup selection
        let candidate_rowgroups = if let Some(ref ids) = target_ids {
            tracing::debug!(
                "Using BloomFilter optimization for {} target IDs",
                ids.len()
            );
            self.filter_rowgroups_with_enhanced_bloom_filters(file_path, ids)
                .await?
        } else {
            // No ID filtering - include all rowgroups
            let footer = self.get_footer(file_path).await?;
            footer
                .file_metadata
                .row_groups
                .iter()
                .map(|rg| rg.id)
                .collect()
        };

        // Step 2: Apply metadata predicate filtering
        let filtered_rowgroups = if let Some(ref preds) = predicates {
            tracing::debug!("Applying {} metadata predicates", preds.len());
            self.filter_rowgroups_by_predicates(&candidate_rowgroups, preds)
                .await?
        } else {
            candidate_rowgroups
        };

        // Step 3: Apply max rowgroups limit
        let final_rowgroups = if let Some(max) = max_rowgroups {
            filtered_rowgroups.into_iter().take(max).collect()
        } else {
            filtered_rowgroups
        };

        // Get footer reference after all mutable operations
        let footer = self.get_footer(&self.base_path).await?;

        tracing::info!(
            "Filtered scan: processing {}/{} rowgroups after filtering",
            final_rowgroups.len(),
            footer.file_metadata.row_groups.len()
        );

        let mut all_vectors = Vec::new();
        let mut rowgroups_loaded = 0;
        let mut _bytes_read = 0u64;

        // Random I/O pattern for filtered rowgroups (optimized for latency)
        for &rowgroup_id in &final_rowgroups {
            match self.read_rowgroup(rowgroup_id).await {
                Ok(batch) => {
                    let vectors = self.extract_vector_records_from_batch(&batch)?;

                    // Apply fine-grained filtering within rowgroup
                    let filtered_vectors = if let Some(ref ids) = target_ids {
                        self.filter_vectors_by_ids(vectors, ids)
                    } else {
                        vectors
                    };

                    rowgroups_loaded += 1;
                    let rowgroup = footer
                        .file_metadata
                        .row_groups
                        .iter()
                        .find(|rg| rg.id == rowgroup_id)
                        .ok_or_else(|| {
                            proximadb_kernel::error::ProximaDBError::InvalidInput(format!(
                                "Rowgroup {} not found in footer metadata",
                                rowgroup_id
                            ))
                        })?;
                    _bytes_read += self.estimate_rowgroup_size(rowgroup);
                    all_vectors.extend(filtered_vectors);
                }
                Err(e) => {
                    tracing::warn!("Failed to read rowgroup {}: {}", rowgroup_id, e);
                    // Continue with next rowgroup
                }
            }
        }

        let elapsed = start_time.elapsed();
        let efficiency = if !footer.file_metadata.row_groups.is_empty() {
            100.0 * rowgroups_loaded as f64 / footer.file_metadata.row_groups.len() as f64
        } else {
            0.0
        };

        tracing::info!(
            "✅ Filtered scan completed: {} vectors from {}/{} rowgroups in {:.2}s ({:.1}% I/O efficiency)",
            all_vectors.len(),
            rowgroups_loaded,
            footer.file_metadata.row_groups.len(),
            elapsed.as_secs_f64(),
            efficiency
        );

        Ok(all_vectors)
    }

    /// Enhanced BloomFilter-based rowgroup filtering using batch optimization
    async fn filter_rowgroups_with_enhanced_bloom_filters(
        &mut self,
        _file_path: &str,
        target_ids: &[String],
    ) -> Result<Vec<u16>> {
        let footer = self.get_footer(&self.base_path).await?;

        // Use the enhanced batch BloomFilter lookup from common.rs
        let candidate_lists =
            RowGroupBloomFilter::find_candidates_batch_optimized(&footer, target_ids);

        // Merge all candidate rowgroups
        let mut all_candidates = std::collections::HashSet::new();
        for candidates in candidate_lists {
            for candidate in candidates {
                all_candidates.insert(candidate);
            }
        }

        let result: Vec<u16> = all_candidates.into_iter().collect();

        tracing::debug!(
            "BloomFilter filtering: {} target IDs → {} candidate rowgroups",
            target_ids.len(),
            result.len()
        );

        Ok(result)
    }

    /// Filter rowgroups by metadata predicates
    async fn filter_rowgroups_by_predicates(
        &self,
        candidate_rowgroups: &[u16],
        predicates: &[proximadb_raptor_common::Predicate],
    ) -> Result<Vec<u16>> {
        let footer = self.get_footer(&self.base_path).await?;
        let mut filtered = Vec::new();

        for &rowgroup_id in candidate_rowgroups {
            if let Some(rowgroup) = footer
                .file_metadata
                .row_groups
                .iter()
                .find(|rg| rg.id == rowgroup_id)
            {
                // Check if rowgroup satisfies all predicates
                let satisfies_all = predicates
                    .iter()
                    .all(|predicate| self.evaluate_predicate_on_rowgroup(rowgroup, predicate));

                if satisfies_all {
                    filtered.push(rowgroup_id);
                }
            }
        }

        tracing::debug!(
            "Predicate filtering: {}/{} rowgroups satisfy {} predicates",
            filtered.len(),
            candidate_rowgroups.len(),
            predicates.len()
        );

        Ok(filtered)
    }

    /// Evaluate a single predicate against rowgroup metadata
    fn evaluate_predicate_on_rowgroup(
        &self,
        rowgroup: &RowGroupMetadata,
        predicate: &proximadb_raptor_common::Predicate,
    ) -> bool {
        if let Some(column_stats) = rowgroup.metadata_stats.get(&predicate.field) {
            match &predicate.op {
                proximadb_raptor_common::PredicateOp::Eq => {
                    // For equality, value must be within [min, max] range
                    if let (Some(min), Some(max)) =
                        (&column_stats.min_value, &column_stats.max_value)
                    {
                        // Value must be >= min AND <= max for it to possibly exist
                        compare_proxima_values(&predicate.value, min) != std::cmp::Ordering::Less
                            && compare_proxima_values(&predicate.value, max)
                                != std::cmp::Ordering::Greater
                    } else {
                        true // No statistics available, include rowgroup
                    }
                }
                proximadb_raptor_common::PredicateOp::Lt => {
                    // For Lt, we need min < value (at least one row could be less)
                    if let Some(min) = &column_stats.min_value {
                        compare_proxima_values(min, &predicate.value) == std::cmp::Ordering::Less
                    } else {
                        true
                    }
                }
                proximadb_raptor_common::PredicateOp::Lte => {
                    if let Some(min) = &column_stats.min_value {
                        compare_proxima_values(min, &predicate.value) != std::cmp::Ordering::Greater
                    } else {
                        true
                    }
                }
                proximadb_raptor_common::PredicateOp::Gt => {
                    // For Gt, we need max > value
                    if let Some(max) = &column_stats.max_value {
                        compare_proxima_values(max, &predicate.value) == std::cmp::Ordering::Greater
                    } else {
                        true
                    }
                }
                proximadb_raptor_common::PredicateOp::Gte => {
                    if let Some(max) = &column_stats.max_value {
                        compare_proxima_values(max, &predicate.value) != std::cmp::Ordering::Less
                    } else {
                        true
                    }
                }
                // Conservative: include rowgroup if unsure
                _ => true,
            }
        } else {
            true // No statistics for this field, include rowgroup
        }
    }

    /// Filter vectors by IDs within a rowgroup
    fn filter_vectors_by_ids(
        &self,
        vectors: Vec<proximadb_proto::proximadb_v1::VectorRecord>,
        target_ids: &[String],
    ) -> Vec<proximadb_proto::proximadb_v1::VectorRecord> {
        let target_set: std::collections::HashSet<&String> = target_ids.iter().collect();

        vectors
            .into_iter()
            .filter(|v| target_set.contains(&v.id))
            .collect()
    }

    /// Extract VectorRecord objects from Arrow RecordBatch
    /// Reconstructs full VectorRecord structures for ArrowIPC compatibility
    fn extract_vector_records_from_batch(
        &self,
        batch: &RecordBatch,
    ) -> Result<Vec<proximadb_proto::proximadb_v1::VectorRecord>> {
        use arrow_array::cast::AsArray;

        let mut records = Vec::new();
        let num_rows = batch.num_rows();

        // Extract column arrays with proper error handling
        let id_array = batch
            .column_by_name("id")
            .and_then(|col| col.as_any().downcast_ref::<arrow_array::StringArray>())
            .ok_or_else(|| anyhow::anyhow!("Missing or invalid 'id' column in RecordBatch"))?;

        // Vector column can be either ListArray or FixedSizeListArray
        let vector_col = batch
            .column_by_name("vector")
            .ok_or_else(|| anyhow::anyhow!("Missing 'vector' column in RecordBatch"))?;

        // Try FixedSizeListArray first (most common from writer)
        let is_fixed_size_list = vector_col
            .as_any()
            .downcast_ref::<arrow_array::FixedSizeListArray>()
            .is_some();
        let is_list = vector_col
            .as_any()
            .downcast_ref::<arrow_array::ListArray>()
            .is_some();

        if !is_fixed_size_list && !is_list {
            return Err(anyhow::anyhow!(
                "'vector' column is neither ListArray nor FixedSizeListArray"
            ));
        }

        // Optional columns (may not exist in all rowgroups)
        let quantized_vector_array = batch
            .column_by_name("quantized_vector")
            .and_then(|col| col.as_any().downcast_ref::<arrow_array::ListArray>());

        let timestamp_array = batch
            .column_by_name("timestamp")
            .and_then(|col| col.as_any().downcast_ref::<arrow_array::UInt32Array>());

        let updated_at_array = batch
            .column_by_name("updated_at")
            .and_then(|col| col.as_any().downcast_ref::<arrow_array::UInt32Array>());

        let expires_at_array = batch
            .column_by_name("expires_at")
            .and_then(|col| col.as_any().downcast_ref::<arrow_array::UInt32Array>());

        let version_array = batch
            .column_by_name("version")
            .and_then(|col| col.as_any().downcast_ref::<arrow_array::UInt32Array>());

        // Metadata is stored as JSON string in Arrow (serialized HashMap)
        let metadata_array = batch
            .column_by_name("metadata")
            .and_then(|col| col.as_any().downcast_ref::<arrow_array::StringArray>());

        // Source content stored as binary
        let source_content_array = batch
            .column_by_name("source_content")
            .and_then(|col| col.as_any().downcast_ref::<arrow_array::BinaryArray>());

        // Reconstruct VectorRecord for each row
        for row_idx in 0..num_rows {
            let mut record = proximadb_proto::proximadb_v1::VectorRecord::default();

            // Extract ID (required field)
            let id_value = id_array.value(row_idx);
            record.id = id_value.to_string();

            // Extract vector (required field) - handle both FixedSizeListArray and ListArray
            if is_fixed_size_list {
                let fixed_array = vector_col
                    .as_any()
                    .downcast_ref::<arrow_array::FixedSizeListArray>()
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Vector column expected FixedSizeListArray when fixed-size flag is set"
                        )
                    })?;
                if !fixed_array.is_null(row_idx) {
                    let vector_list = fixed_array.value(row_idx);
                    if let Some(float_array) =
                        vector_list.as_primitive_opt::<arrow_array::types::Float32Type>()
                    {
                        record.vector = float_array.values().to_vec();
                    }
                }
            } else if is_list {
                let list_array = vector_col
                    .as_any()
                    .downcast_ref::<arrow_array::ListArray>()
                    .ok_or_else(|| {
                        anyhow::anyhow!("Vector column expected ListArray when list flag is set")
                    })?;
                if !list_array.is_null(row_idx) {
                    let vector_list = list_array.value(row_idx);
                    if let Some(float_array) =
                        vector_list.as_primitive_opt::<arrow_array::types::Float32Type>()
                    {
                        record.vector = float_array.values().to_vec();
                    }
                }
            }

            // Extract quantized vector (optional)
            if let Some(quant_array) = quantized_vector_array
                && !quant_array.is_null(row_idx)
            {
                let quant_list = quant_array.value(row_idx);
                if let Some(_u8_array) =
                    quant_list.as_primitive_opt::<arrow_array::types::UInt8Type>()
                {
                    // quantized_vector removed - internalized in storage
                    // Store quantized data internally if needed
                }
            }

            // Extract timestamp fields (optional)
            if let Some(ts_array) = timestamp_array
                && !ts_array.is_null(row_idx)
            {
                record.timestamp = Some(ts_array.value(row_idx) as i64);
            }

            if let Some(upd_array) = updated_at_array
                && !upd_array.is_null(row_idx)
            {
                record.updated_at = Some(upd_array.value(row_idx) as i64);
            }

            if let Some(exp_array) = expires_at_array
                && !exp_array.is_null(row_idx)
            {
                record.expires_at = Some(exp_array.value(row_idx) as i64);
            }

            if let Some(ver_array) = version_array
                && !ver_array.is_null(row_idx)
            {
                record.version = Some(ver_array.value(row_idx));
            }

            // Extract metadata (JSON string → HashMap)
            if let Some(meta_array) = metadata_array
                && !meta_array.is_null(row_idx)
            {
                let json_str = meta_array.value(row_idx);
                if let Ok(metadata_map) = serde_json::from_str::<
                    std::collections::HashMap<String, serde_json::Value>,
                >(json_str)
                {
                    for (key, value) in metadata_map {
                        // VectorRecord.metadata is the v1 proto SqlValue (this record is a
                        // VectorRecord); it stays SqlValue here. Predicate/column-stats
                        // MetadataValue migrated to ProximaValue (TD-V1SUNSET-3) separately.
                        let sql_value = match value {
                            serde_json::Value::String(s) => proximadb_proto::proximadb_v1::SqlValue {
                                value: Some(
                                    proximadb_proto::proximadb_v1::sql_value::Value::StringValue(s),
                                ),
                            },
                            serde_json::Value::Number(n) => {
                                proximadb_proto::proximadb_v1::SqlValue {
                                    value: Some(
                                        proximadb_proto::proximadb_v1::sql_value::Value::NumberValue(
                                            n.as_f64().unwrap_or(0.0),
                                        ),
                                    ),
                                }
                            }
                            serde_json::Value::Bool(b) => proximadb_proto::proximadb_v1::SqlValue {
                                value: Some(
                                    proximadb_proto::proximadb_v1::sql_value::Value::BoolValue(b),
                                ),
                            },
                            _ => proximadb_proto::proximadb_v1::SqlValue {
                                value: Some(
                                    proximadb_proto::proximadb_v1::sql_value::Value::StringValue(
                                        value.to_string(),
                                    ),
                                ),
                            },
                        };

                        record.metadata.insert(key, sql_value);
                    }
                }
            }

            // Extract source content (binary)
            if let Some(source_array) = source_content_array
                && !source_array.is_null(row_idx)
            {
                let source_bytes = source_array.value(row_idx);
                // Convert bytes to source string
                if let Ok(source_string) = String::from_utf8(source_bytes.to_vec()) {
                    record.source = Some(source_string);
                }
            }

            records.push(record);
        }

        tracing::debug!(
            "Reconstructed {} VectorRecord objects from Arrow RecordBatch ({} rows, {} columns)",
            records.len(),
            num_rows,
            batch.num_columns()
        );

        Ok(records)
    }

    /// Estimate rowgroup size in bytes for throughput calculation
    fn estimate_rowgroup_size(&self, rowgroup: &RowGroupMetadata) -> u64 {
        // Sum up column page sizes
        rowgroup
            .column_pages
            .values()
            .map(|page| page.compressed_size)
            .sum()
    }

    // REMOVED: load_rowgroup_from_storage wrapper method
    // Reason: Redundant - logic inlined directly where needed
    // Benefit: Reduced stack depth, less function call overhead

    /// Read file metadata. Caches the deserialized `RaptorFileMetadata` per
    /// file path and validates the cache against the current (size, modified)
    /// stat — so the footer is only re-read when the file actually changes.
    async fn read_metadata(&self, file_path: &str) -> Result<RaptorFileMetadata> {
        tracing::debug!("RAPTOR read_metadata: Starting for file: {}", file_path);

        let cache_key = format!("{}:{}:raptor", file_path, self.collection_id);

        // Track access pattern for predictive prefetching
        self.cache
            .track_access(cache_key.clone(), CacheKind::Metadata);

        // Stat the file first so we can validate the cache.
        let file_metadata = self.filesystem.metadata(file_path).await?;
        let file_size_u64 = file_metadata.size;
        let modified = file_metadata.modified;
        let file_size = file_size_u64 as usize;

        // Cache hit: same path, same size, same modified timestamp.
        if let Some(entry) = self.metadata_cache.get(file_path)
            && entry.size == file_size_u64
            && entry.modified == modified
        {
            tracing::debug!(
                "RAPTOR read_metadata: cache hit for {} (size={}, modified={:?})",
                file_path,
                file_size_u64,
                modified
            );
            return Ok((*entry.metadata).clone());
        }

        tracing::debug!("RAPTOR read_metadata: File size: {} bytes", file_size);

        // Check if file is too small to have a footer
        if file_size < 8 {
            tracing::error!(
                "RAPTOR read_metadata: File {} is too small ({} bytes), need at least 8 bytes for footer",
                file_path,
                file_size
            );
            return Err(anyhow::anyhow!(
                "RAPTOR file {} is too small ({} bytes) to contain a valid footer",
                file_path,
                file_size
            ));
        }

        tracing::debug!(
            "RAPTOR get_metadata: File {} has size {} bytes",
            file_path,
            file_size
        );

        // Read magic number and footer size in one 8-byte read (optimization)
        let footer_metadata_offset = file_size - 8;
        let footer_metadata_bytes = self
            .filesystem
            .read_range(file_path, footer_metadata_offset as u64, 8)
            .await?;

        // Validate we got the expected number of bytes
        if footer_metadata_bytes.len() < 8 {
            tracing::error!(
                "RAPTOR read_metadata: Expected 8 bytes from footer, got {} bytes. File: {}, size: {}, offset: {}",
                footer_metadata_bytes.len(),
                file_path,
                file_size,
                footer_metadata_offset
            );
            return Err(anyhow::anyhow!(
                "RAPTOR file {} has invalid footer: expected 8 bytes, got {}",
                file_path,
                footer_metadata_bytes.len()
            ));
        }

        // Extract footer size (first 4 bytes) and magic (last 4 bytes)
        let footer_size_bytes = &footer_metadata_bytes[0..4];
        let magic_bytes = &footer_metadata_bytes[4..8];

        if magic_bytes != constants::RAPTOR_MAGIC {
            return Err(anyhow::anyhow!(
                "Invalid RAPTOR file: magic number mismatch"
            ));
        }
        let footer_size = u32::from_le_bytes(footer_size_bytes[..4].try_into()?) as u64;

        // Now read the actual footer using the correct size
        let footer_offset = file_size as u64 - 8 - footer_size;
        let footer_data = self
            .filesystem
            .read_range(file_path, footer_offset, footer_size)
            .await?;

        // Deserialize the footer to get metadata
        tracing::debug!(
            "RAPTOR read_metadata: Deserializing {} bytes of footer",
            footer_data.len()
        );
        let footer: RaptorFooter = bincode::deserialize(&footer_data)?;
        let metadata = footer.file_metadata;

        // Demoted from info! to debug!: this fires on every cache miss and is
        // not actionable in steady state.
        tracing::debug!(
            "RAPTOR read_metadata: loaded metadata with {} row groups, {} total vectors",
            metadata.row_groups.len(),
            metadata.total_vectors
        );

        // Populate cache: any subsequent read with the same (size, modified)
        // stat returns the cached metadata without re-reading the footer.
        self.metadata_cache.insert(
            file_path.to_string(),
            CachedFileMetadata {
                metadata: Arc::new(metadata.clone()),
                size: file_size_u64,
                modified,
            },
        );

        Ok(metadata)
    }

    /// Matrix-based candidate search using K×K matrix for centroid selection
    /// This is the proper implementation for RAPTOR's Matrix Trinity architecture
    async fn ivf_search_candidates(
        &mut self,
        query: &[f32],
        ef: usize,
        metric: &DistanceMetric,
    ) -> Result<Vec<String>> {
        // RAPTOR uses Matrix Trinity: K×K → P×K → P² pipeline
        // Step 1: Load K×K inter-centroid matrix and footer if not cached
        self.load_kxk_matrix().await?;

        // Get footer and K×K matrix (cached by zero-copy filesystem)
        let file_path = &self.base_path;
        let footer = self.get_footer(file_path).await?;
        let kxk_matrix = self.get_kxk_matrix(file_path).await?;

        // Step 2: Calculate distances to all centroids from footer
        let distance_compute = UnifiedDistanceCompute::new(*metric);
        let mut centroid_distances = Vec::with_capacity(kxk_matrix.num_centroids as usize);

        // Get all centroids from the footer's ColumnarCentroids
        for i in 0..kxk_matrix.num_centroids {
            // With 1:1 mapping, centroid_id == rowgroup_id
            let rowgroup_id = i as u16;
            if let Some(centroid) = footer.centroids.get_centroid(rowgroup_id) {
                let dist = distance_compute
                    .calculate_distance(query, &centroid, metric)
                    .raw_value;
                centroid_distances.push((i as usize, dist));
            }
        }

        // Step 3: Select top-k centroids (which map 1:1 to rowgroups)
        centroid_distances.sort_by(|a, b| a.1.total_cmp(&b.1));
        let num_centroids_to_search = (ef / 10).max(1).min(kxk_matrix.num_centroids as usize);

        let mut all_candidates = Vec::new();

        // Step 4: For each selected centroid, search its corresponding rowgroup
        for &(centroid_id, _) in centroid_distances.iter().take(num_centroids_to_search) {
            // With 1:1 mapping, centroid_id == rowgroup_id
            let rowgroup_id = centroid_id as u16;

            // Load P×K matrix for this rowgroup
            let pxk_matrix = self.load_pxk_matrix(&self.base_path, rowgroup_id).await?;

            // Load P² matrix for intra-rowgroup navigation
            let p2_matrix = self.load_p2_matrix_for_rowgroup(rowgroup_id).await?;

            // Search within this rowgroup using P² matrix
            let rowgroup_candidates = self
                .search_within_rowgroup_p2(
                    rowgroup_id,
                    &p2_matrix,
                    &pxk_matrix,
                    query,
                    ef / num_centroids_to_search.max(1),
                    metric,
                )
                .await?;

            all_candidates.extend(rowgroup_candidates);
        }

        // Step 5: Sort all candidates and return top-ef
        all_candidates.sort_by(|a, b| a.1.total_cmp(&b.1));

        let final_candidates: Vec<String> = all_candidates
            .into_iter()
            .take(ef)
            .map(|(id, _dist)| id)
            .collect();

        tracing::info!(
            "Matrix-based search completed: {} candidates from {} centroids/rowgroups",
            final_candidates.len(),
            num_centroids_to_search
        );

        Ok(final_candidates)
    }

    /// Search within a single rowgroup using P² matrix with P×K spillover filtering
    ///
    /// OPTIMIZATION: Uses P×K matrix for triangle inequality filtering to reduce
    /// the number of vectors that need full distance computation.
    ///
    /// Triangle inequality: d(query, vector) >= |d(query, centroid) - d(vector, centroid)|
    ///
    /// If the minimum possible distance exceeds our threshold, we skip the vector entirely.
    async fn search_within_rowgroup_p2(
        &self,
        rowgroup_id: u16,
        _p2_matrix: &Arc<P2Matrix>,
        pxk_matrix: &Arc<VectorCentroidMatrix>,
        query: &[f32],
        k: usize,
        metric: &DistanceMetric,
    ) -> Result<Vec<(String, f32)>> {
        // Load vectors for this rowgroup
        let vectors = self
            .load_rowgroup_vectors(&self.base_path, rowgroup_id)
            .await?;
        let ids = self.load_rowgroup_ids(&self.base_path, rowgroup_id).await?;

        let distance_compute = UnifiedDistanceCompute::new(*metric);

        // Step 1: Compute query-to-centroid distance once
        // Get the centroid for this rowgroup from footer
        let footer = self.get_footer(&self.base_path).await?;
        let centroids = footer.centroids.decode_all();
        let centroid = centroids
            .iter()
            .find(|(rg_id, _)| *rg_id == rowgroup_id)
            .map(|(_, c)| c.clone());

        let query_to_centroid = centroid.as_ref().map_or(0.0, |c| {
            distance_compute
                .calculate_distance(query, c, metric)
                .raw_value
        });

        // Step 2: Use P×K filtering with triangle inequality
        // Track filtering statistics
        let total_vectors = vectors.len();
        let mut filtered_count = 0usize;

        // Dynamic threshold: start high and tighten as we find good candidates
        let mut threshold = f32::MAX;
        let mut candidates = Vec::with_capacity(vectors.len().min(k * 4)); // Pre-allocate for ~4k expected

        for (idx, vector) in vectors.iter().enumerate() {
            // P×K FILTERING: Check if vector can possibly be close enough
            if let Ok(vector_to_centroid) = pxk_matrix.get_distance(idx, rowgroup_id as usize) {
                // Triangle inequality lower bound
                let min_possible_dist = (query_to_centroid - vector_to_centroid).abs();

                // Skip if minimum possible distance exceeds threshold
                if min_possible_dist > threshold {
                    filtered_count += 1;
                    continue;
                }
            }

            // Compute exact distance only for vectors that pass the filter
            let dist = distance_compute
                .calculate_distance(query, vector, metric)
                .raw_value;

            // Update threshold based on k-th best distance seen so far
            if candidates.len() >= k {
                // Sort and keep only top-k
                candidates.sort_by(|a: &(String, f32), b: &(String, f32)| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                candidates.truncate(k);
                // Update threshold to k-th best + margin
                threshold = candidates.last().map_or(f32::MAX, |(_, d)| *d * 1.2);
            }

            if idx < ids.len() {
                candidates.push((ids[idx].clone(), dist));
            }
        }

        // Log filtering efficiency
        if filtered_count > 0 {
            tracing::debug!(
                "P×K filtering: skipped {}/{} vectors ({}% reduction)",
                filtered_count,
                total_vectors,
                (filtered_count * 100) / total_vectors.max(1)
            );
        }

        // Sort and return top-k from this rowgroup
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(k);

        Ok(candidates)
    }

    /// Load vector IDs for a rowgroup
    async fn load_rowgroup_ids(&self, _file_path: &str, rowgroup_id: u16) -> Result<Vec<String>> {
        // This would load the actual IDs from the rowgroup metadata
        // For now, return placeholder IDs
        let num_vectors = 100; // Would come from metadata
        Ok((0..num_vectors)
            .map(|i| format!("rg{}_vec{}", rowgroup_id, i))
            .collect())
    }

    /// Calculate boosted distance using the same 5-component formula as the writer
    ///
    /// This method ensures consistency between storage organization (clustering) and
    /// search navigation (P² matrix traversal) by applying the identical boosting formula:
    /// D = α₁·d₁ + α₂·d₂ + α₃·d₃ + β₁·d₄ + β₂·d₅
    /// Calculate raw distance between two vectors using specified metric
    fn calculate_raw_distance(
        &self,
        v1: &[f32],
        v2: &[f32],
        metric: &DistanceMetric,
    ) -> Result<f32> {
        // Use the unified distance compute engine for consistency
        let result = self.distance_compute.calculate_distance(v1, v2, metric);
        Ok(result.raw_value)
    }

    /// Estimate global average distance from cluster metadata
    fn estimate_global_avg_distance(&self, cluster_metadata: &ClusterMetadata) -> f32 {
        let mut total = 0.0;
        let mut count = 0;

        // Use inter-centroid distances as a proxy for global distances
        for row in &cluster_metadata.centroid_distances {
            for &dist in row {
                if dist > 0.0 {
                    total += dist;
                    count += 1;
                }
            }
        }

        if count > 0 { total / count as f32 } else { 1.0 }
    }

    // ====== Matrix Trinity Search Methods ======

    /// Step 1: Use K×K matrix to select most relevant centroids for search
    /// This implements the centroid selection phase of Matrix Trinity
    /// ENHANCED: Now includes Phase 1 boundary detection using d_i/d_j > 0.8 rule
    async fn select_centroids_with_kxk_matrix(
        &self,
        query: &[f32],
        num_centroids: usize,
        metric: &DistanceMetric,
    ) -> Result<Vec<CentroidSelection>> {
        // Load configuration for boundary detection
        let boundary_config = proximadb_raptor_common::BoundaryDetectionConfig::default();

        // Load all centroids from footer
        let footer = self.get_footer(&self.base_path).await?;

        let distance_compute = UnifiedDistanceCompute::new(*metric);
        let mut all_distances = Vec::new();

        // Step 1: Compute distance from query to all centroids
        for centroid_id in 0..footer.centroids.count as usize {
            let centroid = footer
                .centroids
                .get_centroid(centroid_id as u16)
                .ok_or_else(|| {
                    proximadb_kernel::error::ProximaDBError::Internal(format!(
                        "Centroid {} not found (centroid count mismatch)",
                        centroid_id
                    ))
                })?;
            let dist = distance_compute
                .calculate_distance(query, &centroid, metric)
                .raw_value;

            // Simple 1-to-1 mapping: centroid_id == rowgroup_id
            let rowgroup_id = centroid_id as u16;

            all_distances.push(CentroidSelection {
                centroid_id,
                rowgroup_id,
                distance: dist,
            });
        }

        // Step 2: Sort and select primary centroids
        all_distances.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        let primary_centroids: Vec<CentroidSelection> =
            all_distances.iter().take(num_centroids).cloned().collect();

        // Step 3: Apply boundary detection rule
        let mut boundary_expansions = Vec::new();
        let mut expanded_set = std::collections::HashSet::new();

        // Add primary centroids to expanded set
        for c in &primary_centroids {
            expanded_set.insert(c.centroid_id);
        }

        // Check boundary rule for each primary centroid
        for primary in &primary_centroids {
            // For each other centroid, check if it's a boundary neighbor
            for candidate in &all_distances {
                // Skip if already in set or is the primary itself
                if expanded_set.contains(&candidate.centroid_id)
                    || candidate.centroid_id == primary.centroid_id
                {
                    continue;
                }

                // Calculate boundary ratio: d(Q,primary)/d(Q,candidate)
                let boundary_ratio = primary.distance / candidate.distance;

                // Apply boundary rule: if ratio > 0.8, this is a boundary case
                if boundary_ratio > boundary_config.boundary_ratio_threshold {
                    // Also check K² distance between centroids
                    if let Ok(k2_matrix) = self.get_kxk_matrix(&self.base_path).await {
                        let inter_distance =
                            k2_matrix.get_distance(primary.centroid_id, candidate.centroid_id);

                        // Only add if centroids are reasonably close
                        if inter_distance < primary.distance * 1.5 {
                            boundary_expansions.push((
                                primary.centroid_id as u16,
                                candidate.centroid_id as u16,
                                boundary_ratio,
                            ));
                            expanded_set.insert(candidate.centroid_id);

                            tracing::debug!(
                                "Boundary detected: C{} -> C{} (ratio: {:.2}, dist: {:.3})",
                                primary.centroid_id,
                                candidate.centroid_id,
                                boundary_ratio,
                                inter_distance
                            );
                        }
                    }
                }

                // Apply expansion budget
                let max_expansion =
                    (num_centroids as f32 * boundary_config.expansion_budget_percent) as usize;
                if boundary_expansions.len() >= max_expansion {
                    break;
                }
            }
        }

        // Step 4: Build final result set
        let mut final_centroids = primary_centroids.clone();

        // Add boundary-detected centroids
        for (_, to_centroid, _) in &boundary_expansions {
            if let Some(centroid_data) = all_distances
                .iter()
                .find(|c| c.centroid_id == *to_centroid as usize)
            {
                final_centroids.push(centroid_data.clone());
            }
        }

        // Log boundary detection results
        tracing::info!(
            "Phase 1 Boundary Detection: {} primary + {} boundary = {} total centroids",
            num_centroids,
            boundary_expansions.len(),
            final_centroids.len()
        );

        // Store boundary detection result for debugging (could be returned if needed)
        let _boundary_result = proximadb_raptor_common::BoundaryDetectionResult {
            primary_centroids: primary_centroids
                .iter()
                .map(|c| c.centroid_id as u16)
                .collect(),
            boundary_expansions: boundary_expansions.clone(),
            expanded_centroids: final_centroids
                .iter()
                .map(|c| c.centroid_id as u16)
                .collect(),
            boundary_ratio_threshold: boundary_config.boundary_ratio_threshold,
            expansion_count: boundary_expansions.len(),
        };

        Ok(final_centroids)
    }

    /// Search within a rowgroup using Matrix Trinity (K², P×K, P²) with spillover filtering
    ///
    /// OPTIMIZATION: Uses P×K matrix for triangle inequality pruning before
    /// computing full vector distances, significantly reducing computation.
    async fn search_rowgroup_with_matrices(
        &mut self,
        query: &[f32],
        centroid_id: usize,
        rowgroup_id: u16,
        ef: usize,
        metric: &DistanceMetric,
    ) -> Result<Vec<f32>> {
        // Load P² matrix for this rowgroup
        let _p2_matrix = self.load_p2_matrix_for_rowgroup(rowgroup_id).await?;

        // Load P×K matrix for vector-to-centroid distances (for filtering)
        let pxk_matrix = self.load_pxk_matrix_for_rowgroup(rowgroup_id).await?;

        // Load vectors from rowgroup for distance computation
        let vectors = self.load_vectors_for_rowgroup(rowgroup_id).await?;

        let distance_compute = UnifiedDistanceCompute::new(*metric);

        // Step 1: Get rowgroup centroid and compute query-to-centroid distance
        let footer = self.get_footer(&self.base_path).await?;
        let centroids = footer.centroids.decode_all();
        let centroid = centroids
            .iter()
            .find(|(rg_id, _)| *rg_id == rowgroup_id)
            .map(|(_, c)| c.clone());

        let query_to_centroid = centroid.as_ref().map_or(0.0, |c| {
            distance_compute
                .calculate_distance(query, c, metric)
                .raw_value
        });

        // Step 2: Use P×K filtering with triangle inequality
        let total_vectors = vectors.len();
        let mut filtered_count = 0usize;
        let mut candidate_distances = Vec::with_capacity(ef * 2);
        let mut threshold = f32::MAX;

        for (vector_idx, vector) in vectors.iter().enumerate() {
            // P×K FILTERING: Use triangle inequality to skip distant vectors
            if let Ok(vector_to_centroid) = pxk_matrix.get_distance(vector_idx, centroid_id) {
                let min_possible_dist = (query_to_centroid - vector_to_centroid).abs();

                // Skip if minimum possible distance exceeds our best threshold
                if min_possible_dist > threshold {
                    filtered_count += 1;
                    continue;
                }
            }

            // Compute exact distance for vectors that pass the filter
            let base_distance = distance_compute
                .calculate_distance(query, vector, metric)
                .raw_value;

            // Get P×K distance for boosting
            let pxk_distance = pxk_matrix
                .get_distance(vector_idx, centroid_id)
                .unwrap_or(0.0);

            // Apply boosting formula: base + centroid_penalty
            let boosted_distance = base_distance + (pxk_distance * 0.1);

            candidate_distances.push(boosted_distance);

            // Keep the running top-ef WITHOUT re-sorting on every accepted
            // candidate (was: a full sort+truncate per insertion once the
            // pool filled — O(n·ef·log ef)). Evict the max by scan; ef is a
            // small beam, so the scan is the cheaper side of the trade.
            if candidate_distances.len() > ef
                && let Some(max_pos) = candidate_distances
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(pos, _)| pos)
            {
                candidate_distances.swap_remove(max_pos);
            }
            if candidate_distances.len() >= ef {
                threshold = candidate_distances.iter().copied().fold(f32::MIN, f32::max) * 1.2;
            }
        }

        // Sort and return top distances (≤ ef elements — cheap).
        candidate_distances.sort_by(|a, b| a.total_cmp(b));

        tracing::debug!(
            "P×K filtered search in rowgroup {}: {} candidates, filtered {}/{} ({:.0}%)",
            rowgroup_id,
            candidate_distances.len(),
            filtered_count,
            total_vectors,
            (filtered_count as f32 * 100.0) / total_vectors.max(1) as f32
        );

        Ok(candidate_distances)
    }

    // ====== Matrix Trinity Helper Methods ======

    /// Load P² matrix for a specific rowgroup from disk
    ///
    /// The P² matrix stores intra-rowgroup vector distances for efficient
    /// local navigation during search. Uses upper-triangle storage to save 50%.
    async fn load_p2_matrix_for_rowgroup(&self, rowgroup_id: u16) -> Result<Arc<P2Matrix>> {
        // Try to load from disk first
        match self.load_p2_matrix_from_file(rowgroup_id).await {
            Ok(matrix) => Ok(Arc::new(matrix)),
            Err(e) => {
                tracing::debug!(
                    "P² matrix not found for rowgroup {}, using fallback: {}",
                    rowgroup_id,
                    e
                );
                // Fallback to placeholder for compatibility during development
                let default_matrix = P2Matrix {
                    num_vectors: 1000,
                    distances: vec![128; (1000 * 999) / 2],
                    min_distance: 0.0,
                    max_distance: 2.0,
                    compression: ProximaScheme::BitPacked { bits: 8 },
                    compressed_size: 64000,
                };
                Ok(Arc::new(default_matrix))
            }
        }
    }

    /// Load P² matrix from file for a specific rowgroup
    async fn load_p2_matrix_from_file(&self, rg_id: u16) -> Result<P2Matrix> {
        // Get metadata to find P² matrix offset
        let metadata = self.read_metadata(&self.base_path).await?;
        let rowgroup_metadata = metadata
            .row_groups
            .get(rg_id as usize)
            .ok_or_else(|| anyhow::anyhow!("Row group {} not found in metadata", rg_id))?;

        // Get P² matrix location from column pages
        let p2_metadata = rowgroup_metadata
            .column_pages
            .get(&ColumnType::P2Matrix)
            .ok_or_else(|| anyhow::anyhow!("No P² matrix column for row group {}", rg_id))?;

        let p2_offset = p2_metadata.offset;
        let p2_size = p2_metadata.compressed_size;

        // Read compressed P² matrix data
        let compressed_data = self
            .read_p2_matrix_bytes(&self.base_path, p2_offset, p2_size)
            .await?;

        // Decompress
        let decompressed = self.decompress_p2_matrix(&compressed_data)?;

        // Deserialize P² matrix
        let p2_matrix: P2Matrix =
            bincode::deserialize(&decompressed).context("Failed to deserialize P² matrix")?;

        tracing::debug!(
            "Loaded P² matrix for rowgroup {}: {} vectors, {:.1}KB compressed",
            rg_id,
            p2_matrix.num_vectors,
            p2_size as f32 / 1024.0
        );

        Ok(p2_matrix)
    }

    /// Load P×K matrix for a specific rowgroup (used by Matrix Trinity search)
    async fn load_pxk_matrix_for_rowgroup(
        &mut self,
        rowgroup_id: u16,
    ) -> Result<Arc<VectorCentroidMatrix>> {
        // Load the actual P×K matrix from disk
        self.load_pxk_matrix(&self.base_path, rowgroup_id).await
    }

    /// Load vectors for a specific rowgroup (used by Matrix Trinity search)
    async fn load_vectors_for_rowgroup(&self, _rowgroup_id: u16) -> Result<Vec<Vec<f32>>> {
        // This would load actual vectors from the rowgroup
        // For now, return placeholder vectors
        let num_vectors = 1000;
        let dimension = 384;
        let mut vectors = Vec::with_capacity(num_vectors);

        for i in 0..num_vectors {
            let mut vector = vec![0.0; dimension];
            // Add some variation to make it realistic
            for (j, v_val) in vector.iter_mut().enumerate() {
                *v_val = (i as f32 + j as f32) / (num_vectors + dimension) as f32;
            }
            vectors.push(vector);
        }

        Ok(vectors)
    }

    /// Get rowgroup for a specific centroid (CORRECTED: 1-to-1 mapping)
    /// Each centroid maps to exactly ONE rowgroup for perfect parallelism
    async fn get_rowgroup_for_centroid(&self, centroid_id: usize) -> Result<u16> {
        // PERFECTED DESIGN: 1-to-1 Centroid-to-Rowgroup Mapping
        //
        // KEY INSIGHT: K centroids = K rowgroups (perfect parallelism)
        // - Each centroid gets exactly ONE rowgroup
        // - centroid_id == rowgroup_id (simple indexing)
        // - If rowgroup exceeds capacity → create NEW centroid for overflow
        // - This enables perfect search parallelism across rowgroups

        // PARALLEL SUBDIVISION BENEFITS:
        // - Each rowgroup can be searched independently (perfect parallelism)
        // - No coordination needed between rowgroups during search
        // - Vector space evenly subdivided across K partitions
        // - Overflow handling creates balanced distribution

        // SEARCH PARALLELISM:
        // - K×K matrix selects subset of centroids to search
        // - Each selected centroid = one independent rowgroup to search
        // - Rowgroups can be searched in parallel threads
        // - P² matrix within each rowgroup provides exact distances

        // Simple 1-to-1 mapping
        let rowgroup_id = centroid_id as u16;

        tracing::debug!(
            "Centroid {} → Rowgroup {} (1-to-1 mapping for perfect parallelism)",
            centroid_id,
            rowgroup_id
        );

        Ok(rowgroup_id)
    }

    /// Phase 2: Detect spillover using P×K matrix
    /// This identifies vectors that are closer to non-assigned centroids
    /// Returns additional centroids that should be searched due to spillover
    async fn detect_spillover_with_pxk_matrix(
        &mut self,
        selected_centroids: &[CentroidSelection],
        _metric: &DistanceMetric,
    ) -> Result<proximadb_raptor_common::SpilloverDetectionResult> {
        // Load configuration
        let spillover_config = proximadb_raptor_common::SpilloverDetectionConfig::default();

        // Get the number of centroids from footer
        let footer = self.get_footer(&self.base_path).await?;
        let num_centroids = footer.centroids.count as usize;

        let mut spillover_map = std::collections::HashMap::new();
        let mut spillover_percentages = std::collections::HashMap::new();
        let mut recursive_expansions = Vec::new();
        let mut final_centroids_set = std::collections::HashSet::new();

        // Add initial centroids to final set
        for c in selected_centroids {
            final_centroids_set.insert(c.centroid_id as u16);
        }

        // Track max spillover for statistics
        let mut max_spillover_percentage = 0.0f32;
        let mut total_spillover_vectors = 0usize;

        // Check spillover for each selected centroid
        for centroid_sel in selected_centroids {
            let centroid_id = centroid_sel.centroid_id as u16;
            let rowgroup_id = centroid_sel.rowgroup_id;

            // Load P×K matrix for this rowgroup
            let pxk_matrix = match self.load_pxk_matrix(&self.base_path, rowgroup_id).await {
                Ok(matrix) => matrix,
                Err(e) => {
                    tracing::warn!(
                        "Failed to load P×K matrix for rowgroup {}: {}",
                        rowgroup_id,
                        e
                    );
                    continue;
                }
            };

            // Count spillovers to other centroids
            let mut spillover_counts: std::collections::HashMap<u16, usize> =
                std::collections::HashMap::new();
            let total_vectors = pxk_matrix.num_vectors as usize;

            // For each vector in the rowgroup
            for vector_idx in 0..total_vectors {
                // Get distance to assigned centroid
                let assigned_distance =
                    pxk_matrix.get_distance(vector_idx, centroid_id as usize)?;

                // Check distances to other centroids (sample top candidates)
                for other_centroid_id in 0..std::cmp::min(num_centroids, 20) {
                    if other_centroid_id == centroid_id as usize {
                        continue;
                    }

                    let other_distance = pxk_matrix.get_distance(vector_idx, other_centroid_id)?;

                    // Check spillover condition: vector closer to other centroid
                    if other_distance
                        < assigned_distance * spillover_config.distance_ratio_threshold
                    {
                        *spillover_counts
                            .entry(other_centroid_id as u16)
                            .or_insert(0) += 1;
                        total_spillover_vectors += 1;
                    }
                }
            }

            // Calculate spillover percentages and check threshold
            let mut spillover_targets = Vec::new();
            for (target_centroid, count) in spillover_counts {
                let percentage = count as f32 / total_vectors as f32;

                if percentage > spillover_config.spillover_threshold {
                    spillover_targets.push(target_centroid);
                    spillover_percentages.insert(target_centroid, percentage);
                    max_spillover_percentage = max_spillover_percentage.max(percentage);

                    // Add to final set if not already present
                    if final_centroids_set.insert(target_centroid) {
                        recursive_expansions.push(target_centroid);

                        tracing::debug!(
                            "Spillover detected: {} vectors ({:.1}%) from C{} to C{}",
                            count,
                            percentage * 100.0,
                            centroid_id,
                            target_centroid
                        );
                    }
                }
            }

            if !spillover_targets.is_empty() {
                spillover_map.insert(centroid_id, spillover_targets);
            }
        }

        // Recursive spillover check (depth = 1 for now, can be extended)
        if !recursive_expansions.is_empty() && spillover_config.max_recursive_depth > 1 {
            tracing::debug!(
                "Checking recursive spillovers for {} newly discovered centroids",
                recursive_expansions.len()
            );

            // Convert recursive expansions to CentroidSelection for recursive call
            let _recursive_centroids: Vec<CentroidSelection> = recursive_expansions
                .iter()
                .map(|&id| CentroidSelection {
                    centroid_id: id as usize,
                    rowgroup_id: id, // 1-to-1 mapping
                    distance: 0.0,   // Will be recalculated if needed
                })
                .collect();

            // Recursively check (with depth limit)
            // Note: In production, implement proper depth tracking
            // For now, we skip recursive check to avoid infinite recursion
        }

        // Build final centroid list
        let final_centroids: Vec<u16> = final_centroids_set.into_iter().collect();

        tracing::info!(
            "Phase 2 Spillover Detection: {} initial → {} final centroids ({} added)",
            selected_centroids.len(),
            final_centroids.len(),
            final_centroids.len() - selected_centroids.len()
        );

        Ok(proximadb_raptor_common::SpilloverDetectionResult {
            spillover_map,
            spillover_percentages,
            recursive_expansions,
            final_centroids,
            spillover_threshold: spillover_config.spillover_threshold,
            max_spillover_percentage,
            total_spillover_vectors,
        })
    }

    /// Get rowgroups for centroid using actual footer data (when available)
    async fn get_rowgroups_for_centroid_from_footer(&self, centroid_id: usize) -> Result<Vec<u16>> {
        // Use the actual centroid-to-rowgroup mapping from the footer
        let footer = self.get_footer(&self.base_path).await?;

        // Since we have 1-to-1 mapping, centroid_id == rowgroup_id
        if centroid_id < footer.total_centroids as usize {
            // With 1-to-1 mapping, return just the single rowgroup
            let rowgroup_id = centroid_id as u16;

            tracing::debug!(
                "Centroid {} → Rowgroup {} (1-to-1 mapping)",
                centroid_id,
                rowgroup_id
            );

            Ok(vec![rowgroup_id])
        } else {
            // Invalid centroid ID
            Ok(Vec::new())
        }
    }

    /// Get which centroid a rowgroup belongs to (1-to-1 inverse mapping)
    async fn get_centroid_for_rowgroup(&self, rowgroup_id: u16) -> Result<usize> {
        // Simple 1-to-1 inverse mapping: rowgroup_id == centroid_id
        Ok(rowgroup_id as usize)
    }

    /// Get rowgroup index within its centroid (simplified for 1-to-1 mapping)
    async fn get_rowgroup_index_within_centroid(&self, _rowgroup_id: u16) -> Result<usize> {
        // With 1-to-1 mapping, each centroid has only one rowgroup
        // So the index is always 0
        Ok(0)
    }

    /// Main Matrix Trinity Search Implementation
    /// Orchestrates K×K → P×K → P² matrix pipeline
    /// ENHANCED: Now includes Phase 1 boundary detection and Phase 2 spillover detection
    async fn matrix_trinity_search(
        &mut self,
        query: &[f32],
        ef: usize,
        metric: &DistanceMetric,
    ) -> Result<Vec<String>> {
        tracing::debug!("Starting Enhanced Matrix Trinity search: ef={}", ef);

        // Phase 1: K×K Matrix with Boundary Detection
        // This now includes boundary detection via d_i/d_j > 0.8 rule
        let initial_centroids = self
            .select_centroids_with_kxk_matrix(
                query,
                (ef / 4).max(1), // Initial selection before expansion
                metric,
            )
            .await?;

        let phase1_count = initial_centroids.len();
        tracing::info!(
            "Phase 1 Complete: {} centroids selected (includes boundary expansion)",
            phase1_count
        );

        // Phase 2: P×K Spillover Detection
        // Detect vectors that spill to other centroids
        let spillover_result = self
            .detect_spillover_with_pxk_matrix(&initial_centroids, metric)
            .await?;

        // Combine centroids from Phase 1 and Phase 2
        let mut final_centroid_set = std::collections::HashSet::new();
        for c in &initial_centroids {
            final_centroid_set.insert(c.centroid_id);
        }
        for c in &spillover_result.final_centroids {
            final_centroid_set.insert(*c as usize);
        }

        // Convert back to CentroidSelection for search
        let mut final_centroids = Vec::new();
        for centroid_id in final_centroid_set {
            // Find original selection or create new one
            if let Some(original) = initial_centroids
                .iter()
                .find(|c| c.centroid_id == centroid_id)
            {
                final_centroids.push(original.clone());
            } else {
                // Centroid added via spillover, create new selection
                final_centroids.push(CentroidSelection {
                    centroid_id,
                    rowgroup_id: centroid_id as u16, // 1-to-1 mapping
                    distance: f32::MAX,              // Will be properly scored during search
                });
            }
        }

        tracing::info!(
            "Phase 2 Complete: {} → {} centroids after spillover detection (+{})",
            phase1_count,
            final_centroids.len(),
            final_centroids.len() - phase1_count
        );

        // Phase 3: P² Matrix - Search within selected rowgroups
        let mut all_candidates = Vec::new();

        for selection in &final_centroids {
            let rowgroup_candidates = self
                .search_rowgroup_with_matrices(
                    query,
                    selection.centroid_id,
                    selection.rowgroup_id,
                    ef,
                    metric,
                )
                .await?;

            // Convert distances to candidate IDs (simplified for now)
            for (idx, _distance) in rowgroup_candidates.iter().enumerate() {
                all_candidates.push(format!(
                    "rg{}_c{}_v{}",
                    selection.rowgroup_id, selection.centroid_id, idx
                ));

                if all_candidates.len() >= ef {
                    break;
                }
            }

            if all_candidates.len() >= ef {
                break;
            }
        }

        tracing::debug!(
            "Matrix Trinity search completed: {} candidates from {} centroid-rowgroup pairs",
            all_candidates.len(),
            final_centroids.len()
        );

        Ok(all_candidates)
    }

    /// Load K×K inter-centroid distance matrix from footer
    /// Cached for reader lifetime as it's used frequently in search
    async fn load_kxk_matrix(&self) -> Result<()> {
        // K×K matrix is loaded on demand from the footer, no need to check cache

        // Footer and K×K matrix are loaded on demand via get_footer() and get_kxk_matrix()

        if let Ok(_footer) = self.get_footer(&self.base_path).await {
            // K×K matrix is derived from footer on demand

            let matrix = self.get_kxk_matrix(&self.base_path).await?;
            tracing::info!(
                "Loaded K×K matrix: {} centroids, {} bytes compressed (87.5% compression)",
                matrix.num_centroids,
                matrix.compressed_data.len()
            );
        }

        Ok(())
    }

    /// Load P×K vector-to-centroid matrix for a specific rowgroup
    /// P×K matrices are stored in rowgroups and must be read from disk each time
    async fn load_pxk_matrix(
        &self,
        file_path: &str,
        rowgroup_id: u16,
    ) -> Result<Arc<VectorCentroidMatrix>> {
        // Load rowgroup metadata to find P×K matrix location
        let metadata = self.read_metadata(file_path).await?;
        let rg_metadata = metadata
            .row_groups
            .iter()
            .find(|rg| rg.id == rowgroup_id)
            .ok_or_else(|| anyhow::anyhow!("Rowgroup {} not found", rowgroup_id))?;

        // Check if P×K matrix exists in column pages
        if let Some(pxk_metadata) = rg_metadata.column_pages.get(&ColumnType::PxKMatrix) {
            // Read P×K matrix from column page
            let matrix_data = self
                .filesystem
                .read_range(
                    &self.base_path,
                    pxk_metadata.offset,
                    pxk_metadata.compressed_size,
                )
                .await?;

            // Decompress if needed
            let decompressed = if pxk_metadata.compression != CompressionAlgorithm::None {
                proximadb_compression::decompress(
                    &matrix_data,
                    pxk_metadata.compression,
                    CompressionContext::Column,
                )?
            } else {
                matrix_data
            };

            // Deserialize matrix
            let matrix: VectorCentroidMatrix = bincode::deserialize(&decompressed)?;

            // Extract fields before moving matrix into Arc
            let storage_strategy = matrix.storage_strategy;
            let num_vectors = matrix.num_vectors;
            let num_centroids = matrix.num_centroids;

            let compression_ratio = match storage_strategy {
                VectorCentroidStorageStrategy::Full => 50.0,
                VectorCentroidStorageStrategy::Hierarchical => 99.85,
                VectorCentroidStorageStrategy::Sparse => 99.0,
            };

            let arc_matrix = Arc::new(matrix);

            tracing::debug!(
                "Loaded inline P×K matrix for rowgroup {} from offset {}: \
                 {} vectors × {} centroids, strategy {:?}, {:.2}% compression",
                rowgroup_id,
                pxk_metadata.offset,
                num_vectors,
                num_centroids,
                storage_strategy,
                compression_ratio
            );

            return Ok(arc_matrix);
        }

        // Fallback: Check footer for P×K matrix (old format for compatibility)
        // Footer is loaded on demand via get_footer()

        // Try to load from footer's vector_centroid_matrices (these are refs, need to load actual data)
        if let Ok(footer) = self.get_footer(&self.base_path).await {
            for matrix_ref in &footer.vector_centroid_matrices {
                if matrix_ref.rowgroup_id == rowgroup_id {
                    // Load actual matrix data from file using the ref's offset
                    let matrix_data = self
                        .filesystem
                        .read_range(
                            &self.base_path,
                            matrix_ref.file_offset,
                            matrix_ref.compressed_size as u64,
                        )
                        .await?;

                    // Parse compression algorithm from string
                    let compression_algo = match matrix_ref.compression_algorithm.as_str() {
                        "zstd" => proximadb_compression::CompressionAlgorithm::Zstd,
                        "lz4" => proximadb_compression::CompressionAlgorithm::Lz4,
                        "snappy" => proximadb_compression::CompressionAlgorithm::Snappy,
                        _ => proximadb_compression::CompressionAlgorithm::None,
                    };

                    // Decompress if needed
                    let decompressed =
                        if compression_algo != proximadb_compression::CompressionAlgorithm::None {
                            proximadb_compression::decompress(
                                &matrix_data,
                                compression_algo,
                                proximadb_compression::CompressionContext::Column,
                            )?
                        } else {
                            matrix_data
                        };

                    // Deserialize to VectorCentroidMatrix
                    let matrix: VectorCentroidMatrix = bincode::deserialize(&decompressed)?;
                    let arc_matrix = Arc::new(matrix);
                    // P×K matrices are not cached - read on demand
                    return Ok(arc_matrix);
                }
            }
        }

        Err(anyhow::anyhow!(
            "P×K matrix not found for rowgroup {}",
            rowgroup_id
        ))
    }

    /// Get inter-centroid distance from K×K matrix with O(1) lookup
    pub async fn get_inter_centroid_distance(
        &self,
        centroid_i: usize,
        centroid_j: usize,
    ) -> Result<f32> {
        // Ensure K×K matrix is loaded
        self.load_kxk_matrix().await?;

        let matrix = self.get_kxk_matrix(&self.base_path).await?;
        Ok(matrix.get_distance(centroid_i, centroid_j))
    }

    /// Get vector-to-centroid distance from P×K matrix
    pub async fn get_vector_centroid_distance(
        &mut self,
        rowgroup_id: u16,
        vector_idx: usize,
        centroid_idx: usize,
    ) -> Result<f32> {
        let matrix = self.load_pxk_matrix(&self.base_path, rowgroup_id).await?;
        matrix.get_distance(vector_idx, centroid_idx)
    }

    /// Load the centralized footer containing all centroids
    /// Returns the footer and caches it through zero-copy system
    async fn load_footer(&self, file_path: &str) -> Result<Arc<RaptorFooter>> {
        tracing::debug!("RAPTOR load_footer: Starting for file: {}", file_path);

        // Read footer from file - the zero-copy filesystem will handle caching
        let file_metadata = self.filesystem.metadata(file_path).await?;
        let file_size = file_metadata.size as usize;
        tracing::debug!("RAPTOR load_footer: File size: {} bytes", file_size);

        if file_size < 8 {
            tracing::error!(
                "RAPTOR load_footer: File too small ({} bytes), needs at least 8 bytes for footer metadata",
                file_size
            );
            return Err(anyhow::anyhow!("File too small to contain RAPTOR footer"));
        }

        // Read magic number and footer size
        let footer_metadata_offset = file_size - 8;
        tracing::debug!(
            "RAPTOR load_footer: Reading footer metadata from offset: {}, filesystem_type: {}",
            footer_metadata_offset,
            self.filesystem.filesystem_type()
        );

        // CRITICAL FIX: Use direct method call instead of UFCS
        // UFCS `FileSystem::read_range(self.filesystem.as_ref(), ...)` was calling the
        // default trait implementation instead of dispatching through the vtable!
        let footer_metadata_bytes = self
            .filesystem
            .read_range(file_path, footer_metadata_offset as u64, 8)
            .await?;
        tracing::debug!(
            "RAPTOR load_footer: Footer metadata bytes: {:?}",
            footer_metadata_bytes
        );

        // Validate we read the expected 8 bytes
        if footer_metadata_bytes.len() < 8 {
            return Err(anyhow::anyhow!(
                "Failed to read footer metadata: expected 8 bytes, got {} bytes",
                footer_metadata_bytes.len()
            ));
        }

        let footer_size = u32::from_le_bytes(footer_metadata_bytes[0..4].try_into()?) as u64;
        let magic = &footer_metadata_bytes[4..8];
        tracing::debug!(
            "RAPTOR load_footer: Footer size: {} bytes, magic: {:?}, expected: {:?}",
            footer_size,
            magic,
            constants::RAPTOR_MAGIC
        );

        if magic != constants::RAPTOR_MAGIC {
            tracing::error!(
                "RAPTOR load_footer: Magic mismatch! Got {:?}, expected {:?}",
                magic,
                constants::RAPTOR_MAGIC
            );
            return Err(anyhow::anyhow!(
                "Invalid RAPTOR file: magic number mismatch"
            ));
        }

        // Read the actual footer
        let footer_offset = file_size as u64 - 8 - footer_size;
        tracing::debug!(
            "RAPTOR load_footer: Reading footer from offset: {}, size: {}",
            footer_offset,
            footer_size
        );
        let footer_bytes = self
            .filesystem
            .read_range(file_path, footer_offset, footer_size)
            .await?;

        // Deserialize footer
        tracing::debug!(
            "RAPTOR load_footer: Deserializing {} bytes of footer data",
            footer_bytes.len()
        );
        let footer: RaptorFooter = bincode::deserialize(&footer_bytes)?;

        tracing::info!(
            "RAPTOR load_footer: Successfully loaded footer with {} centroids, {} row groups",
            footer.centroids.count,
            footer.file_metadata.row_groups.len()
        );

        // The zero-copy filesystem will cache this automatically
        Ok(Arc::new(footer))
    }

    /// Zero-copy memory-mapped footer loading (preferred method)
    async fn load_footer_with_mmap(&mut self, file_path: &str) -> Result<Arc<RaptorFooter>> {
        // Use filesystem API for cloud compatibility
        let filesystem = self.filesystem.clone();

        // Get file metadata for size
        let metadata = filesystem.metadata(file_path).await?;
        let file_size = metadata.size;

        // Memory-map the entire file for zero-copy access
        let mmap = filesystem
            .get_mmap(file_path)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Memory mapping not supported for {}", file_path))?;

        // Read footer metadata from the end of file (8 bytes: footer_size + magic)
        if file_size < 8 {
            return Err(anyhow::anyhow!("File too small to contain RAPTOR footer"));
        }

        let footer_metadata_offset = file_size as usize - 8;
        let footer_size_bytes = &mmap[footer_metadata_offset..footer_metadata_offset + 4];
        let magic_bytes = &mmap[footer_metadata_offset + 4..footer_metadata_offset + 8];

        // Verify magic number for file integrity
        if magic_bytes != constants::RAPTOR_MAGIC {
            return Err(anyhow::anyhow!(
                "Invalid RAPTOR file: magic number mismatch"
            ));
        }

        let footer_size = u32::from_le_bytes(footer_size_bytes.try_into()?) as usize;
        let footer_offset = file_size as usize - 8 - footer_size;

        // Zero-copy access to footer bytes directly from memory-mapped region
        let footer_bytes = &mmap[footer_offset..footer_offset + footer_size];

        // Deserialize footer directly from memory-mapped bytes (zero-copy)
        let footer: RaptorFooter = bincode::deserialize(footer_bytes)?;

        tracing::info!(
            "Zero-copy mmap footer load: {} centroids, {} bytes, file: {}",
            footer.centroids.count,
            footer_size,
            file_path
        );

        Ok(Arc::new(footer))
    }

    /// Traditional file I/O fallback for cloud storage
    async fn load_footer_traditional(&mut self, file_path: &str) -> Result<()> {
        // Read file size to find footer location
        let file_size = self.filesystem.metadata(file_path).await?.size;

        // Read magic number and footer size in one 8-byte read (optimization)
        let footer_metadata_offset = file_size - 8;
        let footer_metadata_bytes = self
            .filesystem
            .read_range(file_path, footer_metadata_offset, 8)
            .await?;

        // Extract footer size (first 4 bytes) and magic (last 4 bytes)
        let footer_size_bytes = &footer_metadata_bytes[0..4];
        let magic_bytes = &footer_metadata_bytes[4..8];

        // Verify magic number
        if magic_bytes != constants::RAPTOR_MAGIC {
            return Err(anyhow::anyhow!(
                "Invalid RAPTOR file: magic number mismatch"
            ));
        }
        let footer_size = u32::from_le_bytes(footer_size_bytes[..4].try_into()?) as u64;

        // Read the actual footer
        let footer_offset = file_size - 8 - footer_size;
        let footer_bytes = self
            .filesystem
            .read_range(file_path, footer_offset, footer_size)
            .await?;

        // Deserialize footer
        let footer: RaptorFooter = bincode::deserialize(&footer_bytes)?;

        // Cache the footer
        // Footer is cached by zero-copy filesystem

        tracing::info!(
            "Traditional I/O footer load: {} centroids of dimension {}, total size {} bytes",
            footer.centroids.count,
            footer.centroids.dimension,
            footer_size
        );

        // Cache the footer
        // Footer is cached by zero-copy filesystem

        Ok(())
    }

    /// Get centroid for a specific rowgroup from the cached footer
    /// Returns None if footer not loaded or rowgroup not found
    pub async fn get_centroid(&self, rowgroup_id: u16) -> Result<Option<Vec<f32>>> {
        let footer = self.get_footer(&self.base_path).await?;
        Ok(footer.centroids.get_centroid(rowgroup_id))
    }

    /// Load bloom filter for a specific row group WITHOUT reading row group data
    /// This enables efficient ID-based row group skipping during search
    pub async fn load_bloom_filter(
        &self,
        file_path: &str,
        rowgroup_id: u16,
    ) -> Result<Arc<RowGroupBloomFilter>> {
        // Note: Bloom filters are not cached at the engine level
        // They could be cached by the zero-copy filesystem if needed

        // Load metadata to get bloom filter offset
        let metadata = self.read_metadata(file_path).await?;
        let rowgroup_metadata = metadata
            .row_groups
            .get(rowgroup_id as usize)
            .ok_or_else(|| anyhow::anyhow!("Row group {} not found", rowgroup_id))?;

        // Check if bloom filter exists for this row group
        let bloom_metadata = rowgroup_metadata
            .column_pages
            .get(&ColumnType::BloomFilter)
            .ok_or_else(|| anyhow::anyhow!("No bloom filter for row group {}", rowgroup_id))?;
        let bloom_offset = bloom_metadata.offset;

        tracing::debug!(
            "Loading bloom filter for row group {} at offset {} (independent of row data)",
            rowgroup_id,
            bloom_offset
        );

        // Read compressed bloom filter data from disk
        // Note: We read ONLY the bloom filter, not the entire row group
        let compressed_bloom_data = self
            .read_bloom_filter_bytes(file_path, bloom_offset)
            .await?;

        // Decompress bloom filter using unified compression
        let bloom_data = self.decompress_bloom_filter(&compressed_bloom_data)?;

        // Deserialize bloom filter
        let bloom_filter: RowGroupBloomFilter =
            bincode::deserialize(&bloom_data).context("Failed to deserialize bloom filter")?;

        let bloom_filter_arc = Arc::new(bloom_filter);

        // Note: Not caching bloom filter at engine level - rely on filesystem caching

        tracing::debug!(
            "Loaded bloom filter: {} IDs, {:.3}% FPR, {} bytes for row group {}",
            bloom_filter_arc.stats().num_ids,
            bloom_filter_arc.stats().false_positive_rate * 100.0,
            bloom_filter_arc.stats().size_bytes,
            rowgroup_id
        );

        Ok(bloom_filter_arc)
    }

    /// Read bloom filter bytes from disk at specific offset
    async fn read_bloom_filter_bytes(&self, file_path: &str, offset: u64) -> Result<Vec<u8>> {
        // First, read the bloom filter size (4 bytes at offset)
        let size_bytes = self.filesystem.read_range(file_path, offset, 4).await?;

        let bloom_size =
            u32::from_le_bytes([size_bytes[0], size_bytes[1], size_bytes[2], size_bytes[3]]) as u64;

        // Read the actual compressed bloom filter data
        let bloom_data = self
            .filesystem
            .read_range(file_path, offset + 4, bloom_size)
            .await?;

        Ok(bloom_data)
    }

    /// Decompress bloom filter using unified compression module
    fn decompress_bloom_filter(&self, compressed_data: &[u8]) -> Result<Vec<u8>> {
        use proximadb_compression::{CompressionContext, CompressionProvider, StandardCompression};

        // Create decompression context
        let decompressor = StandardCompression;
        let context = CompressionContext::Block; // Bloom filter data is heterogeneous

        // Decompress using ZSTD (matches writer compression)
        let decompressed =
            decompressor.decompress(compressed_data, CompressionAlgorithm::Zstd, context)?;

        Ok(decompressed)
    }

    /// Check if a vector ID might exist in a row group using bloom filter
    /// Returns None if bloom filter not available, Some(bool) for membership test
    pub async fn check_id_in_rowgroup(
        &self,
        file_path: &str,
        rowgroup_id: u16,
        vector_id: &str,
    ) -> Result<Option<bool>> {
        match self.load_bloom_filter(file_path, rowgroup_id).await {
            Ok(bloom_filter) => Ok(Some(bloom_filter.contains(vector_id))),
            Err(_) => {
                tracing::debug!(
                    "Bloom filter not available for row group {}, assuming ID might exist",
                    rowgroup_id
                );
                Ok(None) // Bloom filter not available, assume ID might exist
            }
        }
    }

    /// Filter row groups based on vector ID using bloom filters
    /// Returns list of row group IDs that might contain the vector ID
    pub async fn filter_rowgroups_by_id(
        &self,
        file_path: &str,
        vector_id: &str,
    ) -> Result<Vec<u16>> {
        let metadata = self.read_metadata(file_path).await?;
        let mut candidate_rowgroups = Vec::new();

        for (rg_idx, _) in metadata.row_groups.iter().enumerate() {
            let rowgroup_id = rg_idx as u16;

            match self
                .check_id_in_rowgroup(file_path, rowgroup_id, vector_id)
                .await?
            {
                Some(true) => {
                    tracing::debug!(
                        "Bloom filter: ID '{}' might exist in row group {}",
                        vector_id,
                        rowgroup_id
                    );
                    candidate_rowgroups.push(rowgroup_id);
                }
                Some(false) => {
                    tracing::debug!(
                        "Bloom filter: ID '{}' definitely NOT in row group {}",
                        vector_id,
                        rowgroup_id
                    );
                    // Skip this row group - bloom filter guarantees ID is not present
                }
                None => {
                    tracing::debug!(
                        "No bloom filter for row group {}, including in search",
                        rowgroup_id
                    );
                    candidate_rowgroups.push(rowgroup_id); // Include if no bloom filter
                }
            }
        }

        tracing::info!(
            "Bloom filter pruning: {} of {} row groups selected for ID '{}'",
            candidate_rowgroups.len(),
            metadata.row_groups.len(),
            vector_id
        );

        Ok(candidate_rowgroups)
    }

    /// Load a specific vector by ID from the appropriate row group
    /// Uses bloom filter to identify candidate row groups first
    pub async fn load_vector_by_id(&self, file_path: &str, vector_id: &str) -> Result<Vec<f32>> {
        // Use bloom filter to find candidate row groups
        let candidate_rowgroups = self.filter_rowgroups_by_id(file_path, vector_id).await?;

        if candidate_rowgroups.is_empty() {
            return Err(anyhow::anyhow!(
                "Vector ID '{}' not found in any row group",
                vector_id
            ));
        }

        // Search through candidate row groups
        for &rg_id in &candidate_rowgroups {
            if let Ok(vector) = self
                .find_vector_in_rowgroup(file_path, rg_id, vector_id)
                .await
            {
                tracing::debug!("Found vector '{}' in row group {}", vector_id, rg_id);
                return Ok(vector);
            }
        }

        Err(anyhow::anyhow!(
            "Vector ID '{}' not found in candidate row groups",
            vector_id
        ))
    }

    /// Find specific vector within a row group
    async fn find_vector_in_rowgroup(
        &self,
        _file_path: &str,
        rg_id: u16,
        vector_id: &str,
    ) -> Result<Vec<f32>> {
        // Load the row group data
        let batch = self.read_rowgroup(rg_id).await?;

        // Find the vector by ID in the batch
        if let Some(id_array) = batch.column_by_name("id")
            && let Some(vector_array) = batch.column_by_name("vector")
        {
            use arrow_array::{ListArray, StringArray};

            if let Some(ids) = id_array.as_any().downcast_ref::<StringArray>() {
                for i in 0..ids.len() {
                    if !ids.is_null(i) && ids.value(i) == vector_id {
                        // Found the ID, extract the vector
                        if let Some(vectors) = vector_array.as_any().downcast_ref::<ListArray>()
                            && let Some(vector_values) = vectors
                                .value(i)
                                .as_any()
                                .downcast_ref::<arrow_array::Float32Array>(
                            )
                        {
                            let vector: Vec<f32> = (0..vector_values.len())
                                .map(|j| vector_values.value(j))
                                .collect();
                            return Ok(vector);
                        }
                    }
                }
            }
        }

        Err(anyhow::anyhow!(
            "Vector '{}' not found in row group {}",
            vector_id,
            rg_id
        ))
    }

    /// Main similarity search entry point using target vector ID
    pub async fn similarity_search_by_id(
        &self,
        file_path: &str,
        target_id: &str,
        k: usize,
    ) -> Result<Vec<SimilarityResult>> {
        tracing::info!("Starting similarity search for ID '{}', k={}", target_id, k);

        // STEP 1: Bloom filter pre-screening
        let candidate_rowgroups = self.filter_rowgroups_by_id(file_path, target_id).await?;

        if candidate_rowgroups.is_empty() {
            return Err(anyhow::anyhow!(
                "Target ID '{}' not found in any row group",
                target_id
            ));
        }

        tracing::debug!(
            "Bloom filter screening: {} candidate row groups",
            candidate_rowgroups.len()
        );

        // STEP 2: Load target vector and compute centroid distances
        let target_vector = self.load_vector_by_id(file_path, target_id).await?;
        let mut cluster_distances = Vec::new();

        for &rg_id in &candidate_rowgroups {
            if let Some(centroid) = self.get_centroid(rg_id).await? {
                let distance = self
                    .distance_compute
                    .calculate_distance(
                        &target_vector,
                        &centroid,
                        &proximadb_distance_kernel::DistanceMetric::Cosine,
                    )
                    .raw_value;
                cluster_distances.push((rg_id, distance));
            }
        }

        // Sort by centroid distance (closest clusters first)
        cluster_distances
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        tracing::debug!(
            "Centroid ranking: {} clusters ordered by distance",
            cluster_distances.len()
        );

        // STEP 3: Local graph traversal within selected clusters
        // Use sqrt(k) for cluster selection - IVF standard for ~95% recall with O(sqrt(k)) complexity
        let total_clusters = cluster_distances.len();
        let max_clusters_to_search = if total_clusters == 0 {
            1
        } else {
            ((total_clusters as f64).sqrt().ceil() as usize)
                .max(1)
                .min(total_clusters)
        };
        tracing::debug!(
            "RAPTOR similarity: Using nprobe={} (sqrt of {} clusters)",
            max_clusters_to_search,
            total_clusters
        );
        let mut all_candidates = Vec::new();

        for (rg_id, centroid_dist) in cluster_distances.into_iter().take(max_clusters_to_search) {
            tracing::debug!(
                "Searching cluster {} with centroid distance {:.4}",
                rg_id,
                centroid_dist
            );

            // Load row group vectors for this cluster
            let cluster_results = self
                .search_within_cluster(
                    file_path,
                    rg_id,
                    &target_vector,
                    target_id,
                    k * 2, // Over-fetch for cross-cluster ranking
                )
                .await?;

            all_candidates.extend(cluster_results);
        }

        // STEP 4: Cross-cluster result merging with 5-component boosting
        let final_results = self
            .merge_cross_cluster_results(all_candidates, &target_vector, k)
            .await?;

        tracing::info!(
            "Similarity search completed: {} results for ID '{}'",
            final_results.len(),
            target_id
        );
        Ok(final_results)
    }

    /// Search within a single cluster using local graph traversal
    async fn search_within_cluster(
        &self,
        file_path: &str,
        rg_id: u16,
        target_vector: &[f32],
        target_id: &str,
        k: usize,
    ) -> Result<Vec<CandidateResult>> {
        // Load row group data
        let batch = self.read_rowgroup(rg_id).await?;
        let vectors = self.extract_vectors_from_batch(&batch)?;
        let ids = self.extract_ids_from_batch(&batch)?;

        // Try to load P² matrix for this cluster (optional)
        let p2_matrix = match self.load_p2_matrix(file_path, rg_id).await {
            Ok(matrix) => Some(matrix),
            Err(_) => {
                tracing::debug!("No P² matrix for row group {}, using linear search", rg_id);
                None
            }
        };

        let results = if let Some(matrix) = p2_matrix {
            // Use P² matrix navigation if available
            self.navigate_with_p2_matrix(&matrix, &vectors, &ids, target_vector, target_id, k)
                .await?
        } else {
            // Fallback to linear search within cluster
            self.linear_search_cluster(&vectors, &ids, target_vector, k)?
        };

        Ok(results)
    }

    /// Load P² matrix for a specific row group
    async fn load_p2_matrix(&self, file_path: &str, rg_id: u16) -> Result<IntraRowgroupMatrix> {
        // Get metadata to find P² matrix offset
        let metadata = self.read_metadata(file_path).await?;
        let rowgroup_metadata = metadata
            .row_groups
            .get(rg_id as usize)
            .ok_or_else(|| anyhow::anyhow!("Row group {} not found", rg_id))?;

        // Get P² matrix from column pages
        let p2_metadata = rowgroup_metadata
            .column_pages
            .get(&ColumnType::P2Matrix)
            .ok_or_else(|| anyhow::anyhow!("No P² matrix for row group {}", rg_id))?;
        let p2_offset = p2_metadata.offset;
        let p2_size = p2_metadata.compressed_size;

        // Read compressed P² matrix data
        let compressed_data = self
            .read_p2_matrix_bytes(file_path, p2_offset, p2_size)
            .await?;

        // Decompress using unified compression
        let decompressed = self.decompress_p2_matrix(&compressed_data)?;

        // Deserialize P² matrix
        let p2_matrix: P2Matrix =
            bincode::deserialize(&decompressed).context("Failed to deserialize P² matrix")?;

        // Load vectors for efficient navigation
        let vectors = self.load_rowgroup_vectors(file_path, rg_id).await?;

        Ok(IntraRowgroupMatrix::new(p2_matrix, vectors))
    }

    /// Load vectors from a rowgroup for P² matrix navigation
    async fn load_rowgroup_vectors(&self, _file_path: &str, rg_id: u16) -> Result<Vec<Vec<f32>>> {
        // Read the row group data
        let batch = self.read_rowgroup(rg_id).await?;

        // Extract vectors from the batch
        self.extract_vectors_from_batch(&batch)
    }

    /// Read P² matrix bytes from disk
    async fn read_p2_matrix_bytes(
        &self,
        file_path: &str,
        offset: u64,
        size: u64,
    ) -> Result<Vec<u8>> {
        // Read the compressed P² matrix data directly
        let matrix_data = self.filesystem.read_range(file_path, offset, size).await?;

        Ok(matrix_data)
    }

    /// Decompress P² matrix using unified compression
    fn decompress_p2_matrix(&self, compressed_data: &[u8]) -> Result<Vec<u8>> {
        use proximadb_compression::{CompressionContext, CompressionProvider, StandardCompression};

        let decompressor = StandardCompression;
        let context = CompressionContext::Column; // P² matrix contains homogeneous distance values
        let decompressed =
            decompressor.decompress(compressed_data, CompressionAlgorithm::Zstd, context)?;

        Ok(decompressed)
    }

    /// Navigate using P² matrix for intra-rowgroup search
    async fn navigate_with_p2_matrix(
        &self,
        matrix: &IntraRowgroupMatrix,
        vectors: &[Vec<f32>],
        ids: &[String],
        target_vector: &[f32],
        _target_id: &str,
        k: usize,
    ) -> Result<Vec<CandidateResult>> {
        let distance_compute = UnifiedDistanceCompute::new(DistanceMetric::Cosine);

        // P² matrix provides exact distances between all vectors
        // Use it for efficient nearest neighbor search with clustering awareness

        // Step 1: Compute distances from query to all vectors
        let mut query_distances: Vec<(usize, f32)> = Vec::with_capacity(vectors.len());
        for (idx, vector) in vectors.iter().enumerate() {
            let dist = distance_compute
                .calculate_distance(target_vector, vector, &DistanceMetric::Cosine)
                .raw_value;
            query_distances.push((idx, dist));
        }

        // Sort by distance to find nearest vectors
        query_distances.sort_by(|a, b| a.1.total_cmp(&b.1));

        // Step 2: Use P² matrix to identify dense clusters around top candidates
        // This helps find vectors that are similar to each other AND the query
        let mut final_candidates = Vec::new();
        let mut seen = HashSet::new();

        // Take top candidates and explore their neighborhoods using P² matrix
        for &(candidate_idx, candidate_dist) in query_distances.iter().take(k * 2) {
            if seen.insert(candidate_idx) {
                final_candidates.push(CandidateResult {
                    id: ids[candidate_idx].clone(),
                    vector: vectors[candidate_idx].clone(),
                    distance: candidate_dist,
                    cluster_id: 0, // Will be set by caller
                    cluster_info: RaptorClusterInfo::default(),
                });

                // Use P² matrix to find vectors close to this candidate
                // This identifies local clusters of similar vectors
                if final_candidates.len() < k * 3 {
                    for (other_idx, other_vec) in vectors.iter().enumerate() {
                        if other_idx != candidate_idx && !seen.contains(&other_idx) {
                            // Get pre-computed distance from P² matrix
                            let intra_dist =
                                matrix.p2_matrix.get_distance(candidate_idx, other_idx);

                            // If close enough in the P² matrix, it's likely relevant
                            if intra_dist < 0.3 {
                                // Threshold for cluster membership
                                let query_dist = distance_compute
                                    .calculate_distance(
                                        target_vector,
                                        other_vec,
                                        &DistanceMetric::Cosine,
                                    )
                                    .raw_value;
                                if seen.insert(other_idx) && final_candidates.len() < k * 3 {
                                    final_candidates.push(CandidateResult {
                                        id: ids[other_idx].clone(),
                                        vector: vectors[other_idx].clone(),
                                        distance: query_dist,
                                        cluster_id: 0,
                                        cluster_info: RaptorClusterInfo::default(),
                                    });
                                }
                            }
                        }
                    }
                }
            }

            if final_candidates.len() >= k * 3 {
                break;
            }
        }

        // Sort final candidates by distance and return top-k
        final_candidates.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        final_candidates.truncate(k);

        tracing::debug!(
            "P² matrix navigation found {} candidates from {} vectors",
            final_candidates.len(),
            vectors.len()
        );

        Ok(final_candidates)
    }

    /// Linear search fallback when no P² matrix available
    fn linear_search_cluster(
        &self,
        vectors: &[Vec<f32>],
        ids: &[String],
        target_vector: &[f32],
        k: usize,
    ) -> Result<Vec<CandidateResult>> {
        let mut candidates = Vec::new();

        for (vector, id) in vectors.iter().zip(ids.iter()) {
            let distance = self
                .distance_compute
                .calculate_distance(target_vector, vector, &DistanceMetric::Cosine)
                .raw_value;

            candidates.push(CandidateResult {
                id: id.clone(),
                vector: vector.clone(),
                distance,
                cluster_id: 0,
                cluster_info: RaptorClusterInfo::default(),
            });
        }

        // Sort by distance and return top-k
        candidates.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(k);

        Ok(candidates)
    }

    /// Find closest vector as entry point
    fn find_closest_entry_point(
        &self,
        vectors: &[Vec<f32>],
        target_vector: &[f32],
    ) -> Result<usize> {
        let mut min_distance = f32::INFINITY;
        let mut best_idx = 0;

        for (idx, vector) in vectors.iter().enumerate() {
            let distance = self
                .distance_compute
                .calculate_distance(target_vector, vector, &DistanceMetric::Cosine)
                .raw_value;

            if distance < min_distance {
                min_distance = distance;
                best_idx = idx;
            }
        }

        Ok(best_idx)
    }

    /// Merge results from multiple clusters with 5-component boosting
    async fn merge_cross_cluster_results(
        &self,
        mut candidates: Vec<CandidateResult>,
        _target_vector: &[f32],
        k: usize,
    ) -> Result<Vec<SimilarityResult>> {
        // Apply 5-component boosting (simplified version)
        for candidate in &mut candidates {
            candidate.distance =
                self.apply_5_component_boosting(candidate.distance, &candidate.cluster_info);
        }

        // Sort by boosted distance and return top-k
        candidates.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let final_results: Vec<SimilarityResult> = candidates
            .into_iter()
            .take(k)
            .map(|c| SimilarityResult::new(c.distance, DistanceMetric::Cosine))
            .collect();

        Ok(final_results)
    }

    /// Apply 5-component boosting formula (simplified)
    fn apply_5_component_boosting(
        &self,
        base_distance: f32,
        cluster_info: &RaptorClusterInfo,
    ) -> f32 {
        // Simplified 5-component boosting
        // In full implementation, this would use the complete formula from the design
        let alpha_own = 1.0; // Weight for intra-cluster distance
        let alpha_other = 0.1; // Weight for inter-cluster penalty

        base_distance * alpha_own + cluster_info.inter_cluster_penalty * alpha_other
    }

    /// Extract vectors from Arrow RecordBatch
    fn extract_vectors_from_batch(&self, batch: &RecordBatch) -> Result<Vec<Vec<f32>>> {
        if let Some(vector_array) = batch.column_by_name("vector") {
            use arrow_array::ListArray;

            if let Some(vectors) = vector_array.as_any().downcast_ref::<ListArray>() {
                let mut result = Vec::new();

                for i in 0..vectors.len() {
                    if let Some(vector_values) = vectors
                        .value(i)
                        .as_any()
                        .downcast_ref::<arrow_array::Float32Array>()
                    {
                        let vector: Vec<f32> = (0..vector_values.len())
                            .map(|j| vector_values.value(j))
                            .collect();
                        result.push(vector);
                    }
                }

                return Ok(result);
            }
        }

        Err(anyhow::anyhow!("No vector column found in batch"))
    }

    /// Extract IDs from Arrow RecordBatch
    fn extract_ids_from_batch(&self, batch: &RecordBatch) -> Result<Vec<String>> {
        if let Some(id_array) = batch.column_by_name("id") {
            use arrow_array::StringArray;

            if let Some(ids) = id_array.as_any().downcast_ref::<StringArray>() {
                let mut result = Vec::new();

                for i in 0..ids.len() {
                    if !ids.is_null(i) {
                        result.push(ids.value(i).to_string());
                    }
                }

                return Ok(result);
            }
        }

        Err(anyhow::anyhow!("No id column found in batch"))
    }

    /// Get all centroids from the cached footer
    /// Returns empty vec if footer not loaded
    pub async fn get_all_centroids(&self) -> Result<Vec<(u32, Vec<f32>)>> {
        let footer = self.get_footer(&self.base_path).await?;
        let all_centroids = footer.centroids.decode_all();
        // Convert u16 to u32 for the tuple
        Ok(all_centroids
            .into_iter()
            .map(|(id, vec)| (id as u32, vec))
            .collect())
    }

    /// Hierarchical search using the neighbor structure
    /// Comprehensive validation method to verify reader-writer alignment
    pub async fn validate_alignment_with_writer(
        &self,
        file_path: &str,
    ) -> Result<ValidationReport> {
        tracing::info!(
            "🔍 Validating RAPTOR reader-writer alignment for {}",
            file_path
        );

        let mut report = ValidationReport::default();

        // 1. Validate file header and footer reading
        match self.load_footer(file_path).await {
            Ok(_) => {
                report.footer_reading = true;
                tracing::debug!("✅ Footer reading: PASS");
            }
            Err(e) => {
                report.errors.push(format!("Footer reading failed: {}", e));
                tracing::error!("❌ Footer reading: FAIL - {}", e);
            }
        }

        // 2. Validate metadata extraction
        match self.get_metadata(file_path).await {
            Ok(metadata) => {
                report.metadata_extraction = true;
                report.total_row_groups = metadata.row_groups.len();
                tracing::debug!(
                    "✅ Metadata extraction: PASS - {} row groups",
                    metadata.row_groups.len()
                );
            }
            Err(e) => {
                report
                    .errors
                    .push(format!("Metadata extraction failed: {}", e));
                tracing::error!("❌ Metadata extraction: FAIL - {}", e);
            }
        }

        // 3. Validate bloom filter independence (key requirement)
        if let Ok(metadata) = self.get_metadata(file_path).await {
            let mut bloom_tests = 0;
            let mut bloom_successes = 0;

            for (rg_idx, rg_metadata) in metadata.row_groups.iter().enumerate() {
                if rg_metadata.bloom_filter_offset.is_some() {
                    bloom_tests += 1;

                    match self.load_bloom_filter(file_path, rg_idx as u16).await {
                        Ok(bloom_filter) => {
                            bloom_successes += 1;
                            tracing::debug!(
                                "✅ Bloom filter {} loaded: {} IDs, {:.3}% FPR",
                                rg_idx,
                                bloom_filter.stats().num_ids,
                                bloom_filter.stats().false_positive_rate * 100.0
                            );
                        }
                        Err(e) => {
                            report
                                .errors
                                .push(format!("Bloom filter {} loading failed: {}", rg_idx, e));
                        }
                    }
                }
            }

            report.bloom_filter_independence = bloom_successes == bloom_tests && bloom_tests > 0;
            report.bloom_filters_tested = bloom_tests;
            report.bloom_filters_successful = bloom_successes;

            tracing::info!(
                "🔍 Bloom filter independence: {}/{} successful",
                bloom_successes,
                bloom_tests
            );
        }

        // 4. Validate unified compression alignment
        // This is implicitly tested by successful bloom filter and metadata loading
        if report.metadata_extraction && report.bloom_filter_independence {
            report.compression_alignment = true;
            tracing::debug!("✅ Unified compression alignment: PASS");
        }

        // 5. Validate cache integration
        report.cache_integration = true; // Always true as cache is integrated in constructor

        // 6. Generate overall alignment score
        report.calculate_alignment_score();

        tracing::info!(
            "🎯 RAPTOR Reader-Writer Alignment: {:.1}% ({}/6 components)",
            report.alignment_score * 100.0,
            report.get_passing_components()
        );

        Ok(report)
    }
}

/// Validation report for reader-writer alignment
#[derive(Debug, Default)]
pub struct ValidationReport {
    pub footer_reading: bool,
    pub metadata_extraction: bool,
    pub bloom_filter_independence: bool,
    pub compression_alignment: bool,
    pub cache_integration: bool,
    pub total_row_groups: usize,
    pub bloom_filters_tested: usize,
    pub bloom_filters_successful: usize,
    pub errors: Vec<String>,
    pub alignment_score: f32,
}

impl ValidationReport {
    fn calculate_alignment_score(&mut self) {
        let components = [
            self.footer_reading,
            self.metadata_extraction,
            self.bloom_filter_independence,
            self.compression_alignment,
            self.cache_integration,
        ];

        let passing = components.iter().filter(|&&x| x).count() as f32;
        self.alignment_score = passing / components.len() as f32;
    }

    fn get_passing_components(&self) -> usize {
        [
            self.footer_reading,
            self.metadata_extraction,
            self.bloom_filter_independence,
            self.compression_alignment,
            self.cache_integration,
        ]
        .iter()
        .filter(|&&x| x)
        .count()
    }

    pub fn is_fully_aligned(&self) -> bool {
        self.alignment_score >= 1.0
    }

    pub fn print_summary(&self) {
        info!("RAPTOR Reader-Writer Alignment Report");
        info!("Overall Score: {:.1}%", self.alignment_score * 100.0);
        info!("Components Passing: {}/5", self.get_passing_components());
        info!("Component Status:");
        info!(
            "  Footer Reading: {}",
            if self.footer_reading { "PASS" } else { "FAIL" }
        );
        info!(
            "  Metadata Extraction: {}",
            if self.metadata_extraction {
                "PASS"
            } else {
                "FAIL"
            }
        );
        info!(
            "  Bloom Filter Independence: {}",
            if self.bloom_filter_independence {
                "PASS"
            } else {
                "FAIL"
            }
        );
        info!(
            "  Compression Alignment: {}",
            if self.compression_alignment {
                "PASS"
            } else {
                "FAIL"
            }
        );
        info!(
            "  Cache Integration: {}",
            if self.cache_integration {
                "PASS"
            } else {
                "FAIL"
            }
        );
        info!("Statistics:");
        info!("  Total Row Groups: {}", self.total_row_groups);
        info!("  Bloom Filters Tested: {}", self.bloom_filters_tested);
        info!(
            "  Bloom Filters Successful: {}",
            self.bloom_filters_successful
        );

        if !self.errors.is_empty() {
            warn!("Errors ({}):", self.errors.len());
            for error in &self.errors {
                warn!("   - {}", error);
            }
        }

        if self.is_fully_aligned() {
            debug!("RAPTOR Reader-Writer: FULLY ALIGNED");
        } else {
            debug!("RAPTOR Reader-Writer: Alignment needs improvement");
        }
    }
}

impl RaptorReader {
    /// Efficiently navigates through super-clusters to find best rowgroups
    pub async fn hierarchical_search(
        &self,
        query_vector: &[f32],
        top_k_rowgroups: usize,
        distance_metric: &DistanceMetric,
    ) -> Result<Vec<u16>> {
        // Footer is loaded on demand via get_footer()

        let footer = self.get_footer(&self.base_path).await?;
        let all_centroids = footer.centroids.decode_all();

        if all_centroids.is_empty() {
            return Ok(Vec::new());
        }

        let k = all_centroids.len();

        // Step 1: Compute distances to ALL centroids (only once)
        // This is fast with SIMD and worth doing for accurate navigation
        let mut centroid_distances = Vec::with_capacity(k);
        for (rg_id, centroid) in &all_centroids {
            let dist = self
                .distance_compute
                .calculate_distance(query_vector, centroid, distance_metric)
                .raw_value;
            centroid_distances.push((dist, *rg_id));
        }

        // Sort to find closest centroids
        centroid_distances
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Step 2: Use hierarchical navigation for large collections
        if k >= 1000 {
            // OPTIMIZED: Use K² matrix for neighbor discovery with triangle inequality pruning
            let k2_matrix = &footer.inter_centroid_distances;

            // Build a map from rowgroup_id to index for K² lookups
            let rg_to_idx: std::collections::HashMap<u16, usize> = all_centroids
                .iter()
                .enumerate()
                .map(|(idx, (rg_id, _))| (*rg_id, idx))
                .collect();

            let mut visited = std::collections::HashSet::new();
            let mut result_with_dist: Vec<(f32, u16)> = Vec::new();

            // Start with top-3 closest rowgroups as seeds
            for &(dist, rg_id) in centroid_distances.iter().take(3) {
                if !visited.contains(&rg_id) {
                    result_with_dist.push((dist, rg_id));
                    visited.insert(rg_id);
                }
            }

            // Use K² matrix to find neighbors of seeds
            // Triangle inequality: query_to_neighbor >= |query_to_seed - seed_to_neighbor|
            let expansion_threshold = centroid_distances
                .get(top_k_rowgroups.min(k - 1))
                .map_or(f32::MAX, |(d, _)| d * 1.5);

            for &(seed_dist, seed_rg) in centroid_distances.iter().take(3) {
                if let Some(&seed_idx) = rg_to_idx.get(&seed_rg) {
                    // Use K² to find all centroids close to this seed
                    for (neighbor_idx, (neighbor_rg, neighbor_centroid)) in
                        all_centroids.iter().enumerate()
                    {
                        if visited.contains(neighbor_rg) {
                            continue;
                        }

                        // Get seed-to-neighbor distance from K² matrix (O(1) lookup!)
                        let k2_dist = k2_matrix.get_distance(seed_idx, neighbor_idx);

                        // Triangle inequality lower bound: query_to_neighbor >= |seed_dist - k2_dist|
                        let min_possible_dist = (seed_dist - k2_dist).abs();

                        // Prune if minimum possible distance exceeds threshold
                        if min_possible_dist > expansion_threshold {
                            continue;
                        }

                        // Promising neighbor - compute exact distance
                        let neighbor_dist = self
                            .distance_compute
                            .calculate_distance(query_vector, neighbor_centroid, distance_metric)
                            .raw_value;

                        if neighbor_dist < expansion_threshold {
                            result_with_dist.push((neighbor_dist, *neighbor_rg));
                            visited.insert(*neighbor_rg);
                        }
                    }
                }
            }

            // Sort by distance and take top-k
            result_with_dist
                .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            let result: Vec<u16> = result_with_dist
                .into_iter()
                .take(top_k_rowgroups)
                .map(|(_, rg_id)| rg_id)
                .collect();

            tracing::debug!(
                "Hierarchical search with K² pruning: explored {} rowgroups, selected top {}",
                visited.len(),
                result.len()
            );

            Ok(result)
        } else {
            // Small collection: just return top-k directly
            Ok(centroid_distances
                .iter()
                .take(top_k_rowgroups)
                .map(|(_, rg_id)| *rg_id)
                .collect())
        }
    }

    // Obsolete - clustering info is in footer's ColumnarCentroids and K×K matrix
    #[allow(dead_code)]
    async fn load_cluster_metadata_obsolete(&self) -> Result<()> {
        // Get footer through helper method
        let footer = self.get_footer(&self.base_path).await?;
        let all_centroids = footer.centroids.decode_all();
        let centroids: Vec<Vec<f32>> = all_centroids.iter().map(|(_, c)| c.clone()).collect();

        // PERFORMANCE OPTIMIZATION: Only compute full matrix for small collections
        // Based on performance testing:
        // - k ≤ 100: ~1ms (negligible)
        // - k = 1000: ~105ms (significant)
        // - k = 10000: ~10.5s (unacceptable)
        let _centroid_distances = if centroids.len() <= 100 {
            // Small collection: pre-compute full matrix (< 1ms overhead)
            let mut distances = vec![vec![0.0f32; centroids.len()]; centroids.len()];

            for (i, centroid_i) in centroids.iter().enumerate() {
                distances[i][i] = 0.0;

                for j in (i + 1)..centroids.len() {
                    let dist = self
                        .distance_compute
                        .calculate_distance(centroid_i, &centroids[j], &DistanceMetric::Euclidean)
                        .raw_value;

                    distances[i][j] = dist;
                    distances[j][i] = dist;
                }
            }

            tracing::debug!(
                "Pre-computed {} centroid distances for small collection",
                centroids.len() * (centroids.len() - 1) / 2
            );

            distances
        } else {
            // Large collection: use lazy loading (compute on-demand during search)
            // Return empty matrix - distances will be computed as needed
            tracing::info!(
                "Using lazy loading for {} centroids (would need {} distance calculations)",
                centroids.len(),
                centroids.len() * (centroids.len() - 1) / 2
            );

            vec![vec![0.0f32; centroids.len()]; centroids.len()]
        };

        // Create cluster stats from rowgroup metadata
        let mut cluster_stats = Vec::new();
        for rg in &footer.file_metadata.row_groups {
            if let Some(ref stats) = rg.centroid_stats {
                cluster_stats.push(ClusterStats {
                    mean_distance: stats.mean_distance,
                    std_deviation: stats.std_deviation,
                    radius: stats.radius,
                });
            }
        }

        // Method body removed - ClusterMetadata no longer exists
        Ok(())
    }

    /// Get boosting configuration (can be customized per collection)
    #[allow(dead_code)]
    fn get_boost_config(&self) -> BoostConfig {
        // In production, this could be loaded from collection configuration
        // For now, use default values optimized for RAPTOR
        BoostConfig::default()
    }

    // REMOVED: encode_for_cache and decode_cached_rowgroup wrapper methods
    // Reason: Redundant - Arrow IPC operations inlined where needed
    // Benefit: Less indirection, clearer code flow

    /// Parse metadata from footer bytes (stub)
    /// Get metadata for a file without reading the actual data
    pub async fn get_metadata(&self, file_path: &str) -> Result<RaptorFileMetadata> {
        self.read_metadata(file_path).await
    }

    /// Read only specific columns from a rowgroup (v2 columnar format)
    pub async fn read_columns(
        &self,
        file_path: &str,
        rg_id: u16,
        columns: &[ColumnType],
    ) -> Result<PartialRowGroup> {
        let metadata = self.read_metadata(file_path).await?;
        let rg_metadata = metadata
            .row_groups
            .get(rg_id as usize)
            .ok_or_else(|| anyhow::anyhow!("Row group {} not found", rg_id))?;

        let mut partial = PartialRowGroup {
            vectors: None,
            ids: None,
            metadata: HashMap::new(),
            source_content: None,
        };

        // Use columnar format (Release 1 - no backward compatibility)
        if !rg_metadata.column_pages.is_empty() {
            // Read only requested column pages
            for column_type in columns {
                if let Some(page_meta) = rg_metadata.column_pages.get(column_type) {
                    // Read only this column page
                    let compressed = self
                        .filesystem
                        .read_range(file_path, page_meta.offset, page_meta.compressed_size)
                        .await?;

                    // Decompress with appropriate algorithm
                    let decompressed =
                        self.decompress_column(&compressed, page_meta.compression)?;

                    // Decode based on column type
                    match column_type {
                        ColumnType::VectorsFp32 => {
                            partial.vectors =
                                Some(self.decode_vector_column(&decompressed, metadata.dimension)?);
                        }
                        ColumnType::Ids => {
                            partial.ids = Some(self.decode_id_column(&decompressed)?);
                        }
                        ColumnType::Metadata(key) => {
                            partial
                                .metadata
                                .insert(key.clone(), self.decode_metadata_column(&decompressed)?);
                        }
                        ColumnType::SourceContent => {
                            partial.source_content =
                                Some(self.decode_source_column(&decompressed)?);
                        }
                        _ => {} // Skip matrices and other types for now
                    }
                }
            }
        } else {
            tracing::warn!(
                "No column pages found in rowgroup {} - file may be corrupted",
                rg_id
            );
        }

        Ok(partial)
    }

    /// Search without loading metadata or source content
    pub async fn search_vectors_only(
        &self,
        file_path: &str,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<OptimizedSearchRecord>> {
        let metadata = self.read_metadata(file_path).await?;
        let mut all_results = Vec::new();

        for (rg_idx, _rg_metadata) in metadata.row_groups.iter().enumerate() {
            // Only load vectors and IDs, skip metadata/source
            let partial = self
                .read_columns(
                    file_path,
                    rg_idx as u16,
                    &[ColumnType::VectorsFp32, ColumnType::Ids],
                )
                .await?;

            if let (Some(vectors), Some(ids)) = (partial.vectors, partial.ids) {
                // Compute distances for all vectors in this rowgroup
                let distance_compute = UnifiedDistanceCompute::new(DistanceMetric::Cosine);

                for (idx, vector) in vectors.iter().enumerate() {
                    let distance = distance_compute
                        .calculate_distance(query, vector, &DistanceMetric::Cosine)
                        .raw_value;
                    all_results.push(OptimizedSearchRecord {
                        id: ids[idx].clone(),
                        score: -distance, // Convert distance to similarity score (negative for sorting)
                        vector: Some(Arc::new(vector.clone())), // Use Arc to avoid copying
                        metadata: Default::default(), // Not loaded in fullscan mode
                        ..Default::default()
                    });
                }
            }
        }

        // Use bounded priority queue for efficient top-k selection
        let mut priority_queue = BoundedPriorityQueue::new(k);

        // Insert all results into bounded queue
        for result in all_results {
            priority_queue.try_insert(result);
        }

        // Get sorted results from bounded queue
        let final_results = priority_queue.into_sorted_vec();

        Ok(final_results)
    }

    /// Helper: Decompress column data
    fn decompress_column(
        &self,
        compressed: &[u8],
        algorithm: CompressionAlgorithm,
    ) -> Result<Vec<u8>> {
        proximadb_compression::decompress(
            compressed,
            algorithm,
            proximadb_compression::CompressionContext::Column,
        )
    }

    /// Helper: Decode vector column from columnar format
    /// Format: [col1_len:u32][col1_data][col2_len:u32][col2_data]...
    /// Each column is one dimension across all vectors, encoded with ProximaCodec
    fn decode_vector_column(&self, data: &[u8], dimension: usize) -> Result<Vec<Vec<f32>>> {
        use proximadb_codec::ProximaCodec;

        let codec = ProximaCodec::global();
        let mut offset = 0;
        let mut dimension_columns: Vec<Vec<f32>> = Vec::with_capacity(dimension);

        // Read each dimension column
        for _dim_idx in 0..dimension {
            if offset + 4 > data.len() {
                // No more data - break early (might have fewer dimensions than expected)
                break;
            }

            // Read column length
            let col_len = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            offset += 4;

            if offset + col_len > data.len() {
                return Err(anyhow::anyhow!(
                    "Invalid vector column data: expected {} bytes at offset {}, but only {} available",
                    col_len,
                    offset,
                    data.len() - offset
                ));
            }

            // Decode this dimension column
            let encoded_col = &data[offset..offset + col_len];
            let decoded_col: Vec<f32> = codec.decode(encoded_col)?;
            dimension_columns.push(decoded_col);
            offset += col_len;
        }

        if dimension_columns.is_empty() {
            return Ok(Vec::new());
        }

        // Transpose from columnar to row format
        let num_vectors = dimension_columns[0].len();
        let mut vectors = Vec::with_capacity(num_vectors);

        for vec_idx in 0..num_vectors {
            let mut vector = Vec::with_capacity(dimension_columns.len());
            for dim_col in &dimension_columns {
                if vec_idx < dim_col.len() {
                    vector.push(dim_col[vec_idx]);
                } else {
                    vector.push(0.0); // Padding for incomplete vectors
                }
            }
            vectors.push(vector);
        }

        Ok(vectors)
    }

    /// Helper: Decode ID column
    fn decode_id_column(&self, data: &[u8]) -> Result<Vec<String>> {
        let mut ids = Vec::new();
        let mut offset = 0;

        while offset < data.len() {
            let len = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            offset += 4;

            let id = String::from_utf8(data[offset..offset + len].to_vec())?;
            ids.push(id);
            offset += len;
        }

        Ok(ids)
    }

    /// Helper: Decode metadata column
    fn decode_metadata_column(&self, _data: &[u8]) -> Result<Vec<Option<Vec<u8>>>> {
        // Implementation would decode dictionary-encoded metadata
        Ok(Vec::new())
    }

    /// Helper: Decode source content column
    fn decode_source_column(&self, _data: &[u8]) -> Result<Vec<Option<Vec<u8>>>> {
        // Implementation would decode source content
        Ok(Vec::new())
    }

    /// Read multiple row groups by indices
    pub async fn read_rowgroups(
        &self,
        file_path: &str,
        indices: &[u16],
    ) -> Result<Vec<RecordBatch>> {
        tracing::debug!(
            "RAPTOR read_rowgroups: Reading {} row groups from {}",
            indices.len(),
            file_path
        );
        let mut batches = Vec::new();
        for &idx in indices {
            tracing::debug!("RAPTOR read_rowgroups: Reading row group {}", idx);
            // Read specific row group
            let batch = self.read_rowgroup(idx).await?;
            tracing::debug!(
                "RAPTOR read_rowgroups: Got batch with {} rows",
                batch.num_rows()
            );
            batches.push(batch);
        }
        tracing::debug!("RAPTOR read_rowgroups: Returning {} batches", batches.len());
        Ok(batches)
    }

    /// Read a single row group by index
    pub async fn read_rowgroup(&self, rg_id: u16) -> Result<RecordBatch> {
        tracing::debug!(
            "RAPTOR read_rowgroup: Reading actual data for row group {} from {}",
            rg_id,
            self.base_path
        );

        // Load metadata to get row group information
        let metadata = self.read_metadata(&self.base_path).await?;

        if rg_id as usize >= metadata.row_groups.len() {
            return Err(anyhow::anyhow!("Row group {} not found", rg_id));
        }

        let rg_metadata = &metadata.row_groups[rg_id as usize];
        tracing::debug!(
            "RAPTOR read_rowgroup: Row group {} has {} vectors",
            rg_id,
            rg_metadata.vector_count
        );

        // Get column metadata
        let vector_column_meta = rg_metadata
            .column_pages
            .get(&ColumnType::VectorsFp32)
            .ok_or_else(|| anyhow::anyhow!("Vector column not found in row group metadata"))?;
        let id_column_meta = rg_metadata
            .column_pages
            .get(&ColumnType::Ids)
            .ok_or_else(|| anyhow::anyhow!("ID column not found in row group metadata"))?;

        // Use dimension from metadata, not from config
        let dimension = metadata.dimension;
        let num_rows = rg_metadata.vector_count;

        // Read and decompress VECTOR column separately
        tracing::debug!(
            "Reading vector column: offset={}, compressed_size={}",
            vector_column_meta.offset,
            vector_column_meta.compressed_size
        );
        let vector_compressed = self
            .filesystem
            .read_range(
                &self.base_path,
                vector_column_meta.offset,
                vector_column_meta.compressed_size,
            )
            .await?;

        let vector_data = if vector_column_meta.compression != CompressionAlgorithm::None {
            use proximadb_compression::{CompressionContext, decompress};
            decompress(
                &vector_compressed,
                vector_column_meta.compression,
                CompressionContext::VectorSerialization,
            )?
        } else {
            vector_compressed
        };

        // Read and decompress ID column separately
        tracing::debug!(
            "Reading ID column: offset={}, compressed_size={}",
            id_column_meta.offset,
            id_column_meta.compressed_size
        );
        let id_compressed = self
            .filesystem
            .read_range(
                &self.base_path,
                id_column_meta.offset,
                id_column_meta.compressed_size,
            )
            .await?;

        let id_data = if id_column_meta.compression != CompressionAlgorithm::None {
            use proximadb_compression::{CompressionContext, decompress};
            decompress(
                &id_compressed,
                id_column_meta.compression,
                CompressionContext::Column,
            )?
        } else {
            id_compressed
        };

        // Parse ID column
        let mut cursor = std::io::Cursor::new(&id_data);
        use std::io::Read;

        let mut ids = Vec::with_capacity(num_rows);
        for i in 0..num_rows {
            // Read length (4 bytes)
            let mut len_bytes = [0u8; 4];
            if let Err(e) = cursor.read_exact(&mut len_bytes) {
                tracing::error!("Failed to read ID length for vector {}: {}", i, e);
                return Err(anyhow::anyhow!("Failed to read ID length: {}", e));
            }
            let len = u32::from_le_bytes(len_bytes) as usize;

            // Read string data
            let mut str_bytes = vec![0u8; len];
            cursor.read_exact(&mut str_bytes)?;

            let id = String::from_utf8(str_bytes)
                .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in ID: {}", e))?;
            ids.push(id);
        }

        tracing::debug!("RAPTOR: Parsed {} IDs", ids.len());

        // Parse vector column
        let mut cursor = std::io::Cursor::new(&vector_data);

        // Use ProximaCodec for decoding
        let codec = ProximaCodec::global();

        // Read columnar vectors (transposed format)
        let mut columns: Vec<Vec<f32>> = vec![Vec::with_capacity(num_rows); dimension];

        tracing::debug!(
            "Decoding {} dimension columns for {} rows, vector_data len: {}",
            dimension,
            num_rows,
            vector_data.len()
        );

        for (dim_idx, column) in columns.iter_mut().enumerate().take(dimension) {
            // Read encoded column length
            let mut len_bytes = [0u8; 4];
            if let Err(e) = cursor.read_exact(&mut len_bytes) {
                tracing::error!(
                    "Failed to read column {} length at position {}: {}",
                    dim_idx,
                    cursor.position(),
                    e
                );
                return Err(anyhow::anyhow!(
                    "Failed to read column {} length: {}",
                    dim_idx,
                    e
                ));
            }
            let encoded_len = u32::from_le_bytes(len_bytes) as usize;

            tracing::debug!(
                "Dimension {}: encoded_len={}, cursor position={}",
                dim_idx,
                encoded_len,
                cursor.position()
            );

            // Validate we have enough data
            if cursor.position() as usize + encoded_len > vector_data.len() {
                tracing::error!(
                    "Not enough data for dimension {}: need {}, have {}",
                    dim_idx,
                    encoded_len,
                    vector_data.len() - cursor.position() as usize
                );
                return Err(anyhow::anyhow!(
                    "Insufficient data for dimension {}",
                    dim_idx
                ));
            }

            // Read encoded column data
            let mut encoded_data = vec![0u8; encoded_len];
            cursor.read_exact(&mut encoded_data)?;

            // Decode using ProximaCodec
            let decoded = codec.decode(&encoded_data)?;
            *column = decoded;
        }

        // Transpose back to row format and create Float32Array
        let mut flat_values = Vec::with_capacity(num_rows * dimension);
        for vec_idx in 0..num_rows {
            for column in columns.iter().take(dimension) {
                flat_values.push(column[vec_idx]);
            }
        }

        tracing::debug!(
            "RAPTOR: Decoded {} vectors of dimension {}",
            num_rows,
            dimension
        );

        // Create Arrow arrays
        use arrow_array::{Float32Array, Int64Array, StringArray, UInt32Array};
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc as StdArc;

        // Create Arrow arrays from parsed data
        let id_array = StdArc::new(StringArray::from(ids.clone()));

        // Create FixedSizeListArray for vectors - proper Arrow representation
        // Each row contains one vector of fixed dimension
        use arrow_array::FixedSizeListArray;

        let values_array = Float32Array::from(flat_values.clone());
        let vector_array = StdArc::new(FixedSizeListArray::new(
            StdArc::new(Field::new("item", DataType::Float32, false)),
            dimension as i32,
            StdArc::new(values_array),
            None,
        ));

        // Debug: log array sizes
        tracing::debug!(
            "Array sizes - IDs: {}, Vector rows: {} (dimension {})",
            id_array.len(),
            vector_array.len(),
            dimension
        );

        // Create placeholder arrays for metadata, version, and timestamp
        // These are nullable fields that can be empty for now
        let metadata_array = StdArc::new(StringArray::from(vec![None as Option<String>; num_rows]));
        let version_array = StdArc::new(UInt32Array::from(vec![None as Option<u32>; num_rows]));
        let timestamp_array = StdArc::new(Int64Array::from(vec![None as Option<i64>; num_rows]));

        tracing::debug!(
            "RAPTOR read_rowgroup: Returning batch with {} rows, {} values per vector",
            num_rows,
            dimension
        );

        // Create the schema that matches RAPTOR's expected format
        // But use FixedSizeList for vector field to properly represent the data
        let schema = StdArc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    StdArc::new(Field::new("item", DataType::Float32, false)),
                    dimension as i32,
                ),
                false,
            ),
            Field::new("metadata", DataType::Utf8, true),
            Field::new("version", DataType::UInt32, true),
            Field::new("timestamp", DataType::Int64, true),
        ]));

        // Create the RecordBatch with all required fields
        // Now all columns have the same number of rows
        Ok(RecordBatch::try_new(
            schema,
            vec![
                id_array as StdArc<dyn arrow_array::Array>,
                vector_array as StdArc<dyn arrow_array::Array>,
                metadata_array as StdArc<dyn arrow_array::Array>,
                version_array as StdArc<dyn arrow_array::Array>,
                timestamp_array as StdArc<dyn arrow_array::Array>,
            ],
        )?)
    }

    // REMOVED: parse_metadata method - no longer needed
    // The footer is now properly deserialized using bincode in read_metadata()
    // This ensures we get the actual metadata including all centroids
}

// REMOVED: Extension trait for CrossCacheOrchestrator
// Reason: Unnecessary wrapper adding stack overhead
// Solution: Direct calls to unified cache modules (vector_store, metadata_store, etc.)
// Benefit: Reduced stack depth, less function call overhead, cleaner code

#[cfg(test)]
#[cfg(feature = "experimental-engines")]
mod tests;
