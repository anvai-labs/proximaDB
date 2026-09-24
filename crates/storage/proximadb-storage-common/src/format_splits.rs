//! # File Split Abstraction for Parallel Reading
//!
//! Provides a unified split abstraction for parallel scanning across all
//! ProximaDB storage engines (SST, HELIX, SWIFT, NOVA, VIPER, RAPTOR).
//!
//! FileSplits represent independent units of work that can be processed
//! in parallel, enabling efficient distributed query execution.
//!
//! ## Features
//!
//! - **Scalar Pruning**: Min/max statistics for predicate evaluation
//! - **Vector Pruning**: Centroid-based distance bounds for vector search
//! - **Bloom Filters**: Membership testing for equality predicates
//! - **Locality Hints**: Preferred locations for data-local scheduling
//!
//! ## Usage
//!
//! ```rust,ignore
//! let split = FileSplit::new_block("/data/file.sst".to_string(), 0, 0, 1024, 100);
//!
//! // Check if split can be pruned for scalar predicate
//! if split.can_prune_scalar("price", &ScalarPredicate::GreaterThan(100.0)) {
//!     // Skip this split - no matching rows
//! }
//!
//! // Check if split can be pruned for vector search
//! if split.can_prune_vector(&query_vector, 0.5) {
//!     // Skip this split - no vectors within distance threshold
//! }
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// A split represents an independent unit of work for parallel reading.
/// Each split can be processed by a separate thread/task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSplit {
    /// Unique identifier for this split
    pub split_id: String,
    /// File path containing this split's data
    pub file_path: String,
    /// Byte offset within the file
    pub offset: u64,
    /// Byte length of this split
    pub length: u64,
    /// Split type (determines how to read this split)
    pub split_type: SplitType,
    /// Statistics for query optimization
    pub statistics: SplitStatistics,
    /// Locality hints for scheduling
    pub locality: SplitLocality,
    /// Total byte size of the OBJECT this split lives in, when the planner
    /// already knows it (TD-USUB-8).
    ///
    /// Deliberately separate from [`Self::length`], which means different things
    /// per producer: the PAX segment locators set it to the whole-file size
    /// (their splits are whole-file), while the SST planner sets it to
    /// `file.block_size` per block. Reinterpreting `length` as the object size
    /// would therefore make a footer-tail computation read the wrong bytes on the
    /// SST path — garbage or a spurious error, not a clean failure.
    ///
    /// Object-storage readers need the size before they can locate a trailing
    /// footer, and today they pay a `HEAD` for it even when the planner obtained
    /// it for free from `list()`. Carrying it here removes one of the four
    /// dependent round trips in the cold-read chain, which is the DEPTH term
    /// ADR-033 ranks dominant. `None` ⇒ the reader falls back to the `HEAD`.
    #[serde(default)]
    pub object_size: Option<u64>,
}

/// Type of split - varies by storage engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SplitType {
    /// SST: Block-based split
    Block { block_id: u32, record_count: u64 },
    /// HELIX: Hilbert curve range
    HilbertRange {
        start_code: u64,
        end_code: u64,
        hilbert_order: u8,
    },
    /// SWIFT: SuperBlock (coarse) or DataBlock (fine)
    SuperBlock {
        superblock_id: u32,
        block_count: usize,
        block_ids: Vec<u32>,
    },
    /// NOVA/VIPER/RAPTOR: Parquet row group
    RowGroup {
        row_group_index: usize,
        row_count: i64,
    },
    /// RAPTOR: Z-order curve range
    ZOrderRange { start_code: u64, end_code: u64 },
    /// Generic: Byte range
    ByteRange { estimated_records: usize },
}

/// Statistics for split pruning and optimization
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SplitStatistics {
    /// Row count in this split
    pub row_count: Option<u64>,
    /// Byte size of this split
    pub byte_size: Option<u64>,
    /// Column statistics (for predicate pushdown)
    pub column_stats: HashMap<String, ColumnBounds>,
    /// Centroid for vector search pruning (if applicable)
    pub centroid: Option<Vec<f32>>,
    /// Spatial bounds (for HELIX/SWIFT/RAPTOR)
    pub spatial_bounds: Option<SpatialBounds>,
    /// Bloom filter (serialized) for membership testing
    pub bloom_filter: Option<Vec<u8>>,
}

/// Column bounds for predicate evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnBounds {
    /// Minimum value (JSON encoded)
    pub min: Option<serde_json::Value>,
    /// Maximum value (JSON encoded)
    pub max: Option<serde_json::Value>,
    /// Null count
    pub null_count: u64,
    /// Distinct count (approximate)
    pub distinct_count: Option<u64>,
}

/// Spatial bounds for vector-aware splits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpatialBounds {
    /// HELIX: Hilbert curve bounds
    Hilbert {
        min_code: u64,
        max_code: u64,
        order: u8,
    },
    /// SWIFT: AdaCurve bounds
    AdaCurve { min_code: u64, max_code: u64 },
    /// RAPTOR: Z-order bounds
    ZOrder { min_code: u64, max_code: u64 },
    /// NOVA: Zone map bounds (per dimension)
    ZoneMap {
        /// Dimension index -> (min, max)
        bounds: HashMap<u32, (f32, f32)>,
    },
    /// Bounding box in vector space
    BoundingBox {
        min_corner: Vec<f32>,
        max_corner: Vec<f32>,
    },
}

/// Locality hints for split scheduling
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SplitLocality {
    /// Preferred hosts (for distributed execution)
    pub preferred_hosts: Vec<String>,
    /// Storage tier (hot/warm/cold)
    pub storage_tier: StorageTier,
    /// Cache status
    pub cache_status: CacheStatus,
}

/// Storage tier for tiered storage
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageTier {
    #[default]
    Hot,
    Warm,
    Cold,
    Archive,
}

/// Cache status for optimization
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheStatus {
    /// Data is in local cache
    Cached,
    /// Data is on local disk
    Local,
    /// Data is remote (S3/Azure/GCS)
    Remote,
    #[default]
    Unknown,
}

impl FileSplit {
    /// Create a single split that nominally covers an entire
    /// collection. Used by callers that don't (yet) have shard-aware
    /// split-planning — Spark's `EmbeddedProximaDB::plan_partitions`
    /// returns one such split per partition for its initial
    /// single-partition fallback (TD-097 follow-up will replace this
    /// with real AXIS/SST shard-aware planning that returns the
    /// per-block / per-row-group splits).
    ///
    /// The `file_path` is set to a logical `collection://<name>` URI
    /// (not a real path); downstream readers ignore it because the
    /// scan flows through the canonical record path, not via
    /// arbitrary file reads.
    pub fn whole_collection(collection: &str) -> Self {
        Self {
            split_id: format!("collection:{}:whole", collection),
            file_path: format!("collection://{}", collection),
            offset: 0,
            length: 0,
            split_type: SplitType::ByteRange {
                estimated_records: 0,
            },
            statistics: SplitStatistics::default(),
            locality: SplitLocality::default(),
            object_size: None,
        }
    }

    /// Create a new block-based split (for SST engine)
    /// Record the size of the object this split lives in (TD-USUB-8), so an
    /// object-storage reader can locate the trailing footer without a `HEAD`.
    pub fn with_object_size(mut self, object_size: u64) -> Self {
        self.object_size = Some(object_size);
        self
    }

    pub fn new_block(
        file_path: String,
        block_id: u32,
        offset: u64,
        length: u64,
        record_count: u64,
    ) -> Self {
        Self {
            split_id: format!("{}:block:{}", file_path, block_id),
            file_path,
            offset,
            length,
            split_type: SplitType::Block {
                block_id,
                record_count,
            },
            statistics: SplitStatistics {
                row_count: Some(record_count),
                byte_size: Some(length),
                ..Default::default()
            },
            locality: SplitLocality::default(),
            object_size: None,
        }
    }

    /// Create a new row group split (for NOVA/VIPER/RAPTOR)
    pub fn new_row_group(
        file_path: String,
        row_group_index: usize,
        offset: u64,
        length: u64,
        row_count: i64,
    ) -> Self {
        Self {
            split_id: format!("{}:rg:{}", file_path, row_group_index),
            file_path,
            offset,
            length,
            split_type: SplitType::RowGroup {
                row_group_index,
                row_count,
            },
            statistics: SplitStatistics {
                row_count: Some(row_count as u64),
                byte_size: Some(length),
                ..Default::default()
            },
            locality: SplitLocality::default(),
            object_size: None,
        }
    }

    /// Create a new Hilbert range split (for HELIX)
    pub fn new_hilbert_range(
        file_path: String,
        start_code: u64,
        end_code: u64,
        hilbert_order: u8,
        offset: u64,
        length: u64,
    ) -> Self {
        Self {
            split_id: format!("{}:hilbert:{}:{}", file_path, start_code, end_code),
            file_path,
            offset,
            length,
            split_type: SplitType::HilbertRange {
                start_code,
                end_code,
                hilbert_order,
            },
            statistics: SplitStatistics {
                spatial_bounds: Some(SpatialBounds::Hilbert {
                    min_code: start_code,
                    max_code: end_code,
                    order: hilbert_order,
                }),
                byte_size: Some(length),
                ..Default::default()
            },
            locality: SplitLocality::default(),
            object_size: None,
        }
    }

    /// Create a new SuperBlock split (for SWIFT)
    pub fn new_superblock(
        file_path: String,
        superblock_id: u32,
        block_ids: Vec<u32>,
        offset: u64,
        length: u64,
    ) -> Self {
        let block_count = block_ids.len();
        Self {
            split_id: format!("{}:superblock:{}", file_path, superblock_id),
            file_path,
            offset,
            length,
            split_type: SplitType::SuperBlock {
                superblock_id,
                block_count,
                block_ids,
            },
            statistics: SplitStatistics {
                byte_size: Some(length),
                ..Default::default()
            },
            locality: SplitLocality::default(),
            object_size: None,
        }
    }

    /// Set statistics for this split
    pub fn with_statistics(mut self, stats: SplitStatistics) -> Self {
        self.statistics = stats;
        self
    }

    /// Set locality hints for this split
    pub fn with_locality(mut self, locality: SplitLocality) -> Self {
        self.locality = locality;
        self
    }

    /// Check if this split can be pruned by the given predicate bounds
    pub fn can_prune(
        &self,
        column: &str,
        min: &serde_json::Value,
        max: &serde_json::Value,
    ) -> bool {
        if let Some(bounds) = self.statistics.column_stats.get(column) {
            // If split's max < predicate's min, split can be pruned
            if let (Some(split_max), Some(_pred_min)) = (&bounds.max, min.as_f64())
                && let Some(split_max_val) = split_max.as_f64()
                && let Some(pred_min_val) = min.as_f64()
                && split_max_val < pred_min_val
            {
                return true;
            }
            // If split's min > predicate's max, split can be pruned
            if let (Some(split_min), Some(_pred_max)) = (&bounds.min, max.as_f64())
                && let Some(split_min_val) = split_min.as_f64()
                && let Some(pred_max_val) = max.as_f64()
                && split_min_val > pred_max_val
            {
                return true;
            }
        }
        false // Cannot prune - must read this split
    }

    /// Estimate cost of reading this split (for scheduling)
    pub fn estimated_cost(&self) -> u64 {
        let base_cost = self.statistics.byte_size.unwrap_or(self.length);

        // Apply locality penalty
        let locality_multiplier = match self.locality.cache_status {
            CacheStatus::Cached => 1,
            CacheStatus::Local => 2,
            CacheStatus::Remote => 10,
            CacheStatus::Unknown => 5,
        };

        base_cost * locality_multiplier
    }

    /// Check if this split can be pruned based on a scalar predicate.
    ///
    /// Returns true if the split can definitely be skipped (no matching rows).
    /// Returns false if the split might contain matching rows and must be read.
    pub fn can_prune_scalar(&self, column: &str, predicate: &ScalarPredicate) -> bool {
        if let Some(bounds) = self.statistics.column_stats.get(column) {
            return bounds.can_prune(predicate);
        }
        false // Cannot prune without statistics
    }

    /// Check if this split can be pruned based on vector distance threshold.
    ///
    /// Uses centroid and max_radius to determine if any vectors in this split
    /// could possibly be within the given distance of the query vector.
    ///
    /// Returns true if the split can definitely be skipped.
    /// Returns false if the split might contain matching vectors.
    ///
    /// # Arguments
    /// * `query` - Query vector (must match dimension of stored vectors)
    /// * `max_distance` - Maximum distance threshold for results
    pub fn can_prune_vector(&self, query: &[f32], max_distance: f32) -> bool {
        // Get centroid and max_radius from statistics
        let centroid = match &self.statistics.centroid {
            Some(c) => c,
            None => return false, // Cannot prune without centroid
        };

        // Get max_radius from spatial bounds if available
        let max_radius = self.get_max_radius().unwrap_or(f32::MAX);

        // Calculate distance from query to centroid
        let centroid_distance = euclidean_distance(query, centroid);

        // If centroid_distance - max_radius > max_distance, split can be pruned
        // This is because even the closest vector in the split would be too far
        centroid_distance - max_radius > max_distance
    }

    /// Get the maximum radius from centroid for this split.
    fn get_max_radius(&self) -> Option<f32> {
        // Try to get from spatial bounds
        if let Some(ref spatial) = self.statistics.spatial_bounds
            && let SpatialBounds::BoundingBox {
                min_corner,
                max_corner,
            } = spatial
        {
            // Calculate diagonal as conservative radius estimate
            let diagonal_sq: f32 = min_corner
                .iter()
                .zip(max_corner.iter())
                .map(|(a, b)| (b - a).powi(2))
                .sum();
            return Some(diagonal_sq.sqrt() / 2.0);
        }
        None
    }

    /// Get detailed split cost breakdown for query planning.
    pub fn split_cost(&self) -> SplitCost {
        let io_bytes = self.statistics.byte_size.unwrap_or(self.length);
        let estimated_rows = self.statistics.row_count.unwrap_or(0);

        // Decode complexity based on split type
        let decode_complexity = match &self.split_type {
            SplitType::Block { .. } => 1.0,
            SplitType::HilbertRange { .. } => 1.2, // Hilbert decode overhead
            SplitType::SuperBlock { block_count, .. } => 1.0 + (*block_count as f64 * 0.1),
            SplitType::RowGroup { .. } => 0.8, // Columnar is efficient
            SplitType::ZOrderRange { .. } => 1.3, // Z-order decode overhead
            SplitType::ByteRange { .. } => 1.0,
        };

        SplitCost {
            io_bytes,
            estimated_rows,
            decode_complexity,
        }
    }
}

/// Scalar predicate for split pruning.
#[derive(Debug, Clone)]
pub enum ScalarPredicate {
    /// Equality check
    Equal(ScalarValue),
    /// Not equal check
    NotEqual(ScalarValue),
    /// Less than
    LessThan(ScalarValue),
    /// Less than or equal
    LessThanOrEqual(ScalarValue),
    /// Greater than
    GreaterThan(ScalarValue),
    /// Greater than or equal
    GreaterThanOrEqual(ScalarValue),
    /// Is null check
    IsNull,
    /// Is not null check
    IsNotNull,
    /// In list (for membership)
    In(Vec<ScalarValue>),
    /// Between (inclusive)
    Between(ScalarValue, ScalarValue),
}

/// Scalar value for predicate evaluation.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum ScalarValue {
    Null,
    Bool(bool),
    Int64(i64),
    Float64(f64),
    String(String),
}

impl ScalarValue {
    /// Convert from serde_json::Value.
    pub fn from_json(value: &serde_json::Value) -> Option<Self> {
        match value {
            serde_json::Value::Null => Some(ScalarValue::Null),
            serde_json::Value::Bool(b) => Some(ScalarValue::Bool(*b)),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Some(ScalarValue::Int64(i))
                } else {
                    n.as_f64().map(ScalarValue::Float64)
                }
            }
            serde_json::Value::String(s) => Some(ScalarValue::String(s.clone())),
            _ => None,
        }
    }
}

/// Cost estimate for reading a split.
#[derive(Debug, Clone)]
pub struct SplitCost {
    /// I/O bytes to read
    pub io_bytes: u64,
    /// Estimated number of rows
    pub estimated_rows: u64,
    /// Decode complexity multiplier (1.0 = baseline)
    pub decode_complexity: f64,
}

impl SplitCost {
    /// Calculate total cost score.
    pub fn total_cost(&self) -> f64 {
        (self.io_bytes as f64) * self.decode_complexity
    }
}

impl ColumnBounds {
    /// Check if this column can be pruned based on predicate.
    pub fn can_prune(&self, predicate: &ScalarPredicate) -> bool {
        match predicate {
            ScalarPredicate::Equal(value) => {
                // If value < min or value > max, prune
                if let (Some(min), Some(max)) = (&self.min, &self.max)
                    && let (Some(min_val), Some(max_val)) =
                        (ScalarValue::from_json(min), ScalarValue::from_json(max))
                {
                    return value < &min_val || value > &max_val;
                }
                false
            }
            ScalarPredicate::LessThan(value) => {
                // If min >= value, prune
                if let Some(min) = &self.min
                    && let Some(min_val) = ScalarValue::from_json(min)
                {
                    return &min_val >= value;
                }
                false
            }
            ScalarPredicate::LessThanOrEqual(value) => {
                // If min > value, prune
                if let Some(min) = &self.min
                    && let Some(min_val) = ScalarValue::from_json(min)
                {
                    return &min_val > value;
                }
                false
            }
            ScalarPredicate::GreaterThan(value) => {
                // If max <= value, prune
                if let Some(max) = &self.max
                    && let Some(max_val) = ScalarValue::from_json(max)
                {
                    return &max_val <= value;
                }
                false
            }
            ScalarPredicate::GreaterThanOrEqual(value) => {
                // If max < value, prune
                if let Some(max) = &self.max
                    && let Some(max_val) = ScalarValue::from_json(max)
                {
                    return &max_val < value;
                }
                false
            }
            ScalarPredicate::IsNull => {
                // If null_count == 0, prune
                self.null_count == 0
            }
            ScalarPredicate::IsNotNull => {
                // Cannot prune based on IsNotNull alone
                false
            }
            ScalarPredicate::Between(low, high) => {
                // If max < low or min > high, prune
                if let (Some(min), Some(max)) = (&self.min, &self.max)
                    && let (Some(min_val), Some(max_val)) =
                        (ScalarValue::from_json(min), ScalarValue::from_json(max))
                {
                    return &max_val < low || &min_val > high;
                }
                false
            }
            ScalarPredicate::In(values) => {
                // If all values are outside [min, max], prune
                if let (Some(min), Some(max)) = (&self.min, &self.max)
                    && let (Some(min_val), Some(max_val)) =
                        (ScalarValue::from_json(min), ScalarValue::from_json(max))
                {
                    return values.iter().all(|v| v < &min_val || v > &max_val);
                }
                false
            }
            ScalarPredicate::NotEqual(_) => {
                // Cannot prune based on NotEqual alone
                false
            }
        }
    }
}

/// Calculate Euclidean distance between two vectors.
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::MAX;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Trait for generating splits from storage files
pub trait SplitGenerator: Send + Sync {
    /// Generate splits for a file
    fn generate_splits(&self, file_path: &str) -> anyhow::Result<Vec<FileSplit>>;

    /// Generate splits with target count (for parallelism control)
    fn generate_splits_with_target(
        &self,
        file_path: &str,
        _target_count: usize,
    ) -> anyhow::Result<Vec<FileSplit>> {
        // Default: just generate all splits
        self.generate_splits(file_path)
    }

    /// Generate splits for multiple files
    fn generate_splits_for_files(&self, file_paths: &[String]) -> anyhow::Result<Vec<FileSplit>> {
        let mut all_splits = Vec::new();
        for path in file_paths {
            all_splits.extend(self.generate_splits(path)?);
        }
        Ok(all_splits)
    }
}

/// Split planner for optimizing split assignment
pub struct SplitPlanner {
    /// Maximum split size for memory management
    pub max_split_size_bytes: u64,
    /// Minimum splits per partition
    pub min_splits_per_partition: usize,
    /// Enable split combining for small splits
    pub enable_split_combining: bool,
}

impl Default for SplitPlanner {
    fn default() -> Self {
        Self {
            max_split_size_bytes: 128 * 1024 * 1024, // 128MB
            min_splits_per_partition: 1,
            enable_split_combining: true,
        }
    }
}

impl SplitPlanner {
    /// Plan splits for parallel execution
    pub fn plan_splits(
        &self,
        splits: Vec<FileSplit>,
        target_partitions: usize,
    ) -> Vec<Vec<FileSplit>> {
        if splits.is_empty() {
            return vec![];
        }

        let mut partitions: Vec<Vec<FileSplit>> = vec![vec![]; target_partitions];
        let mut partition_costs: Vec<u64> = vec![0; target_partitions];

        // Sort splits by cost (descending) for better load balancing
        let mut sorted_splits = splits;
        sorted_splits.sort_by_key(|s| std::cmp::Reverse(s.estimated_cost()));

        // Greedy assignment to partition with lowest cost
        for split in sorted_splits {
            let cost = split.estimated_cost();
            let min_idx = partition_costs
                .iter()
                .enumerate()
                .min_by_key(|(_, c)| *c)
                .map_or(0, |(i, _)| i);

            partitions[min_idx].push(split);
            partition_costs[min_idx] += cost;
        }

        // Remove empty partitions
        partitions.retain(|p| !p.is_empty());
        partitions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestSplitGenerator;

    impl SplitGenerator for TestSplitGenerator {
        fn generate_splits(&self, file_path: &str) -> anyhow::Result<Vec<FileSplit>> {
            Ok(vec![FileSplit::new_block(
                file_path.to_string(),
                0,
                0,
                128,
                10,
            )])
        }
    }

    #[test]
    fn test_block_split() {
        let split = FileSplit::new_block("/data/file.sst".to_string(), 0, 0, 1024, 100);

        assert_eq!(split.split_id, "/data/file.sst:block:0");
        assert_eq!(split.file_path, "/data/file.sst");
        assert_eq!(split.offset, 0);
        assert_eq!(split.length, 1024);
        assert_eq!(split.statistics.row_count, Some(100));
        assert_eq!(split.statistics.byte_size, Some(1024));
    }

    #[test]
    fn test_row_group_split() {
        let split = FileSplit::new_row_group("/data/file.parquet".to_string(), 0, 0, 65536, 10000);

        assert!(matches!(split.split_type, SplitType::RowGroup { .. }));
        assert_eq!(split.split_id, "/data/file.parquet:rg:0");
        assert_eq!(split.statistics.row_count, Some(10000));
    }

    #[test]
    fn test_hilbert_superblock_locality_and_cost_multipliers() {
        let hilbert =
            FileSplit::new_hilbert_range("/data/file.helix".to_string(), 10, 20, 8, 32, 256);
        assert_eq!(hilbert.split_id, "/data/file.helix:hilbert:10:20");
        assert!(matches!(
            hilbert.statistics.spatial_bounds,
            Some(SpatialBounds::Hilbert {
                min_code: 10,
                max_code: 20,
                order: 8
            })
        ));
        assert!((hilbert.split_cost().decode_complexity - 1.2).abs() < 0.01);

        let superblock =
            FileSplit::new_superblock("/data/file.swift".to_string(), 7, vec![1, 2, 3], 64, 512);
        assert_eq!(superblock.split_id, "/data/file.swift:superblock:7");
        assert!(matches!(
            superblock.split_type,
            SplitType::SuperBlock { block_count: 3, .. }
        ));
        assert!((superblock.split_cost().decode_complexity - 1.3).abs() < 0.01);

        for (cache_status, expected) in [
            (CacheStatus::Cached, 100),
            (CacheStatus::Local, 200),
            (CacheStatus::Remote, 1000),
            (CacheStatus::Unknown, 500),
        ] {
            let split = FileSplit::new_block("/data/file.sst".to_string(), 0, 0, 100, 1)
                .with_locality(SplitLocality {
                    preferred_hosts: vec!["host1".to_string()],
                    storage_tier: StorageTier::Warm,
                    cache_status,
                });
            assert_eq!(split.estimated_cost(), expected);
            assert_eq!(split.locality.preferred_hosts, vec!["host1".to_string()]);
            assert_eq!(split.locality.storage_tier, StorageTier::Warm);
        }
    }

    #[test]
    fn test_manual_split_variants_and_defaults() {
        assert_eq!(StorageTier::default(), StorageTier::Hot);
        assert_eq!(CacheStatus::default(), CacheStatus::Unknown);

        let zorder = FileSplit {
            split_id: "z".to_string(),
            file_path: "/z".to_string(),
            offset: 0,
            length: 10,
            split_type: SplitType::ZOrderRange {
                start_code: 1,
                end_code: 9,
            },
            statistics: SplitStatistics::default(),
            locality: SplitLocality::default(),
            object_size: None,
        };
        assert!((zorder.split_cost().decode_complexity - 1.3).abs() < 0.01);

        let byte_range = FileSplit {
            split_id: "b".to_string(),
            file_path: "/b".to_string(),
            offset: 5,
            length: 20,
            split_type: SplitType::ByteRange {
                estimated_records: 2,
            },
            statistics: SplitStatistics::default(),
            locality: SplitLocality {
                preferred_hosts: vec![],
                storage_tier: StorageTier::Archive,
                cache_status: CacheStatus::Unknown,
            },
        };
        let cost = byte_range.split_cost();
        assert_eq!(cost.io_bytes, 20);
        assert_eq!(cost.estimated_rows, 0);
        assert_eq!(cost.total_cost(), 20.0);
    }

    #[test]
    fn test_split_planner() {
        let planner = SplitPlanner::default();

        let splits = vec![
            FileSplit::new_block("/f1.sst".to_string(), 0, 0, 1000, 100),
            FileSplit::new_block("/f1.sst".to_string(), 1, 1000, 2000, 200),
            FileSplit::new_block("/f2.sst".to_string(), 0, 0, 1500, 150),
        ];

        let partitions = planner.plan_splits(splits, 2);
        assert_eq!(partitions.len(), 2);
        assert!(planner.plan_splits(vec![], 0).is_empty());
        assert_eq!(planner.max_split_size_bytes, 128 * 1024 * 1024);
        assert_eq!(planner.min_splits_per_partition, 1);
        assert!(planner.enable_split_combining);
    }

    #[test]
    fn test_scalar_predicate_equal() {
        let bounds = ColumnBounds {
            min: Some(serde_json::json!(10)),
            max: Some(serde_json::json!(100)),
            null_count: 0,
            distinct_count: None,
        };

        // Value within range - cannot prune
        assert!(!bounds.can_prune(&ScalarPredicate::Equal(ScalarValue::Int64(50))));

        // Value below range - can prune
        assert!(bounds.can_prune(&ScalarPredicate::Equal(ScalarValue::Int64(5))));

        // Value above range - can prune
        assert!(bounds.can_prune(&ScalarPredicate::Equal(ScalarValue::Int64(150))));

        assert!(!bounds.can_prune(&ScalarPredicate::NotEqual(ScalarValue::Int64(50))));
    }

    #[test]
    fn test_scalar_predicate_less_than() {
        let bounds = ColumnBounds {
            min: Some(serde_json::json!(10)),
            max: Some(serde_json::json!(100)),
            null_count: 0,
            distinct_count: None,
        };

        // predicate: value < 5 (min is 10, so can prune)
        assert!(bounds.can_prune(&ScalarPredicate::LessThan(ScalarValue::Int64(10))));

        // predicate: value < 50 (min is 10, cannot prune)
        assert!(!bounds.can_prune(&ScalarPredicate::LessThan(ScalarValue::Int64(50))));

        assert!(bounds.can_prune(&ScalarPredicate::LessThanOrEqual(ScalarValue::Int64(9))));
        assert!(!bounds.can_prune(&ScalarPredicate::LessThanOrEqual(ScalarValue::Int64(10))));
    }

    #[test]
    fn test_scalar_predicate_greater_than() {
        let bounds = ColumnBounds {
            min: Some(serde_json::json!(10)),
            max: Some(serde_json::json!(100)),
            null_count: 0,
            distinct_count: None,
        };

        // predicate: value > 100 (max is 100, so can prune)
        assert!(bounds.can_prune(&ScalarPredicate::GreaterThan(ScalarValue::Int64(100))));

        // predicate: value > 50 (max is 100, cannot prune)
        assert!(!bounds.can_prune(&ScalarPredicate::GreaterThan(ScalarValue::Int64(50))));

        assert!(
            bounds.can_prune(&ScalarPredicate::GreaterThanOrEqual(ScalarValue::Int64(
                101
            )))
        );
        assert!(
            !bounds.can_prune(&ScalarPredicate::GreaterThanOrEqual(ScalarValue::Int64(
                100
            )))
        );
    }

    #[test]
    fn test_scalar_predicate_between() {
        let bounds = ColumnBounds {
            min: Some(serde_json::json!(10)),
            max: Some(serde_json::json!(100)),
            null_count: 0,
            distinct_count: None,
        };

        // BETWEEN 50 AND 80 - overlaps with [10, 100], cannot prune
        assert!(!bounds.can_prune(&ScalarPredicate::Between(
            ScalarValue::Int64(50),
            ScalarValue::Int64(80)
        )));

        // BETWEEN 200 AND 300 - no overlap, can prune
        assert!(bounds.can_prune(&ScalarPredicate::Between(
            ScalarValue::Int64(200),
            ScalarValue::Int64(300)
        )));

        assert!(bounds.can_prune(&ScalarPredicate::In(vec![
            ScalarValue::Int64(1),
            ScalarValue::Int64(200)
        ])));
        assert!(!bounds.can_prune(&ScalarPredicate::In(vec![
            ScalarValue::Int64(1),
            ScalarValue::Int64(50)
        ])));
    }

    #[test]
    fn test_scalar_predicate_is_null() {
        let bounds_with_nulls = ColumnBounds {
            min: Some(serde_json::json!(10)),
            max: Some(serde_json::json!(100)),
            null_count: 5,
            distinct_count: None,
        };

        let bounds_no_nulls = ColumnBounds {
            min: Some(serde_json::json!(10)),
            max: Some(serde_json::json!(100)),
            null_count: 0,
            distinct_count: None,
        };

        // Has nulls - cannot prune IS NULL
        assert!(!bounds_with_nulls.can_prune(&ScalarPredicate::IsNull));

        // No nulls - can prune IS NULL
        assert!(bounds_no_nulls.can_prune(&ScalarPredicate::IsNull));
        assert!(!bounds_no_nulls.can_prune(&ScalarPredicate::IsNotNull));
    }

    #[test]
    fn test_split_scalar_pruning() {
        let mut split = FileSplit::new_block("/data/file.sst".to_string(), 0, 0, 1024, 100);

        // Add column statistics
        split.statistics.column_stats.insert(
            "price".to_string(),
            ColumnBounds {
                min: Some(serde_json::json!(10.0)),
                max: Some(serde_json::json!(100.0)),
                null_count: 0,
                distinct_count: None,
            },
        );

        // Can prune when value is outside range
        assert!(split.can_prune_scalar(
            "price",
            &ScalarPredicate::GreaterThan(ScalarValue::Float64(100.0))
        ));

        // Cannot prune when value is within range
        assert!(!split.can_prune_scalar(
            "price",
            &ScalarPredicate::GreaterThan(ScalarValue::Float64(50.0))
        ));

        // Cannot prune unknown column
        assert!(
            !split.can_prune_scalar("unknown", &ScalarPredicate::Equal(ScalarValue::Int64(50)))
        );

        assert!(split.can_prune(
            "price",
            &serde_json::json!(200.0),
            &serde_json::json!(300.0)
        ));
        assert!(split.can_prune("price", &serde_json::json!(0.0), &serde_json::json!(5.0)));
        assert!(!split.can_prune("price", &serde_json::json!(50.0), &serde_json::json!(60.0)));
        assert!(!split.can_prune("unknown", &serde_json::json!(0.0), &serde_json::json!(1.0)));
    }

    #[test]
    fn test_split_vector_pruning() {
        let mut split = FileSplit::new_block("/data/file.sst".to_string(), 0, 0, 1024, 100);

        // Add centroid for vector pruning
        split.statistics.centroid = Some(vec![0.0, 0.0, 0.0]);
        split.statistics.spatial_bounds = Some(SpatialBounds::BoundingBox {
            min_corner: vec![-1.0, -1.0, -1.0],
            max_corner: vec![1.0, 1.0, 1.0],
        });

        // Query close to centroid - cannot prune
        let query_close = vec![0.5, 0.5, 0.5];
        assert!(!split.can_prune_vector(&query_close, 10.0));

        // Query very far from centroid - can prune
        let query_far = vec![100.0, 100.0, 100.0];
        assert!(split.can_prune_vector(&query_far, 1.0));

        split.statistics.spatial_bounds = None;
        assert!(!split.can_prune_vector(&query_far, 1.0));
    }

    #[test]
    fn test_split_vector_pruning_no_centroid() {
        let split = FileSplit::new_block("/data/file.sst".to_string(), 0, 0, 1024, 100);

        // Without centroid, cannot prune
        let query = vec![0.5, 0.5, 0.5];
        assert!(!split.can_prune_vector(&query, 1.0));
    }

    #[test]
    fn test_split_cost() {
        let split = FileSplit::new_block("/data/file.sst".to_string(), 0, 0, 1024, 100);
        let cost = split.split_cost();

        assert_eq!(cost.io_bytes, 1024);
        assert_eq!(cost.estimated_rows, 100);
        assert!((cost.decode_complexity - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_split_cost_row_group() {
        let split = FileSplit::new_row_group("/data/file.parquet".to_string(), 0, 0, 65536, 10000);
        let cost = split.split_cost();

        // Row groups are more efficient (columnar)
        assert!((cost.decode_complexity - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];

        let dist = euclidean_distance(&a, &b);
        assert!((dist - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_euclidean_distance_mismatched_dims() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];

        let dist = euclidean_distance(&a, &b);
        assert_eq!(dist, f32::MAX);
    }

    #[test]
    fn test_scalar_value_from_json() {
        assert_eq!(
            ScalarValue::from_json(&serde_json::json!(42)),
            Some(ScalarValue::Int64(42))
        );
        assert_eq!(
            ScalarValue::from_json(&serde_json::json!(3.125)),
            Some(ScalarValue::Float64(3.125))
        );
        assert_eq!(
            ScalarValue::from_json(&serde_json::json!("hello")),
            Some(ScalarValue::String("hello".to_string()))
        );
        assert_eq!(
            ScalarValue::from_json(&serde_json::json!(true)),
            Some(ScalarValue::Bool(true))
        );
        assert_eq!(
            ScalarValue::from_json(&serde_json::Value::Null),
            Some(ScalarValue::Null)
        );
        assert_eq!(ScalarValue::from_json(&serde_json::json!([1, 2])), None);
        assert_eq!(ScalarValue::from_json(&serde_json::json!({"a": 1})), None);
    }

    #[test]
    fn test_split_generator_default_methods() {
        let generator = TestSplitGenerator;

        let targeted = generator
            .generate_splits_with_target("/data/a.sst", 4)
            .unwrap();
        assert_eq!(targeted.len(), 1);
        assert_eq!(targeted[0].file_path, "/data/a.sst");

        let files = vec!["/data/a.sst".to_string(), "/data/b.sst".to_string()];
        let splits = generator.generate_splits_for_files(&files).unwrap();
        assert_eq!(splits.len(), 2);
        assert_eq!(splits[1].file_path, "/data/b.sst");
    }

    #[test]
    fn test_with_statistics_and_spatial_bounds_variants() {
        let stats = SplitStatistics {
            row_count: Some(5),
            byte_size: Some(32),
            column_stats: HashMap::new(),
            centroid: Some(vec![1.0, 2.0]),
            spatial_bounds: Some(SpatialBounds::AdaCurve {
                min_code: 10,
                max_code: 20,
            }),
            bloom_filter: Some(vec![1, 2, 3]),
        };

        let split =
            FileSplit::new_block("/data/file.sst".to_string(), 9, 0, 64, 99).with_statistics(stats);
        assert_eq!(split.statistics.row_count, Some(5));
        assert_eq!(split.statistics.byte_size, Some(32));
        assert_eq!(split.statistics.bloom_filter, Some(vec![1, 2, 3]));
        assert!(matches!(
            split.statistics.spatial_bounds,
            Some(SpatialBounds::AdaCurve {
                min_code: 10,
                max_code: 20
            })
        ));

        let bounds = [
            SpatialBounds::ZOrder {
                min_code: 1,
                max_code: 2,
            },
            SpatialBounds::ZoneMap {
                bounds: HashMap::from([(0, (1.0, 2.0))]),
            },
        ];
        assert_eq!(bounds.len(), 2);
    }
}
