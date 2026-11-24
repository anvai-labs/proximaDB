//! # Pruning Strategies - Pluggable Pruning Interface for Native Compute Engine
//!
//! This module provides trait-based abstractions for different pruning strategies
//! following SOLID principles:
//!
//! - **Single Responsibility**: Each pruner handles one type of pruning
//! - **Interface Segregation**: Separate traits for vector, scalar, spatial, and bloom pruning
//! - **Dependency Inversion**: HeaderCache depends on abstract pruner traits
//!
//! ## Pruning Hierarchy
//!
//! ```text
//! Query arrives
//!      ↓
//! VectorPruner (CentroidTree) → O(log n) coarse elimination by vector distance
//!      ↓
//! SpatialPruner (Hilbert/Z-order) → Spatial locality pruning
//!      ↓
//! ScalarPruner (Min/Max bounds) → Column statistics pruning
//!      ↓
//! BloomChecker (Consolidated) → Point lookup ID filtering
//!      ↓
//! Remaining rowgroups → Actual I/O
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use proximadb::storage::schema::pruning_strategies::{VectorPruner, SchemaPruningResult};
//!
//! let pruner: Arc<dyn VectorPruner> = Arc::new(CentroidTree::build(&centroids, 8));
//! let result = pruner.prune_by_vector(&query, max_distance);
//! // Only read rowgroups in result.included_indices
//! ```

use std::collections::HashSet;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use super::header_cache::{ColumnBounds, ScalarPredicate, SpatialRange};

// ============================================================================
// Pruning Result Types
// ============================================================================

/// Backwards-compat alias for [`SchemaPruningResult`].
pub type PruningResult = SchemaPruningResult;

/// Result of a pruning operation, indicating which rowgroups may contain matches.
#[derive(Debug, Clone)]
pub struct SchemaPruningResult {
    /// Indices of rowgroups that may contain matching data.
    /// Empty means no rowgroups match (can skip entire file).
    pub included_indices: Vec<usize>,

    /// Total rowgroups considered.
    pub total_rowgroups: usize,

    /// Pruning statistics for observability.
    pub stats: PruningStats,
}

impl SchemaPruningResult {
    /// Create a result where all rowgroups are included (no pruning possible).
    pub fn include_all(total: usize) -> Self {
        Self {
            included_indices: (0..total).collect(),
            total_rowgroups: total,
            stats: PruningStats {
                pruning_ratio: 0.0,
                computation_ns: 0,
                method: "none".to_string(),
            },
        }
    }

    /// Create a result with specific included indices.
    pub fn with_indices(
        indices: Vec<usize>,
        total: usize,
        method: &str,
        computation_ns: u64,
    ) -> Self {
        let pruning_ratio = if total > 0 {
            1.0 - (indices.len() as f64 / total as f64)
        } else {
            0.0
        };

        Self {
            included_indices: indices,
            total_rowgroups: total,
            stats: PruningStats {
                pruning_ratio,
                computation_ns,
                method: method.to_string(),
            },
        }
    }

    /// Create a result where no rowgroups match (can skip entire file).
    pub fn include_none(total: usize, method: &str, computation_ns: u64) -> Self {
        Self {
            included_indices: Vec::new(),
            total_rowgroups: total,
            stats: PruningStats {
                pruning_ratio: 1.0,
                computation_ns,
                method: method.to_string(),
            },
        }
    }

    /// Check if any rowgroups were included.
    pub fn has_matches(&self) -> bool {
        !self.included_indices.is_empty()
    }

    /// Get pruned count (rowgroups eliminated).
    pub fn pruned_count(&self) -> usize {
        self.total_rowgroups
            .saturating_sub(self.included_indices.len())
    }

    /// Intersect with another pruning result (AND semantics).
    pub fn intersect(&self, other: &SchemaPruningResult) -> SchemaPruningResult {
        let start = std::time::Instant::now();

        let self_set: HashSet<usize> = self.included_indices.iter().copied().collect();
        let other_set: HashSet<usize> = other.included_indices.iter().copied().collect();

        let intersection: Vec<usize> = self_set.intersection(&other_set).copied().collect();

        let total = self.total_rowgroups.max(other.total_rowgroups);
        let pruning_ratio = if total > 0 {
            1.0 - (intersection.len() as f64 / total as f64)
        } else {
            0.0
        };

        SchemaPruningResult {
            included_indices: intersection,
            total_rowgroups: total,
            stats: PruningStats {
                pruning_ratio,
                computation_ns: start.elapsed().as_nanos() as u64,
                method: format!("{}+{}", self.stats.method, other.stats.method),
            },
        }
    }
}

/// Statistics from pruning operation for observability.
#[derive(Debug, Clone)]
pub struct PruningStats {
    /// Ratio of rowgroups pruned (0.0 = none, 1.0 = all).
    pub pruning_ratio: f64,

    /// Time spent computing pruning decision (nanoseconds).
    pub computation_ns: u64,

    /// Pruning method used (e.g., "centroid_tree", "scalar_bounds", "bloom").
    pub method: String,
}

/// Result of bloom filter membership check.
#[derive(Debug, Clone)]
pub struct BloomCheckResult {
    /// IDs that definitely do not exist (can skip lookup).
    pub definitely_absent: Vec<String>,

    /// IDs that might exist (need actual lookup).
    pub possibly_present: Vec<String>,

    /// False positive rate estimate for this check.
    pub estimated_fpr: f64,
}

impl BloomCheckResult {
    /// Create a result where all IDs might be present (bloom filter disabled or unavailable).
    pub fn all_possibly_present(ids: Vec<String>) -> Self {
        Self {
            definitely_absent: Vec::new(),
            possibly_present: ids,
            estimated_fpr: 1.0,
        }
    }

    /// Create a result from bloom filter checks.
    pub fn from_checks(
        definitely_absent: Vec<String>,
        possibly_present: Vec<String>,
        estimated_fpr: f64,
    ) -> Self {
        Self {
            definitely_absent,
            possibly_present,
            estimated_fpr,
        }
    }
}

// ============================================================================
// Pruner Traits (Interface Segregation Principle)
// ============================================================================

/// Trait for vector-based pruning using centroid distance bounds.
///
/// Implementations use spatial data structures (trees, grids) to quickly
/// identify rowgroups whose vectors are within distance threshold of query.
///
/// ## Performance
///
/// - CentroidTree: O(log n) average case
/// - Linear scan: O(n) - fallback when no index available
///
/// ## Distance Semantics
///
/// max_distance is the L2 (Euclidean) distance threshold.
/// Rowgroups whose centroids are farther than max_distance from query
/// can be safely pruned because all vectors in that rowgroup are
/// farther than max_distance - max_radius from the query.
pub trait VectorPruner: Send + Sync {
    /// Prune rowgroups by vector distance to query.
    ///
    /// # Arguments
    /// * `query` - Query vector (must match dimension of stored vectors)
    /// * `max_distance` - Maximum L2 distance threshold
    ///
    /// # Returns
    /// SchemaPruningResult with indices of rowgroups that may contain vectors
    /// within max_distance of query.
    fn prune_by_vector(&self, query: &[f32], max_distance: f32) -> SchemaPruningResult;

    /// Prune using quantized distance approximation for speed.
    ///
    /// Uses lower-precision distance computation (INT8 or binary)
    /// for faster pruning with slightly higher false positive rate.
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `max_distance` - Maximum L2 distance threshold
    ///
    /// # Returns
    /// SchemaPruningResult with conservative (superset) of matching rowgroups.
    fn prune_quantized(&self, query: &[f32], max_distance: f32) -> SchemaPruningResult {
        // Default: fall back to exact pruning
        self.prune_by_vector(query, max_distance)
    }

    /// Get dimension of vectors this pruner handles.
    fn dimension(&self) -> usize;

    /// Get number of rowgroups/centroids indexed.
    fn num_entries(&self) -> usize;
}

/// Trait for scalar predicate pruning using column min/max bounds.
///
/// Uses column statistics (min, max, null_count) stored in rowgroup
/// metadata to skip rowgroups that cannot contain matching rows.
pub trait ScalarPruner: Send + Sync {
    /// Prune rowgroups by scalar column predicate.
    ///
    /// # Arguments
    /// * `column` - Column name to apply predicate to
    /// * `predicate` - Scalar predicate (Eq, Lt, Gt, Between, etc.)
    ///
    /// # Returns
    /// SchemaPruningResult with indices of rowgroups that may contain matching data.
    fn prune_by_predicate(&self, column: &str, predicate: &ScalarPredicate) -> SchemaPruningResult;

    /// Prune by multiple predicates (AND semantics).
    fn prune_by_predicates(&self, predicates: &[(&str, ScalarPredicate)]) -> SchemaPruningResult {
        if predicates.is_empty() {
            return SchemaPruningResult::include_all(self.num_rowgroups());
        }

        let mut result = self.prune_by_predicate(predicates[0].0, &predicates[0].1);

        for (column, predicate) in predicates.iter().skip(1) {
            let next = self.prune_by_predicate(column, predicate);
            result = result.intersect(&next);
        }

        result
    }

    /// Get number of rowgroups.
    fn num_rowgroups(&self) -> usize;

    /// Get available column names for pruning.
    fn available_columns(&self) -> Vec<String>;
}

/// Trait for spatial range pruning using space-filling curves.
///
/// Different engines use different spatial indexing schemes:
/// - HELIX: Hilbert curves (best locality preservation)
/// - RAPTOR: Z-order/Morton codes (simple, fast)
/// - SWIFT: Adaptive learned curves
/// - NOVA: Zone maps (per-dimension bounds)
pub trait SpatialPruner: Send + Sync {
    /// Prune rowgroups by spatial range overlap.
    ///
    /// # Arguments
    /// * `range` - Query spatial range in engine-specific encoding
    ///
    /// # Returns
    /// SchemaPruningResult with indices of rowgroups whose spatial range overlaps query.
    fn prune_by_spatial_range(&self, range: &SpatialRange) -> SchemaPruningResult;

    /// Get spatial range type used by this pruner.
    fn spatial_type(&self) -> SpatialRangeType;

    /// Get number of rowgroups.
    fn num_rowgroups(&self) -> usize;
}

/// Type of spatial range encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpatialRangeType {
    /// Hilbert curve (HELIX)
    Hilbert,
    /// Z-order/Morton code (RAPTOR)
    ZOrder,
    /// Adaptive learned curve (SWIFT)
    AdaCurve,
    /// Per-dimension zone map (NOVA)
    ZoneMap,
    /// Block range (SST)
    BlockRange,
    /// Parquet rowgroup (NOVA)
    ParquetRowGroup,
}

/// Trait for bloom filter membership testing.
///
/// Used for point lookups by ID to quickly eliminate rowgroups
/// that definitely do not contain the requested IDs.
pub trait BloomChecker: Send + Sync {
    /// Check which IDs might exist in this file/collection.
    ///
    /// # Arguments
    /// * `ids` - IDs to check for potential membership
    ///
    /// # Returns
    /// BloomCheckResult separating definitely absent from possibly present IDs.
    fn check_ids(&self, ids: &[&str]) -> BloomCheckResult;

    /// Check a single ID.
    fn might_contain(&self, id: &str) -> bool {
        let result = self.check_ids(&[id]);
        result.possibly_present.contains(&id.to_string())
    }

    /// Get estimated false positive rate.
    fn false_positive_rate(&self) -> f64;

    /// Get number of items in the bloom filter.
    fn num_items(&self) -> usize;
}

// ============================================================================
// Composite Pruner (combines multiple strategies)
// ============================================================================

/// Composite pruner that combines multiple pruning strategies.
///
/// Applies pruners in order of expected selectivity (most selective first)
/// and short-circuits when no rowgroups remain.
pub struct CompositePruner {
    vector_pruner: Option<Arc<dyn VectorPruner>>,
    scalar_pruner: Option<Arc<dyn ScalarPruner>>,
    spatial_pruner: Option<Arc<dyn SpatialPruner>>,
    bloom_checker: Option<Arc<dyn BloomChecker>>,
    total_rowgroups: usize,
}

impl CompositePruner {
    /// Create a new composite pruner.
    pub fn new(total_rowgroups: usize) -> Self {
        Self {
            vector_pruner: None,
            scalar_pruner: None,
            spatial_pruner: None,
            bloom_checker: None,
            total_rowgroups,
        }
    }

    /// Add vector pruner.
    pub fn with_vector_pruner(mut self, pruner: Arc<dyn VectorPruner>) -> Self {
        self.vector_pruner = Some(pruner);
        self
    }

    /// Add scalar pruner.
    pub fn with_scalar_pruner(mut self, pruner: Arc<dyn ScalarPruner>) -> Self {
        self.scalar_pruner = Some(pruner);
        self
    }

    /// Add spatial pruner.
    pub fn with_spatial_pruner(mut self, pruner: Arc<dyn SpatialPruner>) -> Self {
        self.spatial_pruner = Some(pruner);
        self
    }

    /// Add bloom checker.
    pub fn with_bloom_checker(mut self, checker: Arc<dyn BloomChecker>) -> Self {
        self.bloom_checker = Some(checker);
        self
    }

    /// Apply all configured pruners and return combined result.
    pub fn prune(
        &self,
        query_vector: Option<(&[f32], f32)>,
        predicates: Option<&[(&str, ScalarPredicate)]>,
        spatial_range: Option<&SpatialRange>,
        _id_filter: Option<&[&str]>,
    ) -> SchemaPruningResult {
        let mut result = SchemaPruningResult::include_all(self.total_rowgroups);

        // Apply vector pruning first (typically most selective for vector search)
        if let (Some(pruner), Some((query, max_dist))) = (&self.vector_pruner, query_vector) {
            let vector_result = pruner.prune_by_vector(query, max_dist);
            result = result.intersect(&vector_result);

            // Short-circuit if nothing matches
            if !result.has_matches() {
                return result;
            }
        }

        // Apply spatial pruning
        if let (Some(pruner), Some(range)) = (&self.spatial_pruner, spatial_range) {
            let spatial_result = pruner.prune_by_spatial_range(range);
            result = result.intersect(&spatial_result);

            if !result.has_matches() {
                return result;
            }
        }

        // Apply scalar pruning
        if let (Some(pruner), Some(preds)) = (&self.scalar_pruner, predicates) {
            let scalar_result = pruner.prune_by_predicates(preds);
            result = result.intersect(&scalar_result);

            if !result.has_matches() {
                return result;
            }
        }

        // Bloom filtering applies to IDs, not rowgroups
        // This is handled separately at the ID level

        result
    }
}

// ============================================================================
// Null Implementations (for testing and fallback)
// ============================================================================

/// No-op vector pruner that includes all rowgroups.
pub struct NullVectorPruner {
    /// Total rowgroups
    total: usize,
    /// Vector dimension
    dimension: usize,
}

impl NullVectorPruner {
    pub fn new(total: usize, dimension: usize) -> Self {
        Self { total, dimension }
    }
}

impl VectorPruner for NullVectorPruner {
    fn prune_by_vector(&self, _query: &[f32], _max_distance: f32) -> SchemaPruningResult {
        SchemaPruningResult::include_all(self.total)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn num_entries(&self) -> usize {
        self.total
    }
}

/// No-op scalar pruner that includes all rowgroups.
pub struct NullScalarPruner {
    /// Total rowgroups
    total: usize,
}

impl NullScalarPruner {
    pub fn new(total: usize) -> Self {
        Self { total }
    }
}

impl ScalarPruner for NullScalarPruner {
    fn prune_by_predicate(
        &self,
        _column: &str,
        _predicate: &ScalarPredicate,
    ) -> SchemaPruningResult {
        SchemaPruningResult::include_all(self.total)
    }

    fn num_rowgroups(&self) -> usize {
        self.total
    }

    fn available_columns(&self) -> Vec<String> {
        Vec::new()
    }
}

/// No-op bloom checker that marks all IDs as possibly present.
pub struct NullBloomChecker;

impl BloomChecker for NullBloomChecker {
    fn check_ids(&self, ids: &[&str]) -> BloomCheckResult {
        BloomCheckResult::all_possibly_present(ids.iter().map(|s| s.to_string()).collect())
    }

    fn false_positive_rate(&self) -> f64 {
        1.0
    }

    fn num_items(&self) -> usize {
        0
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pruning_result_include_all() {
        let result = SchemaPruningResult::include_all(10);
        assert_eq!(result.included_indices.len(), 10);
        assert_eq!(result.total_rowgroups, 10);
        assert_eq!(result.pruned_count(), 0);
        assert!(result.has_matches());
    }

    #[test]
    fn test_pruning_result_include_none() {
        let result = SchemaPruningResult::include_none(10, "test", 100);
        assert!(result.included_indices.is_empty());
        assert_eq!(result.total_rowgroups, 10);
        assert_eq!(result.pruned_count(), 10);
        assert!(!result.has_matches());
        assert_eq!(result.stats.pruning_ratio, 1.0);
    }

    #[test]
    fn test_pruning_result_with_indices() {
        let result = SchemaPruningResult::with_indices(vec![1, 3, 5], 10, "test", 100);
        assert_eq!(result.included_indices.len(), 3);
        assert_eq!(result.pruned_count(), 7);
        assert!((result.stats.pruning_ratio - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_pruning_result_intersect() {
        let result1 = SchemaPruningResult::with_indices(vec![0, 1, 2, 3, 4], 10, "a", 100);
        let result2 = SchemaPruningResult::with_indices(vec![2, 3, 4, 5, 6], 10, "b", 100);

        let intersected = result1.intersect(&result2);
        assert_eq!(intersected.included_indices.len(), 3); // 2, 3, 4
        assert!(intersected.included_indices.contains(&2));
        assert!(intersected.included_indices.contains(&3));
        assert!(intersected.included_indices.contains(&4));
    }

    #[test]
    fn test_null_vector_pruner() {
        let pruner = NullVectorPruner::new(10, 128);
        let result = pruner.prune_by_vector(&[0.0; 128], 1.0);
        assert_eq!(result.included_indices.len(), 10);
        assert_eq!(pruner.dimension(), 128);
        assert_eq!(pruner.num_entries(), 10);
    }

    #[test]
    fn test_null_bloom_checker() {
        let checker = NullBloomChecker;
        let result = checker.check_ids(&["id1", "id2", "id3"]);
        assert!(result.definitely_absent.is_empty());
        assert_eq!(result.possibly_present.len(), 3);
        assert_eq!(checker.false_positive_rate(), 1.0);
    }

    #[test]
    fn test_bloom_check_result() {
        let result = BloomCheckResult::from_checks(
            vec!["absent1".to_string()],
            vec!["present1".to_string(), "present2".to_string()],
            0.01,
        );
        assert_eq!(result.definitely_absent.len(), 1);
        assert_eq!(result.possibly_present.len(), 2);
        assert_eq!(result.estimated_fpr, 0.01);
    }

    #[test]
    fn test_composite_pruner_empty() {
        let pruner = CompositePruner::new(10);
        let result = pruner.prune(None, None, None, None);
        assert_eq!(result.included_indices.len(), 10);
    }
}
