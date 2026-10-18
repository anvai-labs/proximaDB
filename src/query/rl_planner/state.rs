//! Planner State Representation
//!
//! Defines the state space for the RL planner, capturing query characteristics,
//! collection properties, and system state.

use serde::{Deserialize, Serialize};

/// Canonical storage engine type (re-exported from `proximadb-storage-common` via
/// `core::types`). The previous local 6-variant copy (a subset missing TST) was
/// consolidated into the single source of truth; the canonical type carries its
/// own `Display`/`as_str`/`from_name`/`FromStr`.
pub use crate::core::types::StorageEngineType;

/// Available index types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IndexType {
    HNSW,
    IVF,
    LSH,
    Annoy,
    PQ,
    Flat,
}

impl std::fmt::Display for IndexType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HNSW => write!(f, "HNSW"),
            Self::IVF => write!(f, "IVF"),
            Self::LSH => write!(f, "LSH"),
            Self::Annoy => write!(f, "Annoy"),
            Self::PQ => write!(f, "PQ"),
            Self::Flat => write!(f, "Flat"),
        }
    }
}

/// Available quantization levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantizationLevel {
    None,
    Binary,
    INT8,
    PQ4,
    PQ8,
    FP16,
}

impl std::fmt::Display for QuantizationLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::Binary => write!(f, "Binary"),
            Self::INT8 => write!(f, "INT8"),
            Self::PQ4 => write!(f, "PQ4"),
            Self::PQ8 => write!(f, "PQ8"),
            Self::FP16 => write!(f, "FP16"),
        }
    }
}

/// Object economy metadata features for routing decisions
///
/// Encodes block-level statistics from VectorObjectEconomyDirectory
/// to enable cost-aware routing and tier-specific optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectEconomyFeatures {
    /// Whether object economy metadata is available
    pub enabled: bool,
    /// Directory freshness (0.0 = stale, 1.0 = fresh)
    pub directory_freshness: f32,
    /// Number of blocks in the SST
    pub block_count: u32,
    /// Average vectors per block
    pub block_density: f32,
    /// Average query-to-centroid distance
    pub avg_centroid_distance: f32,
    /// Z-order code quality (0.0 = poor, 1.0 = excellent clustering)
    pub zorder_code_quality: f32,
    /// Zone map filter effectiveness (0.0 = no filtering, 1.0 = perfect)
    pub zone_map_selectivity: f32,
}

impl ObjectEconomyFeatures {
    /// Create disabled features when no object economy directory is available
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            directory_freshness: 0.0,
            block_count: 0,
            block_density: 0.0,
            avg_centroid_distance: 0.0,
            zorder_code_quality: 0.0,
            zone_map_selectivity: 0.0,
        }
    }
}

impl Default for ObjectEconomyFeatures {
    fn default() -> Self {
        Self::disabled()
    }
}

/// Filter complexity categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum FilterComplexity {
    /// No filter
    #[default]
    None,
    /// Simple equality filter (e.g., category = 'books')
    Simple,
    /// Range filter (e.g., price > 10 AND price < 100)
    Range,
    /// Complex filter with multiple conditions
    Complex,
    /// Full-text search combined with vector
    FullText,
}

impl std::fmt::Display for FilterComplexity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::Simple => write!(f, "Simple"),
            Self::Range => write!(f, "Range"),
            Self::Complex => write!(f, "Complex"),
            Self::FullText => write!(f, "FullText"),
        }
    }
}

/// Complete state representation for RL planner
///
/// Captures all relevant features for decision making:
/// - Query characteristics (dimension, top_k, filters)
/// - Collection properties (size, engine, available indexes)
/// - System state (memory, CPU, cache)
/// - Historical context (recent latencies, recalls)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannerState {
    // ===== Query Characteristics =====
    /// Vector dimension
    pub query_dimension: u32,
    /// Number of results requested
    pub top_k: u32,
    /// Whether query has metadata filter
    pub has_filter: bool,
    /// Estimated filter selectivity (0.0 = selects nothing, 1.0 = selects all)
    pub filter_selectivity: f32,
    /// Filter complexity category
    pub filter_complexity: FilterComplexity,
    /// Requested search mode (from user)
    pub requested_exact: bool,

    // ===== Collection Characteristics =====
    /// Total vectors in collection
    pub collection_size: u64,
    /// Storage engine type
    pub storage_engine: StorageEngineType,
    /// Available indexes for this collection
    pub available_indexes: Vec<IndexType>,
    /// Available quantization levels
    pub available_quantization: Vec<QuantizationLevel>,
    /// Number of SST files or storage segments
    pub num_storage_segments: u32,
    /// Average vectors per segment
    pub avg_vectors_per_segment: u32,

    // ===== System State =====
    /// Memory pressure (0.0 = no pressure, 1.0 = critical)
    pub memory_pressure: f32,
    /// CPU utilization (0.0 - 1.0)
    pub cpu_utilization: f32,
    /// Number of pending queries in queue
    pub pending_queries: u32,
    /// Cache hit rate (0.0 - 1.0)
    pub cache_hit_rate: f32,
    /// Available parallelism (number of threads)
    pub available_parallelism: u32,

    // ===== Historical Context =====
    /// Recent query latencies (milliseconds, last 10 queries)
    pub recent_latencies: Vec<f32>,
    /// Recent recall rates (last 10 queries)
    pub recent_recalls: Vec<f32>,
    /// Recent throughput (QPS, last 10 measurements)
    pub recent_throughput: Vec<f32>,

    // ===== Object Economy Features =====
    /// Object economy metadata for cost-aware routing
    pub object_economy: ObjectEconomyFeatures,
}

impl Default for PlannerState {
    fn default() -> Self {
        Self {
            query_dimension: 768,
            top_k: 10,
            has_filter: false,
            filter_selectivity: 1.0,
            filter_complexity: FilterComplexity::None,
            requested_exact: false,
            collection_size: 10_000,
            storage_engine: StorageEngineType::SST,
            available_indexes: vec![IndexType::Flat],
            available_quantization: vec![QuantizationLevel::None],
            num_storage_segments: 1,
            avg_vectors_per_segment: 10_000,
            memory_pressure: 0.0,
            cpu_utilization: 0.3,
            pending_queries: 0,
            cache_hit_rate: 0.5,
            available_parallelism: num_cpus::get() as u32,
            recent_latencies: vec![10.0; 10],
            recent_recalls: vec![0.95; 10],
            recent_throughput: vec![100.0; 10],
            object_economy: ObjectEconomyFeatures::default(),
        }
    }
}

impl PlannerState {
    /// Create new planner state builder
    pub fn builder() -> PlannerStateBuilder {
        PlannerStateBuilder::default()
    }

    /// Encode state as feature vector for ML model
    pub fn as_feature_vector(&self) -> Vec<f32> {
        let mut features = Vec::with_capacity(50);

        // Query features (normalized)
        features.push(self.query_dimension as f32 / 4096.0); // Max 4096 dim
        features.push(self.top_k as f32 / 1000.0); // Max 1000 top_k
        features.push(if self.has_filter { 1.0 } else { 0.0 });
        features.push(self.filter_selectivity);
        features.push(self.filter_complexity as u8 as f32 / 4.0);
        features.push(if self.requested_exact { 1.0 } else { 0.0 });

        // Collection features (log-scaled for large values)
        // `max(1)`: log10(0) is -inf for an empty collection (TD-RLNAN-1).
        features.push((self.collection_size.max(1) as f32).log10() / 9.0); // Max 1B vectors
        features.push(self.storage_engine as u8 as f32 / 5.0);
        features.push(self.num_storage_segments as f32 / 100.0);
        features.push(self.avg_vectors_per_segment as f32 / 100_000.0);

        // Index availability (one-hot encoded)
        for idx_type in &[
            IndexType::HNSW,
            IndexType::IVF,
            IndexType::LSH,
            IndexType::Annoy,
            IndexType::PQ,
            IndexType::Flat,
        ] {
            features.push(if self.available_indexes.contains(idx_type) {
                1.0
            } else {
                0.0
            });
        }

        // Quantization availability (one-hot encoded)
        for quant in &[
            QuantizationLevel::None,
            QuantizationLevel::Binary,
            QuantizationLevel::INT8,
            QuantizationLevel::PQ4,
            QuantizationLevel::PQ8,
            QuantizationLevel::FP16,
        ] {
            features.push(if self.available_quantization.contains(quant) {
                1.0
            } else {
                0.0
            });
        }

        // System state
        features.push(self.memory_pressure);
        features.push(self.cpu_utilization);
        features.push(self.pending_queries as f32 / 100.0);
        features.push(self.cache_hit_rate);
        features.push(self.available_parallelism as f32 / 64.0);

        // Historical context (aggregated statistics)
        let avg_latency =
            self.recent_latencies.iter().sum::<f32>() / self.recent_latencies.len() as f32;
        let avg_recall = self.recent_recalls.iter().sum::<f32>() / self.recent_recalls.len() as f32;
        let avg_throughput =
            self.recent_throughput.iter().sum::<f32>() / self.recent_throughput.len() as f32;

        features.push(avg_latency / 100.0); // Normalize to ~1.0 for 100ms
        features.push(avg_recall);
        features.push(avg_throughput / 1000.0); // Normalize to ~1.0 for 1000 QPS

        // Latency variance (stability indicator)
        let latency_variance = self
            .recent_latencies
            .iter()
            .map(|l| (l - avg_latency).powi(2))
            .sum::<f32>()
            / self.recent_latencies.len() as f32;
        features.push(latency_variance.sqrt() / avg_latency);

        // Object economy features
        features.push(if self.object_economy.enabled {
            1.0
        } else {
            0.0
        });
        features.push(self.object_economy.directory_freshness);
        features.push(self.object_economy.block_count as f32 / 10000.0);
        features.push(self.object_economy.block_density / 10000.0);
        features.push(self.object_economy.avg_centroid_distance);
        features.push(self.object_economy.zorder_code_quality);
        features.push(self.object_economy.zone_map_selectivity);

        // TD-RLNAN-1: empty histories divide 0 by 0 and caller-supplied ratios
        // can be NaN. A non-finite feature turns a learned weight into NaN,
        // which persists as `null` and makes the saved policy unloadable, so
        // every feature leaves here finite.
        for feature in &mut features {
            if !feature.is_finite() {
                *feature = 0.0;
            }
        }
        features
    }

    /// Compute context hash for caching similar states
    pub fn context_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        // Hash key features that define similar contexts
        self.storage_engine.hash(&mut hasher);
        (self.collection_size / 1000).hash(&mut hasher); // Bucket by 1000s
        (self.query_dimension / 128).hash(&mut hasher); // Bucket by 128s
        self.top_k.hash(&mut hasher);
        self.has_filter.hash(&mut hasher);
        ((self.filter_selectivity * 10.0) as u32).hash(&mut hasher); // 0.1 granularity

        hasher.finish()
    }

    /// Check if collection is large enough for approximate search to be beneficial
    pub fn should_consider_approximate(&self) -> bool {
        self.collection_size > 1000 && !self.requested_exact
    }

    /// Check if HNSW index is available and beneficial
    pub fn can_use_hnsw(&self) -> bool {
        self.available_indexes.contains(&IndexType::HNSW) && self.collection_size > 1000
    }

    /// Check if IVF index is available and beneficial
    pub fn can_use_ivf(&self) -> bool {
        self.available_indexes.contains(&IndexType::IVF) && self.collection_size > 10_000
    }

    /// Check if progressive quantization is beneficial
    pub fn should_use_progressive_quantization(&self) -> bool {
        self.collection_size > 5000
            && (self
                .available_quantization
                .contains(&QuantizationLevel::Binary)
                || self
                    .available_quantization
                    .contains(&QuantizationLevel::INT8))
    }
}

/// Builder for PlannerState
#[derive(Default)]
pub struct PlannerStateBuilder {
    state: PlannerState,
}

impl PlannerStateBuilder {
    pub fn query_dimension(mut self, dim: u32) -> Self {
        self.state.query_dimension = dim;
        self
    }

    pub fn top_k(mut self, k: u32) -> Self {
        self.state.top_k = k;
        self
    }

    pub fn with_filter(mut self, selectivity: f32, complexity: FilterComplexity) -> Self {
        self.state.has_filter = true;
        self.state.filter_selectivity = selectivity;
        self.state.filter_complexity = complexity;
        self
    }

    pub fn collection_size(mut self, size: u64) -> Self {
        self.state.collection_size = size;
        self
    }

    pub fn storage_engine(mut self, engine: StorageEngineType) -> Self {
        self.state.storage_engine = engine;
        self
    }

    pub fn available_indexes(mut self, indexes: Vec<IndexType>) -> Self {
        self.state.available_indexes = indexes;
        self
    }

    pub fn available_quantization(mut self, levels: Vec<QuantizationLevel>) -> Self {
        self.state.available_quantization = levels;
        self
    }

    pub fn memory_pressure(mut self, pressure: f32) -> Self {
        self.state.memory_pressure = pressure;
        self
    }

    pub fn cpu_utilization(mut self, utilization: f32) -> Self {
        self.state.cpu_utilization = utilization;
        self
    }

    pub fn recent_latencies(mut self, latencies: Vec<f32>) -> Self {
        self.state.recent_latencies = latencies;
        self
    }

    pub fn recent_recalls(mut self, recalls: Vec<f32>) -> Self {
        self.state.recent_recalls = recalls;
        self
    }

    pub fn object_economy(mut self, oe: ObjectEconomyFeatures) -> Self {
        self.state.object_economy = oe;
        self
    }

    pub fn build(self) -> PlannerState {
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_state() {
        let state = PlannerState::default();
        assert_eq!(state.query_dimension, 768);
        assert_eq!(state.top_k, 10);
        assert!(!state.has_filter);
    }

    #[test]
    fn test_state_builder() {
        let state = PlannerState::builder()
            .query_dimension(1536)
            .top_k(100)
            .collection_size(1_000_000)
            .storage_engine(StorageEngineType::HELIX)
            .available_indexes(vec![IndexType::HNSW, IndexType::IVF])
            .build();

        assert_eq!(state.query_dimension, 1536);
        assert_eq!(state.top_k, 100);
        assert_eq!(state.collection_size, 1_000_000);
        assert!(matches!(state.storage_engine, StorageEngineType::HELIX));
        assert!(state.can_use_hnsw());
        assert!(state.can_use_ivf());
    }

    #[test]
    fn test_feature_vector() {
        let state = PlannerState::default();
        let features = state.as_feature_vector();

        // Should have consistent length
        assert!(features.len() > 20);

        // All features should be normalized roughly to [0, 1]
        for (i, &f) in features.iter().enumerate() {
            assert!(
                (0.0..=10.0).contains(&f),
                "Feature {} = {} out of range",
                i,
                f
            );
        }
    }

    #[test]
    fn test_context_hash() {
        let state1 = PlannerState::builder()
            .collection_size(10_000)
            .storage_engine(StorageEngineType::SST)
            .build();

        let state2 = PlannerState::builder()
            .collection_size(10_500) // Similar, same bucket
            .storage_engine(StorageEngineType::SST)
            .build();

        let state3 = PlannerState::builder()
            .collection_size(100_000) // Different bucket
            .storage_engine(StorageEngineType::SST)
            .build();

        assert_eq!(state1.context_hash(), state2.context_hash());
        assert_ne!(state1.context_hash(), state3.context_hash());
    }

    #[test]
    fn test_should_consider_approximate() {
        let small = PlannerState::builder().collection_size(500).build();
        let large = PlannerState::builder().collection_size(10_000).build();

        assert!(!small.should_consider_approximate());
        assert!(large.should_consider_approximate());
    }

    #[test]
    fn test_object_economy_features_disabled() {
        let oe = ObjectEconomyFeatures::disabled();
        assert!(!oe.enabled);
        assert_eq!(oe.directory_freshness, 0.0);
    }

    #[test]
    fn test_object_economy_features_in_state() {
        let state = PlannerState::builder()
            .object_economy(ObjectEconomyFeatures {
                enabled: true,
                directory_freshness: 0.9,
                block_count: 100,
                block_density: 1000.0,
                avg_centroid_distance: 0.5,
                zorder_code_quality: 0.8,
                zone_map_selectivity: 0.7,
            })
            .build();

        assert!(state.object_economy.enabled);
        assert_eq!(state.object_economy.block_count, 100);
    }

    #[test]
    fn test_feature_vector_includes_object_economy() {
        let state = PlannerState::builder()
            .object_economy(ObjectEconomyFeatures {
                enabled: true,
                directory_freshness: 0.9,
                block_count: 100,
                block_density: 1000.0,
                avg_centroid_distance: 0.5,
                zorder_code_quality: 0.8,
                zone_map_selectivity: 0.7,
            })
            .build();

        let features = state.as_feature_vector();

        // Object economy contributes the trailing 7 features of the vector.
        // Assert against the actual encoding (rather than a brittle total
        // count) so the test stays meaningful as the base feature set evolves.
        assert!(
            features.len() >= 7,
            "feature vector must include the object-economy block"
        );
        let oe = &features[features.len() - 7..];
        assert_eq!(oe[0], 1.0, "enabled");
        assert!((oe[1] - 0.9).abs() < 1e-6, "directory_freshness");
        assert!((oe[2] - 100.0 / 10_000.0).abs() < 1e-6, "block_count");
        assert!((oe[3] - 1000.0 / 10_000.0).abs() < 1e-6, "block_density");
        assert!((oe[4] - 0.5).abs() < 1e-6, "avg_centroid_distance");
        assert!((oe[5] - 0.8).abs() < 1e-6, "zorder_code_quality");
        assert!((oe[6] - 0.7).abs() < 1e-6, "zone_map_selectivity");
    }
}
