//! Per-Engine Optimization Paths
//!
//! Defines and documents the optimization paths available for each storage engine.
//! These paths represent the discrete action space that the RL planner explores.

pub mod helix_paths;
pub mod nova_paths;
pub mod raptor_paths;
pub mod sst_paths;
pub mod swift_paths;

use super::action::ExecutionAction;
use super::state::StorageEngineType;

/// Get all optimization paths for a given engine
pub fn get_paths(engine: StorageEngineType) -> Vec<OptimizationPath> {
    match engine {
        StorageEngineType::SST => sst_paths::paths(),
        StorageEngineType::HELIX => helix_paths::paths(),
        StorageEngineType::SWIFT => swift_paths::paths(),
        StorageEngineType::NOVA => nova_paths::paths(),
        StorageEngineType::RAPTOR => raptor_paths::paths(),
        // TST (and any engine without a dedicated path set) reuses the SST
        // optimization paths as the safest baseline.
        _ => sst_paths::paths(),
    }
}

/// Optimization path with metadata
#[derive(Debug, Clone)]
pub struct OptimizationPath {
    /// Unique path identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Description of when to use this path
    pub description: String,
    /// The execution action for this path
    pub action: ExecutionAction,
    /// Expected characteristics
    pub expected: PathExpectation,
    /// Recommended use cases
    pub use_cases: Vec<String>,
}

/// Expected performance characteristics
#[derive(Debug, Clone)]
pub struct PathExpectation {
    /// Expected recall range (min, max)
    pub recall_range: (f32, f32),
    /// Expected latency multiplier vs baseline (1.0 = same as baseline)
    pub latency_multiplier: f32,
    /// Expected throughput multiplier vs baseline
    pub throughput_multiplier: f32,
    /// Memory overhead factor (1.0 = no extra memory)
    pub memory_factor: f32,
}

impl Default for PathExpectation {
    fn default() -> Self {
        Self {
            recall_range: (0.95, 1.0),
            latency_multiplier: 1.0,
            throughput_multiplier: 1.0,
            memory_factor: 1.0,
        }
    }
}

impl PathExpectation {
    /// High recall, slower
    pub fn high_recall() -> Self {
        Self {
            recall_range: (0.98, 1.0),
            latency_multiplier: 2.0,
            throughput_multiplier: 0.5,
            memory_factor: 1.5,
        }
    }

    /// Fast but lower recall
    pub fn fast_approximate() -> Self {
        Self {
            recall_range: (0.85, 0.95),
            latency_multiplier: 0.3,
            throughput_multiplier: 3.0,
            memory_factor: 0.8,
        }
    }

    /// Balanced performance
    pub fn balanced() -> Self {
        Self {
            recall_range: (0.92, 0.98),
            latency_multiplier: 0.5,
            throughput_multiplier: 2.0,
            memory_factor: 1.0,
        }
    }
}

/// Path category for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathCategory {
    /// Baseline exact search
    Baseline,
    /// Index-based approximate search
    IndexBased,
    /// Quantization-based filtering
    QuantizationBased,
    /// Block/Zone pruning
    PruningBased,
    /// Combined strategies
    Hybrid,
}

impl OptimizationPath {
    /// Create new optimization path
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
        action: ExecutionAction,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: description.into(),
            action,
            expected: PathExpectation::default(),
            use_cases: Vec::new(),
        }
    }

    /// Add expected performance characteristics
    pub fn with_expectation(mut self, expected: PathExpectation) -> Self {
        self.expected = expected;
        self
    }

    /// Add use cases
    pub fn with_use_cases(mut self, use_cases: Vec<&str>) -> Self {
        self.use_cases = use_cases.into_iter().map(String::from).collect();
        self
    }

    /// Check if path is suitable for given constraints
    pub fn is_suitable(&self, min_recall: f32, max_latency_factor: f32) -> bool {
        self.expected.recall_range.0 >= min_recall
            && self.expected.latency_multiplier <= max_latency_factor
    }

    /// Get category of this path
    pub fn category(&self) -> PathCategory {
        let action = &self.action;

        // Check for index
        let has_index = action.index_strategy.is_some()
            && !matches!(
                action.index_strategy,
                Some(super::action::IndexStrategy::DirectScan)
            );

        // Check for quantization
        let has_quant = action.quantization_stages.len() > 1
            || (action.quantization_stages.len() == 1
                && action.quantization_stages[0] != super::action::QuantizationStage::FP32);

        // Check for pruning
        let has_pruning = !matches!(action.block_pruning, super::action::BlockPruneConfig::Off)
            || action.zone_map_enabled
            || action.bloom_filter_enabled;

        match (has_index, has_quant, has_pruning) {
            (false, false, false) => PathCategory::Baseline,
            (true, false, false) => PathCategory::IndexBased,
            (false, true, false) => PathCategory::QuantizationBased,
            (false, false, true) => PathCategory::PruningBased,
            _ => PathCategory::Hybrid,
        }
    }
}

/// Filter paths by category
pub fn filter_by_category(
    paths: &[OptimizationPath],
    category: PathCategory,
) -> Vec<&OptimizationPath> {
    paths.iter().filter(|p| p.category() == category).collect()
}

/// Get recommended paths for a workload
pub fn recommend_paths(
    engine: StorageEngineType,
    collection_size: u64,
    min_recall: f32,
    latency_sensitive: bool,
) -> Vec<OptimizationPath> {
    let all_paths = get_paths(engine);

    let mut suitable: Vec<OptimizationPath> = all_paths
        .into_iter()
        .filter(|p| {
            let max_latency = if latency_sensitive { 0.5 } else { 2.0 };
            p.is_suitable(min_recall, max_latency)
        })
        .collect();

    // Sort by expected throughput (higher is better)
    suitable.sort_by(|a, b| {
        b.expected
            .throughput_multiplier
            .partial_cmp(&a.expected.throughput_multiplier)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // For small collections, prefer simpler paths
    if collection_size < 10_000 {
        suitable.retain(|p| {
            matches!(
                p.category(),
                PathCategory::Baseline | PathCategory::IndexBased
            )
        });
    }

    suitable
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_paths_all_engines() {
        for engine in &[
            StorageEngineType::SST,
            StorageEngineType::HELIX,
            StorageEngineType::SWIFT,
            StorageEngineType::NOVA,
            StorageEngineType::RAPTOR,
        ] {
            let paths = get_paths(*engine);
            assert!(!paths.is_empty(), "No paths for {:?}", engine);
        }
    }

    #[test]
    fn test_path_category() {
        let baseline = OptimizationPath::new(
            "baseline",
            "Baseline",
            "Direct scan",
            ExecutionAction::default(),
        );
        assert_eq!(baseline.category(), PathCategory::Baseline);

        let indexed = OptimizationPath::new(
            "hnsw",
            "HNSW",
            "HNSW index",
            ExecutionAction::with_hnsw(100),
        );
        assert_eq!(indexed.category(), PathCategory::IndexBased);

        let progressive = OptimizationPath::new(
            "progressive",
            "Progressive",
            "Progressive quantization",
            ExecutionAction::with_progressive_quantization(),
        );
        assert_eq!(progressive.category(), PathCategory::QuantizationBased);
    }

    #[test]
    fn test_recommend_paths() {
        let paths = recommend_paths(StorageEngineType::SST, 100_000, 0.9, false);
        assert!(!paths.is_empty());

        // All recommended paths should meet recall requirement
        for path in &paths {
            assert!(
                path.expected.recall_range.0 >= 0.9,
                "Path {} doesn't meet recall requirement",
                path.name
            );
        }
    }

    #[test]
    fn test_filter_by_category() {
        let paths = get_paths(StorageEngineType::SST);
        let indexed = filter_by_category(&paths, PathCategory::IndexBased);
        assert!(!indexed.is_empty());

        for path in indexed {
            assert_eq!(path.category(), PathCategory::IndexBased);
        }
    }
}
