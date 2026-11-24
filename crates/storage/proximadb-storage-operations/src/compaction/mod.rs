//! Compaction Module
//!
//! Comprehensive compaction framework with pluggable strategies and priority scheduling.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                  CompactionScheduler                         │
//! │  (Priority Queue, Concurrency Control, Rate Limiting)       │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │              CompactionStrategyRegistry                      │
//! │  (Strategy Selection, Cost Estimation, Engine Matching)     │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!         ┌────────────────────┼────────────────────┐
//!         ▼                    ▼                    ▼
//! ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
//! │   Leveled     │   │    Tiered     │   │   Custom      │
//! │  (SST, Nova)  │   │ (Helix, Swift) │   │  (Extension)  │
//! └───────────────┘   └───────────────┘   └───────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use crate::storage::operations::compaction::{
//!     CompactionScheduler, CompactionStrategyRegistry, CompactionFileMetadata,
//! };
//!
//! // Create scheduler
//! let scheduler = CompactionScheduler::new();
//!
//! // Check if compaction needed and schedule
//! let files = get_collection_files(collection_id);
//! scheduler.check_and_schedule(collection_id, "sst", &files).await?;
//!
//! // Execute with custom executor
//! scheduler.execute_next(|plan| async {
//!     // Perform actual compaction
//!     execute_compaction(plan).await
//! }).await?;
//! ```

pub mod manager;
pub mod scheduler;

// Flattened from strategies/ in Phase 2
pub mod leveled;
pub mod tiered;

// Shared types from former strategies/mod.rs
use anyhow::Result;
use async_trait::async_trait;
use std::fmt::Debug;
use std::time::{Duration, Instant};

// Re-export main types
pub use manager::{CompactionManager, CompactionStatus, CompactionType};
pub use scheduler::{CompactionScheduler, SchedulerConfig, SchedulerStats};

// Re-export strategy types (flattened from strategies/ in Phase 2)
pub use leveled::LeveledCompactionStrategy;
pub use tiered::TieredCompactionStrategy;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify all types are exported correctly
        let _scheduler = CompactionScheduler::new();
        let _registry = CompactionStrategyRegistry::new();
        let _file = CompactionFileMetadata::new("test", "/path", 1024);
        let _leveled = LeveledCompactionStrategy::new();
        let _tiered = TieredCompactionStrategy::new();
    }

    #[tokio::test]
    async fn test_integration() {
        let scheduler = CompactionScheduler::new();

        // Create test files
        let files = vec![
            CompactionFileMetadata::new("f1", "/data/f1.sst", 32 * 1024 * 1024).with_level(0),
            CompactionFileMetadata::new("f2", "/data/f2.sst", 32 * 1024 * 1024).with_level(0),
            CompactionFileMetadata::new("f3", "/data/f3.sst", 32 * 1024 * 1024).with_level(0),
            CompactionFileMetadata::new("f4", "/data/f4.sst", 32 * 1024 * 1024).with_level(0),
        ];

        // Check and schedule
        let result = scheduler.check_and_schedule("test", "sst", &files).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_some()); // Should schedule L0 compaction
    }

    #[test]
    fn test_file_metadata_builder() {
        let file = CompactionFileMetadata::new("file1", "/path/to/file.sst", 1024 * 1024)
            .with_level(2)
            .with_entries(1000)
            .with_key_range("aaa", "zzz");

        assert_eq!(file.file_id, "file1");
        assert_eq!(file.level, 2);
        assert_eq!(file.num_entries, 1000);
        assert_eq!(file.min_key, Some("aaa".to_string()));
    }

    #[test]
    fn test_registry_creation() {
        let registry = CompactionStrategyRegistry::new();
        assert!(!registry.all().is_empty());
    }

    #[test]
    fn test_find_strategy_for_engine() {
        let registry = CompactionStrategyRegistry::new();

        // Leveled should apply to SST
        let sst_strategy = registry.find_for_engine("sst");
        assert!(sst_strategy.is_some());
        assert_eq!(sst_strategy.unwrap().name(), "leveled");
    }
}

/// Backwards-compat alias for [`CompactionFileMetadata`].
pub type FileMetadata = CompactionFileMetadata;

/// Metadata about a storage file for compaction decisions
#[derive(Debug, Clone)]
pub struct CompactionFileMetadata {
    /// Unique file identifier
    pub file_id: String,
    /// File path or URL
    pub path: String,
    /// File size in bytes
    pub size_bytes: u64,
    /// Level in LSM tree (for leveled compaction)
    pub level: u32,
    /// Creation timestamp
    pub created_at: Instant,
    /// Number of entries/vectors
    pub num_entries: u64,
    /// Minimum key in file
    pub min_key: Option<String>,
    /// Maximum key in file
    pub max_key: Option<String>,
    /// Tombstone/deletion count
    pub tombstone_count: u64,
    /// Read amplification factor
    pub read_amplification: f64,
}

impl CompactionFileMetadata {
    pub fn new(file_id: impl Into<String>, path: impl Into<String>, size_bytes: u64) -> Self {
        Self {
            file_id: file_id.into(),
            path: path.into(),
            size_bytes,
            level: 0,
            created_at: Instant::now(),
            num_entries: 0,
            min_key: None,
            max_key: None,
            tombstone_count: 0,
            read_amplification: 1.0,
        }
    }

    pub fn with_level(mut self, level: u32) -> Self {
        self.level = level;
        self
    }

    pub fn with_entries(mut self, count: u64) -> Self {
        self.num_entries = count;
        self
    }

    pub fn with_key_range(mut self, min: impl Into<String>, max: impl Into<String>) -> Self {
        self.min_key = Some(min.into());
        self.max_key = Some(max.into());
        self
    }
}

/// Plan for a compaction operation
#[derive(Debug, Clone)]
pub struct CompactionPlan {
    /// Unique plan identifier
    pub plan_id: String,
    /// Collection being compacted
    pub collection_id: String,
    /// Files to be compacted
    pub input_files: Vec<CompactionFileMetadata>,
    /// Target level for output (for leveled)
    pub target_level: u32,
    /// Estimated output size
    pub estimated_output_size: u64,
    /// Priority score (higher = more urgent)
    pub priority: f64,
    /// Strategy that created this plan
    pub strategy_name: String,
    /// Additional strategy-specific parameters
    pub parameters: CompactionParameters,
}

/// Strategy-specific compaction parameters
#[derive(Debug, Clone, Default)]
pub struct CompactionParameters {
    /// Target file size for output
    pub target_file_size_bytes: u64,
    /// Whether to apply re-quantization
    pub apply_requantization: bool,
    /// Compression level (0-9)
    pub compression_level: u8,
    /// Whether to rebuild bloom filters
    pub rebuild_bloom_filters: bool,
    /// Maximum files in output
    pub max_output_files: usize,
}

/// Result of a compaction operation
#[derive(Debug, Clone)]
pub struct CompactionExecutionResult {
    /// Plan that was executed
    pub plan_id: String,
    /// Files that were removed
    pub files_removed: Vec<String>,
    /// Files that were created
    pub files_created: Vec<CompactionFileMetadata>,
    /// Bytes reclaimed
    pub bytes_freed: u64,
    /// Time taken
    pub duration: Duration,
    /// Whether operation succeeded
    pub success: bool,
    /// Error message if failed
    pub error_message: Option<String>,
}

/// Statistics about a file set for cost estimation
#[derive(Debug, Clone, Default)]
pub struct FileStatistics {
    /// Total file count
    pub file_count: usize,
    /// Total bytes across all files
    pub total_bytes: u64,
    /// Files per level
    pub files_per_level: Vec<usize>,
    /// Bytes per level
    pub bytes_per_level: Vec<u64>,
    /// Overall read amplification
    pub read_amplification: f64,
    /// Space amplification (actual / minimum)
    pub space_amplification: f64,
    /// Write amplification
    pub write_amplification: f64,
    /// Age of oldest file
    pub oldest_file_age: Duration,
    /// Tombstone ratio
    pub tombstone_ratio: f64,
}

/// Cost estimate for a compaction plan
#[derive(Debug, Clone)]
pub struct CompactionCostEstimate {
    /// Estimated time to complete
    pub estimated_time: Duration,
    /// Estimated I/O bytes (read + write)
    pub estimated_io_bytes: u64,
    /// Estimated CPU cost (arbitrary units)
    pub estimated_cpu_cost: f64,
    /// Expected space savings
    pub expected_bytes_freed: u64,
    /// Priority score for scheduling
    pub priority_score: f64,
}

/// Core compaction strategy trait
///
/// Implementations select files for compaction and define execution parameters.
/// Follows Strategy pattern for pluggable algorithms.
#[async_trait]
pub trait CompactionStrategy: Send + Sync + Debug {
    /// Strategy name for logging and identification
    fn name(&self) -> &'static str;

    /// Select files for compaction and create a plan
    ///
    /// Returns None if no compaction is needed
    async fn select_files(
        &self,
        collection_id: &str,
        files: &[CompactionFileMetadata],
    ) -> Result<Option<CompactionPlan>>;

    /// Calculate priority score for a potential compaction
    ///
    /// Higher scores indicate more urgent compaction needs
    fn priority_score(&self, stats: &FileStatistics) -> f64;

    /// Estimate the cost of executing a compaction plan
    fn estimate_cost(&self, plan: &CompactionPlan) -> CompactionCostEstimate;

    /// Check if this strategy applies to the given engine type
    fn applies_to_engine(&self, engine_name: &str) -> bool;

    /// Get optimization hints for monitoring/debugging
    fn optimization_hints(&self) -> Vec<String> {
        vec![]
    }
}

/// Registry for compaction strategies
pub struct CompactionStrategyRegistry {
    strategies: Vec<Box<dyn CompactionStrategy>>,
}

impl Default for CompactionStrategyRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl CompactionStrategyRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            strategies: Vec::new(),
        };

        // Register built-in strategies
        registry.register(Box::new(LeveledCompactionStrategy::new()));
        registry.register(Box::new(TieredCompactionStrategy::new()));

        registry
    }

    /// Register a custom compaction strategy
    pub fn register(&mut self, strategy: Box<dyn CompactionStrategy>) {
        tracing::info!("Registered compaction strategy: {}", strategy.name());
        self.strategies.push(strategy);
    }

    /// Find the best strategy for an engine
    pub fn find_for_engine(&self, engine_name: &str) -> Option<&dyn CompactionStrategy> {
        self.strategies
            .iter()
            .find(|s| s.applies_to_engine(engine_name))
            .map(|s| s.as_ref())
    }

    /// Find a strategy by its name
    pub fn find_by_name(&self, strategy_name: &str) -> Option<&dyn CompactionStrategy> {
        self.strategies
            .iter()
            .find(|s| s.name() == strategy_name)
            .map(|s| s.as_ref())
    }

    /// Find a strategy by either name or engine compatibility
    /// (tries strategy name first, then engine name)
    pub fn find(&self, name: &str) -> Option<&dyn CompactionStrategy> {
        self.find_by_name(name)
            .or_else(|| self.find_for_engine(name))
    }

    /// Get all strategies
    pub fn all(&self) -> &[Box<dyn CompactionStrategy>] {
        &self.strategies
    }

    /// Select the best plan across all applicable strategies
    pub async fn select_best_plan(
        &self,
        collection_id: &str,
        engine_name: &str,
        files: &[CompactionFileMetadata],
    ) -> Result<Option<CompactionPlan>> {
        let mut best_plan: Option<(CompactionPlan, f64)> = None;

        for strategy in &self.strategies {
            if !strategy.applies_to_engine(engine_name) {
                continue;
            }

            if let Some(plan) = strategy.select_files(collection_id, files).await? {
                let cost = strategy.estimate_cost(&plan);
                let score = cost.priority_score;

                match &best_plan {
                    None => best_plan = Some((plan, score)),
                    Some((_, best_score)) if score > *best_score => {
                        best_plan = Some((plan, score));
                    }
                    _ => {}
                }
            }
        }

        Ok(best_plan.map(|(plan, _)| plan))
    }
}
