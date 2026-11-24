//! Tiered Compaction Strategy
//!
//! Implements size-tiered compaction, ideal for write-heavy workloads.
//! Used by the HELIX engine (and Parquet-layout engines).

use anyhow::Result;
use async_trait::async_trait;
use std::time::Duration;

use super::{
    CompactionCostEstimate, CompactionParameters, CompactionPlan, CompactionStrategy, FileMetadata,
    FileStatistics,
};

/// Tiered compaction strategy (STCS - Size-Tiered Compaction Strategy)
///
/// # Algorithm
///
/// 1. Group files into size tiers (similar sizes together)
/// 2. When tier has enough files (min_threshold), merge them
/// 3. Creates larger files that move to higher tiers
/// 4. Optimized for write throughput
///
/// # Performance Characteristics
///
/// - **Write Amplification**: O(2-4x) - excellent for writes
/// - **Read Amplification**: O(N) - may need to check multiple files
/// - **Space Amplification**: O(2x) - moderate space usage
///
/// # Best For
///
/// - Write-heavy workloads
/// - Time-series data
/// - Columnar formats (Parquet)
/// - HELIX spatial clustering
#[derive(Debug, Clone)]
pub struct TieredCompactionStrategy {
    /// Minimum files in a tier to trigger compaction
    min_threshold: usize,
    /// Maximum files in a tier
    max_threshold: usize,
    /// Size multiplier between tiers (typically 4)
    size_tier_ratio: f64,
    /// Minimum file size for a tier (bytes)
    min_file_size: u64,
    /// Maximum file size (stop compacting beyond this)
    max_file_size: u64,
    /// Whether to apply clustering optimization (for HELIX)
    enable_clustering: bool,
}

impl Default for TieredCompactionStrategy {
    fn default() -> Self {
        Self {
            min_threshold: 4,
            max_threshold: 32,
            size_tier_ratio: 4.0,
            min_file_size: 4 * 1024 * 1024,   // 4MB
            max_file_size: 512 * 1024 * 1024, // 512MB
            enable_clustering: false,
        }
    }
}

impl TieredCompactionStrategy {
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable clustering for HELIX engine
    pub fn with_clustering(mut self, enable: bool) -> Self {
        self.enable_clustering = enable;
        self
    }

    /// Configure tier thresholds
    pub fn with_thresholds(mut self, min: usize, max: usize) -> Self {
        self.min_threshold = min;
        self.max_threshold = max;
        self
    }

    /// Configure size ratio between tiers
    pub fn with_size_ratio(mut self, ratio: f64) -> Self {
        self.size_tier_ratio = ratio;
        self
    }

    /// Get tier index for a file based on its size
    fn get_tier_index(&self, file_size: u64) -> usize {
        if file_size < self.min_file_size {
            return 0;
        }

        let mut tier = 0;
        let mut tier_size = self.min_file_size;

        while tier_size < file_size && tier_size < self.max_file_size {
            tier_size = (tier_size as f64 * self.size_tier_ratio) as u64;
            tier += 1;
        }

        tier
    }

    /// Group files into size tiers
    fn group_into_tiers<'a>(&self, files: &'a [FileMetadata]) -> Vec<Vec<&'a FileMetadata>> {
        // Calculate max tier based on max file size
        let max_tier = self.get_tier_index(self.max_file_size) + 1;
        let mut tiers: Vec<Vec<&'a FileMetadata>> = vec![Vec::new(); max_tier];

        for file in files {
            let tier = self.get_tier_index(file.size_bytes).min(max_tier - 1);
            tiers[tier].push(file);
        }

        tiers
    }

    /// Calculate average size for a tier
    fn tier_average_size(&self, tier: usize) -> u64 {
        if tier == 0 {
            return self.min_file_size;
        }
        (self.min_file_size as f64 * self.size_tier_ratio.powi(tier as i32)) as u64
    }

    /// Select best tier for compaction
    fn select_tier_for_compaction(
        &self,
        collection_id: &str,
        tiers: &[Vec<&FileMetadata>],
    ) -> Option<CompactionPlan> {
        // Check tiers from smallest to largest
        for (tier_idx, tier_files) in tiers.iter().enumerate() {
            if tier_files.len() < self.min_threshold {
                continue;
            }

            // Skip if files are already at max size
            let avg_size = tier_files.iter().map(|f| f.size_bytes).sum::<u64>()
                / tier_files.len().max(1) as u64;
            if avg_size >= self.max_file_size {
                continue;
            }

            // Select files to compact (up to max_threshold)
            let files_to_compact: Vec<FileMetadata> = tier_files
                .iter()
                .take(self.max_threshold)
                .map(|f| (*f).clone())
                .collect();

            let estimated_output_size: u64 = files_to_compact.iter().map(|f| f.size_bytes).sum();
            let next_tier_avg = self.tier_average_size(tier_idx + 1);

            return Some(CompactionPlan {
                plan_id: format!(
                    "tiered_t{}_{}",
                    tier_idx,
                    chrono::Utc::now().timestamp_millis()
                ),
                collection_id: collection_id.to_string(),
                input_files: files_to_compact,
                target_level: (tier_idx + 1) as u32, // Move to next tier
                estimated_output_size,
                priority: 50.0 + (10.0 / (tier_idx + 1) as f64), // Smaller tiers have higher priority
                strategy_name: "tiered".to_string(),
                parameters: CompactionParameters {
                    target_file_size_bytes: next_tier_avg,
                    apply_requantization: self.enable_clustering,
                    compression_level: 6,
                    rebuild_bloom_filters: true,
                    max_output_files: ((estimated_output_size / next_tier_avg) + 1) as usize,
                },
            });
        }

        None
    }

    /// Check for files that can benefit from re-clustering (HELIX)
    fn select_clustering_compaction(
        &self,
        collection_id: &str,
        files: &[FileMetadata],
    ) -> Option<CompactionPlan> {
        if !self.enable_clustering {
            return None;
        }

        // Find files with poor clustering (high read amplification)
        let fragmented_files: Vec<&FileMetadata> = files
            .iter()
            .filter(|f| f.read_amplification > 2.0)
            .collect();

        if fragmented_files.len() < 2 {
            return None;
        }

        let input_files: Vec<FileMetadata> = fragmented_files
            .iter()
            .take(8) // Limit to 8 files for clustering
            .map(|f| (*f).clone())
            .collect();

        let estimated_output_size: u64 = input_files.iter().map(|f| f.size_bytes).sum();

        Some(CompactionPlan {
            plan_id: format!("tiered_cluster_{}", chrono::Utc::now().timestamp_millis()),
            collection_id: collection_id.to_string(),
            input_files,
            target_level: 0, // Clustering doesn't change levels
            estimated_output_size,
            priority: 30.0, // Lower than regular compaction
            strategy_name: "tiered_clustering".to_string(),
            parameters: CompactionParameters {
                target_file_size_bytes: self.max_file_size / 4,
                apply_requantization: true, // Rebuild spatial index
                compression_level: 6,
                rebuild_bloom_filters: true,
                max_output_files: 4,
            },
        })
    }
}

#[async_trait]
impl CompactionStrategy for TieredCompactionStrategy {
    fn name(&self) -> &'static str {
        "tiered"
    }

    async fn select_files(
        &self,
        collection_id: &str,
        files: &[FileMetadata],
    ) -> Result<Option<CompactionPlan>> {
        if files.is_empty() {
            return Ok(None);
        }

        let tiers = self.group_into_tiers(files);

        // Priority 1: Regular tiered compaction
        if let Some(plan) = self.select_tier_for_compaction(collection_id, &tiers) {
            tracing::debug!(
                "TieredCompaction: selected tier {} compaction with {} files",
                plan.target_level - 1,
                plan.input_files.len()
            );
            return Ok(Some(plan));
        }

        // Priority 2: Clustering compaction (for HELIX)
        if let Some(plan) = self.select_clustering_compaction(collection_id, files) {
            tracing::debug!(
                "TieredCompaction: selected clustering compaction with {} files",
                plan.input_files.len()
            );
            return Ok(Some(plan));
        }

        Ok(None)
    }

    fn priority_score(&self, stats: &FileStatistics) -> f64 {
        let mut score = 0.0;

        // More files = higher priority
        score += (stats.file_count as f64).ln() * 10.0;

        // High space amplification increases priority
        if stats.space_amplification > 2.0 {
            score += (stats.space_amplification - 1.0) * 25.0;
        }

        // High read amplification increases priority
        score += stats.read_amplification * 5.0;

        // Old files should be compacted
        let age_hours = stats.oldest_file_age.as_secs() as f64 / 3600.0;
        if age_hours > 24.0 {
            score += (age_hours / 24.0).min(10.0) * 5.0;
        }

        score
    }

    fn estimate_cost(&self, plan: &CompactionPlan) -> CompactionCostEstimate {
        let input_size: u64 = plan.input_files.iter().map(|f| f.size_bytes).sum();
        let output_size = plan.estimated_output_size;

        // Total I/O = read input + write output
        let total_io = input_size + output_size;

        // Tiered compaction is simpler (concatenation), estimate ~150 MB/s
        let throughput_bytes_per_sec = 150 * 1024 * 1024;
        let mut estimated_seconds = total_io as f64 / throughput_bytes_per_sec as f64;

        // Add extra time for clustering
        if plan.parameters.apply_requantization {
            estimated_seconds *= 1.5;
        }

        CompactionCostEstimate {
            estimated_time: Duration::from_secs_f64(estimated_seconds),
            estimated_io_bytes: total_io,
            estimated_cpu_cost: plan.input_files.len() as f64 * 5.0, // Less CPU than leveled
            expected_bytes_freed: (input_size as f64 * 0.05) as u64, // ~5% reduction (less than leveled)
            priority_score: plan.priority,
        }
    }

    fn applies_to_engine(&self, engine_name: &str) -> bool {
        matches!(
            engine_name.to_lowercase().as_str(),
            "helix" | "swift" | "parquet"
        )
    }

    fn optimization_hints(&self) -> Vec<String> {
        vec![
            format!("min_threshold: {} files", self.min_threshold),
            format!("max_threshold: {} files", self.max_threshold),
            format!("size_tier_ratio: {}", self.size_tier_ratio),
            format!(
                "file_size_range: {} MB - {} MB",
                self.min_file_size / (1024 * 1024),
                self.max_file_size / (1024 * 1024)
            ),
            format!("clustering_enabled: {}", self.enable_clustering),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_files() -> Vec<FileMetadata> {
        vec![
            // Tier 0 (small files, exactly 4MB to stay in tier 0)
            FileMetadata::new("small_1", "/data/small_1.parquet", 4 * 1024 * 1024),
            FileMetadata::new("small_2", "/data/small_2.parquet", 4 * 1024 * 1024),
            FileMetadata::new("small_3", "/data/small_3.parquet", 4 * 1024 * 1024),
            FileMetadata::new("small_4", "/data/small_4.parquet", 4 * 1024 * 1024),
            // Tier 1 (medium files, ~16MB each)
            FileMetadata::new("medium_1", "/data/medium_1.parquet", 16 * 1024 * 1024),
            FileMetadata::new("medium_2", "/data/medium_2.parquet", 16 * 1024 * 1024),
            // Tier 2 (large files, ~64MB each)
            FileMetadata::new("large_1", "/data/large_1.parquet", 64 * 1024 * 1024),
        ]
    }

    #[tokio::test]
    async fn test_tiered_compaction_trigger() {
        let strategy = TieredCompactionStrategy::new();
        let files = create_test_files();

        let plan = strategy.select_files("test", &files).await.unwrap();
        assert!(plan.is_some());

        let plan = plan.unwrap();
        // Should compact the 4 small files in tier 0
        assert_eq!(plan.input_files.len(), 4);
    }

    #[tokio::test]
    async fn test_no_compaction_needed() {
        let strategy = TieredCompactionStrategy::new();
        let files = vec![
            FileMetadata::new("f1", "/data/f1.parquet", 4 * 1024 * 1024),
            FileMetadata::new("f2", "/data/f2.parquet", 5 * 1024 * 1024),
        ];

        let plan = strategy.select_files("test", &files).await.unwrap();
        assert!(plan.is_none()); // Only 2 files, below threshold
    }

    #[test]
    fn test_tier_calculation() {
        let strategy = TieredCompactionStrategy::new();

        assert_eq!(strategy.get_tier_index(2 * 1024 * 1024), 0); // 2MB → tier 0
        assert_eq!(strategy.get_tier_index(4 * 1024 * 1024), 0); // 4MB → tier 0
        assert_eq!(strategy.get_tier_index(16 * 1024 * 1024), 1); // 16MB → tier 1
        assert_eq!(strategy.get_tier_index(64 * 1024 * 1024), 2); // 64MB → tier 2
    }

    #[test]
    fn test_priority_score() {
        let strategy = TieredCompactionStrategy::new();

        let stats = FileStatistics {
            file_count: 20,
            space_amplification: 2.5,
            read_amplification: 3.0,
            oldest_file_age: Duration::from_secs(48 * 3600), // 48 hours
            ..Default::default()
        };

        let score = strategy.priority_score(&stats);
        assert!(score > 50.0); // Should be relatively high priority
    }

    #[test]
    fn test_applies_to_engine() {
        let strategy = TieredCompactionStrategy::new();

        assert!(!strategy.applies_to_engine("viper"));
        assert!(strategy.applies_to_engine("helix"));
        assert!(strategy.applies_to_engine("swift"));
        assert!(!strategy.applies_to_engine("sst"));
    }

    #[tokio::test]
    async fn test_clustering_compaction() {
        let strategy = TieredCompactionStrategy::new().with_clustering(true);

        let mut files = vec![
            FileMetadata::new("frag_1", "/data/frag_1.helix", 32 * 1024 * 1024),
            FileMetadata::new("frag_2", "/data/frag_2.helix", 32 * 1024 * 1024),
            FileMetadata::new("frag_3", "/data/frag_3.helix", 32 * 1024 * 1024),
        ];

        // Set high read amplification to trigger clustering
        for file in &mut files {
            file.read_amplification = 3.5;
        }

        let plan = strategy.select_files("test", &files).await.unwrap();
        assert!(plan.is_some());

        let plan = plan.unwrap();
        assert_eq!(plan.strategy_name, "tiered_clustering");
        assert!(plan.parameters.apply_requantization);
    }
}
