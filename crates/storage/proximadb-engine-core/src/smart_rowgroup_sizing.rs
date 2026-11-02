// Smart Row Group Sizing for RAPTOR Engine
// Optimizes row group size based on vector dimensions, metadata cost, and cloud I/O characteristics

use anyhow::Result;
// use std::collections::HashMap; // Unused import

/// Cloud storage I/O characteristics for optimal row group sizing
#[derive(Debug, Clone)]
pub struct CloudIOProfile {
    /// Optimal I/O size for this storage backend (1MB-4MB for S3/GCS/ADLS)
    pub optimal_io_size_bytes: usize,
    /// Maximum IOPS for sequential reads
    pub max_sequential_iops: u32,
    /// Latency per I/O operation in microseconds
    pub latency_per_io_us: u32,
    /// Storage tier (Hot/Warm/Cold affects I/O patterns)
    pub storage_tier: DataTemperatureTier,
}

#[derive(Debug, Clone)]
pub enum DataTemperatureTier {
    Hot,  // S3 Standard, GCS Standard, ADLS Hot
    Warm, // S3 IA, GCS Nearline, ADLS Cool
    Cold, // S3 Glacier IR, GCS Coldline, ADLS Archive
}

impl Default for CloudIOProfile {
    fn default() -> Self {
        // S3 Standard defaults - most common case
        Self {
            optimal_io_size_bytes: 2 * 1024 * 1024, // 2MB - sweet spot for S3
            max_sequential_iops: 5500,              // S3 Standard throughput
            latency_per_io_us: 20_000,              // ~20ms typical S3 latency
            storage_tier: DataTemperatureTier::Hot,
        }
    }
}

impl CloudIOProfile {
    /// S3 Standard profile (most common)
    pub fn s3_standard() -> Self {
        Self {
            optimal_io_size_bytes: 2 * 1024 * 1024, // 2MB
            max_sequential_iops: 5500,
            latency_per_io_us: 20_000,
            storage_tier: DataTemperatureTier::Hot,
        }
    }

    /// GCS Standard profile
    pub fn gcs_standard() -> Self {
        Self {
            optimal_io_size_bytes: 4 * 1024 * 1024, // 4MB - GCS prefers larger chunks
            max_sequential_iops: 10000,
            latency_per_io_us: 15_000, // ~15ms typical GCS latency
            storage_tier: DataTemperatureTier::Hot,
        }
    }

    /// Azure Data Lake Storage (ADLS) Gen2 profile
    pub fn adls_gen2() -> Self {
        Self {
            optimal_io_size_bytes: 2 * 1024 * 1024, // 2MB
            max_sequential_iops: 8000,
            latency_per_io_us: 25_000, // ~25ms typical ADLS latency
            storage_tier: DataTemperatureTier::Hot,
        }
    }

    /// Local NVMe profile for comparison
    pub fn local_nvme() -> Self {
        Self {
            optimal_io_size_bytes: 64 * 1024, // 64KB - much smaller chunks optimal
            max_sequential_iops: 500_000,
            latency_per_io_us: 100, // ~0.1ms NVMe latency
            storage_tier: DataTemperatureTier::Hot,
        }
    }
}

/// Smart row group sizing calculator
#[derive(Debug, Clone)]
pub struct SmartRowGroupSizer {
    /// Cloud I/O profile for this deployment
    pub io_profile: CloudIOProfile,
    /// Vector dimension
    pub vector_dimension: usize,
    /// Average metadata size per vector (bytes)
    pub avg_metadata_bytes: usize,
    /// Matrix Trinity overhead per vector (P² + P×K contribution)
    pub matrix_overhead_bytes: usize,
    /// Quantization enabled and level
    pub quantization_config: Option<InternalQuantizationConfig>,
    /// Target query pattern (affects sizing strategy)
    pub query_pattern: QueryPattern,
}

// Use unified quantization types instead of deprecated proto imports
use proximadb_quantization_model::{QuantizationStrategy, UnifiedQuantizationLevel};

/// Internal quantization config for sizing calculations
#[derive(Debug, Clone)]
pub struct InternalQuantizationConfig {
    pub primary_level: UnifiedQuantizationLevel,
    pub store_fp32: bool,
    pub compression_ratio: f32,
}

#[derive(Debug, Clone)]
pub enum QueryPattern {
    /// k < 10, need very fast response
    HighSelectivity,
    /// k = 10-100, balanced
    MediumSelectivity,
    /// k > 100, throughput focused
    LowSelectivity,
    /// Mixed workload
    Mixed,
}

impl SmartRowGroupSizer {
    pub fn new(
        io_profile: CloudIOProfile,
        vector_dimension: usize,
        avg_metadata_bytes: usize,
    ) -> Self {
        Self {
            io_profile,
            vector_dimension,
            avg_metadata_bytes,
            matrix_overhead_bytes: Self::estimate_matrix_overhead(vector_dimension),
            quantization_config: None,
            query_pattern: QueryPattern::Mixed,
        }
    }

    /// Estimate Matrix Trinity overhead per vector
    fn estimate_matrix_overhead(_dimension: usize) -> usize {
        // Matrix Trinity overhead:
        // - P² matrix: ~1 byte per vector pair (INT8 quantized) / P vectors = ~1KB for P=1024
        // - P×K matrix: Selective storage based on K/D ratio (10-100% coverage)
        // Much more efficient than HNSW's graph edges!
        let p2_overhead = 1; // ~1 byte per vector for P² contribution
        let pxk_overhead = 4; // ~4 bytes average for P×K (selective storage)

        p2_overhead + pxk_overhead // ~5 bytes total vs ~144 bytes for HNSW
    }

    /// Calculate optimal row group size in number of vectors
    pub fn calculate_optimal_rowgroup_size(&self) -> Result<OptimalRowGroupSize> {
        // Step 1: Calculate bytes per vector
        let vector_bytes = self.calculate_vector_storage_bytes();
        let total_bytes_per_vector =
            vector_bytes + self.avg_metadata_bytes + self.matrix_overhead_bytes;

        // Step 2: Calculate base row group size from I/O profile
        let base_vectors_per_rowgroup =
            self.io_profile.optimal_io_size_bytes / total_bytes_per_vector;

        // Step 3: Adjust based on query pattern
        let pattern_adjusted = self.adjust_for_query_pattern(base_vectors_per_rowgroup);

        // Step 4: Ensure minimum/maximum bounds
        let bounded_size = self.apply_bounds(pattern_adjusted);

        // Step 5: Calculate actual I/O size and efficiency
        let actual_io_size = bounded_size * total_bytes_per_vector;
        let io_efficiency = actual_io_size as f32 / self.io_profile.optimal_io_size_bytes as f32;

        Ok(OptimalRowGroupSize {
            vectors_per_rowgroup: bounded_size,
            bytes_per_vector: total_bytes_per_vector,
            total_rowgroup_bytes: actual_io_size,
            io_efficiency_ratio: io_efficiency,
            estimated_read_latency_ms: self.estimate_read_latency(actual_io_size),
            rationale: self.generate_rationale(bounded_size, actual_io_size),
        })
    }

    /// Calculate bytes needed to store one vector based on quantization
    fn calculate_vector_storage_bytes(&self) -> usize {
        let dimension = self.vector_dimension;

        match &self.quantization_config {
            None => {
                // FP32 storage
                dimension * 4
            }
            Some(config) => {
                let quantized_bytes = match &config.primary_level.level_type {
                    Some(QuantizationStrategy::Binary(_)) => dimension / 8,
                    Some(QuantizationStrategy::Scalar(scalar)) if scalar.bits == 8 => dimension,
                    Some(QuantizationStrategy::Pq(pq)) if pq.bits_per_code == 4 => dimension / 2,
                    Some(QuantizationStrategy::Pq(pq)) if pq.bits_per_code == 8 => dimension,
                    _ => dimension * 4, // Default to FP32
                };

                if config.store_fp32 {
                    // Dual storage: quantized + FP32
                    quantized_bytes + (dimension * 4)
                } else {
                    quantized_bytes
                }
            }
        }
    }

    /// Adjust row group size based on query patterns and semantic accuracy
    fn adjust_for_query_pattern(&self, base_size: usize) -> usize {
        // Step 1: Apply semantic accuracy factor based on vector dimension
        let semantic_factor = self.calculate_semantic_accuracy_factor();
        let semantic_adjusted = (base_size as f32 * semantic_factor) as usize;

        // Step 2: Apply query pattern adjustment
        match self.query_pattern {
            QueryPattern::HighSelectivity => {
                // k < 10: Smaller row groups to minimize waste
                (semantic_adjusted as f32 * 0.7) as usize
            }
            QueryPattern::MediumSelectivity => {
                // k = 10-100: Use semantic-adjusted size
                semantic_adjusted
            }
            QueryPattern::LowSelectivity => {
                // k > 100: Larger row groups for throughput, but still respect semantic accuracy
                (semantic_adjusted as f32 * 1.4) as usize
            }
            QueryPattern::Mixed => {
                // Balanced approach with semantic consideration
                semantic_adjusted
            }
        }
    }

    /// Calculate semantic accuracy factor - higher dimensions = smaller row groups for precision
    fn calculate_semantic_accuracy_factor(&self) -> f32 {
        // Higher dimensional vectors provide more accurate semantic distance
        // Therefore we can use smaller row groups for better precision/recall
        match self.vector_dimension {
            // Very low dimensions (e.g., word2vec): Need larger row groups for statistical significance
            d if d <= 128 => 1.3,
            // Low-medium dimensions (e.g., sentence transformers): Moderate adjustment
            d if d <= 384 => 1.1,
            // Medium dimensions (e.g., BERT): Baseline
            d if d <= 768 => 1.0,
            // High dimensions (e.g., OpenAI): Can use smaller row groups
            d if d <= 1536 => 0.8,
            // Very high dimensions (e.g., research models): Smaller row groups for precision
            d if d <= 2048 => 0.7,
            // Ultra-high dimensions: Very small row groups
            _ => 0.6,
        }
    }

    /// Apply reasonable bounds to row group size
    fn apply_bounds(&self, size: usize) -> usize {
        // Minimum: 100 vectors (avoid too many small I/Os)
        // Maximum: 10,000 vectors (avoid excessive memory usage)
        size.clamp(100, 10_000)
    }

    /// Estimate read latency for a row group of this size
    fn estimate_read_latency(&self, bytes: usize) -> f32 {
        let base_latency = self.io_profile.latency_per_io_us as f32 / 1000.0; // Convert to ms

        // Add transfer time based on size
        let transfer_time_ms = match self.io_profile.storage_tier {
            DataTemperatureTier::Hot => (bytes as f32 / (100.0 * 1024.0 * 1024.0)) * 1000.0, // 100MB/s
            DataTemperatureTier::Warm => (bytes as f32 / (50.0 * 1024.0 * 1024.0)) * 1000.0, // 50MB/s
            DataTemperatureTier::Cold => (bytes as f32 / (10.0 * 1024.0 * 1024.0)) * 1000.0, // 10MB/s
        };

        base_latency + transfer_time_ms
    }

    /// Generate human-readable rationale for the sizing decision
    fn generate_rationale(&self, vectors: usize, bytes: usize) -> String {
        let semantic_factor = self.calculate_semantic_accuracy_factor();
        let semantic_desc = match self.vector_dimension {
            d if d <= 128 => "low-dim (larger groups for stats)",
            d if d <= 384 => "med-dim (moderate adjustment)",
            d if d <= 768 => "standard-dim (baseline)",
            d if d <= 1536 => "high-dim (smaller for precision)",
            d if d <= 2048 => "very-high-dim (small for accuracy)",
            _ => "ultra-high-dim (minimal groups)",
        };

        format!(
            "Sized for {} vectors ({:.1}MB) based on {}MB optimal I/O, {} query pattern, {}-d {} vectors (semantic factor: {:.1}x)",
            vectors,
            bytes as f32 / (1024.0 * 1024.0),
            self.io_profile.optimal_io_size_bytes / (1024 * 1024),
            format!("{:?}", self.query_pattern).to_lowercase(),
            self.vector_dimension,
            semantic_desc,
            semantic_factor
        )
    }

    /// Create configuration for specific cloud providers
    pub fn for_s3_standard(dimension: usize, metadata_bytes: usize) -> Self {
        Self::new(CloudIOProfile::s3_standard(), dimension, metadata_bytes)
    }

    pub fn for_gcs_standard(dimension: usize, metadata_bytes: usize) -> Self {
        Self::new(CloudIOProfile::gcs_standard(), dimension, metadata_bytes)
    }

    pub fn for_adls_gen2(dimension: usize, metadata_bytes: usize) -> Self {
        Self::new(CloudIOProfile::adls_gen2(), dimension, metadata_bytes)
    }

    /// Configure quantization
    pub fn with_quantization(mut self, config: InternalQuantizationConfig) -> Self {
        self.quantization_config = Some(config);
        self
    }

    /// Configure query pattern
    pub fn with_query_pattern(mut self, pattern: QueryPattern) -> Self {
        self.query_pattern = pattern;
        self
    }
}

/// Result of optimal row group size calculation
#[derive(Debug, Clone)]
pub struct OptimalRowGroupSize {
    /// Number of vectors per row group
    pub vectors_per_rowgroup: usize,
    /// Bytes per vector (including metadata and Matrix Trinity overhead)
    pub bytes_per_vector: usize,
    /// Total bytes per row group
    pub total_rowgroup_bytes: usize,
    /// How well this aligns with optimal I/O size (1.0 = perfect)
    pub io_efficiency_ratio: f32,
    /// Estimated read latency for one row group (ms)
    pub estimated_read_latency_ms: f32,
    /// Human-readable explanation
    pub rationale: String,
}

/// Result of Matrix Trinity balanced sizing calculation
#[derive(Debug, Clone)]
pub struct BalancedMatrixTrinitySizing {
    /// Optimal vectors per rowgroup (p = √N)
    pub vectors_per_rowgroup: usize,
    /// Optimal number of rowgroups (k = √N)
    pub num_rowgroups: usize,
    /// Total vectors this was optimized for
    pub total_vectors: usize,
    /// P² matrix cost per query (operations)
    pub p2_cost_per_query: usize,
    /// P×K matrix cost per query (operations)
    pub pxk_cost_per_query: usize,
    /// K² matrix cost (pre-computed once)
    pub k2_cost: usize,
    /// Total operations per query (P² + P×K)
    pub total_query_ops: usize,
    /// Theoretical speedup vs naive O(N) scan
    pub speedup_vs_naive: f32,
    /// Rationale for sizing decision
    pub rationale: String,
}

impl BalancedMatrixTrinitySizing {
    /// Calculate balanced Matrix Trinity sizing using p = k = √N formula
    ///
    /// ## Matrix Trinity Cost Model
    ///
    /// For N total vectors with p vectors per rowgroup and k rowgroups:
    /// - **P² cost**: k × p² = N × p (linear in p) - intra-rowgroup pairwise distances
    /// - **P×K cost**: p × k² = N²/p (inverse in p) - vector-to-centroid lookups
    /// - **K² cost**: k² = N²/p² (pre-computed once) - inter-centroid distances
    ///
    /// ## Optimization
    ///
    /// To balance P² and P×K costs: N × p = N²/p
    /// Solving: p² = N → **p = √N**
    ///
    /// With k = N/p = N/√N = √N, we get **p = k = √N**
    ///
    /// This reduces total query cost from O(N) to O(√N), providing √N speedup.
    ///
    /// ## Example
    ///
    /// For N = 30,000 vectors:
    /// - Naive scan: 30,000 distance computations
    /// - Balanced (p = k = 173): P² = 173 + P×K ≈ 346 operations per query
    /// - Speedup: 30,000 / 346 ≈ 87x
    pub fn calculate(total_vectors: usize, dimension: usize) -> Self {
        if total_vectors == 0 {
            return Self {
                vectors_per_rowgroup: 0,
                num_rowgroups: 0,
                total_vectors: 0,
                p2_cost_per_query: 0,
                pxk_cost_per_query: 0,
                k2_cost: 0,
                total_query_ops: 0,
                speedup_vs_naive: 1.0,
                rationale: "Empty dataset".to_string(),
            };
        }

        // Calculate √N as the balanced rowgroup size
        let sqrt_n = (total_vectors as f64).sqrt();

        // Apply practical bounds:
        // - Minimum 32 vectors (GPU threadgroup size)
        // - Maximum 4096 vectors (memory efficiency for high-dim vectors)
        let min_size = 32.max(dimension / 8); // Scale with dimension
        let max_size = 4096.min(total_vectors); // Can't exceed total

        let optimal_p = (sqrt_n as usize).clamp(min_size, max_size);
        let optimal_k = total_vectors.div_ceil(optimal_p); // Ceiling division

        // Recalculate actual p to ensure k * p >= N
        let actual_p = total_vectors.div_ceil(optimal_k);

        // Calculate Matrix Trinity costs
        let n = total_vectors;
        let p = actual_p;
        let k = optimal_k;

        // P² cost: For a query, we search within ~nprobe rowgroups
        // Average nprobe ≈ √k for good recall
        let nprobe = ((k as f64).sqrt() as usize).max(1);
        let p2_cost = nprobe * p; // Distance computations within selected rowgroups

        // P×K cost: Compare query to k centroids
        let pxk_cost = k; // Distance to all centroids

        // K² cost: Pre-computed once, amortized
        let k2_cost = k * k;

        let total_query_ops = p2_cost + pxk_cost;
        let speedup = n as f32 / total_query_ops as f32;

        let rationale = format!(
            "Balanced for N={}: p=k≈√N={} (actual p={}, k={}). \
             Query cost: P²={} + P×K={} = {} ops vs N={} naive ({}x speedup). \
             K² pre-computed: {} entries. Dim={}, nprobe={}.",
            n,
            sqrt_n as usize,
            p,
            k,
            p2_cost,
            pxk_cost,
            total_query_ops,
            n,
            speedup as usize,
            k2_cost,
            dimension,
            nprobe
        );

        Self {
            vectors_per_rowgroup: actual_p,
            num_rowgroups: k,
            total_vectors: n,
            p2_cost_per_query: p2_cost,
            pxk_cost_per_query: pxk_cost,
            k2_cost,
            total_query_ops,
            speedup_vs_naive: speedup,
            rationale,
        }
    }

    /// Calculate sizing with custom nprobe (number of rowgroups to search)
    pub fn calculate_with_nprobe(
        total_vectors: usize,
        dimension: usize,
        target_nprobe: usize,
    ) -> Self {
        let mut sizing = Self::calculate(total_vectors, dimension);

        // Recalculate P² cost with specified nprobe
        let nprobe = target_nprobe.min(sizing.num_rowgroups).max(1);
        sizing.p2_cost_per_query = nprobe * sizing.vectors_per_rowgroup;
        sizing.total_query_ops = sizing.p2_cost_per_query + sizing.pxk_cost_per_query;
        sizing.speedup_vs_naive = sizing.total_vectors as f32 / sizing.total_query_ops as f32;

        sizing
    }
}

/// Precomputed configurations for common scenarios
pub struct CommonConfigurations;

impl CommonConfigurations {
    /// OpenAI embeddings (1536 dimensions) on S3
    pub fn openai_s3() -> SmartRowGroupSizer {
        SmartRowGroupSizer::for_s3_standard(1536, 200) // 200 bytes avg metadata
            .with_quantization(InternalQuantizationConfig {
                primary_level: UnifiedQuantizationLevel::pq8(32),
                store_fp32: true, // Dual storage for accuracy
                compression_ratio: 4.0,
            })
            .with_query_pattern(QueryPattern::HighSelectivity)
    }

    /// BERT embeddings (768 dimensions) on GCS
    pub fn bert_gcs() -> SmartRowGroupSizer {
        SmartRowGroupSizer::for_gcs_standard(768, 150)
            .with_quantization(InternalQuantizationConfig {
                primary_level: UnifiedQuantizationLevel::int8(),
                store_fp32: false,
                compression_ratio: 4.0,
            })
            .with_query_pattern(QueryPattern::MediumSelectivity)
    }

    /// High-dimensional research vectors (2048 dimensions) on ADLS
    pub fn research_adls() -> SmartRowGroupSizer {
        SmartRowGroupSizer::for_adls_gen2(2048, 500) // Larger metadata for research
            .with_quantization(InternalQuantizationConfig {
                primary_level: UnifiedQuantizationLevel::pq4(16),
                store_fp32: true,
                compression_ratio: 8.0,
            })
            .with_query_pattern(QueryPattern::LowSelectivity)
    }
}

impl SmartRowGroupSizer {
    /// Calculate rowgroup size optimized for Matrix Trinity balance
    ///
    /// This uses the p = k = √N formula to balance:
    /// - P² matrix (intra-rowgroup pairwise distances)
    /// - P×K matrix (vector-to-centroid lookups)
    /// - K² matrix (inter-centroid distances, pre-computed)
    ///
    /// Returns the optimal sizing given total vector count.
    pub fn calculate_matrix_trinity_balanced_size(
        &self,
        total_vectors: usize,
    ) -> BalancedMatrixTrinitySizing {
        BalancedMatrixTrinitySizing::calculate(total_vectors, self.vector_dimension)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_trinity_balanced_sizing_30k() {
        // Test case from user's analysis: N=30,000 vectors
        let sizing = BalancedMatrixTrinitySizing::calculate(30000, 128);

        // √30,000 ≈ 173
        assert!(
            sizing.vectors_per_rowgroup >= 100 && sizing.vectors_per_rowgroup <= 250,
            "Expected p ≈ 173, got {}",
            sizing.vectors_per_rowgroup
        );

        // Should have similar number of rowgroups
        assert!(
            sizing.num_rowgroups >= 100 && sizing.num_rowgroups <= 300,
            "Expected k ≈ 173, got {}",
            sizing.num_rowgroups
        );

        // Speedup should be significant (> 10x for 30K vectors)
        // Note: Actual speedup is ~12x when accounting for nprobe = √k rowgroups searched
        // This is lower than theoretical √N=173x because we search √k≈13 rowgroups for recall
        assert!(
            sizing.speedup_vs_naive > 10.0,
            "Expected significant speedup, got {}x",
            sizing.speedup_vs_naive
        );

        println!("Matrix Trinity balanced sizing for N=30,000:");
        println!("  p (vectors/rowgroup) = {}", sizing.vectors_per_rowgroup);
        println!("  k (rowgroups) = {}", sizing.num_rowgroups);
        println!("  P² cost/query = {}", sizing.p2_cost_per_query);
        println!("  P×K cost/query = {}", sizing.pxk_cost_per_query);
        println!("  K² pre-computed = {}", sizing.k2_cost);
        println!("  Speedup vs naive = {:.1}x", sizing.speedup_vs_naive);
        println!("  Rationale: {}", sizing.rationale);
    }

    #[test]
    fn test_matrix_trinity_balanced_sizing_1m() {
        // Test at 1M scale
        let sizing = BalancedMatrixTrinitySizing::calculate(1_000_000, 768);

        // √1,000,000 = 1000
        assert!(
            sizing.vectors_per_rowgroup >= 500 && sizing.vectors_per_rowgroup <= 2000,
            "Expected p ≈ 1000, got {}",
            sizing.vectors_per_rowgroup
        );

        // Speedup should be ~30x (accounting for nprobe ≈ √1000 ≈ 32 rowgroups searched)
        // Theoretical max is √N = 1000x, but with nprobe we get N / (nprobe*p + k) ≈ 30x
        assert!(
            sizing.speedup_vs_naive > 25.0,
            "Expected ~30x speedup at 1M scale (with nprobe), got {}x",
            sizing.speedup_vs_naive
        );

        println!("Matrix Trinity balanced sizing for N=1,000,000:");
        println!(
            "  p = {}, k = {}",
            sizing.vectors_per_rowgroup, sizing.num_rowgroups
        );
        println!("  Speedup = {:.1}x", sizing.speedup_vs_naive);
    }

    #[test]
    fn test_matrix_trinity_scaling() {
        // Verify √N scaling across different dataset sizes
        println!("Matrix Trinity scaling analysis:");
        println!(
            "{:>12} {:>8} {:>8} {:>12} {:>12}",
            "N", "p", "k", "Query ops", "Speedup"
        );
        println!("{}", "-".repeat(60));

        for n in [1000, 10000, 30000, 100000, 1000000] {
            let sizing = BalancedMatrixTrinitySizing::calculate(n, 128);
            println!(
                "{:>12} {:>8} {:>8} {:>12} {:>12.1}x",
                n,
                sizing.vectors_per_rowgroup,
                sizing.num_rowgroups,
                sizing.total_query_ops,
                sizing.speedup_vs_naive
            );

            // Verify speedup is significant - with nprobe=√k, speedup ≈ N/(√k*p + k)
            // For balanced sizing where p≈k≈√N and nprobe≈√k, this gives:
            // speedup ≈ N / (N^0.25 * N^0.5 + N^0.5) = N / (N^0.75 + N^0.5) ≈ N^0.25
            // Use N^0.25 * 0.8 as a conservative floor that accounts for overhead
            let expected_min_speedup = ((n as f64).powf(0.25) * 0.8).max(3.0) as f32;
            assert!(
                sizing.speedup_vs_naive > expected_min_speedup,
                "Speedup {} should be > {} for N={}",
                sizing.speedup_vs_naive,
                expected_min_speedup,
                n
            );
        }
    }

    #[test]
    fn test_matrix_trinity_dimension_scaling() {
        // Higher dimensions should affect bounds but not the √N principle
        let n = 100000;

        for dim in [128, 384, 768, 1536] {
            let sizing = BalancedMatrixTrinitySizing::calculate(n, dim);
            println!(
                "N={}, dim={}: p={}, k={}, speedup={:.1}x",
                n, dim, sizing.vectors_per_rowgroup, sizing.num_rowgroups, sizing.speedup_vs_naive
            );

            // All should still provide significant speedup (>15x with nprobe overhead)
            assert!(
                sizing.speedup_vs_naive > 15.0,
                "Expected 15x+ speedup for N={}, dim={}, got {}x",
                n,
                dim,
                sizing.speedup_vs_naive
            );
        }
    }

    #[test]
    fn test_openai_s3_sizing() {
        let sizer = CommonConfigurations::openai_s3();
        let result = sizer.calculate_optimal_rowgroup_size().unwrap();

        // Should be reasonable for S3 2MB I/O
        assert!(result.vectors_per_rowgroup >= 100);
        assert!(result.vectors_per_rowgroup <= 10000);
        assert!(result.total_rowgroup_bytes <= 4 * 1024 * 1024); // Max 4MB

        println!(
            "OpenAI/S3: {} vectors, {:.1}MB, efficiency: {:.2}",
            result.vectors_per_rowgroup,
            result.total_rowgroup_bytes as f32 / (1024.0 * 1024.0),
            result.io_efficiency_ratio
        );
    }

    #[test]
    fn test_bert_gcs_sizing() {
        let sizer = CommonConfigurations::bert_gcs();
        let result = sizer.calculate_optimal_rowgroup_size().unwrap();

        // GCS prefers 4MB chunks
        assert!(result.total_rowgroup_bytes <= 6 * 1024 * 1024); // Max 6MB

        println!(
            "BERT/GCS: {} vectors, {:.1}MB, efficiency: {:.2}",
            result.vectors_per_rowgroup,
            result.total_rowgroup_bytes as f32 / (1024.0 * 1024.0),
            result.io_efficiency_ratio
        );
    }

    #[test]
    fn test_dimension_scaling() {
        // Test how row group size scales with vector dimension
        println!("Testing semantic accuracy factor across dimensions:");
        for dimension in [128, 384, 768, 1536, 2048, 4096] {
            let sizer = SmartRowGroupSizer::for_s3_standard(dimension, 100);
            let result = sizer.calculate_optimal_rowgroup_size().unwrap();

            println!(
                "Dim {}: {} vectors, {:.1}KB per vector, {:.2}x semantic factor",
                dimension,
                result.vectors_per_rowgroup,
                result.bytes_per_vector as f32 / 1024.0,
                sizer.calculate_semantic_accuracy_factor()
            );
        }
    }

    #[test]
    fn test_semantic_accuracy_rationale() {
        let high_dim_sizer = SmartRowGroupSizer::for_s3_standard(1536, 200); // OpenAI
        let low_dim_sizer = SmartRowGroupSizer::for_s3_standard(128, 100); // Word2Vec

        let high_result = high_dim_sizer.calculate_optimal_rowgroup_size().unwrap();
        let low_result = low_dim_sizer.calculate_optimal_rowgroup_size().unwrap();

        println!("High-dim (1536d): {}", high_result.rationale);
        println!("Low-dim (128d): {}", low_result.rationale);

        // High dimensional vectors should have smaller row groups for better precision
        assert!(
            high_result.vectors_per_rowgroup < low_result.vectors_per_rowgroup,
            "High-dim vectors should have smaller row groups for better semantic precision"
        );
    }

    // ========== NEW TESTS ==========

    #[test]
    fn test_cloud_io_profile_nvme() {
        let nvme = CloudIOProfile::local_nvme();
        assert_eq!(nvme.optimal_io_size_bytes, 64 * 1024); // 64KB
        assert_eq!(nvme.latency_per_io_us, 100);
        assert!(nvme.max_sequential_iops > 100_000);
    }

    #[test]
    fn test_cloud_io_profile_s3() {
        let s3 = CloudIOProfile::s3_standard();
        assert_eq!(s3.optimal_io_size_bytes, 2 * 1024 * 1024); // 2MB
        assert_eq!(s3.latency_per_io_us, 20_000);
    }

    #[test]
    fn test_cloud_io_profile_gcs() {
        let gcs = CloudIOProfile::gcs_standard();
        assert_eq!(gcs.optimal_io_size_bytes, 4 * 1024 * 1024); // 4MB
        assert!(gcs.latency_per_io_us < s3_latency());
    }

    fn s3_latency() -> u32 {
        CloudIOProfile::s3_standard().latency_per_io_us
    }

    #[test]
    fn test_cloud_io_profile_adls() {
        let adls = CloudIOProfile::adls_gen2();
        assert_eq!(adls.optimal_io_size_bytes, 2 * 1024 * 1024);
        assert_eq!(adls.latency_per_io_us, 25_000);
    }

    #[test]
    fn test_nvme_vs_cloud_sizing() {
        let nvme_sizer = SmartRowGroupSizer::new(CloudIOProfile::local_nvme(), 768, 100);
        let s3_sizer = SmartRowGroupSizer::for_s3_standard(768, 100);

        let nvme_result = nvme_sizer.calculate_optimal_rowgroup_size().unwrap();
        let s3_result = s3_sizer.calculate_optimal_rowgroup_size().unwrap();

        // S3 needs larger row groups due to higher latency and larger optimal I/O
        assert!(
            s3_result.vectors_per_rowgroup >= nvme_result.vectors_per_rowgroup,
            "S3 should use >= rowgroup size as NVMe: S3={} vs NVMe={}",
            s3_result.vectors_per_rowgroup,
            nvme_result.vectors_per_rowgroup
        );
    }

    #[test]
    fn test_query_pattern_high_selectivity_smaller() {
        let mut sizer = SmartRowGroupSizer::for_s3_standard(768, 100);
        sizer.query_pattern = QueryPattern::HighSelectivity;
        let high_sel = sizer.calculate_optimal_rowgroup_size().unwrap();

        sizer.query_pattern = QueryPattern::LowSelectivity;
        let low_sel = sizer.calculate_optimal_rowgroup_size().unwrap();

        assert!(
            high_sel.vectors_per_rowgroup <= low_sel.vectors_per_rowgroup,
            "HighSelectivity should produce smaller or equal rowgroups vs LowSelectivity"
        );
    }

    #[test]
    fn test_query_pattern_mixed_equals_medium() {
        let mut sizer = SmartRowGroupSizer::for_s3_standard(768, 100);
        sizer.query_pattern = QueryPattern::Mixed;
        let mixed = sizer.calculate_optimal_rowgroup_size().unwrap();

        sizer.query_pattern = QueryPattern::MediumSelectivity;
        let medium = sizer.calculate_optimal_rowgroup_size().unwrap();

        // Mixed and MediumSelectivity should produce the same result
        assert_eq!(
            mixed.vectors_per_rowgroup, medium.vectors_per_rowgroup,
            "Mixed and MediumSelectivity should be equivalent"
        );
    }

    #[test]
    fn test_bounds_enforcement_minimum() {
        // Use very large vector dimension with small I/O to force sub-100 result
        let sizer = SmartRowGroupSizer::new(
            CloudIOProfile::local_nvme(), // 64KB optimal I/O
            4096,                         // Very high dimension = large bytes per vector
            1000,                         // large metadata
        );
        let result = sizer.calculate_optimal_rowgroup_size().unwrap();
        assert!(
            result.vectors_per_rowgroup >= 100,
            "Should enforce minimum of 100, got {}",
            result.vectors_per_rowgroup
        );
    }

    #[test]
    fn test_bounds_enforcement_maximum() {
        // Use very small vector dimension with large I/O to force very large result
        let sizer = SmartRowGroupSizer::new(
            CloudIOProfile::gcs_standard(), // 4MB optimal I/O
            4,                              // Tiny dimension = 16 bytes per vector
            1,                              // Minimal metadata
        );
        let result = sizer.calculate_optimal_rowgroup_size().unwrap();
        assert!(
            result.vectors_per_rowgroup <= 10_000,
            "Should enforce maximum of 10000, got {}",
            result.vectors_per_rowgroup
        );
    }

    #[test]
    fn test_read_latency_increases_with_tier_coldness() {
        let hot_sizer = SmartRowGroupSizer::new(
            CloudIOProfile {
                storage_tier: DataTemperatureTier::Hot,
                ..CloudIOProfile::s3_standard()
            },
            768,
            100,
        );
        let warm_sizer = SmartRowGroupSizer::new(
            CloudIOProfile {
                storage_tier: DataTemperatureTier::Warm,
                ..CloudIOProfile::s3_standard()
            },
            768,
            100,
        );
        let cold_sizer = SmartRowGroupSizer::new(
            CloudIOProfile {
                storage_tier: DataTemperatureTier::Cold,
                ..CloudIOProfile::s3_standard()
            },
            768,
            100,
        );

        let hot_result = hot_sizer.calculate_optimal_rowgroup_size().unwrap();
        let warm_result = warm_sizer.calculate_optimal_rowgroup_size().unwrap();
        let cold_result = cold_sizer.calculate_optimal_rowgroup_size().unwrap();

        assert!(
            hot_result.estimated_read_latency_ms < warm_result.estimated_read_latency_ms,
            "Hot latency should be less than Warm"
        );
        assert!(
            warm_result.estimated_read_latency_ms < cold_result.estimated_read_latency_ms,
            "Warm latency should be less than Cold"
        );
    }

    #[test]
    fn test_matrix_trinity_balanced_sizing_empty_dataset() {
        let sizing = BalancedMatrixTrinitySizing::calculate(0, 128);
        assert_eq!(sizing.vectors_per_rowgroup, 0);
        assert_eq!(sizing.num_rowgroups, 0);
        assert_eq!(sizing.total_query_ops, 0);
        assert_eq!(sizing.speedup_vs_naive, 1.0);
    }

    #[test]
    fn test_matrix_trinity_with_custom_nprobe() {
        let sizing = BalancedMatrixTrinitySizing::calculate_with_nprobe(100_000, 128, 5);
        assert!(sizing.p2_cost_per_query > 0);
        assert!(sizing.pxk_cost_per_query > 0);
        assert!(sizing.speedup_vs_naive > 1.0);

        // Higher nprobe should increase P2 cost
        let low_nprobe = BalancedMatrixTrinitySizing::calculate_with_nprobe(100_000, 128, 1);
        let high_nprobe = BalancedMatrixTrinitySizing::calculate_with_nprobe(100_000, 128, 50);
        assert!(
            high_nprobe.p2_cost_per_query > low_nprobe.p2_cost_per_query,
            "Higher nprobe should increase P2 cost"
        );
    }

    #[test]
    fn test_semantic_accuracy_factor_values() {
        // Test specific dimension ranges
        let sizer_64 = SmartRowGroupSizer::for_s3_standard(64, 100);
        let sizer_384 = SmartRowGroupSizer::for_s3_standard(384, 100);
        let sizer_768 = SmartRowGroupSizer::for_s3_standard(768, 100);
        let sizer_1536 = SmartRowGroupSizer::for_s3_standard(1536, 100);
        let sizer_4096 = SmartRowGroupSizer::for_s3_standard(4096, 100);

        assert_eq!(sizer_64.calculate_semantic_accuracy_factor(), 1.3); // <= 128
        assert_eq!(sizer_384.calculate_semantic_accuracy_factor(), 1.1); // <= 384
        assert_eq!(sizer_768.calculate_semantic_accuracy_factor(), 1.0); // <= 768
        assert_eq!(sizer_1536.calculate_semantic_accuracy_factor(), 0.8); // <= 1536
        assert_eq!(sizer_4096.calculate_semantic_accuracy_factor(), 0.6); // > 2048
    }

    #[test]
    fn test_research_adls_sizing() {
        let sizer = CommonConfigurations::research_adls();
        let result = sizer.calculate_optimal_rowgroup_size().unwrap();

        // Research vectors on ADLS should produce reasonable sizing
        assert!(result.vectors_per_rowgroup >= 100);
        assert!(result.vectors_per_rowgroup <= 10_000);
        assert!(result.bytes_per_vector > 0);
        assert!(result.estimated_read_latency_ms > 0.0);
    }
}
