// Google Artus-inspired Bloom Filter Implementation for RAPTOR
// Provides per-column bloom filters based on cardinality and access patterns

use anyhow::Result;
// Use core bloom filter implementation
use crate::core::bloom::factory::BloomFilterFactory;
use crate::core::bloom::{BloomFilterConfig, BloomFilterStrategy};
use std::collections::HashMap;

/// Artus-style column statistics for intelligent bloom filter sizing
#[derive(Debug, Clone)]
pub struct ArtusColumnStats {
    pub column_name: String,
    pub cardinality: usize,
    pub null_ratio: f32,
    pub access_frequency: u64,
    pub selectivity: f32,
    pub data_type: ColumnData,
    pub bloom_benefit_score: f32,
}

#[derive(Debug, Clone)]
pub enum ColumnData {
    Integer,
    Float,
    String,
    Binary,
    Boolean,
    Timestamp,
    Vector,
}

/// Artus-inspired adaptive bloom filter configuration
#[derive(Debug, Clone)]
pub struct ArtusBloomConfig {
    /// Target false positive rate
    pub false_positive_rate: f64,
    /// Minimum cardinality to create bloom filter
    pub min_cardinality: usize,
    /// Maximum memory per bloom filter (bytes)
    pub max_memory_per_filter: usize,
    /// Enable adaptive sizing based on access patterns
    pub adaptive_sizing: bool,
    /// Cardinality threshold for automatic bloom creation
    pub auto_bloom_threshold: f32,
}

impl Default for ArtusBloomConfig {
    fn default() -> Self {
        Self {
            false_positive_rate: 0.01,
            min_cardinality: 100,
            max_memory_per_filter: 1024 * 1024, // 1MB
            adaptive_sizing: true,
            auto_bloom_threshold: 0.7, // Create bloom if cardinality < 70% of rows
        }
    }
}

/// Artus-style column bloom filter manager
pub struct ArtusBloomManager {
    config: ArtusBloomConfig,
    /// Bloom filters per column
    column_blooms: HashMap<String, Box<dyn BloomFilterStrategy>>,
    /// Column statistics for intelligent management
    column_stats: HashMap<String, ArtusColumnStats>,
    /// Serialized bloom filters for persistence
    serialized_blooms: HashMap<String, Vec<u8>>,
}

impl ArtusBloomManager {
    pub fn new(config: ArtusBloomConfig) -> Self {
        Self {
            config,
            column_blooms: HashMap::new(),
            column_stats: HashMap::new(),
            serialized_blooms: HashMap::new(),
        }
    }

    /// Analyze column and decide if bloom filter is beneficial
    pub fn should_create_bloom(&self, stats: &ArtusColumnStats) -> bool {
        // Google Artus-inspired heuristics
        let cardinality_ratio = stats.cardinality as f32 / stats.access_frequency as f32;

        // Create bloom if:
        // 1. Cardinality is within reasonable bounds
        // 2. Column is frequently accessed
        // 3. Selectivity suggests bloom would help
        stats.cardinality >= self.config.min_cardinality
            && stats.cardinality <= 1_000_000  // Not too high cardinality
            && cardinality_ratio < self.config.auto_bloom_threshold
            && stats.selectivity < 0.5  // Selective queries benefit most
            && stats.access_frequency > 10 // Accessed frequently enough
    }

    /// Calculate optimal bloom filter size based on Artus principles
    fn calculate_optimal_size(&self, stats: &ArtusColumnStats) -> (usize, usize) {
        // Artus formula: balance memory vs false positive rate
        let items = stats.cardinality;
        let fp_rate = self.config.false_positive_rate;

        // Optimal number of bits
        let bits = (-(items as f64) * fp_rate.ln() / (2.0_f64.ln().powi(2))).ceil() as usize;

        // Cap at maximum memory
        let capped_bits = bits.min(self.config.max_memory_per_filter * 8);

        // Optimal number of hash functions
        let hash_functions = ((capped_bits as f64 / items as f64) * 2.0_f64.ln()).round() as usize;

        (capped_bits, hash_functions.max(1))
    }

    /// Create bloom filter for a column based on its statistics
    pub fn create_column_bloom(&mut self, stats: ArtusColumnStats) -> Result<()> {
        if !self.should_create_bloom(&stats) {
            tracing::debug!(
                "Skipping bloom filter for column {} (cardinality: {}, selectivity: {})",
                stats.column_name,
                stats.cardinality,
                stats.selectivity
            );
            return Ok(());
        }

        let (bits, hash_functions) = self.calculate_optimal_size(&stats);

        tracing::info!(
            "Creating Artus bloom filter for column {} with {} bits, {} hash functions (cardinality: {})",
            stats.column_name,
            bits,
            hash_functions,
            stats.cardinality
        );

        // Create bloom filter with calculated parameters
        let config = BloomFilterConfig {
            strategy: crate::core::bloom::BloomStrategy::ByteAligned,
            bits_per_key: 10,
            false_positive_rate: Some(self.config.false_positive_rate),
            expected_items: stats.cardinality,
            enabled: true,
            hash_algorithm: crate::core::bloom::HashAlgorithm::Murmur3,
        };
        let bloom = BloomFilterFactory::create(&config);

        self.column_blooms.insert(stats.column_name.clone(), bloom);
        self.column_stats.insert(stats.column_name.clone(), stats);

        Ok(())
    }

    /// Add value to column's bloom filter
    pub fn add_to_bloom(&mut self, column: &str, value: &str) {
        if let Some(bloom) = self.column_blooms.get_mut(column) {
            bloom.insert(value.as_bytes());

            // Update access statistics
            if let Some(stats) = self.column_stats.get_mut(column) {
                stats.access_frequency += 1;
            }
        }
    }

    /// Check if value might exist in column
    pub fn check_bloom(&self, column: &str, value: &str) -> Option<bool> {
        self.column_blooms
            .get(column)
            .map(|bloom| bloom.might_contain(value.as_bytes()))
    }

    /// Batch add values to bloom filter
    pub fn batch_add_to_bloom(&mut self, column: &str, values: &[String]) {
        if let Some(bloom) = self.column_blooms.get_mut(column) {
            for value in values {
                bloom.insert(value.as_bytes());
            }

            // Update statistics
            if let Some(stats) = self.column_stats.get_mut(column) {
                stats.access_frequency += values.len() as u64;
            }
        }
    }

    /// Serialize bloom filters for persistence
    pub fn serialize_blooms(&mut self) -> Result<HashMap<String, Vec<u8>>> {
        let mut serialized = HashMap::new();

        for (column, bloom) in &self.column_blooms {
            // Serialize bloom filter to bytes
            let bytes = self.serialize_bloom(bloom.as_ref())?;
            serialized.insert(column.clone(), bytes);
        }

        self.serialized_blooms = serialized.clone();
        Ok(serialized)
    }

    /// Deserialize bloom filters from storage
    pub fn deserialize_blooms(&mut self, data: HashMap<String, Vec<u8>>) -> Result<()> {
        for (column, bytes) in data {
            let bloom = self.deserialize_bloom(&bytes)?;
            self.column_blooms.insert(column, bloom);
        }

        Ok(())
    }

    /// Get bloom filter statistics
    pub fn get_bloom_stats(&self, column: &str) -> Option<BloomStats> {
        self.column_blooms.get(column).map(|bloom| {
            let stats = self.column_stats.get(column);
            BloomStats {
                column: column.to_string(),
                size_bytes: bloom.memory_usage(),
                num_hash_functions: bloom.hash_count(),
                items_added: stats.map_or(0, |s| s.cardinality),
                false_positive_rate: self.config.false_positive_rate,
                access_frequency: stats.map_or(0, |s| s.access_frequency),
            }
        })
    }

    /// Optimize bloom filters based on access patterns (Artus adaptive strategy)
    pub fn optimize_blooms(&mut self) -> Result<()> {
        if !self.config.adaptive_sizing {
            return Ok(());
        }

        let mut columns_to_resize = Vec::new();

        for (column, stats) in &self.column_stats {
            // Check if bloom needs resizing based on actual vs expected cardinality
            if let Some(bloom) = self.column_blooms.get(column) {
                let current_fp_rate =
                    self.estimate_false_positive_rate(bloom.as_ref(), stats.cardinality);

                // Resize if FP rate deviates significantly
                if (current_fp_rate - self.config.false_positive_rate).abs() > 0.05 {
                    columns_to_resize.push((column.clone(), stats.clone()));
                }
            }
        }

        // Resize identified blooms
        for (column, stats) in columns_to_resize {
            tracing::info!("Optimizing bloom filter for column {}", column);
            self.column_blooms.remove(&column);
            self.create_column_bloom(stats)?;
        }

        Ok(())
    }

    /// Estimate current false positive rate
    fn estimate_false_positive_rate(&self, bloom: &dyn BloomFilterStrategy, items: usize) -> f64 {
        let m = bloom.bit_count() as f64;
        let k = bloom.hash_count() as f64;
        let n = items as f64;

        (1.0_f64 - (-k * n / m).exp()).powf(k)
    }

    /// Serialization for bloom filters — refuses rather than emitting a filter
    /// that answers "absent" for every key.
    ///
    /// This used to write the parameter header followed by
    /// `vec![0u8; bit_count / 8]` — an all-zero bitmap, because
    /// `BloomFilterStrategy` exposes no accessor for its internal bits. A
    /// correct, fully-populated filter therefore round-tripped through
    /// `serialize_bloom` → `deserialize_bloom` into one that reports every key
    /// as *definitely absent*, which is the data-loss direction: callers prune
    /// records that exist. Refusing is strictly safer than returning bytes that
    /// decode to a wrong answer (TD-FACADE-1).
    ///
    /// To implement: either extend `BloomFilterStrategy` with bitmap access, or
    /// use `proximadb_bloom::SstableBloomFilter`, which already has real
    /// `serialize`/`deserialize`. Note its `xxhash64`/`cityhash64` are
    /// themselves placeholders (see TD-FACADE-1), so pick `murmur3_32`.
    fn serialize_bloom(&self, _bloom: &dyn BloomFilterStrategy) -> Result<Vec<u8>> {
        Err(anyhow::anyhow!(
            "RAPTOR bloom serialization is not implemented: BloomFilterStrategy exposes \
             no bitmap accessor, and emitting an all-zero bitmap would make the \
             deserialized filter report every key as absent. See TD-FACADE-1."
        ))
    }

    /// Custom deserialization for bloom filter
    fn deserialize_bloom(&self, bytes: &[u8]) -> Result<Box<dyn BloomFilterStrategy>> {
        if bytes.len() < 16 {
            return Err(anyhow::anyhow!("Invalid bloom filter data"));
        }

        // Read parameters
        let bits = u64::from_le_bytes(bytes[0..8].try_into()?);
        let _hash_functions = u64::from_le_bytes(bytes[8..16].try_into()?);

        // Create bloom with parameters
        // Note: This is simplified - actual implementation would restore bitmap
        let config = BloomFilterConfig {
            strategy: crate::core::bloom::BloomStrategy::ByteAligned,
            bits_per_key: (bits / self.column_stats.len() as u64 / 8) as u32,
            false_positive_rate: Some(0.01),
            expected_items: self.column_stats.values().map(|s| s.cardinality).sum(),
            enabled: true,
            hash_algorithm: crate::core::bloom::HashAlgorithm::Murmur3,
        };
        let bloom = BloomFilterFactory::create(&config);

        Ok(bloom)
    }
}

/// Bloom filter statistics
#[derive(Debug, Clone)]
pub struct BloomStats {
    pub column: String,
    pub size_bytes: usize,
    pub num_hash_functions: usize,
    pub items_added: usize,
    pub false_positive_rate: f64,
    pub access_frequency: u64,
}

/// Multi-column bloom filter for compound predicates
pub struct CompoundBloomFilter {
    #[allow(dead_code)]
    columns: Vec<String>,
    bloom: Box<dyn BloomFilterStrategy>,
    stats: ArtusColumnStats,
}

impl CompoundBloomFilter {
    pub fn new(columns: Vec<String>, cardinality: usize) -> Self {
        let config = BloomFilterConfig {
            strategy: crate::core::bloom::BloomStrategy::ByteAligned,
            bits_per_key: 10,
            false_positive_rate: Some(0.01),
            expected_items: cardinality,
            enabled: true,
            hash_algorithm: crate::core::bloom::HashAlgorithm::Murmur3,
        };
        let bloom = BloomFilterFactory::create(&config);

        let stats = ArtusColumnStats {
            column_name: columns.join("+"),
            cardinality,
            null_ratio: 0.0,
            access_frequency: 0,
            selectivity: 0.5,
            data_type: ColumnData::String,
            bloom_benefit_score: 0.0,
        };

        Self {
            columns,
            bloom,
            stats,
        }
    }

    /// Add compound value (concatenation of column values)
    pub fn add(&mut self, values: &[String]) {
        let compound = values.join("|");
        self.bloom.insert(compound.as_bytes());
        self.stats.access_frequency += 1;
    }

    /// Check compound predicate
    pub fn check(&self, values: &[String]) -> bool {
        let compound = values.join("|");
        self.bloom.might_contain(compound.as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_artus_bloom_creation() {
        let mut manager = ArtusBloomManager::new(ArtusBloomConfig::default());

        let stats = ArtusColumnStats {
            column_name: "user_id".to_string(),
            cardinality: 10000,
            null_ratio: 0.01,
            access_frequency: 20000, // Increased to ensure cardinality_ratio < 0.7
            selectivity: 0.3,
            data_type: ColumnData::String,
            bloom_benefit_score: 0.8,
        };

        manager.create_column_bloom(stats).unwrap();

        // Add values
        manager.add_to_bloom("user_id", "user123");
        manager.add_to_bloom("user_id", "user456");

        // Check values
        assert_eq!(manager.check_bloom("user_id", "user123"), Some(true));
        assert_eq!(manager.check_bloom("user_id", "user999"), Some(false));
    }

    #[test]
    fn test_compound_bloom() {
        let mut compound =
            CompoundBloomFilter::new(vec!["city".to_string(), "category".to_string()], 1000);

        compound.add(&["New York".to_string(), "Electronics".to_string()]);
        compound.add(&["Los Angeles".to_string(), "Books".to_string()]);

        assert!(compound.check(&["New York".to_string(), "Electronics".to_string()]));
        assert!(!compound.check(&["Chicago".to_string(), "Toys".to_string()]));
    }

    // ========== NEW TESTS ==========

    #[test]
    fn test_should_create_bloom_high_cardinality_rejected() {
        let manager = ArtusBloomManager::new(ArtusBloomConfig::default());
        let stats = ArtusColumnStats {
            column_name: "huge_col".to_string(),
            cardinality: 2_000_000, // Too high
            null_ratio: 0.0,
            access_frequency: 10_000_000,
            selectivity: 0.1,
            data_type: ColumnData::String,
            bloom_benefit_score: 0.5,
        };
        assert!(!manager.should_create_bloom(&stats));
    }

    #[test]
    fn test_should_create_bloom_low_cardinality_rejected() {
        let manager = ArtusBloomManager::new(ArtusBloomConfig::default());
        let stats = ArtusColumnStats {
            column_name: "tiny_col".to_string(),
            cardinality: 5, // Below min_cardinality (100)
            null_ratio: 0.0,
            access_frequency: 1000,
            selectivity: 0.1,
            data_type: ColumnData::Integer,
            bloom_benefit_score: 0.5,
        };
        assert!(!manager.should_create_bloom(&stats));
    }

    #[test]
    fn test_should_create_bloom_low_access_rejected() {
        let manager = ArtusBloomManager::new(ArtusBloomConfig::default());
        let stats = ArtusColumnStats {
            column_name: "rare_col".to_string(),
            cardinality: 500,
            null_ratio: 0.0,
            access_frequency: 5, // Below threshold of 10
            selectivity: 0.1,
            data_type: ColumnData::String,
            bloom_benefit_score: 0.5,
        };
        assert!(!manager.should_create_bloom(&stats));
    }

    #[test]
    fn test_should_create_bloom_high_selectivity_rejected() {
        let manager = ArtusBloomManager::new(ArtusBloomConfig::default());
        let stats = ArtusColumnStats {
            column_name: "broad_col".to_string(),
            cardinality: 500,
            null_ratio: 0.0,
            access_frequency: 100_000,
            selectivity: 0.8, // Too high (> 0.5 means not selective)
            data_type: ColumnData::String,
            bloom_benefit_score: 0.5,
        };
        assert!(!manager.should_create_bloom(&stats));
    }

    #[test]
    fn test_should_create_bloom_ideal_conditions() {
        let manager = ArtusBloomManager::new(ArtusBloomConfig::default());
        let stats = ArtusColumnStats {
            column_name: "perfect_col".to_string(),
            cardinality: 10_000,
            null_ratio: 0.01,
            access_frequency: 100_000, // cardinality_ratio = 0.1, well below 0.7
            selectivity: 0.1,
            data_type: ColumnData::String,
            bloom_benefit_score: 0.9,
        };
        assert!(manager.should_create_bloom(&stats));
    }

    #[test]
    fn test_optimal_size_calculation() {
        let manager = ArtusBloomManager::new(ArtusBloomConfig::default());
        let stats = ArtusColumnStats {
            column_name: "test_col".to_string(),
            cardinality: 10_000,
            null_ratio: 0.0,
            access_frequency: 50_000,
            selectivity: 0.1,
            data_type: ColumnData::String,
            bloom_benefit_score: 0.5,
        };
        let (bits, hash_fns) = manager.calculate_optimal_size(&stats);
        assert!(bits > 0, "Should have non-zero bits");
        assert!(hash_fns >= 1, "Should have at least 1 hash function");
        // With 10K items and 1% FPR, optimal bits ~ 95851
        assert!(bits > 50_000, "Bits should be reasonable for 10K items");
    }

    #[test]
    fn test_optimal_size_respects_memory_limit() {
        let config = ArtusBloomConfig {
            max_memory_per_filter: 1024, // 1KB = 8192 bits
            ..Default::default()
        };
        let manager = ArtusBloomManager::new(config);
        let stats = ArtusColumnStats {
            column_name: "big_col".to_string(),
            cardinality: 1_000_000,
            null_ratio: 0.0,
            access_frequency: 5_000_000,
            selectivity: 0.1,
            data_type: ColumnData::String,
            bloom_benefit_score: 0.5,
        };
        let (bits, _) = manager.calculate_optimal_size(&stats);
        assert!(
            bits <= 1024 * 8,
            "Bits {} should not exceed memory limit",
            bits
        );
    }

    #[test]
    fn test_false_positive_rate_estimation() {
        let mut manager = ArtusBloomManager::new(ArtusBloomConfig::default());
        let stats = ArtusColumnStats {
            column_name: "fp_test".to_string(),
            cardinality: 1000,
            null_ratio: 0.0,
            access_frequency: 10_000,
            selectivity: 0.1,
            data_type: ColumnData::String,
            bloom_benefit_score: 0.5,
        };
        manager.create_column_bloom(stats).unwrap();

        if let Some(bloom) = manager.column_blooms.get("fp_test") {
            let fp_rate = manager.estimate_false_positive_rate(bloom.as_ref(), 1000);
            // FP rate should be a valid probability
            assert!(
                (0.0..=1.0).contains(&fp_rate),
                "FP rate {} out of range",
                fp_rate
            );
        }
    }

    #[test]
    fn test_batch_add_to_bloom() {
        let mut manager = ArtusBloomManager::new(ArtusBloomConfig::default());
        let stats = ArtusColumnStats {
            column_name: "batch_col".to_string(),
            cardinality: 1000,
            null_ratio: 0.0,
            access_frequency: 10_000,
            selectivity: 0.1,
            data_type: ColumnData::String,
            bloom_benefit_score: 0.5,
        };
        manager.create_column_bloom(stats).unwrap();

        let values: Vec<String> = (0..100).map(|i| format!("item_{}", i)).collect();
        manager.batch_add_to_bloom("batch_col", &values);

        // Check that added values are found
        assert_eq!(manager.check_bloom("batch_col", "item_0"), Some(true));
        assert_eq!(manager.check_bloom("batch_col", "item_99"), Some(true));
    }

    #[test]
    fn test_check_bloom_nonexistent_column() {
        let manager = ArtusBloomManager::new(ArtusBloomConfig::default());
        assert_eq!(manager.check_bloom("nonexistent", "value"), None);
    }

    #[test]
    fn test_get_bloom_stats() {
        let mut manager = ArtusBloomManager::new(ArtusBloomConfig::default());
        let stats = ArtusColumnStats {
            column_name: "stats_col".to_string(),
            cardinality: 500,
            null_ratio: 0.0,
            access_frequency: 5000,
            selectivity: 0.2,
            data_type: ColumnData::Integer,
            bloom_benefit_score: 0.5,
        };
        manager.create_column_bloom(stats).unwrap();

        let bloom_stats = manager.get_bloom_stats("stats_col");
        assert!(bloom_stats.is_some());
        let bs = bloom_stats.unwrap();
        assert_eq!(bs.column, "stats_col");
        assert!(bs.size_bytes > 0);
        assert!(bs.num_hash_functions > 0);
        assert_eq!(bs.false_positive_rate, 0.01);
    }

    #[test]
    fn test_get_bloom_stats_nonexistent() {
        let manager = ArtusBloomManager::new(ArtusBloomConfig::default());
        assert!(manager.get_bloom_stats("nope").is_none());
    }

    #[test]
    fn test_serialize_deserialize_blooms() {
        let mut manager = ArtusBloomManager::new(ArtusBloomConfig::default());
        let stats = ArtusColumnStats {
            column_name: "ser_col".to_string(),
            cardinality: 1000,
            null_ratio: 0.0,
            access_frequency: 10_000,
            selectivity: 0.1,
            data_type: ColumnData::String,
            bloom_benefit_score: 0.5,
        };
        manager.create_column_bloom(stats).unwrap();
        manager.add_to_bloom("ser_col", "test_value");

        // TD-FACADE-1 pin. This test used to assert only byte *structure*
        // (non-empty, >= 16 bytes) and passed while the payload was an all-zero
        // bitmap — i.e. it was green over a filter that deserialised to
        // "absent for every key", never checking that "test_value" survived.
        // Serialization now refuses until a real bitmap can be written; a
        // future implementation must replace this with a membership round-trip
        // (add -> serialize -> deserialize -> might_contain("test_value")).
        let err = manager
            .serialize_blooms()
            .expect_err("serialization must refuse rather than emit an all-zero bitmap");
        assert!(
            err.to_string().contains("not implemented"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_column_data_types() {
        // Verify all column data type variants can be used in stats
        let types = vec![
            ColumnData::Integer,
            ColumnData::Float,
            ColumnData::String,
            ColumnData::Binary,
            ColumnData::Boolean,
            ColumnData::Timestamp,
            ColumnData::Vector,
        ];
        for dt in types {
            let stats = ArtusColumnStats {
                column_name: format!("{:?}_col", dt),
                cardinality: 500,
                null_ratio: 0.0,
                access_frequency: 5000,
                selectivity: 0.1,
                data_type: dt,
                bloom_benefit_score: 0.5,
            };
            let manager = ArtusBloomManager::new(ArtusBloomConfig::default());
            assert!(manager.should_create_bloom(&stats));
        }
    }

    #[test]
    fn test_artus_bloom_config_default() {
        let config = ArtusBloomConfig::default();
        assert_eq!(config.false_positive_rate, 0.01);
        assert_eq!(config.min_cardinality, 100);
        assert_eq!(config.max_memory_per_filter, 1024 * 1024);
        assert!(config.adaptive_sizing);
        assert!((config.auto_bloom_threshold - 0.7).abs() < 0.001);
    }
}
