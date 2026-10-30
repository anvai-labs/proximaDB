//! Metadata Filter Pushdown Optimization
//!
//! This module provides early metadata filtering at various stages of the search
//! pipeline to reduce unnecessary distance computations.
//!
//! Expected Performance Improvement: 20-30% reduction through early filtering

use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing::debug;

use crate::core::bloom::{BloomFilter, BloomFilterBuilder};
use crate::core::search::{ComparisonOperator, FilterExpression};
use proximadb_records::ProximaRecord;

/// Metadata filter optimizer with pushdown capabilities
pub struct MetadataFilterPushdown {
    /// Bloom filter for quick metadata existence checks
    bloom_filters: HashMap<String, Arc<BloomFilter>>,

    /// Column statistics for selective filtering
    column_stats: HashMap<String, MetadataColumnStatistics>,

    /// Index on frequently filtered columns
    column_indexes: HashMap<String, ColumnIndex>,

    /// Filter selectivity estimator
    #[allow(dead_code)]
    selectivity_estimator: SelectivityEstimator,

    /// Runtime policy for metadata pushdown decisions.
    config: MetadataFilterPushdownConfig,
}

/// Configuration for metadata filter pushdown heuristics.
#[derive(Debug, Clone, PartialEq)]
pub struct MetadataFilterPushdownConfig {
    /// Build per-column bloom filters only below this distinct-value count.
    pub bloom_distinct_value_limit: usize,
    /// Bits per key for generated metadata bloom filters.
    pub bloom_bits_per_key: u32,
    /// False-positive rate for generated metadata bloom filters.
    pub bloom_false_positive_rate: f64,
    /// Selectivity below this threshold uses indexed filtering first.
    pub indexed_selectivity_threshold: f64,
    /// Selectivity below this threshold uses bloom filtering before direct evaluation.
    pub bloom_selectivity_threshold: f64,
    /// Fallback selectivity for columns without statistics.
    pub unknown_column_selectivity: f64,
    /// Fallback selectivity for malformed `IN` predicates.
    pub invalid_in_selectivity: f64,
    /// Fallback selectivity for range-like predicates.
    pub range_selectivity: f64,
    /// Minimum distinct/total ratio required before building an index.
    pub min_index_selectivity: f64,
    /// Maximum distinct values allowed before skipping an index.
    pub max_index_distinct_values: usize,
}

impl Default for MetadataFilterPushdownConfig {
    fn default() -> Self {
        Self {
            bloom_distinct_value_limit: 1000,
            bloom_bits_per_key: 10,
            bloom_false_positive_rate: 0.01,
            indexed_selectivity_threshold: 0.01,
            bloom_selectivity_threshold: 0.1,
            unknown_column_selectivity: 0.5,
            invalid_in_selectivity: 0.5,
            range_selectivity: 0.3,
            min_index_selectivity: 0.01,
            max_index_distinct_values: 10_000,
        }
    }
}

impl MetadataFilterPushdownConfig {
    /// Validate selectivity and bloom policy values.
    pub fn validate(&self) -> std::result::Result<(), String> {
        for (name, value) in [
            ("bloom_false_positive_rate", self.bloom_false_positive_rate),
            (
                "indexed_selectivity_threshold",
                self.indexed_selectivity_threshold,
            ),
            (
                "bloom_selectivity_threshold",
                self.bloom_selectivity_threshold,
            ),
            (
                "unknown_column_selectivity",
                self.unknown_column_selectivity,
            ),
            ("invalid_in_selectivity", self.invalid_in_selectivity),
            ("range_selectivity", self.range_selectivity),
            ("min_index_selectivity", self.min_index_selectivity),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(format!(
                    "metadata pushdown config {name} must be finite and between 0.0 and 1.0, got {value}"
                ));
            }
        }
        if self.bloom_selectivity_threshold < self.indexed_selectivity_threshold {
            return Err(
                "bloom_selectivity_threshold must be >= indexed_selectivity_threshold".to_string(),
            );
        }
        if self.bloom_bits_per_key == 0 {
            return Err("bloom_bits_per_key must be greater than zero".to_string());
        }
        Ok(())
    }
}

/// Statistics for a metadata column
#[derive(Debug, Clone)]
pub struct MetadataColumnStatistics {
    /// Name of the metadata column
    pub column_name: String,
    /// Number of distinct values
    pub distinct_values: usize,
    /// Number of null values
    pub null_count: usize,
    /// Total row count
    pub total_count: usize,
    /// Minimum value in the column
    pub min_value: Option<Value>,
    /// Maximum value in the column
    pub max_value: Option<Value>,
    /// Distribution of values (for selectivity estimation)
    pub value_histogram: HashMap<Value, usize>,
    /// Optional bloom filter for membership testing
    pub bloom_filter: Option<Arc<BloomFilter>>,
}

/// Index structure for fast metadata lookups
pub struct ColumnIndex {
    /// Inverted index: value -> vector IDs
    inverted_index: HashMap<Value, HashSet<String>>,

    /// Range index for numeric columns
    #[allow(dead_code)]
    range_index: Option<RangeIndex>,

    /// Prefix tree for string columns
    #[allow(dead_code)]
    trie_index: Option<TrieIndex>,
}

/// Range index for numeric columns
struct RangeIndex {
    #[allow(dead_code)]
    sorted_entries: Vec<(f64, String)>, // (value, vector_id)
}

/// Trie index for string prefix matching
struct TrieIndex {
    #[allow(dead_code)]
    root: TrieNode,
}

struct TrieNode {
    #[allow(dead_code)]
    children: HashMap<char, Box<TrieNode>>,
    #[allow(dead_code)]
    vector_ids: HashSet<String>,
}

/// Filter selectivity estimator
struct SelectivityEstimator {
    /// Historical selectivity for different filter patterns
    #[allow(dead_code)]
    selectivity_history: HashMap<String, f64>,
}

impl MetadataFilterPushdown {
    /// Create a new metadata filter pushdown optimizer
    pub fn new() -> Self {
        Self::with_config(MetadataFilterPushdownConfig::default())
    }

    /// Create a metadata filter pushdown optimizer with explicit policy.
    pub fn with_config(config: MetadataFilterPushdownConfig) -> Self {
        debug_assert!(
            config.validate().is_ok(),
            "invalid metadata filter pushdown config: {:?}",
            config.validate().err()
        );
        Self {
            bloom_filters: HashMap::new(),
            column_stats: HashMap::new(),
            column_indexes: HashMap::new(),
            selectivity_estimator: SelectivityEstimator::new(),
            config,
        }
    }

    /// Build column statistics and indexes from a batch of records
    pub fn build_statistics(&mut self, records: &[ProximaRecord]) {
        let mut column_data: HashMap<String, Vec<Option<Value>>> = HashMap::new();

        // Collect all metadata values by column
        for record in records {
            let metadata = self.extract_metadata(record);

            for (key, value) in metadata {
                column_data
                    .entry(key.clone())
                    .or_default()
                    .push(Some(value));
            }
        }

        // Build statistics for each column
        for (column_name, values) in column_data {
            let stats = self.compute_column_stats(&column_name, &values);

            // Build bloom filter for the column
            if stats.distinct_values < self.config.bloom_distinct_value_limit {
                use crate::core::bloom::{BloomFilterConfig, BloomStrategy};
                let config = BloomFilterConfig {
                    strategy: BloomStrategy::BitPacked,
                    bits_per_key: self.config.bloom_bits_per_key,
                    false_positive_rate: Some(self.config.bloom_false_positive_rate),
                    expected_items: stats.distinct_values,
                    enabled: true,
                    hash_algorithm: crate::core::bloom::HashAlgorithm::default(),
                };
                let mut bloom_builder = BloomFilterBuilder::new(config);
                for value in values.iter().flatten() {
                    if let Ok(bytes) = serde_json::to_vec(value) {
                        bloom_builder.add(&bytes);
                    }
                }
                let bloom = bloom_builder.build();
                self.bloom_filters
                    .insert(column_name.clone(), Arc::new(bloom));
            }

            // Build index if selective enough
            if self.should_build_index(&stats) {
                let index = self.build_column_index(&column_name, &values, records);
                self.column_indexes.insert(column_name.clone(), index);
            }

            self.column_stats.insert(column_name, stats);
        }
    }

    /// Apply filter pushdown at the WAL level
    pub fn apply_wal_filter(
        &self,
        records: Vec<ProximaRecord>,
        filter: &FilterExpression,
    ) -> Vec<ProximaRecord> {
        // Estimate filter selectivity
        let selectivity = self.estimate_selectivity(filter);

        debug!(
            "Applying WAL filter pushdown with estimated selectivity: {:.2}%",
            selectivity * 100.0
        );

        // Use different strategies based on selectivity
        if selectivity < self.config.indexed_selectivity_threshold {
            // Very selective - use index if available
            self.apply_indexed_filter(records, filter)
        } else if selectivity < self.config.bloom_selectivity_threshold {
            // Moderately selective - use bloom filter first
            self.apply_bloom_then_filter(records, filter)
        } else {
            // Not very selective - apply directly
            self.apply_direct_filter(records, filter)
        }
    }

    /// Apply indexed filtering for very selective filters
    fn apply_indexed_filter(
        &self,
        records: Vec<ProximaRecord>,
        filter: &FilterExpression,
    ) -> Vec<ProximaRecord> {
        // Extract columns used in filter
        let filter_columns = self.extract_filter_columns(filter);

        // Check if we have indexes for these columns
        let indexed_columns: Vec<_> = filter_columns
            .iter()
            .filter(|col| self.column_indexes.contains_key(*col))
            .collect();

        if indexed_columns.is_empty() {
            return self.apply_direct_filter(records, filter);
        }

        // Use index to get candidate IDs
        let candidate_ids = self.get_indexed_candidates(filter, &indexed_columns);

        // Filter records by candidate IDs
        records
            .into_iter()
            .filter(|record| candidate_ids.contains(&record.oid))
            .collect()
    }

    /// Apply bloom filter before full filtering
    fn apply_bloom_then_filter(
        &self,
        records: Vec<ProximaRecord>,
        filter: &FilterExpression,
    ) -> Vec<ProximaRecord> {
        // First pass: bloom filter
        let bloom_candidates = self.bloom_filter_pass(records, filter);

        // Second pass: actual filter evaluation
        self.apply_direct_filter(bloom_candidates, filter)
    }

    /// Bloom filter pass for quick elimination
    fn bloom_filter_pass(
        &self,
        records: Vec<ProximaRecord>,
        filter: &FilterExpression,
    ) -> Vec<ProximaRecord> {
        records
            .into_iter()
            .filter(|_record| {
                // Check if record might match based on bloom filters
                self.check_bloom_filters(filter)
            })
            .collect()
    }

    /// Check bloom filters for a filter expression
    fn check_bloom_filters(&self, filter: &FilterExpression) -> bool {
        match filter {
            FilterExpression::Comparison {
                field,
                operator,
                value,
            } => {
                if let Some(bloom) = self.bloom_filters.get(field) {
                    match operator {
                        ComparisonOperator::Equals => {
                            if let Ok(value_bytes) = serde_json::to_vec(value) {
                                bloom.as_ref().might_contain(&value_bytes)
                            } else {
                                false
                            }
                        }
                        _ => true, // Can't use bloom filter for other operators
                    }
                } else {
                    true // No bloom filter, can't eliminate
                }
            }
            FilterExpression::And(exprs) => exprs.iter().all(|expr| self.check_bloom_filters(expr)),
            FilterExpression::Or(exprs) => exprs.iter().any(|expr| self.check_bloom_filters(expr)),
            FilterExpression::Not(expr) => !self.check_bloom_filters(expr),
        }
    }

    /// Apply direct filter evaluation
    fn apply_direct_filter(
        &self,
        records: Vec<ProximaRecord>,
        filter: &FilterExpression,
    ) -> Vec<ProximaRecord> {
        use crate::core::search::json_comparison::evaluate_filter;

        records
            .into_iter()
            .filter(|record| {
                let metadata = self.extract_metadata_hashmap(record);
                evaluate_filter(filter, &metadata)
            })
            .collect()
    }

    /// Estimate filter selectivity
    fn estimate_selectivity(&self, filter: &FilterExpression) -> f64 {
        match filter {
            FilterExpression::Comparison {
                field,
                operator,
                value,
            } => {
                if let Some(stats) = self.column_stats.get(field) {
                    self.estimate_comparison_selectivity(stats, operator, value)
                } else {
                    self.config.unknown_column_selectivity
                }
            }
            FilterExpression::And(exprs) => {
                // Multiply selectivities (assuming independence)
                exprs
                    .iter()
                    .map(|expr| self.estimate_selectivity(expr))
                    .product()
            }
            FilterExpression::Or(exprs) => {
                // Combine selectivities
                let selectivities: Vec<f64> = exprs
                    .iter()
                    .map(|expr| self.estimate_selectivity(expr))
                    .collect();

                // P(A or B) = P(A) + P(B) - P(A and B)
                1.0 - selectivities.iter().map(|s| 1.0 - s).product::<f64>()
            }
            FilterExpression::Not(expr) => 1.0 - self.estimate_selectivity(expr),
        }
    }

    /// Estimate selectivity for a comparison
    fn estimate_comparison_selectivity(
        &self,
        stats: &MetadataColumnStatistics,
        operator: &ComparisonOperator,
        value: &Value,
    ) -> f64 {
        match operator {
            ComparisonOperator::Equals => {
                if let Some(count) = stats.value_histogram.get(value) {
                    *count as f64 / stats.total_count as f64
                } else {
                    1.0 / stats.distinct_values.max(1) as f64
                }
            }
            ComparisonOperator::NotEquals => {
                1.0 - self.estimate_comparison_selectivity(
                    stats,
                    &ComparisonOperator::Equals,
                    value,
                )
            }
            ComparisonOperator::IsNull => stats.null_count as f64 / stats.total_count as f64,
            ComparisonOperator::IsNotNull => {
                1.0 - (stats.null_count as f64 / stats.total_count as f64)
            }
            ComparisonOperator::In => {
                if let Value::Array(values) = value {
                    values
                        .iter()
                        .map(|v| {
                            self.estimate_comparison_selectivity(
                                stats,
                                &ComparisonOperator::Equals,
                                v,
                            )
                        })
                        .sum::<f64>()
                        .min(1.0)
                } else {
                    self.config.invalid_in_selectivity
                }
            }
            _ => self.config.range_selectivity,
        }
    }

    /// Extract metadata from a record's props as JSON map
    fn extract_metadata(&self, record: &ProximaRecord) -> HashMap<String, Value> {
        crate::core::search::sql_value_filter::proxima_tree_to_json_map(&record.props)
    }

    /// Extract metadata as HashMap for filter evaluation
    fn extract_metadata_hashmap(&self, record: &ProximaRecord) -> HashMap<String, Value> {
        self.extract_metadata(record)
    }

    /// Compute statistics for a column
    fn compute_column_stats(
        &self,
        column_name: &str,
        values: &[Option<Value>],
    ) -> MetadataColumnStatistics {
        let mut distinct_values = HashSet::new();
        let mut null_count = 0;
        let mut value_histogram = HashMap::new();
        let mut min_value: Option<Value> = None;
        let mut max_value: Option<Value> = None;

        for value_opt in values {
            match value_opt {
                Some(value) => {
                    distinct_values.insert(value.clone());
                    *value_histogram.entry(value.clone()).or_insert(0) += 1;

                    // Update min/max for comparable values
                    let update_min = match min_value.as_ref() {
                        Some(current_min) => self.compare_values(value, current_min) < 0,
                        None => true,
                    };
                    if update_min {
                        min_value = Some(value.clone());
                    }
                    let update_max = match max_value.as_ref() {
                        Some(current_max) => self.compare_values(value, current_max) > 0,
                        None => true,
                    };
                    if update_max {
                        max_value = Some(value.clone());
                    }
                }
                None => null_count += 1,
            }
        }

        MetadataColumnStatistics {
            column_name: column_name.to_string(),
            distinct_values: distinct_values.len(),
            null_count,
            total_count: values.len(),
            min_value,
            max_value,
            value_histogram,
            bloom_filter: None,
        }
    }

    /// Compare two JSON values
    fn compare_values(&self, a: &Value, b: &Value) -> i32 {
        use crate::core::search::json_comparison::compare_json_values;
        use std::cmp::Ordering;

        match compare_json_values(a, b) {
            Ordering::Less => -1,
            Ordering::Equal => 0,
            Ordering::Greater => 1,
        }
    }

    /// Check if we should build an index for a column
    fn should_build_index(&self, stats: &MetadataColumnStatistics) -> bool {
        // Build index if column is selective and frequently used
        let selectivity = stats.distinct_values as f64 / stats.total_count as f64;
        selectivity > self.config.min_index_selectivity
            && stats.distinct_values < self.config.max_index_distinct_values
    }

    /// Build column index
    fn build_column_index(
        &self,
        _column_name: &str,
        values: &[Option<Value>],
        records: &[ProximaRecord],
    ) -> ColumnIndex {
        let mut inverted_index = HashMap::new();

        for (i, value_opt) in values.iter().enumerate() {
            if let Some(value) = value_opt
                && let Some(record) = records.get(i)
                && !record.oid.is_empty()
            {
                inverted_index
                    .entry(value.clone())
                    .or_insert_with(HashSet::new)
                    .insert(record.oid.clone());
            }
        }

        ColumnIndex {
            inverted_index,
            range_index: None,
            trie_index: None,
        }
    }

    /// Extract columns used in a filter
    fn extract_filter_columns(&self, filter: &FilterExpression) -> HashSet<String> {
        use crate::core::search::filter_extraction::extract_filter_columns;
        extract_filter_columns(filter)
    }

    /// Get candidate IDs using indexes
    fn get_indexed_candidates(
        &self,
        filter: &FilterExpression,
        indexed_columns: &[&String],
    ) -> HashSet<String> {
        match filter {
            FilterExpression::Comparison {
                field,
                operator,
                value,
            } => {
                if indexed_columns.contains(&field) {
                    if let Some(index) = self.column_indexes.get(field) {
                        self.get_candidates_from_index(index, operator, value)
                    } else {
                        HashSet::new()
                    }
                } else {
                    HashSet::new()
                }
            }
            FilterExpression::And(exprs) => {
                // Intersection of candidates
                exprs
                    .iter()
                    .map(|expr| self.get_indexed_candidates(expr, indexed_columns))
                    .reduce(|a, b| a.intersection(&b).cloned().collect())
                    .unwrap_or_default()
            }
            FilterExpression::Or(exprs) => {
                // Union of candidates
                exprs
                    .iter()
                    .flat_map(|expr| self.get_indexed_candidates(expr, indexed_columns))
                    .collect()
            }
            FilterExpression::Not(_) => {
                // Can't use index for NOT
                HashSet::new()
            }
        }
    }

    /// Get candidates from a single index
    fn get_candidates_from_index(
        &self,
        index: &ColumnIndex,
        operator: &ComparisonOperator,
        value: &Value,
    ) -> HashSet<String> {
        match operator {
            ComparisonOperator::Equals => {
                index.inverted_index.get(value).cloned().unwrap_or_default()
            }
            ComparisonOperator::In => {
                if let Value::Array(values) = value {
                    values
                        .iter()
                        .flat_map(|v| index.inverted_index.get(v).cloned().unwrap_or_default())
                        .collect()
                } else {
                    HashSet::new()
                }
            }
            _ => HashSet::new(), // Other operators need range index
        }
    }
}

impl Default for MetadataFilterPushdown {
    fn default() -> Self {
        Self::new()
    }
}

impl SelectivityEstimator {
    fn new() -> Self {
        Self {
            selectivity_history: HashMap::new(),
        }
    }
}

/// Builder for creating per-column bloom filters on metadata values
pub struct MetadataBloomBuilder {
    builders: HashMap<String, BloomFilterBuilder>,
    expected_items: usize,
    config: MetadataBloomBuilderConfig,
}

/// Configuration for standalone metadata bloom construction.
#[derive(Debug, Clone, PartialEq)]
pub struct MetadataBloomBuilderConfig {
    pub false_positive_rate: f64,
    pub bits_per_key: u32,
}

impl Default for MetadataBloomBuilderConfig {
    fn default() -> Self {
        Self {
            false_positive_rate: 0.01,
            bits_per_key: 10,
        }
    }
}

impl MetadataBloomBuilderConfig {
    pub fn validate(&self) -> std::result::Result<(), String> {
        if !self.false_positive_rate.is_finite() || !(0.0..=1.0).contains(&self.false_positive_rate)
        {
            return Err(format!(
                "metadata bloom false_positive_rate must be finite and between 0.0 and 1.0, got {}",
                self.false_positive_rate
            ));
        }
        if self.bits_per_key == 0 {
            return Err("metadata bloom bits_per_key must be greater than zero".to_string());
        }
        Ok(())
    }
}

impl MetadataBloomBuilder {
    /// Create a new metadata bloom filter builder for the expected number of items
    pub fn new(expected_items: usize) -> Self {
        Self::with_config(expected_items, MetadataBloomBuilderConfig::default())
    }

    /// Create a metadata bloom filter builder with explicit policy.
    pub fn with_config(expected_items: usize, config: MetadataBloomBuilderConfig) -> Self {
        debug_assert!(
            config.validate().is_ok(),
            "invalid metadata bloom builder config: {:?}",
            config.validate().err()
        );
        Self {
            builders: HashMap::new(),
            expected_items,
            config,
        }
    }

    /// Add a record's metadata to bloom filters
    pub fn add_record(&mut self, record: &ProximaRecord) {
        use crate::core::bloom::{BloomFilterConfig, BloomStrategy};
        use crate::core::search::sql_value_filter::proxima_tree_to_json_map;

        for (key, value) in proxima_tree_to_json_map(&record.props) {
            let config = BloomFilterConfig {
                strategy: BloomStrategy::BitPacked,
                bits_per_key: self.config.bits_per_key,
                false_positive_rate: Some(self.config.false_positive_rate),
                expected_items: self.expected_items,
                enabled: true,
                hash_algorithm: crate::core::bloom::HashAlgorithm::default(),
            };
            let builder = self
                .builders
                .entry(key.clone())
                .or_insert_with(|| BloomFilterBuilder::new(config));

            let serialized = serde_json::to_vec(&value).unwrap_or_default();
            builder.add(&serialized);
        }
    }

    /// Build all bloom filters
    pub fn build(self) -> HashMap<String, BloomFilter> {
        self.builders
            .into_iter()
            .map(|(key, builder)| (key, builder.build()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn column_stats(distinct_values: usize, total_count: usize) -> MetadataColumnStatistics {
        MetadataColumnStatistics {
            column_name: "category".to_string(),
            distinct_values,
            null_count: 0,
            total_count,
            min_value: None,
            max_value: None,
            value_histogram: HashMap::new(),
            bloom_filter: None,
        }
    }

    #[test]
    fn test_metadata_pushdown_config_validation() {
        let mut config = MetadataFilterPushdownConfig::default();
        assert!(config.validate().is_ok());

        config.bloom_selectivity_threshold = 0.001;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_metadata_pushdown_uses_configured_selectivity_policy() {
        let mut pushdown = MetadataFilterPushdown::with_config(MetadataFilterPushdownConfig {
            unknown_column_selectivity: 0.72,
            invalid_in_selectivity: 0.61,
            range_selectivity: 0.41,
            ..Default::default()
        });
        pushdown
            .column_stats
            .insert("known".to_string(), column_stats(10, 100));

        let unknown = FilterExpression::Comparison {
            field: "missing".to_string(),
            operator: ComparisonOperator::Equals,
            value: Value::String("electronics".to_string()),
        };
        assert_eq!(pushdown.estimate_selectivity(&unknown), 0.72);

        let malformed_in = FilterExpression::Comparison {
            field: "known".to_string(),
            operator: ComparisonOperator::In,
            value: Value::String("not-an-array".to_string()),
        };
        assert_eq!(pushdown.estimate_selectivity(&malformed_in), 0.61);

        let range = FilterExpression::Comparison {
            field: "known".to_string(),
            operator: ComparisonOperator::GreaterThan,
            value: Value::Number(serde_json::Number::from(100)),
        };
        assert_eq!(pushdown.estimate_selectivity(&range), 0.41);
    }

    #[test]
    fn test_metadata_pushdown_uses_configured_index_policy() {
        let pushdown = MetadataFilterPushdown::with_config(MetadataFilterPushdownConfig {
            min_index_selectivity: 0.2,
            max_index_distinct_values: 50,
            ..Default::default()
        });

        assert!(pushdown.should_build_index(&column_stats(25, 100)));
        assert!(!pushdown.should_build_index(&column_stats(10, 100)));
        assert!(!pushdown.should_build_index(&column_stats(60, 100)));
    }

    #[test]
    fn test_metadata_bloom_builder_config_validation() {
        let mut config = MetadataBloomBuilderConfig::default();
        assert!(config.validate().is_ok());

        config.bits_per_key = 0;
        assert!(config.validate().is_err());

        let builder = MetadataBloomBuilder::with_config(
            128,
            MetadataBloomBuilderConfig {
                false_positive_rate: 0.05,
                bits_per_key: 12,
            },
        );
        assert_eq!(builder.config.false_positive_rate, 0.05);
        assert_eq!(builder.config.bits_per_key, 12);
    }

    #[test]
    fn test_selectivity_estimation() {
        let pushdown = MetadataFilterPushdown::new();

        let filter = FilterExpression::And(vec![
            FilterExpression::Comparison {
                field: "category".to_string(),
                operator: ComparisonOperator::Equals,
                value: Value::String("electronics".to_string()),
            },
            FilterExpression::Comparison {
                field: "price".to_string(),
                operator: ComparisonOperator::GreaterThan,
                value: Value::Number(serde_json::Number::from(100)),
            },
        ]);

        let selectivity = pushdown.estimate_selectivity(&filter);
        assert!((0.0..=1.0).contains(&selectivity));
    }

    #[test]
    fn test_bloom_filter_building() {
        use proximadb_data_model::ProximaValue;
        use proximadb_records::{
            EmbeddingCell, LabelSet, ProximaRecord, ProximaTree, ProximaTreeNode,
        };

        let mut builder = MetadataBloomBuilder::new(1000);

        let mut props = ProximaTree::new();
        props.insert(
            "category".to_string(),
            ProximaTreeNode::Value(ProximaValue::String("electronics".to_string())),
        );
        props.insert(
            "price".to_string(),
            ProximaTreeNode::Value(ProximaValue::Float64(99.99)),
        );

        let now_ns = 0i64;
        let record = ProximaRecord {
            oid: "test1".to_string(),
            local_id: None,
            tid: None,
            variation_id: None,
            record_version: 1,
            spec_version: 1,
            tenant_id: String::new(),
            permitted_principals: Vec::new(),
            rls_policy_id: None,
            created_at_ns: now_ns,
            updated_at_ns: now_ns,
            valid_from_ns: None,
            valid_to_ns: None,
            origin: None,
            actor: None,
            method: None,
            memory_type: None,
            props,
            refs: Vec::new(),
            edge: None,
            embeddings: vec![EmbeddingCell::new_fp32(
                "default",
                "dense_vector",
                3,
                vec![1.0, 2.0, 3.0],
            )],
            sequence: None,
            labels: LabelSet::new(),
            ..Default::default()
        };

        builder.add_record(&record);
        let bloom_filters = builder.build();

        assert_eq!(bloom_filters.len(), 2);
        assert!(bloom_filters.contains_key("category"));
        assert!(bloom_filters.contains_key("price"));
    }
}
