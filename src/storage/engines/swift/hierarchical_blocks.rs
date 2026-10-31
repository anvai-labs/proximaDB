// Hierarchical block structure and metadata indexing for SST
// Clean implementation with no backward compatibility

use anyhow::Result;
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::storage::engines::core::formats::proximablocks::{ProximaDataBlock, SuperBlock};

/// Metadata index for efficient filtering
#[derive(Debug)]
pub struct MetadataIndex {
    /// Column-specific indexes
    column_indexes: HashMap<String, ColumnIndex>,

    /// Composite indexes for common query patterns
    #[allow(dead_code)]
    composite_indexes: Vec<CompositeIndex>,

    /// Table-level statistics
    table_stats: SwiftHierarchicalTableStatistics,

    /// Filterable columns configuration
    filterable_columns: HashSet<String>,
}

/// Index for a single column
#[derive(Debug)]
pub enum ColumnIndex {
    /// For categorical columns with low cardinality
    Inverted {
        value_to_blocks: HashMap<serde_json::Value, BitSet>,
        cardinality: u32,
    },

    /// For numeric columns
    BTree {
        tree: BTreeMap<OrderedValue, BitSet>,
        min: f64,
        max: f64,
        histogram: Histogram,
    },

    /// For text columns with full-text search
    FullText {
        token_to_blocks: HashMap<String, BitSet>,
        total_tokens: u64,
    },
}

/// Composite index for multiple columns
#[derive(Debug)]
pub struct CompositeIndex {
    pub columns: Vec<String>,
    pub index_type: CompositeIndexType,
    pub data: BTreeMap<Vec<serde_json::Value>, BitSet>,
}

#[derive(Debug)]
pub enum CompositeIndexType {
    Compound, // All columns must match
    Covering, // Index contains all needed data
    Partial,  // Index with WHERE clause
}

/// Wrapper for f64 that implements Ord by treating NaN as greater than all other values
pub use proximadb_records::TotalOrderedF64 as OrderedFloat;

/// Ordered wrapper for JSON values that preserves type-aware ordering
#[derive(Debug, Clone, PartialEq)]
pub enum OrderedValue {
    Null,
    Bool(bool),
    Number(OrderedFloat),
    String(String),
}

impl Eq for OrderedValue {}

impl PartialOrd for OrderedValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedValue {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use OrderedValue::*;
        match (self, other) {
            (Null, Null) => std::cmp::Ordering::Equal,
            (Null, _) => std::cmp::Ordering::Less,
            (_, Null) => std::cmp::Ordering::Greater,

            (Bool(a), Bool(b)) => a.cmp(b),
            (Bool(_), _) => std::cmp::Ordering::Less,
            (_, Bool(_)) => std::cmp::Ordering::Greater,

            (Number(a), Number(b)) => a.cmp(b),
            (Number(_), String(_)) => std::cmp::Ordering::Less,
            (String(_), Number(_)) => std::cmp::Ordering::Greater,

            (String(a), String(b)) => a.cmp(b),
        }
    }
}

impl From<serde_json::Value> for OrderedValue {
    fn from(val: serde_json::Value) -> Self {
        match val {
            serde_json::Value::Null => OrderedValue::Null,
            serde_json::Value::Bool(b) => OrderedValue::Bool(b),
            serde_json::Value::Number(n) => {
                OrderedValue::Number(OrderedFloat(n.as_f64().unwrap_or(0.0)))
            }
            serde_json::Value::String(s) => OrderedValue::String(s),
            serde_json::Value::Array(_) | serde_json::Value::Object(_) => {
                // For complex types, fall back to string representation
                OrderedValue::String(val.to_string())
            }
        }
    }
}

/// Bit set for tracking which blocks contain matching records
#[derive(Debug, Clone)]
pub struct BitSet {
    bits: Vec<u64>,
    size: usize,
}

impl BitSet {
    pub fn new(size: usize) -> Self {
        let n_words = size.div_ceil(64);
        Self {
            bits: vec![0; n_words],
            size,
        }
    }

    pub fn set(&mut self, idx: usize) {
        if idx < self.size {
            let word = idx / 64;
            let bit = idx % 64;
            self.bits[word] |= 1u64 << bit;
        }
    }

    pub fn test(&self, idx: usize) -> bool {
        if idx >= self.size {
            return false;
        }
        let word = idx / 64;
        let bit = idx % 64;
        (self.bits[word] & (1u64 << bit)) != 0
    }

    pub fn count(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn intersect(&self, other: &BitSet) -> BitSet {
        let mut result = BitSet::new(self.size.min(other.size));
        for i in 0..result.bits.len().min(self.bits.len()).min(other.bits.len()) {
            result.bits[i] = self.bits[i] & other.bits[i];
        }
        result
    }

    pub fn union(&self, other: &BitSet) -> BitSet {
        let mut result = BitSet::new(self.size.max(other.size));
        for i in 0..self.bits.len() {
            result.bits[i] = self.bits[i];
        }
        for i in 0..other.bits.len() {
            if i < result.bits.len() {
                result.bits[i] |= other.bits[i];
            }
        }
        result
    }
}

/// Histogram for numeric column statistics
#[derive(Debug, Clone)]
pub struct Histogram {
    pub buckets: Vec<HistogramBucket>,
    pub total_count: u64,
}

#[derive(Debug, Clone)]
pub struct HistogramBucket {
    pub min: f64,
    pub max: f64,
    pub count: u64,
}

/// Backwards-compat alias for [`SwiftHierarchicalTableStatistics`].
pub type TableStatistics = SwiftHierarchicalTableStatistics;

/// Table-level statistics
#[derive(Debug, Clone)]
pub struct SwiftHierarchicalTableStatistics {
    pub total_records: u64,
    pub total_blocks: u64,
    pub total_superblocks: u64,
    pub column_stats: HashMap<String, GlobalColumnStats>,
}

#[derive(Debug, Clone)]
pub struct GlobalColumnStats {
    pub null_ratio: f64,
    pub cardinality: u64,
    pub avg_size_bytes: u64,
}

#[derive(Debug, Clone)]
pub enum Data {
    Integer,
    Float,
    String,
    Boolean,
    Array,
    Object,
}

impl MetadataIndex {
    pub fn new() -> Self {
        Self {
            column_indexes: HashMap::new(),
            composite_indexes: Vec::new(),
            table_stats: SwiftHierarchicalTableStatistics {
                total_records: 0,
                total_blocks: 0,
                total_superblocks: 0,
                column_stats: HashMap::new(),
            },
            filterable_columns: HashSet::new(),
        }
    }

    /// Build index from superblocks
    pub fn build_from_superblocks(&mut self, superblocks: &[SuperBlock]) -> Result<()> {
        // Update table statistics
        self.table_stats.total_superblocks = superblocks.len() as u64;

        for (sb_idx, superblock) in superblocks.iter().enumerate() {
            for (b_idx, block) in superblock.blocks.iter().enumerate() {
                self.table_stats.total_blocks += 1;
                self.table_stats.total_records += block.records.len() as u64;

                // Index each block's metadata
                self.index_block(sb_idx, b_idx, block)?;
            }
        }

        // Build composite indexes for common patterns
        self.build_composite_indexes()?;

        Ok(())
    }

    /// Index a single block's metadata
    fn index_block(&mut self, sb_idx: usize, b_idx: usize, block: &ProximaDataBlock) -> Result<()> {
        let block_id = sb_idx * 64 + b_idx;

        // Process each record's metadata
        for record in &block.records {
            if record.props.is_empty() {
                continue;
            }

            // Flatten the canonical property tree to BARE JSON scalars. The
            // `ProximaTreeNode` enum is externally tagged, so serializing a node
            // directly (serde_json::to_value) yields `{"Value":{"String":...}}`
            // — which never matches the bare query values used by
            // `find_blocks_with_value` / `find_blocks_in_range`. The canonical
            // `proxima_tree_to_json_map` unwraps each node to its scalar value.
            let json_props =
                crate::core::search::sql_value_filter::proxima_tree_to_json_map(&record.props);
            for (key, json_value) in &json_props {
                // Only index filterable columns
                if !self.filterable_columns.contains(key) {
                    continue;
                }

                // Get or create column index
                let new_index = if !self.column_indexes.contains_key(key) {
                    Some(self.create_column_index(json_value))
                } else {
                    None
                };

                if let Some(idx) = new_index {
                    self.column_indexes.insert(key.clone(), idx);
                }

                // Update index with this value
                if let Some(column_index) = self.column_indexes.get_mut(key) {
                    Self::update_column_index_static(column_index, json_value, block_id)?;
                }
            }
        }

        Ok(())
    }

    /// Create appropriate index type based on value type
    fn create_column_index(&self, sample_value: &serde_json::Value) -> ColumnIndex {
        match sample_value {
            serde_json::Value::Number(_) => ColumnIndex::BTree {
                tree: BTreeMap::new(),
                min: f64::MAX,
                max: f64::MIN,
                histogram: Histogram {
                    buckets: Vec::new(),
                    total_count: 0,
                },
            },
            serde_json::Value::String(_) => {
                // Use inverted index for strings by default
                ColumnIndex::Inverted {
                    value_to_blocks: HashMap::new(),
                    cardinality: 0,
                }
            }
            _ => {
                // Default to inverted index
                ColumnIndex::Inverted {
                    value_to_blocks: HashMap::new(),
                    cardinality: 0,
                }
            }
        }
    }

    /// Update column index with a new value
    fn update_column_index_static(
        index: &mut ColumnIndex,
        value: &serde_json::Value,
        block_id: usize,
    ) -> Result<()> {
        match index {
            ColumnIndex::Inverted {
                value_to_blocks,
                cardinality,
            } => {
                let bitset = value_to_blocks.entry(value.clone()).or_insert_with(|| {
                    *cardinality += 1;
                    BitSet::new(10000) // Assume max 10k blocks
                });
                bitset.set(block_id);
            }
            ColumnIndex::BTree {
                tree,
                min,
                max,
                histogram,
            } => {
                if let Some(num) = value.as_f64() {
                    *min = min.min(num);
                    *max = max.max(num);

                    let ordered = OrderedValue::from(value.clone());
                    let bitset = tree.entry(ordered).or_insert_with(|| BitSet::new(10000));
                    bitset.set(block_id);

                    histogram.total_count += 1;
                }
            }
            ColumnIndex::FullText {
                token_to_blocks,
                total_tokens,
            } => {
                if let Some(text) = value.as_str() {
                    // Simple tokenization (in production, use proper tokenizer)
                    for token in text.split_whitespace() {
                        let bitset = token_to_blocks
                            .entry(token.to_lowercase())
                            .or_insert_with(|| BitSet::new(10000));
                        bitset.set(block_id);
                        *total_tokens += 1;
                    }
                }
            }
        }
        Ok(())
    }

    /// Build composite indexes for common query patterns
    fn build_composite_indexes(&mut self) -> Result<()> {
        // Example: Build composite index for (category, price) if both exist
        if self.column_indexes.contains_key("category") && self.column_indexes.contains_key("price")
        {
            // Implementation would go here
        }
        Ok(())
    }

    /// Find blocks matching a metadata filter
    pub fn find_matching_blocks(&self, filter: &super::MetadataFilter) -> Result<BitSet> {
        let mut result = BitSet::new(self.table_stats.total_blocks as usize);

        // Start with all blocks
        for i in 0..self.table_stats.total_blocks as usize {
            result.set(i);
        }

        // Apply each condition
        for condition in &filter.conditions {
            let matching = self.evaluate_condition(condition)?;
            result = result.intersect(&matching);
        }

        Ok(result)
    }

    /// Evaluate a single filter condition
    fn evaluate_condition(&self, condition: &super::FilterCondition) -> Result<BitSet> {
        use super::FilterCondition;

        match condition {
            FilterCondition::Equals(column, value) => self.find_blocks_with_value(column, value),
            FilterCondition::Range(column, min, max) => self.find_blocks_in_range(column, min, max),
            FilterCondition::In(column, values) => {
                let mut result = BitSet::new(self.table_stats.total_blocks as usize);
                for value in values {
                    let matches = self.find_blocks_with_value(column, value)?;
                    result = result.union(&matches);
                }
                Ok(result)
            }
            FilterCondition::IsNull(column) => {
                // Find blocks where column is null
                self.find_blocks_without_column(column)
            }
            FilterCondition::IsNotNull(column) => {
                // Find blocks where column is not null
                let without = self.find_blocks_without_column(column)?;
                let mut all = BitSet::new(self.table_stats.total_blocks as usize);
                for i in 0..self.table_stats.total_blocks as usize {
                    if !without.test(i) {
                        all.set(i);
                    }
                }
                Ok(all)
            }
        }
    }

    fn find_blocks_with_value(&self, column: &str, value: &serde_json::Value) -> Result<BitSet> {
        if let Some(index) = self.column_indexes.get(column) {
            match index {
                ColumnIndex::Inverted {
                    value_to_blocks, ..
                } => Ok(value_to_blocks
                    .get(value)
                    .cloned()
                    .unwrap_or_else(|| BitSet::new(self.table_stats.total_blocks as usize))),
                ColumnIndex::BTree { tree, .. } => {
                    let ordered = OrderedValue::from(value.clone());
                    Ok(tree
                        .get(&ordered)
                        .cloned()
                        .unwrap_or_else(|| BitSet::new(self.table_stats.total_blocks as usize)))
                }
                _ => Ok(BitSet::new(self.table_stats.total_blocks as usize)),
            }
        } else {
            Ok(BitSet::new(self.table_stats.total_blocks as usize))
        }
    }

    fn find_blocks_in_range(
        &self,
        column: &str,
        min: &serde_json::Value,
        max: &serde_json::Value,
    ) -> Result<BitSet> {
        if let Some(ColumnIndex::BTree { tree, .. }) = self.column_indexes.get(column) {
            let min_ordered = OrderedValue::from(min.clone());
            let max_ordered = OrderedValue::from(max.clone());

            let mut result = BitSet::new(self.table_stats.total_blocks as usize);

            // With proper type-aware ordering, we can directly use the range
            for (_value, bitset) in tree.range(min_ordered..=max_ordered) {
                result = result.union(bitset);
            }
            Ok(result)
        } else {
            Ok(BitSet::new(self.table_stats.total_blocks as usize))
        }
    }

    fn find_blocks_without_column(&self, _column: &str) -> Result<BitSet> {
        // This would track which blocks don't have the column
        // For simplicity, returning empty set
        Ok(BitSet::new(self.table_stats.total_blocks as usize))
    }
}

impl Default for MetadataIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::proximadb_v1::VectorRecord;

    #[test]
    fn test_bitset_operations() {
        let mut bs1 = BitSet::new(100);
        let mut bs2 = BitSet::new(100);

        bs1.set(10);
        bs1.set(20);
        bs1.set(30);

        bs2.set(20);
        bs2.set(30);
        bs2.set(40);

        // Test intersection
        let intersection = bs1.intersect(&bs2);
        assert!(intersection.test(20));
        assert!(intersection.test(30));
        assert!(!intersection.test(10));
        assert!(!intersection.test(40));
        assert_eq!(intersection.count(), 2);

        // Test union
        let union = bs1.union(&bs2);
        assert!(union.test(10));
        assert!(union.test(20));
        assert!(union.test(30));
        assert!(union.test(40));
        assert_eq!(union.count(), 4);
    }

    #[test]
    fn test_metadata_index() {
        let mut index = MetadataIndex::new();
        index.filterable_columns.insert("category".to_string());
        index.filterable_columns.insert("price".to_string());

        // Create test block
        let block = ProximaDataBlock {
            encoding_marker: 0x00,
            encoding_metadata: None,
            block_id: 0,
            encoded_vectors: None,
            vector_layout: crate::storage::engines::core::formats::proximablocks::VectorEncodingLayout::Auto,
            records: vec![VectorRecord {
                id: "1".to_string(),
                vector: vec![1.0, 2.0, 3.0],
                metadata: {
                    let mut meta = std::collections::HashMap::new();
                    meta.insert("category".to_string(), crate::proto::proximadb_v1::SqlValue {
                        value: Some(crate::proto::proximadb_v1::sql_value::Value::StringValue("electronics".to_string())),
                    });
                    meta.insert("price".to_string(), crate::proto::proximadb_v1::SqlValue {
                        value: Some(crate::proto::proximadb_v1::sql_value::Value::NumberValue(99.99)),
                    });
                    meta
                },
                timestamp: Some(0),
                updated_at: None,
                expires_at: None,
                version: None,
                source: Some("test".to_string()),
            }
            .into()],
            quantized_vectors: None,
            quantization_level: None,
            quantized_section: None,
            metadata: crate::storage::engines::core::formats::proximablocks::block_structures::ProximaBlockMetadata {
                record_count: 1,
                size_bytes: 0,
                compressed_size: 0,
                timestamp: 0,
                compaction_level: 0,
                has_deletes: false,
                has_updates: false,
                version_range: (0, 0),
                column_stats: std::collections::HashMap::new(),
                quantization_stats: crate::storage::engines::core::formats::proximablocks::block_structures::QuantizationStatistics::default(),
                data_checksum: 0,
                metadata_checksum: 0,
            },
            compression_config: crate::storage::engines::core::formats::proximablocks::block_structures::BlockCompressionConfig {
                algorithm: proximadb_compression::CompressionAlgorithm::Lz4,
                compression_level: 1,
                enable_vector_compression: true,
                enable_metadata_compression: true,
                compression_threshold_bytes: 8192,
                dictionary_compression: false,
                vector_layout: crate::storage::engines::core::formats::proximablocks::VectorEncodingLayout::Auto,
                metadata_algorithm: None,
            },
            compression_algorithm: proximadb_compression::CompressionAlgorithm::Lz4,
            uncompressed_size: 0,
            bloom_filter: None,
            block_bloom_filter: None,
            id_range: ("1".to_string(), "1".to_string()),
            timestamp_range: (0, 0),
            statistics: crate::storage::engines::core::formats::proximablocks::block_structures::BlockStatistics {
                read_count: 0,
                write_count: 0,
                search_count: 0,
                cache_hits: 0,
                cache_misses: 0,
                avg_read_time_ms: 0.0,
                avg_search_time_ms: 0.0,
                last_accessed_at: 0,
            },
            metadata_stats: None,
            has_deletes: false,
        };

        // Index the block
        index.index_block(0, 0, &block).unwrap();

        // Test finding blocks with specific value
        let matches = index
            .find_blocks_with_value("category", &serde_json::json!("electronics"))
            .unwrap();
        assert!(matches.test(0));

        // Test finding blocks in range
        // Now that OrderedValue properly handles numeric types, this should work correctly
        let matches = index
            .find_blocks_in_range("price", &serde_json::json!(50.0), &serde_json::json!(150.0))
            .unwrap();
        assert!(matches.test(0));
    }

    // ========================================================================
    // BitSet extended tests
    // ========================================================================

    #[test]
    fn test_bitset_empty() {
        let bs = BitSet::new(64);
        assert_eq!(bs.count(), 0);
        assert!(!bs.test(0));
        assert!(!bs.test(63));
    }

    #[test]
    fn test_bitset_size_zero() {
        let bs = BitSet::new(0);
        assert_eq!(bs.count(), 0);
        // Out-of-bounds test should return false
        assert!(!bs.test(0));
    }

    #[test]
    fn test_bitset_set_out_of_bounds() {
        let mut bs = BitSet::new(10);
        bs.set(100); // Out of bounds, should be silently ignored
        assert_eq!(bs.count(), 0);
    }

    #[test]
    fn test_bitset_test_out_of_bounds() {
        let bs = BitSet::new(10);
        assert!(!bs.test(10));
        assert!(!bs.test(100));
        assert!(!bs.test(usize::MAX));
    }

    #[test]
    fn test_bitset_single_bit() {
        let mut bs = BitSet::new(128);
        bs.set(0);
        assert!(bs.test(0));
        assert!(!bs.test(1));
        assert_eq!(bs.count(), 1);
    }

    #[test]
    fn test_bitset_cross_word_boundary() {
        let mut bs = BitSet::new(128);
        bs.set(63); // Last bit of first word
        bs.set(64); // First bit of second word
        assert!(bs.test(63));
        assert!(bs.test(64));
        assert_eq!(bs.count(), 2);
    }

    #[test]
    fn test_bitset_all_bits_set() {
        let mut bs = BitSet::new(64);
        for i in 0..64 {
            bs.set(i);
        }
        assert_eq!(bs.count(), 64);
    }

    #[test]
    fn test_bitset_intersect_different_sizes() {
        let mut bs1 = BitSet::new(200);
        let mut bs2 = BitSet::new(50);

        bs1.set(10);
        bs1.set(30);
        bs1.set(150); // Only in bs1 (beyond bs2 size)

        bs2.set(10);
        bs2.set(40); // Only in bs2

        let intersection = bs1.intersect(&bs2);
        assert!(intersection.test(10)); // Common
        assert!(!intersection.test(30)); // Only in bs1
        assert!(!intersection.test(40)); // Only in bs2
        assert_eq!(intersection.count(), 1);
    }

    #[test]
    fn test_bitset_union_different_sizes() {
        let mut bs1 = BitSet::new(64);
        let mut bs2 = BitSet::new(200);

        bs1.set(5);
        bs2.set(5);
        bs2.set(150); // Beyond bs1 size

        let union = bs1.union(&bs2);
        assert!(union.test(5));
        assert!(union.test(150));
        assert_eq!(union.count(), 2);
    }

    #[test]
    fn test_bitset_intersect_empty() {
        let bs1 = BitSet::new(100);
        let bs2 = BitSet::new(100);
        let intersection = bs1.intersect(&bs2);
        assert_eq!(intersection.count(), 0);
    }

    #[test]
    fn test_bitset_union_empty() {
        let bs1 = BitSet::new(100);
        let bs2 = BitSet::new(100);
        let union = bs1.union(&bs2);
        assert_eq!(union.count(), 0);
    }

    // ========================================================================
    // OrderedFloat tests
    // ========================================================================

    #[test]
    fn test_ordered_float_normal_comparison() {
        assert!(OrderedFloat(1.0) < OrderedFloat(2.0));
        assert!(OrderedFloat(2.0) > OrderedFloat(1.0));
        assert_eq!(OrderedFloat(1.0), OrderedFloat(1.0));
    }

    #[test]
    fn test_ordered_float_negative() {
        assert!(OrderedFloat(-1.0) < OrderedFloat(0.0));
        assert!(OrderedFloat(-2.0) < OrderedFloat(-1.0));
    }

    #[test]
    fn test_ordered_float_nan_handling() {
        let nan = OrderedFloat(f64::NAN);
        let one = OrderedFloat(1.0);

        // NaN is treated as greater than any non-NaN value via Ord
        assert!(nan > one);
        assert!(one < nan);

        // NaN == NaN via Ord (cmp returns Equal), but PartialEq uses f64's NaN != NaN
        let nan2 = OrderedFloat(f64::NAN);
        assert_eq!(nan.cmp(&nan2), std::cmp::Ordering::Equal);
    }

    // ========================================================================
    // OrderedValue tests
    // ========================================================================

    #[test]
    fn test_ordered_value_null_is_smallest() {
        assert!(OrderedValue::Null < OrderedValue::Bool(false));
        assert!(OrderedValue::Null < OrderedValue::Number(OrderedFloat(-1000.0)));
        assert!(OrderedValue::Null < OrderedValue::String("".to_string()));
    }

    #[test]
    fn test_ordered_value_type_ordering() {
        // Null < Bool < Number < String
        assert!(OrderedValue::Null < OrderedValue::Bool(false));
        assert!(OrderedValue::Bool(true) < OrderedValue::Number(OrderedFloat(0.0)));
        assert!(OrderedValue::Number(OrderedFloat(999.0)) < OrderedValue::String("a".to_string()));
    }

    #[test]
    fn test_ordered_value_bool_ordering() {
        assert!(OrderedValue::Bool(false) < OrderedValue::Bool(true));
        assert_eq!(OrderedValue::Bool(true), OrderedValue::Bool(true));
    }

    #[test]
    fn test_ordered_value_number_ordering() {
        assert!(OrderedValue::Number(OrderedFloat(1.0)) < OrderedValue::Number(OrderedFloat(2.0)));
        assert_eq!(
            OrderedValue::Number(OrderedFloat(3.5)),
            OrderedValue::Number(OrderedFloat(3.5))
        );
    }

    #[test]
    fn test_ordered_value_string_ordering() {
        assert!(OrderedValue::String("a".to_string()) < OrderedValue::String("b".to_string()));
        assert_eq!(
            OrderedValue::String("hello".to_string()),
            OrderedValue::String("hello".to_string())
        );
    }

    #[test]
    fn test_ordered_value_from_json() {
        assert_eq!(
            OrderedValue::from(serde_json::Value::Null),
            OrderedValue::Null
        );
        assert_eq!(
            OrderedValue::from(serde_json::Value::Bool(true)),
            OrderedValue::Bool(true)
        );
        assert_eq!(
            OrderedValue::from(serde_json::json!(42.0)),
            OrderedValue::Number(OrderedFloat(42.0))
        );
        assert_eq!(
            OrderedValue::from(serde_json::Value::String("test".to_string())),
            OrderedValue::String("test".to_string())
        );
    }

    #[test]
    fn test_ordered_value_from_json_array() {
        let arr = serde_json::json!([1, 2, 3]);
        let ordered = OrderedValue::from(arr);
        // Arrays fall back to string representation
        assert!(matches!(ordered, OrderedValue::String(_)));
    }

    // ========================================================================
    // Histogram tests
    // ========================================================================

    #[test]
    fn test_histogram_empty() {
        let hist = Histogram {
            buckets: Vec::new(),
            total_count: 0,
        };
        assert_eq!(hist.total_count, 0);
        assert!(hist.buckets.is_empty());
    }

    #[test]
    fn test_histogram_with_buckets() {
        let hist = Histogram {
            buckets: vec![
                HistogramBucket {
                    min: 0.0,
                    max: 10.0,
                    count: 50,
                },
                HistogramBucket {
                    min: 10.0,
                    max: 20.0,
                    count: 30,
                },
                HistogramBucket {
                    min: 20.0,
                    max: 30.0,
                    count: 20,
                },
            ],
            total_count: 100,
        };
        assert_eq!(hist.buckets.len(), 3);
        let bucket_sum: u64 = hist.buckets.iter().map(|b| b.count).sum();
        assert_eq!(bucket_sum, hist.total_count);
    }

    // ========================================================================
    // MetadataIndex creation tests
    // ========================================================================

    #[test]
    fn test_metadata_index_default() {
        let index = MetadataIndex::default();
        assert_eq!(index.table_stats.total_records, 0);
        assert_eq!(index.table_stats.total_blocks, 0);
    }

    #[test]
    fn test_metadata_index_filterable_columns() {
        let mut index = MetadataIndex::new();
        index.filterable_columns.insert("category".to_string());
        index.filterable_columns.insert("price".to_string());
        assert!(index.filterable_columns.contains("category"));
        assert!(index.filterable_columns.contains("price"));
        assert!(!index.filterable_columns.contains("unknown"));
    }
}
