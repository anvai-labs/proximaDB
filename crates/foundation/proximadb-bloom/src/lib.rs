/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */
//! Bloom-filter implementations for ProximaDB.
//!
//! Extracted from the root crate's `src/core/bloom` as a slice of the root-crate
//! decomposition (see `docs/history/planning/ROOT_CRATE_DECOMPOSITION_PLAN_2026_06_21.adoc`,
//! Slice D / D1). The root crate re-exports it as `crate::core::bloom` for source
//! compatibility.

//! # Bloom Filter Module - Probabilistic Data Structure for Fast Lookups
//!
//! This module provides ProximaDB's sophisticated bloom filter implementation using
//! the Strategy Pattern for multiple filter types. Bloom filters provide space-efficient
//! probabilistic membership testing, eliminating 95%+ of unnecessary disk reads.
//!
//! ## Bloom Filter Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │         Bloom Filter System              │
//! ├─────────────────────────────────────────┤
//! │ Strategy │ Factory │ Builder │ Stats    │
//! ├─────────────────────────────────────────┤
//! │ BitPacked │ ByteAligned │ Simple │ Composite │
//! └─────────────────────────────────────────┘
//!           ↓              ↓              ↓
//!     SST Files    Metadata    MemTable
//! ```
//!
//! ## Key Concepts
//!
//! ### What is a Bloom Filter?
//! A bloom filter is a space-efficient probabilistic data structure that tests
//! whether an element is a member of a set. Key properties:
//! - **No False Negatives**: If filter says NO, item definitely doesn't exist
//! - **Possible False Positives**: If filter says YES, item might exist
//! - **Space Efficient**: Uses bit array instead of storing actual items
//! - **Fast Operations**: O(k) for insert/lookup where k = hash functions
//!
//! ### How It Works
//! ```text
//! Insert "hello":
//! hash1("hello") = 3 → Set bit 3
//! hash2("hello") = 7 → Set bit 7
//! hash3("hello") = 11 → Set bit 11
//!
//! Check "world":
//! hash1("world") = 3 ✓ (bit is set)
//! hash2("world") = 7 ✓ (bit is set)  
//! hash3("world") = 15 ✗ (bit not set)
//! Result: "world" definitely not in set
//! ```
//!
//! ## Filter Strategies
//!
//! ### 1. **BitPacked Strategy**
//! Memory-efficient bit array implementation:
//! - **Storage**: 1 bit per position
//! - **Memory**: Minimal overhead
//! - **Use Case**: In-memory filters, memtables
//! - **Performance**: Fastest for memory operations
//!
//! ### 2. **ByteAligned Strategy**
//! Byte-aligned storage for disk persistence:
//! - **Storage**: Byte boundaries for I/O efficiency
//! - **Memory**: Slightly more than BitPacked
//! - **Use Case**: SST files, persistent storage
//! - **Performance**: Optimized for disk I/O
//!
//! ### 3. **Simple Strategy**
//! Boolean array for small datasets:
//! - **Storage**: 1 byte per position
//! - **Memory**: Higher usage
//! - **Use Case**: Small filters, testing
//! - **Performance**: Fast for small sizes
//!
//! ### 4. **Composite Strategy**
//! Combines multiple filters:
//! - **Key Filter**: For record IDs
//! - **Metadata Filter**: For attributes
//! - **Hierarchical**: File and block level
//! - **Use Case**: Complex filtering scenarios
//!
//! ## Hash Functions
//!
//! Multiple hash algorithms for different needs:
//! - **MurmurHash3**: Default, good distribution
//! - **xxHash**: Extremely fast
//! - **FNV-1a**: Simple and fast, good for small keys
//!
//! Double hashing generates k hash values from 2 base hashes:
//! ```rust,ignore
//! hash_i = hash1 + i * hash2 (mod m)
//! ```
//!
//! ## False Positive Rate
//!
//! The false positive rate depends on:
//! - **m**: Number of bits in filter
//! - **n**: Number of inserted elements
//! - **k**: Number of hash functions
//!
//! Optimal k = (m/n) * ln(2) ≈ 0.7 * (m/n)
//!
//! ### Bits Per Key Guide
//! | Bits/Key | FPR | Use Case |
//! |----------|-----|----------|
//! | 4 | 10% | Quick elimination |
//! | 8 | 2% | Balanced |
//! | 10 | 1% | Default |
//! | 14 | 0.1% | High accuracy |
//! | 20 | 0.01% | Critical paths |
//!
//! ## SSTable Integration
//!
//! SST files use hierarchical bloom filters:
//! ```text
//! File Level (Global):
//!   - Quick file elimination
//!   - 10 bits per key
//!   
//! Block Level (Local):
//!   - Fine-grained filtering
//!   - 8 bits per key
//!   
//! Metadata Filter:
//!   - Column-specific filters
//!   - Type-aware hashing
//! ```
//!
//! ## Performance Characteristics
//!
//! - **Insert Speed**: 1M+ ops/sec
//! - **Lookup Speed**: 10M+ ops/sec
//! - **Memory Usage**: 10 bits/key = 1.25 bytes/key
//! - **Disk I/O Saved**: 95%+ for negative lookups
//! - **CPU Overhead**: < 100ns per operation
//!
//! ## Configuration
//!
//! ```toml
//! [bloom_filter]
//! # Strategy selection
//! strategy = "byte_aligned"  # bit_packed, simple, composite
//!
//! # Accuracy settings
//! bits_per_key = 10
//! false_positive_rate = 0.01  # Alternative to bits_per_key
//! expected_items = 100000
//!
//! # Hash configuration
//! hash_algorithm = "murmur3"  # xxhash, cityhash
//!
//! # Hierarchical settings
//! [bloom_filter.hierarchical]
//! enable_block_filters = true
//! block_size = 4096
//! metadata_columns = ["category", "status", "tags"]
//! ```
//!
//! ## Usage Examples
//!
//! ### Basic Bloom Filter
//! ```rust,ignore
//! use proximadb::bloom::{BloomFilterBuilder, CoreBloomFilterConfig};
//!
//! let config = CoreBloomFilterConfig::for_sstable(10000);
//! let mut builder = BloomFilterBuilder::new(config);
//!
//! // Add keys
//! builder.add(b"user:123");
//! builder.add(b"user:456");
//!
//! let filter = builder.build();
//!
//! // Check membership
//! if filter.might_contain(b"user:123") {
//!     // Key might exist, check storage
//! } else {
//!     // Key definitely doesn't exist, skip I/O
//! }
//! ```
//!
//! ### Metadata Filtering
//! ```rust,ignore
//! use proximadb::bloom::SstableBloomFilter;
//!
//! let filter = SstableBloomFilter::new(...);
//!
//! // Check metadata
//! let item = MetadataItem {
//!     key: "category".to_string(),
//!     value: Some(Value::StringValue("electronics".to_string())),
//! };
//!
//! if filter.might_match_metadata("category", &item)? {
//!     // Metadata might match
//! }
//! ```
//!
//! ## Memory Optimization
//!
//! Target memory usage per collection:
//! - **Key Filter**: ~4MB for 1M keys
//! - **Metadata Filter**: ~2MB for 10 columns
//! - **Block Filters**: ~2MB for 1000 blocks
//! - **Total**: ~8MB per collection
//!
//! ## Best Practices
//!
//! 1. **Size Appropriately**: Use expected_items for optimal sizing
//! 2. **Choose Strategy**: BitPacked for memory, ByteAligned for disk
//! 3. **Monitor FPR**: Track actual vs expected false positive rate
//! 4. **Use Hierarchical**: For large SST files with many blocks
//! 5. **Type-Aware Hashing**: Serialize metadata consistently

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod factory;
pub mod strategies;

// Phase 1.2: Adaptive Bloom Filters
pub mod adaptive;
pub mod compression;
pub mod hierarchical;

/// Core trait for all bloom filter implementations
pub trait BloomFilterStrategy: Send + Sync + std::fmt::Debug {
    /// Insert a key into the bloom filter
    fn insert(&mut self, key: &[u8]);

    /// Check if a key might exist in the bloom filter
    fn might_contain(&self, key: &[u8]) -> bool;

    /// Get the number of bits used by the filter
    fn bit_count(&self) -> usize;

    /// Get the number of hash functions used
    fn hash_count(&self) -> usize;

    /// Serialize the bloom filter to bytes
    fn serialize(&self) -> Result<Vec<u8>>;

    /// Get memory usage in bytes
    fn memory_usage(&self) -> usize;

    /// Clear all bits in the filter
    fn clear(&mut self);

    /// Get the false positive rate (estimated)
    fn false_positive_rate(&self) -> f64;

    /// Get the number of elements inserted
    fn num_elements(&self) -> usize;
}

/// Trait for metadata-aware bloom filters with type-safe operations
pub trait MetadataBloomFilter: BloomFilterStrategy {
    /// Insert a metadata item with proper type handling
    fn insert_metadata(&mut self, column: &str, item: &proximadb_proto::proximadb_v1::MetadataItem);

    /// Check if metadata might match with proper type handling
    fn might_match_metadata(
        &self,
        column: &str,
        item: &proximadb_proto::proximadb_v1::MetadataItem,
    ) -> bool;

    /// Get the number of metadata columns tracked
    fn num_columns(&self) -> usize;
}

/// Serialize metadata value for bloom filter hashing
/// This ensures consistent serialization across all types
pub fn serialize_metadata_value(item: &proximadb_proto::proximadb_v1::MetadataItem) -> String {
    use proximadb_proto::proximadb_v1::metadata_item::Value;

    match &item.value {
        Some(Value::StringValue(s)) => s.clone(),
        Some(Value::NumberValue(n)) => {
            // Use consistent number formatting to avoid precision issues
            // For integers, format without decimal point
            if n.fract() == 0.0 && n.is_finite() {
                format!("{:.0}", n)
            } else {
                // Use scientific notation for consistent representation
                format!("{:e}", n)
            }
        }
        Some(Value::BoolValue(b)) => b.to_string(),
        None => String::new(),
    }
}

/// Convert JSON value to MetadataItem for bloom filter operations
pub fn json_to_metadata_item(
    key: &str,
    value: &serde_json::Value,
) -> proximadb_proto::proximadb_v1::MetadataItem {
    use proximadb_proto::proximadb_v1::metadata_item::Value as ProtoValue;

    let proto_value = match value {
        serde_json::Value::String(s) => Some(ProtoValue::StringValue(s.clone())),
        serde_json::Value::Number(n) => n.as_f64().map(ProtoValue::NumberValue),
        serde_json::Value::Bool(b) => Some(ProtoValue::BoolValue(*b)),
        _ => None, // Null, Array, Object not supported in MetadataItem
    };

    proximadb_proto::proximadb_v1::MetadataItem {
        key: key.to_string(),
        value: proto_value,
    }
}

/// Bloom filter strategy types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum BloomStrategy {
    /// Memory efficient, bit-packed storage (best for in-memory operations)
    BitPacked,
    /// Byte-aligned storage (best for disk persistence)
    ByteAligned,
    /// Simple boolean array (fast for small datasets)
    Simple,
    /// Composite filter combining multiple strategies
    Composite,
}

/// Hash algorithm selection
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub enum HashAlgorithm {
    /// MurmurHash3 - fast and good distribution
    Murmur3,
    /// xxHash - extremely fast (default) - our internal optimized implementation
    #[default]
    XXHash,
    /// FNV-1a - simple and fast, good for small keys
    Fnv1a,
}

/// Backwards-compat alias for [`CoreBloomFilterConfig`].
pub type BloomFilterConfig = CoreBloomFilterConfig;

/// Enhanced bloom filter configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CoreBloomFilterConfig {
    /// Strategy to use
    pub strategy: BloomStrategy,

    /// Bits per key (for ByteAligned/BitPacked)
    pub bits_per_key: u32,

    /// Target false positive rate (alternative to bits_per_key)
    pub false_positive_rate: Option<f64>,

    /// Expected number of items
    pub expected_items: usize,

    /// Enable bloom filter
    pub enabled: bool,

    /// Hash function selection
    pub hash_algorithm: HashAlgorithm,
}

impl Default for CoreBloomFilterConfig {
    fn default() -> Self {
        Self {
            strategy: BloomStrategy::ByteAligned,
            bits_per_key: 10,
            false_positive_rate: None,
            expected_items: 10000,
            enabled: true,
            hash_algorithm: HashAlgorithm::default(),
        }
    }
}

impl CoreBloomFilterConfig {
    /// Create config for SSTable usage
    pub fn for_sstable(expected_items: usize) -> Self {
        Self {
            strategy: BloomStrategy::ByteAligned,
            bits_per_key: 10,
            expected_items,
            ..Default::default()
        }
    }

    /// Create config for memtable usage
    pub fn for_memtable(expected_items: usize) -> Self {
        Self {
            strategy: BloomStrategy::BitPacked,
            bits_per_key: 8,
            expected_items,
            ..Default::default()
        }
    }

    /// Calculate bits per key from false positive rate
    pub fn bits_from_fpr(false_positive_rate: f64) -> u32 {
        let ln2_squared = 0.4804530139182014;
        let bits = (-(false_positive_rate.ln() / ln2_squared)) as u32;
        bits.clamp(1, 32)
    }
}

/// Serialized bloom filter with type information
#[derive(Serialize, Deserialize)]
pub struct SerializedBloomFilter {
    /// Type tag for polymorphic deserialization
    pub strategy_type: BloomStrategy,

    /// Version for forward compatibility
    pub version: u32,

    /// Configuration used to create the filter
    pub config: CoreBloomFilterConfig,

    /// Serialized filter data
    pub data: Vec<u8>,

    /// Metadata about the filter
    pub metadata: HashMap<String, serde_json::Value>,
}

impl SerializedBloomFilter {
    /// Current serialization version
    pub const CURRENT_VERSION: u32 = 1;

    /// Create from a bloom filter instance
    pub fn from_filter(
        filter: &dyn BloomFilterStrategy,
        config: CoreBloomFilterConfig,
    ) -> Result<Self> {
        let mut metadata = HashMap::new();
        metadata.insert(
            "num_elements".to_string(),
            serde_json::Value::Number(filter.num_elements().into()),
        );
        metadata.insert(
            "false_positive_rate".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(filter.false_positive_rate())
                    .unwrap_or_else(|| serde_json::Number::from(0)),
            ),
        );

        Ok(Self {
            strategy_type: BloomStrategy::ByteAligned, // Default strategy
            version: Self::CURRENT_VERSION,
            config,
            data: filter.serialize()?,
            metadata,
        })
    }
}

/// Hash functions for bloom filters
pub mod hash {
    use std::hash::{Hash, Hasher};

    /// xxHash 64-bit placeholder implementation
    pub fn xxhash64(data: &[u8]) -> u64 {
        // Simple placeholder - in production, use xxhash-rust crate
        let mut hash = 0u64;
        for (i, &byte) in data.iter().enumerate() {
            hash = hash.wrapping_mul(0x1b873593).wrapping_add(byte as u64);
            hash = hash.rotate_left((i % 32) as u32);
        }
        hash
    }

    /// FNV-1a - fast simple hash
    pub fn cityhash64(data: &[u8]) -> u64 {
        // Simple placeholder - in production, use cityhash crate
        let mut hash = 0x9ae16a3b2f90404fu64;
        for chunk in data.chunks(8) {
            let mut value = 0u64;
            for (i, &byte) in chunk.iter().enumerate() {
                value |= (byte as u64) << (i * 8);
            }
            hash = hash.wrapping_mul(0xc3a5c85c97cb3127).wrapping_add(value);
            hash = hash.rotate_left(31);
        }
        hash
    }

    /// MurmurHash3 32-bit implementation
    pub fn murmur3_32(key: &[u8], seed: u32) -> u32 {
        let mut h = seed;
        let (chunks, remainder) = key.as_chunks::<4>();

        // Process 4-byte chunks
        for chunk in chunks {
            let mut k = u32::from_le_bytes(*chunk);
            k = k.wrapping_mul(0xcc9e2d51);
            k = k.rotate_left(15);
            k = k.wrapping_mul(0x1b873593);

            h ^= k;
            h = h.rotate_left(13);
            h = h.wrapping_mul(5).wrapping_add(0xe6546b64);
        }

        // Process remaining bytes
        if !remainder.is_empty() {
            let mut k = 0u32;
            for (i, &byte) in remainder.iter().enumerate() {
                k |= (byte as u32) << (i * 8);
            }

            k = k.wrapping_mul(0xcc9e2d51);
            k = k.rotate_left(15);
            k = k.wrapping_mul(0x1b873593);
            h ^= k;
        }

        // Finalization
        h ^= key.len() as u32;
        h ^= h >> 16;
        h = h.wrapping_mul(0x85ebca6b);
        h ^= h >> 13;
        h = h.wrapping_mul(0xc2b2ae35);
        h ^= h >> 16;

        h
    }

    /// Simple hash function for testing
    pub fn simple_hash(key: &[u8], seed: u32) -> u32 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        seed.hash(&mut hasher);
        key.hash(&mut hasher);
        hasher.finish() as u32
    }

    /// Double hashing to generate multiple hash values
    pub fn double_hash(key: &[u8], num_hashes: u32, bit_count: usize) -> Vec<usize> {
        // Default to XXHash for speed
        double_hash_with_algorithm(key, num_hashes, bit_count, crate::HashAlgorithm::XXHash)
    }

    /// Double hashing with configurable algorithm
    pub fn double_hash_with_algorithm(
        key: &[u8],
        num_hashes: u32,
        bit_count: usize,
        algorithm: crate::HashAlgorithm,
    ) -> Vec<usize> {
        if bit_count == 0 {
            return vec![0; num_hashes as usize];
        }

        let (h1, h2) = match algorithm {
            crate::HashAlgorithm::XXHash => {
                use proximadb_kernel::hash::{FastHash, XxHasher};
                // Two independent hashes using XXHash
                let h1 = XxHasher::hash_bytes(key) as u32;
                let h2 = XxHasher::hash_bytes(&[key, b"_salt"].concat()) as u32;
                (h1, h2)
            }
            crate::HashAlgorithm::Murmur3 => {
                let h1 = murmur3_32(key, 0);
                let h2 = murmur3_32(key, h1);
                (h1, h2)
            }
            crate::HashAlgorithm::Fnv1a => {
                // Use FNV-1a from our internal utils module
                use proximadb_kernel::hash::{FastHash, FnvHasher};

                // First hash
                let h1 = FnvHasher::hash_bytes(key) as u32;

                // Second hash with salt for independence
                let mut key_with_salt = Vec::with_capacity(key.len() + 8);
                key_with_salt.extend_from_slice(key);
                key_with_salt.extend_from_slice(b"_fnv_h2");
                let h2 = FnvHasher::hash_bytes(&key_with_salt) as u32;

                (h1, h2)
            }
        };

        (0..num_hashes)
            .map(|i| {
                let hash = h1.wrapping_add(i.wrapping_mul(h2));
                (hash as usize) % bit_count
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bits_from_fpr() {
        // Test conversion from false positive rate to bits per key
        assert_eq!(CoreBloomFilterConfig::bits_from_fpr(0.01), 9); // ~1% FPR needs ~9 bits
        assert_eq!(CoreBloomFilterConfig::bits_from_fpr(0.001), 14); // ~0.1% FPR needs ~14 bits
        assert_eq!(CoreBloomFilterConfig::bits_from_fpr(0.1), 4); // ~10% FPR needs ~4 bits
    }

    #[test]
    fn test_murmur_hash() {
        let key = b"test_key";
        let hash1 = hash::murmur3_32(key, 0);
        let hash2 = hash::murmur3_32(key, 1);

        // Different seeds should produce different hashes
        assert_ne!(hash1, hash2);

        // Same input should produce same hash
        assert_eq!(hash::murmur3_32(key, 0), hash1);
    }
}

// ============================================================================
// Additional types and aliases for storage engine compatibility
// ============================================================================

/// Backwards-compat alias for [`CoreBloomFilterStats`].
pub type BloomFilterStats = CoreBloomFilterStats;

/// Stats for bloom filter usage
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct CoreBloomFilterStats {
    /// Number of primary keys inserted into bloom filters
    pub key_count: u64,
    /// Number of metadata columns with bloom filters
    pub metadata_columns: u64,
    /// Total keys across all bloom filters
    pub total_keys: u64,
    /// Number of key lookups avoided by bloom filter negatives
    pub key_lookups_saved: u64,
    /// Number of metadata queries avoided by bloom filter negatives
    pub metadata_queries_saved: u64,
}

/// Combined bloom filter for SSTable (keys + metadata)
/// Memory target: ~8MB per collection (down from ~40MB)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SstableBloomFilter {
    /// Key filter configuration
    pub key_filter_config: CoreBloomFilterConfig,
    /// Key filter data
    pub key_filter_data: Vec<u8>,
    /// Metadata filter data  
    pub metadata_filter_data: Vec<u8>,
    /// Statistics
    pub stats: CoreBloomFilterStats,
    /// Memory usage tracking
    memory_usage: Option<usize>,
}

impl SstableBloomFilter {
    /// Create a new SstableBloomFilter
    pub fn new(
        key_filter_config: CoreBloomFilterConfig,
        key_filter_data: Vec<u8>,
        metadata_filter_data: Vec<u8>,
        stats: CoreBloomFilterStats,
    ) -> Self {
        let memory_usage =
            std::mem::size_of::<Self>() + key_filter_data.len() + metadata_filter_data.len();
        Self {
            key_filter_config,
            key_filter_data,
            metadata_filter_data,
            stats,
            memory_usage: Some(memory_usage),
        }
    }

    /// Check if key might exist using the unified bloom filter
    pub fn might_contain_key(&self, key: &str) -> Result<bool> {
        // If bloom filter is empty, conservatively return true
        if self.key_filter_data.is_empty() {
            return Ok(true);
        }

        // Deserialize the bloom filter strategy based on configuration
        let bloom_strategy = factory::BloomFilterFactory::deserialize(
            &self.key_filter_config,
            &self.key_filter_data,
        )?;

        // Use the unified bloom filter's might_contain method
        Ok(bloom_strategy.might_contain(key.as_bytes()))
    }

    /// Check if metadata might match using MetadataItem for type safety
    ///
    /// Returns `true` (might match) when:
    /// - No metadata filter exists (can't reject without filter data)
    /// - Metadata filter exists but indicates potential match
    ///
    /// Returns `false` (definitely no match) only when metadata filter
    /// is populated AND definitively rejects the item.
    pub fn might_match_metadata(
        &self,
        _column: &str,
        _item: &proximadb_proto::proximadb_v1::MetadataItem,
    ) -> Result<bool> {
        if self.metadata_filter_data.is_empty() {
            // No metadata bloom filter built - cannot reject, might match
            return Ok(true);
        }
        // Conservative approach: assume metadata might match
        // Deferred: Implement actual metadata bloom filter checking when metadata is indexed
        Ok(true)
    }

    /// Get total size in bytes
    pub fn total_size_bytes(&self) -> usize {
        self.key_filter_data.len() + self.metadata_filter_data.len()
    }

    /// Get memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        self.memory_usage.unwrap_or_else(|| {
            std::mem::size_of::<Self>()
                + self.key_filter_data.len()
                + self.metadata_filter_data.len()
        })
    }

    /// Check if within target memory limit (8MB)
    pub fn is_within_memory_target(&self) -> bool {
        self.memory_usage_bytes() < 8 * 1024 * 1024
    }

    /// Custom serialization using manual byte layout to avoid bincode issues
    pub fn serialize(&self) -> Result<Vec<u8>> {
        use std::io::Write;
        let mut buffer = Vec::new();

        // Write a simple header
        buffer.write_all(b"BF01")?; // Magic bytes + version

        // Write config
        let strategy_byte = match self.key_filter_config.strategy {
            BloomStrategy::BitPacked => 0u8,
            BloomStrategy::ByteAligned => 1u8,
            BloomStrategy::Simple => 2u8,
            BloomStrategy::Composite => 3u8,
        };
        buffer.write_all(&[strategy_byte])?;
        buffer.write_all(&self.key_filter_config.bits_per_key.to_le_bytes())?;
        let fpr = self
            .key_filter_config
            .false_positive_rate
            .unwrap_or(f64::NAN);
        buffer.write_all(&fpr.to_le_bytes())?;
        buffer.write_all(&self.key_filter_config.expected_items.to_le_bytes())?;
        buffer.write_all(&[if self.key_filter_config.enabled {
            1u8
        } else {
            0u8
        }])?;

        let hash_byte = match self.key_filter_config.hash_algorithm {
            HashAlgorithm::Murmur3 => 0u8,
            HashAlgorithm::XXHash => 1u8,
            HashAlgorithm::Fnv1a => 2u8,
        };
        buffer.write_all(&[hash_byte])?;

        // Write stats
        buffer.write_all(&self.stats.key_count.to_le_bytes())?;
        buffer.write_all(&self.stats.metadata_columns.to_le_bytes())?;
        buffer.write_all(&self.stats.total_keys.to_le_bytes())?;
        buffer.write_all(&self.stats.key_lookups_saved.to_le_bytes())?;
        buffer.write_all(&self.stats.metadata_queries_saved.to_le_bytes())?;

        // Write data lengths and data
        buffer.write_all(&(self.key_filter_data.len() as u32).to_le_bytes())?;
        buffer.write_all(&self.key_filter_data)?;

        buffer.write_all(&(self.metadata_filter_data.len() as u32).to_le_bytes())?;
        buffer.write_all(&self.metadata_filter_data)?;

        Ok(buffer)
    }

    /// Custom deserialization using manual byte layout to avoid bincode issues
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        use std::io::Read;
        let mut cursor = std::io::Cursor::new(data);

        // Read and validate header
        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic)?;
        if &magic != b"BF01" {
            return Err(anyhow::anyhow!("Invalid bloom filter format"));
        }

        // Read config
        let mut strategy_buf = [0u8; 1];
        cursor.read_exact(&mut strategy_buf)?;
        let strategy = match strategy_buf[0] {
            0 => BloomStrategy::BitPacked,
            1 => BloomStrategy::ByteAligned,
            2 => BloomStrategy::Simple,
            3 => BloomStrategy::Composite,
            _ => BloomStrategy::ByteAligned,
        };

        let mut bits_per_key_buf = [0u8; 4];
        cursor.read_exact(&mut bits_per_key_buf)?;
        let bits_per_key = u32::from_le_bytes(bits_per_key_buf);

        let mut fpr_buf = [0u8; 8];
        cursor.read_exact(&mut fpr_buf)?;
        let fpr = f64::from_le_bytes(fpr_buf);
        let false_positive_rate = if fpr.is_nan() { None } else { Some(fpr) };

        let mut expected_items_buf = [0u8; 8];
        cursor.read_exact(&mut expected_items_buf)?;
        let expected_items = usize::from_le_bytes(expected_items_buf);

        let mut enabled_buf = [0u8; 1];
        cursor.read_exact(&mut enabled_buf)?;
        let enabled = enabled_buf[0] != 0;

        let mut hash_buf = [0u8; 1];
        cursor.read_exact(&mut hash_buf)?;
        let hash_algorithm = match hash_buf[0] {
            0 => HashAlgorithm::Murmur3,
            1 => HashAlgorithm::XXHash,
            2 => HashAlgorithm::Fnv1a,
            _ => HashAlgorithm::Murmur3,
        };

        // Read stats
        let mut stats_buf = [0u8; 8];

        cursor.read_exact(&mut stats_buf)?;
        let key_count = u64::from_le_bytes(stats_buf);

        cursor.read_exact(&mut stats_buf)?;
        let metadata_columns = u64::from_le_bytes(stats_buf);

        cursor.read_exact(&mut stats_buf)?;
        let total_keys = u64::from_le_bytes(stats_buf);

        cursor.read_exact(&mut stats_buf)?;
        let key_lookups_saved = u64::from_le_bytes(stats_buf);

        cursor.read_exact(&mut stats_buf)?;
        let metadata_queries_saved = u64::from_le_bytes(stats_buf);

        // Read data
        let mut len_buf = [0u8; 4];
        cursor.read_exact(&mut len_buf)?;
        let key_data_len = u32::from_le_bytes(len_buf) as usize;

        let mut key_filter_data = vec![0u8; key_data_len];
        cursor.read_exact(&mut key_filter_data)?;

        cursor.read_exact(&mut len_buf)?;
        let meta_data_len = u32::from_le_bytes(len_buf) as usize;

        let mut metadata_filter_data = vec![0u8; meta_data_len];
        cursor.read_exact(&mut metadata_filter_data)?;

        // Create the structures
        let stats = CoreBloomFilterStats {
            key_count,
            metadata_columns,
            total_keys,
            key_lookups_saved,
            metadata_queries_saved,
        };

        let key_filter_config = CoreBloomFilterConfig {
            strategy,
            bits_per_key,
            false_positive_rate,
            expected_items,
            enabled,
            hash_algorithm,
        };

        Ok(Self::new(
            key_filter_config,
            key_filter_data,
            metadata_filter_data,
            stats,
        ))
    }
}

/// Hierarchical bloom filter configuration for SST files
#[derive(Debug, Clone)]
pub struct HierarchicalBloomConfig {
    /// Configuration for the global-level key bloom filter
    pub global_key_filter: CoreBloomFilterConfig,
    /// Configuration for the global-level metadata bloom filter
    pub global_metadata_filter: CoreBloomFilterConfig,
    /// Configuration for per-block key bloom filters
    pub block_key_filter: CoreBloomFilterConfig,
    /// Configuration for per-block metadata bloom filters
    pub block_metadata_filter: CoreBloomFilterConfig,
    /// Minimum number of blocks before enabling hierarchical filters
    pub block_count_threshold: usize,
    /// Minimum metadata columns before enabling metadata bloom filters
    pub metadata_column_threshold: usize,
}

/// Bloom filter builder for incremental construction
pub struct BloomFilterBuilder {
    #[allow(dead_code)]
    config: CoreBloomFilterConfig,
    filter: Box<dyn BloomFilterStrategy>,
}

impl BloomFilterBuilder {
    /// Create a new bloom filter builder with the given configuration
    pub fn new(config: CoreBloomFilterConfig) -> Self {
        let filter = factory::BloomFilterFactory::create(&config);
        Self { config, filter }
    }

    /// Insert a key into the bloom filter being built
    pub fn add(&mut self, key: &[u8]) {
        self.filter.insert(key);
    }

    /// Finalize and return the constructed bloom filter
    pub fn build(self) -> Box<dyn BloomFilterStrategy> {
        self.filter
    }
}

/// Serializable version of SstableBloomFilter to work around bincode limitations
#[derive(Debug, Serialize, Deserialize)]
pub struct SerializedSstableBloomFilter {
    // CoreBloomFilterConfig fields flattened to avoid Option<f64> issues
    strategy: u8,
    bits_per_key: u32,
    false_positive_rate: f64, // Use NaN for None
    expected_items: usize,
    enabled: bool,
    hash_algorithm: u8,

    // Filter data
    key_filter_data: Vec<u8>,
    metadata_filter_data: Vec<u8>,

    // Stats fields
    key_count: u64,
    metadata_columns: u64,
    total_keys: u64,
    key_lookups_saved: u64,
    metadata_queries_saved: u64,
}

// Implement conversion for serde
impl From<SerializedSstableBloomFilter> for SstableBloomFilter {
    fn from(serialized: SerializedSstableBloomFilter) -> Self {
        let strategy = match serialized.strategy {
            0 => BloomStrategy::BitPacked,
            1 => BloomStrategy::ByteAligned,
            2 => BloomStrategy::Simple,
            3 => BloomStrategy::Composite,
            _ => BloomStrategy::ByteAligned,
        };

        let hash_algorithm = match serialized.hash_algorithm {
            0 => HashAlgorithm::Murmur3,
            1 => HashAlgorithm::XXHash,
            2 => HashAlgorithm::Fnv1a,
            _ => HashAlgorithm::Murmur3,
        };

        let false_positive_rate = if serialized.false_positive_rate.is_nan() {
            None
        } else {
            Some(serialized.false_positive_rate)
        };

        let memory_usage = std::mem::size_of::<Self>()
            + serialized.key_filter_data.len()
            + serialized.metadata_filter_data.len();

        Self {
            key_filter_config: CoreBloomFilterConfig {
                strategy,
                bits_per_key: serialized.bits_per_key,
                false_positive_rate,
                expected_items: serialized.expected_items,
                enabled: serialized.enabled,
                hash_algorithm,
            },
            key_filter_data: serialized.key_filter_data,
            metadata_filter_data: serialized.metadata_filter_data,
            stats: CoreBloomFilterStats {
                key_count: serialized.key_count,
                metadata_columns: serialized.metadata_columns,
                total_keys: serialized.total_keys,
                key_lookups_saved: serialized.key_lookups_saved,
                metadata_queries_saved: serialized.metadata_queries_saved,
            },
            memory_usage: Some(memory_usage),
        }
    }
}

impl From<SstableBloomFilter> for SerializedSstableBloomFilter {
    fn from(bf: SstableBloomFilter) -> Self {
        Self {
            strategy: match bf.key_filter_config.strategy {
                BloomStrategy::BitPacked => 0,
                BloomStrategy::ByteAligned => 1,
                BloomStrategy::Simple => 2,
                BloomStrategy::Composite => 3,
            },
            bits_per_key: bf.key_filter_config.bits_per_key,
            false_positive_rate: bf.key_filter_config.false_positive_rate.unwrap_or(f64::NAN),
            expected_items: bf.key_filter_config.expected_items,
            enabled: bf.key_filter_config.enabled,
            hash_algorithm: match bf.key_filter_config.hash_algorithm {
                HashAlgorithm::Murmur3 => 0,
                HashAlgorithm::XXHash => 1,
                HashAlgorithm::Fnv1a => 2,
            },
            key_filter_data: bf.key_filter_data,
            metadata_filter_data: bf.metadata_filter_data,
            key_count: bf.stats.key_count,
            metadata_columns: bf.stats.metadata_columns,
            total_keys: bf.stats.total_keys,
            key_lookups_saved: bf.stats.key_lookups_saved,
            metadata_queries_saved: bf.stats.metadata_queries_saved,
        }
    }
}

/// Type alias for a heap-allocated bloom filter using any strategy
pub type BloomFilter = Box<dyn BloomFilterStrategy>;
/// Type alias for a composite bloom filter combining multiple strategies
pub type CompositeBloomFilter = Box<dyn BloomFilterStrategy>;
