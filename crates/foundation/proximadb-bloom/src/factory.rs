/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! Bloom filter factory for creating appropriate implementations

use anyhow::Result;
use std::sync::Arc;

use super::strategies::{
    BitPackedBloomFilter, ByteAlignedBloomFilter, CompositeBloomFilter, SimpleBloomFilter,
};
use super::{BloomFilterConfig, BloomFilterStrategy, BloomStrategy, SerializedBloomFilter};

/// Factory for creating bloom filter instances
pub struct BloomFilterFactory;

impl BloomFilterFactory {
    /// Create a bloom filter based on configuration
    pub fn create(config: &BloomFilterConfig) -> Box<dyn BloomFilterStrategy> {
        if !config.enabled || config.expected_items == 0 {
            return Box::new(NoOpBloomFilter);
        }

        // Default to ByteAligned strategy for general use
        Box::new(ByteAlignedBloomFilter::new(config.expected_items, config))
    }

    /// Create a bloom filter for SSTable usage
    pub fn for_sstable(expected_items: usize) -> Box<dyn BloomFilterStrategy> {
        let config = BloomFilterConfig::for_sstable(expected_items);
        Self::create(&config)
    }

    /// Create a bloom filter for memtable usage
    pub fn for_memtable(expected_items: usize) -> Box<dyn BloomFilterStrategy> {
        let config = BloomFilterConfig::for_memtable(expected_items);
        Self::create(&config)
    }

    /// Create from serialized data
    pub fn from_serialized(
        serialized: &SerializedBloomFilter,
    ) -> Result<Box<dyn BloomFilterStrategy>> {
        if serialized.version != SerializedBloomFilter::CURRENT_VERSION {
            return Err(anyhow::anyhow!(
                "Unsupported bloom filter version: {}",
                serialized.version
            ));
        }

        Self::deserialize_strategy(serialized.strategy_type, &serialized.data)
    }

    /// Create a thread-safe bloom filter
    pub fn create_concurrent(config: &BloomFilterConfig) -> Arc<dyn BloomFilterStrategy> {
        Arc::from(Self::create(config))
    }

    /// Deserialize bloom filter from raw bytes using config to determine strategy
    pub fn deserialize(
        config: &BloomFilterConfig,
        data: &[u8],
    ) -> Result<Box<dyn BloomFilterStrategy>> {
        if data.is_empty() {
            return Ok(Box::new(NoOpBloomFilter));
        }

        Self::deserialize_strategy(config.strategy, data)
    }

    fn deserialize_strategy(
        strategy: BloomStrategy,
        data: &[u8],
    ) -> Result<Box<dyn BloomFilterStrategy>> {
        let filter: Box<dyn BloomFilterStrategy> = match strategy {
            BloomStrategy::BitPacked => Box::new(BitPackedBloomFilter::from_bytes(data)?),
            BloomStrategy::ByteAligned => Box::new(ByteAlignedBloomFilter::from_bytes(data)?),
            BloomStrategy::Simple => Box::new(SimpleBloomFilter::from_bytes(data)?),
            BloomStrategy::Composite => Box::new(CompositeBloomFilter::from_bytes(data)?),
        };

        Ok(filter)
    }
}

/// No-op bloom filter for when bloom filters are disabled
#[derive(Debug)]
struct NoOpBloomFilter;

impl BloomFilterStrategy for NoOpBloomFilter {
    fn insert(&mut self, _key: &[u8]) {}

    fn might_contain(&self, _key: &[u8]) -> bool {
        true // Always return true to avoid false negatives
    }

    fn bit_count(&self) -> usize {
        0
    }

    fn hash_count(&self) -> usize {
        0
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        Ok(vec![])
    }

    fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>()
    }

    fn clear(&mut self) {}

    fn false_positive_rate(&self) -> f64 {
        1.0
    }

    fn num_elements(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetadataBloomFilter;

    const STRATEGIES: [(BloomStrategy, &str); 4] = [
        (BloomStrategy::BitPacked, "BitPackedBloomFilter"),
        (BloomStrategy::ByteAligned, "ByteAlignedBloomFilter"),
        (BloomStrategy::Simple, "SimpleBloomFilter"),
        (BloomStrategy::Composite, "CompositeBloomFilter"),
    ];

    fn strategy_fixture(strategy: BloomStrategy) -> Box<dyn BloomFilterStrategy> {
        let config = BloomFilterConfig {
            strategy,
            expected_items: 8,
            bits_per_key: 8,
            ..Default::default()
        };
        // Construct concrete implementations: create() always chooses ByteAligned.
        let mut filter: Box<dyn BloomFilterStrategy> = match strategy {
            BloomStrategy::BitPacked => Box::new(BitPackedBloomFilter::new(8, &config)),
            BloomStrategy::ByteAligned => Box::new(ByteAlignedBloomFilter::new(8, &config)),
            BloomStrategy::Simple => Box::new(SimpleBloomFilter::new(64, &config)),
            BloomStrategy::Composite => {
                let mut composite = CompositeBloomFilter::new(8, &config);
                composite.insert_metadata(
                    "kind",
                    &crate::json_to_metadata_item("kind", &serde_json::json!("fixture")),
                );
                Box::new(composite)
            }
        };
        filter.insert(b"first");
        filter.insert(b"second");
        filter
    }

    fn serialized_fixture(strategy: BloomStrategy, data: Vec<u8>) -> SerializedBloomFilter {
        SerializedBloomFilter {
            strategy_type: strategy,
            version: SerializedBloomFilter::CURRENT_VERSION,
            // Deliberately disagree with the type tag and ignore creation settings.
            config: BloomFilterConfig {
                strategy: if strategy == BloomStrategy::Simple {
                    BloomStrategy::Composite
                } else {
                    BloomStrategy::Simple
                },
                enabled: false,
                expected_items: 0,
                bits_per_key: 1,
                ..Default::default()
            },
            data,
            metadata: Default::default(),
        }
    }

    #[test]
    fn deserialization_entrypoints_preserve_all_strategy_payloads() -> Result<()> {
        for (strategy, _) in STRATEGIES {
            let original = strategy_fixture(strategy);
            let bytes = original.serialize()?;
            let serialized = serialized_fixture(strategy, bytes.clone());
            let config = BloomFilterConfig {
                strategy,
                ..serialized.config.clone()
            };

            for restored in [
                BloomFilterFactory::from_serialized(&serialized)?,
                BloomFilterFactory::deserialize(&config, &bytes)?,
            ] {
                assert_eq!(restored.serialize()?, bytes, "{strategy:?}");
                assert_eq!(restored.bit_count(), original.bit_count(), "{strategy:?}");
                assert_eq!(restored.hash_count(), original.hash_count(), "{strategy:?}");
                assert_eq!(restored.num_elements(), 2, "{strategy:?}");
                assert_eq!(
                    restored.false_positive_rate(),
                    original.false_positive_rate()
                );
                assert!(restored.might_contain(b"first"), "{strategy:?}");
                assert!(restored.might_contain(b"second"), "{strategy:?}");
            }
        }
        Ok(())
    }

    #[test]
    fn deserialization_entrypoints_preserve_malformed_data_errors() {
        for (strategy, name) in STRATEGIES {
            let serialized = serialized_fixture(strategy, vec![0]);
            let config = BloomFilterConfig {
                strategy,
                ..serialized.config.clone()
            };
            let expected =
                format!("Failed to deserialize {name}: io error: unexpected end of file");
            assert_eq!(
                BloomFilterFactory::from_serialized(&serialized)
                    .unwrap_err()
                    .to_string(),
                expected
            );
            assert_eq!(
                BloomFilterFactory::deserialize(&config, &serialized.data)
                    .unwrap_err()
                    .to_string(),
                expected
            );
        }
    }

    #[test]
    fn serialized_version_rejection_precedes_payload_decoding() -> Result<()> {
        for (strategy, _) in STRATEGIES {
            for data in [vec![], vec![0], strategy_fixture(strategy).serialize()?] {
                for version in [0, SerializedBloomFilter::CURRENT_VERSION + 1] {
                    let mut serialized = serialized_fixture(strategy, data.clone());
                    serialized.version = version;
                    assert_eq!(
                        BloomFilterFactory::from_serialized(&serialized)
                            .unwrap_err()
                            .to_string(),
                        format!("Unsupported bloom filter version: {version}")
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn empty_raw_data_is_noop_but_empty_serialized_payload_is_rejected() -> Result<()> {
        for (strategy, name) in STRATEGIES {
            let serialized = serialized_fixture(strategy, vec![]);
            let config = BloomFilterConfig {
                strategy,
                ..serialized.config.clone()
            };
            let mut raw = BloomFilterFactory::deserialize(&config, &[])?;
            raw.insert(b"first");
            raw.clear();
            assert!(raw.might_contain(b"any key"));
            assert_eq!(raw.bit_count(), 0);
            assert_eq!(raw.hash_count(), 0);
            assert_eq!(raw.num_elements(), 0);
            assert_eq!(raw.false_positive_rate(), 1.0);
            assert!(raw.serialize()?.is_empty());
            assert_eq!(
                BloomFilterFactory::from_serialized(&serialized)
                    .unwrap_err()
                    .to_string(),
                format!("Failed to deserialize {name}: io error: unexpected end of file")
            );
        }
        Ok(())
    }

    #[test]
    fn test_factory_creation() {
        // Test each strategy
        let strategies = vec![
            BloomStrategy::BitPacked,
            BloomStrategy::ByteAligned,
            BloomStrategy::Simple,
            BloomStrategy::Composite,
        ];

        for strategy in strategies {
            let config = BloomFilterConfig {
                strategy,
                expected_items: 100,
                ..Default::default()
            };

            let filter = BloomFilterFactory::create(&config);
            assert_eq!(filter.num_elements(), 0);
        }
    }

    #[test]
    fn test_specialized_factories() {
        let sstable_filter = BloomFilterFactory::for_sstable(1000);
        assert!(sstable_filter.bit_count() > 0);

        let memtable_filter = BloomFilterFactory::for_memtable(500);
        assert!(memtable_filter.bit_count() > 0);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let config = BloomFilterConfig::default();
        let mut filter = BloomFilterFactory::create(&config);

        // Add some data
        filter.insert(b"test1");
        filter.insert(b"test2");

        // Serialize
        let serialized =
            SerializedBloomFilter::from_filter(filter.as_ref(), config.clone()).unwrap();

        // Deserialize
        let restored = BloomFilterFactory::from_serialized(&serialized).unwrap();

        // Verify
        assert!(restored.might_contain(b"test1"));
        assert!(restored.might_contain(b"test2"));
        assert!(!restored.might_contain(b"test3"));
    }
}
