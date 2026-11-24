/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! Engine Capabilities Module
//!
//! Centralized module for checking storage engine capabilities and feature support.
//! This module provides static methods to check what features each engine supports
//! without needing to instantiate engine instances.
//!
//! ## Architecture Note
//!
//! This module provides a static API that delegates to the trait-based capability
//! system in `trait_components::capabilities`. This approach:
//! - Maintains backward compatibility with existing static API consumers
//! - Uses the trait-based system as the single source of truth (OCP compliant)
//! - Avoids duplication of capability definitions

use crate::proto::proximadb_v1::CompressionAlgorithm;
use crate::storage::trait_components::capabilities::CapabilityFactory;
use std::collections::HashSet;

// Re-export StorageEngine for external use
pub use crate::proto::proximadb_v1::StorageEngine;

/// Engine capabilities checker - provides static methods for feature support queries
pub struct EngineCapabilities;

impl EngineCapabilities {
    /// Check if a compression algorithm is supported by a given storage engine
    ///
    /// Delegates to the trait-based capability system for OCP compliance.
    pub fn is_compression_supported(
        engine: StorageEngine,
        algorithm: CompressionAlgorithm,
    ) -> bool {
        let caps = CapabilityFactory::from_proto_engine(engine);
        caps.is_compression_supported(algorithm)
    }

    /// Get all supported compression algorithms for a storage engine
    ///
    /// Delegates to the trait-based capability system for OCP compliance.
    pub fn get_supported_compression_algorithms(
        engine: StorageEngine,
    ) -> HashSet<CompressionAlgorithm> {
        let caps = CapabilityFactory::from_proto_engine(engine);
        caps.supported_compression()
    }

    /// Get unsupported compression algorithms for a storage engine
    pub fn get_unsupported_compression_algorithms(
        engine: StorageEngine,
    ) -> Vec<CompressionAlgorithm> {
        let supported = Self::get_supported_compression_algorithms(engine);
        let mut unsupported = Vec::new();

        // Check all possible algorithms
        let all_algorithms = vec![
            CompressionAlgorithm::CompressionNone,
            CompressionAlgorithm::CompressionZstd,
            CompressionAlgorithm::CompressionLz4,
            CompressionAlgorithm::CompressionSnappy,
            CompressionAlgorithm::CompressionGzip,
            CompressionAlgorithm::CompressionBrotli,
            CompressionAlgorithm::CompressionBzip2,
            CompressionAlgorithm::CompressionDeflate,
            CompressionAlgorithm::CompressionXz,
            CompressionAlgorithm::CompressionZlib,
            CompressionAlgorithm::CompressionLzo,
            CompressionAlgorithm::CompressionLz4hc,
            CompressionAlgorithm::CompressionLzma,
        ];

        for algo in all_algorithms {
            if !supported.contains(&algo) {
                unsupported.push(algo);
            }
        }

        unsupported
    }

    /// Get recommended compression algorithm for a use case
    pub fn get_recommended_compression(
        engine: StorageEngine,
        priority: CompressionPriority,
    ) -> CompressionAlgorithm {
        match (engine, priority) {
            // SST engine recommendations
            (StorageEngine::Sst, CompressionPriority::Speed) => {
                CompressionAlgorithm::CompressionLz4
            }
            (StorageEngine::Sst, CompressionPriority::Balanced) => {
                CompressionAlgorithm::CompressionZstd
            }
            (StorageEngine::Sst, CompressionPriority::Ratio) => {
                CompressionAlgorithm::CompressionBrotli
            }

            // HELIX and SWIFT engine recommendations (SST-based, use LZ4 for speed)
            (StorageEngine::Helix, CompressionPriority::Speed)
            | (StorageEngine::Swift, CompressionPriority::Speed) => {
                CompressionAlgorithm::CompressionLz4
            }
            (StorageEngine::Helix, CompressionPriority::Balanced)
            | (StorageEngine::Swift, CompressionPriority::Balanced) => {
                CompressionAlgorithm::CompressionZstd
            }
            (StorageEngine::Helix, CompressionPriority::Ratio)
            | (StorageEngine::Swift, CompressionPriority::Ratio) => {
                CompressionAlgorithm::CompressionBrotli
            }

            // NOVA and RAPTOR engine recommendations (columnar-based, use ZSTD)
            (StorageEngine::Nova, CompressionPriority::Speed)
            | (StorageEngine::Raptor, CompressionPriority::Speed) => {
                CompressionAlgorithm::CompressionSnappy
            }
            (StorageEngine::Nova, CompressionPriority::Balanced)
            | (StorageEngine::Raptor, CompressionPriority::Balanced) => {
                CompressionAlgorithm::CompressionZstd
            }
            (StorageEngine::Nova, CompressionPriority::Ratio)
            | (StorageEngine::Raptor, CompressionPriority::Ratio) => {
                CompressionAlgorithm::CompressionBrotli
            }

            // TST recommendations prioritize balanced archival efficiency.
            (StorageEngine::Tst, CompressionPriority::Speed) => {
                CompressionAlgorithm::CompressionLz4
            }
            (StorageEngine::Tst, CompressionPriority::Balanced) => {
                CompressionAlgorithm::CompressionZstd
            }
            (StorageEngine::Tst, CompressionPriority::Ratio) => {
                CompressionAlgorithm::CompressionBrotli
            }

            // Default
            _ => CompressionAlgorithm::CompressionNone,
        }
    }

    /// Get optimal compression level for an algorithm
    pub fn get_optimal_compression_level(
        algorithm: CompressionAlgorithm,
        priority: CompressionPriority,
    ) -> i32 {
        match (algorithm, priority) {
            // ZSTD levels (1-22)
            (CompressionAlgorithm::CompressionZstd, CompressionPriority::Speed) => 1,
            (CompressionAlgorithm::CompressionZstd, CompressionPriority::Balanced) => 3,
            (CompressionAlgorithm::CompressionZstd, CompressionPriority::Ratio) => 9,

            // LZ4 doesn't have levels in lz4_flex
            (CompressionAlgorithm::CompressionLz4, _) => 0,
            (CompressionAlgorithm::CompressionLz4hc, _) => 9,

            // Gzip/Deflate/Zlib levels (0-9)
            (CompressionAlgorithm::CompressionGzip, CompressionPriority::Speed) => 1,
            (CompressionAlgorithm::CompressionGzip, CompressionPriority::Balanced) => 6,
            (CompressionAlgorithm::CompressionGzip, CompressionPriority::Ratio) => 9,

            (CompressionAlgorithm::CompressionDeflate, CompressionPriority::Speed) => 1,
            (CompressionAlgorithm::CompressionDeflate, CompressionPriority::Balanced) => 6,
            (CompressionAlgorithm::CompressionDeflate, CompressionPriority::Ratio) => 9,

            (CompressionAlgorithm::CompressionZlib, CompressionPriority::Speed) => 1,
            (CompressionAlgorithm::CompressionZlib, CompressionPriority::Balanced) => 6,
            (CompressionAlgorithm::CompressionZlib, CompressionPriority::Ratio) => 9,

            // Brotli levels (0-11)
            (CompressionAlgorithm::CompressionBrotli, CompressionPriority::Speed) => 1,
            (CompressionAlgorithm::CompressionBrotli, CompressionPriority::Balanced) => 4,
            (CompressionAlgorithm::CompressionBrotli, CompressionPriority::Ratio) => 11,

            // Bzip2 levels (1-9)
            (CompressionAlgorithm::CompressionBzip2, CompressionPriority::Speed) => 1,
            (CompressionAlgorithm::CompressionBzip2, CompressionPriority::Balanced) => 5,
            (CompressionAlgorithm::CompressionBzip2, CompressionPriority::Ratio) => 9,

            // XZ/LZMA levels (0-9)
            (CompressionAlgorithm::CompressionXz, CompressionPriority::Speed) => 0,
            (CompressionAlgorithm::CompressionXz, CompressionPriority::Balanced) => 6,
            (CompressionAlgorithm::CompressionXz, CompressionPriority::Ratio) => 9,

            (CompressionAlgorithm::CompressionLzma, CompressionPriority::Speed) => 0,
            (CompressionAlgorithm::CompressionLzma, CompressionPriority::Balanced) => 6,
            (CompressionAlgorithm::CompressionLzma, CompressionPriority::Ratio) => 9,

            // Snappy doesn't have levels
            (CompressionAlgorithm::CompressionSnappy, _) => 0,

            // No compression
            (CompressionAlgorithm::CompressionNone, _) => 0,

            // LZO not supported
            (CompressionAlgorithm::CompressionLzo, _) => 0,
        }
    }

    /// Get the engine name as a string
    pub fn get_engine_name(engine: StorageEngine) -> &'static str {
        match engine {
            StorageEngine::Sst => "SST",
            StorageEngine::Viper => "VIPER (removed, ADR-093)",
            StorageEngine::Helix => "HELIX",
            StorageEngine::Nova => "NOVA",
            StorageEngine::Swift => "SWIFT",
            StorageEngine::Raptor => "RAPTOR",
            StorageEngine::Tst => "TST",
            _ => "Unknown",
        }
    }

    /// Convert integer engine type to StorageEngine enum
    pub fn engine_from_int(engine_type: i32) -> StorageEngine {
        StorageEngine::try_from(engine_type).unwrap_or(StorageEngine::Unspecified)
    }

    // ========================================================================
    // Search Optimization Capabilities (for RL Planner Integration)
    // ========================================================================

    /// Get supported index types for a storage engine
    pub fn get_supported_index_types(engine: StorageEngine) -> Vec<SearchIndexType> {
        match engine {
            StorageEngine::Sst => vec![
                SearchIndexType::Flat,
                SearchIndexType::HNSW,
                SearchIndexType::IVF,
            ],
            StorageEngine::Helix => vec![
                SearchIndexType::Flat,
                SearchIndexType::HNSW,
                SearchIndexType::IVF,
                SearchIndexType::HilbertCurve, // Specialized for HELIX
            ],
            StorageEngine::Swift => vec![
                SearchIndexType::Flat,
                SearchIndexType::HNSW,
                SearchIndexType::AdaCurve, // Learned space-filling curve (uses Hilbert internally)
            ],
            StorageEngine::Nova => vec![
                SearchIndexType::Flat,
                SearchIndexType::IVF,
                SearchIndexType::ZoneMap,
            ],
            StorageEngine::Raptor => vec![
                SearchIndexType::Flat,
                SearchIndexType::IVF,
                SearchIndexType::AdaptiveMatrix,
            ],
            _ => vec![SearchIndexType::Flat],
        }
    }

    /// Get supported quantization levels for a storage engine
    pub fn get_supported_quantization_levels(
        engine: StorageEngine,
    ) -> Vec<SearchQuantizationLevel> {
        match engine {
            StorageEngine::Sst => vec![
                SearchQuantizationLevel::FP32,
                SearchQuantizationLevel::INT8,
                SearchQuantizationLevel::Binary,
            ],
            StorageEngine::Helix => vec![
                SearchQuantizationLevel::FP32,
                SearchQuantizationLevel::INT8,
                SearchQuantizationLevel::Binary,
                SearchQuantizationLevel::PQ8,
            ],
            StorageEngine::Swift => vec![
                SearchQuantizationLevel::FP32,
                SearchQuantizationLevel::INT8,
                SearchQuantizationLevel::Binary,
                SearchQuantizationLevel::PQ4,
                SearchQuantizationLevel::PQ8,
            ],
            StorageEngine::Nova => vec![
                SearchQuantizationLevel::FP32,
                SearchQuantizationLevel::INT8,
                SearchQuantizationLevel::Binary,
            ],
            StorageEngine::Raptor => {
                vec![SearchQuantizationLevel::FP32, SearchQuantizationLevel::INT8]
            }
            _ => vec![SearchQuantizationLevel::FP32],
        }
    }

    /// Get supported pruning strategies for a storage engine
    pub fn get_supported_pruning_strategies(engine: StorageEngine) -> Vec<SearchPruningStrategy> {
        match engine {
            StorageEngine::Sst => vec![
                SearchPruningStrategy::None,
                SearchPruningStrategy::BloomFilter,
                SearchPruningStrategy::BlockCentroid,
            ],
            StorageEngine::Helix => vec![
                SearchPruningStrategy::None,
                SearchPruningStrategy::HilbertRange,
                SearchPruningStrategy::ZoneMap,
                SearchPruningStrategy::BlockCentroid,
            ],
            StorageEngine::Swift => vec![
                SearchPruningStrategy::None,
                SearchPruningStrategy::AdaCurvePruning, // Learned curve-based pruning
                SearchPruningStrategy::BlockCentroid,
                SearchPruningStrategy::SuperblockSignature,
            ],
            StorageEngine::Nova => vec![
                SearchPruningStrategy::None,
                SearchPruningStrategy::ZoneMap,
                SearchPruningStrategy::ColumnStats,
            ],
            StorageEngine::Raptor => vec![
                SearchPruningStrategy::None,
                SearchPruningStrategy::AdaptiveTier,
            ],
            _ => vec![SearchPruningStrategy::None],
        }
    }

    /// Check if engine supports progressive search pipeline
    ///
    /// Delegates to the trait-based capability system for OCP compliance.
    pub fn supports_progressive_search(engine: StorageEngine) -> bool {
        let caps = CapabilityFactory::from_proto_engine(engine);
        caps.supports_progressive_quantization()
    }

    /// Get recommended search configuration for a workload
    pub fn get_search_recommendations(
        engine: StorageEngine,
        collection_size: u64,
        latency_sensitive: bool,
    ) -> SearchRecommendation {
        let use_index = collection_size > 5000;
        let use_progressive = Self::supports_progressive_search(engine) && collection_size > 10000;

        let index_type = if use_index {
            match engine {
                StorageEngine::Helix => SearchIndexType::HilbertCurve,
                StorageEngine::Raptor => SearchIndexType::AdaptiveMatrix,
                _ => SearchIndexType::IVF, // Default to IVF for large collections
            }
        } else {
            SearchIndexType::Flat
        };

        let quantization = if latency_sensitive && collection_size > 10000 {
            SearchQuantizationLevel::INT8
        } else {
            SearchQuantizationLevel::FP32
        };

        let pruning = match engine {
            StorageEngine::Sst if collection_size > 10000 => SearchPruningStrategy::BloomFilter,
            StorageEngine::Helix => SearchPruningStrategy::HilbertRange,
            StorageEngine::Swift => SearchPruningStrategy::AdaCurvePruning, // Learned curve pruning
            StorageEngine::Nova => SearchPruningStrategy::ZoneMap,
            _ => SearchPruningStrategy::None,
        };

        SearchRecommendation {
            index_type,
            quantization,
            pruning,
            use_progressive,
            expected_recall: if use_index { 0.95 } else { 1.0 },
            expected_latency_factor: if use_progressive {
                0.3
            } else if use_index {
                0.5
            } else {
                1.0
            },
        }
    }
}

/// Index types supported by storage engines
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SearchIndexType {
    /// No index, full scan
    Flat,
    /// Hierarchical Navigable Small World graph
    HNSW,
    /// Inverted File index with clustering
    IVF,
    /// Locality Sensitive Hashing
    LSH,
    /// Product Quantization based index
    PQ,
    /// HELIX-specific Hilbert curve ordering
    HilbertCurve,
    /// SWIFT AdaCurve - learned space-filling curve (uses Hilbert internally)
    AdaCurve,
    /// Zone map based pruning (NOVA)
    ZoneMap,
    /// Adaptive matrix structure (RAPTOR)
    AdaptiveMatrix,
}

/// Quantization levels for search
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SearchQuantizationLevel {
    /// Full 32-bit floating point
    FP32,
    /// 8-bit integer quantization
    INT8,
    /// 1-bit binary quantization
    Binary,
    /// 4-bit product quantization
    PQ4,
    /// 8-bit product quantization
    PQ8,
}

/// Pruning strategies for search
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SearchPruningStrategy {
    /// No pruning
    None,
    /// Bloom filter based pruning (SST)
    BloomFilter,
    /// Block centroid distance pruning
    BlockCentroid,
    /// Hilbert range pruning (HELIX)
    HilbertRange,
    /// AdaCurve (learned curve) pruning (SWIFT) - uses Hilbert internally
    AdaCurvePruning,
    /// Zone map pruning (NOVA)
    ZoneMap,
    /// Column statistics pruning (NOVA)
    ColumnStats,
    /// Superblock signature pruning (SWIFT)
    SuperblockSignature,
    /// Adaptive tier pruning (RAPTOR)
    AdaptiveTier,
}

/// Search recommendation from capabilities
#[derive(Debug, Clone)]
pub struct SearchRecommendation {
    /// Recommended index type
    pub index_type: SearchIndexType,
    /// Recommended quantization level
    pub quantization: SearchQuantizationLevel,
    /// Recommended pruning strategy
    pub pruning: SearchPruningStrategy,
    /// Whether to use progressive search
    pub use_progressive: bool,
    /// Expected recall with these settings
    pub expected_recall: f32,
    /// Expected latency factor (1.0 = baseline)
    pub expected_latency_factor: f32,
}

/// Compression priority for choosing algorithms and levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionPriority {
    /// Optimize for compression/decompression speed
    Speed,
    /// Balance between speed and compression ratio
    Balanced,
    /// Optimize for best compression ratio
    Ratio,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sst_compression_support() {
        // SST should support all algorithms except LZO
        assert!(EngineCapabilities::is_compression_supported(
            StorageEngine::Sst,
            CompressionAlgorithm::CompressionZstd
        ));
        assert!(EngineCapabilities::is_compression_supported(
            StorageEngine::Sst,
            CompressionAlgorithm::CompressionBrotli
        ));
        assert!(!EngineCapabilities::is_compression_supported(
            StorageEngine::Sst,
            CompressionAlgorithm::CompressionLzo
        ));
    }

    #[test]
    fn test_compression_recommendations() {
        // Test speed optimized recommendations
        assert_eq!(
            EngineCapabilities::get_recommended_compression(
                StorageEngine::Sst,
                CompressionPriority::Speed
            ),
            CompressionAlgorithm::CompressionLz4
        );

        // Test ratio optimized recommendations
        assert_eq!(
            EngineCapabilities::get_recommended_compression(
                StorageEngine::Sst,
                CompressionPriority::Ratio
            ),
            CompressionAlgorithm::CompressionBrotli
        );
    }

    #[test]
    fn test_compression_levels() {
        // Test ZSTD levels
        assert_eq!(
            EngineCapabilities::get_optimal_compression_level(
                CompressionAlgorithm::CompressionZstd,
                CompressionPriority::Speed
            ),
            1
        );
        assert_eq!(
            EngineCapabilities::get_optimal_compression_level(
                CompressionAlgorithm::CompressionZstd,
                CompressionPriority::Ratio
            ),
            9
        );

        // Test Brotli levels
        assert_eq!(
            EngineCapabilities::get_optimal_compression_level(
                CompressionAlgorithm::CompressionBrotli,
                CompressionPriority::Ratio
            ),
            11
        );
    }

}
