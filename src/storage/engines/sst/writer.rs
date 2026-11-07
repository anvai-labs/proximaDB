#![allow(dead_code)]
/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! SSTable Writer with Bloom Filter and Atomic Write Support
//!
//! Creates optimized SSTable files with bloom filters, indexes, and block-based storage.
//! Uses unified atomic write strategies for cross-cloud compatibility.
//!
//! PROXIMA ENCODING INTEGRATION:
//! ================================
//! This writer intelligently chooses encoding schemes per ProximaDataBlock based on data analysis:
//!
//! 1. ENCODING DECISION PROCESS:
//!    - Analyze vector statistics (range, deltas, patterns)
//!    - Choose optimal encoding (BitPacked, Delta, FrameOfReference, etc.)
//!    - Transpose vectors for columnar encoding within blocks
//!    - Write encoding marker as first byte of block
//!
//! 2. VECTOR ENCODING STRATEGY:
//!    a) Collect block of vectors (typically 500-1000)
//!    b) Transpose to columnar layout: vectors[N][D] → columns[D][N]
//!    c) Analyze each dimension column independently
//!    d) Apply dimension-specific encoding:
//!       - Constant dimensions → Run-length encoding
//!       - Small range → Frame of Reference
//!       - Sequential → Delta encoding
//!       - General → BitPacking
//!
//! 3. ENCODING MARKERS (1 byte):
//!    0x00: Raw/Uncompressed (backward compatible)
//!    0x10: Proxima BitPacked
//!    0x20: Proxima Delta
//!    0x30: Proxima FrameOfReference
//!    0x40: Proxima PatchedBase (for outliers)
//!    0x50: Proxima Dictionary
//!    0x60: Proxima RunLength
//!    0xF0-0xFF: Reserved for future encodings
//!
//! 4. METADATA ENCODING:
//!    - Timestamps: Always delta encoded
//!    - IDs: Dictionary encoded if repetitive
//!    - Versions: BitPacked (small range)
//!
//! 5. ADAPTIVE STRATEGY:
//!    - Monitor encoding effectiveness
//!    - Fall back to raw if encoding increases size
//!    - Track statistics for future optimization

use anyhow::Result;
use std::path::Path;
use std::sync::Arc;

use crate::storage::persistence::filesystem::FilesystemFactory;

use crate::core::bloom::{BloomFilterConfig, HashAlgorithm};
use crate::storage::engines::core::formats::proximablocks::ProximaBlockMetadata;

// UnifiedProximaSIMD functionality is now in ProximaCodec

/// ✅ SST-specific metadata using Proxima composition pattern (like HELIX)
/// This follows the same pattern as HelixBlockMetadata but for SST engine optimizations
#[derive(Debug, Clone)]
pub struct SstBlockMetadata {
    /// ✅ Base Proxima metadata - REUSE all auto-generated features!
    /// This includes: bloom filters, metadata statistics, range tracking, delete detection,
    /// SIMD encoding, compression, and all other automatic capabilities
    pub proxima_metadata: ProximaBlockMetadata,

    /// ✅ SST-specific additions only
    pub sst_specific_data: SstSpecificData,
}

/// SST engine-specific optimizations that complement Proxima capabilities
#[derive(Debug, Clone)]
pub struct SstSpecificData {
    /// Three-stage filtering support (Bloom → Quantized → Full precision)
    pub three_stage_filtering: bool,
    /// Row-based storage optimization
    pub row_based_optimization: bool,
    /// Real-time query support
    pub real_time_query_support: bool,
}
// Using unified quantization engine directly from compute module
// use crate::core::bloom::{
//     BloomFilterConfig, BloomStrategy, BloomFilterStrategy, HashAlgorithm,
//     factory::BloomFilterFactory,
// };
// ✅ REMOVED: CompositeBloomFilterBuilder - Proxima provides bloom filters automatically
use crate::proto::proximadb_v1::CompressionConfig;

use proximadb_compression::StandardCompression;

/// Proxima encoding markers as constants
mod encoding_markers {
    #[allow(dead_code)]
    pub const RAW: u8 = 0x00; // Raw/Uncompressed
    #[allow(dead_code)]
    pub const BITPACKED: u8 = 0x10; // Proxima BitPacked
    #[allow(dead_code)]
    pub const DELTA: u8 = 0x20; // Proxima Delta encoding
    #[allow(dead_code)]
    pub const FRAME_OF_REF: u8 = 0x30; // Proxima FrameOfReference
    #[allow(dead_code)]
    pub const PATCHED_BASE: u8 = 0x40; // Proxima PatchedBase
    #[allow(dead_code)]
    pub const DICTIONARY: u8 = 0x50; // Proxima Dictionary
    #[allow(dead_code)]
    pub const RUN_LENGTH: u8 = 0x60; // Proxima RunLength
}

/// Cache-line alignment constant (64 bytes)
///
/// ## Why 64-byte alignment?
/// - Modern CPUs use 64-byte cache lines (Intel/AMD/ARM)
/// - SIMD operations (AVX2/AVX-512/NEON) work best on aligned data
/// - Memory-mapped reads can be used directly without copying to aligned buffers
///
/// ## Overhead Analysis (Audited December 2024)
/// - Typical 263KB block with 51 bytes padding = 0.019% overhead
/// - This negligible overhead enables direct SIMD operations on mmap'd data
///
/// ## Wire Format
/// Each block is written as: [length:4][data:N][padding:0-63]
/// Reader must skip padding after each block (see sst_query_engine.rs:3803-3810)
const CACHE_LINE_SIZE: usize = 64;

/// Block index entry for random access
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct BlockIndexEntry {
    block_id: u32,
    offset: u64,
    size: u32,
    record_count: u32,
    min_id: String,
    max_id: String,
}

/// SSTable writer with atomic write optimization and quantization support
/// MIGRATED: Now uses universal adapters to eliminate code duplication
pub struct SstableWriter {
    /// Output file path
    path: std::path::PathBuf,
    /// Block size for data organization
    block_size: usize,
    /// Bloom filter configuration
    bloom_config: BloomFilterConfig,
    /// Filesystem factory for atomic writes
    filesystem: Arc<FilesystemFactory>,
    /// Direct compression provider (no adapter indirection)
    #[allow(dead_code)]
    compression_provider: StandardCompression,
    /// Compression configuration from flush parameters
    compression_config: Option<CompressionConfig>,
    // SIMD encoding now handled internally by ProximaDataBlock
}

impl SstableWriter {
    /// Create a new SSTable writer with collection-specific configuration
    pub fn new_with_config<P: AsRef<Path>>(
        path: P,
        block_size: usize,
        filesystem: Arc<FilesystemFactory>,
        _collection_config: Option<&crate::proto::proximadb_v1::Collection>,
    ) -> Self {
        // Initialize compression provider directly
        let compression_provider = StandardCompression;

        // Compression configuration can be added here if needed
        // SIMD encoding now handled internally by ProximaDataBlock

        Self {
            path: path.as_ref().to_path_buf(),
            block_size,
            bloom_config: BloomFilterConfig {
                strategy: crate::core::bloom::BloomStrategy::ByteAligned,
                expected_items: 10000,
                false_positive_rate: Some(0.01),
                bits_per_key: 8,
                enabled: true,
                hash_algorithm: HashAlgorithm::Murmur3,
            },
            filesystem,
            compression_provider,
            compression_config: None,
        }
    }

    /// Create a new SSTable writer with filesystem support for atomic writes
    /// Quantization is enabled by default as it's part of the SST file layout
    pub fn new<P: AsRef<Path>>(
        path: P,
        block_size: usize,
        filesystem: Arc<FilesystemFactory>,
    ) -> Self {
        Self::new_with_config(path, block_size, filesystem, None)
    }

    /// MIGRATION: Create SSTable writer with universal adapters
    /// Both compression and quantization use universal adapters for code deduplication
    pub fn with_compression<P: AsRef<Path>>(
        path: P,
        block_size: usize,
        filesystem: Arc<FilesystemFactory>,
        compression_config: Option<CompressionConfig>,
    ) -> Self {
        // Initialize universal compression adapter
        let compression_provider = StandardCompression;

        Self {
            path: path.as_ref().to_path_buf(),
            block_size,
            bloom_config: BloomFilterConfig::default(),
            filesystem,
            compression_provider,
            compression_config,
        }
    }

    // Removed with_quantization() method - quantization is ALWAYS enabled
    // as it's integral to the SST file layout and provides PQ sorting for
    // better compression and selectivity

    /// Set bloom filter configuration
    pub fn with_bloom_config(mut self, config: BloomFilterConfig) -> Self {
        self.bloom_config = config;
        self
    }

    /// Set compression configuration (SDK-driven)
    pub fn with_compression_config(mut self, config: Option<CompressionConfig>) -> Self {
        // Update compression configuration (stored in compression_config field)
        self.compression_config = config;
        self
    }

    /// Write `records` as a native PAX segment (ADR-049 M1-2). ProximaRecord-native:
    /// no ProximaRecord→VectorRecord→streaming round-trip — records go straight to
    /// the PAX encoder. PAX encodes via `std::fs` to a local staging file, then we
    /// promote to the target URL through the filesystem abstraction (handles
    /// `file://` + object store), mirroring the flush PaxBlock arm. Replaces the
    /// wrong-way `write_sorted_proxima_records` adapter at the storage/index/tier
    /// call sites (AXIS stores, tier data movement).
    pub async fn write_pax_segment<I>(&self, records: I, collection_id: &str) -> Result<()>
    where
        I: Iterator<Item = proximadb_records::ProximaRecord> + Send,
    {
        use crate::storage::engines::sst::segment_format::write_pax_segment_with_f32_tier;
        use proximadb_block_format::VectorQuant;

        let recs: Vec<proximadb_records::ProximaRecord> = records.collect();
        if recs.is_empty() {
            return Ok(());
        }
        let embedding_count = recs.first().map(|r| r.embeddings.len().max(1)).unwrap_or(1);

        // Resolve the target URL (bare path → file://) so the filesystem
        // abstraction handles both local and object-store destinations.
        let path_str = self.path.to_string_lossy().to_string();
        let fs_url = if path_str.contains("://") {
            path_str
        } else {
            format!("file://{}", path_str)
        };
        // PAX writes via std::fs to a local staging file, then we promote.
        let staging = std::env::temp_dir().join(format!(
            "proximadb-pax-{}.pax",
            proximadb_kernel::uuid::Uuid::new_v4()
        ));
        write_pax_segment_with_f32_tier(
            &staging,
            &recs,
            collection_id,
            embedding_count,
            VectorQuant::RaBitQ,
            false,
            None,
            Some(fs_url.as_str()),
        )?;
        let bytes = std::fs::read(&staging)
            .map_err(|e| anyhow::anyhow!("read PAX staging {}: {e}", staging.display()))?;
        let _ = std::fs::remove_file(&staging);
        self.filesystem
            .write(&fs_url, &bytes, None)
            .await
            .map_err(|e| anyhow::anyhow!("promote PAX segment to {fs_url}: {e}"))?;
        Ok(())
    }

    // (M1-3 9b: streaming writer tree + its dead helpers removed; PAX bridge is the
    //  sole write path. ProximaDataBlock handles SIMD encoding internally — use
    //  ProximaDataBlock::new_with_engine_profile(records, config, EngineProfile::SST).)
}

#[cfg(test)]
#[cfg_attr(test, path = "writer_tests.rs")]
mod tests;
