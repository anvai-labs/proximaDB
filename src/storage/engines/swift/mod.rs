//! # SWIFT Engine - ⚠️ DEPRECATED - INCOMPLETE
//!
//! **WARNING**: This engine is DEPRECATED and INCOMPLETE.
//!
//! ## Deprecation Notice (2026-04-03)
//!
//! SWIFT is deprecated and will be removed in v1.0 unless contributors complete the
//! remaining implementation. See `/docs/storage/EXPERIMENTAL_ENGINES_STATUS.md` for details.
//!
//! ## Current Status
//!
//! - **Status**: INCOMPLETE - Not Production Ready
//! - **Feature Flag**: `experimental-engines` required
//! - **Tests**: 41 tests (insufficient coverage)
//! - **DEFERRED Items**: 30+ critical features missing
//! - **Recommendation**: Use SST, VIPER, HELIX, or NOVA instead
//!
//! ## Migration Guide
//!
//! For hierarchical storage needs, use application-level hierarchy with production engines:
//! - **SST**: For range queries and efficient access patterns
//! - **NOVA**: For columnar analytics with zone maps
//! - **Multiple Collections**: Use collection naming for hierarchy
//!
//! Example:
//! ```rust,ignore
//! // Instead of SWIFT hierarchy:
//! // swift_engine.create_department_hierarchy(&org_structure).await?;
//!
//! // Use multiple SST collections:
//! let marketing = sst_engine.create_collection("tenant_marketing").await?;
//! let sales = sst_engine.create_collection("tenant_sales").await?;
//! ```
//!
//! ## SWIFT Engine: Storage With Indexed Fast Traversal
//!
//! SWIFT is ProximaDB's hierarchical storage engine featuring a three-tier architecture for organized data management.
//!
//! ### Hierarchical Capabilities (incomplete):
//! 1. **Three-Tier Architecture**: SuperBlock, DataBlock, Records hierarchy
//! 2. **Hierarchical Indexing**: Multi-level navigation with O(log n) access patterns
//! 3. **Large-Scale Support**: Designed for datasets from millions to billions of vectors
//! 4. **Proxima Integration**: SIMD-optimized encoding with compression
//! 5. **Incremental Operations**: Non-disruptive updates and expansions
//!
//! **STATUS**: INCOMPLETE - Not recommended for production use
//!
//! ## 🎯 OPTIMAL USE CASES
//!
//! SWIFT excels in scenarios requiring hierarchical organization and large-scale management:
//!
//! ### ✅ **Enterprise Content Management Systems**
//! ```rust,ignore
//! // Digital asset libraries with departmental organization
//! let media_vectors = load_enterprise_assets(); // 50M+ digital assets
//! swift_engine.create_department_hierarchy(&org_structure).await; // Department SuperBlocks
//! swift_engine.flush_with_hierarchy(media_vectors, &department_mapping).await;
//! let department_search = swift_engine.search_within_hierarchy(
//!     "marketing",
//!     brand_query,
//!     100
//! ).await; // Fast departmental isolation
//! ```
//!
//! ### ✅ **Multi-Tenant SaaS Platforms**
//! ```rust,ignore
//! // Complete tenant isolation with hierarchical storage
//! for tenant_batch in enterprise_tenants {
//!     swift_engine.create_tenant_superblock(
//!         &tenant_batch.tenant_id,
//!         &tenant_batch.storage_tier,
//!         tenant_batch.vectors
//!     ).await; // Isolated SuperBlocks per tenant
//! }
//! let tenant_results = swift_engine.search_tenant_isolated(
//!     &tenant_id,
//!     query,
//!     TenantSearchConfig::new().respect_quotas(true)
//! ).await; // Guaranteed isolation and resource limits
//! ```
//!
//! ### ✅ **Version-Controlled Document Systems**
//! ```rust,ignore
//! // Document versioning with efficient historical access
//! for document_version in document_history {
//!     swift_engine.append_document_version(
//!         &document_version.document_id,
//!         &document_version.version,
//!         document_version.embedding
//!     ).await; // Hierarchical version storage
//! }
//! let version_timeline = swift_engine.get_document_evolution(
//!     &document_id,
//!     DateRange::last_year()
//! ).await; // Efficient temporal queries
//! ```
//!
//! ### ✅ **Geospatial Data Organization**
//! ```rust,ignore
//! // Geographic hierarchy for location-based services
//! let location_vectors = load_poi_embeddings(); // Points of interest
//! swift_engine.create_geographic_hierarchy(
//!     GeographicConfig::new()
//!         .country_level() // Top-level SuperBlocks
//!         .state_level()   // Mid-level DataBlocks
//!         .city_level()    // Record-level organization
//! ).await;
//! let nearby_pois = swift_engine.search_geographic(
//!     user_location,
//!     radius_km,
//!     GeoSearchConfig::new().respect_hierarchy(true)
//! ).await; // Hierarchical geographic pruning
//! ```
//!
//! ## 🏗️ **THREE-TIER ARCHITECTURE OVERVIEW**
//!
//! ### **SuperBlock Level (Top Tier)**
//! - **Purpose**: High-level organizational units (departments, tenants, regions)
//! - **Optimization**: Coarse-grained filtering and resource isolation
//! - **Benefit**: Efficient top-level pruning and access control
//!
//! ### **DataBlock Level (Mid Tier)**
//! - **Purpose**: Fine-grained organization within SuperBlocks (projects, versions, categories)
//! - **Optimization**: Mid-level indexing with Proxima compression
//! - **Benefit**: Balanced granularity for most query patterns
//!
//! ### **Record Level (Bottom Tier)**
//! - **Purpose**: Individual vector storage with full metadata
//! - **Optimization**: SIMD-optimized encoding and compression
//! - **Benefit**: Maximum storage efficiency and fast access
//!
//! ## ❌ **NOT OPTIMAL FOR:**
//!
//! - **Flat Data Structures**: SST or VIPER better for non-hierarchical data
//! - **Simple Use Cases**: Overhead not justified for basic vector storage
//! - **Real-Time Analytics**: NOVA better for complex analytical workloads
//! - **Memory-Constrained Systems**: Three-tier overhead requires adequate RAM
//!
//! ## 📊 PERFORMANCE CHARACTERISTICS
//!
//! - **Query Performance**: Excellent (hierarchical pruning reduces search space)
//! - **Write Performance**: Good (batch operations at SuperBlock level)
//! - **Storage Efficiency**: Good (Proxima compression + hierarchical organization)
//! - **Memory Usage**: Moderate (intelligent tier-based loading)
//! - **Scalability**: Outstanding (linear scaling with proper hierarchy design)
//!
//! ## How SWIFT Leverages Common Modules
//!
//! ### 1. Row-Based Module Integration (`proximablocks::`)
//! - **Hierarchical Blocks**: Uses `SuperBlock` and `DataBlock` from proximablocks for
//!   its unique three-tier hierarchy (SuperBlock → DataBlock → Records)
//! - **Index Structures**: Leverages `HierarchicalIndex` and `MultiLevelIndex` for
//!   efficient navigation of its deep block structure
//! - **Batch Operations**: Uses `RowBasedBatchOperations` with custom strategies
//!   optimized for hierarchical traversal
//! - **Compression**: Shared compression infrastructure with per-superblock tuning
//!
//! ### 2. SST Module Synergy (`sst::`)
//! - **Bloom Filters**: Reuses SST's `SstableBloomFilter` implementation
//! - **Compaction Logic**: Shares compaction strategies with SST
//! - **Reader Patterns**: Borrows efficient reading patterns from SST
//!
//! ### 3. Universal Adapter Integration (`universal::`)
//! - **Progressive Search**: Uses universal's Binary → INT8 → PQ → FP32 pipeline
//! - **Distance Computation**: All searches go through UniversalDistanceAdapter
//! - **Format Conversion**: Automatic conversion between hierarchical and flat formats
//! - **Hardware Optimization**: SIMD acceleration via universal adapter
//!
//! ### 4. Compute Module Integration (`compute::`)
//! - **Quantization**: Replaced local quantization_blocks with unified `StorageQuantizationEngine`
//! - **Distance Metrics**: Full suite of 13 metrics from `UnifiedDistanceCompute`
//! - **Memory Management**: Shared `VectorMemoryPool` for buffer reuse
//!
//! ## SWIFT-Specific Optimizations
//! - **Three-Tier Hierarchy**: Unique SuperBlock → DataBlock → Records structure
//! - **Large-Scale Design**: Optimized for datasets with millions to billions of vectors
//! - **Efficient Metadata**: Optimized metadata overhead per vector
//! - **Logarithmic Navigation**: O(log n) navigation through hierarchical indexes

pub mod engine;
pub mod extraction;
pub mod hierarchical_blocks;
pub mod id_index;
pub mod pca_manager; // PCA caching for spatial encoding
// NOTE: quantization_blocks removed - using unified quantization from compute module
pub mod batch_operations;
pub mod optimized_operations;
pub mod progressive_search;
pub mod progressive_stages; // ISP-compliant progressive search stages
pub mod stages;
pub mod superblock_cache;
pub mod unified_metadata_serializer {
    pub use crate::storage::engines::core::proximablocks_compact_serializer::*;
}
pub mod proximablocks_compact_reader;
pub mod proximablocks_compact_strategy_reader;

// Re-export main engine type and cache
#[allow(deprecated)]
pub use engine::SwiftEngine;
pub use proximablocks_compact_strategy_reader::{
    CachedSWIFTReader, DirectSWIFTReader, UnifiedSWIFTReader,
};
pub use superblock_cache::{
    CachedSuperBlockMetadata, OptimalTreePath, SwiftSuperBlockCache, TreeNavigationHints,
};

// Re-export SOLID progressive search stages
pub use stages::{
    SwiftBinaryStage, SwiftFp32Stage, SwiftInt8Stage, SwiftProgressivePipelineBuilder,
};

use anyhow::{Result, anyhow};
// use std::collections::HashMap; // Unused import
use std::sync::Arc;
use tracing::{debug, info};

use crate::proto::proximadb_v1::VectorRecord;
use proximadb_compression::CompressionAlgorithm;
use proximadb_records::ProximaRecord;

// SYNERGY: Reuse row-based bloom filter structures (shared with SST)
// Proxima encoding for columnar vector optimization

// ProximaCodec system for encoding/decoding
use crate::storage::engines::core::formats::proximablocks::engine_profile::EngineProfile;
use crate::storage::engines::core::ops::proximacodec::types::ProximaScheme;
// NOTE: Quantization now uses unified engine from compute module

// Import Proxima common structures (SWIFT uses hierarchical structure)
// Note: ProximaDataBlock provides the block structure with encoding support
use crate::storage::engines::core::formats::proximablocks::block_structures::{
    ProximaBlockMetadata, ProximaDataBlock,
};

/// ✅ SWIFT-specific metadata using Proxima composition pattern (like HELIX and SST)
/// This follows the same pattern as HelixBlockMetadata and SstBlockMetadata but for SWIFT SuperBlock optimizations
#[derive(Debug, Clone)]
pub struct SwiftSuperBlockMetadata {
    /// ✅ Base Proxima metadata - REUSE all auto-generated features!
    /// This includes: bloom filters, metadata statistics, range tracking, delete detection,
    /// SIMD encoding, compression, and all other automatic capabilities
    pub proxima_metadata: ProximaBlockMetadata,

    /// ✅ SWIFT-specific hierarchical additions only
    pub swift_specific_data: SwiftSpecificData,
}

/// SWIFT engine-specific hierarchical optimizations that complement Proxima capabilities
#[derive(Debug, Clone)]
pub struct SwiftSpecificData {
    /// Three-tier hierarchical structure (SuperBlock → DataBlock → Records)
    pub hierarchical_structure: bool,
    /// Large-scale vector optimization (millions to billions)
    pub large_scale_optimization: bool,
    /// Efficient metadata storage design
    pub efficient_metadata_storage: bool,
    /// Optimized hierarchical traversal support
    pub optimized_traversal: bool,
}

/// SWIFT-specific SuperBlock structure - hierarchical container for multiple data blocks
#[derive(Debug)]
pub struct SuperBlock {
    pub superblock_id: usize,
    pub name: String,
    pub blocks: Vec<ProximaDataBlock>,
    pub superblock_encoding_marker: u8,
    pub centroid: Vec<f32>,
    /// FP16 quantized superblock centroid (50% storage reduction, <0.1% error)
    pub centroid_fp16: Option<Vec<u16>>,
    /// Per-block centroids aligned with `blocks`
    pub block_centroids: Vec<Vec<f32>>,
    /// FP16 quantized per-block centroids (50% storage reduction)
    pub block_centroids_fp16: Option<Vec<Vec<u16>>>,
    pub quantized_signature: Vec<u8>,
    /// ✅ Now uses SWIFT composition metadata instead of manual bloom filter
    pub swift_metadata: SwiftSuperBlockMetadata,
    pub record_count: u32, // Track total records in this superblock
    /// AdaCurve code for learned space-filling curve (hierarchical spatial indexing)
    pub adacurve_code: Option<u64>,
}

impl SuperBlock {
    /// ✅ REFACTORED: Create SuperBlock using Proxima composition pattern
    pub fn new(id: usize, name: String) -> Self {
        // ✅ Initialize with Proxima capabilities (will be set when blocks are added)
        let default_proxima_metadata = ProximaBlockMetadata::default();

        let swift_metadata = SwiftSuperBlockMetadata {
            proxima_metadata: default_proxima_metadata,
            swift_specific_data: SwiftSpecificData {
                hierarchical_structure: true,
                large_scale_optimization: true,
                efficient_metadata_storage: true,
                optimized_traversal: true,
            },
        };

        Self {
            superblock_id: id,
            name,
            blocks: Vec::new(),
            superblock_encoding_marker: 0x00,
            centroid: Vec::new(),
            centroid_fp16: None,
            block_centroids: Vec::new(),
            block_centroids_fp16: None,
            quantized_signature: Vec::new(),
            swift_metadata,
            record_count: 0,
            adacurve_code: None, // Will be populated during clustering
        }
    }

    /// ✅ REFACTORED: Add block and aggregate Proxima metadata automatically
    pub fn add_block(&mut self, block: ProximaDataBlock) {
        let prev_count = self.record_count as f32;
        let block_count = block.metadata.record_count as f32;

        // ✅ Update SuperBlock metadata using Proxima auto-generated metadata
        self.record_count += block.metadata.record_count;

        // ✅ Aggregate Proxima metadata from all blocks
        if !self.blocks.is_empty() {
            // Merge column statistics
            for (column, block_stats) in &block.metadata.column_stats {
                if let Some(existing_stats) = self
                    .swift_metadata
                    .proxima_metadata
                    .column_stats
                    .get_mut(column)
                {
                    // Update min/max values
                    if let (Some(block_min), Some(existing_min)) =
                        (&block_stats.min_value, &existing_stats.min_value)
                    {
                        // Use JSON comparison for consistency
                        use std::cmp::Ordering;
                        let cmp = match (block_min, existing_min) {
                            (serde_json::Value::Number(n1), serde_json::Value::Number(n2)) => n1
                                .as_f64()
                                .partial_cmp(&n2.as_f64())
                                .unwrap_or(Ordering::Equal),
                            (serde_json::Value::String(s1), serde_json::Value::String(s2)) => {
                                s1.cmp(s2)
                            }
                            _ => Ordering::Equal,
                        };
                        if cmp == Ordering::Less {
                            existing_stats.min_value = block_stats.min_value.clone();
                        }
                    }
                    // Similar logic for max values
                    existing_stats.null_count += block_stats.null_count;
                } else {
                    // Add new column statistics
                    self.swift_metadata
                        .proxima_metadata
                        .column_stats
                        .insert(column.clone(), block_stats.clone());
                }
            }

            // Update aggregate metadata
            self.swift_metadata.proxima_metadata.record_count += block.metadata.record_count;
            self.swift_metadata.proxima_metadata.size_bytes += block.metadata.size_bytes;
            self.swift_metadata.proxima_metadata.compressed_size += block.metadata.compressed_size;
        } else {
            // First block - initialize metadata
            self.swift_metadata.proxima_metadata = block.metadata.clone();
        }

        // Update centroid with weighted mean across blocks
        let block_centroid = compute_block_centroid(&block);

        // Store FP32 centroid first (before any potential move)
        self.block_centroids.push(block_centroid.clone());

        // NEW: Compute and store FP16 centroid (50% storage reduction)
        if !block_centroid.is_empty() {
            let fp16_centroid = crate::storage::engines::sst::fp32_to_fp16(&block_centroid);
            if let Some(ref mut fp16_vec) = self.block_centroids_fp16 {
                fp16_vec.push(fp16_centroid);
            } else {
                self.block_centroids_fp16 = Some(vec![fp16_centroid]);
            }
        }

        // Update superblock centroid with weighted mean
        if !block_centroid.is_empty() {
            if self.centroid.is_empty() {
                self.centroid = block_centroid;
            } else if self.centroid.len() == block_centroid.len() {
                let total = prev_count + block_count;
                if total > 0.0 {
                    for (i, val) in block_centroid.iter().enumerate() {
                        self.centroid[i] =
                            (self.centroid[i] * prev_count + val * block_count) / total;
                    }
                }
            }
        }

        self.blocks.push(block);
    }
}

/// Placeholder for quantized index - now handled by unified compute module
#[derive(Debug)]
pub struct QuantizedIndex {
    #[allow(dead_code)]
    dimension: usize,
}

impl QuantizedIndex {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

fn compute_block_centroid(block: &ProximaDataBlock) -> Vec<f32> {
    let first = match block.records.first() {
        Some(r) => r
            .embeddings
            .first()
            .map_or(&[][..], |embedding| embedding.as_fp32_slice()),
        None => return Vec::new(),
    };
    let dim = first.len();
    if dim == 0 {
        return Vec::new();
    }
    let mut sum = vec![0f32; dim];
    let mut count = 0f32;
    for record in &block.records {
        let vector = record
            .embeddings
            .first()
            .map_or(&[][..], |embedding| embedding.as_fp32_slice());
        if vector.len() != dim {
            return Vec::new();
        }
        for (i, v) in vector.iter().enumerate() {
            sum[i] += *v;
        }
        count += 1.0;
    }
    if count == 0.0 {
        return Vec::new();
    }
    sum.into_iter().map(|s| s / count).collect()
}

/// SWIFT file structure - hierarchical superblock design
#[derive(Debug)]
pub struct SwiftFile {
    /// File header containing all metadata
    pub header: SwiftHeader,

    /// Three-tier hierarchy for billion-scale vectors
    pub superblocks: Vec<SuperBlock>,

    /// Global indexes for different access patterns
    pub id_index: id_index::IdIndex,
    pub quantized_index: QuantizedIndex,
    pub metadata_index: hierarchical_blocks::MetadataIndex,

    /// Memory management
    #[allow(dead_code)]
    memory_manager: Arc<MemoryManager>,
    // Note: simd_encoder removed - encoding now done via ProximaCodec per-operation
}

/// Magic constant for SWIFT files (4 bytes)
pub const SWIFT_MAGIC: [u8; 4] = *b"SWFT";

/// SWIFT header - all metadata in one place
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SwiftHeader {
    // File identification
    pub magic: [u8; 4],
    pub version: u32,

    // Collection information
    pub collection_id: String,
    pub timestamp: i64,
    pub compaction_level: u8,

    // Vector configuration
    pub dimension: usize,
    pub distance_metric: String,
    pub quantization: QuantizationConfig,

    // Record counts
    pub total_records: u64,
    pub deleted_records: u64,

    // Layout information
    pub superblock_count: u32,
    pub blocks_per_superblock: u32,
    pub records_per_block: u32,

    // Index offsets
    pub superblock_offset: u64,
    pub id_index_offset: u64,
    pub quantized_index_offset: u64,
    pub metadata_index_offset: u64,

    // Checksums
    pub header_checksum: u32,
    pub file_checksum: u64,
}

impl Default for SwiftHeader {
    fn default() -> Self {
        Self {
            magic: SWIFT_MAGIC,
            version: 1,
            collection_id: String::new(),
            timestamp: 0,
            compaction_level: 0,
            dimension: 0,
            distance_metric: "cosine".to_string(), // Deferred: Use proper enum conversion
            quantization: QuantizationConfig::default(),
            total_records: 0,
            deleted_records: 0,
            superblock_count: 0,
            blocks_per_superblock: 10,
            records_per_block: 512, // Reduced from 1000 for better granularity
            superblock_offset: 4096, // After header
            id_index_offset: 0,
            quantized_index_offset: 0,
            metadata_index_offset: 0,
            header_checksum: 0,
            file_checksum: 0,
        }
    }
}

/// DEPRECATED: Being replaced with unified config
/// Use crate::core::unified_config::EngineQuantizationConfig instead
#[derive(Debug, Clone)]
pub struct QuantizationConfigOLD {
    // Binary quantization
    pub enable_binary: bool,
    pub binary_threshold: f32,

    // INT8 quantization
    pub enable_int8: bool,
    pub int8_scale: f32,
    pub int8_zero_point: i8,

    // Product Quantization
    pub enable_pq: bool,
    pub pq_segments: u8,
    pub pq_bits: u8,
    pub pq_codebooks: Vec<Codebook>,

    // Compression
    pub compression_algorithm: CompressionAlgorithm,
    pub compression_level: u8,
}

impl Default for QuantizationConfigOLD {
    fn default() -> Self {
        Self {
            enable_binary: false,
            binary_threshold: 0.5,
            enable_int8: false,
            int8_scale: 1.0,
            int8_zero_point: 0,
            enable_pq: false,
            pq_segments: 8,
            pq_bits: 8,
            pq_codebooks: Vec::new(),
            compression_algorithm: CompressionAlgorithm::None,
            compression_level: 0,
        }
    }
}

/// Use proto-generated config directly - no more duplicates!
pub use crate::proto::proximadb_v1::QuantizationConfig;

/// PQ Codebook
#[derive(Debug, Clone)]
pub struct Codebook {
    pub segment_id: u8,
    pub dimension: usize,
    pub centroids: Vec<Vec<f32>>,
    pub distance_table: Vec<Vec<f32>>,
}

// SuperBlock and DataBlock are now imported from proximablocks common module
// Additional SWIFT-specific fields can be added via composition if needed

/// Backwards-compat alias for [`SwiftColumnStats`].
pub type ColumnStats = SwiftColumnStats;

/// Column statistics for metadata filtering
#[derive(Debug, Clone)]
pub struct SwiftColumnStats {
    pub column_name: String,
    pub null_count: u32,
    pub distinct_count: u32,
    pub min_value: serde_json::Value,
    pub max_value: serde_json::Value,
}

/// Memory manager for efficient resource usage
#[derive(Debug)]
pub struct MemoryManager {
    #[allow(dead_code)]
    max_memory_bytes: usize,
    #[allow(dead_code)]
    current_usage: std::sync::atomic::AtomicUsize,
}

impl SwiftFile {
    /// ✅ REFACTORED: Build blocks using Proxima composition pattern (like HELIX and SST)
    /// Proxima automatically handles quantization, encoding, bloom filters, and metadata statistics!
    pub fn build_blocks_from_records_with_adapters(
        &mut self,
        records: Vec<VectorRecord>,
        _quantization_engine: Option<
            &crate::compute::quantization::storage_engine::StorageQuantizationEngine,
        >,
        _quantization_config: Option<
            &crate::compute::quantization::storage_engine::StorageQuantizationConfig,
        >,
    ) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }

        // Group records into blocks (~2000 vectors per block)
        let records_per_block = self.header.records_per_block as usize;

        let canonical_records: Vec<ProximaRecord> =
            records.iter().map(ProximaRecord::from).collect();

        for (block_id, chunk) in canonical_records.chunks(records_per_block).enumerate() {
            // ✅ Proxima automatically provides:
            // - 🔍 Automatic Bloom Filter Generation
            // - 📊 Automatic Metadata Statistics
            // - 📝 Automatic Range Tracking
            // - 🧠 Automatic Delete Detection
            // - ⚡ Automatic SIMD Encoding
            // - 🗜️ Automatic Compression
            // - 🚀 Automatic Quantization (if enabled)
            // Use centralized compression config conversion from Proxima
            use crate::storage::engines::core::formats::proximablocks::compression_config::RowBasedCompressionConfig;
            let mut compression_config =
                RowBasedCompressionConfig::create_block_config_from_proto(None); // Deferred: Pass actual compression config

            // Enable SIMD optimization for SWIFT (low-latency focus)
            compression_config.vector_layout = crate::storage::engines::core::formats::proximablocks::VectorEncodingLayout::TransposeFieldEncodedAndCompressedVector;

            // Create block with SWIFT engine profile for optimized SIMD encoding
            let mut block = ProximaDataBlock::new_with_engine_profile(
                chunk.to_vec(),
                compression_config,
                EngineProfile::Swift,
            );

            // ✅ Add quantized columns for progressive search (Binary → INT8 → FP32)
            // This enables 10-50x speedup by filtering 95% of candidates with Hamming distance
            use crate::storage::engines::core::formats::proximablocks::block_structures::QuantizedSection;

            let vectors: Vec<Vec<f32>> = chunk
                .iter()
                .filter_map(|r| {
                    r.embeddings
                        .first()
                        .map(|embedding| embedding.values.to_fp32_owned())
                })
                .collect();
            if !vectors.is_empty() && !vectors[0].is_empty() {
                let dimension = vectors[0].len();

                // Compute binary quantization (1-bit per dimension, 32x compression)
                let binary_vectors: Vec<Vec<u8>> = vectors
                    .iter()
                    .map(|v| {
                        let mut binary = vec![0u8; dimension.div_ceil(8)];
                        for (i, &val) in v.iter().enumerate() {
                            if val > 0.0 {
                                binary[i / 8] |= 1 << (i % 8);
                            }
                        }
                        binary
                    })
                    .collect();

                // Compute INT8 quantization (4x compression, ~95% recall)
                let (min_val, max_val) = vectors
                    .iter()
                    .flat_map(|v| v.iter())
                    .fold((f32::MAX, f32::MIN), |(min, max), &val| {
                        (min.min(val), max.max(val))
                    });

                let scale = if (max_val - min_val).abs() > 1e-8 {
                    255.0 / (max_val - min_val)
                } else {
                    1.0
                };

                let int8_vectors: Vec<Vec<i8>> = vectors
                    .iter()
                    .map(|v| {
                        v.iter()
                            .map(|&val| {
                                let normalized = ((val - min_val) * scale).clamp(0.0, 255.0) as u8;
                                (normalized as i16 - 128) as i8
                            })
                            .collect()
                    })
                    .collect();

                block.quantized_section = Some(QuantizedSection {
                    binary_vectors: Some(binary_vectors),
                    int8_vectors: Some(int8_vectors),
                    pq_vectors: None,
                    codebooks: None,
                });

                block.metadata.quantization_stats.has_binary = true;
                block.metadata.quantization_stats.has_int8 = true;
            }
            // Note: Proxima automatically handles bloom filters during construction

            // Update ID index
            for (idx, record) in chunk.iter().enumerate() {
                if !record.oid.is_empty() {
                    self.id_index
                        .add(record.oid.clone(), block_id as u32, idx)?;
                }
            }

            // Group blocks into superblocks (64 blocks per superblock)
            let superblock_id = block_id / 64;
            if self.superblocks.len() <= superblock_id {
                // ✅ Create SuperBlock using Proxima composition pattern
                let mut superblock =
                    SuperBlock::new(superblock_id, format!("swift_sb_{}", superblock_id));

                // PROXIMA: Set SuperBlock-level encoding for hierarchical compression
                superblock.superblock_encoding_marker = 0x80; // SWIFT SuperBlock encoding

                // Initialize SWIFT-specific fields
                superblock.centroid = vec![0.0; self.header.dimension];
                superblock.block_centroids = Vec::new();
                superblock.quantized_signature = Vec::new();

                // ✅ Proxima will automatically provide bloom filters when blocks are added!

                self.superblocks.push(superblock);
            }

            // ✅ Use the new add_block method that leverages Proxima metadata
            self.superblocks[superblock_id].add_block(block);
        }

        // Update header statistics
        self.header.total_records = records.len() as u64;
        self.header.superblock_count = self.superblocks.len() as u32;

        // Build metadata indexes
        // Deferred: Fix SuperBlock type mismatch - MetadataIndex expects different SuperBlock type
        // self.metadata_index.build_from_superblocks(&self.superblocks[..])?;

        Ok(())
    }

    /// ✅ REFACTORED: Build blocks with compression using Proxima composition pattern
    pub fn build_blocks_from_records_with_compression(
        &mut self,
        records: Vec<VectorRecord>,
        compression_config: Option<crate::proto::proximadb_v1::CompressionConfig>,
    ) -> Result<()> {
        let canonical_records: Vec<ProximaRecord> =
            records.iter().map(ProximaRecord::from).collect();
        self.build_blocks_from_proxima_records_with_compression(
            canonical_records,
            compression_config,
        )
    }

    /// Build blocks directly from the canonical ProximaRecord envelope.
    ///
    /// ADR-009/010: SWIFT is a rebuildable projection/access method over the
    /// ProximaRecord spine. Protocol VectorRecord conversion is kept only in
    /// legacy compatibility methods above.
    pub fn build_blocks_from_proxima_records_with_compression(
        &mut self,
        records: Vec<ProximaRecord>,
        compression_config: Option<crate::proto::proximadb_v1::CompressionConfig>,
    ) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }

        // === NEW: Create blocks and prepare for AdaCurves clustering ===
        let records_per_block = self.header.records_per_block as usize;
        let mut blocks = Vec::new();

        // Helper structure to hold centroid info for clustering
        struct BlockWithCentroid {
            centroid: Vec<f32>,
        }
        let mut block_centroids = Vec::new();

        for chunk in records.chunks(records_per_block) {
            use crate::storage::engines::core::formats::proximablocks::compression_config::RowBasedCompressionConfig;
            let mut block_compression_config =
                RowBasedCompressionConfig::create_block_config_from_proto(
                    compression_config.as_ref(),
                );

            block_compression_config.vector_layout = crate::storage::engines::core::formats::proximablocks::VectorEncodingLayout::GroupedFieldEncodedAndCompressedVector;

            let block = ProximaDataBlock::new_with_engine_profile(
                chunk.to_vec(),
                block_compression_config,
                EngineProfile::Swift,
            );

            // Compute and store centroid for clustering
            let centroid = compute_block_centroid(&block);
            block_centroids.push(BlockWithCentroid {
                centroid: centroid.clone(),
            });
            blocks.push(block);
        }

        // === Cluster blocks using unified PCA + spatial encoding infrastructure ===
        // Uses shared SpatialClusteringPipeline for consistent behavior across engines
        // AdaCurve falls back to Hilbert (0.95 locality) which is superior to Z-Order
        info!(
            "🔬 SWIFT: Applying unified PCA + spatial clustering to {} blocks",
            blocks.len()
        );

        use crate::storage::engines::core::formats::proximablocks::spatial_encoding::SpatialCode;
        use crate::storage::engines::core::formats::proximablocks::spatial_traits::CurveType;
        use crate::storage::engines::core::pca::cluster_blocks_sync;

        let dimension = if let Some(first_centroid) = block_centroids.first() {
            first_centroid.centroid.len()
        } else {
            self.header.dimension
        };

        // Use min(32, dimension) for optimal clustering
        let target_dims = dimension.min(32);

        // Extract centroids for clustering
        let centroids: Vec<Vec<f32>> = block_centroids
            .iter()
            .map(|bc| bc.centroid.clone())
            .collect();

        // Use unified clustering (AdaCurve uses Hilbert internally for better locality)
        let clustering_result = cluster_blocks_sync(&centroids, CurveType::AdaCurve, target_dims);

        // Reorder blocks by spatial code
        let clustered_blocks: Vec<ProximaDataBlock> = clustering_result
            .sorted_indices
            .iter()
            .map(|&i| blocks[i].clone())
            .collect();

        // Convert SpatialCode to u64 for SWIFT's adacurve_code storage
        fn spatial_code_to_u64(code: &SpatialCode) -> u64 {
            match code {
                SpatialCode::Code64(v) => *v,
                SpatialCode::Code128(v) => *v as u64,
                _ => 0,
            }
        }

        let adacurve_codes: Vec<u64> = clustering_result
            .sorted_indices
            .iter()
            .map(|&i| spatial_code_to_u64(&clustering_result.spatial_codes[i]))
            .collect();

        info!(
            "🔬 SWIFT: Spatial clustering complete - codes range: {} to {}",
            adacurve_codes.iter().min().unwrap_or(&0),
            adacurve_codes.iter().max().unwrap_or(&0)
        );

        // Use clustered blocks for superblock construction
        let blocks_with_score: Vec<(ProximaDataBlock, f32, u64)> = clustered_blocks
            .into_iter()
            .zip(adacurve_codes.iter())
            .map(|(block, &code)| (block, code as f32, code))
            .collect();

        // Reinitialize superblocks for this build
        self.superblocks.clear();

        let mut superblock_codes: std::collections::HashMap<usize, Vec<u64>> =
            std::collections::HashMap::new();

        for (block_id, (mut block, _, adacurve_code)) in blocks_with_score.into_iter().enumerate() {
            let block_id = block_id as u32;
            // Assign deterministic block_id (preserves ID ordering inside blocks)
            block.block_id = block_id;

            // Update ID index with clustered block ordering
            for (idx, record) in block.records.iter().enumerate() {
                if !record.oid.is_empty() {
                    self.id_index.add(record.oid.clone(), block_id, idx)?;
                }
            }

            let superblock_id = (block_id / 64) as usize;
            if self.superblocks.len() <= superblock_id {
                let mut superblock =
                    SuperBlock::new(superblock_id, format!("swift_sb_{}", superblock_id));
                superblock.superblock_encoding_marker = 0x80; // SWIFT SuperBlock encoding
                superblock.centroid = vec![0.0; self.header.dimension];
                superblock.block_centroids = Vec::new();
                superblock.quantized_signature = Vec::new();
                self.superblocks.push(superblock);
            }

            // Track AdaCurve codes per superblock for aggregation
            superblock_codes
                .entry(superblock_id)
                .or_default()
                .push(adacurve_code);

            self.superblocks[superblock_id].add_block(block);
        }

        // Populate superblock AdaCurve codes (use average of block codes)
        for (sb_id, codes) in &superblock_codes {
            if let Some(superblock) = self.superblocks.get_mut(*sb_id) {
                let avg_code = if codes.is_empty() {
                    0
                } else {
                    (codes.iter().map(|&c| c as u128).sum::<u128>() / codes.len() as u128) as u64
                };
                superblock.adacurve_code = Some(avg_code);
            }
        }

        // Update header statistics
        self.header.total_records = records.len() as u64;
        self.header.superblock_count = self.superblocks.len() as u32;

        // Build metadata indexes
        // Deferred: Fix SuperBlock type mismatch - MetadataIndex expects different SuperBlock type
        // self.metadata_index.build_from_superblocks(&self.superblocks[..])?;

        Ok(())
    }

    /// Load a record at a specific location
    pub fn load_record_at_location(
        &self,
        location: &id_index::RecordLocation,
    ) -> Result<VectorRecord> {
        let superblock_id = location.superblock_idx;
        let block_idx = location.block_idx as usize;

        if superblock_id as usize >= self.superblocks.len() {
            return Err(anyhow!("Superblock {} not found", superblock_id));
        }

        let superblock = &self.superblocks[superblock_id as usize];
        if block_idx >= superblock.blocks.len() {
            return Err(anyhow!("Block {} not found in superblock", block_idx));
        }

        let block = &superblock.blocks[block_idx];
        if location.offset_in_block as usize >= block.records.len() {
            return Err(anyhow!(
                "Record offset {} out of bounds",
                location.offset_in_block
            ));
        }

        Ok(VectorRecord::from(
            &block.records[location.offset_in_block as usize],
        ))
    }

    /// Create a new SWIFT file - clean slate, no legacy
    pub fn new(collection_id: String, dimension: usize, distance_metric: String) -> Self {
        let header = SwiftHeader {
            magic: SWIFT_MAGIC,
            version: 1,
            collection_id,
            timestamp: chrono::Utc::now().timestamp(),
            compaction_level: 0,
            dimension,
            distance_metric,
            quantization: QuantizationConfig::default(),
            total_records: 0,
            deleted_records: 0,
            superblock_count: 0,
            blocks_per_superblock: 64,
            records_per_block: 512, // Reduced from 2000 for better granularity
            superblock_offset: 0,
            id_index_offset: 0,
            quantized_index_offset: 0,
            metadata_index_offset: 0,
            header_checksum: 0,
            file_checksum: 0,
        };

        Self {
            header,
            superblocks: Vec::new(),
            id_index: id_index::IdIndex::new(),
            // Quantized index now handled by unified compute module
            quantized_index: QuantizedIndex::new(dimension),
            metadata_index: hierarchical_blocks::MetadataIndex::new(),
            memory_manager: Arc::new(MemoryManager {
                max_memory_bytes: 4 * 1024 * 1024 * 1024, // 4GB
                current_usage: std::sync::atomic::AtomicUsize::new(0),
            }),
            // Note: simd_encoder removed - encoding now done via ProximaCodec per-operation
        }
    }

    /// Mode 1: AXIS-driven ID lookup
    pub async fn get_by_ids(&self, ids: &[String]) -> Result<Vec<VectorRecord>> {
        batch_operations::get_records_by_ids(self, ids).await
    }

    /// Mode 2: Index-free similarity search with progressive refinement
    pub async fn search_without_index(
        &self,
        query: &[f32],
        top_k: usize,
        filter: Option<SwiftMetadataFilter>,
        prune: &crate::core::search::BlockPruneConfig,
    ) -> Result<Vec<ProximaRecord>> {
        progressive_search::search_progressive(self, query, top_k, filter, prune).await
    }

    /// Serialize SwiftFile to bytes for disk persistence
    /// Uses Proxima block serialization similar to SST for optimal performance
    pub fn serialize(&self) -> Result<Vec<u8>> {
        use bytes::BytesMut;

        let mut buffer = BytesMut::new();

        // Write header with magic and version
        buffer.extend_from_slice(&self.header.magic);
        buffer.extend_from_slice(&self.header.version.to_le_bytes());
        buffer.extend_from_slice(&(self.header.collection_id.len() as u32).to_le_bytes());
        buffer.extend_from_slice(self.header.collection_id.as_bytes());
        buffer.extend_from_slice(&self.header.timestamp.to_le_bytes());
        buffer.extend_from_slice(&self.header.dimension.to_le_bytes());
        buffer.extend_from_slice(&self.header.total_records.to_le_bytes());
        buffer.extend_from_slice(&(self.superblocks.len() as u32).to_le_bytes());

        // Write superblocks with Proxima optimization
        for superblock in &self.superblocks {
            // Write superblock metadata
            buffer.extend_from_slice(&(superblock.superblock_id as u32).to_le_bytes());
            buffer.extend_from_slice(&superblock.record_count.to_le_bytes());
            buffer.extend_from_slice(&(superblock.blocks.len() as u32).to_le_bytes());

            // Serialize each Proxima block efficiently
            for block in &superblock.blocks {
                // Proxima blocks already have built-in serialization
                // Get serialized block and bloom filter in parallel
                let (block_bytes, bloom_data) = block.serialize_with_bloom_sync()?;

                // Write block size then data
                buffer.extend_from_slice(&(block_bytes.len() as u32).to_le_bytes());
                buffer.extend_from_slice(&block_bytes);

                // Store bloom filter for later aggregation if generated
                if let Some(bloom) = bloom_data {
                    debug!(
                        "✅ SWIFT: Generated bloom filter for block: {} bytes",
                        bloom.len()
                    );
                }
            }

            // ✅ Write aggregated bloom filter from Proxima blocks
            // Aggregate bloom filters from all blocks in superblock
            let mut has_bloom = false;
            for block in &superblock.blocks {
                if block.bloom_filter.is_some() {
                    has_bloom = true;
                    break;
                }
            }

            if has_bloom {
                buffer.extend_from_slice(&1u8.to_le_bytes()); // Has bloom filter
                // For now, just use first block's bloom filter as representative
                // Deferred: Implement proper bloom filter aggregation
                if let Some(first_block) = superblock.blocks.first() {
                    if let Some(ref bloom) = first_block.bloom_filter {
                        let bloom_bytes: Vec<u8> = bloom.serialize()?;
                        buffer.extend_from_slice(&(bloom_bytes.len() as u32).to_le_bytes());
                        buffer.extend_from_slice(&bloom_bytes);
                    } else {
                        buffer.extend_from_slice(&0u32.to_le_bytes()); // Empty bloom
                    }
                } else {
                    buffer.extend_from_slice(&0u32.to_le_bytes()); // Empty bloom
                }
            } else {
                buffer.extend_from_slice(&0u8.to_le_bytes()); // No bloom filter
            }

            // Write block centroids (mandatory, aligned with blocks)
            buffer.extend_from_slice(&(superblock.block_centroids.len() as u32).to_le_bytes());
            for centroid in &superblock.block_centroids {
                buffer.extend_from_slice(&(centroid.len() as u32).to_le_bytes());
                for v in centroid {
                    buffer.extend_from_slice(&v.to_le_bytes());
                }
            }
        }

        Ok(buffer.to_vec())
    }

    /// Write SwiftFile to disk using filesystem abstraction (follows SST pattern)
    pub async fn write_to_disk(
        &self,
        filesystem_factory: &crate::storage::persistence::filesystem::FilesystemFactory,
        path: &str,
    ) -> Result<u64> {
        use crate::storage::persistence::filesystem::atomic_strategy::AtomicWriteExecutorFactory;

        let data = self.serialize()?;
        let bytes_written = data.len() as u64;

        // Get filesystem based on path (same as SST writer.rs:351)
        let path_str = path;
        let (_scheme, fs_url) = if path_str.contains("://") {
            let parts: Vec<&str> = path_str.splitn(2, "://").collect();
            (parts[0], path_str.to_string())
        } else {
            ("file", format!("file://{}", path_str))
        };
        let fs = filesystem_factory.get_filesystem(&fs_url)?;

        // Use atomic writer for safe persistence (same as SST writer.rs:584)
        let atomic_writer = AtomicWriteExecutorFactory::create_production_executor();
        atomic_writer.write_atomic(&*fs, path, &data, None).await?;

        Ok(bytes_written)
    }

    /// Deserialize SwiftFile from bytes with optional collection config for type-safe metadata
    pub fn deserialize(
        data: &[u8],
        collection: Option<&crate::proto::proximadb_v1::Collection>,
    ) -> Result<Self> {
        use bytes::Buf;
        use std::io::{Cursor, Read};

        let mut cursor = Cursor::new(data);

        // Read header
        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic)?;

        if magic != SWIFT_MAGIC {
            return Err(anyhow!("Invalid SWIFT file magic"));
        }

        let version = cursor.get_u32_le();
        let collection_id_len = cursor.get_u32_le() as usize;
        let mut collection_id = vec![0u8; collection_id_len];
        cursor.read_exact(&mut collection_id)?;
        let collection_id = String::from_utf8(collection_id)?;

        let timestamp = cursor.get_i64_le();
        let dimension = cursor.get_u64_le() as usize;
        let total_records = cursor.get_u64_le();
        let superblock_count = cursor.get_u32_le() as usize;

        // Create header
        let header = SwiftHeader {
            magic,
            version,
            collection_id: collection_id.clone(),
            timestamp,
            dimension,
            total_records,
            superblock_count: superblock_count as u32,
            compaction_level: 0,
            distance_metric: "euclidean".to_string(),
            quantization: QuantizationConfig::default(),
            deleted_records: 0,
            blocks_per_superblock: 64,
            records_per_block: 512, // Reduced from 2000 for better granularity
            superblock_offset: 0,
            id_index_offset: 0,
            quantized_index_offset: 0,
            metadata_index_offset: 0,
            header_checksum: 0,
            file_checksum: 0,
        };

        // Read superblocks
        let mut superblocks = Vec::with_capacity(superblock_count);

        for _ in 0..superblock_count {
            let superblock_id = cursor.get_u32_le() as usize;
            let record_count = cursor.get_u32_le();
            let block_count = cursor.get_u32_le() as usize;

            let mut superblock = SuperBlock::new(superblock_id, format!("sb_{}", superblock_id));
            superblock.record_count = record_count;

            // Read Proxima blocks
            for _ in 0..block_count {
                let block_size = cursor.get_u32_le() as usize;
                let mut block_data = vec![0u8; block_size];
                cursor.read_exact(&mut block_data)?;

                // Deserialize Proxima block with collection config for type-safe metadata
                let block = ProximaDataBlock::deserialize(&block_data, collection)?;
                superblock.blocks.push(block);
            }

            // Read bloom filter flag (for backward compatibility)
            let has_bloom = cursor.get_u8() == 1;
            if has_bloom {
                let bloom_size = cursor.get_u32_le() as usize;
                let mut bloom_data = vec![0u8; bloom_size];
                cursor.read_exact(&mut bloom_data)?;
                // ✅ Bloom filters are now stored in Proxima blocks, skip legacy bloom data
                // The blocks already have their bloom filters from deserialization
            }

            // Read block centroids
            let centroid_count = cursor.get_u32_le() as usize;
            for _ in 0..centroid_count {
                let len = cursor.get_u32_le() as usize;
                let mut buf = vec![0u8; len * 4];
                cursor.read_exact(&mut buf)?;
                let mut centroid = Vec::with_capacity(len);
                for chunk in buf.as_chunks::<4>().0 {
                    centroid.push(f32::from_le_bytes(*chunk));
                }
                superblock.block_centroids.push(centroid);
            }

            // Recompute superblock centroid if possible (weighted by block records)
            if !superblock.block_centroids.is_empty() {
                let mut agg = vec![0f32; superblock.block_centroids[0].len()];
                let mut total = 0f32;
                for (block, centroid) in superblock
                    .blocks
                    .iter()
                    .zip(superblock.block_centroids.iter())
                {
                    if centroid.len() != agg.len() {
                        continue;
                    }
                    let w = block.metadata.record_count as f32;
                    total += w;
                    for (i, v) in centroid.iter().enumerate() {
                        agg[i] += *v * w;
                    }
                }
                if total > 0.0 {
                    for v in &mut agg {
                        *v /= total;
                    }
                    superblock.centroid = agg;
                }
            }

            superblocks.push(superblock);
        }

        Ok(SwiftFile {
            header,
            superblocks,
            id_index: id_index::IdIndex::new(),
            quantized_index: QuantizedIndex::new(dimension),
            metadata_index: hierarchical_blocks::MetadataIndex::new(),
            memory_manager: Arc::new(MemoryManager {
                max_memory_bytes: 4 * 1024 * 1024 * 1024,
                current_usage: std::sync::atomic::AtomicUsize::new(0),
            }),
            // Note: simd_encoder removed - encoding now done via ProximaCodec per-operation
        })
    }

    /// Read SwiftFile from disk using filesystem (follows SST pattern)
    pub async fn read_from_disk(
        filesystem_factory: &crate::storage::persistence::filesystem::FilesystemFactory,
        path: &str,
        collection: Option<&crate::proto::proximadb_v1::Collection>,
    ) -> Result<Self> {
        // Get filesystem based on path (same pattern as SST reader)
        let path_str = path;
        let (_scheme, fs_url) = if path_str.contains("://") {
            let parts: Vec<&str> = path_str.splitn(2, "://").collect();
            (parts[0], path_str.to_string())
        } else {
            ("file", format!("file://{}", path_str))
        };

        let fs = filesystem_factory.get_filesystem(&fs_url)?;
        let data = fs.read(path).await?;

        // Deserialize from data with collection config
        Self::deserialize(&data, collection)
    }

    /// Read SwiftFile using unified caching filesystem (for optimized queries)
    pub async fn read_from_disk_with_cache(
        filesystem_factory: &crate::storage::persistence::filesystem::FilesystemFactory,
        path: &str,
        collection_id: &str,
        collection: Option<&crate::proto::proximadb_v1::Collection>,
    ) -> Result<Self> {
        use crate::storage::persistence::filesystem::FileSystem;

        // Create UnifiedCachingFilesystem (same as SST unified_reader.rs:47)
        let base_fs = filesystem_factory.get_filesystem("file://")?;
        let unified_fs = std::sync::Arc::new(
            crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem::new(
                base_fs,
                collection_id.to_string(),
                "swift".to_string(),
            ),
        );

        // Read with caching (unified_fs implements FileSystem trait)
        let data = unified_fs
            .read(path)
            .await
            .map_err(|e| anyhow!("Failed to read file: {}", e))?;

        // Deserialize from cached data with collection config
        Self::deserialize(&data, collection)
    }

    /// PROXIMA: Optimize SuperBlock encoding for columnar SIMD and hierarchical compression
    /// Uses columnar layout for maximum SIMD efficiency and optimized I/O
    #[allow(dead_code)]
    fn finalize_superblock_encoding(&mut self) {
        // use proximadb_hardware_caps::HardwareCapabilities; // Unused import

        let hw_caps = proximadb_hardware_caps::get_hardware_capabilities();

        for superblock in &mut self.superblocks {
            // Columnar analysis: transpose vectors to analyze dimension-wise
            let mut all_vectors = Vec::new();
            // INT-2.5b: as_fp32_cow returns Cow<'_, [f32]> which is
            // owned for non-Fp32 variants — the temporary would drop
            // before we use it. Materialize the owned Vec<f32> and
            // hold it for the duration of the analysis.
            let mut owned_vectors: Vec<Vec<f32>> = Vec::new();
            for block in &superblock.blocks {
                for record in &block.records {
                    if let Some(embedding) = record.embeddings.first() {
                        owned_vectors.push(embedding.as_fp32_cow().into_owned());
                    }
                }
            }
            all_vectors.extend(owned_vectors.iter().map(|v| v.as_slice()));

            if all_vectors.is_empty() {
                continue;
            }

            let dimension = all_vectors[0].len();
            let vector_count = all_vectors.len();

            // COLUMNAR ANALYSIS: Transpose to dimension-major layout for SIMD
            let mut columnar_stats = Vec::new();
            for dim in 0..dimension {
                let mut values: Vec<f32> = all_vectors.iter().map(|v| v[dim]).collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let min_val = values[0];
                let max_val = values[values.len() - 1];
                let range = max_val - min_val;

                // Calculate delta compression potential
                let mut sum_delta = 0.0f32;
                for i in 1..values.len() {
                    sum_delta += (values[i] - values[i - 1]).abs();
                }
                let avg_delta = sum_delta / (values.len() - 1) as f32;

                columnar_stats.push((min_val, max_val, range, avg_delta));
            }

            // SIMD-optimized scheme selection based on hardware capabilities
            let (marker, _scheme) = if hw_caps.cpu.simd.has_avx512 {
                // AVX-512: 16x f32 SIMD operations
                Self::select_avx512_scheme(&columnar_stats, vector_count)
            } else if hw_caps.cpu.simd.has_avx2 {
                // AVX2: 8x f32 SIMD operations
                Self::select_avx2_scheme(&columnar_stats, vector_count)
            } else if hw_caps.cpu.simd.has_sse {
                // SSE: 4x f32 SIMD operations
                Self::select_sse_scheme(&columnar_stats, vector_count)
            } else {
                // Fallback: scalar optimized
                Self::select_scalar_scheme(&columnar_stats, vector_count)
            };

            // Calculate global statistics for metadata
            let (_global_min, _global_max) = columnar_stats.iter().fold(
                (f32::MAX, f32::MIN),
                |(min_acc, max_acc), &(min_val, max_val, _, _)| {
                    (min_acc.min(min_val), max_acc.max(max_val))
                },
            );

            superblock.superblock_encoding_marker = marker;
            // ✅ Update Proxima metadata in the new composition structure
            // This metadata will be aggregated from blocks automatically via add_block()

            // Update child blocks to inherit SuperBlock columnar encoding
            for block in &mut superblock.blocks {
                // Child blocks inherit columnar SIMD optimization
                if block.encoding_marker == 0x00 || block.encoding_marker == 0x10 {
                    block.encoding_marker = 0xFF; // Inherit columnar SIMD from SuperBlock
                    block.encoding_metadata = None; // Use SuperBlock columnar metadata
                }
            }

            // NEW: Compute FP16 centroid for superblock (50% storage reduction)
            if !superblock.centroid.is_empty() {
                superblock.centroid_fp16 = Some(crate::storage::engines::sst::fp32_to_fp16(
                    &superblock.centroid,
                ));
            }
        }
    }

    /// AVX-512 optimized scheme selection for 16-wide SIMD
    #[allow(dead_code)]
    fn select_avx512_scheme(
        stats: &[(f32, f32, f32, f32)],
        _vector_count: usize,
    ) -> (u8, ProximaScheme) {
        let avg_range =
            stats.iter().map(|(_, _, range, _)| *range).sum::<f32>() / stats.len() as f32;
        let avg_delta =
            stats.iter().map(|(_, _, _, delta)| *delta).sum::<f32>() / stats.len() as f32;

        if avg_range < 1e-6 {
            (0x96, ProximaScheme::RunLength) // AVX-512 run-length
        } else if avg_delta < avg_range / 16.0 {
            // Excellent delta compression for AVX-512
            (
                0x92,
                ProximaScheme::Delta {
                    base: stats[0].0 as i64,
                },
            )
        } else if avg_range < 32.0 {
            // Frame-of-reference for 16-wide SIMD
            (
                0x93,
                ProximaScheme::FrameOfReference {
                    reference: stats[0].0 as i64,
                    bits: (avg_range.log2().ceil() as u8).max(6), // Optimized for AVX-512
                },
            )
        } else {
            // High-precision BitPacking for AVX-512
            (
                0x91,
                ProximaScheme::BitPacked {
                    bits: (avg_range.log2().ceil() as u8).max(10),
                },
            )
        }
    }

    /// AVX2 optimized scheme selection for 8-wide SIMD
    #[allow(dead_code)]
    fn select_avx2_scheme(
        stats: &[(f32, f32, f32, f32)],
        _vector_count: usize,
    ) -> (u8, ProximaScheme) {
        let avg_range =
            stats.iter().map(|(_, _, range, _)| *range).sum::<f32>() / stats.len() as f32;
        let avg_delta =
            stats.iter().map(|(_, _, _, delta)| *delta).sum::<f32>() / stats.len() as f32;

        if avg_range < 1e-6 {
            (0x86, ProximaScheme::RunLength) // AVX2 run-length
        } else if avg_delta < avg_range / 8.0 {
            // Good delta compression for AVX2
            (
                0x82,
                ProximaScheme::Delta {
                    base: stats[0].0 as i64,
                },
            )
        } else if avg_range < 64.0 {
            // Frame-of-reference for 8-wide SIMD
            (
                0x83,
                ProximaScheme::FrameOfReference {
                    reference: stats[0].0 as i64,
                    bits: (avg_range.log2().ceil() as u8).max(8),
                },
            )
        } else {
            // Standard BitPacking for AVX2
            (
                0x81,
                ProximaScheme::BitPacked {
                    bits: (avg_range.log2().ceil() as u8).max(12),
                },
            )
        }
    }

    /// SSE optimized scheme selection for 4-wide SIMD
    #[allow(dead_code)]
    fn select_sse_scheme(
        stats: &[(f32, f32, f32, f32)],
        _vector_count: usize,
    ) -> (u8, ProximaScheme) {
        let avg_range =
            stats.iter().map(|(_, _, range, _)| *range).sum::<f32>() / stats.len() as f32;
        let avg_delta =
            stats.iter().map(|(_, _, _, delta)| *delta).sum::<f32>() / stats.len() as f32;

        if avg_range < 1e-6 {
            (0x76, ProximaScheme::RunLength) // SSE run-length
        } else if avg_delta < avg_range / 4.0 {
            // Modest delta compression for SSE
            (
                0x72,
                ProximaScheme::Delta {
                    base: stats[0].0 as i64,
                },
            )
        } else if avg_range < 128.0 {
            // Conservative frame-of-reference for 4-wide SIMD
            (
                0x73,
                ProximaScheme::FrameOfReference {
                    reference: stats[0].0 as i64,
                    bits: (avg_range.log2().ceil() as u8).max(10),
                },
            )
        } else {
            // Basic BitPacking for SSE
            (
                0x71,
                ProximaScheme::BitPacked {
                    bits: (avg_range.log2().ceil() as u8).max(14),
                },
            )
        }
    }

    /// Scalar optimized scheme selection (no SIMD)
    #[allow(dead_code)]
    fn select_scalar_scheme(
        stats: &[(f32, f32, f32, f32)],
        _vector_count: usize,
    ) -> (u8, ProximaScheme) {
        let avg_range =
            stats.iter().map(|(_, _, range, _)| *range).sum::<f32>() / stats.len() as f32;
        let avg_delta =
            stats.iter().map(|(_, _, _, delta)| *delta).sum::<f32>() / stats.len() as f32;

        if avg_range < 1e-6 {
            (0x66, ProximaScheme::RunLength) // Scalar run-length
        } else if avg_delta < avg_range / 2.0 {
            // Simple delta compression
            (
                0x62,
                ProximaScheme::Delta {
                    base: stats[0].0 as i64,
                },
            )
        } else if avg_range < 256.0 {
            // Basic frame-of-reference
            (
                0x63,
                ProximaScheme::FrameOfReference {
                    reference: stats[0].0 as i64,
                    bits: (avg_range.log2().ceil() as u8).max(12),
                },
            )
        } else {
            // Conservative BitPacking
            (
                0x61,
                ProximaScheme::BitPacked {
                    bits: (avg_range.log2().ceil() as u8).max(16),
                },
            )
        }
    }
}

/// Backwards-compat alias for [`SwiftMetadataFilter`].
pub type MetadataFilter = SwiftMetadataFilter;

/// Metadata filter for queries
#[derive(Debug, Clone)]
pub struct SwiftMetadataFilter {
    pub conditions: Vec<FilterCondition>,
}

#[derive(Debug, Clone)]
pub enum FilterCondition {
    Equals(String, serde_json::Value),
    Range(String, serde_json::Value, serde_json::Value),
    In(String, Vec<serde_json::Value>),
    IsNull(String),
    IsNotNull(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swift_file_creation() {
        let sst = SwiftFile::new("test_collection".to_string(), 768, "cosine".to_string());

        assert_eq!(sst.header.collection_id, "test_collection");
        assert_eq!(sst.header.dimension, 768);
        assert_eq!(sst.header.version, 1);
        assert_eq!(sst.header.magic, SWIFT_MAGIC);
    }

    #[test]
    fn test_quantization_config_default() {
        let config = QuantizationConfig::default();
        // Proto bool fields default to false
        assert!(!config.enabled.unwrap_or(false));
    }
}
