//! # RAPTOR Engine - ⚠️ DEPRECATED - EXPERIMENTAL
//!
//! **WARNING**: This engine is DEPRECATED and EXPERIMENTAL.
//!
//! ## Deprecation Notice (2026-04-03)
//!
//! RAPTOR is deprecated and will be removed in v1.0 unless contributors complete the
//! remaining implementation. See `/docs/storage/EXPERIMENTAL_ENGINES_STATUS.md` for details.
//!
//! ## Current Status
//!
//! - **Status**: EXPERIMENTAL - Not Production Ready
//! - **Feature Flag**: `experimental-engines` required
//! - **Tests**: 23 tests (insufficient coverage)
//! - **DEFERRED Items**: 35+ optimization tasks remaining
//! - **Recommendation**: Use SST, VIPER, HELIX, or NOVA instead
//!
//! ## Migration Guide
//!
//! For adaptive workload optimization, use production engines with appropriate configuration:
//! - **VIPER**: For vector storage with Proxima encoding and compression
//! - **HELIX**: For high-dimensional data with PCA dimension reduction
//! - **SST**: For static workloads with efficient range queries
//!
//! Example:
//! ```rust,ignore
//! // Instead of RAPTOR adaptive mode:
//! // raptor_engine.enable_adaptive_mode(true).await?;
//!
//! // Use VIPER with optimized configuration:
//! let viper_config = ViperConfig::optimized_for_workload(&workload_profile);
//! let viper_engine = ViperEngine::new(viper_config).await?;
//! ```
//!
//! ## RAPTOR Storage Engine - Row-Aligned Predicated Tensor Optimized Repository
//!
//! RAPTOR is ProximaDB's adaptive storage engine featuring the Matrix Trinity architecture for intelligent workload optimization.
//!
//! ### Adaptive Capabilities (experimental):
//! 1. **Matrix Trinity Architecture**: P²+K²+P×K matrix system for search navigation
//! 2. **Workload Adaptation**: Optimization based on query patterns and data distribution
//! 3. **Smart Resource Management**: Adaptive row group sizing and memory-efficient operations
//! 4. **Intelligent Compaction**: Consolidation with pattern-aware optimization
//!
//! **STATUS**: EXPERIMENTAL - Not recommended for production use
//!
//! ## 🎯 OPTIMAL USE CASES
//!
//! RAPTOR excels in dynamic environments requiring workload adaptation:
//!
//! ### ✅ **Dynamic Recommendation Systems**
//! ```rust,ignore
//! // E-commerce platforms with changing user preferences
//! let user_embeddings = load_user_behavior_vectors(); // 512D user profiles
//! raptor_engine.flush_with_adaptation(user_embeddings).await; // Adapts to usage patterns
//! let recommendations = raptor_engine.search_adaptive(user_query, 20).await; // Smart k-sizing
//! ```
//!
//! ### ✅ **Multi-Tenant SaaS Platforms**
//! ```rust,ignore
//! // Different tenants with varying query patterns
//! for tenant_batch in tenant_data_batches {
//!     raptor_engine.configure_adaptive_params(&tenant_batch.tenant_id,
//!         &tenant_batch.workload_profile).await; // Per-tenant optimization
//!     raptor_engine.flush(tenant_batch.vectors).await; // Adaptive row sizing
//! }
//! ```
//!
//! ### ✅ **Research and Development Workloads**
//! ```rust,ignore
//! // Experimental datasets with unknown query patterns
//! let research_vectors = load_experimental_embeddings(); // Variable dimensions
//! raptor_engine.enable_adaptive_mode(true).await; // Learn optimal parameters
//! let results = raptor_engine.search_with_learning(query, k).await; // Improve over time
//! ```
//!
//! ## 🚀 **MATRIX TRINITY ARCHITECTURE**
//!
//! RAPTOR's core innovation is the Matrix Trinity system:
//!
//! ### **P² Matrix (Intra-RowGroup)**
//! - **Purpose**: Pairwise distances within row groups for local navigation
//! - **Optimization**: SIMD-accelerated distance computation with Proxima compression
//! - **Benefit**: O(1) neighbor lookup within clusters
//!
//! ### **K² Matrix (Inter-Centroid)**
//! - **Purpose**: Centroid-to-centroid distances for global navigation
//! - **Optimization**: Sparse storage for distant centroids with intelligent pruning
//! - **Benefit**: Efficient cluster-to-cluster traversal
//!
//! ### **P×K Matrix (Vector-to-Centroid)**
//! - **Purpose**: Adaptive coverage based on workload patterns
//! - **Optimization**: Dynamic sparsity with boundary detection
//! - **Benefit**: Learned query pattern optimization
//!
//! ## ❌ **NOT OPTIMAL FOR:**
//!
//! - **Static Workloads**: SST or VIPER better for predictable patterns
//! - **Memory-Constrained Systems**: Matrix storage requires significant RAM
//! - **Simple Point Queries**: HELIX spatial locality may be more efficient
//! - **Append-Only Workloads**: NOVA columnar analytics may be preferable
//!
//! ## 📊 PERFORMANCE CHARACTERISTICS
//!
//! - **Query Performance**: Excellent (adaptive optimization improves over time)
//! - **Write Performance**: Good (intelligent batching with adaptive row sizing)
//! - **Storage Efficiency**: Moderate (matrix overhead balanced by compression)
//! - **Memory Usage**: High (matrices cached for performance)
//! - **Adaptation Speed**: Fast (learns patterns within 1000s of queries)

/// Magic constant for RAPTOR files (4 bytes)
pub const RAPTOR_MAGIC: [u8; 4] = *b"RPTR";

// Common types module - MUST be first to avoid circular dependencies
pub mod common;
pub mod config;
pub mod constants;

// Core modules
pub mod adaptive_pxk;
pub mod consolidated_compactor;
pub mod consolidated_reader;
pub mod engine;
pub mod extraction;
// matrix_builder hoisted to the proximadb-raptor-engine crate (writer-only dep).
// metadata_serializer removed - functionality consolidated into unified_metadata_serializer
pub mod unified_metadata_serializer {
    pub use crate::storage::engines::core::matrix_trinity_serializer::*;
}
pub mod writer;
// ivf_manager removed - obsolete with Matrix Trinity (P² + K² + P×K)
pub mod artus_bloom;
pub mod rowgroup_manager;
pub use proximadb_engine_core::smart_rowgroup_sizing; // extracted to engine-core (TD-DECOMP-69)

// Re-export commonly used types from common module
pub use common::{
    BloomFilterMetadata,
    BoostingStrategy, // Added boundary/correction types
    CachePolicy,
    ColumnEncoding,
    ColumnPageMetadata,
    ColumnStats,
    ColumnType,       // Added columnar page types
    ColumnnarIdIndex, // Added bloom filter types
    ConfidenceAssessment,
    ConfidenceSignals,
    CorrectionStrategy,
    FieldDescriptor,
    IoStrategy,
    K2Matrix,
    LocalityCluster,
    MetadataColumn,
    MetadataData,
    MetadataValue,
    P2Matrix,
    Predicate,
    PredicateOp,
    ProximaScheme,
    RaptorFileMetadata,
    ReadPattern,
    RowGroup,
    RowGroupBloomFilter,
    RowGroupMetadata,
    SchemaDescriptor,
    SpilloverInfo,
    VectorCentroidMatrix, // Matrix Trinity architecture
    VectorEncoding,
    VectorStats,
};

// Export consolidated modules instead of deprecated ones
pub use config::{AccuracyLevel, CompactionConfig, CompressionStrategy, PxKStrategy, RaptorConfig};
pub use consolidated_compactor::RaptorCompactor;
pub use consolidated_reader::{IntraRowgroupMatrix, RaptorReader}; // Use consolidated reader
#[allow(deprecated)]
pub use engine::RaptorEngine;
pub use writer::RaptorWriter; // Use consolidated compactor
// IvfManager removed - Matrix Trinity handles clustering via centroids
pub use adaptive_pxk::{AdaptivePxKStorage, BoundaryInfo, SelectionReason, VectorSelection};
pub use common::{
    ColumnarBlock, MetadataColumns, ProximaEncodedData, QuantizationParams, QuantizedColumnarData,
    TransposedVectors,
};
pub use rowgroup_manager::RowGroups;
pub use smart_rowgroup_sizing::{
    CloudIOProfile, CommonConfigurations, OptimalRowGroupSize, SmartRowGroupSizer,
};
