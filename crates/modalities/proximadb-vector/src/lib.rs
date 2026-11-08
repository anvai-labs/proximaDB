//! # ProximaDB Vector Modality
//!
//! This crate contains vector search engines and indexing, quantization, ANN, scoring,
//! and vector operations services for the ProximaDB vector database.
//!
//! ## Architecture
//!
//! The vector modality is organized into several key modules:
//!
//! - **`distance`** - Distance metrics (L2, Cosine, Dot Product, Manhattan) with SIMD acceleration
//! - **`quantization`** - Vector quantization (Scalar, Product, Binary)
//! - **`index`** - Vector indexing algorithms (HNSW, IVF, PQ, Annoy, LSH)
//! - **`search`** - Vector similarity search and ANN algorithms
//! - **`service`** - Vector service implementing VectorQueryService trait (Phase 3)
//!
//! ## Foundation
//!
//! This crate serves as the foundation for vector operations across ProximaDB,
//! providing reusable contracts and implementations for:
//!
//! - Storage engines that need vector similarity search
//! - Query executors that need vector operations
//! - Index builders that need quantization and distance metrics
//!
//! ## Phase 3: Modality Runtime Extraction
//!
//! As of Phase 3, this crate provides a root-independent vector query runtime through the `service` module:
//! - `VectorServiceImpl` implements the stable `VectorQueryService` trait
//! - Scores caller-supplied canonical `ProximaRecord` values without fabricating results
//! - Keeps the server production path on `VectorOperationsService` until AXIS is exposed behind a narrow contract
//! - Supports gradual migration through trait object injection without creating a separate durable vector store
//!
//! ## Dependencies
//!
//! - `proximadb-kernel` - Core error types and foundational contracts
//! - `proximadb-proto` - Protocol buffer types for VectorRecord
//! - `proximadb-query-filter` - Filter expression contracts
//! - `proximadb-vector-query` - Vector query service contract
//! - `arrow` - Columnar data structures for vector operations

pub mod distance;
pub mod index;
pub mod quantization;
pub mod transform_index;

// Re-export common types for convenience. The duplicate distance/search
// engine (SimilarityResult, UnifiedDistanceCompute, BruteForceSearch,
// VectorServiceImpl) was removed under TD-SEARCH-1 — the canonical compute
// lives in `proximadb-distance-kernel`; this crate keeps only the metric
// enum + proto conversions.
pub use distance::DistanceMetric;

pub use quantization::{
    ProtoQuantizationType, QuantizationConfig, QuantizationLevel, QuantizedVectorData,
};

pub use index::{
    FlatIndex, IndexError, IndexParameters, IndexStats, IndexType, Neighbor, VectorIndex,
    VectorIndexConfig,
};

pub use transform_index::{DisentangledVectorProjection, TransformProjectionSpec};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_module_imports() {
        let _metric = DistanceMetric::Euclidean;
    }

    #[test]
    fn proto_metric_conversion_round_trips() {
        for metric in [
            DistanceMetric::Cosine,
            DistanceMetric::Euclidean,
            DistanceMetric::DotProduct,
            DistanceMetric::Manhattan,
        ] {
            let proto = distance::internal_distance_to_proto(metric);
            assert_eq!(distance::proto_distance_to_internal(proto), metric);
        }
    }
}
