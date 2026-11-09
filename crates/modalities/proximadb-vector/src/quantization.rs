//! # Vector Quantization
//!
//! This module provides vector quantization for compression and fast approximate search.
//!
//! ## Quantization Types
//!
//! - **Binary** - 1-bit quantization (0/1)
//! - **Scalar (Int8)** - 8-bit scalar quantization
//! - **Product (PQ)** - Product quantization with sub-vector codebooks
//!
//! ## Note on Engine Consolidation
//!
//! The production quantization engine (7,556 lines) remains in `src/compute/quantization/`
//! due to complex dependencies on storage, core, and compute infrastructure.
//! This will be migrated to vector modality in Phase 6B after dependency untangling.

pub mod binary;
pub mod compile_time;
pub mod hardware_accelerated;
pub mod internal_types;
pub mod product;
pub mod scalar;
/// 2-bit Sign-Magnitude quantizer (QuIVer, arXiv 2605.02171). 16:1 vs float32.
pub mod sign_magnitude;
pub mod smart_defaults;

/// TurboQuant data-oblivious quantizer (ADR-021, arXiv:2504.19874).
/// See `docs/12-design/TURBOQUANT_HLD_2026_05_30.adoc` and
/// `docs/12-design/TURBOQUANT_LLD_2026_05_30.adoc` for design intent and
/// per-phase status. Implementation lands across P2-P4.
#[cfg(feature = "experimental-turboquant")]
pub mod turboquant;

use serde::{Deserialize, Serialize};

// Re-export proto types for compatibility
pub use proximadb_proto::v1::{QuantizationConfig, QuantizationLevel};

// The proto wire enum is re-exported under a QUALIFIED name (2026-11-08):
// `internal_types.rs` glob-brings the types-crate `QuantizationType` (the
// internal SSOT) into the sibling module, and both enums share the
// Binary/Scalar/Product variant names — a bare-name export of BOTH meant a
// wrong-import compiled silently and only mis-decoded on the divergent
// variants (proto Uniform vs types TurboQuant).
pub use proximadb_proto::v1::quantization_level::QuantizationType as ProtoQuantizationType;

// Re-export implementations
pub use binary::{BinaryQuantizer, BinaryVector};
pub use hardware_accelerated::AcceleratedQuantization;
pub use product::{PQCodebook, PQVector, ProductQuantizer};
pub use scalar::{Int8Vector, ScalarQuantizer};
pub use smart_defaults::QuantizationSmartDefaults;

/// Quantized vector data
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QuantizedVectorData {
    /// Binary quantized (1 bit per dimension)
    Binary(BinaryVector),
    /// Scalar int8 quantized (8 bits per dimension)
    Int8(Int8Vector),
    /// Product quantized (codebook indices)
    PQ(PQVector),
}

impl QuantizedVectorData {
    pub fn quantization_type(&self) -> ProtoQuantizationType {
        match self {
            QuantizedVectorData::Binary(_) => ProtoQuantizationType::Binary,
            QuantizedVectorData::Int8(_) => ProtoQuantizationType::Scalar,
            QuantizedVectorData::PQ(_) => ProtoQuantizationType::Product,
        }
    }

    pub fn size_bytes(&self) -> usize {
        match self {
            QuantizedVectorData::Binary(v) => v.data.len(),
            QuantizedVectorData::Int8(v) => v.data.len(),
            QuantizedVectorData::PQ(v) => v.indices.len() * std::mem::size_of::<u8>(),
        }
    }
}

/// Common quantization trait
pub trait Quantizer<T, U> {
    fn train(&mut self, vectors: &[T]) -> Result<(), String>;
    fn quantize(&self, vector: &[T]) -> Result<U, String>;
    fn unquantize(&self, quantized: &U) -> Vec<T>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_type() {
        let binary = QuantizedVectorData::Binary(BinaryVector {
            data: vec![0b10101010],
        });
        assert_eq!(binary.quantization_type(), ProtoQuantizationType::Binary);

        let int8 = QuantizedVectorData::Int8(Int8Vector {
            data: vec![1i8, 2, 3],
            scale: 0.01,
            min: vec![-1.0, -1.0, -1.0],
        });
        assert_eq!(int8.quantization_type(), ProtoQuantizationType::Scalar);
    }

    #[test]
    fn test_size_bytes() {
        let binary = QuantizedVectorData::Binary(BinaryVector {
            data: vec![0b10101010],
        });
        assert_eq!(binary.size_bytes(), 1);

        let int8 = QuantizedVectorData::Int8(Int8Vector {
            data: vec![1i8, 2, 3],
            scale: 0.01,
            min: vec![-1.0, -1.0, -1.0],
        });
        assert_eq!(int8.size_bytes(), 3);
    }
}
