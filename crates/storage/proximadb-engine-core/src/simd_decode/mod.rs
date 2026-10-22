// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! SIMD-Accelerated Decode Pipeline for Native Compute Engine
//!
//! This module provides hardware-accelerated decoding operations for the
//! ProximaDB storage engine. It automatically selects the best available
//! SIMD backend at runtime.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    DecoderFactory                           │
//! │  (Selects best backend: AVX2 > NEON > SSE > Scalar)        │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    SimdDecoder Trait                        │
//! │  - decode_bitpacked_i64()                                   │
//! │  - decode_bitpacked_i32()                                   │
//! │  - decode_delta_f32()                                       │
//! │  - decode_delta_i64()                                       │
//! └─────────────────────────────────────────────────────────────┘
//!          │              │              │              │
//!          ▼              ▼              ▼              ▼
//!     ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
//!     │  AVX2  │    │  NEON  │    │  SSE   │    │ Scalar │
//!     │ x86_64 │    │ ARM64  │    │ x86_64 │    │  All   │
//!     └────────┘    └────────┘    └────────┘    └────────┘
//! ```
//!
//! # Module Structure
//!
//! - **traits.rs**: Core `SimdDecoder` trait and `DecoderFactory`
//! - **common.rs**: Scalar fallbacks shared by every target (variable-bit
//!   extraction, delta-header parsing, prefix sum, f32 reconstruction)
//! - **bitpacked_scalar.rs**: Portable scalar fallback (reference implementation)
//! - **bitpacked_avx2.rs**: AVX2 implementation for x86_64
//! - **bitpacked_neon.rs**: NEON implementation for ARM64
//! - **delta_simd.rs**: Delta encoding with SIMD prefix sum
//! - **fused_quantization.rs**: Fused Binary/INT4/INT8 -> FP32 pipeline
//! - **test_support.rs**: Golden-vector battery shared by all per-target tests
//!   (test-only)
//!
//! # Usage
//!
//! ```rust,ignore
//! use crate::storage::engines::core::ops::simd_decode::{DecoderFactory, SimdDecoder};
//!
//! // Get the best available decoder
//! let decoder = DecoderFactory::best_available();
//! println!("Using {} decoder", decoder.name());
//!
//! // Decode bitpacked data
//! let mut output = vec![0i64; 1000];
//! let count = decoder.decode_bitpacked_i64(&packed_data, 8, &mut output)?;
//!
//! // Or use specific functions directly
//! use crate::storage::engines::core::ops::simd_decode::fused_quantization::*;
//!
//! let mut f32_output = vec![0.0f32; 1000];
//! fused_decode_int8_to_f32(&int8_data, &mut f32_output, &params)?;
//! ```
//!
//! # Performance
//!
//! | Operation        | Scalar   | AVX2     | NEON     |
//! |-----------------|----------|----------|----------|
//! | 8-bit decode    | 1.0x     | 6-8x     | 3-4x     |
//! | 16-bit decode   | 1.0x     | 6-8x     | 3-4x     |
//! | 32-bit decode   | 1.0x     | 6-8x     | 3-4x     |
//! | Delta decode    | 1.0x     | 3-5x     | 2-4x     |
//! | INT8 -> FP32    | 1.0x     | 4-6x     | 2-4x     |
//!
//! # SOLID Principles
//!
//! - **S**ingle Responsibility: Each decoder handles one platform
//! - **O**pen/Closed: Add new decoders by implementing SimdDecoder trait
//! - **L**iskov Substitution: All decoders are interchangeable via trait
//! - **I**nterface Segregation: Minimal trait with essential operations
//! - **D**ependency Inversion: Use DecoderFactory to get Arc<dyn SimdDecoder>
//!
//! # TDD Testing
//!
//! All SIMD implementations are tested against the scalar reference:
//!
//! ```rust,ignore
//! #[test]
//! fn test_simd_vs_scalar_equivalence() {
//!     let simd_decoder = DecoderFactory::best_available();
//!     let scalar_decoder = DecoderFactory::scalar();
//!
//!     let packed = create_test_data();
//!
//!     let mut simd_output = vec![0i64; 100];
//!     let mut scalar_output = vec![0i64; 100];
//!
//!     simd_decoder.decode_bitpacked_i64(&packed, 8, &mut simd_output)?;
//!     scalar_decoder.decode_bitpacked_i64(&packed, 8, &mut scalar_output)?;
//!
//!     assert_eq!(simd_output, scalar_output);
//! }
//! ```

// Sub-modules
pub mod bitpacked_scalar;
mod common;
pub mod delta_simd;
pub mod fused_quantization;
mod traits;

#[cfg(test)]
pub mod test_support;

// Platform-specific modules
#[cfg(target_arch = "x86_64")]
pub mod bitpacked_avx2;

#[cfg(target_arch = "aarch64")]
pub mod bitpacked_neon;

// Re-exports for convenient access
pub use bitpacked_scalar::ScalarDecoder;
pub use traits::{CpuFeatures, DecoderFactory, SimdDecoder};

#[cfg(target_arch = "x86_64")]
pub use bitpacked_avx2::Avx2Decoder;

#[cfg(target_arch = "aarch64")]
pub use bitpacked_neon::NeonDecoder;

// Delta decode functions
pub use delta_simd::{delta_decode_f32, delta_decode_i32_prefix_sum, delta_decode_i64_prefix_sum};

// Fused quantization functions
pub use fused_quantization::{
    QuantizationParams, fused_decode_binary_to_f32, fused_decode_int4_to_f32,
    fused_decode_int8_to_f32, progressive_decode_binary_int8_f32,
};

/// Get the best available decoder for the current platform
///
/// This is a convenience function that wraps `DecoderFactory::best_available()`.
///
/// # Example
/// ```rust,ignore
/// let decoder = simd_decode::best_decoder();
/// println!("Using {} with {:?}", decoder.name(), decoder.supported_features());
/// ```
pub fn best_decoder() -> std::sync::Arc<dyn SimdDecoder> {
    DecoderFactory::best_available()
}

/// Check if SIMD acceleration is available
///
/// Returns true if any SIMD instruction set (AVX2, NEON, SSE) is available.
pub fn has_simd_support() -> bool {
    CpuFeatures::detect().has_simd()
}

/// Get detected CPU features
pub fn detected_features() -> CpuFeatures {
    CpuFeatures::detect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_best_decoder_returns_valid() {
        let decoder = best_decoder();
        assert!(!decoder.name().is_empty());
        println!(
            "Best decoder: {} ({:?})",
            decoder.name(),
            decoder.supported_features()
        );
    }

    #[test]
    fn test_has_simd_support() {
        let has_simd = has_simd_support();
        let features = detected_features();

        println!("SIMD support: {}", has_simd);
        println!("Features: {:?}", features);

        // Verify consistency
        assert_eq!(has_simd, features.has_simd());
    }

    #[test]
    fn test_decoder_factory_scalar() {
        let decoder = DecoderFactory::scalar();
        assert_eq!(decoder.name(), "Scalar");
        assert!(!decoder.has_acceleration());
    }

    #[test]
    fn test_simd_vs_scalar_correctness() {
        // The best-available decoder must match the scalar reference on the
        // shared golden bit-width sweep.
        test_support::assert_matches_scalar_reference(best_decoder().as_ref());
    }

    #[test]
    fn test_delta_functions() {
        // Test delta decode
        let base = 1.0f32;
        let values = [1.0f32, 2.0, 3.0, 4.0];
        let base_bits = base.to_bits() as i64;
        let deltas: Vec<i64> = values
            .iter()
            .map(|&v| (v.to_bits() as i64) - base_bits)
            .collect();

        let mut output = vec![0.0f32; 4];
        let count = delta_decode_f32(&deltas, base, &mut output).unwrap();

        assert_eq!(count, 4);
        for (i, (&expected, &actual)) in values.iter().zip(output.iter()).enumerate() {
            assert!(
                (expected - actual).abs() < 1e-6,
                "Delta decode mismatch at {}: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_fused_quantization_functions() {
        // Test INT8 -> FP32
        let input = [0u8, 127, 128, 255]; // 0, 127, -128, -1 as signed
        let mut output = vec![0.0f32; 4];
        let params = QuantizationParams::symmetric(0.01);

        let count = fused_decode_int8_to_f32(&input, &mut output, &params).unwrap();
        assert_eq!(count, 4);

        let expected = [0.0f32, 1.27, -1.28, -0.01];
        for (i, (&e, &a)) in expected.iter().zip(output.iter()).enumerate() {
            assert!(
                (e - a).abs() < 1e-4,
                "INT8->FP32 mismatch at {}: expected {}, got {}",
                i,
                e,
                a
            );
        }
    }

    #[test]
    fn test_binary_decode() {
        let input = [0b11110000u8];
        let mut output = vec![0.0f32; 8];

        let count = fused_decode_binary_to_f32(&input, &mut output, true).unwrap();
        assert_eq!(count, 8);

        // First 4 bits are 0 (bipolar: -1), next 4 are 1 (bipolar: +1)
        for i in 0..4 {
            assert_eq!(output[i], -1.0, "Binary decode mismatch at {}", i);
        }
        for i in 4..8 {
            assert_eq!(output[i], 1.0, "Binary decode mismatch at {}", i);
        }
    }
}
