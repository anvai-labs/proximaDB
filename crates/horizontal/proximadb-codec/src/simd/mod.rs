// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! SIMD-Accelerated Encoding/Decoding for ProximaCodec
//!
//! This module provides hardware-accelerated implementations of encoding schemes
//! using SIMD intrinsics (AVX2, AVX-512, NEON, SSE) for 2-5x performance improvement.
//!
//! ## Architecture
//!
//! ```text
//! ProximaCodec (codec.rs)
//!   ├─> Handles: Wire format, markers, counts, type routing
//!   └─> Delegates to:
//!       ├─> SIMD Encoders (this module) - Hardware-accelerated
//!       └─> Baseline Encoders (baseline/) - Portable fallback
//! ```
//!
//! ## Module Organization
//!
//! - **backend**: Hardware detection and memory pool management
//! - **delta**: Delta encoding/decoding (most common)
//! - **bitpack**: Bitpacking encoding/decoding
//! - **frame_of_reference**: Frame-of-reference encoding/decoding
//! - **zigzag**: Zigzag encoding/decoding
//! - **pfor_delta**: PForDelta encoding/decoding
//! - **double_delta**: Double delta encoding/decoding
//! - **encoder**: SimdEncoder (RawEncoder trait implementation)
//! - **decoder**: SimdDecoder (RawDecoder trait implementation)
//!
//! ## Supported Schemes
//!
//! ### High Priority (Implemented)
//! - ✅ Delta: f32→i64 conversion + delta encoding (2-4x compression)
//! - ✅ BitPacked: Variable bit-width packing (1.5-3x compression)
//! - ✅ PForDelta: Patched frame-of-reference (3-6x compression)
//! - ✅ Zigzag: Signed integer interleaving (2-3x compression)
//! - ✅ DoubleDelta: Delta of deltas for time-series (3-8x compression)
//!
//! ## Design Principles
//!
//! 1. **Raw Data Transformation Only**: Encoders/decoders work on raw byte arrays
//! 2. **Hardware Detection**: Automatic SIMD backend selection (AVX-512 > AVX2 > NEON > SSE > Scalar)
//! 3. **Graceful Fallback**: If SIMD fails, fall back to baseline
//! 4. **Memory Pooling**: Zero-allocation hot paths using VectorMemoryPool

// Backend detection and memory management
pub mod backend;

// SIMD implementation file (contains all the actual SIMD functions)
#[allow(clippy::module_inception)]
pub mod simd;

// Organized re-exports for better API navigation
pub mod bitpack;
pub mod delta;
pub mod double_delta;
pub mod frame_of_reference;
pub mod pfor_delta;
pub mod zigzag;

// Encoder/decoder trait implementations
pub mod decoder;
pub mod encoder;

// Re-export all SIMD functions from simd.rs for backward compatibility
pub use simd::{
    get_cached_backend,
    // Backend functions
    get_simd_backend,
    get_simd_info,
    has_simd_support,
    simd_bitpack_decode_f32,
    // Bitpacking
    simd_bitpack_encode_f32,
    simd_delta_decode_f32,
    // Delta encoding
    simd_delta_encode_f32,
    simd_double_delta_decode_f32,
    // Double delta
    simd_double_delta_encode_f32,
    simd_frame_of_reference_decode_f32,
    // Frame-of-reference
    simd_frame_of_reference_encode_f32,
    simd_pfor_delta_decode_f32,
    // PForDelta
    simd_pfor_delta_encode_f32,
    simd_zigzag_decode_f32,
    // Zigzag
    simd_zigzag_encode_f32,
};

// Re-export backend module for direct access
pub use backend as hardware;

// Re-export encoder/decoder types
pub use decoder::SimdDecoder;
pub use encoder::SimdEncoder;

// Encoder and decoder advertise the same scheme-level capability.
fn supports_scheme(scheme: &crate::types::ProximaScheme) -> bool {
    use crate::types::ProximaScheme;

    let backend = get_simd_backend();
    if !backend.has_acceleration() {
        return false;
    }

    matches!(
        scheme,
        ProximaScheme::Delta { .. }
            | ProximaScheme::BitPacked { .. }
            | ProximaScheme::FrameOfReference { .. }
            | ProximaScheme::Zigzag { .. }
            | ProximaScheme::PForDelta { .. }
    )
}
