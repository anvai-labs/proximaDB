// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Baseline encoding/decoding algorithms — pure-Rust, no SIMD/GPU.
//!
//! Contains the reference implementations for all encoding schemes.
//! Used as fallback when SIMD/GPU is unavailable and as ground truth for tests.

pub mod decoder;
pub mod encoder;
pub mod functions;

pub use decoder::BaselineDecoder;
pub use encoder::BaselineEncoder;
pub use functions::*;

// Encoder and decoder advertise the same scheme-level capability.
fn supports_scheme(scheme: &crate::types::ProximaScheme) -> bool {
    use crate::types::ProximaScheme;

    matches!(
        scheme,
        ProximaScheme::Raw
            | ProximaScheme::Delta { .. }
            | ProximaScheme::BitPacked { .. }
            | ProximaScheme::FrameOfReference { .. }
            | ProximaScheme::SparseBitmap
            | ProximaScheme::SparseCOO
            | ProximaScheme::RunLength
            | ProximaScheme::PForDelta { .. }
            | ProximaScheme::Zigzag { .. }
            | ProximaScheme::DoubleDelta { .. }
            | ProximaScheme::PForDoubleDelta { .. }
            | ProximaScheme::Gorilla
            | ProximaScheme::VByte
            | ProximaScheme::Dictionary
            | ProximaScheme::Simple8b
            | ProximaScheme::Adaptive
    )
}
