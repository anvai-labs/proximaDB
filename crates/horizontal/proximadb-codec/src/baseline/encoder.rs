// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Baseline encoder - Pure Rust implementation (always available)
//!
//! DEFERRED: Phase 2.1 - Extract encoding functions from old ProximaEncoder

use crate::traits::RawEncoder;
use crate::types::ProximaScheme;
use anyhow::Result;

use crate::functions;

/// Baseline encoder - Pure Rust, no SIMD, always available
pub struct BaselineEncoder;

impl RawEncoder for BaselineEncoder {
    fn name(&self) -> &'static str {
        "baseline"
    }

    fn supports(&self, scheme: &ProximaScheme) -> bool {
        super::supports_scheme(scheme)
    }

    fn encode_f32(&self, values: &[f32], scheme: &ProximaScheme) -> Result<Vec<u8>> {
        match scheme {
            ProximaScheme::Delta { base } => functions::delta::encode_f32(values, *base),
            ProximaScheme::BitPacked { bits } => functions::bitpack::encode_f32(values, *bits),
            ProximaScheme::FrameOfReference { reference, .. } => {
                functions::frame_of_ref::encode_f32(values, *reference)
            }
            ProximaScheme::SparseBitmap => functions::sparse_bitmap::encode_f32(values),
            ProximaScheme::SparseCOO => functions::sparse_coo::encode_f32(values),
            ProximaScheme::Raw => functions::raw::encode_f32(values),
            ProximaScheme::RunLength => functions::run_length::encode_f32(values),
            ProximaScheme::PForDelta { base, .. } => {
                functions::pfor_delta::encode_f32(values, *base)
            }
            ProximaScheme::Zigzag { bits } => functions::zigzag::encode_f32(values, *bits),
            ProximaScheme::DoubleDelta { .. } => functions::double_delta::encode_f32(values),
            ProximaScheme::PForDoubleDelta { base, .. } => {
                functions::pfor_double_delta::encode_f32(values, *base)
            }
            ProximaScheme::Gorilla => functions::gorilla::encode_f32(values),
            ProximaScheme::VByte => functions::vbyte::encode_f32(values),
            ProximaScheme::Dictionary => functions::dictionary::encode_f32(values),
            ProximaScheme::Simple8b => functions::simple8b::encode_f32(values),
            ProximaScheme::Sq8 | ProximaScheme::RaBitQ => Err(anyhow::anyhow!(
                "quantized vector scheme needs per-column params; use the PAX block path"
            )),
            ProximaScheme::Adaptive => functions::adaptive::encode_f32(values),
        }
    }

    fn encode_i64(&self, values: &[i64], scheme: &ProximaScheme) -> Result<Vec<u8>> {
        match scheme {
            ProximaScheme::Delta { base } => functions::delta::encode_i64(values, *base),
            ProximaScheme::BitPacked { bits } => functions::bitpack::encode_i64(values, *bits),
            ProximaScheme::FrameOfReference { reference, .. } => {
                functions::frame_of_ref::encode_i64(values, *reference)
            }
            ProximaScheme::SparseBitmap => functions::sparse_bitmap::encode_i64(values),
            ProximaScheme::SparseCOO => functions::sparse_coo::encode_i64(values),
            ProximaScheme::Raw => functions::raw::encode_i64(values),
            ProximaScheme::RunLength => functions::run_length::encode_i64(values),
            ProximaScheme::PForDelta { base, .. } => {
                functions::pfor_delta::encode_i64(values, *base)
            }
            ProximaScheme::Zigzag { bits } => functions::zigzag::encode_i64(values, *bits),
            ProximaScheme::DoubleDelta { .. } => functions::double_delta::encode_i64(values),
            ProximaScheme::PForDoubleDelta { base, .. } => {
                functions::pfor_double_delta::encode_i64(values, *base)
            }
            ProximaScheme::Gorilla => functions::gorilla::encode_i64(values),
            ProximaScheme::VByte => functions::vbyte::encode_i64(values),
            ProximaScheme::Dictionary => functions::dictionary::encode_i64(values),
            ProximaScheme::Simple8b => functions::simple8b::encode_i64(values),
            ProximaScheme::Sq8 | ProximaScheme::RaBitQ => Err(anyhow::anyhow!(
                "quantized vector scheme needs per-column params; use the PAX block path"
            )),
            ProximaScheme::Adaptive => functions::adaptive::encode_i64(values),
        }
    }

    fn encode_i32(&self, values: &[i32], scheme: &ProximaScheme) -> Result<Vec<u8>> {
        match scheme {
            ProximaScheme::Delta { base } => functions::delta::encode_i32(values, *base),
            ProximaScheme::BitPacked { bits } => functions::bitpack::encode_i32(values, *bits),
            ProximaScheme::FrameOfReference { reference, .. } => {
                functions::frame_of_ref::encode_i32(values, *reference)
            }
            ProximaScheme::SparseBitmap => functions::sparse_bitmap::encode_i32(values),
            ProximaScheme::SparseCOO => functions::sparse_coo::encode_i32(values),
            ProximaScheme::Raw => functions::raw::encode_i32(values),
            ProximaScheme::RunLength => functions::run_length::encode_i32(values),
            ProximaScheme::PForDelta { base, .. } => {
                functions::pfor_delta::encode_i32(values, *base)
            }
            ProximaScheme::Zigzag { bits } => functions::zigzag::encode_i32(values, *bits),
            ProximaScheme::DoubleDelta { .. } => functions::double_delta::encode_i32(values),
            ProximaScheme::PForDoubleDelta { base, .. } => {
                functions::pfor_double_delta::encode_i32(values, *base)
            }
            ProximaScheme::Gorilla => functions::gorilla::encode_i32(values),
            ProximaScheme::VByte => functions::vbyte::encode_i32(values),
            ProximaScheme::Dictionary => functions::dictionary::encode_i32(values),
            ProximaScheme::Simple8b => functions::simple8b::encode_i32(values),
            ProximaScheme::Sq8 | ProximaScheme::RaBitQ => Err(anyhow::anyhow!(
                "quantized vector scheme needs per-column params; use the PAX block path"
            )),
            ProximaScheme::Adaptive => functions::adaptive::encode_i32(values),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_supports() {
        let encoder = BaselineEncoder;

        // Should support all implemented schemes
        assert!(encoder.supports(&ProximaScheme::Raw));
        assert!(encoder.supports(&ProximaScheme::Delta { base: 0 }));
        assert!(encoder.supports(&ProximaScheme::BitPacked { bits: 16 }));
        assert!(encoder.supports(&ProximaScheme::FrameOfReference {
            reference: 0,
            bits: 16
        }));
        assert!(encoder.supports(&ProximaScheme::SparseBitmap));
        assert!(encoder.supports(&ProximaScheme::SparseCOO));
        assert!(encoder.supports(&ProximaScheme::RunLength));
        assert!(encoder.supports(&ProximaScheme::PForDelta {
            majority_bits: 16,
            base: 0
        }));
        assert!(encoder.supports(&ProximaScheme::Zigzag { bits: 32 }));
        assert!(encoder.supports(&ProximaScheme::DoubleDelta {
            first_value: 0,
            first_delta: 0
        }));
        assert!(encoder.supports(&ProximaScheme::Gorilla));
        assert!(encoder.supports(&ProximaScheme::VByte));
        assert!(encoder.supports(&ProximaScheme::Dictionary));
        assert!(encoder.supports(&ProximaScheme::Simple8b));
        assert!(encoder.supports(&ProximaScheme::Adaptive));
    }

    #[test]
    fn test_encode_f32_delta() {
        let encoder = BaselineEncoder;
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let scheme = ProximaScheme::Delta { base: 0 };

        let result = encoder.encode_f32(&values, &scheme);
        assert!(result.is_ok());

        let encoded = result.unwrap();
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_encode_i64_delta() {
        let encoder = BaselineEncoder;
        let values = vec![100i64, 200, 300, 400];
        let scheme = ProximaScheme::Delta { base: 0 };

        let result = encoder.encode_i64(&values, &scheme);
        assert!(result.is_ok());
    }

    #[test]
    fn test_encode_f32_bitpacked() {
        let encoder = BaselineEncoder;
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let scheme = ProximaScheme::BitPacked { bits: 32 }; // Full precision

        let result = encoder.encode_f32(&values, &scheme);
        assert!(result.is_ok());

        let encoded = result.unwrap();
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_encode_i64_bitpacked() {
        let encoder = BaselineEncoder;
        let values = vec![100i64, 200, 300, 400];
        let scheme = ProximaScheme::BitPacked { bits: 16 }; // 16-bit packing

        let result = encoder.encode_i64(&values, &scheme);
        assert!(result.is_ok());
    }

    #[test]
    fn test_encode_i32_bitpacked() {
        let encoder = BaselineEncoder;
        let values = vec![1i32, 2, 3, 4, 5];
        let scheme = ProximaScheme::BitPacked { bits: 8 }; // 8-bit packing

        let result = encoder.encode_i32(&values, &scheme);
        assert!(result.is_ok());
    }

    #[test]
    fn test_encode_f32_for() {
        let encoder = BaselineEncoder;
        let values = vec![100.0f32, 101.0, 102.0, 103.0];
        let scheme = ProximaScheme::FrameOfReference {
            reference: 100.0f32.to_bits() as i64,
            bits: 32,
        };

        let result = encoder.encode_f32(&values, &scheme);
        assert!(result.is_ok());

        let encoded = result.unwrap();
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_encode_i64_for() {
        let encoder = BaselineEncoder;
        let values = vec![1000i64, 1001, 1002, 1003];
        let scheme = ProximaScheme::FrameOfReference {
            reference: 1000,
            bits: 16,
        };

        let result = encoder.encode_i64(&values, &scheme);
        assert!(result.is_ok());
    }

    #[test]
    fn test_encode_i32_for() {
        let encoder = BaselineEncoder;
        let values = vec![500i32, 501, 502, 503];
        let scheme = ProximaScheme::FrameOfReference {
            reference: 500,
            bits: 16,
        };

        let result = encoder.encode_i32(&values, &scheme);
        assert!(result.is_ok());
    }
}
