// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Baseline decoder - Pure Rust implementation (always available)
//!
//! Phase 2.5 - Implemented using raw decoding functions

use crate::traits::RawDecoder;
use crate::types::ProximaScheme;
use anyhow::Result;

use crate::functions;

/// Baseline decoder - Pure Rust, no SIMD, always available
pub struct BaselineDecoder;

impl RawDecoder for BaselineDecoder {
    fn name(&self) -> &'static str {
        "baseline"
    }

    fn supports(&self, scheme: &ProximaScheme) -> bool {
        super::supports_scheme(scheme)
    }

    fn decode_f32(&self, data: &[u8], scheme: &ProximaScheme, count: usize) -> Result<Vec<f32>> {
        match scheme {
            ProximaScheme::Delta { .. } => functions::delta::decode_f32(data, count),
            ProximaScheme::BitPacked { bits } => functions::bitpack::decode_f32(data, *bits, count),
            ProximaScheme::FrameOfReference { .. } => {
                functions::frame_of_ref::decode_f32(data, count)
            }
            ProximaScheme::SparseBitmap => functions::sparse_bitmap::decode_f32(data, count),
            ProximaScheme::SparseCOO => functions::sparse_coo::decode_f32(data, count),
            ProximaScheme::Raw => functions::raw::decode_f32(data),
            ProximaScheme::RunLength => functions::run_length::decode_f32(data),
            ProximaScheme::PForDelta { .. } => functions::pfor_delta::decode_f32(data, count),
            ProximaScheme::Zigzag { .. } => functions::zigzag::decode_f32(data, count),
            ProximaScheme::DoubleDelta { .. } => functions::double_delta::decode_f32(data, count),
            ProximaScheme::PForDoubleDelta { .. } => {
                functions::pfor_double_delta::decode_f32(data, count)
            }
            ProximaScheme::Gorilla => functions::gorilla::decode_f32(data, count),
            ProximaScheme::VByte => functions::vbyte::decode_f32(data, count),
            ProximaScheme::Dictionary => functions::dictionary::decode_f32(data, count),
            ProximaScheme::Simple8b => functions::simple8b::decode_f32(data, count),
            ProximaScheme::Sq8 | ProximaScheme::RaBitQ => Err(anyhow::anyhow!(
                "quantized vector scheme needs per-column params; use the PAX block path"
            )),
            ProximaScheme::Adaptive => functions::adaptive::decode_f32(data, count),
        }
    }

    fn decode_i64(&self, data: &[u8], scheme: &ProximaScheme, count: usize) -> Result<Vec<i64>> {
        match scheme {
            ProximaScheme::Delta { .. } => functions::delta::decode_i64(data, count),
            ProximaScheme::BitPacked { bits } => functions::bitpack::decode_i64(data, *bits, count),
            ProximaScheme::FrameOfReference { .. } => {
                functions::frame_of_ref::decode_i64(data, count)
            }
            ProximaScheme::SparseBitmap => functions::sparse_bitmap::decode_i64(data, count),
            ProximaScheme::SparseCOO => functions::sparse_coo::decode_i64(data, count),
            ProximaScheme::Raw => functions::raw::decode_i64(data),
            ProximaScheme::RunLength => functions::run_length::decode_i64(data),
            ProximaScheme::PForDelta { .. } => functions::pfor_delta::decode_i64(data, count),
            ProximaScheme::Zigzag { .. } => functions::zigzag::decode_i64(data, count),
            ProximaScheme::DoubleDelta { .. } => functions::double_delta::decode_i64(data, count),
            ProximaScheme::PForDoubleDelta { .. } => {
                functions::pfor_double_delta::decode_i64(data, count)
            }
            ProximaScheme::Gorilla => functions::gorilla::decode_i64(data, count),
            ProximaScheme::VByte => functions::vbyte::decode_i64(data, count),
            ProximaScheme::Dictionary => functions::dictionary::decode_i64(data, count),
            ProximaScheme::Simple8b => functions::simple8b::decode_i64(data, count),
            ProximaScheme::Sq8 | ProximaScheme::RaBitQ => Err(anyhow::anyhow!(
                "quantized vector scheme needs per-column params; use the PAX block path"
            )),
            ProximaScheme::Adaptive => functions::adaptive::decode_i64(data, count),
        }
    }

    fn decode_i32(&self, data: &[u8], scheme: &ProximaScheme, count: usize) -> Result<Vec<i32>> {
        match scheme {
            ProximaScheme::Delta { .. } => functions::delta::decode_i32(data, count),
            ProximaScheme::BitPacked { bits } => functions::bitpack::decode_i32(data, *bits, count),
            ProximaScheme::FrameOfReference { .. } => {
                functions::frame_of_ref::decode_i32(data, count)
            }
            ProximaScheme::SparseBitmap => functions::sparse_bitmap::decode_i32(data, count),
            ProximaScheme::SparseCOO => functions::sparse_coo::decode_i32(data, count),
            ProximaScheme::Raw => functions::raw::decode_i32(data),
            ProximaScheme::RunLength => functions::run_length::decode_i32(data),
            ProximaScheme::PForDelta { .. } => functions::pfor_delta::decode_i32(data, count),
            ProximaScheme::Zigzag { .. } => functions::zigzag::decode_i32(data, count),
            ProximaScheme::DoubleDelta { .. } => functions::double_delta::decode_i32(data, count),
            ProximaScheme::PForDoubleDelta { .. } => {
                functions::pfor_double_delta::decode_i32(data, count)
            }
            ProximaScheme::Gorilla => functions::gorilla::decode_i32(data, count),
            ProximaScheme::VByte => functions::vbyte::decode_i32(data, count),
            ProximaScheme::Dictionary => functions::dictionary::decode_i32(data, count),
            ProximaScheme::Simple8b => functions::simple8b::decode_i32(data, count),
            ProximaScheme::Sq8 | ProximaScheme::RaBitQ => Err(anyhow::anyhow!(
                "quantized vector scheme needs per-column params; use the PAX block path"
            )),
            ProximaScheme::Adaptive => functions::adaptive::decode_i32(data, count),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_supports() {
        let decoder = BaselineDecoder;

        // Should support all implemented schemes
        assert!(decoder.supports(&ProximaScheme::Raw));
        assert!(decoder.supports(&ProximaScheme::Delta { base: 0 }));
        assert!(decoder.supports(&ProximaScheme::BitPacked { bits: 16 }));
        assert!(decoder.supports(&ProximaScheme::FrameOfReference {
            reference: 0,
            bits: 16
        }));
        assert!(decoder.supports(&ProximaScheme::SparseBitmap));
        assert!(decoder.supports(&ProximaScheme::SparseCOO));
        assert!(decoder.supports(&ProximaScheme::RunLength));
        assert!(decoder.supports(&ProximaScheme::PForDelta {
            majority_bits: 16,
            base: 0
        }));
        assert!(decoder.supports(&ProximaScheme::Zigzag { bits: 32 }));
        assert!(decoder.supports(&ProximaScheme::DoubleDelta {
            first_value: 0,
            first_delta: 0
        }));
        assert!(decoder.supports(&ProximaScheme::Gorilla));
        assert!(decoder.supports(&ProximaScheme::VByte));
        assert!(decoder.supports(&ProximaScheme::Dictionary));
        assert!(decoder.supports(&ProximaScheme::Simple8b));
        assert!(decoder.supports(&ProximaScheme::Adaptive));
    }

    #[test]
    fn test_decode_f32_delta() {
        let decoder = BaselineDecoder;

        // Manually create encoded delta data matching our format:
        // [base:4 bytes][bits:1 byte][bitpacked_deltas...]
        // For values [1.0, 2.0, 3.0, 4.0] with base=0

        // First encode using encoder
        use crate::impls::baseline::encoder::BaselineEncoder;
        use crate::traits::RawEncoder;

        let encoder = BaselineEncoder;
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let scheme = ProximaScheme::Delta { base: 0 };

        let encoded = encoder
            .encode_f32(&values, &scheme)
            .unwrap_or_else(|e| panic!("Failed to encode f32 values: {}", e));

        // Now decode
        let result = decoder.decode_f32(&encoded, &scheme, values.len());
        assert!(result.is_ok());

        let decoded = result.expect("Decoding f32 should succeed after is_ok() check");
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_decode_i64_delta() {
        let decoder = BaselineDecoder;

        // First encode using encoder
        use crate::impls::baseline::encoder::BaselineEncoder;
        use crate::traits::RawEncoder;

        let encoder = BaselineEncoder;
        let values = vec![100i64, 200, 300, 400];
        let scheme = ProximaScheme::Delta { base: 0 };

        let encoded = encoder
            .encode_i64(&values, &scheme)
            .unwrap_or_else(|e| panic!("Failed to encode i64 values: {}", e));

        // Now decode
        let result = decoder.decode_i64(&encoded, &scheme, values.len());
        assert!(result.is_ok());

        let decoded = result.expect("Decoding i64 should succeed after is_ok() check");
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_decode_i32_delta() {
        let decoder = BaselineDecoder;

        // First encode using encoder
        use crate::impls::baseline::encoder::BaselineEncoder;
        use crate::traits::RawEncoder;

        let encoder = BaselineEncoder;
        let values = vec![10i32, 20, 30, 40];
        let scheme = ProximaScheme::Delta { base: 0 };

        let encoded = encoder
            .encode_i32(&values, &scheme)
            .unwrap_or_else(|e| panic!("Failed to encode i32 values: {}", e));

        // Now decode
        let result = decoder.decode_i32(&encoded, &scheme, values.len());
        assert!(result.is_ok());

        let decoded = result.expect("Decoding i32 should succeed after is_ok() check");
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_decode_f32_bitpacked() {
        let decoder = BaselineDecoder;

        // First encode using encoder
        use crate::impls::baseline::encoder::BaselineEncoder;
        use crate::traits::RawEncoder;

        let encoder = BaselineEncoder;
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let scheme = ProximaScheme::BitPacked { bits: 32 };

        let encoded = encoder
            .encode_f32(&values, &scheme)
            .unwrap_or_else(|e| panic!("Failed to encode f32 values: {}", e));

        // Now decode
        let result = decoder.decode_f32(&encoded, &scheme, values.len());
        assert!(result.is_ok());

        let decoded = result.expect("Decoding f32 should succeed after is_ok() check");
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_decode_i64_bitpacked() {
        let decoder = BaselineDecoder;

        use crate::impls::baseline::encoder::BaselineEncoder;
        use crate::traits::RawEncoder;

        let encoder = BaselineEncoder;
        let values = vec![100i64, 200, 300, 400];
        let scheme = ProximaScheme::BitPacked { bits: 16 };

        let encoded = encoder
            .encode_i64(&values, &scheme)
            .unwrap_or_else(|e| panic!("Failed to encode i64 values: {}", e));

        // Now decode
        let result = decoder.decode_i64(&encoded, &scheme, values.len());
        assert!(result.is_ok());

        let decoded = result.expect("Decoding i64 should succeed after is_ok() check");
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_decode_i32_bitpacked() {
        let decoder = BaselineDecoder;

        use crate::impls::baseline::encoder::BaselineEncoder;
        use crate::traits::RawEncoder;

        let encoder = BaselineEncoder;
        let values = vec![10i32, 20, 30, 40, 50];
        let scheme = ProximaScheme::BitPacked { bits: 8 };

        let encoded = encoder
            .encode_i32(&values, &scheme)
            .unwrap_or_else(|e| panic!("Failed to encode i32 values: {}", e));

        // Now decode
        let result = decoder.decode_i32(&encoded, &scheme, values.len());
        assert!(result.is_ok());

        let decoded = result.expect("Decoding i32 should succeed after is_ok() check");
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_decode_f32_for() {
        let decoder = BaselineDecoder;

        use crate::impls::baseline::encoder::BaselineEncoder;
        use crate::traits::RawEncoder;

        let encoder = BaselineEncoder;
        let values = vec![100.0f32, 101.0, 102.0, 103.0];
        let scheme = ProximaScheme::FrameOfReference {
            reference: 100.0f32.to_bits() as i64,
            bits: 32,
        };

        let encoded = encoder
            .encode_f32(&values, &scheme)
            .unwrap_or_else(|e| panic!("Failed to encode f32 values: {}", e));

        // Now decode
        let result = decoder.decode_f32(&encoded, &scheme, values.len());
        assert!(result.is_ok());

        let decoded = result.expect("Decoding f32 should succeed after is_ok() check");
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_decode_i64_for() {
        let decoder = BaselineDecoder;

        use crate::impls::baseline::encoder::BaselineEncoder;
        use crate::traits::RawEncoder;

        let encoder = BaselineEncoder;
        let values = vec![1000i64, 1001, 1002, 1003];
        let scheme = ProximaScheme::FrameOfReference {
            reference: 1000,
            bits: 16,
        };

        let encoded = encoder
            .encode_i64(&values, &scheme)
            .unwrap_or_else(|e| panic!("Failed to encode i64 values: {}", e));

        // Now decode
        let result = decoder.decode_i64(&encoded, &scheme, values.len());
        assert!(result.is_ok());

        let decoded = result.expect("Decoding i64 should succeed after is_ok() check");
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_decode_i32_for() {
        let decoder = BaselineDecoder;

        use crate::impls::baseline::encoder::BaselineEncoder;
        use crate::traits::RawEncoder;

        let encoder = BaselineEncoder;
        let values = vec![500i32, 501, 502, 503, 504];
        let scheme = ProximaScheme::FrameOfReference {
            reference: 500,
            bits: 16,
        };

        let encoded = encoder
            .encode_i32(&values, &scheme)
            .unwrap_or_else(|e| panic!("Failed to encode i32 values: {}", e));

        // Now decode
        let result = decoder.decode_i32(&encoded, &scheme, values.len());
        assert!(result.is_ok());

        let decoded = result.expect("Decoding i32 should succeed after is_ok() check");
        assert_eq!(decoded, values);
    }
}
