// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! SIMD Decoder - Hardware-accelerated decoding using NEON/AVX2/AVX512
//!
//! This decoder uses SIMD intrinsics to accelerate decoding operations.
//! It automatically detects the best available SIMD backend (NEON on ARM,
//! AVX2/AVX512 on x86_64) and falls back to scalar if SIMD is unavailable.

use anyhow::Result;
use tracing::trace;

use crate::simd::simd_bitpack_decode_f32;
use crate::traits::RawDecoder;
use crate::types::ProximaScheme;

/// SIMD-accelerated decoder
///
/// Supports:
/// - Delta decoding (NEON/AVX2)
/// - BitPacked decoding (AVX2)
///
/// Falls back to baseline implementation for unsupported schemes.
pub struct SimdDecoder;

impl RawDecoder for SimdDecoder {
    fn name(&self) -> &'static str {
        "SIMD"
    }

    fn supports(&self, scheme: &ProximaScheme) -> bool {
        super::supports_scheme(scheme)
    }

    fn decode_f32(&self, data: &[u8], scheme: &ProximaScheme, count: usize) -> Result<Vec<f32>> {
        match scheme {
            ProximaScheme::Delta { .. } => {
                trace!(
                    "🚀 [SIMD] Decoding {} bytes with Delta (count={})",
                    data.len(),
                    count
                );

                // Use baseline implementation for wire format compatibility
                use crate::impls::baseline::functions::delta;
                let values = delta::decode_f32(data, count)?;

                trace!(
                    "✅ [SIMD] Delta decoded {} bytes → {} values (using baseline format)",
                    data.len(),
                    values.len()
                );
                Ok(values)
            }

            ProximaScheme::BitPacked { bits } => {
                trace!(
                    "🚀 [SIMD] Decoding {} bytes with BitPacked ({}b/val, count={})",
                    data.len(),
                    bits,
                    count
                );

                // Decode using SIMD BitPacked
                let values = simd_bitpack_decode_f32(data, *bits, count)?;

                trace!(
                    "✅ [SIMD] BitPacked decoded {} bytes → {} values",
                    data.len(),
                    values.len()
                );
                Ok(values)
            }

            ProximaScheme::FrameOfReference { reference, bits } => {
                trace!(
                    "🚀 [SIMD] Decoding {} bytes with FrameOfReference (ref={}, {}b/val, count={})",
                    data.len(),
                    reference,
                    bits,
                    count
                );

                // Decode using SIMD FrameOfReference
                let values = crate::simd::simd_frame_of_reference_decode_f32(
                    data, *reference, *bits, count,
                )?;

                trace!(
                    "✅ [SIMD] FrameOfReference decoded {} bytes → {} values",
                    data.len(),
                    values.len()
                );
                Ok(values)
            }

            ProximaScheme::Zigzag { bits } => {
                trace!(
                    "🚀 [SIMD] Decoding {} bytes with Zigzag ({}b/val, count={})",
                    data.len(),
                    bits,
                    count
                );

                // Decode using SIMD Zigzag
                let values = crate::simd::simd_zigzag_decode_f32(data, *bits, count)?;

                trace!(
                    "✅ [SIMD] Zigzag decoded {} bytes → {} values",
                    data.len(),
                    values.len()
                );
                Ok(values)
            }

            ProximaScheme::PForDelta {
                majority_bits,
                base,
            } => {
                trace!(
                    "🚀 [SIMD] Decoding {} bytes with PForDelta ({}b majority, base={}, count={})",
                    data.len(),
                    majority_bits,
                    base,
                    count
                );

                // Decode using SIMD PForDelta
                let values =
                    crate::simd::simd_pfor_delta_decode_f32(data, *majority_bits, *base, count)?;

                trace!(
                    "✅ [SIMD] PForDelta decoded {} bytes → {} values",
                    data.len(),
                    values.len()
                );
                Ok(values)
            }

            _ => {
                anyhow::bail!("SIMD decoder does not support scheme: {}", scheme.name())
            }
        }
    }

    fn decode_i64(&self, data: &[u8], scheme: &ProximaScheme, count: usize) -> Result<Vec<i64>> {
        match scheme {
            ProximaScheme::Delta { .. } => {
                trace!(
                    "🚀 [SIMD] Decoding {} bytes as i64 with Delta (count={})",
                    data.len(),
                    count
                );

                // Use baseline implementation for wire format compatibility
                use crate::impls::baseline::functions::delta;
                let values = delta::decode_i64(data, count)?;

                trace!(
                    "✅ [SIMD] Delta decoded {} bytes → {} values (using baseline format)",
                    data.len(),
                    values.len()
                );
                Ok(values)
            }

            _ => {
                anyhow::bail!(
                    "SIMD decoder does not support scheme: {} for i64",
                    scheme.name()
                )
            }
        }
    }

    fn decode_i32(&self, data: &[u8], scheme: &ProximaScheme, count: usize) -> Result<Vec<i32>> {
        match scheme {
            ProximaScheme::Delta { .. } => {
                trace!(
                    "🚀 [SIMD] Decoding {} bytes as i32 with Delta (count={})",
                    data.len(),
                    count
                );

                // Use baseline implementation for wire format compatibility
                use crate::impls::baseline::functions::delta;
                let values = delta::decode_i32(data, count)?;

                trace!(
                    "✅ [SIMD] Delta decoded {} bytes → {} values (using baseline format)",
                    data.len(),
                    values.len()
                );
                Ok(values)
            }

            _ => {
                anyhow::bail!(
                    "SIMD decoder does not support scheme: {} for i32",
                    scheme.name()
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_decoder_supports() {
        let decoder = SimdDecoder;

        // Should support: Delta, BitPacked, FrameOfReference, Zigzag, PForDelta
        assert!(decoder.supports(&ProximaScheme::Delta { base: 0 }));
        assert!(decoder.supports(&ProximaScheme::BitPacked { bits: 8 }));
        assert!(decoder.supports(&ProximaScheme::FrameOfReference {
            reference: 0,
            bits: 8
        }));
        assert!(decoder.supports(&ProximaScheme::Zigzag { bits: 8 }));
        assert!(decoder.supports(&ProximaScheme::PForDelta {
            majority_bits: 8,
            base: 0
        }));

        // Should not support other schemes
        assert!(!decoder.supports(&ProximaScheme::RunLength));
        assert!(!decoder.supports(&ProximaScheme::Dictionary));
        assert!(!decoder.supports(&ProximaScheme::Simple8b));
        assert!(!decoder.supports(&ProximaScheme::VByte));
    }

    #[test]
    fn test_simd_delta_decode() {
        use crate::impls::simd::encoder::SimdEncoder;
        use crate::traits::RawEncoder;

        let encoder = SimdEncoder;
        let decoder = SimdDecoder;
        // Use 32 values minimum to demonstrate compression benefits
        let values: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();

        // Encode
        let encoded = encoder
            .encode_f32(&values, &ProximaScheme::Delta { base: 0 })
            .unwrap();

        // Decode (count = number of values)
        let decoded = decoder
            .decode_f32(&encoded, &ProximaScheme::Delta { base: 0 }, values.len())
            .unwrap();

        // Verify round-trip
        assert_eq!(values, decoded);
    }

    #[test]
    fn test_simd_bitpack_decode() {
        use crate::impls::simd::encoder::SimdEncoder;
        use crate::traits::RawEncoder;

        let encoder = SimdEncoder;
        let decoder = SimdDecoder;
        let values = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        // Encode with 32 bits for lossless f32 (BitPacked with bits < 32 is lossy)
        let encoded = encoder
            .encode_f32(&values, &ProximaScheme::BitPacked { bits: 32 })
            .unwrap();

        // Decode (count = number of values)
        let decoded = decoder
            .decode_f32(
                &encoded,
                &ProximaScheme::BitPacked { bits: 32 },
                values.len(),
            )
            .unwrap();

        // Verify lossless round-trip
        assert_eq!(values, decoded);
    }
}
