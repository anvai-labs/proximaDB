// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Scalar Fallback Decoder
//!
//! This module provides a portable scalar implementation of the SIMD decoder
//! that works on all platforms. It serves as:
//!
//! 1. **Fallback**: Used when no SIMD acceleration is available
//! 2. **Reference**: Correctness baseline for SIMD implementations
//! 3. **Testing**: Used to verify SIMD implementations produce identical results
//!
//! # Performance
//!
//! The scalar decoder processes one value at a time. While slower than SIMD,
//! it handles all edge cases correctly and serves as the reference implementation.
//!
//! # Bit Width Support
//!
//! Supports all bit widths from 1-64 for i64 output and 1-32 for i32 output.

use anyhow::{Result, bail};

use super::traits::{CpuFeatures, SimdDecoder};

/// Scalar (non-SIMD) decoder implementation
///
/// This is the fallback decoder used when no SIMD acceleration is available.
/// It processes values one at a time using standard Rust operations.
#[derive(Debug, Clone, Copy)]
pub struct ScalarDecoder;

impl ScalarDecoder {
    /// Create a new scalar decoder
    pub fn new() -> Self {
        Self
    }
}

impl Default for ScalarDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl SimdDecoder for ScalarDecoder {
    fn decode_bitpacked_i64(
        &self,
        input: &[u8],
        bits_per_value: u8,
        output: &mut [i64],
    ) -> Result<usize> {
        if bits_per_value == 0 {
            bail!("bits_per_value must be > 0");
        }
        if bits_per_value > 64 {
            bail!("bits_per_value must be <= 64, got {}", bits_per_value);
        }

        let bits = bits_per_value as usize;

        // Calculate how many values we can decode from the input
        let total_bits = input.len() * 8;
        let max_values = total_bits / bits;
        let count = max_values.min(output.len());

        if count == 0 {
            return Ok(0);
        }

        // Create mask for extracting bits
        let mask: u64 = if bits == 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };

        // Decode each value
        for (i, out_val) in output.iter_mut().enumerate().take(count) {
            let bit_offset = i * bits;
            let byte_offset = bit_offset / 8;
            let bit_in_byte = bit_offset % 8;

            // Read enough bytes to cover all bits (up to 9 bytes for 64-bit values crossing boundaries)
            let bytes_needed = (bit_in_byte + bits).div_ceil(8);
            if byte_offset + bytes_needed > input.len() {
                // Partial read at end of input
                break;
            }

            // Assemble the value from bytes
            let mut value: u64 = 0;
            for j in 0..bytes_needed.min(8) {
                if byte_offset + j < input.len() {
                    value |= (input[byte_offset + j] as u64) << (j * 8);
                }
            }

            // Handle 9th byte for values that cross 8-byte boundary
            if bytes_needed > 8 && byte_offset + 8 < input.len() {
                let extra_bits = (bit_in_byte + bits).saturating_sub(64);
                let extra_byte = input[byte_offset + 8] as u64;
                // First shift and mask the main 8-byte value
                value >>= bit_in_byte;
                // Then add the extra bits from 9th byte (only the relevant bits)
                if extra_bits > 0 && extra_bits < 64 {
                    value |= (extra_byte & ((1u64 << extra_bits) - 1)) << (bits - extra_bits);
                }
                value &= mask;
            } else {
                // Shift to align and mask
                value = (value >> bit_in_byte) & mask;
            }

            // Sign-extend if the high bit is set (for signed interpretation)
            // Note: The caller decides whether to interpret as signed or unsigned
            *out_val = value as i64;
        }

        Ok(count)
    }

    fn decode_bitpacked_i32(
        &self,
        input: &[u8],
        bits_per_value: u8,
        output: &mut [i32],
    ) -> Result<usize> {
        if bits_per_value == 0 {
            bail!("bits_per_value must be > 0");
        }
        if bits_per_value > 32 {
            bail!(
                "bits_per_value must be <= 32 for i32 output, got {}",
                bits_per_value
            );
        }

        let bits = bits_per_value as usize;

        // Calculate how many values we can decode
        let total_bits = input.len() * 8;
        let max_values = total_bits / bits;
        let count = max_values.min(output.len());

        if count == 0 {
            return Ok(0);
        }

        // Create mask
        let mask: u32 = if bits == 32 {
            u32::MAX
        } else {
            (1u32 << bits) - 1
        };

        // Decode each value
        for (i, out_val) in output.iter_mut().enumerate().take(count) {
            let bit_offset = i * bits;
            let byte_offset = bit_offset / 8;
            let bit_in_byte = bit_offset % 8;

            // Read enough bytes (up to 5 for 32-bit values crossing boundaries)
            let bytes_needed = (bit_in_byte + bits).div_ceil(8);
            if byte_offset + bytes_needed > input.len() {
                break;
            }

            // Assemble value from bytes (using u64 for intermediate to handle overflow)
            let mut value: u64 = 0;
            for j in 0..bytes_needed.min(8) {
                if byte_offset + j < input.len() {
                    value |= (input[byte_offset + j] as u64) << (j * 8);
                }
            }

            // Shift to align and mask
            let extracted = ((value >> bit_in_byte) & (mask as u64)) as u32;
            *out_val = extracted as i32;
        }

        Ok(count)
    }

    fn decode_delta_f32(&self, input: &[u8], base: f32, output: &mut [f32]) -> Result<usize> {
        // Delta encoding format: packed deltas as i32 (reinterpreted from f32 bits)
        // Each delta is the difference in the bit representation

        if output.is_empty() {
            return Ok(0);
        }

        // Parse the input format:
        // [count: u32][bits: u8][packed_deltas: bytes]
        let Some(header) = super::common::parse_delta_header(input)? else {
            return Ok(0);
        };

        let actual_count = header.count.min(output.len());

        // Create temporary i64 buffer for deltas
        let mut deltas = vec![0i64; actual_count];
        let decoded = self.decode_bitpacked_i64(header.packed, header.bits, &mut deltas)?;

        // Reconstruct values: value = base + delta
        super::common::reconstruct_delta_f32(
            &deltas,
            base,
            &mut output[..decoded.min(actual_count)],
        );

        Ok(decoded.min(actual_count))
    }

    fn decode_delta_i64(&self, input: &[u8], base: i64, output: &mut [i64]) -> Result<usize> {
        // Delta encoding format: packed deltas
        // Each delta is the difference from the previous value

        if output.is_empty() {
            return Ok(0);
        }

        // Parse header
        let Some(header) = super::common::parse_delta_header(input)? else {
            return Ok(0);
        };

        let actual_count = header.count.min(output.len());

        // Decode deltas, then apply prefix sum to reconstruct values:
        // values[0] = base + deltas[0]; values[i] = values[i-1] + deltas[i]
        let decoded =
            self.decode_bitpacked_i64(header.packed, header.bits, &mut output[..actual_count])?;
        super::common::prefix_sum_i64(&mut output[..decoded], base);

        Ok(decoded)
    }

    fn supported_features(&self) -> CpuFeatures {
        // Scalar decoder has no SIMD features
        CpuFeatures::default()
    }

    fn name(&self) -> &'static str {
        "Scalar"
    }

    fn has_acceleration(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd_decode::test_support;

    #[test]
    fn test_scalar_golden_decode_battery() {
        // The reference implementation defines the golden vectors every
        // hardware target must reproduce.
        test_support::run_golden_decode_battery(&ScalarDecoder::new());
    }

    #[test]
    fn test_scalar_decode_bitpacked_8bit() {
        // 8-bit values, byte-aligned
        test_support::assert_decodes_i64(
            &ScalarDecoder::new(),
            &[0, 1, 127, 255, 128, 64, 32, 16],
            8,
        );
    }

    #[test]
    fn test_scalar_decode_bitpacked_4bit() {
        // 4-bit values (0-15)
        test_support::assert_decodes_i64(
            &ScalarDecoder::new(),
            &[0, 1, 7, 15, 8, 4, 2, 1, 0, 15, 10, 5],
            4,
        );
    }

    #[test]
    fn test_scalar_decode_bitpacked_1bit() {
        // 1-bit values (0 or 1)
        test_support::assert_decodes_i64(
            &ScalarDecoder::new(),
            &[1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1],
            1,
        );
    }

    #[test]
    fn test_scalar_decode_bitpacked_17bit() {
        // 17-bit values (crossing byte boundaries)
        test_support::assert_decodes_i64(
            &ScalarDecoder::new(),
            &[0, 1, 65535, 131071, 100000, 50000, 25000, 12500],
            17,
        );
    }

    #[test]
    fn test_scalar_decode_bitpacked_32bit() {
        // 32-bit values
        test_support::assert_decodes_i64(
            &ScalarDecoder::new(),
            &[0, 1, u32::MAX as u64, 2147483648, 1000000000],
            32,
        );
    }

    #[test]
    fn test_scalar_decode_bitpacked_i32() {
        test_support::assert_decodes_i32(
            &ScalarDecoder::new(),
            &[0, 1, 127, 255, 65535, 100000, 200000, 300000],
            20,
        );
    }

    #[test]
    fn test_scalar_invalid_bits() {
        let decoder = ScalarDecoder::new();

        let mut output = vec![0i64; 10];

        // 0 bits is invalid
        assert!(
            decoder
                .decode_bitpacked_i64(&[0u8; 10], 0, &mut output)
                .is_err()
        );

        // 65 bits is invalid for i64
        assert!(
            decoder
                .decode_bitpacked_i64(&[0u8; 100], 65, &mut output)
                .is_err()
        );

        // 33 bits is invalid for i32
        let mut output_i32 = vec![0i32; 10];
        assert!(
            decoder
                .decode_bitpacked_i32(&[0u8; 100], 33, &mut output_i32)
                .is_err()
        );
    }

    #[test]
    fn test_scalar_empty_input() {
        let decoder = ScalarDecoder::new();

        let mut output = vec![0i64; 10];
        let count = decoder.decode_bitpacked_i64(&[], 8, &mut output).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_scalar_name_and_features() {
        let decoder = ScalarDecoder::new();

        assert_eq!(decoder.name(), "Scalar");
        assert!(!decoder.has_acceleration());
        assert!(!decoder.supported_features().has_simd());
    }

    #[test]
    fn test_scalar_decode_various_bit_widths() {
        let decoder = ScalarDecoder::new();

        // Test multiple bit widths
        for bits in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 20, 24, 28, 32] {
            let max_val = if bits >= 64 {
                u64::MAX
            } else {
                (1u64 << bits) - 1
            };
            let values: Vec<u64> = (0..16).map(|i| i * max_val / 16).collect();
            test_support::assert_decodes_i64(&decoder, &values, bits);
        }
    }
}
