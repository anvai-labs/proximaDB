// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! BitPacked encoding - Baseline (pure scalar) implementation
//!
//! **ARCHITECTURE NOTE**: This is the BASELINE implementation.
//! - NO SIMD intrinsics allowed
//! - NO GPU code
//! - Pure portable Rust only
//!
//! For SIMD acceleration, see: `src/storage/engines/core/ops/proximacodec/simd.rs`
//! For GPU acceleration, see: `src/storage/engines/core/ops/proximacodec/impls/gpu/`
//!
//! Packs integer values using a fixed bit width.
//! Returns ONLY the compressed data - headers are added by WireFormatManager.

use anyhow::Result;

/// Encode f32 values using bitpacking (raw, no headers)
///
/// # Algorithm
/// 1. Convert f32 to u32 (via to_bits)
/// 2. Pack using specified bit width
///
/// # Format (raw data only, NO headers)
/// ```text
/// [bitpacked_values...]
/// ```text
///
/// # Parameters
/// - `values`: f32 slice to encode
/// - `bits`: Bit width for packing (1-32)
///
/// # Returns
/// Raw encoded bytes (NO scheme marker, NO count header)
pub fn encode_f32(values: &[f32], bits: u8) -> Result<Vec<u8>> {
    if bits == 0 || bits > 32 {
        return Err(anyhow::anyhow!(
            "Bit width {} is invalid (must be 1-32)",
            bits
        ));
    }

    if values.is_empty() {
        return Ok(Vec::new());
    }

    // LOSSLESS: Convert f32 to IEEE 754 bit pattern (pure scalar)
    // This preserves exact f32 representation, including fractional parts
    // Round-trip: f32 → to_bits() → u32 → from_bits() → f32 (exact match)
    let u32_values: Vec<u32> = values.iter().map(|&v| v.to_bits()).collect();

    // Bit-pack
    bitpack_u32(&u32_values, bits)
}

/// Encode i64 values using bitpacking (raw, no headers)
pub fn encode_i64(values: &[i64], bits: u8) -> Result<Vec<u8>> {
    if bits == 0 || bits > 64 {
        return Err(anyhow::anyhow!(
            "Bit width {} is invalid (must be 1-64)",
            bits
        ));
    }

    if values.is_empty() {
        return Ok(Vec::new());
    }

    // Convert to u64 for bitpacking
    let u64_values: Vec<u64> = values.iter().map(|&v| v as u64).collect();

    bitpack_u64(&u64_values, bits)
}

/// Encode i32 values using bitpacking (raw, no headers)
pub fn encode_i32(values: &[i32], bits: u8) -> Result<Vec<u8>> {
    if bits == 0 || bits > 32 {
        return Err(anyhow::anyhow!(
            "Bit width {} is invalid (must be 1-32)",
            bits
        ));
    }

    if values.is_empty() {
        return Ok(Vec::new());
    }

    // Convert to u32 for bitpacking
    let u32_values: Vec<u32> = values.iter().map(|&v| v as u32).collect();

    bitpack_u32(&u32_values, bits)
}

/// Decode f32 values from bitpacked data
pub fn decode_f32(data: &[u8], bits: u8, count: usize) -> Result<Vec<f32>> {
    if bits > 32 {
        return Err(anyhow::anyhow!("Bit width {} exceeds 32", bits));
    }

    // Unpack to u32
    let u32_values = unbitpack_u32(data, bits, count)?;

    // LOSSLESS: Convert u32 bit pattern back to f32
    // This preserves exact f32 representation (reverse of to_bits())
    let values: Vec<f32> = u32_values.iter().map(|&v| f32::from_bits(v)).collect();

    Ok(values)
}

/// Decode i64 values from bitpacked data
pub fn decode_i64(data: &[u8], bits: u8, count: usize) -> Result<Vec<i64>> {
    if bits > 64 {
        return Err(anyhow::anyhow!("Bit width {} exceeds 64", bits));
    }

    // Unpack to u64
    let u64_values = unbitpack_u64(data, bits, count)?;

    // Convert u64 to i64
    let values: Vec<i64> = u64_values.iter().map(|&v| v as i64).collect();

    Ok(values)
}

/// Decode i32 values from bitpacked data
pub fn decode_i32(data: &[u8], bits: u8, count: usize) -> Result<Vec<i32>> {
    if bits > 32 {
        return Err(anyhow::anyhow!("Bit width {} exceeds 32", bits));
    }

    // Unpack to u32
    let u32_values = unbitpack_u32(data, bits, count)?;

    // Convert u32 to i32
    let values: Vec<i32> = u32_values.iter().map(|&v| v as i32).collect();

    Ok(values)
}

// ===== Bit-packing helpers =====

/// Integer primitive that can be laid into the little-endian bit stream
/// (redundancy pass 3: the u32/u64 cores were stamped twice here and again
/// in zigzag.rs; one generic core replaces all four copies).
trait PackPrim: Copy + Default {
    /// 32 for u32, 64 for u64.
    const MAX_BITS: u8;
    /// `(self >> pos) & 1` as u32.
    fn bit(self, pos: u8) -> u32;
    /// `self | ((bit as Self) << pos)`.
    fn with_bit(self, pos: u8, bit: u32) -> Self;
}

impl PackPrim for u32 {
    const MAX_BITS: u8 = 32;
    fn bit(self, pos: u8) -> u32 {
        (self >> pos) & 1
    }
    fn with_bit(self, pos: u8, bit: u32) -> Self {
        self | (bit << pos)
    }
}

impl PackPrim for u64 {
    const MAX_BITS: u8 = 64;
    fn bit(self, pos: u8) -> u32 {
        ((self >> pos) & 1) as u32
    }
    fn with_bit(self, pos: u8, bit: u32) -> Self {
        self | ((bit as u64) << pos)
    }
}

/// Single packing core (little-endian bit stream, LSB-first values).
/// Byte-identical to the two width-specialized fns it replaced; the golden
/// tests pin the layout.
fn pack_bits<T: PackPrim>(values: &[T], bits: u8) -> Result<Vec<u8>> {
    if bits == 0 {
        return Ok(Vec::new());
    }
    if bits > T::MAX_BITS {
        return Err(anyhow::anyhow!(
            "Bit width {} exceeds {}",
            bits,
            T::MAX_BITS
        ));
    }

    let total_bits = values.len() * bits as usize;
    let total_bytes = total_bits.div_ceil(8);
    let mut result = vec![0u8; total_bytes];

    let mut bit_offset = 0;
    for &value in values {
        for bit_pos in 0..bits {
            let bit = value.bit(bit_pos);
            let byte_idx = bit_offset / 8;
            let bit_idx = bit_offset % 8;

            if byte_idx < result.len() {
                result[byte_idx] |= (bit as u8) << bit_idx;
            }

            bit_offset += 1;
        }
    }

    Ok(result)
}

/// Single unpacking core (zero-fills values whose bits run past the data).
fn unpack_bits<T: PackPrim>(data: &[u8], bits: u8, count: usize) -> Result<Vec<T>> {
    if bits == 0 {
        return Ok(vec![T::default(); count]);
    }
    if bits > T::MAX_BITS {
        return Err(anyhow::anyhow!(
            "Bit width {} exceeds {}",
            bits,
            T::MAX_BITS
        ));
    }

    let mut result = Vec::with_capacity(count);
    let mut bit_offset = 0;

    for _ in 0..count {
        let mut value = T::default();

        for bit_pos in 0..bits {
            let byte_idx = bit_offset / 8;
            let bit_idx = bit_offset % 8;

            if byte_idx < data.len() {
                let bit = (data[byte_idx] >> bit_idx) & 1;
                value = value.with_bit(bit_pos, bit as u32);
            }

            bit_offset += 1;
        }

        result.push(value);
    }

    Ok(result)
}

/// Bit-pack u32 values (simple implementation, no SIMD)
///
/// **Note**: Made public for use by simd.rs helpers that need bitpacking
/// for intermediate results (PForDelta, FrameOfReference, etc.)
pub fn bitpack_u32(values: &[u32], bits: u8) -> Result<Vec<u8>> {
    pack_bits(values, bits)
}

/// Bit-pack u64 values
pub(crate) fn bitpack_u64(values: &[u64], bits: u8) -> Result<Vec<u8>> {
    pack_bits(values, bits)
}

/// Unpack u32 values from bit-packed data
///
/// **Note**: Made public for use by simd.rs helpers that need bitunpacking
/// for intermediate results (PForDelta, FrameOfReference, etc.)
pub fn unbitpack_u32(data: &[u8], bits: u8, count: usize) -> Result<Vec<u32>> {
    unpack_bits(data, bits, count)
}

/// Unpack u64 values from bit-packed data
pub(crate) fn unbitpack_u64(data: &[u8], bits: u8, count: usize) -> Result<Vec<u64>> {
    unpack_bits(data, bits, count)
}

// ===== Signed bit-packing helpers (with sign extension) =====

/// Bit-pack i32 values (with sign extension support)
///
/// **Note**: Made public for use by encoding schemes that need signed bitpacking
/// (FrameOfReference, Delta, PatchedBase, etc.)
///
/// This is identical to bitpack_u32 but accepts i32 input for convenience.
pub(crate) fn bitpack_i32(values: &[i32], bits: u8) -> Result<Vec<u8>> {
    if bits > 32 {
        return Err(anyhow::anyhow!("Bit width {} exceeds 32", bits));
    }

    if bits == 0 {
        return Ok(Vec::new());
    }

    let total_bits = values.len() * bits as usize;
    let total_bytes = total_bits.div_ceil(8);
    let mut result = vec![0u8; total_bytes];

    let mut bit_offset = 0;
    for &value in values {
        let value_u32 = value as u32;

        for bit_pos in 0..bits {
            let bit = (value_u32 >> bit_pos) & 1;
            let byte_idx = bit_offset / 8;
            let bit_idx = bit_offset % 8;

            if byte_idx < result.len() {
                result[byte_idx] |= (bit as u8) << bit_idx;
            }

            bit_offset += 1;
        }
    }

    Ok(result)
}

/// Bit-pack i64 values (with sign extension support)
///
/// **Note**: Made public for use by encoding schemes that need signed bitpacking
pub(crate) fn bitpack_i64(values: &[i64], bits: u8) -> Result<Vec<u8>> {
    if bits > 64 {
        return Err(anyhow::anyhow!("Bit width {} exceeds 64", bits));
    }

    if bits == 0 {
        return Ok(Vec::new());
    }

    let total_bits = values.len() * bits as usize;
    let total_bytes = total_bits.div_ceil(8);
    let mut result = vec![0u8; total_bytes];

    let mut bit_offset = 0;
    for &value in values {
        let value_u64 = value as u64;

        for bit_pos in 0..bits {
            let bit = (value_u64 >> bit_pos) & 1;
            let byte_idx = bit_offset / 8;
            let bit_idx = bit_offset % 8;

            if byte_idx < result.len() {
                result[byte_idx] |= (bit as u8) << bit_idx;
            }

            bit_offset += 1;
        }
    }

    Ok(result)
}

/// Unpack i32 values from bit-packed data (with sign extension)
///
/// **Note**: Made public for use by encoding schemes that need signed unpacking
/// (FrameOfReference, Delta, PatchedBase, etc.)
///
/// **Sign Extension**: If the high bit is set and bits < 32, the value is
/// sign-extended to preserve negative values correctly.
///
/// # Example
/// ```ignore
/// // 5-bit signed value: 11111 (binary) = -1 (signed)
/// // Without sign extension: 31 (unsigned)
/// // With sign extension: 0xFFFFFFFF = -1 (i32)
/// ```text
pub fn unbitpack_i32(data: &[u8], bits: u8, count: usize) -> Result<Vec<i32>> {
    if bits > 32 {
        return Err(anyhow::anyhow!("Bit width {} exceeds 32", bits));
    }

    if bits == 0 {
        return Ok(vec![0; count]);
    }

    let mut result = Vec::with_capacity(count);
    let mut bit_offset = 0;

    for _ in 0..count {
        let mut value = 0u32;

        for bit_pos in 0..bits {
            let byte_idx = bit_offset / 8;
            let bit_idx = bit_offset % 8;

            if byte_idx < data.len() {
                let bit = (data[byte_idx] >> bit_idx) & 1;
                value |= (bit as u32) << bit_pos;
            }

            bit_offset += 1;
        }

        // Sign extend: if high bit is set, extend with 1s
        let signed_value = if bits < 32 && (value & (1 << (bits - 1))) != 0 {
            // Sign bit is set - extend with 1s
            let mask = !0u32 << bits;
            (value | mask) as i32
        } else {
            value as i32
        };

        result.push(signed_value);
    }

    Ok(result)
}

/// Unpack i64 values from bit-packed data (with sign extension)
///
/// **Note**: Made public for use by encoding schemes that need signed unpacking
///
/// **Sign Extension**: If the high bit is set and bits < 64, the value is
/// sign-extended to preserve negative values correctly.
pub(crate) fn unbitpack_i64(data: &[u8], bits: u8, count: usize) -> Result<Vec<i64>> {
    if bits > 64 {
        return Err(anyhow::anyhow!("Bit width {} exceeds 64", bits));
    }

    if bits == 0 {
        return Ok(vec![0; count]);
    }

    let mut result = Vec::with_capacity(count);
    let mut bit_offset = 0;

    for _ in 0..count {
        let mut value = 0u64;

        for bit_pos in 0..bits {
            let byte_idx = bit_offset / 8;
            let bit_idx = bit_offset % 8;

            if byte_idx < data.len() {
                let bit = (data[byte_idx] >> bit_idx) & 1;
                value |= (bit as u64) << bit_pos;
            }

            bit_offset += 1;
        }

        // Sign extend: if high bit is set, extend with 1s
        let signed_value = if bits < 64 && (value & (1 << (bits - 1))) != 0 {
            // Sign bit is set - extend with 1s
            let mask = !0u64 << bits;
            (value | mask) as i64
        } else {
            value as i64
        };

        result.push(signed_value);
    }

    Ok(result)
}

// ===== Unsigned bitpacking variants (NO sign extension) =====
//
// These are used by PFor schemes that treat bitpacked values as unsigned
// and handle sign/negative values via separate patches.

/// Unpack i32 values WITHOUT sign extension (for PFor schemes)
///
/// This variant treats all unpacked values as unsigned and casts directly to i32.
/// Used by pfor_delta and pfor_double_delta where patches handle full precision.
#[allow(dead_code)]
pub fn unbitpack_i32_unsigned(data: &[u8], bits: u8, count: usize) -> Result<Vec<i32>> {
    if bits > 32 {
        return Err(anyhow::anyhow!("Bit width {} exceeds 32", bits));
    }

    if bits == 0 {
        return Ok(vec![0; count]);
    }

    let mut result = Vec::with_capacity(count);
    let mut bit_offset = 0;

    for _ in 0..count {
        let mut value = 0u32;

        for bit_pos in 0..bits {
            let byte_idx = bit_offset / 8;
            let bit_idx = bit_offset % 8;

            if byte_idx < data.len() {
                let bit = (data[byte_idx] >> bit_idx) & 1;
                value |= (bit as u32) << bit_pos;
            }

            bit_offset += 1;
        }

        // NO sign extension - direct cast
        result.push(value as i32);
    }

    Ok(result)
}

/// Unpack i64 values WITHOUT sign extension (for PFor schemes)
///
/// This variant treats all unpacked values as unsigned and casts directly to i64.
/// Used by pfor_delta and pfor_double_delta where patches handle full precision.
pub(crate) fn unbitpack_i64_unsigned(data: &[u8], bits: u8, count: usize) -> Result<Vec<i64>> {
    if bits > 64 {
        return Err(anyhow::anyhow!("Bit width {} exceeds 64", bits));
    }

    if bits == 0 {
        return Ok(vec![0; count]);
    }

    let mut result = Vec::with_capacity(count);
    let mut bit_offset = 0;

    for _ in 0..count {
        let mut value = 0u64;

        for bit_pos in 0..bits {
            let byte_idx = bit_offset / 8;
            let bit_idx = bit_offset % 8;

            if byte_idx < data.len() {
                let bit = (data[byte_idx] >> bit_idx) & 1;
                value |= (bit as u64) << bit_pos;
            }

            bit_offset += 1;
        }

        // NO sign extension - direct cast
        result.push(value as i64);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Golden-byte pins (redundancy pass 3) ====================
    // Hand-computed wire bytes; any change here is a FORMAT change. Bit
    // layout: little-endian stream, value k occupies offsets [k*bits,
    // (k+1)*bits), value bit j at stream offset k*bits+j, byte = offset/8
    // (LSB-first). These pin the layout BEFORE the PackPrim consolidation.

    /// [1,2,3] @ 2 bits: 1(01),2(10),3(11) -> bits {0,3,4,5} -> 1+8+16+32 = 0x39
    #[test]
    fn golden_bitpack_u32_three_values_two_bits() {
        assert_eq!(bitpack_u32(&[1, 2, 3], 2).unwrap(), vec![0x39]);
    }

    /// [5,0,6,7,1] @ 3 bits: v0=101 v1=000 v2=110 v3=111 v4=001 ->
    /// byte0 = 2^0+2^2+2^7 = 0x85; byte1 = 2^0..2^4 = 0x1F
    #[test]
    fn golden_bitpack_u32_five_values_three_bits() {
        assert_eq!(bitpack_u32(&[5, 0, 6, 7, 1], 3).unwrap(), vec![0x85, 0x1F]);
    }

    /// Full-width packing is plain little-endian.
    #[test]
    fn golden_bitpack_u32_full_width_is_le() {
        assert_eq!(
            bitpack_u32(&[0x1234_5678], 32).unwrap(),
            vec![0x78, 0x56, 0x34, 0x12]
        );
    }

    /// Masked low bits only: 255 & 0b1111 = 0x0F (equivalence witness that
    /// bitpack's masked write matches an unmasked low-bits write).
    #[test]
    fn golden_bitpack_u32_masks_to_width() {
        assert_eq!(bitpack_u32(&[255], 4).unwrap(), vec![0x0F]);
    }

    #[test]
    fn golden_unbitpack_round_goldens() {
        assert_eq!(unbitpack_u32(&[0x39], 2, 3).unwrap(), vec![1, 2, 3]);
        assert_eq!(
            unbitpack_u32(&[0x85, 0x1F], 3, 5).unwrap(),
            vec![5, 0, 6, 7, 1]
        );
        // Zero-fill contract: reads past the data end yield 0.
        assert_eq!(unbitpack_u32(&[0xFF], 4, 3).unwrap(), vec![15, 15, 0]);
    }

    #[test]
    fn test_bitpack_f32_roundtrip() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let bits = 32; // Full precision

        let encoded = encode_f32(&values, bits).expect("Failed to encode f32 values");
        let decoded =
            decode_f32(&encoded, bits, values.len()).expect("Failed to decode f32 values");

        assert_eq!(values, decoded);
    }

    #[test]
    fn test_bitpack_i64_roundtrip() {
        let values = vec![10i64, 20, 30, 40, 50];
        let bits = 16; // 16 bits per value

        let encoded = encode_i64(&values, bits).expect("Failed to encode i64 values");
        let decoded =
            decode_i64(&encoded, bits, values.len()).expect("Failed to decode i64 values");

        assert_eq!(values, decoded);
    }

    #[test]
    fn test_bitpack_i32_roundtrip() {
        let values = vec![100i32, 200, 300, 400, 500];
        let bits = 16; // 16 bits per value

        let encoded = encode_i32(&values, bits).expect("Failed to encode i32 values");
        let decoded =
            decode_i32(&encoded, bits, values.len()).expect("Failed to decode i32 values");

        assert_eq!(values, decoded);
    }

    #[test]
    fn test_bitpack_small_width() {
        // Values that fit in 4 bits (0-15)
        let values = vec![1i32, 5, 10, 15, 3];
        let bits = 4;

        let encoded =
            encode_i32(&values, bits).expect("Failed to encode i32 values with small bit width");
        let decoded = decode_i32(&encoded, bits, values.len())
            .expect("Failed to decode i32 values with small bit width");

        assert_eq!(values, decoded);

        // Verify compression
        let original_bytes = values.len() * 4; // 4 bytes per i32
        let compressed_bytes = encoded.len();
        assert!(
            compressed_bytes < original_bytes,
            "Expected compression: {} < {}",
            compressed_bytes,
            original_bytes
        );
    }

    #[test]
    fn test_bitpack_empty() {
        let values: Vec<i64> = vec![];
        let bits = 16;

        let encoded = encode_i64(&values, bits).expect("Failed to encode empty i64 values");
        assert!(encoded.is_empty());

        let decoded = decode_i64(&encoded, bits, 0).expect("Failed to decode empty i64 values");
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_bitpack_single_bit() {
        // Binary values (0 or 1)
        let values = vec![0i32, 1, 0, 1, 1, 0];
        let bits = 1;

        let encoded =
            encode_i32(&values, bits).expect("Failed to encode i32 values with single bit");
        let decoded = decode_i32(&encoded, bits, values.len())
            .expect("Failed to decode i32 values with single bit");

        assert_eq!(values, decoded);

        // Verify extreme compression
        let original_bytes = values.len() * 4;
        let compressed_bytes = encoded.len();
        assert_eq!(compressed_bytes, 1); // 6 bits fit in 1 byte
        assert!(compressed_bytes < original_bytes);
    }
}
