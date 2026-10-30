// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Frame of Reference (FOR) encoding - Baseline (pure scalar) implementation
//!
//! **ARCHITECTURE NOTE**: This is the BASELINE implementation.
//! - NO SIMD intrinsics allowed
//! - NO GPU code
//! - Pure portable Rust only
//!
//! For SIMD acceleration, see: `src/storage/engines/core/ops/proximacodec/simd.rs`
//! For GPU acceleration, see: `src/storage/engines/core/ops/proximacodec/impls/gpu/`
//!
//! Stores a base value and encodes offsets from that base using fixed bit width.
//! Best for clustered data with small variance.
//! Returns ONLY the compressed data - headers are added by WireFormatManager.

use anyhow::Result;

use super::helpers;
use super::helpers::WirePrim;

// ===== Core wire format encoding functions =====

/// Single FOR encode core (redundancy pass 4: the i32/i64 wire twins were
/// byte-identical modulo the base slot and offset widths; golden tests pin
/// the layout).
fn encode_frame_of_ref_wire<W: WirePrim>(wire_values: &[W], base: W) -> Result<Vec<u8>> {
    if wire_values.is_empty() {
        return Ok(Vec::new());
    }

    let mut result = Vec::new();

    // Store base value (WIRE_BYTES for the width)
    base.write_le(&mut result);

    // Compute offsets from base
    let offsets: Vec<W> = wire_values.iter().map(|&v| v.wrap_sub(base)).collect();

    // Find optimal bit width for offsets
    let max_offset_bits = offsets.iter().map(|&o| o.abs_bit_len()).max().unwrap_or(0);

    let bits = if max_offset_bits == 0 {
        1
    } else {
        // Add 1 bit for sign, but cap at the width
        (max_offset_bits + 1).min(W::BITS)
    };

    // Store bit width
    result.push(bits);

    // Bit-pack the offsets (delegate to shared helper)
    let packed = W::pack_signed(&offsets, bits)?;
    result.extend(packed);

    Ok(result)
}

// ===== Public API (thin wrappers using generic helpers) =====

/// Encode f32 values using Frame of Reference (raw, no headers)
///
/// # Algorithm
/// 1. Convert f32 to i32 (via to_bits)
/// 2. Compute offsets from base
/// 3. Pack offsets using specified bit width
///
/// # Format (raw data only, NO headers)
/// ```text
/// [base:4 bytes][bitpacked_offsets...]
/// ```text
///
/// # Parameters
/// - `values`: f32 slice to encode
/// - `base`: Base value for offset calculation (as i64)
///
/// # Returns
/// Raw encoded bytes (NO scheme marker, NO count header)
///
/// # Note
/// The bit width is determined automatically from the maximum offset.
/// The bit width is stored in the wire header (not in raw data).
pub fn encode_f32(values: &[f32], base: i64) -> Result<Vec<u8>> {
    helpers::encode_generic(values, |wire_values| {
        encode_frame_of_ref_wire(wire_values, base as i32)
    })
}

/// Encode i64 values using Frame of Reference (raw, no headers)
pub fn encode_i64(values: &[i64], base: i64) -> Result<Vec<u8>> {
    helpers::encode_generic(values, |wire_values| {
        encode_frame_of_ref_wire(wire_values, base)
    })
}

/// Encode i32 values using Frame of Reference (raw, no headers)
pub fn encode_i32(values: &[i32], base: i64) -> Result<Vec<u8>> {
    helpers::encode_generic(values, |wire_values| {
        encode_frame_of_ref_wire(wire_values, base as i32)
    })
}

// ===== Core wire format decoding functions =====

/// Single FOR decode core; header is [base:WIRE_BYTES][bits:1].
fn decode_frame_of_ref_wire<W: WirePrim>(data: &[u8], count: usize) -> Result<Vec<W>> {
    if count == 0 {
        return Ok(Vec::new());
    }

    if data.len() < W::WIRE_BYTES + 1 {
        return Err(anyhow::anyhow!("FOR decode: insufficient data"));
    }

    // Read base
    let base = W::read_le(data)?;

    // Read bits (1 byte)
    let bits = data[W::WIRE_BYTES];

    // Unpack offsets (delegate to shared helper with sign extension)
    let offsets = W::unpack_signed(&data[W::WIRE_BYTES + 1..], bits, count)?;

    // Reconstruct values (base + offset, wrapping)
    Ok(offsets.into_iter().map(|o| base.wrap_add(o)).collect())
}

// ===== Compatibility helpers for SIMD module =====

/// Parse FrameOfReference header for f32 data
///
/// **Helper for SIMD**: Made public to allow SIMD module to reuse wire format parsing.
///
/// # Returns
/// (base_i32, bits, offset_to_packed_data)
pub fn parse_header_f32(data: &[u8]) -> Result<(i32, u8, usize)> {
    if data.len() < 5 {
        return Err(anyhow::anyhow!("FOR decode: insufficient data"));
    }

    // Read base (4 bytes)
    let base_i32 = i32::from_le_bytes([data[0], data[1], data[2], data[3]]);

    // Read bits (1 byte)
    let bits = data[4];

    Ok((base_i32, bits, 5))
}

/// Reconstruct f32 values from offsets and base (scalar)
///
/// **Helper for SIMD**: Made public to allow SIMD module to provide vectorized version.
///
/// # Algorithm
/// 1. Add base to each offset: `value_bits = base + offset`
/// 2. Reinterpret i32 bits as f32
///
/// # Arguments
/// * `offsets` - Unpacked i32 offsets
/// * `base_i32` - Base value (f32 bit representation as i32)
pub fn reconstruct_values_scalar_f32(offsets: &[i32], base_i32: i32) -> Vec<f32> {
    helpers::reconstruct_f32_from_i32(offsets, base_i32)
}

// ===== Public API (thin wrappers using generic helpers) =====

/// Decode f32 values from Frame of Reference encoded data
pub fn decode_f32(data: &[u8], count: usize) -> Result<Vec<f32>> {
    helpers::decode_generic::<f32>(data, count, decode_frame_of_ref_wire::<i32>)
}

/// Decode i64 values from Frame of Reference encoded data
pub fn decode_i64(data: &[u8], count: usize) -> Result<Vec<i64>> {
    helpers::decode_generic::<i64>(data, count, decode_frame_of_ref_wire::<i64>)
}

/// Decode i32 values from Frame of Reference encoded data
pub fn decode_i32(data: &[u8], count: usize) -> Result<Vec<i32>> {
    helpers::decode_generic::<i32>(data, count, decode_frame_of_ref_wire::<i32>)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Golden-byte pins (redundancy pass 4) ====================
    // Wire layout: [base:WIRE_BYTES LE][bits:1 byte][bitpacked offsets]. Bit
    // layout of the packed part: little-endian stream, value k occupies
    // offsets [k*bits,(k+1)*bits), LSB-first. These pin the CURRENT bytes of
    // both widths BEFORE the WirePrim consolidation; any change is a
    // wire-format change.

    /// i32: [10,11,12] base 10 -> offsets [0,1,2]; max |offset| 2 needs 2
    /// bits + 1 sign = 3 bits. Packed: v1 bit0 -> stream bit 3, v2 bit1 ->
    /// stream bit 7 => 0x88, second byte 0x00.
    #[test]
    fn golden_for_i32_wire_bytes() {
        let encoded = encode_i32(&[10, 11, 12], 10).unwrap();
        assert_eq!(encoded, vec![10, 0, 0, 0, 3, 0x88, 0x00]);
        assert_eq!(decode_i32(&encoded, 3).unwrap(), vec![10, 11, 12]);
    }

    /// i64: [100,102] base 100 -> offsets [0,2]; 2 bits + sign = 3 bits.
    /// Packed: v1 (=2, 0b010) bit1 -> stream bit 4 => 0x10, one byte total.
    #[test]
    fn golden_for_i64_wire_bytes() {
        let encoded = encode_i64(&[100, 102], 100).unwrap();
        assert_eq!(encoded, vec![100, 0, 0, 0, 0, 0, 0, 0, 3, 0x10]);
        assert_eq!(decode_i64(&encoded, 2).unwrap(), vec![100, 102]);
    }

    /// Constant data pins the bits==1 special case (max offset 0 -> 1 bit).
    #[test]
    fn golden_for_i32_constant_offsets_zero_bits() {
        let encoded = encode_i32(&[7, 7], 7).unwrap();
        assert_eq!(encoded, vec![7, 0, 0, 0, 1, 0x00]);
        assert_eq!(decode_i32(&encoded, 2).unwrap(), vec![7, 7]);
    }

    #[test]
    fn test_for_f32_roundtrip() {
        // Clustered values around 100.0
        let values = vec![100.0f32, 101.0, 102.0, 103.0, 104.0];
        let base = 100.0f32.to_bits() as i64;

        let encoded = encode_f32(&values, base).expect("FOR f32 encoding should succeed");
        let decoded = decode_f32(&encoded, values.len()).expect("FOR f32 decoding should succeed");

        assert_eq!(values, decoded);
    }

    #[test]
    fn test_for_i64_roundtrip() {
        // Clustered values around 1000
        let values = vec![1000i64, 1001, 1002, 1003, 1004];
        let base = 1000;

        let encoded = encode_i64(&values, base).expect("FOR i64 encoding should succeed");
        let decoded = decode_i64(&encoded, values.len()).expect("FOR i64 decoding should succeed");

        assert_eq!(values, decoded);
    }

    #[test]
    fn test_for_i32_roundtrip() {
        // Clustered values around 500
        let values = vec![500i32, 501, 502, 503, 504];
        let base = 500;

        let encoded = encode_i32(&values, base).expect("FOR i32 encoding should succeed");
        let decoded = decode_i32(&encoded, values.len()).expect("FOR i32 decoding should succeed");

        assert_eq!(values, decoded);
    }

    #[test]
    fn test_for_compression() {
        // Values in range 1000-1015 (need 4 bits for offsets)
        let values: Vec<i64> = (1000..1016).collect();
        let base = 1000;

        let encoded = encode_i64(&values, base).expect("FOR i64 encoding should succeed");
        let decoded = decode_i64(&encoded, values.len()).expect("FOR i64 decoding should succeed");

        assert_eq!(values, decoded);

        // Verify compression
        // Original: 16 values × 8 bytes = 128 bytes
        // Encoded: 8 bytes (base) + 1 byte (bits) + 16×4 bits = 8 + 1 + 8 = 17 bytes
        let original_bytes = values.len() * 8;
        assert!(
            encoded.len() < original_bytes,
            "Should compress: {} < {}",
            encoded.len(),
            original_bytes
        );
    }

    #[test]
    fn test_for_constant_data() {
        // All same value - offsets are all 0
        let values = vec![42i32; 100];
        let base = 42;

        let encoded = encode_i32(&values, base).expect("FOR i32 encoding should succeed");
        let decoded = decode_i32(&encoded, values.len()).expect("FOR i32 decoding should succeed");

        assert_eq!(values, decoded);

        // Verify extreme compression (all offsets = 0, stored as 1 bit each)
        let original_bytes = values.len() * 4;
        assert!(encoded.len() < original_bytes / 10); // Should be much smaller
    }

    #[test]
    fn test_for_empty() {
        let values: Vec<i64> = vec![];
        let base = 0;

        let encoded =
            encode_i64(&values, base).expect("FOR i64 encoding should succeed for empty input");
        assert!(encoded.is_empty());

        let decoded =
            decode_i64(&encoded, 0).expect("FOR i64 decoding should succeed for empty input");
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_for_single_value() {
        let values = vec![42i64];
        let base = 42;

        let encoded =
            encode_i64(&values, base).expect("FOR i64 encoding should succeed for single value");
        let decoded =
            decode_i64(&encoded, 1).expect("FOR i64 decoding should succeed for single value");

        assert_eq!(values, decoded);
    }

    #[test]
    fn test_for_negative_offsets() {
        // Base in middle, some values below
        let values = vec![95i32, 96, 100, 104, 105];
        let base = 100;

        let encoded = encode_i32(&values, base)
            .expect("FOR i32 encoding should succeed with negative offsets");
        let decoded = decode_i32(&encoded, values.len())
            .expect("FOR i32 decoding should succeed with negative offsets");

        assert_eq!(values, decoded);
    }

    #[test]
    fn test_for_optimal_base() {
        // Base = minimum value for best compression
        let values = vec![1000i64, 1001, 1002, 1050, 1100];
        let base = 1000; // Minimum

        let encoded =
            encode_i64(&values, base).expect("FOR i64 encoding should succeed with optimal base");
        let decoded = decode_i64(&encoded, values.len())
            .expect("FOR i64 decoding should succeed with optimal base");

        assert_eq!(values, decoded);

        // With optimal base, max offset is 100, needs ~7 bits
        // Much better than encoding full values
    }
}
