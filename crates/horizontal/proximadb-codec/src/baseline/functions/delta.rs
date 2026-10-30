// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Delta encoding - Baseline (pure scalar) implementation
//!
//! **ARCHITECTURE NOTE**: This is the BASELINE implementation.
//! - NO SIMD intrinsics allowed
//! - NO GPU code
//! - Pure portable Rust only
//!
//! For SIMD acceleration, see: `src/storage/engines/core/ops/proximacodec/simd.rs`
//! For GPU acceleration, see: `src/storage/engines/core/ops/proximacodec/impls/gpu/`
//!
//! Computes differences from a base value and bit-packs the deltas.
//! Returns ONLY the compressed data - headers are added by WireFormatManager.

use anyhow::Result;

// ===== Bitpacking delegation to shared helpers =====
//
// All bitpacking operations now use the shared helpers in bitpack.rs
// to avoid code duplication and ensure consistent sign extension behavior.

use super::bitpack;
use super::helpers;
use super::helpers::WirePrim;

// ===== Core wire format encoding functions =====

/// Single delta encode core (redundancy pass 4: the i32-base and i64-base
/// twins differed only in the base slot width — deltas were already i64 on
/// both sides; golden tests pin the layout).
///
/// Uses i64 deltas to prevent overflow when computing differences (the i32
/// path widens via `WirePrim::widen`; for i64 the widen is the identity and
/// the subtraction wraps exactly as the old native core did).
fn encode_delta_wire<W: WirePrim>(wire_values: &[W], base: W) -> Result<Vec<u8>> {
    if wire_values.is_empty() {
        return Ok(Vec::new());
    }

    let mut result = Vec::new();

    // Store base value (WIRE_BYTES for the width)
    base.write_le(&mut result);

    // Compute deltas in i64 (NO OVERFLOW!)
    let deltas: Vec<i64> = wire_values
        .iter()
        .map(|&v| v.widen().wrapping_sub(base.widen()))
        .collect();

    // Find optimal bit width for deltas
    let max_delta_bits = deltas.iter().map(|&d| d.abs_bit_len()).max().unwrap_or(0);

    let bits = if max_delta_bits == 0 {
        1
    } else {
        // Add 1 bit for sign, but cap at 64
        (max_delta_bits + 1).min(64)
    };

    result.push(bits);

    // Bit-pack the deltas
    let packed = bitpack::bitpack_i64(&deltas, bits)?;
    result.extend(packed);

    Ok(result)
}

// ===== Public API (thin wrappers using generic helpers) =====

/// Encode f32 values using delta encoding (raw, no headers)
///
/// # Algorithm
/// 1. Convert f32 to i32 (via to_bits)
/// 2. Compute deltas from base in i64 (prevents overflow)
/// 3. Find optimal bit width
/// 4. Bit-pack deltas
///
/// # Format (raw data only, NO headers)
/// ```text
/// [base:4 bytes][bits:1 byte][bitpacked_deltas...]
/// ```text
///
/// # Parameters
/// - `values`: f32 slice to encode
/// - `base`: Base value for delta calculation (as i64)
///
/// # Returns
/// Raw encoded bytes (NO scheme marker, NO count header)
pub fn encode_f32(values: &[f32], base: i64) -> Result<Vec<u8>> {
    helpers::encode_generic(values, |wire_values| {
        encode_delta_wire(wire_values, base as i32)
    })
}

/// Encode i64 values using delta encoding (raw, no headers)
pub fn encode_i64(values: &[i64], base: i64) -> Result<Vec<u8>> {
    helpers::encode_generic(values, |wire_values| encode_delta_wire(wire_values, base))
}

/// Encode i32 values using delta encoding (raw, no headers)
pub fn encode_i32(values: &[i32], base: i64) -> Result<Vec<u8>> {
    helpers::encode_generic(values, |wire_values| {
        encode_delta_wire(wire_values, base as i32)
    })
}

// ===== Core wire format decoding functions =====

/// Single delta decode core; header is [base:WIRE_BYTES][bits:1], the packed
/// payload is always i64 deltas (narrowed back to W on reconstruction).
fn decode_delta_wire<W: WirePrim>(data: &[u8], count: usize) -> Result<Vec<W>> {
    if count == 0 {
        return Ok(Vec::new());
    }

    if data.len() < W::WIRE_BYTES + 1 {
        return Err(anyhow::anyhow!("Delta decode: insufficient data"));
    }

    // Read base
    let base = W::read_le(data)?;

    // Read bits (1 byte)
    let bits = data[W::WIRE_BYTES];

    // Unpack deltas (i64)
    let deltas = bitpack::unbitpack_i64(&data[W::WIRE_BYTES + 1..], bits, count)?;

    // Reconstruct values: widen the base, add each delta in i64, narrow back
    // (truncating for i32 exactly like the old reconstruct helper).
    Ok(deltas
        .into_iter()
        .map(|d| W::from_wide(base.widen().wrapping_add(d)))
        .collect())
}

// ===== Public API (thin wrappers using generic helpers) =====

/// Decode f32 values from delta-encoded data
pub fn decode_f32(data: &[u8], count: usize) -> Result<Vec<f32>> {
    helpers::decode_generic::<f32>(data, count, decode_delta_wire::<i32>)
}

/// Decode i64 values from delta-encoded data
pub fn decode_i64(data: &[u8], count: usize) -> Result<Vec<i64>> {
    helpers::decode_generic::<i64>(data, count, decode_delta_wire::<i64>)
}

/// Decode i32 values from delta-encoded data
pub fn decode_i32(data: &[u8], count: usize) -> Result<Vec<i32>> {
    helpers::decode_generic::<i32>(data, count, decode_delta_wire::<i32>)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Golden-byte pins (redundancy pass 4) ====================
    // Wire layout: [base:WIRE_BYTES LE][bits:1 byte][bitpacked i64 deltas].
    // Both widths store i64 deltas; only the base slot width differs. These
    // pin the CURRENT bytes of both widths BEFORE the WirePrim consolidation;
    // any change is a wire-format change.

    /// i32 path (4-byte base): [10,13] base 10 -> i64 deltas [0,3], 2 bits +
    /// sign = 3 bits; packed v1=011 -> stream bits 3,4 => 0x18.
    #[test]
    fn golden_delta_i32_base_i64_deltas_wire_bytes() {
        let encoded = encode_i32(&[10, 13], 10).unwrap();
        assert_eq!(encoded, vec![10, 0, 0, 0, 3, 0x18]);
        assert_eq!(decode_i32(&encoded, 2).unwrap(), vec![10, 13]);
    }

    /// i64 path (8-byte base): [5,8] base 5 -> deltas [0,3], 3 bits, 0x18.
    #[test]
    fn golden_delta_i64_wire_bytes() {
        let encoded = encode_i64(&[5, 8], 5).unwrap();
        assert_eq!(encoded, vec![5, 0, 0, 0, 0, 0, 0, 0, 3, 0x18]);
        assert_eq!(decode_i64(&encoded, 2).unwrap(), vec![5, 8]);
    }

    /// Constant data pins the bits==1 special case.
    #[test]
    fn golden_delta_i64_constant_one_bit() {
        let encoded = encode_i64(&[9, 9], 9).unwrap();
        assert_eq!(encoded, vec![9, 0, 0, 0, 0, 0, 0, 0, 1, 0x00]);
        assert_eq!(decode_i64(&encoded, 2).unwrap(), vec![9, 9]);
    }

    #[test]
    fn test_delta_f32_roundtrip() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let base = 0;

        let encoded = encode_f32(&values, base).expect("Failed to encode f32 values");
        let decoded = decode_f32(&encoded, values.len()).expect("Failed to decode f32 values");

        assert_eq!(values, decoded);
    }

    #[test]
    fn test_delta_i64_roundtrip() {
        let values = vec![100i64, 200, 300, 400, 500];
        let base = 0;

        let encoded = encode_i64(&values, base).expect("Failed to encode i64 values");
        let decoded = decode_i64(&encoded, values.len()).expect("Failed to decode i64 values");

        assert_eq!(values, decoded);
    }

    #[test]
    fn test_delta_sequential() {
        let values: Vec<i64> = (0..1000).collect();
        let base = 0;

        let encoded = encode_i64(&values, base).expect("Failed to encode sequential i64 values");
        let decoded =
            decode_i64(&encoded, values.len()).expect("Failed to decode sequential i64 values");

        assert_eq!(values, decoded);

        // Verify compression
        let original_bytes = values.len() * 8;
        assert!(
            encoded.len() < original_bytes,
            "Should compress sequential data"
        );
    }

    // ===== OVERFLOW EDGE CASE TESTS =====
    // These tests verify that the i64 delta fix prevents overflow

    #[test]
    fn test_overflow_i32_extremes() {
        // Test with i32::MAX and i32::MIN - would overflow with i32 deltas
        let values = vec![i32::MIN, i32::MAX, i32::MIN, i32::MAX];
        let base = 0;

        let encoded = encode_i32(&values, base).expect("Failed to encode i32 extreme values");
        let decoded =
            decode_i32(&encoded, values.len()).expect("Failed to decode i32 extreme values");

        assert_eq!(values, decoded, "Failed to roundtrip i32 extremes");
    }

    #[test]
    fn test_overflow_f32_extreme_bit_patterns() {
        // Test f32 values with extreme bit patterns that would cause i32 delta overflow
        let values = vec![
            f32::from_bits(i32::MAX as u32),
            f32::from_bits(i32::MIN as u32),
            f32::from_bits(0),
            f32::from_bits(i32::MAX as u32),
        ];
        let base = 0;

        let encoded = encode_f32(&values, base).expect("Failed to encode f32 extreme bit patterns");
        let decoded =
            decode_f32(&encoded, values.len()).expect("Failed to decode f32 extreme bit patterns");

        assert_eq!(values.len(), decoded.len());
        for (orig, dec) in values.iter().zip(decoded.iter()) {
            assert_eq!(
                orig.to_bits(),
                dec.to_bits(),
                "Failed to roundtrip extreme f32 bit pattern: orig={:08x}, dec={:08x}",
                orig.to_bits(),
                dec.to_bits()
            );
        }
    }

    #[test]
    fn test_overflow_alternating_extremes() {
        // Worst case: alternating between extremes causes maximum deltas
        let values = vec![i32::MIN, i32::MAX, i32::MIN, i32::MAX, i32::MIN, i32::MAX];
        let base = 0;

        let encoded =
            encode_i32(&values, base).expect("Failed to encode alternating extreme i32 values");
        let decoded = decode_i32(&encoded, values.len())
            .expect("Failed to decode alternating extreme i32 values");

        assert_eq!(values, decoded, "Failed to handle alternating extremes");
    }

    #[test]
    fn test_overflow_i64_full_range() {
        // Test i64 values across full range
        let values = vec![i64::MIN, i64::MIN / 2, 0i64, i64::MAX / 2, i64::MAX];
        let base = 0;

        let encoded = encode_i64(&values, base).expect("Failed to encode i64 full range values");
        let decoded =
            decode_i64(&encoded, values.len()).expect("Failed to decode i64 full range values");

        assert_eq!(values, decoded, "Failed to handle i64 extremes");
    }

    #[test]
    fn test_overflow_f32_special_values() {
        // Test special f32 values with extreme bit patterns
        let values = vec![
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::MAX,
            f32::MIN,
            0.0f32,
        ];
        let base = 0;

        let encoded = encode_f32(&values, base).expect("Failed to encode f32 special values");
        let decoded =
            decode_f32(&encoded, values.len()).expect("Failed to decode f32 special values");

        assert_eq!(values.len(), decoded.len());
        for (orig, dec) in values.iter().zip(decoded.iter()) {
            // Compare bit patterns since NAN != NAN
            assert_eq!(
                orig.to_bits(),
                dec.to_bits(),
                "Failed to roundtrip special f32 value: orig={:08x}, dec={:08x}",
                orig.to_bits(),
                dec.to_bits()
            );
        }
    }
}
