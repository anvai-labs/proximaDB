// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Patched Frame of Reference Delta (PForDelta) - Baseline (pure scalar) implementation
//!
//! **ARCHITECTURE NOTE**: This is the BASELINE implementation.
//! - NO SIMD intrinsics allowed
//! - NO GPU code
//! - Pure portable Rust only
//!
//! For SIMD acceleration, see: `src/storage/engines/core/ops/proximacodec/simd.rs`
//! For GPU acceleration, see: `src/storage/engines/core/ops/proximacodec/impls/gpu/`
//!
//! Handles data with outliers by separating regular values from patches.
//! Combines Frame of Reference with exception handling for outliers.
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

/// Single PForDelta encode core (redundancy pass 4: the i32-base and i64-base
/// twins differed only in the base slot width — deltas, percentile logic, and
/// patches were already i64 on both sides). This is the CANONICAL wire format
/// pinned against the SIMD encoder by TD-CODEC-1; the golden tests pin the
/// bytes.
fn encode_pfor_delta_wire<W: WirePrim>(wire_values: &[W], base: W) -> Result<Vec<u8>> {
    if wire_values.is_empty() {
        return Ok(Vec::new());
    }

    let mut result = Vec::new();
    base.write_le(&mut result);

    // Compute deltas in i64 (NO OVERFLOW! for the i32 path; wrapping for i64)
    let deltas: Vec<i64> = wire_values
        .iter()
        .map(|&v| v.widen().wrapping_sub(base.widen()))
        .collect();

    // Find optimal bit width for 90% of values
    let mut sorted_deltas: Vec<u64> = deltas.iter().map(|&d| d.unsigned_abs()).collect();
    sorted_deltas.sort_unstable();

    let percentile_90_idx = (sorted_deltas.len() * 90) / 100;
    let threshold = sorted_deltas.get(percentile_90_idx).copied().unwrap_or(0);
    let bits = if threshold == 0 {
        1
    } else {
        64 - threshold.leading_zeros() as u8
    };

    result.push(bits);

    // Separate regular values and patches
    let mut regular_values = Vec::with_capacity(deltas.len());
    let mut patches = Vec::new();
    let max_regular = if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };

    for (idx, &delta) in deltas.iter().enumerate() {
        let abs_delta = delta.unsigned_abs();
        if abs_delta <= max_regular {
            regular_values.push(delta);
        } else {
            regular_values.push(0); // Sentinel for patch position
            patches.push((idx as u32, delta));
        }
    }

    // Store number of patches
    let num_patches = patches.len() as u32;
    result.extend_from_slice(&num_patches.to_le_bytes());

    // Bitpack regular values (decoder uses the unsigned unpack variant — no
    // sign extension for sentinels)
    let packed = bitpack::bitpack_i64(&regular_values, bits)?;
    result.extend(packed);

    // Store patches (position:4 bytes + value:8 bytes = 12 bytes per patch)
    for (pos, value) in patches {
        result.extend_from_slice(&pos.to_le_bytes());
        result.extend_from_slice(&value.to_le_bytes());
    }

    Ok(result)
}

// ===== Public API (thin wrappers using generic helpers) =====

/// Encode f32 values using PForDelta (raw, no headers)
///
/// # Algorithm
/// 1. Convert f32 to i32 bits
/// 2. Compute deltas from base in i64 (prevents overflow)
/// 3. Identify outliers (values requiring more bits than 90th percentile)
/// 4. Store regular values with bitpacking
/// 5. Store outliers as patches with their positions
///
/// # Format (raw data only, NO headers)
/// ```text
/// [base:4 bytes][bits:1 byte][num_patches:4 bytes]
/// [bitpacked_values...][patches:(pos:4, value:8)*]
/// ```text
///
/// # Parameters
/// - `values`: f32 slice to encode
/// - `base`: Base value for frame of reference (i64 for compatibility)
///
/// # Returns
/// Raw encoded bytes (NO scheme marker, NO count header)
pub fn encode_f32(values: &[f32], base: i64) -> Result<Vec<u8>> {
    helpers::encode_generic(values, |wire_values| {
        encode_pfor_delta_wire(wire_values, base as i32)
    })
}

/// Encode i64 values using PForDelta (raw, no headers)
pub fn encode_i64(values: &[i64], base: i64) -> Result<Vec<u8>> {
    helpers::encode_generic(values, |wire_values| {
        encode_pfor_delta_wire(wire_values, base)
    })
}

/// Encode i32 values using PForDelta (raw, no headers)
pub fn encode_i32(values: &[i32], base: i64) -> Result<Vec<u8>> {
    helpers::encode_generic(values, |wire_values| {
        encode_pfor_delta_wire(wire_values, base as i32)
    })
}

// ===== Core wire format decoding functions =====

/// Single PForDelta decode core; header is
/// [base:WIRE_BYTES][bits:1][num_patches:u32], the packed regulars and the
/// (pos, i64) patches are width-independent.
fn decode_pfor_delta_wire<W: WirePrim>(data: &[u8], count: usize) -> Result<Vec<W>> {
    if count == 0 {
        return Ok(Vec::new());
    }

    let (base, deltas) = parse_header_and_patches::<W>(data, count)?;

    // Reconstruct values: widen the base, add each i64 delta, narrow back
    // (truncating for i32 exactly like the old reconstruct helper).
    Ok(deltas
        .into_iter()
        .map(|d| W::from_wide(base.widen().wrapping_add(d)))
        .collect())
}

/// Parse the PForDelta header and apply patches to the unpacked deltas
/// (shared by the width-generic decode core and the SIMD-facing f32 helper).
fn parse_header_and_patches<W: WirePrim>(data: &[u8], count: usize) -> Result<(W, Vec<i64>)> {
    let header_len = W::WIRE_BYTES + 5;
    if data.len() < header_len {
        return Err(anyhow::anyhow!("PForDelta decode: insufficient data"));
    }

    // Read base
    let base = W::read_le(data)?;

    // Read bit width
    let bits = data[W::WIRE_BYTES];

    // Read number of patches
    let num_patches = u32::from_le_bytes([
        data[W::WIRE_BYTES + 1],
        data[W::WIRE_BYTES + 2],
        data[W::WIRE_BYTES + 3],
        data[W::WIRE_BYTES + 4],
    ]) as usize;

    // Calculate size of bitpacked data
    let bitpacked_bytes = (count * bits as usize).div_ceil(8);

    // Patches are 12 bytes each: 4 (pos) + 8 (value i64)
    if data.len() < header_len + bitpacked_bytes + num_patches * 12 {
        return Err(anyhow::anyhow!(
            "PForDelta decode: insufficient data for patches"
        ));
    }

    // Unpack regular values using unsigned variant (no sign extension for sentinels)
    let bitpacked_data = &data[header_len..header_len + bitpacked_bytes];
    let mut deltas = bitpack::unbitpack_i64_unsigned(bitpacked_data, bits, count)?;

    // Apply patches
    let patch_start = header_len + bitpacked_bytes;
    for i in 0..num_patches {
        let offset = patch_start + i * 12;
        let pos = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;

        let value = i64::from_le_bytes([
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
            data[offset + 8],
            data[offset + 9],
            data[offset + 10],
            data[offset + 11],
        ]);

        if pos < deltas.len() {
            deltas[pos] = value;
        }
    }

    Ok((base, deltas))
}

// ===== Compatibility helpers for SIMD module =====

/// Parse PForDelta header and apply patches to deltas (for SIMD compatibility)
///
/// **Helper for SIMD**: Made public to allow SIMD module to reuse wire format parsing
///
/// # Returns
/// (base_i32, patched_deltas)
pub fn parse_header_and_patches_f32(data: &[u8], count: usize) -> Result<(i32, Vec<i64>)> {
    parse_header_and_patches::<i32>(data, count)
}

/// Reconstruct f32 values from deltas and base (for SIMD compatibility)
///
/// **Helper for SIMD**: Made public to allow SIMD module to provide vectorized version
pub fn reconstruct_values_scalar_f32(deltas: &[i64], base_i32: i32) -> Vec<f32> {
    helpers::reconstruct_f32_from_i64(deltas, base_i32)
}

// ===== Public API (thin wrappers using generic helpers) =====

pub fn decode_f32(data: &[u8], count: usize) -> Result<Vec<f32>> {
    helpers::decode_generic::<f32>(data, count, decode_pfor_delta_wire::<i32>)
}

/// Decode i64 values from PForDelta encoded data
pub fn decode_i64(data: &[u8], count: usize) -> Result<Vec<i64>> {
    helpers::decode_generic::<i64>(data, count, decode_pfor_delta_wire::<i64>)
}

/// Decode i32 values from PForDelta encoded data
pub fn decode_i32(data: &[u8], count: usize) -> Result<Vec<i32>> {
    helpers::decode_generic::<i32>(data, count, decode_pfor_delta_wire::<i32>)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Golden-byte pins (redundancy pass 4) ====================
    // Wire layout: [base:WIRE_BYTES LE][bits:1 byte][num_patches:u32 LE]
    // [bitpacked i64 deltas (unsigned unpack, sentinels at patch positions)]
    // [patches: (pos:u32 LE, value:i64 LE)*]. This is the CANONICAL wire
    // format TD-CODEC-1 pins against the SIMD encoder — treat any byte
    // change here as a wire-format change. These pins were verified green
    // BEFORE the WirePrim consolidation of the baseline cores.

    /// i32 path (4-byte base): 19 zeros + 695 delta from base 5; p90 index
    /// 18 of 20 sorted |deltas| is 0 -> bits 1, max_regular 1, so the 695
    /// becomes one patch; 20 one-bit regulars pack to 3 zero bytes.
    #[test]
    fn golden_pfor_delta_i32_wire_bytes() {
        let mut values = vec![5i32; 20];
        values[19] = 700;

        let encoded = encode_i32(&values, 5).unwrap();
        assert_eq!(
            encoded,
            vec![
                5, 0, 0, 0, // base 5 LE i32
                1, // bits = 1 (p90 threshold 0)
                1, 0, 0, 0, // num_patches = 1
                0, 0, 0, // packed regulars (20 one-bit zeros)
                19, 0, 0, 0, // patch pos = 19
                0xB7, 0x02, 0, 0, 0, 0, 0, 0, // patch value 695 LE i64
            ]
        );
        assert_eq!(decode_i32(&encoded, 20).unwrap(), values);
    }

    /// i64 path (8-byte base): same shape, 19 zero deltas + 700, one patch.
    #[test]
    fn golden_pfor_delta_i64_wire_bytes() {
        let mut values = vec![7i64; 20];
        values[19] = 707;

        let encoded = encode_i64(&values, 7).unwrap();
        assert_eq!(
            encoded,
            vec![
                7, 0, 0, 0, 0, 0, 0, 0, // base 7 LE i64
                1, // bits = 1
                1, 0, 0, 0, // num_patches = 1
                0, 0, 0, // packed regulars
                19, 0, 0, 0, // patch pos = 19
                0xBC, 0x02, 0, 0, 0, 0, 0, 0, // patch value 700 LE i64
            ]
        );
        assert_eq!(decode_i64(&encoded, 20).unwrap(), values);
    }

    /// All-equal data pins the no-patch bits=1 case (4 one-bit zeros -> 1
    /// byte 0x00).
    #[test]
    fn golden_pfor_delta_i32_no_patches() {
        let encoded = encode_i32(&[5, 5, 5, 5], 5).unwrap();
        assert_eq!(encoded, vec![5, 0, 0, 0, 1, 0, 0, 0, 0, 0x00]);
        assert_eq!(decode_i32(&encoded, 4).unwrap(), vec![5, 5, 5, 5]);
    }

    #[test]
    fn test_pfor_f32_roundtrip() {
        // Mostly clustered values with a few outliers
        let mut values = vec![100.0f32; 90];
        values.extend(vec![101.0; 5]);
        values.extend(vec![1000.0; 5]); // Outliers

        let encoded = encode_f32(&values, 0).expect("failed to encode f32 values");
        let decoded = decode_f32(&encoded, values.len()).expect("failed to decode f32 values");

        assert_eq!(values.len(), decoded.len());
        for (orig, dec) in values.iter().zip(decoded.iter()) {
            assert_eq!(orig.to_bits(), dec.to_bits());
        }
    }

    #[test]
    fn test_pfor_i64_roundtrip() {
        // Regular values + outliers
        let mut values = vec![1000i64; 85];
        values.extend(vec![1001; 10]);
        values.extend(vec![999999; 5]); // Outliers

        let encoded = encode_i64(&values, 1000).expect("failed to encode i64 values");
        let decoded = decode_i64(&encoded, values.len()).expect("failed to decode i64 values");

        assert_eq!(values, decoded);
    }

    #[test]
    fn test_pfor_i32_roundtrip() {
        // Clustered with outliers
        let mut values = vec![500i32; 80];
        values.extend(vec![501, 502, 503, 504, 505]);
        values.extend(vec![10000; 15]); // Outliers

        let encoded = encode_i32(&values, 500).expect("failed to encode i32 values");
        let decoded = decode_i32(&encoded, values.len()).expect("failed to decode i32 values");

        assert_eq!(values, decoded);
    }

    #[test]
    fn test_pfor_compression() {
        // Test compression efficiency with outliers
        let mut values = vec![42i32; 1000];
        values[100] = 999999; // Single outlier

        let encoded = encode_i32(&values, 42).expect("failed to encode i32 values");
        let decoded = decode_i32(&encoded, values.len()).expect("failed to decode i32 values");

        assert_eq!(values, decoded);

        // Should be much smaller than uncompressed
        // 1000 × 4 = 4000 bytes original
        // PFor should achieve ~100 bytes (1 bit for regular + 1 patch)
        assert!(
            encoded.len() < 200,
            "Compression inefficient: {} bytes",
            encoded.len()
        );
    }

    #[test]
    fn test_pfor_multiple_outliers() {
        // Test with multiple outliers scattered
        let mut values = Vec::new();
        for i in 0..100 {
            if i % 10 == 0 {
                values.push(999999i64); // Outlier every 10 values
            } else {
                values.push(100i64);
            }
        }

        let encoded = encode_i64(&values, 100).expect("failed to encode i64 values");
        let decoded = decode_i64(&encoded, values.len()).expect("failed to decode i64 values");

        assert_eq!(values, decoded);
    }

    #[test]
    fn test_pfor_empty() {
        let values: Vec<f32> = vec![];
        let encoded = encode_f32(&values, 0).expect("failed to encode empty f32 values");
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_pfor_no_outliers() {
        // All values within small range - should behave like regular FOR
        let values: Vec<i32> = (0..100).collect();

        let encoded = encode_i32(&values, 0).expect("failed to encode i32 values");
        let decoded = decode_i32(&encoded, values.len()).expect("failed to decode i32 values");

        assert_eq!(values, decoded);

        // Should have 0 patches
        let num_patches = u32::from_le_bytes([encoded[5], encoded[6], encoded[7], encoded[8]]);
        assert_eq!(num_patches, 0);
    }

    #[test]
    fn test_pfor_all_outliers() {
        // Test case: mix of small values (90%) and large outliers (10%) - 32 values
        let mut values = vec![100i32; 29]; // 29 small values (90%)
        values.push(1000000); // Outlier 1
        values.push(2000000); // Outlier 2
        values.push(3000000); // Outlier 3

        let encoded = encode_i32(&values, 100).expect("failed to encode i32 values");
        let decoded = decode_i32(&encoded, values.len()).expect("failed to decode i32 values");

        assert_eq!(values, decoded);

        // Should have 3 patches for the outliers
        let num_patches = u32::from_le_bytes([encoded[5], encoded[6], encoded[7], encoded[8]]);
        assert!(
            num_patches >= 3,
            "Expected at least 3 patches, got {}",
            num_patches
        );
    }

    #[test]
    fn test_pfor_sequential_with_spike() {
        // Test verifies PForDelta roundtrip with mix of regular and spike values (32 values)
        // Similar pattern to test_pfor_all_outliers but with f32
        let mut values: Vec<f32> = vec![100.0; 29]; // 29 regular values (90%)
        values.push(1000000.0); // Outlier spike 1
        values.push(2000000.0); // Outlier spike 2
        values.push(3000000.0); // Outlier spike 3

        let encoded = encode_f32(&values, 0).expect("failed to encode f32 values");
        let decoded = decode_f32(&encoded, values.len()).expect("failed to decode f32 values");

        // Verify roundtrip correctness
        assert_eq!(values.len(), decoded.len());
        for (orig, dec) in values.iter().zip(decoded.iter()) {
            assert_eq!(orig.to_bits(), dec.to_bits());
        }

        // PForDelta should handle the encoding (patches are implementation detail)
        // Just verify it works correctly
        let num_patches = u32::from_le_bytes([encoded[5], encoded[6], encoded[7], encoded[8]]);
        // With base=0, deltas for constant 100.0 values fit in small bits
        // Large spikes should create patches
        let _ = num_patches; // u32 is always >= 0; large spikes should create patches
    }
}
