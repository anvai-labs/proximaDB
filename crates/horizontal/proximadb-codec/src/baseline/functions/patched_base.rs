// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Patched Base Encoding - Baseline (pure scalar) implementation
//!
//! **ARCHITECTURE NOTE**: This is the BASELINE implementation.
//! - NO SIMD intrinsics allowed
//! - NO GPU code
//! - Pure portable Rust only
//!
//! For SIMD acceleration, see: `src/storage/engines/core/ops/proximacodec/simd.rs`
//! For GPU acceleration, see: `src/storage/engines/core/ops/proximacodec/impls/gpu/`
//!
//! ## Algorithm Overview
//!
//! Patched Base encoding handles data with outliers by separating regular values
//! from patches (outliers). Unlike PForDelta which uses percentile-based threshold,
//! PatchedBase uses a fixed threshold based on the patch_bits parameter.
//!
//! ## Key Differences from PForDelta
//!
//! - **PForDelta**: Dynamic threshold (90th percentile), optimal for unknown distributions
//! - **PatchedBase**: Fixed threshold (2^patch_bits), optimal when bit width is known
//!
//! ## Wire Format
//!
//! ```text
//! [base:4 bytes][patch_bits:1 byte][num_patches:4 bytes]
//! [bitpacked_values...][patches:(pos:4, value:8)*]
//! ```
//!
//! Note: All values are stored (not just regular), patches indicate which need full precision.

use anyhow::Result;

// ===== Bitpacking delegation to shared helpers =====
//
// All bitpacking operations now use the shared helpers in bitpack.rs
// to avoid code duplication and ensure consistent sign extension behavior.

use super::bitpack;
use super::helpers;
use super::helpers::WirePrim;

// ===== Core wire format encoding functions =====

/// Single patched-base encode core (redundancy pass 4: the i32-base and
/// i64-base twins differed only in the base slot width — deltas, threshold,
/// and patches were already i64 on both sides; golden tests pin the layout).
fn encode_patched_base_wire<W: WirePrim>(
    wire_values: &[W],
    base: W,
    patch_bits: u8,
) -> Result<Vec<u8>> {
    if wire_values.is_empty() {
        return Ok(Vec::new());
    }

    if patch_bits > 63 {
        return Err(anyhow::anyhow!("Patch bits {} exceeds 63", patch_bits));
    }

    let mut result = Vec::new();
    base.write_le(&mut result);
    result.push(patch_bits);

    // Compute deltas in i64 (NO OVERFLOW! for the i32 path; wrapping for i64)
    let deltas: Vec<i64> = wire_values
        .iter()
        .map(|&v| v.widen().wrapping_sub(base.widen()))
        .collect();

    // Threshold for patching
    let threshold = if patch_bits >= 63 {
        i64::MAX
    } else {
        1i64 << patch_bits
    };

    // Identify patches
    let mut patches = Vec::new();
    for (idx, &delta) in deltas.iter().enumerate() {
        if delta.unsigned_abs() >= threshold as u64 {
            patches.push((idx as u32, delta));
        }
    }

    // Store number of patches
    let num_patches = patches.len() as u32;
    result.extend_from_slice(&num_patches.to_le_bytes());

    // Bitpack all values
    let packed = bitpack::bitpack_i64(&deltas, patch_bits)?;
    result.extend(packed);

    // Store patches (position:4 bytes + value:8 bytes = 12 bytes per patch)
    for (pos, value) in patches {
        result.extend_from_slice(&pos.to_le_bytes());
        result.extend_from_slice(&value.to_le_bytes());
    }

    Ok(result)
}

// ===== Public API (thin wrappers using generic helpers) =====

/// Encode f32 values using Patched Base (raw, no headers)
///
/// # Algorithm
/// 1. Convert f32 to i32 bits, compute deltas in i64 (NO OVERFLOW!)
/// 2. Identify outliers: |delta| >= 2^patch_bits
/// 3. Store all values bitpacked at patch_bits width
/// 4. Store outlier patches with full precision
///
/// # Parameters
/// - `values`: f32 slice to encode
/// - `base`: Base value for delta calculation (i64 for compatibility)
/// - `patch_bits`: Bit width for regular values (threshold = 2^patch_bits)
///
/// # Returns
/// Raw encoded bytes (NO scheme marker, NO count header)
pub fn encode_f32(values: &[f32], base: i64, patch_bits: u8) -> Result<Vec<u8>> {
    helpers::encode_generic(values, |wire_values| {
        encode_patched_base_wire(wire_values, base as i32, patch_bits)
    })
}

/// Encode i64 values using Patched Base (raw, no headers)
pub fn encode_i64(values: &[i64], base: i64, patch_bits: u8) -> Result<Vec<u8>> {
    helpers::encode_generic(values, |wire_values| {
        encode_patched_base_wire(wire_values, base, patch_bits)
    })
}

/// Encode i32 values using Patched Base (raw, no headers)
pub fn encode_i32(values: &[i32], base: i64, patch_bits: u8) -> Result<Vec<u8>> {
    helpers::encode_generic(values, |wire_values| {
        encode_patched_base_wire(wire_values, base as i32, patch_bits)
    })
}

// ===== Core wire format decoding functions =====

/// Single patched-base decode core; header is
/// [base:WIRE_BYTES][patch_bits:1][num_patches:u32], the packed payload and
/// the (pos, i64) patches are width-independent.
fn decode_patched_base_wire<W: WirePrim>(data: &[u8], count: usize) -> Result<Vec<W>> {
    if count == 0 {
        return Ok(Vec::new());
    }

    let header_len = W::WIRE_BYTES + 5;
    if data.len() < header_len {
        return Err(anyhow::anyhow!("PatchedBase decode: insufficient data"));
    }

    // Read base
    let base = W::read_le(data)?;

    // Read patch bit width
    let patch_bits = data[W::WIRE_BYTES];

    // Read number of patches
    let num_patches = u32::from_le_bytes([
        data[W::WIRE_BYTES + 1],
        data[W::WIRE_BYTES + 2],
        data[W::WIRE_BYTES + 3],
        data[W::WIRE_BYTES + 4],
    ]) as usize;

    // Calculate size of bitpacked data
    let bitpacked_bytes = (count * patch_bits as usize).div_ceil(8);

    // Patches are 12 bytes each: 4 (pos) + 8 (value i64)
    if data.len() < header_len + bitpacked_bytes + num_patches * 12 {
        return Err(anyhow::anyhow!(
            "PatchedBase decode: insufficient data for patches"
        ));
    }

    // Unpack all values
    let bitpacked_data = &data[header_len..header_len + bitpacked_bytes];
    let mut deltas = bitpack::unbitpack_i64(bitpacked_data, patch_bits, count)?;

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

    // Reconstruct values: widen the base, add each i64 delta, narrow back
    // (truncating for i32 exactly like the old reconstruct helper).
    Ok(deltas
        .into_iter()
        .map(|d| W::from_wide(base.widen().wrapping_add(d)))
        .collect())
}

// ===== Public API (thin wrappers using generic helpers) =====

/// Decode f32 values from Patched Base encoded data
pub fn decode_f32(data: &[u8], count: usize) -> Result<Vec<f32>> {
    helpers::decode_generic::<f32>(data, count, decode_patched_base_wire::<i32>)
}

/// Decode i64 values from Patched Base encoded data
pub fn decode_i64(data: &[u8], count: usize) -> Result<Vec<i64>> {
    helpers::decode_generic::<i64>(data, count, decode_patched_base_wire::<i64>)
}

/// Decode i32 values from Patched Base encoded data
pub fn decode_i32(data: &[u8], count: usize) -> Result<Vec<i32>> {
    helpers::decode_generic::<i32>(data, count, decode_patched_base_wire::<i32>)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Golden-byte pins (redundancy pass 4) ====================
    // Wire layout: [base:WIRE_BYTES LE][patch_bits:1][num_patches:u32 LE]
    // [bitpacked i64 deltas][patches: (pos:u32 LE, value:i64 LE)*]. Both
    // widths store i64 deltas/patches; only the base slot width differs.
    // These pin the CURRENT bytes of both widths BEFORE the WirePrim
    // consolidation; any change is a wire-format change.

    /// i32 path (4-byte base): [5,700] base 5, patch_bits 4 -> deltas
    /// [0,695]; 695 >= 2^4 so one patch; packed low nibble of 695
    /// (0b0111) -> 0x70; patch value 695 = 0x2B7.
    #[test]
    fn golden_patched_base_i32_wire_bytes() {
        let encoded = encode_i32(&[5, 700], 5, 4).unwrap();
        assert_eq!(
            encoded,
            vec![
                5, 0, 0, 0, // base 5 LE i32
                4, // patch_bits
                1, 0, 0, 0,    // num_patches = 1
                0x70, // packed deltas: low nibble of 695 = 0b0111
                1, 0, 0, 0, // patch pos = 1
                0xB7, 0x02, 0, 0, 0, 0, 0, 0, // patch value 695 LE i64
            ]
        );
        assert_eq!(decode_i32(&encoded, 2).unwrap(), vec![5, 700]);
    }

    /// i64 path (8-byte base): [100,4296] base 100, patch_bits 4 -> delta
    /// 4196 = 0x1064; low nibble 0x4 -> bit 6 -> 0x40.
    #[test]
    fn golden_patched_base_i64_wire_bytes() {
        let encoded = encode_i64(&[100, 4296], 100, 4).unwrap();
        assert_eq!(
            encoded,
            vec![
                100, 0, 0, 0, 0, 0, 0, 0, // base 100 LE i64
                4, // patch_bits
                1, 0, 0, 0,    // num_patches = 1
                0x40, // packed deltas: low nibble of 4196 = 0x4
                1, 0, 0, 0, // patch pos = 1
                0x64, 0x10, 0, 0, 0, 0, 0, 0, // patch value 4196 LE i64
            ]
        );
        assert_eq!(decode_i64(&encoded, 2).unwrap(), vec![100, 4296]);
    }

    #[test]
    fn test_patched_base_f32_roundtrip() {
        // Mix of regular and outlier values
        let mut values = vec![100.0f32; 85];
        values.extend(vec![101.0; 10]);
        values.extend(vec![10000.0; 5]); // Outliers

        let encoded = encode_f32(&values, 0, 8).expect("Failed to encode f32 values");
        let decoded = decode_f32(&encoded, values.len()).expect("Failed to decode f32 values");

        assert_eq!(values.len(), decoded.len());
        for (orig, dec) in values.iter().zip(decoded.iter()) {
            assert_eq!(orig.to_bits(), dec.to_bits());
        }
    }

    #[test]
    fn test_patched_base_i64_roundtrip() {
        // Regular values + outliers
        let mut values = vec![1000i64; 90];
        values.extend(vec![1001; 5]);
        values.extend(vec![999999; 5]); // Outliers

        let encoded = encode_i64(&values, 1000, 10).expect("Failed to encode i64 values");
        let decoded = decode_i64(&encoded, values.len()).expect("Failed to decode i64 values");

        assert_eq!(values, decoded);
    }

    #[test]
    fn test_patched_base_i32_roundtrip() {
        // Clustered with outliers
        let mut values = vec![500i32; 80];
        values.extend(vec![501, 502, 503, 504, 505]);
        values.extend(vec![50000; 15]); // Outliers

        let encoded = encode_i32(&values, 500, 8).expect("Failed to encode i32 values");
        let decoded = decode_i32(&encoded, values.len()).expect("Failed to decode i32 values");

        assert_eq!(values, decoded);
    }

    #[test]
    fn test_patched_base_no_outliers() {
        // All values within threshold - should have 0 patches
        let values: Vec<i32> = (0..100).collect();

        let encoded = encode_i32(&values, 0, 10).expect("Failed to encode i32 values");
        let decoded = decode_i32(&encoded, values.len()).expect("Failed to decode i32 values");

        assert_eq!(values, decoded);

        // Should have 0 patches
        let num_patches = u32::from_le_bytes([encoded[5], encoded[6], encoded[7], encoded[8]]);
        assert_eq!(num_patches, 0);
    }

    #[test]
    fn test_patched_base_all_outliers() {
        // All values exceed threshold
        let values = vec![10000i32; 32];

        let encoded = encode_i32(&values, 0, 4).expect("Failed to encode i32 values"); // threshold = 16, all exceed
        let decoded = decode_i32(&encoded, values.len()).expect("Failed to decode i32 values");

        assert_eq!(values, decoded);

        // Should have many patches
        let num_patches = u32::from_le_bytes([encoded[5], encoded[6], encoded[7], encoded[8]]);
        assert!(num_patches > 0, "Expected patches for outliers");
    }

    #[test]
    fn test_patched_base_overflow_protection() {
        // Test with i32 extremes - would overflow with i32 deltas
        let values = vec![i32::MIN, i32::MAX, i32::MIN, i32::MAX];

        let encoded = encode_i32(&values, 0, 32).expect("Failed to encode i32 values");
        let decoded = decode_i32(&encoded, values.len()).expect("Failed to decode i32 values");

        assert_eq!(values, decoded, "Failed to roundtrip i32 extremes");
    }

    #[test]
    fn test_patched_base_empty() {
        let values: Vec<f32> = vec![];
        let encoded = encode_f32(&values, 0, 8).expect("Failed to encode f32 values");
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_patched_base_compression() {
        // Test compression efficiency with few outliers
        let mut values = vec![42i32; 1000];
        values[100] = 999999; // Single outlier
        values[500] = 888888; // Another outlier

        let encoded = encode_i32(&values, 42, 8).expect("Failed to encode i32 values");
        let decoded = decode_i32(&encoded, values.len()).expect("Failed to decode i32 values");

        assert_eq!(values, decoded);

        // Should be much smaller than uncompressed
        // 1000 × 4 = 4000 bytes original
        // PatchedBase with 8 bits: ~1000 bytes (data) + ~24 bytes (patches) + 9 bytes (header)
        // = ~1033 bytes, which is ~4x compression
        assert!(
            encoded.len() < 1500,
            "Compression inefficient: {} bytes (expected ~1000-1100)",
            encoded.len()
        );
    }
}
