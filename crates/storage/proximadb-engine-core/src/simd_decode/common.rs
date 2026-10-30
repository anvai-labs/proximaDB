// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Scalar fallbacks shared by every `SimdDecoder` target.
//!
//! The byte-aligned SIMD kernels (real AVX2/NEON intrinsics) stay in their
//! per-target modules by design; the pieces below are the scalar work every
//! target performs identically — non-byte-aligned bit extraction, delta-header
//! parsing, prefix-sum finalization, and f32 bit-pattern reconstruction — so
//! they are defined once and cannot drift between targets.

use anyhow::{Result, bail};

/// Parsed `[count: u32 LE][bits: u8][packed deltas...]` delta payload header.
pub(crate) struct DeltaHeader<'a> {
    /// Value count advertised by the encoder.
    pub(crate) count: usize,
    /// Bits per packed delta value.
    pub(crate) bits: u8,
    /// Packed delta payload following the 5-byte header.
    pub(crate) packed: &'a [u8],
}

/// Parse the 5-byte delta header from the front of `input`.
///
/// Returns `Ok(None)` when the input is empty or the advertised count is zero
/// (nothing to decode); errors when the header is truncated.
pub(crate) fn parse_delta_header(input: &[u8]) -> Result<Option<DeltaHeader<'_>>> {
    if input.is_empty() {
        return Ok(None);
    }
    if input.len() < 5 {
        bail!("Input too short for delta header");
    }
    let count = u32::from_le_bytes([input[0], input[1], input[2], input[3]]) as usize;
    let bits = input[4];
    if count == 0 {
        return Ok(None);
    }
    Ok(Some(DeltaHeader {
        count,
        bits,
        packed: &input[5..],
    }))
}

/// Extract non-byte-aligned bit widths (scalar path shared by all targets).
pub(crate) fn decode_variable_bits_i64(
    input: &[u8],
    bits: u8,
    output: &mut [i64],
) -> Result<usize> {
    let bits_usize = bits as usize;
    let total_bits = input.len() * 8;
    let max_values = total_bits / bits_usize;
    let count = max_values.min(output.len());

    if count == 0 {
        return Ok(0);
    }

    let mask: u64 = if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };

    for (i, out_val) in output.iter_mut().enumerate().take(count) {
        let bit_offset = i * bits_usize;
        let byte_offset = bit_offset / 8;
        let bit_in_byte = bit_offset % 8;

        let bytes_needed = (bit_in_byte + bits_usize).div_ceil(8);
        if byte_offset + bytes_needed > input.len() {
            break;
        }

        let mut value: u64 = 0;
        for j in 0..bytes_needed.min(8) {
            if byte_offset + j < input.len() {
                value |= (input[byte_offset + j] as u64) << (j * 8);
            }
        }

        *out_val = ((value >> bit_in_byte) & mask) as i64;
    }

    Ok(count)
}

/// Extract non-byte-aligned bit widths into i32 (scalar path shared by all
/// targets).
pub(crate) fn decode_variable_bits_i32(
    input: &[u8],
    bits: u8,
    output: &mut [i32],
) -> Result<usize> {
    let bits_usize = bits as usize;
    let total_bits = input.len() * 8;
    let max_values = total_bits / bits_usize;
    let count = max_values.min(output.len());

    if count == 0 {
        return Ok(0);
    }

    let mask: u32 = if bits == 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    };

    for (i, out_val) in output.iter_mut().enumerate().take(count) {
        let bit_offset = i * bits_usize;
        let byte_offset = bit_offset / 8;
        let bit_in_byte = bit_offset % 8;

        let bytes_needed = (bit_in_byte + bits_usize).div_ceil(8);
        if byte_offset + bytes_needed > input.len() {
            break;
        }

        let mut value: u64 = 0;
        for j in 0..bytes_needed.min(8) {
            if byte_offset + j < input.len() {
                value |= (input[byte_offset + j] as u64) << (j * 8);
            }
        }

        *out_val = (((value >> bit_in_byte) as u32) & mask) as i32;
    }

    Ok(count)
}

/// Reconstruct f32 values from bit-pattern deltas (`value_bits = base + delta`).
pub(crate) fn reconstruct_delta_f32(deltas: &[i64], base: f32, output: &mut [f32]) {
    let base_bits = base.to_bits() as i64;
    for (delta, out_val) in deltas.iter().zip(output.iter_mut()) {
        *out_val = f32::from_bits((base_bits + delta) as u32);
    }
}

/// Apply the sequential prefix sum `out[i] += out[i-1]` seeded with `base`.
pub(crate) fn prefix_sum_i64(output: &mut [i64], base: i64) {
    if output.is_empty() {
        return;
    }

    output[0] += base;

    for i in 1..output.len() {
        output[i] += output[i - 1];
    }
}
