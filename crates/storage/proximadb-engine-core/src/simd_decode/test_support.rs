// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Shared golden-vector test support for the per-target SIMD decoder batteries.
//!
//! Every per-target test module (scalar reference, AVX2, NEON) drives the SAME
//! golden corpora through these helpers, so all targets run identical vectors:
//! a target-specific divergence fails its own battery instead of silently
//! drifting from the others.

use super::bitpacked_scalar::ScalarDecoder;
use super::traits::SimdDecoder;

/// Pack `values` at `bits` per value, LSB-first across byte boundaries —
/// the encoder side of the bitpacked contract under test.
pub fn create_bitpacked_data(values: &[u64], bits: u8) -> Vec<u8> {
    let total_bits = values.len() * bits as usize;
    let byte_count = total_bits.div_ceil(8);
    let mut result = vec![0u8; byte_count];

    let mask = if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };

    for (i, &value) in values.iter().enumerate() {
        let bit_offset = i * bits as usize;
        let byte_offset = bit_offset / 8;
        let bit_in_byte = bit_offset % 8;

        let masked_value = value & mask;

        let mut remaining_bits = bits as usize;
        let mut current_byte = byte_offset;
        let mut current_bit = bit_in_byte;
        let mut value_bits = masked_value;

        while remaining_bits > 0 {
            let bits_in_byte = (8 - current_bit).min(remaining_bits);
            let byte_mask = ((1u64 << bits_in_byte) - 1) as u8;
            result[current_byte] |= ((value_bits as u8) & byte_mask) << current_bit;

            value_bits >>= bits_in_byte;
            remaining_bits -= bits_in_byte;
            current_byte += 1;
            current_bit = 0;
        }
    }

    result
}

/// Golden corpus: identity ramp `0..len`.
pub fn ramp(len: usize) -> Vec<u64> {
    (0..len).map(|i| i as u64).collect()
}

/// Golden corpus: `i * stride` ramp for `0..len`.
pub fn scaled_ramp(len: usize, stride: u64) -> Vec<u64> {
    (0..len as u64).map(|i| i * stride).collect()
}

/// Assert `decoder` round-trips `values` (packed at `bits`) into i64 output.
pub fn assert_decodes_i64(decoder: &dyn SimdDecoder, values: &[u64], bits: u8) {
    let label = decoder.name();
    let packed = create_bitpacked_data(values, bits);

    let mut output = vec![0i64; values.len()];
    let count = decoder
        .decode_bitpacked_i64(&packed, bits, &mut output)
        .unwrap_or_else(|e| panic!("{label}: decode_bitpacked_i64({bits} bits) failed: {e}"));

    assert_eq!(
        count,
        values.len(),
        "{label}: count mismatch at {bits} bits"
    );
    for (i, (&expected, &actual)) in values.iter().zip(output.iter()).enumerate() {
        assert_eq!(
            actual as u64, expected,
            "{label}: mismatch at index {i} for {bits} bits"
        );
    }
}

/// Assert `decoder` round-trips `values` (packed at `bits`) into i32 output.
pub fn assert_decodes_i32(decoder: &dyn SimdDecoder, values: &[u64], bits: u8) {
    let label = decoder.name();
    let packed = create_bitpacked_data(values, bits);

    let mut output = vec![0i32; values.len()];
    let count = decoder
        .decode_bitpacked_i32(&packed, bits, &mut output)
        .unwrap_or_else(|e| panic!("{label}: decode_bitpacked_i32({bits} bits) failed: {e}"));

    assert_eq!(
        count,
        values.len(),
        "{label}: count mismatch at {bits} bits"
    );
    for (i, (&expected, &actual)) in values.iter().zip(output.iter()).enumerate() {
        assert_eq!(
            actual as u64, expected,
            "{label}: mismatch at index {i} for {bits} bits"
        );
    }
}

/// Assert `decoder` matches the scalar reference across the golden bit-width
/// sweep (byte-aligned and byte-crossing widths, 100-value corpus).
pub fn assert_matches_scalar_reference(decoder: &dyn SimdDecoder) {
    let label = decoder.name();
    let reference = ScalarDecoder::new();

    for bits in [4u8, 8, 12, 16, 20, 24, 32] {
        let values: Vec<u64> = (0..100).map(|i| i as u64 % (1 << bits.min(63))).collect();
        let packed = create_bitpacked_data(&values, bits);

        let mut target_output = vec![0i64; values.len()];
        let mut reference_output = vec![0i64; values.len()];

        let target_count = decoder
            .decode_bitpacked_i64(&packed, bits, &mut target_output)
            .unwrap_or_else(|e| panic!("{label}: decode {bits} bits failed: {e}"));
        let reference_count = reference
            .decode_bitpacked_i64(&packed, bits, &mut reference_output)
            .unwrap_or_else(|e| panic!("Scalar: decode {bits} bits failed: {e}"));

        assert_eq!(
            target_count, reference_count,
            "{label} vs Scalar: count mismatch for {bits} bits"
        );
        assert_eq!(
            target_output, reference_output,
            "{label} vs Scalar: output mismatch for {bits} bits"
        );
    }
}

/// Run the golden decode battery every hardware target must pass:
/// byte-aligned widths (8/16/32), an i32 decode, and equivalence with the
/// scalar reference across the bit-width sweep.
pub fn run_golden_decode_battery(decoder: &dyn SimdDecoder) {
    // Byte-aligned widths (the AVX2/NEON fast paths)
    assert_decodes_i64(decoder, &scaled_ramp(32, 1), 8);
    assert_decodes_i64(decoder, &scaled_ramp(32, 1_000), 16);
    assert_decodes_i64(decoder, &scaled_ramp(32, 100_000), 32);
    assert_decodes_i32(decoder, &scaled_ramp(32, 100), 16);

    // Byte-crossing and non-aligned widths must match the scalar reference
    assert_matches_scalar_reference(decoder);
}
