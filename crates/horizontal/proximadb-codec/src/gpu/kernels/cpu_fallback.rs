// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Shared CPU fallback cores for the deferred GPU kernels (redundancy pass 3).
//!
//! Every scheme module (`cuda`, `rocm`, `metal`, `opencl`) ships the same
//! story: real GPU kernel compilation is DEFERRED, so each encode/decode path
//! currently executes a scalar CPU fallback. Those fallback loops were pasted
//! verbatim into every scheme function (41 clone blocks at the pass-3 triage;
//! `rocm.rs` was a literal fork of `opencl.rs`). This module holds the ONE
//! copy of each scalar core and delegates to [`crate::baseline::functions`]
//! wherever the wire layout already matches — the baseline bit-packer packs
//! the same little-endian, LSB-first bit stream these fallbacks implement.
//!
//! ## Layout
//!
//! Value `i` occupies stream bits `[i*bits, (i+1)*bits)`; stream bit `s` lives
//! at byte `s/8`, bit `s%8`. Bytes are therefore byte-identical to the
//! historical per-scheme loops for every width those loops handled correctly.
//!
//! ## Truncation fix (pass-3 latent bug)
//!
//! The historical loops wrote/read at most TWO bytes per value, silently
//! dropping the high bits of any value whose packed span crossed a third byte
//! — e.g. `bits = 20` at odd byte alignments, or full `bits = 32` words.
//! Delegating to the baseline core writes/reads every bit, so those widths
//! now round-trip exactly (see `test_bitpack_roundtrip_bits_20`).
//!
//! This module is intentionally NOT feature-gated: like `kernels::utils` it
//! compiles in default builds so its round-trip tests run everywhere. The
//! scheme modules that call it stay gated on `feature = "gpu"`/platform.

use anyhow::Result;

use crate::baseline::functions::bitpack;

/// Delta-encode `f32` values against a base (`delta[i] = values[i] - base`).
pub fn delta_encode_f32(values: &[f32], base: f32) -> Vec<i64> {
    values.iter().map(|&v| (v - base) as i64).collect()
}

/// Reverse of [`delta_encode_f32`] (`value[i] = delta[i] + base`).
pub fn delta_decode_f32(deltas: &[i64], base: f32) -> Vec<f32> {
    deltas.iter().map(|&d| d as f32 + base).collect()
}

/// BitPacked encoding: pack the low `bits` bits of each value's IEEE 754
/// word into the LSB-first bit stream.
pub fn bitpack_encode_f32(values: &[f32], bits: u8) -> Result<Vec<u8>> {
    let words: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
    bitpack::bitpack_u32(&words, bits)
}

/// BitPacked decoding: unpack `count` `bits`-wide words and reinterpret each
/// as an `f32` via `from_bits`.
pub fn bitpack_decode_f32(packed: &[u8], bits: u8, count: usize) -> Result<Vec<f32>> {
    let words = bitpack::unbitpack_u32(packed, bits, count)?;
    Ok(words.into_iter().map(f32::from_bits).collect())
}

/// FrameOfReference encoding: offsets from `reference`, low `bits` of each
/// offset's two's-complement word bit-packed.
pub fn frame_of_reference_encode_f32(values: &[f32], reference: i64, bits: u8) -> Result<Vec<u8>> {
    let reference_f32 = reference as f32;
    let words: Vec<u32> = values
        .iter()
        .map(|&v| ((v - reference_f32) as i64) as u32)
        .collect();
    bitpack::bitpack_u32(&words, bits)
}

/// FrameOfReference decoding: unpack the offsets and add `reference` back.
pub fn frame_of_reference_decode_f32(
    packed: &[u8],
    reference: i64,
    bits: u8,
    count: usize,
) -> Result<Vec<f32>> {
    let words = bitpack::unbitpack_u32(packed, bits, count)?;
    let reference_f32 = reference as f32;
    Ok(words
        .into_iter()
        .map(|w| (w as i32) as f32 + reference_f32)
        .collect())
}

/// Zigzag encoding: 32-bit zigzag of each value's IEEE 754 word
/// (`zz = (n << 1) ^ (n >> 31)`), bit-packed at `bits` per value.
pub fn zigzag_encode_f32(values: &[f32], bits: u8) -> Result<Vec<u8>> {
    let words: Vec<u32> = values
        .iter()
        .map(|&v| {
            let n = v.to_bits() as i32;
            ((n << 1) ^ (n >> 31)) as u32
        })
        .collect();
    bitpack::bitpack_u32(&words, bits)
}

/// Zigzag decoding: unpack the zigzag words and reverse the mapping
/// (`n = (zz >>> 1) ^ -(zz & 1)`).
pub fn zigzag_decode_f32(packed: &[u8], bits: u8, count: usize) -> Result<Vec<f32>> {
    let words = bitpack::unbitpack_u32(packed, bits, count)?;
    Ok(words
        .into_iter()
        .map(|w| {
            let zz = w as i32;
            let n = ((zz as u32) >> 1) as i32 ^ -(zz & 1);
            f32::from_bits(n as u32)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_roundtrip() {
        let values = vec![10.0f32, 20.0, 30.0, 40.0];
        let base = 5.0f32;

        let deltas = delta_encode_f32(&values, base);
        let decoded = delta_decode_f32(&deltas, base);

        for (original, recovered) in values.iter().zip(decoded.iter()) {
            assert!((original - recovered).abs() < 0.01);
        }
    }

    #[test]
    fn test_bitpack_layout_is_lsb_first() {
        // Golden layout pin: value i occupies stream bits [i*bits, (i+1)*bits),
        // LSB-first within the little-endian byte stream.
        let values = [
            f32::from_bits(0b0000_1010u32),
            f32::from_bits(0b0000_0101u32),
        ];
        let packed = bitpack_encode_f32(&values, 4).unwrap();
        assert_eq!(packed, vec![0b0101_1010u8]);

        let words = [0xFFu32, 0x0F];
        let values: Vec<f32> = words.iter().map(|&w| f32::from_bits(w)).collect();
        let packed = bitpack_encode_f32(&values, 8).unwrap();
        assert_eq!(packed, vec![0xFFu8, 0x0F]);
    }

    #[test]
    fn test_bitpack_roundtrip_bits_20() {
        // Pass-3 latent bug: the historical per-scheme loops wrote at most two
        // bytes per value, so bits>16 at odd alignments (20 % 8 = 4) silently
        // dropped the top bits. The shared core must round-trip all 20 bits.
        let mask = (1u32 << 20) - 1;
        let words = [0x000F_FFFFu32, 0x000A_5A5A, 0x0000_0001, 0x0008_0001];
        let values: Vec<f32> = words.iter().map(|&w| f32::from_bits(w)).collect();

        let packed = bitpack_encode_f32(&values, 20).unwrap();
        // 4 values x 20 bits = 80 bits = 10 bytes (the old loops also sized
        // the buffer correctly; they just never wrote/-read the high bits).
        assert_eq!(packed.len(), 10);

        let decoded = bitpack_decode_f32(&packed, 20, values.len()).unwrap();
        for (i, value) in decoded.iter().enumerate() {
            assert_eq!(
                value.to_bits(),
                words[i] & mask,
                "20-bit word {} must round-trip exactly",
                i
            );
        }
    }

    #[test]
    fn test_bitpack_roundtrip_bits_32() {
        // Full words need four bytes per value on misaligned entries — also
        // truncated by the historical two-byte loops.
        let words = [u32::MAX, 0x1234_5678, 0x8000_0001];
        let values: Vec<f32> = words.iter().map(|&w| f32::from_bits(w)).collect();

        let packed = bitpack_encode_f32(&values, 32).unwrap();
        let decoded = bitpack_decode_f32(&packed, 32, values.len()).unwrap();
        for (i, value) in decoded.iter().enumerate() {
            assert_eq!(value.to_bits(), words[i], "full word {} must survive", i);
        }
    }

    #[test]
    fn test_frame_of_reference_roundtrip() {
        let values = vec![100.0f32, 105.0, 110.0, 115.0, 120.0];
        let reference = 100i64;
        let bits = 8u8;

        let packed = frame_of_reference_encode_f32(&values, reference, bits).unwrap();
        let decoded =
            frame_of_reference_decode_f32(&packed, reference, bits, values.len()).unwrap();

        assert_eq!(decoded.len(), values.len());
        for (original, recovered) in values.iter().zip(decoded.iter()) {
            assert!((original - recovered).abs() < 1.0);
        }
    }

    #[test]
    fn test_zigzag_roundtrip() {
        // Zigzag operates on the IEEE word, not the float value: ordinary
        // floats like -10.0 zigzag to ~2^31 and cannot fit 16 bits by design.
        // Pick words whose zigzag fits: positive n zigzags to 2n, n = -1 to 1.
        // (0xFFFFFFFF is a negative NaN pattern; comparisons are bitwise, so
        // the payload survives verbatim.)
        let words = [
            0x0000_0000u32,
            0x0000_0001,
            0x0000_000A,
            0x0000_0064,
            0xFFFF_FFFF,
        ];
        let values: Vec<f32> = words.iter().map(|&w| f32::from_bits(w)).collect();

        let packed = zigzag_encode_f32(&values, 16).unwrap();
        let decoded = zigzag_decode_f32(&packed, 16, values.len()).unwrap();

        assert_eq!(decoded.len(), values.len());
        for (original, recovered) in values.iter().zip(decoded.iter()) {
            assert_eq!(original.to_bits(), recovered.to_bits());
        }
    }
}
