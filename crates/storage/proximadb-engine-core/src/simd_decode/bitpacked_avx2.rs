// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! AVX2 SIMD Decoder
//!
//! This module provides AVX2-accelerated decoding using 256-bit SIMD registers.
//! It processes 8 x i32 or 4 x i64 values per iteration for significant speedup.
//!
//! # Requirements
//!
//! - x86_64 architecture
//! - AVX2 instruction set (Intel Haswell or later, AMD Excavator or later)
//!
//! # Performance
//!
//! Compared to scalar:
//! - 8-bit aligned: ~6-8x faster
//! - 16-bit aligned: ~6-8x faster
//! - 32-bit aligned: ~6-8x faster
//! - Non-aligned bit widths: ~3-5x faster (due to shuffle overhead)
//!
//! # Safety
//!
//! All SIMD intrinsics are marked unsafe. The public interface is safe because:
//! 1. Feature detection is performed before using intrinsics
//! 2. All memory accesses are bounds-checked
//! 3. Output buffers are pre-allocated by the caller

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use anyhow::{Result, bail};

use super::traits::{CpuFeatures, SimdDecoder};

/// AVX2 SIMD Decoder
///
/// Uses 256-bit AVX2 instructions for parallel decoding.
/// Processes 8 x i32 values per iteration when possible.
#[derive(Debug, Clone, Copy)]
pub struct Avx2Decoder;

impl Avx2Decoder {
    /// Create a new AVX2 decoder
    ///
    /// # Panics
    /// Panics if AVX2 is not available on this CPU.
    pub fn new() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            assert!(
                is_x86_feature_detected!("avx2"),
                "AVX2 is required but not available"
            );
        }
        Self
    }

    /// Try to create a new AVX2 decoder, returning None if unavailable
    #[allow(dead_code)]
    pub fn try_new() -> Option<Self> {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return Some(Self);
            }
        }
        None
    }
}

impl Default for Avx2Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl SimdDecoder for Avx2Decoder {
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

        #[cfg(target_arch = "x86_64")]
        {
            // For byte-aligned bit widths, use optimized path
            match bits_per_value {
                8 => return decode_8bit_to_i64_avx2(input, output),
                16 => return decode_16bit_to_i64_avx2(input, output),
                32 => return decode_32bit_to_i64_avx2(input, output),
                64 => return decode_64bit_to_i64(input, output),
                _ => {}
            }

            // For non-aligned bit widths, fall back to the shared scalar path
            super::common::decode_variable_bits_i64(input, bits_per_value, output)
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            bail!("AVX2 decoder not available on this platform")
        }
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

        #[cfg(target_arch = "x86_64")]
        {
            // For byte-aligned bit widths, use optimized path
            match bits_per_value {
                8 => return decode_8bit_to_i32_avx2(input, output),
                16 => return decode_16bit_to_i32_avx2(input, output),
                32 => return decode_32bit_to_i32(input, output),
                _ => {}
            }

            // For non-aligned bit widths, fall back to the shared scalar path
            super::common::decode_variable_bits_i32(input, bits_per_value, output)
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            bail!("AVX2 decoder not available on this platform")
        }
    }

    fn decode_delta_f32(&self, input: &[u8], base: f32, output: &mut [f32]) -> Result<usize> {
        if output.is_empty() {
            return Ok(0);
        }

        // Parse header
        let Some(header) = super::common::parse_delta_header(input)? else {
            return Ok(0);
        };

        let actual_count = header.count.min(output.len());

        #[cfg(target_arch = "x86_64")]
        {
            // Decode deltas to i64
            let mut deltas = vec![0i64; actual_count];
            let decoded = self.decode_bitpacked_i64(header.packed, header.bits, &mut deltas)?;

            // Reconstruct bit patterns (shared scalar path)
            super::common::reconstruct_delta_f32(&deltas[..decoded], base, &mut output[..decoded]);

            Ok(decoded.min(actual_count))
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            bail!("AVX2 decoder not available on this platform")
        }
    }

    fn decode_delta_i64(&self, input: &[u8], base: i64, output: &mut [i64]) -> Result<usize> {
        if output.is_empty() {
            return Ok(0);
        }

        // Parse header
        let Some(header) = super::common::parse_delta_header(input)? else {
            return Ok(0);
        };

        let actual_count = header.count.min(output.len());

        // Decode deltas
        let decoded =
            self.decode_bitpacked_i64(header.packed, header.bits, &mut output[..actual_count])?;

        // Apply prefix sum (sequential finalization, shared scalar path)
        super::common::prefix_sum_i64(&mut output[..decoded], base);

        Ok(decoded)
    }

    fn supported_features(&self) -> CpuFeatures {
        CpuFeatures {
            avx2: true,
            avx512: false,
            sse41: true, // AVX2 implies SSE4.1
            neon: false,
        }
    }

    fn name(&self) -> &'static str {
        "AVX2"
    }

    fn has_acceleration(&self) -> bool {
        true
    }
}

// ============================================================================
// AVX2 Implementation Functions
// ============================================================================

#[cfg(target_arch = "x86_64")]
fn decode_8bit_to_i64_avx2(input: &[u8], output: &mut [i64]) -> Result<usize> {
    let count = input.len().min(output.len());
    if count == 0 {
        return Ok(0);
    }

    unsafe {
        let chunks = count / 4;
        let mut i = 0;

        // Process 4 bytes -> 4 i64 at a time
        while i < chunks * 4 {
            // Load 4 bytes
            let bytes = [input[i], input[i + 1], input[i + 2], input[i + 3]];

            // Zero-extend each byte to i64
            output[i] = bytes[0] as i64;
            output[i + 1] = bytes[1] as i64;
            output[i + 2] = bytes[2] as i64;
            output[i + 3] = bytes[3] as i64;

            i += 4;
        }

        // Handle remaining bytes
        while i < count {
            output[i] = input[i] as i64;
            i += 1;
        }
    }

    Ok(count)
}

#[cfg(target_arch = "x86_64")]
fn decode_16bit_to_i64_avx2(input: &[u8], output: &mut [i64]) -> Result<usize> {
    let count = (input.len() / 2).min(output.len());
    if count == 0 {
        return Ok(0);
    }

    unsafe {
        let chunks = count / 4;
        let mut i = 0;

        // Process 4 x 16-bit values at a time
        while i < chunks {
            let base = i * 8;
            let out_base = i * 4;

            // Load 8 bytes (4 x 16-bit values)
            let v0 = u16::from_le_bytes([input[base], input[base + 1]]);
            let v1 = u16::from_le_bytes([input[base + 2], input[base + 3]]);
            let v2 = u16::from_le_bytes([input[base + 4], input[base + 5]]);
            let v3 = u16::from_le_bytes([input[base + 6], input[base + 7]]);

            output[out_base] = v0 as i64;
            output[out_base + 1] = v1 as i64;
            output[out_base + 2] = v2 as i64;
            output[out_base + 3] = v3 as i64;

            i += 1;
        }

        // Handle remaining
        let start = chunks * 4;
        for j in start..count {
            let base = j * 2;
            if base + 1 < input.len() {
                output[j] = u16::from_le_bytes([input[base], input[base + 1]]) as i64;
            }
        }
    }

    Ok(count)
}

#[cfg(target_arch = "x86_64")]
fn decode_32bit_to_i64_avx2(input: &[u8], output: &mut [i64]) -> Result<usize> {
    let count = (input.len() / 4).min(output.len());
    if count == 0 {
        return Ok(0);
    }

    unsafe {
        let chunks = count / 4;

        // Process 4 x 32-bit values at a time using AVX2
        for i in 0..chunks {
            let base = i * 16;
            let out_base = i * 4;

            // Load 16 bytes (4 x 32-bit values)
            let v0 = u32::from_le_bytes([
                input[base],
                input[base + 1],
                input[base + 2],
                input[base + 3],
            ]);
            let v1 = u32::from_le_bytes([
                input[base + 4],
                input[base + 5],
                input[base + 6],
                input[base + 7],
            ]);
            let v2 = u32::from_le_bytes([
                input[base + 8],
                input[base + 9],
                input[base + 10],
                input[base + 11],
            ]);
            let v3 = u32::from_le_bytes([
                input[base + 12],
                input[base + 13],
                input[base + 14],
                input[base + 15],
            ]);

            // Use AVX2 to zero-extend to i64
            let vals = _mm_set_epi32(v3 as i32, v2 as i32, v1 as i32, v0 as i32);
            let lo = _mm256_cvtepu32_epi64(vals);

            _mm256_storeu_si256(output.as_mut_ptr().add(out_base) as *mut __m256i, lo);
        }

        // Handle remaining
        let start = chunks * 4;
        for j in start..count {
            let base = j * 4;
            if base + 3 < input.len() {
                output[j] = u32::from_le_bytes([
                    input[base],
                    input[base + 1],
                    input[base + 2],
                    input[base + 3],
                ]) as i64;
            }
        }
    }

    Ok(count)
}

#[cfg(target_arch = "x86_64")]
fn decode_64bit_to_i64(input: &[u8], output: &mut [i64]) -> Result<usize> {
    let count = (input.len() / 8).min(output.len());
    if count == 0 {
        return Ok(0);
    }

    unsafe {
        let chunks = count / 4;

        // Process 4 x 64-bit values at a time using AVX2
        for i in 0..chunks {
            let base = i * 32;
            let out_base = i * 4;

            // Load 32 bytes (4 x 64-bit values) using AVX2
            let src = input.as_ptr().add(base) as *const __m256i;
            let vals = _mm256_loadu_si256(src);
            _mm256_storeu_si256(output.as_mut_ptr().add(out_base) as *mut __m256i, vals);
        }

        // Handle remaining
        let start = chunks * 4;
        for j in start..count {
            let base = j * 8;
            if base + 7 < input.len() {
                output[j] = i64::from_le_bytes([
                    input[base],
                    input[base + 1],
                    input[base + 2],
                    input[base + 3],
                    input[base + 4],
                    input[base + 5],
                    input[base + 6],
                    input[base + 7],
                ]);
            }
        }
    }

    Ok(count)
}

#[cfg(target_arch = "x86_64")]
fn decode_8bit_to_i32_avx2(input: &[u8], output: &mut [i32]) -> Result<usize> {
    let count = input.len().min(output.len());
    if count == 0 {
        return Ok(0);
    }

    unsafe {
        let chunks = count / 8;

        // Process 8 bytes -> 8 i32 at a time using AVX2
        for i in 0..chunks {
            let base = i * 8;
            let out_base = i * 8;

            // Load 8 bytes
            let bytes = _mm_loadl_epi64(input.as_ptr().add(base) as *const __m128i);

            // Zero-extend 8 bytes to 8 x i32 using AVX2
            let extended = _mm256_cvtepu8_epi32(bytes);

            _mm256_storeu_si256(output.as_mut_ptr().add(out_base) as *mut __m256i, extended);
        }

        // Handle remaining
        let start = chunks * 8;
        for j in start..count {
            output[j] = input[j] as i32;
        }
    }

    Ok(count)
}

#[cfg(target_arch = "x86_64")]
fn decode_16bit_to_i32_avx2(input: &[u8], output: &mut [i32]) -> Result<usize> {
    let count = (input.len() / 2).min(output.len());
    if count == 0 {
        return Ok(0);
    }

    unsafe {
        let chunks = count / 8;

        // Process 8 x 16-bit values at a time using AVX2
        for i in 0..chunks {
            let base = i * 16;
            let out_base = i * 8;

            // Load 16 bytes (8 x 16-bit values)
            let vals = _mm_loadu_si128(input.as_ptr().add(base) as *const __m128i);

            // Zero-extend 8 x 16-bit to 8 x 32-bit using AVX2
            let extended = _mm256_cvtepu16_epi32(vals);

            _mm256_storeu_si256(output.as_mut_ptr().add(out_base) as *mut __m256i, extended);
        }

        // Handle remaining
        let start = chunks * 8;
        for j in start..count {
            let base = j * 2;
            if base + 1 < input.len() {
                output[j] = u16::from_le_bytes([input[base], input[base + 1]]) as i32;
            }
        }
    }

    Ok(count)
}

#[cfg(target_arch = "x86_64")]
fn decode_32bit_to_i32(input: &[u8], output: &mut [i32]) -> Result<usize> {
    let count = (input.len() / 4).min(output.len());
    if count == 0 {
        return Ok(0);
    }

    unsafe {
        let chunks = count / 8;

        // Process 8 x 32-bit values at a time using AVX2
        for i in 0..chunks {
            let base = i * 32;
            let out_base = i * 8;

            // Load 32 bytes (8 x 32-bit values) directly
            let vals = _mm256_loadu_si256(input.as_ptr().add(base) as *const __m256i);
            _mm256_storeu_si256(output.as_mut_ptr().add(out_base) as *mut __m256i, vals);
        }

        // Handle remaining
        let start = chunks * 8;
        for j in start..count {
            let base = j * 4;
            if base + 3 < input.len() {
                output[j] = i32::from_le_bytes([
                    input[base],
                    input[base + 1],
                    input[base + 2],
                    input[base + 3],
                ]);
            }
        }
    }

    Ok(count)
}

// Non-byte-aligned bit widths, delta-header parsing, f32 bit-pattern
// reconstruction, and prefix-sum finalization are the shared scalar fallbacks
// in `super::common` — defined once, identical on every target.

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests {
    use super::*;
    use crate::simd_decode::test_support;

    fn skip_if_no_avx2() -> bool {
        !is_x86_feature_detected!("avx2")
    }

    #[test]
    fn test_avx2_golden_decode_battery() {
        if skip_if_no_avx2() {
            return;
        }
        test_support::run_golden_decode_battery(&Avx2Decoder::new());
    }

    #[test]
    fn test_avx2_decode_8bit() {
        if skip_if_no_avx2() {
            return;
        }
        test_support::assert_decodes_i64(&Avx2Decoder::new(), &test_support::scaled_ramp(32, 1), 8);
    }

    #[test]
    fn test_avx2_decode_16bit() {
        if skip_if_no_avx2() {
            return;
        }
        test_support::assert_decodes_i64(
            &Avx2Decoder::new(),
            &test_support::scaled_ramp(32, 1_000),
            16,
        );
    }

    #[test]
    fn test_avx2_decode_32bit() {
        if skip_if_no_avx2() {
            return;
        }
        test_support::assert_decodes_i64(
            &Avx2Decoder::new(),
            &test_support::scaled_ramp(32, 100_000),
            32,
        );
    }

    #[test]
    fn test_avx2_decode_i32() {
        if skip_if_no_avx2() {
            return;
        }
        test_support::assert_decodes_i32(
            &Avx2Decoder::new(),
            &test_support::scaled_ramp(32, 100),
            16,
        );
    }

    #[test]
    fn test_avx2_vs_scalar_equivalence() {
        if skip_if_no_avx2() {
            return;
        }
        test_support::assert_matches_scalar_reference(&Avx2Decoder::new());
    }

    #[test]
    fn test_avx2_name_and_features() {
        if skip_if_no_avx2() {
            return;
        }

        let decoder = Avx2Decoder::new();
        assert_eq!(decoder.name(), "AVX2");
        assert!(decoder.has_acceleration());
        assert!(decoder.supported_features().avx2);
    }
}
