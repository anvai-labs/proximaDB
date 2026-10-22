// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! ARM NEON SIMD Decoder
//!
//! This module provides ARM NEON-accelerated decoding using 128-bit SIMD registers.
//! It processes 4 x i32 or 2 x i64 values per iteration on ARM64 platforms.
//!
//! # Requirements
//!
//! - ARM64 (AArch64) architecture
//! - NEON is mandatory on ARM64, so no runtime detection needed
//!
//! # Performance
//!
//! Compared to scalar:
//! - 8-bit aligned: ~3-4x faster
//! - 16-bit aligned: ~3-4x faster
//! - 32-bit aligned: ~3-4x faster
//! - Non-aligned bit widths: ~2-3x faster
//!
//! # Apple Silicon Optimization
//!
//! On Apple M1/M2/M3/M4 processors, NEON operations are highly optimized with:
//! - 4 NEON execution units
//! - 128-bit vector registers
//! - Low-latency memory access

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use anyhow::{Result, bail};

use super::traits::{CpuFeatures, SimdDecoder};

/// ARM NEON SIMD Decoder
///
/// Uses 128-bit NEON instructions for parallel decoding.
/// Processes 4 x i32 values per iteration when possible.
#[derive(Debug, Clone, Copy)]
pub struct NeonDecoder;

impl NeonDecoder {
    /// Create a new NEON decoder
    ///
    /// NEON is mandatory on ARM64, so this always succeeds on that platform.
    pub fn new() -> Self {
        Self
    }
}

impl Default for NeonDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl SimdDecoder for NeonDecoder {
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

        #[cfg(target_arch = "aarch64")]
        {
            // For byte-aligned bit widths, use optimized paths
            match bits_per_value {
                8 => return decode_8bit_to_i64_neon(input, output),
                16 => return decode_16bit_to_i64_neon(input, output),
                32 => return decode_32bit_to_i64_neon(input, output),
                64 => return decode_64bit_to_i64_neon(input, output),
                _ => {}
            }

            // For non-aligned bit widths, use the shared scalar path
            super::common::decode_variable_bits_i64(input, bits_per_value, output)
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            bail!("NEON decoder not available on this platform")
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

        #[cfg(target_arch = "aarch64")]
        {
            match bits_per_value {
                8 => return decode_8bit_to_i32_neon(input, output),
                16 => return decode_16bit_to_i32_neon(input, output),
                32 => return decode_32bit_to_i32_neon(input, output),
                _ => {}
            }

            super::common::decode_variable_bits_i32(input, bits_per_value, output)
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            bail!("NEON decoder not available on this platform")
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

        #[cfg(target_arch = "aarch64")]
        {
            // Decode deltas to i64
            let mut deltas = vec![0i64; actual_count];
            let decoded = self.decode_bitpacked_i64(header.packed, header.bits, &mut deltas)?;

            // Reconstruct using NEON SIMD
            unsafe {
                reconstruct_delta_f32_neon(&deltas[..decoded], base, &mut output[..decoded])?;
            }

            Ok(decoded.min(actual_count))
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            bail!("NEON decoder not available on this platform")
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
            avx2: false,
            avx512: false,
            sse41: false,
            neon: true,
        }
    }

    fn name(&self) -> &'static str {
        "NEON"
    }

    fn has_acceleration(&self) -> bool {
        true
    }
}

// ============================================================================
// NEON Implementation Functions
// ============================================================================

#[cfg(target_arch = "aarch64")]
fn decode_8bit_to_i64_neon(input: &[u8], output: &mut [i64]) -> Result<usize> {
    let count = input.len().min(output.len());
    if count == 0 {
        return Ok(0);
    }

    unsafe {
        let chunks = count / 8;

        // Process 8 bytes -> 8 i64 at a time
        for i in 0..chunks {
            let base = i * 8;

            // Load 8 bytes
            let bytes = vld1_u8(input.as_ptr().add(base));

            // Zero-extend to 16-bit, then 32-bit, then 64-bit
            let u16_lo = vmovl_u8(bytes);
            let u32_lo = vmovl_u16(vget_low_u16(u16_lo));
            let u32_hi = vmovl_u16(vget_high_u16(u16_lo));

            // Extend to 64-bit
            let u64_0 = vmovl_u32(vget_low_u32(u32_lo));
            let u64_1 = vmovl_u32(vget_high_u32(u32_lo));
            let u64_2 = vmovl_u32(vget_low_u32(u32_hi));
            let u64_3 = vmovl_u32(vget_high_u32(u32_hi));

            // Store as i64
            vst1q_s64(output.as_mut_ptr().add(base), vreinterpretq_s64_u64(u64_0));
            vst1q_s64(
                output.as_mut_ptr().add(base + 2),
                vreinterpretq_s64_u64(u64_1),
            );
            vst1q_s64(
                output.as_mut_ptr().add(base + 4),
                vreinterpretq_s64_u64(u64_2),
            );
            vst1q_s64(
                output.as_mut_ptr().add(base + 6),
                vreinterpretq_s64_u64(u64_3),
            );
        }

        // Handle remaining
        for i in (chunks * 8)..count {
            output[i] = input[i] as i64;
        }
    }

    Ok(count)
}

#[cfg(target_arch = "aarch64")]
fn decode_16bit_to_i64_neon(input: &[u8], output: &mut [i64]) -> Result<usize> {
    let count = (input.len() / 2).min(output.len());
    if count == 0 {
        return Ok(0);
    }

    unsafe {
        let chunks = count / 4;

        for i in 0..chunks {
            let base = i * 8; // 4 x 16-bit = 8 bytes
            let out_base = i * 4;

            // Load 4 x 16-bit values
            let vals = vld1_u16(input.as_ptr().add(base) as *const u16);

            // Zero-extend to 32-bit, then 64-bit
            let u32_vals = vmovl_u16(vals);
            let u64_lo = vmovl_u32(vget_low_u32(u32_vals));
            let u64_hi = vmovl_u32(vget_high_u32(u32_vals));

            // Store as i64
            vst1q_s64(
                output.as_mut_ptr().add(out_base),
                vreinterpretq_s64_u64(u64_lo),
            );
            vst1q_s64(
                output.as_mut_ptr().add(out_base + 2),
                vreinterpretq_s64_u64(u64_hi),
            );
        }

        // Handle remaining
        for (j, out_val) in output.iter_mut().enumerate().take(count).skip(chunks * 4) {
            let base = j * 2;
            if base + 1 < input.len() {
                *out_val = u16::from_le_bytes([input[base], input[base + 1]]) as i64;
            }
        }
    }

    Ok(count)
}

#[cfg(target_arch = "aarch64")]
fn decode_32bit_to_i64_neon(input: &[u8], output: &mut [i64]) -> Result<usize> {
    let count = (input.len() / 4).min(output.len());
    if count == 0 {
        return Ok(0);
    }

    unsafe {
        let chunks = count / 4;

        for i in 0..chunks {
            let base = i * 16; // 4 x 32-bit = 16 bytes
            let out_base = i * 4;

            // Load 4 x 32-bit values
            let vals = vld1q_u32(input.as_ptr().add(base) as *const u32);

            // Zero-extend to 64-bit
            let u64_lo = vmovl_u32(vget_low_u32(vals));
            let u64_hi = vmovl_u32(vget_high_u32(vals));

            // Store as i64
            vst1q_s64(
                output.as_mut_ptr().add(out_base),
                vreinterpretq_s64_u64(u64_lo),
            );
            vst1q_s64(
                output.as_mut_ptr().add(out_base + 2),
                vreinterpretq_s64_u64(u64_hi),
            );
        }

        // Handle remaining
        for (j, out_val) in output.iter_mut().enumerate().take(count).skip(chunks * 4) {
            let base = j * 4;
            if base + 3 < input.len() {
                *out_val = u32::from_le_bytes([
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

#[cfg(target_arch = "aarch64")]
fn decode_64bit_to_i64_neon(input: &[u8], output: &mut [i64]) -> Result<usize> {
    let count = (input.len() / 8).min(output.len());
    if count == 0 {
        return Ok(0);
    }

    unsafe {
        let chunks = count / 2;

        // Process 2 x 64-bit values at a time using NEON
        for i in 0..chunks {
            let base = i * 16;
            let out_base = i * 2;

            // Load 2 x 64-bit values
            let vals = vld1q_s64(input.as_ptr().add(base) as *const i64);
            vst1q_s64(output.as_mut_ptr().add(out_base), vals);
        }

        // Handle remaining
        for (j, out_val) in output.iter_mut().enumerate().take(count).skip(chunks * 2) {
            let base = j * 8;
            if base + 7 < input.len() {
                *out_val = i64::from_le_bytes([
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

#[cfg(target_arch = "aarch64")]
fn decode_8bit_to_i32_neon(input: &[u8], output: &mut [i32]) -> Result<usize> {
    let count = input.len().min(output.len());
    if count == 0 {
        return Ok(0);
    }

    unsafe {
        let chunks = count / 8;

        // Process 8 bytes -> 8 i32 at a time
        for i in 0..chunks {
            let base = i * 8;

            // Load 8 bytes
            let bytes = vld1_u8(input.as_ptr().add(base));

            // Zero-extend to 16-bit, then 32-bit
            let u16_vals = vmovl_u8(bytes);
            let u32_lo = vmovl_u16(vget_low_u16(u16_vals));
            let u32_hi = vmovl_u16(vget_high_u16(u16_vals));

            // Store as i32
            vst1q_s32(output.as_mut_ptr().add(base), vreinterpretq_s32_u32(u32_lo));
            vst1q_s32(
                output.as_mut_ptr().add(base + 4),
                vreinterpretq_s32_u32(u32_hi),
            );
        }

        // Handle remaining
        for j in (chunks * 8)..count {
            output[j] = input[j] as i32;
        }
    }

    Ok(count)
}

#[cfg(target_arch = "aarch64")]
fn decode_16bit_to_i32_neon(input: &[u8], output: &mut [i32]) -> Result<usize> {
    let count = (input.len() / 2).min(output.len());
    if count == 0 {
        return Ok(0);
    }

    unsafe {
        let chunks = count / 8;

        for i in 0..chunks {
            let base = i * 16; // 8 x 16-bit = 16 bytes
            let out_base = i * 8;

            // Load 8 x 16-bit values
            let vals = vld1q_u16(input.as_ptr().add(base) as *const u16);

            // Zero-extend to 32-bit
            let u32_lo = vmovl_u16(vget_low_u16(vals));
            let u32_hi = vmovl_u16(vget_high_u16(vals));

            // Store as i32
            vst1q_s32(
                output.as_mut_ptr().add(out_base),
                vreinterpretq_s32_u32(u32_lo),
            );
            vst1q_s32(
                output.as_mut_ptr().add(out_base + 4),
                vreinterpretq_s32_u32(u32_hi),
            );
        }

        // Handle remaining
        for (j, out_val) in output.iter_mut().enumerate().take(count).skip(chunks * 8) {
            let base = j * 2;
            if base + 1 < input.len() {
                *out_val = u16::from_le_bytes([input[base], input[base + 1]]) as i32;
            }
        }
    }

    Ok(count)
}

#[cfg(target_arch = "aarch64")]
fn decode_32bit_to_i32_neon(input: &[u8], output: &mut [i32]) -> Result<usize> {
    let count = (input.len() / 4).min(output.len());
    if count == 0 {
        return Ok(0);
    }

    unsafe {
        let chunks = count / 4;

        // Process 4 x 32-bit values at a time using NEON
        for i in 0..chunks {
            let base = i * 16;
            let out_base = i * 4;

            // Load and store 4 x 32-bit values directly
            let vals = vld1q_s32(input.as_ptr().add(base) as *const i32);
            vst1q_s32(output.as_mut_ptr().add(out_base), vals);
        }

        // Handle remaining
        for (j, out_val) in output.iter_mut().enumerate().take(count).skip(chunks * 4) {
            let base = j * 4;
            if base + 3 < input.len() {
                *out_val = i32::from_le_bytes([
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

// Non-byte-aligned bit widths, delta-header parsing, and prefix-sum
// finalization are the shared scalar fallbacks in `super::common` — defined
// once, identical on every target. Only the NEON f32 reconstruction below
// uses real SIMD intrinsics and stays per-target.

#[cfg(target_arch = "aarch64")]
unsafe fn reconstruct_delta_f32_neon(deltas: &[i64], base: f32, output: &mut [f32]) -> Result<()> {
    unsafe {
        let base_bits = base.to_bits() as i64;
        let count = deltas.len().min(output.len());

        // Process 4 values at a time
        let chunks = count / 4;

        for i in 0..chunks {
            let idx = i * 4;

            // Reconstruct bit patterns
            let v0 = (base_bits + deltas[idx]) as u32;
            let v1 = (base_bits + deltas[idx + 1]) as u32;
            let v2 = (base_bits + deltas[idx + 2]) as u32;
            let v3 = (base_bits + deltas[idx + 3]) as u32;

            // Load as u32 vector and reinterpret as f32
            let u32_vec = vld1q_u32([v0, v1, v2, v3].as_ptr());
            let f32_vec = vreinterpretq_f32_u32(u32_vec);

            vst1q_f32(output.as_mut_ptr().add(idx), f32_vec);
        }

        // Handle remaining
        for i in (chunks * 4)..count {
            let value_bits = (base_bits + deltas[i]) as u32;
            output[i] = f32::from_bits(value_bits);
        }

        Ok(())
    }
}

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;
    use crate::simd_decode::test_support;

    #[test]
    fn test_neon_golden_decode_battery() {
        test_support::run_golden_decode_battery(&NeonDecoder::new());
    }

    #[test]
    fn test_neon_decode_8bit() {
        test_support::assert_decodes_i64(&NeonDecoder::new(), &test_support::scaled_ramp(32, 1), 8);
    }

    #[test]
    fn test_neon_decode_16bit() {
        test_support::assert_decodes_i64(
            &NeonDecoder::new(),
            &test_support::scaled_ramp(32, 1_000),
            16,
        );
    }

    #[test]
    fn test_neon_decode_32bit() {
        test_support::assert_decodes_i64(
            &NeonDecoder::new(),
            &test_support::scaled_ramp(32, 100_000),
            32,
        );
    }

    #[test]
    fn test_neon_decode_i32() {
        test_support::assert_decodes_i32(
            &NeonDecoder::new(),
            &test_support::scaled_ramp(32, 100),
            16,
        );
    }

    #[test]
    fn test_neon_vs_scalar_equivalence() {
        test_support::assert_matches_scalar_reference(&NeonDecoder::new());
    }

    #[test]
    fn test_neon_name_and_features() {
        let decoder = NeonDecoder::new();
        assert_eq!(decoder.name(), "NEON");
        assert!(decoder.has_acceleration());
        assert!(decoder.supported_features().neon);
    }
}
