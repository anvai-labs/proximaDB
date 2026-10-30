// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! OpenCL Kernels - Cross-platform GPU acceleration
//!
//! This module provides OpenCL implementations for encoding/decoding.
//! OpenCL is a cross-platform standard that works on NVIDIA, AMD, Intel,
//! and Apple GPUs.
//!
//! ## Architecture
//!
//! - **OpenCL C**: Kernels written in OpenCL C
//! - **Platform Detection**: Automatic device selection
//! - **Work-group Size**: Configurable (default 256)
//! - **Local Memory**: 16-64 KB per work-group
//!
//! ## Platform Support
//!
//! - NVIDIA: Via CUDA compatibility layer
//! - AMD: Native OpenCL implementation
//! - Intel: Integrated GPU support
//! - Apple: Legacy support (pre-M1)
//!
//! Real OpenCL kernel compilation is DEFERRED; every function below runs the
//! shared scalar CPU fallback from [`super::cpu_fallback`] (redundancy pass 3:
//! the fallback loops used to be pasted per scheme).

use anyhow::Result;
use tracing::{debug, trace};

use super::cpu_fallback;
use super::utils::GpuBatchConfig;
use proximadb_hardware_caps::HardwareBackend;

/// OpenCL context wrapper
pub struct OpenCLContext {
    config: GpuBatchConfig,
}

impl OpenCLContext {
    /// Create new OpenCL context
    pub fn new(total_vectors: usize, dimension: usize) -> Result<Self> {
        let config =
            GpuBatchConfig::for_backend(&HardwareBackend::OpenCL, total_vectors, dimension);

        debug!(
            "🌐 [OpenCL] Initializing context: {} vectors, dim={}",
            total_vectors, dimension
        );

        Ok(Self { config })
    }

    /// Get batch configuration
    pub fn config(&self) -> &GpuBatchConfig {
        &self.config
    }
}

// ============================================================================
// DELTA ENCODING/DECODING
// ============================================================================

/// OpenCL Delta encoding for f32
///
/// # OpenCL Kernel
/// ```opencl
/// __kernel void delta_encode_f32(
///     __global const float* input,
///     __global long* output,
///     const float base,
///     const int n
/// ) {
///     int gid = get_global_id(0);
///     if (gid < n) {
///         output[gid] = (long)(input[gid] - base);
///     }
/// }
/// ```text
pub fn opencl_delta_encode_f32(values: &[f32], base: f32) -> Result<Vec<i64>> {
    trace!(
        "🔧 [OpenCL] Delta encode: {} values, base={}",
        values.len(),
        base
    );

    // Deferred: Real OpenCL implementation using clCreateBuffer/clEnqueueNDRangeKernel
    let deltas = cpu_fallback::delta_encode_f32(values, base);

    debug!(
        "✅ [OpenCL] Delta encoded {} values → {} deltas",
        values.len(),
        deltas.len()
    );
    Ok(deltas)
}

/// OpenCL Delta decoding for f32
///
/// # OpenCL Kernel
/// ```opencl
/// __kernel void delta_decode_f32(
///     __global const long* input,
///     __global float* output,
///     const float base,
///     const int n
/// ) {
///     int gid = get_global_id(0);
///     if (gid < n) {
///         output[gid] = (float)input[gid] + base;
///     }
/// }
/// ```text
pub fn opencl_delta_decode_f32(deltas: &[i64], base: f32) -> Result<Vec<f32>> {
    trace!(
        "🔧 [OpenCL] Delta decode: {} deltas, base={}",
        deltas.len(),
        base
    );

    // Deferred: Real OpenCL implementation
    let values = cpu_fallback::delta_decode_f32(deltas, base);

    debug!(
        "✅ [OpenCL] Delta decoded {} deltas → {} values",
        deltas.len(),
        values.len()
    );
    Ok(values)
}

// ============================================================================
// BITPACKED ENCODING/DECODING
// ============================================================================

/// OpenCL BitPacked encoding for f32
///
/// # OpenCL Kernel
/// ```opencl
/// __kernel void bitpack_encode_f32(
///     __global const float* input,
///     __global uchar* output,
///     const uint bits,
///     const int n
/// ) {
///     int gid = get_global_id(0);
///     if (gid < n) {
///         uint val = as_uint(input[gid]);
///         uint mask = (1u << bits) - 1u;
///         uint packed = val & mask;
///
///         size_t bit_offset = gid * bits;
///         size_t byte_offset = bit_offset / 8;
///         size_t bit_in_byte = bit_offset % 8;
///
///         atomic_or(&output[byte_offset], packed << bit_in_byte);
///     }
/// }
/// ```text
pub fn opencl_bitpack_encode_f32(values: &[f32], bits: u8) -> Result<Vec<u8>> {
    trace!(
        "🔧 [OpenCL] BitPacked encode: {} values, {}b/val",
        values.len(),
        bits
    );

    // Deferred: Real OpenCL implementation with atomic operations
    let result = cpu_fallback::bitpack_encode_f32(values, bits)?;

    debug!(
        "✅ [OpenCL] BitPacked encoded {} values → {} bytes",
        values.len(),
        result.len()
    );
    Ok(result)
}

/// OpenCL BitPacked decoding for f32
pub fn opencl_bitpack_decode_f32(packed: &[u8], bits: u8, count: usize) -> Result<Vec<f32>> {
    trace!(
        "🔧 [OpenCL] BitPacked decode: {} bytes, {}b/val, count={}",
        packed.len(),
        bits,
        count
    );

    // Deferred: Real OpenCL implementation
    let result = cpu_fallback::bitpack_decode_f32(packed, bits, count)?;

    debug!(
        "✅ [OpenCL] BitPacked decoded {} bytes → {} values",
        packed.len(),
        result.len()
    );
    Ok(result)
}

// ============================================================================
// FRAME OF REFERENCE ENCODING/DECODING
// ============================================================================

/// OpenCL FrameOfReference encoding
pub fn opencl_frame_of_reference_encode_f32(
    values: &[f32],
    reference: i64,
    bits: u8,
) -> Result<Vec<u8>> {
    trace!(
        "🔧 [OpenCL] FrameOfReference encode: {} values, ref={}, {}b/val",
        values.len(),
        reference,
        bits
    );

    // Deferred: Real OpenCL implementation
    let result = cpu_fallback::frame_of_reference_encode_f32(values, reference, bits)?;

    debug!(
        "✅ [OpenCL] FrameOfReference encoded {} values → {} bytes",
        values.len(),
        result.len()
    );
    Ok(result)
}

/// OpenCL FrameOfReference decoding
pub fn opencl_frame_of_reference_decode_f32(
    packed: &[u8],
    reference: i64,
    bits: u8,
    count: usize,
) -> Result<Vec<f32>> {
    trace!(
        "🔧 [OpenCL] FrameOfReference decode: {} bytes, ref={}, {}b/val, count={}",
        packed.len(),
        reference,
        bits,
        count
    );

    // Deferred: Real OpenCL implementation
    let result = cpu_fallback::frame_of_reference_decode_f32(packed, reference, bits, count)?;

    debug!(
        "✅ [OpenCL] FrameOfReference decoded {} bytes → {} values",
        packed.len(),
        result.len()
    );
    Ok(result)
}

// ============================================================================
// ZIGZAG ENCODING/DECODING
// ============================================================================

/// OpenCL Zigzag encoding
///
/// # OpenCL Kernel
/// ```opencl
/// __kernel void zigzag_encode_f32(
///     __global const float* input,
///     __global int* output,
///     const int n
/// ) {
///     int gid = get_global_id(0);
///     if (gid < n) {
///         int n = as_int(input[gid]);
///         output[gid] = (n << 1) ^ (n >> 31);
///     }
/// }
/// ```text
pub fn opencl_zigzag_encode_f32(values: &[f32], bits: u8) -> Result<Vec<u8>> {
    trace!(
        "🔧 [OpenCL] Zigzag encode: {} values, {}b/val",
        values.len(),
        bits
    );

    // Deferred: Real OpenCL implementation
    let result = cpu_fallback::zigzag_encode_f32(values, bits)?;

    debug!(
        "✅ [OpenCL] Zigzag encoded {} values → {} bytes",
        values.len(),
        result.len()
    );
    Ok(result)
}

/// OpenCL Zigzag decoding
pub fn opencl_zigzag_decode_f32(packed: &[u8], bits: u8, count: usize) -> Result<Vec<f32>> {
    trace!(
        "🔧 [OpenCL] Zigzag decode: {} bytes, {}b/val, count={}",
        packed.len(),
        bits,
        count
    );

    // Deferred: Real OpenCL implementation
    let result = cpu_fallback::zigzag_decode_f32(packed, bits, count)?;

    debug!(
        "✅ [OpenCL] Zigzag decoded {} bytes → {} values",
        packed.len(),
        result.len()
    );
    Ok(result)
}

// ============================================================================
// PFOR-DELTA ENCODING/DECODING
// ============================================================================

/// OpenCL PForDelta encoding (stub - complex kernel)
pub fn opencl_pfor_delta_encode_f32(
    values: &[f32],
    majority_bits: u8,
    base: i64,
) -> Result<Vec<u8>> {
    trace!(
        "🔧 [OpenCL] PForDelta encode: {} values, {}b majority, base={}",
        values.len(),
        majority_bits,
        base
    );

    // Deferred: Real OpenCL implementation
    anyhow::bail!("OpenCL PForDelta encoding not yet implemented - use SIMD fallback")
}

/// OpenCL PForDelta decoding (stub - complex kernel)
pub fn opencl_pfor_delta_decode_f32(
    data: &[u8],
    majority_bits: u8,
    base: i64,
    count: usize,
) -> Result<Vec<f32>> {
    trace!(
        "🔧 [OpenCL] PForDelta decode: {} bytes, {}b majority, base={}, count={}",
        data.len(),
        majority_bits,
        base,
        count
    );

    // Deferred: Real OpenCL implementation
    anyhow::bail!("OpenCL PForDelta decoding not yet implemented - use SIMD fallback")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opencl_context_creation() {
        let ctx = OpenCLContext::new(10000, 128);
        assert!(ctx.is_ok());

        let ctx = ctx.unwrap();
        assert!(ctx.config().batch_size >= 10000);
        assert_eq!(ctx.config().threads_per_block, 256);
    }

    #[test]
    fn test_opencl_delta_encode() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let result = opencl_delta_encode_f32(&values, 0.0);
        assert!(result.is_ok());

        let deltas = result.unwrap();
        assert_eq!(deltas.len(), 4);
    }

    #[test]
    fn test_opencl_delta_roundtrip() {
        let values = vec![10.0f32, 20.0, 30.0, 40.0];
        let base = 5.0;

        let deltas = opencl_delta_encode_f32(&values, base).unwrap();
        let decoded = opencl_delta_decode_f32(&deltas, base).unwrap();

        for (original, recovered) in values.iter().zip(decoded.iter()) {
            assert!((original - recovered).abs() < 0.01);
        }
    }

    #[test]
    fn test_opencl_bitpack_roundtrip_bits_20() {
        // Regression for the shared-fallback truncation fix: 20-bit words at
        // odd byte alignments must round-trip through the OpenCL fallback.
        let words = [0x000F_FFFFu32, 0x000A_5A5A, 0x0000_0001, 0x0008_0001];
        let values: Vec<f32> = words.iter().map(|&w| f32::from_bits(w)).collect();
        let mask = (1u32 << 20) - 1;

        let packed = opencl_bitpack_encode_f32(&values, 20).unwrap();
        let decoded = opencl_bitpack_decode_f32(&packed, 20, values.len()).unwrap();

        for (i, value) in decoded.iter().enumerate() {
            assert_eq!(value.to_bits(), words[i] & mask, "word {}", i);
        }
    }
}
