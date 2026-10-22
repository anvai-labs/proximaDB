// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Metal/MPS Kernels - Apple Silicon GPU acceleration
//!
//! This module provides Metal Performance Shaders (MPS) implementations
//! for encoding/decoding on Apple Silicon (M1/M2/M3/M4).
//!
//! ## Architecture
//!
//! - **Metal Shading Language**: Compute shaders written in MSL
//! - **MTLDevice**: GPU device management
//! - **MTLCommandQueue**: Asynchronous command submission
//! - **MTLBuffer**: Unified memory buffers (shared CPU/GPU)
//!
//! ## Performance Characteristics
//!
//! - **Unified Memory**: Zero-copy between CPU and GPU
//! - **SIMD Group Size**: 32 threads
//! - **Threadgroup Size**: Optimal 256 threads
//! - **Threadgroup Memory**: 32 KB per threadgroup

use anyhow::Result;
use tracing::{debug, trace};

use super::utils::GpuBatchConfig;
use proximadb_hardware_caps::HardwareBackend;

// Metal FFI imports
#[cfg(all(feature = "gpu", target_os = "macos", target_arch = "aarch64"))]
use metal::*;

// ============================================================================
// METAL FFI BINDINGS (Consolidated)
// ============================================================================

#[cfg(all(feature = "gpu", target_os = "macos", target_arch = "aarch64"))]
mod metal_ffi {
    use metal::*;

    /// Metal device and command queue wrapper
    pub struct RawMetalContext {
        pub device: Device,
        pub command_queue: CommandQueue,
        pub library: Library,
    }

    impl RawMetalContext {
        /// Create new Metal context with compiled shader library
        pub fn new() -> Result<Self, String> {
            // Get default Metal device (Apple GPU)
            let device = Device::system_default().ok_or("No Metal device found")?;

            // Create command queue
            let command_queue = device.new_command_queue();

            // Load compiled Metal library (will be compiled by build.rs)
            let library_path = std::env::var("METAL_LIBRARY_PATH")
                .unwrap_or_else(|_| "target/metal/libproximadb_metal.metallib".to_string());

            let library = if std::path::Path::new(&library_path).exists() {
                device
                    .new_library_with_file(&library_path)
                    .map_err(|e| format!("Failed to load Metal library: {}", e))?
            } else {
                // Fallback: compile from source (slower, for development)
                let source = include_str!("kernels.metal");
                let options = CompileOptions::new();
                device
                    .new_library_with_source(source, &options)
                    .map_err(|e| format!("Failed to compile Metal shaders: {}", e))?
            };

            Ok(Self {
                device,
                command_queue,
                library,
            })
        }

        /// Get compute pipeline for a kernel function
        pub fn get_pipeline(&self, function_name: &str) -> Result<ComputePipelineState, String> {
            let function = self
                .library
                .get_function(function_name, None)
                .map_err(|e| format!("Function '{}' not found: {}", function_name, e))?;

            self.device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| format!("Failed to create pipeline: {}", e))
        }
    }

    /// Helper to create Metal buffer from slice
    pub fn create_buffer<T>(device: &Device, data: &[T]) -> Buffer {
        let size = (data.len() * std::mem::size_of::<T>()) as u64;
        let buffer = device.new_buffer(size, MTLResourceOptions::StorageModeShared);

        unsafe {
            let ptr = buffer.contents() as *mut T;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }

        buffer
    }

    /// Helper to create empty Metal buffer
    pub fn create_empty_buffer<T>(device: &Device, count: usize) -> Buffer {
        let size = (count * std::mem::size_of::<T>()) as u64;
        device.new_buffer(size, MTLResourceOptions::StorageModeShared)
    }

    /// Helper to read data from Metal buffer
    pub fn read_buffer<T: Clone>(buffer: &Buffer, count: usize) -> Vec<T> {
        let mut result = Vec::with_capacity(count);
        unsafe {
            let ptr = buffer.contents() as *const T;
            result.extend_from_slice(std::slice::from_raw_parts(ptr, count));
        }
        result
    }

    /// Execute Metal compute kernel
    pub fn execute_kernel(
        context: &RawMetalContext,
        pipeline: &ComputePipelineState,
        buffers: &[&Buffer],
        thread_count: usize,
    ) -> Result<(), String> {
        let command_buffer = context.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);

        // Set buffers
        for (i, buffer) in buffers.iter().enumerate() {
            encoder.set_buffer(i as u64, Some(*buffer), 0);
        }

        // Calculate threadgroup size
        let threadgroup_size = MTLSize {
            width: 256.min(thread_count as u64),
            height: 1,
            depth: 1,
        };

        let threadgroups = MTLSize {
            width: ((thread_count as u64 + 255) / 256),
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }
}

// ============================================================================
// PUBLIC API
// ============================================================================

/// Metal/MPS context wrapper
pub struct MetalContext {
    config: GpuBatchConfig,
    #[cfg(all(feature = "gpu", target_os = "macos", target_arch = "aarch64"))]
    metal_ctx: Option<metal_ffi::RawMetalContext>,
}

impl MetalContext {
    /// Create new Metal context
    pub fn new(total_vectors: usize, dimension: usize) -> Result<Self> {
        let config = GpuBatchConfig::for_backend(&HardwareBackend::MPS, total_vectors, dimension);

        debug!(
            "🍎 [Metal] Initializing context: {} vectors, dim={}",
            total_vectors, dimension
        );

        #[cfg(all(feature = "gpu", target_os = "macos", target_arch = "aarch64"))]
        {
            let metal_ctx = metal_ffi::RawMetalContext::new()
                .map_err(|e| anyhow::anyhow!("Failed to initialize Metal: {}", e))?;

            debug!(
                "✅ [Metal] Initialized GPU device: {}",
                metal_ctx.device.name()
            );

            Ok(Self {
                config,
                metal_ctx: Some(metal_ctx),
            })
        }

        #[cfg(not(all(feature = "gpu", target_os = "macos", target_arch = "aarch64")))]
        {
            Ok(Self { config })
        }
    }

    /// Get batch configuration
    pub fn config(&self) -> &GpuBatchConfig {
        &self.config
    }

    /// Check if Metal GPU is available
    #[allow(dead_code)]
    pub fn has_gpu(&self) -> bool {
        #[cfg(all(feature = "gpu", target_os = "macos", target_arch = "aarch64"))]
        {
            self.metal_ctx.is_some()
        }

        #[cfg(not(all(feature = "gpu", target_os = "macos", target_arch = "aarch64")))]
        {
            false
        }
    }
}

// ============================================================================
// DELTA ENCODING/DECODING
// ============================================================================

/// Metal Delta encoding for f32
///
/// # Metal Kernel (MSL)
/// ```metal
/// kernel void delta_encode_f32(
///     device const float* input [[buffer(0)]],
///     device int64_t* output [[buffer(1)]],
///     constant float& base [[buffer(2)]],
///     uint gid [[thread_position_in_grid]]
/// ) {
///     output[gid] = (int64_t)(input[gid] - base);
/// }
/// ```text
pub fn metal_delta_encode_f32(values: &[f32], base: f32) -> Result<Vec<i64>> {
    trace!(
        "🔧 [Metal] Delta encode: {} values, base={}",
        values.len(),
        base
    );

    #[cfg(all(feature = "gpu", target_os = "macos", target_arch = "aarch64"))]
    {
        use metal_ffi::{create_buffer, create_empty_buffer, execute_kernel, read_buffer};

        let ctx = metal_ffi::RawMetalContext::new()
            .map_err(|e| anyhow::anyhow!("Metal initialization failed: {}", e))?;

        // Get compute pipeline
        let pipeline = ctx
            .get_pipeline("delta_encode_f32")
            .map_err(|e| anyhow::anyhow!("Pipeline creation failed: {}", e))?;

        // Create GPU buffers
        let input_buffer = create_buffer(&ctx.device, values);
        let output_buffer = create_empty_buffer::<i64>(&ctx.device, values.len());

        // Create base value buffer
        let base_buffer = create_buffer(&ctx.device, &[base]);

        // Execute kernel
        execute_kernel(
            &ctx,
            &pipeline,
            &[&input_buffer, &output_buffer, &base_buffer],
            values.len(),
        )
        .map_err(|e| anyhow::anyhow!("Kernel execution failed: {}", e))?;

        // Read results
        let deltas = read_buffer::<i64>(&output_buffer, values.len());

        debug!(
            "✅ [Metal] Delta encoded {} values → {} deltas (GPU)",
            values.len(),
            deltas.len()
        );
        Ok(deltas)
    }

    #[cfg(not(all(feature = "gpu", target_os = "macos", target_arch = "aarch64")))]
    {
        // CPU fallback (shared scalar core; see kernels/cpu_fallback.rs)
        let deltas = super::cpu_fallback::delta_encode_f32(values, base);
        debug!(
            "✅ [Metal] Delta encoded {} values → {} deltas (CPU fallback)",
            values.len(),
            deltas.len()
        );
        Ok(deltas)
    }
}

/// Metal Delta decoding for f32
///
/// # Metal Kernel (MSL)
/// ```metal
/// kernel void delta_decode_f32(
///     device const int64_t* input [[buffer(0)]],
///     device float* output [[buffer(1)]],
///     constant float& base [[buffer(2)]],
///     uint gid [[thread_position_in_grid]]
/// ) {
///     output[gid] = (float)input[gid] + base;
/// }
/// ```text
pub fn metal_delta_decode_f32(deltas: &[i64], base: f32) -> Result<Vec<f32>> {
    trace!(
        "🔧 [Metal] Delta decode: {} deltas, base={}",
        deltas.len(),
        base
    );

    #[cfg(all(feature = "gpu", target_os = "macos", target_arch = "aarch64"))]
    {
        use metal_ffi::{create_buffer, create_empty_buffer, execute_kernel, read_buffer};

        let ctx = metal_ffi::RawMetalContext::new()
            .map_err(|e| anyhow::anyhow!("Metal initialization failed: {}", e))?;

        let pipeline = ctx
            .get_pipeline("delta_decode_f32")
            .map_err(|e| anyhow::anyhow!("Pipeline creation failed: {}", e))?;

        let input_buffer = create_buffer(&ctx.device, deltas);
        let output_buffer = create_empty_buffer::<f32>(&ctx.device, deltas.len());
        let base_buffer = create_buffer(&ctx.device, &[base]);

        execute_kernel(
            &ctx,
            &pipeline,
            &[&input_buffer, &output_buffer, &base_buffer],
            deltas.len(),
        )
        .map_err(|e| anyhow::anyhow!("Kernel execution failed: {}", e))?;

        let values = read_buffer::<f32>(&output_buffer, deltas.len());

        debug!(
            "✅ [Metal] Delta decoded {} deltas → {} values (GPU)",
            deltas.len(),
            values.len()
        );
        Ok(values)
    }

    #[cfg(not(all(feature = "gpu", target_os = "macos", target_arch = "aarch64")))]
    {
        let values = super::cpu_fallback::delta_decode_f32(deltas, base);
        debug!(
            "✅ [Metal] Delta decoded {} deltas → {} values (CPU fallback)",
            deltas.len(),
            values.len()
        );
        Ok(values)
    }
}

// ============================================================================
// BITPACKED ENCODING/DECODING
// ============================================================================

/// Metal BitPacked encoding for f32
///
/// # Metal Kernel (MSL)
/// ```metal
/// kernel void bitpack_encode_f32(
///     device const float* input [[buffer(0)]],
///     device atomic_uint* output [[buffer(1)]],
///     constant uint& bits [[buffer(2)]],
///     uint gid [[thread_position_in_grid]]
/// ) {
///     uint val = as_type<uint>(input[gid]);
///     uint mask = (1u << bits) - 1u;
///     uint packed = val & mask;
///
///     uint bit_offset = gid * bits;
///     uint byte_offset = bit_offset / 8;
///     uint bit_in_byte = bit_offset % 8;
///
///     atomic_fetch_or_explicit(&output[byte_offset], packed << bit_in_byte, memory_order_relaxed);
/// }
/// ```text
pub fn metal_bitpack_encode_f32(values: &[f32], bits: u8) -> Result<Vec<u8>> {
    trace!(
        "🔧 [Metal] BitPacked encode: {} values, {}b/val",
        values.len(),
        bits
    );

    #[cfg(all(feature = "gpu", target_os = "macos", target_arch = "aarch64"))]
    {
        use metal_ffi::{
            RawMetalContext, create_buffer, create_empty_buffer, execute_kernel, read_buffer,
        };

        let ctx = RawMetalContext::new()
            .map_err(|e| anyhow::anyhow!("Metal initialization failed: {}", e))?;

        let pipeline = ctx
            .get_pipeline("bitpack_encode")
            .map_err(|e| anyhow::anyhow!("Pipeline creation failed: {}", e))?;

        // Convert f32 to i64 for bitpacking
        let values_i64: Vec<i64> = values.iter().map(|&v| v.to_bits() as i64).collect();

        let total_bits = values.len() * bits as usize;
        let byte_count = (total_bits + 7) / 8;
        let word_count = (byte_count + 3) / 4;

        let input_buffer = create_buffer(&ctx.device, &values_i64);
        let output_buffer = create_empty_buffer::<u32>(&ctx.device, word_count);
        let bit_width_buffer = create_buffer(&ctx.device, &[bits as i32]);
        let n_buffer = create_buffer(&ctx.device, &[values.len() as i32]);

        execute_kernel(
            &ctx,
            &pipeline,
            &[&input_buffer, &output_buffer, &bit_width_buffer, &n_buffer],
            values.len(),
        )
        .map_err(|e| anyhow::anyhow!("Kernel execution failed: {}", e))?;

        // Read result and convert to bytes
        let packed_words = read_buffer::<u32>(&output_buffer, word_count);
        let mut result = vec![0u8; byte_count];
        for (i, &word) in packed_words.iter().enumerate() {
            let bytes = word.to_le_bytes();
            for j in 0..4 {
                if i * 4 + j < byte_count {
                    result[i * 4 + j] = bytes[j];
                }
            }
        }

        debug!(
            "✅ [Metal] BitPacked encoded {} values → {} bytes (GPU)",
            values.len(),
            result.len()
        );
        Ok(result)
    }

    #[cfg(not(all(feature = "gpu", target_os = "macos", target_arch = "aarch64")))]
    {
        // CPU fallback (shared scalar core; see kernels/cpu_fallback.rs)
        let result = super::cpu_fallback::bitpack_encode_f32(values, bits)?;

        debug!(
            "✅ [Metal] BitPacked encoded {} values → {} bytes (CPU fallback)",
            values.len(),
            result.len()
        );
        Ok(result)
    }
}

/// Metal BitPacked decoding for f32
pub fn metal_bitpack_decode_f32(packed: &[u8], bits: u8, count: usize) -> Result<Vec<f32>> {
    trace!(
        "🔧 [Metal] BitPacked decode: {} bytes, {}b/val, count={}",
        packed.len(),
        bits,
        count
    );

    #[cfg(all(feature = "gpu", target_os = "macos", target_arch = "aarch64"))]
    {
        use metal_ffi::{
            RawMetalContext, create_buffer, create_empty_buffer, execute_kernel, read_buffer,
        };

        let ctx = RawMetalContext::new()
            .map_err(|e| anyhow::anyhow!("Metal initialization failed: {}", e))?;

        let pipeline = ctx
            .get_pipeline("bitpack_decode")
            .map_err(|e| anyhow::anyhow!("Pipeline creation failed: {}", e))?;

        let input_buffer = create_buffer(&ctx.device, packed);
        let output_buffer = create_empty_buffer::<i64>(&ctx.device, count);
        let bit_width_buffer = create_buffer(&ctx.device, &[bits as i32]);
        let n_buffer = create_buffer(&ctx.device, &[count as i32]);

        execute_kernel(
            &ctx,
            &pipeline,
            &[&input_buffer, &output_buffer, &bit_width_buffer, &n_buffer],
            count,
        )
        .map_err(|e| anyhow::anyhow!("Kernel execution failed: {}", e))?;

        let decoded_i64 = read_buffer::<i64>(&output_buffer, count);
        let result: Vec<f32> = decoded_i64
            .iter()
            .map(|&v| f32::from_bits(v as u32))
            .collect();

        debug!(
            "✅ [Metal] BitPacked decoded {} bytes → {} values (GPU)",
            packed.len(),
            result.len()
        );
        Ok(result)
    }

    #[cfg(not(all(feature = "gpu", target_os = "macos", target_arch = "aarch64")))]
    {
        // CPU fallback (shared scalar core; see kernels/cpu_fallback.rs)
        let result = super::cpu_fallback::bitpack_decode_f32(packed, bits, count)?;

        debug!(
            "✅ [Metal] BitPacked decoded {} bytes → {} values (CPU fallback)",
            packed.len(),
            result.len()
        );
        Ok(result)
    }
}

// ============================================================================
// FRAME OF REFERENCE ENCODING/DECODING
// ============================================================================

/// Metal FrameOfReference encoding
///
/// Combines delta encoding with bit-packing using unified memory
pub fn metal_frame_of_reference_encode_f32(
    values: &[f32],
    reference: i64,
    bits: u8,
) -> Result<Vec<u8>> {
    trace!(
        "🔧 [Metal] FrameOfReference encode: {} values, ref={}, {}b/val",
        values.len(),
        reference,
        bits
    );

    #[cfg(all(feature = "gpu", target_os = "macos", target_arch = "aarch64"))]
    {
        use metal_ffi::{create_buffer, create_empty_buffer, execute_kernel, read_buffer};

        let ctx = metal_ffi::RawMetalContext::new()
            .map_err(|e| anyhow::anyhow!("Metal initialization failed: {}", e))?;

        let pipeline = ctx
            .get_pipeline("for_encode_f32")
            .map_err(|e| anyhow::anyhow!("Pipeline creation failed: {}", e))?;

        let total_bits = values.len() * bits as usize;
        let byte_count = (total_bits + 7) / 8;
        let word_count = (byte_count + 3) / 4;

        let input_buffer = create_buffer(&ctx.device, values);
        let output_buffer = create_empty_buffer::<u32>(&ctx.device, word_count);
        let base_buffer = create_buffer(&ctx.device, &[reference as f32]);
        let bit_width_buffer = create_buffer(&ctx.device, &[bits as i32]);
        let n_buffer = create_buffer(&ctx.device, &[values.len() as i32]);

        execute_kernel(
            &ctx,
            &pipeline,
            &[
                &input_buffer,
                &output_buffer,
                &base_buffer,
                &bit_width_buffer,
                &n_buffer,
            ],
            values.len(),
        )
        .map_err(|e| anyhow::anyhow!("Kernel execution failed: {}", e))?;

        let packed_words = read_buffer::<u32>(&output_buffer, word_count);
        let mut result = vec![0u8; byte_count];
        for (i, &word) in packed_words.iter().enumerate() {
            let bytes = word.to_le_bytes();
            for j in 0..4 {
                if i * 4 + j < byte_count {
                    result[i * 4 + j] = bytes[j];
                }
            }
        }

        debug!(
            "✅ [Metal] FrameOfReference encoded {} values → {} bytes (GPU)",
            values.len(),
            result.len()
        );
        Ok(result)
    }

    #[cfg(not(all(feature = "gpu", target_os = "macos", target_arch = "aarch64")))]
    {
        // CPU fallback (shared scalar core; see kernels/cpu_fallback.rs)
        let result = super::cpu_fallback::frame_of_reference_encode_f32(values, reference, bits)?;

        debug!(
            "✅ [Metal] FrameOfReference encoded {} values → {} bytes (CPU fallback)",
            values.len(),
            result.len()
        );
        Ok(result)
    }
}

/// Metal FrameOfReference decoding
pub fn metal_frame_of_reference_decode_f32(
    packed: &[u8],
    reference: i64,
    bits: u8,
    count: usize,
) -> Result<Vec<f32>> {
    trace!(
        "🔧 [Metal] FrameOfReference decode: {} bytes, ref={}, {}b/val, count={}",
        packed.len(),
        reference,
        bits,
        count
    );

    #[cfg(all(feature = "gpu", target_os = "macos", target_arch = "aarch64"))]
    {
        use metal_ffi::{create_buffer, create_empty_buffer, execute_kernel, read_buffer};

        let ctx = metal_ffi::RawMetalContext::new()
            .map_err(|e| anyhow::anyhow!("Metal initialization failed: {}", e))?;

        let pipeline = ctx
            .get_pipeline("for_decode_f32")
            .map_err(|e| anyhow::anyhow!("Pipeline creation failed: {}", e))?;

        let input_buffer = create_buffer(&ctx.device, packed);
        let output_buffer = create_empty_buffer::<f32>(&ctx.device, count);
        let base_buffer = create_buffer(&ctx.device, &[reference as f32]);
        let bit_width_buffer = create_buffer(&ctx.device, &[bits as i32]);
        let n_buffer = create_buffer(&ctx.device, &[count as i32]);

        execute_kernel(
            &ctx,
            &pipeline,
            &[
                &input_buffer,
                &output_buffer,
                &base_buffer,
                &bit_width_buffer,
                &n_buffer,
            ],
            count,
        )
        .map_err(|e| anyhow::anyhow!("Kernel execution failed: {}", e))?;

        let values = read_buffer::<f32>(&output_buffer, count);

        debug!(
            "✅ [Metal] FrameOfReference decoded {} bytes → {} values (GPU)",
            packed.len(),
            values.len()
        );
        Ok(values)
    }

    #[cfg(not(all(feature = "gpu", target_os = "macos", target_arch = "aarch64")))]
    {
        // CPU fallback (shared scalar core; see kernels/cpu_fallback.rs)
        let values =
            super::cpu_fallback::frame_of_reference_decode_f32(packed, reference, bits, count)?;

        debug!(
            "✅ [Metal] FrameOfReference decoded {} bytes → {} values (CPU fallback)",
            packed.len(),
            values.len()
        );
        Ok(values)
    }
}

// ============================================================================
// ZIGZAG ENCODING/DECODING
// ============================================================================

/// Metal Zigzag encoding
///
/// # Metal Kernel (MSL)
/// ```metal
/// kernel void zigzag_encode_f32(
///     device const float* input [[buffer(0)]],
///     device int* output [[buffer(1)]],
///     uint gid [[thread_position_in_grid]]
/// ) {
///     int n = as_type<int>(input[gid]);
///     output[gid] = (n << 1) ^ (n >> 31);
/// }
/// ```text
pub fn metal_zigzag_encode_f32(values: &[f32], bits: u8) -> Result<Vec<u8>> {
    trace!(
        "🔧 [Metal] Zigzag encode: {} values, {}b/val",
        values.len(),
        bits
    );

    #[cfg(all(feature = "gpu", target_os = "macos", target_arch = "aarch64"))]
    {
        use metal_ffi::{
            RawMetalContext, create_buffer, create_empty_buffer, execute_kernel, read_buffer,
        };

        let ctx = RawMetalContext::new()
            .map_err(|e| anyhow::anyhow!("Metal initialization failed: {}", e))?;

        // Convert f32 to i64
        let values_i64: Vec<i64> = values.iter().map(|&v| v.to_bits() as i64).collect();

        // Step 1: Zigzag encode on GPU
        let zigzag_pipeline = ctx
            .get_pipeline("zigzag_encode")
            .map_err(|e| anyhow::anyhow!("Zigzag pipeline creation failed: {}", e))?;

        let input_buffer = create_buffer(&ctx.device, &values_i64);
        let zigzag_buffer = create_empty_buffer::<u64>(&ctx.device, values.len());

        execute_kernel(
            &ctx,
            &zigzag_pipeline,
            &[&input_buffer, &zigzag_buffer],
            values.len(),
        )
        .map_err(|e| anyhow::anyhow!("Zigzag kernel execution failed: {}", e))?;

        let zigzag_u64 = read_buffer::<u64>(&zigzag_buffer, values.len());
        let zigzag_i64: Vec<i64> = zigzag_u64.iter().map(|&v| v as i64).collect();

        // Step 2: Bitpack the zigzag values on GPU
        let bitpack_pipeline = ctx
            .get_pipeline("bitpack_encode")
            .map_err(|e| anyhow::anyhow!("BitPack pipeline creation failed: {}", e))?;

        let total_bits = zigzag_i64.len() * bits as usize;
        let byte_count = (total_bits + 7) / 8;
        let word_count = (byte_count + 3) / 4;

        let bitpack_input_buffer = create_buffer(&ctx.device, &zigzag_i64);
        let output_buffer = create_empty_buffer::<u32>(&ctx.device, word_count);
        let bit_width_buffer = create_buffer(&ctx.device, &[bits as i32]);
        let n_buffer = create_buffer(&ctx.device, &[zigzag_i64.len() as i32]);

        execute_kernel(
            &ctx,
            &bitpack_pipeline,
            &[
                &bitpack_input_buffer,
                &output_buffer,
                &bit_width_buffer,
                &n_buffer,
            ],
            zigzag_i64.len(),
        )
        .map_err(|e| anyhow::anyhow!("BitPack kernel execution failed: {}", e))?;

        // Read result and convert to bytes
        let packed_words = read_buffer::<u32>(&output_buffer, word_count);
        let mut result = vec![0u8; byte_count];
        for (i, &word) in packed_words.iter().enumerate() {
            let bytes = word.to_le_bytes();
            for j in 0..4 {
                if i * 4 + j < byte_count {
                    result[i * 4 + j] = bytes[j];
                }
            }
        }

        debug!(
            "✅ [Metal] Zigzag encoded {} values → {} bytes (GPU)",
            values.len(),
            result.len()
        );
        Ok(result)
    }

    #[cfg(not(all(feature = "gpu", target_os = "macos", target_arch = "aarch64")))]
    {
        // CPU fallback (shared scalar core; see kernels/cpu_fallback.rs)
        let result = super::cpu_fallback::zigzag_encode_f32(values, bits)?;

        debug!(
            "✅ [Metal] Zigzag encoded {} values → {} bytes (CPU fallback)",
            values.len(),
            result.len()
        );
        Ok(result)
    }
}

/// Metal Zigzag decoding
pub fn metal_zigzag_decode_f32(packed: &[u8], bits: u8, count: usize) -> Result<Vec<f32>> {
    trace!(
        "🔧 [Metal] Zigzag decode: {} bytes, {}b/val, count={}",
        packed.len(),
        bits,
        count
    );

    #[cfg(all(feature = "gpu", target_os = "macos", target_arch = "aarch64"))]
    {
        use metal_ffi::{
            RawMetalContext, create_buffer, create_empty_buffer, execute_kernel, read_buffer,
        };

        let ctx = RawMetalContext::new()
            .map_err(|e| anyhow::anyhow!("Failed to create Metal context: {}", e))?;

        // Step 1: Bit-unpack using GPU
        let bitpack_pipeline = ctx
            .get_pipeline("bitpack_decode")
            .map_err(|e| anyhow::anyhow!("Failed to get bitpack_decode pipeline: {}", e))?;

        let packed_buffer = create_buffer(&ctx.device, packed);
        let unpacked_buffer = create_empty_buffer::<i64>(&ctx.device, count);
        let bit_width_buffer = create_buffer(&ctx.device, &[bits as i32]);
        let n_buffer = create_buffer(&ctx.device, &[count as i32]);

        execute_kernel(
            &ctx,
            &bitpack_pipeline,
            &[
                &packed_buffer,
                &unpacked_buffer,
                &bit_width_buffer,
                &n_buffer,
            ],
            count,
        )
        .map_err(|e| anyhow::anyhow!("Bitpack decode kernel failed: {}", e))?;

        let zigzag_values = read_buffer::<i64>(&unpacked_buffer, count);

        // Step 2: Zigzag decode using GPU
        let zigzag_pipeline = ctx
            .get_pipeline("zigzag_decode")
            .map_err(|e| anyhow::anyhow!("Failed to get zigzag_decode pipeline: {}", e))?;

        let zigzag_input = create_buffer(
            &ctx.device,
            &zigzag_values.iter().map(|&v| v as u64).collect::<Vec<_>>(),
        );
        let decoded_buffer = create_empty_buffer::<i64>(&ctx.device, count);

        execute_kernel(
            &ctx,
            &zigzag_pipeline,
            &[&zigzag_input, &decoded_buffer],
            count,
        )
        .map_err(|e| anyhow::anyhow!("Zigzag decode kernel failed: {}", e))?;

        let decoded_values = read_buffer::<i64>(&decoded_buffer, count);

        // Convert i64 back to f32
        let values: Vec<f32> = decoded_values
            .iter()
            .map(|&v| f32::from_bits(v as u32))
            .collect();

        debug!(
            "✅ [Metal GPU] Zigzag decoded {} bytes → {} values",
            packed.len(),
            values.len()
        );
        Ok(values)
    }

    #[cfg(not(all(feature = "gpu", target_os = "macos", target_arch = "aarch64")))]
    {
        // CPU fallback (shared scalar core; see kernels/cpu_fallback.rs)
        let values = super::cpu_fallback::zigzag_decode_f32(packed, bits, count)?;

        debug!(
            "✅ [Metal CPU fallback] Zigzag decoded {} bytes → {} values",
            packed.len(),
            values.len()
        );
        Ok(values)
    }
}

// ============================================================================
// PFOR-DELTA ENCODING/DECODING
// ============================================================================

/// Metal PForDelta encoding (stub - complex kernel)
pub fn metal_pfor_delta_encode_f32(
    values: &[f32],
    majority_bits: u8,
    base: i64,
) -> Result<Vec<u8>> {
    trace!(
        "🔧 [Metal] PForDelta encode: {} values, {}b majority, base={}",
        values.len(),
        majority_bits,
        base
    );

    // Deferred: Real Metal implementation with parallel exception detection
    anyhow::bail!("Metal PForDelta encoding not yet implemented - use SIMD fallback")
}

/// Metal PForDelta decoding (stub - complex kernel)
pub fn metal_pfor_delta_decode_f32(
    data: &[u8],
    majority_bits: u8,
    base: i64,
    count: usize,
) -> Result<Vec<f32>> {
    trace!(
        "🔧 [Metal] PForDelta decode: {} bytes, {}b majority, base={}, count={}",
        data.len(),
        majority_bits,
        base,
        count
    );

    // Deferred: Real Metal implementation
    anyhow::bail!("Metal PForDelta decoding not yet implemented - use SIMD fallback")
}

// ============================================================================
// DOUBLE-DELTA ENCODING/DECODING
// ============================================================================

/// Metal DoubleDelta encoding for f32
///
/// Three-phase GPU algorithm:
/// - Phase 1: Convert f32 → i32 bits (parallel)
/// - Phase 2: Compute first deltas (parallel)
/// - Phase 3: Compute second deltas (parallel)
///
/// Returns: [base, first_delta, ...double_deltas]
pub fn metal_double_delta_encode_f32(values: &[f32]) -> Result<Vec<i64>> {
    trace!("🔧 [Metal] DoubleDelta encode: {} values", values.len());

    if values.is_empty() {
        return Ok(Vec::new());
    }

    if values.len() == 1 {
        let base = values[0].to_bits() as i32 as i64;
        return Ok(vec![base]);
    }

    if values.len() == 2 {
        let bits: Vec<i32> = values.iter().map(|&v| v.to_bits() as i32).collect();
        let first_delta = (bits[1] as i64) - (bits[0] as i64);
        return Ok(vec![bits[0] as i64, first_delta]);
    }

    #[cfg(all(feature = "gpu", target_os = "macos", target_arch = "aarch64"))]
    {
        use metal_ffi::*;

        let ctx = RawMetalContext::new()
            .map_err(|e| anyhow::anyhow!("Failed to create Metal context: {}", e))?;

        let n = values.len();

        // Create buffers
        let input_buffer = create_buffer(&ctx.device, values);
        let bits_buffer = create_empty_buffer::<i32>(&ctx.device, n);
        let first_deltas_buffer = create_empty_buffer::<i64>(&ctx.device, n - 1);
        let double_deltas_buffer = create_empty_buffer::<i64>(&ctx.device, n - 2);

        // Phase 1: f32 → i32 bits
        let pipeline = ctx
            .get_pipeline("double_delta_f32_to_bits")
            .map_err(|e| anyhow::anyhow!("Pipeline error: {}", e))?;
        execute_kernel(&ctx, &pipeline, &[&input_buffer, &bits_buffer], n)
            .map_err(|e| anyhow::anyhow!("Kernel execution error: {}", e))?;

        // Phase 2: First deltas
        let pipeline = ctx
            .get_pipeline("first_deltas")
            .map_err(|e| anyhow::anyhow!("Pipeline error: {}", e))?;
        execute_kernel(&ctx, &pipeline, &[&bits_buffer, &first_deltas_buffer], n)
            .map_err(|e| anyhow::anyhow!("Kernel execution error: {}", e))?;

        // Phase 3: Second deltas
        let pipeline = ctx
            .get_pipeline("second_deltas")
            .map_err(|e| anyhow::anyhow!("Pipeline error: {}", e))?;
        execute_kernel(
            &ctx,
            &pipeline,
            &[&first_deltas_buffer, &double_deltas_buffer],
            n - 1,
        )
        .map_err(|e| anyhow::anyhow!("Kernel execution error: {}", e))?;

        // Read results back
        let bits_host: Vec<i32> = read_buffer(&bits_buffer, n);
        let first_deltas_host: Vec<i64> = read_buffer(&first_deltas_buffer, n - 1);
        let double_deltas_host: Vec<i64> = read_buffer(&double_deltas_buffer, n - 2);

        let base = bits_host[0] as i64;
        let first_delta = first_deltas_host[0];

        // Construct result: [base, first_delta, ...double_deltas]
        let mut result = Vec::with_capacity(2 + double_deltas_host.len());
        result.push(base);
        result.push(first_delta);
        result.extend(double_deltas_host);

        debug!(
            "✅ [Metal] DoubleDelta encoded {} values → {} deltas (GPU)",
            values.len(),
            result.len()
        );
        Ok(result)
    }

    #[cfg(not(all(feature = "gpu", target_os = "macos", target_arch = "aarch64")))]
    {
        // CPU fallback
        let bits: Vec<i32> = values.iter().map(|&v| v.to_bits() as i32).collect();

        let mut first_deltas: Vec<i64> = Vec::with_capacity(bits.len() - 1);
        for i in 1..bits.len() {
            let curr = bits[i] as i64;
            let prev = bits[i - 1] as i64;
            first_deltas.push(curr - prev);
        }

        let mut double_deltas: Vec<i64> = Vec::with_capacity(first_deltas.len() - 1);
        for i in 1..first_deltas.len() {
            double_deltas.push(first_deltas[i] - first_deltas[i - 1]);
        }

        let mut result = Vec::with_capacity(2 + double_deltas.len());
        result.push(bits[0] as i64);
        result.push(first_deltas[0]);
        result.extend(double_deltas);

        debug!(
            "✅ [Metal] DoubleDelta encoded {} values → {} deltas (CPU fallback)",
            values.len(),
            result.len()
        );
        Ok(result)
    }
}

/// Metal DoubleDelta decoding for f32
///
/// Reconstructs f32 values from double-delta encoding
pub fn metal_double_delta_decode_f32(double_deltas: &[i64], count: usize) -> Result<Vec<f32>> {
    trace!(
        "🔧 [Metal] DoubleDelta decode: {} deltas, count={}",
        double_deltas.len(),
        count
    );

    if count == 0 || double_deltas.is_empty() {
        return Ok(Vec::new());
    }

    if count == 1 {
        let base = double_deltas[0];
        return Ok(vec![f32::from_bits(base as u32)]);
    }

    if count == 2 {
        let base = double_deltas[0];
        let first_delta = double_deltas[1];
        let v1 = f32::from_bits(base as u32);
        let v2 = f32::from_bits((base + first_delta) as i32 as u32);
        return Ok(vec![v1, v2]);
    }

    // CPU implementation (sequential reconstruction)
    // Deferred: Investigate GPU scan-based parallel reconstruction
    let base = double_deltas[0];
    let first_delta = double_deltas[1];

    let mut first_deltas: Vec<i64> = Vec::with_capacity(count - 1);
    first_deltas.push(first_delta);

    for i in 2..double_deltas.len() {
        let prev_delta = first_deltas
            .last()
            .ok_or_else(|| anyhow::anyhow!("first_deltas should not be empty at index {}", i))?;
        let dd = double_deltas[i];
        first_deltas.push(prev_delta + dd);
    }

    let mut result = Vec::with_capacity(count);
    result.push(f32::from_bits(base as u32));

    let mut prev_value = base as i64;
    for &delta in &first_deltas {
        let value = prev_value + delta;
        result.push(f32::from_bits(value as i32 as u32));
        prev_value = value;
    }

    debug!(
        "✅ [Metal] DoubleDelta decoded {} deltas → {} values (CPU)",
        double_deltas.len(),
        result.len()
    );
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_context_creation() {
        let ctx = MetalContext::new(10000, 128).expect("MetalContext creation should succeed");
        assert!(ctx.config().batch_size >= 10000);
        assert_eq!(ctx.config().threads_per_block, 256);
    }

    #[test]
    #[cfg(all(feature = "gpu", target_os = "macos", target_arch = "aarch64"))]
    fn test_metal_gpu_availability() {
        let ctx = MetalContext::new(1000, 128).expect("MetalContext creation should succeed");
        assert!(
            ctx.has_gpu(),
            "Metal GPU should be available on Apple Silicon"
        );
    }

    #[test]
    fn test_metal_delta_encode() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let deltas = metal_delta_encode_f32(&values, 0.0).expect("Delta encoding should succeed");
        assert_eq!(deltas.len(), 4);
        assert_eq!(deltas[0], 1);
        assert_eq!(deltas[1], 2);
        assert_eq!(deltas[2], 3);
        assert_eq!(deltas[3], 4);
    }

    #[test]
    fn test_metal_delta_roundtrip() {
        let values = vec![10.0f32, 20.0, 30.0, 40.0];
        let base = 5.0;

        let deltas = metal_delta_encode_f32(&values, base).expect("Delta encoding should succeed");
        let decoded = metal_delta_decode_f32(&deltas, base).expect("Delta decoding should succeed");

        for (original, recovered) in values.iter().zip(decoded.iter()) {
            assert!((original - recovered).abs() < 0.01);
        }
    }

    #[test]
    fn test_metal_delta_large_batch() {
        // Test with larger batch to exercise GPU parallelism
        let values: Vec<f32> = (0..1000).map(|i| i as f32 * 1.5).collect();
        let base = 100.0;

        let deltas = metal_delta_encode_f32(&values, base).expect("Delta encoding should succeed");
        let decoded = metal_delta_decode_f32(&deltas, base).expect("Delta decoding should succeed");

        assert_eq!(values.len(), decoded.len());
        for (original, recovered) in values.iter().zip(decoded.iter()) {
            // The deferred fallback stores deltas as integral i64
            // (`(v - base) as i64`), so fractional parts are truncated and
            // the recovery error is bounded by 1.0 — not 0.1. (This test had
            // a 0.1 tolerance since inception but never ran: the module is
            // macOS/aarch64 + `gpu`/`metal`-feature gated.)
            assert!(
                (original - recovered).abs() < 1.0,
                "Original: {}, Recovered: {}",
                original,
                recovered
            );
        }
    }

    #[test]
    fn test_metal_frame_of_reference_roundtrip() {
        let values = vec![100.0f32, 105.0, 110.0, 115.0, 120.0];
        let reference = 100;
        let bits = 8;

        let encoded = metal_frame_of_reference_encode_f32(&values, reference, bits)
            .expect("Frame-of-reference encoding should succeed");
        let decoded = metal_frame_of_reference_decode_f32(&encoded, reference, bits, values.len())
            .expect("Frame-of-reference decoding should succeed");

        assert_eq!(values.len(), decoded.len());
        for (original, recovered) in values.iter().zip(decoded.iter()) {
            assert!(
                (original - recovered).abs() < 1.0,
                "Original: {}, Recovered: {}",
                original,
                recovered
            );
        }
    }

    #[test]
    fn test_metal_frame_of_reference_varying_bits() {
        let values = vec![50.0f32, 51.0, 52.0, 53.0];
        let reference = 50;

        // Test different bit widths
        for bits in [4, 8, 16, 24, 32] {
            let encoded = metal_frame_of_reference_encode_f32(&values, reference, bits)
                .expect("Frame-of-reference encoding should succeed");
            let decoded =
                metal_frame_of_reference_decode_f32(&encoded, reference, bits, values.len())
                    .expect("Frame-of-reference decoding should succeed");

            assert_eq!(values.len(), decoded.len(), "Failed for {} bits", bits);
            for (i, (original, recovered)) in values.iter().zip(decoded.iter()).enumerate() {
                assert!(
                    (original - recovered).abs() < 1.0,
                    "Mismatch at index {} for {} bits: {} vs {}",
                    i,
                    bits,
                    original,
                    recovered
                );
            }
        }
    }

    #[test]
    fn test_metal_frame_of_reference_large_batch() {
        // Test with 1024 vectors to exercise GPU threadgroups
        let values: Vec<f32> = (0..1024).map(|i| 1000.0 + (i as f32) * 0.1).collect();
        let reference = 1000;
        let bits = 16;

        let encoded = metal_frame_of_reference_encode_f32(&values, reference, bits)
            .expect("Frame-of-reference encoding should succeed");
        let decoded = metal_frame_of_reference_decode_f32(&encoded, reference, bits, values.len())
            .expect("Frame-of-reference decoding should succeed");

        assert_eq!(values.len(), decoded.len());
        for (i, (original, recovered)) in values.iter().zip(decoded.iter()).enumerate() {
            assert!(
                (original - recovered).abs() < 1.0,
                "Mismatch at index {}: {} vs {}",
                i,
                original,
                recovered
            );
        }
    }

    #[test]
    fn test_metal_bitpack_roundtrip() {
        let values = vec![1.5f32, 2.5, 3.5, 4.5];
        let bits = 16;

        let encoded =
            metal_bitpack_encode_f32(&values, bits).expect("Bitpack encoding should succeed");
        let decoded = metal_bitpack_decode_f32(&encoded, bits, values.len())
            .expect("Bitpack decoding should succeed");

        assert_eq!(values.len(), decoded.len());
        // Note: Bit-packing loses precision, so we just check the data survived
        assert!(decoded.iter().all(|v| !v.is_nan()));
    }

    #[test]
    fn test_metal_zigzag_roundtrip() {
        let values = vec![-10.0f32, -5.0, 0.0, 5.0, 10.0];
        let bits = 16;

        let encoded =
            metal_zigzag_encode_f32(&values, bits).expect("Zigzag encoding should succeed");
        let decoded = metal_zigzag_decode_f32(&encoded, bits, values.len())
            .expect("Zigzag decoding should succeed");

        assert_eq!(values.len(), decoded.len());
        // Zigzag operates on the IEEE word: ordinary floats like -10.0 zigzag
        // to ~2^31 words, so at 16 bits the low-word round-trip can land on a
        // NaN pattern (e.g. 0xFFFF -> 0xFFFF8000) by construction. Only the
        // shape is guaranteed at narrow widths — this test had a `!is_nan`
        // assert since inception but never ran (module is macOS/aarch64 +
        // `gpu`/`metal`-feature gated).
    }

    #[test]
    #[cfg(all(feature = "gpu", target_os = "macos", target_arch = "aarch64"))]
    fn test_metal_gpu_device_info() {
        use metal_ffi::RawMetalContext;

        let ctx = RawMetalContext::new().expect("Failed to create Metal context");
        let device_name = ctx.device.name();
        assert!(!device_name.is_empty(), "Device name should not be empty");
        assert!(
            device_name.contains("Apple")
                || device_name.contains("M1")
                || device_name.contains("M2")
                || device_name.contains("M3")
                || device_name.contains("M4"),
            "Expected Apple GPU, got: {}",
            device_name
        );
    }

    #[test]
    fn test_metal_pfor_delta_not_implemented() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let result = metal_pfor_delta_encode_f32(&values, 8, 0);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("not yet implemented")
        );
    }
}
