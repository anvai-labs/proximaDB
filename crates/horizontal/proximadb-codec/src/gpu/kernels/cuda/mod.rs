// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! CUDA Kernels - NVIDIA GPU acceleration
//!
//! This module provides CUDA kernel implementations for encoding/decoding.
//! It uses Rust FFI to call CUDA kernels written in CUDA C/C++.
//!
//! ## Architecture
//!
//!
//! - **Kernel Files**: CUDA kernels are in `kernels.cu`
//! - **FFI Bindings**: Rust functions call CUDA C code via FFI (see `ffi.rs`)
//! - **Memory Management**: Uses unified memory where available
//! - **Async Execution**: Supports CUDA streams for overlapping compute/transfer
//!
//! ## Build Requirements
//!
//! - CUDA Toolkit 11.0+
//! - nvcc compiler
//! - `cuda-sys` crate for bindings

use anyhow::{Result, anyhow};
use tracing::{debug, trace, warn};

use super::utils::GpuBatchConfig;
use proximadb_hardware_caps::HardwareBackend;

// Import CUDA FFI bindings when available
#[cfg(all(feature = "gpu", target_os = "linux", target_arch = "x86_64"))]
mod ffi;
#[cfg(all(feature = "gpu", target_os = "linux", target_arch = "x86_64"))]
use ffi::*;

// ============================================================================
// GPU MEMORY MANAGEMENT (RAII wrapper)
// ============================================================================

#[cfg(all(feature = "gpu", target_os = "linux", target_arch = "x86_64"))]
struct GpuMemory<T> {
    ptr: *mut T,
    size: usize,
}

#[cfg(all(feature = "gpu", target_os = "linux", target_arch = "x86_64"))]
impl<T> GpuMemory<T> {
    fn allocate(count: usize) -> Result<Self> {
        let size = count * std::mem::size_of::<T>();
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();

        unsafe {
            let error = cudaMalloc(&mut ptr as *mut *mut std::ffi::c_void, size);
            check_cuda_error(error).map_err(|e| anyhow!("cudaMalloc failed: {}", e))?;
        }

        Ok(Self {
            ptr: ptr as *mut T,
            size,
        })
    }

    fn copy_from_host(&mut self, data: &[T]) -> Result<()> {
        unsafe {
            let error = cudaMemcpy(
                self.ptr as *mut std::ffi::c_void,
                data.as_ptr() as *const std::ffi::c_void,
                data.len() * std::mem::size_of::<T>(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            );
            check_cuda_error(error).map_err(|e| anyhow!("cudaMemcpy H2D failed: {}", e))?;
        }
        Ok(())
    }

    fn copy_to_host(&self, data: &mut [T]) -> Result<()> {
        unsafe {
            let error = cudaMemcpy(
                data.as_mut_ptr() as *mut std::ffi::c_void,
                self.ptr as *const std::ffi::c_void,
                data.len() * std::mem::size_of::<T>(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
            );
            check_cuda_error(error).map_err(|e| anyhow!("cudaMemcpy D2H failed: {}", e))?;
        }
        Ok(())
    }

    fn as_ptr(&self) -> *const T {
        self.ptr
    }

    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

#[cfg(all(feature = "gpu", target_os = "linux", target_arch = "x86_64"))]
impl<T> Drop for GpuMemory<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let error = cudaFree(self.ptr as *mut std::ffi::c_void);
                if error != CUDA_SUCCESS {
                    warn!("cudaFree failed: {}", get_last_cuda_error());
                }
            }
        }
    }
}

/// CUDA context wrapper
pub struct CudaContext {
    config: GpuBatchConfig,
}

impl CudaContext {
    /// Create new CUDA context
    pub fn new(total_vectors: usize, dimension: usize) -> Result<Self> {
        let config = GpuBatchConfig::for_backend(&HardwareBackend::CUDA, total_vectors, dimension);

        debug!(
            "🚀 [CUDA] Initializing context: {} vectors, dim={}",
            total_vectors, dimension
        );

        Ok(Self { config })
    }
}

// ============================================================================
// DELTA ENCODING/DECODING
// ============================================================================

/// CUDA Delta encoding for f32 values (to i64 deltas)
pub fn cuda_delta_encode_f32(values: &[f32], base: f32) -> Result<Vec<i64>> {
    trace!(
        "🔧 [CUDA] Delta encode: {} values (base={})",
        values.len(),
        base
    );

    #[cfg(all(feature = "gpu", target_os = "linux", target_arch = "x86_64"))]
    {
        // Allocate GPU memory
        let mut input_gpu = GpuMemory::<f32>::allocate(values.len())?;
        let mut output_gpu = GpuMemory::<i64>::allocate(values.len())?;

        // Copy input to GPU
        input_gpu.copy_from_host(values)?;

        // Launch CUDA kernel
        unsafe {
            cuda_delta_encode_f32(
                input_gpu.as_ptr(),
                output_gpu.as_mut_ptr(),
                base,
                values.len() as i32,
                std::ptr::null_mut(),
            );

            let error = cudaDeviceSynchronize();
            check_cuda_error(error).map_err(|e| anyhow!("CUDA kernel failed: {}", e))?;
        }

        // Copy result back to host
        let mut deltas = vec![0i64; values.len()];
        output_gpu.copy_to_host(&mut deltas)?;

        debug!(
            "✅ [CUDA] Delta encoded {} values → {} deltas (GPU)",
            values.len(),
            deltas.len()
        );
        Ok(deltas)
    }

    #[cfg(not(all(feature = "gpu", target_os = "linux", target_arch = "x86_64")))]
    {
        let deltas = super::cpu_fallback::delta_encode_f32(values, base);
        debug!(
            "✅ [CUDA] Delta encoded {} values → {} deltas (CPU fallback)",
            values.len(),
            deltas.len()
        );
        Ok(deltas)
    }
}

/// CUDA Delta decoding for f32 values from i64 deltas
pub fn cuda_delta_decode_f32(deltas: &[i64], base: f32) -> Result<Vec<f32>> {
    trace!(
        "🔧 [CUDA] Delta decode: {} deltas (base={})",
        deltas.len(),
        base
    );

    #[cfg(all(feature = "gpu", target_os = "linux", target_arch = "x86_64"))]
    {
        // Allocate GPU memory
        let mut input_gpu = GpuMemory::<i64>::allocate(deltas.len())?;
        let mut output_gpu = GpuMemory::<f32>::allocate(deltas.len())?;

        // Copy input to GPU
        input_gpu.copy_from_host(deltas)?;

        // Launch CUDA kernel
        unsafe {
            cuda_delta_decode_f32(
                input_gpu.as_ptr(),
                output_gpu.as_mut_ptr(),
                base,
                deltas.len() as i32,
                std::ptr::null_mut(),
            );

            let error = cudaDeviceSynchronize();
            check_cuda_error(error).map_err(|e| anyhow!("CUDA kernel failed: {}", e))?;
        }

        // Copy result back to host
        let mut values = vec![0.0f32; deltas.len()];
        output_gpu.copy_to_host(&mut values)?;

        debug!(
            "✅ [CUDA] Delta decoded {} deltas → {} values (GPU)",
            deltas.len(),
            values.len()
        );
        Ok(values)
    }

    #[cfg(not(all(feature = "gpu", target_os = "linux", target_arch = "x86_64")))]
    {
        let values = super::cpu_fallback::delta_decode_f32(deltas, base);
        debug!(
            "✅ [CUDA] Delta decoded {} deltas → {} values (CPU fallback)",
            deltas.len(),
            values.len()
        );
        Ok(values)
    }
}

// ============================================================================
// BITPACKED ENCODING/DECODING
// ============================================================================

/// CUDA BitPacked encoding for f32
pub fn cuda_bitpack_encode_f32(values: &[f32], bits: u8) -> Result<Vec<u8>> {
    trace!(
        "🔧 [CUDA] BitPacked encode: {} values, {}b/val",
        values.len(),
        bits
    );

    #[cfg(all(feature = "gpu", target_os = "linux", target_arch = "x86_64"))]
    {
        let n = values.len();
        let output_size = ((n * bits as usize) + 7) / 8;

        // Convert f32 to i64 for bitpacking
        let values_i64: Vec<i64> = values.iter().map(|&v| v.to_bits() as i64).collect();

        // Allocate GPU memory
        let mut input_gpu = GpuMemory::<i64>::allocate(n)?;
        let mut output_gpu = GpuMemory::<u8>::allocate(output_size)?;

        // Copy input to GPU
        input_gpu.copy_from_host(&values_i64)?;

        // Clear output buffer
        let zero_buffer = vec![0u8; output_size];
        output_gpu.copy_from_host(&zero_buffer)?;

        // Launch CUDA kernel
        unsafe {
            cuda_bitpack_encode(
                input_gpu.as_ptr(),
                output_gpu.as_mut_ptr(),
                bits as i32,
                n as i32,
                std::ptr::null_mut(),
            );

            let error = cudaDeviceSynchronize();
            check_cuda_error(error).map_err(|e| anyhow!("CUDA kernel failed: {}", e))?;
        }

        // Copy result back to host
        let mut output = vec![0u8; output_size];
        output_gpu.copy_to_host(&mut output)?;

        debug!(
            "✅ [CUDA] BitPacked encoded {} values → {} bytes (GPU)",
            values.len(),
            output.len()
        );
        Ok(output)
    }

    #[cfg(not(all(feature = "gpu", target_os = "linux", target_arch = "x86_64")))]
    {
        // CPU fallback (shared scalar core; see kernels/cpu_fallback.rs)
        let result = super::cpu_fallback::bitpack_encode_f32(values, bits)?;

        debug!(
            "✅ [CUDA] BitPacked encoded {} values → {} bytes (CPU fallback)",
            values.len(),
            result.len()
        );
        Ok(result)
    }
}

/// CUDA BitPacked decoding for f32
pub fn cuda_bitpack_decode_f32(packed: &[u8], bits: u8, count: usize) -> Result<Vec<f32>> {
    trace!(
        "🔧 [CUDA] BitPacked decode: {} bytes, {}b/val, count={}",
        packed.len(),
        bits,
        count
    );

    #[cfg(all(feature = "gpu", target_os = "linux", target_arch = "x86_64"))]
    {
        // Allocate GPU memory
        let mut input_gpu = GpuMemory::<u8>::allocate(packed.len())?;
        let mut output_gpu = GpuMemory::<i64>::allocate(count)?;

        // Copy input to GPU
        input_gpu.copy_from_host(packed)?;

        // Launch CUDA kernel
        unsafe {
            cuda_bitpack_decode(
                input_gpu.as_ptr(),
                output_gpu.as_mut_ptr(),
                bits as i32,
                count as i32,
                std::ptr::null_mut(),
            );

            let error = cudaDeviceSynchronize();
            check_cuda_error(error).map_err(|e| anyhow!("CUDA kernel failed: {}", e))?;
        }

        // Copy result back to host
        let mut output_i64 = vec![0i64; count];
        output_gpu.copy_to_host(&mut output_i64)?;

        let output: Vec<f32> = output_i64
            .into_iter()
            .map(|v| f32::from_bits(v as u32))
            .collect();
        debug!(
            "✅ [CUDA] BitPacked decoded {} bytes → {} values (GPU)",
            packed.len(),
            output.len()
        );
        Ok(output)
    }

    #[cfg(not(all(feature = "gpu", target_os = "linux", target_arch = "x86_64")))]
    {
        // CPU fallback (shared scalar core; see kernels/cpu_fallback.rs)
        let output = super::cpu_fallback::bitpack_decode_f32(packed, bits, count)?;

        debug!(
            "✅ [CUDA] BitPacked decoded {} bytes → {} values (CPU fallback)",
            packed.len(),
            output.len()
        );
        Ok(output)
    }
}

// ============================================================================
// FRAME OF REFERENCE ENCODING/DECODING
// ============================================================================

pub fn cuda_frame_of_reference_encode_f32(
    values: &[f32],
    reference: i64,
    bits: u8,
) -> Result<Vec<u8>> {
    trace!(
        "🔧 [CUDA] FrameOfReference encode: {} values, ref={}, {}b/val",
        values.len(),
        reference,
        bits
    );

    #[cfg(all(feature = "gpu", target_os = "linux", target_arch = "x86_64"))]
    {
        let n = values.len();
        let output_size = ((n * bits as usize) + 7) / 8;
        let reference_f32 = reference as f32;

        // Allocate GPU memory
        let mut input_gpu = GpuMemory::<f32>::allocate(n)?;
        let mut output_gpu = GpuMemory::<u8>::allocate(output_size)?;

        // Copy input to GPU
        input_gpu.copy_from_host(values)?;

        // Clear output buffer
        let zero_buffer = vec![0u8; output_size];
        output_gpu.copy_from_host(&zero_buffer)?;

        // Launch CUDA kernel
        unsafe {
            cuda_for_encode_f32(
                input_gpu.as_ptr(),
                output_gpu.as_mut_ptr(),
                reference_f32,
                bits as i32,
                n as i32,
                std::ptr::null_mut(),
            );

            let error = cudaDeviceSynchronize();
            check_cuda_error(error).map_err(|e| anyhow!("CUDA kernel failed: {}", e))?;
        }

        // Copy result back to host
        let mut output = vec![0u8; output_size];
        output_gpu.copy_to_host(&mut output)?;

        debug!(
            "✅ [CUDA] FrameOfReference encoded {} values → {} bytes (GPU)",
            values.len(),
            output.len()
        );
        Ok(output)
    }

    #[cfg(not(all(feature = "gpu", target_os = "linux", target_arch = "x86_64")))]
    {
        // CPU fallback (shared scalar core; see kernels/cpu_fallback.rs)
        let result = super::cpu_fallback::frame_of_reference_encode_f32(values, reference, bits)?;

        debug!(
            "✅ [CUDA] FrameOfReference encoded {} values → {} bytes (CPU fallback)",
            values.len(),
            result.len()
        );
        Ok(result)
    }
}

pub fn cuda_frame_of_reference_decode_f32(
    packed: &[u8],
    reference: i64,
    bits: u8,
    count: usize,
) -> Result<Vec<f32>> {
    trace!(
        "🔧 [CUDA] FrameOfReference decode: {} bytes, ref={}, {}b/val, count={}",
        packed.len(),
        reference,
        bits,
        count
    );

    #[cfg(all(feature = "gpu", target_os = "linux", target_arch = "x86_64"))]
    {
        let reference_f32 = reference as f32;

        // Allocate GPU memory
        let mut input_gpu = GpuMemory::<u8>::allocate(packed.len())?;
        let mut output_gpu = GpuMemory::<f32>::allocate(count)?;

        // Copy input to GPU
        input_gpu.copy_from_host(packed)?;

        // Launch CUDA kernel
        unsafe {
            cuda_for_decode_f32(
                input_gpu.as_ptr(),
                output_gpu.as_mut_ptr(),
                reference_f32,
                bits as i32,
                count as i32,
                std::ptr::null_mut(),
            );

            let error = cudaDeviceSynchronize();
            check_cuda_error(error).map_err(|e| anyhow!("CUDA kernel failed: {}", e))?;
        }

        // Copy result back to host
        let mut output = vec![0.0f32; count];
        output_gpu.copy_to_host(&mut output)?;

        debug!(
            "✅ [CUDA] FrameOfReference decoded {} bytes → {} values (GPU)",
            packed.len(),
            output.len()
        );
        Ok(output)
    }

    #[cfg(not(all(feature = "gpu", target_os = "linux", target_arch = "x86_64")))]
    {
        // CPU fallback (shared scalar core; see kernels/cpu_fallback.rs)
        let decoded =
            super::cpu_fallback::frame_of_reference_decode_f32(packed, reference, bits, count)?;

        debug!(
            "✅ [CUDA] FrameOfReference decoded {} bytes → {} values (CPU fallback)",
            packed.len(),
            decoded.len()
        );
        Ok(decoded)
    }
}

// ============================================================================
// ZIGZAG ENCODING/DECODING
// ============================================================================

pub fn cuda_zigzag_encode(values: &[i64]) -> Result<Vec<u64>> {
    trace!("🔧 [CUDA] Zigzag encode: {} values", values.len());

    #[cfg(all(feature = "gpu", target_os = "linux", target_arch = "x86_64"))]
    {
        let mut input_gpu = GpuMemory::<i64>::allocate(values.len())?;
        let mut output_gpu = GpuMemory::<u64>::allocate(values.len())?;

        input_gpu.copy_from_host(values)?;

        unsafe {
            cuda_zigzag_encode(
                input_gpu.as_ptr(),
                output_gpu.as_mut_ptr(),
                values.len() as i32,
                std::ptr::null_mut(),
            );
            let error = cudaDeviceSynchronize();
            check_cuda_error(error).map_err(|e| anyhow!("CUDA kernel failed: {}", e))?;
        }

        let mut output = vec![0u64; values.len()];
        output_gpu.copy_to_host(&mut output)?;
        debug!("✅ [CUDA] Zigzag encoded {} values (GPU)", values.len());
        Ok(output)
    }

    #[cfg(not(all(feature = "gpu", target_os = "linux", target_arch = "x86_64")))]
    {
        // CPU fallback
        let output: Vec<u64> = values
            .iter()
            .map(|&v| ((v << 1) ^ (v >> 63)) as u64)
            .collect();
        debug!(
            "✅ [CUDA] Zigzag encoded {} values (CPU fallback)",
            values.len()
        );
        Ok(output)
    }
}

// ============================================================================
// ERROR HANDLING
// ============================================================================

#[cfg(all(feature = "gpu", target_os = "linux", target_arch = "x86_64"))]
fn get_last_cuda_error() -> String {
    unsafe {
        let err = cudaGetLastError();
        let c_str = cudaGetErrorString(err);
        if c_str.is_null() {
            return "Unknown CUDA error".to_string();
        }
        std::ffi::CStr::from_ptr(c_str)
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(all(feature = "gpu", target_os = "linux", target_arch = "x86_64"))]
fn check_cuda_error(code: i32) -> Result<()> {
    if code == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(anyhow!(get_last_cuda_error()))
    }
}

// ============================================================================
// TESTS (basic structure; GPU tests likely ignored by default)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_context_creation() {
        let _ = CudaContext::new(10000, 128);
    }
}
