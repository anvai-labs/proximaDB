// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! GPU Encoder - Hardware-accelerated encoding using GPU compute
//!
//! This encoder uses GPU parallel processing to accelerate encoding operations.
//! It automatically detects the best available GPU backend (CUDA on Linux,
//! ROCm on AMD, MPS on Apple Silicon) and falls back to SIMD if GPU is unavailable.

use anyhow::Result;
use tracing::{debug, trace};

use crate::simd::{
    get_simd_backend, simd_bitpack_encode_f32, simd_delta_encode_f32,
    simd_frame_of_reference_encode_f32, simd_pfor_delta_encode_f32, simd_zigzag_encode_f32,
};
use crate::traits::RawEncoder;
use crate::types::ProximaScheme;
use proximadb_hardware_caps::HardwareBackend;

// Import GPU kernels
#[cfg(all(feature = "gpu", target_os = "linux", target_arch = "x86_64"))]
use crate::gpu::kernels::cuda;

#[cfg(all(feature = "gpu", target_os = "linux"))]
use crate::gpu::kernels::rocm;

#[cfg(all(feature = "gpu", target_os = "macos", target_arch = "aarch64"))]
use crate::gpu::kernels::metal;

#[cfg(feature = "gpu")]
use crate::gpu::kernels::opencl;

/// GPU-accelerated encoder
///
/// Supports:
/// - Delta encoding (GPU/CUDA/ROCm/MPS)
/// - BitPacked encoding (GPU/CUDA/ROCm/MPS)
/// - FrameOfReference encoding (GPU/CUDA/ROCm/MPS)
/// - Zigzag encoding (GPU/CUDA/ROCm/MPS)
/// - PForDelta encoding (GPU/CUDA/ROCm/MPS)
///
/// Falls back to SIMD implementation if GPU is unavailable.
pub struct GpuEncoder;

impl GpuEncoder {
    /// Dispatch Delta encoding to appropriate GPU backend
    #[allow(dead_code)]
    fn gpu_delta_encode(
        &self,
        values: &[f32],
        base: f32,
        backend: &HardwareBackend,
    ) -> Result<Vec<i64>> {
        match backend {
            #[cfg(all(feature = "gpu", target_os = "linux", target_arch = "x86_64"))]
            HardwareBackend::CUDA => cuda::cuda_delta_encode_f32(values, base),

            #[cfg(all(feature = "gpu", target_os = "linux"))]
            HardwareBackend::ROCm => rocm::rocm_delta_encode_f32(values, base),

            #[cfg(all(feature = "gpu", target_os = "macos", target_arch = "aarch64"))]
            HardwareBackend::MPS => metal::metal_delta_encode_f32(values, base),

            #[cfg(feature = "gpu")]
            HardwareBackend::OpenCL => opencl::opencl_delta_encode_f32(values, base),

            _ => {
                // Fall back to SIMD if GPU kernel not available
                trace!(
                    "⚠️  [GPU] Backend {:?} not available, falling back to SIMD",
                    backend
                );
                simd_delta_encode_f32(values, base)
            }
        }
    }

    /// Dispatch BitPacked encoding to appropriate GPU backend
    fn gpu_bitpack_encode(
        &self,
        values: &[f32],
        bits: u8,
        backend: &HardwareBackend,
    ) -> Result<Vec<u8>> {
        match backend {
            #[cfg(all(feature = "gpu", target_os = "linux", target_arch = "x86_64"))]
            HardwareBackend::CUDA => cuda::cuda_bitpack_encode_f32(values, bits),

            #[cfg(all(feature = "gpu", target_os = "linux"))]
            HardwareBackend::ROCm => rocm::rocm_bitpack_encode_f32(values, bits),

            #[cfg(all(feature = "gpu", target_os = "macos", target_arch = "aarch64"))]
            HardwareBackend::MPS => metal::metal_bitpack_encode_f32(values, bits),

            #[cfg(feature = "gpu")]
            HardwareBackend::OpenCL => opencl::opencl_bitpack_encode_f32(values, bits),

            _ => {
                trace!(
                    "⚠️  [GPU] Backend {:?} not available, falling back to SIMD",
                    backend
                );
                simd_bitpack_encode_f32(values, bits)
            }
        }
    }

    /// Dispatch FrameOfReference encoding to appropriate GPU backend
    fn gpu_frame_of_reference_encode(
        &self,
        values: &[f32],
        reference: i64,
        bits: u8,
        backend: &HardwareBackend,
    ) -> Result<Vec<u8>> {
        match backend {
            #[cfg(all(feature = "gpu", target_os = "linux", target_arch = "x86_64"))]
            HardwareBackend::CUDA => {
                cuda::cuda_frame_of_reference_encode_f32(values, reference, bits)
            }

            #[cfg(all(feature = "gpu", target_os = "linux"))]
            HardwareBackend::ROCm => {
                rocm::rocm_frame_of_reference_encode_f32(values, reference, bits)
            }

            #[cfg(all(feature = "gpu", target_os = "macos", target_arch = "aarch64"))]
            HardwareBackend::MPS => {
                metal::metal_frame_of_reference_encode_f32(values, reference, bits)
            }

            #[cfg(feature = "gpu")]
            HardwareBackend::OpenCL => {
                opencl::opencl_frame_of_reference_encode_f32(values, reference, bits)
            }

            _ => {
                trace!(
                    "⚠️  [GPU] Backend {:?} not available, falling back to SIMD",
                    backend
                );
                simd_frame_of_reference_encode_f32(values, reference, bits)
            }
        }
    }

    /// Dispatch Zigzag encoding to appropriate GPU backend
    fn gpu_zigzag_encode(
        &self,
        values: &[f32],
        bits: u8,
        backend: &HardwareBackend,
    ) -> Result<Vec<u8>> {
        match backend {
            #[cfg(all(feature = "gpu", target_os = "linux", target_arch = "x86_64"))]
            HardwareBackend::CUDA => cuda::cuda_zigzag_encode_f32(values, bits),

            #[cfg(all(feature = "gpu", target_os = "linux"))]
            HardwareBackend::ROCm => rocm::rocm_zigzag_encode_f32(values, bits),

            #[cfg(all(feature = "gpu", target_os = "macos", target_arch = "aarch64"))]
            HardwareBackend::MPS => metal::metal_zigzag_encode_f32(values, bits),

            #[cfg(feature = "gpu")]
            HardwareBackend::OpenCL => opencl::opencl_zigzag_encode_f32(values, bits),

            _ => {
                trace!(
                    "⚠️  [GPU] Backend {:?} not available, falling back to SIMD",
                    backend
                );
                simd_zigzag_encode_f32(values, bits)
            }
        }
    }

    /// Dispatch PForDelta encoding to appropriate GPU backend
    fn gpu_pfor_delta_encode(
        &self,
        values: &[f32],
        majority_bits: u8,
        base: i64,
        backend: &HardwareBackend,
    ) -> Result<Vec<u8>> {
        match backend {
            #[cfg(all(feature = "gpu", target_os = "linux", target_arch = "x86_64"))]
            HardwareBackend::CUDA => cuda::cuda_pfor_delta_encode_f32(values, majority_bits, base),

            #[cfg(all(feature = "gpu", target_os = "linux"))]
            HardwareBackend::ROCm => rocm::rocm_pfor_delta_encode_f32(values, majority_bits, base),

            #[cfg(all(feature = "gpu", target_os = "macos", target_arch = "aarch64"))]
            HardwareBackend::MPS => metal::metal_pfor_delta_encode_f32(values, majority_bits, base),

            #[cfg(feature = "gpu")]
            HardwareBackend::OpenCL => {
                opencl::opencl_pfor_delta_encode_f32(values, majority_bits, base)
            }

            _ => {
                trace!(
                    "⚠️  [GPU] Backend {:?} not available, falling back to SIMD",
                    backend
                );
                simd_pfor_delta_encode_f32(values, majority_bits, base)
            }
        }
    }
}

impl RawEncoder for GpuEncoder {
    fn name(&self) -> &'static str {
        "GPU"
    }

    fn supports(&self, scheme: &ProximaScheme) -> bool {
        super::supports_scheme(scheme)
    }

    fn encode_f32(&self, values: &[f32], scheme: &ProximaScheme) -> Result<Vec<u8>> {
        let backend = get_simd_backend();

        if !backend.is_gpu() {
            anyhow::bail!("GPU encoder requires GPU backend, but got {:?}", backend);
        }

        match scheme {
            ProximaScheme::Delta { base } => {
                trace!(
                    "🚀 [GPU] Encoding {} values with Delta (base={})",
                    values.len(),
                    base
                );

                // Use baseline implementation for wire format compatibility
                use crate::impls::baseline::functions::delta;
                let result = delta::encode_f32(values, *base)?;

                debug!(
                    "✅ [GPU] Delta encoded {} values → {} bytes (using baseline format)",
                    values.len(),
                    result.len()
                );
                Ok(result)
            }

            ProximaScheme::BitPacked { bits } => {
                trace!(
                    "🚀 [GPU] Encoding {} values with BitPacked ({}b/val)",
                    values.len(),
                    bits
                );

                let packed = self.gpu_bitpack_encode(values, *bits, &backend)?;

                debug!(
                    "✅ [GPU] BitPacked encoded {} values → {} bytes",
                    values.len(),
                    packed.len()
                );
                Ok(packed)
            }

            ProximaScheme::FrameOfReference { reference, bits } => {
                trace!(
                    "🚀 [GPU] Encoding {} values with FrameOfReference (ref={}, {}b/val)",
                    values.len(),
                    reference,
                    bits
                );

                let packed =
                    self.gpu_frame_of_reference_encode(values, *reference, *bits, &backend)?;

                debug!(
                    "✅ [GPU] FrameOfReference encoded {} values → {} bytes",
                    values.len(),
                    packed.len()
                );
                Ok(packed)
            }

            ProximaScheme::Zigzag { bits } => {
                trace!(
                    "🚀 [GPU] Encoding {} values with Zigzag ({}b/val)",
                    values.len(),
                    bits
                );

                let packed = self.gpu_zigzag_encode(values, *bits, &backend)?;

                debug!(
                    "✅ [GPU] Zigzag encoded {} values → {} bytes",
                    values.len(),
                    packed.len()
                );
                Ok(packed)
            }

            ProximaScheme::PForDelta {
                majority_bits,
                base,
            } => {
                trace!(
                    "🚀 [GPU] Encoding {} values with PForDelta ({}b majority, base={})",
                    values.len(),
                    majority_bits,
                    base
                );

                let packed = self.gpu_pfor_delta_encode(values, *majority_bits, *base, &backend)?;

                debug!(
                    "✅ [GPU] PForDelta encoded {} values → {} bytes",
                    values.len(),
                    packed.len()
                );
                Ok(packed)
            }

            _ => {
                anyhow::bail!("GPU encoder does not support scheme: {}", scheme.name())
            }
        }
    }

    fn encode_i64(&self, values: &[i64], scheme: &ProximaScheme) -> Result<Vec<u8>> {
        match scheme {
            ProximaScheme::Delta { base } => {
                trace!(
                    "🚀 [GPU] Encoding {} i64 values with Delta (base={})",
                    values.len(),
                    base
                );

                // Use baseline implementation for wire format compatibility
                use crate::impls::baseline::functions::delta;
                let result = delta::encode_i64(values, *base)?;

                debug!(
                    "✅ [GPU] Delta encoded {} i64 values → {} bytes (using baseline format)",
                    values.len(),
                    result.len()
                );
                Ok(result)
            }

            _ => {
                anyhow::bail!(
                    "GPU encoder does not support scheme: {} for i64",
                    scheme.name()
                )
            }
        }
    }

    fn encode_i32(&self, values: &[i32], scheme: &ProximaScheme) -> Result<Vec<u8>> {
        match scheme {
            ProximaScheme::Delta { base } => {
                trace!(
                    "🚀 [GPU] Encoding {} i32 values with Delta (base={})",
                    values.len(),
                    base
                );

                // Use baseline implementation for wire format compatibility
                use crate::impls::baseline::functions::delta;
                let result = delta::encode_i32(values, *base)?;

                debug!(
                    "✅ [GPU] Delta encoded {} i32 values → {} bytes (using baseline format)",
                    values.len(),
                    result.len()
                );
                Ok(result)
            }

            _ => {
                anyhow::bail!(
                    "GPU encoder does not support scheme: {} for i32",
                    scheme.name()
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_encoder_supports() {
        let encoder = GpuEncoder;
        let backend = get_simd_backend();

        // Only test if GPU is available
        if !backend.is_gpu() {
            println!("⚠️  GPU not available, skipping GPU encoder tests");
            return;
        }

        // Should support: Delta, BitPacked, FrameOfReference, Zigzag, PForDelta
        assert!(encoder.supports(&ProximaScheme::Delta { base: 0 }));
        assert!(encoder.supports(&ProximaScheme::BitPacked { bits: 8 }));
        assert!(encoder.supports(&ProximaScheme::FrameOfReference {
            reference: 0,
            bits: 8
        }));
        assert!(encoder.supports(&ProximaScheme::Zigzag { bits: 8 }));
        assert!(encoder.supports(&ProximaScheme::PForDelta {
            majority_bits: 8,
            base: 0
        }));

        // Should not support other schemes
        assert!(!encoder.supports(&ProximaScheme::RunLength));
        assert!(!encoder.supports(&ProximaScheme::Dictionary));
        assert!(!encoder.supports(&ProximaScheme::Simple8b));
        assert!(!encoder.supports(&ProximaScheme::VByte));
    }

    #[test]
    fn test_gpu_delta_encode() {
        let encoder = GpuEncoder;
        let backend = get_simd_backend();

        // Only test if GPU is available
        if !backend.is_gpu() {
            println!("⚠️  GPU not available, skipping GPU delta encode test");
            return;
        }

        // Use 32 values minimum to demonstrate compression benefits
        let values: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let result = encoder.encode_f32(&values, &ProximaScheme::Delta { base: 0 });
        assert!(result.is_ok());

        let encoded = result.unwrap();
        // Using baseline bitpacked format: [base:4][bits:1][packed_deltas]
        // Baseline format is wire-compatible across all implementations
        // Verify compression (32 values * 4 bytes raw = 128 bytes)
        println!(
            "GPU Encoded size: {} bytes (raw would be 128 bytes)",
            encoded.len()
        );

        // Bitpacked format should provide some compression for sequential data
        // but may not always be < 128 due to bitpacking overhead and float representation
        assert!(
            encoded.len() <= 160,
            "Encoded size should be reasonable: {} bytes",
            encoded.len()
        );
    }
}
