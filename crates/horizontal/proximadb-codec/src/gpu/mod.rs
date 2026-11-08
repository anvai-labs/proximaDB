// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! GPU encoder/decoder - Platform-specific implementations
//!
//! This module provides GPU-accelerated encoding/decoding with
//! conditional compilation based on platform and features:
//!
//! - CUDA: Linux, Windows (#[cfg(feature = "gpu")])
//! - ROCm: Linux (#[cfg(feature = "gpu")])
//! - MPS (Metal): macOS (#[cfg(feature = "gpu")])
//! - OpenCL: Cross-platform fallback (#[cfg(feature = "gpu")])
//!
//! ## Architecture
//!
//! The GPU encoder/decoder uses a fallback strategy:
//! 1. **GPU Detection**: Checks for available GPU backend (CUDA/ROCm/MPS/OpenCL)
//! 2. **Kernel Selection**: Routes to platform-specific GPU kernels
//! 3. **SIMD Fallback**: Falls back to SIMD if GPU unavailable or scheme unsupported
//!
//! ## Current Implementation Status
//!
//! - ✅ **API Layer**: Complete - GpuEncoder and GpuDecoder traits implemented
//! - ✅ **Backend Detection**: Complete - Integrated with HardwareBackend
//! - ✅ **GPU Kernel Architecture**: Complete - All 4 backends with kernel dispatch
//! - ✅ **Memory Utilities**: Complete - GpuBatchConfig and GpuBuffer infrastructure
//! - ⏳ **Real GPU Compilation**: DEFERRED - Kernels use CPU fallback until real GPU compilation added
//! - ⏳ **VectorMemoryPool Integration**: DEFERRED - Integrate with memory pooling system
//!
//! ## GPU Kernel Implementation
//!
//! GPU kernels are implemented in `kernels/` subdirectory:
//! - `kernels/cuda.rs` - NVIDIA CUDA kernels (Linux x86_64)
//! - `kernels/rocm.rs` - AMD ROCm/HIP kernels (Linux)
//! - `kernels/metal.rs` - Apple Metal Shaders (macOS ARM64)
//! - `kernels/opencl.rs` - Cross-platform OpenCL kernels
//! - `kernels/utils.rs` - Common GPU utilities and batch configuration
//!
//! Each kernel module provides:
//! - Delta encoding/decoding
//! - BitPacked encoding/decoding
//! - FrameOfReference encoding/decoding
//! - Zigzag encoding/decoding
//! - PForDelta encoding/decoding (stub - returns error)
//!
//! ## Next Steps for Real GPU Acceleration
//!
//! 1. Add CUDA kernel compilation using nvcc
//! 2. Add Metal shader compilation using metal compiler
//! 3. Add OpenCL kernel compilation using OpenCL runtime
//! 4. Add ROCm/HIP kernel compilation using hipcc
//! 5. Integrate with VectorMemoryPool for GPU memory management
//! 6. Implement GPU batching with optimal batch sizes per backend

pub mod batching;
pub mod decoder;
pub mod encoder;
pub mod examples;
pub mod kernels;

pub use batching::{BatchPerformanceEstimator, BatchingStrategy, GpuBatchIterator, GpuBatchSizer};
pub use decoder::GpuDecoder;
pub use encoder::GpuEncoder;

// Encoder and decoder advertise the same scheme-level capability.
fn supports_scheme(scheme: &crate::types::ProximaScheme) -> bool {
    use crate::types::ProximaScheme;

    let backend = crate::simd::get_simd_backend();
    if !backend.is_gpu() {
        return false;
    }

    matches!(
        scheme,
        ProximaScheme::Delta { .. }
            | ProximaScheme::BitPacked { .. }
            | ProximaScheme::FrameOfReference { .. }
            | ProximaScheme::Zigzag { .. }
            | ProximaScheme::PForDelta { .. }
    )
}
