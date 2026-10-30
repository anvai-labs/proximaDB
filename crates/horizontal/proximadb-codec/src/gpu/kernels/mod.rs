// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! GPU Kernels - Platform-specific GPU compute implementations
//!
//! This module provides GPU kernel implementations for encoding/decoding:
//!
//! ## Supported Platforms
//!
//! - **CUDA**: NVIDIA GPUs (Linux, Windows)
//! - **ROCm**: AMD GPUs (Linux)
//! - **Metal/MPS**: Apple Silicon GPUs (macOS)
//! - **OpenCL**: Cross-platform fallback
//!
//! ## Architecture
//!
//! Each kernel module implements:
//! - Delta encoding/decoding
//! - BitPacked encoding/decoding
//! - FrameOfReference encoding/decoding
//! - Zigzag encoding/decoding
//! - PForDelta encoding/decoding
//!
//! ## Memory Management
//!
//! GPU kernels use `VectorMemoryPool` for:
//! - Device memory allocation
//! - Host-device transfers
//! - Zero-copy pinned memory where supported
//!
//! ## Batching Strategy
//!
//! GPU kernels process data in batches optimized for:
//! - Warp/wavefront sizes (32/64 threads)
//! - Shared memory capacity
//! - Register pressure
//! - Memory coalescing

#[cfg(all(feature = "gpu", target_os = "linux", target_arch = "x86_64"))]
pub mod cuda;

#[cfg(all(feature = "gpu", target_os = "linux"))]
pub mod rocm;

#[cfg(all(
    any(feature = "gpu", feature = "metal"),
    target_os = "macos",
    target_arch = "aarch64"
))]
pub mod metal;

#[cfg(feature = "gpu")]
pub mod opencl;

// Shared scalar CPU fallback cores the deferred scheme kernels delegate to
// (compiles un-gated so its round-trip tests run in default builds).
pub mod cpu_fallback;

// Common GPU utilities
pub mod utils;
