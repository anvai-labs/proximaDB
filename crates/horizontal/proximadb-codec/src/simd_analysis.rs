// SIMD-accelerated analysis functions for ProximaCodec pattern detection
//
// This module provides hardware-accelerated implementations of common analysis
// operations used in encoding scheme selection. Falls back to scalar operations
// when SIMD is unavailable.

/// SIMD-accelerated min/max detection for f32 slices
///
/// Uses AVX2 on x86_64 or NEON on ARM64 for 6-8x speedup over scalar operations.
/// Automatically falls back to scalar implementation on unsupported platforms.
pub fn simd_min_max_f32(data: &[f32]) -> (f32, f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { simd_min_max_f32_avx2(data) }
        } else {
            scalar_min_max_f32(data)
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { simd_min_max_f32_neon(data) }
        } else {
            scalar_min_max_f32(data)
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        scalar_min_max_f32(data)
    }
}

/// Scalar fallback for min/max detection
#[inline]
fn scalar_min_max_f32(data: &[f32]) -> (f32, f32) {
    let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    (min, max)
}

/// AVX2 implementation for x86_64
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_min_max_f32_avx2(data: &[f32]) -> (f32, f32) {
    unsafe {
        use std::arch::x86_64::*;

        if data.is_empty() {
            return (f32::INFINITY, f32::NEG_INFINITY);
        }

        let mut min_vec = _mm256_set1_ps(f32::INFINITY);
        let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);

        let (chunks, remainder) = data.as_chunks::<8>();

        // Process 8 f32 values per iteration
        for chunk in chunks {
            let vals = _mm256_loadu_ps(chunk.as_ptr());
            min_vec = _mm256_min_ps(min_vec, vals);
            max_vec = _mm256_max_ps(max_vec, vals);
        }

        // Horizontal reduction: 8 lanes -> 1 value
        let min = horizontal_min_f32_avx2(min_vec);
        let max = horizontal_max_f32_avx2(max_vec);

        // Handle remainder with scalar operations
        remainder.iter().fold((min, max), |(a_min, a_max), &b| {
            (a_min.min(b), a_max.max(b))
        })
    }
}

/// Horizontal min reduction for AVX2
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_min_f32_avx2(vec: std::arch::x86_64::__m256) -> f32 {
    // All intrinsics here are register-only (cast/extract/min/shuffle/cvtss) and
    // therefore safe; only memory loads (handled by callers) need an unsafe block.
    use std::arch::x86_64::*;

    // Extract high and low 128-bit lanes
    let low = _mm256_castps256_ps128(vec);
    let high = _mm256_extractf128_ps(vec, 1);

    // Min across lanes
    let min128 = _mm_min_ps(low, high);

    // Shuffle and reduce 128-bit to scalar
    let shuffle1 = _mm_shuffle_ps(min128, min128, 0b00_00_11_10);
    let min64 = _mm_min_ps(min128, shuffle1);

    let shuffle2 = _mm_shuffle_ps(min64, min64, 0b00_00_00_01);
    let min32 = _mm_min_ps(min64, shuffle2);

    _mm_cvtss_f32(min32)
}

/// Horizontal max reduction for AVX2
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_max_f32_avx2(vec: std::arch::x86_64::__m256) -> f32 {
    // Register-only intrinsics (safe); see horizontal_min_f32_avx2.
    use std::arch::x86_64::*;

    let low = _mm256_castps256_ps128(vec);
    let high = _mm256_extractf128_ps(vec, 1);

    let max128 = _mm_max_ps(low, high);

    let shuffle1 = _mm_shuffle_ps(max128, max128, 0b00_00_11_10);
    let max64 = _mm_max_ps(max128, shuffle1);

    let shuffle2 = _mm_shuffle_ps(max64, max64, 0b00_00_00_01);
    let max32 = _mm_max_ps(max64, shuffle2);

    _mm_cvtss_f32(max32)
}

/// NEON implementation for ARM64
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn simd_min_max_f32_neon(data: &[f32]) -> (f32, f32) {
    unsafe {
        use std::arch::aarch64::*;

        if data.is_empty() {
            return (f32::INFINITY, f32::NEG_INFINITY);
        }

        let mut min_vec = vdupq_n_f32(f32::INFINITY);
        let mut max_vec = vdupq_n_f32(f32::NEG_INFINITY);

        let (chunks, remainder) = data.as_chunks::<4>();

        // Process 4 f32 values per iteration (NEON is 128-bit)
        for chunk in chunks {
            let vals = vld1q_f32(chunk.as_ptr());
            min_vec = vminq_f32(min_vec, vals);
            max_vec = vmaxq_f32(max_vec, vals);
        }

        // Horizontal reduction: 4 lanes -> 1 value
        let min = vminvq_f32(min_vec);
        let max = vmaxvq_f32(max_vec);

        // Handle remainder
        remainder.iter().fold((min, max), |(a_min, a_max), &b| {
            (a_min.min(b), a_max.max(b))
        })
    }
}

/// SIMD-accelerated zero counting for f32 slices
///
/// Counts values where |val| < threshold using SIMD mask operations and popcount.
/// Expected 8-10x speedup over scalar branch-heavy counting.
pub fn simd_zero_count_f32(data: &[f32], threshold: f32) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { simd_zero_count_f32_avx2(data, threshold) }
        } else {
            scalar_zero_count_f32(data, threshold)
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { simd_zero_count_f32_neon(data, threshold) }
        } else {
            scalar_zero_count_f32(data, threshold)
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        scalar_zero_count_f32(data, threshold)
    }
}

/// Scalar fallback for zero counting
#[inline]
fn scalar_zero_count_f32(data: &[f32], threshold: f32) -> usize {
    data.iter().filter(|&&v| v.abs() < threshold).count()
}

/// AVX2 implementation for zero counting
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_zero_count_f32_avx2(data: &[f32], threshold: f32) -> usize {
    unsafe {
        use std::arch::x86_64::*;

        let threshold_vec = _mm256_set1_ps(threshold);
        let neg_threshold_vec = _mm256_set1_ps(-threshold);
        let mut count = 0;

        let (chunks, remainder) = data.as_chunks::<8>();

        for chunk in chunks {
            let vals = _mm256_loadu_ps(chunk.as_ptr());

            // Check if -threshold < val < threshold
            let gt_neg = _mm256_cmp_ps(vals, neg_threshold_vec, _CMP_GT_OQ);
            let lt_pos = _mm256_cmp_ps(vals, threshold_vec, _CMP_LT_OQ);
            let is_near_zero = _mm256_and_ps(gt_neg, lt_pos);

            // Count set bits in mask
            let mask = _mm256_movemask_ps(is_near_zero);
            count += mask.count_ones() as usize;
        }

        // Handle remainder with scalar
        count + remainder.iter().filter(|&&v| v.abs() < threshold).count()
    }
}

/// NEON implementation for zero counting
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn simd_zero_count_f32_neon(data: &[f32], threshold: f32) -> usize {
    unsafe {
        use std::arch::aarch64::*;

        let threshold_vec = vdupq_n_f32(threshold);
        let neg_threshold_vec = vdupq_n_f32(-threshold);
        let mut count = 0;

        let (chunks, remainder) = data.as_chunks::<4>();

        for chunk in chunks {
            let vals = vld1q_f32(chunk.as_ptr());

            // Check if -threshold < val < threshold
            let gt_neg = vcgtq_f32(vals, neg_threshold_vec);
            let lt_pos = vcltq_f32(vals, threshold_vec);
            let is_near_zero = vandq_u32(gt_neg, lt_pos);

            // Count set lanes - each true lane has all bits set (0xFFFFFFFF)
            // Extract individual lanes and count how many are non-zero
            let lane0 = vgetq_lane_u32(is_near_zero, 0);
            let lane1 = vgetq_lane_u32(is_near_zero, 1);
            let lane2 = vgetq_lane_u32(is_near_zero, 2);
            let lane3 = vgetq_lane_u32(is_near_zero, 3);

            count += (lane0 != 0) as usize;
            count += (lane1 != 0) as usize;
            count += (lane2 != 0) as usize;
            count += (lane3 != 0) as usize;
        }

        count + remainder.iter().filter(|&&v| v.abs() < threshold).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_min_max_basic() {
        let data = vec![5.0, 2.0, 8.0, 1.0, 9.0, 3.0];
        let (min, max) = simd_min_max_f32(&data);
        assert_eq!(min, 1.0);
        assert_eq!(max, 9.0);
    }

    #[test]
    fn test_simd_min_max_empty() {
        let data: Vec<f32> = vec![];
        let (min, max) = simd_min_max_f32(&data);
        assert_eq!(min, f32::INFINITY);
        assert_eq!(max, f32::NEG_INFINITY);
    }

    #[test]
    fn test_simd_min_max_single() {
        let data = vec![42.0];
        let (min, max) = simd_min_max_f32(&data);
        assert_eq!(min, 42.0);
        assert_eq!(max, 42.0);
    }

    #[test]
    fn test_simd_min_max_large() {
        // Test with size > 8 to trigger SIMD path
        let data: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.1).collect();
        let (min, max) = simd_min_max_f32(&data);
        assert!((min - 0.0).abs() < 0.001);
        assert!((max - 102.3).abs() < 0.001);
    }

    #[test]
    fn test_simd_zero_count_basic() {
        let data = vec![0.0, 1e-10, 0.5, 1e-9, 2.0, -1e-10];
        let count = simd_zero_count_f32(&data, 1e-9);
        assert_eq!(count, 3); // 0.0, 1e-10, -1e-10
    }

    #[test]
    fn test_simd_zero_count_none() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let count = simd_zero_count_f32(&data, 1e-9);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_simd_zero_count_all() {
        let data = vec![1e-15, -1e-15, 0.0, 1e-20];
        let count = simd_zero_count_f32(&data, 1e-9);
        assert_eq!(count, 4);
    }

    #[test]
    fn test_simd_zero_count_large() {
        // Test with size > 8 to trigger SIMD path
        let mut data = vec![1.0; 1024];
        for i in (0..1024).step_by(10) {
            data[i] = 0.0;
        }
        let count = simd_zero_count_f32(&data, 1e-9);
        assert_eq!(count, 103); // 0, 10, 20, ..., 1020
    }
}
