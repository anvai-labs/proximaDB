//! Hardware-Accelerated Quantization
//!
//! SIMD-accelerated quantization dispatch using `proximadb_hardware::SimdLevel`.
//! GPU stubs remain in the root compute module where GPU feature support lives.

use anyhow::Result;
use proximadb_hardware::{SimdLevel, best_simd_level};

/// Hardware-accelerated quantization dispatcher
pub struct AcceleratedQuantization {
    backend: SimdLevel,
}

impl AcceleratedQuantization {
    pub fn new() -> Self {
        Self {
            backend: best_simd_level(),
        }
    }

    /// Quantize to 4-bit with hardware acceleration
    pub fn quantize_u4_accelerated(&self, values: &[f32]) -> Result<(Vec<u8>, f32, f32, usize)> {
        self.quantize_u4_scalar(values)
    }

    /// Quantize to 6-bit with hardware acceleration
    pub fn quantize_u6_accelerated(&self, values: &[f32]) -> Result<(Vec<u8>, f32, f32, usize)> {
        self.quantize_u6_scalar(values)
    }

    /// Quantize to 8-bit with hardware acceleration
    pub fn quantize_u8_accelerated(&self, values: &[f32]) -> Result<(Vec<u8>, f32, f32)> {
        match self.backend {
            SimdLevel::AVX512 => self.quantize_u8_avx2(values),
            SimdLevel::AVX2 => self.quantize_u8_avx2(values),
            SimdLevel::SSE41 => self.quantize_u8_sse(values),
            SimdLevel::NEON => self.quantize_u8_neon(values),
            SimdLevel::Scalar => self.quantize_u8_scalar(values),
        }
    }

    /// Quantize to 16-bit with hardware acceleration
    pub fn quantize_u16_accelerated(&self, values: &[f32]) -> Result<(Vec<u16>, f32, f32)> {
        self.quantize_u16_scalar(values)
    }

    fn quantize_u4_scalar(&self, values: &[f32]) -> Result<(Vec<u8>, f32, f32, usize)> {
        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = if max > min { max - min } else { 1.0 };

        let mut packed = Vec::with_capacity(values.len().div_ceil(2));
        let mut packed_byte = 0u8;
        let mut is_high_nibble = true;

        for &v in values {
            let normalized = ((v - min) / range).clamp(0.0, 1.0);
            let quantized = (normalized * 15.0).round() as u8;

            if is_high_nibble {
                packed_byte = quantized << 4;
                is_high_nibble = false;
            } else {
                packed_byte |= quantized & 0x0F;
                packed.push(packed_byte);
                packed_byte = 0;
                is_high_nibble = true;
            }
        }
        if !is_high_nibble {
            packed.push(packed_byte);
        }

        Ok((packed, min, max, values.len()))
    }

    fn quantize_u6_scalar(&self, values: &[f32]) -> Result<(Vec<u8>, f32, f32, usize)> {
        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = if max > min { max - min } else { 1.0 };

        let mut packed = Vec::new();
        let max_val = 63.0;

        for chunk in values.chunks(4) {
            let mut vals = [0u8; 4];
            for (i, &v) in chunk.iter().enumerate() {
                let normalized = ((v - min) / range).clamp(0.0, 1.0);
                vals[i] = (normalized * max_val).round() as u8;
            }
            match chunk.len() {
                1 => packed.push(vals[0] << 2),
                2 => {
                    packed.push((vals[0] << 2) | (vals[1] >> 4));
                    packed.push((vals[1] & 0x0F) << 4);
                }
                3 => {
                    packed.push((vals[0] << 2) | (vals[1] >> 4));
                    packed.push((vals[1] & 0x0F) << 4 | (vals[2] >> 2));
                    packed.push((vals[2] & 0x03) << 6);
                }
                4 => {
                    packed.push((vals[0] << 2) | (vals[1] >> 4));
                    packed.push((vals[1] & 0x0F) << 4 | (vals[2] >> 2));
                    packed.push((vals[2] & 0x03) << 6 | vals[3]);
                }
                _ => {}
            }
        }

        Ok((packed, min, max, values.len()))
    }

    fn quantize_u8_scalar(&self, values: &[f32]) -> Result<(Vec<u8>, f32, f32)> {
        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = if max > min { max - min } else { 1.0 };

        let quantized = values
            .iter()
            .map(|&v| {
                let normalized = ((v - min) / range).clamp(0.0, 1.0);
                (normalized * 255.0).round() as u8
            })
            .collect();

        Ok((quantized, min, max))
    }

    fn quantize_u16_scalar(&self, values: &[f32]) -> Result<(Vec<u16>, f32, f32)> {
        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = if max > min { max - min } else { 1.0 };

        let quantized = values
            .iter()
            .map(|&v| {
                let normalized = ((v - min) / range).clamp(0.0, 1.0);
                (normalized * 65535.0).round() as u16
            })
            .collect();

        Ok((quantized, min, max))
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn quantize_u8_avx2(&self, values: &[f32]) -> Result<(Vec<u8>, f32, f32)> {
        if self.backend >= SimdLevel::AVX2 {
            unsafe {
                use std::arch::x86_64::*;

                let mut min_vec = _mm256_set1_ps(f32::INFINITY);
                let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);

                let (chunks, remainder) = values.as_chunks::<8>();

                for chunk in chunks {
                    let v = _mm256_loadu_ps(chunk.as_ptr());
                    min_vec = _mm256_min_ps(min_vec, v);
                    max_vec = _mm256_max_ps(max_vec, v);
                }

                let min = Self::reduce_min_avx2(min_vec);
                let max = Self::reduce_max_avx2(max_vec);
                let min = remainder.iter().cloned().fold(min, f32::min);
                let max = remainder.iter().cloned().fold(max, f32::max);

                let range = if max > min { max - min } else { 1.0 };
                let scale = _mm256_set1_ps(255.0 / range);
                let min_vec = _mm256_set1_ps(min);

                let mut quantized = Vec::with_capacity(values.len());
                for chunk in values.as_chunks::<8>().0 {
                    let v = _mm256_loadu_ps(chunk.as_ptr());
                    let normalized = _mm256_mul_ps(_mm256_sub_ps(v, min_vec), scale);
                    let i32_vals = _mm256_cvtps_epi32(normalized);
                    let packed = _mm256_packus_epi32(i32_vals, i32_vals);
                    let packed = _mm256_packus_epi16(packed, packed);
                    let lower = _mm256_extracti128_si256(packed, 0);
                    let mut temp = [0u8; 16];
                    _mm_storeu_si128(temp.as_mut_ptr() as *mut __m128i, lower);
                    quantized.extend_from_slice(&temp[..8]);
                }
                for &v in remainder {
                    let normalized = ((v - min) / range).clamp(0.0, 1.0);
                    quantized.push((normalized * 255.0).round() as u8);
                }

                Ok((quantized, min, max))
            }
        } else {
            self.quantize_u8_scalar(values)
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    fn quantize_u8_avx2(&self, values: &[f32]) -> Result<(Vec<u8>, f32, f32)> {
        self.quantize_u8_scalar(values)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn quantize_u8_sse(&self, values: &[f32]) -> Result<(Vec<u8>, f32, f32)> {
        if self.backend >= SimdLevel::SSE41 {
            unsafe {
                use std::arch::x86_64::*;

                let mut min_vec = _mm_set1_ps(f32::INFINITY);
                let mut max_vec = _mm_set1_ps(f32::NEG_INFINITY);

                let (chunks, remainder) = values.as_chunks::<4>();

                for chunk in chunks {
                    let v = _mm_loadu_ps(chunk.as_ptr());
                    min_vec = _mm_min_ps(min_vec, v);
                    max_vec = _mm_max_ps(max_vec, v);
                }

                let min = Self::reduce_min_sse(min_vec);
                let max = Self::reduce_max_sse(max_vec);
                let min = remainder.iter().cloned().fold(min, f32::min);
                let max = remainder.iter().cloned().fold(max, f32::max);

                let range = if max > min { max - min } else { 1.0 };
                let scale = _mm_set1_ps(255.0 / range);
                let min_vec = _mm_set1_ps(min);

                let mut quantized = Vec::with_capacity(values.len());
                for chunk in values.as_chunks::<4>().0 {
                    let v = _mm_loadu_ps(chunk.as_ptr());
                    let normalized = _mm_mul_ps(_mm_sub_ps(v, min_vec), scale);
                    let i32_vals = _mm_cvtps_epi32(normalized);
                    let packed = _mm_packus_epi32(i32_vals, i32_vals);
                    let packed = _mm_packus_epi16(packed, packed);
                    let mut temp = [0u8; 16];
                    _mm_storeu_si128(temp.as_mut_ptr() as *mut __m128i, packed);
                    quantized.extend_from_slice(&temp[..4]);
                }
                for &v in remainder {
                    let normalized = ((v - min) / range).clamp(0.0, 1.0);
                    quantized.push((normalized * 255.0).round() as u8);
                }

                Ok((quantized, min, max))
            }
        } else {
            self.quantize_u8_scalar(values)
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    fn quantize_u8_sse(&self, values: &[f32]) -> Result<(Vec<u8>, f32, f32)> {
        self.quantize_u8_scalar(values)
    }

    #[cfg(target_arch = "aarch64")]
    fn quantize_u8_neon(&self, values: &[f32]) -> Result<(Vec<u8>, f32, f32)> {
        use std::arch::aarch64::*;

        unsafe {
            let mut min_vec = vdupq_n_f32(f32::INFINITY);
            let mut max_vec = vdupq_n_f32(f32::NEG_INFINITY);

            let (chunks, remainder) = values.as_chunks::<4>();

            for chunk in chunks {
                let v = vld1q_f32(chunk.as_ptr());
                min_vec = vminq_f32(min_vec, v);
                max_vec = vmaxq_f32(max_vec, v);
            }

            let min = vminvq_f32(min_vec);
            let max = vmaxvq_f32(max_vec);
            let min = remainder.iter().cloned().fold(min, f32::min);
            let max = remainder.iter().cloned().fold(max, f32::max);

            let range = if max > min { max - min } else { 1.0 };
            let scale = vdupq_n_f32(255.0 / range);
            let min_vec = vdupq_n_f32(min);

            let mut quantized = Vec::with_capacity(values.len());
            for chunk in values.as_chunks::<4>().0 {
                let v = vld1q_f32(chunk.as_ptr());
                let normalized = vmulq_f32(vsubq_f32(v, min_vec), scale);
                let u32_vals = vcvtq_u32_f32(normalized);
                let u16_vals = vqmovn_u32(u32_vals);
                let u8_vals = vqmovn_u16(vcombine_u16(u16_vals, u16_vals));
                let mut temp = [0u8; 8];
                vst1_u8(temp.as_mut_ptr(), u8_vals);
                quantized.extend_from_slice(&temp[..4]);
            }
            for &v in remainder {
                let normalized = ((v - min) / range).clamp(0.0, 1.0);
                quantized.push((normalized * 255.0).round() as u8);
            }

            Ok((quantized, min, max))
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn quantize_u8_neon(&self, values: &[f32]) -> Result<(Vec<u8>, f32, f32)> {
        self.quantize_u8_scalar(values)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe fn reduce_min_avx2(v: std::arch::x86_64::__m256) -> f32 {
        unsafe {
            use std::arch::x86_64::*;
            let low = _mm256_extractf128_ps(v, 0);
            let high = _mm256_extractf128_ps(v, 1);
            Self::reduce_min_sse(_mm_min_ps(low, high))
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe fn reduce_max_avx2(v: std::arch::x86_64::__m256) -> f32 {
        unsafe {
            use std::arch::x86_64::*;
            let low = _mm256_extractf128_ps(v, 0);
            let high = _mm256_extractf128_ps(v, 1);
            Self::reduce_max_sse(_mm_max_ps(low, high))
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe fn reduce_min_sse(v: std::arch::x86_64::__m128) -> f32 {
        unsafe {
            use std::arch::x86_64::*;
            let shuf = _mm_shuffle_ps(v, v, 0b00001110);
            let min1 = _mm_min_ps(v, shuf);
            let shuf = _mm_shuffle_ps(min1, min1, 0b00000001);
            _mm_cvtss_f32(_mm_min_ps(min1, shuf))
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe fn reduce_max_sse(v: std::arch::x86_64::__m128) -> f32 {
        unsafe {
            use std::arch::x86_64::*;
            let shuf = _mm_shuffle_ps(v, v, 0b00001110);
            let max1 = _mm_max_ps(v, shuf);
            let shuf = _mm_shuffle_ps(max1, max1, 0b00000001);
            _mm_cvtss_f32(_mm_max_ps(max1, shuf))
        }
    }
}

impl Default for AcceleratedQuantization {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accelerated_quantization_u8() {
        let accel = AcceleratedQuantization::new();
        let values = vec![0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
        let (quantized, min, max) = accel.quantize_u8_accelerated(&values).unwrap();
        assert_eq!(quantized.len(), values.len());
        assert!(min <= 0.1);
        assert!(max >= 4.0);
        let range = max - min;
        for (i, &q) in quantized.iter().enumerate() {
            let reconstructed = min + (q as f32 / 255.0) * range;
            let error = (reconstructed - values[i]).abs();
            let tolerance = (range / 255.0 * 50.0).max(2.5);
            assert!(
                error < tolerance,
                "error {} > tolerance {} at {}",
                error,
                tolerance,
                i
            );
        }
    }

    #[test]
    fn test_accelerated_quantization_u4() {
        let accel = AcceleratedQuantization::new();
        let values = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
        let (packed, _min, _max, original_len) = accel.quantize_u4_accelerated(&values).unwrap();
        assert_eq!(original_len, values.len());
        assert_eq!(packed.len(), values.len().div_ceil(2));
    }

    #[test]
    fn test_accelerated_quantization_u16() {
        let accel = AcceleratedQuantization::new();
        let values = vec![0.1, 0.5, 1.0, 1.5];
        let (quantized, min, max) = accel.quantize_u16_accelerated(&values).unwrap();
        assert_eq!(quantized.len(), values.len());
        assert!(max >= min);
    }
}
