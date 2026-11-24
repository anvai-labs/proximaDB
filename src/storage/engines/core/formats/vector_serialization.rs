/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! Unified Vector Serialization Utilities
//!
//! This module provides centralized, high-performance vector serialization
//! that can be shared across all storage engines (SST, HELIX, NOVA, SWIFT, RAPTOR).
//!
//! ## Design Principles
//!
//! 1. **Zero-copy for fixed dimensions**: Common embedding dimensions (64, 128, 256, 384, 512, 768, 1024, 1536, 2048)
//!    use bytemuck for zero-copy serialization/deserialization.
//!
//! 2. **Fallback for variable dimensions**: Non-standard dimensions use a length-prefixed format
//!    with direct memory copy.
//!
//! 3. **SIMD-friendly alignment**: Data is aligned for optimal SIMD operations.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use crate::storage::engines::core::formats::vector_serialization::VectorSerializer;
//!
//! // Serialize
//! let vector = vec![1.0f32, 2.0, 3.0, 4.0];
//! let bytes = VectorSerializer::serialize(&vector);
//!
//! // Deserialize with known dimension
//! let restored = VectorSerializer::deserialize(&bytes, Some(4))?;
//!
//! // Deserialize with unknown dimension (reads length prefix)
//! let restored = VectorSerializer::deserialize(&bytes, None)?;
//! ```

use anyhow::Result;

/// Common embedding dimensions that get zero-copy treatment
const FIXED_DIMENSIONS: [usize; 9] = [64, 128, 256, 384, 512, 768, 1024, 1536, 2048];

/// Unified vector serialization with zero-copy optimization for common dimensions
pub struct VectorSerializer;

impl VectorSerializer {
    /// Check if a dimension qualifies for zero-copy optimization
    #[inline(always)]
    pub fn is_fixed_dimension(dim: usize) -> bool {
        FIXED_DIMENSIONS.contains(&dim)
    }

    /// Get all supported fixed dimensions
    pub fn fixed_dimensions() -> &'static [usize] {
        &FIXED_DIMENSIONS
    }

    /// Serialize a vector to bytes with zero-copy optimization for fixed dimensions
    ///
    /// # Format
    /// - Fixed dimensions: Raw f32 bytes (no length prefix)
    /// - Variable dimensions: [length: u32][f32 data...]
    #[inline(always)]
    pub fn serialize(vector: &[f32]) -> Vec<u8> {
        let dim = vector.len();

        if Self::is_fixed_dimension(dim) {
            // FASTEST: Zero-copy cast from f32 slice to u8 slice
            // Safe because f32 has no padding and is repr(C) compatible
            let byte_slice: &[u8] = bytemuck::cast_slice(vector);
            byte_slice.to_vec() // Single memcpy
        } else {
            // Variable dimensions: length prefix + data
            let byte_len = dim * 4; // f32 = 4 bytes
            let mut result = Vec::with_capacity(byte_len + 4); // +4 for length prefix

            // Write length prefix for variable dimensions
            result.extend_from_slice(&(dim as u32).to_le_bytes());

            // Direct memory copy using bytemuck (safe, no unsafe block needed)
            let byte_slice: &[u8] = bytemuck::cast_slice(vector);
            result.extend_from_slice(byte_slice);

            result
        }
    }

    /// Deserialize bytes to a vector with zero-copy optimization
    ///
    /// # Arguments
    /// * `data` - The serialized bytes
    /// * `expected_dim` - If Some, use fixed dimension format. If None, read length prefix.
    ///
    /// # Returns
    /// The deserialized vector, or an error if the data is malformed
    #[inline(always)]
    pub fn deserialize(data: &[u8], expected_dim: Option<usize>) -> Result<Vec<f32>> {
        match expected_dim {
            Some(dim) if Self::is_fixed_dimension(dim) => {
                // FASTEST: Zero-copy cast for fixed dimensions
                let expected_bytes = dim * 4;
                if data.len() != expected_bytes {
                    return Err(anyhow::anyhow!(
                        "Fixed dimension size mismatch: expected {} bytes for dim {}, got {}",
                        expected_bytes,
                        dim,
                        data.len()
                    ));
                }

                // Safe zero-copy cast
                let float_slice: &[f32] = bytemuck::cast_slice(data);
                Ok(float_slice.to_vec()) // Single memcpy
            }
            Some(dim) => {
                // Known dimension but not in fixed list - expect length prefix
                if data.len() < 4 {
                    return Err(anyhow::anyhow!(
                        "Insufficient data for variable dimension vector header"
                    ));
                }

                let stored_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
                if stored_len != dim {
                    return Err(anyhow::anyhow!(
                        "Dimension mismatch: expected {}, got {} in header",
                        dim,
                        stored_len
                    ));
                }

                let expected_bytes = dim * 4 + 4; // +4 for length prefix
                if data.len() != expected_bytes {
                    return Err(anyhow::anyhow!(
                        "Variable dimension size mismatch: expected {} bytes, got {}",
                        expected_bytes,
                        data.len()
                    ));
                }

                let float_slice: &[f32] = bytemuck::cast_slice(&data[4..]);
                Ok(float_slice.to_vec())
            }
            None => {
                // Unknown dimension - try to detect format
                // First check if it could be a length-prefixed format
                if data.len() >= 4 {
                    let potential_len =
                        u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
                    let expected_bytes = potential_len * 4 + 4;

                    if data.len() == expected_bytes && potential_len > 0 && potential_len < 100000 {
                        // Looks like length-prefixed format
                        let float_slice: &[f32] = bytemuck::cast_slice(&data[4..]);
                        return Ok(float_slice.to_vec());
                    }
                }

                // Try as raw bytes (fixed dimension without prefix)
                if data.len().is_multiple_of(4) {
                    let float_slice: &[f32] = bytemuck::cast_slice(data);
                    Ok(float_slice.to_vec())
                } else {
                    Err(anyhow::anyhow!(
                        "Invalid vector data: length {} is not divisible by 4",
                        data.len()
                    ))
                }
            }
        }
    }

    /// Simple deserialization for when you know data is raw f32 bytes (no length prefix)
    ///
    /// This is the fastest path - direct bytemuck cast with minimal validation.
    #[inline(always)]
    pub fn deserialize_raw(data: &[u8]) -> Result<Vec<f32>> {
        if !data.len().is_multiple_of(4) {
            return Err(anyhow::anyhow!(
                "Invalid vector binary data: length {} is not divisible by 4",
                data.len()
            ));
        }

        let float_slice: &[f32] = bytemuck::cast_slice(data);
        Ok(float_slice.to_vec())
    }

    /// Serialize multiple vectors efficiently
    ///
    /// For batch operations, this method serializes vectors with a count header.
    /// Format: [count: u32][dim: u32][vector1 bytes][vector2 bytes]...
    pub fn serialize_batch(vectors: &[Vec<f32>]) -> Result<Vec<u8>> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        let dim = vectors[0].len();
        // Verify all vectors have same dimension
        if !vectors.iter().all(|v| v.len() == dim) {
            return Err(anyhow::anyhow!(
                "All vectors in batch must have same dimension"
            ));
        }

        let count = vectors.len();
        let bytes_per_vector = dim * 4;
        let total_size = 8 + count * bytes_per_vector; // 8 = count(4) + dim(4)

        let mut result = Vec::with_capacity(total_size);
        result.extend_from_slice(&(count as u32).to_le_bytes());
        result.extend_from_slice(&(dim as u32).to_le_bytes());

        for vector in vectors {
            let byte_slice: &[u8] = bytemuck::cast_slice(vector);
            result.extend_from_slice(byte_slice);
        }

        Ok(result)
    }

    /// Deserialize a batch of vectors
    pub fn deserialize_batch(data: &[u8]) -> Result<Vec<Vec<f32>>> {
        if data.len() < 8 {
            return Err(anyhow::anyhow!("Insufficient data for batch header"));
        }

        let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let dim = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;

        let bytes_per_vector = dim * 4;
        let expected_size = 8 + count * bytes_per_vector;

        if data.len() != expected_size {
            return Err(anyhow::anyhow!(
                "Batch size mismatch: expected {} bytes, got {}",
                expected_size,
                data.len()
            ));
        }

        let mut vectors = Vec::with_capacity(count);
        let vector_data = &data[8..];

        for i in 0..count {
            let start = i * bytes_per_vector;
            let end = start + bytes_per_vector;
            let float_slice: &[f32] = bytemuck::cast_slice(&vector_data[start..end]);
            vectors.push(float_slice.to_vec());
        }

        Ok(vectors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_dimension_roundtrip() {
        for dim in VectorSerializer::fixed_dimensions() {
            let vector: Vec<f32> = (0..*dim).map(|i| i as f32 * 0.1).collect();
            let bytes = VectorSerializer::serialize(&vector);

            // Fixed dimensions have no length prefix
            assert_eq!(bytes.len(), dim * 4);

            let restored = VectorSerializer::deserialize(&bytes, Some(*dim)).unwrap();
            assert_eq!(vector, restored);
        }
    }

    #[test]
    fn test_variable_dimension_roundtrip() {
        // Use a dimension not in the fixed list
        let dim = 100;
        assert!(!VectorSerializer::is_fixed_dimension(dim));

        let vector: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
        let bytes = VectorSerializer::serialize(&vector);

        // Variable dimensions have 4-byte length prefix
        assert_eq!(bytes.len(), dim * 4 + 4);

        let restored = VectorSerializer::deserialize(&bytes, Some(dim)).unwrap();
        assert_eq!(vector, restored);
    }

    #[test]
    fn test_auto_detect_format() {
        // Test with length prefix
        let vector: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // dim=5, not in fixed list
        let bytes = VectorSerializer::serialize(&vector);
        let restored = VectorSerializer::deserialize(&bytes, None).unwrap();
        assert_eq!(vector, restored);

        // Test with fixed dimension (no prefix)
        let vector: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let bytes = VectorSerializer::serialize(&vector);
        let restored = VectorSerializer::deserialize(&bytes, None).unwrap();
        assert_eq!(vector, restored);
    }

    #[test]
    fn test_deserialize_raw() {
        let vector: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes: &[u8] = bytemuck::cast_slice(&vector);
        let restored = VectorSerializer::deserialize_raw(bytes).unwrap();
        assert_eq!(vector, restored);
    }

    #[test]
    fn test_batch_serialization() {
        let vectors: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
        ];

        let bytes = VectorSerializer::serialize_batch(&vectors).unwrap();
        let restored = VectorSerializer::deserialize_batch(&bytes).unwrap();

        assert_eq!(vectors, restored);
    }

    #[test]
    fn test_empty_batch() {
        let vectors: Vec<Vec<f32>> = vec![];
        let bytes = VectorSerializer::serialize_batch(&vectors).unwrap();
        assert!(bytes.is_empty());
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let vector: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes = VectorSerializer::serialize(&vector);

        // Try to deserialize with wrong dimension
        let result = VectorSerializer::deserialize(&bytes, Some(128));
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_data_error() {
        // Data not divisible by 4
        let invalid_data = vec![1u8, 2, 3];
        let result = VectorSerializer::deserialize_raw(&invalid_data);
        assert!(result.is_err());
    }
}
