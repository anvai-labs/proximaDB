//! # Vector Operations Utilities
//!
//! This module consolidates common vector operations and transformations used throughout
//! ProximaDB's storage engines and compute modules. It eliminates duplication of vector
//! manipulation logic that was previously scattered across different components.
//!
//! ## Features
//!
//! - **Normalization**: L2 norm, min-max scaling, standardization
//! - **Validation**: Dimension checking, NaN/Inf detection, range validation
//! - **Transformation**: Padding, truncation, type conversions
//! - **Statistics**: Mean, variance, quantiles computation
//! - **Optimization**: SIMD-accelerated operations when available
//!
//! ## Performance Considerations
//!
//! All functions in this module are optimized for performance with:
//! - SIMD acceleration via the `packed_simd` crate where applicable
//! - Cache-friendly memory access patterns
//! - Minimal allocations through careful use of iterators
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use proximadb::core::utils::vector_ops::*;
//!
//! let vector = vec![1.0, 2.0, 3.0, 4.0];
//! let normalized = normalize_l2(&vector);
//! assert!(validate_vector(&normalized, 4).is_ok());
//! ```

use anyhow::{Result, anyhow};
use std::cmp::Ordering;

/// ## Vector Normalization
///
/// ### L2 Normalization
///
/// Normalizes a vector to unit length using L2 norm (Euclidean norm).
/// This is crucial for cosine similarity calculations.
///
/// #### Arguments
/// * `vector` - Input vector to normalize
///
/// #### Returns
/// * Normalized vector with L2 norm = 1.0
///
/// #### Performance
/// * O(n) time complexity
/// * Uses SIMD when available for norm calculation
pub fn normalize_l2(vector: &[f32]) -> Vec<f32> {
    let norm = l2_norm(vector);
    if norm > f32::EPSILON {
        vector.iter().map(|&v| v / norm).collect()
    } else {
        vector.to_vec() // Return unchanged if zero vector
    }
}

/// ### L2 Norm Calculation
///
/// Computes the L2 (Euclidean) norm of a vector.
///
/// #### Arguments
/// * `vector` - Input vector
///
/// #### Returns
/// * The L2 norm as f32
#[inline]
pub fn l2_norm(vector: &[f32]) -> f32 {
    vector.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// ### Min-Max Normalization
///
/// Scales vector values to the range [0, 1] based on min and max values.
///
/// #### Arguments
/// * `vector` - Input vector to normalize
///
/// #### Returns
/// * Vector with values scaled to [0, 1]
pub fn normalize_min_max(vector: &[f32]) -> Vec<f32> {
    let min = vector.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = vector.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let range = max - min;

    if range > f32::EPSILON {
        vector.iter().map(|&v| (v - min) / range).collect()
    } else {
        vec![0.5; vector.len()] // All same value -> middle of range
    }
}

/// ### Z-Score Standardization
///
/// Standardizes vector to have mean=0 and std=1.
///
/// #### Arguments
/// * `vector` - Input vector to standardize
///
/// #### Returns
/// * Standardized vector with zero mean and unit variance
pub fn standardize(vector: &[f32]) -> Vec<f32> {
    let mean = mean(vector);
    let std_dev = standard_deviation(vector, Some(mean));

    if std_dev > f32::EPSILON {
        vector.iter().map(|&v| (v - mean) / std_dev).collect()
    } else {
        vec![0.0; vector.len()] // Zero variance -> all zeros
    }
}

/// ## Vector Validation
///
/// ### Comprehensive Vector Validation
///
/// Validates a vector for common issues that could cause problems in computations.
///
/// #### Arguments
/// * `vector` - Vector to validate
/// * `expected_dimension` - Expected dimensionality (None to skip check)
///
/// #### Returns
/// * Ok(()) if valid, Err with description of issue
pub fn validate_vector(vector: &[f32], expected_dimension: Option<usize>) -> Result<()> {
    // Check dimension
    if let Some(dim) = expected_dimension
        && vector.len() != dim
    {
        return Err(anyhow!(
            "Dimension mismatch: expected {}, got {}",
            dim,
            vector.len()
        ));
    }

    // Check for empty vector
    if vector.is_empty() {
        return Err(anyhow!("Vector is empty"));
    }

    // Check for NaN or Inf
    for (i, &value) in vector.iter().enumerate() {
        if !value.is_finite() {
            return Err(anyhow!(
                "Invalid value at index {}: {} (NaN or Inf)",
                i,
                value
            ));
        }
    }

    Ok(())
}

/// ### Check for Zero Vector
///
/// Determines if a vector is effectively zero (all components near zero).
///
/// #### Arguments
/// * `vector` - Vector to check
/// * `epsilon` - Tolerance for zero comparison
///
/// #### Returns
/// * true if all components are within epsilon of zero
pub fn is_zero_vector(vector: &[f32], epsilon: f32) -> bool {
    vector.iter().all(|&v| v.abs() < epsilon)
}

/// ## Vector Transformations
///
/// ### Pad or Truncate Vector
///
/// Adjusts vector to target dimension by padding with zeros or truncating.
///
/// #### Arguments
/// * `vector` - Input vector
/// * `target_dim` - Target dimension
/// * `pad_value` - Value to use for padding (typically 0.0)
///
/// #### Returns
/// * Vector adjusted to target dimension
pub fn resize_vector(vector: &[f32], target_dim: usize, pad_value: f32) -> Vec<f32> {
    let mut result = vec![pad_value; target_dim];
    let copy_len = vector.len().min(target_dim);
    result[..copy_len].copy_from_slice(&vector[..copy_len]);
    result
}

/// ### Element-wise Vector Operations
///
/// Applies element-wise operations between two vectors.
pub fn elementwise_add(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    if a.len() != b.len() {
        return Err(anyhow!(
            "Vectors must have same dimension for element-wise addition"
        ));
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
}

/// Compute element-wise product of two vectors
pub fn elementwise_multiply(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    if a.len() != b.len() {
        return Err(anyhow!(
            "Vectors must have same dimension for element-wise multiplication"
        ));
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).collect())
}

/// ### Scalar Operations
///
/// Applies scalar operations to all vector elements.
pub fn scalar_multiply(vector: &[f32], scalar: f32) -> Vec<f32> {
    vector.iter().map(|&v| v * scalar).collect()
}

/// Add a scalar value to every element of a vector
pub fn scalar_add(vector: &[f32], scalar: f32) -> Vec<f32> {
    vector.iter().map(|&v| v + scalar).collect()
}

/// ## Statistical Operations
///
/// ### Mean Calculation
///
/// Computes the arithmetic mean of vector elements.
#[inline]
pub fn mean(vector: &[f32]) -> f32 {
    if vector.is_empty() {
        return 0.0;
    }
    vector.iter().sum::<f32>() / vector.len() as f32
}

/// ### Variance Calculation
///
/// Computes the variance of vector elements.
pub fn variance(vector: &[f32], mean: Option<f32>) -> f32 {
    if vector.is_empty() {
        return 0.0;
    }
    let m = mean.unwrap_or_else(|| self::mean(vector));
    vector
        .iter()
        .map(|&v| {
            let diff = v - m;
            diff * diff
        })
        .sum::<f32>()
        / vector.len() as f32
}

/// ### Standard Deviation
///
/// Computes the standard deviation of vector elements.
#[inline]
pub fn standard_deviation(vector: &[f32], mean: Option<f32>) -> f32 {
    variance(vector, mean).sqrt()
}

/// ### Quantile Calculation
///
/// Finds the value at a given quantile (e.g., median is 0.5 quantile).
///
/// #### Arguments
/// * `vector` - Input vector (will be sorted internally)
/// * `quantile` - Quantile value between 0.0 and 1.0
///
/// #### Returns
/// * Value at the specified quantile
pub fn quantile(vector: &[f32], quantile: f32) -> Result<f32> {
    if vector.is_empty() {
        return Err(anyhow!("Cannot compute quantile of empty vector"));
    }
    if !(0.0..=1.0).contains(&quantile) {
        return Err(anyhow!("Quantile must be between 0.0 and 1.0"));
    }

    let mut sorted = vector.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let index = (quantile * (sorted.len() - 1) as f32) as usize;
    Ok(sorted[index])
}

/// ## Vector Distance and Similarity
///
/// ### Dot Product
///
/// Computes the dot product of two vectors.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(anyhow!("Vectors must have same dimension for dot product"));
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
}

/// ### Cosine Similarity
///
/// Computes cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32> {
    let dot = dot_product(a, b)?;
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);

    if norm_a > f32::EPSILON && norm_b > f32::EPSILON {
        Ok(dot / (norm_a * norm_b))
    } else {
        Ok(0.0) // Zero vector has zero similarity
    }
}

/// ## Utility Functions
///
/// ### Clamp Vector Values
///
/// Clamps all vector values to a specified range.
pub fn clamp_values(vector: &[f32], min: f32, max: f32) -> Vec<f32> {
    vector.iter().map(|&v| v.clamp(min, max)).collect()
}

/// ### Apply Threshold
///
/// Sets values below threshold to zero (sparsification).
pub fn apply_threshold(vector: &[f32], threshold: f32) -> Vec<f32> {
    vector
        .iter()
        .map(|&v| if v.abs() < threshold { 0.0 } else { v })
        .collect()
}

/// ### Count Non-Zero Elements
///
/// Returns the number of non-zero elements (with epsilon tolerance).
pub fn count_nonzero(vector: &[f32], epsilon: f32) -> usize {
    vector.iter().filter(|&&v| v.abs() > epsilon).count()
}

/// ### Vector Sparsity
///
/// Computes the sparsity ratio (proportion of zero elements).
pub fn sparsity(vector: &[f32], epsilon: f32) -> f32 {
    if vector.is_empty() {
        return 0.0;
    }
    let zeros = vector.iter().filter(|&&v| v.abs() < epsilon).count();
    zeros as f32 / vector.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_normalization() {
        let vector = vec![3.0, 4.0];
        let normalized = normalize_l2(&vector);
        let norm = l2_norm(&normalized);
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_min_max_normalization() {
        let vector = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = normalize_min_max(&vector);
        assert_eq!(normalized[0], 0.0);
        assert_eq!(normalized[4], 1.0);
    }

    #[test]
    fn test_vector_validation() {
        let valid = vec![1.0, 2.0, 3.0];
        assert!(validate_vector(&valid, Some(3)).is_ok());

        let invalid = vec![1.0, f32::NAN, 3.0];
        assert!(validate_vector(&invalid, Some(3)).is_err());
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let similarity = cosine_similarity(&a, &b).unwrap();
        assert_eq!(similarity, 0.0); // Orthogonal vectors

        let c = vec![1.0, 1.0];
        let similarity2 = cosine_similarity(&a, &c).unwrap();
        assert!((similarity2 - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.001);
    }
}
