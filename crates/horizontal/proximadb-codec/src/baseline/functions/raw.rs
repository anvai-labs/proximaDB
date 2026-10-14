// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Raw (Identity) Encoding - No transformation
//!
//! Simply serializes data to bytes without any encoding transformation.
//! Best for high-entropy data (e.g., normalized embeddings) where integer
//! transformations cause expansion.
//!
//! Compression is applied separately at higher layers (GroupFieldEncoded/TransposeFieldEncoded).

use anyhow::Result;

/// Encode f32 values as raw bytes (no transformation)
///
/// # Arguments
/// * `values` - f32 values to serialize
///
/// # Returns
/// Raw byte serialization (4 bytes per f32)
pub fn encode_f32(values: &[f32]) -> Result<Vec<u8>> {
    if values.is_empty() {
        return Ok(Vec::new());
    }

    // Serialize f32 → bytes (4 bytes per value)
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for &val in values {
        bytes.extend_from_slice(&val.to_le_bytes());
    }

    Ok(bytes)
}

/// Decode f32 values from raw bytes
///
/// # Arguments
/// * `data` - Raw bytes (4 bytes per f32)
///
/// # Returns
/// Decoded f32 values
pub fn decode_f32(data: &[u8]) -> Result<Vec<f32>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    if !data.len().is_multiple_of(4) {
        return Err(anyhow::anyhow!(
            "Invalid raw bytes length for f32: {} (must be multiple of 4)",
            data.len()
        ));
    }

    // Deserialize bytes → f32
    let mut values = Vec::with_capacity(data.len() / 4);
    for chunk in data.as_chunks::<4>().0 {
        values.push(f32::from_le_bytes(*chunk));
    }

    Ok(values)
}

/// Encode f64 values as raw bytes (no transformation)
pub fn encode_f64(values: &[f64]) -> Result<Vec<u8>> {
    if values.is_empty() {
        return Ok(Vec::new());
    }

    let mut bytes = Vec::with_capacity(values.len() * 8);
    for &val in values {
        bytes.extend_from_slice(&val.to_le_bytes());
    }

    Ok(bytes)
}

/// Decode f64 values from raw bytes
pub fn decode_f64(data: &[u8]) -> Result<Vec<f64>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    if !data.len().is_multiple_of(8) {
        return Err(anyhow::anyhow!(
            "Invalid raw bytes length for f64: {} (must be multiple of 8)",
            data.len()
        ));
    }

    let mut values = Vec::with_capacity(data.len() / 8);
    for chunk in data.as_chunks::<8>().0 {
        values.push(f64::from_le_bytes(*chunk));
    }

    Ok(values)
}

/// Encode i64 values as raw bytes (no transformation)
pub fn encode_i64(values: &[i64]) -> Result<Vec<u8>> {
    if values.is_empty() {
        return Ok(Vec::new());
    }

    let mut bytes = Vec::with_capacity(values.len() * 8);
    for &val in values {
        bytes.extend_from_slice(&val.to_le_bytes());
    }

    Ok(bytes)
}

/// Decode i64 values from raw bytes
pub fn decode_i64(data: &[u8]) -> Result<Vec<i64>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    if !data.len().is_multiple_of(8) {
        return Err(anyhow::anyhow!(
            "Invalid raw bytes length for i64: {} (must be multiple of 8)",
            data.len()
        ));
    }

    let mut values = Vec::with_capacity(data.len() / 8);
    for chunk in data.as_chunks::<8>().0 {
        values.push(i64::from_le_bytes(*chunk));
    }

    Ok(values)
}

/// Encode i32 values as raw bytes (no transformation)
pub fn encode_i32(values: &[i32]) -> Result<Vec<u8>> {
    if values.is_empty() {
        return Ok(Vec::new());
    }

    let mut bytes = Vec::with_capacity(values.len() * 4);
    for &val in values {
        bytes.extend_from_slice(&val.to_le_bytes());
    }

    Ok(bytes)
}

/// Decode i32 values from raw bytes
pub fn decode_i32(data: &[u8]) -> Result<Vec<i32>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    if !data.len().is_multiple_of(4) {
        return Err(anyhow::anyhow!(
            "Invalid raw bytes length for i32: {} (must be multiple of 4)",
            data.len()
        ));
    }

    let mut values = Vec::with_capacity(data.len() / 4);
    for chunk in data.as_chunks::<4>().0 {
        values.push(i32::from_le_bytes(*chunk));
    }

    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raw_f32_roundtrip() {
        let values = vec![0.1, 0.2, 0.3, -0.5, 1.0];
        let encoded = encode_f32(&values).unwrap();
        let decoded = decode_f32(&encoded).unwrap();

        assert_eq!(values, decoded);
        assert_eq!(encoded.len(), values.len() * 4); // No compression
    }

    #[test]
    fn test_raw_normalized_embeddings() {
        // Normalized embeddings in [-1, 1]
        let values: Vec<f32> = (0..256).map(|i| ((i % 200) as f32 / 100.0) - 1.0).collect();

        let encoded = encode_f32(&values).unwrap();
        let decoded = decode_f32(&encoded).unwrap();

        assert_eq!(values, decoded);
        assert_eq!(encoded.len(), values.len() * 4); // Identity encoding
    }

    #[test]
    fn test_raw_i64_roundtrip() {
        let values = vec![1i64, -2, 3, -100, 999];
        let encoded = encode_i64(&values).unwrap();
        let decoded = decode_i64(&encoded).unwrap();

        assert_eq!(values, decoded);
        assert_eq!(encoded.len(), values.len() * 8);
    }

    #[test]
    fn test_raw_f64_roundtrip() {
        let values = vec![0.1f64, 0.2, -0.3, f64::INFINITY, f64::NEG_INFINITY];
        let encoded = encode_f64(&values).unwrap();
        let decoded = decode_f64(&encoded).unwrap();

        assert_eq!(values.len(), decoded.len());
        for (orig, dec) in values.iter().zip(decoded.iter()) {
            assert_eq!(orig.to_bits(), dec.to_bits());
        }
        assert_eq!(encoded.len(), values.len() * 8);
    }

    #[test]
    fn test_raw_i32_roundtrip() {
        let values = vec![1i32, -2, 3, -100, 999];
        let encoded = encode_i32(&values).unwrap();
        let decoded = decode_i32(&encoded).unwrap();

        assert_eq!(values, decoded);
        assert_eq!(encoded.len(), values.len() * 4);
    }
}
