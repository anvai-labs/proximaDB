// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Sparse Bitmap encoding - Raw implementation (no headers)
//!
//! For sparse data, stores only non-zero indices and values.
//! Best for vectors with many zeros (>90% sparsity).
//! Returns ONLY the compressed data - headers are added by WireFormatManager.

use anyhow::Result;

use super::helpers;
use super::helpers::WirePrim;

// ===== Core wire format encoding functions =====

/// Single sparse-bitmap core (redundancy pass 4: the i32/i64 wire twins were
/// byte-identical modulo the value slot width; golden tests pin the layout).
fn encode_sparse_bitmap_wire<W: WirePrim>(wire_values: &[W]) -> Result<Vec<u8>> {
    if wire_values.is_empty() {
        return Ok(Vec::new());
    }

    // Find non-zero indices
    let mut nonzero_indices = Vec::new();
    let mut nonzero_values = Vec::new();

    for (idx, &val) in wire_values.iter().enumerate() {
        if !val.is_zero() {
            nonzero_indices.push(idx);
            nonzero_values.push(val);
        }
    }

    let mut result = Vec::new();

    // Store count of non-zero values
    let num_nonzero = nonzero_indices.len() as u32;
    result.extend_from_slice(&num_nonzero.to_le_bytes());

    // Create bitmap
    let bitmap_bytes = wire_values.len().div_ceil(8);
    let mut bitmap = vec![0u8; bitmap_bytes];

    for &idx in &nonzero_indices {
        let byte_idx = idx / 8;
        let bit_idx = idx % 8;
        if byte_idx < bitmap.len() {
            bitmap[byte_idx] |= 1u8 << bit_idx;
        }
    }

    result.extend(&bitmap);

    // Store non-zero values
    for &val in &nonzero_values {
        val.write_le(&mut result);
    }

    Ok(result)
}

// ===== Public API (thin wrappers using generic helpers) =====

/// Encode f32 sparse values using bitmap format (raw, no headers)
///
/// # Algorithm
/// 1. Identify non-zero indices
/// 2. Create bitmap of non-zero positions
/// 3. Store non-zero values sequentially
///
/// # Format (raw data only, NO headers)
/// ```text
/// [num_nonzero:4 bytes][bitmap_bytes...][nonzero_values...]
/// ```text
///
/// # Parameters
/// - `values`: f32 slice to encode (may be sparse)
///
/// # Returns
/// Raw encoded bytes (NO scheme marker, NO count header)
pub fn encode_f32(values: &[f32]) -> Result<Vec<u8>> {
    helpers::encode_generic(values, encode_sparse_bitmap_wire::<i32>)
}

/// Encode i64 sparse values using bitmap format (raw, no headers)
pub fn encode_i64(values: &[i64]) -> Result<Vec<u8>> {
    helpers::encode_generic(values, encode_sparse_bitmap_wire::<i64>)
}

/// Encode i32 sparse values using bitmap format (raw, no headers)
pub fn encode_i32(values: &[i32]) -> Result<Vec<u8>> {
    helpers::encode_generic(values, encode_sparse_bitmap_wire::<i32>)
}

// ===== Core wire format decoding functions =====

/// Single bitmap decode core; non-zero values are `W::WIRE_BYTES` each.
fn decode_sparse_bitmap_wire<W: WirePrim>(data: &[u8], count: usize) -> Result<Vec<W>> {
    if count == 0 {
        return Ok(Vec::new());
    }

    if data.len() < 4 {
        return Err(anyhow::anyhow!("SparseBitmap decode: insufficient data"));
    }

    // Read number of non-zero values
    let num_nonzero = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

    // Calculate bitmap size
    let bitmap_bytes = count.div_ceil(8);

    if data.len() < 4 + bitmap_bytes {
        return Err(anyhow::anyhow!(
            "SparseBitmap decode: insufficient bitmap data"
        ));
    }

    let bitmap = &data[4..4 + bitmap_bytes];
    let values_start = 4 + bitmap_bytes;

    if data.len() < values_start + num_nonzero * W::WIRE_BYTES {
        return Err(anyhow::anyhow!(
            "SparseBitmap decode: insufficient value data"
        ));
    }

    // Initialize result with zeros
    let mut result = vec![W::default(); count];

    // Read non-zero values
    let mut value_idx = 0;
    for (idx, res_val) in result.iter_mut().enumerate() {
        let byte_idx = idx / 8;
        let bit_idx = idx % 8;

        if byte_idx < bitmap.len()
            && (bitmap[byte_idx] & (1u8 << bit_idx)) != 0
            && value_idx < num_nonzero
        {
            let offset = values_start + value_idx * W::WIRE_BYTES;
            *res_val = W::read_le(&data[offset..])?;
            value_idx += 1;
        }
    }

    Ok(result)
}

// ===== Public API (thin wrappers using generic helpers) =====

/// Decode f32 values from sparse bitmap encoded data
pub fn decode_f32(data: &[u8], count: usize) -> Result<Vec<f32>> {
    helpers::decode_generic::<f32>(data, count, decode_sparse_bitmap_wire::<i32>)
}

/// Decode i64 values from sparse bitmap encoded data
pub fn decode_i64(data: &[u8], count: usize) -> Result<Vec<i64>> {
    helpers::decode_generic::<i64>(data, count, decode_sparse_bitmap_wire::<i64>)
}

/// Decode i32 values from sparse bitmap encoded data
pub fn decode_i32(data: &[u8], count: usize) -> Result<Vec<i32>> {
    helpers::decode_generic::<i32>(data, count, decode_sparse_bitmap_wire::<i32>)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Golden-byte pins (redundancy pass 4) ====================
    // Wire layout: [num_nonzero:u32 LE][bitmap: count.div_ceil(8) bytes, LSB
    // first][non-zero values: WIRE_BYTES LE each, in index order]. These pin
    // the CURRENT bytes of both widths BEFORE the WirePrim consolidation; any
    // change is a wire-format change.

    /// i32: count=10 with non-zeros at 1 (=3) and 8 (=7): bitmap bits 1 and 8
    /// -> bytes 0x02, 0x01.
    #[test]
    fn golden_sparse_bitmap_i32_wire_bytes() {
        let mut values = vec![0i32; 10];
        values[1] = 3;
        values[8] = 7;

        let encoded = encode_i32(&values).unwrap();
        assert_eq!(
            encoded,
            vec![
                2, 0, 0, 0, // num_nonzero = 2
                0x02, 0x01, // bitmap: bit1 | bit8
                3, 0, 0, 0, // value 3
                7, 0, 0, 0, // value 7
            ]
        );
        assert_eq!(decode_i32(&encoded, 10).unwrap(), values);
    }

    /// i64: count=9 with a single non-zero 6 at index 0: bitmap 0x01,0x00.
    #[test]
    fn golden_sparse_bitmap_i64_wire_bytes() {
        let mut values = vec![0i64; 9];
        values[0] = 6;

        let encoded = encode_i64(&values).unwrap();
        assert_eq!(
            encoded,
            vec![
                1, 0, 0, 0, // num_nonzero = 1
                0x01, 0x00, // bitmap: bit0
                6, 0, 0, 0, 0, 0, 0, 0, // value 6 LE i64
            ]
        );
        assert_eq!(decode_i64(&encoded, 9).unwrap(), values);
    }

    #[test]
    fn test_sparse_bitmap_f32_roundtrip() {
        // 90% sparse vector
        let mut values = vec![0.0f32; 100];
        values[10] = 1.5;
        values[25] = 2.5;
        values[50] = 3.5;
        values[75] = 4.5;
        values[99] = 5.5;

        let encoded = encode_f32(&values).expect("Failed to encode f32 values");
        let decoded = decode_f32(&encoded, values.len()).expect("Failed to decode f32 values");

        assert_eq!(values, decoded);
    }

    #[test]
    fn test_sparse_bitmap_i64_roundtrip() {
        // Very sparse vector
        let mut values = vec![0i64; 1000];
        values[100] = 42;
        values[500] = 100;
        values[999] = 200;

        let encoded = encode_i64(&values).expect("Failed to encode i64 values");
        let decoded = decode_i64(&encoded, values.len()).expect("Failed to decode i64 values");

        assert_eq!(values, decoded);
    }

    #[test]
    fn test_sparse_bitmap_compression() {
        // Highly sparse vector (99% zeros)
        let mut values = vec![0i32; 10000];
        for i in 0..10 {
            values[i * 1000] = (i as i32 + 1) * 10;
        }

        let encoded = encode_i32(&values).expect("Failed to encode i32 values");
        let decoded = decode_i32(&encoded, values.len()).expect("Failed to decode i32 values");

        assert_eq!(values, decoded);

        // Verify compression
        // Original: 10000 × 4 bytes = 40000 bytes
        // Encoded: 4 (count) + 1250 (bitmap) + 10×4 (values) = ~1294 bytes
        let original_bytes = values.len() * 4;
        assert!(
            encoded.len() < original_bytes / 10,
            "Should compress sparse data: {} < {}",
            encoded.len(),
            original_bytes / 10
        );
    }

    #[test]
    fn test_sparse_bitmap_all_zeros() {
        let values = vec![0i64; 100];

        let encoded = encode_i64(&values).expect("Failed to encode i64 values");
        let decoded = decode_i64(&encoded, values.len()).expect("Failed to decode i64 values");

        assert_eq!(values, decoded);

        // Should be very small (just count + bitmap of zeros)
        assert!(encoded.len() < 50);
    }

    #[test]
    fn test_sparse_bitmap_dense() {
        // Not sparse at all
        let values: Vec<i32> = (1..=100).collect();

        let encoded = encode_i32(&values).expect("Failed to encode i32 values");
        let decoded = decode_i32(&encoded, values.len()).expect("Failed to decode i32 values");

        assert_eq!(values, decoded);

        // Dense data won't compress well
        let original_bytes = values.len() * 4;
        // Will be larger due to bitmap overhead
        assert!(encoded.len() >= original_bytes);
    }

    #[test]
    fn test_sparse_bitmap_empty() {
        let values: Vec<f32> = vec![];

        let encoded = encode_f32(&values).expect("Failed to encode f32 values");
        assert!(encoded.is_empty());

        let decoded = decode_f32(&encoded, 0).expect("Failed to decode f32 values");
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_sparse_bitmap_single_nonzero() {
        let mut values = vec![0.0f32; 1000];
        values[500] = 42.0;

        let encoded = encode_f32(&values).expect("Failed to encode f32 values");
        let decoded = decode_f32(&encoded, values.len()).expect("Failed to decode f32 values");

        assert_eq!(values, decoded);

        // Very small: count + bitmap + 1 value
        assert!(encoded.len() < 200);
    }
}
