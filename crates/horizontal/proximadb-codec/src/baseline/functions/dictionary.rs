// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Dictionary Encoding - Raw implementation (no headers)
//!
//! Maps unique values to codes, stores mapping + codes.
//! Excellent for data with low cardinality (many repeated values).
//! Best for categorical data, status codes, repeated strings.
//! Returns ONLY the compressed data - headers are added by WireFormatManager.

use anyhow::Result;
use std::collections::HashMap;

use super::helpers;
use super::helpers::WirePrim;

// ===== Core wire format encoding functions =====

/// Single dictionary core (redundancy pass 4: the i32/i64 wire twins were
/// byte-identical modulo the dictionary value slot width; golden tests pin
/// the layout).
fn encode_dictionary_wire<W: WirePrim>(wire_values: &[W]) -> Result<Vec<u8>> {
    if wire_values.is_empty() {
        return Ok(Vec::new());
    }

    // Build dictionary
    let mut dictionary = Vec::new();
    let mut value_to_code = HashMap::new();

    for &value in wire_values {
        if let std::collections::hash_map::Entry::Vacant(e) = value_to_code.entry(value) {
            let code = dictionary.len() as u32;
            e.insert(code);
            dictionary.push(value);
        }
    }

    let mut result = Vec::new();

    // Store dictionary size
    let num_unique = dictionary.len() as u32;
    result.extend_from_slice(&num_unique.to_le_bytes());

    // Store dictionary values
    for &value in &dictionary {
        value.write_le(&mut result);
    }

    // Store codes (using variable-length encoding for efficiency)
    for &value in wire_values {
        let code = value_to_code[&value];
        encode_varint(&mut result, code as u64);
    }

    Ok(result)
}

// ===== Public API (thin wrappers using generic helpers) =====

/// Encode f32 values using dictionary encoding (raw, no headers)
///
/// # Algorithm
/// 1. Build dictionary of unique values
/// 2. Map each value to a code (index in dictionary)
/// 3. Store dictionary + codes
///
/// # Format (raw data only, NO headers)
/// ```text
/// [num_unique:4 bytes][dictionary: value*][codes: varint*]
/// ```text
///
/// # Parameters
/// - `values`: f32 slice to encode
///
/// # Returns
/// Raw encoded bytes (NO scheme marker, NO count header)
pub fn encode_f32(values: &[f32]) -> Result<Vec<u8>> {
    helpers::encode_generic(values, encode_dictionary_wire::<i32>)
}

/// Encode i64 values using dictionary encoding (raw, no headers)
pub fn encode_i64(values: &[i64]) -> Result<Vec<u8>> {
    helpers::encode_generic(values, encode_dictionary_wire::<i64>)
}

/// Encode i32 values using dictionary encoding (raw, no headers)
pub fn encode_i32(values: &[i32]) -> Result<Vec<u8>> {
    helpers::encode_generic(values, encode_dictionary_wire::<i32>)
}

// ===== Core wire format decoding functions =====

/// Single dictionary decode core; dictionary entries are `W::WIRE_BYTES` each.
fn decode_dictionary_wire<W: WirePrim>(data: &[u8], count: usize) -> Result<Vec<W>> {
    if count == 0 {
        return Ok(Vec::new());
    }

    if data.len() < 4 {
        return Err(anyhow::anyhow!("Dictionary decode: insufficient data"));
    }

    // Read dictionary size
    let num_unique = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

    // Read dictionary values
    let dict_bytes = 4 + num_unique * W::WIRE_BYTES;
    if data.len() < dict_bytes {
        return Err(anyhow::anyhow!(
            "Dictionary decode: insufficient dictionary data"
        ));
    }

    let mut dictionary = Vec::with_capacity(num_unique);
    for i in 0..num_unique {
        let offset = 4 + i * W::WIRE_BYTES;
        dictionary.push(W::read_le(&data[offset..])?);
    }

    // Decode codes
    let mut result = Vec::with_capacity(count);
    let mut offset = dict_bytes;

    for _ in 0..count {
        let (code, bytes_read) = decode_varint(&data[offset..])?;
        if code as usize >= dictionary.len() {
            return Err(anyhow::anyhow!("Dictionary decode: invalid code"));
        }
        result.push(dictionary[code as usize]);
        offset += bytes_read;
    }

    Ok(result)
}

// ===== Public API (thin wrappers using generic helpers) =====

/// Decode f32 values from dictionary encoded data
pub fn decode_f32(data: &[u8], count: usize) -> Result<Vec<f32>> {
    helpers::decode_generic::<f32>(data, count, decode_dictionary_wire::<i32>)
}

/// Decode i64 values from dictionary encoded data
pub fn decode_i64(data: &[u8], count: usize) -> Result<Vec<i64>> {
    helpers::decode_generic::<i64>(data, count, decode_dictionary_wire::<i64>)
}

/// Decode i32 values from dictionary encoded data
pub fn decode_i32(data: &[u8], count: usize) -> Result<Vec<i32>> {
    helpers::decode_generic::<i32>(data, count, decode_dictionary_wire::<i32>)
}

// ===== VarInt helpers =====

fn encode_varint(buf: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        buf.push(((value & 0x7F) | 0x80) as u8);
        value >>= 7;
    }
    buf.push(value as u8);
}

fn decode_varint(data: &[u8]) -> Result<(u64, usize)> {
    let mut value = 0u64;
    let mut shift = 0;
    let mut bytes_read = 0;

    for &byte in data {
        bytes_read += 1;
        value |= ((byte & 0x7F) as u64) << shift;

        if byte & 0x80 == 0 {
            return Ok((value, bytes_read));
        }

        shift += 7;
        if shift >= 64 {
            return Err(anyhow::anyhow!("VarInt decode: overflow"));
        }
    }

    Err(anyhow::anyhow!("VarInt decode: incomplete"))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Golden-byte pins (redundancy pass 4) ====================
    // Wire layout: [num_unique:u32 LE][unique values, first-appearance order,
    // WIRE_BYTES LE each][codes: varint]*. These pin the CURRENT bytes of both
    // widths BEFORE the WirePrim consolidation; any change is a wire-format
    // change.

    /// i32: [30,10,30] -> dict [30,10], codes [0,1,0].
    #[test]
    fn golden_dictionary_i32_wire_bytes() {
        let encoded = encode_i32(&[30, 10, 30]).unwrap();
        assert_eq!(encoded, vec![2, 0, 0, 0, 30, 0, 0, 0, 10, 0, 0, 0, 0, 1, 0]);
        assert_eq!(decode_i32(&encoded, 3).unwrap(), vec![30, 10, 30]);
    }

    /// i64: [1000,7,1000] -> dict [1000,7] in 8-byte slots, codes [0,1,0].
    #[test]
    fn golden_dictionary_i64_wire_bytes() {
        let encoded = encode_i64(&[1000, 7, 1000]).unwrap();
        assert_eq!(
            encoded,
            vec![
                2, 0, 0, 0, // num_unique = 2
                0xE8, 0x03, 0, 0, 0, 0, 0, 0, // 1000 LE i64
                7, 0, 0, 0, 0, 0, 0, 0, // 7 LE i64
                0, 1, 0, // varint codes
            ]
        );
        assert_eq!(decode_i64(&encoded, 3).unwrap(), vec![1000, 7, 1000]);
    }

    #[test]
    fn test_dictionary_repeated_values() {
        // Many repetitions of few unique values
        let mut values = Vec::new();
        values.extend(vec![1i32; 100]);
        values.extend(vec![2i32; 100]);
        values.extend(vec![3i32; 100]);

        let encoded =
            encode_i32(&values).expect("failed to encode i32 values for repeated values test");
        let decoded = decode_i32(&encoded, values.len())
            .expect("failed to decode i32 values for repeated values test");

        assert_eq!(values, decoded);

        // Should compress well: 3 unique values + 300 small codes
        // Original: 1200 bytes, Encoded: ~316 bytes (dict + codes)
        assert!(encoded.len() < 400);
    }

    #[test]
    fn test_dictionary_categorical() {
        // Simulated status codes
        let values = vec![
            200i32, 200, 404, 200, 500, 200, 404, 200, 200, 404, 500, 200, 200, 404, 200, 200,
        ];

        let encoded =
            encode_i32(&values).expect("failed to encode i32 values for categorical test");
        let decoded = decode_i32(&encoded, values.len())
            .expect("failed to decode i32 values for categorical test");

        assert_eq!(values, decoded);

        // 3 unique values (200, 404, 500)
        // Format: 4 bytes (count) + 3*4 bytes (dict) + 16*1 bytes (codes) = 32 bytes
        // Original: 16*4 = 64 bytes
        let original_size = values.len() * 4;
        assert!(encoded.len() <= original_size); // Should not expand much
    }

    #[test]
    fn test_dictionary_f32_roundtrip() {
        let values = vec![
            1.0f32, 2.0, 1.0, 3.0, 2.0, 1.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 3.0,
        ];

        let encoded = encode_f32(&values).expect("failed to encode f32 values for roundtrip test");
        let decoded = decode_f32(&encoded, values.len())
            .expect("failed to decode f32 values for roundtrip test");

        assert_eq!(values.len(), decoded.len());
        for (orig, dec) in values.iter().zip(decoded.iter()) {
            assert_eq!(orig.to_bits(), dec.to_bits());
        }
    }

    #[test]
    fn test_dictionary_i64_roundtrip() {
        let values = vec![
            1000i64, 2000, 1000, 3000, 2000, 1000, 3000, 2000, 1000, 1000, 2000, 3000, 1000, 2000,
            1000, 3000,
        ];

        let encoded = encode_i64(&values).expect("failed to encode i64 values for roundtrip test");
        let decoded = decode_i64(&encoded, values.len())
            .expect("failed to decode i64 values for roundtrip test");

        assert_eq!(values, decoded);
    }

    #[test]
    fn test_dictionary_empty() {
        let values: Vec<i32> = vec![];
        let encoded = encode_i32(&values).expect("failed to encode empty i32 values");
        assert!(encoded.is_empty());

        let decoded = decode_i32(&encoded, 0).expect("failed to decode empty i32 values");
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_dictionary_single_unique() {
        // All same value
        let values = vec![42i32; 1000];

        let encoded =
            encode_i32(&values).expect("failed to encode i32 values for single unique test");
        let decoded = decode_i32(&encoded, values.len())
            .expect("failed to decode i32 values for single unique test");

        assert_eq!(values, decoded);

        // 1 unique value + 1000 zero codes
        // Original: 4000 bytes, Encoded: ~1008 bytes
        assert!(encoded.len() < 1100);
    }

    #[test]
    fn test_dictionary_many_unique() {
        // All unique values - worst case for dictionary
        let values: Vec<i32> = (0..100).collect();

        let encoded =
            encode_i32(&values).expect("failed to encode i32 values for many unique test");
        let decoded = decode_i32(&encoded, values.len())
            .expect("failed to decode i32 values for many unique test");

        assert_eq!(values, decoded);

        // Dictionary overhead makes this larger than original
        // 100 unique values × 4 bytes = 400 bytes dict
        // 100 codes × ~1 byte = 100 bytes codes
        // Total: ~504 bytes vs 400 bytes original
    }

    #[test]
    fn test_dictionary_compression_ratio() {
        // Low cardinality - ideal for dictionary
        let mut values = Vec::new();
        for _ in 0..1000 {
            values.push(1i32);
            values.push(2i32);
            values.push(3i32);
            values.push(4i32);
            values.push(5i32);
        }

        let encoded =
            encode_i32(&values).expect("failed to encode i32 values for compression ratio test");
        let decoded = decode_i32(&encoded, values.len())
            .expect("failed to decode i32 values for compression ratio test");

        assert_eq!(values, decoded);

        // 5 unique values + 5000 small codes
        // Original: 20000 bytes
        // Encoded: ~5024 bytes (5 × 4 + 5000 × 1)
        let original_size = values.len() * 4;
        let compression_ratio = original_size as f64 / encoded.len() as f64;
        assert!(compression_ratio > 3.5);
    }

    #[test]
    fn test_dictionary_user_ratings() {
        // Simulated 1-5 star ratings
        let values = vec![
            5i32, 4, 5, 3, 4, 5, 5, 4, 3, 5, 4, 5, 2, 4, 5, 5, 4, 5, 3, 4, 5, 5, 4, 5, 1, 4, 5, 5,
            4, 5,
        ];

        let encoded =
            encode_i32(&values).expect("failed to encode i32 values for user ratings test");
        let decoded = decode_i32(&encoded, values.len())
            .expect("failed to decode i32 values for user ratings test");

        assert_eq!(values, decoded);

        // 5 unique ratings, should compress well
        let original_size = values.len() * 4;
        assert!(encoded.len() < original_size / 2);
    }

    #[test]
    fn test_dictionary_alternating() {
        // Alternating pattern
        let mut values = Vec::new();
        for i in 0..100 {
            values.push(if i % 2 == 0 { 1i32 } else { 2i32 });
        }

        let encoded =
            encode_i32(&values).expect("failed to encode i32 values for alternating pattern test");
        let decoded = decode_i32(&encoded, values.len())
            .expect("failed to decode i32 values for alternating pattern test");

        assert_eq!(values, decoded);

        // 2 unique values, excellent compression
        let original_size = values.len() * 4;
        let compression_ratio = original_size as f64 / encoded.len() as f64;
        assert!(compression_ratio > 3.0);
    }
}
