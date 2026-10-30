// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Zigzag Encoding - Raw implementation (no headers)
//!
//! Maps signed integers to unsigned for better compression.
//! Transforms signed values so small absolute values map to small unsigned values.
//! Best for signed integers with small absolute values (e.g., -5 to +5).
//! Returns ONLY the compressed data - headers are added by WireFormatManager.

use anyhow::Result;

use super::helpers;

// ===== Core wire format encoding functions =====

/// Core encoding logic for i32 wire type (used by f32 and i32)
fn encode_zigzag_i32_wire(wire_values: &[i32], bits: u8) -> Result<Vec<u8>> {
    if wire_values.is_empty() {
        return Ok(Vec::new());
    }

    let mut result = Vec::new();
    result.push(bits);

    // Apply zigzag encoding
    let zigzag_values: Vec<u32> = wire_values.iter().map(|&v| zigzag_encode_i32(v)).collect();

    // Bit-pack the zigzag encoded values
    let packed = super::bitpack::bitpack_u32(&zigzag_values, bits)?;
    result.extend(packed);

    Ok(result)
}

/// Core encoding logic for i64 wire type
fn encode_zigzag_i64_wire(wire_values: &[i64], bits: u8) -> Result<Vec<u8>> {
    if wire_values.is_empty() {
        return Ok(Vec::new());
    }

    let mut result = Vec::new();
    result.push(bits);

    // Apply zigzag encoding
    let zigzag_values: Vec<u64> = wire_values.iter().map(|&v| zigzag_encode_i64(v)).collect();

    // Bit-pack the zigzag encoded values
    let packed = super::bitpack::bitpack_u64(&zigzag_values, bits)?;
    result.extend(packed);

    Ok(result)
}

// ===== Public API (thin wrappers using generic helpers) =====

/// Zigzag encode f32 values (raw, no headers)
///
/// # Algorithm
/// 1. Convert f32 to signed integers (interpret bits as i32)
/// 2. Apply zigzag transformation: (n << 1) ^ (n >> 31)
/// 3. Bit-pack the unsigned values
///
/// # Format (raw data only, NO headers)
/// ```text
/// [bits:1 byte][bitpacked_zigzag_values...]
/// ```text
///
/// # Parameters
/// - `values`: f32 slice to encode
/// - `bits`: Bit width for packing (typically determined by max value)
///
/// # Returns
/// Raw encoded bytes (NO scheme marker, NO count header)
pub fn encode_f32(values: &[f32], bits: u8) -> Result<Vec<u8>> {
    helpers::encode_generic(values, |wire_values| {
        encode_zigzag_i32_wire(wire_values, bits)
    })
}

/// Zigzag encode i64 values (raw, no headers)
pub fn encode_i64(values: &[i64], bits: u8) -> Result<Vec<u8>> {
    helpers::encode_generic(values, |wire_values| {
        encode_zigzag_i64_wire(wire_values, bits)
    })
}

/// Zigzag encode i32 values (raw, no headers)
pub fn encode_i32(values: &[i32], bits: u8) -> Result<Vec<u8>> {
    helpers::encode_generic(values, |wire_values| {
        encode_zigzag_i32_wire(wire_values, bits)
    })
}

// ===== Core wire format decoding functions =====

/// Core decoding logic for i32 wire type
fn decode_zigzag_i32_wire(data: &[u8], count: usize) -> Result<Vec<i32>> {
    if data.is_empty() {
        return Err(anyhow::anyhow!("Zigzag decode: insufficient data"));
    }

    // Read bit width
    let bits = data[0];

    // Unpack zigzag values
    let zigzag_values = super::bitpack::unbitpack_u32(&data[1..], bits, count)?;

    // Decode zigzag
    let result = zigzag_values
        .iter()
        .map(|&zz| zigzag_decode_u32(zz))
        .collect();

    Ok(result)
}

/// Core decoding logic for i64 wire type
fn decode_zigzag_i64_wire(data: &[u8], count: usize) -> Result<Vec<i64>> {
    if data.is_empty() {
        return Err(anyhow::anyhow!("Zigzag decode: insufficient data"));
    }

    // Read bit width
    let bits = data[0];

    // Unpack zigzag values
    let zigzag_values = super::bitpack::unbitpack_u64(&data[1..], bits, count)?;

    // Decode zigzag
    let result = zigzag_values
        .iter()
        .map(|&zz| zigzag_decode_u64(zz))
        .collect();

    Ok(result)
}

// ===== Public API (thin wrappers using generic helpers) =====

/// Decode f32 values from zigzag encoded data
pub fn decode_f32(data: &[u8], count: usize) -> Result<Vec<f32>> {
    helpers::decode_generic::<f32>(data, count, decode_zigzag_i32_wire)
}

/// Decode i64 values from zigzag encoded data
pub fn decode_i64(data: &[u8], count: usize) -> Result<Vec<i64>> {
    helpers::decode_generic::<i64>(data, count, decode_zigzag_i64_wire)
}

/// Decode i32 values from zigzag encoded data
pub fn decode_i32(data: &[u8], count: usize) -> Result<Vec<i32>> {
    helpers::decode_generic::<i32>(data, count, decode_zigzag_i32_wire)
}

// ===== Zigzag encoding/decoding helpers =====

/// Zigzag encode i32 to u32
/// Transforms: 0 → 0, -1 → 1, 1 → 2, -2 → 3, 2 → 4, ...
fn zigzag_encode_i32(n: i32) -> u32 {
    ((n << 1) ^ (n >> 31)) as u32
}

/// Zigzag decode u32 to i32
fn zigzag_decode_u32(n: u32) -> i32 {
    ((n >> 1) as i32) ^ (-((n & 1) as i32))
}

/// Zigzag encode i64 to u64
fn zigzag_encode_i64(n: i64) -> u64 {
    ((n << 1) ^ (n >> 63)) as u64
}

/// Zigzag decode u64 to i64
fn zigzag_decode_u64(n: u64) -> i64 {
    ((n >> 1) as i64) ^ (-((n & 1) as i64))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Golden-byte pin (redundancy pass 3) ====================
    // Cross-module witness: zigzag's wire is [bits:1][packed...]. [-1,1] @2
    // bits: zz(-1)=1(01), zz(1)=2(10) -> stream 1,0,0,1 -> 2^0+2^3 = 0x09.
    // Pins that zigzag's private bit-packing stays byte-identical to
    // bitpack's (the copies are scheduled for consolidation).
    #[test]
    fn golden_zigzag_encode_i32_wire_bytes() {
        assert_eq!(encode_i32(&[-1, 1], 2).unwrap(), vec![0x02, 0x09]);
    }

    #[test]
    fn test_zigzag_encoding() {
        // Test zigzag transformation for i32
        assert_eq!(zigzag_encode_i32(0), 0);
        assert_eq!(zigzag_encode_i32(-1), 1);
        assert_eq!(zigzag_encode_i32(1), 2);
        assert_eq!(zigzag_encode_i32(-2), 3);
        assert_eq!(zigzag_encode_i32(2), 4);
        assert_eq!(zigzag_encode_i32(-64), 127);
        assert_eq!(zigzag_encode_i32(64), 128);

        // Test roundtrip
        for n in -1000..1000 {
            assert_eq!(zigzag_decode_u32(zigzag_encode_i32(n)), n);
        }
    }

    #[test]
    fn test_zigzag_i32_roundtrip() -> Result<()> {
        // Small signed values around zero
        let values = vec![-5i32, -2, 0, 1, 3, 5, -10, 8];

        let encoded = encode_i32(&values, 8)?;
        let decoded = decode_i32(&encoded, values.len())?;

        assert_eq!(values, decoded);
        Ok(())
    }

    #[test]
    fn test_zigzag_i64_roundtrip() -> Result<()> {
        // Negative and positive values
        let values = vec![-100i64, -50, -1, 0, 1, 50, 100, -75, 25];

        let encoded = encode_i64(&values, 16)?;
        let decoded = decode_i64(&encoded, values.len())?;

        assert_eq!(values, decoded);
        Ok(())
    }

    #[test]
    fn test_zigzag_f32_roundtrip() -> Result<()> {
        // f32 values (interpreted as bit patterns)
        let values = vec![1.0f32, -1.0, 2.5, -2.5, 0.0, 10.0, -10.0];

        let encoded = encode_f32(&values, 32)?;
        let decoded = decode_f32(&encoded, values.len())?;

        assert_eq!(values.len(), decoded.len());
        for (orig, dec) in values.iter().zip(decoded.iter()) {
            assert_eq!(orig.to_bits(), dec.to_bits());
        }
        Ok(())
    }

    #[test]
    fn test_zigzag_compression() -> Result<()> {
        // Test compression efficiency for small signed values
        let values: Vec<i32> = (-50..50).collect();

        let encoded = encode_i32(&values, 8)?;
        let decoded = decode_i32(&encoded, values.len())?;

        assert_eq!(values, decoded);

        // Should be much smaller than uncompressed
        // 100 values × 4 bytes = 400 bytes original
        // Zigzag with 8 bits: 1 + (100 × 8 / 8) = 101 bytes
        assert!(
            encoded.len() < 150,
            "Compression inefficient: {} bytes",
            encoded.len()
        );
        Ok(())
    }

    #[test]
    fn test_zigzag_all_negative() -> Result<()> {
        // All negative values
        let values = vec![-100i32, -200, -300, -400, -500];

        let encoded = encode_i32(&values, 16)?;
        let decoded = decode_i32(&encoded, values.len())?;

        assert_eq!(values, decoded);
        Ok(())
    }

    #[test]
    fn test_zigzag_all_positive() -> Result<()> {
        // All positive values
        let values = vec![100i32, 200, 300, 400, 500];

        let encoded = encode_i32(&values, 16)?;
        let decoded = decode_i32(&encoded, values.len())?;

        assert_eq!(values, decoded);
        Ok(())
    }

    #[test]
    fn test_zigzag_zeros() -> Result<()> {
        // All zeros - should compress to minimum
        let values = vec![0i32; 100];

        let encoded = encode_i32(&values, 1)?;
        let decoded = decode_i32(&encoded, values.len())?;

        assert_eq!(values, decoded);

        // Should be very small: 1 byte (bits) + 13 bytes (100 bits packed)
        assert!(encoded.len() < 20);
        Ok(())
    }

    #[test]
    fn test_zigzag_empty() -> Result<()> {
        let values: Vec<i32> = vec![];
        let encoded = encode_i32(&values, 8)?;
        assert!(encoded.is_empty());
        Ok(())
    }

    #[test]
    fn test_zigzag_single_value() -> Result<()> {
        let values = vec![-42i32];

        let encoded = encode_i32(&values, 8)?;
        let decoded = decode_i32(&encoded, values.len())?;

        assert_eq!(values, decoded);
        Ok(())
    }

    #[test]
    fn test_zigzag_large_range() -> Result<()> {
        // Values with large positive and negative values
        let values = vec![i32::MIN, i32::MIN / 2, -1, 0, 1, i32::MAX / 2, i32::MAX];

        let encoded = encode_i32(&values, 32)?;
        let decoded = decode_i32(&encoded, values.len())?;

        assert_eq!(values, decoded);
        Ok(())
    }
}
