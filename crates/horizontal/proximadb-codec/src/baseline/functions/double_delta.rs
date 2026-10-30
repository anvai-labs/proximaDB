// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Double Delta Encoding - Raw implementation (no headers)
//!
//! Second-order delta encoding: compresses differences of differences.
//! Excellent for data with constant or near-constant rate of change.
//! Best for time series with linear trends or smooth curves.
//! Returns ONLY the compressed data - headers are added by WireFormatManager.

use anyhow::Result;

// ===== Bitpacking delegation to shared helpers =====
//
// All bitpacking operations now use the shared helpers in bitpack.rs
// to avoid code duplication and ensure consistent sign extension behavior.

use super::helpers;
use super::helpers::WirePrim;

// ===== Core wire format encoding functions =====

/// Single double-delta encode core (redundancy pass 4: the i32/i64 native
/// cores and the f32 hybrid (i32 base + i64 deltas) differed only in the
/// base/delta slot widths; golden tests pin all three layouts).
///
/// `W` is the base value domain, `D` the delta domain. i32 data uses
/// `<i32, i32>`, i64 `<i64, i64>`, and the f32 path `<i32, i64>` (i64 deltas
/// prevent overflow when differencing i32 bit patterns).
fn encode_double_delta_wire<W: WirePrim, D: WirePrim>(wire_values: &[W]) -> Result<Vec<u8>> {
    if wire_values.is_empty() {
        return Ok(Vec::new());
    }

    if wire_values.len() == 1 {
        let mut result = Vec::new();
        wire_values[0].write_le(&mut result);
        return Ok(result);
    }

    let mut result = Vec::new();
    let base = wire_values[0];
    base.write_le(&mut result);

    if wire_values.len() == 2 {
        let first_delta = D::from_wide(wire_values[1].widen()).wrap_sub(D::from_wide(base.widen()));
        first_delta.write_le(&mut result);
        result.push(0);
        return Ok(result);
    }

    // Compute first-order deltas (i64 widening prevents overflow for the f32
    // path; wrapping arithmetic for the native-width paths)
    let mut deltas: Vec<D> = Vec::with_capacity(wire_values.len() - 1);
    for i in 1..wire_values.len() {
        let curr = D::from_wide(wire_values[i].widen());
        let prev = D::from_wide(wire_values[i - 1].widen());
        deltas.push(curr.wrap_sub(prev));
    }

    let first_delta = deltas[0];
    first_delta.write_le(&mut result);

    // Compute second-order deltas (double deltas)
    let mut double_deltas: Vec<D> = Vec::with_capacity(deltas.len() - 1);
    for i in 1..deltas.len() {
        double_deltas.push(deltas[i].wrap_sub(deltas[i - 1]));
    }

    if double_deltas.is_empty() {
        result.push(0);
        return Ok(result);
    }

    let max_abs = double_deltas
        .iter()
        .map(|&dd| dd.abs_bit_len())
        .max()
        .unwrap_or(0);

    let bits = if max_abs == 0 {
        1
    } else {
        (max_abs + 1).min(D::BITS)
    };

    result.push(bits);

    let packed = D::pack_signed(&double_deltas, bits)?;
    result.extend(packed);

    Ok(result)
}

// ===== Public API (thin wrappers using generic helpers) =====

/// Encode f32 values using double delta (raw, no headers)
///
/// # Algorithm
/// 1. Compute first-order deltas: delta[i] = value[i] - value[i-1]
/// 2. Compute second-order deltas: double_delta[i] = delta[i] - delta[i-1]
/// 3. Bit-pack the double deltas with minimal bits
///
/// # Format (raw data only, NO headers)
/// ```text
/// [base:4 bytes][first_delta:8 bytes i64][bits:1 byte][bitpacked_double_deltas...]
/// ```text
///
/// # Parameters
/// - `values`: f32 slice to encode
///
/// # Returns
/// Raw encoded bytes (NO scheme marker, NO count header)
pub fn encode_f32(values: &[f32]) -> Result<Vec<u8>> {
    helpers::encode_generic(values, encode_double_delta_wire::<i32, i64>)
}

/// Encode i64 values using double delta (raw, no headers)
pub fn encode_i64(values: &[i64]) -> Result<Vec<u8>> {
    helpers::encode_generic(values, encode_double_delta_wire::<i64, i64>)
}

/// Encode i32 values using double delta (raw, no headers)
pub fn encode_i32(values: &[i32]) -> Result<Vec<u8>> {
    helpers::encode_generic(values, encode_double_delta_wire::<i32, i32>)
}

// ===== Core wire format decoding functions =====

/// Single double-delta decode core; header is
/// [base:W::WIRE_BYTES][first_delta:D::WIRE_BYTES][bits:1].
fn decode_double_delta_wire<W: WirePrim, D: WirePrim>(data: &[u8], count: usize) -> Result<Vec<W>> {
    if count == 0 {
        return Ok(Vec::new());
    }

    if data.len() < W::WIRE_BYTES {
        return Err(anyhow::anyhow!("DoubleDelta decode: insufficient data"));
    }

    let base = W::read_le(data)?;

    if count == 1 {
        return Ok(vec![base]);
    }

    let header_len = W::WIRE_BYTES + D::WIRE_BYTES + 1;
    if data.len() < header_len {
        return Err(anyhow::anyhow!(
            "DoubleDelta decode: insufficient data for first delta"
        ));
    }

    let first_delta = D::read_le(&data[W::WIRE_BYTES..])?;

    // Reconstruct a value from the running delta state (D-domain arithmetic,
    // narrowed back to W exactly like the old per-width cores).
    let value_of = |delta_state: D| W::from_wide(base.widen().wrap_add(delta_state.widen()));

    if count == 2 {
        return Ok(vec![base, value_of(first_delta)]);
    }

    let bits = data[header_len - 1];

    if bits == 0 {
        let mut result = Vec::with_capacity(count);
        result.push(base);

        let mut current = first_delta;
        result.push(value_of(current));

        for _ in 2..count {
            current = current.wrap_add(first_delta);
            result.push(value_of(current));
        }

        return Ok(result);
    }

    let num_double_deltas = count - 2;
    let double_deltas = D::unpack_signed(&data[header_len..], bits, num_double_deltas)?;

    let mut result = Vec::with_capacity(count);
    result.push(base);

    let mut prev_delta = first_delta;

    // Second value in the D domain (full precision — the f32 hybrid keeps
    // accumulating in i64); only the pushed value narrows to W.
    let second_value = D::from_wide(base.widen()).wrap_add(first_delta);
    result.push(W::from_wide(second_value.widen()));
    let mut prev_value = second_value;

    for &dd in &double_deltas {
        let delta = prev_delta.wrap_add(dd);
        let value = prev_value.wrap_add(delta);
        result.push(W::from_wide(value.widen()));
        prev_value = value;
        prev_delta = delta;
    }

    Ok(result)
}

// ===== Public API (thin wrappers using generic helpers) =====

/// Decode f32 values from double delta encoded data
pub fn decode_f32(data: &[u8], count: usize) -> Result<Vec<f32>> {
    helpers::decode_generic::<f32>(data, count, decode_double_delta_wire::<i32, i64>)
}

/// Decode i64 values from double delta encoded data
pub fn decode_i64(data: &[u8], count: usize) -> Result<Vec<i64>> {
    helpers::decode_generic::<i64>(data, count, decode_double_delta_wire::<i64, i64>)
}

/// Decode i32 values from double delta encoded data
pub fn decode_i32(data: &[u8], count: usize) -> Result<Vec<i32>> {
    helpers::decode_generic::<i32>(data, count, decode_double_delta_wire::<i32, i32>)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Golden-byte pins (redundancy pass 4) ====================
    // Wire layout: [base:W bytes][first_delta:D bytes][bits:1 byte]
    // [bitpacked double deltas]. The i32 core uses W=i32/D=i32, the i64 core
    // W=i64/D=i64, and the f32 path W=i32/D=i64 (4-byte base, 8-byte delta).
    // These pin the CURRENT bytes of all three BEFORE the WirePrim
    // consolidation; any change is a wire-format change.

    /// i32 core: [100,110,125] -> base 100, first_delta 10, double deltas
    /// [5]; 3 bits + sign = 4 bits; packed 0b0101 -> 0x05.
    #[test]
    fn golden_double_delta_i32_wire_bytes() {
        let encoded = encode_i32(&[100, 110, 125]).unwrap();
        assert_eq!(encoded, vec![100, 0, 0, 0, 10, 0, 0, 0, 4, 0x05]);
        assert_eq!(decode_i32(&encoded, 3).unwrap(), vec![100, 110, 125]);
    }

    /// i64 core: [1000,1010,1025] -> base 1000 (8B), first_delta 10 (8B),
    /// double deltas [5] @ 4 bits -> 0x05.
    #[test]
    fn golden_double_delta_i64_wire_bytes() {
        let encoded = encode_i64(&[1000, 1010, 1025]).unwrap();
        assert_eq!(
            encoded,
            vec![
                0xE8, 0x03, 0, 0, 0, 0, 0, 0, // base 1000 LE i64
                10, 0, 0, 0, 0, 0, 0, 0, // first_delta 10 LE i64
                4, 0x05, // bits + packed double delta
            ]
        );
        assert_eq!(decode_i64(&encoded, 3).unwrap(), vec![1000, 1010, 1025]);
    }

    /// f32 hybrid (W=i32, D=i64): [1.0, 2.0] pins the MIXED widths — 4-byte
    /// base (0x3F800000) followed by an 8-byte first_delta
    /// (0x40000000-0x3F800000 = 0x00800000) and a zero bits byte.
    #[test]
    fn golden_double_delta_f32_hybrid_widths() {
        let encoded = encode_f32(&[1.0, 2.0]).unwrap();
        assert_eq!(
            encoded,
            vec![
                0x00, 0x00, 0x80, 0x3F, // base = 1.0f32 bits LE
                0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, // first_delta i64
                0x00, // bits = 0 (no double deltas)
            ]
        );
        let decoded = decode_f32(&encoded, 2).unwrap();
        assert_eq!(decoded[0].to_bits(), 1.0f32.to_bits());
        assert_eq!(decoded[1].to_bits(), 2.0f32.to_bits());
    }

    #[test]
    fn test_double_delta_linear() {
        // Linear sequence: constant delta, zero double delta (32+ values)
        let values: Vec<i32> = (0..32).map(|i| 100 + i * 10).collect();

        let encoded = encode_i32(&values).expect("Failed to encode linear values");
        let decoded = decode_i32(&encoded, values.len()).expect("Failed to decode linear values");

        assert_eq!(values, decoded);

        // Should be small: base + first_delta + 1 bit for zero double deltas
        assert!(encoded.len() < 30, "Should compress linear data well");
    }

    #[test]
    fn test_double_delta_quadratic() {
        // Quadratic sequence: linear delta, constant double delta (32+ values)
        // values: 0, 1, 4, 9, 16, 25... (squares)
        // deltas: 1, 3, 5, 7, 9... (odd numbers)
        // double deltas: 2, 2, 2, 2... (constant)
        let values: Vec<i32> = (0..32).map(|i| i * i).collect();

        let encoded = encode_i32(&values).expect("Failed to encode quadratic values");
        let decoded =
            decode_i32(&encoded, values.len()).expect("Failed to decode quadratic values");

        assert_eq!(values, decoded);
    }

    #[test]
    fn test_double_delta_timestamps() {
        // Simulated timestamps with constant rate
        let mut values = Vec::new();
        let mut ts = 1000000i64;
        for _ in 0..100 {
            values.push(ts);
            ts += 1000; // Constant increment
        }

        let encoded = encode_i64(&values).expect("Failed to encode timestamp values");
        let decoded =
            decode_i64(&encoded, values.len()).expect("Failed to decode timestamp values");

        assert_eq!(values, decoded);

        // Should compress very well (constant delta)
        let original_size = values.len() * 8;
        assert!(
            encoded.len() < original_size / 10,
            "Should compress constant-rate timestamps well: {} vs {}",
            encoded.len(),
            original_size
        );
    }

    #[test]
    fn test_double_delta_f32_roundtrip() {
        // Use values with consistent bit-pattern deltas (32+ values)
        let values: Vec<f32> = (0..32).map(|i| 100.0 + i as f32 * 0.5).collect();

        let encoded = encode_f32(&values).expect("Failed to encode f32 values");
        let decoded = decode_f32(&encoded, values.len()).expect("Failed to decode f32 values");

        assert_eq!(values.len(), decoded.len());
        // DoubleDelta on f32 may have minor precision loss due to bit-level delta encoding
        for (orig, dec) in values.iter().zip(decoded.iter()) {
            let diff = (orig - dec).abs();
            assert!(diff < 0.01, "Expected {}, got {}", orig, dec);
        }
    }

    #[test]
    fn test_double_delta_i64_roundtrip() {
        let values: Vec<i64> = (0..32).map(|i| 1000 + i * 10).collect();

        let encoded = encode_i64(&values).expect("Failed to encode i64 values");
        let decoded = decode_i64(&encoded, values.len()).expect("Failed to decode i64 values");

        assert_eq!(values, decoded);
    }

    #[test]
    fn test_double_delta_empty() {
        let values: Vec<i32> = vec![];
        let encoded = encode_i32(&values).expect("Failed to encode empty values");
        assert!(encoded.is_empty());

        let decoded = decode_i32(&encoded, 0).expect("Failed to decode empty values");
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_double_delta_single_value() {
        let values = vec![42i32];

        let encoded = encode_i32(&values).expect("Failed to encode single value");
        let decoded = decode_i32(&encoded, values.len()).expect("Failed to decode single value");

        assert_eq!(values, decoded);
    }

    #[test]
    fn test_double_delta_two_values() {
        let values = vec![10i32, 20];

        let encoded = encode_i32(&values).expect("Failed to encode two values");
        let decoded = decode_i32(&encoded, values.len()).expect("Failed to decode two values");

        assert_eq!(values, decoded);
    }

    #[test]
    fn test_double_delta_random() {
        // Random data - worst case for double delta (32+ values)
        let values: Vec<i32> = vec![
            100, 50, 200, 75, 150, 25, 180, 90, 120, 160, 80, 140, 60, 190, 110, 130, 170, 40, 210,
            95, 125, 155, 85, 145, 65, 185, 105, 135, 165, 45, 195, 115,
        ];

        let encoded = encode_i32(&values).expect("Failed to encode random values");
        let decoded = decode_i32(&encoded, values.len()).expect("Failed to decode random values");

        assert_eq!(values, decoded);
    }

    #[test]
    fn test_double_delta_sensor_data() {
        // Simulated sensor data with slight variations
        let mut values = Vec::new();
        let mut temp = 20.0f32;
        for i in 0..100 {
            temp += 0.1 + (i as f32 * 0.01); // Gradual increase with acceleration
            values.push(temp);
        }

        let encoded = encode_f32(&values).expect("Failed to encode sensor data");
        let decoded = decode_f32(&encoded, values.len()).expect("Failed to decode sensor data");

        assert_eq!(values.len(), decoded.len());
        for (orig, dec) in values.iter().zip(decoded.iter()) {
            let diff = (orig - dec).abs();
            assert!(diff < 0.1, "Expected {}, got {}", orig, dec);
        }
    }

    #[test]
    fn test_double_delta_constant() {
        // All same value - ultimate compression
        let values = vec![42i32; 100];

        let encoded = encode_i32(&values).expect("Failed to encode constant values");
        let decoded = decode_i32(&encoded, values.len()).expect("Failed to decode constant values");

        assert_eq!(values, decoded);

        // Constant data: base (4) + first_delta=0 (4) + bits (1) = 9 bytes minimum
        // Actual may be slightly more due to implementation details
        assert!(encoded.len() < 30, "Encoded {} bytes", encoded.len());
    }

    // ===== OVERFLOW EDGE CASE TESTS =====
    // These tests verify that the i64 delta fix prevents overflow

    #[test]
    fn test_overflow_i32_extremes() {
        // Test with i32::MAX and i32::MIN - would overflow with i32 deltas
        // Delta between i32::MAX and i32::MIN is 2^32, which exceeds i32 range
        let values = vec![i32::MIN, i32::MAX, i32::MIN, i32::MAX];

        let encoded = encode_i32(&values).expect("Failed to encode i32 extreme values");
        let decoded =
            decode_i32(&encoded, values.len()).expect("Failed to decode i32 extreme values");

        assert_eq!(values, decoded, "Failed to roundtrip i32 extremes");
    }

    #[test]
    fn test_overflow_f32_extreme_bit_patterns() {
        // Test f32 values with extreme bit patterns that would cause i32 delta overflow
        let values = vec![
            f32::from_bits(i32::MAX as u32), // Positive extreme
            f32::from_bits(i32::MIN as u32), // Negative extreme
            f32::from_bits(0),               // Zero
            f32::from_bits(i32::MAX as u32), // Back to positive
        ];

        let encoded = encode_f32(&values).expect("Failed to encode f32 extreme bit patterns");
        let decoded =
            decode_f32(&encoded, values.len()).expect("Failed to decode f32 extreme bit patterns");

        assert_eq!(values.len(), decoded.len());
        for (orig, dec) in values.iter().zip(decoded.iter()) {
            assert_eq!(
                orig.to_bits(),
                dec.to_bits(),
                "Failed to roundtrip extreme f32 bit pattern: orig={:08x}, dec={:08x}",
                orig.to_bits(),
                dec.to_bits()
            );
        }
    }

    #[test]
    fn test_overflow_maximum_delta_sequence() {
        // Create a sequence where consecutive deltas are at i32 boundary
        // This would definitely overflow with i32 delta computation
        let values = vec![
            0i32,
            i32::MAX / 2,
            i32::MAX,
            i32::MAX / 2,
            0,
            i32::MIN / 2,
            i32::MIN,
        ];

        let encoded = encode_i32(&values).expect("Failed to encode maximum delta sequence");
        let decoded =
            decode_i32(&encoded, values.len()).expect("Failed to decode maximum delta sequence");

        assert_eq!(values, decoded, "Failed to handle large delta transitions");
    }

    #[test]
    fn test_overflow_f32_infinity_nan() {
        // Test special f32 values with extreme bit patterns
        let values = vec![
            0.0f32,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::MAX,
            f32::MIN,
        ];

        let encoded = encode_f32(&values).expect("Failed to encode f32 special values");
        let decoded =
            decode_f32(&encoded, values.len()).expect("Failed to decode f32 special values");

        assert_eq!(values.len(), decoded.len());
        for (orig, dec) in values.iter().zip(decoded.iter()) {
            // Compare bit patterns since NAN != NAN
            assert_eq!(
                orig.to_bits(),
                dec.to_bits(),
                "Failed to roundtrip special f32 value: orig={:08x}, dec={:08x}",
                orig.to_bits(),
                dec.to_bits()
            );
        }
    }

    #[test]
    fn test_overflow_i64_full_range() {
        // Test i64 values across full range
        let values = vec![i64::MIN, i64::MIN / 2, 0i64, i64::MAX / 2, i64::MAX, 0];

        let encoded = encode_i64(&values).expect("Failed to encode i64 full range values");
        let decoded =
            decode_i64(&encoded, values.len()).expect("Failed to decode i64 full range values");

        assert_eq!(values, decoded, "Failed to handle i64 extremes");
    }

    #[test]
    fn test_overflow_alternating_extremes() {
        // Worst case: alternating between extremes
        // Each delta is maximum possible value
        let values = vec![
            i32::MIN,
            i32::MAX,
            i32::MIN,
            i32::MAX,
            i32::MIN,
            i32::MAX,
            i32::MIN,
            i32::MAX,
        ];

        let encoded = encode_i32(&values).expect("Failed to encode alternating extremes");
        let decoded =
            decode_i32(&encoded, values.len()).expect("Failed to decode alternating extremes");

        assert_eq!(values, decoded, "Failed to handle alternating extremes");
    }
}
