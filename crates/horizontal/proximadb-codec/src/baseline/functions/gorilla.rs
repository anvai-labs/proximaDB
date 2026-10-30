// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Gorilla Compression - Raw implementation (no headers)
//!
//! Facebook's Gorilla compression using XOR-based encoding for floats.
//! Excellent for time-series floats where consecutive values are similar.
//! Stores only the XOR difference between consecutive values.
//! Returns ONLY the compressed data - headers are added by WireFormatManager.

use anyhow::Result;

use super::helpers;

/// Unsigned XOR primitive for the Gorilla bit stream (u32 for the f32/i32
/// wire, u64 for f64/i64).
///
/// Redundancy pass 4 (PackPrim/WirePrim precedent): the two wire cores were
/// byte-identical modulo the value width and the per-width control-field
/// sizes; the golden tests pin both layouts. The trait stays local to this
/// codec because only Gorilla works in the unsigned XOR domain.
trait GorillaPrim: Copy + std::ops::BitAnd<Output = Self> {
    /// Wire slot size in bytes (4 for u32, 8 for u64).
    const WIRE_BYTES: usize;
    /// Total value bits (32 for u32, 64 for u64).
    const BITS: u8;
    /// Wire field width for the leading-zero count (5 for u32, 6 for u64).
    const LEADING_BITS: u8;
    /// Wire field width for the meaningful-bits count (6 for u32, 7 for u64).
    const MEANINGFUL_BITS: u8;

    /// Append `self` little-endian to `out`.
    fn write_le(self, out: &mut Vec<u8>);
    /// Read `self` little-endian from the first `WIRE_BYTES` bytes of `data`.
    fn read_le(data: &[u8]) -> Result<Self>;
    fn bitxor(self, other: Self) -> Self;
    fn leading_zeros(self) -> u8;
    fn trailing_zeros(self) -> u8;
    fn is_zero(self) -> bool;
    /// Low `width` bits set (`width` may equal `BITS`).
    fn low_mask(width: u8) -> Self;
    fn shl(self, n: u8) -> Self;
    fn shr(self, n: u8) -> Self;
    fn to_u64(self) -> u64;
    fn from_u64(v: u64) -> Self;
}

impl GorillaPrim for u32 {
    const WIRE_BYTES: usize = 4;
    const BITS: u8 = 32;
    const LEADING_BITS: u8 = 5;
    const MEANINGFUL_BITS: u8 = 6;

    fn write_le(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }

    fn read_le(data: &[u8]) -> Result<Self> {
        let bytes: [u8; 4] = data
            .get(..4)
            .and_then(|s| s.try_into().ok())
            .ok_or_else(|| anyhow::anyhow!("wire read: need 4 bytes, got {}", data.len()))?;
        Ok(Self::from_le_bytes(bytes))
    }

    fn bitxor(self, other: Self) -> Self {
        self ^ other
    }

    fn leading_zeros(self) -> u8 {
        self.leading_zeros() as u8
    }

    fn trailing_zeros(self) -> u8 {
        self.trailing_zeros() as u8
    }

    fn is_zero(self) -> bool {
        self == 0
    }

    fn low_mask(width: u8) -> Self {
        if width >= 32 {
            u32::MAX
        } else {
            (1u32 << width) - 1
        }
    }

    fn shl(self, n: u8) -> Self {
        self << n
    }

    fn shr(self, n: u8) -> Self {
        self >> n
    }

    fn to_u64(self) -> u64 {
        self as u64
    }

    fn from_u64(v: u64) -> Self {
        v as u32
    }
}

impl GorillaPrim for u64 {
    const WIRE_BYTES: usize = 8;
    const BITS: u8 = 64;
    const LEADING_BITS: u8 = 6;
    const MEANINGFUL_BITS: u8 = 7;

    fn write_le(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }

    fn read_le(data: &[u8]) -> Result<Self> {
        let bytes: [u8; 8] = data
            .get(..8)
            .and_then(|s| s.try_into().ok())
            .ok_or_else(|| anyhow::anyhow!("wire read: need 8 bytes, got {}", data.len()))?;
        Ok(Self::from_le_bytes(bytes))
    }

    fn bitxor(self, other: Self) -> Self {
        self ^ other
    }

    fn leading_zeros(self) -> u8 {
        self.leading_zeros() as u8
    }

    fn trailing_zeros(self) -> u8 {
        self.trailing_zeros() as u8
    }

    fn is_zero(self) -> bool {
        self == 0
    }

    fn low_mask(width: u8) -> Self {
        if width >= 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        }
    }

    fn shl(self, n: u8) -> Self {
        self << n
    }

    fn shr(self, n: u8) -> Self {
        self >> n
    }

    fn to_u64(self) -> u64 {
        self
    }

    fn from_u64(v: u64) -> Self {
        v
    }
}

// ===== Core wire format encoding functions =====

/// Single Gorilla encode core over the unsigned XOR domain.
fn encode_gorilla_wire<P: GorillaPrim>(values: &[P]) -> Result<Vec<u8>> {
    if values.is_empty() {
        return Ok(Vec::new());
    }

    let mut result = Vec::new();
    values[0].write_le(&mut result);

    if values.len() == 1 {
        return Ok(result);
    }

    let mut bit_writer = BitWriter::new();
    let mut prev_value = values[0];
    let mut prev_leading_zeros = 0u8;
    let mut prev_trailing_zeros = 0u8;

    for &current in &values[1..] {
        let xor = prev_value.bitxor(current);

        if xor.is_zero() {
            bit_writer.write_bit(false);
        } else {
            bit_writer.write_bit(true);

            let leading_zeros = xor.leading_zeros();
            let trailing_zeros = xor.trailing_zeros();
            let meaningful_bits = P::BITS - leading_zeros - trailing_zeros;

            if leading_zeros >= prev_leading_zeros
                && trailing_zeros >= prev_trailing_zeros
                && prev_leading_zeros + prev_trailing_zeros < P::BITS
            {
                bit_writer.write_bit(false);
                let block_size = P::BITS - prev_leading_zeros - prev_trailing_zeros;
                let mask = P::low_mask(block_size);
                let value_bits = xor.shr(prev_trailing_zeros) & mask;
                bit_writer.write_bits(value_bits.to_u64(), block_size);
            } else {
                bit_writer.write_bit(true);
                bit_writer.write_bits(leading_zeros as u64, P::LEADING_BITS);
                bit_writer.write_bits(meaningful_bits as u64, P::MEANINGFUL_BITS);
                let mask = P::low_mask(meaningful_bits);
                let value_bits = xor.shr(trailing_zeros) & mask;
                bit_writer.write_bits(value_bits.to_u64(), meaningful_bits);
                prev_leading_zeros = leading_zeros;
                prev_trailing_zeros = trailing_zeros;
            }
        }

        prev_value = current;
    }

    let compressed = bit_writer.finish();
    result.extend(compressed);

    Ok(result)
}

/// Core encoding logic for u32 wire type (used by f32 and i32): cast into the
/// unsigned XOR domain and run the shared core.
fn encode_gorilla_u32_wire(wire_values: &[i32]) -> Result<Vec<u8>> {
    let u32_values: Vec<u32> = wire_values.iter().map(|&v| v as u32).collect();
    encode_gorilla_wire(&u32_values)
}

/// Core encoding logic for u64 wire type (used by i64): identity cast.
fn encode_gorilla_u64_wire(wire_values: &[i64]) -> Result<Vec<u8>> {
    let u64_values: Vec<u64> = wire_values.iter().map(|&v| v as u64).collect();
    encode_gorilla_wire(&u64_values)
}

// ===== Public API (thin wrappers using generic helpers) =====

/// Encode f32 values using Gorilla compression (raw, no headers)
///
/// # Algorithm
/// 1. Store first value as-is
/// 2. For each subsequent value:
///    - XOR with previous value
///    - If XOR == 0: store single control bit (0)
///    - If XOR != 0: store control bit (1) + compressed XOR
///
/// # Format (raw data only, NO headers)
/// ```text
/// [first_value:4 bytes][compressed_stream...]
/// ```text
///
/// # Parameters
/// - `values`: f32 slice to encode
///
/// # Returns
/// Raw encoded bytes (NO scheme marker, NO count header)
pub fn encode_f32(values: &[f32]) -> Result<Vec<u8>> {
    helpers::encode_generic(values, encode_gorilla_u32_wire)
}

/// Encode f64 values using Gorilla compression (raw, no headers)
pub fn encode_f64(values: &[f64]) -> Result<Vec<u8>> {
    helpers::encode_generic(values, encode_gorilla_u64_wire)
}

/// Encode i64 values using Gorilla-style XOR compression
pub fn encode_i64(values: &[i64]) -> Result<Vec<u8>> {
    helpers::encode_generic(values, encode_gorilla_u64_wire)
}

/// Encode i32 values using Gorilla-style XOR compression
pub fn encode_i32(values: &[i32]) -> Result<Vec<u8>> {
    helpers::encode_generic(values, encode_gorilla_u32_wire)
}

// ===== Core wire format decoding functions =====

/// Single Gorilla decode core over the unsigned XOR domain.
fn decode_gorilla_wire<P: GorillaPrim>(data: &[u8], count: usize) -> Result<Vec<P>> {
    if count == 0 {
        return Ok(Vec::new());
    }

    if data.len() < P::WIRE_BYTES {
        return Err(anyhow::anyhow!("Gorilla decode: insufficient data"));
    }

    let first_value = P::read_le(data)?;
    let mut result = Vec::with_capacity(count);
    result.push(first_value);

    if count == 1 {
        return Ok(result);
    }

    let mut bit_reader = BitReader::new(&data[P::WIRE_BYTES..]);
    let mut prev_value = first_value;
    let mut prev_leading_zeros = 0u8;
    let mut prev_trailing_zeros = 0u8;

    for _ in 1..count {
        let control_bit = bit_reader.read_bit()?;

        if !control_bit {
            result.push(prev_value);
        } else {
            let use_prev_block = !bit_reader.read_bit()?;

            let xor = if use_prev_block {
                let block_size = P::BITS - prev_leading_zeros - prev_trailing_zeros;
                let value_bits = P::from_u64(bit_reader.read_bits(block_size)?);
                value_bits.shl(prev_trailing_zeros)
            } else {
                let leading_zeros = bit_reader.read_bits(P::LEADING_BITS)? as u8;
                let meaningful_bits = bit_reader.read_bits(P::MEANINGFUL_BITS)? as u8;
                let value_bits = P::from_u64(bit_reader.read_bits(meaningful_bits)?);
                let trailing_zeros = P::BITS - leading_zeros - meaningful_bits;
                prev_leading_zeros = leading_zeros;
                prev_trailing_zeros = trailing_zeros;
                value_bits.shl(trailing_zeros)
            };

            let current_value = prev_value.bitxor(xor);
            result.push(current_value);
            prev_value = current_value;
        }
    }

    Ok(result)
}

/// Core decoding logic for u32 wire type: decode the unsigned domain and
/// cast back to i32.
fn decode_gorilla_u32_wire(data: &[u8], count: usize) -> Result<Vec<i32>> {
    let values = decode_gorilla_wire::<u32>(data, count)?;
    Ok(values.into_iter().map(|v| v as i32).collect())
}

/// Core decoding logic for u64 wire type: cast back to i64.
fn decode_gorilla_u64_wire(data: &[u8], count: usize) -> Result<Vec<i64>> {
    let values = decode_gorilla_wire::<u64>(data, count)?;
    Ok(values.into_iter().map(|v| v as i64).collect())
}

// ===== Public API (thin wrappers using generic helpers) =====

/// Decode f32 values from Gorilla compressed data
pub fn decode_f32(data: &[u8], count: usize) -> Result<Vec<f32>> {
    helpers::decode_generic::<f32>(data, count, decode_gorilla_u32_wire)
}

/// Decode f64 values from Gorilla compressed data
pub fn decode_f64(data: &[u8], count: usize) -> Result<Vec<f64>> {
    helpers::decode_generic::<f64>(data, count, decode_gorilla_u64_wire)
}

/// Decode i64 values from Gorilla compressed data
pub fn decode_i64(data: &[u8], count: usize) -> Result<Vec<i64>> {
    helpers::decode_generic::<i64>(data, count, decode_gorilla_u64_wire)
}

/// Decode i32 values from Gorilla compressed data
pub fn decode_i32(data: &[u8], count: usize) -> Result<Vec<i32>> {
    helpers::decode_generic::<i32>(data, count, decode_gorilla_u32_wire)
}

// ===== Bit-level I/O helpers =====

struct BitWriter {
    bytes: Vec<u8>,
    current_byte: u8,
    bit_pos: u8,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            current_byte: 0,
            bit_pos: 0,
        }
    }

    fn write_bit(&mut self, bit: bool) {
        if bit {
            self.current_byte |= 1 << (7 - self.bit_pos);
        }
        self.bit_pos += 1;

        if self.bit_pos == 8 {
            self.bytes.push(self.current_byte);
            self.current_byte = 0;
            self.bit_pos = 0;
        }
    }

    fn write_bits(&mut self, value: u64, num_bits: u8) {
        for i in (0..num_bits).rev() {
            let bit = (value >> i) & 1 == 1;
            self.write_bit(bit);
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bit_pos > 0 {
            self.bytes.push(self.current_byte);
        }
        self.bytes
    }
}

struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    fn read_bit(&mut self) -> Result<bool> {
        if self.byte_pos >= self.data.len() {
            return Err(anyhow::anyhow!("BitReader: unexpected end of data"));
        }

        let bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1 == 1;
        self.bit_pos += 1;

        if self.bit_pos == 8 {
            self.byte_pos += 1;
            self.bit_pos = 0;
        }

        Ok(bit)
    }

    fn read_bits(&mut self, num_bits: u8) -> Result<u64> {
        let mut value = 0u64;
        for _ in 0..num_bits {
            value = (value << 1) | (self.read_bit()? as u64);
        }
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Golden-byte pins (redundancy pass 4) ====================
    // Wire layout: [first value:WIRE_BYTES LE][bit stream, MSB-first per byte].
    // Per non-zero XOR: '1' marker, then either '0' (reuse previous block:
    // value bits only) or '1' (leading-zeros count in LEADING_BITS bits,
    // meaningful-bits count in MEANINGFUL_BITS bits, then the meaningful
    // bits). The u32 stream uses 5/6-bit count fields, the u64 stream 6/7.
    // These pin the CURRENT bytes of both widths BEFORE the GorillaPrim
    // consolidation; any change is a wire-format change.

    /// u32: [1,2] -> xor 3 reuses the initial full-width block: stream is
    /// '1','0' + 32 value bits (thirty 0s then '11') = 34 bits -> 5 bytes
    /// 0x80,0,0,0,0xC0.
    #[test]
    fn golden_gorilla_u32_wire_bytes() {
        let encoded = encode_i32(&[1, 2]).unwrap();
        assert_eq!(encoded, vec![1, 0, 0, 0, 0x80, 0, 0, 0, 0xC0]);
        assert_eq!(decode_i32(&encoded, 2).unwrap(), vec![1, 2]);
    }

    /// u32, identical values: [1,1] -> single '0' control bit -> one 0x00
    /// byte after the 4-byte header.
    #[test]
    fn golden_gorilla_u32_repeat_zero_bit() {
        let encoded = encode_i32(&[1, 1]).unwrap();
        assert_eq!(encoded, vec![1, 0, 0, 0, 0x00]);
        assert_eq!(decode_i32(&encoded, 2).unwrap(), vec![1, 1]);
    }

    /// u64: [1,2] -> same shape with a 64-bit block: '1','0' + 64 value bits
    /// (62 zeros then '11') = 66 bits -> 9 bytes, last byte 0xC0.
    #[test]
    fn golden_gorilla_u64_wire_bytes() {
        let encoded = encode_i64(&[1, 2]).unwrap();
        assert_eq!(
            encoded,
            vec![
                1, 0, 0, 0, 0, 0, 0, 0, // first value 1 LE i64
                0x80, 0, 0, 0, 0, 0, 0, 0, 0xC0, // bit stream
            ]
        );
        assert_eq!(decode_i64(&encoded, 2).unwrap(), vec![1, 2]);
    }

    #[test]
    fn test_gorilla_identical_values() -> Result<()> {
        // All same value - best case for Gorilla
        let values = vec![42.0f32; 100];

        let encoded = encode_f32(&values)?;
        let decoded = decode_f32(&encoded, values.len())?;

        assert_eq!(values.len(), decoded.len());
        for (orig, dec) in values.iter().zip(decoded.iter()) {
            assert_eq!(orig.to_bits(), dec.to_bits());
        }

        // Should be very small: first value + 100 zero bits
        assert!(encoded.len() < 20);
        Ok(())
    }

    #[test]
    fn test_gorilla_similar_values() -> Result<()> {
        // Similar values (typical time-series, 32+ values)
        let values: Vec<f32> = (0..32).map(|i| 20.0 + (i as f32 * 0.1)).collect();

        let encoded = encode_f32(&values)?;
        let decoded = decode_f32(&encoded, values.len())?;

        assert_eq!(values.len(), decoded.len());
        // Gorilla optimized for floats - should roundtrip exactly
        for (orig, dec) in values.iter().zip(decoded.iter()) {
            let diff = (orig - dec).abs();
            assert!(diff < 0.01, "Expected {}, got {}", orig, dec);
        }
        Ok(())
    }

    #[test]
    fn test_gorilla_i64_roundtrip() -> Result<()> {
        let values: Vec<i64> = (0..32).map(|i| 1000 + i).collect();

        let encoded = encode_i64(&values)?;
        let decoded = decode_i64(&encoded, values.len())?;

        assert_eq!(values, decoded);
        Ok(())
    }

    #[test]
    fn test_gorilla_f64_roundtrip() -> Result<()> {
        let values: Vec<f64> = (0..32).map(|i| -10_000.0 + (i as f64 * 0.125)).collect();

        let encoded = encode_f64(&values)?;
        let decoded = decode_f64(&encoded, values.len())?;

        assert_eq!(values.len(), decoded.len());
        for (orig, dec) in values.iter().zip(decoded.iter()) {
            assert_eq!(orig.to_bits(), dec.to_bits());
        }
        Ok(())
    }

    #[test]
    fn test_gorilla_i32_roundtrip() -> Result<()> {
        let values: Vec<i32> = (0..32).map(|i| 100 + i).collect();

        let encoded = encode_i32(&values)?;
        let decoded = decode_i32(&encoded, values.len())?;

        assert_eq!(values, decoded);
        Ok(())
    }

    #[test]
    fn test_gorilla_empty() -> Result<()> {
        let values: Vec<f32> = vec![];
        let encoded = encode_f32(&values)?;
        assert!(encoded.is_empty());

        let decoded = decode_f32(&encoded, 0)?;
        assert!(decoded.is_empty());
        Ok(())
    }

    #[test]
    fn test_gorilla_single_value() -> Result<()> {
        let values = vec![42.0f32];

        let encoded = encode_f32(&values)?;
        let decoded = decode_f32(&encoded, values.len())?;

        assert_eq!(values.len(), decoded.len());
        assert_eq!(values[0].to_bits(), decoded[0].to_bits());
        Ok(())
    }

    #[test]
    fn test_gorilla_temperature_sensor() -> Result<()> {
        // Simulated temperature sensor data
        let mut values = Vec::new();
        let mut temp = 20.0f32;
        for _ in 0..100 {
            values.push(temp);
            temp += 0.1; // Gradual increase
        }

        let encoded = encode_f32(&values)?;
        let decoded = decode_f32(&encoded, values.len())?;

        assert_eq!(values.len(), decoded.len());

        // Gorilla optimized for time-series floats
        for (orig, dec) in values.iter().zip(decoded.iter()) {
            let diff = (orig - dec).abs();
            assert!(diff < 0.01, "Expected {}, got {}", orig, dec);
        }
        Ok(())
    }

    #[test]
    fn test_gorilla_timestamps() -> Result<()> {
        // Simulated timestamps with small increments
        let mut values = Vec::new();
        let mut ts = 1000000i64;
        for _ in 0..100 {
            values.push(ts);
            ts += 1000; // +1 second
        }

        let encoded = encode_i64(&values)?;
        let decoded = decode_i64(&encoded, values.len())?;

        assert_eq!(values, decoded);
        Ok(())
    }
}
