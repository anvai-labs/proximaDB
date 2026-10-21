// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! Generic type conversion helpers for ProximaCodec encoding schemes
//!
//! This module provides traits and functions for generic type handling across
//! encoding schemes, reducing code duplication from 96 type-specific functions
//! to ~32 generic implementations.

use anyhow::Result;

use super::bitpack;

/// Fixed-width wire integer primitive (i32 or i64).
///
/// Redundancy pass 4 (following the `PackPrim` precedent in `bitpack.rs`):
/// every codec's `_i32_wire`/`_i64_wire` twin cores differed only in the
/// integer width of their wire slots, base values, and packed payloads. This
/// trait factors out that width so each codec keeps ONE generic core; the
/// golden-byte tests in each codec pin the layout the cores must preserve.
///
/// `ToWireFormat` (below) maps application types onto these primitives
/// (f32 -> i32, f64 -> i64); `WirePrim` parameterizes the core over which of
/// the two widths it operates on. Width operations are added to the trait
/// incrementally, alongside the codec whose consolidation needs them.
pub trait WirePrim: Copy + Default + std::fmt::Debug + PartialEq + Eq + std::hash::Hash {
    /// Wire slot size in bytes (4 for i32, 8 for i64).
    const WIRE_BYTES: usize;
    /// Total value bits (32 for i32, 64 for i64).
    const BITS: u8;

    /// Append `self` little-endian to `out`.
    fn write_le(self, out: &mut Vec<u8>);
    /// Read `self` little-endian from the first `WIRE_BYTES` bytes of `data`.
    /// Errors (never panics) when `data` is too short.
    fn read_le(data: &[u8]) -> Result<Self>;
    /// `self == 0` (sparse codecs' non-zero test).
    fn is_zero(self) -> bool;
    /// `self.wrapping_sub(rhs)` in the native width (trait spellings `wrap_*`
    /// avoid shadowing the inherent integer methods wherever the trait is in
    /// scope).
    fn wrap_sub(self, rhs: Self) -> Self;
    /// `self.wrapping_add(rhs)` in the native width.
    fn wrap_add(self, rhs: Self) -> Self;
    /// Bit length of `self.unsigned_abs()` in the native width
    /// (`BITS - leading_zeros`), so 0 maps to 0.
    fn abs_bit_len(self) -> u8;
    /// Widen to i64 (identity for i64). Lets a narrow base share the i64
    /// delta arithmetic of the delta/patched/pfor family.
    fn widen(self) -> i64;
    /// Narrow an i64 back (wrapping truncation for i32).
    fn from_wide(w: i64) -> Self;
    /// Signed bit-pack at `bits` width (delegates to `bitpack`).
    fn pack_signed(values: &[Self], bits: u8) -> Result<Vec<u8>>;
    /// Signed bit-unpack with sign extension (delegates to `bitpack`).
    fn unpack_signed(data: &[u8], bits: u8, count: usize) -> Result<Vec<Self>>;
}

impl WirePrim for i32 {
    const WIRE_BYTES: usize = 4;
    const BITS: u8 = 32;

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

    fn is_zero(self) -> bool {
        self == 0
    }

    fn wrap_sub(self, rhs: Self) -> Self {
        i32::wrapping_sub(self, rhs)
    }

    fn wrap_add(self, rhs: Self) -> Self {
        i32::wrapping_add(self, rhs)
    }

    fn abs_bit_len(self) -> u8 {
        32 - self.unsigned_abs().leading_zeros() as u8
    }

    fn widen(self) -> i64 {
        self as i64
    }

    fn from_wide(w: i64) -> Self {
        w as i32
    }

    fn pack_signed(values: &[Self], bits: u8) -> Result<Vec<u8>> {
        bitpack::bitpack_i32(values, bits)
    }

    fn unpack_signed(data: &[u8], bits: u8, count: usize) -> Result<Vec<Self>> {
        bitpack::unbitpack_i32(data, bits, count)
    }
}

impl WirePrim for i64 {
    const WIRE_BYTES: usize = 8;
    const BITS: u8 = 64;

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

    fn is_zero(self) -> bool {
        self == 0
    }

    fn wrap_sub(self, rhs: Self) -> Self {
        i64::wrapping_sub(self, rhs)
    }

    fn wrap_add(self, rhs: Self) -> Self {
        i64::wrapping_add(self, rhs)
    }

    fn abs_bit_len(self) -> u8 {
        64 - self.unsigned_abs().leading_zeros() as u8
    }

    fn widen(self) -> i64 {
        self
    }

    fn from_wide(w: i64) -> Self {
        w
    }

    fn pack_signed(values: &[Self], bits: u8) -> Result<Vec<u8>> {
        bitpack::bitpack_i64(values, bits)
    }

    fn unpack_signed(data: &[u8], bits: u8, count: usize) -> Result<Vec<Self>> {
        bitpack::unbitpack_i64(data, bits, count)
    }
}

/// Generic type conversion trait for wire format serialization
///
/// Converts application types (f32, f64, i32, i64) to wire representation types
/// suitable for bitpacking and encoding operations.
///
/// # Wire Format Types
/// - `f32 → i32` via `to_bits()` for lossless bit-level encoding
/// - `f64 → i64` via `to_bits()` for lossless bit-level encoding
/// - `i32 → i32` direct pass-through (identity conversion)
/// - `i64 → i64` direct pass-through (identity conversion)
pub trait ToWireFormat: Copy {
    /// The wire representation type (i32 or i64)
    type WireType: Copy + std::fmt::Debug;

    /// Convert value to wire format
    fn to_wire(&self) -> Self::WireType;

    /// Convert value from wire format back to original type
    fn from_wire(wire: Self::WireType) -> Self;
}

impl ToWireFormat for f32 {
    type WireType = i32;

    #[inline]
    fn to_wire(&self) -> i32 {
        self.to_bits() as i32
    }

    #[inline]
    fn from_wire(wire: i32) -> f32 {
        f32::from_bits(wire as u32)
    }
}

impl ToWireFormat for f64 {
    type WireType = i64;

    #[inline]
    fn to_wire(&self) -> i64 {
        self.to_bits() as i64
    }

    #[inline]
    fn from_wire(wire: i64) -> f64 {
        f64::from_bits(wire as u64)
    }
}

impl ToWireFormat for i32 {
    type WireType = i32;

    #[inline]
    fn to_wire(&self) -> i32 {
        *self
    }

    #[inline]
    fn from_wire(wire: i32) -> i32 {
        wire
    }
}

impl ToWireFormat for i64 {
    type WireType = i64;

    #[inline]
    fn to_wire(&self) -> i64 {
        *self
    }

    #[inline]
    fn from_wire(wire: i64) -> i64 {
        wire
    }
}

/// Generic encoder wrapper that converts input types to wire format
///
/// This function encapsulates the pattern of:
/// 1. Convert input values to wire format (e.g., f32 → i32 via to_bits)
/// 2. Encode wire values using scheme-specific logic
/// 3. Return encoded bytes
///
/// # Example
/// ```ignore
/// pub fn encode_f32(values: &[f32], base: i64) -> Result<Vec<u8>> {
///     encode_generic(values, |wire_values| {
///         encode_frame_of_ref_wire(wire_values, base)
///     })
/// }
/// ```text
#[inline]
pub fn encode_generic<T: ToWireFormat>(
    values: &[T],
    encode_fn: impl Fn(&[T::WireType]) -> Result<Vec<u8>>,
) -> Result<Vec<u8>> {
    let wire_values: Vec<T::WireType> = values.iter().map(|v| v.to_wire()).collect();
    encode_fn(&wire_values)
}

/// Generic decoder wrapper that converts wire format back to output types
///
/// This function encapsulates the pattern of:
/// 1. Decode bytes to wire format values (i32 or i64)
/// 2. Convert wire values back to application types (e.g., i32 → f32 via from_bits)
/// 3. Return decoded values
///
/// # Example
/// ```ignore
/// pub fn decode_f32(data: &[u8], count: usize) -> Result<Vec<f32>> {
///     decode_generic(data, count, |d, c| {
///         decode_delta_wire(d, c)
///     })
/// }
/// ```text
#[inline]
pub fn decode_generic<T: ToWireFormat>(
    data: &[u8],
    count: usize,
    decode_fn: impl Fn(&[u8], usize) -> Result<Vec<T::WireType>>,
) -> Result<Vec<T>> {
    let wire_values = decode_fn(data, count)?;
    Ok(wire_values.iter().map(|&w| T::from_wire(w)).collect())
}

/// Reconstruction helper for f32 from i32 offsets
///
/// Used by FrameOfReference and similar schemes where offsets are stored
/// as i32 values and need to be added to a base value.
///
/// # Arguments
/// * `offsets` - i32 offsets from base value
/// * `base` - Base value in i32 representation
///
/// # Returns
/// Vec of f32 values reconstructed via from_bits
#[inline]
pub fn reconstruct_f32_from_i32(offsets: &[i32], base: i32) -> Vec<f32> {
    offsets
        .iter()
        .map(|&offset| f32::from_bits(base.wrapping_add(offset) as u32))
        .collect()
}

/// Reconstruction helper for f32 from i64 deltas
///
/// Used by PForDelta and similar schemes where deltas are stored
/// as i64 values (to avoid overflow during delta calculation).
///
/// # Arguments
/// * `deltas` - i64 deltas from base value
/// * `base` - Base value in i32 representation
///
/// # Returns
/// Vec of f32 values reconstructed via from_bits
#[inline]
pub fn reconstruct_f32_from_i64(deltas: &[i64], base: i32) -> Vec<f32> {
    deltas
        .iter()
        .map(|&delta| {
            let val_i64 = (base as i64) + delta; // i64 arithmetic - no overflow
            f32::from_bits(val_i64 as i32 as u32)
        })
        .collect()
}

/// Reconstruction helper for i32 from i64 deltas
///
/// Converts i64 deltas to i32 values, handling potential overflow.
#[inline]
pub fn reconstruct_i32_from_i64(deltas: &[i64], base: i32) -> Vec<i32> {
    deltas
        .iter()
        .map(|&delta| {
            let val_i64 = (base as i64) + delta;
            val_i64 as i32 // Truncate to i32
        })
        .collect()
}

/// Reconstruction helper for i64 from i64 deltas
///
/// Direct addition with wrapping semantics for signed integers.
#[inline]
pub fn reconstruct_i64_from_i64(deltas: &[i64], base: i64) -> Vec<i64> {
    deltas
        .iter()
        .map(|&delta| base.wrapping_add(delta))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wire_prim_le_roundtrip_both_widths() {
        let mut buf = Vec::new();
        (-5i32).write_le(&mut buf);
        assert_eq!(buf, (-5i32).to_le_bytes().to_vec());
        assert_eq!(
            i32::read_le(&[1, 2, 3, 4]).unwrap(),
            i32::from_le_bytes([1, 2, 3, 4])
        );

        let mut buf = Vec::new();
        (-7i64).write_le(&mut buf);
        assert_eq!(buf, (-7i64).to_le_bytes().to_vec());
        assert_eq!(
            i64::read_le(&[1, 2, 3, 4, 5, 6, 7, 8]).unwrap(),
            i64::from_le_bytes([1, 2, 3, 4, 5, 6, 7, 8])
        );

        // Short buffers error instead of panicking.
        assert!(i32::read_le(&[1, 2, 3]).is_err());
        assert!(i64::read_le(&[1, 2, 3]).is_err());
    }

    #[test]
    fn test_wire_prim_width_ops() {
        // abs_bit_len: 0 -> 0, native-width bit lengths.
        assert_eq!(0i32.abs_bit_len(), 0);
        assert_eq!(2i32.abs_bit_len(), 2);
        assert_eq!((-2i32).abs_bit_len(), 2);
        assert_eq!(0i64.abs_bit_len(), 0);
        assert_eq!((1i64 << 40).abs_bit_len(), 41);
        assert_eq!(i64::MIN.abs_bit_len(), 64);

        // wrap_* match the inherent wrapping arithmetic.
        assert_eq!(i32::MIN.wrap_sub(1), i32::MAX);
        assert_eq!(i32::MAX.wrap_add(1), i32::MIN);
        assert_eq!(i64::MIN.wrap_sub(1), i64::MAX);
        assert_eq!(i64::MAX.wrap_add(1), i64::MIN);

        // pack_signed/unpack_signed delegate to the bitpack helpers; the
        // signed unpack SIGN-EXTENDS (2 = 10b -> -2, 3 = 11b -> -1 at 2 bits).
        let packed = i32::pack_signed(&[1, 2, 3], 2).unwrap();
        assert_eq!(packed, vec![0x39]);
        assert_eq!(i32::unpack_signed(&packed, 2, 3).unwrap(), vec![1, -2, -1]);
        let packed64 = i64::pack_signed(&[1, 2, 3], 2).unwrap();
        assert_eq!(packed64, vec![0x39]);
        assert_eq!(
            i64::unpack_signed(&packed64, 2, 3).unwrap(),
            vec![1, -2, -1]
        );
    }

    #[test]
    fn test_to_wire_format_f32() {
        let value = 1.5f32;
        let wire = value.to_wire();
        let roundtrip = f32::from_wire(wire);
        assert_eq!(value, roundtrip);
    }

    #[test]
    fn test_to_wire_format_f64() {
        let value = -12345.6789f64;
        let wire = value.to_wire();
        let roundtrip = f64::from_wire(wire);
        assert_eq!(value.to_bits(), roundtrip.to_bits());
    }

    #[test]
    fn test_to_wire_format_i32() {
        let value = -42i32;
        let wire = value.to_wire();
        let roundtrip = i32::from_wire(wire);
        assert_eq!(value, roundtrip);
    }

    #[test]
    fn test_to_wire_format_i64() {
        let value = -123456789i64;
        let wire = value.to_wire();
        let roundtrip = i64::from_wire(wire);
        assert_eq!(value, roundtrip);
    }

    #[test]
    fn test_encode_generic_f32() {
        let values = vec![1.0f32, 2.0, 3.0];

        // Simple mock encoder that just stores wire values as bytes
        let result = encode_generic(&values, |wire_values| {
            assert_eq!(wire_values.len(), 3);
            assert_eq!(wire_values[0], 1.0f32.to_bits() as i32);
            Ok(vec![1, 2, 3]) // Mock bytes
        });

        assert!(result.is_ok());
    }

    #[test]
    fn test_decode_generic_f32() {
        let data = vec![1, 2, 3];

        // Mock decoder that returns wire format values
        let result = decode_generic::<f32>(&data, 2, |_d, _count| {
            Ok(vec![1.0f32.to_bits() as i32, 2.0f32.to_bits() as i32])
        });

        assert!(result.is_ok());
        let decoded = result.unwrap();
        assert_eq!(decoded, vec![1.0f32, 2.0]);
    }

    #[test]
    fn test_reconstruct_f32_from_i32() {
        let base = 1.0f32.to_bits() as i32;
        let offsets = vec![0, (2.0f32.to_bits() as i32) - base]; // 0.0 offset, 1.0 offset
        let result = reconstruct_f32_from_i32(&offsets, base);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 1.0f32);
        assert_eq!(result[1], 2.0f32);
    }

    #[test]
    fn test_reconstruct_f32_from_i64() {
        let base = 1.0f32.to_bits() as i32;
        let deltas = vec![0i64, (2.0f32.to_bits() as i32 - base) as i64];
        let result = reconstruct_f32_from_i64(&deltas, base);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 1.0f32);
        assert_eq!(result[1], 2.0f32);
    }

    #[test]
    fn test_reconstruct_i64_from_i64() {
        let base = 1000i64;
        let deltas = vec![0, 500, -250];
        let result = reconstruct_i64_from_i64(&deltas, base);

        assert_eq!(result, vec![1000, 1500, 750]);
    }
}
