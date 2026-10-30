// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! # `key_codec` — typed, order-preserving composite key encoding (F0 · ADR-072 D9)
//!
//! Replaces the defective `encode_primary_key_tuple` string-join (which had
//! documented `\u{1f}`-separator collisions, was non-order-preserving, and
//! type-lossy). The encoding is calibrated against the FoundationDB tuple layer
//! and CockroachDB key encoding; see `TD-FOUNDATION-1` Appendix A. Three
//! properties hold, and they are the point:
//!
//! 1. **Prefix-free / self-terminating** per column — the concatenation of
//!    columns is unambiguous, so two *distinct* tuples can never encode to the
//!    same bytes (the collision the old codec had).
//! 2. **Order-preserving** per type — lexicographic byte order equals value
//!    order, so composite-key range scans work.
//! 3. **Type-tagged** — self-describing and robust to schema evolution / NULLs.
//!
//! Layout: `tenant-field ‖ keyspace-field ‖ col₀ ‖ col₁ ‖ …`, where each column
//! is `<type-tag> <order-preserving payload>` and the leading tenant/keyspace
//! string fields make each tenant's keyspace a contiguous, isolated range.
//!
//! Scope (MVP): all-ascending, binary/UTF-8 collation. `Decimal`, locale-aware
//! collation, and per-column descending order are deferred (see the errors and
//! `TD-FOUNDATION-1` Appendix A §"phase 2"). Structured/embedding values are
//! rejected as key components.

use crate::{ProximaType, ProximaValue, TimeUnit};
use anyhow::{Result, bail};

// Type tags. A fixed column has a single type, so order-preservation only has to
// hold *within* a tag; the tag exists for prefix-freeness / no cross-type
// collision and self-description. `0x05` is reserved for `Decimal` (phase 2).
const TAG_NULL: u8 = 0x00;
const TAG_BOOL: u8 = 0x01;
const TAG_SINT: u8 = 0x02;
const TAG_UINT: u8 = 0x03;
const TAG_FLOAT: u8 = 0x04;
const TAG_STRING: u8 = 0x06;
const TAG_BINARY: u8 = 0x07;
const TAG_UUID: u8 = 0x08;
const TAG_ULID: u8 = 0x09;

const ESCAPE: u8 = 0xFF; // 0x00 -> 0x00 0xFF; a lone 0x00 terminates a field.

/// A decoded composite key: the tenant/keyspace scope plus the column values.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedIdentity {
    pub tenant: String,
    pub keyspace: String,
    pub values: Vec<ProximaValue>,
}

/// Encode a tenant/keyspace-scoped composite key from `values`.
///
/// Returns an error for value kinds that cannot form a key component
/// (structured/embedding types, `Decimal` until phase 2, and non-finite floats).
pub fn encode_identity(tenant: &str, keyspace: &str, values: &[ProximaValue]) -> Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(values.len() * 9 + tenant.len() + keyspace.len() + 2);
    encode_str_field(&mut buf, tenant.as_bytes());
    encode_str_field(&mut buf, keyspace.as_bytes());
    for v in values {
        encode_value(&mut buf, v)?;
    }
    Ok(buf)
}

/// Decode a key produced by [`encode_identity`] given the column types (the
/// tag records the type *family* but a fixed-width type still needs its declared
/// width, so the schema's `types` must be supplied). `Time`/`Timestamp*` decode
/// back in nanoseconds (the canonical unit encoding normalizes to).
pub fn decode_identity(types: &[ProximaType], bytes: &[u8]) -> Result<DecodedIdentity> {
    let mut r = Reader { b: bytes, pos: 0 };
    let tenant = String::from_utf8(r.read_str_field()?)?;
    let keyspace = String::from_utf8(r.read_str_field()?)?;
    let mut values = Vec::with_capacity(types.len());
    for t in types {
        values.push(r.read_value(t)?);
    }
    if r.pos != bytes.len() {
        bail!(
            "key_codec: {} trailing byte(s) after {} column(s)",
            bytes.len() - r.pos,
            types.len()
        );
    }
    Ok(DecodedIdentity {
        tenant,
        keyspace,
        values,
    })
}

/// Decode only the `(tenant, keyspace)` scope prefix of a key produced by
/// [`encode_identity`] — no column types needed. Used to route a stored claim
/// back to its table (TD-DML-3 claim reconciliation).
pub fn decode_identity_scope(bytes: &[u8]) -> Result<(String, String)> {
    let mut r = Reader { b: bytes, pos: 0 };
    let tenant = String::from_utf8(r.read_str_field()?)?;
    let keyspace = String::from_utf8(r.read_str_field()?)?;
    Ok((tenant, keyspace))
}

// --- encoding ---------------------------------------------------------------

/// Escaped, `0x00`-terminated bytes with no type tag — used for the positional
/// tenant/keyspace prefix fields (always strings).
fn encode_str_field(buf: &mut Vec<u8>, bytes: &[u8]) {
    for &b in bytes {
        buf.push(b);
        if b == 0x00 {
            buf.push(ESCAPE);
        }
    }
    buf.push(0x00);
}

/// A type-tagged, escaped, `0x00`-terminated field (strings / binary).
fn encode_escaped(buf: &mut Vec<u8>, tag: u8, bytes: &[u8]) {
    buf.push(tag);
    for &b in bytes {
        buf.push(b);
        if b == 0x00 {
            buf.push(ESCAPE);
        }
    }
    buf.push(0x00);
}

fn encode_value(buf: &mut Vec<u8>, v: &ProximaValue) -> Result<()> {
    match v {
        ProximaValue::Null => buf.push(TAG_NULL),
        ProximaValue::Boolean(b) => {
            buf.push(TAG_BOOL);
            buf.push(u8::from(*b));
        }
        // Signed integers: flip the high bit so the two's-complement range maps
        // onto a monotone unsigned big-endian range (min -> 0x00.., max -> 0xFF..).
        ProximaValue::Int8(x) => {
            buf.push(TAG_SINT);
            buf.push((*x as u8) ^ 0x80);
        }
        ProximaValue::Int16(x) => {
            buf.push(TAG_SINT);
            buf.extend_from_slice(&((*x as u16) ^ 0x8000).to_be_bytes());
        }
        ProximaValue::Int32(x) => {
            buf.push(TAG_SINT);
            buf.extend_from_slice(&((*x as u32) ^ 0x8000_0000).to_be_bytes());
        }
        ProximaValue::Int64(x) => {
            buf.push(TAG_SINT);
            buf.extend_from_slice(&((*x as u64) ^ 0x8000_0000_0000_0000).to_be_bytes());
        }
        // Date is days-since-epoch (i32).
        ProximaValue::Date(x) => {
            buf.push(TAG_SINT);
            buf.extend_from_slice(&((*x as u32) ^ 0x8000_0000).to_be_bytes());
        }
        // Temporal i64 + unit: normalize to nanoseconds first, or `1s` and
        // `1000ms` (the same instant) would encode differently.
        ProximaValue::Time(x, u)
        | ProximaValue::Timestamp(x, u)
        | ProximaValue::TimestampTz(x, u) => {
            let nanos = to_nanos(*x, *u)?;
            buf.push(TAG_SINT);
            buf.extend_from_slice(&((nanos as u64) ^ 0x8000_0000_0000_0000).to_be_bytes());
        }
        // Unsigned integers are already monotone as big-endian.
        ProximaValue::UInt8(x) => {
            buf.push(TAG_UINT);
            buf.push(*x);
        }
        ProximaValue::UInt16(x) => {
            buf.push(TAG_UINT);
            buf.extend_from_slice(&x.to_be_bytes());
        }
        ProximaValue::UInt32(x) => {
            buf.push(TAG_UINT);
            buf.extend_from_slice(&x.to_be_bytes());
        }
        ProximaValue::UInt64(x) => {
            buf.push(TAG_UINT);
            buf.extend_from_slice(&x.to_be_bytes());
        }
        // IEEE big-endian with the sign-aware bit flip (correct total order,
        // incl. negatives). NaN is rejected; -0.0 is canonicalized to 0.0 so it
        // does not form a distinct key from 0.0.
        ProximaValue::Float16(x) | ProximaValue::Float32(x) => {
            buf.push(TAG_FLOAT);
            buf.extend_from_slice(&encode_f32(*x)?.to_be_bytes());
        }
        ProximaValue::Float64(x) => {
            buf.push(TAG_FLOAT);
            buf.extend_from_slice(&encode_f64(*x)?.to_be_bytes());
        }
        ProximaValue::String(s) | ProximaValue::Symbol(s) => {
            encode_escaped(buf, TAG_STRING, s.as_bytes());
        }
        ProximaValue::Binary(b) | ProximaValue::BinaryVector(b) => {
            encode_escaped(buf, TAG_BINARY, b);
        }
        ProximaValue::Uuid(bytes) => {
            buf.push(TAG_UUID);
            buf.extend_from_slice(bytes);
        }
        ProximaValue::ULID(bytes) => {
            buf.push(TAG_ULID);
            buf.extend_from_slice(bytes);
        }
        // Order-preserving decimal encoding is phase 2 (needs sign/exp/digits).
        ProximaValue::Decimal(_) => {
            bail!("key_codec: Decimal keys are not supported yet (phase 2)")
        }
        // Structured / embedding values have no well-defined identity ordering.
        ProximaValue::Json(_)
        | ProximaValue::Jsonb(_)
        | ProximaValue::Array(_)
        | ProximaValue::Map(_)
        | ProximaValue::Struct(_)
        | ProximaValue::DenseVector(_)
        | ProximaValue::SparseVector { .. } => {
            bail!("key_codec: {} is not a valid key component", type_name(v))
        }
    }
    Ok(())
}

fn to_nanos(x: i64, u: TimeUnit) -> Result<i64> {
    let factor: i64 = match u {
        TimeUnit::Second => 1_000_000_000,
        TimeUnit::Millisecond => 1_000_000,
        TimeUnit::Microsecond => 1_000,
        TimeUnit::Nanosecond => 1,
    };
    x.checked_mul(factor)
        .ok_or_else(|| anyhow::anyhow!("key_codec: temporal value overflows i64 nanoseconds"))
}

fn encode_f32(x: f32) -> Result<u32> {
    if x.is_nan() {
        bail!("key_codec: NaN is not a valid key component");
    }
    let x = if x == 0.0 { 0.0f32 } else { x }; // canonicalize -0.0 -> 0.0
    let bits = x.to_bits();
    Ok(if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits ^ 0x8000_0000
    })
}

fn encode_f64(x: f64) -> Result<u64> {
    if x.is_nan() {
        bail!("key_codec: NaN is not a valid key component");
    }
    let x = if x == 0.0 { 0.0f64 } else { x };
    let bits = x.to_bits();
    Ok(if bits & 0x8000_0000_0000_0000 != 0 {
        !bits
    } else {
        bits ^ 0x8000_0000_0000_0000
    })
}

fn type_name(v: &ProximaValue) -> &'static str {
    match v {
        ProximaValue::Json(_) => "Json",
        ProximaValue::Jsonb(_) => "Jsonb",
        ProximaValue::Array(_) => "Array",
        ProximaValue::Map(_) => "Map",
        ProximaValue::Struct(_) => "Struct",
        ProximaValue::DenseVector(_) => "DenseVector",
        ProximaValue::SparseVector { .. } => "SparseVector",
        _ => "value",
    }
}

// --- decoding ---------------------------------------------------------------

struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.b.len() {
            bail!(
                "key_codec: truncated key (wanted {} bytes at {})",
                n,
                self.pos
            );
        }
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    /// Read an escaped, `0x00`-terminated field: `0x00 0xFF` unescapes to `0x00`,
    /// a lone `0x00` terminates.
    fn read_str_field(&mut self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        loop {
            let b = self.byte()?;
            if b == 0x00 {
                if self.pos < self.b.len() && self.b[self.pos] == ESCAPE {
                    self.pos += 1;
                    out.push(0x00);
                } else {
                    return Ok(out);
                }
            } else {
                out.push(b);
            }
        }
    }

    fn expect_tag(&mut self, want: u8) -> Result<()> {
        let got = self.byte()?;
        if got != want {
            bail!(
                "key_codec: tag mismatch (got 0x{:02x}, want 0x{:02x})",
                got,
                want
            );
        }
        Ok(())
    }

    fn read_value(&mut self, t: &ProximaType) -> Result<ProximaValue> {
        // A NULL is encoded as TAG_NULL regardless of the column's declared type
        // (a nullable key column), so resolve it before dispatching on the type.
        if self.pos < self.b.len() && self.b[self.pos] == TAG_NULL {
            self.pos += 1;
            return Ok(ProximaValue::Null);
        }
        match t {
            ProximaType::Boolean => {
                self.expect_tag(TAG_BOOL)?;
                Ok(ProximaValue::Boolean(self.byte()? != 0))
            }
            ProximaType::Int8 => {
                self.expect_tag(TAG_SINT)?;
                Ok(ProximaValue::Int8((self.byte()? ^ 0x80) as i8))
            }
            ProximaType::Int16 => {
                self.expect_tag(TAG_SINT)?;
                let u = u16::from_be_bytes(self.take(2)?.try_into().unwrap()) ^ 0x8000;
                Ok(ProximaValue::Int16(u as i16))
            }
            ProximaType::Int32 => {
                self.expect_tag(TAG_SINT)?;
                let u = u32::from_be_bytes(self.take(4)?.try_into().unwrap()) ^ 0x8000_0000;
                Ok(ProximaValue::Int32(u as i32))
            }
            ProximaType::Int64 => {
                self.expect_tag(TAG_SINT)?;
                let u =
                    u64::from_be_bytes(self.take(8)?.try_into().unwrap()) ^ 0x8000_0000_0000_0000;
                Ok(ProximaValue::Int64(u as i64))
            }
            ProximaType::Date => {
                self.expect_tag(TAG_SINT)?;
                let u = u32::from_be_bytes(self.take(4)?.try_into().unwrap()) ^ 0x8000_0000;
                Ok(ProximaValue::Date(u as i32))
            }
            ProximaType::Time(_) => {
                self.read_temporal(|v| ProximaValue::Time(v, TimeUnit::Nanosecond))
            }
            ProximaType::Timestamp(_) => {
                self.read_temporal(|v| ProximaValue::Timestamp(v, TimeUnit::Nanosecond))
            }
            ProximaType::TimestampTz(_) => {
                self.read_temporal(|v| ProximaValue::TimestampTz(v, TimeUnit::Nanosecond))
            }
            ProximaType::UInt8 => {
                self.expect_tag(TAG_UINT)?;
                Ok(ProximaValue::UInt8(self.byte()?))
            }
            ProximaType::UInt16 => {
                self.expect_tag(TAG_UINT)?;
                Ok(ProximaValue::UInt16(u16::from_be_bytes(
                    self.take(2)?.try_into().unwrap(),
                )))
            }
            ProximaType::UInt32 => {
                self.expect_tag(TAG_UINT)?;
                Ok(ProximaValue::UInt32(u32::from_be_bytes(
                    self.take(4)?.try_into().unwrap(),
                )))
            }
            ProximaType::UInt64 => {
                self.expect_tag(TAG_UINT)?;
                Ok(ProximaValue::UInt64(u64::from_be_bytes(
                    self.take(8)?.try_into().unwrap(),
                )))
            }
            ProximaType::Float16 | ProximaType::Float32 => {
                self.expect_tag(TAG_FLOAT)?;
                let enc = u32::from_be_bytes(self.take(4)?.try_into().unwrap());
                let bits = if enc & 0x8000_0000 != 0 {
                    enc ^ 0x8000_0000
                } else {
                    !enc
                };
                let f = f32::from_bits(bits);
                Ok(if matches!(t, ProximaType::Float16) {
                    ProximaValue::Float16(f)
                } else {
                    ProximaValue::Float32(f)
                })
            }
            ProximaType::Float64 => {
                self.expect_tag(TAG_FLOAT)?;
                let enc = u64::from_be_bytes(self.take(8)?.try_into().unwrap());
                let bits = if enc & 0x8000_0000_0000_0000 != 0 {
                    enc ^ 0x8000_0000_0000_0000
                } else {
                    !enc
                };
                Ok(ProximaValue::Float64(f64::from_bits(bits)))
            }
            ProximaType::String => {
                self.expect_tag(TAG_STRING)?;
                Ok(ProximaValue::String(String::from_utf8(
                    self.read_str_field()?,
                )?))
            }
            ProximaType::Symbol => {
                self.expect_tag(TAG_STRING)?;
                Ok(ProximaValue::Symbol(String::from_utf8(
                    self.read_str_field()?,
                )?))
            }
            ProximaType::Binary => {
                self.expect_tag(TAG_BINARY)?;
                Ok(ProximaValue::Binary(self.read_str_field()?))
            }
            ProximaType::Uuid => {
                self.expect_tag(TAG_UUID)?;
                Ok(ProximaValue::Uuid(self.take(16)?.try_into().unwrap()))
            }
            ProximaType::ULID => {
                self.expect_tag(TAG_ULID)?;
                Ok(ProximaValue::ULID(self.take(16)?.try_into().unwrap()))
            }
            other => bail!("key_codec: {:?} is not a decodable key component", other),
        }
    }

    fn read_temporal(&mut self, ctor: impl Fn(i64) -> ProximaValue) -> Result<ProximaValue> {
        self.expect_tag(TAG_SINT)?;
        let u = u64::from_be_bytes(self.take(8)?.try_into().unwrap()) ^ 0x8000_0000_0000_0000;
        Ok(ctor(u as i64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc(v: ProximaValue) -> Vec<u8> {
        encode_identity("t", "ks", &[v]).unwrap()
    }

    /// For each type, a value-sorted sample must encode to lexicographically
    /// ascending bytes (order preservation), and distinct values must not collide.
    fn assert_order_preserving(sorted: Vec<ProximaValue>) {
        let encoded: Vec<Vec<u8>> = sorted.into_iter().map(enc).collect();
        for w in encoded.windows(2) {
            assert!(
                w[0] < w[1],
                "not order-preserving: {:?} !< {:?}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn signed_ints_order_across_sign() {
        assert_order_preserving(vec![
            ProximaValue::Int32(i32::MIN),
            ProximaValue::Int32(-1000),
            ProximaValue::Int32(-1),
            ProximaValue::Int32(0),
            ProximaValue::Int32(1),
            ProximaValue::Int32(1000),
            ProximaValue::Int32(i32::MAX),
        ]);
        assert_order_preserving(vec![
            ProximaValue::Int8(i8::MIN),
            ProximaValue::Int8(-1),
            ProximaValue::Int8(0),
            ProximaValue::Int8(127),
        ]);
    }

    #[test]
    fn unsigned_ints_order() {
        assert_order_preserving(vec![
            ProximaValue::UInt64(0),
            ProximaValue::UInt64(1),
            ProximaValue::UInt64(u64::MAX / 2),
            ProximaValue::UInt64(u64::MAX),
        ]);
    }

    #[test]
    fn floats_order_incl_negatives() {
        assert_order_preserving(vec![
            ProximaValue::Float64(f64::NEG_INFINITY),
            ProximaValue::Float64(-1e10),
            ProximaValue::Float64(-1.5),
            ProximaValue::Float64(0.0),
            ProximaValue::Float64(1.5),
            ProximaValue::Float64(1e10),
            ProximaValue::Float64(f64::INFINITY),
        ]);
    }

    #[test]
    fn neg_zero_equals_zero() {
        assert_eq!(
            enc(ProximaValue::Float64(-0.0)),
            enc(ProximaValue::Float64(0.0))
        );
        assert_eq!(
            enc(ProximaValue::Float32(-0.0)),
            enc(ProximaValue::Float32(0.0))
        );
    }

    #[test]
    fn nan_is_rejected() {
        assert!(encode_identity("t", "ks", &[ProximaValue::Float64(f64::NAN)]).is_err());
    }

    #[test]
    fn strings_order_and_escape_embedded_nul() {
        assert_order_preserving(vec![
            ProximaValue::String("a".into()),
            ProximaValue::String("ab".into()),
            ProximaValue::String("b".into()),
        ]);
        // An embedded 0x00 must not act as a terminator: ("a\0b") and ("a","b")
        // must be distinguishable, and round-trip.
        let s = ProximaValue::String("a\u{0}b".into());
        let bytes = enc(s.clone());
        let back = decode_identity(&[ProximaType::String], &bytes).unwrap();
        assert_eq!(back.values, vec![s]);
    }

    #[test]
    fn composite_no_collision_across_boundary() {
        // The classic separator bug: ("a","bc") vs ("ab","c") must differ.
        let k1 = encode_identity(
            "t",
            "ks",
            &[
                ProximaValue::String("a".into()),
                ProximaValue::String("bc".into()),
            ],
        )
        .unwrap();
        let k2 = encode_identity(
            "t",
            "ks",
            &[
                ProximaValue::String("ab".into()),
                ProximaValue::String("c".into()),
            ],
        )
        .unwrap();
        assert_ne!(k1, k2);
    }

    #[test]
    fn tenant_scopes_are_isolated_and_ordered() {
        let a = encode_identity("tenant-a", "ks", &[ProximaValue::Int32(9999)]).unwrap();
        let b = encode_identity("tenant-b", "ks", &[ProximaValue::Int32(-1)]).unwrap();
        // Different tenants never collide, and the whole of tenant-a sorts before
        // tenant-b regardless of the column value.
        assert_ne!(a, b);
        assert!(a < b);
    }

    #[test]
    fn timeunit_normalized_to_same_instant() {
        let one_second = enc(ProximaValue::Timestamp(1, TimeUnit::Second));
        let thousand_millis = enc(ProximaValue::Timestamp(1_000, TimeUnit::Millisecond));
        assert_eq!(
            one_second, thousand_millis,
            "same instant must encode identically"
        );
    }

    #[test]
    fn structured_types_are_rejected() {
        assert!(encode_identity("t", "ks", &[ProximaValue::DenseVector(vec![1.0])]).is_err());
        assert!(encode_identity("t", "ks", &[ProximaValue::Array(vec![])]).is_err());
        assert!(encode_identity("t", "ks", &[ProximaValue::Decimal("1.0".into())]).is_err());
    }

    #[test]
    fn round_trip_scalars() {
        let cols = [
            (ProximaType::Int64, ProximaValue::Int64(-42)),
            (ProximaType::UInt32, ProximaValue::UInt32(7)),
            (ProximaType::Boolean, ProximaValue::Boolean(true)),
            (ProximaType::Float64, ProximaValue::Float64(3.5)),
            (
                ProximaType::String,
                ProximaValue::String("hÉllo\u{0}world".into()),
            ),
            (ProximaType::Uuid, ProximaValue::Uuid([7u8; 16])),
            (ProximaType::Date, ProximaValue::Date(-5)),
        ];
        let types: Vec<ProximaType> = cols.iter().map(|(t, _)| t.clone()).collect();
        let values: Vec<ProximaValue> = cols.iter().map(|(_, v)| v.clone()).collect();
        let bytes = encode_identity("acme", "users", &values).unwrap();
        let back = decode_identity(&types, &bytes).unwrap();
        assert_eq!(back.tenant, "acme");
        assert_eq!(back.keyspace, "users");
        assert_eq!(back.values, values);
    }

    #[test]
    fn round_trip_temporal_in_nanos() {
        let bytes =
            encode_identity("t", "ks", &[ProximaValue::Timestamp(2, TimeUnit::Second)]).unwrap();
        let back =
            decode_identity(&[ProximaType::Timestamp(TimeUnit::Nanosecond)], &bytes).unwrap();
        assert_eq!(
            back.values,
            vec![ProximaValue::Timestamp(2_000_000_000, TimeUnit::Nanosecond)]
        );
    }

    #[test]
    fn null_encodes_and_round_trips() {
        // A NULL in a nullable Int32 key column decodes back to Null.
        let bytes = encode_identity("t", "ks", &[ProximaValue::Null]).unwrap();
        let back = decode_identity(&[ProximaType::Int32], &bytes).unwrap();
        assert_eq!(back.values, vec![ProximaValue::Null]);
    }
}
