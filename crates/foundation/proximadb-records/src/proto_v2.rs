/*
 * Copyright 2025 Vijaykumar Singh
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

//! Canonical v2 protobuf codecs for the rich record/value boundary.
//!
//! This module keeps REST, gRPC, and internal handlers aligned on the same
//! `ProximaRecord`/`ProximaValue` contract from the multimodal overhaul spec
//! (§3 record envelope, §4.2 canonical type system, ADR-001/ADR-002).

use std::collections::HashMap;

use anyhow::{Result, anyhow, bail};
use proximadb_data_model::{ProximaValue, TimeUnit};
use proximadb_proto::proximadb_v2 as v2;

use crate::{EmbeddingCell, ProximaRecord, ProximaTree, ProximaTreeNode};

fn bytes16(bytes: &[u8], type_name: &str) -> Result<[u8; 16]> {
    bytes
        .try_into()
        .map_err(|_| anyhow!("{type_name} requires exactly 16 bytes, got {}", bytes.len()))
}

fn ns_to_millis(ns: i64) -> i64 {
    ns / 1_000_000
}

/// Convert a v2 protobuf `TypedValue` into the canonical Rust value enum.
pub fn typed_value_to_proxima(value: &v2::TypedValue) -> Result<ProximaValue> {
    use v2::typed_value::Value;

    let Some(inner) = value.value.as_ref() else {
        return Ok(ProximaValue::Null);
    };

    Ok(match inner {
        Value::TextValue(v) => ProximaValue::String(v.clone()),
        Value::IntegerValue(v) => ProximaValue::Int64(*v),
        Value::FloatValue(v) => ProximaValue::Float64(*v),
        Value::DecimalValue(v) => ProximaValue::Decimal(String::from_utf8_lossy(&v.value).into()),
        Value::BooleanValue(v) => ProximaValue::Boolean(*v),
        Value::TimestampValue(v) => ProximaValue::Timestamp(*v, TimeUnit::Microsecond),
        Value::TimestampTzValue(v) => {
            ProximaValue::TimestampTz(v.timestamp_us, TimeUnit::Microsecond)
        }
        Value::DateValue(v) => ProximaValue::Date(*v),
        Value::TimeValue(v) => ProximaValue::Time(*v, TimeUnit::Microsecond),
        Value::DurationValue(v) => ProximaValue::Timestamp(*v, TimeUnit::Microsecond),
        Value::IntervalValue(v) => ProximaValue::Struct(HashMap::from([
            ("months".to_string(), ProximaValue::Int32(v.months)),
            ("days".to_string(), ProximaValue::Int32(v.days)),
            ("nanos".to_string(), ProximaValue::Int64(v.nanos)),
        ])),
        Value::UuidValue(v) => ProximaValue::Uuid(bytes16(v, "uuid")?),
        Value::BinaryValue(v) => ProximaValue::Binary(v.clone()),
        Value::JsonValue(v) => ProximaValue::Json(serde_json::from_str(v)?),
        Value::TextArray(v) => {
            ProximaValue::Array(v.values.iter().cloned().map(ProximaValue::String).collect())
        }
        Value::IntegerArray(v) => {
            ProximaValue::Array(v.values.iter().copied().map(ProximaValue::Int64).collect())
        }
        Value::FloatArray(v) => ProximaValue::Array(
            v.values
                .iter()
                .copied()
                .map(ProximaValue::Float64)
                .collect(),
        ),
        Value::BooleanArray(v) => ProximaValue::Array(
            v.values
                .iter()
                .copied()
                .map(ProximaValue::Boolean)
                .collect(),
        ),
        Value::UuidArray(v) => ProximaValue::Array(
            v.values
                .iter()
                .map(|bytes| bytes16(bytes, "uuid").map(ProximaValue::Uuid))
                .collect::<Result<Vec<_>>>()?,
        ),
        Value::StringStringMap(v) => ProximaValue::Map(
            v.entries
                .iter()
                .map(|(k, v)| (k.clone(), ProximaValue::String(v.clone())))
                .collect(),
        ),
        Value::StringIntegerMap(v) => ProximaValue::Map(
            v.entries
                .iter()
                .map(|(k, v)| (k.clone(), ProximaValue::Int64(*v)))
                .collect(),
        ),
        Value::StringFloatMap(v) => ProximaValue::Map(
            v.entries
                .iter()
                .map(|(k, v)| (k.clone(), ProximaValue::Float64(*v)))
                .collect(),
        ),
        Value::GeoPoint(v) => ProximaValue::Struct(HashMap::from([
            ("latitude".to_string(), ProximaValue::Float64(v.latitude)),
            ("longitude".to_string(), ProximaValue::Float64(v.longitude)),
        ])),
        Value::GeoPolygon(v) => ProximaValue::Array(
            v.points
                .iter()
                .map(|point| {
                    ProximaValue::Struct(HashMap::from([
                        (
                            "latitude".to_string(),
                            ProximaValue::Float64(point.latitude),
                        ),
                        (
                            "longitude".to_string(),
                            ProximaValue::Float64(point.longitude),
                        ),
                    ]))
                })
                .collect(),
        ),
        Value::VectorValue(v) => {
            ProximaValue::DenseVector(v.values.iter().map(|v| *v as f32).collect())
        }
        Value::SparseVectorValue(v) => ProximaValue::SparseVector {
            indices: v.indices.clone(),
            values: v.values.clone(),
        },
        Value::Int8Value(v) => ProximaValue::Int8((*v).try_into()?),
        Value::Int16Value(v) => ProximaValue::Int16((*v).try_into()?),
        Value::Int32Value(v) => ProximaValue::Int32(*v),
        Value::Uint8Value(v) => ProximaValue::UInt8((*v).try_into()?),
        Value::Uint16Value(v) => ProximaValue::UInt16((*v).try_into()?),
        Value::Uint32Value(v) => ProximaValue::UInt32(*v),
        Value::Uint64Value(v) => ProximaValue::UInt64(*v),
        Value::Float32Value(v) => ProximaValue::Float32(*v),
        Value::SymbolValue(v) => ProximaValue::Symbol(v.clone()),
        Value::JsonbValue(v) => ProximaValue::Jsonb(ProximaValue::from_jsonb_slice(v)?),
        Value::UlidValue(v) => ProximaValue::ULID(bytes16(v, "ulid")?),
        Value::ArrayValue(v) => ProximaValue::Array(
            v.values
                .iter()
                .map(typed_value_to_proxima)
                .collect::<Result<Vec<_>>>()?,
        ),
        Value::MapValue(v) => ProximaValue::Map(typed_map_to_proxima(&v.entries)?),
        Value::StructValue(v) => ProximaValue::Struct(typed_map_to_proxima(&v.entries)?),
        Value::BinaryVectorValue(v) => ProximaValue::BinaryVector(v.clone()),
        Value::IsNull(true) => ProximaValue::Null,
        Value::IsNull(false) => bail!("is_null=false is not a value"),
    })
}

fn typed_map_to_proxima(
    map: &HashMap<String, v2::TypedValue>,
) -> Result<HashMap<String, ProximaValue>> {
    map.iter()
        .map(|(k, v)| typed_value_to_proxima(v).map(|value| (k.clone(), value)))
        .collect()
}

/// Convert a canonical Rust value into a v2 protobuf `TypedValue`.
pub fn proxima_value_to_typed_value(value: &ProximaValue) -> v2::TypedValue {
    use v2::ColumnDataType;
    use v2::typed_value::Value;

    let (declared_type, value) = match value {
        ProximaValue::Boolean(v) => (ColumnDataType::Boolean, Value::BooleanValue(*v)),
        ProximaValue::Int8(v) => (ColumnDataType::Int8, Value::Int8Value(*v as i32)),
        ProximaValue::Int16(v) => (ColumnDataType::Int16, Value::Int16Value(*v as i32)),
        ProximaValue::Int32(v) => (ColumnDataType::Int32, Value::Int32Value(*v)),
        ProximaValue::Int64(v) => (ColumnDataType::Integer, Value::IntegerValue(*v)),
        ProximaValue::UInt8(v) => (ColumnDataType::Uint8, Value::Uint8Value(*v as u32)),
        ProximaValue::UInt16(v) => (ColumnDataType::Uint16, Value::Uint16Value(*v as u32)),
        ProximaValue::UInt32(v) => (ColumnDataType::Uint32, Value::Uint32Value(*v)),
        ProximaValue::UInt64(v) => (ColumnDataType::Uint64, Value::Uint64Value(*v)),
        ProximaValue::Float16(v) | ProximaValue::Float32(v) => {
            (ColumnDataType::Float32, Value::Float32Value(*v))
        }
        ProximaValue::Float64(v) => (ColumnDataType::Float, Value::FloatValue(*v)),
        ProximaValue::Decimal(v) => (
            ColumnDataType::Decimal,
            Value::DecimalValue(v2::DecimalValue {
                value: v.as_bytes().to_vec(),
                precision: 38,
                scale: 18,
            }),
        ),
        ProximaValue::String(v) => (ColumnDataType::Text, Value::TextValue(v.clone())),
        ProximaValue::Symbol(v) => (ColumnDataType::Symbol, Value::SymbolValue(v.clone())),
        ProximaValue::Binary(v) => (ColumnDataType::Binary, Value::BinaryValue(v.clone())),
        ProximaValue::Date(v) => (ColumnDataType::Date, Value::DateValue(*v)),
        ProximaValue::Time(v, _) => (ColumnDataType::Time, Value::TimeValue(*v)),
        ProximaValue::Timestamp(v, _) => (ColumnDataType::Timestamp, Value::TimestampValue(*v)),
        ProximaValue::TimestampTz(v, _) => (
            ColumnDataType::TimestampTz,
            Value::TimestampTzValue(v2::TimestampTzValue {
                timestamp_us: *v,
                timezone: "UTC".to_string(),
            }),
        ),
        ProximaValue::Uuid(v) => (ColumnDataType::Uuid, Value::UuidValue(v.to_vec())),
        ProximaValue::ULID(v) => (ColumnDataType::Ulid, Value::UlidValue(v.to_vec())),
        ProximaValue::Json(v) => (
            ColumnDataType::Json,
            Value::JsonValue(serde_json::to_string(v).unwrap_or_else(|_| "null".to_string())),
        ),
        ProximaValue::Jsonb(v) => (
            ColumnDataType::Jsonb,
            Value::JsonbValue(ProximaValue::to_jsonb_vec(v).unwrap_or_default()),
        ),
        ProximaValue::Array(v) => (
            ColumnDataType::ArrayAny,
            Value::ArrayValue(v2::TypedValueArray {
                values: v.iter().map(proxima_value_to_typed_value).collect(),
            }),
        ),
        ProximaValue::Map(v) => (
            ColumnDataType::MapStringAny,
            Value::MapValue(v2::TypedValueMap {
                entries: proxima_map_to_typed_map(v),
            }),
        ),
        ProximaValue::Struct(v) => (
            ColumnDataType::Struct,
            Value::StructValue(v2::TypedValueMap {
                entries: proxima_map_to_typed_map(v),
            }),
        ),
        ProximaValue::DenseVector(v) => (
            ColumnDataType::Vector,
            Value::VectorValue(v2::FloatArray {
                values: v.iter().map(|v| *v as f64).collect(),
            }),
        ),
        ProximaValue::SparseVector { indices, values } => (
            ColumnDataType::SparseVector,
            Value::SparseVectorValue(v2::SparseVector {
                indices: indices.clone(),
                values: values.clone(),
                dimension: 0,
            }),
        ),
        ProximaValue::BinaryVector(v) => (
            ColumnDataType::BinaryVector,
            Value::BinaryVectorValue(v.clone()),
        ),
        ProximaValue::Null => (ColumnDataType::ColumnTypeUnspecified, Value::IsNull(true)),
    };

    v2::TypedValue {
        declared_type: declared_type as i32,
        value: Some(value),
    }
}

fn proxima_map_to_typed_map(
    map: &HashMap<String, ProximaValue>,
) -> HashMap<String, v2::TypedValue> {
    map.iter()
        .map(|(k, v)| (k.clone(), proxima_value_to_typed_value(v)))
        .collect()
}

/// Convert a v2 protobuf record into the canonical Rust `ProximaRecord`.
/// Smallest accepted `expires_at_ms`: 1e11 ms = 1973-03-03T09:46:40Z.
///
/// Below this is not a plausible expiry for a record written today, and the
/// common way to produce one is passing epoch SECONDS where the field is
/// MILLISECONDS (1_700_000_000 s read as ms is January 1970). Either way the
/// write would succeed while the record is already dead.
pub const MIN_PLAUSIBLE_EXPIRES_AT_MS: i64 = 100_000_000_000;

/// Map a client-supplied record expiry (epoch ms) onto `valid_to_ns`.
///
/// The single rule for every inbound surface (REST v2 and gRPC v2).
/// Rejected, so the write fails instead of silently producing a dead record:
/// * `<= 0`: `valid_to_ns = Some(0)` is the canonical tombstone
///   (`is_record_dead`), so 0 would delete-on-insert;
/// * `< MIN_PLAUSIBLE_EXPIRES_AT_MS`: almost certainly seconds, not ms.
///
/// A plausible expiry that is merely in the past is accepted (clock skew);
/// the read path then treats the record as expired. Values past year 2262
/// saturate to `i64::MAX` ns, which reads as "never expires".
pub fn expires_at_ms_to_valid_to_ns(
    expires_at_ms: Option<i64>,
) -> std::result::Result<Option<i64>, String> {
    match expires_at_ms {
        None => Ok(None),
        Some(ms) if ms <= 0 => Err(format!(
            "expires_at_ms must be a positive epoch-millisecond timestamp, got {ms} \
             (0 is the internal tombstone encoding; use the delete API)"
        )),
        Some(ms) if ms < MIN_PLAUSIBLE_EXPIRES_AT_MS => Err(format!(
            "expires_at_ms {ms} is before 1973; it looks like epoch seconds, but the \
             field is epoch milliseconds"
        )),
        Some(ms) => Ok(Some(ms.saturating_mul(1_000_000))),
    }
}

pub fn proto_record_to_envelope(proto: &v2::ProximaRecord) -> Result<ProximaRecord> {
    let mut record = ProximaRecord::default();
    record.oid = proto.id.clone();
    record.record_version = proto.version.unwrap_or(0) as u64;
    record.created_at_ns = proto.timestamp_ms.saturating_mul(1_000_000);
    record.updated_at_ns = proto
        .updated_at_ms
        .map(|v| v.saturating_mul(1_000_000))
        .unwrap_or(record.created_at_ns);
    record.valid_to_ns =
        expires_at_ms_to_valid_to_ns(proto.expires_at_ms).map_err(|e| anyhow!(e))?;
    record.origin = proto.source.clone();
    record.actor = proto.created_by.clone();
    record.variation_id = proto.schema_id.clone();

    record.props = proto
        .props
        .iter()
        .map(|(k, v)| {
            typed_value_to_proxima(v).map(|value| (k.clone(), ProximaTreeNode::Value(value)))
        })
        .collect::<Result<ProximaTree>>()?;

    for text in &proto.text_fields {
        record.props.insert(
            text.name.clone(),
            ProximaTreeNode::Value(ProximaValue::String(text.content.clone())),
        );
    }

    if !proto.vector.is_empty() {
        record.embeddings.push(EmbeddingCell::new_fp32(
            "default",
            "dense_vector",
            proto.vector_dimension.unwrap_or(proto.vector.len() as u32),
            proto.vector.clone(),
        ));
    }

    if let Some(sparse) = proto.sparse_vector.as_ref() {
        record.props.insert(
            "_sparse_vector".to_string(),
            ProximaTreeNode::Value(ProximaValue::SparseVector {
                indices: sparse.indices.clone(),
                values: sparse.values.clone(),
            }),
        );
    }

    Ok(record)
}

/// Convert a canonical Rust `ProximaRecord` into the v2 protobuf record shape.
pub fn envelope_to_proto_record(record: &ProximaRecord) -> v2::ProximaRecord {
    // INT-2.5b: v2 proto's `vector: Vec<f32>` is a v1-style fp32 field.
    // Promote non-Fp32 variants on the way out; native precision-aware
    // proto fields land in a future v3 proto.
    let vector: Vec<f32> = record
        .embeddings
        .first()
        .map(|embedding| embedding.values.to_fp32_owned())
        .unwrap_or_default();

    v2::ProximaRecord {
        id: record.oid.clone(),
        vector_dimension: (!vector.is_empty()).then_some(vector.len() as u32),
        vector,
        sparse_vector: None,
        props: record
            .props
            .iter()
            .filter_map(|(k, v)| match v {
                ProximaTreeNode::Value(value) => {
                    Some((k.clone(), proxima_value_to_typed_value(value)))
                }
                ProximaTreeNode::Object(_) => None,
            })
            .collect(),
        text_fields: Vec::new(),
        timestamp_ms: ns_to_millis(record.created_at_ns),
        updated_at_ms: Some(ns_to_millis(record.updated_at_ns)),
        expires_at_ms: record.valid_to_ns.map(ns_to_millis),
        version: Some(record.record_version as u32),
        source: record.origin.clone(),
        source_type: None,
        schema_id: record.variation_id.clone(),
        partition_key: None,
        partition_values: HashMap::new(),
        created_by: record.actor.clone(),
        updated_by: None,
        custom_metadata: HashMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_value_round_trips_rich_values() {
        let values = vec![
            ProximaValue::UInt64(42),
            ProximaValue::Float32(1.5),
            ProximaValue::Symbol("tag".to_string()),
            ProximaValue::ULID([7; 16]),
            ProximaValue::Jsonb(serde_json::json!({"nested": [1, true]})),
            ProximaValue::Array(vec![
                ProximaValue::Int8(-3),
                ProximaValue::String("x".into()),
            ]),
            ProximaValue::Struct(HashMap::from([(
                "answer".to_string(),
                ProximaValue::Int32(42),
            )])),
            ProximaValue::BinaryVector(vec![0b1010_1010]),
        ];

        for value in values {
            let typed = proxima_value_to_typed_value(&value);
            let round_tripped = typed_value_to_proxima(&typed).expect("decode typed value");
            assert_eq!(round_tripped, value);
        }
    }

    #[test]
    fn typed_value_decodes_wire_only_scalar_array_map_and_geo_shapes() {
        use v2::typed_value::Value;

        let uuid = [9_u8; 16];
        let cases = vec![
            (
                Value::TextValue("hello".to_string()),
                ProximaValue::String("hello".to_string()),
            ),
            (Value::IntegerValue(42), ProximaValue::Int64(42)),
            (Value::FloatValue(3.5), ProximaValue::Float64(3.5)),
            (
                Value::DecimalValue(v2::DecimalValue {
                    value: b"12.3400".to_vec(),
                    precision: 6,
                    scale: 4,
                }),
                ProximaValue::Decimal("12.3400".to_string()),
            ),
            (Value::BooleanValue(true), ProximaValue::Boolean(true)),
            (
                Value::TimestampValue(123),
                ProximaValue::Timestamp(123, TimeUnit::Microsecond),
            ),
            (
                Value::TimestampTzValue(v2::TimestampTzValue {
                    timestamp_us: 456,
                    timezone: "America/Chicago".to_string(),
                }),
                ProximaValue::TimestampTz(456, TimeUnit::Microsecond),
            ),
            (Value::DateValue(20_000), ProximaValue::Date(20_000)),
            (
                Value::TimeValue(987),
                ProximaValue::Time(987, TimeUnit::Microsecond),
            ),
            (
                Value::DurationValue(654),
                ProximaValue::Timestamp(654, TimeUnit::Microsecond),
            ),
            (
                Value::IntervalValue(v2::IntervalValue {
                    months: 2,
                    days: 3,
                    nanos: 4,
                }),
                ProximaValue::Struct(HashMap::from([
                    ("months".to_string(), ProximaValue::Int32(2)),
                    ("days".to_string(), ProximaValue::Int32(3)),
                    ("nanos".to_string(), ProximaValue::Int64(4)),
                ])),
            ),
            (Value::UuidValue(uuid.to_vec()), ProximaValue::Uuid(uuid)),
            (
                Value::BinaryValue(vec![1, 2, 3]),
                ProximaValue::Binary(vec![1, 2, 3]),
            ),
            (
                Value::JsonValue(r#"{"nested":true}"#.to_string()),
                ProximaValue::Json(serde_json::json!({"nested": true})),
            ),
            (
                Value::TextArray(v2::TextArray {
                    values: vec!["a".to_string(), "b".to_string()],
                }),
                ProximaValue::Array(vec![
                    ProximaValue::String("a".to_string()),
                    ProximaValue::String("b".to_string()),
                ]),
            ),
            (
                Value::IntegerArray(v2::IntegerArray { values: vec![1, 2] }),
                ProximaValue::Array(vec![ProximaValue::Int64(1), ProximaValue::Int64(2)]),
            ),
            (
                Value::FloatArray(v2::FloatArray {
                    values: vec![1.25, 2.5],
                }),
                ProximaValue::Array(vec![
                    ProximaValue::Float64(1.25),
                    ProximaValue::Float64(2.5),
                ]),
            ),
            (
                Value::BooleanArray(v2::BooleanArray {
                    values: vec![true, false],
                }),
                ProximaValue::Array(vec![
                    ProximaValue::Boolean(true),
                    ProximaValue::Boolean(false),
                ]),
            ),
            (
                Value::UuidArray(v2::UuidArray {
                    values: vec![uuid.to_vec()],
                }),
                ProximaValue::Array(vec![ProximaValue::Uuid(uuid)]),
            ),
            (
                Value::StringStringMap(v2::StringStringMap {
                    entries: HashMap::from([("k".to_string(), "v".to_string())]),
                }),
                ProximaValue::Map(HashMap::from([(
                    "k".to_string(),
                    ProximaValue::String("v".to_string()),
                )])),
            ),
            (
                Value::StringIntegerMap(v2::StringIntegerMap {
                    entries: HashMap::from([("n".to_string(), 7)]),
                }),
                ProximaValue::Map(HashMap::from([("n".to_string(), ProximaValue::Int64(7))])),
            ),
            (
                Value::StringFloatMap(v2::StringFloatMap {
                    entries: HashMap::from([("f".to_string(), 1.5)]),
                }),
                ProximaValue::Map(HashMap::from([(
                    "f".to_string(),
                    ProximaValue::Float64(1.5),
                )])),
            ),
            (
                Value::GeoPoint(v2::GeoPoint {
                    latitude: 41.0,
                    longitude: -87.0,
                    altitude: Some(12.0),
                }),
                ProximaValue::Struct(HashMap::from([
                    ("latitude".to_string(), ProximaValue::Float64(41.0)),
                    ("longitude".to_string(), ProximaValue::Float64(-87.0)),
                ])),
            ),
            (
                Value::GeoPolygon(v2::GeoPolygon {
                    points: vec![v2::GeoPoint {
                        latitude: 1.0,
                        longitude: 2.0,
                        altitude: None,
                    }],
                }),
                ProximaValue::Array(vec![ProximaValue::Struct(HashMap::from([
                    ("latitude".to_string(), ProximaValue::Float64(1.0)),
                    ("longitude".to_string(), ProximaValue::Float64(2.0)),
                ]))]),
            ),
            (
                Value::VectorValue(v2::FloatArray {
                    values: vec![0.25, 0.5],
                }),
                ProximaValue::DenseVector(vec![0.25, 0.5]),
            ),
            (
                Value::SparseVectorValue(v2::SparseVector {
                    indices: vec![2, 4],
                    values: vec![0.2, 0.4],
                    dimension: 8,
                }),
                ProximaValue::SparseVector {
                    indices: vec![2, 4],
                    values: vec![0.2, 0.4],
                },
            ),
            (Value::Int16Value(-123), ProximaValue::Int16(-123)),
            (Value::Uint8Value(200), ProximaValue::UInt8(200)),
            (Value::Uint16Value(65_000), ProximaValue::UInt16(65_000)),
            (Value::Uint32Value(700), ProximaValue::UInt32(700)),
        ];

        for (wire, expected) in cases {
            let typed = v2::TypedValue {
                declared_type: 0,
                value: Some(wire),
            };
            assert_eq!(typed_value_to_proxima(&typed).unwrap(), expected);
        }
    }

    #[test]
    fn typed_value_decodes_nested_map_null_and_reports_invalid_wire_values() {
        use v2::typed_value::Value;

        assert_eq!(
            typed_value_to_proxima(&v2::TypedValue::default()).unwrap(),
            ProximaValue::Null
        );
        assert_eq!(
            typed_value_to_proxima(&v2::TypedValue {
                declared_type: 0,
                value: Some(Value::MapValue(v2::TypedValueMap {
                    entries: HashMap::from([(
                        "inner".to_string(),
                        proxima_value_to_typed_value(&ProximaValue::String("v".to_string())),
                    )]),
                })),
            })
            .unwrap(),
            ProximaValue::Map(HashMap::from([(
                "inner".to_string(),
                ProximaValue::String("v".to_string()),
            )]))
        );
        assert_eq!(
            typed_value_to_proxima(&v2::TypedValue {
                declared_type: 0,
                value: Some(Value::IsNull(true)),
            })
            .unwrap(),
            ProximaValue::Null
        );

        for invalid in [
            Value::UuidValue(vec![1, 2, 3]),
            Value::UuidArray(v2::UuidArray {
                values: vec![vec![1, 2, 3]],
            }),
            Value::JsonValue("{not-json".to_string()),
            Value::Int8Value(i8::MAX as i32 + 1),
            Value::Uint8Value(u8::MAX as u32 + 1),
            Value::IsNull(false),
        ] {
            assert!(
                typed_value_to_proxima(&v2::TypedValue {
                    declared_type: 0,
                    value: Some(invalid),
                })
                .is_err()
            );
        }
    }

    #[test]
    fn proto_record_uses_props_not_legacy_metadata() {
        let mut proto = v2::ProximaRecord {
            id: "r1".to_string(),
            vector: vec![0.1, 0.2],
            timestamp_ms: 10,
            ..Default::default()
        };
        proto.props.insert(
            "price".to_string(),
            proxima_value_to_typed_value(&ProximaValue::Decimal("12.34".to_string())),
        );

        let envelope = proto_record_to_envelope(&proto).expect("decode proto record");
        assert_eq!(envelope.oid, "r1");
        assert_eq!(envelope.embeddings[0].as_fp32_slice(), &[0.1, 0.2]);
        assert_eq!(
            envelope.props.get("price"),
            Some(&ProximaTreeNode::Value(ProximaValue::Decimal(
                "12.34".to_string()
            )))
        );
    }

    #[test]
    fn proto_record_to_envelope_preserves_text_sparse_temporal_and_provenance_fields() {
        let mut proto = v2::ProximaRecord {
            id: "r2".to_string(),
            vector: vec![1.0, 2.0],
            vector_dimension: Some(16),
            sparse_vector: Some(v2::SparseVector {
                indices: vec![1, 9],
                values: vec![0.5, 0.9],
                dimension: 32,
            }),
            timestamp_ms: 100,
            updated_at_ms: Some(200),
            // A plausible epoch-ms expiry (the old fixture, 300 ms, is now
            // rejected as below MIN_PLAUSIBLE_EXPIRES_AT_MS).
            expires_at_ms: Some(1_700_000_000_300),
            version: Some(4),
            source: Some("cdc".to_string()),
            schema_id: Some("shape-a".to_string()),
            created_by: Some("alice".to_string()),
            text_fields: vec![v2::TextField {
                name: "body".to_string(),
                content: "hello body".to_string(),
                storage_hint: 0,
                chunk_count: None,
                chunk_reference: None,
            }],
            ..Default::default()
        };
        proto.props.insert(
            "status".to_string(),
            proxima_value_to_typed_value(&ProximaValue::String("ready".to_string())),
        );

        let envelope = proto_record_to_envelope(&proto).unwrap();

        assert_eq!(envelope.oid, "r2");
        assert_eq!(envelope.record_version, 4);
        assert_eq!(envelope.created_at_ns, 100_000_000);
        assert_eq!(envelope.updated_at_ns, 200_000_000);
        assert_eq!(envelope.valid_to_ns, Some(1_700_000_000_300_000_000));
        assert_eq!(envelope.origin.as_deref(), Some("cdc"));
        assert_eq!(envelope.actor.as_deref(), Some("alice"));
        assert_eq!(envelope.variation_id.as_deref(), Some("shape-a"));
        assert_eq!(envelope.embeddings[0].dim, 16);
        assert_eq!(
            envelope.props.get("body"),
            Some(&ProximaTreeNode::Value(ProximaValue::String(
                "hello body".to_string()
            )))
        );
        assert_eq!(
            envelope.props.get("_sparse_vector"),
            Some(&ProximaTreeNode::Value(ProximaValue::SparseVector {
                indices: vec![1, 9],
                values: vec![0.5, 0.9],
            }))
        );
    }

    #[test]
    fn proto_record_to_envelope_rejects_non_positive_expiry() {
        // expires_at_ms = 0 used to become valid_to_ns = Some(0): the
        // tombstone. The insert "succeeded" and the record was invisible.
        for bad in [0_i64, -5] {
            let proto = v2::ProximaRecord {
                id: "r".to_string(),
                vector: vec![1.0],
                expires_at_ms: Some(bad),
                ..Default::default()
            };
            let err = proto_record_to_envelope(&proto).unwrap_err().to_string();
            assert!(err.contains("expires_at_ms"), "{err}");
        }
        let ok = v2::ProximaRecord {
            id: "r".to_string(),
            vector: vec![1.0],
            expires_at_ms: Some(1_700_000_000_000),
            ..Default::default()
        };
        assert_eq!(
            proto_record_to_envelope(&ok).unwrap().valid_to_ns,
            Some(1_700_000_000_000_000_000)
        );
        // Epoch SECONDS in the ms field: rejected, not written as a 1970 expiry.
        let secs = v2::ProximaRecord {
            id: "r".to_string(),
            vector: vec![1.0],
            expires_at_ms: Some(1_700_000_000),
            ..Default::default()
        };
        let err = proto_record_to_envelope(&secs).unwrap_err().to_string();
        assert!(err.contains("epoch seconds"), "{err}");
        // The floor itself is accepted; one below it is not.
        assert!(expires_at_ms_to_valid_to_ns(Some(MIN_PLAUSIBLE_EXPIRES_AT_MS)).is_ok());
        assert!(expires_at_ms_to_valid_to_ns(Some(MIN_PLAUSIBLE_EXPIRES_AT_MS - 1)).is_err());
        // Far future saturates to "never expires", never to the tombstone.
        assert_eq!(
            expires_at_ms_to_valid_to_ns(Some(i64::MAX)),
            Ok(Some(i64::MAX))
        );
        let none = v2::ProximaRecord {
            id: "r".to_string(),
            vector: vec![1.0],
            ..Default::default()
        };
        assert_eq!(proto_record_to_envelope(&none).unwrap().valid_to_ns, None);
    }

    #[test]
    fn envelope_to_proto_record_uses_only_value_props_and_preserves_record_fields() {
        let mut nested = ProximaTree::new();
        nested.insert(
            "ignored".to_string(),
            ProximaTreeNode::Value(ProximaValue::String("nested".to_string())),
        );
        let mut record = ProximaRecord {
            oid: "r3".to_string(),
            record_version: 8,
            created_at_ns: 1_234_000_000,
            updated_at_ns: 2_345_000_000,
            valid_to_ns: Some(3_456_000_000),
            origin: Some("api".to_string()),
            actor: Some("bob".to_string()),
            variation_id: Some("shape-b".to_string()),
            ..ProximaRecord::default()
        };
        record.props.insert(
            "flat".to_string(),
            ProximaTreeNode::Value(ProximaValue::Int64(11)),
        );
        record
            .props
            .insert("nested".to_string(), ProximaTreeNode::Object(nested));
        record
            .embeddings
            .push(EmbeddingCell::new_fp32("m", "text", 2, vec![0.1, 0.2]));

        let proto = envelope_to_proto_record(&record);

        assert_eq!(proto.id, "r3");
        assert_eq!(proto.vector, vec![0.1, 0.2]);
        assert_eq!(proto.vector_dimension, Some(2));
        assert_eq!(proto.timestamp_ms, 1_234);
        assert_eq!(proto.updated_at_ms, Some(2_345));
        assert_eq!(proto.expires_at_ms, Some(3_456));
        assert_eq!(proto.version, Some(8));
        assert_eq!(proto.source.as_deref(), Some("api"));
        assert_eq!(proto.created_by.as_deref(), Some("bob"));
        assert_eq!(proto.schema_id.as_deref(), Some("shape-b"));
        assert!(proto.props.contains_key("flat"));
        assert!(!proto.props.contains_key("nested"));
    }
}
