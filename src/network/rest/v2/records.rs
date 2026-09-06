/*
 * Copyright 2025 ProximaDB
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

//! ProximaRecord REST handlers for v2 API
//!
//! This module provides REST endpoints for inserting and searching ProximaRecords,
//! the v2 record type with full type system support.
//!
//! ## Endpoints
//!
//! - `POST /api/v2/collections/{collection}/records/batch` - Insert ProximaRecords
//! - `POST /api/v2/collections/{collection}/search` - Search with typed filters
//!
//! ## ProximaRecord Structure
//!
//! ProximaRecord replaces VectorRecord with:
//! - `props`: Canonical rich fields (INTEGER, FLOAT, DECIMAL, UUID, JSONB, etc.)
//! - `text_fields`: Dedicated TEXT column storage with chunking support
//! - Schema validation at insert time (when enabled)

use axum::{
    Json,
    extract::{Extension, Path, Query, State},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, error, info};
use utoipa::{IntoParams, ToSchema};

use crate::api_handlers::{
    RichFilterCondition, RichFilterOperator, RichRecordBatchRequest, RichRecordDeleteBatchRequest,
    RichRecordGetRequest, RichSearchRequest,
};
use crate::errors::{ApiError, ApiResult};
use crate::network::auth::middleware::DataPlaneCapability;
use crate::network::middleware::tenant::TenantContext;
use crate::network::rest::canonical::handlers::AppState;
use crate::services::{
    WriteDurabilityRequirement, WriteIntent, WriteLaneRouter, WriteOperationKind,
};
use proximadb_data_model::{ProximaValue, TimeUnit};
use proximadb_records::{EmbeddingCell, ProximaRecord, ProximaTreeNode};

#[cfg(test)]
/// Convert a JSON value to SqlValue for legacy conversion tests.
fn json_to_sql_value(value: &serde_json::Value) -> crate::proto::proximadb_v1::SqlValue {
    use crate::proto::proximadb_v1::sql_value::Value;
    use crate::proto::proximadb_v1::{SqlArray, SqlObject, SqlValue};

    let inner = match value {
        serde_json::Value::Null => Value::NullValue(0),
        serde_json::Value::Bool(b) => Value::BoolValue(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int64Value(i)
            } else if let Some(f) = n.as_f64() {
                Value::NumberValue(f)
            } else {
                Value::StringValue(n.to_string())
            }
        }
        serde_json::Value::String(s) => Value::StringValue(s.clone()),
        serde_json::Value::Array(arr) => {
            let values: Vec<SqlValue> = arr.iter().map(json_to_sql_value).collect();
            Value::ArrayValue(SqlArray { values })
        }
        serde_json::Value::Object(obj) => {
            let fields: HashMap<String, SqlValue> = obj
                .iter()
                .map(|(k, v)| (k.clone(), json_to_sql_value(v)))
                .collect();
            Value::ObjectValue(SqlObject { fields })
        }
    };

    SqlValue { value: Some(inner) }
}

#[cfg(test)]
/// Convert SqlValue back to JSON for legacy conversion tests.
fn sql_value_to_json(
    value: &crate::proto::proximadb_v1::SqlValue,
) -> Result<serde_json::Value, ApiError> {
    // TD-PROTO-2 consolidation: delegates to the canonical rendering
    // (non-finite floats render as null — the old 500-error path is gone;
    // pinned in records' contract test).
    Ok(proximadb_records::conversions::sql_value_to_json(value))
}

#[cfg(test)]
/// Convert a JSON value to a FilterClause value.
fn json_to_filter_clause_value(
    value: &serde_json::Value,
) -> Option<crate::proto::proximadb_v1::filter_clause::Value> {
    use crate::proto::proximadb_v1::filter_clause::Value;

    match value {
        serde_json::Value::String(s) => Some(Value::StringValue(s.clone())),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(Value::IntValue(i))
            } else {
                n.as_f64().map(Value::DoubleValue)
            }
        }
        serde_json::Value::Bool(b) => Some(Value::BoolValue(*b)),
        _ => None, // Arrays and objects not directly supported in FilterClause
    }
}

#[cfg(test)]
/// Convert TypedFilter list to FilterClause list for legacy MetadataFilter tests.
///
/// Supports the following operators:
/// - eq: Equals
/// - neq: Not equals
/// - gt: Greater than
/// - gte: Greater than or equal
/// - lt: Less than
/// - lte: Less than or equal
/// - contains: String/array contains (substring match)
/// - in: Value is in a list
/// - between: Value is between two bounds (converted to gte + lte)
/// - starts_with: String starts with prefix (converted to contains)
/// - ends_with: String ends with suffix (converted to contains)
fn convert_typed_filters_to_clauses(
    typed_filters: &[TypedFilter],
) -> Result<Vec<crate::proto::proximadb_v1::FilterClause>, ApiError> {
    use crate::proto::proximadb_v1::{ComparisonOp, FilterClause, filter_clause::Value};

    let mut clauses = Vec::new();

    for filter in typed_filters {
        let value_json = rest_value_payload(&filter.value);
        let value_upper_json = filter.value_upper.as_ref().map(rest_value_payload);

        match filter.op.as_str() {
            "eq" => {
                if let Some(value) = json_to_filter_clause_value(&value_json) {
                    clauses.push(FilterClause {
                        field: filter.field.clone(),
                        op: ComparisonOp::Eq as i32,
                        value: Some(value),
                    });
                }
            }
            "neq" => {
                if let Some(value) = json_to_filter_clause_value(&value_json) {
                    clauses.push(FilterClause {
                        field: filter.field.clone(),
                        op: ComparisonOp::Ne as i32,
                        value: Some(value),
                    });
                }
            }
            "gt" => {
                if let Some(value) = json_to_filter_clause_value(&value_json) {
                    clauses.push(FilterClause {
                        field: filter.field.clone(),
                        op: ComparisonOp::Gt as i32,
                        value: Some(value),
                    });
                }
            }
            "gte" => {
                if let Some(value) = json_to_filter_clause_value(&value_json) {
                    clauses.push(FilterClause {
                        field: filter.field.clone(),
                        op: ComparisonOp::Gte as i32,
                        value: Some(value),
                    });
                }
            }
            "lt" => {
                if let Some(value) = json_to_filter_clause_value(&value_json) {
                    clauses.push(FilterClause {
                        field: filter.field.clone(),
                        op: ComparisonOp::Lt as i32,
                        value: Some(value),
                    });
                }
            }
            "lte" => {
                if let Some(value) = json_to_filter_clause_value(&value_json) {
                    clauses.push(FilterClause {
                        field: filter.field.clone(),
                        op: ComparisonOp::Lte as i32,
                        value: Some(value),
                    });
                }
            }
            "between" => {
                // "between" requires both value and value_upper
                // Convert to two clauses: field >= value AND field <= value_upper
                let value_upper = value_upper_json.as_ref().ok_or_else(|| {
                    ApiError::InvalidArgument(
                        "Filter operator 'between' requires 'value_upper' to be specified"
                            .to_string(),
                    )
                })?;

                if let Some(lower_value) = json_to_filter_clause_value(&value_json) {
                    clauses.push(FilterClause {
                        field: filter.field.clone(),
                        op: ComparisonOp::Gte as i32,
                        value: Some(lower_value),
                    });
                }

                if let Some(upper_value) = json_to_filter_clause_value(value_upper) {
                    clauses.push(FilterClause {
                        field: filter.field.clone(),
                        op: ComparisonOp::Lte as i32,
                        value: Some(upper_value),
                    });
                }
            }
            "contains" => {
                // Contains for string substring matching
                if let Some(value) = json_to_filter_clause_value(&value_json) {
                    clauses.push(FilterClause {
                        field: filter.field.clone(),
                        op: ComparisonOp::Contains as i32,
                        value: Some(value),
                    });
                }
            }
            "starts_with" => {
                // starts_with is implemented using Contains operator
                // The backend should interpret this as prefix matching
                // We encode the intent by using Contains with the prefix value
                if let serde_json::Value::String(s) = &value_json {
                    clauses.push(FilterClause {
                        field: filter.field.clone(),
                        op: ComparisonOp::Contains as i32,
                        value: Some(Value::StringValue(format!("^{}", s))),
                    });
                } else {
                    return Err(ApiError::InvalidArgument(
                        "Filter operator 'starts_with' requires a string value".to_string(),
                    ));
                }
            }
            "ends_with" => {
                // ends_with is implemented using Contains operator
                // The backend should interpret this as suffix matching
                // We encode the intent by using Contains with the suffix value
                if let serde_json::Value::String(s) = &value_json {
                    clauses.push(FilterClause {
                        field: filter.field.clone(),
                        op: ComparisonOp::Contains as i32,
                        value: Some(Value::StringValue(format!("{}$", s))),
                    });
                } else {
                    return Err(ApiError::InvalidArgument(
                        "Filter operator 'ends_with' requires a string value".to_string(),
                    ));
                }
            }
            "in" => {
                // "in" operator: value should be an array
                // We use the In comparison operator
                if let serde_json::Value::Array(arr) = &value_json {
                    // For the "in" operator, we need to pass the array of values
                    // The FilterClause only supports single values, so we convert
                    // the array to a comma-separated string representation
                    // that the backend can parse
                    let values_str: Vec<String> = arr
                        .iter()
                        .filter_map(|v| match v {
                            serde_json::Value::String(s) => Some(format!("\"{}\"", s)),
                            serde_json::Value::Number(n) => Some(n.to_string()),
                            serde_json::Value::Bool(b) => Some(b.to_string()),
                            _ => None,
                        })
                        .collect();

                    clauses.push(FilterClause {
                        field: filter.field.clone(),
                        op: ComparisonOp::In as i32,
                        value: Some(Value::StringValue(format!("[{}]", values_str.join(",")))),
                    });
                } else {
                    return Err(ApiError::InvalidArgument(
                        "Filter operator 'in' requires an array value".to_string(),
                    ));
                }
            }
            _ => {
                // Unknown operator - this should have been caught in validation
                return Err(ApiError::InvalidArgument(format!(
                    "Unsupported filter operator: {}",
                    filter.op
                )));
            }
        }
    }

    Ok(clauses)
}

/// Request to insert ProximaRecords
///
/// ## Example JSON
///
/// ```json
/// {
///     "records": [
///         {
///             "id": "doc_1",
///             "vector": [0.1, 0.2, 0.3],
///             "props": {
///                 "category": "electronics",
///                 "price": 299.99,
///                 "in_stock": true
///             },
///             "text_fields": [
///                 {
///                     "name": "description",
///                     "content": "A detailed product description...",
///                     "storage_hint": "adaptive"
///                 }
///             ],
///             "source": "catalog_import"
///         }
///     ],
///     "validate_schema": true
/// }
/// ```
/// Not supported in v0.2: optimistic versioning, conditional-write predicates,
/// update-by-filter, delete-by-filter, and patch / partial-record update. The
/// live contract is exposed via `WriteContractHealth` on the route-health
/// diagnostic endpoint (`GET /api/v2/_diagnostics/collections/{id}/route-health`);
/// see also `docs/SUPPORTED_SURFACE.adoc` "Not Supported in v0.2".
#[derive(Debug, Deserialize, ToSchema)]
pub struct InsertRecordsRequest {
    /// Records to insert
    pub records: Vec<ProximaRecordInput>,
    /// Whether to validate against collection schema (default: true)
    pub validate_schema: Option<bool>,
    /// Whether existing records with the same ID should be replaced.
    ///
    /// The current durable record write path is idempotent/upsert-oriented; this
    /// field is accepted so SDKs can expose explicit upsert intent without
    /// leaving the OpenAPI contract.
    pub upsert: Option<bool>,
}

/// Input format for ProximaRecord (JSON-friendly)
///
/// This is the JSON-serializable input format for ProximaRecord.
/// It uses serde_json::Value for typed fields to support dynamic typing
/// at the API boundary, with validation happening during conversion.
#[derive(Debug, Deserialize, ToSchema)]
pub struct ProximaRecordInput {
    /// Record ID (optional, will be auto-generated if not provided)
    pub id: Option<String>,
    /// Vector embedding (required)
    #[schema(value_type = Vec<f32>)]
    pub vector: Vec<f32>,
    /// Canonical rich property map.
    #[schema(value_type = Option<HashMap<String, Value>>)]
    pub props: Option<HashMap<String, RestProximaValue>>,
    /// Dedicated TEXT fields with storage hints
    pub text_fields: Option<Vec<TextFieldInput>>,
}

/// REST value shape for the canonical `ProximaValue` type system.
///
/// Scalars may be sent directly for ergonomic JSON. Rich or ambiguous values use
/// `{ "type": "...", "value": ... }`, e.g. `{ "type": "jsonb", "value": {...} }`.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum RestProximaValue {
    Typed {
        #[serde(rename = "type")]
        type_name: String,
        value: serde_json::Value,
    },
    Inferred(serde_json::Value),
}

fn rest_value_to_proxima(value: &RestProximaValue) -> Result<ProximaValue, ApiError> {
    match value {
        RestProximaValue::Inferred(value) => infer_json_value(value),
        RestProximaValue::Typed { type_name, value } => typed_json_value(type_name, value),
    }
}

fn infer_json_value(value: &serde_json::Value) -> Result<ProximaValue, ApiError> {
    Ok(match value {
        serde_json::Value::Null => ProximaValue::Null,
        serde_json::Value::Bool(v) => ProximaValue::Boolean(*v),
        serde_json::Value::Number(v) => {
            if let Some(i) = v.as_i64() {
                ProximaValue::Int64(i)
            } else if let Some(u) = v.as_u64() {
                ProximaValue::UInt64(u)
            } else if let Some(f) = v.as_f64() {
                ProximaValue::Float64(f)
            } else {
                return Err(ApiError::InvalidArgument(format!("Invalid number: {v}")));
            }
        }
        serde_json::Value::String(v) => ProximaValue::String(v.clone()),
        serde_json::Value::Array(values) => ProximaValue::Array(
            values
                .iter()
                .map(infer_json_value)
                .collect::<Result<Vec<_>, _>>()?,
        ),
        serde_json::Value::Object(values) => ProximaValue::Struct(
            values
                .iter()
                .map(|(k, v)| infer_json_value(v).map(|value| (k.clone(), value)))
                .collect::<Result<HashMap<_, _>, _>>()?,
        ),
    })
}

fn typed_json_value(type_name: &str, value: &serde_json::Value) -> Result<ProximaValue, ApiError> {
    let normalized = type_name.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "null" => Ok(ProximaValue::Null),
        "bool" | "boolean" => value
            .as_bool()
            .map(ProximaValue::Boolean)
            .ok_or_else(|| ApiError::InvalidArgument("boolean requires true or false".to_string())),
        "string" | "utf8" => value
            .as_str()
            .map(|v| ProximaValue::String(v.to_string()))
            .ok_or_else(|| ApiError::InvalidArgument("string requires a string".to_string())),
        "json" => Ok(ProximaValue::Json(value.clone())),
        "jsonb" => Ok(ProximaValue::Jsonb(value.clone())),
        "decimal" => Ok(ProximaValue::Decimal(match value {
            serde_json::Value::String(v) => v.clone(),
            _ => value.to_string(),
        })),
        "symbol" => value
            .as_str()
            .map(|v| ProximaValue::Symbol(v.to_string()))
            .ok_or_else(|| ApiError::InvalidArgument("symbol requires a string".to_string())),
        "uuid" => value
            .as_str()
            .ok_or_else(|| ApiError::InvalidArgument("uuid requires a string".to_string()))
            .and_then(|v| {
                uuid::Uuid::parse_str(v)
                    .map(|uuid| ProximaValue::Uuid(*uuid.as_bytes()))
                    .map_err(|e| ApiError::InvalidArgument(format!("invalid uuid: {e}")))
            }),
        "ulid" => parse_hex_16("ulid", value).map(ProximaValue::ULID),
        "int8" => parse_i64("int8", value).and_then(|v| {
            i8::try_from(v)
                .map(ProximaValue::Int8)
                .map_err(|_| ApiError::InvalidArgument("int8 out of range".to_string()))
        }),
        "int16" => parse_i64("int16", value).and_then(|v| {
            i16::try_from(v)
                .map(ProximaValue::Int16)
                .map_err(|_| ApiError::InvalidArgument("int16 out of range".to_string()))
        }),
        "int32" => parse_i64("int32", value).and_then(|v| {
            i32::try_from(v)
                .map(ProximaValue::Int32)
                .map_err(|_| ApiError::InvalidArgument("int32 out of range".to_string()))
        }),
        "int64" | "integer" => parse_i64("int64", value).map(ProximaValue::Int64),
        "uint8" => parse_u64("uint8", value).and_then(|v| {
            u8::try_from(v)
                .map(ProximaValue::UInt8)
                .map_err(|_| ApiError::InvalidArgument("uint8 out of range".to_string()))
        }),
        "uint16" => parse_u64("uint16", value).and_then(|v| {
            u16::try_from(v)
                .map(ProximaValue::UInt16)
                .map_err(|_| ApiError::InvalidArgument("uint16 out of range".to_string()))
        }),
        "uint32" => parse_u64("uint32", value).and_then(|v| {
            u32::try_from(v)
                .map(ProximaValue::UInt32)
                .map_err(|_| ApiError::InvalidArgument("uint32 out of range".to_string()))
        }),
        "uint64" => value
            .as_u64()
            .or_else(|| value.as_str().and_then(|v| v.parse().ok()))
            .map(ProximaValue::UInt64)
            .ok_or_else(|| {
                ApiError::InvalidArgument("uint64 requires an unsigned integer".to_string())
            }),
        "float16" => value
            .as_f64()
            .map(|v| ProximaValue::Float16(v as f32))
            .ok_or_else(|| ApiError::InvalidArgument("float16 requires a number".to_string())),
        "float32" => value
            .as_f64()
            .map(|v| ProximaValue::Float32(v as f32))
            .ok_or_else(|| ApiError::InvalidArgument("float32 requires a number".to_string())),
        "float64" | "float" | "number" => value
            .as_f64()
            .map(ProximaValue::Float64)
            .ok_or_else(|| ApiError::InvalidArgument("float64 requires a number".to_string())),
        "date" => parse_i64("date", value).and_then(|v| {
            i32::try_from(v)
                .map(ProximaValue::Date)
                .map_err(|_| ApiError::InvalidArgument("date out of range".to_string()))
        }),
        "time" => parse_temporal(value).map(|(v, unit)| ProximaValue::Time(v, unit)),
        "timestamp" => parse_temporal(value).map(|(v, unit)| ProximaValue::Timestamp(v, unit)),
        "timestamptz" | "timestamp_tz" => {
            parse_temporal(value).map(|(v, unit)| ProximaValue::TimestampTz(v, unit))
        }
        "binary_vector" => bytes_from_json_array(value).map(ProximaValue::BinaryVector),
        "binary" => bytes_from_json_array(value).map(ProximaValue::Binary),
        "array" => value
            .as_array()
            .ok_or_else(|| ApiError::InvalidArgument("array requires an array".to_string()))?
            .iter()
            .map(infer_json_value)
            .collect::<Result<Vec<_>, _>>()
            .map(ProximaValue::Array),
        "map" => typed_object_to_proxima(value).map(ProximaValue::Map),
        "struct" => typed_object_to_proxima(value).map(ProximaValue::Struct),
        "dense_vector" | "vector" => value
            .as_array()
            .ok_or_else(|| ApiError::InvalidArgument("vector requires an array".to_string()))?
            .iter()
            .map(|v| {
                v.as_f64().map(|n| n as f32).ok_or_else(|| {
                    ApiError::InvalidArgument("vector values must be numeric".to_string())
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(ProximaValue::DenseVector),
        "sparse_vector" => {
            let obj = value.as_object().ok_or_else(|| {
                ApiError::InvalidArgument(
                    "sparse_vector requires {\"indices\": [...], \"values\": [...]}".to_string(),
                )
            })?;
            let indices = obj
                .get("indices")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    ApiError::InvalidArgument("sparse_vector indices must be an array".to_string())
                })?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .and_then(|v| u32::try_from(v).ok())
                        .ok_or_else(|| {
                            ApiError::InvalidArgument(
                                "sparse_vector indices must be uint32 values".to_string(),
                            )
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let values = obj
                .get("values")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    ApiError::InvalidArgument("sparse_vector values must be an array".to_string())
                })?
                .iter()
                .map(|v| {
                    v.as_f64().map(|v| v as f32).ok_or_else(|| {
                        ApiError::InvalidArgument(
                            "sparse_vector values must be numeric".to_string(),
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(ProximaValue::SparseVector { indices, values })
        }
        _ => infer_json_value(value),
    }
}

fn typed_object_to_proxima(
    value: &serde_json::Value,
) -> Result<HashMap<String, ProximaValue>, ApiError> {
    value
        .as_object()
        .ok_or_else(|| ApiError::InvalidArgument("object type requires a JSON object".to_string()))?
        .iter()
        .map(|(key, value)| {
            serde_json::from_value::<RestProximaValue>(value.clone())
                .map_err(|e| ApiError::InvalidArgument(format!("invalid typed object value: {e}")))
                .and_then(|value| rest_value_to_proxima(&value))
                .map(|value| (key.clone(), value))
        })
        .collect()
}

fn parse_i64(type_name: &str, value: &serde_json::Value) -> Result<i64, ApiError> {
    value
        .as_i64()
        .or_else(|| value.as_str().and_then(|v| v.parse().ok()))
        .ok_or_else(|| ApiError::InvalidArgument(format!("{type_name} requires an integer")))
}

fn parse_u64(type_name: &str, value: &serde_json::Value) -> Result<u64, ApiError> {
    value
        .as_u64()
        .or_else(|| value.as_str().and_then(|v| v.parse().ok()))
        .ok_or_else(|| {
            ApiError::InvalidArgument(format!("{type_name} requires an unsigned integer"))
        })
}

fn parse_temporal(value: &serde_json::Value) -> Result<(i64, TimeUnit), ApiError> {
    if let Some(object) = value.as_object() {
        let value = object
            .get("value")
            .ok_or_else(|| {
                ApiError::InvalidArgument("temporal value requires a value field".to_string())
            })
            .and_then(|value| parse_i64("temporal value", value))?;
        let unit = object
            .get("unit")
            .and_then(|unit| unit.as_str())
            .map(parse_time_unit)
            .transpose()?
            .unwrap_or(TimeUnit::Nanosecond);
        Ok((value, unit))
    } else {
        parse_i64("temporal value", value).map(|value| (value, TimeUnit::Nanosecond))
    }
}

fn parse_time_unit(unit: &str) -> Result<TimeUnit, ApiError> {
    match unit.trim().to_ascii_lowercase().as_str() {
        "s" | "sec" | "second" | "seconds" => Ok(TimeUnit::Second),
        "ms" | "millisecond" | "milliseconds" => Ok(TimeUnit::Millisecond),
        "us" | "microsecond" | "microseconds" => Ok(TimeUnit::Microsecond),
        "ns" | "nanosecond" | "nanoseconds" => Ok(TimeUnit::Nanosecond),
        other => Err(ApiError::InvalidArgument(format!(
            "unsupported temporal unit: {other}"
        ))),
    }
}

fn parse_hex_16(type_name: &str, value: &serde_json::Value) -> Result<[u8; 16], ApiError> {
    let raw = value
        .as_str()
        .ok_or_else(|| ApiError::InvalidArgument(format!("{type_name} requires a string")))?;
    let decoded = hex::decode(raw.trim())
        .map_err(|e| ApiError::InvalidArgument(format!("{type_name} must be 16 hex bytes: {e}")))?;
    decoded.try_into().map_err(|_| {
        ApiError::InvalidArgument(format!("{type_name} must decode to exactly 16 bytes"))
    })
}

pub(crate) fn proxima_value_to_rest_value(value: &ProximaValue) -> RestProximaValue {
    let typed = |type_name: &str, value: serde_json::Value| RestProximaValue::Typed {
        type_name: type_name.to_string(),
        value,
    };

    match value {
        ProximaValue::Boolean(value) => typed("boolean", serde_json::Value::Bool(*value)),
        ProximaValue::Int8(value) => typed("int8", serde_json::json!(value)),
        ProximaValue::Int16(value) => typed("int16", serde_json::json!(value)),
        ProximaValue::Int32(value) => typed("int32", serde_json::json!(value)),
        ProximaValue::Int64(value) => typed("int64", serde_json::json!(value)),
        ProximaValue::UInt8(value) => typed("uint8", serde_json::json!(value)),
        ProximaValue::UInt16(value) => typed("uint16", serde_json::json!(value)),
        ProximaValue::UInt32(value) => typed("uint32", serde_json::json!(value)),
        ProximaValue::UInt64(value) => typed("uint64", serde_json::json!(value)),
        ProximaValue::Float16(value) => typed("float16", serde_json::json!(value)),
        ProximaValue::Float32(value) => typed("float32", serde_json::json!(value)),
        ProximaValue::Float64(value) => typed("float64", serde_json::json!(value)),
        ProximaValue::Decimal(value) => typed("decimal", serde_json::Value::String(value.clone())),
        ProximaValue::String(value) => typed("string", serde_json::Value::String(value.clone())),
        ProximaValue::Symbol(value) => typed("symbol", serde_json::Value::String(value.clone())),
        ProximaValue::Binary(value) => typed("binary", bytes_to_json(value)),
        ProximaValue::Date(value) => typed("date", serde_json::json!(value)),
        ProximaValue::Time(value, unit) => typed("time", temporal_to_json(*value, *unit)),
        ProximaValue::Timestamp(value, unit) => typed("timestamp", temporal_to_json(*value, *unit)),
        ProximaValue::TimestampTz(value, unit) => {
            typed("timestamptz", temporal_to_json(*value, *unit))
        }
        ProximaValue::Uuid(value) => typed(
            "uuid",
            serde_json::Value::String(uuid::Uuid::from_bytes(*value).to_string()),
        ),
        ProximaValue::ULID(value) => typed("ulid", serde_json::Value::String(hex::encode(value))),
        ProximaValue::Json(value) => typed("json", value.clone()),
        ProximaValue::Jsonb(value) => typed("jsonb", value.clone()),
        ProximaValue::Array(values) => typed(
            "array",
            serde_json::Value::Array(
                values
                    .iter()
                    .map(|value| rest_value_to_json_value(&proxima_value_to_rest_value(value)))
                    .collect(),
            ),
        ),
        ProximaValue::Map(values) => typed("map", proxima_map_to_json(values)),
        ProximaValue::Struct(values) => typed("struct", proxima_map_to_json(values)),
        ProximaValue::DenseVector(values) => typed("dense_vector", serde_json::json!(values)),
        ProximaValue::SparseVector { indices, values } => typed(
            "sparse_vector",
            serde_json::json!({
                "indices": indices,
                "values": values,
            }),
        ),
        ProximaValue::BinaryVector(value) => typed("binary_vector", bytes_to_json(value)),
        ProximaValue::Null => typed("null", serde_json::Value::Null),
    }
}

#[cfg(test)]
fn rest_value_payload(value: &RestProximaValue) -> serde_json::Value {
    match value {
        RestProximaValue::Typed { value, .. } | RestProximaValue::Inferred(value) => value.clone(),
    }
}

fn rest_value_to_json_value(value: &RestProximaValue) -> serde_json::Value {
    serde_json::to_value(value).unwrap_or(serde_json::Value::Null)
}

fn proxima_map_to_json(values: &HashMap<String, ProximaValue>) -> serde_json::Value {
    serde_json::Value::Object(
        values
            .iter()
            .map(|(key, value)| {
                (
                    key.clone(),
                    rest_value_to_json_value(&proxima_value_to_rest_value(value)),
                )
            })
            .collect(),
    )
}

fn temporal_to_json(value: i64, unit: TimeUnit) -> serde_json::Value {
    serde_json::json!({
        "value": value,
        "unit": match unit {
            TimeUnit::Second => "second",
            TimeUnit::Millisecond => "millisecond",
            TimeUnit::Microsecond => "microsecond",
            TimeUnit::Nanosecond => "nanosecond",
        },
    })
}

fn bytes_to_json(value: &[u8]) -> serde_json::Value {
    serde_json::Value::Array(
        value
            .iter()
            .map(|value| serde_json::Value::Number((*value as u64).into()))
            .collect(),
    )
}

fn bytes_from_json_array(value: &serde_json::Value) -> Result<Vec<u8>, ApiError> {
    value
        .as_array()
        .ok_or_else(|| {
            ApiError::InvalidArgument("binary values require an array of bytes".to_string())
        })?
        .iter()
        .map(|v| {
            v.as_u64()
                .filter(|v| *v <= u8::MAX as u64)
                .map(|v| v as u8)
                .ok_or_else(|| {
                    ApiError::InvalidArgument("binary byte values must be 0..255".to_string())
                })
        })
        .collect()
}

/// Convert TypedFilter list → optimizer Predicate list for the planner.
///
/// Best-effort: filter ops we can't map to a `PredicateOp` are dropped, since
/// the planner can fall through to policy defaults. The selectivity estimator
/// treats an empty predicate list as "selectivity = 1.0" (full scan).
fn typed_filters_to_predicates(
    filters: &[TypedFilter],
) -> Vec<crate::query::federated::optimizer::Predicate> {
    use crate::query::federated::optimizer::{Predicate, PredicateOp};
    filters
        .iter()
        .filter_map(|f| {
            let op = match f.op.as_str() {
                "eq" => PredicateOp::Eq,
                "neq" => PredicateOp::Ne,
                "gt" => PredicateOp::Gt,
                "gte" => PredicateOp::Ge,
                "lt" => PredicateOp::Lt,
                "lte" => PredicateOp::Le,
                "between" => PredicateOp::Between,
                "in" => PredicateOp::In,
                _ => return None,
            };
            let value = rest_value_to_predicate_value(&f.value);
            Some(Predicate {
                column: f.field.clone(),
                op,
                value,
            })
        })
        .collect()
}

fn rest_value_to_predicate_value(
    v: &RestProximaValue,
) -> crate::query::federated::optimizer::PredicateValue {
    use crate::query::federated::optimizer::PredicateValue;
    // Project through ProximaValue so we share the same JSON-inference logic
    // already used for query execution. Anything we can't infer collapses to
    // Null — the selectivity estimator handles Null gracefully via the
    // fallback policy.
    let pv = match rest_value_to_proxima(v) {
        Ok(p) => p,
        Err(_) => return PredicateValue::Null,
    };
    match pv {
        ProximaValue::String(s) => PredicateValue::String(s),
        ProximaValue::Int64(i) => PredicateValue::Int(i),
        ProximaValue::UInt64(u) => PredicateValue::Int(u as i64),
        ProximaValue::Float64(f) => PredicateValue::Float(f),
        ProximaValue::Boolean(b) => PredicateValue::Bool(b),
        ProximaValue::Array(items) => PredicateValue::List(
            items
                .into_iter()
                .filter_map(proxima_to_predicate_value)
                .collect(),
        ),
        _ => PredicateValue::Null,
    }
}

fn proxima_to_predicate_value(
    p: ProximaValue,
) -> Option<crate::query::federated::optimizer::PredicateValue> {
    use crate::query::federated::optimizer::PredicateValue;
    Some(match p {
        ProximaValue::String(s) => PredicateValue::String(s),
        ProximaValue::Int64(i) => PredicateValue::Int(i),
        ProximaValue::UInt64(u) => PredicateValue::Int(u as i64),
        ProximaValue::Float64(f) => PredicateValue::Float(f),
        ProximaValue::Boolean(b) => PredicateValue::Bool(b),
        ProximaValue::Null => PredicateValue::Null,
        _ => return None,
    })
}

/// Operators accepted on the typed `{field,op,value}` filter surface, shared by
/// `/search` and `/records/scan` validation.
const VALID_TYPED_FILTER_OPS: [&str; 11] = [
    "eq",
    "neq",
    "gt",
    "gte",
    "lt",
    "lte",
    "contains",
    "between",
    "in",
    "starts_with",
    "ends_with",
];

/// Validate a list of typed `{field,op,value}` filters before lowering them.
fn validate_typed_filters(filters: &[TypedFilter]) -> Result<(), ApiError> {
    for filter in filters {
        if filter.field.is_empty() {
            return Err(ApiError::InvalidArgument(
                "Filter field name cannot be empty".to_string(),
            ));
        }
        if !VALID_TYPED_FILTER_OPS.contains(&filter.op.as_str()) {
            return Err(ApiError::InvalidArgument(format!(
                "Invalid filter operator '{}'. Valid operators: {:?}",
                filter.op, VALID_TYPED_FILTER_OPS
            )));
        }
        if filter.op == "between" && filter.value_upper.is_none() {
            return Err(ApiError::InvalidArgument(
                "Filter operator 'between' requires 'value_upper' to be specified".to_string(),
            ));
        }
    }
    Ok(())
}

/// Parse a metadata filter supplied as loose JSON into the canonical
/// [`FilterExpression`] consumed by the scan/search engines.
///
/// Two shapes are accepted so the same field can mirror `/search` and the
/// simple equality maps that multi-tenant callers send:
///   - **array** — `[{ "field": .., "op": .., "value": .. }, ..]`, the typed
///     filter list (full operator set), reusing the search lowering;
///   - **object** — `{ "field": value, .. }`, treated as an AND of equality
///     comparisons (the common `account_id`-scoping case).
///
/// Returns `Ok(None)` when there is nothing to filter on.
fn parse_metadata_filter(
    value: &serde_json::Value,
) -> Result<Option<crate::core::search::FilterExpression>, ApiError> {
    use crate::core::search::{ComparisonOperator, FilterExpression};

    match value {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::Array(items) if items.is_empty() => Ok(None),
        serde_json::Value::Object(map) if map.is_empty() => Ok(None),
        serde_json::Value::Array(_) => {
            let typed: Vec<TypedFilter> = serde_json::from_value(value.clone())
                .map_err(|e| ApiError::InvalidArgument(format!("invalid filter list: {e}")))?;
            validate_typed_filters(&typed)?;
            let rich = typed
                .iter()
                .map(typed_filter_to_rich)
                .collect::<Result<Vec<_>, ApiError>>()?;
            Ok(crate::services::operations::vectors::rich_filters_to_filter_expression(&rich))
        }
        serde_json::Value::Object(map) => {
            let conditions: Vec<FilterExpression> = map
                .iter()
                .map(|(field, v)| FilterExpression::Comparison {
                    field: field.clone(),
                    operator: ComparisonOperator::Equals,
                    value: v.clone(),
                })
                .collect();
            Ok(match conditions.len() {
                0 => None,
                1 => conditions.into_iter().next(),
                _ => Some(FilterExpression::And(conditions)),
            })
        }
        _ => Err(ApiError::InvalidArgument(
            "filter must be an object or an array of {field,op,value}".to_string(),
        )),
    }
}

fn typed_filter_to_rich(filter: &TypedFilter) -> Result<RichFilterCondition, ApiError> {
    let operator = match filter.op.as_str() {
        "eq" => RichFilterOperator::Eq,
        "neq" => RichFilterOperator::Ne,
        "gt" => RichFilterOperator::Gt,
        "gte" => RichFilterOperator::Gte,
        "lt" => RichFilterOperator::Lt,
        "lte" => RichFilterOperator::Lte,
        "contains" => RichFilterOperator::Contains,
        "starts_with" => RichFilterOperator::StartsWith,
        "ends_with" => RichFilterOperator::EndsWith,
        "between" => RichFilterOperator::Between,
        "in" => RichFilterOperator::In,
        _ => {
            return Err(ApiError::InvalidArgument(format!(
                "Unsupported filter operator: {}",
                filter.op
            )));
        }
    };

    let value = rest_value_to_proxima(&filter.value)?;
    let value_upper = filter
        .value_upper
        .as_ref()
        .map(rest_value_to_proxima)
        .transpose()?;
    let value_list = match (&operator, &value) {
        (RichFilterOperator::In, ProximaValue::Array(values)) => values.clone(),
        _ => Vec::new(),
    };

    Ok(RichFilterCondition {
        field: filter.field.clone(),
        operator,
        value,
        value_upper,
        value_list,
    })
}

/// Input format for TEXT fields
///
/// TEXT fields are stored in dedicated columns with optional chunking
/// for large content. The storage hint helps optimize storage strategy.
#[derive(Debug, Deserialize, ToSchema)]
pub struct TextFieldInput {
    /// Field name
    pub name: String,
    /// Text content
    pub content: String,
    /// Storage strategy hint
    ///
    /// - "inline": Store inline in main column (<4KB)
    /// - "chunked": Split into chunks with embeddings (4KB-1MB)
    /// - "sidecar": Store in separate sidecar file (>1MB)
    /// - "adaptive": Auto-select based on content size (default)
    pub storage_hint: Option<String>,
}

/// Response for insert operation
#[derive(Debug, Serialize, ToSchema)]
pub struct InsertRecordsResponse {
    /// Number of successfully inserted records
    pub inserted_count: usize,
    /// Number of failed records
    pub failed_count: usize,
    /// Detailed errors for failed records
    pub errors: Vec<InsertError>,
    /// IDs of successfully inserted records
    pub inserted_ids: Vec<String>,
}

/// Error details for a failed record insertion
#[derive(Debug, Serialize, ToSchema)]
pub struct InsertError {
    /// Index of the record in the request
    pub index: usize,
    /// Record ID (if provided)
    pub id: Option<String>,
    /// Error message
    pub error: String,
}

/// POST /api/v2/collections/{collection}/records/batch
///
/// Insert ProximaRecords into a collection with typed field support.
///
/// ## Request Body
///
/// See [`InsertRecordsRequest`] for the expected JSON format.
///
/// ## Response
///
/// Returns [`InsertRecordsResponse`] with counts and any errors.
///
/// ## Errors
///
/// - `400 Bad Request`: Invalid request format or validation error
/// - `404 Not Found`: Collection does not exist
/// - `500 Internal Server Error`: Storage or processing error
#[utoipa::path(
    post,
    path = "/api/v2/collections/{collection_id}/records/batch",
    tag = "Records",
    operation_id = "insertRecords",
    summary = "Insert or upsert ProximaRecord batches.",
    params(
        ("collection_id" = String, Path, description = "Collection name/ID."),
    ),
    request_body = InsertRecordsRequest,
    responses(
        (status = 200, description = "Batch write result.", body = InsertRecordsResponse),
        (status = 400, description = "Invalid request.", body = crate::network::rest::openapi::ErrorResponse),
    ),
)]
pub async fn insert_records(
    Path(collection): Path<String>,
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    capability: Option<Extension<DataPlaneCapability>>,
    Json(request): Json<InsertRecordsRequest>,
) -> ApiResult<Json<InsertRecordsResponse>> {
    info!(
        "V2 API: Inserting {} records into collection '{}'",
        request.records.len(),
        collection
    );

    // Validate collection exists
    if collection.is_empty() {
        return Err(ApiError::InvalidArgument(
            "Collection name is required".to_string(),
        ));
    }

    if request.records.is_empty() {
        return Err(ApiError::InvalidArgument(
            "At least one record is required".to_string(),
        ));
    }

    // Slice 4 of tenant-pod-affinity: gate writes against the
    // primary-pod registry BEFORE any storage work. Two outcomes:
    //
    // * `Allow`: either no binding for (tenant, collection) — legacy
    //   behavior — or the binding matches this pod. Increment the
    //   appropriate `WRITES_ALLOWED_TOTAL` counter and proceed.
    // * `Misrouted`: a binding exists pointing elsewhere. Return 421
    //   Misdirected Request with the target pod so the client SDK
    //   re-routes. Continuing here would land the write in this pod's
    //   memtable where reads on the primary pod would never find it
    //   (the 3-stage search at services/operations/vectors/legacy.rs
    //   :2827-2858 is local-memtable-only on stage 1).
    match crate::cluster::primary_pod_registry::consult_for_write(
        &state.primary_pod_registry,
        &state.self_pod_id,
        &tenant.tenant_id,
        &collection,
    ) {
        crate::cluster::primary_pod_registry::WriteRoutingDecision::Allow => {
            if state
                .primary_pod_registry
                .is_assigned(&tenant.tenant_id, &collection)
            {
                crate::metrics::primary_pod_metrics::record_allowed_bound(&tenant.tenant_id);
            } else {
                crate::metrics::primary_pod_metrics::record_allowed_unbounded(&tenant.tenant_id);
            }
        }
        crate::cluster::primary_pod_registry::WriteRoutingDecision::Misrouted { target_pod } => {
            crate::metrics::primary_pod_metrics::record_misrouted(&tenant.tenant_id);
            tracing::warn!(
                target = "proximadb.primary_pod.misroute",
                self_pod = %state.self_pod_id,
                target_pod = %target_pod,
                tenant_id = %tenant.tenant_id,
                collection_id = %collection,
                "v2 insert misrouted — client SDK should retry against the primary pod"
            );
            return Err(ApiError::Misdirected {
                target_pod,
                tenant_id: tenant.tenant_id.clone(),
                collection_id: collection,
            });
        }
    }
    if let Some(Extension(capability)) = capability.as_ref() {
        capability
            .ensure_record_count(request.records.len())
            .map_err(ApiError::InvalidArgument)?;
    }

    let validate_schema = request.validate_schema.unwrap_or_else(|| {
        debug!("No schema validation preference provided, defaulting to true");
        true
    });
    if request.upsert.unwrap_or(false) {
        debug!("Record batch requested explicit upsert semantics");
    }
    debug!(
        "Schema validation: {}",
        if validate_schema {
            "enabled"
        } else {
            "disabled"
        }
    );

    let total_records = request.records.len();
    let mut inserted_ids = Vec::with_capacity(total_records);
    let mut errors = Vec::new();
    let mut rich_records = Vec::with_capacity(total_records);

    for (index, record) in request.records.into_iter().enumerate() {
        // Validate vector is not empty
        if record.vector.is_empty() {
            errors.push(InsertError {
                index,
                id: record.id,
                error: "Vector cannot be empty".to_string(),
            });
            continue;
        }

        // Generate ID if not provided
        let record_id = record.id.unwrap_or_else(|| {
            let new_id = uuid::Uuid::new_v4().to_string();
            debug!("Generated new UUID for record: {}", new_id);
            new_id
        });

        let mut props = HashMap::new();

        if let Some(input_props) = record.props {
            for (key, value) in input_props {
                let proxima_value = rest_value_to_proxima(&value)?;
                props.insert(key, ProximaTreeNode::Value(proxima_value));
            }
        }

        if let Some(text_fields) = record.text_fields {
            for text_field in text_fields {
                props.insert(
                    text_field.name,
                    ProximaTreeNode::Value(ProximaValue::String(text_field.content)),
                );
            }
        }

        let now_ns = chrono::Utc::now()
            .timestamp_millis()
            .saturating_mul(1_000_000);
        let vector = record.vector;
        let dim = vector.len() as u32;
        let rich_record = ProximaRecord {
            oid: record_id.clone(),
            created_at_ns: now_ns,
            updated_at_ns: now_ns,
            origin: Some("v2_api".to_string()),
            props,
            embeddings: vec![EmbeddingCell {
                model_id: "default".to_string(),
                modality: "dense_vector".to_string(),
                dim,
                values: proximadb_records::EmbeddingValues::Fp32(vector),
                ..Default::default()
            }],
            ..ProximaRecord::default()
        };

        rich_records.push(rich_record);
        inserted_ids.push(record_id);
    }

    // Early return if all records failed validation
    if rich_records.is_empty() {
        return Ok(Json(InsertRecordsResponse {
            inserted_count: 0,
            failed_count: errors.len(),
            errors,
            inserted_ids: vec![],
        }));
    }

    let batch_request = RichRecordBatchRequest {
        collection_id: collection.clone(),
        records: rich_records,
    };

    let intent = WriteIntent::new(&collection, WriteOperationKind::Insert)
        .with_durability(WriteDurabilityRequirement::WalRequired)
        .with_row_count_hint(inserted_ids.len() as u64);
    let lane = WriteLaneRouter::new().route(&intent);
    debug!(
        collection_id = %collection,
        write_lane = ?lane.lane,
        guards = ?lane.required_guards,
        "REST v2 insert_records write-lane decision"
    );

    match state
        .record_ops
        .handle_record_batch_for_tenant(batch_request, Some(&tenant.tenant_id))
        .await
    {
        Ok(resp) => {
            // v0.2 release-readiness audit (round 2): a missing collection
            // was returning HTTP 200 with `BatchOperationResult::failure` in
            // the body, which broke the v2 INSERT data-path silently for
            // weeks. Map the well-known NOT_FOUND error_code onto a real
            // HTTP 404 here so SDK consumers and curl users see the same
            // signal. Other failure codes still flow through the body, which
            // matches the batched per-record contract.
            if !resp.success {
                if resp.error_code.as_deref() == Some("NOT_FOUND") {
                    return Err(ApiError::NotFound(
                        resp.errors
                            .into_iter()
                            .next()
                            .unwrap_or_else(|| format!("Collection '{}' not found", collection)),
                    ));
                }
                // ADR-069 S4: the whole batch was shed by write-admission
                // backpressure — surface 429 / RESOURCE_EXHAUSTED so the client
                // retries with backoff. (`WAL_BACKPRESSURE` is raised inside the
                // write path; `WAL_LANE_REJECTED` is the #951 pre-write lane
                // rejection. No partial results to represent — unlike a partial
                // success, the batch was wholly rejected.)
                if matches!(
                    resp.error_code.as_deref(),
                    Some("WAL_BACKPRESSURE") | Some("WAL_LANE_REJECTED")
                ) {
                    return Err(ApiError::ResourceExhausted(
                        resp.errors.into_iter().next().unwrap_or_else(|| {
                            "WAL write-admission backpressure; retry after a flush drains"
                                .to_string()
                        }),
                    ));
                }
            }
            // Check for success - if successful, all records were inserted
            let validation_error_count = errors.len();
            let success_count = if resp.success {
                resp.metrics.successful_count as usize
            } else {
                0
            }
            .min(inserted_ids.len());
            let service_failed_count = inserted_ids
                .len()
                .saturating_sub(success_count)
                .max(resp.metrics.failed_count.max(0) as usize)
                .max(resp.errors.len());
            errors.extend(
                resp.errors
                    .into_iter()
                    .enumerate()
                    .map(|(idx, error)| InsertError {
                        index: success_count + idx,
                        id: None,
                        error,
                    }),
            );

            let response = InsertRecordsResponse {
                inserted_count: success_count,
                failed_count: validation_error_count + service_failed_count,
                errors,
                inserted_ids: if resp.success {
                    resp.vector_ids
                } else {
                    vec![]
                },
            };

            info!(
                "V2 API: Insert complete - {} inserted, {} failed",
                response.inserted_count, response.failed_count
            );

            Ok(Json(response))
        }
        Err(e) => {
            error!("V2 API: Batch insert failed: {}", e);
            // ADR-069 S4: promote a WalBackpressure in the chain to a retryable
            // status (429 / RESOURCE_EXHAUSTED) instead of a non-retryable 500.
            // NOTE: the batch path folds WAL-lane rejections into Ok(success=false)
            // (#951), so this arm sees backpressure only on Err-propagating inserts;
            // end-to-end 429 for the batch path is the TD-WAL-1 S4 residual.
            Err(ApiError::from_write_error("Insert failed", e))
        }
    }
}

/// Search request with typed filters
///
/// ## Example JSON
///
/// ```json
/// {
///     "vector": [0.1, 0.2, 0.3],
///     "top_k": 10,
///     "filters": [
///         {"field": "category", "op": "eq", "value": "electronics"},
///         {"field": "price", "op": "lt", "value": 500},
///         {"field": "in_stock", "op": "eq", "value": true}
///     ],
///     "include_text": true
/// }
/// ```
#[derive(Debug, Deserialize, ToSchema)]
pub struct TypedSearchRequest {
    /// Query vector
    #[schema(value_type = Vec<f32>, min_items = 1)]
    pub vector: Vec<f32>,
    /// Number of results to return
    #[schema(minimum = 1)]
    pub top_k: usize,
    /// Typed filters with operator support
    pub filters: Option<Vec<TypedFilter>>,
    /// Whether to include TEXT fields in results (default: false)
    ///
    /// TEXT fields can be large, so they are excluded by default.
    /// Set to true to include them in the response.
    pub include_text: Option<bool>,
    /// Whether to include the vector in results (default: false)
    ///
    /// Vector data can be large, so it is excluded by default.
    pub include_vector: Option<bool>,
    /// Return the SearchPlanTrace + a human-readable route explain
    /// in the response (LLD §1 contract). Defaults to `false` so
    /// non-debug requests don't pay the JSON serialization cost of
    /// the ~30-field trace envelope.
    pub debug: Option<bool>,
}

/// A typed filter for search operations
///
/// Supports various comparison operators with type-safe values.
#[derive(Debug, Deserialize, ToSchema)]
pub struct TypedFilter {
    /// Field name to filter on
    pub field: String,
    /// Comparison operator
    ///
    /// Supported operators:
    /// - "eq": Equals
    /// - "neq": Not equals
    /// - "gt": Greater than
    /// - "gte": Greater than or equal
    /// - "lt": Less than
    /// - "lte": Less than or equal
    /// - "contains": String/array contains
    /// - "between": Value is between two bounds (requires value_upper)
    /// - "in": Value is in a list
    /// - "starts_with": String starts with prefix
    /// - "ends_with": String ends with suffix
    pub op: String,
    /// Filter value (type depends on field type)
    #[schema(value_type = Value)]
    pub value: RestProximaValue,
    /// Upper bound for "between" operator
    #[schema(value_type = Option<Value>)]
    pub value_upper: Option<RestProximaValue>,
}

/// Search result with typed fields
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct TypedSearchResult {
    /// Record ID
    pub id: String,
    /// Similarity score (0.0 - 1.0 for cosine, distance for L2)
    pub score: f32,
    /// Vector embedding (if requested)
    #[schema(value_type = Option<Vec<f32>>)]
    pub vector: Option<Vec<f32>>,
    /// Rich properties from the record
    #[schema(value_type = HashMap<String, Value>)]
    pub props: HashMap<String, RestProximaValue>,
    /// TEXT fields (if include_text is true)
    pub text_fields: Option<Vec<TextFieldOutput>>,
}

/// Output format for TEXT fields
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct TextFieldOutput {
    /// Field name
    pub name: String,
    /// Text content (may be truncated for large content)
    pub content: String,
    /// Number of chunks (for chunked storage)
    pub chunk_count: Option<u32>,
    /// Whether content was truncated
    pub truncated: bool,
}

/// Search response with typed results
#[derive(Debug, Serialize, ToSchema)]
pub struct TypedSearchResponse {
    /// Search results
    pub results: Vec<TypedSearchResult>,
    /// Total number of matching documents (before top_k limit)
    pub total_matches: Option<u64>,
    /// Search latency in milliseconds
    pub latency_ms: u64,
    /// Request ID for tracing
    pub request_id: String,
    /// TD-064: predicate-aware shortfall — present when a filtered search
    /// returned fewer than the requested `top_k` after the WAL+AXIS+storage
    /// merge. **Always emitted when present (NOT `debug`-gated)**, unlike
    /// `search_plan_trace`: a silent `<top_k` under a tenant/RLS filter is
    /// fail-open, so the client must be able to distinguish "fewer than k
    /// records match my filter" from "the engine returned my full top-k".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predicate_shortfall: Option<PredicateShortfallWire>,
    /// SearchPlanTrace (LLD §10) — the per-query telemetry envelope that
    /// upstream gateways consume for metering and planner-v2 training.
    /// Phase 0 emits a stub trace populated from request_id + latency; later
    /// phases fill in the per-stage counters. Only emitted when the request
    /// sets `debug=true` (LLD §1 contract).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Option<Value>)]
    pub search_plan_trace: Option<crate::observability::search_plan_trace::SearchPlanTrace>,
    /// Human-readable explain summary derived from the SearchPlanTrace.
    /// Only emitted when the request sets `debug=true` — gives an on-call
    /// operator a one-glance view of the plan, cache result, and any
    /// actionable hints (high scan fraction, repair triggered, ...).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Option<Value>)]
    pub explain: Option<crate::observability::route_explain::RouteExplain>,
}

/// TD-064: predicate-aware recall shortfall (REST wire shape).
///
/// Mirrors `observability::search_plan_trace::PredicateShortfall` as an
/// explicit REST/OpenAPI schema so the field is typed and documented on the
/// wire without forcing the observability type (or the slim `IndexQueryResult`
/// DTO) to derive `ToSchema`.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct PredicateShortfallWire {
    /// The `top_k` value the caller asked for.
    pub requested_k: u32,
    /// Results actually returned after predicate filtering + merge.
    pub returned_k: u32,
    /// Candidate pool size considered before the predicate (oversample budget).
    pub oversample_pool: u32,
    /// ADR-011 mode that produced the shortfall (`post_filter`, `inline`, or
    /// `pre_filter`).
    pub ann_filtering_mode: String,
}

impl From<&crate::observability::search_plan_trace::PredicateShortfall> for PredicateShortfallWire {
    fn from(sf: &crate::observability::search_plan_trace::PredicateShortfall) -> Self {
        Self {
            requested_k: sf.requested_k,
            returned_k: sf.returned_k,
            oversample_pool: sf.oversample_pool,
            ann_filtering_mode: sf.ann_filtering_mode.clone(),
        }
    }
}

/// POST /api/v2/collections/{collection}/search
///
/// Search a collection with typed filters.
///
/// ## Request Body
///
/// See [`TypedSearchRequest`] for the expected JSON format.
///
/// ## Response
///
/// Returns [`TypedSearchResponse`] with ranked results.
///
/// ## Errors
///
/// - `400 Bad Request`: Invalid request format or filter error
/// - `404 Not Found`: Collection does not exist
/// - `500 Internal Server Error`: Search execution error
#[utoipa::path(
    post,
    path = "/api/v2/collections/{collection_id}/search",
    tag = "Search",
    operation_id = "searchRecords",
    summary = "Search records with vector similarity and typed filters.",
    params(
        ("collection_id" = String, Path, description = "Collection name/ID."),
    ),
    request_body = TypedSearchRequest,
    responses(
        (status = 200, description = "Search results.", body = TypedSearchResponse),
        (status = 400, description = "Invalid request.", body = crate::network::rest::openapi::ErrorResponse),
    ),
)]
pub async fn search_with_typed_filters(
    Path(collection): Path<String>,
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    Json(request): Json<TypedSearchRequest>,
) -> ApiResult<Json<TypedSearchResponse>> {
    let start_time = std::time::Instant::now();
    let request_id = uuid::Uuid::new_v4().to_string();

    info!(
        "V2 API: Search request {} for collection '{}', top_k={}",
        request_id, collection, request.top_k
    );

    // Validate request
    if collection.is_empty() {
        return Err(ApiError::InvalidArgument(
            "Collection name is required".to_string(),
        ));
    }

    if request.vector.is_empty() {
        return Err(ApiError::InvalidArgument(
            "Query vector is required".to_string(),
        ));
    }

    if request.top_k == 0 {
        return Err(ApiError::InvalidArgument(
            "top_k must be greater than 0".to_string(),
        ));
    }
    // v0.2 release-readiness audit (round 2): cap top_k to a defensive
    // upper bound so a malformed client (or a malicious one) cannot ask
    // the server to allocate an unbounded result buffer. The cap is
    // intentionally generous — production HNSW workloads top out far
    // below this — but small enough that a single bad request can't
    // exhaust memory. Operators can raise it via an env var if they
    // need a wider window for benchmark or batch-export shapes.
    let max_top_k: usize = std::env::var("PROXIMADB_SEARCH_MAX_TOP_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);
    if request.top_k > max_top_k {
        return Err(ApiError::InvalidArgument(format!(
            "top_k={} exceeds server cap {} (set PROXIMADB_SEARCH_MAX_TOP_K to override)",
            request.top_k, max_top_k
        )));
    }

    // Validate filters if provided
    if let Some(ref filters) = request.filters {
        validate_typed_filters(filters)?;
    }

    let include_text = request.include_text.unwrap_or_else(|| {
        debug!("No include_text preference provided, defaulting to false");
        false
    });
    debug!(
        "Include TEXT fields: {}, filters: {:?}",
        include_text,
        request.filters.as_ref().map(|f| f.len())
    );

    let filters = request
        .filters
        .as_ref()
        .map(|filters| {
            filters
                .iter()
                .map(typed_filter_to_rich)
                .collect::<Result<Vec<_>, ApiError>>()
        })
        .transpose()?
        .unwrap_or_default();

    let search_request = RichSearchRequest {
        collection_id: collection.clone(),
        query_vector: request.vector.clone(),
        top_k: request.top_k as u32,
        filters,
    };

    // TD-064: wrap the search in a predicate-diagnostics scope so any
    // AxisManager-deep shortfall surfaces here without needing every
    // intermediate service/proto type to carry a predicate_shortfall field.
    // Capture the quantized-route downgrade INSIDE the scope — the task-local
    // binding ends when the scoped future completes, so it must be taken here
    // (TD-075 / F2: surfaced in EXPLAIN below).
    let (
        search_outcome,
        quantized_route_downgraded,
        cold_stage1_only,
        turboquant_hints,
        vector_bounds_pruned_blocks,
        egress_bytes,
        object_store_gets,
        object_store_bytes_read,
    ) = crate::observability::io_trace::instrument_with_stable_tenant(
        Some(tenant.tenant_id.clone()),
        tenant.tenant_stable_id,
        "rest.v2.records.search",
        crate::observability::predicate_diagnostics::scope(async {
            let outcome = state
                .record_ops
                .handle_record_search_for_tenant(
                    search_request,
                    proximadb_runtime::PortIdentity::from(&tenant),
                )
                .await;
            let downgraded =
                crate::observability::predicate_diagnostics::take_quantized_downgrade();
            // ADR-023 T-E: cold-start Stage-1-only serving (also taken in-scope).
            let cold_stage1 = crate::observability::predicate_diagnostics::take_cold_stage1_only();
            // TD-040 S3e: SST vector-bounds prune count (also taken in-scope —
            // the task-local binding ends when this future completes).
            let vb_pruned =
                crate::observability::predicate_diagnostics::take_vector_bounds_pruned();
            // Phase K (Quantization Trait Convergence Plan): TurboQuant
            // EXPLAIN hints recorded by `score_turboquant`. Taken INSIDE
            // the scope because the task-local binding ends when this
            // future completes — same constraint as the take above.
            // `None` is the common case (most searches don't run
            // through TurboQuant scoring); the slot is skipped at
            // serialization time via the `skip_serializing_if` on
            // `SearchPlanHints.turboquant` (Phase J) and
            // `VectorHints.turboquant` (Phase F).
            let tq_hints = crate::observability::predicate_diagnostics::take_turboquant_hints();
            // C3 egress delivery: read the per-query cross-AZ bytes from the
            // active io_trace BEFORE this scope closes (the task-local binding
            // ends when this future completes — same constraint as the takes
            // above). 0 on the free same-AZ path. Flowed into the response
            // SearchPlanTrace as actual_egress_gb (the KEU billing quantity).
            // TD-IOTRACE-2: read the physical object-store meters alongside
            // egress — range_gets (the accurate ranged-GET count) + bytes_read —
            // BEFORE this scope closes (the task-local binding ends when this
            // future completes). Flowed into the response SearchPlanTrace as
            // object_store_gets / object_store_bytes_read (the physical KRU
            // billing input, TD-IOTRACE-3). 0 when no I/O occurred.
            let (egress_bytes, object_store_gets, object_store_bytes_read) =
                crate::observability::io_trace::snapshot()
                    .map(|s| (s.egress_bytes, s.range_gets, s.bytes_read))
                    .unwrap_or((0, 0, 0));
            (
                outcome,
                downgraded,
                cold_stage1,
                tq_hints,
                vb_pruned,
                egress_bytes,
                object_store_gets,
                object_store_bytes_read,
            )
        }),
    )
    .await;

    match search_outcome {
        Ok(resp) => {
            let latency_ms = start_time.elapsed().as_millis() as u64;

            let results: Vec<TypedSearchResult> = resp
                .results
                .iter()
                .map(|r| {
                    let props: HashMap<String, RestProximaValue> = r
                        .props
                        .iter()
                        .map(|(k, v)| (k.clone(), proxima_value_to_rest_value(v)))
                        .collect();

                    Ok(TypedSearchResult {
                        id: r.id.clone(),
                        score: r.score as f32,
                        vector: if request.include_vector.unwrap_or_else(|| {
                            debug!("No include_vector preference provided, defaulting to false");
                            false
                        }) {
                            Some(r.vector.clone())
                        } else {
                            None
                        },
                        props,
                        text_fields: if include_text {
                            // Extract text fields from metadata
                            Some(vec![]) // Would be populated from text storage
                        } else {
                            None
                        },
                    })
                })
                .collect::<Result<_, ApiError>>()?;

            let total_matches = resp.total_found as u64;

            // ── Phase 1: run the deterministic planner via the
            // process-wide PlanCache ─────────────────────────────
            // Bundles selectivity + GLS + filter-strategy + route
            // choice + cache lookup. Identical query shapes within
            // the same corpus version reuse the cached PlanOutput
            // instead of recomputing.
            let predicates = match request.filters.as_ref() {
                Some(filters) => typed_filters_to_predicates(filters.as_slice()),
                None => Vec::new(),
            };
            let field_stats =
                crate::query::federated::optimizer::selectivity::FieldStatistics::default();
            let plan_policy =
                crate::query::federated::optimizer::PredicateSelectivityPolicy::default();
            let tier_record =
                crate::catalog::tenant_tier::TenantTierRecord::fail_safe(&tenant.tenant_id);
            // GLS sampling requires neighborhood centroids that we
            // don't yet surface; pass an empty slice. The cached
            // planner short-circuits to a fresh compute when GLS
            // samples are non-empty, so this contract holds when
            // Phase 5's stats refresher lands.
            let cached_inputs =
                crate::query::federated::optimizer::cached_plan_builder::CachedPlanInputs {
                    plan_inputs:
                        crate::query::federated::optimizer::plan_builder::PlanBuilderInputs {
                            predicates: &predicates,
                            field_stats: &field_stats,
                            policy: &plan_policy,
                            gls_samples: &[],
                            dim: request.vector.len(),
                            recall_target: 0.9,
                            collection_gb: 0.0,
                            tier: &tier_record,
                        },
                    // Pull the current corpus_version from the
                    // process-wide registry. Catalog write paths
                    // bump this on schema/segment/stats changes;
                    // the cache invalidates on the next lookup
                    // when the stamped version no longer matches.
                    // Defaults to 1 for any (tenant, collection)
                    // the registry has never seen.
                    corpus_version: crate::catalog::CorpusVersionRegistry::global()
                        .current(&tenant.tenant_id, &collection)
                        .await,
                };
            let cached_plan =
                crate::query::federated::optimizer::cached_plan_builder::build_for_search_cached_with_collection(
                    crate::query::cache::plan_cache::PlanCache::global(),
                    &cached_inputs,
                    &collection,
                )
                .await;

            // Build the Phase-1 SearchPlanTrace via the centralized
            // TraceBuilder helper. AXIS-level counters
            // (block_fill_pct, tunneled_nodes, ...) stay zero until
            // Phase 3's engine wiring fills them in; the builder
            // accepts the default IndexStats today and emits a
            // valid trace.
            let cache_result = if cached_plan.cache_hit {
                crate::observability::search_plan_trace::CacheResult::Hit
            } else {
                crate::observability::search_plan_trace::CacheResult::Miss
            };
            let trace = crate::observability::search_plan_trace_builder::build(
                crate::observability::search_plan_trace_builder::TraceBuilderInputs {
                    trace_id: request_id.clone(),
                    tenant_id: tenant.tenant_id.clone(),
                    collection_name: collection.clone(),
                    filter_strategy: cached_plan.plan.filter_strategy.clone(),
                    index_route: cached_plan.plan.index_route.clone(),
                    estimated_selectivity: cached_plan.plan.estimated_selectivity,
                    gls_score: cached_plan.plan.gls_score,
                    latency_ms: latency_ms as f64,
                    index_stats: crate::core::service_types::IndexStats::default(),
                    candidate_count: results.len() as u32,
                    rerank_count: results.len() as u32,
                    repair_count: 0,
                    sure_signals: crate::observability::search_plan_trace::SureSignals::default(),
                    cache_result,
                    failure_class: None,
                    // bytes_per_vector: the trace builder needs this
                    // for actual_scan_gb derivation. Pass 0.0 until
                    // engine instrumentation lands (Phase 3); the
                    // builder skips the derivation and leaves
                    // actual_scan_gb at 0.0.
                    bytes_per_vector: 0.0,
                    // C3 egress (Dimension 2): per-query cross-AZ bytes captured
                    // in-scope above → actual_egress_gb (the KEU billing quantity
                    // the control plane prices). 0 on the free same-AZ path.
                    egress_bytes,
                    // TD-IOTRACE-2: physical object-store meters from the
                    // io_trace snapshot (captured in-scope above). The physical
                    // KRU billing input (TD-IOTRACE-3).
                    object_store_gets,
                    object_store_bytes_read,
                    // TD-064: shortfall is now a first-class field on the
                    // service response (`resp.predicate_shortfall`),
                    // recomputed authoritatively against the final merged
                    // result. Feed the same value into the debug-gated
                    // EXPLAIN/trace so both surfaces agree.
                    predicate_shortfall: resp.predicate_shortfall.clone(),
                    // Phase K: TurboQuant EXPLAIN payload pulled from
                    // the same task-local bus. `None` for searches
                    // that didn't route through TurboQuant scoring;
                    // present with the full 9-field payload otherwise.
                    turboquant_explain: turboquant_hints.clone(),
                },
            );

            // Emit the populated trace as a structured tracing span so
            // existing log aggregators (Loki, etc.) ingest it without
            // needing a dedicated billing sink yet. Tier-aware
            // sampling + retention live downstream of this — every
            // search emits one structured event; the gateway's log
            // pipeline samples per its own policy.
            //
            // Fields land as labels on the log entry. trace_id keeps
            // the entry correlatable with the response's
            // search_plan_trace block when debug=true; tenant_id +
            // collection are kept as bounded-cardinality labels.
            // Enum fields use Debug formatting (they have stable
            // snake_case Display via serde, but the tracing macro
            // doesn't auto-convert; the Debug shape is good enough
            // for log search).
            tracing::info!(
                trace_id = %trace.trace_id,
                tenant_id = %trace.tenant_id,
                collection = %trace.collection_name,
                filter_strategy = ?trace.filter_strategy,
                index_route = ?trace.index_route,
                cache_result = ?trace.cache_result,
                latency_ms = trace.latency_ms,
                candidate_count = trace.candidate_count,
                "search_plan_trace"
            );

            // Gate the trace + explain emission on debug=true per LLD §1.
            // Non-debug responses keep the JSON payload tight; debug
            // responses surface the full trace + a human-readable route
            // explain (route_explain::build over the populated trace).
            let debug_requested = request.debug.unwrap_or(false);
            let (trace_field, explain_field) = if debug_requested {
                // TD-064 / LLD §5: surface the recall-probe gate state to
                // EXPLAIN so the route_explain hint chain can emit
                // RECALL_PROBE_CLOSED when the model wanted the quantized
                // route but the gate isn't open for this (tenant, collection)
                // scope. Only paid on debug=true requests — non-debug
                // searches skip the lock acquisition entirely.
                let recall_probe_open = match &state.recall_probe_gate {
                    Some(gate) => {
                        let scope = crate::catalog::ProbeScope::new(
                            tenant.tenant_id.clone(),
                            collection.clone(),
                        );
                        Some(gate.is_open(&scope).await)
                    }
                    None => None,
                };
                let explain = crate::observability::route_explain::build(
                    &crate::observability::route_explain::ExplainInputs {
                        trace: &trace,
                        // collection_gb not yet plumbed from the
                        // catalog; pass 0.0 and accept the
                        // scan-fraction-zero fallback until
                        // collection-size hydration lands.
                        corpus_gb: 0.0,
                        recall_probe_open,
                        quantized_route_downgraded,
                        cold_stage1_only,
                        vector_bounds_pruned_blocks,
                    },
                );
                (Some(trace.clone()), Some(explain))
            } else {
                (None, None)
            };
            let response = TypedSearchResponse {
                results: results.clone(),
                total_matches: Some(total_matches),
                latency_ms,
                request_id: request_id.clone(),
                // TD-064: typed, top-level, always-on shortfall (mapped from
                // the service response field). `None` on the happy path.
                predicate_shortfall: resp.predicate_shortfall.as_ref().map(Into::into),
                search_plan_trace: trace_field,
                explain: explain_field,
            };

            info!(
                "V2 API: Search {} completed in {}ms, {} results",
                request_id,
                latency_ms,
                response.results.len()
            );

            Ok(Json(response))
        }
        Err(e) => {
            error!("V2 API: Search failed: {}", e);
            if e.to_string().contains("not found") {
                Err(ApiError::CollectionNotFound(collection))
            } else {
                Err(ApiError::Internal(format!("Search failed: {}", e)))
            }
        }
    }
}

/// Query parameters for getting a single record
#[derive(Debug, Deserialize, IntoParams)]
#[into_params(parameter_in = Query)]
pub struct GetRecordV2Query {
    /// Whether to include the vector in the response
    pub include_vector: Option<bool>,
    /// Whether to include TEXT fields in the response
    pub include_text: Option<bool>,
}

/// Response for getting a single record
#[derive(Debug, Serialize, ToSchema)]
pub struct RecordV2Response {
    /// Record ID
    pub id: String,
    /// Vector embedding (if requested)
    #[schema(value_type = Option<Vec<f32>>)]
    pub vector: Option<Vec<f32>>,
    /// Rich properties from the record
    #[schema(value_type = HashMap<String, Value>)]
    pub props: HashMap<String, RestProximaValue>,
    /// TEXT fields (if include_text is true)
    pub text_fields: Option<Vec<TextFieldOutput>>,
    /// Record version
    pub version: Option<u64>,
    /// Record timestamp
    pub timestamp: Option<i64>,
}

/// Response for deleting a single record.
#[derive(Debug, Serialize, ToSchema)]
pub struct DeleteRecordV2Response {
    /// Whether the delete tombstone was accepted.
    pub success: bool,
    /// Deleted record ID.
    pub id: String,
    /// Processing latency in microseconds.
    pub processing_time_us: i64,
}

/// GET /api/v2/collections/{collection_id}/records/{record_id}
///
/// Get a single record by ID.
///
/// ## Path Parameters
///
/// - `collection_id`: Collection name/ID
/// - `record_id`: Record ID
///
/// ## Query Parameters
///
/// - `include_vector`: Whether to include the vector (default: true)
/// - `include_text`: Whether to include TEXT fields (default: false)
///
/// ## Response
///
/// Returns [`RecordV2Response`] with record details.
///
/// ## Errors
///
/// - `404 Not Found`: Collection or record does not exist
/// - `500 Internal Server Error`: Retrieval failed
#[utoipa::path(
    get,
    path = "/api/v2/collections/{collection_id}/records/{record_id}",
    tag = "Records",
    operation_id = "getRecord",
    summary = "Get a record by ID.",
    params(
        ("collection_id" = String, Path, description = "Collection name/ID."),
        ("record_id" = String, Path, description = "Record ID."),
        GetRecordV2Query,
    ),
    responses(
        (status = 200, description = "Record.", body = RecordV2Response),
        (status = 404, description = "Resource not found.", body = crate::network::rest::openapi::ErrorResponse),
    ),
)]
pub async fn get_record_v2(
    Path((collection_id, record_id)): Path<(String, String)>,
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    Query(params): Query<GetRecordV2Query>,
) -> ApiResult<Json<RecordV2Response>> {
    debug!(
        "V2 API: Getting record '{}' from collection '{}'",
        record_id, collection_id
    );

    if collection_id.is_empty() {
        return Err(ApiError::InvalidArgument(
            "Collection ID is required".to_string(),
        ));
    }

    if record_id.is_empty() {
        return Err(ApiError::InvalidArgument(
            "Record ID is required".to_string(),
        ));
    }

    let include_vector = params.include_vector.unwrap_or_else(|| {
        debug!("No include_vector preference provided, defaulting to true");
        true
    });
    let include_text = params.include_text.unwrap_or_else(|| {
        debug!("No include_text preference provided, defaulting to false");
        false
    });

    match state
        .record_ops
        .handle_record_get_for_tenant(
            RichRecordGetRequest {
                collection_id: collection_id.clone(),
                record_id: record_id.clone(),
                include_vector,
                include_props: true,
            },
            proximadb_runtime::PortIdentity::from(&tenant),
        )
        .await
    {
        Ok(Some(record)) => {
            let props: HashMap<String, RestProximaValue> = record
                .props
                .iter()
                .map(|(k, v)| (k.clone(), proxima_value_to_rest_value(v)))
                .collect();

            let response = RecordV2Response {
                id: record.id,
                vector: if include_vector {
                    Some(record.vector)
                } else {
                    None
                },
                props,
                text_fields: if include_text {
                    Some(vec![]) // Would be populated from text storage
                } else {
                    None
                },
                version: record.version.map(|v| v as u64),
                timestamp: record.timestamp,
            };

            Ok(Json(response))
        }
        Ok(None) => Err(ApiError::NotFound(format!(
            "Record '{}' not found in collection '{}'",
            record_id, collection_id
        ))),
        Err(e) => {
            if e.to_string().contains("not found") {
                Err(ApiError::NotFound(format!(
                    "Record '{}' not found in collection '{}'",
                    record_id, collection_id
                )))
            } else {
                Err(ApiError::Internal(format!("Failed to get record: {}", e)))
            }
        }
    }
}

/// DELETE /api/v2/collections/{collection_id}/records/{record_id}
///
/// Delete a single record by writing a tombstone through the rich record path.
#[utoipa::path(
    delete,
    path = "/api/v2/collections/{collection_id}/records/{record_id}",
    tag = "Records",
    operation_id = "deleteRecord",
    summary = "Delete a record by ID.",
    params(
        ("collection_id" = String, Path, description = "Collection name/ID."),
        ("record_id" = String, Path, description = "Record ID."),
    ),
    responses(
        (status = 200, description = "Delete accepted.", body = DeleteRecordV2Response),
    ),
)]
pub async fn delete_record_v2(
    Path((collection_id, record_id)): Path<(String, String)>,
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
) -> ApiResult<Json<DeleteRecordV2Response>> {
    debug!(
        "V2 API: Deleting record '{}' from collection '{}'",
        record_id, collection_id
    );

    if collection_id.is_empty() {
        return Err(ApiError::InvalidArgument(
            "Collection ID is required".to_string(),
        ));
    }

    if record_id.is_empty() {
        return Err(ApiError::InvalidArgument(
            "Record ID is required".to_string(),
        ));
    }

    // Primary-pod write-router gate (symmetric with `insert_records`). Deletes
    // are mutations too: a delete accepted on a displaced pod would never reach
    // the primary's memtable, leaving the record live on the primary. Gate
    // BEFORE any storage work.
    match crate::cluster::primary_pod_registry::consult_for_write(
        &state.primary_pod_registry,
        &state.self_pod_id,
        &tenant.tenant_id,
        &collection_id,
    ) {
        crate::cluster::primary_pod_registry::WriteRoutingDecision::Allow => {
            if state
                .primary_pod_registry
                .is_assigned(&tenant.tenant_id, &collection_id)
            {
                crate::metrics::primary_pod_metrics::record_allowed_bound(&tenant.tenant_id);
            } else {
                crate::metrics::primary_pod_metrics::record_allowed_unbounded(&tenant.tenant_id);
            }
        }
        crate::cluster::primary_pod_registry::WriteRoutingDecision::Misrouted { target_pod } => {
            crate::metrics::primary_pod_metrics::record_misrouted(&tenant.tenant_id);
            tracing::warn!(
                target = "proximadb.primary_pod.misroute",
                self_pod = %state.self_pod_id,
                target_pod = %target_pod,
                tenant_id = %tenant.tenant_id,
                collection_id = %collection_id,
                "v2 delete misrouted — client SDK should retry against the primary pod"
            );
            return Err(ApiError::Misdirected {
                target_pod,
                tenant_id: tenant.tenant_id.clone(),
                collection_id: collection_id.clone(),
            });
        }
    }

    let intent = WriteIntent::new(&collection_id, WriteOperationKind::Delete)
        .with_durability(WriteDurabilityRequirement::WalRequired)
        .with_row_count_hint(1);
    let lane = WriteLaneRouter::new().route(&intent);
    debug!(
        collection_id = %collection_id,
        write_lane = ?lane.lane,
        guards = ?lane.required_guards,
        "REST v2 delete_record write-lane decision"
    );

    match state
        .record_ops
        .handle_record_delete_batch_for_tenant(
            RichRecordDeleteBatchRequest {
                collection_id: collection_id.clone(),
                record_ids: vec![record_id.clone()],
            },
            Some(&tenant.tenant_id),
        )
        .await
    {
        Ok(result) if result.success => Ok(Json(DeleteRecordV2Response {
            success: true,
            id: record_id,
            processing_time_us: result.metrics.processing_time_us,
        })),
        Ok(result) => Err(ApiError::Internal(format!(
            "Delete failed: {}",
            result
                .errors
                .first()
                .cloned()
                .unwrap_or_else(|| "unknown error".to_string())
        ))),
        Err(e) => {
            if e.to_string().contains("not found") {
                Err(ApiError::NotFound(format!(
                    "Record '{}' not found in collection '{}'",
                    record_id, collection_id
                )))
            } else {
                Err(ApiError::Internal(format!(
                    "Failed to delete record: {}",
                    e
                )))
            }
        }
    }
}

// ── TD-099: paginated records scan ──────────────────────────────────
//
// Cursor-paginated table-scan over a collection. Spec'd at
// `POST /api/v2/collections/{collection_id}/records/scan`
// (operationId `scanRecords`). The Hadoop connector's RecordReader and
// Spark's planInputPartitions use this to drain a collection without
// the similarity bias of `searchRecords`.
//
// Storage-engine delegation (TD-099 acceptance 2) shipped in
// `c8050ab1a`: the handler drives `UnifiedHandlers::handle_record_scan_for_tenant`
// → `VectorOperationsService::scan_records_with_tenant_context` (the
// same WAL/memtable scan used by the rest of the v2 surface).
//
// Cursor pagination (TD-099 acceptance 3, sub-slice a): the handler
// now stable-sorts records by `(updated_at_ns, oid)` then filters
// strictly after the inbound cursor's tuple. Cursors are opaque
// `base64(rmp_serde(ScanCursor))` blobs; stale cursors (epoch > 24h)
// return 410, collection-mismatched cursors return 400. The
// `VectorOperationsService` still materializes the full collection
// (cursor is applied at the handler layer); pushing cursor into the
// WAL streaming layer is a separate slice — see TD-099 acceptance (3).

// ScanCursor + apply_scan_cursor moved to `src/services/scan_cursor.rs`
// (shared codec — used by both this REST handler and
// `EmbeddedProximaDB::scan_records`). REST-specific error mapping
// (HTTP 400 / 410) happens at the handler boundary; the codec itself
// stays protocol-agnostic.
pub(crate) use crate::services::scan_cursor::{ScanCursor, ScanCursorDecodeError};

/// Map the shared codec's decode errors onto our HTTP-mapped
/// ApiError variants. Kept thin so the codec stays protocol-neutral.
fn decode_scan_cursor(
    raw: &str,
    requested_collection: &str,
    now_ns: i64,
) -> Result<ScanCursor, ApiError> {
    ScanCursor::decode(raw, requested_collection, now_ns).map_err(|e| match e {
        ScanCursorDecodeError::Expired => ApiError::Gone(e.to_string()),
        ScanCursorDecodeError::CollectionMismatch { .. } | ScanCursorDecodeError::Malformed(_) => {
            ApiError::InvalidArgument(e.to_string())
        }
        ScanCursorDecodeError::ContentRevisionMismatch { .. }
        | ScanCursorDecodeError::IncarnationMismatch => ApiError::Conflict(e.to_string()),
    })
}

fn validate_scan_content_revision(
    cursor: Option<&ScanCursor>,
    current_revision: u64,
    current_token: &str,
) -> Result<(), ApiError> {
    if let Some(cursor) = cursor {
        cursor
            .validate_content_revision(current_revision, current_token)
            .map_err(|error| ApiError::Conflict(error.to_string()))?;
    }
    Ok(())
}

/// Body of `POST /records/scan`. Mirrors the OpenAPI `ScanRecordsRequest`
/// schema; all fields optional so an empty `{}` returns the first page.
#[derive(Debug, Deserialize, Default, ToSchema)]
pub struct ScanRecordsRequest {
    /// Opaque continuation token returned as `next_cursor` from a
    /// prior page. Omit / null to start from the beginning.
    #[serde(default)]
    pub cursor: Option<String>,
    /// Max records to return in this page. Server enforces upper bound.
    #[serde(default)]
    pub limit: Option<u32>,
    /// Metadata filter applied (before the limit) to the scanned page.
    /// Accepts either the typed list form `[{field,op,value}]` (mirrors
    /// `searchRecords.filters`) or a simple equality map `{field: value}`.
    #[serde(default)]
    #[schema(value_type = Option<Value>)]
    pub filter: Option<serde_json::Value>,
    #[serde(default)]
    pub include_vector: Option<bool>,
    #[serde(default)]
    pub include_text: Option<bool>,
}

/// Response shape for `scanRecords`. Empty `records` + null
/// `next_cursor` signals end-of-scan. Each record matches the OpenAPI
/// `RecordResponse` schema (same shape used by `getRecord`).
#[derive(Debug, Serialize, ToSchema)]
pub struct ScanRecordsResponse {
    pub records: Vec<RecordV2Response>,
    pub content_revision: u64,
    pub content_revision_token: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scanned_count: Option<u64>,
}

/// Server-side cap on a single scan page. Callers asking for more get
/// silently clamped — keeps a wild `limit=u32::MAX` from materializing
/// the whole collection in one round trip.
const SCAN_RECORDS_MAX_PAGE: usize = 10_000;
/// Default `limit` when the caller omits one. Mirrors the Hadoop SDK
/// default in `clients/rust/src/connectors/hadoop.rs`.
const SCAN_RECORDS_DEFAULT_LIMIT: usize = 1_000;

/// POST /api/v2/collections/{collection_id}/records/scan
///
/// Returns the next page of records (TD-099 acceptance 2 + 3, live).
/// Cursor-based pagination: pass the response's `next_cursor` back as
/// `cursor` to fetch the next page; a `null` cursor means the scan is
/// exhausted. Cursors are minted and validated against the caller-facing
/// collection id in the URL path, expire after 24h (HTTP 410), and reject
/// cross-collection reuse (HTTP 400).
#[utoipa::path(
    post,
    path = "/api/v2/collections/{collection_id}/records/scan",
    tag = "Records",
    operation_id = "scanRecords",
    summary = "Paginated scan of records in a collection.",
    params(
        ("collection_id" = String, Path, description = "Collection name/ID."),
    ),
    request_body = ScanRecordsRequest,
    responses(
        (status = 200, description = "Page of records + optional next cursor.", body = ScanRecordsResponse),
        (status = 400, description = "Invalid request (including a cursor minted for a different collection).", body = crate::network::rest::openapi::ErrorResponse),
        (status = 404, description = "Resource not found.", body = crate::network::rest::openapi::ErrorResponse),
        (status = 409, description = "Collection changed during the scan; restart pagination.", body = crate::network::rest::openapi::ErrorResponse),
        (status = 410, description = "Cursor expired (older than 24h); restart the scan.", body = crate::network::rest::openapi::ErrorResponse),
    ),
)]
pub async fn scan_records(
    Path(collection_id): Path<String>,
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    Json(request): Json<ScanRecordsRequest>,
) -> ApiResult<Json<ScanRecordsResponse>> {
    debug!("V2 API: scan_records for collection '{}'", collection_id);

    if collection_id.is_empty() {
        return Err(ApiError::InvalidArgument(
            "Collection ID is required".to_string(),
        ));
    }

    let include_vector = request.include_vector.unwrap_or(true);
    let include_text = request.include_text.unwrap_or(false);
    let include_props = true;
    let effective_limit = clamp_scan_limit(request.limit);
    let now_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(0);

    let (canonical_collection_id, _) = state
        .record_ops
        .resolve_collection_identity_for_tenant(&collection_id, Some(&tenant.tenant_id))
        .await
        .map_err(|error| ApiError::Internal(format!("collection resolution failed: {error}")))?
        .ok_or_else(|| ApiError::NotFound(format!("Collection '{collection_id}' not found")))?;
    let (content_revision, content_revision_token) = state
        .record_ops
        .content_revision_snapshot(Some(&tenant.tenant_id), &canonical_collection_id)
        .await;

    let inbound_cursor: Option<ScanCursor> = match request.cursor.as_deref() {
        Some(raw) if !raw.is_empty() => {
            Some(decode_scan_cursor(raw, &canonical_collection_id, now_ns)?)
        }
        _ => None,
    };
    validate_scan_content_revision(
        inbound_cursor.as_ref(),
        content_revision,
        &content_revision_token,
    )?;

    // Metadata filter is parsed into the canonical FilterExpression and pushed
    // into the scan predicate (applied before the limit). Previously this field
    // was accepted but silently ignored — a cross-tenant leak for callers that
    // scope a shared collection with an `account_id`/`tenant_id` filter.
    let metadata_filter = match request.filter.as_ref() {
        Some(value) => parse_metadata_filter(value)?,
        None => None,
    };

    // The scan merges flushed segments + unflushed WAL, filters dead records
    // and the metadata predicate, then pages by the (updated_at_ns, oid)
    // keyset cursor. NOTE: this path fully materializes the collection per
    // page today — the TD-099(3d) WAL push-down is NOT wired here (immutable
    // engines expose only read_all_records); restoring it is tracked in
    // TECHNICAL_DEBT.adoc under TD-099.
    let (page, next_cursor) = state
        .record_ops
        .handle_record_scan_paginated_for_tenant(
            &canonical_collection_id,
            inbound_cursor.as_ref(),
            effective_limit,
            include_vector,
            include_props,
            Some(&tenant.tenant_id),
            metadata_filter.as_ref(),
            now_ns,
        )
        .await
        .map_err(|e| {
            if e.to_string().contains("not found") {
                ApiError::NotFound(format!("Collection '{}' not found", collection_id))
            } else {
                ApiError::Internal(format!("scan_records failed: {}", e))
            }
        })?;

    let scanned_count = page.len() as u64;
    let serialized: Vec<RecordV2Response> = page
        .into_iter()
        .map(|record| proxima_record_to_response(record, include_vector, include_text))
        .collect();

    let revision_after = state
        .record_ops
        .content_revision_snapshot(Some(&tenant.tenant_id), &canonical_collection_id)
        .await;
    if revision_after != (content_revision, content_revision_token.clone()) {
        return Err(ApiError::Conflict(
            "collection changed while scanning; restart pagination".to_string(),
        ));
    }

    let next_cursor_str = match next_cursor {
        Some(mut cursor) => {
            cursor.stamp_content_revision(content_revision, content_revision_token.clone());
            Some(
                cursor
                    .encode()
                    .map_err(|e| ApiError::Internal(format!("cursor encode failed: {e}")))?,
            )
        }
        None => None,
    };

    Ok(Json(ScanRecordsResponse {
        records: serialized,
        content_revision,
        content_revision_token,
        next_cursor: next_cursor_str,
        scanned_count: Some(scanned_count),
    }))
}

/// Clamp the caller's requested `limit` into the
/// `[1, SCAN_RECORDS_MAX_PAGE]` range. `None` defaults to
/// `SCAN_RECORDS_DEFAULT_LIMIT`. Pure for testing.
fn clamp_scan_limit(requested: Option<u32>) -> usize {
    match requested {
        None => SCAN_RECORDS_DEFAULT_LIMIT,
        Some(0) => SCAN_RECORDS_DEFAULT_LIMIT,
        Some(n) => (n as usize).min(SCAN_RECORDS_MAX_PAGE),
    }
}

/// Convert a canonical `ProximaRecord` into the `RecordResponse`
/// wire shape used by both `getRecord` and `scanRecords`. Honors
/// `include_vector` + `include_text`; props always populated.
fn proxima_record_to_response(
    record: ProximaRecord,
    include_vector: bool,
    include_text: bool,
) -> RecordV2Response {
    let vector = if include_vector {
        record
            .embeddings
            .into_iter()
            .next()
            .map(|cell| cell.values.to_fp32_owned())
    } else {
        None
    };

    let props: HashMap<String, RestProximaValue> = record
        .props
        .into_iter()
        .filter_map(|(k, node)| match node {
            ProximaTreeNode::Value(v) => Some((k, proxima_value_to_rest_value(&v))),
            _ => None,
        })
        .collect();

    let version = if record.record_version == 0 {
        None
    } else {
        Some(record.record_version)
    };
    let timestamp = if record.updated_at_ns == 0 {
        None
    } else {
        // ProximaRecord timestamps are nanoseconds; the v2 response
        // shape historically uses the same ns granularity (see
        // get_record_v2). Pass through unchanged.
        Some(record.updated_at_ns)
    };

    RecordV2Response {
        id: if record.oid.is_empty() {
            record.local_id.unwrap_or_else(|| "unknown".to_string())
        } else {
            record.oid
        },
        vector,
        props,
        text_fields: if include_text { Some(vec![]) } else { None },
        version,
        timestamp,
    }
}

/// Query parameters for the CDC change-feed endpoint.
#[derive(Debug, serde::Deserialize)]
pub struct ChangesQuery {
    /// Return changes with WAL sequence number strictly greater than this (default 0 = all).
    #[serde(default)]
    pub since_lsn: u64,
}

/// `GET /api/v2/collections/{collection_id}/changes?since_lsn=N` — CDC change-feed.
///
/// Returns row-level changes (op + lsn + after-image) for the table, ordered by lsn, from
/// the unified canonical store — so it reflects writes made over ANY surface (REST/gRPC/
/// pgwire). The cursor is the `lsn`: pass the highest seen `lsn` back as `since_lsn` to poll
/// incrementally.
pub async fn get_changes(
    Path(collection_id): Path<String>,
    State(state): State<AppState>,
    Query(q): Query<ChangesQuery>,
) -> ApiResult<Json<serde_json::Value>> {
    let changes = state
        .record_ops
        .table_changes(&collection_id, q.since_lsn)
        .await
        .map_err(|e| ApiError::Internal(format!("change-feed read failed: {e}")))?;
    let next_lsn = changes.iter().map(|c| c.lsn).max();
    Ok(Json(serde_json::json!({
        "collection": collection_id,
        "since_lsn": q.since_lsn,
        "count": changes.len(),
        "next_lsn": next_lsn,
        "changes": changes,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rv(value: serde_json::Value) -> RestProximaValue {
        RestProximaValue::Inferred(value)
    }

    #[test]
    fn parse_metadata_filter_object_form_is_equality_and() {
        use crate::core::search::{ComparisonOperator, FilterExpression};

        // Single key → a single Equals comparison (the account-scoping case).
        let expr = parse_metadata_filter(&serde_json::json!({ "account_id": "acctA" }))
            .expect("valid")
            .expect("some");
        match expr {
            FilterExpression::Comparison {
                field,
                operator,
                value,
            } => {
                assert_eq!(field, "account_id");
                assert_eq!(operator, ComparisonOperator::Equals);
                assert_eq!(value, serde_json::json!("acctA"));
            }
            other => panic!("expected single comparison, got {other:?}"),
        }

        // Multiple keys → an AND of equality comparisons.
        let expr = parse_metadata_filter(&serde_json::json!({ "account_id": "acctA", "tier": 2 }))
            .expect("valid")
            .expect("some");
        match expr {
            FilterExpression::And(conditions) => assert_eq!(conditions.len(), 2),
            other => panic!("expected AND, got {other:?}"),
        }
    }

    #[test]
    fn parse_metadata_filter_array_form_lowers_typed_ops() {
        use crate::core::search::{ComparisonOperator, FilterExpression};

        let expr = parse_metadata_filter(&serde_json::json!([
            { "field": "name", "op": "starts_with", "value": "ac" }
        ]))
        .expect("valid")
        .expect("some");
        match expr {
            FilterExpression::Comparison {
                operator, field, ..
            } => {
                assert_eq!(field, "name");
                // Must NOT collapse to Contains.
                assert_eq!(operator, ComparisonOperator::StartsWith);
            }
            other => panic!("expected comparison, got {other:?}"),
        }
    }

    #[test]
    fn parse_metadata_filter_empty_and_invalid() {
        assert!(
            parse_metadata_filter(&serde_json::Value::Null)
                .unwrap()
                .is_none()
        );
        assert!(
            parse_metadata_filter(&serde_json::json!({}))
                .unwrap()
                .is_none()
        );
        assert!(
            parse_metadata_filter(&serde_json::json!([]))
                .unwrap()
                .is_none()
        );
        // Bad operator in the typed-list form is rejected.
        assert!(
            parse_metadata_filter(&serde_json::json!([
                { "field": "x", "op": "bogus", "value": 1 }
            ]))
            .is_err()
        );
        // A bare scalar is neither an object nor a list.
        assert!(parse_metadata_filter(&serde_json::json!("nope")).is_err());
    }

    #[test]
    fn test_clamp_scan_limit_defaults_when_missing_or_zero() {
        assert_eq!(clamp_scan_limit(None), SCAN_RECORDS_DEFAULT_LIMIT);
        assert_eq!(clamp_scan_limit(Some(0)), SCAN_RECORDS_DEFAULT_LIMIT);
    }

    #[test]
    fn test_clamp_scan_limit_passes_through_small_values() {
        assert_eq!(clamp_scan_limit(Some(1)), 1);
        assert_eq!(clamp_scan_limit(Some(42)), 42);
        assert_eq!(
            clamp_scan_limit(Some(SCAN_RECORDS_MAX_PAGE as u32)),
            SCAN_RECORDS_MAX_PAGE
        );
    }

    #[test]
    fn test_clamp_scan_limit_caps_oversize_requests() {
        assert_eq!(
            clamp_scan_limit(Some(u32::MAX)),
            SCAN_RECORDS_MAX_PAGE,
            "oversize limit must be clamped to the page cap"
        );
        assert_eq!(
            clamp_scan_limit(Some(SCAN_RECORDS_MAX_PAGE as u32 + 1)),
            SCAN_RECORDS_MAX_PAGE,
        );
    }

    #[test]
    fn test_scan_records_request_deserializes_bare_body() {
        // OpenAPI spec marks every field optional; a bare {} must parse.
        let parsed: ScanRecordsRequest = serde_json::from_str("{}").unwrap();
        assert!(parsed.cursor.is_none());
        assert!(parsed.limit.is_none());
        assert!(parsed.filter.is_none());
        assert!(parsed.include_vector.is_none());
        assert!(parsed.include_text.is_none());
    }

    #[test]
    fn test_scan_records_request_round_trips_full_body() {
        let body = r#"{
            "cursor": "abc",
            "limit": 250,
            "filter": {"label": "active"},
            "include_vector": false,
            "include_text": true
        }"#;
        let parsed: ScanRecordsRequest = serde_json::from_str(body).unwrap();
        assert_eq!(parsed.cursor.as_deref(), Some("abc"));
        assert_eq!(parsed.limit, Some(250));
        assert_eq!(parsed.include_vector, Some(false));
        assert_eq!(parsed.include_text, Some(true));
        assert!(parsed.filter.is_some());
    }

    #[test]
    fn test_proxima_record_to_response_extracts_oid_props_and_vector() {
        use proximadb_data_model::ProximaValue;
        use proximadb_records::{EmbeddingCell, ProximaRecord, ProximaTreeNode};

        let mut record = ProximaRecord::default();
        record.oid = "rec-1".to_string();
        record.record_version = 7;
        record.updated_at_ns = 1_700_000_000_000_000_000;
        record.props.insert(
            "label".to_string(),
            ProximaTreeNode::Value(ProximaValue::String("active".to_string())),
        );
        record.embeddings.push(EmbeddingCell::new_fp32(
            "model-x",
            "vector",
            3,
            vec![0.1, 0.2, 0.3],
        ));

        let resp = proxima_record_to_response(record, true, false);
        assert_eq!(resp.id, "rec-1");
        assert_eq!(resp.version, Some(7));
        assert_eq!(resp.timestamp, Some(1_700_000_000_000_000_000));
        assert_eq!(resp.vector.as_ref().map(|v| v.len()), Some(3));
        assert!(resp.props.contains_key("label"));
        assert!(resp.text_fields.is_none(), "include_text=false → None");
    }

    #[test]
    fn test_proxima_record_to_response_omits_vector_when_not_requested() {
        use proximadb_records::{EmbeddingCell, ProximaRecord};

        let mut record = ProximaRecord::default();
        record.oid = "rec-2".to_string();
        record.embeddings.push(EmbeddingCell::new_fp32(
            "model-x",
            "vector",
            2,
            vec![1.0, 2.0],
        ));

        let resp = proxima_record_to_response(record, false, true);
        assert!(resp.vector.is_none(), "include_vector=false → None");
        match resp.text_fields {
            Some(fields) => assert_eq!(fields.len(), 0, "include_text=true → empty Vec"),
            None => panic!("include_text=true must yield Some(vec)"),
        }
    }

    // Cursor codec + apply_scan_cursor tests live with the shared
    // module: `src/services/scan_cursor.rs::tests`. The handler-side
    // tests below only verify the REST-specific error mapping
    // (decode_scan_cursor → ApiError variants → HTTP status codes).

    fn fixture_cursor_raw() -> String {
        ScanCursor {
            collection_id: "col-a".to_string(),
            last_updated_at_ns: 1_700_000_000_000_000_000,
            last_oid: "rec-077".to_string(),
            epoch_ns: 1_700_000_000_000_000_000,
            content_revision: None,
            content_revision_token: None,
        }
        .encode()
        .expect("encode")
    }

    #[test]
    fn test_decode_scan_cursor_maps_stale_to_gone_410() {
        let raw = fixture_cursor_raw();
        // 25 hours later, past the 24h ceiling.
        let now_ns = 1_700_000_000_000_000_000_i64 + 25 * 3_600 * 1_000_000_000;
        match decode_scan_cursor(&raw, "col-a", now_ns) {
            Err(ApiError::Gone(_)) => {}
            other => panic!("expected ApiError::Gone, got {:?}", other),
        }
    }

    #[test]
    fn test_decode_scan_cursor_maps_mismatch_and_malformed_to_400() {
        let raw = fixture_cursor_raw();
        let now_ns = 1_700_000_000_000_000_000_i64;
        // Collection mismatch.
        match decode_scan_cursor(&raw, "col-OTHER", now_ns) {
            Err(ApiError::InvalidArgument(msg)) => {
                assert!(
                    msg.contains("col-a"),
                    "msg must surface issuing collection: {msg}"
                );
                assert!(msg.contains("col-OTHER"), "msg must surface target: {msg}");
            }
            other => panic!("expected ApiError::InvalidArgument, got {:?}", other),
        }
        // Malformed base64.
        assert!(matches!(
            decode_scan_cursor("not!valid!base64", "col-a", now_ns),
            Err(ApiError::InvalidArgument(_))
        ));
    }

    #[test]
    fn scan_cursor_revision_rejects_mixed_era_pages() {
        let mut cursor =
            ScanCursor::decode(&fixture_cursor_raw(), "col-a", 1_700_000_000_000_000_000)
                .expect("cursor");
        cursor.content_revision = Some(41);
        cursor.content_revision_token = Some("epoch-a:41".into());
        assert!(validate_scan_content_revision(Some(&cursor), 41, "epoch-a:41").is_ok());
        assert!(matches!(
            validate_scan_content_revision(Some(&cursor), 42, "epoch-a:42"),
            Err(ApiError::Conflict(_))
        ));
    }

    #[test]
    fn test_insert_request_deserialization() {
        let json = r#"{
            "records": [
                {
                    "id": "doc_1",
                    "vector": [0.1, 0.2, 0.3],
                    "props": {
                        "category": "test",
                        "price": 99.99
                    },
                    "text_fields": [
                        {
                            "name": "content",
                            "content": "Test content",
                            "storage_hint": "adaptive"
                        }
                    ]
                }
            ],
            "validate_schema": true
        }"#;

        let request: InsertRecordsRequest = serde_json::from_str(json)
            .expect("Failed to deserialize InsertRecordsRequest from test JSON");
        assert_eq!(request.records.len(), 1);
        assert_eq!(request.records[0].id, Some("doc_1".to_string()));
        assert_eq!(request.records[0].vector.len(), 3);
        assert_eq!(request.validate_schema, Some(true));
    }

    #[test]
    fn test_search_request_deserialization() {
        let json = r#"{
            "vector": [0.1, 0.2, 0.3],
            "top_k": 10,
            "filters": [
                {"field": "category", "op": "eq", "value": "electronics"},
                {"field": "price", "op": "lt", "value": 500}
            ],
            "include_text": true
        }"#;

        let request: TypedSearchRequest = serde_json::from_str(json)
            .expect("Failed to deserialize TypedSearchRequest from test JSON");
        assert_eq!(request.vector.len(), 3);
        assert_eq!(request.top_k, 10);
        assert_eq!(request.include_text, Some(true));

        let filters = request.filters.as_ref().expect("filters should be Some");
        assert_eq!(filters.len(), 2);
        assert_eq!(filters[0].field, "category");
        assert_eq!(filters[0].op, "eq");
    }

    #[test]
    fn test_typed_filter_between_validation() {
        let json = r#"{
            "field": "price",
            "op": "between",
            "value": {"type": "decimal", "value": "100.00"},
            "value_upper": {"type": "decimal", "value": "500.00"}
        }"#;

        let filter: TypedFilter =
            serde_json::from_str(json).expect("Failed to deserialize TypedFilter from test JSON");
        assert_eq!(filter.op, "between");
        assert!(filter.value_upper.is_some());
    }

    #[test]
    fn test_typed_filter_to_rich_preserves_explicit_decimal() {
        let filter = TypedFilter {
            field: "price".to_string(),
            op: "between".to_string(),
            value: RestProximaValue::Typed {
                type_name: "decimal".to_string(),
                value: serde_json::json!("10.50"),
            },
            value_upper: Some(RestProximaValue::Typed {
                type_name: "decimal".to_string(),
                value: serde_json::json!("20.75"),
            }),
        };

        let rich = typed_filter_to_rich(&filter).expect("typed filter should convert");

        assert!(matches!(rich.value, ProximaValue::Decimal(v) if v == "10.50"));
        assert!(matches!(rich.value_upper, Some(ProximaValue::Decimal(v)) if v == "20.75"));
    }

    #[test]
    fn test_proxima_value_to_rest_value_preserves_rich_type() {
        let value = ProximaValue::Jsonb(serde_json::json!({"tags": ["a", "b"]}));
        let rest = proxima_value_to_rest_value(&value);

        assert_eq!(
            rest,
            RestProximaValue::Typed {
                type_name: "jsonb".to_string(),
                value: serde_json::json!({"tags": ["a", "b"]}),
            }
        );
    }

    #[test]
    fn test_nested_rest_values_preserve_element_types() {
        let value = ProximaValue::Array(vec![ProximaValue::Decimal("10.50".to_string())]);
        let rest = proxima_value_to_rest_value(&value);

        assert_eq!(
            rest,
            RestProximaValue::Typed {
                type_name: "array".to_string(),
                value: serde_json::json!([{"type": "decimal", "value": "10.50"}]),
            }
        );
    }

    #[test]
    fn test_convert_eq_filter() {
        use crate::proto::proximadb_v1::ComparisonOp;

        let filters = vec![TypedFilter {
            field: "status".to_string(),
            op: "eq".to_string(),
            value: rv(serde_json::json!("active")),
            value_upper: None,
        }];

        let clauses = convert_typed_filters_to_clauses(&filters)
            .expect("Failed to convert typed filters to clauses");
        assert_eq!(clauses.len(), 1);
        assert_eq!(clauses[0].field, "status");
        assert_eq!(clauses[0].op, ComparisonOp::Eq as i32);
    }

    #[test]
    fn test_convert_range_filters() {
        use crate::proto::proximadb_v1::ComparisonOp;

        let filters = vec![
            TypedFilter {
                field: "price".to_string(),
                op: "gt".to_string(),
                value: rv(serde_json::json!(100)),
                value_upper: None,
            },
            TypedFilter {
                field: "price".to_string(),
                op: "gte".to_string(),
                value: rv(serde_json::json!(100)),
                value_upper: None,
            },
            TypedFilter {
                field: "price".to_string(),
                op: "lt".to_string(),
                value: rv(serde_json::json!(500)),
                value_upper: None,
            },
            TypedFilter {
                field: "price".to_string(),
                op: "lte".to_string(),
                value: rv(serde_json::json!(500)),
                value_upper: None,
            },
        ];

        let clauses = convert_typed_filters_to_clauses(&filters)
            .expect("Failed to convert typed filters to clauses");
        assert_eq!(clauses.len(), 4);
        assert_eq!(clauses[0].op, ComparisonOp::Gt as i32);
        assert_eq!(clauses[1].op, ComparisonOp::Gte as i32);
        assert_eq!(clauses[2].op, ComparisonOp::Lt as i32);
        assert_eq!(clauses[3].op, ComparisonOp::Lte as i32);
    }

    #[test]
    fn test_convert_between_filter() {
        use crate::proto::proximadb_v1::ComparisonOp;

        let filters = vec![TypedFilter {
            field: "price".to_string(),
            op: "between".to_string(),
            value: rv(serde_json::json!(100)),
            value_upper: Some(rv(serde_json::json!(500))),
        }];

        let clauses = convert_typed_filters_to_clauses(&filters)
            .expect("Failed to convert typed filters to clauses");
        // between is converted to two clauses: gte and lte
        assert_eq!(clauses.len(), 2);
        assert_eq!(clauses[0].field, "price");
        assert_eq!(clauses[0].op, ComparisonOp::Gte as i32);
        assert_eq!(clauses[1].field, "price");
        assert_eq!(clauses[1].op, ComparisonOp::Lte as i32);
    }

    #[test]
    fn test_convert_between_filter_missing_upper() {
        let filters = vec![TypedFilter {
            field: "price".to_string(),
            op: "between".to_string(),
            value: rv(serde_json::json!(100)),
            value_upper: None, // Missing upper bound
        }];

        let result = convert_typed_filters_to_clauses(&filters);
        assert!(result.is_err());
    }

    #[test]
    fn test_convert_contains_filter() {
        use crate::proto::proximadb_v1::ComparisonOp;

        let filters = vec![TypedFilter {
            field: "description".to_string(),
            op: "contains".to_string(),
            value: rv(serde_json::json!("search term")),
            value_upper: None,
        }];

        let clauses = convert_typed_filters_to_clauses(&filters)
            .expect("Failed to convert typed filters to clauses");
        assert_eq!(clauses.len(), 1);
        assert_eq!(clauses[0].op, ComparisonOp::Contains as i32);
    }

    #[test]
    fn test_convert_starts_with_filter() {
        use crate::proto::proximadb_v1::{ComparisonOp, filter_clause::Value};

        let filters = vec![TypedFilter {
            field: "name".to_string(),
            op: "starts_with".to_string(),
            value: rv(serde_json::json!("pre")),
            value_upper: None,
        }];

        let clauses = convert_typed_filters_to_clauses(&filters)
            .expect("Failed to convert typed filters to clauses");
        assert_eq!(clauses.len(), 1);
        assert_eq!(clauses[0].op, ComparisonOp::Contains as i32);
        // Verify the value is prefixed with ^
        if let Some(Value::StringValue(s)) = &clauses[0].value {
            assert_eq!(s, "^pre");
        } else {
            panic!("Expected StringValue");
        }
    }

    #[test]
    fn test_convert_ends_with_filter() {
        use crate::proto::proximadb_v1::{ComparisonOp, filter_clause::Value};

        let filters = vec![TypedFilter {
            field: "name".to_string(),
            op: "ends_with".to_string(),
            value: rv(serde_json::json!("suffix")),
            value_upper: None,
        }];

        let clauses = convert_typed_filters_to_clauses(&filters)
            .expect("Failed to convert typed filters to clauses");
        assert_eq!(clauses.len(), 1);
        assert_eq!(clauses[0].op, ComparisonOp::Contains as i32);
        // Verify the value is suffixed with $
        if let Some(Value::StringValue(s)) = &clauses[0].value {
            assert_eq!(s, "suffix$");
        } else {
            panic!("Expected StringValue");
        }
    }

    #[test]
    fn test_convert_in_filter() {
        use crate::proto::proximadb_v1::{ComparisonOp, filter_clause::Value};

        let filters = vec![TypedFilter {
            field: "status".to_string(),
            op: "in".to_string(),
            value: rv(serde_json::json!(["active", "pending", "review"])),
            value_upper: None,
        }];

        let clauses = convert_typed_filters_to_clauses(&filters)
            .expect("Failed to convert typed filters to clauses");
        assert_eq!(clauses.len(), 1);
        assert_eq!(clauses[0].op, ComparisonOp::In as i32);
        // Verify the value is a JSON array string
        if let Some(Value::StringValue(s)) = &clauses[0].value {
            assert!(s.starts_with('['));
            assert!(s.ends_with(']'));
            assert!(s.contains("active"));
            assert!(s.contains("pending"));
            assert!(s.contains("review"));
        } else {
            panic!("Expected StringValue");
        }
    }

    #[test]
    fn test_convert_in_filter_with_numbers() {
        use crate::proto::proximadb_v1::{ComparisonOp, filter_clause::Value};

        let filters = vec![TypedFilter {
            field: "priority".to_string(),
            op: "in".to_string(),
            value: rv(serde_json::json!([1, 2, 3])),
            value_upper: None,
        }];

        let clauses = convert_typed_filters_to_clauses(&filters)
            .expect("Failed to convert typed filters to clauses");
        assert_eq!(clauses.len(), 1);
        assert_eq!(clauses[0].op, ComparisonOp::In as i32);
        if let Some(Value::StringValue(s)) = &clauses[0].value {
            assert!(s.contains("1"));
            assert!(s.contains("2"));
            assert!(s.contains("3"));
        } else {
            panic!("Expected StringValue");
        }
    }

    #[test]
    fn test_convert_in_filter_non_array_error() {
        let filters = vec![TypedFilter {
            field: "status".to_string(),
            op: "in".to_string(),
            value: rv(serde_json::json!("not_an_array")),
            value_upper: None,
        }];

        let result = convert_typed_filters_to_clauses(&filters);
        assert!(result.is_err());
    }

    #[test]
    fn test_convert_neq_filter() {
        use crate::proto::proximadb_v1::ComparisonOp;

        let filters = vec![TypedFilter {
            field: "status".to_string(),
            op: "neq".to_string(),
            value: rv(serde_json::json!("deleted")),
            value_upper: None,
        }];

        let clauses = convert_typed_filters_to_clauses(&filters)
            .expect("Failed to convert typed filters to clauses");
        assert_eq!(clauses.len(), 1);
        assert_eq!(clauses[0].op, ComparisonOp::Ne as i32);
    }

    #[test]
    fn test_convert_multiple_filters() {
        use crate::proto::proximadb_v1::ComparisonOp;

        let filters = vec![
            TypedFilter {
                field: "category".to_string(),
                op: "eq".to_string(),
                value: rv(serde_json::json!("electronics")),
                value_upper: None,
            },
            TypedFilter {
                field: "price".to_string(),
                op: "lt".to_string(),
                value: rv(serde_json::json!(1000)),
                value_upper: None,
            },
            TypedFilter {
                field: "in_stock".to_string(),
                op: "eq".to_string(),
                value: rv(serde_json::json!(true)),
                value_upper: None,
            },
        ];

        let clauses = convert_typed_filters_to_clauses(&filters)
            .expect("Failed to convert typed filters to clauses");
        assert_eq!(clauses.len(), 3);
        assert_eq!(clauses[0].op, ComparisonOp::Eq as i32);
        assert_eq!(clauses[1].op, ComparisonOp::Lt as i32);
        assert_eq!(clauses[2].op, ComparisonOp::Eq as i32);
    }

    #[test]
    fn test_json_to_filter_clause_value_types() {
        use crate::proto::proximadb_v1::filter_clause::Value;

        // String
        let string_val = json_to_filter_clause_value(&serde_json::json!("test"));
        assert!(matches!(string_val, Some(Value::StringValue(_))));

        // Integer
        let int_val = json_to_filter_clause_value(&serde_json::json!(42));
        assert!(matches!(int_val, Some(Value::IntValue(42))));

        // Float
        let float_val = json_to_filter_clause_value(&serde_json::json!(3.5));
        assert!(matches!(float_val, Some(Value::DoubleValue(_))));

        // Boolean
        let bool_val = json_to_filter_clause_value(&serde_json::json!(true));
        assert!(matches!(bool_val, Some(Value::BoolValue(true))));

        // Null returns None
        let null_val = json_to_filter_clause_value(&serde_json::json!(null));
        assert!(null_val.is_none());

        // Array returns None (not directly supported)
        let array_val = json_to_filter_clause_value(&serde_json::json!([1, 2, 3]));
        assert!(array_val.is_none());

        // Object returns None (not directly supported)
        let object_val = json_to_filter_clause_value(&serde_json::json!({"key": "value"}));
        assert!(object_val.is_none());
    }

    // =========================================================================
    // Tests for json_to_sql_value
    // =========================================================================

    #[test]
    fn test_json_to_sql_value_null() {
        use crate::proto::proximadb_v1::sql_value::Value;
        let val = json_to_sql_value(&serde_json::json!(null));
        assert!(matches!(val.value, Some(Value::NullValue(0))));
    }

    #[test]
    fn test_json_to_sql_value_bool() {
        use crate::proto::proximadb_v1::sql_value::Value;
        let val = json_to_sql_value(&serde_json::json!(true));
        assert!(matches!(val.value, Some(Value::BoolValue(true))));

        let val = json_to_sql_value(&serde_json::json!(false));
        assert!(matches!(val.value, Some(Value::BoolValue(false))));
    }

    #[test]
    fn test_json_to_sql_value_integer() {
        use crate::proto::proximadb_v1::sql_value::Value;
        let val = json_to_sql_value(&serde_json::json!(42));
        assert!(matches!(val.value, Some(Value::Int64Value(42))));

        let val = json_to_sql_value(&serde_json::json!(-100));
        assert!(matches!(val.value, Some(Value::Int64Value(-100))));
    }

    #[test]
    fn test_json_to_sql_value_float() {
        use crate::proto::proximadb_v1::sql_value::Value;
        let val = json_to_sql_value(&serde_json::json!(3.5));
        match val.value {
            Some(Value::NumberValue(f)) => assert!((f - 3.5).abs() < f64::EPSILON),
            other => panic!("Expected NumberValue, got {:?}", other),
        }
    }

    #[test]
    fn test_json_to_sql_value_string() {
        use crate::proto::proximadb_v1::sql_value::Value;
        let val = json_to_sql_value(&serde_json::json!("hello world"));
        match val.value {
            Some(Value::StringValue(ref s)) => assert_eq!(s, "hello world"),
            other => panic!("Expected StringValue, got {:?}", other),
        }
    }

    #[test]
    fn test_json_to_sql_value_array() {
        use crate::proto::proximadb_v1::sql_value::Value;
        let val = json_to_sql_value(&serde_json::json!([1, "two", true]));
        match val.value {
            Some(Value::ArrayValue(ref arr)) => {
                assert_eq!(arr.values.len(), 3);
                assert!(matches!(arr.values[0].value, Some(Value::Int64Value(1))));
                assert!(matches!(arr.values[1].value, Some(Value::StringValue(_))));
                assert!(matches!(arr.values[2].value, Some(Value::BoolValue(true))));
            }
            other => panic!("Expected ArrayValue, got {:?}", other),
        }
    }

    #[test]
    fn test_json_to_sql_value_object() {
        use crate::proto::proximadb_v1::sql_value::Value;
        let val = json_to_sql_value(&serde_json::json!({"key": "value", "num": 42}));
        match val.value {
            Some(Value::ObjectValue(ref obj)) => {
                assert_eq!(obj.fields.len(), 2);
                assert!(obj.fields.contains_key("key"));
                assert!(obj.fields.contains_key("num"));
            }
            other => panic!("Expected ObjectValue, got {:?}", other),
        }
    }

    #[test]
    fn test_json_to_sql_value_nested_array() {
        use crate::proto::proximadb_v1::sql_value::Value;
        let val = json_to_sql_value(&serde_json::json!([[1, 2], [3, 4]]));
        match val.value {
            Some(Value::ArrayValue(ref arr)) => {
                assert_eq!(arr.values.len(), 2);
                // Each element should be an ArrayValue
                assert!(matches!(arr.values[0].value, Some(Value::ArrayValue(_))));
            }
            other => panic!("Expected nested ArrayValue, got {:?}", other),
        }
    }

    // =========================================================================
    // Tests for sql_value_to_json (the v2 records version)
    // =========================================================================

    #[test]
    fn test_sql_value_to_json_null() {
        use crate::proto::proximadb_v1::{SqlValue, sql_value::Value};
        let sv = SqlValue {
            value: Some(Value::NullValue(0)),
        };
        let result = sql_value_to_json(&sv).expect("Should convert null");
        assert!(result.is_null());
    }

    #[test]
    fn test_sql_value_to_json_bool() {
        use crate::proto::proximadb_v1::{SqlValue, sql_value::Value};
        let sv = SqlValue {
            value: Some(Value::BoolValue(true)),
        };
        let result = sql_value_to_json(&sv).expect("Should convert bool");
        assert_eq!(result, serde_json::json!(true));
    }

    #[test]
    fn test_sql_value_to_json_int64() {
        use crate::proto::proximadb_v1::{SqlValue, sql_value::Value};
        let sv = SqlValue {
            value: Some(Value::Int64Value(999)),
        };
        let result = sql_value_to_json(&sv).expect("Should convert int64");
        assert_eq!(result, serde_json::json!(999));
    }

    #[test]
    fn test_sql_value_to_json_number() {
        use crate::proto::proximadb_v1::{SqlValue, sql_value::Value};
        let sv = SqlValue {
            value: Some(Value::NumberValue(2.5)),
        };
        let result = sql_value_to_json(&sv).expect("Should convert number");
        assert!((result.as_f64().expect("Should be f64") - 2.5).abs() < 0.001);
    }

    #[test]
    fn test_sql_value_to_json_string() {
        use crate::proto::proximadb_v1::{SqlValue, sql_value::Value};
        let sv = SqlValue {
            value: Some(Value::StringValue("test_str".to_string())),
        };
        let result = sql_value_to_json(&sv).expect("Should convert string");
        assert_eq!(result, serde_json::json!("test_str"));
    }

    #[test]
    fn test_sql_value_to_json_bytes() {
        use crate::proto::proximadb_v1::{SqlValue, sql_value::Value};
        let sv = SqlValue {
            value: Some(Value::BytesValue(vec![0, 1, 255])),
        };
        let result = sql_value_to_json(&sv).expect("Should convert bytes");
        // Canonical rendering — bytes are base64 (was int-array).
        assert_eq!(result, serde_json::json!("AAH/"));
    }

    #[test]
    fn test_sql_value_to_json_array() {
        use crate::proto::proximadb_v1::{SqlArray, SqlValue, sql_value::Value};
        let sv = SqlValue {
            value: Some(Value::ArrayValue(SqlArray {
                values: vec![
                    SqlValue {
                        value: Some(Value::Int64Value(1)),
                    },
                    SqlValue {
                        value: Some(Value::Int64Value(2)),
                    },
                ],
            })),
        };
        let result = sql_value_to_json(&sv).expect("Should convert array");
        assert_eq!(result, serde_json::json!([1, 2]));
    }

    #[test]
    fn test_sql_value_to_json_object() {
        use crate::proto::proximadb_v1::{SqlObject, SqlValue, sql_value::Value};
        let mut fields = HashMap::new();
        fields.insert(
            "name".to_string(),
            SqlValue {
                value: Some(Value::StringValue("alice".to_string())),
            },
        );
        let sv = SqlValue {
            value: Some(Value::ObjectValue(SqlObject { fields })),
        };
        let result = sql_value_to_json(&sv).expect("Should convert object");
        assert_eq!(result["name"], serde_json::json!("alice"));
    }

    #[test]
    fn test_sql_value_to_json_none_value() {
        use crate::proto::proximadb_v1::SqlValue;
        let sv = SqlValue { value: None };
        let result = sql_value_to_json(&sv).expect("Should convert None to null");
        assert!(result.is_null());
    }

    #[test]
    fn test_sql_value_to_json_nan_returns_error() {
        use crate::proto::proximadb_v1::{SqlValue, sql_value::Value};
        let sv = SqlValue {
            value: Some(Value::NumberValue(f64::NAN)),
        };
        let result = sql_value_to_json(&sv);
        // Canonical rendering — non-finite floats render as null (the old
        // 500-error path is gone; pinned in records' contract test).
        assert_eq!(
            result.expect("canonical rendering is infallible"),
            serde_json::json!(null)
        );
    }

    // =========================================================================
    // Tests for json_to_sql_value -> sql_value_to_json roundtrip
    // =========================================================================

    #[test]
    fn test_json_sql_value_roundtrip_string() {
        let original = serde_json::json!("roundtrip_test");
        let sql_val = json_to_sql_value(&original);
        let result = sql_value_to_json(&sql_val).expect("Roundtrip should succeed");
        assert_eq!(original, result);
    }

    #[test]
    fn test_json_sql_value_roundtrip_integer() {
        let original = serde_json::json!(42);
        let sql_val = json_to_sql_value(&original);
        let result = sql_value_to_json(&sql_val).expect("Roundtrip should succeed");
        assert_eq!(original, result);
    }

    #[test]
    fn test_json_sql_value_roundtrip_bool() {
        let original = serde_json::json!(true);
        let sql_val = json_to_sql_value(&original);
        let result = sql_value_to_json(&sql_val).expect("Roundtrip should succeed");
        assert_eq!(original, result);
    }

    #[test]
    fn test_json_sql_value_roundtrip_null() {
        let original = serde_json::json!(null);
        let sql_val = json_to_sql_value(&original);
        let result = sql_value_to_json(&sql_val).expect("Roundtrip should succeed");
        assert_eq!(original, result);
    }

    #[test]
    fn test_json_sql_value_roundtrip_nested_object() {
        let original = serde_json::json!({"a": 1, "b": "two", "c": [true, null]});
        let sql_val = json_to_sql_value(&original);
        let result = sql_value_to_json(&sql_val).expect("Roundtrip should succeed");
        assert_eq!(original, result);
    }

    // =========================================================================
    // Tests for unsupported filter operators
    // =========================================================================

    #[test]
    fn test_convert_unsupported_filter_operator() {
        let filters = vec![TypedFilter {
            field: "x".to_string(),
            op: "regex".to_string(),
            value: rv(serde_json::json!(".*")),
            value_upper: None,
        }];

        let result = convert_typed_filters_to_clauses(&filters);
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.expect_err("Should be error"));
        assert!(err_msg.contains("Unsupported"));
    }

    #[test]
    fn test_convert_starts_with_non_string_error() {
        let filters = vec![TypedFilter {
            field: "name".to_string(),
            op: "starts_with".to_string(),
            value: rv(serde_json::json!(123)),
            value_upper: None,
        }];

        let result = convert_typed_filters_to_clauses(&filters);
        assert!(result.is_err());
    }

    #[test]
    fn test_convert_ends_with_non_string_error() {
        let filters = vec![TypedFilter {
            field: "name".to_string(),
            op: "ends_with".to_string(),
            value: rv(serde_json::json!(true)),
            value_upper: None,
        }];

        let result = convert_typed_filters_to_clauses(&filters);
        assert!(result.is_err());
    }

    #[test]
    fn test_convert_empty_filters_returns_empty() {
        let filters: Vec<TypedFilter> = vec![];
        let clauses =
            convert_typed_filters_to_clauses(&filters).expect("Empty filters should succeed");
        assert!(clauses.is_empty());
    }

    // ============================================================
    // json_to_sql_value / sql_value_to_json conversion tests
    // ============================================================

    #[test]
    fn test_json_to_sql_null() {
        let sv = json_to_sql_value(&serde_json::Value::Null);
        assert!(matches!(
            sv.value,
            Some(crate::proto::proximadb_v1::sql_value::Value::NullValue(_))
        ));
    }

    #[test]
    fn test_json_to_sql_bool() {
        let sv = json_to_sql_value(&serde_json::json!(true));
        assert!(matches!(
            sv.value,
            Some(crate::proto::proximadb_v1::sql_value::Value::BoolValue(
                true
            ))
        ));
    }

    #[test]
    fn test_json_to_sql_integer() {
        let sv = json_to_sql_value(&serde_json::json!(42));
        assert!(matches!(
            sv.value,
            Some(crate::proto::proximadb_v1::sql_value::Value::Int64Value(42))
        ));
    }

    #[test]
    fn test_json_to_sql_float() {
        let sv = json_to_sql_value(&serde_json::json!(3.5));
        if let Some(crate::proto::proximadb_v1::sql_value::Value::NumberValue(f)) = sv.value {
            assert!((f - 3.5).abs() < 1e-10);
        } else {
            panic!("Expected NumberValue");
        }
    }

    #[test]
    fn test_json_to_sql_string() {
        let sv = json_to_sql_value(&serde_json::json!("hello"));
        assert!(matches!(
            sv.value,
            Some(crate::proto::proximadb_v1::sql_value::Value::StringValue(_))
        ));
    }

    #[test]
    fn test_json_to_sql_array() {
        let sv = json_to_sql_value(&serde_json::json!([1, 2, 3]));
        if let Some(crate::proto::proximadb_v1::sql_value::Value::ArrayValue(arr)) = sv.value {
            assert_eq!(arr.values.len(), 3);
        } else {
            panic!("Expected ArrayValue");
        }
    }

    #[test]
    fn test_json_to_sql_object() {
        let sv = json_to_sql_value(&serde_json::json!({"key": "val"}));
        if let Some(crate::proto::proximadb_v1::sql_value::Value::ObjectValue(obj)) = sv.value {
            assert!(obj.fields.contains_key("key"));
        } else {
            panic!("Expected ObjectValue");
        }
    }

    #[test]
    fn test_sql_to_json_roundtrip_null() {
        let sv = json_to_sql_value(&serde_json::Value::Null);
        let json = sql_value_to_json(&sv).unwrap();
        assert!(json.is_null());
    }

    #[test]
    fn test_sql_to_json_roundtrip_bool() {
        let original = serde_json::json!(false);
        let sv = json_to_sql_value(&original);
        let json = sql_value_to_json(&sv).unwrap();
        assert_eq!(json, original);
    }

    #[test]
    fn test_sql_to_json_roundtrip_integer() {
        let original = serde_json::json!(12345);
        let sv = json_to_sql_value(&original);
        let json = sql_value_to_json(&sv).unwrap();
        assert_eq!(json, original);
    }

    #[test]
    fn test_sql_to_json_roundtrip_string() {
        let original = serde_json::json!("test_value");
        let sv = json_to_sql_value(&original);
        let json = sql_value_to_json(&sv).unwrap();
        assert_eq!(json, original);
    }

    #[test]
    fn test_sql_to_json_roundtrip_array() {
        let original = serde_json::json!(["a", "b", "c"]);
        let sv = json_to_sql_value(&original);
        let json = sql_value_to_json(&sv).unwrap();
        assert_eq!(json, original);
    }

    #[test]
    fn test_sql_to_json_roundtrip_nested_object() {
        let original = serde_json::json!({"nested": {"deep": true}});
        let sv = json_to_sql_value(&original);
        let json = sql_value_to_json(&sv).unwrap();
        assert_eq!(json, original);
    }

    #[test]
    fn test_sql_to_json_none_value() {
        use crate::proto::proximadb_v1::SqlValue;
        let sv = SqlValue { value: None };
        let json = sql_value_to_json(&sv).unwrap();
        assert!(json.is_null());
    }

    // ============================================================
    // json_to_filter_clause_value tests
    // ============================================================

    #[test]
    fn test_filter_clause_string() {
        let val = json_to_filter_clause_value(&serde_json::json!("active"));
        assert!(val.is_some());
    }

    #[test]
    fn test_filter_clause_integer() {
        let val = json_to_filter_clause_value(&serde_json::json!(42));
        assert!(val.is_some());
    }

    #[test]
    fn test_filter_clause_float() {
        let val = json_to_filter_clause_value(&serde_json::json!(3.5));
        assert!(val.is_some());
    }

    #[test]
    fn test_filter_clause_bool() {
        let val = json_to_filter_clause_value(&serde_json::json!(true));
        assert!(val.is_some());
    }

    #[test]
    fn test_filter_clause_array_unsupported() {
        let val = json_to_filter_clause_value(&serde_json::json!([1, 2]));
        assert!(val.is_none());
    }

    #[test]
    fn test_filter_clause_null_unsupported() {
        let val = json_to_filter_clause_value(&serde_json::Value::Null);
        assert!(val.is_none());
    }

    // ============================================================
    // V2 Record Request Parsing
    // ============================================================

    #[test]
    fn test_v2_record_request_parsing() {
        let json = r#"{
            "records": [
                {
                    "id": "rec_001",
                    "vector": [0.1, 0.2, 0.3, 0.4],
                    "props": {
                        "name": "Widget",
                        "price": 29.99,
                        "in_stock": true,
                        "quantity": 42
                    },
                    "text_fields": [
                        {
                            "name": "description",
                            "content": "A high-quality widget for all purposes",
                            "storage_hint": "adaptive"
                        }
                    ]
                },
                {
                    "vector": [0.5, 0.6, 0.7, 0.8],
                    "props": {
                        "name": "Gadget"
                    }
                }
            ],
            "validate_schema": false
        }"#;

        let request: InsertRecordsRequest =
            serde_json::from_str(json).expect("Failed to parse V2 record request");

        assert_eq!(request.records.len(), 2);
        assert_eq!(request.validate_schema, Some(false));

        // First record has all fields
        let rec0 = &request.records[0];
        assert_eq!(rec0.id, Some("rec_001".to_string()));
        assert_eq!(rec0.vector.len(), 4);
        assert!(rec0.props.is_some());
        let props = rec0.props.as_ref().expect("props should be Some");
        assert!(props.contains_key("name"));
        assert!(props.contains_key("price"));
        assert!(props.contains_key("in_stock"));
        assert!(props.contains_key("quantity"));

        let text_fields = rec0
            .text_fields
            .as_ref()
            .expect("text_fields should be Some");
        assert_eq!(text_fields.len(), 1);
        assert_eq!(text_fields[0].name, "description");
        assert_eq!(text_fields[0].storage_hint, Some("adaptive".to_string()));

        // Second record has auto-generated ID (None)
        let rec1 = &request.records[1];
        assert!(rec1.id.is_none());
        assert_eq!(rec1.vector.len(), 4);
        assert!(rec1.text_fields.is_none());
        assert!(rec1.props.is_some());
    }

    // ============================================================
    // V2 Record Response Serialization
    // ============================================================

    #[test]
    fn test_v2_record_response_serialization() {
        let response = InsertRecordsResponse {
            inserted_count: 3,
            failed_count: 1,
            errors: vec![InsertError {
                index: 2,
                id: Some("bad_rec".to_string()),
                error: "Vector cannot be empty".to_string(),
            }],
            inserted_ids: vec!["id_1".to_string(), "id_2".to_string(), "id_3".to_string()],
        };

        let json_str =
            serde_json::to_string(&response).expect("Failed to serialize InsertRecordsResponse");
        let parsed: serde_json::Value =
            serde_json::from_str(&json_str).expect("Failed to parse serialized response");

        assert_eq!(parsed["inserted_count"], 3);
        assert_eq!(parsed["failed_count"], 1);
        assert_eq!(
            parsed["inserted_ids"]
                .as_array()
                .expect("Expected array")
                .len(),
            3
        );

        let errors = parsed["errors"].as_array().expect("Expected errors array");
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0]["index"], 2);
        assert_eq!(errors[0]["id"], "bad_rec");
        assert_eq!(errors[0]["error"], "Vector cannot be empty");
    }

    // ============================================================
    // V2 Batch Request Parsing
    // ============================================================

    #[test]
    fn test_v2_batch_request_parsing() {
        // A batch with multiple records of varying completeness
        let json = r#"{
            "records": [
                {"vector": [0.1, 0.2], "props": {"a": 1}},
                {"vector": [0.3, 0.4], "props": {"b": 2}},
                {"vector": [0.5, 0.6], "props": {"c": 3}},
                {"vector": [0.7, 0.8]}
            ]
        }"#;

        let request: InsertRecordsRequest =
            serde_json::from_str(json).expect("Failed to parse batch request");

        assert_eq!(request.records.len(), 4);
        // validate_schema defaults to None when omitted
        assert!(request.validate_schema.is_none());

        // All records should have valid vectors
        for rec in &request.records {
            assert_eq!(rec.vector.len(), 2);
        }

        // Only first three have props
        assert!(request.records[0].props.is_some());
        assert!(request.records[1].props.is_some());
        assert!(request.records[2].props.is_some());
        assert!(request.records[3].props.is_none());
    }

    // ============================================================
    // V2 Search Request Parsing
    // ============================================================

    #[test]
    fn test_v2_search_request_parsing() {
        let json = r#"{
            "vector": [0.1, 0.2, 0.3, 0.4, 0.5],
            "top_k": 25,
            "filters": [
                {"field": "category", "op": "eq", "value": "electronics"},
                {"field": "price", "op": "gte", "value": 10.0},
                {"field": "price", "op": "lte", "value": 1000.0},
                {"field": "brand", "op": "in", "value": ["Apple", "Samsung"]},
                {"field": "description", "op": "contains", "value": "wireless"}
            ],
            "include_text": true,
            "include_vector": false
        }"#;

        let request: TypedSearchRequest =
            serde_json::from_str(json).expect("Failed to parse V2 search request");

        assert_eq!(request.vector.len(), 5);
        assert_eq!(request.top_k, 25);
        assert_eq!(request.include_text, Some(true));
        assert_eq!(request.include_vector, Some(false));

        let filters = request.filters.as_ref().expect("filters should be Some");
        assert_eq!(filters.len(), 5);

        // Verify filter field names and operators
        assert_eq!(filters[0].field, "category");
        assert_eq!(filters[0].op, "eq");
        assert_eq!(filters[1].field, "price");
        assert_eq!(filters[1].op, "gte");
        assert_eq!(filters[2].field, "price");
        assert_eq!(filters[2].op, "lte");
        assert_eq!(filters[3].field, "brand");
        assert_eq!(filters[3].op, "in");
        assert!(rest_value_payload(&filters[3].value).is_array());
        assert_eq!(filters[4].field, "description");
        assert_eq!(filters[4].op, "contains");

        // Verify no filters have value_upper unless between
        for filter in filters {
            assert!(filter.value_upper.is_none());
        }
    }

    // ============================================================
    // V2 Schema Request Parsing
    // ============================================================

    #[test]
    fn test_v2_schema_request_parsing() {
        // Test that UpdateSchemaRequest deserializes correctly
        // UpdateSchemaRequest uses #[serde(flatten)] on SchemaDefinition
        let json = r#"{
            "columns": [
                {
                    "name": "category",
                    "data_type": "text",
                    "nullable": true,
                    "indexed": true,
                    "filterable": true
                },
                {
                    "name": "price",
                    "data_type": "float",
                    "nullable": false,
                    "filterable": true
                },
                {
                    "name": "embedding",
                    "data_type": "vector",
                    "vector_dimension": 768
                }
            ],
            "enforcement": "hybrid",
            "allow_additional_fields": true,
            "force": false
        }"#;

        let request: super::super::schema::UpdateSchemaRequest =
            serde_json::from_str(json).expect("Failed to parse schema update request");

        assert_eq!(request.schema.columns.len(), 3);
        assert_eq!(request.schema.enforcement, Some("hybrid".to_string()));
        assert_eq!(request.schema.allow_additional_fields, Some(true));
        assert_eq!(request.force, Some(false));

        // Verify column details
        let col0 = &request.schema.columns[0];
        assert_eq!(col0.name, "category");
        assert_eq!(col0.data_type, "text");
        assert_eq!(col0.nullable, Some(true));
        assert_eq!(col0.indexed, Some(true));

        let col1 = &request.schema.columns[1];
        assert_eq!(col1.name, "price");
        assert_eq!(col1.data_type, "float");
        assert_eq!(col1.nullable, Some(false));

        let col2 = &request.schema.columns[2];
        assert_eq!(col2.name, "embedding");
        assert_eq!(col2.data_type, "vector");
        assert_eq!(col2.vector_dimension, Some(768));
    }

    // ============================================================
    // V2 Error Response Format
    // ============================================================

    #[test]
    fn test_v2_error_response_format() {
        // Verify InsertError and InsertRecordsResponse serialization
        // when all records fail
        let response = InsertRecordsResponse {
            inserted_count: 0,
            failed_count: 2,
            errors: vec![
                InsertError {
                    index: 0,
                    id: Some("rec_a".to_string()),
                    error: "Vector cannot be empty".to_string(),
                },
                InsertError {
                    index: 1,
                    id: None,
                    error: "Dimension mismatch: expected 768, got 128".to_string(),
                },
            ],
            inserted_ids: vec![],
        };

        let json_str =
            serde_json::to_string(&response).expect("Failed to serialize error response");
        let parsed: serde_json::Value =
            serde_json::from_str(&json_str).expect("Failed to parse serialized error response");

        assert_eq!(parsed["inserted_count"], 0);
        assert_eq!(parsed["failed_count"], 2);
        assert!(
            parsed["inserted_ids"]
                .as_array()
                .expect("Expected array")
                .is_empty()
        );

        let errors = parsed["errors"].as_array().expect("Expected errors array");
        assert_eq!(errors.len(), 2);

        // First error has an id
        assert_eq!(errors[0]["index"], 0);
        assert_eq!(errors[0]["id"], "rec_a");
        assert!(
            errors[0]["error"]
                .as_str()
                .expect("Expected string")
                .contains("empty")
        );

        // Second error has null id
        assert_eq!(errors[1]["index"], 1);
        assert!(errors[1]["id"].is_null());
        assert!(
            errors[1]["error"]
                .as_str()
                .expect("Expected string")
                .contains("Dimension mismatch")
        );

        // Also verify the search response serializes correctly
        let search_resp = TypedSearchResponse {
            results: vec![TypedSearchResult {
                id: "doc_1".to_string(),
                score: 0.95,
                vector: None,
                props: {
                    let mut m = HashMap::new();
                    m.insert(
                        "category".to_string(),
                        RestProximaValue::Typed {
                            type_name: "string".to_string(),
                            value: serde_json::json!("test"),
                        },
                    );
                    m
                },
                text_fields: None,
            }],
            total_matches: Some(100),
            latency_ms: 5,
            request_id: "req-123".to_string(),
            predicate_shortfall: None,
            search_plan_trace: None,
            explain: None,
        };

        let search_json =
            serde_json::to_string(&search_resp).expect("Failed to serialize search response");
        let search_parsed: serde_json::Value =
            serde_json::from_str(&search_json).expect("Failed to parse serialized search response");

        assert_eq!(
            search_parsed["results"]
                .as_array()
                .expect("Expected array")
                .len(),
            1
        );
        assert_eq!(search_parsed["total_matches"], 100);
        assert_eq!(search_parsed["latency_ms"], 5);
        assert_eq!(search_parsed["request_id"], "req-123");
    }
}
