// Custom serde implementations for protobuf types
// Handles oneof types and SDK compatibility for complex types like QuantizationConfig
use crate::proto::proximadb_v1::{PropertyValue, property_value::Value as PropertyValueVariant};
use crate::proto::proximadb_v1::{QuantizationConfig, QuantizationLevel};
use crate::proto::proximadb_v1::{SqlValue, sql_value::Value as SqlValueVariant};
use crate::utils::encoding::{base64_decode, base64_encode};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A genuine JSONB payload is valid canonical MessagePack by construction —
/// used to gate the `{"jsonb_value": <base64>}` tagged form against generic
/// single-key objects whose string merely happens to be base64-decodable.
fn valid_jsonb_bytes(bytes: &[u8]) -> bool {
    rmp_serde::from_slice::<serde_json::Value>(bytes).is_ok()
}

// ============================================================================
// QuantizationConfig - SDK Compatibility Shim
// ============================================================================
//
// The Python SDK sends flat quantization config:
// {
//   "enabled": true,
//   "type": "product",  // or "binary", "scalar"
//   "bits": 16,
//   "num_subvectors": 16,
//   "bits_per_subvector": 8,
//   "bits_per_vector": 128
// }
//
// But the proto expects nested custom_levels:
// {
//   "enabled": true,
//   "strategy": 2,
//   "custom_levels": [
//     {"level_id": "...", "type": 1, "bits": 16, "num_subvectors": 16}
//   ]
// }
//
// This custom deserializer accepts BOTH formats for backward compatibility.

impl<'de> Deserialize<'de> for QuantizationConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use crate::proto::proximadb_v1::quantization_config::Strategy;
        use crate::proto::proximadb_v1::quantization_level::QuantizationType;

        // Helper struct that accepts both proto and SDK fields
        #[derive(serde::Deserialize)]
        struct QuantizationConfigHelper {
            // Standard proto fields
            enabled: Option<bool>,
            strategy: Option<i32>,
            #[serde(default)]
            custom_levels: Vec<QuantizationLevel>,
            enable_progressive_search: Option<bool>,
            binary_filter_selectivity: Option<f32>,
            int8_ranking_selectivity: Option<f32>,
            pq_ranking_selectivity: Option<f32>,
            training_sample_size: Option<u32>,
            quality_threshold: Option<f32>,
            enable_adaptive_training: Option<bool>,
            optimize_for_storage: Option<bool>,
            optimize_for_memory: Option<bool>,
            enable_simd_acceleration: Option<bool>,
            enable_binary: Option<bool>,
            enable_int8: Option<bool>,
            enable_pq: Option<bool>,
            pq_segments: Option<u32>,
            pq_bits: Option<u32>,
            pq_codebooks: Option<u32>,
            binary_threshold: Option<f32>,
            int8_threshold: Option<f32>,
            pq_threshold: Option<f32>,
            // Phase N: TurboQuant opt-in. Defaults to `None` ⇒ false.
            enable_turboquant: Option<bool>,

            // SDK compatibility fields (flat structure)
            #[serde(alias = "type")]
            quantization_type: Option<String>,
            bits: Option<u32>,
            num_subvectors: Option<u32>,
            bits_per_subvector: Option<u32>,
            bits_per_vector: Option<u32>,
        }

        let helper = QuantizationConfigHelper::deserialize(deserializer)?;

        // Determine custom_levels: use proto format if provided, else construct from SDK fields
        let custom_levels = if !helper.custom_levels.is_empty() {
            // Proto format - use as-is
            helper.custom_levels
        } else if helper.quantization_type.is_some()
            || helper.bits.is_some()
            || helper.num_subvectors.is_some()
        {
            // SDK format - construct custom_levels from flat fields
            let quant_type_str = helper.quantization_type.as_deref().unwrap_or("none");
            let quant_type = match quant_type_str.to_lowercase().as_str() {
                "binary" => QuantizationType::Binary as i32,
                "int8" | "scalar" => QuantizationType::Scalar as i32,
                "product" | "pq" => QuantizationType::Product as i32,
                "uniform" => QuantizationType::Uniform as i32,
                "none" => QuantizationType::None as i32,
                _ => QuantizationType::None as i32,
            };

            let bits = helper
                .bits
                .or(helper.bits_per_subvector)
                .or(helper.bits_per_vector)
                .unwrap_or(8);

            let num_subvectors = helper.num_subvectors.unwrap_or(
                if quant_type == QuantizationType::Product as i32 {
                    16
                } else {
                    1
                },
            );

            vec![QuantizationLevel {
                level_id: format!("sdk_level_{}", quant_type_str),
                r#type: quant_type,
                bits,
                num_subvectors,
                adaptive_subvectors: false,
                scale: 1.0,
                offset: 0.0,
                clamp_values: true,
                threshold: 0.0,
                sign_based: false,
                enable_in_storage: true,
                enable_in_index: true,
                search_priority: 0,
                min_recall: 0.95,
                enable_validation: true,
            }]
        } else {
            // No quantization levels specified
            vec![]
        };

        // Determine strategy based on custom_levels or explicit strategy
        let strategy = helper.strategy.or({
            if !custom_levels.is_empty() {
                Some(Strategy::CustomLevels as i32)
            } else {
                Some(Strategy::SmartDefaults as i32)
            }
        });

        Ok(QuantizationConfig {
            enabled: helper.enabled,
            strategy,
            custom_levels,
            enable_progressive_search: helper.enable_progressive_search,
            binary_filter_selectivity: helper.binary_filter_selectivity,
            int8_ranking_selectivity: helper.int8_ranking_selectivity,
            pq_ranking_selectivity: helper.pq_ranking_selectivity,
            training_sample_size: helper.training_sample_size,
            quality_threshold: helper.quality_threshold,
            enable_adaptive_training: helper.enable_adaptive_training,
            optimize_for_storage: helper.optimize_for_storage,
            optimize_for_memory: helper.optimize_for_memory,
            enable_simd_acceleration: helper.enable_simd_acceleration,
            enable_binary: helper.enable_binary,
            enable_int8: helper.enable_int8,
            enable_pq: helper.enable_pq,
            pq_segments: helper.pq_segments,
            pq_bits: helper.pq_bits,
            pq_codebooks: helper.pq_codebooks,
            binary_threshold: helper.binary_threshold,
            int8_threshold: helper.int8_threshold,
            pq_threshold: helper.pq_threshold,
            // Phase N: TurboQuant opt-in helper field — same defaulting
            // behavior as `enable_binary` / `enable_int8` / `enable_pq`.
            enable_turboquant: helper.enable_turboquant,
        })
    }
}
impl Serialize for SqlValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(1))?;

        match &self.value {
            Some(SqlValueVariant::StringValue(v)) => {
                map.serialize_entry("string_value", v)?;
            }
            Some(SqlValueVariant::NumberValue(v)) => {
                map.serialize_entry("number_value", v)?;
            }
            Some(SqlValueVariant::BoolValue(v)) => {
                map.serialize_entry("bool_value", v)?;
            }
            Some(SqlValueVariant::Int64Value(v)) => {
                map.serialize_entry("int64_value", v)?;
            }
            Some(SqlValueVariant::BytesValue(v)) => {
                map.serialize_entry("bytes_value", &base64_encode(v))?;
            }
            Some(SqlValueVariant::NullValue(_)) => {
                map.serialize_entry("null_value", &serde_json::Value::Null)?;
            }
            Some(SqlValueVariant::ArrayValue(v)) => {
                map.serialize_entry("array_value", v)?;
            }
            Some(SqlValueVariant::ObjectValue(v)) => {
                map.serialize_entry("object_value", v)?;
            }
            Some(SqlValueVariant::JsonbValue(v)) => {
                map.serialize_entry("jsonb_value", &base64_encode(v))?;
            }
            None => {
                map.serialize_entry("null_value", &serde_json::Value::Null)?;
            }
        }

        map.end()
    }
}
impl<'de> Deserialize<'de> for SqlValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // First, deserialize as a raw JSON value to handle both formats
        let json_value = serde_json::Value::deserialize(deserializer)?;

        // Convert JSON value to SqlValue
        // Supports BOTH:
        // 1. Proto format: {"string_value": "test"}, {"number_value": 42}
        // 2. Simple format: "test", 42, true, null (MVP-friendly)
        let value = match &json_value {
            // Non-object kinds are exactly the shared simple-format
            // conversion (see json_to_sql_value below).
            serde_json::Value::Object(obj) => {
                // Check for proto-style wrapped values first
                if let Some(v) = obj.get("string_value")
                    && let Some(s) = v.as_str()
                {
                    return Ok(SqlValue {
                        value: Some(SqlValueVariant::StringValue(s.to_string())),
                    });
                }
                if let Some(v) = obj.get("number_value")
                    && let Some(f) = v.as_f64()
                {
                    return Ok(SqlValue {
                        value: Some(SqlValueVariant::NumberValue(f)),
                    });
                }
                if let Some(v) = obj.get("bool_value")
                    && let Some(b) = v.as_bool()
                {
                    return Ok(SqlValue {
                        value: Some(SqlValueVariant::BoolValue(b)),
                    });
                }
                if let Some(v) = obj.get("int64_value")
                    && let Some(i) = v.as_i64()
                {
                    return Ok(SqlValue {
                        value: Some(SqlValueVariant::Int64Value(i)),
                    });
                }
                if let Some(v) = obj.get("bytes_value")
                    && let Some(s) = v.as_str()
                {
                    let bytes = base64_decode(s).map_err(serde::de::Error::custom)?;
                    return Ok(SqlValue {
                        value: Some(SqlValueVariant::BytesValue(bytes)),
                    });
                }
                if obj.len() == 1
                    && let Some(v) = obj.get("jsonb_value")
                    && let Some(s) = v.as_str()
                    && let Ok(bytes) = base64_decode(s)
                    // Round-trip gate: a genuine JSONB payload is valid
                    // MessagePack by construction. A generic object whose
                    // literal key happens to be "jsonb_value" with an
                    // arbitrary base64-ish string usually is not, and falls
                    // through to ObjectValue instead of being reshaped.
                    && valid_jsonb_bytes(&bytes)
                {
                    // The tagged form commits only for a SINGLE-KEY object
                    // whose value decodes to valid JSONB bytes: sibling keys
                    // must never be silently dropped, and a generic object
                    // that merely contains a "jsonb_value" key falls through
                    // to the generic-object handling below instead of 422-ing
                    // or being reshaped. (Residual ambiguity mirrors the
                    // pre-existing bytes_value tagged form.)
                    return Ok(SqlValue {
                        value: Some(SqlValueVariant::JsonbValue(bytes)),
                    });
                }
                if obj.contains_key("null_value") {
                    return Ok(SqlValue {
                        value: Some(SqlValueVariant::NullValue(0)),
                    });
                }
                if let Some(v) = obj.get("array_value") {
                    let arr: crate::proto::proximadb_v1::SqlArray =
                        serde_json::from_value(v.clone()).map_err(serde::de::Error::custom)?;
                    return Ok(SqlValue {
                        value: Some(SqlValueVariant::ArrayValue(arr)),
                    });
                }
                if let Some(v) = obj.get("object_value") {
                    let object: crate::proto::proximadb_v1::SqlObject =
                        serde_json::from_value(v.clone()).map_err(serde::de::Error::custom)?;
                    return Ok(SqlValue {
                        value: Some(SqlValueVariant::ObjectValue(object)),
                    });
                }

                // Treat as a generic object (MVP-friendly nested object):
                // the shared conversion's Object branch does exactly this.
                return Ok(json_to_sql_value(&json_value));
            }
            // Every non-object kind is the shared simple-format conversion.
            _ => json_to_sql_value(&json_value).value,
        };

        Ok(SqlValue { value })
    }
}

/// Simple-format conversion from `serde_json::Value` to [`SqlValue`].
///
/// Canonical copy (2026-11-07) of a conversion that previously existed in
/// five drifting variants across the workspace (api document ingest,
/// embedded, REST v2 tests, arrow conversion, and this crate's
/// `Deserialize` impl). Objects become `ObjectValue` and arrays
/// `ArrayValue` — proto-style tagged objects (`{"string_value": ...}`) are
/// NOT unwrapped here; only [`Deserialize for SqlValue`] does that.
///
/// Number handling: integers that fit `i64` become exact `Int64Value`;
/// everything else approximates as f64 `NumberValue` (for u64 >
/// `i64::MAX` the alternative shipped previously was a silent wraparound).
/// The `StringValue` fallback is unreachable for numbers serde_json can
/// parse without `arbitrary_precision`; it exists so the fn is total and
/// lossless if that feature is ever enabled (pinned by the test below).
pub fn json_to_sql_value(json: &serde_json::Value) -> SqlValue {
    let value = match json {
        serde_json::Value::String(s) => SqlValueVariant::StringValue(s.clone()),
        serde_json::Value::Bool(b) => SqlValueVariant::BoolValue(*b),
        serde_json::Value::Null => SqlValueVariant::NullValue(0),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                SqlValueVariant::Int64Value(i)
            } else {
                n.as_f64()
                    .map(SqlValueVariant::NumberValue)
                    .unwrap_or(SqlValueVariant::StringValue(n.to_string()))
            }
        }
        serde_json::Value::Array(arr) => {
            SqlValueVariant::ArrayValue(crate::proto::proximadb_v1::SqlArray {
                values: arr.iter().map(json_to_sql_value).collect(),
            })
        }
        serde_json::Value::Object(obj) => {
            SqlValueVariant::ObjectValue(crate::proto::proximadb_v1::SqlObject {
                fields: obj
                    .iter()
                    .map(|(k, v)| (k.clone(), json_to_sql_value(v)))
                    .collect(),
            })
        }
    };
    SqlValue { value: Some(value) }
}

impl Serialize for PropertyValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        // Use serialize_struct instead of serialize_map for bincode compatibility
        let mut state = serializer.serialize_struct("PropertyValue", 9)?;

        // Serialize all fields as Options (matching PropertyValueHelper in deserialize)
        match &self.value {
            Some(PropertyValueVariant::StringValue(v)) => {
                state.serialize_field("string_value", &Some(v))?;
                state.serialize_field("int_value", &None::<i64>)?;
                state.serialize_field("double_value", &None::<f64>)?;
                state.serialize_field("bool_value", &None::<bool>)?;
                state.serialize_field("bytes_value", &None::<String>)?;
                state.serialize_field(
                    "array_value",
                    &None::<crate::proto::proximadb_v1::PropertyArray>,
                )?;
                state.serialize_field(
                    "object_value",
                    &None::<crate::proto::proximadb_v1::PropertyObject>,
                )?;
                state.serialize_field("vector_value", &None::<Vec<f32>>)?;
                state.serialize_field("null_value", &None::<serde_json::Value>)?;
            }
            Some(PropertyValueVariant::IntValue(v)) => {
                state.serialize_field("string_value", &None::<String>)?;
                state.serialize_field("int_value", &Some(v))?;
                state.serialize_field("double_value", &None::<f64>)?;
                state.serialize_field("bool_value", &None::<bool>)?;
                state.serialize_field("bytes_value", &None::<String>)?;
                state.serialize_field(
                    "array_value",
                    &None::<crate::proto::proximadb_v1::PropertyArray>,
                )?;
                state.serialize_field(
                    "object_value",
                    &None::<crate::proto::proximadb_v1::PropertyObject>,
                )?;
                state.serialize_field("vector_value", &None::<Vec<f32>>)?;
                state.serialize_field("null_value", &None::<serde_json::Value>)?;
            }
            Some(PropertyValueVariant::DoubleValue(v)) => {
                state.serialize_field("string_value", &None::<String>)?;
                state.serialize_field("int_value", &None::<i64>)?;
                state.serialize_field("double_value", &Some(v))?;
                state.serialize_field("bool_value", &None::<bool>)?;
                state.serialize_field("bytes_value", &None::<String>)?;
                state.serialize_field(
                    "array_value",
                    &None::<crate::proto::proximadb_v1::PropertyArray>,
                )?;
                state.serialize_field(
                    "object_value",
                    &None::<crate::proto::proximadb_v1::PropertyObject>,
                )?;
                state.serialize_field("vector_value", &None::<Vec<f32>>)?;
                state.serialize_field("null_value", &None::<serde_json::Value>)?;
            }
            Some(PropertyValueVariant::BoolValue(v)) => {
                state.serialize_field("string_value", &None::<String>)?;
                state.serialize_field("int_value", &None::<i64>)?;
                state.serialize_field("double_value", &None::<f64>)?;
                state.serialize_field("bool_value", &Some(v))?;
                state.serialize_field("bytes_value", &None::<String>)?;
                state.serialize_field(
                    "array_value",
                    &None::<crate::proto::proximadb_v1::PropertyArray>,
                )?;
                state.serialize_field(
                    "object_value",
                    &None::<crate::proto::proximadb_v1::PropertyObject>,
                )?;
                state.serialize_field("vector_value", &None::<Vec<f32>>)?;
                state.serialize_field("null_value", &None::<serde_json::Value>)?;
            }
            Some(PropertyValueVariant::BytesValue(v)) => {
                state.serialize_field("string_value", &None::<String>)?;
                state.serialize_field("int_value", &None::<i64>)?;
                state.serialize_field("double_value", &None::<f64>)?;
                state.serialize_field("bool_value", &None::<bool>)?;
                state.serialize_field("bytes_value", &Some(base64_encode(v)))?;
                state.serialize_field(
                    "array_value",
                    &None::<crate::proto::proximadb_v1::PropertyArray>,
                )?;
                state.serialize_field(
                    "object_value",
                    &None::<crate::proto::proximadb_v1::PropertyObject>,
                )?;
                state.serialize_field("vector_value", &None::<Vec<f32>>)?;
                state.serialize_field("null_value", &None::<serde_json::Value>)?;
            }
            Some(PropertyValueVariant::ArrayValue(v)) => {
                state.serialize_field("string_value", &None::<String>)?;
                state.serialize_field("int_value", &None::<i64>)?;
                state.serialize_field("double_value", &None::<f64>)?;
                state.serialize_field("bool_value", &None::<bool>)?;
                state.serialize_field("bytes_value", &None::<String>)?;
                state.serialize_field("array_value", &Some(v))?;
                state.serialize_field(
                    "object_value",
                    &None::<crate::proto::proximadb_v1::PropertyObject>,
                )?;
                state.serialize_field("vector_value", &None::<Vec<f32>>)?;
                state.serialize_field("null_value", &None::<serde_json::Value>)?;
            }
            Some(PropertyValueVariant::ObjectValue(v)) => {
                state.serialize_field("string_value", &None::<String>)?;
                state.serialize_field("int_value", &None::<i64>)?;
                state.serialize_field("double_value", &None::<f64>)?;
                state.serialize_field("bool_value", &None::<bool>)?;
                state.serialize_field("bytes_value", &None::<String>)?;
                state.serialize_field(
                    "array_value",
                    &None::<crate::proto::proximadb_v1::PropertyArray>,
                )?;
                state.serialize_field("object_value", &Some(v))?;
                state.serialize_field("vector_value", &None::<Vec<f32>>)?;
                state.serialize_field("null_value", &None::<serde_json::Value>)?;
            }
            Some(PropertyValueVariant::VectorValue(v)) => {
                state.serialize_field("string_value", &None::<String>)?;
                state.serialize_field("int_value", &None::<i64>)?;
                state.serialize_field("double_value", &None::<f64>)?;
                state.serialize_field("bool_value", &None::<bool>)?;
                state.serialize_field("bytes_value", &None::<String>)?;
                state.serialize_field(
                    "array_value",
                    &None::<crate::proto::proximadb_v1::PropertyArray>,
                )?;
                state.serialize_field(
                    "object_value",
                    &None::<crate::proto::proximadb_v1::PropertyObject>,
                )?;
                state.serialize_field("vector_value", &Some(&v.values))?;
                state.serialize_field("null_value", &None::<serde_json::Value>)?;
            }
            None => {
                state.serialize_field("string_value", &None::<String>)?;
                state.serialize_field("int_value", &None::<i64>)?;
                state.serialize_field("double_value", &None::<f64>)?;
                state.serialize_field("bool_value", &None::<bool>)?;
                state.serialize_field("bytes_value", &None::<String>)?;
                state.serialize_field(
                    "array_value",
                    &None::<crate::proto::proximadb_v1::PropertyArray>,
                )?;
                state.serialize_field(
                    "object_value",
                    &None::<crate::proto::proximadb_v1::PropertyObject>,
                )?;
                state.serialize_field("vector_value", &None::<Vec<f32>>)?;
                state.serialize_field("null_value", &Some(serde_json::Value::Null))?;
            }
        }

        state.end()
    }
}
impl<'de> Deserialize<'de> for PropertyValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct PropertyValueHelper {
            string_value: Option<String>,
            int_value: Option<i64>,
            double_value: Option<f64>,
            bool_value: Option<bool>,
            bytes_value: Option<String>, // base64 encoded
            array_value: Option<crate::proto::proximadb_v1::PropertyArray>,
            object_value: Option<crate::proto::proximadb_v1::PropertyObject>,
            vector_value: Option<Vec<f32>>,
            #[allow(dead_code)]
            null_value: Option<serde_json::Value>,
        }

        let helper = PropertyValueHelper::deserialize(deserializer)?;

        let value = if let Some(v) = helper.string_value {
            Some(PropertyValueVariant::StringValue(v))
        } else if let Some(v) = helper.int_value {
            Some(PropertyValueVariant::IntValue(v))
        } else if let Some(v) = helper.double_value {
            Some(PropertyValueVariant::DoubleValue(v))
        } else if let Some(v) = helper.bool_value {
            Some(PropertyValueVariant::BoolValue(v))
        } else if let Some(v) = helper.bytes_value {
            let bytes = base64_decode(&v).map_err(serde::de::Error::custom)?;
            Some(PropertyValueVariant::BytesValue(bytes))
        } else if let Some(v) = helper.array_value {
            Some(PropertyValueVariant::ArrayValue(v))
        } else if let Some(v) = helper.object_value {
            Some(PropertyValueVariant::ObjectValue(v))
        } else {
            helper.vector_value.map(|v| {
                PropertyValueVariant::VectorValue(crate::proto::proximadb_v1::VectorData {
                    values: v,
                })
            })
        };

        Ok(PropertyValue { value })
    }
}
impl Serialize for crate::proto::proximadb_v1::SourceContent {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use crate::proto::proximadb_v1::source_content::Data;
        use serde::ser::SerializeMap;

        let mut map = serializer.serialize_map(Some(1))?;

        match &self.data {
            Some(Data::TextContent(v)) => map.serialize_entry("text_content", v)?,
            Some(Data::BinaryContent(v)) => {
                map.serialize_entry("binary_content", &base64_encode(v))?
            }
            Some(Data::ExternalReference(v)) => map.serialize_entry("external_reference", v)?,
            None => map.serialize_entry("null_content", &serde_json::Value::Null)?,
        }

        map.end()
    }
}
impl Serialize for crate::proto::proximadb_v1::FilterClause {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use crate::proto::proximadb_v1::filter_clause::Value;
        use serde::ser::SerializeMap;

        let mut map = serializer.serialize_map(Some(3))?;
        map.serialize_entry("field", &self.field)?;
        map.serialize_entry("op", &self.op)?;

        match &self.value {
            Some(Value::StringValue(v)) => map.serialize_entry("string_value", v)?,
            Some(Value::IntValue(v)) => map.serialize_entry("int_value", v)?,
            Some(Value::DoubleValue(v)) => map.serialize_entry("double_value", v)?,
            Some(Value::BoolValue(v)) => map.serialize_entry("bool_value", v)?,
            None => map.serialize_entry("null_value", &serde_json::Value::Null)?,
        }

        map.end()
    }
}

// Custom serde for TypedField (has oneof value)
impl Serialize for crate::proto::proximadb_v1::TypedField {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("TypedField", 3)?;

        // Serialize the oneof value field
        match &self.value {
            Some(crate::proto::proximadb_v1::typed_field::Value::StringValue(v)) => {
                state.serialize_field("value", &serde_json::json!({"string_value": v}))?;
            }
            Some(crate::proto::proximadb_v1::typed_field::Value::IntValue(v)) => {
                state.serialize_field("value", &serde_json::json!({"int_value": v}))?;
            }
            Some(crate::proto::proximadb_v1::typed_field::Value::DoubleValue(v)) => {
                state.serialize_field("value", &serde_json::json!({"double_value": v}))?;
            }
            Some(crate::proto::proximadb_v1::typed_field::Value::BoolValue(v)) => {
                state.serialize_field("value", &serde_json::json!({"bool_value": v}))?;
            }
            Some(crate::proto::proximadb_v1::typed_field::Value::StringArray(v)) => {
                state.serialize_field("value", &serde_json::json!({"string_array": v}))?;
            }
            Some(crate::proto::proximadb_v1::typed_field::Value::TimestampValueMs(v)) => {
                state.serialize_field("value", &serde_json::json!({"timestamp_value_ms": v}))?;
            }
            None => {
                state.serialize_field("value", &serde_json::Value::Null)?;
            }
        }

        // Serialize the boolean fields
        state.serialize_field("indexed", &self.indexed)?;
        state.serialize_field("filterable", &self.filterable)?;

        state.end()
    }
}

impl<'de> Deserialize<'de> for crate::proto::proximadb_v1::TypedField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct TypedFieldHelper {
            value: Option<serde_json::Value>,
            indexed: Option<bool>,
            filterable: Option<bool>,
        }

        let helper = TypedFieldHelper::deserialize(deserializer)?;

        let value = if let Some(val) = helper.value {
            if let Some(obj) = val.as_object() {
                if let Some(string_val) = obj.get("string_value") {
                    string_val.as_str().map(|s| {
                        crate::proto::proximadb_v1::typed_field::Value::StringValue(s.to_string())
                    })
                } else if let Some(int_val) = obj.get("int_value") {
                    int_val
                        .as_i64()
                        .map(crate::proto::proximadb_v1::typed_field::Value::IntValue)
                } else if let Some(double_val) = obj.get("double_value") {
                    double_val
                        .as_f64()
                        .map(crate::proto::proximadb_v1::typed_field::Value::DoubleValue)
                } else if let Some(bool_val) = obj.get("bool_value") {
                    bool_val
                        .as_bool()
                        .map(crate::proto::proximadb_v1::typed_field::Value::BoolValue)
                } else if let Some(timestamp_val) = obj.get("timestamp_value_ms") {
                    timestamp_val
                        .as_i64()
                        .map(crate::proto::proximadb_v1::typed_field::Value::TimestampValueMs)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        Ok(crate::proto::proximadb_v1::TypedField {
            value,
            indexed: helper.indexed.unwrap_or(false),
            filterable: helper.filterable.unwrap_or(false),
        })
    }
}

// Custom serde for MetadataValue (has oneof value)
impl Serialize for crate::proto::proximadb_v1::MetadataValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(1))?;

        match &self.value {
            Some(crate::proto::proximadb_v1::metadata_value::Value::StringValue(v)) => {
                map.serialize_entry("string_value", v)?;
            }
            Some(crate::proto::proximadb_v1::metadata_value::Value::IntValue(v)) => {
                map.serialize_entry("int_value", v)?;
            }
            Some(crate::proto::proximadb_v1::metadata_value::Value::DoubleValue(v)) => {
                map.serialize_entry("double_value", v)?;
            }
            Some(crate::proto::proximadb_v1::metadata_value::Value::BoolValue(v)) => {
                map.serialize_entry("bool_value", v)?;
            }
            None => {
                map.serialize_entry("null_value", &serde_json::Value::Null)?;
            }
        }

        map.end()
    }
}

impl<'de> Deserialize<'de> for crate::proto::proximadb_v1::MetadataValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct MetadataValueHelper {
            string_value: Option<String>,
            int_value: Option<i64>,
            double_value: Option<f64>,
            bool_value: Option<bool>,
            #[allow(dead_code)]
            null_value: Option<serde_json::Value>,
        }

        let helper = MetadataValueHelper::deserialize(deserializer)?;

        let value = if let Some(v) = helper.string_value {
            Some(crate::proto::proximadb_v1::metadata_value::Value::StringValue(v))
        } else if let Some(v) = helper.int_value {
            Some(crate::proto::proximadb_v1::metadata_value::Value::IntValue(
                v,
            ))
        } else if let Some(v) = helper.double_value {
            Some(crate::proto::proximadb_v1::metadata_value::Value::DoubleValue(v))
        } else {
            helper
                .bool_value
                .map(crate::proto::proximadb_v1::metadata_value::Value::BoolValue)
        };

        Ok(crate::proto::proximadb_v1::MetadataValue { value })
    }
}

// FilterClause Deserialize implementation
impl<'de> Deserialize<'de> for crate::proto::proximadb_v1::FilterClause {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct FilterClauseHelper {
            field: String,
            op: i32,
            string_value: Option<String>,
            int_value: Option<i64>,
            double_value: Option<f64>,
            bool_value: Option<bool>,
            #[allow(dead_code)]
            null_value: Option<serde_json::Value>,
        }

        let helper = FilterClauseHelper::deserialize(deserializer)?;

        let value = if let Some(v) = helper.string_value {
            Some(crate::proto::proximadb_v1::filter_clause::Value::StringValue(v))
        } else if let Some(v) = helper.int_value {
            Some(crate::proto::proximadb_v1::filter_clause::Value::IntValue(
                v,
            ))
        } else if let Some(v) = helper.double_value {
            Some(crate::proto::proximadb_v1::filter_clause::Value::DoubleValue(v))
        } else {
            helper
                .bool_value
                .map(crate::proto::proximadb_v1::filter_clause::Value::BoolValue)
        };

        Ok(crate::proto::proximadb_v1::FilterClause {
            field: helper.field,
            op: helper.op,
            value,
        })
    }
}

// ============================================================================
// Graph Types - Custom Serde for Complex Types with PropertyValue
// ============================================================================

use crate::proto::proximadb_v1::{Edge, Node};

// Node - has PropertyValue fields
impl Serialize for Node {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("Node", 6)?;
        state.serialize_field("id", &self.id)?;
        state.serialize_field("labels", &self.labels)?;

        // Sort HashMap keys for deterministic bincode serialization
        let sorted_properties: std::collections::BTreeMap<_, _> = self.properties.iter().collect();
        state.serialize_field("properties", &sorted_properties)?;

        state.serialize_field("embedding", &self.embedding)?;
        state.serialize_field("created_at_ms", &self.created_at_ms)?;
        state.serialize_field("updated_at_ms", &self.updated_at_ms)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for Node {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct NodeHelper {
            id: String,
            labels: Vec<String>,
            properties: std::collections::HashMap<String, PropertyValue>,
            embedding: Option<crate::proto::proximadb_v1::EmbeddingVersion>,
            created_at_ms: i64,
            updated_at_ms: i64,
        }

        let helper = NodeHelper::deserialize(deserializer)?;
        Ok(Node {
            id: helper.id,
            labels: helper.labels,
            properties: helper.properties,
            embedding: helper.embedding,
            created_at_ms: helper.created_at_ms,
            updated_at_ms: helper.updated_at_ms,
        })
    }
}

// Edge - has PropertyValue fields
impl Serialize for Edge {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("Edge", 8)?;
        state.serialize_field("id", &self.id)?;
        state.serialize_field("from_node_id", &self.from_node_id)?;
        state.serialize_field("to_node_id", &self.to_node_id)?;
        state.serialize_field("edge_type", &self.edge_type)?;

        // Sort HashMap keys for deterministic bincode serialization
        let sorted_properties: std::collections::BTreeMap<_, _> = self.properties.iter().collect();
        state.serialize_field("properties", &sorted_properties)?;

        state.serialize_field("weight", &self.weight)?;
        state.serialize_field("created_at_ms", &self.created_at_ms)?;
        state.serialize_field("updated_at_ms", &self.updated_at_ms)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for Edge {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct EdgeHelper {
            id: String,
            from_node_id: String,
            to_node_id: String,
            edge_type: String,
            properties: std::collections::HashMap<String, PropertyValue>,
            weight: Option<f64>,
            created_at_ms: i64,
            updated_at_ms: i64,
        }

        let helper = EdgeHelper::deserialize(deserializer)?;
        Ok(Edge {
            id: helper.id,
            from_node_id: helper.from_node_id,
            to_node_id: helper.to_node_id,
            edge_type: helper.edge_type,
            properties: helper.properties,
            weight: helper.weight,
            created_at_ms: helper.created_at_ms,
            updated_at_ms: helper.updated_at_ms,
        })
    }
}

// GraphCollection and CreateGraphRequest now use auto-generated serde
// since all their nested types have serde derives in build.rs

// PropertyConstraint - has oneof so needs custom serde
use crate::proto::proximadb_v1::{PropertyConstraint, property_constraint::Constraint};

impl Serialize for PropertyConstraint {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(1))?;

        match &self.constraint {
            Some(Constraint::StringConstraint(v)) => {
                map.serialize_entry("string_constraint", v)?;
            }
            Some(Constraint::NumericConstraint(v)) => {
                map.serialize_entry("numeric_constraint", v)?;
            }
            Some(Constraint::ArrayConstraint(v)) => {
                map.serialize_entry("array_constraint", v)?;
            }
            Some(Constraint::RegexConstraint(v)) => {
                map.serialize_entry("regex_constraint", v)?;
            }
            None => {
                map.serialize_entry("null_constraint", &serde_json::Value::Null)?;
            }
        }

        map.end()
    }
}

impl<'de> Deserialize<'de> for PropertyConstraint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;

        if let Some(obj) = value.as_object() {
            if let Some(v) = obj.get("string_constraint") {
                let constraint =
                    serde_json::from_value(v.clone()).map_err(serde::de::Error::custom)?;
                return Ok(PropertyConstraint {
                    constraint: Some(Constraint::StringConstraint(constraint)),
                });
            }
            if let Some(v) = obj.get("numeric_constraint") {
                let constraint =
                    serde_json::from_value(v.clone()).map_err(serde::de::Error::custom)?;
                return Ok(PropertyConstraint {
                    constraint: Some(Constraint::NumericConstraint(constraint)),
                });
            }
            if let Some(v) = obj.get("array_constraint") {
                let constraint =
                    serde_json::from_value(v.clone()).map_err(serde::de::Error::custom)?;
                return Ok(PropertyConstraint {
                    constraint: Some(Constraint::ArrayConstraint(constraint)),
                });
            }
            if let Some(v) = obj.get("regex_constraint") {
                let constraint =
                    serde_json::from_value(v.clone()).map_err(serde::de::Error::custom)?;
                return Ok(PropertyConstraint {
                    constraint: Some(Constraint::RegexConstraint(constraint)),
                });
            }
        }

        Ok(PropertyConstraint { constraint: None })
    }
}

#[cfg(test)]
mod json_to_sql_value_tests {
    use super::*;

    /// Pins the number-fallback decision. The reachable u64-above-i64::MAX
    /// case must approximate as f64 `NumberValue` — the pre-consolidation
    /// embedded path silently wrapped it to `Int64Value(i as i64)`.
    #[test]
    fn huge_u64_approximates_as_number_value() {
        let big = serde_json::json!(u64::MAX);
        assert_eq!(
            json_to_sql_value(&big).value,
            Some(SqlValueVariant::NumberValue(u64::MAX as f64))
        );

        // Small u64s still hit the exact i64 path first.
        let small = serde_json::json!(42u64);
        assert_eq!(
            json_to_sql_value(&small).value,
            Some(SqlValueVariant::Int64Value(42))
        );

        let neg = serde_json::json!(-7i64);
        assert_eq!(
            json_to_sql_value(&neg).value,
            Some(SqlValueVariant::Int64Value(-7))
        );
    }

    /// Simple forms + nesting convert without surprises.
    #[test]
    fn simple_and_nested_forms_convert() {
        assert_eq!(
            json_to_sql_value(&serde_json::json!("s")).value,
            Some(SqlValueVariant::StringValue("s".into()))
        );
        assert_eq!(
            json_to_sql_value(&serde_json::json!(true)).value,
            Some(SqlValueVariant::BoolValue(true))
        );
        assert_eq!(
            json_to_sql_value(&serde_json::json!(null)).value,
            Some(SqlValueVariant::NullValue(0))
        );

        let nested = serde_json::json!({"a": [1, "x", true], "b": {"c": 2.5}});
        let Some(SqlValueVariant::ObjectValue(obj)) = json_to_sql_value(&nested).value else {
            panic!("top level must convert to ObjectValue");
        };
        let Some(SqlValue {
            value: Some(SqlValueVariant::ArrayValue(arr)),
        }) = obj.fields.get("a")
        else {
            panic!("field 'a' must convert to ArrayValue");
        };
        assert_eq!(arr.values.len(), 3);
        let Some(SqlValue {
            value: Some(SqlValueVariant::ObjectValue(inner)),
        }) = obj.fields.get("b")
        else {
            panic!("field 'b' must convert to nested ObjectValue");
        };
        assert_eq!(
            inner.fields.get("c").and_then(|v| v.value.clone()),
            Some(SqlValueVariant::NumberValue(2.5))
        );
    }
}
