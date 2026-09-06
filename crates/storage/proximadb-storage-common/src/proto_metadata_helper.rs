// Helper functions for proto metadata conversion with repeated MetadataItem

use proximadb_proto::proximadb_v1::{MetadataItem, SqlValue, metadata_item};
use std::collections::HashMap;

/// Convert repeated MetadataItem to serde_json Value map for REST API compatibility
///
/// This properly handles the typed metadata values from the proto definition,
/// preserving type information for accurate comparisons and filtering.
pub fn proto_metadata_to_json(metadata: &[MetadataItem]) -> HashMap<String, serde_json::Value> {
    metadata
        .iter()
        .map(|item| {
            let value = match &item.value {
                Some(metadata_item::Value::StringValue(s)) => serde_json::Value::String(s.clone()),
                Some(metadata_item::Value::NumberValue(n)) => serde_json::Number::from_f64(*n)
                    .map_or_else(
                        || serde_json::Value::String(n.to_string()),
                        serde_json::Value::Number,
                    ),
                Some(metadata_item::Value::BoolValue(b)) => serde_json::Value::Bool(*b),
                None => serde_json::Value::Null,
            };
            (item.key.clone(), value)
        })
        .collect()
}

/// Convert serde_json Value map to repeated MetadataItem
pub fn json_metadata_to_proto(metadata: &HashMap<String, serde_json::Value>) -> Vec<MetadataItem> {
    metadata
        .iter()
        .map(|(k, v)| {
            let value = match v {
                serde_json::Value::String(s) => Some(metadata_item::Value::StringValue(s.clone())),
                serde_json::Value::Number(n) => {
                    if let Some(f) = n.as_f64() {
                        Some(metadata_item::Value::NumberValue(f))
                    } else {
                        // Fallback for very large integers
                        Some(metadata_item::Value::StringValue(n.to_string()))
                    }
                }
                serde_json::Value::Bool(b) => Some(metadata_item::Value::BoolValue(*b)),
                serde_json::Value::Null => None,
                other => Some(metadata_item::Value::StringValue(other.to_string())), // Arrays and objects become JSON strings
            };
            MetadataItem {
                key: k.clone(),
                value,
            }
        })
        .collect()
}

/// Convert repeated MetadataItem to HashMap<String, String> for internal use
/// This converts all values to strings for backward compatibility
pub fn proto_metadata_to_hashmap(metadata: &[MetadataItem]) -> HashMap<String, String> {
    metadata
        .iter()
        .map(|item| {
            let value_str = match &item.value {
                Some(metadata_item::Value::StringValue(s)) => s.clone(),
                Some(metadata_item::Value::NumberValue(n)) => n.to_string(),
                Some(metadata_item::Value::BoolValue(b)) => b.to_string(),
                None => String::new(),
            };
            (item.key.clone(), value_str)
        })
        .collect()
}

/// Convert HashMap<String, String> to repeated MetadataItem
/// All values are stored as strings since the source is HashMap<String, String>
pub fn hashmap_to_proto_metadata(metadata: &HashMap<String, String>) -> Vec<MetadataItem> {
    metadata
        .iter()
        .map(|(k, v)| {
            // Try to parse the string value to determine its type
            let value = if let Ok(n) = v.parse::<f64>() {
                Some(metadata_item::Value::NumberValue(n))
            } else if let Ok(b) = v.parse::<bool>() {
                Some(metadata_item::Value::BoolValue(b))
            } else {
                // Default to string for all other cases (including empty strings)
                Some(metadata_item::Value::StringValue(v.clone()))
            };

            MetadataItem {
                key: k.clone(),
                value,
            }
        })
        .collect()
}

/// Convert Vec<MetadataItem> to HashMap<String, SqlValue> for proto v1 compatibility
pub fn proto_metadata_to_sqlvalue_hashmap(metadata: &[MetadataItem]) -> HashMap<String, SqlValue> {
    metadata
        .iter()
        .map(|item| {
            let sql_value = match &item.value {
                Some(metadata_item::Value::StringValue(s)) => SqlValue {
                    value: Some(
                        proximadb_proto::proximadb_v1::sql_value::Value::StringValue(s.clone()),
                    ),
                },
                Some(metadata_item::Value::NumberValue(n)) => SqlValue {
                    value: Some(proximadb_proto::proximadb_v1::sql_value::Value::NumberValue(*n)),
                },
                Some(metadata_item::Value::BoolValue(b)) => SqlValue {
                    value: Some(proximadb_proto::proximadb_v1::sql_value::Value::BoolValue(
                        *b,
                    )),
                },
                None => SqlValue {
                    value: Some(proximadb_proto::proximadb_v1::sql_value::Value::NullValue(
                        0,
                    )), // 0 represents NULL_VALUE enum
                },
            };
            (item.key.clone(), sql_value)
        })
        .collect()
}

/// Convert SqlValue to MetadataItem for compatibility
pub fn sqlvalue_to_metadata_item(key: String, sql_value: &SqlValue) -> MetadataItem {
    let value = match &sql_value.value {
        Some(proximadb_proto::proximadb_v1::sql_value::Value::StringValue(s)) => {
            Some(metadata_item::Value::StringValue(s.clone()))
        }
        Some(proximadb_proto::proximadb_v1::sql_value::Value::NumberValue(n)) => {
            Some(metadata_item::Value::NumberValue(*n))
        }
        Some(proximadb_proto::proximadb_v1::sql_value::Value::BoolValue(b)) => {
            Some(metadata_item::Value::BoolValue(*b))
        }
        Some(proximadb_proto::proximadb_v1::sql_value::Value::Int64Value(i)) => {
            Some(metadata_item::Value::NumberValue(*i as f64))
        }
        _ => None, // BytesValue, NullValue, ArrayValue, ObjectValue not supported in MetadataItem
    };
    MetadataItem { key, value }
}

/// Convert HashMap<String, SqlValue> to serde_json Value map for REST API compatibility
pub fn sqlvalue_metadata_to_json(
    metadata: &HashMap<String, SqlValue>,
) -> HashMap<String, serde_json::Value> {
    metadata
        .iter()
        .map(|(key, sql_value)| {
            (
                key.clone(),
                proximadb_records::conversions::sql_value_to_json(sql_value),
            )
        })
        .collect()
}
