//! Strongly-typed metadata structures for ProximaDB
//!
//! This module provides high-performance, type-safe metadata handling
//! to replace the overhead of serde_json::Value with zero-cost abstractions.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;
use std::sync::Arc;

/// Strongly-typed metadata value enum matching protobuf definition
/// This provides zero-cost abstraction over dynamic metadata
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    /// UTF-8 string value (reference-counted for cheap cloning)
    String(Arc<str>),
    /// 64-bit floating-point number
    Number(f64),
    /// Boolean value
    Bool(bool),
    /// Null / absent value
    Null,
    // Future: Array(Vec<MetadataValue>),
    // Future: Object(HashMap<String, MetadataValue>),
}

// Custom Serialize implementation for MetadataValue
impl Serialize for MetadataValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            MetadataValue::String(s) => serializer.serialize_str(s),
            MetadataValue::Number(n) => serializer.serialize_f64(*n),
            MetadataValue::Bool(b) => serializer.serialize_bool(*b),
            MetadataValue::Null => serializer.serialize_none(),
        }
    }
}

// Custom Deserialize implementation for MetadataValue
impl<'de> Deserialize<'de> for MetadataValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        Ok(MetadataValue::from_json(value))
    }
}

impl MetadataValue {
    /// Convert from serde_json::Value (for backward compatibility)
    pub fn from_json(value: serde_json::Value) -> Self {
        match value {
            serde_json::Value::String(s) => MetadataValue::String(Arc::from(s.as_str())),
            serde_json::Value::Number(n) => MetadataValue::Number(n.as_f64().unwrap_or(0.0)),
            serde_json::Value::Bool(b) => MetadataValue::Bool(b),
            serde_json::Value::Null => MetadataValue::Null,
            // For now, convert arrays and objects to strings
            // In production, we'd implement full support
            v => MetadataValue::String(Arc::from(v.to_string().as_str())),
        }
    }

    /// Convert to serde_json::Value (for backward compatibility)
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            MetadataValue::String(s) => serde_json::Value::String(s.to_string()),
            MetadataValue::Number(n) => {
                serde_json::json!(*n)
            }
            MetadataValue::Bool(b) => serde_json::Value::Bool(*b),
            MetadataValue::Null => serde_json::Value::Null,
        }
    }

    /// Get as string if possible
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetadataValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Get as number if possible
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            MetadataValue::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// Get as bool if possible
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            MetadataValue::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

/// Optimized metadata storage with strongly-typed values
/// Uses Arc internally to minimize cloning overhead
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TypedMetadata {
    // Use Arc to enable cheap cloning
    inner: Arc<HashMap<String, MetadataValue>>,
}

// Custom Serialize - serialize as regular HashMap
impl Serialize for TypedMetadata {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.inner.as_ref().serialize(serializer)
    }
}

// Custom Deserialize - deserialize into Arc<HashMap>
impl<'de> Deserialize<'de> for TypedMetadata {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let map = HashMap::<String, MetadataValue>::deserialize(deserializer)?;
        Ok(TypedMetadata {
            inner: Arc::new(map),
        })
    }
}

impl TypedMetadata {
    /// Create new empty metadata
    pub fn new() -> Self {
        Self {
            inner: Arc::new(HashMap::new()),
        }
    }

    /// Create from a HashMap of MetadataValue
    pub fn from_map(map: HashMap<String, MetadataValue>) -> Self {
        Self {
            inner: Arc::new(map),
        }
    }

    /// Convert from HashMap<String, serde_json::Value> for compatibility
    pub fn from_json_map(json_map: HashMap<String, serde_json::Value>) -> Self {
        let mut map = HashMap::with_capacity(json_map.len());
        for (key, value) in json_map {
            map.insert(key, MetadataValue::from_json(value));
        }
        Self::from_map(map)
    }

    /// Convert from HashMap with string interning for deduplication
    /// This method should be used when loading metadata from storage
    pub async fn from_json_map_interned(
        json_map: HashMap<String, serde_json::Value>,
        interner: &proximadb_interner::StringInterner,
    ) -> Self {
        let mut map = HashMap::with_capacity(json_map.len());
        for (key, value) in json_map {
            // Intern string values for deduplication
            let interned_value = match value {
                serde_json::Value::String(s) => {
                    let interned = interner.intern(&s).await;
                    MetadataValue::String(interned)
                }
                other => MetadataValue::from_json(other),
            };
            map.insert(key, interned_value);
        }
        Self::from_map(map)
    }

    /// Convert to HashMap<String, serde_json::Value> for compatibility
    pub fn to_json_map(&self) -> HashMap<String, serde_json::Value> {
        let mut json_map = HashMap::with_capacity(self.inner.len());
        for (key, value) in self.inner.iter() {
            json_map.insert(key.clone(), value.to_json());
        }
        json_map
    }

    /// Get a value by key
    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.inner.get(key)
    }

    /// Check if contains key
    pub fn contains_key(&self, key: &str) -> bool {
        self.inner.contains_key(key)
    }

    /// Get inner map reference
    pub fn as_map(&self) -> &HashMap<String, MetadataValue> {
        &self.inner
    }

    /// Number of metadata entries
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Create a mutable version (performs Arc::make_mut)
    pub fn make_mut(&mut self) -> &mut HashMap<String, MetadataValue> {
        Arc::make_mut(&mut self.inner)
    }

    /// Insert a key-value pair (creates new Arc)
    pub fn insert(&mut self, key: String, value: MetadataValue) {
        Arc::make_mut(&mut self.inner).insert(key, value);
    }

    /// Remove a key (creates new Arc if needed)
    pub fn remove(&mut self, key: &str) -> Option<MetadataValue> {
        Arc::make_mut(&mut self.inner).remove(key)
    }

    /// Iterator over key-value pairs
    pub fn iter(&self) -> impl Iterator<Item = (&String, &MetadataValue)> {
        self.inner.iter()
    }

    /// Clear all metadata entries
    pub fn clear(&mut self) {
        Arc::make_mut(&mut self.inner).clear();
    }
}

/// Convert from protobuf MetadataItem
impl From<&proximadb_proto::proximadb_v1::MetadataItem> for MetadataValue {
    fn from(item: &proximadb_proto::proximadb_v1::MetadataItem) -> Self {
        use proximadb_proto::proximadb_v1::metadata_item::Value;

        match &item.value {
            Some(Value::StringValue(s)) => MetadataValue::String(Arc::from(s.as_str())),
            Some(Value::NumberValue(n)) => MetadataValue::Number(*n),
            Some(Value::BoolValue(b)) => MetadataValue::Bool(*b),
            None => MetadataValue::Null,
        }
    }
}

/// Convert from v1 protobuf SqlValue (used in v1 vector types)
impl From<&proximadb_proto::proximadb_v1::SqlValue> for MetadataValue {
    fn from(item: &proximadb_proto::proximadb_v1::SqlValue) -> Self {
        use proximadb_proto::proximadb_v1::sql_value::Value;
        match &item.value {
            Some(Value::StringValue(s)) => MetadataValue::String(Arc::from(s.as_str())),
            Some(Value::NumberValue(n)) => MetadataValue::Number(*n),
            Some(Value::BoolValue(b)) => MetadataValue::Bool(*b),
            Some(Value::Int64Value(i)) => MetadataValue::Number(*i as f64),
            Some(Value::BytesValue(_)) => MetadataValue::String(Arc::from("[binary]")),
            Some(Value::NullValue(_)) => MetadataValue::Null,
            Some(Value::ArrayValue(_)) => MetadataValue::String(Arc::from("[array]")),
            Some(Value::ObjectValue(_)) => MetadataValue::String(Arc::from("[object]")),
            // TD-PROTO-2: MetadataValue feeds equality/comparison — a
            // placeholder made structured comparisons always miss; decode
            // to the canonical JSON text instead.
            Some(Value::JsonbValue(b)) => MetadataValue::String(Arc::from(
                crate::ProximaValue::jsonb_to_json_string_lossy(b),
            )),
            None => MetadataValue::Null,
        }
    }
}

/// Convert to protobuf SqlValue (used in v1 vector types)
impl From<&MetadataValue> for proximadb_proto::proximadb_v1::SqlValue {
    fn from(value: &MetadataValue) -> Self {
        use proximadb_proto::proximadb_v1::sql_value::Value;

        let proto_value = match value {
            MetadataValue::String(s) => Some(Value::StringValue(s.to_string())),
            MetadataValue::Number(n) => Some(Value::NumberValue(*n)),
            MetadataValue::Bool(b) => Some(Value::BoolValue(*b)),
            MetadataValue::Null => Some(Value::NullValue(prost_types::NullValue::default().into())),
        };

        proximadb_proto::proximadb_v1::SqlValue { value: proto_value }
    }
}

// Duplicate From implementation removed - keeping the one at line 244

/// Builder pattern for efficient metadata construction
pub struct TypedMetadataBuilder {
    map: HashMap<String, MetadataValue>,
}

impl TypedMetadataBuilder {
    /// Create an empty metadata builder
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    /// Create a metadata builder with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            map: HashMap::with_capacity(capacity),
        }
    }

    /// Insert a string metadata value
    pub fn insert_string(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.map.insert(
            key.into(),
            MetadataValue::String(Arc::from(value.into().as_str())),
        );
        self
    }

    /// Insert a numeric metadata value
    pub fn insert_number(mut self, key: impl Into<String>, value: f64) -> Self {
        self.map.insert(key.into(), MetadataValue::Number(value));
        self
    }

    /// Insert a boolean metadata value
    pub fn insert_bool(mut self, key: impl Into<String>, value: bool) -> Self {
        self.map.insert(key.into(), MetadataValue::Bool(value));
        self
    }

    /// Finalize and return the constructed typed metadata
    pub fn build(self) -> TypedMetadata {
        TypedMetadata::from_map(self.map)
    }
}

impl Default for TypedMetadataBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_typed_metadata_basic() {
        let metadata = TypedMetadataBuilder::new()
            .insert_string("name", "test")
            .insert_number("score", 0.95)
            .insert_bool("active", true)
            .build();

        assert_eq!(metadata.get("name").and_then(|v| v.as_str()), Some("test"));
        assert_eq!(metadata.get("score").and_then(|v| v.as_f64()), Some(0.95));
        assert_eq!(metadata.get("active").and_then(|v| v.as_bool()), Some(true));
    }

    #[test]
    fn test_cheap_cloning() {
        let metadata1 = TypedMetadataBuilder::new()
            .insert_string("key", "value")
            .build();

        let metadata2 = metadata1.clone();

        // Both should point to the same Arc
        assert!(Arc::ptr_eq(&metadata1.inner, &metadata2.inner));
    }

    #[test]
    fn test_json_compatibility() {
        let mut json_map = HashMap::new();
        json_map.insert("string".to_string(), serde_json::json!("value"));
        json_map.insert("number".to_string(), serde_json::json!(42.0));
        json_map.insert("bool".to_string(), serde_json::json!(true));

        let metadata = TypedMetadata::from_json_map(json_map.clone());
        let back_to_json = metadata.to_json_map();

        assert_eq!(back_to_json.get("string"), json_map.get("string"));
        assert_eq!(back_to_json.get("number"), json_map.get("number"));
        assert_eq!(back_to_json.get("bool"), json_map.get("bool"));
    }
}
