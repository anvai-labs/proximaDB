//! # Arrow Conversion Utilities
//!
//! Provides conversion between Arrow RecordBatches and ProximaDB internal types.
//! This enables the storage-compute separation by standardizing on Arrow as the
//! data exchange format.
//!
//! ## Key Conversions
//!
//! - VectorRecord ↔ RecordBatch
//! - VectorBatch ↔ RecordBatch
//! - Metadata ↔ Arrow Arrays
//! - Filter expressions ↔ Arrow compute predicates

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use arrow_array::{
    Array, ArrayRef, Float32Array, Int32Array, Int64Array, ListArray, RecordBatch, StringArray,
    builder::{Float32Builder, ListBuilder},
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use serde_json::Value as JsonValue;

use super::traits::VectorBatch;
use crate::proto::proximadb_v1::{SqlArray, SqlObject, VectorRecord};

// Use proto SqlValue type
use crate::proto::proximadb_v1::SqlValue;
use crate::proto::proximadb_v1::sql_value::Value as ProtoSqlValueInner;

// ============================================================================
// Schema Definitions
// ============================================================================

/// Standard schema for vector data in Arrow format
pub fn vector_schema(dimension: usize) -> ArrowSchema {
    ArrowSchema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                dimension as i32,
            ),
            false,
        ),
        Field::new("metadata", DataType::Utf8, true), // JSON encoded
        Field::new("version", DataType::Int64, true),
        Field::new("timestamp", DataType::Int64, true),
    ])
}

/// Standard schema for vector data with flat vector (variable dimension)
pub fn vector_schema_flat() -> ArrowSchema {
    ArrowSchema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, false))),
            false,
        ),
        Field::new("dimension", DataType::Int32, false),
        Field::new("metadata", DataType::Utf8, true), // JSON encoded
        Field::new("version", DataType::Int64, true),
        Field::new("timestamp", DataType::Int64, true),
    ])
}

/// Standard schema for document data
pub fn document_schema() -> ArrowSchema {
    ArrowSchema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("document", DataType::Utf8, false), // JSON encoded
        Field::new("version", DataType::Int64, true),
        Field::new("created_at", DataType::Int64, true),
        Field::new("updated_at", DataType::Int64, true),
    ])
}

/// Standard schema for graph nodes
pub fn graph_node_schema() -> ArrowSchema {
    ArrowSchema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new(
            "labels",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, false))),
            false,
        ),
        Field::new("properties", DataType::Utf8, true), // JSON encoded
    ])
}

/// Standard schema for graph edges
pub fn graph_edge_schema() -> ArrowSchema {
    ArrowSchema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("source_id", DataType::Utf8, false),
        Field::new("target_id", DataType::Utf8, false),
        Field::new("edge_type", DataType::Utf8, false),
        Field::new("properties", DataType::Utf8, true), // JSON encoded
        Field::new("weight", DataType::Float64, true),
    ])
}

// ============================================================================
// VectorBatch ↔ RecordBatch Conversions
// ============================================================================

/// Convert VectorBatch to Arrow RecordBatch
pub fn vector_batch_to_record_batch(batch: &VectorBatch) -> Result<RecordBatch> {
    let num_vectors = batch.ids.len();
    let dimension = batch.dimension;

    // Build ID array
    let id_array: ArrayRef = Arc::new(StringArray::from(batch.ids.clone()));

    // Build vector array (as List<Float32>) with non-nullable items to match schema
    let item_field = Arc::new(Field::new("item", DataType::Float32, false));
    let mut vector_builder = ListBuilder::new(Float32Builder::new()).with_field(item_field);
    for i in 0..num_vectors {
        let start = i * dimension;
        let end = start + dimension;
        let vec_slice = &batch.vectors[start..end];

        let values = vector_builder.values();
        for &v in vec_slice {
            values.append_value(v);
        }
        vector_builder.append(true);
    }
    let vector_array: ArrayRef = Arc::new(vector_builder.finish());

    // Build dimension array
    let dimension_array: ArrayRef = Arc::new(Int32Array::from(vec![dimension as i32; num_vectors]));

    // Build metadata array (JSON encoded)
    let metadata_array: ArrayRef = if let Some(ref metadata) = batch.metadata {
        Arc::new(StringArray::from(
            metadata
                .iter()
                .map(|m| serde_json::to_string(m).unwrap_or_default())
                .collect::<Vec<_>>(),
        ))
    } else {
        Arc::new(StringArray::from(vec![None::<String>; num_vectors]))
    };

    // Build version array (placeholder)
    let version_array: ArrayRef = Arc::new(Int64Array::from(vec![None::<i64>; num_vectors]));

    // Build timestamp array (placeholder)
    let timestamp_array: ArrayRef = Arc::new(Int64Array::from(vec![None::<i64>; num_vectors]));

    let schema = vector_schema_flat();
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            id_array,
            vector_array,
            dimension_array,
            metadata_array,
            version_array,
            timestamp_array,
        ],
    )
    .context("Failed to create RecordBatch from VectorBatch")
}

/// Convert Arrow RecordBatch to VectorBatch
pub fn record_batch_to_vector_batch(batch: &RecordBatch) -> Result<VectorBatch> {
    // Extract ID column
    let id_col = batch
        .column_by_name("id")
        .ok_or_else(|| anyhow!("Missing 'id' column"))?;
    let ids: Vec<String> = id_col
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow!("'id' column is not StringArray"))?
        .iter()
        .map(|v| v.unwrap_or_default().to_string())
        .collect();

    // Extract vector column
    let vector_col = batch
        .column_by_name("vector")
        .ok_or_else(|| anyhow!("Missing 'vector' column"))?;

    // Handle both List and FixedSizeList
    let (vectors, dimension) =
        if let Some(list_array) = vector_col.as_any().downcast_ref::<ListArray>() {
            let mut all_vectors = Vec::new();
            let mut dim = 0;

            for i in 0..list_array.len() {
                if list_array.is_valid(i) {
                    let value = list_array.value(i);
                    let float_array = value
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .ok_or_else(|| anyhow!("Vector values are not Float32Array"))?;

                    dim = float_array.len();
                    for j in 0..float_array.len() {
                        all_vectors.push(float_array.value(j));
                    }
                }
            }
            (all_vectors, dim)
        } else {
            return Err(anyhow!("'vector' column has unsupported type"));
        };

    // Extract metadata column if present
    let metadata = batch.column_by_name("metadata").and_then(|col| {
        col.as_any().downcast_ref::<StringArray>().map(|arr| {
            arr.iter()
                .map(|v| {
                    v.and_then(|s| serde_json::from_str(s).ok())
                        .unwrap_or_default()
                })
                .collect::<Vec<HashMap<String, JsonValue>>>()
        })
    });

    Ok(VectorBatch {
        ids,
        vectors,
        dimension,
        metadata,
    })
}

// ============================================================================
// VectorRecord ↔ RecordBatch Conversions
// ============================================================================

/// Convert Vec<VectorRecord> to Arrow RecordBatch
pub fn vector_records_to_record_batch(records: &[VectorRecord]) -> Result<RecordBatch> {
    if records.is_empty() {
        return Err(anyhow!("Cannot create RecordBatch from empty records"));
    }

    let dimension = records[0].vector.len();
    let num_records = records.len();

    // Build arrays
    let mut ids = Vec::with_capacity(num_records);
    let mut vectors = Vec::with_capacity(num_records * dimension);
    let mut metadata_json = Vec::with_capacity(num_records);

    for record in records {
        ids.push(record.id.clone());
        vectors.extend(record.vector.iter().copied());

        // Convert metadata to JSON
        let meta: HashMap<String, JsonValue> = record
            .metadata
            .iter()
            .map(|(k, v)| (k.clone(), sql_value_to_json(v)))
            .collect();
        metadata_json.push(serde_json::to_string(&meta).unwrap_or_default());
    }

    // Create VectorBatch and convert
    let batch = VectorBatch {
        ids,
        vectors,
        dimension,
        metadata: Some(
            metadata_json
                .iter()
                .map(|s| serde_json::from_str(s).unwrap_or_default())
                .collect(),
        ),
    };

    vector_batch_to_record_batch(&batch)
}

/// Convert Arrow RecordBatch to Vec<VectorRecord>
pub fn record_batch_to_vector_records(batch: &RecordBatch) -> Result<Vec<VectorRecord>> {
    let vector_batch = record_batch_to_vector_batch(batch)?;

    let mut records = Vec::with_capacity(vector_batch.ids.len());
    let dimension = vector_batch.dimension;

    for (i, id) in vector_batch.ids.iter().enumerate() {
        let start = i * dimension;
        let end = start + dimension;
        let vector = vector_batch.vectors[start..end].to_vec();

        let metadata = vector_batch
            .metadata
            .as_ref()
            .and_then(|m| m.get(i))
            .map(|m| {
                m.iter()
                    .map(|(k, v)| (k.clone(), json_to_sql_value(v)))
                    .collect()
            })
            .unwrap_or_default();

        records.push(VectorRecord {
            id: id.clone(),
            vector,
            metadata,
            ..Default::default()
        });
    }

    Ok(records)
}

// ============================================================================
// SqlValue ↔ JSON Conversions
// ============================================================================

/// Convert proto SqlValue to JSON Value
pub fn sql_value_to_json(value: &SqlValue) -> JsonValue {
    // TD-PROTO-2 consolidation: the canonical rendering moved to
    // proximadb-records (foundation-reachable from every crate); this is the
    // compat re-export for existing root-crate callers.
    proximadb_records::conversions::sql_value_to_json(value)
}

/// Convert JSON Value to proto SqlValue
pub fn json_to_sql_value(value: &JsonValue) -> SqlValue {
    let inner = match value {
        JsonValue::Null => Some(ProtoSqlValueInner::NullValue(0)),
        JsonValue::Bool(b) => Some(ProtoSqlValueInner::BoolValue(*b)),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(ProtoSqlValueInner::Int64Value(i))
            } else if let Some(f) = n.as_f64() {
                Some(ProtoSqlValueInner::NumberValue(f))
            } else {
                Some(ProtoSqlValueInner::NullValue(0))
            }
        }
        JsonValue::String(s) => Some(ProtoSqlValueInner::StringValue(s.clone())),
        JsonValue::Array(arr) => Some(ProtoSqlValueInner::ArrayValue(SqlArray {
            values: arr.iter().map(json_to_sql_value).collect(),
        })),
        JsonValue::Object(obj) => Some(ProtoSqlValueInner::ObjectValue(SqlObject {
            fields: obj
                .iter()
                .map(|(k, v)| (k.clone(), json_to_sql_value(v)))
                .collect(),
        })),
    };
    SqlValue { value: inner }
}

// ============================================================================
// Filter Expression to Arrow Predicate
// ============================================================================

use super::traits::FilterExpression;

/// Convert FilterExpression to a displayable string (for debugging)
pub fn filter_to_string(filter: &FilterExpression) -> String {
    match filter {
        FilterExpression::Comparison { column, op, value } => {
            let op_str = match op {
                super::traits::ComparisonOp::Eq => "=",
                super::traits::ComparisonOp::Ne => "!=",
                super::traits::ComparisonOp::Lt => "<",
                super::traits::ComparisonOp::Le => "<=",
                super::traits::ComparisonOp::Gt => ">",
                super::traits::ComparisonOp::Ge => ">=",
                super::traits::ComparisonOp::Like => "LIKE",
            };
            format!("{} {} {}", column, op_str, value)
        }
        FilterExpression::And(filters) => {
            let parts: Vec<String> = filters.iter().map(filter_to_string).collect();
            format!("({})", parts.join(" AND "))
        }
        FilterExpression::Or(filters) => {
            let parts: Vec<String> = filters.iter().map(filter_to_string).collect();
            format!("({})", parts.join(" OR "))
        }
        FilterExpression::Not(inner) => {
            format!("NOT {}", filter_to_string(inner))
        }
        FilterExpression::IsNull { column } => {
            format!("{} IS NULL", column)
        }
        FilterExpression::IsNotNull { column } => {
            format!("{} IS NOT NULL", column)
        }
        FilterExpression::In { column, values } => {
            let vals: Vec<String> = values.iter().map(|v| v.to_string()).collect();
            format!("{} IN ({})", column, vals.join(", "))
        }
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Get Arrow DataType for a proto SqlValue
pub fn sql_value_to_arrow_type(value: &SqlValue) -> DataType {
    match &value.value {
        None => DataType::Null,
        Some(inner) => match inner {
            ProtoSqlValueInner::NullValue(_) => DataType::Null,
            ProtoSqlValueInner::BoolValue(_) => DataType::Boolean,
            ProtoSqlValueInner::Int64Value(_) => DataType::Int64,
            ProtoSqlValueInner::NumberValue(_) => DataType::Float64,
            ProtoSqlValueInner::StringValue(_) => DataType::Utf8,
            ProtoSqlValueInner::BytesValue(_) => DataType::Binary,
            ProtoSqlValueInner::JsonbValue(_) => DataType::Binary,
            ProtoSqlValueInner::ArrayValue(arr) => {
                if let Some(first) = arr.values.first() {
                    DataType::List(Arc::new(Field::new(
                        "item",
                        sql_value_to_arrow_type(first),
                        true,
                    )))
                } else {
                    DataType::List(Arc::new(Field::new("item", DataType::Null, true)))
                }
            }
            ProtoSqlValueInner::ObjectValue(_) => DataType::Utf8, // JSON encoded
        },
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_batch_roundtrip() {
        let original = VectorBatch {
            ids: vec!["v1".to_string(), "v2".to_string()],
            vectors: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            dimension: 3,
            metadata: Some(vec![
                [("key".to_string(), JsonValue::String("value1".to_string()))]
                    .into_iter()
                    .collect(),
                [("key".to_string(), JsonValue::String("value2".to_string()))]
                    .into_iter()
                    .collect(),
            ]),
        };

        let record_batch = vector_batch_to_record_batch(&original).unwrap();
        let recovered = record_batch_to_vector_batch(&record_batch).unwrap();

        assert_eq!(original.ids, recovered.ids);
        assert_eq!(original.dimension, recovered.dimension);
        assert_eq!(original.vectors.len(), recovered.vectors.len());

        // Check vectors are approximately equal (float comparison)
        for (a, b) in original.vectors.iter().zip(recovered.vectors.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sql_value_json_roundtrip() {
        // Create a proto SqlValue with object value
        let mut fields = std::collections::HashMap::new();
        fields.insert(
            "string".to_string(),
            SqlValue {
                value: Some(ProtoSqlValueInner::StringValue("hello".to_string())),
            },
        );
        fields.insert(
            "number".to_string(),
            SqlValue {
                value: Some(ProtoSqlValueInner::Int64Value(42)),
            },
        );
        fields.insert(
            "float".to_string(),
            SqlValue {
                value: Some(ProtoSqlValueInner::NumberValue(3.5)),
            },
        );
        fields.insert(
            "bool".to_string(),
            SqlValue {
                value: Some(ProtoSqlValueInner::BoolValue(true)),
            },
        );
        fields.insert(
            "null".to_string(),
            SqlValue {
                value: Some(ProtoSqlValueInner::NullValue(0)),
            },
        );

        let original = SqlValue {
            value: Some(ProtoSqlValueInner::ObjectValue(SqlObject { fields })),
        };

        let json = sql_value_to_json(&original);
        let recovered = json_to_sql_value(&json);

        // Compare as JSON since SqlValue doesn't implement PartialEq for all nested types
        let original_json = sql_value_to_json(&original);
        let recovered_json = sql_value_to_json(&recovered);
        assert_eq!(original_json, recovered_json);
    }

    #[test]
    fn test_vector_schema() {
        let schema = vector_schema(128);
        assert_eq!(schema.fields().len(), 5);
        assert!(schema.field_with_name("id").is_ok());
        assert!(schema.field_with_name("vector").is_ok());
        assert!(schema.field_with_name("metadata").is_ok());
    }

    #[test]
    fn test_filter_to_string() {
        let filter = FilterExpression::And(vec![
            FilterExpression::Comparison {
                column: "price".to_string(),
                op: super::super::traits::ComparisonOp::Gt,
                value: JsonValue::Number(100.into()),
            },
            FilterExpression::IsNotNull {
                column: "name".to_string(),
            },
        ]);

        let s = filter_to_string(&filter);
        assert!(s.contains("price > 100"));
        assert!(s.contains("name IS NOT NULL"));
        assert!(s.contains("AND"));
    }
}
