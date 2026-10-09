// Copyright 2025 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Arrow <-> Proto codec for Flight protocol
//!
//! Reuses existing Arrow infrastructure from arrow_ipc_scanner.rs and columnar_io.rs
//! for maximum code reuse and consistency.

use anyhow::{Context, Result};
use arrow_array::{
    Array, ArrayRef, BinaryArray, BooleanArray, Date32Array, Decimal128Array, FixedSizeListArray,
    Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, LargeBinaryArray,
    LargeStringArray, ListArray, RecordBatch, StringArray, StructArray, Time32MillisecondArray,
    Time32SecondArray, Time64MicrosecondArray, Time64NanosecondArray, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray, UInt8Array,
    UInt16Array, UInt32Array, UInt64Array,
};
use arrow_flight::{FlightData, FlightDescriptor, Ticket};
use arrow_ipc::writer::IpcWriteOptions;
use arrow_schema::{DataType, Field, Fields, Schema, TimeUnit as ArrowTimeUnit};
use proximadb_data_model::{ProximaValue, TimeUnit};
use proximadb_records::conversions::json_to_proxima;
use proximadb_records::{EdgeShape, EmbeddingCell, ProximaRecord, ProximaTreeNode};
use proximadb_runtime::rich_search::{
    RichFilterCondition, RichFilterOperator, RichSearchRequest, RichSearchResult,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::proto::proximadb_v1::{SqlValue, VectorSearchRequest};

/// Write mode for Arrow IPC operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum WriteMode {
    /// Use WAL for durability.
    #[default]
    WAL,
    /// Direct engine write request. The current Flight service accepts this
    /// token for forward compatibility but falls back to WAL-backed writes.
    Direct,
}

/// Logical batch write operation requested over Arrow Flight.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FlightWriteOperation {
    /// Insert or replace records. This is the current v2 rich-record write path.
    #[default]
    Upsert,
    /// Insert records. Routed through the same rich-record path until strict
    /// insert-only semantics are available below the protocol layer.
    Insert,
    /// Delete records by id/oid using the v2 tombstone path.
    Delete,
}

impl From<FlightWriteOperation> for crate::services::WriteOperationKind {
    fn from(op: FlightWriteOperation) -> Self {
        match op {
            FlightWriteOperation::Upsert => Self::Upsert,
            FlightWriteOperation::Insert => Self::Insert,
            FlightWriteOperation::Delete => Self::Delete,
        }
    }
}

impl From<WriteMode> for crate::services::WriteDurabilityRequirement {
    fn from(mode: WriteMode) -> Self {
        match mode {
            WriteMode::WAL => Self::WalRequired,
            WriteMode::Direct => Self::DirectCommitAllowed,
        }
    }
}

impl FlightWriteOperation {
    pub fn from_token(token: &str) -> Option<Self> {
        let token = token.trim().to_ascii_lowercase();
        let token = token
            .strip_prefix("batch_write_mode_")
            .unwrap_or(token.as_str());
        match token {
            "upsert" | "batch_upsert" | "bulk_upsert" => Some(Self::Upsert),
            "insert" | "batch_insert" | "bulk_insert" => Some(Self::Insert),
            "delete" | "batch_delete" | "bulk_delete" => Some(Self::Delete),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Upsert => "upsert",
            Self::Insert => "insert",
            Self::Delete => "delete",
        }
    }
}

/// Metadata extracted from FlightDescriptor
#[derive(Debug, Clone)]
pub struct FlightRequestMetadata {
    /// Target collection for the Arrow Flight operation
    pub collection_id: String,
    /// Logical batch operation requested by the client
    pub operation: FlightWriteOperation,
    /// Write mode (WAL or direct engine write)
    pub write_mode: WriteMode,
    /// Whether to trigger compaction after the write completes
    pub trigger_compaction: bool,
}

/// Arrow-Proto bidirectional codec
///
/// Reuses existing conversion functions from arrow_ipc_scanner.rs
pub struct ArrowProtoCodec;

/// Canonical v2 vector-search ticket for Arrow Flight `DoGet` (TD-FLIGHT-1).
///
/// The pre-TD-FLIGHT-1 ticket was a JSON-serialized v1 `VectorSearchRequest`
/// proto, whose key shape (`queries`/`include_fields`) never matched the only
/// client that built it (the Python SDK sent `query`/`include_vectors`) — so the
/// query vector was silently dropped and Flight search returned garbage. This is
/// the clean canonical v2 contract: a self-describing JSON object discriminated
/// by `"type": "vector_search"`, carrying exactly what the canonical
/// `RichSearchRequest` needs. `do_get` routes it ahead of the removed v1
/// fallback, and `bulk_search` builds one per query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlightSearchTicket {
    /// Discriminator — must be the literal `"vector_search"`.
    #[serde(rename = "type")]
    pub ticket_type: String,
    /// Collection name or stable catalog id (resolved by the canonical service).
    pub collection_id: String,
    /// Query vector.
    pub query_vector: Vec<f32>,
    /// Top-k (defaults to 10 when absent on the wire).
    #[serde(default = "default_top_k")]
    pub top_k: u32,
    /// Typed metadata filters (defaults to none).
    #[serde(default)]
    pub filters: Vec<FlightFilter>,
    /// Whether to include result vectors in the response (defaults to false).
    #[serde(default)]
    pub include_vector: bool,
}

fn default_top_k() -> u32 {
    10
}

/// A single typed filter on the v2 Flight ticket — mirrors
/// `RichFilterCondition` but with JSON-deserializable values (the ticket is a
/// serde JSON blob, not a typed struct on the client side).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlightFilter {
    /// Field path / attribute name.
    pub field: String,
    /// Operator: eq, ne, gt, gte, lt, lte, between, in, not_in, contains,
    /// starts_with, ends_with.
    pub op: String,
    #[serde(default)]
    pub value: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value_upper: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub value_list: Vec<serde_json::Value>,
}

impl FlightSearchTicket {
    /// Parse the ticket only if it carries the v2 discriminator. Returns
    /// `None` for any other ticket shape (file/graph tickets, or the removed
    /// v1 VectorSearchRequest JSON) so `do_get` can fall through cleanly.
    pub fn from_ticket(ticket: &Ticket) -> Option<Self> {
        let parsed = serde_json::from_slice::<serde_json::Value>(&ticket.ticket).ok()?;
        let is_v2 = parsed
            .get("type")
            .and_then(|v| v.as_str())
            .is_some_and(|t| t == "vector_search");
        if !is_v2 {
            return None;
        }
        serde_json::from_value(parsed).ok()
    }

    /// Translate to the canonical transport-neutral rich request consumed by
    /// `RecordSearchPort::search_record`. Mirrors the gRPC/REST filter lowering.
    pub fn to_rich_request(&self) -> Result<RichSearchRequest> {
        let filters = self
            .filters
            .iter()
            .map(flight_filter_to_rich)
            .collect::<Result<Vec<_>>>()?;
        Ok(RichSearchRequest {
            collection_id: self.collection_id.clone(),
            query_vector: self.query_vector.clone(),
            top_k: self.top_k,
            filters,
        })
    }
}

/// Lower one Flight ticket filter to a canonical `RichFilterCondition`.
fn flight_filter_to_rich(filter: &FlightFilter) -> Result<RichFilterCondition> {
    let operator = match filter.op.as_str() {
        "eq" => RichFilterOperator::Eq,
        "ne" => RichFilterOperator::Ne,
        "gt" => RichFilterOperator::Gt,
        "gte" => RichFilterOperator::Gte,
        "lt" => RichFilterOperator::Lt,
        "lte" => RichFilterOperator::Lte,
        "between" => RichFilterOperator::Between,
        "in" => RichFilterOperator::In,
        "not_in" => RichFilterOperator::NotIn,
        "contains" => RichFilterOperator::Contains,
        "starts_with" => RichFilterOperator::StartsWith,
        "ends_with" => RichFilterOperator::EndsWith,
        other => anyhow::bail!("unsupported Flight filter operator: {other}"),
    };
    let value = json_value_to_proxima(&filter.value);
    let value_upper = filter.value_upper.as_ref().map(json_value_to_proxima);
    let value_list = filter
        .value_list
        .iter()
        .map(json_value_to_proxima)
        .collect();
    Ok(RichFilterCondition {
        field: filter.field.clone(),
        operator,
        value,
        value_upper,
        value_list,
    })
}

/// JSON value → ProximaValue, treating JSON null as `ProximaValue::Null`.
fn json_value_to_proxima(value: &serde_json::Value) -> ProximaValue {
    if value.is_null() {
        ProximaValue::Null
    } else {
        json_to_proxima(value)
    }
}

impl ArrowProtoCodec {
    /// Create standard vector schema for ProximaDB
    ///
    /// Schema:
    /// - id: Utf8 (required)
    /// - vector: FixedSizeList<Float32>(dimension) (required)
    /// - metadata: Struct<key: Utf8, value: Utf8> (optional)
    /// - timestamp: Int64 (optional)
    /// - score: Float32 (for search results, optional)
    pub fn create_vector_schema(dimension: usize) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, false)),
                    dimension as i32,
                ),
                false,
            ),
            Field::new(
                "metadata",
                DataType::Struct(Fields::from(vec![
                    Field::new("key", DataType::Utf8, true),
                    Field::new("value", DataType::Utf8, true),
                ])),
                true,
            ),
            Field::new("timestamp", DataType::Int64, true),
            Field::new("score", DataType::Float32, true),
        ]))
    }

    /// Convert Arrow RecordBatches to canonical ProximaRecord envelopes.
    ///
    /// Implements the rich Arrow Flight ingestion boundary for the multimodal
    /// record envelope described in MULTIMODAL_OVERHAUL_SPEC §3 and §9.2. The
    /// columns `id`/`oid`, `vector`, tenant/provenance fields, and graph edge
    /// topology fields are lifted into first-class envelope fields. Remaining
    /// supported Arrow scalar columns become typed `props` entries.
    pub fn batches_to_proxima_records(batches: Vec<RecordBatch>) -> Result<Vec<ProximaRecord>> {
        let mut all_records = Vec::new();

        for batch in batches {
            let records = Self::batch_to_proxima_records(&batch)?;
            all_records.extend(records);
        }

        Ok(all_records)
    }

    /// Extract record ids from Arrow batches for Flight delete operations.
    ///
    /// The v2 record envelope uses `oid`; vector-era clients commonly send
    /// `id`. Both are accepted so the Flight boundary can serve rich records
    /// and existing vector clients without a second protocol.
    pub fn batches_to_record_ids(batches: Vec<RecordBatch>) -> Result<Vec<String>> {
        let mut record_ids = Vec::new();

        for batch in batches {
            let ids = Self::batch_to_record_ids(&batch)?;
            record_ids.extend(ids);
        }

        Ok(record_ids)
    }

    fn batch_to_record_ids(batch: &RecordBatch) -> Result<Vec<String>> {
        let id_array = batch
            .column_by_name("id")
            .or_else(|| batch.column_by_name("oid"))
            .context("Missing 'id' or 'oid' column")?;

        if let Some(array) = id_array.as_any().downcast_ref::<StringArray>() {
            return (0..batch.num_rows())
                .map(|row| {
                    if array.is_null(row) {
                        Err(anyhow::anyhow!("Record id is null at row {}", row))
                    } else {
                        Ok(array.value(row).to_string())
                    }
                })
                .collect::<Result<Vec<_>>>();
        }

        if let Some(array) = id_array.as_any().downcast_ref::<LargeStringArray>() {
            return (0..batch.num_rows())
                .map(|row| {
                    if array.is_null(row) {
                        Err(anyhow::anyhow!("Record id is null at row {}", row))
                    } else {
                        Ok(array.value(row).to_string())
                    }
                })
                .collect::<Result<Vec<_>>>();
        }

        Err(anyhow::anyhow!("'id'/'oid' column is not Utf8/LargeUtf8"))
    }

    fn batch_to_proxima_records(batch: &RecordBatch) -> Result<Vec<ProximaRecord>> {
        let ids = Self::required_string_column_values(
            batch
                .column_by_name("id")
                .or_else(|| batch.column_by_name("oid"))
                .context("Missing 'id' or 'oid' column")?,
            "id/oid",
            batch.num_rows(),
        )?;

        let mut vectors = if let Some(vector_array) = batch.column_by_name("vector") {
            Some(Self::extract_vectors(vector_array, batch.num_rows())?)
        } else {
            None
        };

        let edge_sources = Self::optional_string_values(batch, "source_id")?;
        let edge_targets = Self::optional_string_values(batch, "target_id")?;
        let edge_types = Self::optional_string_values(batch, "edge_type")?;
        let edge_weights = Self::optional_float64_values(batch, "weight")?;

        let mut records = Vec::with_capacity(batch.num_rows());
        for row in 0..batch.num_rows() {
            let oid = ids[row].clone();
            let mut record = ProximaRecord {
                oid: oid.clone(),
                local_id: Some(oid),
                ..ProximaRecord::default()
            };

            if let Some(tenant_id) = Self::string_value(batch, "tenant_id", row)? {
                record.tenant_id = tenant_id;
            }
            let origin = Self::string_value(batch, "origin", row)?;
            let origin = match origin {
                Some(origin) => Some(origin),
                None => Self::string_value(batch, "source", row)?,
            };
            if let Some(origin) = origin {
                record.origin = Some(origin);
            }
            if let Some(created_at_ns) = Self::int64_value(batch, "created_at_ns", row)? {
                record.created_at_ns = created_at_ns;
            } else if let Some(timestamp_ms) = Self::int64_value(batch, "timestamp", row)? {
                record.created_at_ns = timestamp_ms * 1_000_000;
            }
            if let Some(updated_at_ns) = Self::int64_value(batch, "updated_at_ns", row)? {
                record.updated_at_ns = updated_at_ns;
            } else {
                record.updated_at_ns = record.created_at_ns;
            }

            if let Some(values) = vectors.as_mut() {
                let row_values = std::mem::take(&mut values[row]);
                let dim = row_values.len() as u32;
                record.embeddings.push(EmbeddingCell {
                    model_id: "default".to_string(),
                    modality: "dense_vector".to_string(),
                    dim,
                    values: proximadb_records::EmbeddingValues::Fp32(row_values),
                    ..Default::default()
                });
            }

            if let (Some(source_id), Some(target_id), Some(edge_type)) = (
                edge_sources.get(row).and_then(Clone::clone),
                edge_targets.get(row).and_then(Clone::clone),
                edge_types.get(row).and_then(Clone::clone),
            ) {
                record.edge = Some(EdgeShape {
                    source_id,
                    target_id,
                    edge_type,
                    weight: edge_weights.get(row).copied().flatten(),
                });
            }

            for (column_index, field) in batch.schema().fields().iter().enumerate() {
                let name = field.name();
                if Self::is_reserved_record_column(name) {
                    continue;
                }

                let array = batch.column(column_index);
                if let Some(value) = Self::arrow_value_to_proxima(array, row)? {
                    record
                        .props
                        .insert(name.clone(), ProximaTreeNode::Value(value));
                }
            }

            if let Some(metadata) = batch.column_by_name("metadata") {
                for (key, value) in Self::metadata_props(metadata, row)? {
                    record.props.insert(key, ProximaTreeNode::Value(value));
                }
            }

            records.push(record);
        }

        Ok(records)
    }

    fn is_reserved_record_column(name: &str) -> bool {
        matches!(
            name,
            "id" | "oid"
                | "local_id"
                | "vector"
                | "metadata"
                | "tenant_id"
                | "origin"
                | "source"
                | "created_at_ns"
                | "updated_at_ns"
                | "timestamp"
                | "source_id"
                | "target_id"
                | "edge_type"
                | "weight"
        )
    }

    fn string_value(batch: &RecordBatch, column: &str, row: usize) -> Result<Option<String>> {
        let Some(array) = batch.column_by_name(column) else {
            return Ok(None);
        };
        Self::optional_string_value(array, column, row)
    }

    fn int64_value(batch: &RecordBatch, column: &str, row: usize) -> Result<Option<i64>> {
        let Some(array) = batch.column_by_name(column) else {
            return Ok(None);
        };
        let array = array
            .as_any()
            .downcast_ref::<Int64Array>()
            .with_context(|| format!("'{column}' column is not Int64Array"))?;
        if array.is_null(row) {
            Ok(None)
        } else {
            Ok(Some(array.value(row)))
        }
    }

    fn optional_string_values(batch: &RecordBatch, column: &str) -> Result<Vec<Option<String>>> {
        let Some(array) = batch.column_by_name(column) else {
            return Ok(vec![None; batch.num_rows()]);
        };
        (0..batch.num_rows())
            .map(|row| Self::optional_string_value(array, column, row))
            .collect::<Result<Vec<_>>>()
    }

    fn optional_string_value(array: &ArrayRef, column: &str, row: usize) -> Result<Option<String>> {
        if let Some(array) = array.as_any().downcast_ref::<StringArray>() {
            return Ok((!array.is_null(row)).then(|| array.value(row).to_string()));
        }
        if let Some(array) = array.as_any().downcast_ref::<LargeStringArray>() {
            return Ok((!array.is_null(row)).then(|| array.value(row).to_string()));
        }
        Err(anyhow::anyhow!("'{column}' column is not Utf8/LargeUtf8"))
    }

    fn required_string_column_values(
        array: &ArrayRef,
        column: &str,
        num_rows: usize,
    ) -> Result<Vec<String>> {
        (0..num_rows)
            .map(|row| {
                Self::optional_string_value(array, column, row)?
                    .ok_or_else(|| anyhow::anyhow!("Record id is null at row {}", row))
            })
            .collect()
    }

    fn optional_float64_values(batch: &RecordBatch, column: &str) -> Result<Vec<Option<f64>>> {
        let Some(array) = batch.column_by_name(column) else {
            return Ok(vec![None; batch.num_rows()]);
        };
        if let Some(array) = array.as_any().downcast_ref::<Float64Array>() {
            return Ok((0..batch.num_rows())
                .map(|row| {
                    if array.is_null(row) {
                        None
                    } else {
                        Some(array.value(row))
                    }
                })
                .collect());
        }
        if let Some(array) = array.as_any().downcast_ref::<Float32Array>() {
            return Ok((0..batch.num_rows())
                .map(|row| {
                    if array.is_null(row) {
                        None
                    } else {
                        Some(array.value(row) as f64)
                    }
                })
                .collect());
        }
        Err(anyhow::anyhow!("'{column}' column is not Float32/Float64"))
    }

    fn arrow_value_to_proxima(array: &ArrayRef, row: usize) -> Result<Option<ProximaValue>> {
        if array.is_null(row) {
            return Ok(Some(ProximaValue::Null));
        }

        if let Some(array) = array.as_any().downcast_ref::<StringArray>() {
            return Ok(Some(ProximaValue::String(array.value(row).to_string())));
        }
        if let Some(array) = array.as_any().downcast_ref::<LargeStringArray>() {
            return Ok(Some(ProximaValue::String(array.value(row).to_string())));
        }
        if let Some(array) = array.as_any().downcast_ref::<BooleanArray>() {
            return Ok(Some(ProximaValue::Boolean(array.value(row))));
        }
        if let Some(array) = array.as_any().downcast_ref::<Int8Array>() {
            return Ok(Some(ProximaValue::Int8(array.value(row))));
        }
        if let Some(array) = array.as_any().downcast_ref::<Int16Array>() {
            return Ok(Some(ProximaValue::Int16(array.value(row))));
        }
        if let Some(array) = array.as_any().downcast_ref::<Int32Array>() {
            return Ok(Some(ProximaValue::Int32(array.value(row))));
        }
        if let Some(array) = array.as_any().downcast_ref::<Int64Array>() {
            return Ok(Some(ProximaValue::Int64(array.value(row))));
        }
        if let Some(array) = array.as_any().downcast_ref::<UInt8Array>() {
            return Ok(Some(ProximaValue::UInt8(array.value(row))));
        }
        if let Some(array) = array.as_any().downcast_ref::<UInt16Array>() {
            return Ok(Some(ProximaValue::UInt16(array.value(row))));
        }
        if let Some(array) = array.as_any().downcast_ref::<UInt32Array>() {
            return Ok(Some(ProximaValue::UInt32(array.value(row))));
        }
        if let Some(array) = array.as_any().downcast_ref::<UInt64Array>() {
            return Ok(Some(ProximaValue::UInt64(array.value(row))));
        }
        if let Some(array) = array.as_any().downcast_ref::<Float32Array>() {
            return Ok(Some(ProximaValue::Float32(array.value(row))));
        }
        if let Some(array) = array.as_any().downcast_ref::<Float64Array>() {
            return Ok(Some(ProximaValue::Float64(array.value(row))));
        }
        if let Some(array) = array.as_any().downcast_ref::<Decimal128Array>() {
            let scale = match array.data_type() {
                DataType::Decimal128(_, scale) => *scale,
                _ => 0,
            };
            return Ok(Some(ProximaValue::Decimal(Self::format_decimal128(
                array.value(row),
                scale,
            ))));
        }
        if let Some(array) = array.as_any().downcast_ref::<BinaryArray>() {
            return Ok(Some(ProximaValue::Binary(array.value(row).to_vec())));
        }
        if let Some(array) = array.as_any().downcast_ref::<LargeBinaryArray>() {
            return Ok(Some(ProximaValue::Binary(array.value(row).to_vec())));
        }
        if let Some(array) = array.as_any().downcast_ref::<Date32Array>() {
            return Ok(Some(ProximaValue::Date(array.value(row))));
        }
        if let Some(array) = array.as_any().downcast_ref::<Time32SecondArray>() {
            return Ok(Some(ProximaValue::Time(
                array.value(row) as i64,
                TimeUnit::Second,
            )));
        }
        if let Some(array) = array.as_any().downcast_ref::<Time32MillisecondArray>() {
            return Ok(Some(ProximaValue::Time(
                array.value(row) as i64,
                TimeUnit::Millisecond,
            )));
        }
        if let Some(array) = array.as_any().downcast_ref::<Time64MicrosecondArray>() {
            return Ok(Some(ProximaValue::Time(
                array.value(row),
                TimeUnit::Microsecond,
            )));
        }
        if let Some(array) = array.as_any().downcast_ref::<Time64NanosecondArray>() {
            return Ok(Some(ProximaValue::Time(
                array.value(row),
                TimeUnit::Nanosecond,
            )));
        }
        if let Some(value) = Self::timestamp_value_to_proxima(array, row) {
            return Ok(Some(value));
        }
        if matches!(array.data_type(), DataType::FixedSizeList(_, _)) {
            return Ok(Some(ProximaValue::DenseVector(
                Self::extract_vectors(array, row + 1)?
                    .get(row)
                    .cloned()
                    .unwrap_or_default(),
            )));
        }
        if matches!(array.data_type(), DataType::List(_)) {
            return Ok(Some(ProximaValue::DenseVector(
                Self::extract_vectors(array, row + 1)?
                    .get(row)
                    .cloned()
                    .unwrap_or_default(),
            )));
        }

        Ok(None)
    }

    fn timestamp_value_to_proxima(array: &ArrayRef, row: usize) -> Option<ProximaValue> {
        let (unit, has_timezone) = match array.data_type() {
            DataType::Timestamp(unit, timezone) => (*unit, timezone.is_some()),
            _ => return None,
        };

        let value = match unit {
            ArrowTimeUnit::Second => array
                .as_any()
                .downcast_ref::<TimestampSecondArray>()?
                .value(row),
            ArrowTimeUnit::Millisecond => array
                .as_any()
                .downcast_ref::<TimestampMillisecondArray>()?
                .value(row),
            ArrowTimeUnit::Microsecond => array
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()?
                .value(row),
            ArrowTimeUnit::Nanosecond => array
                .as_any()
                .downcast_ref::<TimestampNanosecondArray>()?
                .value(row),
        };
        let unit = Self::arrow_time_unit_to_proxima(unit);

        if has_timezone {
            Some(ProximaValue::TimestampTz(value, unit))
        } else {
            Some(ProximaValue::Timestamp(value, unit))
        }
    }

    fn arrow_time_unit_to_proxima(unit: ArrowTimeUnit) -> TimeUnit {
        match unit {
            ArrowTimeUnit::Second => TimeUnit::Second,
            ArrowTimeUnit::Millisecond => TimeUnit::Millisecond,
            ArrowTimeUnit::Microsecond => TimeUnit::Microsecond,
            ArrowTimeUnit::Nanosecond => TimeUnit::Nanosecond,
        }
    }

    fn format_decimal128(value: i128, scale: i8) -> String {
        if scale == 0 {
            return value.to_string();
        }

        if scale < 0 {
            return format!("{}{}", value, "0".repeat((-scale) as usize));
        }

        let negative = value < 0;
        let digits = value.abs().to_string();
        let scale = scale as usize;
        let decimal = if digits.len() <= scale {
            format!("0.{}{}", "0".repeat(scale - digits.len()), digits)
        } else {
            let split = digits.len() - scale;
            format!("{}.{}", &digits[..split], &digits[split..])
        };

        if negative {
            format!("-{}", decimal)
        } else {
            decimal
        }
    }

    fn metadata_props(array: &ArrayRef, row: usize) -> Result<Vec<(String, ProximaValue)>> {
        if array.is_null(row) {
            return Ok(Vec::new());
        }

        if let Some(string_array) = array.as_any().downcast_ref::<StringArray>() {
            let json: serde_json::Value = serde_json::from_str(string_array.value(row))?;
            if let serde_json::Value::Object(map) = json {
                return Ok(map
                    .iter()
                    .map(|(key, value)| (key.clone(), json_to_proxima(value)))
                    .collect());
            }
            return Ok(Vec::new());
        }

        if let Some(struct_array) = array.as_any().downcast_ref::<StructArray>() {
            let key_array = struct_array
                .column_by_name("key")
                .context("Missing 'key' field in metadata struct")?
                .as_any()
                .downcast_ref::<StringArray>()
                .context("'key' field not StringArray")?;
            let value_array = struct_array
                .column_by_name("value")
                .context("Missing 'value' field in metadata struct")?
                .as_any()
                .downcast_ref::<StringArray>()
                .context("'value' field not StringArray")?;
            if key_array.is_null(row) || value_array.is_null(row) {
                return Ok(Vec::new());
            }
            return Ok(vec![(
                key_array.value(row).to_string(),
                ProximaValue::String(value_array.value(row).to_string()),
            )]);
        }

        Ok(Vec::new())
    }

    /// Extract vectors from Arrow array (handles multiple formats)
    fn extract_vectors(array: &ArrayRef, num_rows: usize) -> Result<Vec<Vec<f32>>> {
        // Try FixedSizeList<Float32> first (standard format)
        if let Some(list_array) = array.as_any().downcast_ref::<FixedSizeListArray>() {
            let value_length = list_array.value_length() as usize;
            if let Some(flat_array) = list_array.values().as_any().downcast_ref::<Float32Array>() {
                return Ok((0..num_rows)
                    .map(|i| {
                        let start = i * value_length;
                        flat_array.values()[start..start + value_length].to_vec()
                    })
                    .collect());
            }
            if let Some(flat_array) = list_array.values().as_any().downcast_ref::<Float64Array>() {
                return Ok((0..num_rows)
                    .map(|i| {
                        let start = i * value_length;
                        flat_array.values()[start..start + value_length]
                            .iter()
                            .map(|value| *value as f32)
                            .collect()
                    })
                    .collect());
            }
            return Err(anyhow::anyhow!(
                "FixedSizeList values are not Float32/Float64"
            ));
        }

        if let Some(list_array) = array.as_any().downcast_ref::<ListArray>() {
            if let Some(values) = list_array.values().as_any().downcast_ref::<Float32Array>() {
                return Ok((0..num_rows)
                    .map(|i| {
                        let start = list_array.value_offsets()[i] as usize;
                        let end = list_array.value_offsets()[i + 1] as usize;
                        values.values()[start..end].to_vec()
                    })
                    .collect());
            }
            if let Some(values) = list_array.values().as_any().downcast_ref::<Float64Array>() {
                return Ok((0..num_rows)
                    .map(|i| {
                        let start = list_array.value_offsets()[i] as usize;
                        let end = list_array.value_offsets()[i + 1] as usize;
                        values.values()[start..end]
                            .iter()
                            .map(|value| *value as f32)
                            .collect()
                    })
                    .collect());
            }
            return Err(anyhow::anyhow!("List values are not Float32/Float64"));
        }

        // Fallback: Binary format (serialized f32 vectors)
        if let Some(binary_array) = array.as_any().downcast_ref::<BinaryArray>() {
            return (0..num_rows)
                .map(|i| {
                    let bytes = binary_array.value(i);
                    Self::deserialize_vector(bytes)
                })
                .collect();
        }

        // Fallback: Direct Float32Array (for 1D vectors)
        if let Some(float_array) = array.as_any().downcast_ref::<Float32Array>() {
            let dim = float_array.len() / num_rows;
            return Ok((0..num_rows)
                .map(|i| float_array.values()[i * dim..(i + 1) * dim].to_vec())
                .collect());
        }

        Err(anyhow::anyhow!(
            "Unsupported vector column type: {:?}",
            array.data_type()
        ))
    }

    /// Deserialize f32 vector from binary (little-endian)
    fn deserialize_vector(bytes: &[u8]) -> Result<Vec<f32>> {
        if !bytes.len().is_multiple_of(4) {
            return Err(anyhow::anyhow!("Invalid vector binary data length"));
        }

        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| {
                let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
                f32::from_le_bytes(arr)
            })
            .collect())
    }

    /// Convert ProximaRecords to Arrow RecordBatch (for write/insert operations)
    pub fn vector_records_to_batch(
        records: Vec<ProximaRecord>,
        dimension: usize,
    ) -> Result<RecordBatch> {
        if records.is_empty() {
            return Err(anyhow::anyhow!("Cannot create batch from empty records"));
        }

        let schema = Self::create_vector_schema(dimension);

        let id_array = StringArray::from_iter_values(records.iter().map(|r| r.oid.as_str()));

        let mut vector_values = Vec::with_capacity(records.len() * dimension);
        for record in &records {
            let vals = record
                .embeddings
                .first()
                .map(|e| e.as_fp32_slice())
                .unwrap_or(&[]);
            vector_values.extend_from_slice(vals);
            if vals.len() < dimension {
                vector_values.extend(std::iter::repeat_n(0.0f32, dimension - vals.len()));
            }
        }
        let flat_array = Arc::new(Float32Array::from(vector_values)) as ArrayRef;
        let vector_field = Arc::new(Field::new("item", DataType::Float32, false));
        let vector_array =
            FixedSizeListArray::new(vector_field, dimension as i32, flat_array, None);

        let metadata_array = Self::build_metadata_struct_array(&records)?;

        let timestamp_array = Int64Array::from(
            records
                .iter()
                .map(|r| {
                    if r.created_at_ns == 0 {
                        None
                    } else {
                        Some(r.created_at_ns / 1_000_000)
                    }
                })
                .collect::<Vec<Option<i64>>>(),
        );

        let score_array = Float32Array::from(vec![None; records.len()]);

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(id_array),
                Arc::new(vector_array),
                Arc::new(metadata_array),
                Arc::new(timestamp_array),
                Arc::new(score_array),
            ],
        )
        .context("Failed to create RecordBatch")
    }

    /// Build StructArray for metadata from ProximaRecord props
    fn build_metadata_struct_array(records: &[ProximaRecord]) -> Result<StructArray> {
        let mut keys: Vec<Option<String>> = Vec::new();
        let mut values: Vec<Option<String>> = Vec::new();

        for record in records {
            if record.props.is_empty() {
                keys.push(None);
                values.push(None);
            } else if let Some((key, node)) = record.props.iter().next() {
                let val_str = match node {
                    ProximaTreeNode::Value(pv) => Some(format!("{pv:?}")),
                    _ => None,
                };
                keys.push(Some(key.clone()));
                values.push(val_str);
            } else {
                keys.push(None);
                values.push(None);
            }
        }

        let key_array = StringArray::from(keys.iter().map(|k| k.as_deref()).collect::<Vec<_>>());
        let value_array =
            StringArray::from(values.iter().map(|v| v.as_deref()).collect::<Vec<_>>());

        Ok(StructArray::from(vec![
            (
                Arc::new(Field::new("key", DataType::Utf8, true)),
                Arc::new(key_array) as ArrayRef,
            ),
            (
                Arc::new(Field::new("value", DataType::Utf8, true)),
                Arc::new(value_array) as ArrayRef,
            ),
        ]))
    }

    /// Convert protocol search results to an Arrow RecordBatch.
    ///
    /// Search still receives a v1 protocol result shape at this edge, but the
    /// Arrow handler must not rehydrate the legacy vector-only envelope as an
    /// internal representation.
    pub fn search_results_to_batch(
        results: &[crate::proto::proximadb_v1::SearchVectorRecord],
        dimension: usize,
    ) -> Result<RecordBatch> {
        if results.is_empty() {
            return Err(anyhow::anyhow!(
                "Cannot create batch from empty search results"
            ));
        }

        // Search responses omit the vector payload unless the request set
        // `include_fields.vector = true`, so `result.vector` is commonly empty.
        // The shared write schema marks `vector` non-nullable, but a result row
        // with no vector must serialize as a NULL list — otherwise the
        // `FixedSizeList` child buffer (which must hold exactly `len*dimension`
        // floats) underflows and `RecordBatch::try_new` fails, crashing the
        // Flight DoGet search *response*. Derive a search-specific schema with a
        // nullable `vector` field and emit per-row validity.
        let schema = {
            let base = Self::create_vector_schema(dimension);
            let fields: Vec<Field> = base
                .fields()
                .iter()
                .map(|f| {
                    if f.name() == "vector" {
                        f.as_ref().clone().with_nullable(true)
                    } else {
                        f.as_ref().clone()
                    }
                })
                .collect();
            Arc::new(Schema::new(fields))
        };

        let id_array = StringArray::from_iter_values(results.iter().map(|r| r.id.as_str()));

        let mut vector_values = Vec::with_capacity(results.len() * dimension);
        let mut vector_present = Vec::with_capacity(results.len());
        for result in results {
            if result.vector.len() == dimension {
                vector_values.extend_from_slice(&result.vector);
                vector_present.push(true);
            } else {
                // No (or mismatched) vector for this row: emit a NULL list but
                // still reserve `dimension` backing slots to keep the child
                // buffer length at `len*dimension`.
                vector_values.extend(std::iter::repeat_n(0.0f32, dimension));
                vector_present.push(false);
            }
        }
        let flat_array = Arc::new(Float32Array::from(vector_values)) as ArrayRef;
        let vector_field = Arc::new(Field::new("item", DataType::Float32, false));
        let vector_nulls = vector_present
            .iter()
            .any(|present| !present)
            .then(|| arrow_buffer::NullBuffer::from(vector_present));
        let vector_array =
            FixedSizeListArray::new(vector_field, dimension as i32, flat_array, vector_nulls);

        let mut meta_keys: Vec<Option<&str>> = Vec::new();
        let mut meta_vals: Vec<Option<String>> = Vec::new();
        for result in results {
            if result.metadata.is_empty() {
                meta_keys.push(None);
                meta_vals.push(None);
            } else if let Some((key, value)) = result.metadata.iter().next() {
                meta_keys.push(Some(key.as_str()));
                meta_vals.push(Some(Self::sql_value_to_string(value)));
            } else {
                meta_keys.push(None);
                meta_vals.push(None);
            }
        }
        let meta_key_array = StringArray::from(meta_keys);
        let meta_val_array =
            StringArray::from(meta_vals.iter().map(|v| v.as_deref()).collect::<Vec<_>>());
        let metadata_array = StructArray::from(vec![
            (
                Arc::new(Field::new("key", DataType::Utf8, true)),
                Arc::new(meta_key_array) as ArrayRef,
            ),
            (
                Arc::new(Field::new("value", DataType::Utf8, true)),
                Arc::new(meta_val_array) as ArrayRef,
            ),
        ]);

        let timestamp_array = Int64Array::from(
            results
                .iter()
                .map(|r| r.timestamp)
                .collect::<Vec<Option<i64>>>(),
        );

        let score_array = Float32Array::from(
            results
                .iter()
                .map(|r| Some(r.score as f32))
                .collect::<Vec<Option<f32>>>(),
        );

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(id_array),
                Arc::new(vector_array),
                Arc::new(metadata_array),
                Arc::new(timestamp_array),
                Arc::new(score_array),
            ],
        )
        .context("Failed to create RecordBatch from search results")
    }

    /// Encode canonical v2 search results (`RichSearchResult`) into a Flight
    /// `RecordBatch` with full-fidelity props (TD-FLIGHT-1). Replaces the v1
    /// `search_results_to_batch`, which collapsed each row's props to a single
    /// key/value pair.
    ///
    /// Schema (the canonical v2 Flight search response contract):
    /// - `id` Utf8 (non-null)
    /// - `score` Float32 (nullable)
    /// - `vector` nullable `FixedSizeList<Float32>` — null per row when the
    ///   request set `include_vector=false`; the child buffer still reserves
    ///   `dimension` slots per row (same constraint as `search_results_to_batch`)
    /// - `properties` nullable Utf8 — the FULL `props` map serialized as one
    ///   JSON object per row; round-trips losslessly via `metadata_props`' JSON
    ///   branch (codec.rs `metadata_props`)
    /// - `timestamp` Int64 (nullable), `version` UInt32 (nullable), `source`
    ///   Utf8 (nullable)
    ///
    /// Empty results yield a zero-row batch of this schema (no panic).
    pub fn rich_search_results_to_batch(
        results: &[RichSearchResult],
        include_vector: bool,
    ) -> Result<RecordBatch> {
        let dimension = if include_vector {
            results.iter().map(|r| r.vector.len()).max().unwrap_or(0)
        } else {
            0
        };

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("score", DataType::Float32, true),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, false)),
                    dimension as i32,
                ),
                true,
            ),
            Field::new("properties", DataType::Utf8, true),
            Field::new("timestamp", DataType::Int64, true),
            Field::new("version", DataType::UInt32, true),
            Field::new("source", DataType::Utf8, true),
        ]));

        let id_array = StringArray::from_iter_values(results.iter().map(|r| r.id.as_str()));
        let score_array = Float32Array::from(
            results
                .iter()
                .map(|r| Some(r.score as f32))
                .collect::<Vec<_>>(),
        );

        // Vector column: per-row validity, reserving `dimension` slots per row
        // even when the row carries no vector (keeps the child buffer length at
        // len*dimension — see `search_results_to_batch` for the same constraint).
        let mut vector_values = Vec::with_capacity(results.len() * dimension);
        let mut vector_present = Vec::with_capacity(results.len());
        for r in results {
            if include_vector && dimension > 0 && r.vector.len() == dimension {
                vector_values.extend_from_slice(&r.vector);
                vector_present.push(true);
            } else {
                vector_values.extend(std::iter::repeat_n(0.0f32, dimension));
                vector_present.push(false);
            }
        }
        let flat_array = Arc::new(Float32Array::from(vector_values)) as ArrayRef;
        let vector_field = Arc::new(Field::new("item", DataType::Float32, false));
        let vector_nulls = vector_present
            .iter()
            .any(|present| !present)
            .then(|| arrow_buffer::NullBuffer::from(vector_present));
        let vector_array =
            FixedSizeListArray::new(vector_field, dimension as i32, flat_array, vector_nulls);

        let properties_array = StringArray::from(
            results
                .iter()
                .map(|r| {
                    if r.props.is_empty() {
                        None
                    } else {
                        serde_json::to_string(&r.props).ok()
                    }
                })
                .collect::<Vec<_>>(),
        );
        let timestamp_array =
            Int64Array::from(results.iter().map(|r| r.timestamp).collect::<Vec<_>>());
        let version_array =
            UInt32Array::from(results.iter().map(|r| r.version).collect::<Vec<_>>());
        let source_array = StringArray::from(
            results
                .iter()
                .map(|r| r.source.as_deref())
                .collect::<Vec<_>>(),
        );

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(id_array),
                Arc::new(score_array),
                Arc::new(vector_array),
                Arc::new(properties_array),
                Arc::new(timestamp_array),
                Arc::new(version_array),
                Arc::new(source_array),
            ],
        )
        .context("Failed to create RecordBatch from v2 search results")
    }

    /// Convert SqlValue to string
    fn sql_value_to_string(value: &SqlValue) -> String {
        use crate::proto::proximadb_v1::sql_value::Value;

        match &value.value {
            Some(Value::StringValue(s)) => s.clone(),
            Some(Value::Int64Value(i)) => i.to_string(),
            Some(Value::NumberValue(f)) => f.to_string(),
            Some(Value::BoolValue(b)) => b.to_string(),
            Some(Value::NullValue(_)) => "null".to_string(),
            Some(Value::BytesValue(b)) => format!("{:?}", b),
            // TD-PROTO-2: canonical compact JSON (same representation as
            // REST), not a Rust Debug byte dump.
            Some(Value::JsonbValue(b)) => {
                proximadb_data_model::ProximaValue::jsonb_to_json_string_lossy(b)
            }
            Some(Value::ArrayValue(_)) => "[array]".to_string(),
            Some(Value::ObjectValue(_)) => "{object}".to_string(),
            None => "null".to_string(),
        }
    }

    /// Parse FlightDescriptor to extract request metadata
    pub fn parse_descriptor(descriptor: &FlightDescriptor) -> Result<FlightRequestMetadata> {
        let path_operation = descriptor
            .path
            .first()
            .and_then(|first_path| FlightWriteOperation::from_token(first_path));

        let path_collection_id = match (descriptor.path.first(), path_operation) {
            (Some(_), Some(_)) => Some(
                descriptor
                    .path
                    .get(1)
                    .context("FlightDescriptor operation path is missing collection id")?
                    .clone(),
            ),
            (Some(first_path), None) => Some(first_path.clone()),
            (None, None) => None,
            (None, Some(_)) => unreachable!("path operation requires a first path segment"),
        };

        let (collection_id, operation, write_mode, trigger_compaction) =
            if !descriptor.cmd.is_empty() {
                let cmd: HashMap<String, serde_json::Value> =
                    serde_json::from_slice(&descriptor.cmd)?;
                let collection_id = cmd
                    .get("collection_id")
                    .or_else(|| cmd.get("collection"))
                    .and_then(Self::json_string)
                    .or(path_collection_id)
                    .context("FlightDescriptor is missing collection id")?;
                let operation = cmd
                    .get("operation")
                    .or_else(|| cmd.get("write_operation"))
                    .and_then(Self::json_string)
                    .and_then(|operation| FlightWriteOperation::from_token(&operation))
                    .or(path_operation)
                    .unwrap_or_default();
                let mode = match cmd.get("write_mode").and_then(Self::json_string).as_deref() {
                    Some("direct") => WriteMode::Direct,
                    _ => WriteMode::WAL,
                };
                let compact = cmd
                    .get("trigger_compaction")
                    .and_then(Self::json_bool)
                    .unwrap_or(false);
                (collection_id, operation, mode, compact)
            } else {
                (
                    path_collection_id.context("FlightDescriptor is missing collection id")?,
                    path_operation.unwrap_or_default(),
                    WriteMode::WAL,
                    false,
                )
            };

        Ok(FlightRequestMetadata {
            collection_id,
            operation,
            write_mode,
            trigger_compaction,
        })
    }

    fn json_string(value: &serde_json::Value) -> Option<String> {
        value.as_str().map(ToOwned::to_owned)
    }

    fn json_bool(value: &serde_json::Value) -> Option<bool> {
        value
            .as_bool()
            .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
    }

    /// Parse Ticket to extract search request
    pub fn ticket_to_search_request(ticket: &Ticket) -> Result<VectorSearchRequest> {
        serde_json::from_slice(&ticket.ticket).context("Failed to parse search request from ticket")
    }

    /// Convert FlightData to RecordBatch
    pub fn flight_data_to_batch(data: &FlightData) -> Result<RecordBatch> {
        arrow_flight::utils::flight_data_to_arrow_batch(
            data,
            Arc::new(Schema::empty()), // Schema sent separately
            &Default::default(),
        )
        .context("Failed to convert FlightData to RecordBatch")
    }

    /// Convert a FlightData stream containing an IPC schema followed by record
    /// batch messages into Arrow RecordBatches.
    ///
    /// PyArrow DoPut/DoExchange clients commonly send a descriptor-only first
    /// message and then a schema message before data batches. Descriptor-only
    /// messages have an empty `data_header` and are ignored here.
    pub fn flight_data_stream_to_batches(messages: &[FlightData]) -> Result<Vec<RecordBatch>> {
        let arrow_messages: Vec<FlightData> = messages
            .iter()
            .filter(|message| !message.data_header.is_empty())
            .cloned()
            .collect();

        if arrow_messages.is_empty() {
            return Ok(Vec::new());
        }

        arrow_flight::utils::flight_data_to_batches(&arrow_messages)
            .context("Failed to convert FlightData stream to RecordBatches")
    }

    /// Convert RecordBatch to FlightData
    pub fn batch_to_flight_data(
        batch: &RecordBatch,
        _options: &IpcWriteOptions,
    ) -> Result<Vec<FlightData>> {
        Ok(arrow_flight::utils::batches_to_flight_data(
            &batch.schema(),
            vec![batch.clone()],
        )?)
    }

    /// Convert RecordBatch to FlightData with compression support
    ///
    /// This method encodes a RecordBatch into Arrow IPC FlightData format,
    /// applying the specified compression codec if provided.
    ///
    /// ## Parameters
    /// - `batch`: The RecordBatch to encode
    /// - `compression`: Optional compression type (LZ4_FRAME or ZSTD)
    ///
    /// ## Returns
    /// A vector of FlightData messages including schema and data
    ///
    /// ## Example
    /// ```rust,ignore
    /// use arrow_ipc::CompressionType;
    ///
    /// let flight_data = ArrowProtoCodec::batch_to_flight_data_with_compression(
    ///     &batch,
    ///     Some(CompressionType::LZ4_FRAME),
    /// )?;
    /// ```
    pub fn batch_to_flight_data_with_compression(
        batch: &RecordBatch,
        compression: Option<arrow_ipc::CompressionType>,
    ) -> Result<Vec<FlightData>> {
        use arrow_ipc::writer::{DictionaryTracker, IpcDataGenerator, IpcWriteContext};

        let schema = batch.schema();

        // Create write options with compression if specified
        let options = match compression {
            Some(codec) => IpcWriteOptions::default()
                .try_with_compression(Some(codec))
                .context("Failed to configure compression")?,
            None => IpcWriteOptions::default(),
        };

        // Create schema message
        let schema_flight_data: FlightData =
            arrow_flight::SchemaAsIpc::new(&schema, &options).into();

        // Encode the batch with compression
        let data_gen = IpcDataGenerator::default();
        let mut dictionary_tracker = DictionaryTracker::new(false);
        let mut compression_context = IpcWriteContext::default();

        let (encoded_dictionaries, encoded_batch) = data_gen.encode(
            batch,
            &mut dictionary_tracker,
            &options,
            &mut compression_context,
        )?;

        // Build the result: schema + dictionaries + data
        let mut stream = Vec::with_capacity(1 + encoded_dictionaries.len() + 1);
        stream.push(schema_flight_data);
        stream.extend(encoded_dictionaries.into_iter().map(Into::into));
        stream.push(encoded_batch.into());

        Ok(stream)
    }

    /// Convert multiple RecordBatches to FlightData with compression support
    ///
    /// This is a convenience method for encoding multiple batches with the same
    /// compression settings. The schema is sent only once at the beginning.
    pub fn batches_to_flight_data_with_compression(
        batches: &[RecordBatch],
        compression: Option<arrow_ipc::CompressionType>,
    ) -> Result<Vec<FlightData>> {
        if batches.is_empty() {
            return Ok(Vec::new());
        }

        use arrow_ipc::writer::{DictionaryTracker, IpcDataGenerator, IpcWriteContext};

        let schema = batches[0].schema();

        // Create write options with compression if specified
        let options = match compression {
            Some(codec) => IpcWriteOptions::default()
                .try_with_compression(Some(codec))
                .context("Failed to configure compression")?,
            None => IpcWriteOptions::default(),
        };

        // Create schema message
        let schema_flight_data: FlightData =
            arrow_flight::SchemaAsIpc::new(&schema, &options).into();

        let data_gen = IpcDataGenerator::default();
        let mut dictionary_tracker = DictionaryTracker::new(false);
        let mut compression_context = IpcWriteContext::default();

        // Estimate capacity
        let mut stream = Vec::with_capacity(1 + batches.len() * 2);
        stream.push(schema_flight_data);

        // Encode each batch
        for batch in batches {
            let (encoded_dictionaries, encoded_batch) = data_gen.encode(
                batch,
                &mut dictionary_tracker,
                &options,
                &mut compression_context,
            )?;

            stream.extend(encoded_dictionaries.into_iter().map(Into::into));
            stream.push(encoded_batch.into());
        }

        Ok(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{
        ArrayRef, BooleanArray, Date32Array, Decimal128Array, FixedSizeListArray, Float32Array,
        Float64Array, Int8Array, Int64Array, LargeBinaryArray, LargeStringArray, ListArray,
        StringArray, Time64MicrosecondArray, TimestampNanosecondArray, UInt64Array,
        types::Float32Type,
    };
    use arrow_schema::{DataType, Field};

    #[test]
    fn test_create_vector_schema() {
        let schema = ArrowProtoCodec::create_vector_schema(384);
        assert_eq!(schema.fields().len(), 5);
        assert_eq!(schema.field(0).name(), "id");
        assert_eq!(schema.field(1).name(), "vector");
    }

    // ---- TD-FLIGHT-1: canonical v2 ticket + encoder ----

    fn mk_ticket(json: serde_json::Value) -> Ticket {
        Ticket {
            ticket: serde_json::to_vec(&json).expect("serialize ticket").into(),
        }
    }

    fn mk_result(id: &str, score: f64, props: Vec<(&str, ProximaValue)>) -> RichSearchResult {
        RichSearchResult {
            id: id.to_string(),
            score,
            similarity: None,
            vector: vec![],
            props: props.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
            version: None,
            timestamp: None,
            source: None,
        }
    }

    #[test]
    fn test_flight_search_ticket_discriminator() {
        let v2 = mk_ticket(serde_json::json!({
            "type": "vector_search",
            "collection_id": "c1",
            "query_vector": [0.1, 0.2],
            "top_k": 5
        }));
        let parsed = FlightSearchTicket::from_ticket(&v2).expect("v2 ticket parses");
        assert_eq!(parsed.collection_id, "c1");
        assert_eq!(parsed.query_vector, vec![0.1_f32, 0.2]);
        assert_eq!(parsed.top_k, 5);
        assert!(!parsed.include_vector);
        assert!(parsed.filters.is_empty());

        // Non-v2 tickets return None rather than mis-parsing.
        let file_ticket = mk_ticket(serde_json::json!({
            "type": "arrow_file", "collection_id": "c1", "file_path": "/x"
        }));
        assert!(FlightSearchTicket::from_ticket(&file_ticket).is_none());

        // Old v1 VectorSearchRequest shape (no "type") is rejected.
        let v1 = mk_ticket(serde_json::json!({
            "collection_id": "c1", "queries": [], "top_k": 3
        }));
        assert!(FlightSearchTicket::from_ticket(&v1).is_none());
    }

    #[test]
    fn test_flight_search_ticket_to_rich_request() {
        let ticket = FlightSearchTicket {
            ticket_type: "vector_search".to_string(),
            collection_id: "c1".to_string(),
            query_vector: vec![1.0, 2.0, 3.0],
            top_k: 7,
            filters: vec![
                FlightFilter {
                    field: "year".into(),
                    op: "gte".into(),
                    value: serde_json::json!(2020),
                    value_upper: None,
                    value_list: vec![],
                },
                FlightFilter {
                    field: "tag".into(),
                    op: "in".into(),
                    value: serde_json::json!(null),
                    value_upper: None,
                    value_list: vec![serde_json::json!("a"), serde_json::json!("b")],
                },
                FlightFilter {
                    field: "price".into(),
                    op: "between".into(),
                    value: serde_json::json!(10),
                    value_upper: Some(serde_json::json!(100)),
                    value_list: vec![],
                },
            ],
            include_vector: true,
        };
        let req = ticket.to_rich_request().expect("translate");
        assert_eq!(req.collection_id, "c1");
        assert_eq!(req.top_k, 7);
        assert_eq!(req.filters.len(), 3);
        assert_eq!(req.filters[0].operator, RichFilterOperator::Gte);
        assert_eq!(req.filters[1].operator, RichFilterOperator::In);
        assert_eq!(req.filters[1].value_list.len(), 2);
        assert_eq!(req.filters[2].operator, RichFilterOperator::Between);
        assert!(req.filters[2].value_upper.is_some());

        // Unsupported operator is rejected, not silently coerced.
        let bad = FlightFilter {
            field: "x".into(),
            op: "weird".into(),
            value: serde_json::json!(1),
            value_upper: None,
            value_list: vec![],
        };
        let bad_ticket = FlightSearchTicket {
            ticket_type: "vector_search".into(),
            collection_id: "c".into(),
            query_vector: vec![],
            top_k: 1,
            filters: vec![bad],
            include_vector: false,
        };
        assert!(bad_ticket.to_rich_request().is_err());
    }

    #[test]
    fn test_rich_search_results_to_batch_empty() {
        let batch =
            ArrowProtoCodec::rich_search_results_to_batch(&[], false).expect("empty encodes");
        assert_eq!(batch.num_rows(), 0);
        // The canonical v2 schema (7 typed columns), including the new
        // full-fidelity `properties` column.
        let field_names: Vec<String> = batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().to_string())
            .collect();
        assert_eq!(
            field_names,
            vec![
                "id".to_string(),
                "score".to_string(),
                "vector".to_string(),
                "properties".to_string(),
                "timestamp".to_string(),
                "version".to_string(),
                "source".to_string(),
            ]
        );
    }

    #[test]
    fn test_rich_search_results_to_batch_preserves_full_props() {
        let props = vec![
            ("category", ProximaValue::String("books".into())),
            ("year", ProximaValue::Int64(2024)),
        ];
        let results = vec![mk_result("r1", 0.91, props)];
        let batch = ArrowProtoCodec::rich_search_results_to_batch(&results, false).expect("encode");
        assert_eq!(batch.num_rows(), 1);
        let props_col = batch.column_by_name("properties").expect("properties col");
        let sa = props_col
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("properties is StringArray");
        // The full map round-trips as JSON — BOTH keys survive (the v1 encoder
        // collapsed props to a single key/value, dropping the second).
        let json: serde_json::Value = serde_json::from_str(sa.value(0)).expect("valid json");
        assert!(json.get("category").is_some());
        assert!(json.get("year").is_some());
    }

    #[test]
    fn test_rich_search_results_to_batch_vector_nullable() {
        let mut r1 = mk_result("r1", 1.0, vec![]);
        r1.vector = vec![0.1, 0.2, 0.3];
        let r2 = mk_result("r2", 0.5, vec![]);
        let batch = ArrowProtoCodec::rich_search_results_to_batch(&[r1, r2], true).expect("encode");
        let vcol = batch.column_by_name("vector").expect("vector col");
        assert_eq!(vcol.null_count(), 1, "second row has no vector => null");
        assert!(!vcol.is_null(0));
        assert!(vcol.is_null(1));
    }

    /// Regression: a search response with NO vector payload (the common case
    /// when `include_fields.vector` is unset) must still encode — the vector
    /// column becomes an all-null FixedSizeList rather than underflowing the
    /// child buffer and failing `RecordBatch::try_new` (which crashed Flight
    /// DoGet search responses).
    #[test]
    fn test_search_results_to_batch_without_vectors_builds_null_column() {
        use crate::proto::proximadb_v1::SearchVectorRecord;
        let results = vec![
            SearchVectorRecord {
                id: "a".to_string(),
                score: 1.0,
                ..Default::default()
            },
            SearchVectorRecord {
                id: "b".to_string(),
                score: 0.5,
                ..Default::default()
            },
        ];
        let batch = ArrowProtoCodec::search_results_to_batch(&results, 384)
            .expect("must encode search results that carry no vectors");
        assert_eq!(batch.num_rows(), 2);
        let vector_col = batch.column_by_name("vector").expect("vector column");
        assert_eq!(vector_col.null_count(), 2, "both rows have a null vector");
    }

    /// A search response carrying vectors on some rows but not others must
    /// preserve the present vectors and null only the absent rows.
    #[test]
    fn test_search_results_to_batch_mixed_vectors() {
        use crate::proto::proximadb_v1::SearchVectorRecord;
        let results = vec![
            SearchVectorRecord {
                id: "a".to_string(),
                score: 1.0,
                vector: vec![1.0, 2.0, 3.0],
                ..Default::default()
            },
            SearchVectorRecord {
                id: "b".to_string(),
                score: 0.5,
                ..Default::default()
            },
        ];
        let batch = ArrowProtoCodec::search_results_to_batch(&results, 3)
            .expect("must encode mixed-vector search results");
        assert_eq!(batch.num_rows(), 2);
        let vector_col = batch.column_by_name("vector").expect("vector column");
        assert_eq!(vector_col.null_count(), 1, "only the second row is null");
        let lists = vector_col
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .expect("FixedSizeList");
        assert!(lists.is_valid(0));
        assert!(!lists.is_valid(1));
    }

    #[test]
    fn test_deserialize_vector() {
        let vec = vec![1.0f32, 2.0, 3.0];
        let mut bytes = Vec::new();
        for v in &vec {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        let result = ArrowProtoCodec::deserialize_vector(&bytes).unwrap();
        assert_eq!(result, vec);
    }

    /// Helper to create a test RecordBatch
    fn create_test_batch(num_rows: usize, dimension: i32) -> RecordBatch {
        // Create id column
        let ids: Vec<String> = (0..num_rows).map(|i| format!("id_{}", i)).collect();
        let id_array = StringArray::from(ids);

        // Create vector column
        let flat_values: Vec<f32> = (0..(num_rows * dimension as usize))
            .map(|i| (i as f32) * 0.1)
            .collect();
        let values_array = Arc::new(Float32Array::from(flat_values)) as ArrayRef;
        let vector_field = Arc::new(Field::new("item", DataType::Float32, false));
        let vector_array = FixedSizeListArray::new(vector_field, dimension, values_array, None);

        // Create timestamp column
        let timestamps: Vec<i64> = (0..num_rows).map(|i| i as i64 * 1000).collect();
        let timestamp_array = Int64Array::from(timestamps);

        // Create schema
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, false)),
                    dimension,
                ),
                false,
            ),
            Field::new("timestamp", DataType::Int64, true),
        ]));

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(id_array),
                Arc::new(vector_array),
                Arc::new(timestamp_array),
            ],
        )
        .expect("Failed to create test batch")
    }

    #[test]
    fn test_batches_to_proxima_records_preserves_rich_arrow_values() {
        let ids = StringArray::from(vec!["doc_1"]);
        let flat_values = Arc::new(Float32Array::from(vec![0.1, 0.2, 0.3])) as ArrayRef;
        let vector_array = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            3,
            flat_values,
            None,
        );
        let title = StringArray::from(vec!["Quarterly report"]);
        let views = Int64Array::from(vec![42]);
        let published = BooleanArray::from(vec![true]);
        let metadata = StringArray::from(vec![r#"{"category":"finance","score":9.5}"#]);
        let source_id = StringArray::from(vec!["node_a"]);
        let target_id = StringArray::from(vec!["node_b"]);
        let edge_type = StringArray::from(vec!["cites"]);
        let weight = Float64Array::from(vec![0.75]);

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 3),
                false,
            ),
            Field::new("title", DataType::Utf8, false),
            Field::new("views", DataType::Int64, false),
            Field::new("published", DataType::Boolean, false),
            Field::new("metadata", DataType::Utf8, true),
            Field::new("source_id", DataType::Utf8, false),
            Field::new("target_id", DataType::Utf8, false),
            Field::new("edge_type", DataType::Utf8, false),
            Field::new("weight", DataType::Float64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(ids),
                Arc::new(vector_array),
                Arc::new(title),
                Arc::new(views),
                Arc::new(published),
                Arc::new(metadata),
                Arc::new(source_id),
                Arc::new(target_id),
                Arc::new(edge_type),
                Arc::new(weight),
            ],
        )
        .unwrap();

        let records = ArrowProtoCodec::batches_to_proxima_records(vec![batch]).unwrap();
        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert_eq!(record.oid, "doc_1");
        assert_eq!(
            record.embeddings[0].values,
            proximadb_records::EmbeddingValues::Fp32(vec![0.1, 0.2, 0.3])
        );
        assert_eq!(
            record.props.get("title"),
            Some(&ProximaTreeNode::Value(ProximaValue::String(
                "Quarterly report".to_string()
            )))
        );
        assert_eq!(
            record.props.get("views"),
            Some(&ProximaTreeNode::Value(ProximaValue::Int64(42)))
        );
        assert_eq!(
            record.props.get("published"),
            Some(&ProximaTreeNode::Value(ProximaValue::Boolean(true)))
        );
        assert_eq!(
            record.props.get("category"),
            Some(&ProximaTreeNode::Value(ProximaValue::String(
                "finance".to_string()
            )))
        );
        assert_eq!(
            record.edge.as_ref().map(|edge| edge.edge_type.as_str()),
            Some("cites")
        );
    }

    #[test]
    fn test_batches_to_proxima_records_preserves_extended_arrow_types() {
        let ids = StringArray::from(vec!["rich_1"]);
        let vector_array = ListArray::from_iter_primitive::<Float32Type, _, _>(vec![Some(vec![
            Some(1.0),
            Some(2.0),
            Some(3.5),
        ])]);
        let small_int = Int8Array::from(vec![-8]);
        let unsigned_big = UInt64Array::from(vec![9_223_372_036_854_775_808_u64]);
        let large_text = LargeStringArray::from(vec!["large text"]);
        let large_binary = LargeBinaryArray::from_vec(vec![b"bytes".as_slice()]);
        let decimal = Decimal128Array::from(vec![12345_i128])
            .with_precision_and_scale(10, 2)
            .unwrap();
        let date = Date32Array::from(vec![19_000]);
        let time = Time64MicrosecondArray::from(vec![86_400_000_000_i64]);
        let timestamp = TimestampNanosecondArray::from(vec![1_700_000_000_000_000_000_i64]);

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                false,
            ),
            Field::new("small_int", DataType::Int8, false),
            Field::new("unsigned_big", DataType::UInt64, false),
            Field::new("large_text", DataType::LargeUtf8, false),
            Field::new("large_binary", DataType::LargeBinary, false),
            Field::new("decimal", DataType::Decimal128(10, 2), false),
            Field::new("date", DataType::Date32, false),
            Field::new(
                "time",
                DataType::Time64(arrow_schema::TimeUnit::Microsecond),
                false,
            ),
            Field::new(
                "event_ts",
                DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, None),
                false,
            ),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(ids),
                Arc::new(vector_array),
                Arc::new(small_int),
                Arc::new(unsigned_big),
                Arc::new(large_text),
                Arc::new(large_binary),
                Arc::new(decimal),
                Arc::new(date),
                Arc::new(time),
                Arc::new(timestamp),
            ],
        )
        .unwrap();

        let records = ArrowProtoCodec::batches_to_proxima_records(vec![batch]).unwrap();
        let record = &records[0];
        assert_eq!(
            record.embeddings[0].values,
            proximadb_records::EmbeddingValues::Fp32(vec![1.0, 2.0, 3.5])
        );
        assert_eq!(
            record.props.get("small_int"),
            Some(&ProximaTreeNode::Value(ProximaValue::Int8(-8)))
        );
        assert_eq!(
            record.props.get("unsigned_big"),
            Some(&ProximaTreeNode::Value(ProximaValue::UInt64(
                9_223_372_036_854_775_808_u64
            )))
        );
        assert_eq!(
            record.props.get("large_text"),
            Some(&ProximaTreeNode::Value(ProximaValue::String(
                "large text".to_string()
            )))
        );
        assert_eq!(
            record.props.get("large_binary"),
            Some(&ProximaTreeNode::Value(ProximaValue::Binary(
                b"bytes".to_vec()
            )))
        );
        assert_eq!(
            record.props.get("decimal"),
            Some(&ProximaTreeNode::Value(ProximaValue::Decimal(
                "123.45".to_string()
            )))
        );
        assert_eq!(
            record.props.get("date"),
            Some(&ProximaTreeNode::Value(ProximaValue::Date(19_000)))
        );
        assert_eq!(
            record.props.get("time"),
            Some(&ProximaTreeNode::Value(ProximaValue::Time(
                86_400_000_000,
                TimeUnit::Microsecond
            )))
        );
        assert_eq!(
            record.props.get("event_ts"),
            Some(&ProximaTreeNode::Value(ProximaValue::Timestamp(
                1_700_000_000_000_000_000,
                TimeUnit::Nanosecond
            )))
        );
    }

    #[test]
    fn test_batches_to_proxima_records_accepts_large_utf8_envelope_columns() {
        let ids = LargeStringArray::from(vec!["rich_large_1"]);
        let tenant_ids = LargeStringArray::from(vec!["tenant-a"]);
        let origins = LargeStringArray::from(vec!["pyarrow"]);
        let source_ids = LargeStringArray::from(vec!["node-a"]);
        let target_ids = LargeStringArray::from(vec!["node-b"]);
        let edge_types = LargeStringArray::from(vec!["depends_on"]);

        let schema = Arc::new(Schema::new(vec![
            Field::new("oid", DataType::LargeUtf8, false),
            Field::new("tenant_id", DataType::LargeUtf8, false),
            Field::new("origin", DataType::LargeUtf8, false),
            Field::new("source_id", DataType::LargeUtf8, false),
            Field::new("target_id", DataType::LargeUtf8, false),
            Field::new("edge_type", DataType::LargeUtf8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(ids),
                Arc::new(tenant_ids),
                Arc::new(origins),
                Arc::new(source_ids),
                Arc::new(target_ids),
                Arc::new(edge_types),
            ],
        )
        .unwrap();

        let records = ArrowProtoCodec::batches_to_proxima_records(vec![batch]).unwrap();
        let record = &records[0];

        assert_eq!(record.oid, "rich_large_1");
        assert_eq!(record.local_id.as_deref(), Some("rich_large_1"));
        assert_eq!(record.tenant_id, "tenant-a");
        assert_eq!(record.origin.as_deref(), Some("pyarrow"));
        assert_eq!(
            record.edge.as_ref().map(|edge| {
                (
                    edge.source_id.as_str(),
                    edge.target_id.as_str(),
                    edge.edge_type.as_str(),
                )
            }),
            Some(("node-a", "node-b", "depends_on"))
        );
    }

    #[test]
    fn test_flight_write_operation_accepts_v2_enum_tokens() {
        assert_eq!(
            FlightWriteOperation::from_token("UPSERT"),
            Some(FlightWriteOperation::Upsert)
        );
        assert_eq!(
            FlightWriteOperation::from_token("BATCH_WRITE_MODE_DELETE"),
            Some(FlightWriteOperation::Delete)
        );
    }

    #[test]
    fn test_batch_to_flight_data_no_compression() {
        let batch = create_test_batch(10, 4);
        let flight_data = ArrowProtoCodec::batch_to_flight_data_with_compression(&batch, None)
            .expect("Failed to convert batch");

        // Should have schema + data messages
        assert!(!flight_data.is_empty());
        // First message should be schema
        assert!(!flight_data[0].data_header.is_empty());
    }

    #[test]
    fn test_batch_to_flight_data_with_lz4_compression() {
        let batch = create_test_batch(100, 128);
        let flight_data = ArrowProtoCodec::batch_to_flight_data_with_compression(
            &batch,
            Some(arrow_ipc::CompressionType::LZ4_FRAME),
        )
        .expect("Failed to convert batch with LZ4");

        assert!(!flight_data.is_empty());
    }

    #[test]
    fn test_batch_to_flight_data_with_zstd_compression() {
        let batch = create_test_batch(100, 128);
        let flight_data = ArrowProtoCodec::batch_to_flight_data_with_compression(
            &batch,
            Some(arrow_ipc::CompressionType::ZSTD),
        )
        .expect("Failed to convert batch with ZSTD");

        assert!(!flight_data.is_empty());
    }

    #[test]
    fn test_batches_to_flight_data_empty() {
        let flight_data = ArrowProtoCodec::batches_to_flight_data_with_compression(&[], None)
            .expect("Failed to convert empty batches");

        assert!(flight_data.is_empty());
    }

    #[test]
    fn test_batches_to_flight_data_multiple_batches() {
        let batch1 = create_test_batch(50, 64);
        let batch2 = create_test_batch(30, 64);
        let batches = vec![batch1, batch2];

        let flight_data = ArrowProtoCodec::batches_to_flight_data_with_compression(
            &batches,
            Some(arrow_ipc::CompressionType::LZ4_FRAME),
        )
        .expect("Failed to convert batches");

        // Should have schema + data for each batch
        assert!(flight_data.len() >= 3); // At least schema + 2 data messages
    }

    #[test]
    fn test_flight_ticket_serialization() {
        // Create a VectorSearchRequest and serialize it into a Ticket
        let search_request = VectorSearchRequest {
            collection_id: "test_collection".to_string(),
            top_k: 10,
            ..Default::default()
        };

        let ticket_bytes =
            serde_json::to_vec(&search_request).expect("Failed to serialize search request");
        let ticket = Ticket {
            ticket: ticket_bytes.into(),
        };

        // Deserialize back
        let parsed = ArrowProtoCodec::ticket_to_search_request(&ticket)
            .expect("Failed to parse ticket back to search request");
        assert_eq!(parsed.collection_id, "test_collection");
        assert_eq!(parsed.top_k, 10);
    }

    #[test]
    fn test_flight_descriptor_creation() {
        // Create a FlightDescriptor with a collection path and parse it
        let descriptor = FlightDescriptor {
            r#type: 0,
            path: vec!["my_collection".to_string()],
            cmd: Default::default(),
        };

        let metadata =
            ArrowProtoCodec::parse_descriptor(&descriptor).expect("Failed to parse descriptor");
        assert_eq!(metadata.collection_id, "my_collection");
        assert_eq!(metadata.operation, FlightWriteOperation::Upsert);
        assert_eq!(metadata.write_mode, WriteMode::WAL);
        assert!(!metadata.trigger_compaction);
    }

    #[test]
    fn test_arrow_batch_to_proxima_records() {
        let batch = create_test_batch(3, 4);

        let records = ArrowProtoCodec::batches_to_proxima_records(vec![batch])
            .expect("Failed to convert batch to ProximaRecords");

        assert_eq!(records.len(), 3);
        assert_eq!(records[0].oid, "id_0");
        assert_eq!(records[1].oid, "id_1");
        assert_eq!(records[2].oid, "id_2");

        assert_eq!(records[0].embeddings[0].values.len(), 4);
        assert_eq!(records[1].embeddings[0].values.len(), 4);
    }

    #[test]
    fn test_arrow_batch_large_utf8_ids() {
        let ids = LargeStringArray::from(vec!["vec_large_1", "vec_large_2"]);
        let flat_values = Arc::new(Float32Array::from(vec![0.1, 0.2, 0.3, 0.4])) as ArrayRef;
        let vector_array = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            2,
            flat_values,
            None,
        );

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::LargeUtf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 2),
                false,
            ),
        ]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(vector_array)]).unwrap();

        let records = ArrowProtoCodec::batches_to_proxima_records(vec![batch])
            .expect("LargeUtf8 ids should decode");

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].oid, "vec_large_1");
        assert_eq!(records[1].oid, "vec_large_2");
        assert_eq!(
            records[1].embeddings[0].values,
            proximadb_records::EmbeddingValues::Fp32(vec![0.3, 0.4])
        );
    }

    #[test]
    fn test_vector_records_to_arrow_batch() {
        let records = vec![
            ProximaRecord {
                oid: "vec_a".to_string(),
                embeddings: vec![EmbeddingCell {
                    model_id: "default".to_string(),
                    modality: "dense_vector".to_string(),
                    values: proximadb_records::EmbeddingValues::Fp32(vec![1.0, 2.0, 3.0]),
                    dim: 3,
                    ..Default::default()
                }],
                created_at_ns: 100_000_000,
                ..Default::default()
            },
            ProximaRecord {
                oid: "vec_b".to_string(),
                embeddings: vec![EmbeddingCell {
                    model_id: "default".to_string(),
                    modality: "dense_vector".to_string(),
                    values: proximadb_records::EmbeddingValues::Fp32(vec![4.0, 5.0, 6.0]),
                    dim: 3,
                    ..Default::default()
                }],
                created_at_ns: 200_000_000,
                ..Default::default()
            },
        ];

        let batch = ArrowProtoCodec::vector_records_to_batch(records, 3)
            .expect("Failed to convert records to batch");

        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 5);

        let id_col = batch
            .column_by_name("id")
            .expect("Missing id column")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("id is not StringArray");
        assert_eq!(id_col.value(0), "vec_a");
        assert_eq!(id_col.value(1), "vec_b");
    }

    #[test]
    fn test_multimodal_codec_vector() {
        // Test round-trip: ProximaRecords -> RecordBatch -> ProximaRecords
        let original_records = vec![
            ProximaRecord {
                oid: "rt_1".to_string(),
                embeddings: vec![EmbeddingCell {
                    model_id: "default".to_string(),
                    modality: "dense_vector".to_string(),
                    values: proximadb_records::EmbeddingValues::Fp32(vec![0.5, 1.5, 2.5, 3.5]),
                    dim: 4,
                    ..Default::default()
                }],
                ..Default::default()
            },
            ProximaRecord {
                oid: "rt_2".to_string(),
                embeddings: vec![EmbeddingCell {
                    model_id: "default".to_string(),
                    modality: "dense_vector".to_string(),
                    values: proximadb_records::EmbeddingValues::Fp32(vec![4.5, 5.5, 6.5, 7.5]),
                    dim: 4,
                    ..Default::default()
                }],
                ..Default::default()
            },
        ];

        let batch = ArrowProtoCodec::vector_records_to_batch(original_records.clone(), 4)
            .expect("Failed to encode to batch");

        let decoded = ArrowProtoCodec::batches_to_proxima_records(vec![batch])
            .expect("Failed to decode from batch");

        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].oid, "rt_1");
        assert_eq!(decoded[1].oid, "rt_2");
        assert_eq!(
            decoded[0].embeddings[0].values,
            proximadb_records::EmbeddingValues::Fp32(vec![0.5, 1.5, 2.5, 3.5])
        );
        assert_eq!(
            decoded[1].embeddings[0].values,
            proximadb_records::EmbeddingValues::Fp32(vec![4.5, 5.5, 6.5, 7.5])
        );
    }

    #[test]
    fn test_multimodal_codec_document() {
        // Test document schema from multimodal_codec and verify field types
        use crate::network::arrow_ipc::multimodal_codec::document_schema;

        let schema = document_schema();
        assert_eq!(schema.fields().len(), 5);

        // Verify round-trip: build a RecordBatch from document schema, read it back
        let id_array = StringArray::from(vec!["doc_1", "doc_2"]);
        let doc_array = StringArray::from(vec![r#"{"title":"Hello"}"#, r#"{"title":"World"}"#]);
        let version_array = Int64Array::from(vec![1i64, 2]);
        let collection_array = StringArray::from(vec!["col_a", "col_a"]);
        let updated_array = Int64Array::from(vec![1000i64, 2000]);

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(id_array) as ArrayRef,
                Arc::new(doc_array) as ArrayRef,
                Arc::new(version_array) as ArrayRef,
                Arc::new(collection_array) as ArrayRef,
                Arc::new(updated_array) as ArrayRef,
            ],
        )
        .expect("Failed to create document batch");

        assert_eq!(batch.num_rows(), 2);

        let ids = batch
            .column_by_name("id")
            .expect("Missing id")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("id not StringArray");
        assert_eq!(ids.value(0), "doc_1");
        assert_eq!(ids.value(1), "doc_2");
    }

    #[test]
    fn test_flight_info_metadata() {
        // Verify that FlightDescriptor with command metadata is parsed correctly
        let cmd = serde_json::json!({
            "write_mode": "direct",
            "trigger_compaction": "true"
        });

        let descriptor = FlightDescriptor {
            r#type: 0,
            path: vec!["high_perf_collection".to_string()],
            cmd: serde_json::to_vec(&cmd)
                .expect("Failed to serialize cmd")
                .into(),
        };

        let metadata = ArrowProtoCodec::parse_descriptor(&descriptor)
            .expect("Failed to parse descriptor with metadata");
        assert_eq!(metadata.collection_id, "high_perf_collection");
        assert_eq!(metadata.operation, FlightWriteOperation::Upsert);
        assert_eq!(metadata.write_mode, WriteMode::Direct);
        assert!(metadata.trigger_compaction);
    }

    #[test]
    fn test_flight_descriptor_delete_operation_from_path() {
        let descriptor = FlightDescriptor {
            r#type: 0,
            path: vec!["delete".to_string(), "records".to_string()],
            cmd: Default::default(),
        };

        let metadata = ArrowProtoCodec::parse_descriptor(&descriptor)
            .expect("Failed to parse delete descriptor");
        assert_eq!(metadata.collection_id, "records");
        assert_eq!(metadata.operation, FlightWriteOperation::Delete);
        assert_eq!(metadata.write_mode, WriteMode::WAL);
    }

    #[test]
    fn test_flight_descriptor_operation_from_command() {
        let descriptor = FlightDescriptor {
            r#type: 0,
            path: vec!["records".to_string()],
            cmd: serde_json::to_vec(&serde_json::json!({
                "operation": "bulk_delete",
                "write_mode": "direct",
                "trigger_compaction": true
            }))
            .expect("Failed to serialize cmd")
            .into(),
        };

        let metadata = ArrowProtoCodec::parse_descriptor(&descriptor)
            .expect("Failed to parse command descriptor");
        assert_eq!(metadata.collection_id, "records");
        assert_eq!(metadata.operation, FlightWriteOperation::Delete);
        assert_eq!(metadata.write_mode, WriteMode::Direct);
        assert!(metadata.trigger_compaction);
    }

    #[test]
    fn test_flight_descriptor_collection_from_command() {
        let descriptor = FlightDescriptor {
            r#type: 0,
            path: Vec::new(),
            cmd: serde_json::to_vec(&serde_json::json!({
                "collection_id": "records",
                "operation": "upsert",
                "write_mode": "wal",
                "trigger_compaction": false
            }))
            .expect("Failed to serialize cmd")
            .into(),
        };

        let metadata = ArrowProtoCodec::parse_descriptor(&descriptor)
            .expect("Failed to parse command descriptor");
        assert_eq!(metadata.collection_id, "records");
        assert_eq!(metadata.operation, FlightWriteOperation::Upsert);
        assert_eq!(metadata.write_mode, WriteMode::WAL);
        assert!(!metadata.trigger_compaction);
    }

    #[test]
    fn test_batches_to_record_ids_accepts_oid() {
        let schema = Arc::new(Schema::new(vec![Field::new("oid", DataType::Utf8, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec!["r1", "r2"])) as ArrayRef],
        )
        .expect("Failed to create id batch");

        let ids =
            ArrowProtoCodec::batches_to_record_ids(vec![batch]).expect("Failed to extract ids");
        assert_eq!(ids, vec!["r1".to_string(), "r2".to_string()]);
    }

    #[test]
    fn test_flight_data_stream_to_batches_skips_descriptor_only_message() {
        let batch = create_test_batch(2, 4);
        let mut messages = vec![FlightData {
            flight_descriptor: Some(FlightDescriptor::new_path(vec![
                "bulk_upsert".to_string(),
                "records".to_string(),
            ])),
            data_header: Default::default(),
            app_metadata: Default::default(),
            data_body: Default::default(),
        }];
        messages.extend(
            ArrowProtoCodec::batch_to_flight_data(&batch, &Default::default())
                .expect("Failed to encode batch"),
        );

        let decoded = ArrowProtoCodec::flight_data_stream_to_batches(&messages)
            .expect("Failed to decode stream");
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].num_rows(), 2);
    }

    #[test]
    fn test_empty_batch_handling() {
        // Verify empty records produce an error (not a panic)
        let result = ArrowProtoCodec::vector_records_to_batch(vec![], 128);
        assert!(result.is_err(), "Empty records should return an error");

        // Verify empty batch list conversion works without panic
        let result = ArrowProtoCodec::batches_to_proxima_records(vec![]);
        assert!(result.is_ok());
        assert!(result.expect("Should succeed for empty batches").is_empty());

        // Verify empty batch list to flight data doesn't panic
        let result = ArrowProtoCodec::batches_to_flight_data_with_compression(&[], None);
        assert!(result.is_ok());
        assert!(
            result
                .expect("Should succeed for empty flight data")
                .is_empty()
        );
    }

    #[test]
    fn test_compression_reduces_size() {
        // Create a batch with repetitive data that compresses well
        let num_rows = 1000;
        let dimension = 128;

        // Create highly compressible data (all zeros)
        let ids: Vec<String> = (0..num_rows).map(|i| format!("id_{}", i)).collect();
        let id_array = StringArray::from(ids);

        let flat_values: Vec<f32> = vec![0.0f32; num_rows * dimension];
        let values_array = Arc::new(Float32Array::from(flat_values)) as ArrayRef;
        let vector_field = Arc::new(Field::new("item", DataType::Float32, false));
        let vector_array =
            FixedSizeListArray::new(vector_field, dimension as i32, values_array, None);

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, false)),
                    dimension as i32,
                ),
                false,
            ),
        ]));

        let batch = RecordBatch::try_new(schema, vec![Arc::new(id_array), Arc::new(vector_array)])
            .expect("Failed to create batch");

        // Get uncompressed size
        let uncompressed =
            ArrowProtoCodec::batches_to_flight_data_with_compression(&[batch.clone()], None)
                .expect("Failed uncompressed");

        // Get compressed size
        let compressed = ArrowProtoCodec::batches_to_flight_data_with_compression(
            &[batch],
            Some(arrow_ipc::CompressionType::LZ4_FRAME),
        )
        .expect("Failed compressed");

        // Calculate total data size
        let uncompressed_size: usize = uncompressed.iter().map(|fd| fd.data_body.len()).sum();
        let compressed_size: usize = compressed.iter().map(|fd| fd.data_body.len()).sum();

        // Compressed should be smaller for highly repetitive data
        // Note: For very small data, compression overhead might make it larger
        if uncompressed_size > 1024 {
            assert!(
                compressed_size <= uncompressed_size,
                "Compressed ({}) should be <= uncompressed ({}) for repetitive data",
                compressed_size,
                uncompressed_size
            );
        }
    }
}
