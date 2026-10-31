// Copyright 2025 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! gRPC V2 ProximaRecord service implementation
//!
//! This module implements the ProximaRecordService gRPC service for the V2 API.
//! It provides:
//! - Batch record operations (insert, upsert, update, delete)
//! - Search with typed filters
//! - Schema management (create, get, list, evolve)

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, Streaming};
use tracing::{debug, error, info, trace, warn};

use crate::api_handlers::{
    RichFilterCondition, RichFilterOperator, RichRecordBatchRequest, RichRecordDeleteBatchRequest,
    RichRecordGetRequest, RichSearchRequest, RichSearchResponse,
    record_ops_service::RecordOpsService,
};
use crate::network::grpc::auth as grpc_auth;
use crate::proto::proximadb_v1::{CollectionOperation, CollectionRequest};
use crate::proto::proximadb_v2::{
    self, BackpressureLevel, BackpressureSignal, BatchError, BatchWriteMode,
    BatchWriteStreamRequest, BatchWriteStreamResponse, CreateSchemaRequest, CreateSchemaResponse,
    EvolveSchemaRequest, EvolveSchemaResponse, GetSchemaRequest, GetSchemaResponse,
    ListSchemasRequest, ListSchemasResponse, PredicateShortfall, ProximaRecordBatch,
    ProximaRecordBatchResponse, TypedSearchRequest, TypedSearchResponse, TypedSearchResult,
    V2Collection, V2CollectionConfig, V2DeleteCollectionRequest, V2DeleteCollectionResponse,
    V2GetCollectionRequest, V2ListCollectionsRequest, V2ListCollectionsResponse, V2QueryRequest,
    V2QueryResponse, V2QueryRow, proxima_record_service_server::ProximaRecordService,
    proxima_record_service_server::ProximaRecordServiceServer,
};
use crate::services::operations::BatchOperationResult;
use crate::services::{
    WriteDurabilityRequirement, WriteIntent, WriteLaneRouter, WriteOperationKind,
};
use proximadb_data_model::{ProximaType, TimeUnit, VectorElement};
use proximadb_records::proto_v2::{
    proto_record_to_envelope, proxima_value_to_typed_value, typed_value_to_proxima,
};
use proximadb_runtime::{
    CollectionSchemaColumn, CollectionSchemaEnforcement, CollectionSchemaMetadata,
    CollectionSchemaUpdate, CollectionTextStorage,
};

/// gRPC V2 ProximaRecord service implementation
///
/// This service provides the V2 API with:
/// - Rich typed fields (TEXT, INTEGER, FLOAT, DECIMAL, UUID, etc.)
/// - Schema enforcement modes (STRICT, FLEXIBLE, HYBRID)
/// - Typed filtering with range, equality, and CONTAINS operators
pub struct ProximaRecordServiceImpl {
    /// Port-backed handler for collection lifecycle ops + tenant-scoped SQL.
    api_handlers: Arc<dyn proximadb_runtime::ApiHandlersPort>,
    /// Single record authority (reads + writes) — TD-104. Replaces the ROOT
    /// `UnifiedHandlers` the service previously reached through.
    record_ops: Arc<RecordOpsService>,
    /// Shared segment registry — after each successful batch write, records are
    /// flushed to a `.pax` file and the resulting `SegmentMeta` is registered here
    /// so the Iceberg REST server can serve accurate snapshot stats.
    segment_registry: Option<Arc<crate::catalog::SegmentRegistry>>,
    /// Slice 6.1: primary-pod write router. When `Some`, `insert_records`
    /// consults [`crate::cluster::primary_pod_registry::consult_for_write`]
    /// before any storage work and returns `tonic::Status::failed_precondition`
    /// (the gRPC analog of HTTP 421) when the write belongs on a different pod.
    /// `None` keeps the legacy "no gate" behavior for embedded / unit-test
    /// constructions that don't carry a SharedServices instance.
    primary_pod_gate: Option<PrimaryPodGate>,
}

/// Bundle of primary-pod gate inputs so the constructor takes one
/// argument and the field tells a clear story. Mirrors AppState's
/// `primary_pod_registry` + `self_pod_id` pair from the REST side.
#[derive(Clone)]
struct PrimaryPodGate {
    registry: Arc<crate::cluster::primary_pod_registry::PrimaryPodRegistry>,
    self_pod_id: String,
}

/// Slice 6.1 testable helper. Returns `Ok(())` when the request is
/// allowed (either gate is unconfigured or the registry decides
/// `Allow`), or the misrouted `tonic::Status` to return verbatim
/// from the handler. Free function rather than method so unit tests
/// don't have to construct a full `ProximaRecordServiceImpl`.
fn check_primary_pod_gate(
    gate: &Option<PrimaryPodGate>,
    tenant_id: &str,
    collection_id: &str,
) -> Result<(), Status> {
    let Some(gate) = gate else {
        return Ok(());
    };
    match crate::cluster::primary_pod_registry::consult_for_write(
        &gate.registry,
        &gate.self_pod_id,
        tenant_id,
        collection_id,
    ) {
        crate::cluster::primary_pod_registry::WriteRoutingDecision::Allow => {
            if gate.registry.is_assigned(tenant_id, collection_id) {
                crate::metrics::primary_pod_metrics::record_allowed_bound(tenant_id);
            } else {
                crate::metrics::primary_pod_metrics::record_allowed_unbounded(tenant_id);
            }
            Ok(())
        }
        crate::cluster::primary_pod_registry::WriteRoutingDecision::Misrouted { target_pod } => {
            crate::metrics::primary_pod_metrics::record_misrouted(tenant_id);
            tracing::warn!(
                target = "proximadb.primary_pod.misroute",
                self_pod = %gate.self_pod_id,
                target_pod = %target_pod,
                tenant_id = %tenant_id,
                collection_id = %collection_id,
                "gRPC v2 insert_records misrouted — client SDK should retry against the primary pod"
            );
            let api_err = crate::errors::ApiError::Misdirected {
                target_pod,
                tenant_id: tenant_id.to_string(),
                collection_id: collection_id.to_string(),
            };
            Err(api_err.into())
        }
    }
}

fn grpc_record_schema_to_runtime_update(
    collection_id: &str,
    schema: proximadb_v2::RecordSchema,
) -> Result<CollectionSchemaUpdate, Status> {
    let schema_id = if schema.schema_id.is_empty() {
        format!("schema_{}_{}", collection_id, uuid::Uuid::new_v4())
    } else {
        schema.schema_id
    };
    let schema_version = if schema.schema_version.is_empty() {
        "1.0.0".to_string()
    } else {
        schema.schema_version
    };

    let mut columns = Vec::with_capacity(schema.columns.len());
    for column in schema.columns {
        let name = column.name.clone();
        let (data_type, text_storage) = grpc_column_type_to_proxima_type(&column)?;
        columns.push(CollectionSchemaColumn {
            name,
            data_type,
            nullable: column.nullable,
            indexed: column.indexed,
            filterable: column.filterable,
            text_storage,
            max_length: column.max_length,
        });
    }

    Ok(CollectionSchemaUpdate {
        schema_id,
        schema_version,
        enforcement: grpc_enforcement_to_runtime(schema.enforcement_mode),
        auto_evolve: schema.allow_additional_fields,
        columns,
    })
}

/// Derive a minimal canonical schema (ADR-047 authority) from a v2 collection
/// config: one primary `embedding` `DenseVector{Float32, dim}` column (mirroring
/// the storage-provisioned `config.dimension`) plus the declared scalar
/// `filterable_columns` as `String`. Richer/multi-vector schemas are layered on
/// via `CreateSchema`.
///
/// Used by v2 `create_collection` so the catalog's `collection.canonical_schema`
/// sidecar is authoritative for v2-created collections — not just the v1-derived
/// narrow view.
fn v2_config_to_canonical_columns(
    cfg: &proximadb_v2::V2CollectionConfig,
) -> Vec<CollectionSchemaColumn> {
    let mut columns = Vec::with_capacity(1 + cfg.filterable_columns.len());
    columns.push(CollectionSchemaColumn {
        name: "embedding".to_string(),
        data_type: ProximaType::DenseVector {
            element: proximadb_data_model::VectorElement::Float32,
            dim: cfg.dimension as usize,
        },
        nullable: false,
        indexed: true,
        filterable: false,
        text_storage: None,
        max_length: None,
    });
    for name in &cfg.filterable_columns {
        if name.is_empty() {
            continue;
        }
        columns.push(CollectionSchemaColumn {
            name: name.clone(),
            data_type: ProximaType::String,
            nullable: true,
            indexed: false,
            filterable: true,
            text_storage: None,
            max_length: None,
        });
    }
    columns
}

fn runtime_metadata_to_grpc_schema(
    metadata: &CollectionSchemaMetadata,
) -> proximadb_v2::RecordSchema {
    let columns = metadata
        .columns
        .iter()
        .map(|column| proximadb_v2::TypedColumn {
            name: column.name.clone(),
            data_type: proxima_type_to_grpc_column_type(&column.data_type, column.text_storage)
                as i32,
            nullable: column.nullable,
            indexed: column.indexed,
            filterable: column.filterable,
            unique: false,
            max_length: column.max_length,
            min_length: None,
            min_value: None,
            max_value: None,
            min_float_value: None,
            max_float_value: None,
            regex_pattern: None,
            default_value: None,
            decimal_precision: match &column.data_type {
                ProximaType::Decimal { precision, .. } => Some(*precision as u32),
                _ => None,
            },
            decimal_scale: match &column.data_type {
                ProximaType::Decimal { scale, .. } => Some(*scale as u32),
                _ => None,
            },
            array_max_items: None,
            json_max_depth: None,
            json_schema: None,
            timezone: None,
            vector_dimension: match &column.data_type {
                ProximaType::DenseVector { dim, .. } | ProximaType::BinaryVector { dim } => {
                    Some(*dim as u32)
                }
                _ => None,
            },
            vector_element_type: match &column.data_type {
                ProximaType::DenseVector { element, .. }
                | ProximaType::SparseVector { element } => {
                    Some(proxima_vector_element_to_grpc(*element) as i32)
                }
                _ => None,
            },
            text_storage_strategy: None,
            enable_fulltext_index: None,
            enable_ngram_bloom: None,
            ngram_size: None,
            description: None,
            annotations: HashMap::new(),
        })
        .collect::<Vec<_>>();

    proximadb_v2::RecordSchema {
        schema_id: metadata
            .schema_id
            .clone()
            .unwrap_or_else(|| format!("schema_{}", metadata.collection_id)),
        schema_version: metadata
            .schema_version
            .clone()
            .unwrap_or_else(|| "1.0.0".to_string()),
        schema_name: String::new(),
        columns,
        enforcement_mode: runtime_enforcement_to_grpc(metadata.enforcement.unwrap_or_default())
            as i32,
        allow_additional_fields: metadata.auto_evolve,
        parent_schema_id: None,
        evolution_rules: Vec::new(),
        created_at: metadata.created_at_ms,
        created_by: None,
        description: None,
        annotations: HashMap::new(),
    }
}

fn grpc_enforcement_to_runtime(value: i32) -> CollectionSchemaEnforcement {
    match proximadb_v2::SchemaEnforcementMode::try_from(value)
        .unwrap_or(proximadb_v2::SchemaEnforcementMode::Hybrid)
    {
        proximadb_v2::SchemaEnforcementMode::Strict => CollectionSchemaEnforcement::Strict,
        proximadb_v2::SchemaEnforcementMode::Flexible => CollectionSchemaEnforcement::Flexible,
        _ => CollectionSchemaEnforcement::Hybrid,
    }
}

fn runtime_enforcement_to_grpc(
    value: CollectionSchemaEnforcement,
) -> proximadb_v2::SchemaEnforcementMode {
    match value {
        CollectionSchemaEnforcement::Strict => proximadb_v2::SchemaEnforcementMode::Strict,
        CollectionSchemaEnforcement::Flexible => proximadb_v2::SchemaEnforcementMode::Flexible,
        CollectionSchemaEnforcement::Hybrid => proximadb_v2::SchemaEnforcementMode::Hybrid,
    }
}

fn grpc_column_type_to_proxima_type(
    column: &proximadb_v2::TypedColumn,
) -> Result<(ProximaType, Option<CollectionTextStorage>), Status> {
    let data_type = proximadb_v2::ColumnDataType::try_from(column.data_type)
        .unwrap_or(proximadb_v2::ColumnDataType::ColumnTypeUnspecified);
    let ty = match data_type {
        proximadb_v2::ColumnDataType::ColumnTypeUnspecified => {
            return Err(Status::invalid_argument(format!(
                "Schema column '{}' has unspecified datatype",
                column.name
            )));
        }
        proximadb_v2::ColumnDataType::Text => {
            return Ok((ProximaType::String, Some(CollectionTextStorage::Inline)));
        }
        proximadb_v2::ColumnDataType::TextLarge => {
            return Ok((ProximaType::String, Some(CollectionTextStorage::Large)));
        }
        proximadb_v2::ColumnDataType::Integer => ProximaType::Int64,
        proximadb_v2::ColumnDataType::Float => ProximaType::Float64,
        proximadb_v2::ColumnDataType::Decimal => {
            let precision = column.decimal_precision.unwrap_or(38);
            let scale = column.decimal_scale.unwrap_or(18);
            if precision == 0 || precision > 38 {
                return Err(Status::invalid_argument(format!(
                    "Schema column '{}' decimal precision must be between 1 and 38",
                    column.name
                )));
            }
            if scale > precision {
                return Err(Status::invalid_argument(format!(
                    "Schema column '{}' decimal scale cannot exceed precision",
                    column.name
                )));
            }
            ProximaType::Decimal {
                precision: precision as u8,
                scale: scale as u8,
            }
        }
        proximadb_v2::ColumnDataType::Boolean => ProximaType::Boolean,
        proximadb_v2::ColumnDataType::Timestamp => ProximaType::Timestamp(TimeUnit::Nanosecond),
        proximadb_v2::ColumnDataType::TimestampTz => ProximaType::TimestampTz(TimeUnit::Nanosecond),
        proximadb_v2::ColumnDataType::Date => ProximaType::Date,
        proximadb_v2::ColumnDataType::Time => ProximaType::Time(TimeUnit::Nanosecond),
        proximadb_v2::ColumnDataType::Duration => ProximaType::Duration(TimeUnit::Nanosecond),
        proximadb_v2::ColumnDataType::Interval => ProximaType::Interval(TimeUnit::Nanosecond),
        proximadb_v2::ColumnDataType::Uuid => ProximaType::Uuid,
        proximadb_v2::ColumnDataType::Binary | proximadb_v2::ColumnDataType::BinaryLarge => {
            ProximaType::Binary
        }
        proximadb_v2::ColumnDataType::Json => ProximaType::Json,
        proximadb_v2::ColumnDataType::ArrayText => {
            ProximaType::Array(Box::new(ProximaType::String))
        }
        proximadb_v2::ColumnDataType::ArrayInteger => {
            ProximaType::Array(Box::new(ProximaType::Int64))
        }
        proximadb_v2::ColumnDataType::ArrayFloat => {
            ProximaType::Array(Box::new(ProximaType::Float64))
        }
        proximadb_v2::ColumnDataType::ArrayBoolean => {
            ProximaType::Array(Box::new(ProximaType::Boolean))
        }
        proximadb_v2::ColumnDataType::ArrayUuid => ProximaType::Array(Box::new(ProximaType::Uuid)),
        proximadb_v2::ColumnDataType::ArrayAny => ProximaType::Array(Box::new(ProximaType::Json)),
        proximadb_v2::ColumnDataType::MapStringString => ProximaType::Map {
            key: Box::new(ProximaType::String),
            value: Box::new(ProximaType::String),
        },
        proximadb_v2::ColumnDataType::MapStringAny => ProximaType::Map {
            key: Box::new(ProximaType::String),
            value: Box::new(ProximaType::Json),
        },
        proximadb_v2::ColumnDataType::MapStringInteger => ProximaType::Map {
            key: Box::new(ProximaType::String),
            value: Box::new(ProximaType::Int64),
        },
        proximadb_v2::ColumnDataType::MapStringFloat => ProximaType::Map {
            key: Box::new(ProximaType::String),
            value: Box::new(ProximaType::Float64),
        },
        proximadb_v2::ColumnDataType::GeoPoint => ProximaType::Point,
        proximadb_v2::ColumnDataType::GeoPolygon => ProximaType::GeographyPoint,
        proximadb_v2::ColumnDataType::Vector => ProximaType::DenseVector {
            element: grpc_vector_element_to_proxima(column.vector_element_type),
            dim: column.vector_dimension.unwrap_or(0) as usize,
        },
        proximadb_v2::ColumnDataType::SparseVector => ProximaType::SparseVector {
            element: grpc_vector_element_to_proxima(column.vector_element_type),
        },
        proximadb_v2::ColumnDataType::Int8 => ProximaType::Int8,
        proximadb_v2::ColumnDataType::Int16 => ProximaType::Int16,
        proximadb_v2::ColumnDataType::Int32 => ProximaType::Int32,
        proximadb_v2::ColumnDataType::Uint8 => ProximaType::UInt8,
        proximadb_v2::ColumnDataType::Uint16 => ProximaType::UInt16,
        proximadb_v2::ColumnDataType::Uint32 => ProximaType::UInt32,
        proximadb_v2::ColumnDataType::Uint64 => ProximaType::UInt64,
        proximadb_v2::ColumnDataType::Float32 => ProximaType::Float32,
        proximadb_v2::ColumnDataType::Symbol => ProximaType::Symbol,
        proximadb_v2::ColumnDataType::Jsonb => ProximaType::Jsonb,
        proximadb_v2::ColumnDataType::Ulid => ProximaType::ULID,
        proximadb_v2::ColumnDataType::Struct => ProximaType::Struct { fields: Vec::new() },
        proximadb_v2::ColumnDataType::BinaryVector => ProximaType::BinaryVector {
            dim: column.vector_dimension.unwrap_or(0) as usize,
        },
    };
    Ok((ty, None))
}

/// Map the wire `VectorElementType` (optional) into the canonical
/// `VectorElement`. `None`/`Unspecified` fall back to `Float32` to preserve the
/// historical default (pre-field behavior).
fn grpc_vector_element_to_proxima(value: Option<i32>) -> VectorElement {
    use proximadb_v2::VectorElementType as E;
    match value
        .and_then(|v| E::try_from(v).ok())
        .unwrap_or(E::VectorElementFloat32)
    {
        E::VectorElementFloat16 => VectorElement::Float16,
        E::VectorElementBfloat16 => VectorElement::BFloat16,
        E::VectorElementFloat64 => VectorElement::Float64,
        E::VectorElementInt8 => VectorElement::Int8,
        E::VectorElementFloat32 | E::Unspecified => VectorElement::Float32,
    }
}

fn proxima_vector_element_to_grpc(value: VectorElement) -> proximadb_v2::VectorElementType {
    use proximadb_v2::VectorElementType as E;
    match value {
        VectorElement::Float16 => E::VectorElementFloat16,
        VectorElement::BFloat16 => E::VectorElementBfloat16,
        VectorElement::Float32 => E::VectorElementFloat32,
        VectorElement::Float64 => E::VectorElementFloat64,
        VectorElement::Int8 => E::VectorElementInt8,
    }
}

fn proxima_type_to_grpc_column_type(
    value: &ProximaType,
    text_storage: Option<CollectionTextStorage>,
) -> proximadb_v2::ColumnDataType {
    match value {
        ProximaType::String => match text_storage {
            Some(CollectionTextStorage::Large) => proximadb_v2::ColumnDataType::TextLarge,
            _ => proximadb_v2::ColumnDataType::Text,
        },
        ProximaType::Int8 => proximadb_v2::ColumnDataType::Int8,
        ProximaType::Int16 => proximadb_v2::ColumnDataType::Int16,
        ProximaType::Int32 => proximadb_v2::ColumnDataType::Int32,
        ProximaType::Int64 => proximadb_v2::ColumnDataType::Integer,
        ProximaType::UInt8 => proximadb_v2::ColumnDataType::Uint8,
        ProximaType::UInt16 => proximadb_v2::ColumnDataType::Uint16,
        ProximaType::UInt32 => proximadb_v2::ColumnDataType::Uint32,
        ProximaType::UInt64 => proximadb_v2::ColumnDataType::Uint64,
        ProximaType::Float16 | ProximaType::Float32 => proximadb_v2::ColumnDataType::Float32,
        ProximaType::Float64 => proximadb_v2::ColumnDataType::Float,
        ProximaType::Decimal { .. } => proximadb_v2::ColumnDataType::Decimal,
        ProximaType::Boolean => proximadb_v2::ColumnDataType::Boolean,
        ProximaType::Timestamp(_) => proximadb_v2::ColumnDataType::Timestamp,
        ProximaType::TimestampTz(_) => proximadb_v2::ColumnDataType::TimestampTz,
        ProximaType::Date => proximadb_v2::ColumnDataType::Date,
        ProximaType::Time(_) => proximadb_v2::ColumnDataType::Time,
        ProximaType::Duration(_) => proximadb_v2::ColumnDataType::Duration,
        ProximaType::Interval(_) => proximadb_v2::ColumnDataType::Interval,
        ProximaType::Uuid => proximadb_v2::ColumnDataType::Uuid,
        ProximaType::ULID => proximadb_v2::ColumnDataType::Ulid,
        ProximaType::Binary => proximadb_v2::ColumnDataType::Binary,
        ProximaType::Json => proximadb_v2::ColumnDataType::Json,
        ProximaType::Jsonb => proximadb_v2::ColumnDataType::Jsonb,
        ProximaType::Array(inner) => match inner.as_ref() {
            ProximaType::String => proximadb_v2::ColumnDataType::ArrayText,
            ProximaType::Int64 => proximadb_v2::ColumnDataType::ArrayInteger,
            ProximaType::Float64 => proximadb_v2::ColumnDataType::ArrayFloat,
            ProximaType::Boolean => proximadb_v2::ColumnDataType::ArrayBoolean,
            ProximaType::Uuid => proximadb_v2::ColumnDataType::ArrayUuid,
            _ => proximadb_v2::ColumnDataType::ArrayAny,
        },
        ProximaType::Map { key, value } => match (key.as_ref(), value.as_ref()) {
            (ProximaType::String, ProximaType::String) => {
                proximadb_v2::ColumnDataType::MapStringString
            }
            (ProximaType::String, ProximaType::Int64) => {
                proximadb_v2::ColumnDataType::MapStringInteger
            }
            (ProximaType::String, ProximaType::Float64) => {
                proximadb_v2::ColumnDataType::MapStringFloat
            }
            (ProximaType::String, _) => proximadb_v2::ColumnDataType::MapStringAny,
            _ => proximadb_v2::ColumnDataType::MapStringAny,
        },
        ProximaType::Struct { .. } => proximadb_v2::ColumnDataType::Struct,
        ProximaType::Point => proximadb_v2::ColumnDataType::GeoPoint,
        ProximaType::GeographyPoint => proximadb_v2::ColumnDataType::GeoPolygon,
        ProximaType::DenseVector { .. } => proximadb_v2::ColumnDataType::Vector,
        ProximaType::SparseVector { .. } => proximadb_v2::ColumnDataType::SparseVector,
        ProximaType::BinaryVector { .. } => proximadb_v2::ColumnDataType::BinaryVector,
        ProximaType::Symbol => proximadb_v2::ColumnDataType::Symbol,
        ProximaType::Null => proximadb_v2::ColumnDataType::Json,
    }
}

/// Streaming response type for SearchStream
pub type SearchStreamStream =
    Pin<Box<dyn tokio_stream::Stream<Item = Result<TypedSearchResult, Status>> + Send>>;

/// Streaming response type for BatchWriteStream
pub type BatchWriteStreamStream =
    Pin<Box<dyn tokio_stream::Stream<Item = Result<BatchWriteStreamResponse, Status>> + Send>>;

/// Channel buffer size for streaming operations
const STREAM_BUFFER_SIZE: usize = 128;

/// Buffer utilization thresholds for backpressure (percentage)
/// NONE: 0-25% - No backpressure, client can send freely
const BACKPRESSURE_LOW_THRESHOLD: u32 = 25;
/// LOW: 25-50% - Light backpressure, suggest small delays
const BACKPRESSURE_MEDIUM_THRESHOLD: u32 = 50;
/// MEDIUM: 50-75% - Moderate backpressure, suggest medium delays
const BACKPRESSURE_HIGH_THRESHOLD: u32 = 75;
/// HIGH: 75-90% - Heavy backpressure, suggest significant delays
const BACKPRESSURE_CRITICAL_THRESHOLD: u32 = 90;
/// CRITICAL: 90%+ - Critical backpressure, suggest maximum delays
///
/// Suggested delays in milliseconds for each backpressure level
const DELAY_NONE_MS: u32 = 0;
const DELAY_LOW_MS: u32 = 10;
const DELAY_MEDIUM_MS: u32 = 50;
const DELAY_HIGH_MS: u32 = 100;
const DELAY_CRITICAL_MS: u32 = 500;

/// Maximum pending items before blocking (for flow control)
const MAX_PENDING_ITEMS: u32 = 1000;

fn grpc_filters_to_rich(
    filters: &[proximadb_v2::TypedFilterCondition],
) -> Result<Vec<RichFilterCondition>, Status> {
    filters
        .iter()
        .map(|filter| {
            let operator = match proximadb_v2::TypedFilterOperator::try_from(filter.operator)
                .unwrap_or(proximadb_v2::TypedFilterOperator::TypedFilterOpUnspecified)
            {
                proximadb_v2::TypedFilterOperator::Eq => RichFilterOperator::Eq,
                proximadb_v2::TypedFilterOperator::Ne => RichFilterOperator::Ne,
                proximadb_v2::TypedFilterOperator::Gt => RichFilterOperator::Gt,
                proximadb_v2::TypedFilterOperator::Gte => RichFilterOperator::Gte,
                proximadb_v2::TypedFilterOperator::Lt => RichFilterOperator::Lt,
                proximadb_v2::TypedFilterOperator::Lte => RichFilterOperator::Lte,
                proximadb_v2::TypedFilterOperator::Between => RichFilterOperator::Between,
                proximadb_v2::TypedFilterOperator::In => RichFilterOperator::In,
                proximadb_v2::TypedFilterOperator::NotIn => RichFilterOperator::NotIn,
                proximadb_v2::TypedFilterOperator::Contains => RichFilterOperator::Contains,
                other => {
                    return Err(Status::invalid_argument(format!(
                        "unsupported v2 search filter operator: {}",
                        other.as_str_name()
                    )));
                }
            };

            let value = filter
                .value
                .as_ref()
                .ok_or_else(|| Status::invalid_argument("filter value is required"))
                .and_then(|value| {
                    typed_value_to_proxima(value)
                        .map_err(|e| Status::invalid_argument(format!("invalid filter value: {e}")))
                })?;
            let value_upper = filter
                .value_upper
                .as_ref()
                .map(typed_value_to_proxima)
                .transpose()
                .map_err(|e| {
                    Status::invalid_argument(format!("invalid upper filter value: {e}"))
                })?;
            let value_list = filter
                .value_list
                .iter()
                .map(typed_value_to_proxima)
                .collect::<anyhow::Result<Vec<_>>>()
                .map_err(|e| Status::invalid_argument(format!("invalid filter value list: {e}")))?;

            Ok(RichFilterCondition {
                field: filter.field_name.clone(),
                operator,
                value,
                value_upper,
                value_list,
            })
        })
        .collect()
}

/// Streaming pipeline latency metrics for observability
///
/// Tracks latency at various stages of the streaming write pipeline:
/// - Queue wait time: Time spent waiting in the buffer queue
/// - Processing time: Time spent processing individual records
/// - Ack send time: Time spent sending acknowledgments back to client
#[derive(Debug)]
pub struct StreamingPipelineMetrics {
    /// Total batches processed
    pub batches_processed: AtomicU64,
    /// Total records processed
    pub records_processed: AtomicU64,
    /// Total queue wait time in microseconds (cumulative)
    pub queue_wait_time_us: AtomicU64,
    /// Total processing time in microseconds (cumulative)
    pub processing_time_us: AtomicU64,
    /// Total ack send time in microseconds (cumulative)
    pub ack_send_time_us: AtomicU64,
    /// Maximum observed queue depth
    pub max_queue_depth: AtomicU32,
    /// Number of times backpressure was applied
    pub backpressure_events: AtomicU64,
    /// Number of times CRITICAL backpressure was reached
    pub critical_backpressure_events: AtomicU64,
    /// Current backpressure level (as i32 for atomic access)
    pub current_backpressure_level: AtomicU32,
}

impl StreamingPipelineMetrics {
    /// Create a new metrics instance
    pub fn new() -> Self {
        Self {
            batches_processed: AtomicU64::new(0),
            records_processed: AtomicU64::new(0),
            queue_wait_time_us: AtomicU64::new(0),
            processing_time_us: AtomicU64::new(0),
            ack_send_time_us: AtomicU64::new(0),
            max_queue_depth: AtomicU32::new(0),
            backpressure_events: AtomicU64::new(0),
            critical_backpressure_events: AtomicU64::new(0),
            current_backpressure_level: AtomicU32::new(0),
        }
    }

    /// Record queue wait time
    pub fn record_queue_wait(&self, duration: Duration) {
        self.queue_wait_time_us
            .fetch_add(duration.as_micros() as u64, Ordering::Relaxed);
    }

    /// Record processing time
    pub fn record_processing_time(&self, duration: Duration) {
        self.processing_time_us
            .fetch_add(duration.as_micros() as u64, Ordering::Relaxed);
    }

    /// Record ack send time
    pub fn record_ack_send_time(&self, duration: Duration) {
        self.ack_send_time_us
            .fetch_add(duration.as_micros() as u64, Ordering::Relaxed);
    }

    /// Update max queue depth if current is higher
    pub fn update_max_queue_depth(&self, current_depth: u32) {
        loop {
            let current_max = self.max_queue_depth.load(Ordering::Relaxed);
            if current_depth <= current_max {
                break;
            }
            if self
                .max_queue_depth
                .compare_exchange_weak(
                    current_max,
                    current_depth,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                break;
            }
        }
    }

    /// Record a backpressure event
    pub fn record_backpressure(&self, level: BackpressureLevel) {
        self.backpressure_events.fetch_add(1, Ordering::Relaxed);
        self.current_backpressure_level
            .store(level as u32, Ordering::Relaxed);
        if level == BackpressureLevel::BackpressureCritical {
            self.critical_backpressure_events
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Increment batch counter
    pub fn increment_batches(&self) {
        self.batches_processed.fetch_add(1, Ordering::Relaxed);
    }

    /// Add to record counter
    pub fn add_records(&self, count: u64) {
        self.records_processed.fetch_add(count, Ordering::Relaxed);
    }

    /// Get average queue wait time in microseconds (0 if no batches processed)
    pub fn avg_queue_wait_us(&self) -> f64 {
        let batches = self.batches_processed.load(Ordering::Relaxed);
        if batches == 0 {
            0.0
        } else {
            self.queue_wait_time_us.load(Ordering::Relaxed) as f64 / batches as f64
        }
    }

    /// Get average processing time per record in microseconds (0 if no records processed)
    pub fn avg_processing_time_per_record_us(&self) -> f64 {
        let records = self.records_processed.load(Ordering::Relaxed);
        if records == 0 {
            0.0
        } else {
            self.processing_time_us.load(Ordering::Relaxed) as f64 / records as f64
        }
    }

    /// Get summary statistics as a formatted string
    pub fn summary(&self) -> String {
        format!(
            "StreamingMetrics {{ batches: {}, records: {}, avg_queue_wait_us: {:.2}, avg_proc_per_record_us: {:.2}, max_depth: {}, bp_events: {}, critical_bp: {} }}",
            self.batches_processed.load(Ordering::Relaxed),
            self.records_processed.load(Ordering::Relaxed),
            self.avg_queue_wait_us(),
            self.avg_processing_time_per_record_us(),
            self.max_queue_depth.load(Ordering::Relaxed),
            self.backpressure_events.load(Ordering::Relaxed),
            self.critical_backpressure_events.load(Ordering::Relaxed),
        )
    }
}

impl Default for StreamingPipelineMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Flow control state for managing bounded channel backpressure
#[derive(Debug)]
struct FlowControlState {
    /// Current number of pending items in the processing queue
    pending_items: AtomicU32,
    /// Whether flow control is currently active (blocking new sends)
    flow_control_active: std::sync::atomic::AtomicBool,
    /// Channel capacity for utilization calculation
    channel_capacity: usize,
}

impl FlowControlState {
    fn new(channel_capacity: usize) -> Self {
        Self {
            pending_items: AtomicU32::new(0),
            flow_control_active: std::sync::atomic::AtomicBool::new(false),
            channel_capacity,
        }
    }

    /// Add pending items and check if flow control should be activated
    fn add_pending(&self, count: u32) -> bool {
        let new_pending = self.pending_items.fetch_add(count, Ordering::SeqCst) + count;
        if new_pending >= MAX_PENDING_ITEMS {
            self.flow_control_active.store(true, Ordering::SeqCst);
            true
        } else {
            false
        }
    }

    /// Remove pending items and check if flow control can be deactivated
    fn remove_pending(&self, count: u32) {
        let new_pending = self.pending_items.fetch_sub(count, Ordering::SeqCst) - count;
        // Deactivate flow control when below 75% of max
        if new_pending < (MAX_PENDING_ITEMS * 3 / 4) {
            self.flow_control_active.store(false, Ordering::SeqCst);
        }
    }

    /// Check if flow control is currently active
    fn is_active(&self) -> bool {
        self.flow_control_active.load(Ordering::SeqCst)
    }

    /// Get current pending count
    fn pending(&self) -> u32 {
        self.pending_items.load(Ordering::Relaxed)
    }

    /// Calculate buffer utilization percentage based on both pending items and channel usage
    fn calculate_utilization(&self, channel_remaining_capacity: usize) -> u32 {
        // Calculate pending-based utilization
        let pending_util =
            (self.pending() as f64 / MAX_PENDING_ITEMS as f64 * 100.0).min(100.0) as u32;

        // Calculate channel-based utilization
        let channel_used = self
            .channel_capacity
            .saturating_sub(channel_remaining_capacity);
        let channel_util =
            (channel_used as f64 / self.channel_capacity as f64 * 100.0).min(100.0) as u32;

        // Return the higher of the two utilization metrics
        pending_util.max(channel_util)
    }
}

impl ProximaRecordServiceImpl {
    /// Create a new ProximaRecordServiceImpl
    pub fn new(
        api_handlers: Arc<dyn proximadb_runtime::ApiHandlersPort>,
        record_ops: Arc<RecordOpsService>,
    ) -> Self {
        Self {
            api_handlers,
            record_ops,
            segment_registry: None,
            primary_pod_gate: None,
        }
    }

    /// Attach a shared segment registry to enable PAX write-through on batch inserts.
    pub fn with_segment_registry(mut self, registry: Arc<crate::catalog::SegmentRegistry>) -> Self {
        self.segment_registry = Some(registry);
        self
    }

    /// Slice 6.1: attach the primary-pod write router. Once set,
    /// `insert_records` consults the registry on every call and
    /// rejects misrouted writes with `Status::failed_precondition`
    /// before any storage work. SharedServices passes the same
    /// `Arc<PrimaryPodRegistry>` it gives to AppState so the REST
    /// and gRPC surfaces see identical routing decisions.
    pub fn with_primary_pod_gate(
        mut self,
        registry: Arc<crate::cluster::primary_pod_registry::PrimaryPodRegistry>,
        self_pod_id: String,
    ) -> Self {
        self.primary_pod_gate = Some(PrimaryPodGate {
            registry,
            self_pod_id,
        });
        self
    }

    /// Convert to a tonic server
    pub fn into_server(self) -> ProximaRecordServiceServer<Self> {
        ProximaRecordServiceServer::new(self)
    }

    /// Spawn a background PAX write for the given records and register the resulting
    /// `SegmentMeta` with the shared `SegmentRegistry`.
    ///
    /// This is fire-and-forget: the gRPC response is sent immediately after the
    /// canonical engine write succeeds; the PAX file is written asynchronously.
    fn spawn_pax_write(
        registry: Arc<crate::catalog::SegmentRegistry>,
        collection_id: String,
        records: Vec<proximadb_records::ProximaRecord>,
    ) {
        use proximadb_block_format::{BlockCompression, BlockMode};
        use proximadb_storage_common::collection_path::slug_for;
        use proximadb_storage_common::pax_block::PaxSegmentWriter;

        tokio::task::spawn_blocking(move || {
            let slug = slug_for(&collection_id);
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            let path =
                std::path::PathBuf::from(format!("/tmp/proximadb/pax/{}/{}.pax", slug, now_ms));

            let mut writer = PaxSegmentWriter::new(
                &path,
                BlockMode::Pax,
                BlockCompression::None,
                &collection_id,
                0, // schema_fingerprint: 0 until catalog integration
                0, // embedding_count: 0 for now (metadata-only projection)
                None,
            );

            for record in &records {
                if let Err(e) = writer.add_record(record) {
                    warn!(
                        "PAX write: add_record failed for collection '{}': {}",
                        collection_id, e
                    );
                    return;
                }
            }

            match writer.finish() {
                Ok(meta) => {
                    debug!(
                        "PAX write: {} rows, {} bytes → {:?}",
                        meta.row_count, meta.size_bytes, meta.path
                    );
                    registry.register(collection_id, meta);
                }
                Err(e) => {
                    warn!("PAX write: finish failed: {}", e);
                }
            }
        });
    }

    /// Convert ProximaRecordBatch to the canonical internal rich-record request.
    fn convert_to_rich_batch(
        &self,
        batch: &ProximaRecordBatch,
    ) -> Result<RichRecordBatchRequest, Status> {
        let mut records = Vec::with_capacity(batch.records.len());

        for record in &batch.records {
            let envelope = proto_record_to_envelope(record)
                .map_err(|e| Status::invalid_argument(format!("invalid ProximaRecord: {e}")))?;
            records.push(envelope);
        }

        Ok(RichRecordBatchRequest {
            collection_id: batch.collection_id.clone(),
            records,
        })
    }

    fn batch_result_to_response(
        result: BatchOperationResult,
        record_ids: &[String],
        default_error_code: &str,
    ) -> ProximaRecordBatchResponse {
        let failed_count = result.metrics.failed_count.max(result.errors.len() as i64);
        let total_processed = result
            .metrics
            .total_processed
            .max(result.metrics.successful_count + failed_count);
        let errors = result
            .errors
            .iter()
            .enumerate()
            .map(|(index, error)| BatchError {
                record_index: index as i32,
                record_id: record_ids.get(index).cloned().unwrap_or_default(),
                error_code: result
                    .error_code
                    .clone()
                    .unwrap_or_else(|| default_error_code.to_string()),
                error_message: error.clone(),
            })
            .collect();

        ProximaRecordBatchResponse {
            success: result.success,
            total_processed,
            success_count: result.metrics.successful_count,
            failed_count,
            inserted_ids: result.vector_ids,
            errors,
            processing_time_us: result.metrics.processing_time_us,
        }
    }

    /// Convert search results to TypedSearchResult
    fn convert_search_results(
        &self,
        results: &RichSearchResponse,
        include_vector: bool,
    ) -> Vec<TypedSearchResult> {
        results
            .results
            .iter()
            .map(|r| {
                let props: HashMap<String, proximadb_v2::TypedValue> = r
                    .props
                    .iter()
                    .map(|(k, v)| (k.clone(), proxima_value_to_typed_value(v)))
                    .collect();

                TypedSearchResult {
                    id: r.id.clone(),
                    score: r.score,
                    props,
                    vector: if include_vector {
                        r.vector.clone()
                    } else {
                        vec![]
                    },
                    text_fields: vec![], // Would be populated from text storage
                    timestamp_ms: r.timestamp,
                    version: r.version,
                    source: r.source.clone(),
                }
            })
            .collect()
    }

    /// Calculate backpressure signal based on buffer utilization
    ///
    /// Backpressure levels and thresholds:
    /// - NONE (0-25%): No pressure, client can send freely (delay: 0ms)
    /// - LOW (25-50%): Light pressure, suggest small delays (delay: 10ms)
    /// - MEDIUM (50-75%): Moderate pressure, suggest medium delays (delay: 50ms)
    /// - HIGH (75-90%): Heavy pressure, suggest significant delays (delay: 100ms)
    /// - CRITICAL (90%+): Critical pressure, suggest maximum delays (delay: 500ms)
    ///
    /// The suggested delay is a hint to the client about how long to wait before
    /// sending the next batch. Clients that respect this signal help prevent
    /// buffer overflow and ensure smooth throughput.
    ///
    /// Note: This is the simpler version. For adaptive backpressure with flow control,
    /// use `calculate_dynamic_backpressure` instead.
    #[allow(dead_code)]
    fn calculate_backpressure(buffer_percent: u32) -> BackpressureSignal {
        let (level, suggested_delay_ms) = if buffer_percent >= BACKPRESSURE_CRITICAL_THRESHOLD {
            (BackpressureLevel::BackpressureCritical, DELAY_CRITICAL_MS)
        } else if buffer_percent >= BACKPRESSURE_HIGH_THRESHOLD {
            (BackpressureLevel::BackpressureHigh, DELAY_HIGH_MS)
        } else if buffer_percent >= BACKPRESSURE_MEDIUM_THRESHOLD {
            (BackpressureLevel::BackpressureMedium, DELAY_MEDIUM_MS)
        } else if buffer_percent >= BACKPRESSURE_LOW_THRESHOLD {
            (BackpressureLevel::BackpressureLow, DELAY_LOW_MS)
        } else {
            (BackpressureLevel::BackpressureNone, DELAY_NONE_MS)
        };

        BackpressureSignal {
            level: level as i32,
            suggested_delay_ms,
            buffer_percent,
        }
    }

    /// Calculate dynamic backpressure with adaptive delays
    ///
    /// This version adjusts the suggested delay dynamically based on the
    /// exact buffer utilization percentage, providing smoother backpressure
    /// transitions and better flow control.
    fn calculate_dynamic_backpressure(
        buffer_percent: u32,
        flow_control: &FlowControlState,
        channel_capacity: usize,
    ) -> BackpressureSignal {
        // Get the combined utilization considering both pending items and channel usage
        let combined_util = flow_control.calculate_utilization(channel_capacity.saturating_sub(
            (channel_capacity as f64 * (1.0 - buffer_percent as f64 / 100.0)) as usize,
        ));

        // Use the higher of the two utilization values
        let effective_percent = buffer_percent.max(combined_util);

        // Determine level and base delay
        let (level, base_delay) = if effective_percent >= BACKPRESSURE_CRITICAL_THRESHOLD {
            (BackpressureLevel::BackpressureCritical, DELAY_CRITICAL_MS)
        } else if effective_percent >= BACKPRESSURE_HIGH_THRESHOLD {
            (BackpressureLevel::BackpressureHigh, DELAY_HIGH_MS)
        } else if effective_percent >= BACKPRESSURE_MEDIUM_THRESHOLD {
            (BackpressureLevel::BackpressureMedium, DELAY_MEDIUM_MS)
        } else if effective_percent >= BACKPRESSURE_LOW_THRESHOLD {
            (BackpressureLevel::BackpressureLow, DELAY_LOW_MS)
        } else {
            (BackpressureLevel::BackpressureNone, DELAY_NONE_MS)
        };

        // Calculate adaptive delay within the level's range
        // For smoother transitions, interpolate between level boundaries
        let suggested_delay_ms = match level {
            BackpressureLevel::BackpressureCritical => {
                // Scale from 500ms to 1000ms as we approach 100%
                let scale = (effective_percent - BACKPRESSURE_CRITICAL_THRESHOLD) as f64
                    / (100 - BACKPRESSURE_CRITICAL_THRESHOLD) as f64;
                base_delay + (scale * 500.0) as u32
            }
            BackpressureLevel::BackpressureHigh => {
                // Scale from 100ms to 500ms
                let scale = (effective_percent - BACKPRESSURE_HIGH_THRESHOLD) as f64
                    / (BACKPRESSURE_CRITICAL_THRESHOLD - BACKPRESSURE_HIGH_THRESHOLD) as f64;
                base_delay + (scale * (DELAY_CRITICAL_MS - DELAY_HIGH_MS) as f64) as u32
            }
            BackpressureLevel::BackpressureMedium => {
                // Scale from 50ms to 100ms
                let scale = (effective_percent - BACKPRESSURE_MEDIUM_THRESHOLD) as f64
                    / (BACKPRESSURE_HIGH_THRESHOLD - BACKPRESSURE_MEDIUM_THRESHOLD) as f64;
                base_delay + (scale * (DELAY_HIGH_MS - DELAY_MEDIUM_MS) as f64) as u32
            }
            BackpressureLevel::BackpressureLow => {
                // Scale from 10ms to 50ms
                let scale = (effective_percent - BACKPRESSURE_LOW_THRESHOLD) as f64
                    / (BACKPRESSURE_MEDIUM_THRESHOLD - BACKPRESSURE_LOW_THRESHOLD) as f64;
                base_delay + (scale * (DELAY_MEDIUM_MS - DELAY_LOW_MS) as f64) as u32
            }
            BackpressureLevel::BackpressureNone => DELAY_NONE_MS,
        };

        // Add extra delay if flow control is active (hard limit reached)
        let final_delay = if flow_control.is_active() {
            suggested_delay_ms.max(DELAY_CRITICAL_MS)
        } else {
            suggested_delay_ms
        };

        BackpressureSignal {
            level: level as i32,
            suggested_delay_ms: final_delay,
            buffer_percent: effective_percent,
        }
    }
}

#[tonic::async_trait]
impl ProximaRecordService for ProximaRecordServiceImpl {
    // =========================================================================
    // Record Operations
    // =========================================================================

    /// Insert records into a collection
    async fn insert_records(
        &self,
        request: Request<ProximaRecordBatch>,
    ) -> Result<Response<ProximaRecordBatchResponse>, Status> {
        let tenant_id = grpc_auth::resolved_tenant_id(&request)?;
        grpc_auth::enforce_data_plane_request(
            &request,
            "ingest",
            &request.get_ref().collection_id,
            request.get_ref().records.len(),
            Some(prost::Message::encoded_len(request.get_ref()) as u64),
        )?;
        let batch = request.into_inner();
        info!(
            "V2 gRPC: InsertRecords - collection='{}', records={}",
            batch.collection_id,
            batch.records.len()
        );

        // Validate write mode
        if batch.write_mode != BatchWriteMode::Insert as i32
            && batch.write_mode != BatchWriteMode::Unspecified as i32
        {
            return Err(Status::invalid_argument(
                "InsertRecords requires INSERT or UNSPECIFIED write mode",
            ));
        }

        // Slice 6.1: primary-pod write-router gate. Symmetric with the
        // REST v2 wiring at `src/network/rest/v2/records.rs:1032`. The
        // gate runs AFTER auth + write-mode validation (so a malformed
        // request still gets the right error) and BEFORE the lane
        // router / storage path (so a misroute never touches the WAL).
        // ApiError → tonic::Status conversion at `src/errors/mod.rs:135`
        // adds `x-primary-pod` / `x-tenant-id` / `x-collection-id`
        // trailing metadata so SDKs can parse the redirect target.
        check_primary_pod_gate(&self.primary_pod_gate, &tenant_id, &batch.collection_id)?;

        let record_ids: Vec<String> = batch
            .records
            .iter()
            .map(|record| record.id.clone())
            .collect();
        let rich_batch = self.convert_to_rich_batch(&batch)?;
        let pax_records = if self.segment_registry.is_some() {
            rich_batch.records.clone()
        } else {
            Vec::new()
        };
        let collection_id = rich_batch.collection_id.clone();

        let intent = WriteIntent::new(&collection_id, WriteOperationKind::Insert)
            .with_durability(WriteDurabilityRequirement::WalRequired)
            .with_row_count_hint(record_ids.len() as u64);
        let lane = WriteLaneRouter::new().route(&intent);
        debug!(
            collection_id = %collection_id,
            write_lane = ?lane.lane,
            guards = ?lane.required_guards,
            "gRPC InsertRecords write-lane decision"
        );
        lane.require_wal_lane("gRPC InsertRecords")
            .map_err(|e| Status::invalid_argument(e.to_string()))?;

        match self
            .record_ops
            .handle_record_batch_for_tenant(rich_batch, Some(tenant_id.as_str()))
            .await
        {
            Ok(result) => {
                if let Some(registry) = &self.segment_registry
                    && !pax_records.is_empty()
                {
                    Self::spawn_pax_write(registry.clone(), collection_id, pax_records);
                }
                Ok(Response::new(Self::batch_result_to_response(
                    result,
                    &record_ids,
                    "WRITE_FAILED",
                )))
            }
            Err(e) => {
                error!("V2 gRPC: InsertRecords failed: {}", e);
                Err(Status::internal(format!("Insert failed: {}", e)))
            }
        }
    }

    /// Upsert records (insert or update)
    async fn upsert_records(
        &self,
        request: Request<ProximaRecordBatch>,
    ) -> Result<Response<ProximaRecordBatchResponse>, Status> {
        let tenant_id = grpc_auth::resolved_tenant_id(&request)?;
        grpc_auth::enforce_data_plane_request(
            &request,
            "ingest",
            &request.get_ref().collection_id,
            request.get_ref().records.len(),
            Some(prost::Message::encoded_len(request.get_ref()) as u64),
        )?;
        let batch = request.into_inner();
        info!(
            "V2 gRPC: UpsertRecords - collection='{}', records={}",
            batch.collection_id,
            batch.records.len()
        );

        // Primary-pod write-router gate (symmetric with `insert_records`). Every
        // mutation must gate — not just insert — or a displaced pod silently
        // accepts upserts that the new primary's reader never sees. Runs after
        // auth/validation, before the lane router / storage path.
        check_primary_pod_gate(&self.primary_pod_gate, &tenant_id, &batch.collection_id)?;

        let record_ids: Vec<String> = batch
            .records
            .iter()
            .map(|record| record.id.clone())
            .collect();
        let rich_batch = self.convert_to_rich_batch(&batch)?;
        let pax_records = if self.segment_registry.is_some() {
            rich_batch.records.clone()
        } else {
            Vec::new()
        };
        let collection_id = rich_batch.collection_id.clone();

        let intent = WriteIntent::new(&collection_id, WriteOperationKind::Upsert)
            .with_durability(WriteDurabilityRequirement::WalRequired)
            .with_row_count_hint(record_ids.len() as u64);
        let lane = WriteLaneRouter::new().route(&intent);
        debug!(
            collection_id = %collection_id,
            write_lane = ?lane.lane,
            guards = ?lane.required_guards,
            "gRPC UpsertRecords write-lane decision"
        );
        lane.require_wal_lane("gRPC UpsertRecords")
            .map_err(|e| Status::invalid_argument(e.to_string()))?;

        match self
            .record_ops
            .handle_record_batch_for_tenant(rich_batch, Some(tenant_id.as_str()))
            .await
        {
            Ok(result) => {
                if let Some(registry) = &self.segment_registry
                    && !pax_records.is_empty()
                {
                    Self::spawn_pax_write(registry.clone(), collection_id, pax_records);
                }
                Ok(Response::new(Self::batch_result_to_response(
                    result,
                    &record_ids,
                    "WRITE_FAILED",
                )))
            }
            Err(e) => {
                error!("V2 gRPC: UpsertRecords failed: {}", e);
                Err(Status::internal(format!("Upsert failed: {}", e)))
            }
        }
    }

    /// Update existing records
    async fn update_records(
        &self,
        request: Request<ProximaRecordBatch>,
    ) -> Result<Response<ProximaRecordBatchResponse>, Status> {
        let tenant_id = grpc_auth::resolved_tenant_id(&request)?;
        grpc_auth::enforce_data_plane_request(
            &request,
            "ingest",
            &request.get_ref().collection_id,
            request.get_ref().records.len(),
            Some(prost::Message::encoded_len(request.get_ref()) as u64),
        )?;
        let batch = request.into_inner();
        info!(
            "V2 gRPC: UpdateRecords - collection='{}', records={}",
            batch.collection_id,
            batch.records.len()
        );

        // Primary-pod write-router gate (symmetric with `insert_records`).
        check_primary_pod_gate(&self.primary_pod_gate, &tenant_id, &batch.collection_id)?;

        let record_ids: Vec<String> = batch
            .records
            .iter()
            .map(|record| record.id.clone())
            .collect();
        let rich_batch = self.convert_to_rich_batch(&batch)?;
        let collection_id = rich_batch.collection_id.clone();

        let intent = WriteIntent::new(&collection_id, WriteOperationKind::Update)
            .with_durability(WriteDurabilityRequirement::WalRequired)
            .with_row_count_hint(record_ids.len() as u64);
        let lane = WriteLaneRouter::new().route(&intent);
        debug!(
            collection_id = %collection_id,
            write_lane = ?lane.lane,
            guards = ?lane.required_guards,
            "gRPC UpdateRecords write-lane decision"
        );
        lane.require_wal_lane("gRPC UpdateRecords")
            .map_err(|e| Status::invalid_argument(e.to_string()))?;

        match self
            .record_ops
            .handle_record_batch_for_tenant(rich_batch, Some(tenant_id.as_str()))
            .await
        {
            Ok(result) => Ok(Response::new(Self::batch_result_to_response(
                result,
                &record_ids,
                "WRITE_FAILED",
            ))),
            Err(e) => {
                error!("V2 gRPC: UpdateRecords failed: {}", e);
                Err(Status::internal(format!("Update failed: {}", e)))
            }
        }
    }

    /// Delete records by ID
    async fn delete_records(
        &self,
        request: Request<ProximaRecordBatch>,
    ) -> Result<Response<ProximaRecordBatchResponse>, Status> {
        let tenant_id = grpc_auth::resolved_tenant_id(&request)?;
        grpc_auth::enforce_data_plane_request(
            &request,
            "ingest",
            &request.get_ref().collection_id,
            request.get_ref().records.len(),
            Some(prost::Message::encoded_len(request.get_ref()) as u64),
        )?;
        let batch = request.into_inner();
        info!(
            "V2 gRPC: DeleteRecords - collection='{}', records={}",
            batch.collection_id,
            batch.records.len()
        );

        // Primary-pod write-router gate (symmetric with `insert_records`).
        check_primary_pod_gate(&self.primary_pod_gate, &tenant_id, &batch.collection_id)?;

        let record_ids: Vec<String> = batch
            .records
            .iter()
            .map(|record| record.id.clone())
            .collect();
        if let Some((index, _)) = record_ids
            .iter()
            .enumerate()
            .find(|(_, record_id)| record_id.is_empty())
        {
            return Err(Status::invalid_argument(format!(
                "DeleteRecords requires record id at index {index}"
            )));
        }

        let intent = WriteIntent::new(&batch.collection_id, WriteOperationKind::Delete)
            .with_durability(WriteDurabilityRequirement::WalRequired)
            .with_row_count_hint(record_ids.len() as u64);
        let lane = WriteLaneRouter::new().route(&intent);
        debug!(
            collection_id = %batch.collection_id,
            write_lane = ?lane.lane,
            guards = ?lane.required_guards,
            "gRPC DeleteRecords write-lane decision"
        );
        lane.require_wal_lane("gRPC DeleteRecords")
            .map_err(|e| Status::invalid_argument(e.to_string()))?;

        match self
            .record_ops
            .handle_record_delete_batch_for_tenant(
                RichRecordDeleteBatchRequest {
                    collection_id: batch.collection_id.clone(),
                    record_ids: record_ids.clone(),
                },
                Some(tenant_id.as_str()),
            )
            .await
        {
            Ok(result) => Ok(Response::new(Self::batch_result_to_response(
                result,
                &record_ids,
                "DELETE_FAILED",
            ))),
            Err(e) => {
                error!("V2 gRPC: DeleteRecords failed: {}", e);
                if e.to_string().contains("not found") {
                    Err(Status::not_found(format!(
                        "Collection not found: {}",
                        batch.collection_id
                    )))
                } else {
                    Err(Status::internal(format!("Delete failed: {}", e)))
                }
            }
        }
    }

    // =========================================================================
    // Search Operations
    // =========================================================================

    /// Search with typed filters
    async fn search(
        &self,
        request: Request<TypedSearchRequest>,
    ) -> Result<Response<TypedSearchResponse>, Status> {
        let identity = grpc_auth::port_identity(&request)?;
        grpc_auth::enforce_data_plane_request(
            &request,
            "search",
            &request.get_ref().collection_id,
            request.get_ref().top_k as usize,
            Some(prost::Message::encoded_len(request.get_ref()) as u64),
        )?;
        // After auth: unauthenticated requests get the auth error, not the cap.
        check_search_top_k(request.get_ref().top_k)?;
        let req = request.into_inner();
        debug!(
            "V2 gRPC: Search - collection='{}', top_k={}",
            req.collection_id, req.top_k
        );

        let search_request = RichSearchRequest {
            collection_id: req.collection_id.clone(),
            query_vector: req.query_vector.clone(),
            top_k: req.top_k,
            filters: grpc_filters_to_rich(&req.filters)?,
        };

        // TD-064: scope the search in a predicate-diagnostics context so
        // any AxisManager-deep shortfall is captured here. Phase K: also
        // lift the TurboQuant EXPLAIN payload — task-local binding ends
        // when the scoped future returns, so the take MUST happen inside.
        let (search_outcome, _turboquant_hints) =
            crate::observability::predicate_diagnostics::scope(async {
                let outcome = self
                    .record_ops
                    .handle_record_search_for_tenant(search_request, identity.as_borrowed())
                    .await;
                let tq = crate::observability::predicate_diagnostics::take_turboquant_hints();
                (outcome, tq)
            })
            .await;

        match search_outcome {
            Ok(resp) => {
                let include_vector = req.include_vector;

                let results = self.convert_search_results(&resp, include_vector);
                let total_found = resp.total_found;

                // TD-064: typed, top-level shortfall carried on the service
                // response (`resp.predicate_shortfall`), recomputed against the
                // final WAL+AXIS+storage merge. Replaces the old `search_stats`
                // map stuffing — that path called `take_shortfall()` OUTSIDE
                // the scope above (task-local binding already ended), so it was
                // always empty; this reads the authoritative field instead.
                let predicate_shortfall = resp.predicate_shortfall.map(|sf| PredicateShortfall {
                    requested_k: sf.requested_k,
                    returned_k: sf.returned_k,
                    oversample_pool: sf.oversample_pool,
                    ann_filtering_mode: sf.ann_filtering_mode,
                });

                Ok(Response::new(TypedSearchResponse {
                    results,
                    total_found,
                    search_time_us: 0, // Would need timing
                    collection_id: Some(req.collection_id),
                    search_stats: HashMap::new(),
                    predicate_shortfall,
                }))
            }
            Err(e) if crate::services::operations::vectors::is_client_argument_error(&e) => {
                // Invalid query vector or over-cap top_k (TD-SECSCAN-1).
                Err(Status::invalid_argument(e.to_string()))
            }
            Err(e) => {
                error!("V2 gRPC: Search failed: {}", e);
                if e.to_string().contains("not found") {
                    Err(Status::not_found(format!(
                        "Collection not found: {}",
                        req.collection_id
                    )))
                } else {
                    Err(Status::internal(format!("Search failed: {}", e)))
                }
            }
        }
    }

    /// Streaming search - returns results as a stream
    type SearchStreamStream = SearchStreamStream;

    async fn search_stream(
        &self,
        request: Request<TypedSearchRequest>,
    ) -> Result<Response<Self::SearchStreamStream>, Status> {
        let identity = grpc_auth::port_identity(&request)?;
        grpc_auth::enforce_data_plane_request(
            &request,
            "search",
            &request.get_ref().collection_id,
            request.get_ref().top_k as usize,
            Some(prost::Message::encoded_len(request.get_ref()) as u64),
        )?;
        // After auth: unauthenticated requests get the auth error, not the cap.
        check_search_top_k(request.get_ref().top_k)?;
        let req = request.into_inner();
        debug!(
            "V2 gRPC: SearchStream - collection='{}', top_k={}",
            req.collection_id, req.top_k
        );

        let search_request = RichSearchRequest {
            collection_id: req.collection_id.clone(),
            query_vector: req.query_vector.clone(),
            top_k: req.top_k,
            filters: grpc_filters_to_rich(&req.filters)?,
        };

        let include_vector = req.include_vector;

        // Execute search
        let response = self
            .record_ops
            .handle_record_search_for_tenant(search_request, identity.as_borrowed())
            .await
            .map_err(|e| {
                if crate::services::operations::vectors::is_client_argument_error(&e) {
                    Status::invalid_argument(e.to_string())
                } else {
                    Status::internal(format!("Search stream failed: {}", e))
                }
            })?;

        // Create a channel for streaming results
        let (tx, rx) = tokio::sync::mpsc::channel(128);

        // Convert results
        let results = self.convert_search_results(&response, include_vector);

        // Spawn a task to send results through the channel
        tokio::spawn(async move {
            for result in results {
                if tx.send(Ok(result)).await.is_err() {
                    // Client disconnected
                    break;
                }
            }
        });

        let stream = ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(stream) as SearchStreamStream))
    }

    // =========================================================================
    // Streaming Write Operations
    // =========================================================================

    /// Bidirectional streaming for high-throughput batch writes
    ///
    /// This method provides:
    /// - Client streaming: Send batches of records continuously
    /// - Server streaming: Receive acknowledgments with backpressure signals
    /// - Flow control: Bounded channels prevent memory overflow
    /// - Dynamic backpressure: Multi-level pressure signals with adaptive delays
    /// - Latency metrics: Pipeline timing for observability
    /// - Error handling: Per-record errors are reported back to the client
    ///
    /// ## Backpressure Levels
    ///
    /// The server communicates backpressure to clients through the `BackpressureSignal`:
    /// - **NONE (0-25%)**: No pressure, client can send freely
    /// - **LOW (25-50%)**: Light pressure, suggest 10-50ms delays
    /// - **MEDIUM (50-75%)**: Moderate pressure, suggest 50-100ms delays
    /// - **HIGH (75-90%)**: Heavy pressure, suggest 100-500ms delays
    /// - **CRITICAL (90%+)**: Critical pressure, suggest 500-1000ms delays
    ///
    /// Clients that respect the `suggested_delay_ms` help maintain smooth throughput
    /// and prevent buffer overflow.
    type BatchWriteStreamStream = BatchWriteStreamStream;

    async fn batch_write_stream(
        &self,
        request: Request<Streaming<BatchWriteStreamRequest>>,
    ) -> Result<Response<Self::BatchWriteStreamStream>, Status> {
        let tenant_id = grpc_auth::resolved_tenant_id(&request)?;
        let data_plane_capability = grpc_auth::data_plane_capability(&request);
        let mut inbound = request.into_inner();

        // Create a bounded channel for response streaming with flow control
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<BatchWriteStreamResponse, Status>>(
            STREAM_BUFFER_SIZE,
        );

        // Shared state for tracking progress
        let total_processed = Arc::new(AtomicI64::new(0));
        let success_count = Arc::new(AtomicI64::new(0));
        let failed_count = Arc::new(AtomicI64::new(0));

        // Flow control state for managing backpressure
        let flow_control = Arc::new(FlowControlState::new(STREAM_BUFFER_SIZE));

        // Pipeline metrics for observability
        let metrics = Arc::new(StreamingPipelineMetrics::new());

        // Clone record-ops for the processing task (the streaming write path
        // drives batch insert/delete through RecordOpsService — TD-104).
        let record_ops = Arc::clone(&self.record_ops);
        let tenant_id = tenant_id.clone();
        let data_plane_capability = data_plane_capability.clone();

        // Clone metrics for the spawned task
        let metrics_clone = Arc::clone(&metrics);
        let flow_control_clone = Arc::clone(&flow_control);

        // Spawn task to process incoming stream
        tokio::spawn(async move {
            let stream_start_time = Instant::now();
            let metrics = metrics_clone;
            let flow_control = flow_control_clone;

            while let Some(batch_result) = inbound.next().await {
                let batch_receive_time = Instant::now();

                let batch = match batch_result {
                    Ok(b) => b,
                    Err(e) => {
                        error!("V2 gRPC: BatchWriteStream - stream error: {}", e);
                        let _ = tx
                            .send(Err(Status::internal(format!("Stream error: {}", e))))
                            .await;
                        break;
                    }
                };

                let batch_size = batch.records.len();
                if let Some(capability) = data_plane_capability.as_ref()
                    && let Err(status) = grpc_auth::validate_data_plane_capability(
                        capability,
                        "ingest",
                        &batch.collection_id,
                        batch_size,
                        Some(prost::Message::encoded_len(&batch) as u64),
                    )
                {
                    let _ = tx.send(Err(status)).await;
                    break;
                }
                debug!(
                    "V2 gRPC: BatchWriteStream - processing batch for collection='{}', records={}",
                    batch.collection_id, batch_size
                );

                // Track pending items for flow control
                let flow_control_activated = flow_control.add_pending(batch_size as u32);
                if flow_control_activated {
                    warn!(
                        "V2 gRPC: BatchWriteStream - flow control activated, pending items: {}",
                        flow_control.pending()
                    );
                }

                // Update max queue depth metric
                metrics.update_max_queue_depth(flow_control.pending());

                // Calculate queue wait time (time from receive to start processing)
                let queue_wait_time = batch_receive_time.elapsed();
                metrics.record_queue_wait(queue_wait_time);

                // Route batch intent — individual records may carry per-record modes.
                let stream_intent =
                    WriteIntent::new(&batch.collection_id, WriteOperationKind::Upsert)
                        .with_durability(WriteDurabilityRequirement::WalRequired)
                        .with_row_count_hint(batch_size as u64);
                let stream_lane = WriteLaneRouter::new().route(&stream_intent);
                debug!(
                    collection_id = %batch.collection_id,
                    write_lane = ?stream_lane.lane,
                    guards = ?stream_lane.required_guards,
                    batch_size,
                    "gRPC BatchWriteStream write-lane decision"
                );
                if let Err(e) = stream_lane.require_wal_lane("gRPC BatchWriteStream") {
                    error!("Rejecting non-WAL lane in streaming batch: {}", e);
                    break;
                }

                // Start processing timer
                let processing_start = Instant::now();

                // Collect sequences for acknowledgment
                let mut acked_sequences: Vec<u64> = Vec::with_capacity(batch_size);
                let mut batch_errors: Vec<BatchError> = Vec::new();

                // Process the batch with OPTIMISTIC GROUPING: consecutive
                // records with the same write mode go through the write
                // pipeline as ONE call. The pipeline's fixed per-call costs
                // (tenant-context load, collection resolution, schema
                // validation, WAL-lane gate, lease check, cache
                // invalidation) were previously paid PER RECORD — O(N)
                // round-trips of fixed work. The batch API is aggregate-only
                // (no per-record attribution on failure), so any group that
                // fails is retried record-by-record below, reproducing the
                // exact per-record outcomes of the old one-call-per-record
                // path; every mode is idempotent by oid, so a retry after a
                // partially-applied group failure converges to the same
                // state.
                let records = &batch.records;
                let mut run_start = 0usize;
                while run_start < records.len() {
                    let run_mode = BatchWriteMode::try_from(records[run_start].write_mode);
                    let mut run_end = run_start + 1;
                    while run_end < records.len()
                        && BatchWriteMode::try_from(records[run_end].write_mode) == run_mode
                    {
                        run_end += 1;
                    }
                    let run = &records[run_start..run_end];

                    // Convert the run's records up front; MISSING_RECORD and
                    // INVALID_RECORD keep their per-record errors exactly as
                    // the one-call-per-record path produced them.
                    let mut envelopes: Vec<(usize, proximadb_records::ProximaRecord)> =
                        Vec::with_capacity(run.len());
                    for (offset, stream_record) in run.iter().enumerate() {
                        let idx = run_start + offset;
                        total_processed.fetch_add(1, Ordering::SeqCst);
                        match &stream_record.record {
                            None => {
                                batch_errors.push(BatchError {
                                    record_index: idx as i32,
                                    record_id: String::new(),
                                    error_code: "MISSING_RECORD".to_string(),
                                    error_message: "StreamWriteRecord has no record".to_string(),
                                });
                                failed_count.fetch_add(1, Ordering::SeqCst);
                            }
                            Some(record) => match proto_record_to_envelope(record) {
                                Ok(envelope) => envelopes.push((idx, envelope)),
                                Err(e) => {
                                    batch_errors.push(BatchError {
                                        record_index: idx as i32,
                                        record_id: record.id.clone(),
                                        error_code: "INVALID_RECORD".to_string(),
                                        error_message: e.to_string(),
                                    });
                                    failed_count.fetch_add(1, Ordering::SeqCst);
                                }
                            },
                        }
                    }

                    if envelopes.is_empty() {
                        run_start = run_end;
                        continue;
                    }

                    // ONE pipeline call for the whole run (moves the
                    // envelopes — no per-record clones on the happy path).
                    let member_indices: Vec<usize> =
                        envelopes.iter().map(|(idx, _)| *idx).collect();
                    let group_result = match run_mode {
                        Ok(BatchWriteMode::Insert)
                        | Ok(BatchWriteMode::Unspecified)
                        | Ok(BatchWriteMode::Upsert)
                        | Ok(BatchWriteMode::Update) => {
                            record_ops
                                .handle_record_batch_for_tenant(
                                    RichRecordBatchRequest {
                                        collection_id: batch.collection_id.clone(),
                                        records: envelopes.into_iter().map(|(_, e)| e).collect(),
                                    },
                                    Some(tenant_id.as_str()),
                                )
                                .await
                        }
                        Ok(BatchWriteMode::Delete) => {
                            record_ops
                                .handle_record_delete_batch_for_tenant(
                                    RichRecordDeleteBatchRequest {
                                        collection_id: batch.collection_id.clone(),
                                        record_ids: member_indices
                                            .iter()
                                            .filter_map(|idx| {
                                                records[*idx].record.as_ref().map(|r| r.id.clone())
                                            })
                                            .collect(),
                                    },
                                    Some(tenant_id.as_str()),
                                )
                                .await
                        }
                        Err(_) => Err(anyhow::anyhow!(
                            "Invalid write_mode: {}",
                            records[run_start].write_mode
                        )),
                    };

                    if matches!(&group_result, Ok(resp) if resp.success) {
                        // Whole run acknowledged at once.
                        for idx in &member_indices {
                            acked_sequences.push(records[*idx].client_sequence);
                            success_count.fetch_add(1, Ordering::SeqCst);
                        }
                        trace!(
                            "V2 gRPC: batched {} same-mode records in one pipeline call",
                            member_indices.len()
                        );
                    } else {
                        // Aggregate failure — the batch API carries no
                        // per-record attribution, so retry each member
                        // individually. This reproduces the previous
                        // one-call-per-record outcomes exactly; every mode is
                        // idempotent by oid, so re-applying records after a
                        // partially-applied group failure converges to the
                        // same state.
                        for idx in &member_indices {
                            let stream_record = &records[*idx];
                            let Some(record) = &stream_record.record else {
                                // Conversion succeeded above, so this arm is
                                // unreachable; fail loudly rather than ack.
                                batch_errors.push(BatchError {
                                    record_index: *idx as i32,
                                    record_id: String::new(),
                                    error_code: "INTERNAL_ERROR".to_string(),
                                    error_message: "record vanished during retry".to_string(),
                                });
                                failed_count.fetch_add(1, Ordering::SeqCst);
                                continue;
                            };
                            let single_result = match run_mode {
                                Ok(BatchWriteMode::Insert)
                                | Ok(BatchWriteMode::Unspecified)
                                | Ok(BatchWriteMode::Upsert)
                                | Ok(BatchWriteMode::Update) => {
                                    match proto_record_to_envelope(record) {
                                        Ok(envelope) => {
                                            record_ops
                                                .handle_record_batch_for_tenant(
                                                    RichRecordBatchRequest {
                                                        collection_id: batch.collection_id.clone(),
                                                        records: vec![envelope],
                                                    },
                                                    Some(tenant_id.as_str()),
                                                )
                                                .await
                                        }
                                        Err(e) => Err(anyhow::anyhow!(e.to_string())),
                                    }
                                }
                                Ok(BatchWriteMode::Delete) => {
                                    record_ops
                                        .handle_record_delete_batch_for_tenant(
                                            RichRecordDeleteBatchRequest {
                                                collection_id: batch.collection_id.clone(),
                                                record_ids: vec![record.id.clone()],
                                            },
                                            Some(tenant_id.as_str()),
                                        )
                                        .await
                                }
                                Err(_) => Err(anyhow::anyhow!(
                                    "Invalid write_mode: {}",
                                    stream_record.write_mode
                                )),
                            };

                            match single_result {
                                Ok(resp) if resp.success => {
                                    acked_sequences.push(stream_record.client_sequence);
                                    success_count.fetch_add(1, Ordering::SeqCst);
                                }
                                Ok(resp) => {
                                    batch_errors.push(BatchError {
                                        record_index: *idx as i32,
                                        record_id: record.id.clone(),
                                        error_code: "WRITE_FAILED".to_string(),
                                        error_message: if resp.errors.is_empty() {
                                            "Unknown error".to_string()
                                        } else {
                                            resp.errors.join("; ")
                                        },
                                    });
                                    failed_count.fetch_add(1, Ordering::SeqCst);
                                }
                                Err(e) => {
                                    batch_errors.push(BatchError {
                                        record_index: *idx as i32,
                                        record_id: record.id.clone(),
                                        error_code: "INTERNAL_ERROR".to_string(),
                                        error_message: e.to_string(),
                                    });
                                    failed_count.fetch_add(1, Ordering::SeqCst);
                                }
                            }
                        }
                    }

                    run_start = run_end;
                }

                // Record processing time for the batch
                let processing_time = processing_start.elapsed();
                metrics.record_processing_time(processing_time);

                // Update pending count after processing
                flow_control.remove_pending(batch_size as u32);

                // Update metrics
                metrics.increment_batches();
                metrics.add_records(batch_size as u64);

                // Calculate dynamic backpressure based on channel utilization and flow control state
                let channel_remaining = tx.capacity();
                let buffer_usage =
                    ((STREAM_BUFFER_SIZE - channel_remaining) * 100 / STREAM_BUFFER_SIZE) as u32;

                let backpressure = Self::calculate_dynamic_backpressure(
                    buffer_usage,
                    &flow_control,
                    STREAM_BUFFER_SIZE,
                );

                // Record backpressure event if not NONE
                if backpressure.level != BackpressureLevel::BackpressureNone as i32
                    && let Ok(level) = BackpressureLevel::try_from(backpressure.level)
                {
                    metrics.record_backpressure(level);

                    // Log significant backpressure events
                    if backpressure.level >= BackpressureLevel::BackpressureHigh as i32 {
                        warn!(
                            "V2 gRPC: BatchWriteStream - high backpressure: level={:?}, buffer={}%, delay={}ms",
                            level, backpressure.buffer_percent, backpressure.suggested_delay_ms
                        );
                    }
                }

                // Prepare acknowledgment response
                let response = BatchWriteStreamResponse {
                    acked_sequences,
                    backpressure: Some(backpressure),
                    total_processed: total_processed.load(Ordering::SeqCst),
                    success_count: success_count.load(Ordering::SeqCst),
                    failed_count: failed_count.load(Ordering::SeqCst),
                    errors: batch_errors,
                    server_timestamp_ms: chrono::Utc::now().timestamp_millis(),
                };

                // Send acknowledgment response and track timing
                let ack_send_start = Instant::now();
                let send_result = tx.send(Ok(response)).await;
                metrics.record_ack_send_time(ack_send_start.elapsed());

                if send_result.is_err() {
                    debug!("V2 gRPC: BatchWriteStream - client disconnected");
                    break;
                }

                // If flow control is active, yield to allow other tasks to progress
                if flow_control.is_active() {
                    tokio::task::yield_now().await;
                }
            }

            // Log final metrics summary
            info!(
                "V2 gRPC: BatchWriteStream - completed. Total processed: {}, Success: {}, Failed: {}, Duration: {:?}, Metrics: {}",
                total_processed.load(Ordering::SeqCst),
                success_count.load(Ordering::SeqCst),
                failed_count.load(Ordering::SeqCst),
                stream_start_time.elapsed(),
                metrics.summary()
            );
        });

        let stream = ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(stream) as BatchWriteStreamStream))
    }

    // =========================================================================
    // Schema Operations
    // =========================================================================

    /// Create a new schema for a collection
    async fn create_schema(
        &self,
        request: Request<CreateSchemaRequest>,
    ) -> Result<Response<CreateSchemaResponse>, Status> {
        let tenant_id = grpc_auth::resolved_tenant_id(&request)?;
        let req = request.into_inner();
        info!("V2 gRPC: CreateSchema - collection='{}'", req.collection_id);

        let schema = req
            .schema
            .ok_or_else(|| Status::invalid_argument("schema is required"))?;
        let update = grpc_record_schema_to_runtime_update(&req.collection_id, schema)?;
        let schema_id = update.schema_id.clone();

        self.api_handlers
            .update_collection_schema_metadata(&req.collection_id, update, Some(tenant_id.as_str()))
            .await
            .map_err(|e| {
                if let Some(rejected) =
                    crate::storage::metadata::collection_mapping::schema_declaration_rejection(&e)
                {
                    // TD-RESTGET-1: the caller's schema, not the server.
                    Status::invalid_argument(rejected.to_string())
                } else if e.to_string().contains("not found") {
                    Status::not_found(format!("Collection not found: {}", req.collection_id))
                } else {
                    Status::internal(format!("Failed to create schema: {}", e))
                }
            })?;

        Ok(Response::new(CreateSchemaResponse {
            success: true,
            schema_id,
            error_message: None,
        }))
    }

    /// Get schema for a collection
    async fn get_schema(
        &self,
        request: Request<GetSchemaRequest>,
    ) -> Result<Response<GetSchemaResponse>, Status> {
        let tenant_id = grpc_auth::resolved_tenant_id(&request)?;
        let req = request.into_inner();
        debug!("V2 gRPC: GetSchema - collection='{}'", req.collection_id);

        let metadata = self
            .api_handlers
            .get_collection_schema_metadata(&req.collection_id, Some(tenant_id.as_str()))
            .await
            .map_err(|e| {
                if e.to_string().contains("not found") {
                    Status::not_found(format!("Collection not found: {}", req.collection_id))
                } else {
                    Status::internal(format!("Failed to get collection schema: {}", e))
                }
            })?
            .ok_or_else(|| {
                Status::not_found(format!("Collection not found: {}", req.collection_id))
            })?;

        // TD-GRPCSCHEMA-1: the persisted schema is the authority (ADR-047) —
        // serve it regardless of the ProximaRecord flag, which gates record
        // ENFORCEMENT on writes, not the existence of a schema. The flag is
        // settable at create and surfaced below since TD-GRPCSCHEMA-2.
        let schema = Some(runtime_metadata_to_grpc_schema(&metadata));

        Ok(Response::new(GetSchemaResponse {
            success: true,
            schema,
            error_message: None,
            // TD-GRPCSCHEMA-2: surface the mode — the schema is served either
            // way; this says whether write enforcement is ON.
            record_enabled: Some(metadata.enabled),
        }))
    }

    /// List all schemas for a collection
    async fn list_schemas(
        &self,
        request: Request<ListSchemasRequest>,
    ) -> Result<Response<ListSchemasResponse>, Status> {
        let tenant_id = grpc_auth::resolved_tenant_id(&request)?;
        let req = request.into_inner();
        debug!("V2 gRPC: ListSchemas - collection='{}'", req.collection_id);

        let metadata = self
            .api_handlers
            .get_collection_schema_metadata(&req.collection_id, Some(tenant_id.as_str()))
            .await
            .map_err(|e| {
                if e.to_string().contains("not found") {
                    Status::not_found(format!("Collection not found: {}", req.collection_id))
                } else {
                    Status::internal(format!("Failed to list schemas: {}", e))
                }
            })?;

        // TD-GRPCSCHEMA-1: same rule as GetSchema — the persisted schema is
        // served regardless of the ProximaRecord flag.
        let schemas = metadata
            .map(|metadata| vec![runtime_metadata_to_grpc_schema(&metadata)])
            .unwrap_or_default();
        let total_count = schemas.len() as i64;
        Ok(Response::new(ListSchemasResponse {
            schemas,
            total_count,
        }))
    }

    /// Evolve schema with compatibility checks
    async fn evolve_schema(
        &self,
        request: Request<EvolveSchemaRequest>,
    ) -> Result<Response<EvolveSchemaResponse>, Status> {
        let tenant_id = grpc_auth::resolved_tenant_id(&request)?;
        let req = request.into_inner();
        info!(
            "V2 gRPC: EvolveSchema - collection='{}', base_schema='{}'",
            req.collection_id, req.base_schema_id
        );

        let new_schema = req
            .new_schema
            .ok_or_else(|| Status::invalid_argument("new_schema is required"))?;
        // Reuse the CreateSchema conversion — it type-checks every column
        // (`grpc_column_type_to_proxima_type`), so a dry run still rejects a
        // malformed schema before reporting success.
        let update = grpc_record_schema_to_runtime_update(&req.collection_id, new_schema)?;
        let new_schema_id = update.schema_id.clone();

        if req.dry_run {
            debug!("V2 gRPC: EvolveSchema - dry run; validating without persisting");
            return Ok(Response::new(EvolveSchemaResponse {
                success: true,
                new_schema_id,
                records_to_migrate: 0,
                compatibility_warnings: Vec::new(),
                error_message: None,
            }));
        }

        if req.migrate_existing {
            // The schema evolves (persisted below); real per-record migration is
            // deferred. Be honest about it so callers don't assume records moved.
            warn!(
                "V2 gRPC: EvolveSchema: migrate_existing requested for '{}' but record \
                 migration is not yet implemented — schema will evolve, existing records \
                 are not rewritten",
                req.collection_id
            );
        }

        // Persist the evolved canonical schema to the catalog sidecar (authority flows
        // through the runtime `UnifiedHandlers` impl of `ApiHandlersPort`).
        self.api_handlers
            .update_collection_schema_metadata(&req.collection_id, update, Some(tenant_id.as_str()))
            .await
            .map_err(|e| {
                if let Some(rejected) =
                    crate::storage::metadata::collection_mapping::schema_declaration_rejection(&e)
                {
                    // TD-RESTGET-1: the caller's schema, not the server.
                    Status::invalid_argument(rejected.to_string())
                } else if e.to_string().contains("not found") {
                    Status::not_found(format!("Collection not found: {}", req.collection_id))
                } else {
                    Status::internal(format!("Failed to evolve schema: {}", e))
                }
            })?;

        Ok(Response::new(EvolveSchemaResponse {
            success: true,
            new_schema_id,
            records_to_migrate: 0,
            compatibility_warnings: Vec::new(),
            error_message: None,
        }))
    }

    // ---- Collection management (v2 parity with v1 CollectionService) ----
    // Delegate to the same UnifiedHandlers collection path the v1/REST surfaces
    // use; map V2 messages <-> v1 CollectionConfig/Collection.

    async fn create_collection(
        &self,
        request: Request<V2CollectionConfig>,
    ) -> Result<Response<V2Collection>, Status> {
        let tenant_id = grpc_auth::resolved_tenant_id(&request)?;
        let cfg = request.into_inner();
        info!("V2 gRPC: CreateCollection - name='{}'", cfg.name);

        let v1_config = crate::proto::proximadb_v1::CollectionConfig {
            name: cfg.name.clone(),
            dimension: cfg.dimension,
            distance_metric: Some(distance_metric_from_str(&cfg.distance_metric)),
            storage_engine: Some(storage_engine_from_str(&cfg.storage_engine)),
            filterable_columns: cfg
                .filterable_columns
                .iter()
                .map(|n| crate::proto::proximadb_v1::FilterableColumnSpec {
                    name: n.clone(),
                    ..Default::default()
                })
                .collect(),
            description: if cfg.description.is_empty() {
                None
            } else {
                Some(cfg.description.clone())
            },
            index_configs: cfg
                .index_specs
                .iter()
                .map(|s| crate::proto::proximadb_v1::IndexConfig {
                    index_name: if s.algorithm.is_empty() {
                        "default".to_string()
                    } else {
                        s.algorithm.clone()
                    },
                    algorithm: indexing_algorithm_from_str(&s.algorithm),
                    hnsw_config: s
                        .hnsw
                        .as_ref()
                        .map(|h| crate::proto::proximadb_v1::HnswConfig {
                            m: h.m,
                            ef_construction: h.ef_construction,
                            ef_search: h.ef_search,
                            ..Default::default()
                        }),
                    ivf_config: s
                        .ivf
                        .as_ref()
                        .map(|i| crate::proto::proximadb_v1::IvfConfig {
                            n_lists: i.n_lists,
                            n_probe: i.n_probe,
                            ..Default::default()
                        }),
                    is_primary: Some(s.is_primary),
                    ..Default::default()
                })
                .collect(),
            quantization: cfg.quantization.as_ref().map(|q| {
                crate::proto::proximadb_v1::QuantizationConfig {
                    enabled: Some(q.enabled),
                    strategy: Some(quantization_strategy_from_str(&q.strategy)),
                    ..Default::default()
                }
            }),
            ..Default::default()
        };
        // TD-GRPCSCHEMA-2: record mode at create (REST parity).
        let mut v1_config = v1_config;
        v1_config.enable_proxima_record = cfg.enable_proxima_record;
        // TD-TIER-1 SC1 (review round 1): the v2 create MUST NOT silently drop
        // the desired storage class — an unknown label is an actionable
        // InvalidArgument here, and a known label flows into the v1 config so
        // manager admission applies identically to REST, gRPC and Flight.
        v1_config.storage_class = v2_storage_class_field(&cfg)?;
        // TD-TIER-1 SC5: same strict parse as REST; admission is the manager's.
        v1_config.default_access_tier = v2_access_tier_field(&cfg)?;

        let req = CollectionRequest {
            operation: CollectionOperation::CollectionCreate as i32,
            collection_id: Some(cfg.name.clone()),
            collection_config: Some(v1_config),
            query_params: Default::default(),
            options: Default::default(),
            migration_config: Default::default(),
        };
        let resp = self
            .api_handlers
            .handle_collection_operation_for_tenant(req, Some(tenant_id.as_str()))
            .await
            .map_err(|e| Status::internal(format!("CreateCollection failed: {e}")))?;
        let coll = resp
            .collection
            .ok_or_else(|| Status::internal("CreateCollection returned no collection"))?;

        // ADR-047 / collection-as-table: make the catalog (canonical schema sidecar)
        // the authority for this v2-created collection. Storage was provisioned above
        // via the v1 bridge; persist a minimal canonical schema derived from the
        // V2CollectionConfig so v2 GetSchema returns the catalog-authoritative view
        // (richer/multi-vector schemas can be layered on via CreateSchema). The
        // authority flows through the runtime `UnifiedHandlers` impl of
        // `ApiHandlersPort` (the sole production handler since TD-104 S3-f
        // deleted the legacy root twin).
        // Best-effort: the read path already tolerates an absent sidecar (legacy
        // fallback), so a failure here must not fail the already-provisioned collection.
        let canonical_update = CollectionSchemaUpdate {
            schema_id: format!("schema_{}", cfg.name),
            schema_version: "1.0.0".to_string(),
            enforcement: CollectionSchemaEnforcement::Hybrid,
            auto_evolve: true,
            columns: v2_config_to_canonical_columns(&cfg),
        };
        if let Err(e) = self
            .api_handlers
            .update_collection_schema_metadata_with_record_mode(
                &cfg.name,
                canonical_update,
                Some(tenant_id.as_str()),
                // The caller's create-time choice owns the mode; the canonical
                // persist step is persistence, not a schema update that turns
                // record mode on (TD-GRPCSCHEMA-2). UNSET means OFF — REST
                // create's default, and the behavior every pre-field gRPC
                // client had (round 1 review: None would have meant ON).
                Some(cfg.enable_proxima_record.unwrap_or(false)),
            )
            .await
        {
            warn!(
                "V2 CreateCollection: catalog-authoritative schema persist failed for '{}' \
                 (storage provisioned; v1-derived view remains usable): {e}",
                cfg.name
            );
        }

        Ok(Response::new(collection_to_v2(coll)))
    }

    async fn get_collection(
        &self,
        request: Request<V2GetCollectionRequest>,
    ) -> Result<Response<V2Collection>, Status> {
        let tenant_id = grpc_auth::resolved_tenant_id(&request)?;
        let req_in = request.into_inner();
        let req = CollectionRequest {
            operation: CollectionOperation::CollectionGet as i32,
            collection_id: Some(req_in.collection_id.clone()),
            collection_config: None,
            query_params: Default::default(),
            options: Default::default(),
            migration_config: Default::default(),
        };
        let resp = self
            .api_handlers
            .handle_collection_operation_for_tenant(req, Some(tenant_id.as_str()))
            .await
            .map_err(|e| Status::internal(format!("GetCollection failed: {e}")))?;
        let coll = resp.collection.ok_or_else(|| {
            Status::not_found(format!("Collection not found: {}", req_in.collection_id))
        })?;
        Ok(Response::new(collection_to_v2(coll)))
    }

    async fn list_collections(
        &self,
        request: Request<V2ListCollectionsRequest>,
    ) -> Result<Response<V2ListCollectionsResponse>, Status> {
        let tenant_id = grpc_auth::resolved_tenant_id(&request)?;
        let _ = request.into_inner();
        let req = CollectionRequest {
            operation: CollectionOperation::CollectionList as i32,
            collection_id: None,
            collection_config: None,
            query_params: Default::default(),
            options: Default::default(),
            migration_config: Default::default(),
        };
        let resp = self
            .api_handlers
            .handle_collection_operation_for_tenant(req, Some(tenant_id.as_str()))
            .await
            .map_err(|e| Status::internal(format!("ListCollections failed: {e}")))?;
        let collections = resp.collections.into_iter().map(collection_to_v2).collect();
        Ok(Response::new(V2ListCollectionsResponse { collections }))
    }

    async fn delete_collection(
        &self,
        request: Request<V2DeleteCollectionRequest>,
    ) -> Result<Response<V2DeleteCollectionResponse>, Status> {
        let tenant_id = grpc_auth::resolved_tenant_id(&request)?;
        let req_in = request.into_inner();
        info!("V2 gRPC: DeleteCollection - id='{}'", req_in.collection_id);
        let req = CollectionRequest {
            operation: CollectionOperation::CollectionDelete as i32,
            collection_id: Some(req_in.collection_id.clone()),
            collection_config: None,
            query_params: Default::default(),
            options: Default::default(),
            migration_config: Default::default(),
        };
        let resp = self
            .api_handlers
            .handle_collection_operation_for_tenant(req, Some(tenant_id.as_str()))
            .await
            .map_err(|e| {
                // A drop that failed leaves the collection in place: the
                // caller must retry, so report it as such (never success).
                match proximadb_search_types::client_argument::collection_drop_failed(&e) {
                    Some(failed) => Status::unavailable(failed.to_string()),
                    None => Status::internal(format!("DeleteCollection failed: {e}")),
                }
            })?;
        // Nothing deleted: no such collection for this tenant (missing and
        // another tenant's are answered alike). It used to say success.
        if !resp.success {
            return Err(Status::not_found(format!(
                "Collection '{}' not found",
                req_in.collection_id
            )));
        }
        Ok(Response::new(V2DeleteCollectionResponse { success: true }))
    }

    // ---- SQL query (v2 parity with v1 QueryService.ExecuteQuery) ----

    async fn execute_query(
        &self,
        request: Request<V2QueryRequest>,
    ) -> Result<Response<V2QueryResponse>, Status> {
        // TD-ABAC-5: capture the authenticated subject before the request is moved;
        // threaded to ABAC enforcement at the relational read boundary.
        let identity = grpc_auth::port_identity(&request)?;
        let q = request.into_inner();
        let collection = if q.collection_id.is_empty() {
            None
        } else {
            Some(q.collection_id.clone())
        };
        let resp = self
            .api_handlers
            .execute_sql(q.query, None, collection, identity.as_borrowed())
            .await
            .map_err(|e| {
                // A DML lock conflict → ABORTED (retryable); other errors → INTERNAL.
                if let Some((resource, holder)) = crate::errors::extract_dml_lock_conflict(&e) {
                    let msg = match holder {
                        Some(h) => format!("DML lock conflict on {resource} held by {h}"),
                        None => format!("DML lock conflict on {resource}"),
                    };
                    Status::aborted(msg)
                } else {
                    Status::internal(format!("ExecuteQuery failed: {e}"))
                }
            })?;
        let rows_returned = resp.rows_affected.unwrap_or(resp.rows.len() as u64);
        let rows = resp
            .rows
            .into_iter()
            .map(|row| {
                let values = row
                    .into_iter()
                    .enumerate()
                    .map(|(index, value)| {
                        let column = resp
                            .columns
                            .get(index)
                            .cloned()
                            .unwrap_or_else(|| format!("column_{index}"));
                        (
                            column,
                            proximadb_records::proto_v2::proxima_value_to_typed_value(&value),
                        )
                    })
                    .collect();
                V2QueryRow { values }
            })
            .collect();
        Ok(Response::new(V2QueryResponse {
            rows,
            columns: resp.columns,
            rows_returned,
            execution_time_ms: resp.execution_time_ms,
        }))
    }

    // ---- Point record get-by-id (v2 parity with v1 VectorService.VectorGet) ----

    async fn get_record(
        &self,
        request: Request<proximadb_v2::GetRecordRequest>,
    ) -> Result<Response<proximadb_v2::GetRecordResponse>, Status> {
        let identity = grpc_auth::port_identity(&request)?;
        let req = request.into_inner();
        let result = self
            .record_ops
            .handle_record_get_for_tenant(
                RichRecordGetRequest {
                    collection_id: req.collection_id,
                    record_id: req.id,
                    include_vector: req.include_vector,
                    include_props: true,
                },
                identity.as_borrowed(),
            )
            .await
            .map_err(|e| Status::internal(format!("GetRecord failed: {e}")))?;
        match result {
            Some(record) => {
                let props = record
                    .props
                    .iter()
                    .map(|(k, v)| (k.clone(), proxima_value_to_typed_value(v)))
                    .collect();
                Ok(Response::new(proximadb_v2::GetRecordResponse {
                    found: true,
                    record: Some(proximadb_v2::ProximaRecord {
                        id: record.id,
                        vector: record.vector,
                        props,
                        ..Default::default()
                    }),
                }))
            }
            None => Ok(Response::new(proximadb_v2::GetRecordResponse {
                found: false,
                record: None,
            })),
        }
    }
}

// ---- V2 <-> v1 mapping helpers for the collection + query RPCs ----

fn distance_metric_from_str(s: &str) -> i32 {
    crate::proto::proximadb_v1::DistanceMetric::from_str_name(&s.to_ascii_uppercase())
        .unwrap_or(crate::proto::proximadb_v1::DistanceMetric::Cosine) as i32
}

fn storage_engine_from_str(s: &str) -> i32 {
    crate::proto::proximadb_v1::StorageEngine::from_str_name(&s.to_ascii_uppercase())
        .unwrap_or(crate::proto::proximadb_v1::StorageEngine::Sst) as i32
}

fn indexing_algorithm_from_str(s: &str) -> i32 {
    crate::proto::proximadb_v1::IndexingAlgorithm::from_str_name(&s.to_ascii_uppercase())
        .unwrap_or(crate::proto::proximadb_v1::IndexingAlgorithm::Hnsw) as i32
}

fn quantization_strategy_from_str(s: &str) -> i32 {
    crate::proto::proximadb_v1::quantization_config::Strategy::from_str_name(
        &s.to_ascii_uppercase(),
    )
    .unwrap_or(crate::proto::proximadb_v1::quantization_config::Strategy::SmartDefaults) as i32
}

fn indexing_algorithm_to_str(v: i32) -> String {
    crate::proto::proximadb_v1::IndexingAlgorithm::try_from(v)
        .map(|a| a.as_str_name().to_ascii_lowercase())
        .unwrap_or_default()
}

fn quantization_strategy_to_str(v: Option<i32>) -> String {
    v.and_then(|s| crate::proto::proximadb_v1::quantization_config::Strategy::try_from(s).ok())
        .map(|s| s.as_str_name().to_ascii_lowercase())
        .unwrap_or_default()
}

/// TD-TIER-1 SC1: map the v1 observed status onto the v2 read model. The
/// state label mirrors the REST surface ("active" | "not_active"; anything
/// unrecognized reads as "unspecified", never a false "active").
fn v2_storage_class_status_from(
    _cfg: &crate::proto::proximadb_v1::CollectionConfig,
    status: crate::proto::proximadb_v1::StorageClassStatus,
) -> proximadb_v2::V2StorageClassStatus {
    proximadb_v2::V2StorageClassStatus {
        state: crate::services::collection::storage_class::state_label(status.state).to_string(),
        reason: status.reason,
        policy_generation: status.policy_generation,
        reserved_bytes: status.reserved_bytes,
        resident_bytes: status.resident_bytes,
        covered_bytes: status.covered_bytes,
        covered_generation: status.covered_generation,
    }
}

/// TD-TIER-1 SC1: parse the v2 create request's `storage_class` label onto
/// the v1 config discriminant. `None` (absent/empty/unspecified) ⇒ the manager
/// resolves and persists BALANCED; a known label maps to its enum value so
/// admission applies; anything else is an `InvalidArgument` naming the valid
/// labels — never the silent default the strict REST surface already rejects
/// (TD-RESTGET-1 class).
fn v2_storage_class_field(cfg: &proximadb_v2::V2CollectionConfig) -> Result<Option<i32>, Status> {
    match cfg.storage_class.as_deref() {
        None | Some("") | Some("unspecified") => Ok(None),
        Some(raw) => {
            let class = crate::services::collection::storage_class::parse_storage_class(raw)
                .map_err(|e| Status::invalid_argument(e.to_string()))?;
            if class == crate::proto::proximadb_v1::StorageClass::Unspecified {
                Ok(None)
            } else {
                Ok(Some(class as i32))
            }
        }
    }
}

/// TD-TIER-1 SC5: parse the v2 create request's `default_access_tier` label
/// (strict, like `storage_class`): unset/empty ⇒ the account default.
fn v2_access_tier_field(cfg: &proximadb_v2::V2CollectionConfig) -> Result<Option<i32>, Status> {
    let Some(raw) = cfg.default_access_tier.as_deref() else {
        return Ok(None);
    };
    let tier = crate::services::collection::storage_class::parse_access_tier(raw)
        .map_err(|e| Status::invalid_argument(e.to_string()))?;
    Ok((tier != crate::proto::proximadb_v1::AccessTier::Unspecified).then_some(tier as i32))
}

fn collection_to_v2(c: crate::proto::proximadb_v1::Collection) -> V2Collection {
    let cfg = c.config.unwrap_or_default();
    let distance_metric = cfg
        .distance_metric
        .and_then(|m| crate::proto::proximadb_v1::DistanceMetric::try_from(m).ok())
        .map(|m| m.as_str_name().to_ascii_lowercase())
        .unwrap_or_default();
    let storage_engine = cfg
        .storage_engine
        .and_then(|e| crate::proto::proximadb_v1::StorageEngine::try_from(e).ok())
        .map(|e| e.as_str_name().to_ascii_lowercase())
        .unwrap_or_default();
    // TD-TIER-1 SC1: echo the resolved desired storage class ("economy" |
    // "balanced" | "performance"; absent ⇒ balanced) AND the observed status —
    // desired and observed are distinct surfaces.
    let storage_class = crate::services::collection::storage_class::storage_class_label(
        crate::services::collection::storage_class::storage_class_or_balanced(&cfg),
    )
    .to_string();
    let storage_class_status = c
        .storage_class_status
        .map(|status| v2_storage_class_status_from(&cfg, status))
        .unwrap_or_else(|| {
            v2_storage_class_status_from(
                &cfg,
                crate::services::collection::storage_class::storage_class_status(&cfg),
            )
        });
    V2Collection {
        storage_class_status: Some(storage_class_status),
        id: c.id,
        config: Some(V2CollectionConfig {
            // TD-GRPCSCHEMA-2: round-trip the mode on reads too.
            enable_proxima_record: cfg.enable_proxima_record,
            name: cfg.name,
            dimension: cfg.dimension,
            distance_metric,
            storage_engine,
            filterable_columns: cfg.filterable_columns.into_iter().map(|f| f.name).collect(),
            storage_class: Some(storage_class),
            default_access_tier: crate::services::collection::storage_class::access_tier_label(
                cfg.default_access_tier,
            )
            .map(str::to_string),
            description: cfg.description.unwrap_or_default(),
            index_specs: cfg
                .index_configs
                .into_iter()
                .map(|ic| proximadb_v2::V2IndexSpec {
                    algorithm: indexing_algorithm_to_str(ic.algorithm),
                    hnsw: ic.hnsw_config.map(|h| proximadb_v2::V2HnswConfig {
                        m: h.m,
                        ef_construction: h.ef_construction,
                        ef_search: h.ef_search,
                    }),
                    ivf: ic.ivf_config.map(|i| proximadb_v2::V2IvfConfig {
                        n_lists: i.n_lists,
                        n_probe: i.n_probe,
                    }),
                    is_primary: ic.is_primary.unwrap_or(false),
                })
                .collect(),
            quantization: cfg
                .quantization
                .map(|q| proximadb_v2::V2QuantizationConfig {
                    enabled: q.enabled.unwrap_or(false),
                    strategy: quantization_strategy_to_str(q.strategy),
                }),
        }),
        created_at: c.created_at,
        updated_at: c.updated_at,
    }
}

/// Reject a `top_k` above the shared server cap with `InvalidArgument`
/// (TD-SECSCAN-1 §5) — the same limit REST and Arrow Flight enforce.
fn check_search_top_k(top_k: u32) -> Result<(), Status> {
    proximadb_search_types::top_k_limit::check_top_k(u64::from(top_k))
        .map_err(|e| Status::invalid_argument(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TD-TIER-1 SC1 (review round 1): the v2 create MUST read
    /// `V2CollectionConfig.storage_class` — before, a client setting
    /// "performance" silently got balanced (the TD-RESTGET-1 silent-default
    /// class). Absent/empty/unspecified ⇒ `None` (the manager resolves and
    /// persists BALANCED); known labels map onto the v1 discriminants so
    /// manager admission applies identically to REST, gRPC and Flight;
    /// anything else is an actionable InvalidArgument naming the vocabulary.
    #[test]
    fn v2_create_storage_class_is_mapped_not_dropped() {
        let cfg = |class: Option<&str>| proximadb_v2::V2CollectionConfig {
            storage_class: class.map(ToString::to_string),
            ..Default::default()
        };

        for absent in [None, Some(""), Some("unspecified")] {
            assert_eq!(
                v2_storage_class_field(&cfg(absent)).unwrap(),
                None,
                "absent/unspecified must resolve to None"
            );
        }

        assert_eq!(
            v2_storage_class_field(&cfg(Some("performance"))).unwrap(),
            Some(crate::proto::proximadb_v1::StorageClass::Performance as i32)
        );
        assert_eq!(
            v2_storage_class_field(&cfg(Some("ECONOMY"))).unwrap(),
            Some(crate::proto::proximadb_v1::StorageClass::Economy as i32)
        );
        // The proto enum-name form parses too (parity with the REST surface).
        assert_eq!(
            v2_storage_class_field(&cfg(Some("storage_class_balanced"))).unwrap(),
            Some(crate::proto::proximadb_v1::StorageClass::Balanced as i32)
        );

        let error = v2_storage_class_field(&cfg(Some("perfromance")))
            .expect_err("an unknown label must be rejected, not defaulted");
        assert_eq!(
            error.code(),
            tonic::Code::InvalidArgument,
            "{}",
            error.message()
        );
        let message = error.message();
        assert!(message.contains("perfromance"), "{message}");
        assert!(message.contains("balanced"), "{message}");
    }

    #[test]
    fn v2_config_to_canonical_columns_emits_embedding_plus_filterable_scalars() {
        let cfg = proximadb_v2::V2CollectionConfig {
            storage_class: None,
            default_access_tier: None,
            name: "c".to_string(),
            dimension: 128,
            distance_metric: "cosine".to_string(),
            storage_engine: "sst".to_string(),
            filterable_columns: vec!["tag".to_string(), "owner".to_string(), String::new()],
            description: String::new(),
            index_specs: Vec::new(),
            quantization: None,
            enable_proxima_record: None,
        };
        let cols = v2_config_to_canonical_columns(&cfg);
        // Primary vector column mirrors config.dimension; empty filterable names skipped.
        assert_eq!(cols.len(), 3, "embedding + 2 non-empty filterable columns");
        assert_eq!(cols[0].name, "embedding");
        match &cols[0].data_type {
            ProximaType::DenseVector { element, dim } => {
                assert_eq!(*dim, 128);
                assert!(matches!(
                    element,
                    proximadb_data_model::VectorElement::Float32
                ));
            }
            other => panic!("expected DenseVector, got {other:?}"),
        }
        assert!(cols[0].indexed && !cols[0].filterable);
        // Declared filterable columns projected as String + filterable.
        assert_eq!(cols[1].name, "tag");
        assert!(matches!(cols[1].data_type, ProximaType::String));
        assert!(cols[1].filterable);
        assert_eq!(cols[2].name, "owner");
    }

    #[test]
    fn test_typed_value_conversion() {
        // Test would require mocking UnifiedHandlers
        // For now, just verify the module compiles
    }

    #[test]
    fn grpc_schema_round_trips_vector_dim_and_element() {
        // TypedColumn now carries vector_dimension + vector_element_type, so
        // Dense/Sparse/Binary vector columns round-trip dim and element across
        // metadata -> RecordSchema -> CollectionSchemaUpdate (previously dim
        // dropped to 0 and element was hardcoded Float32).
        let dense = CollectionSchemaColumn {
            name: "dense".to_string(),
            data_type: ProximaType::DenseVector {
                element: VectorElement::Float16,
                dim: 128,
            },
            nullable: true,
            indexed: false,
            filterable: false,
            text_storage: None,
            max_length: None,
        };
        let sparse = CollectionSchemaColumn {
            name: "sparse".to_string(),
            data_type: ProximaType::SparseVector {
                element: VectorElement::Int8,
            },
            nullable: true,
            indexed: false,
            filterable: false,
            text_storage: None,
            max_length: None,
        };
        let binary = CollectionSchemaColumn {
            name: "binary".to_string(),
            data_type: ProximaType::BinaryVector { dim: 256 },
            nullable: true,
            indexed: false,
            filterable: false,
            text_storage: None,
            max_length: None,
        };
        let metadata = CollectionSchemaMetadata {
            collection_id: "c".to_string(),
            enabled: true,
            columns: vec![dense, sparse, binary],
            ..Default::default()
        };

        let schema = runtime_metadata_to_grpc_schema(&metadata);
        // Forward map stamps dimension + element onto the wire.
        let wire = |n: &str| schema.columns.iter().find(|c| c.name == n).unwrap();
        assert_eq!(wire("dense").vector_dimension, Some(128));
        assert_eq!(
            wire("dense").vector_element_type,
            Some(proximadb_v2::VectorElementType::VectorElementFloat16 as i32)
        );
        assert_eq!(wire("sparse").vector_dimension, None);
        assert_eq!(
            wire("sparse").vector_element_type,
            Some(proximadb_v2::VectorElementType::VectorElementInt8 as i32)
        );
        assert_eq!(wire("binary").vector_dimension, Some(256));
        assert_eq!(wire("binary").vector_element_type, None);

        // Inverse map reads them back into the canonical ProximaType.
        let update = grpc_record_schema_to_runtime_update("c", schema).unwrap();
        let dense = update.columns.iter().find(|c| c.name == "dense").unwrap();
        let sparse = update.columns.iter().find(|c| c.name == "sparse").unwrap();
        let binary = update.columns.iter().find(|c| c.name == "binary").unwrap();
        assert!(matches!(
            &dense.data_type,
            ProximaType::DenseVector { element, dim }
                if *element == VectorElement::Float16 && *dim == 128
        ));
        assert!(matches!(
            &sparse.data_type,
            ProximaType::SparseVector { element } if *element == VectorElement::Int8
        ));
        assert!(matches!(
            &binary.data_type,
            ProximaType::BinaryVector { dim } if *dim == 256
        ));
    }

    #[test]
    fn test_batch_result_response_maps_error_record_ids() {
        let record_ids = vec!["record-1".to_string()];
        let response = ProximaRecordServiceImpl::batch_result_to_response(
            BatchOperationResult::failure("write failed".to_string(), "BAD_WRITE".to_string()),
            &record_ids,
            "WRITE_FAILED",
        );

        assert!(!response.success);
        assert_eq!(response.total_processed, 1);
        assert_eq!(response.failed_count, 1);
        assert_eq!(response.errors.len(), 1);
        assert_eq!(response.errors[0].record_id, "record-1");
        assert_eq!(response.errors[0].error_code, "BAD_WRITE");
    }

    #[test]
    fn test_batch_result_response_preserves_success_metrics() {
        let response = ProximaRecordServiceImpl::batch_result_to_response(
            BatchOperationResult::success(
                vec!["record-1".to_string(), "record-2".to_string()],
                crate::services::operations::OperationMetrics {
                    total_processed: 2,
                    successful_count: 2,
                    failed_count: 0,
                    updated_count: 0,
                    processing_time_us: 42,
                    wal_write_time_us: 7,
                    index_update_time_us: 3,
                },
            ),
            &["record-1".to_string(), "record-2".to_string()],
            "WRITE_FAILED",
        );

        assert!(response.success);
        assert_eq!(response.total_processed, 2);
        assert_eq!(response.success_count, 2);
        assert_eq!(response.failed_count, 0);
        assert_eq!(response.inserted_ids, ["record-1", "record-2"]);
        assert_eq!(response.processing_time_us, 42);
        assert!(response.errors.is_empty());
    }

    #[test]
    fn test_backpressure_level_none() {
        // 0-25% buffer usage should result in NONE level
        for percent in 0..=24 {
            let signal = ProximaRecordServiceImpl::calculate_backpressure(percent);
            assert_eq!(
                signal.level,
                BackpressureLevel::BackpressureNone as i32,
                "Expected NONE at {}%",
                percent
            );
            assert_eq!(signal.suggested_delay_ms, DELAY_NONE_MS);
            assert_eq!(signal.buffer_percent, percent);
        }
    }

    #[test]
    fn test_backpressure_level_low() {
        // 25-49% buffer usage should result in LOW level
        for percent in 25..50 {
            let signal = ProximaRecordServiceImpl::calculate_backpressure(percent);
            assert_eq!(
                signal.level,
                BackpressureLevel::BackpressureLow as i32,
                "Expected LOW at {}%",
                percent
            );
            assert_eq!(signal.suggested_delay_ms, DELAY_LOW_MS);
        }
    }

    #[test]
    fn test_backpressure_level_medium() {
        // 50-74% buffer usage should result in MEDIUM level
        for percent in 50..75 {
            let signal = ProximaRecordServiceImpl::calculate_backpressure(percent);
            assert_eq!(
                signal.level,
                BackpressureLevel::BackpressureMedium as i32,
                "Expected MEDIUM at {}%",
                percent
            );
            assert_eq!(signal.suggested_delay_ms, DELAY_MEDIUM_MS);
        }
    }

    #[test]
    fn test_backpressure_level_high() {
        // 75-89% buffer usage should result in HIGH level
        for percent in 75..90 {
            let signal = ProximaRecordServiceImpl::calculate_backpressure(percent);
            assert_eq!(
                signal.level,
                BackpressureLevel::BackpressureHigh as i32,
                "Expected HIGH at {}%",
                percent
            );
            assert_eq!(signal.suggested_delay_ms, DELAY_HIGH_MS);
        }
    }

    #[test]
    fn test_backpressure_level_critical() {
        // 90-100% buffer usage should result in CRITICAL level
        for percent in 90..=100 {
            let signal = ProximaRecordServiceImpl::calculate_backpressure(percent);
            assert_eq!(
                signal.level,
                BackpressureLevel::BackpressureCritical as i32,
                "Expected CRITICAL at {}%",
                percent
            );
            assert_eq!(signal.suggested_delay_ms, DELAY_CRITICAL_MS);
        }
    }

    #[test]
    fn test_dynamic_backpressure_scaling() {
        let flow_control = FlowControlState::new(STREAM_BUFFER_SIZE);

        // Test that dynamic backpressure scales delays within level ranges
        // Note: We use values that produce consistent levels after utilization calculation
        let signal_low = ProximaRecordServiceImpl::calculate_dynamic_backpressure(
            35, // 35% - should result in LOW level after calculation
            &flow_control,
            STREAM_BUFFER_SIZE,
        );
        // With no pending items and buffer at 35%, effective level depends on utilization calculation
        // The dynamic backpressure considers both buffer usage and channel capacity
        assert!(signal_low.level >= BackpressureLevel::BackpressureLow as i32);
        // Delay should be at least the LOW base
        assert!(signal_low.suggested_delay_ms >= DELAY_LOW_MS);

        let signal_high = ProximaRecordServiceImpl::calculate_dynamic_backpressure(
            80, // 80% - HIGH level
            &flow_control,
            STREAM_BUFFER_SIZE,
        );
        assert_eq!(
            signal_high.level,
            BackpressureLevel::BackpressureHigh as i32
        );
        // Delay should be between HIGH base (100) and CRITICAL base (500)
        assert!(signal_high.suggested_delay_ms >= DELAY_HIGH_MS);
        assert!(signal_high.suggested_delay_ms <= DELAY_CRITICAL_MS);
    }

    #[test]
    fn test_flow_control_state() {
        let flow_control = FlowControlState::new(100);

        // Initially not active
        assert!(!flow_control.is_active());
        assert_eq!(flow_control.pending(), 0);

        // Add some pending items
        flow_control.add_pending(500);
        assert_eq!(flow_control.pending(), 500);
        assert!(!flow_control.is_active()); // Still below MAX_PENDING_ITEMS

        // Add more to trigger flow control
        flow_control.add_pending(600);
        assert_eq!(flow_control.pending(), 1100);
        assert!(flow_control.is_active()); // Should be active now

        // Remove items to deactivate
        flow_control.remove_pending(300);
        assert_eq!(flow_control.pending(), 800);
        // Still active because above 75% of MAX_PENDING_ITEMS (750)
        assert!(flow_control.is_active());

        flow_control.remove_pending(200);
        assert_eq!(flow_control.pending(), 600);
        // Now below 75% threshold (750), should be deactivated
        assert!(!flow_control.is_active());
    }

    #[test]
    fn test_flow_control_utilization_calculation() {
        let flow_control = FlowControlState::new(100);

        // With no pending items and full channel capacity
        let util = flow_control.calculate_utilization(100);
        assert_eq!(util, 0);

        // Add pending items
        flow_control.add_pending(500); // 50% of MAX_PENDING_ITEMS
        let util = flow_control.calculate_utilization(100);
        assert_eq!(util, 50);

        // With reduced channel capacity
        let util = flow_control.calculate_utilization(30); // 70% channel usage
        // Should return max of pending util (50%) and channel util (70%)
        assert_eq!(util, 70);
    }

    #[test]
    fn test_streaming_pipeline_metrics() {
        let metrics = StreamingPipelineMetrics::new();

        // Initial state
        assert_eq!(metrics.batches_processed.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.records_processed.load(Ordering::Relaxed), 0);

        // Record some operations
        metrics.increment_batches();
        metrics.add_records(10);
        metrics.record_queue_wait(Duration::from_micros(100));
        metrics.record_processing_time(Duration::from_micros(500));
        metrics.record_ack_send_time(Duration::from_micros(50));

        assert_eq!(metrics.batches_processed.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.records_processed.load(Ordering::Relaxed), 10);
        assert_eq!(metrics.queue_wait_time_us.load(Ordering::Relaxed), 100);
        assert_eq!(metrics.processing_time_us.load(Ordering::Relaxed), 500);
        assert_eq!(metrics.ack_send_time_us.load(Ordering::Relaxed), 50);

        // Test averages
        assert!((metrics.avg_queue_wait_us() - 100.0).abs() < f64::EPSILON);
        assert!((metrics.avg_processing_time_per_record_us() - 50.0).abs() < f64::EPSILON);

        // Test max queue depth tracking
        metrics.update_max_queue_depth(50);
        assert_eq!(metrics.max_queue_depth.load(Ordering::Relaxed), 50);

        metrics.update_max_queue_depth(30); // Lower, should not update
        assert_eq!(metrics.max_queue_depth.load(Ordering::Relaxed), 50);

        metrics.update_max_queue_depth(100); // Higher, should update
        assert_eq!(metrics.max_queue_depth.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn test_streaming_pipeline_metrics_backpressure_tracking() {
        let metrics = StreamingPipelineMetrics::new();

        // No backpressure events initially
        assert_eq!(metrics.backpressure_events.load(Ordering::Relaxed), 0);
        assert_eq!(
            metrics.critical_backpressure_events.load(Ordering::Relaxed),
            0
        );

        // Record some backpressure events
        metrics.record_backpressure(BackpressureLevel::BackpressureLow);
        assert_eq!(metrics.backpressure_events.load(Ordering::Relaxed), 1);
        assert_eq!(
            metrics.critical_backpressure_events.load(Ordering::Relaxed),
            0
        );

        metrics.record_backpressure(BackpressureLevel::BackpressureMedium);
        assert_eq!(metrics.backpressure_events.load(Ordering::Relaxed), 2);
        assert_eq!(
            metrics.critical_backpressure_events.load(Ordering::Relaxed),
            0
        );

        // Critical events should be counted separately
        metrics.record_backpressure(BackpressureLevel::BackpressureCritical);
        assert_eq!(metrics.backpressure_events.load(Ordering::Relaxed), 3);
        assert_eq!(
            metrics.critical_backpressure_events.load(Ordering::Relaxed),
            1
        );

        metrics.record_backpressure(BackpressureLevel::BackpressureCritical);
        assert_eq!(
            metrics.critical_backpressure_events.load(Ordering::Relaxed),
            2
        );

        // Check current level is stored
        assert_eq!(
            metrics.current_backpressure_level.load(Ordering::Relaxed),
            BackpressureLevel::BackpressureCritical as u32
        );
    }

    #[test]
    fn test_streaming_pipeline_metrics_summary() {
        let metrics = StreamingPipelineMetrics::new();

        // Add some data
        metrics.increment_batches();
        metrics.increment_batches();
        metrics.add_records(100);
        metrics.record_queue_wait(Duration::from_micros(200));
        metrics.record_processing_time(Duration::from_micros(1000));
        metrics.update_max_queue_depth(50);
        metrics.record_backpressure(BackpressureLevel::BackpressureHigh);

        let summary = metrics.summary();

        // Verify summary contains expected fields
        assert!(summary.contains("batches: 2"));
        assert!(summary.contains("records: 100"));
        assert!(summary.contains("max_depth: 50"));
        assert!(summary.contains("bp_events: 1"));
    }

    #[test]
    fn test_dynamic_backpressure_with_flow_control_active() {
        let flow_control = FlowControlState::new(STREAM_BUFFER_SIZE);

        // Activate flow control by exceeding threshold
        flow_control.add_pending(MAX_PENDING_ITEMS + 100);
        assert!(flow_control.is_active());

        // Even at low buffer usage, delay should be at least CRITICAL when flow control is active
        let signal = ProximaRecordServiceImpl::calculate_dynamic_backpressure(
            10, // Only 10% buffer usage
            &flow_control,
            STREAM_BUFFER_SIZE,
        );

        // Should have at least CRITICAL delay due to flow control being active
        assert!(signal.suggested_delay_ms >= DELAY_CRITICAL_MS);
    }

    #[test]
    fn test_backpressure_constants() {
        // Verify threshold ordering
        assert!(BACKPRESSURE_LOW_THRESHOLD < BACKPRESSURE_MEDIUM_THRESHOLD);
        assert!(BACKPRESSURE_MEDIUM_THRESHOLD < BACKPRESSURE_HIGH_THRESHOLD);
        assert!(BACKPRESSURE_HIGH_THRESHOLD < BACKPRESSURE_CRITICAL_THRESHOLD);
        assert!(BACKPRESSURE_CRITICAL_THRESHOLD <= 100);

        // Verify delay ordering
        assert!(DELAY_NONE_MS < DELAY_LOW_MS);
        assert!(DELAY_LOW_MS < DELAY_MEDIUM_MS);
        assert!(DELAY_MEDIUM_MS < DELAY_HIGH_MS);
        assert!(DELAY_HIGH_MS < DELAY_CRITICAL_MS);
    }

    // ── Slice 6.1: primary-pod gate ─────────────────────────────────

    use crate::cluster::primary_pod_registry::{AssignmentReason, PrimaryPodRegistry};

    fn gate(registry: Arc<PrimaryPodRegistry>, self_pod_id: &str) -> Option<PrimaryPodGate> {
        Some(PrimaryPodGate {
            registry,
            self_pod_id: self_pod_id.to_string(),
        })
    }

    #[test]
    fn gate_unconfigured_allows_writes() {
        // Legacy / embedded mode: ProximaRecordServiceImpl::new without
        // with_primary_pod_gate must NOT reject any write. Locks
        // backwards-compat for paths that bypass SharedServices.
        assert!(check_primary_pod_gate(&None, "tenant-a", "coll-1").is_ok());
    }

    #[test]
    fn gate_allows_when_no_binding_exists() {
        let registry = Arc::new(PrimaryPodRegistry::new());
        let g = gate(registry, "pod-self");
        assert!(check_primary_pod_gate(&g, "tenant-a", "coll-1").is_ok());
    }

    #[test]
    fn gate_allows_when_binding_matches_self_pod() {
        let registry = Arc::new(PrimaryPodRegistry::new());
        registry.assign("tenant-a", "coll-1", "pod-self", AssignmentReason::Create);
        let g = gate(registry, "pod-self");
        assert!(check_primary_pod_gate(&g, "tenant-a", "coll-1").is_ok());
    }

    #[test]
    fn gate_rejects_misrouted_write_with_failed_precondition_and_metadata() {
        let registry = Arc::new(PrimaryPodRegistry::new());
        registry.assign(
            "tenant-a",
            "coll-1",
            "pod-other",
            AssignmentReason::Operator,
        );
        let g = gate(registry, "pod-self");
        let status = check_primary_pod_gate(&g, "tenant-a", "coll-1")
            .expect_err("must reject misrouted write");

        // Code: gRPC FailedPrecondition is the chosen analog of HTTP 421.
        assert_eq!(status.code(), tonic::Code::FailedPrecondition);

        // Trailing metadata: client SDKs parse these to re-route
        // without scraping the human message. Locking each header in
        // catches any future refactor of the ApiError → Status adapter
        // that would silently drop one.
        let md = status.metadata();
        assert_eq!(
            md.get("x-primary-pod").unwrap().to_str().unwrap(),
            "pod-other"
        );
        assert_eq!(md.get("x-tenant-id").unwrap().to_str().unwrap(), "tenant-a");
        assert_eq!(
            md.get("x-collection-id").unwrap().to_str().unwrap(),
            "coll-1"
        );
    }

    #[test]
    fn gate_scopes_per_tenant_collection_pair() {
        // Bind (tenant-a, coll-1) elsewhere. A write to
        // (tenant-a, coll-2) or (tenant-b, coll-1) must still pass —
        // bindings don't bleed across the composite key.
        let registry = Arc::new(PrimaryPodRegistry::new());
        registry.assign(
            "tenant-a",
            "coll-1",
            "pod-other",
            AssignmentReason::Operator,
        );
        let g = gate(registry, "pod-self");

        assert!(check_primary_pod_gate(&g, "tenant-a", "coll-2").is_ok());
        assert!(check_primary_pod_gate(&g, "tenant-b", "coll-1").is_ok());
        // And the original combination still rejects, to prove the
        // first two passes aren't a global "gate always allows" bug.
        assert!(check_primary_pod_gate(&g, "tenant-a", "coll-1").is_err());
    }
}
