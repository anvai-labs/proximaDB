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

//! Collection management with schema support for v2 API
//!
//! This module provides REST endpoints for creating and managing collections
//! with ProximaRecord support and typed schema definitions.
//!
//! ## Endpoints
//!
//! - `POST /api/v2/collections` - Create collection with schema
//! - `GET /api/v2/collections/{id}` - Get collection details
//!
//! ## Schema Enforcement Modes
//!
//! - **Strict**: All columns must match schema exactly
//! - **Flexible**: Schema on read, no validation at insert
//! - **Hybrid**: Core columns enforced, additional fields allowed (default)

use axum::{
    Json,
    extract::{Extension, Path, Query, State},
};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};
use utoipa::{IntoParams, ToSchema};

use crate::errors::{ApiError, ApiResult};
// AnnIndexAdvisor trait needs to be in scope so the recall-tune
// handler's IVF dispatch arm can call `.advise(...)` on
// IvfIndexAdvisor (P2.4 commit). HnswIndexAdvisor / HmgiIndexAdvisor
// (P3) similarly require the trait in scope.
#[cfg(feature = "axis")]
use crate::index::axis::management::AnnIndexAdvisor;
use crate::network::middleware::tenant::TenantContext;
use crate::network::rest::canonical::handlers::AppState;
use crate::proto::proximadb_v1::{CollectionConfig, CollectionOperation, CollectionRequest};

fn collection_storage_engine_label(storage_engine: Option<i32>) -> &'static str {
    match storage_engine {
        Some(raw) if raw != 0 => crate::core::conversions::storage_engine_to_string(raw),
        _ => "auto",
    }
}

fn collection_distance_metric_label(distance_metric: Option<i32>) -> &'static str {
    match distance_metric {
        Some(raw) if raw != 0 => crate::core::conversions::distance_metric_to_string(raw),
        _ => "cosine",
    }
}

fn non_negative_stat(value: i64) -> u64 {
    u64::try_from(value).unwrap_or(0)
}

/// Record count for a wire response: `Some` only when the server actually knows.
///
/// An absent stats block means the counter was never written, which is NOT the
/// same as an empty collection — the write path does not call
/// `MetadataStore::update_stats`, so a populated collection routinely has none.
/// Both endpoints used to coerce that into `0` (`unwrap_or_default()` on the
/// detail path, `map_or(0, ..)` on the list path), producing a confident wrong
/// answer indistinguishable from "empty". That cost one downstream project a
/// false durability bug report before the cause was found (#1527).
fn record_count_for_wire(
    stats: Option<&crate::proto::proximadb_v1::CollectionStats>,
) -> Option<u64> {
    stats.map(|s| non_negative_stat(s.vector_count))
}

fn collection_create_failure_error(collection_name: &str, error_code: Option<&str>) -> ApiError {
    let code = error_code.unwrap_or_default();
    let lower_code = code.to_ascii_lowercase();
    if code.contains("COLLECTION_EXISTS") || lower_code.contains("already exists") {
        return ApiError::AlreadyExists(format!("Collection '{}' already exists", collection_name));
    }
    ApiError::Internal(format!(
        "Failed to create collection '{}': {}",
        collection_name,
        if code.is_empty() {
            "unknown error"
        } else {
            code
        }
    ))
}

/// Map a proto `EmbeddingPrecision` discriminant (carried on
/// `CollectionConfig.canonical_embedding_precision`) to its stable string
/// label. Reuses the proto enum's `as_str_name()` so the REST surface emits
/// the same canonical labels ("EMBEDDING_PRECISION_FP16", …) as the gRPC /
/// Arrow Flight surfaces. Returns `None` for the unset / Unspecified default
/// so the JSON field is omitted when no explicit precision was chosen.
fn collection_embedding_precision_label(precision: Option<i32>) -> Option<String> {
    use crate::proto::proximadb_v1::EmbeddingPrecision;
    let raw = precision?;
    match EmbeddingPrecision::try_from(raw) {
        Ok(EmbeddingPrecision::Unspecified) | Err(_) => None,
        Ok(p) => Some(p.as_str_name().to_string()),
    }
}

/// Request to create a collection with schema support
///
/// ## Example JSON
///
/// ```json
/// {
///     "name": "products",
///     "dimension": 768,
///     "engine": "sst",
///     "schema": {
///         "columns": [
///             {"name": "category", "data_type": "text", "indexed": true},
///             {"name": "price", "data_type": "float", "filterable": true},
///             {"name": "description", "data_type": "text", "max_length": 10000}
///         ],
///         "enforcement": "hybrid",
///         "allow_additional_fields": true
///     },
///     "enable_proxima_record": true
/// }
/// ```
#[derive(Debug, Deserialize, ToSchema)]
pub struct CreateCollectionV2Request {
    /// Collection name (required)
    #[schema(min_length = 1)]
    pub name: String,
    /// Vector dimension (required)
    #[schema(minimum = 1)]
    pub dimension: u32,
    /// Storage engine selection
    ///
    /// Options: "auto", "sst", "helix", "viper", "swift", "nova", "raptor", "tst"
    /// Default: "auto" (system selects optimal engine)
    pub engine: Option<String>,
    /// Schema definition with column types
    pub schema: Option<SchemaDefinition>,
    /// Enable ProximaRecord support for this collection
    ///
    /// When enabled:
    /// - Records can use rich props and text_fields
    /// - Schema validation is applied at insert time
    /// - TEXT columns are stored in dedicated columnar format
    ///
    /// Default: false (backward compatible with v1)
    pub enable_proxima_record: Option<bool>,
    /// Distance metric for vector similarity
    ///
    /// Options: "cosine", "euclidean", "dot_product"
    /// Default: "cosine"
    pub distance_metric: Option<String>,
    /// Initial capacity hint for pre-allocation
    pub initial_capacity: Option<u64>,
    /// Canonical embedding precision for stored vectors
    ///
    /// Options: "fp32" (default), "fp16", "bf16", "int8", "uint8".
    /// Accepts the same string forms as the gRPC / Arrow Flight surfaces
    /// (e.g. "half", "float16", "EMBEDDING_PRECISION_FP16"). The DDL
    /// service applies the same normalisation, so SDKs and pgwire
    /// converge on the same enum discriminant.
    pub canonical_embedding_precision: Option<String>,
    /// Index configurations (e.g. an explicit IVF or HNSW index).
    ///
    /// Restores v1 parity: the v1 proto create accepted `index_configs`, which
    /// drive `active_algorithm_for` (e.g. an IVF index → recall-tune dispatches
    /// to the IVF arm). When omitted, the engine selects a default (HNSW).
    pub index_configs: Option<Vec<IndexConfigInput>>,
    /// Operator metadata tags, e.g. `"recall_target:0.95"`,
    /// `"target_vector_count:1000"`, `"modalities:text,image"`. Consumed by the
    /// recall advisor / route-health (`services/collection/recall_target.rs`).
    pub tags: Option<Vec<String>>,
    /// Quantization config (gRPC-v2 parity). When omitted, quantization is
    /// left unset (engine default).
    pub quantization: Option<QuantizationConfigInput>,
    /// Per-collection cold/warm search routing policy (ADR-028). When omitted,
    /// the system decides cost-derived (`mode = "auto"`). Owners with hard SLAs
    /// override here; query-time precedence is per-query `search_mode` > this
    /// policy > cost-model auto-default.
    pub index_policy: Option<IndexPolicyInput>,
}

/// REST input for the per-collection index routing policy (mirrors proto
/// `IndexPolicy`; ADR-028).
#[derive(Debug, Deserialize, ToSchema)]
pub struct IndexPolicyInput {
    /// Routing mode: "auto" (default) | "exact" | "ivf" | "hnsw" | "helix".
    /// "auto"/omitted ⇒ cost-derived. "exact" ⇒ always brute-force (no index).
    /// The engine modes force the collection onto that engine/route.
    pub mode: Option<String>,
    /// Cold-start index warming: "auto" (default) | "eager" | "lazy" | "never".
    /// Index/auto modes only — rejected when `mode = "exact"`.
    pub rehydrate: Option<String>,
    /// Exact-scan byte budget override (bytes). 0/omitted ⇒ cost-model default
    /// for the storage class.
    pub byte_budget: Option<u64>,
    /// nprobe override for index modes. 0/omitted ⇒ cost-derived. Rejected when
    /// `mode = "exact"`.
    pub nprobe: Option<u32>,
}

/// ADR-028: validate a REST `IndexPolicyInput` and map it onto the proto
/// `IndexPolicy`, fail-closed. Modes and rehydrate strings are checked against
/// allowlists, and the validity matrix is enforced: `rehydrate`/`nprobe` are
/// index/auto concepts and are rejected when `mode = "exact"` (exact builds no
/// index). Strings are normalized to lowercase so downstream consumers match
/// case-insensitively.
fn build_index_policy(
    input: IndexPolicyInput,
) -> Result<crate::proto::proximadb_v1::IndexPolicy, ApiError> {
    let mode = input
        .mode
        .as_deref()
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "auto".to_string());
    let valid_modes = ["auto", "exact", "ivf", "hnsw", "helix"];
    if !valid_modes.contains(&mode.as_str()) {
        return Err(ApiError::InvalidArgument(format!(
            "Invalid index_policy.mode '{}'. Valid modes: {:?}",
            mode, valid_modes
        )));
    }

    let rehydrate = input
        .rehydrate
        .as_deref()
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "auto".to_string());
    let valid_rehydrate = ["auto", "eager", "lazy", "never"];
    if !valid_rehydrate.contains(&rehydrate.as_str()) {
        return Err(ApiError::InvalidArgument(format!(
            "Invalid index_policy.rehydrate '{}'. Valid values: {:?}",
            rehydrate, valid_rehydrate
        )));
    }

    let nprobe = input.nprobe.unwrap_or(0);

    // Validity matrix (fail-closed): exact builds no index, so index-only knobs
    // are contradictory with it.
    if mode == "exact" {
        if rehydrate != "auto" {
            return Err(ApiError::InvalidArgument(
                "index_policy.rehydrate is invalid with mode='exact' (exact builds no index to rehydrate)".to_string(),
            ));
        }
        if nprobe != 0 {
            return Err(ApiError::InvalidArgument(
                "index_policy.nprobe is invalid with mode='exact' (exact builds no index)"
                    .to_string(),
            ));
        }
    }

    Ok(crate::proto::proximadb_v1::IndexPolicy {
        mode,
        rehydrate,
        byte_budget: input.byte_budget.unwrap_or(0),
        nprobe,
    })
}

/// REST input for a single index config (mirrors proto `IndexConfig`).
#[derive(Debug, Deserialize, ToSchema)]
pub struct IndexConfigInput {
    /// Optional index name (defaults to `index_<n>`).
    pub index_name: Option<String>,
    /// Algorithm: "hnsw", "ivf", "pq", "flat", "annoy", "lsh".
    pub algorithm: String,
    /// Free-form algorithm parameters.
    #[serde(default)]
    pub parameters: std::collections::HashMap<String, String>,
    /// HNSW tuning (when algorithm == "hnsw").
    pub hnsw_config: Option<HnswConfigInput>,
    /// IVF tuning (when algorithm == "ivf").
    pub ivf_config: Option<IvfConfigInput>,
    /// Mark this index as the collection's primary ANN index (gRPC-v2 parity).
    pub is_primary: Option<bool>,
}

/// REST input for quantization config (mirrors proto `QuantizationConfig`;
/// gRPC-v2 parity with `V2QuantizationConfig`).
#[derive(Debug, Deserialize, ToSchema)]
pub struct QuantizationConfigInput {
    /// Enable quantization for this collection.
    pub enabled: Option<bool>,
    /// Strategy: "smart_defaults" (default) | "minimal" | "aggressive" | "custom_levels".
    pub strategy: Option<String>,
}

/// REST input for HNSW index params (mirrors proto `HnswConfig`).
#[derive(Debug, Deserialize, ToSchema)]
pub struct HnswConfigInput {
    pub m: Option<u32>,
    pub ef_construction: Option<u32>,
    pub ef_search: Option<u32>,
}

/// REST input for IVF index params (mirrors proto `IvfConfig`).
#[derive(Debug, Deserialize, ToSchema)]
pub struct IvfConfigInput {
    pub n_lists: Option<u32>,
    pub n_probe: Option<u32>,
}

/// Schema definition for a collection
///
/// Defines the typed columns and enforcement rules for ProximaRecord support.
#[derive(Debug, Deserialize, Serialize, Clone, ToSchema)]
pub struct SchemaDefinition {
    /// Column definitions
    pub columns: Vec<RestColumnDefinition>,
    /// Schema enforcement mode
    ///
    /// - "strict": All columns must match schema exactly
    /// - "flexible": Schema on read, no validation at insert
    /// - "hybrid": Core columns enforced, additional fields allowed (default)
    pub enforcement: Option<String>,
    /// Allow additional fields not defined in schema
    ///
    /// Only applies in "hybrid" mode.
    /// Default: true
    pub allow_additional_fields: Option<bool>,
}

/// Backwards-compat alias for [`RestColumnDefinition`].
pub type ColumnDefinition = RestColumnDefinition;

/// Column definition for schema
#[derive(Debug, Deserialize, Serialize, Clone, ToSchema)]
pub struct RestColumnDefinition {
    /// Column name
    pub name: String,
    /// Data type
    ///
    /// Supported types:
    /// - "text": Variable-length UTF-8 text
    /// - "text_large": Large text with sidecar storage
    /// - "integer": 64-bit signed integer
    /// - "float": 64-bit floating point
    /// - "decimal": 128-bit decimal (precision, scale)
    /// - "boolean": True/false
    /// - "timestamp": Microseconds since epoch
    /// - "timestamp_tz": Timestamp with timezone
    /// - "date": Days since epoch
    /// - "time": Microseconds since midnight
    /// - "uuid": RFC 4122 UUID
    /// - "binary": Raw bytes
    /// - "json": Validated JSON
    /// - "array_text", "array_integer", "array_float", "array_boolean"
    /// - "map_string_string", "map_string_any"
    /// - "geo_point": Latitude/longitude point
    /// - "vector": Fixed-dimension vector (specify dimension)
    pub data_type: String,
    /// Whether null values are allowed
    ///
    /// Default: true
    pub nullable: Option<bool>,
    /// Create secondary index for this column
    ///
    /// Improves query performance for equality/range filters.
    /// Default: false
    pub indexed: Option<bool>,
    /// Enable filtering on this column
    ///
    /// When true, the column can be used in WHERE clauses.
    /// Default: true for indexed columns, false otherwise
    pub filterable: Option<bool>,
    /// Maximum length for TEXT/BINARY columns
    ///
    /// Default: no limit
    pub max_length: Option<u32>,
    /// Precision for DECIMAL type (1-38)
    pub precision: Option<u8>,
    /// Scale for DECIMAL type (0-precision)
    pub scale: Option<u8>,
    /// Dimension for VECTOR type
    pub vector_dimension: Option<u32>,
}

/// Parse a v2 REST column definition's `data_type` string into the canonical
/// [`proximadb_data_model::ProximaType`] (ADR-024 Step 5).
///
/// This is the SINGLE source for the type vocabulary the v2 collection API
/// accepts: the accepted set is exactly what maps to a `ProximaType`, rather than
/// a separate hardcoded allowlist. Type-specific validation (decimal
/// precision/scale, vector dimension) is performed here so the REST surface and
/// the catalog/storage layers share one type authority.
pub fn parse_rest_data_type(
    column: &RestColumnDefinition,
) -> Result<proximadb_data_model::ProximaType, ApiError> {
    use proximadb_data_model::{ProximaType, TimeUnit, VectorElement};
    let ty = match column.data_type.as_str() {
        "text" | "text_large" => ProximaType::String,
        "int8" => ProximaType::Int8,
        "int16" => ProximaType::Int16,
        "int32" => ProximaType::Int32,
        "integer" => ProximaType::Int64,
        "uint8" => ProximaType::UInt8,
        "uint16" => ProximaType::UInt16,
        "uint32" => ProximaType::UInt32,
        "uint64" => ProximaType::UInt64,
        "float32" => ProximaType::Float32,
        "float" => ProximaType::Float64,
        "decimal" => {
            let precision = column.precision.ok_or_else(|| {
                ApiError::InvalidArgument(format!(
                    "Column '{}' with type 'decimal' requires precision",
                    column.name
                ))
            })?;
            let scale = column.scale.ok_or_else(|| {
                ApiError::InvalidArgument(format!(
                    "Column '{}' with type 'decimal' requires scale",
                    column.name
                ))
            })?;
            if precision == 0 || precision > 38 {
                return Err(ApiError::InvalidArgument(format!(
                    "Column '{}': decimal precision must be between 1 and 38",
                    column.name
                )));
            }
            if scale > precision {
                return Err(ApiError::InvalidArgument(format!(
                    "Column '{}': decimal scale cannot exceed precision",
                    column.name
                )));
            }
            ProximaType::Decimal { precision, scale }
        }
        "boolean" => ProximaType::Boolean,
        "timestamp" => ProximaType::Timestamp(TimeUnit::Nanosecond),
        "timestamp_tz" => ProximaType::TimestampTz(TimeUnit::Nanosecond),
        "date" => ProximaType::Date,
        "time" => ProximaType::Time(TimeUnit::Nanosecond),
        "duration" => ProximaType::Duration(TimeUnit::Nanosecond),
        "interval" => ProximaType::Interval(TimeUnit::Nanosecond),
        "uuid" => ProximaType::Uuid,
        "ulid" => ProximaType::ULID,
        "binary" | "binary_large" => ProximaType::Binary,
        "json" => ProximaType::Json,
        "jsonb" => ProximaType::Jsonb,
        "symbol" => ProximaType::Symbol,
        "array_text" => ProximaType::Array(Box::new(ProximaType::String)),
        "array_integer" => ProximaType::Array(Box::new(ProximaType::Int64)),
        "array_float" => ProximaType::Array(Box::new(ProximaType::Float64)),
        "array_boolean" => ProximaType::Array(Box::new(ProximaType::Boolean)),
        "array_uuid" => ProximaType::Array(Box::new(ProximaType::Uuid)),
        "array_any" => ProximaType::Array(Box::new(ProximaType::Json)),
        "map_string_string" => ProximaType::Map {
            key: Box::new(ProximaType::String),
            value: Box::new(ProximaType::String),
        },
        "map_string_any" => ProximaType::Map {
            key: Box::new(ProximaType::String),
            value: Box::new(ProximaType::Json),
        },
        "map_string_integer" => ProximaType::Map {
            key: Box::new(ProximaType::String),
            value: Box::new(ProximaType::Int64),
        },
        "map_string_float" => ProximaType::Map {
            key: Box::new(ProximaType::String),
            value: Box::new(ProximaType::Float64),
        },
        "struct" => ProximaType::Struct { fields: Vec::new() },
        "geo_point" => ProximaType::Point,
        "geo_polygon" => ProximaType::GeographyPoint,
        "vector" => {
            let dim = column.vector_dimension.ok_or_else(|| {
                ApiError::InvalidArgument(format!(
                    "Column '{}' with type 'vector' requires vector_dimension",
                    column.name
                ))
            })? as usize;
            ProximaType::DenseVector {
                element: VectorElement::Float32,
                dim,
            }
        }
        "sparse_vector" => ProximaType::SparseVector {
            element: VectorElement::Float32,
        },
        "binary_vector" => {
            let dim = column.vector_dimension.ok_or_else(|| {
                ApiError::InvalidArgument(format!(
                    "Column '{}' with type 'binary_vector' requires vector_dimension",
                    column.name
                ))
            })? as usize;
            ProximaType::BinaryVector { dim }
        }
        other => {
            return Err(ApiError::InvalidArgument(format!(
                "Invalid data type '{}' for column '{}'",
                other, column.name
            )));
        }
    };
    Ok(ty)
}

/// Response for collection creation
#[derive(Debug, Serialize, ToSchema)]
pub struct CreateCollectionV2Response {
    /// Collection ID (same as name)
    pub collection_id: String,
    /// Collection name
    pub name: String,
    /// Vector dimension
    pub dimension: u32,
    /// Selected storage engine
    pub engine: String,
    /// Whether ProximaRecord is enabled
    pub proxima_record_enabled: bool,
    /// Schema ID (if schema was defined)
    pub schema_id: Option<String>,
    /// Creation timestamp
    pub created_at: String,
}

/// POST /api/v2/collections
///
/// Create a new collection with optional schema support.
///
/// ## Request Body
///
/// See [`CreateCollectionV2Request`] for the expected JSON format.
///
/// ## Response
///
/// Returns [`CreateCollectionV2Response`] with collection details.
///
/// ## Errors
///
/// - `400 Bad Request`: Invalid request or schema
/// - `409 Conflict`: Collection already exists
/// - `500 Internal Server Error`: Creation failed
#[utoipa::path(
    post,
    path = "/api/v2/collections",
    tag = "Collections",
    operation_id = "createCollection",
    summary = "Create a collection with optional schema.",
    request_body = CreateCollectionV2Request,
    responses(
        (status = 200, description = "Collection created.", body = CreateCollectionV2Response),
        (status = 400, description = "Invalid request.", body = crate::network::rest::openapi::ErrorResponse),
    ),
)]
pub async fn create_collection_v2(
    Extension(tenant): Extension<TenantContext>,
    State(state): State<AppState>,
    Json(mut request): Json<CreateCollectionV2Request>,
) -> ApiResult<Json<CreateCollectionV2Response>> {
    info!(
        "V2 API: Creating collection '{}' with dimension {}",
        request.name, request.dimension
    );

    // Validate request
    if request.name.is_empty() {
        return Err(ApiError::InvalidArgument(
            "Collection name is required".to_string(),
        ));
    }

    if request.dimension == 0 {
        return Err(ApiError::InvalidArgument(
            "Vector dimension must be greater than 0".to_string(),
        ));
    }

    // Validate engine if specified
    let engine = request.engine.as_deref().unwrap_or("auto");
    // "viper" deliberately excluded (TD-VIPER-1 S2 / ADR-093): it's no
    // longer selectable for new collections. Reject it here with a proper
    // 400 rather than letting it fall through to CollectionService's
    // generic 500 (ApiError::Internal) three layers down.
    let valid_engines = ["auto", "sst", "helix", "swift", "nova", "raptor", "tst"];
    if !valid_engines.contains(&engine) {
        let hint = if engine == "viper" {
            " (VIPER is deprecated, ADR-093 — use sst/nova/helix instead)"
        } else {
            ""
        };
        return Err(ApiError::InvalidArgument(format!(
            "Invalid storage engine '{}'{hint}. Valid engines: {:?}",
            engine, valid_engines
        )));
    }

    // Validate distance metric if specified
    if let Some(ref metric) = request.distance_metric {
        let valid_metrics = ["cosine", "euclidean", "dot_product"];
        if !valid_metrics.contains(&metric.as_str()) {
            return Err(ApiError::InvalidArgument(format!(
                "Invalid distance metric '{}'. Valid metrics: {:?}",
                metric, valid_metrics
            )));
        }
    }

    // Validate schema if provided
    let schema_id = if let Some(ref schema) = request.schema {
        // Validate enforcement mode
        if let Some(ref enforcement) = schema.enforcement {
            let valid_modes = ["strict", "flexible", "hybrid"];
            if !valid_modes.contains(&enforcement.as_str()) {
                return Err(ApiError::InvalidArgument(format!(
                    "Invalid schema enforcement mode '{}'. Valid modes: {:?}",
                    enforcement, valid_modes
                )));
            }
        }

        // Validate columns
        for column in &schema.columns {
            if column.name.is_empty() {
                return Err(ApiError::InvalidArgument(
                    "Column name cannot be empty".to_string(),
                ));
            }

            // Validate the column type by parsing it into the canonical
            // ProximaType — the accepted vocabulary derives from ProximaType
            // (ADR-024 Step 5), and decimal precision/scale + vector dimension
            // are validated inside the parser. No separate hardcoded allowlist.
            let _ = parse_rest_data_type(column)?;
        }

        // Generate schema ID
        Some(uuid::Uuid::new_v4().to_string())
    } else {
        None
    };

    let proxima_record_enabled = request.enable_proxima_record.unwrap_or(false);

    debug!(
        "Creating collection: engine={}, proxima_record={}, schema={:?}",
        engine, proxima_record_enabled, schema_id
    );

    // Create collection config for unified handlers
    // Map engine name to StorageEngine enum value
    let storage_engine_value = if engine == "auto" {
        crate::proto::proximadb_v1::StorageEngine::Unspecified as i32
    } else {
        crate::core::conversions::parse_storage_engine(engine)
            .map(|engine| engine as i32)
            .map_err(|e| ApiError::InvalidArgument(e.to_string()))?
    };

    // Map distance metric name to DistanceMetric enum value
    let distance_metric_value = match request.distance_metric.as_deref() {
        Some(metric) => Some(
            crate::core::conversions::parse_distance_metric(metric)
                .map(|metric| metric as i32)
                .map_err(|e| ApiError::InvalidArgument(e.to_string()))?,
        ),
        None => Some(crate::proto::proximadb_v1::DistanceMetric::Cosine as i32),
    };

    // Map canonical_embedding_precision label to the proto discriminant
    // (matches the dispatch the DDL service / Arrow Flight DoAction use, so
    // every protocol resolves the same string to the same enum value).
    let canonical_embedding_precision = request
        .canonical_embedding_precision
        .as_deref()
        .and_then(|raw| {
            use crate::proto::proximadb_v1::EmbeddingPrecision;
            let key = raw.trim().to_ascii_lowercase();
            let stripped = key.strip_prefix("embedding_precision_").unwrap_or(&key);
            match stripped {
                "unspecified" => Some(EmbeddingPrecision::Unspecified),
                "fp32" | "f32" | "float32" => Some(EmbeddingPrecision::Fp32),
                "fp16" | "f16" | "half" | "float16" => Some(EmbeddingPrecision::Fp16),
                "bf16" | "bfloat16" => Some(EmbeddingPrecision::Bf16),
                "int8" | "i8" | "int8_scalar" => Some(EmbeddingPrecision::Int8),
                "uint8" | "u8" | "uint8_scalar" => Some(EmbeddingPrecision::Uint8),
                _ => None,
            }
        })
        .map(|p| p as i32);

    // v1-parity: convert REST index_configs → proto IndexConfig so an explicit
    // IVF/HNSW index is persisted and read back by `active_algorithm_for`.
    let index_configs = match request.index_configs.take() {
        None => Vec::new(),
        Some(inputs) => {
            use crate::proto::proximadb_v1::{
                HnswConfig, IndexConfig, IndexingAlgorithm, IvfConfig,
            };
            let mut out = Vec::with_capacity(inputs.len());
            for (idx, cfg) in inputs.into_iter().enumerate() {
                let algorithm = match cfg.algorithm.trim().to_ascii_lowercase().as_str() {
                    "hnsw" => IndexingAlgorithm::Hnsw,
                    "ivf" => IndexingAlgorithm::Ivf,
                    "pq" => IndexingAlgorithm::Pq,
                    "flat" => IndexingAlgorithm::Flat,
                    "annoy" => IndexingAlgorithm::Annoy,
                    "lsh" => IndexingAlgorithm::Lsh,
                    other => {
                        return Err(ApiError::InvalidArgument(format!(
                            "unknown index algorithm '{}' (expected hnsw|ivf|pq|flat|annoy|lsh)",
                            other
                        )));
                    }
                };
                out.push(IndexConfig {
                    index_name: cfg.index_name.unwrap_or_else(|| format!("index_{}", idx)),
                    algorithm: algorithm as i32,
                    parameters: cfg.parameters,
                    hnsw_config: cfg.hnsw_config.map(|h| HnswConfig {
                        m: h.m,
                        ef_construction: h.ef_construction,
                        ef_search: h.ef_search,
                        ..Default::default()
                    }),
                    ivf_config: cfg.ivf_config.map(|i| IvfConfig {
                        n_lists: i.n_lists,
                        n_probe: i.n_probe,
                        ..Default::default()
                    }),
                    is_primary: cfg.is_primary,
                    ..Default::default()
                });
            }
            out
        }
    };

    // gRPC-v2 parity: map the optional quantization config onto the proto.
    let quantization = request.quantization.take().map(|q| {
        use crate::proto::proximadb_v1::{QuantizationConfig, quantization_config::Strategy};
        QuantizationConfig {
            enabled: q.enabled,
            strategy: q.strategy.map(|s| {
                Strategy::from_str_name(&s.to_ascii_uppercase()).unwrap_or(Strategy::SmartDefaults)
                    as i32
            }),
            ..Default::default()
        }
    });

    // ADR-028: validate + map the optional index routing policy. Fail-closed on
    // nonsensical combinations (the mode×rehydrate/nprobe validity matrix) so a
    // bad policy is rejected at create rather than silently ignored at query time.
    let index_policy = match request.index_policy.take() {
        None => None,
        Some(p) => Some(build_index_policy(p)?),
    };

    let mut collection_config = CollectionConfig {
        name: request.name.clone(),
        dimension: request.dimension,
        storage_engine: Some(storage_engine_value),
        distance_metric: distance_metric_value,
        canonical_embedding_precision,
        index_configs,
        quantization,
        index_policy,
        tags: request.tags.take().unwrap_or_default(),
        enable_proxima_record: Some(proxima_record_enabled),
        ..Default::default()
    };

    // TD-122: persist the typed schema (ProximaRecord) set at create time so a
    // read-after-create GetCollection echoes it. Reuses the same SchemaDefinition
    // → proto mapping as the update-schema endpoint.
    if let Some(schema) = request.schema.take() {
        super::schema::apply_schema_definition(
            &mut collection_config,
            &schema,
            schema_id.clone().unwrap_or_default(),
            "1.0.0".to_string(),
        );
    }

    let collection_request = CollectionRequest {
        operation: CollectionOperation::CollectionCreate as i32,
        collection_id: Some(request.name.clone()),
        collection_config: Some(collection_config),
        query_params: Default::default(),
        options: Default::default(),
        migration_config: Default::default(),
    };

    // Create collection via unified handlers
    match state
        .api_handlers
        .handle_collection_operation_for_tenant(collection_request, Some(&tenant.tenant_id))
        .await
    {
        Ok(resp) if !resp.success => {
            // The unified handler returns Ok(CollectionResponse) even when the
            // create FAILED (success=false carries the reason in error_code).
            // Previously this arm fell through and returned 200 with the echoed
            // request — masking a half-registered collection (GET dimension:0,
            // absent from LIST). Honor the failure: surface a real HTTP error.
            Err(collection_create_failure_error(
                &request.name,
                resp.error_code.as_deref(),
            ))
        }
        Ok(resp) => {
            // #176 follow-up: `collection_id` is the collection's canonical UUID
            // (`Collection.id`), NOT the request echo. Both the UUID and the user
            // `name` resolve to the same collection on every endpoint (get/insert/
            // search/delete) via `CollectionService::collection` → `get_native_proto`,
            // so returning the UUID keeps a `create → use collection_id` flow working.
            // Fall back to the request name only if the handler somehow omitted the
            // created collection (it always populates it on success).
            let canonical_id = resp
                .collection
                .as_ref()
                .map(|c| c.id.clone())
                .filter(|id| !id.is_empty())
                .unwrap_or_else(|| request.name.clone());
            let response = CreateCollectionV2Response {
                collection_id: canonical_id,
                name: request.name,
                dimension: request.dimension,
                engine: engine.to_string(),
                proxima_record_enabled,
                schema_id,
                created_at: chrono::Utc::now().to_rfc3339(),
            };

            info!(
                "V2 API: Collection '{}' created successfully",
                response.collection_id
            );

            Ok(Json(response))
        }
        Err(e) => {
            if e.to_string().contains("already exists") {
                Err(ApiError::AlreadyExists(format!(
                    "Collection '{}' already exists",
                    request.name
                )))
            } else {
                Err(ApiError::Internal(format!(
                    "Failed to create collection: {}",
                    e
                )))
            }
        }
    }
}

/// Collection details response
#[derive(Debug, Serialize, ToSchema)]
pub struct CollectionV2Response {
    /// Collection ID
    pub collection_id: String,
    /// Collection name
    pub name: String,
    /// Vector dimension
    pub dimension: u32,
    /// Storage engine
    pub engine: String,
    /// Distance metric
    pub distance_metric: String,
    /// Whether ProximaRecord is enabled
    pub proxima_record_enabled: bool,
    /// Canonical embedding precision label for stored vectors.
    ///
    /// Stable string matching the proto `EmbeddingPrecision::as_str_name()`
    /// (e.g. "EMBEDDING_PRECISION_FP16"). `None` when the collection didn't
    /// set a non-default (Unspecified/Fp32) precision.
    pub canonical_embedding_precision: Option<String>,
    /// Schema definition (if defined)
    pub schema: Option<SchemaDefinition>,
    /// Collection statistics
    pub stats: CollectionStatsV2,
    /// Monotonic server-owned content revision for cache revalidation.
    pub content_revision: u64,
    /// Opaque validator that also fences server restarts.
    pub content_revision_token: String,
    /// Per-index config (HNSW/IVF params, is_primary) as persisted at create
    /// time. Mirrors the gRPC-v2 `GetCollection` `index_specs` (TD-122 parity).
    pub index_specs: Vec<IndexSpecOutput>,
    /// Quantization config as persisted at create time, or `None` when unset.
    pub quantization: Option<QuantizationConfigOutput>,
    /// Creation timestamp
    pub created_at: String,
    /// Last update timestamp
    pub updated_at: Option<String>,
}

/// REST output for a single index config (mirrors gRPC `V2IndexSpec`).
#[derive(Debug, Serialize, ToSchema)]
pub struct IndexSpecOutput {
    /// Algorithm: "hnsw" | "ivf" | "pq" | "flat" | "annoy" | "lsh".
    pub algorithm: String,
    /// HNSW params (present when the index is HNSW).
    pub hnsw: Option<HnswConfigOutput>,
    /// IVF params (present when the index is IVF).
    pub ivf: Option<IvfConfigOutput>,
    /// Whether this is the collection's primary ANN index.
    pub is_primary: bool,
}

/// REST output for HNSW params (mirrors gRPC `V2HnswConfig`).
#[derive(Debug, Serialize, ToSchema)]
pub struct HnswConfigOutput {
    pub m: Option<u32>,
    pub ef_construction: Option<u32>,
    pub ef_search: Option<u32>,
}

/// REST output for IVF params (mirrors gRPC `V2IvfConfig`).
#[derive(Debug, Serialize, ToSchema)]
pub struct IvfConfigOutput {
    pub n_lists: Option<u32>,
    pub n_probe: Option<u32>,
}

/// REST output for quantization config (mirrors gRPC `V2QuantizationConfig`).
#[derive(Debug, Serialize, ToSchema)]
pub struct QuantizationConfigOutput {
    pub enabled: bool,
    /// Strategy label, e.g. "smart_defaults" | "minimal" | "aggressive".
    pub strategy: String,
}

/// Collection statistics for v2 API
#[derive(Debug, Serialize, ToSchema)]
pub struct CollectionStatsV2 {
    /// Total number of records, or `null` when the server does not know.
    ///
    /// `null` is a real answer here, not a gap. The counter is maintained by
    /// `MetadataStore::update_stats`, which the record write path does not call,
    /// so a collection that has never had stats written carries none. Reporting
    /// `0` in that case is a *confident wrong answer* — indistinguishable from a
    /// genuinely empty collection, and it cost one downstream project a false
    /// durability bug report before the cause was found (#1527).
    pub record_count: Option<u64>,
    /// Total storage size in bytes
    pub storage_size_bytes: u64,
    /// Number of indexed fields
    pub indexed_fields: u32,
    /// Number of TEXT fields with dedicated storage
    pub text_field_count: u32,
}

/// GET /api/v2/collections/{collection_id}
///
/// Get collection details including schema and statistics.
///
/// ## Path Parameters
///
/// - `collection_id`: Collection name/ID
///
/// ## Response
///
/// Returns [`CollectionV2Response`] with collection details.
///
/// ## Errors
///
/// - `404 Not Found`: Collection does not exist
/// - `500 Internal Server Error`: Retrieval failed
#[utoipa::path(
    get,
    path = "/api/v2/collections/{collection_id}",
    tag = "Collections",
    operation_id = "getCollection",
    summary = "Get collection details.",
    params(
        ("collection_id" = String, Path, description = "Collection name/ID."),
    ),
    responses(
        (status = 200, description = "Collection details.", body = CollectionV2Response),
        (status = 404, description = "Resource not found.", body = crate::network::rest::openapi::ErrorResponse),
    ),
)]
pub async fn get_collection_v2(
    Path(collection_id): Path<String>,
    Extension(tenant): Extension<TenantContext>,
    State(state): State<AppState>,
) -> ApiResult<Json<CollectionV2Response>> {
    debug!("V2 API: Getting collection '{}'", collection_id);

    if collection_id.is_empty() {
        return Err(ApiError::InvalidArgument(
            "Collection ID is required".to_string(),
        ));
    }

    // Get collection via unified handlers
    let request = CollectionRequest {
        operation: CollectionOperation::CollectionGet as i32,
        collection_id: Some(collection_id.clone()),
        collection_config: None,
        query_params: Default::default(),
        options: Default::default(),
        migration_config: Default::default(),
    };

    match state
        .api_handlers
        .handle_collection_operation_for_tenant(request, Some(&tenant.tenant_id))
        .await
    {
        Ok(resp) => {
            // Extract collection info from response
            let collection = resp.collection.unwrap_or_default();
            let config = collection.config.unwrap_or_default();
            // Read the count off the Option *before* unwrapping: an absent stats
            // block means "unknown", and `unwrap_or_default()` below would turn
            // that into a reported 0.
            let record_count = record_count_for_wire(collection.stats.as_ref());
            let stats = collection.stats.unwrap_or_default();

            let engine_str = collection_storage_engine_label(config.storage_engine);
            let distance_metric_str = collection_distance_metric_label(config.distance_metric);

            // TD-122 parity: surface the persisted per-index + quantization config
            // (same vocabulary as the gRPC `collection_to_v2` mapper).
            let index_specs = config
                .index_configs
                .iter()
                .map(|ic| IndexSpecOutput {
                    algorithm: crate::proto::proximadb_v1::IndexingAlgorithm::try_from(
                        ic.algorithm,
                    )
                    .map(|a| a.as_str_name().to_ascii_lowercase())
                    .unwrap_or_default(),
                    hnsw: ic.hnsw_config.as_ref().map(|h| HnswConfigOutput {
                        m: h.m,
                        ef_construction: h.ef_construction,
                        ef_search: h.ef_search,
                    }),
                    ivf: ic.ivf_config.as_ref().map(|i| IvfConfigOutput {
                        n_lists: i.n_lists,
                        n_probe: i.n_probe,
                    }),
                    is_primary: ic.is_primary.unwrap_or(false),
                })
                .collect();
            let quantization = config
                .quantization
                .as_ref()
                .map(|q| QuantizationConfigOutput {
                    enabled: q.enabled.unwrap_or(false),
                    strategy: q
                        .strategy
                        .and_then(|s| {
                            crate::proto::proximadb_v1::quantization_config::Strategy::try_from(s)
                                .ok()
                        })
                        .map(|s| s.as_str_name().to_ascii_lowercase())
                        .unwrap_or_default(),
                });

            // TD-122: surface the persisted ProximaRecord schema + flags that
            // CreateCollection set (reconstructed from the catalog asset).
            let proxima_record_enabled = config.enable_proxima_record.unwrap_or(false);
            // Text columns + enforcement come from build_existing_schema; the
            // scalar filterable columns are appended so the view is complete.
            let mut schema = super::schema::build_existing_schema(&config);
            let scalar_columns: Vec<RestColumnDefinition> = config
                .filterable_columns
                .iter()
                .map(|f| RestColumnDefinition {
                    name: f.name.clone(),
                    data_type: super::schema::filterable_type_to_rest(f.data_type).to_string(),
                    nullable: Some(true),
                    indexed: Some(f.indexed),
                    filterable: Some(true),
                    max_length: None,
                    precision: None,
                    scale: None,
                    vector_dimension: None,
                })
                .collect();
            if !scalar_columns.is_empty() {
                match &mut schema {
                    Some(existing) => {
                        for col in scalar_columns {
                            if !existing.columns.iter().any(|c| c.name == col.name) {
                                existing.columns.push(col);
                            }
                        }
                    }
                    None => {
                        schema = Some(SchemaDefinition {
                            columns: scalar_columns,
                            enforcement: None,
                            allow_additional_fields: Some(true),
                        });
                    }
                }
            }
            let indexed_fields = config
                .filterable_columns
                .iter()
                .filter(|c| c.indexed)
                .count() as u32;
            let text_field_count = config.text_columns.len() as u32;

            let name = if config.name.is_empty() {
                collection_id.clone()
            } else {
                config.name.clone()
            };
            // #176 follow-up: return the canonical UUID (`Collection.id`), not the
            // path echo (which may be the user-supplied name). The path identifier
            // resolved to this collection by name OR UUID, and the returned UUID is
            // itself a valid lookup key on every endpoint, so name- and UUID-based
            // lookup both keep working. Fall back to the path echo only if the
            // handler returned a collection without an id.
            let canonical_id = if collection.id.is_empty() {
                collection_id.clone()
            } else {
                collection.id.clone()
            };
            let (content_revision, content_revision_token) = state
                .record_ops
                .content_revision_snapshot(Some(&tenant.tenant_id), &canonical_id)
                .await;
            let response = CollectionV2Response {
                collection_id: canonical_id,
                name,
                dimension: config.dimension,
                engine: engine_str.to_string(),
                distance_metric: distance_metric_str.to_string(),
                proxima_record_enabled,
                canonical_embedding_precision: collection_embedding_precision_label(
                    config.canonical_embedding_precision,
                ),
                schema,
                stats: CollectionStatsV2 {
                    record_count,
                    storage_size_bytes: non_negative_stat(stats.data_size_bytes),
                    indexed_fields,
                    text_field_count,
                },
                content_revision,
                content_revision_token,
                index_specs,
                quantization,
                created_at: chrono::Utc::now().to_rfc3339(),
                updated_at: None,
            };

            Ok(Json(response))
        }
        Err(e) => {
            if e.to_string().contains("not found") {
                Err(ApiError::CollectionNotFound(collection_id))
            } else {
                Err(ApiError::Internal(format!(
                    "Failed to get collection: {}",
                    e
                )))
            }
        }
    }
}

/// Query parameters for listing collections
#[derive(Debug, Deserialize, IntoParams)]
#[into_params(parameter_in = Query)]
pub struct ListCollectionsV2Query {
    /// Maximum number of collections to return (default: 100)
    pub limit: Option<u32>,
    /// Offset for pagination (default: 0)
    pub offset: Option<u32>,
    /// Whether to include statistics
    pub include_stats: Option<bool>,
}

/// Response for listing collections
#[derive(Debug, Serialize, ToSchema)]
pub struct ListCollectionsV2Response {
    /// List of collections
    pub collections: Vec<CollectionV2Summary>,
    /// Total count of collections
    pub total: u64,
    /// Limit used in this request
    pub limit: u32,
    /// Offset used in this request
    pub offset: u32,
    /// Whether there are more results
    pub has_more: bool,
}

/// Response for deleting a collection through the v2 API.
#[derive(Debug, Serialize, ToSchema)]
pub struct DeleteCollectionV2Response {
    /// Whether the delete request was accepted.
    pub success: bool,
    /// Deleted collection ID.
    pub collection_id: String,
}

/// Summary of a collection for list operations
#[derive(Debug, Serialize, ToSchema)]
pub struct CollectionV2Summary {
    /// Collection ID
    pub collection_id: String,
    /// Collection name
    pub name: String,
    /// Vector dimension
    pub dimension: u32,
    /// Storage engine
    pub engine: String,
    /// Whether ProximaRecord is enabled
    pub proxima_record_enabled: bool,
    /// Record count (if include_stats is true)
    pub record_count: Option<u64>,
}

/// GET /api/v2/collections
///
/// List all collections with pagination.
///
/// ## Query Parameters
///
/// - `limit`: Maximum number of collections to return (default: 100)
/// - `offset`: Offset for pagination (default: 0)
/// - `include_stats`: Whether to include collection statistics
///
/// ## Response
///
/// Returns [`ListCollectionsV2Response`] with collection list.
///
/// ## Errors
///
/// - `500 Internal Server Error`: List operation failed
#[utoipa::path(
    get,
    path = "/api/v2/collections",
    tag = "Collections",
    operation_id = "listCollections",
    summary = "List collections.",
    params(ListCollectionsV2Query),
    responses(
        (status = 200, description = "Collection page.", body = ListCollectionsV2Response),
    ),
)]
pub async fn list_collections_v2(
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    Query(params): Query<ListCollectionsV2Query>,
) -> ApiResult<Json<ListCollectionsV2Response>> {
    let limit = params.limit.unwrap_or(100);
    let offset = params.offset.unwrap_or(0);
    let include_stats = params.include_stats.unwrap_or(false);

    debug!(
        "V2 API: Listing collections, limit={}, offset={}, include_stats={}",
        limit, offset, include_stats
    );

    let mut query_params = std::collections::HashMap::new();
    query_params.insert("limit".to_string(), limit.to_string());
    query_params.insert("offset".to_string(), offset.to_string());

    let mut options = std::collections::HashMap::new();
    options.insert("include_stats".to_string(), include_stats);

    let request = CollectionRequest {
        operation: CollectionOperation::CollectionList as i32,
        collection_id: None,
        collection_config: None,
        query_params,
        options,
        migration_config: Default::default(),
    };

    match state
        .api_handlers
        .handle_collection_operation_for_tenant(request, Some(&tenant.tenant_id))
        .await
    {
        Ok(resp) => {
            let collections: Vec<CollectionV2Summary> = resp
                .collections
                .iter()
                .map(|c| {
                    let cfg = c.config.as_ref();
                    // Issue #1125: use the canonical proto-enum label (the same
                    // helper GET-single/create use). The old hardcoded int map
                    // disagreed with the proto values (Sst=2, Helix=4), so an
                    // sst collection listed as "helix" after restart.
                    let engine_str =
                        collection_storage_engine_label(cfg.and_then(|c| c.storage_engine));
                    CollectionV2Summary {
                        collection_id: c.id.clone(),
                        name: cfg
                            .map(|c| c.name.clone())
                            .filter(|n| !n.is_empty())
                            .unwrap_or_else(|| c.id.clone()),
                        dimension: cfg.map_or(0, |cfg| cfg.dimension),
                        engine: engine_str.to_string(),
                        // Issue #1125: was hardcoded `false`; read the persisted flag.
                        proxima_record_enabled: cfg
                            .and_then(|c| c.enable_proxima_record)
                            .unwrap_or(false),
                        // `map_or(0, ..)` reported 0 for a collection whose
                        // stats were never written — the same fabricated answer
                        // the detail endpoint gave. Absent stats now stay absent.
                        record_count: if include_stats {
                            record_count_for_wire(c.stats.as_ref())
                        } else {
                            None
                        },
                    }
                })
                .collect();

            let total_count = resp.total_count as u64;
            let has_more = (offset as u64 + collections.len() as u64) < total_count;

            let response = ListCollectionsV2Response {
                collections,
                total: total_count,
                limit,
                offset,
                has_more,
            };

            Ok(Json(response))
        }
        Err(e) => Err(ApiError::Internal(format!(
            "Failed to list collections: {}",
            e
        ))),
    }
}

/// DELETE /api/v2/collections/{collection_id}
///
/// Delete a collection by ID/name. This v2 route keeps SDK lifecycle methods on
/// the ProximaRecord-era API while delegating to the existing collection control
/// plane.
#[utoipa::path(
    delete,
    path = "/api/v2/collections/{collection_id}",
    tag = "Collections",
    operation_id = "deleteCollection",
    summary = "Delete a collection.",
    params(
        ("collection_id" = String, Path, description = "Collection name/ID."),
    ),
    responses(
        (status = 200, description = "Collection deleted.", body = DeleteCollectionV2Response),
        (status = 404, description = "Resource not found.", body = crate::network::rest::openapi::ErrorResponse),
    ),
)]
pub async fn delete_collection_v2(
    Path(collection_id): Path<String>,
    Extension(tenant): Extension<TenantContext>,
    State(state): State<AppState>,
) -> ApiResult<Json<DeleteCollectionV2Response>> {
    info!("V2 API: Deleting collection '{}'", collection_id);

    if collection_id.is_empty() {
        return Err(ApiError::InvalidArgument(
            "Collection ID is required".to_string(),
        ));
    }

    let request = CollectionRequest {
        operation: CollectionOperation::CollectionDelete as i32,
        collection_id: Some(collection_id.clone()),
        collection_config: None,
        query_params: Default::default(),
        options: Default::default(),
        migration_config: Default::default(),
    };

    state
        .api_handlers
        .handle_collection_operation_for_tenant(request, Some(&tenant.tenant_id))
        .await
        .map_err(|e| {
            if e.to_string().contains("not found") {
                ApiError::CollectionNotFound(collection_id.clone())
            } else {
                ApiError::Internal(format!("Failed to delete collection: {}", e))
            }
        })?;

    Ok(Json(DeleteCollectionV2Response {
        success: true,
        collection_id,
    }))
}

// ---------------------------------------------------------------------------
// Route-health diagnostic endpoint (experimental, v1)
//
// Exposes a machine-readable capability contract for a single collection.
// Each field reports state that is verifiable from the codebase today; gaps
// are surfaced as typed `degraded_reasons` rather than hidden behind
// optimistic defaults. Lives under `/_diagnostics/collections/...` to signal
// that the JSON shape may evolve before it graduates to `/collections/...`.
// ---------------------------------------------------------------------------

/// Top-level route-health response. Versioned via `schema_version` so the
/// shape can evolve without breaking consumers that have pinned the contract.
#[derive(Debug, Serialize, PartialEq)]
pub struct CollectionRouteHealthV2 {
    pub schema_version: &'static str,
    pub stability: &'static str,
    pub collection_id: String,
    pub engine: String,
    pub dimension: u32,
    pub distance_metric: String,
    pub record_count: u64,
    pub storage_size_bytes: u64,
    pub index_size_bytes: u64,

    pub filtered_ann: FilteredAnnHealth,
    pub writes: WriteContractHealth,
    pub freshness: FreshnessHealth,
    pub object_economy: ObjectEconomyHealth,
    pub recall_probe: RecallProbeHealth,
    pub pinning: PinningHealth,
    pub discovery: DiscoveryHealth,
    pub recall_drift: RecallDriftHealth,
    pub suspension: SuspensionHealth,
    pub cold_serving: ColdServingHealth,

    pub degraded_reasons: Vec<DegradedReason>,
}

/// Surface for `index::axis::management::recall_drift` on the
/// route-health endpoint. Reports whether the AXIS HNSW params the
/// index was built with still match the advisor's recommendation
/// for the **current** corpus size + the operator's `recall_target`.
///
/// `wired = false` means the collection has no `recall_target:` tag
/// — the advisor never had a recommendation to drift from. All
/// other fields are `None` / `"unwired"` in that state.
///
/// When `wired = true`, the live fields populate. `kind` is one of:
///   * `"none"`           — advised params unchanged; no action.
///   * `"ef_search_only"` — only `ef_search` shifted; hot-swappable.
///   * `"rebuild_required"` — `m` or `ef_construction` changed; a
///     `/recluster` is the resolution.
#[derive(Debug, Serialize, PartialEq)]
pub struct RecallDriftHealth {
    /// True when the collection has a `recall_target:<float>` tag —
    /// only then does the advisor have a baseline to drift from.
    pub wired: bool,
    /// The advisor's `recall_target` (parsed from tags). `None` when
    /// `wired = false`.
    pub recall_target: Option<f32>,
    /// The N the advisor was sized against. Derived from the
    /// `target_vector_count:` tag (operator-supplied steady-state
    /// hint) — falls back to 100K (calibration anchor) when absent.
    /// `None` when `wired = false`.
    pub baseline_vector_count: Option<u64>,
    /// Current vector count from the collection stats. `None` when
    /// `wired = false`.
    pub current_vector_count: Option<u64>,
    /// One of "none" / "ef_search_only" / "rebuild_required" /
    /// "unwired".
    pub kind: &'static str,
    /// True iff `kind = "rebuild_required"` — a hint that the
    /// operator should call `/recluster` to realize the recall
    /// target at the current N.
    pub needs_rebuild: bool,
    /// True iff `kind = "ef_search_only"` — a hint that
    /// AXIS could fix the drift in-place by hot-swapping the live
    /// `ef_search` (not yet wired; tracked as a follow-up).
    pub hot_swap_possible: bool,
    /// Free-text summary suitable for operator dashboards. Empty
    /// string when `wired = false`.
    pub summary: String,
    /// Advisor's recommendation for the **baseline** N (what the
    /// index *was* sized against). `None` when `wired = false`.
    pub baseline_params: Option<RecallAdvisedParams>,
    /// Advisor's recommendation for the **current** N (what the
    /// index *should* be sized against now). `None` when
    /// `wired = false`. Compare with `baseline_params` to see
    /// exactly which knob drifted.
    pub current_params: Option<RecallAdvisedParams>,
    /// Operator-facing **next-step pointer**. One of:
    ///
    /// * `"none"` — no drift, no action.
    /// * `"call_recall_tune"` — `ef_search_only` drift; POST
    ///   `/api/v2/_diagnostics/collections/:id/recall-tune` resolves
    ///   in-place at zero rebuild cost. (The RecallDriftSweeper
    ///   already does this automatically every 5 min by default;
    ///   the action is the explicit knob for operators who want
    ///   to drive it sooner.)
    /// * `"call_recluster"` — `rebuild_required` drift; POST
    ///   `/api/v2/_diagnostics/collections/:id/recluster` to
    ///   trigger the recall-aware HNSW rebuild. This path is **not**
    ///   automated — the rebuild reads every record and consumes
    ///   minutes of CPU + memory, so an operator drives it.
    /// * `"set_recall_target_tag"` — collection has no
    ///   `recall_target:` tag; the adaptive stack is dormant.
    ///   Add the tag (e.g. `recall_target:0.95`) and the next
    ///   create-collection / drift sweep will start populating.
    /// * `"raise_max_ef_or_bump_m"` — the advisor's recommended ef
    ///   was **clamped** by the operator's `max_ef_search:` tag
    ///   and `projected_recall_at_clamped_ef` falls below
    ///   `recall_target`. Three resolutions: raise the cap (more
    ///   latency), bump m via /recluster (less ef needed), or
    ///   accept the projected recall. This action wins even when
    ///   `kind == "none"` — a "no-drift" status that's silently
    ///   clamped is misleading.
    ///
    /// The literals are stable identifiers — dashboard / runbook
    /// templates can switch on them without parsing the human
    /// `summary`.
    pub recommended_action: &'static str,
    /// The operator's `max_ef_search:` cap, if any. `None` when no
    /// such tag is set on the collection.
    pub max_ef_search: Option<u32>,
    /// True when the advisor's recommended ef was capped down to
    /// `max_ef_search`. The `current_params.ef_search` reflects the
    /// clamped value; `projected_recall_at_clamped_ef` reports the
    /// recall the index will actually deliver at that ef.
    pub clamped_by_max_ef: bool,
    /// When `clamped_by_max_ef = true`, the recall the index will
    /// actually achieve at the clamped ef. Typically lower than
    /// `recall_target` — that gap is exactly what
    /// `recommended_action="raise_max_ef_or_bump_m"` is signaling.
    pub projected_recall_at_clamped_ef: Option<f32>,
    /// **Which algorithm** the advisor sized for this collection
    /// (P1: "hnsw" / "ivf"). Stable literal — matches
    /// `SupportedAlgorithm::label()`. `"hnsw"` for any collection
    /// whose drift-detection path is HNSW-specific today
    /// (the existing detector covers HNSW only in P1; IVF
    /// drift / hot-swap / recluster surfaces ship in P2). Dashboard
    /// filters can switch on this without parsing
    /// `current_params`.
    pub algorithm: &'static str,
}

/// Snapshot of `(m, ef_construction, ef_search)` exposed on the
/// route-health endpoint. Lets operators see the advisor's
/// recommendation without an extra POST /recall-tune.
#[derive(Debug, Serialize, PartialEq)]
pub struct RecallAdvisedParams {
    pub m: u32,
    pub ef_construction: u32,
    pub ef_search: u32,
}

impl RecallDriftHealth {
    /// "wired = false" state — used when the collection has no
    /// `recall_target:` tag so the advisor never had a baseline.
    pub fn unwired() -> Self {
        Self {
            wired: false,
            recall_target: None,
            baseline_vector_count: None,
            current_vector_count: None,
            kind: "unwired",
            needs_rebuild: false,
            hot_swap_possible: false,
            summary: String::new(),
            baseline_params: None,
            current_params: None,
            recommended_action: "set_recall_target_tag",
            max_ef_search: None,
            clamped_by_max_ef: false,
            projected_recall_at_clamped_ef: None,
            // Default to "hnsw" — the legacy detector covers HNSW
            // only; collections without recall_target tags still
            // assume HNSW from the rest of the stack. IVF reports
            // its own discriminator only when the algorithm-aware
            // detector lands (P2).
            algorithm: "hnsw",
        }
    }
}

/// Stable action literal for the latency-budget conflict.
pub const ACTION_RAISE_MAX_EF_OR_BUMP_M: &str = "raise_max_ef_or_bump_m";

/// Detect the active ANN algorithm for a collection by walking its
/// `index_configs`. Returns the first algorithm-discriminator label
/// the route-health / recall-tune / recluster handlers dispatch on:
///
/// * `"ivf"` if any IndexConfig declares `IndexingAlgorithm::Ivf`
///   (P2.4 dispatch). Recall-tune calls `apply_ivf_nprobe_hot_swap`;
///   recluster calls `rebuild_and_swap_ivf_index_for_recall_target`.
/// * `"hmgi"` (P3+) when an HMGI auto-synthesized IndexConfig is
///   present — exists as a stable literal here so the response
///   shape admits HMGI before the handler dispatch ships.
/// * `"hnsw"` otherwise (default + every legacy collection).
///
/// The literal matches `SupportedAlgorithm::label()` so dashboards
/// + SIEM filters can switch on the same string across surfaces.
#[cfg_attr(not(feature = "axis"), allow(dead_code))]
pub(super) fn active_algorithm_for(
    config: &crate::proto::proximadb_v1::CollectionConfig,
) -> &'static str {
    use crate::proto::proximadb_v1::IndexingAlgorithm;
    // P3: HMGI lives in the AXIS-internal IndexAlgorithm enum but
    // doesn't have a proto-IndexingAlgorithm wire counterpart yet.
    // Detect HMGI via the `modalities:` collection tag — if the
    // operator declared ≥ 2 modalities, the advisor's selector
    // routed to HMGI (or fell back gracefully). This matches the
    // route-health response shape's literal mapping.
    let modality_count = crate::services::collection::recall_target::parse_modalities(config).len();
    if modality_count >= 2 {
        return "hmgi";
    }
    for idx in &config.index_configs {
        let algo =
            IndexingAlgorithm::try_from(idx.algorithm).unwrap_or(IndexingAlgorithm::Unspecified);
        if matches!(algo, IndexingAlgorithm::Ivf) {
            return "ivf";
        }
    }
    "hnsw"
}

/// Map a drift `(kind, clamped)` pair to the operator-facing
/// next-step pointer. Shared by route-health and recall-tune so
/// both surfaces agree on the recommended path.
///
/// `clamped == true` ALWAYS wins over kind, even when kind="none" —
/// a "no-drift" status that's silently clamped is misleading; the
/// real choice is "raise the cap or bump m".
#[cfg_attr(not(feature = "axis"), allow(dead_code))]
pub(super) fn recommended_action_for(kind: &str, clamped: bool) -> &'static str {
    if clamped {
        return ACTION_RAISE_MAX_EF_OR_BUMP_M;
    }
    match kind {
        "none" => "none",
        "ef_search_only" => "call_recall_tune",
        "rebuild_required" => "call_recluster",
        _ => "set_recall_target_tag",
    }
}

/// Filtered-ANN capability state. Reflects the current AXIS HNSW predicate
/// path: ID filters and ProximaRecord-backed metadata predicates are both
/// evaluated during traversal, then reapplied as a residual guard. See
/// `AxisManager::query_hnsw_with_predicate` in
/// `src/index/axis/management/manager.rs:933` for the live mechanism.
#[derive(Debug, Serialize, PartialEq)]
pub struct FilteredAnnHealth {
    /// ID-based filter clauses are evaluated inside the index traversal.
    pub id_predicate_supported: bool,
    /// Non-ID metadata predicates are evaluated against ProximaRecord-derived
    /// metadata during AXIS HNSW traversal.
    pub record_aware_predicates: bool,
    /// `MetadataFilterPushdown` infrastructure (bloom filters, column stats,
    /// selectivity estimator) exists in `src/core/search/metadata_filter_pushdown.rs`.
    pub predicate_pushdown_infrastructure_present: bool,
    /// Whether that infrastructure is wired into the default query path.
    /// Today: container is built, runtime integration is minimal.
    pub predicate_pushdown_default_wired: bool,
    /// Whether the planner discloses post-filter shortfall in EXPLAIN /
    /// result metadata.
    pub post_filter_shortfall_disclosure: bool,
    /// Tracking ID for the open design item. Lets clients correlate the
    /// state reported here with the design doc.
    pub td_064_status: &'static str,
}

/// Write-contract state. Mirrors the modes actually wired through the
/// v2 records surface, not aspirational batch-mode enums.
#[derive(Debug, Serialize, PartialEq)]
pub struct WriteContractHealth {
    pub insert: bool,
    pub upsert: bool,
    pub update: bool,
    pub delete: bool,
    /// Conditional writes (compare-and-set semantics) are not wired today.
    pub conditional_write: bool,
    /// Filter writes (delete/update where predicate) are not wired today.
    pub filter_write: bool,
    /// Patch (partial property update) is not wired today.
    pub patch: bool,
}

/// Freshness state. Two layers:
///
/// * `search_request_modes` — what each individual search request can ask for
///   today. All three `VectorFreshnessMode` variants (Strong, BoundedStale,
///   StaleOk) are wired in the search path via `should_scan_delta_with_time`.
/// * `collection_level_modes_wired` — whether a *default* freshness mode is
///   stored on the collection / projection (so callers don't have to set it
///   per request). Still `false`: only per-projection `ProjectionFreshness`
///   exists, no collection-default catalog field.
#[derive(Debug, Serialize, PartialEq)]
pub struct FreshnessHealth {
    pub scope: &'static str,
    pub collection_level_modes_wired: bool,
    pub search_request_modes: SearchFreshnessModes,
    pub notes: &'static str,
}

/// Per-request freshness modes the search path actually honors. Each flag
/// corresponds to a verifiable arm of `VectorFreshnessMode::should_scan_delta_with_time`.
#[derive(Debug, Serialize, PartialEq)]
pub struct SearchFreshnessModes {
    /// `Strong`: WAL/memtable delta is always merged. Default.
    pub strong: bool,
    /// `BoundedStale { max_staleness_ms }`: directory state up to the bound
    /// is accepted without merging the delta.
    pub bounded_stale: bool,
    /// `StaleOk`: skip the WAL delta entirely.
    pub stale_ok: bool,
    /// Whether `BoundedStale` is wired with a real time-bound check (vs
    /// silently degrading to Strong). True since commit e34a06225 wired
    /// `freshness_watermark_ns` through `scan_wal_delta_if_needed`.
    pub bounded_stale_time_bound_check: bool,
}

/// Collection pinning state (Phase 6 control surface). Reflects whether
/// an operator has explicitly pinned this collection to a storage tier via
/// `CollectionPinRegistry`. Absence of a pin means the access-pattern
/// tiering policy decides placement — there's nothing wrong about that;
/// it's just the default path.
#[derive(Debug, Serialize, PartialEq)]
pub struct PinningHealth {
    /// Whether the pinning registry is reachable from `AppState` for this
    /// deployment. Always `true` today — `AppState.pin_registry` defaults
    /// to a fresh empty registry when `SharedServices` doesn't inject one.
    pub registry_in_app_state: bool,
    /// Current pin state for this collection. `None` when the operator has
    /// not pinned it — the tiering policy decides placement on its own.
    pub pin: Option<PinDetails>,
}

/// Operator-set pin override on a single collection. Mirrors the `PinState`
/// in `src/storage/collection_pinning.rs` but uses a stable string label
/// for the target so the JSON contract doesn't track the internal enum
/// discriminant directly.
#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct PinDetails {
    /// Stable lowercase label: "memory" | "nvme_ssd" | "cloud".
    pub target: &'static str,
    /// Number of replicas the operator requested. `1` = no replication.
    pub replicas: u32,
    /// Wall-clock nanoseconds when the pin was last applied. Lets
    /// dashboards render "pinned X minutes ago" without re-querying.
    pub pinned_at_ns: i64,
}

/// Phase 8 F4a (TD-094) suspend-resume state. A suspended collection has had its
/// in-memory IVF index evicted to free memory while its persisted `ivf.bin` +
/// catalog metadata remain; the next query (or an explicit resume) warm-loads it.
#[derive(Debug, Serialize, PartialEq)]
pub struct SuspensionHealth {
    /// Whether suspend/resume state is observable for this deployment (the AXIS
    /// manager is reachable). `false` ⇒ the other fields are best-effort defaults.
    pub observable: bool,
    /// Whether the collection is currently suspended (index evicted, not yet
    /// warm-loaded back).
    pub suspended: bool,
    /// Whether an IVF index is currently resident in memory for this collection.
    pub in_memory_index: bool,
    /// Whether a persisted `ivf.bin` exists on disk (the resumability signal).
    pub resumable_from_disk: bool,
    pub notes: &'static str,
}

impl SuspensionHealth {
    /// Default block when the AXIS manager is not reachable (e.g. HELIX-only).
    fn unobservable() -> Self {
        Self {
            observable: false,
            suspended: false,
            in_memory_index: false,
            resumable_from_disk: false,
            notes: "AXIS manager not reachable; suspend/resume state unavailable",
        }
    }
}

/// ADR-023 R3 (c) cold→warm serving state of a loaded IVF index. While a binary
/// index is `ColdBinaryOnly` with `warm_clusters_fetched < warm_clusters_total`,
/// the collection serves reduced-recall (on-probe Stage-1 + per-cluster fp32
/// rerank) as the fp32 tier streams in via byte-range GETs; `FullTwoStage` (or
/// `fetched == total`) is full recall. Surfaced so operators see the cold-start
/// window and the object-store warm-fill progress.
#[derive(Debug, Serialize, PartialEq)]
pub struct ColdServingHealth {
    /// Whether cold-serving state is observable (an IVF index is loaded and the
    /// AXIS manager is reachable). `false` ⇒ the other fields are defaults.
    pub observable: bool,
    /// `"FullTwoStage"` (full recall) or `"ColdBinaryOnly"` (cold window), or
    /// `None` when not observable.
    pub serving_state: Option<&'static str>,
    /// Per-cluster fp32 tiers fetched so far (the warm-fill numerator).
    pub warm_clusters_fetched: usize,
    /// Total per-cluster fp32 tiers for a binary index; `0` for whole-file /
    /// non-binary loads (no per-cluster warm tracking).
    pub warm_clusters_total: usize,
    /// `true` while serving the cold window at reduced recall (`ColdBinaryOnly`
    /// with `fetched < total`).
    pub serving_reduced_recall: bool,
    pub notes: &'static str,
}

impl ColdServingHealth {
    /// Default block when no IVF index is loaded or the AXIS manager is absent.
    fn unobservable() -> Self {
        Self {
            observable: false,
            serving_state: None,
            warm_clusters_fetched: 0,
            warm_clusters_total: 0,
            serving_reduced_recall: false,
            notes: "no IVF index loaded; cold-serving state unavailable",
        }
    }

    /// Build from `AxisManager::cold_serving_status` output.
    #[cfg(feature = "axis")]
    fn from_status(
        state: crate::index::axis::IvfServingState,
        fetched: usize,
        total: usize,
    ) -> Self {
        use crate::index::axis::IvfServingState;
        let serving_state = match state {
            IvfServingState::FullTwoStage => "FullTwoStage",
            IvfServingState::ColdBinaryOnly => "ColdBinaryOnly",
        };
        let serving_reduced_recall =
            matches!(state, IvfServingState::ColdBinaryOnly) && fetched < total;
        Self {
            observable: true,
            serving_state: Some(serving_state),
            warm_clusters_fetched: fetched,
            warm_clusters_total: total,
            serving_reduced_recall,
            notes: if serving_reduced_recall {
                "serving cold window: Stage-1 + on-probe fp32 rerank while warm tier streams"
            } else {
                "full two-stage serving (fp32 warm tier resident)"
            },
        }
    }
}

/// Object-economy directory state. The directory format and sidecar live in
/// the SST engine; route-health reports cached in-process status when present
/// without forcing an object-storage read.
#[derive(Debug, Serialize, PartialEq)]
pub struct ObjectEconomyHealth {
    pub eligible: bool,
    pub directory_format_present: bool,
    pub live_status_in_app_state: bool,
    pub live_status: &'static str,
    pub route_hint: Option<&'static str>,
    pub notes: &'static str,
}

/// Recall-probe gate state. The `RecallProbeGate` exists with a complete
/// state machine but is not wired into the query path or `AppState` yet.
#[derive(Debug, Serialize, PartialEq)]
pub struct RecallProbeHealth {
    pub implementation_present: bool,
    pub wired_to_query_path: bool,
    pub live_state_in_app_state: bool,
    /// Per-scope (tenant + collection) gate-open state when the gate is
    /// reachable from `AppState`. `None` when the gate isn't wired into
    /// `AppState` for this deployment, or when the scope has never been
    /// observed (default-closed). The value is taken from
    /// `RecallProbeGate::is_open` at request time and reflects the most
    /// recent probe outcome held in memory.
    pub gate_open: Option<bool>,
    pub notes: &'static str,
}

/// Continuous Discovery (Phase 8 F1) state. Surfaces the per-collection
/// `discovery_active` projection freshness driven by the snapshot-publish
/// coordinator when a discovery job republishes a refined snapshot.
#[derive(Debug, Serialize, PartialEq)]
pub struct DiscoveryHealth {
    pub implementation_present: bool,
    pub live_state_in_app_state: bool,
    /// Freshness of the discovery projection ("fresh" / "updating" /
    /// "rebuild_required" / ...) when the discovery service is wired and a
    /// projection exists; `None` otherwise.
    pub active_projection_freshness: Option<String>,
    /// Pinned-snapshot lineage of the last republish (e.g. "wal:3..9").
    pub source_range: Option<String>,
    pub notes: &'static str,
}

impl DiscoveryHealth {
    fn unwired() -> Self {
        Self {
            implementation_present: true,
            live_state_in_app_state: false,
            active_projection_freshness: None,
            source_range: None,
            notes: "Continuous Discovery (F1) is implemented; the discovery \
                    service is not wired into AppState for this deployment.",
        }
    }

    fn wired(active_projection_freshness: Option<String>, source_range: Option<String>) -> Self {
        Self {
            implementation_present: true,
            live_state_in_app_state: true,
            active_projection_freshness,
            source_range,
            notes: "Discovery service is reachable from AppState; freshness \
                    reflects the discovery_active projection's latest republish state.",
        }
    }
}

/// Typed reasons the collection's route is degraded. Closed enum — adding a
/// reason is a contract change, which is the point.
#[derive(Debug, Serialize, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DegradedReason {
    FilteredAnnRecordPredicateBridgePartial,
    PostFilterShortfallNotDisclosed,
    ObjectEconomyLiveStatusNotReachable,
    ObjectEconomyDirectoryDegraded,
    RecallProbeNotWired,
    FreshnessModesNotCollectionLevel,
    ConditionalWritesUnsupported,
    FilterWritesUnsupported,
}

/// Compute the degraded-reasons list from already-built capability substructs.
///
/// Pure function so each `if !flag` branch can be exercised on both sides
/// without instantiating the full handler. Order is part of the contract —
/// the JSON snapshot test depends on it.
fn compute_degraded_reasons(
    filtered_ann: &FilteredAnnHealth,
    writes: &WriteContractHealth,
    freshness: &FreshnessHealth,
    object_economy: &ObjectEconomyHealth,
    recall_probe: &RecallProbeHealth,
) -> Vec<DegradedReason> {
    let mut reasons = Vec::new();
    if !filtered_ann.record_aware_predicates {
        reasons.push(DegradedReason::FilteredAnnRecordPredicateBridgePartial);
    }
    if !filtered_ann.post_filter_shortfall_disclosure {
        reasons.push(DegradedReason::PostFilterShortfallNotDisclosed);
    }
    if !object_economy.live_status_in_app_state {
        reasons.push(DegradedReason::ObjectEconomyLiveStatusNotReachable);
    }
    if object_economy.eligible
        && object_economy.live_status_in_app_state
        && object_economy.live_status != "loaded"
    {
        reasons.push(DegradedReason::ObjectEconomyDirectoryDegraded);
    }
    if !recall_probe.wired_to_query_path {
        reasons.push(DegradedReason::RecallProbeNotWired);
    }
    if !freshness.collection_level_modes_wired {
        reasons.push(DegradedReason::FreshnessModesNotCollectionLevel);
    }
    if !writes.conditional_write {
        reasons.push(DegradedReason::ConditionalWritesUnsupported);
    }
    if !writes.filter_write {
        reasons.push(DegradedReason::FilterWritesUnsupported);
    }
    reasons
}

/// Build the route-health response from the resolved collection facts.
///
/// Kept pure so it can be unit-tested without spinning up `AppState`.
/// The `engine` and `distance_metric` strings are the same labels the
/// existing `get_collection_v2` handler returns, so contracts stay aligned.
#[cfg(test)]
fn build_route_health(
    collection_id: String,
    engine: String,
    dimension: u32,
    distance_metric: String,
    record_count: u64,
    storage_size_bytes: u64,
    index_size_bytes: u64,
) -> CollectionRouteHealthV2 {
    build_route_health_with_live_state(
        collection_id,
        engine,
        dimension,
        distance_metric,
        record_count,
        storage_size_bytes,
        index_size_bytes,
        None,
        RecallProbeLiveState::Unwired,
        None,
    )
}

/// Per-scope recall-probe state resolved at handler time. `Unwired` means
/// the gate isn't reachable from `AppState` for this deployment, so the
/// route-health response reports `live_state_in_app_state: false` and
/// `gate_open: None`. `Wired { gate_open }` flips `live_state_in_app_state`
/// to true and exposes the actual gate state for the requested scope.
#[derive(Debug, Clone, Copy)]
enum RecallProbeLiveState {
    Unwired,
    Wired { gate_open: bool },
}

fn build_route_health_with_live_state(
    collection_id: String,
    engine: String,
    dimension: u32,
    distance_metric: String,
    record_count: u64,
    storage_size_bytes: u64,
    index_size_bytes: u64,
    cached_object_economy_status: Option<&'static str>,
    recall_probe_state: RecallProbeLiveState,
    pin_state: Option<crate::storage::collection_pinning::PinState>,
) -> CollectionRouteHealthV2 {
    let filtered_ann = FilteredAnnHealth {
        id_predicate_supported: true,
        // AxisManager::query_hnsw_with_predicate builds a metadata map from
        // collection_vectors and evaluates metadata predicates during HNSW
        // traversal, then reapplies the same expression as a residual guard.
        // See src/index/axis/management/manager.rs:933 for the live path.
        // (The dead AxisMetadataLookup scaffold that previously sat alongside
        // it was removed; its placeholder state was a confusion source.)
        record_aware_predicates: true,
        predicate_pushdown_infrastructure_present: true,
        predicate_pushdown_default_wired: false,
        // Genuinely wired: predicate_diagnostics::scope + take_shortfall
        // are called by REST records.rs and gRPC record_service.rs, the
        // captured PredicateShortfall is set on SearchPlanTrace via
        // mark_predicate_shortfall, and axis_predicate_shortfall_total
        // counts every event. Verified 2026-05-28.
        post_filter_shortfall_disclosure: true,
        td_064_status: "record_predicate_and_shortfall_wired",
    };
    let writes = WriteContractHealth {
        insert: true,
        upsert: true,
        update: true,
        delete: true,
        conditional_write: false,
        filter_write: false,
        patch: false,
    };
    let freshness = FreshnessHealth {
        scope: "projection_only",
        collection_level_modes_wired: false,
        search_request_modes: SearchFreshnessModes {
            strong: true,
            bounded_stale: true,
            stale_ok: true,
            bounded_stale_time_bound_check: true,
        },
        notes: "Per-request freshness modes (strong/bounded_stale/stale_ok) \
                are honored by the search path; collection-default modes are \
                not yet stored on the catalog.",
    };
    let object_economy_eligible = engine == "sst";
    let object_economy_live_status = if object_economy_eligible {
        cached_object_economy_status.unwrap_or("not_checked")
    } else {
        "not_applicable"
    };
    let object_economy_live_status_in_app_state =
        object_economy_eligible && cached_object_economy_status.is_some();
    let object_economy = ObjectEconomyHealth {
        eligible: object_economy_eligible,
        directory_format_present: object_economy_eligible,
        live_status_in_app_state: object_economy_live_status_in_app_state,
        live_status: object_economy_live_status,
        route_hint: object_economy_eligible.then_some("object_economy"),
        notes: if object_economy_eligible && cached_object_economy_status.is_some() {
            "VectorObjectEconomyDirectory status was read from the in-process \
             cache without object-storage I/O."
        } else if object_economy_eligible {
            "VectorObjectEconomyDirectory exists for SST; no cached live status \
             is currently present in AppState."
        } else {
            "Vector object-economy directory is currently SST-specific."
        },
    };
    let (recall_probe_in_app_state, recall_probe_gate_open) = match recall_probe_state {
        RecallProbeLiveState::Unwired => (false, None),
        RecallProbeLiveState::Wired { gate_open } => (true, Some(gate_open)),
    };
    let recall_probe = RecallProbeHealth {
        implementation_present: true,
        // TD-075 (Phase 8 F2): the AXIS IVF query path now consults
        // `RecallProbeGate::is_open` before selecting the quantized route
        // (`AxisManager::query` -> `query_ivf` -> `decide_quantized_route`).
        wired_to_query_path: true,
        live_state_in_app_state: recall_probe_in_app_state,
        gate_open: recall_probe_gate_open,
        notes: if recall_probe_in_app_state {
            "RecallProbeGate is reachable from AppState and consulted by the \
             AXIS IVF query path (TD-075): a closed gate routes to exact \
             search; an open gate enables the quantized accelerator. gate_open \
             reflects the latest probe outcome for this scope. The production \
             observer that feeds PASS/FAIL is a separate (Phase 5) follow-up."
        } else {
            "RecallProbeGate state machine exists and is consulted by the AXIS \
             query path, but is not wired into AppState for this deployment."
        },
    };

    let pinning = PinningHealth {
        registry_in_app_state: true,
        pin: pin_state.map(|ps| PinDetails {
            target: ps.target.label(),
            replicas: ps.replicas,
            pinned_at_ns: ps.pinned_at_ns,
        }),
    };

    let degraded_reasons = compute_degraded_reasons(
        &filtered_ann,
        &writes,
        &freshness,
        &object_economy,
        &recall_probe,
    );

    CollectionRouteHealthV2 {
        schema_version: "v1",
        stability: "experimental",
        collection_id,
        engine,
        dimension,
        distance_metric,
        record_count,
        storage_size_bytes,
        index_size_bytes,
        filtered_ann,
        writes,
        freshness,
        object_economy,
        recall_probe,
        pinning,
        // Default unwired; the async handler patches this with live
        // coordinator state. Builder callers (tests) get the unwired block.
        discovery: DiscoveryHealth::unwired(),
        // Default unwired; the async handler patches when the
        // collection has a `recall_target:` tag. Tests can patch via
        // `health.recall_drift = …` directly.
        recall_drift: RecallDriftHealth::unwired(),
        // Default unobservable; the async handler patches this with live
        // suspend/resume state from the AXIS manager (F4a).
        suspension: SuspensionHealth::unobservable(),
        // Default unobservable; the async handler patches this with live
        // cold→warm serving state from the AXIS manager (ADR-023 R3 (c)).
        cold_serving: ColdServingHealth::unobservable(),
        degraded_reasons,
    }
}

fn object_economy_status_label(
    status: &crate::storage::engines::sst::object_economy_directory::DirectoryLoadStatus,
) -> &'static str {
    use crate::storage::engines::sst::object_economy_directory::DirectoryLoadStatus;

    match status {
        DirectoryLoadStatus::Loaded => "loaded",
        DirectoryLoadStatus::Missing => "missing",
        DirectoryLoadStatus::Corrupt(_) => "corrupt",
        DirectoryLoadStatus::Mismatch { .. } => "mismatch",
    }
}

/// Map a proto `FilterableDataType` to the canonical `ProximaType` (ADR-024)
/// serde label the statistics envelope's `data_type` field carries. Unit
/// variants only — parametrized temporal/decimal types degrade to their base
/// label (still a schema-valid string under the frozen contract's
/// `oneOf[string,object]`).
fn filterable_type_to_proxima(v: i32) -> &'static str {
    use crate::proto::proximadb_v1::FilterableDataType as F;
    match F::try_from(v) {
        Ok(F::FilterableInteger) => "Int64",
        Ok(F::FilterableFloat) => "Float64",
        Ok(F::FilterableDecimal) => "Decimal",
        Ok(F::FilterableBoolean) => "Boolean",
        Ok(F::FilterableDatetime) => "Timestamp",
        Ok(F::FilterableTimestampTz) => "TimestampTz",
        Ok(F::FilterableDate) => "Date",
        Ok(F::FilterableTime) => "Time",
        Ok(F::FilterableUuid) => "Uuid",
        _ => "String",
    }
}

/// Map the engine's maintained `GraphStats` to the envelope's graph modality
/// block (ADR-037 TD-175). Reuses `GraphOperationsService::get_stats` — the
/// canonical accessor for the engine's incrementally-maintained node/edge/label
/// counters — rather than recomputing topology.
fn graph_stats_to_modality(
    g: crate::graph::GraphStats,
) -> crate::core::statistics::GraphStatistics {
    use crate::core::statistics::{EdgeTypeCount, GraphStatistics, LabelCount};
    GraphStatistics {
        total_nodes: g.total_nodes,
        total_edges: g.total_edges,
        average_degree: Some(g.average_degree),
        max_degree: Some(g.max_degree),
        connected_components: Some(g.connected_components),
        label_counts: g
            .label_stats
            .into_iter()
            .map(|l| LabelCount {
                label: l.label,
                count: l.count,
            })
            .collect(),
        edge_type_counts: g
            .edge_type_stats
            .into_iter()
            .map(|e| EdgeTypeCount {
                edge_type: e.edge_type,
                count: e.count,
            })
            .collect(),
    }
}

/// GET /api/v2/_diagnostics/collections/{collection_id}/statistics
///
/// ADR-037 (TD-174): emit the v1 [`StatisticsEnvelope`] — the modality-neutral,
/// **units-only** boundary object the agent-facing catalog consumes (AnvaiOps
/// ADR-0021 reads *this*, never the underlying data). It is projected from the
/// resident per-collection summary maintained at the flush/compaction write
/// boundary (the sibling of the KSU meter), overlaid with the authoritative
/// live record/size counts. It is **never a corpus scan** — an agent reads the
/// tiny envelope instead of the data.
///
/// Carries units and distributions only: no pricing, descriptions, quality
/// grade, PII, or per-account scope (those are AnvaiOps policy). Freshness is a
/// FACT (a flush/compaction watermark), never an SLA. Tenant-scoped.
///
/// ## Errors
///
/// - `404 Not Found`: Collection does not exist
/// - `500 Internal Server Error`: Lookup failed
#[utoipa::path(
    get,
    path = "/api/v2/_diagnostics/collections/{collection_id}/statistics",
    tag = "Collections",
    operation_id = "getCollectionStatistics",
    summary = "Agent-facing statistics envelope (ADR-037 statistics-envelope.v1).",
    params(
        ("collection_id" = String, Path, description = "Collection name/ID."),
    ),
    responses(
        (status = 200, description = "Statistics envelope (statistics-envelope.v1; canonical schema in docs/12-design/contracts)."),
        (status = 404, description = "Resource not found.", body = crate::network::rest::openapi::ErrorResponse),
    ),
)]
pub async fn get_collection_statistics_v2(
    Path(collection_id): Path<String>,
    Extension(tenant): Extension<TenantContext>,
    State(state): State<AppState>,
) -> ApiResult<Json<crate::core::statistics::StatisticsEnvelope>> {
    if collection_id.is_empty() {
        return Err(ApiError::InvalidArgument(
            "Collection ID is required".to_string(),
        ));
    }

    let request = CollectionRequest {
        operation: CollectionOperation::CollectionGet as i32,
        collection_id: Some(collection_id.clone()),
        collection_config: None,
        query_params: Default::default(),
        options: Default::default(),
        migration_config: Default::default(),
    };

    let resp = state
        .api_handlers
        .handle_collection_operation_for_tenant(request, Some(&tenant.tenant_id))
        .await
        .map_err(|e| {
            if e.to_string().contains("not found") {
                ApiError::CollectionNotFound(collection_id.clone())
            } else {
                ApiError::Internal(format!("Failed to get collection: {}", e))
            }
        })?;

    let collection = resp.collection.unwrap_or_default();
    let config = collection.config.unwrap_or_default();
    let stats = collection.stats.unwrap_or_default();
    let canonical_id = if collection.id.is_empty() {
        collection_id.clone()
    } else {
        collection.id.clone()
    };

    // TD-175: best-effort graph modality block from the engine's maintained
    // graph counters (the `get_stats` accessor); skipped when the collection has
    // no graph topology so vector-only collections carry no spurious graph block.
    let graph_modality = state
        .graph_operations_service
        .get_stats(&canonical_id)
        .await
        .ok()
        .map(graph_stats_to_modality)
        .filter(|g| g.total_nodes > 0 || g.total_edges > 0);

    let registry = crate::core::statistics::statistics_registry();
    // Overlay the authoritative live counts/sizes + schema-level field facts onto
    // the resident summary (creating it if no flush has populated one yet — then
    // freshness honestly reads as the epoch "no snapshot" watermark). Per-field
    // distributions and modality sketches fill in as flush/compaction observes
    // records (TD-174); this read never scans the corpus.
    registry.update(&canonical_id, move |summary| {
        summary.set_record_count(non_negative_stat(stats.vector_count));
        summary.set_sizes(
            non_negative_stat(stats.data_size_bytes),
            Some(non_negative_stat(stats.index_size_bytes)),
        );
        for col in &config.filterable_columns {
            summary.set_field_schema(
                &col.name,
                serde_json::Value::String(filterable_type_to_proxima(col.data_type).to_string()),
                Some(col.indexed),
                Some(true),
            );
        }
        if config.dimension > 0 {
            summary.set_vector_meta(
                config.dimension,
                collection_distance_metric_label(config.distance_metric),
            );
        }
        if let Some(g) = graph_modality {
            summary.set_graph_stats(g);
        }
    });

    let envelope = registry.envelope(&canonical_id).unwrap_or_else(|| {
        crate::core::statistics::StatisticsSummary::new(&canonical_id).to_envelope()
    });

    Ok(Json(envelope))
}

/// GET /api/v2/_diagnostics/collections/{collection_id}/route-health
///
/// Experimental: returns a machine-readable capability contract describing
/// what the collection's route can guarantee today. The endpoint does not
/// perform any search/write side effects; it resolves collection facts via
/// the same `CollectionGet` path as `get_collection_v2` and otherwise
/// reports static, code-verified capability state.
///
/// ## Errors
///
/// - `404 Not Found`: Collection does not exist
/// - `500 Internal Server Error`: Lookup failed
pub async fn get_collection_route_health_v2(
    Path(collection_id): Path<String>,
    Extension(tenant): Extension<TenantContext>,
    State(state): State<AppState>,
) -> ApiResult<Json<CollectionRouteHealthV2>> {
    debug!(
        "V2 API: route-health for collection '{}' (experimental)",
        collection_id
    );

    if collection_id.is_empty() {
        return Err(ApiError::InvalidArgument(
            "Collection ID is required".to_string(),
        ));
    }

    let request = CollectionRequest {
        operation: CollectionOperation::CollectionGet as i32,
        collection_id: Some(collection_id.clone()),
        collection_config: None,
        query_params: Default::default(),
        options: Default::default(),
        migration_config: Default::default(),
    };

    let resp = state
        .api_handlers
        .handle_collection_operation_for_tenant(request, Some(&tenant.tenant_id))
        .await
        .map_err(|e| {
            if e.to_string().contains("not found") {
                ApiError::CollectionNotFound(collection_id.clone())
            } else {
                ApiError::Internal(format!("Failed to get collection: {}", e))
            }
        })?;

    let collection = resp.collection.unwrap_or_default();
    let config = collection.config.unwrap_or_default();
    let stats = collection.stats.unwrap_or_default();

    let engine_str = collection_storage_engine_label(config.storage_engine).to_string();
    let distance_metric_str = collection_distance_metric_label(config.distance_metric).to_string();
    let cached_object_economy_status = if engine_str == "sst" {
        state
            .vector_operations_service
            .cached_object_economy_directory_status(&collection_id)
            .as_ref()
            .map(object_economy_status_label)
    } else {
        None
    };

    let recall_probe_state = match &state.recall_probe_gate {
        Some(gate) => {
            let scope =
                crate::catalog::ProbeScope::new(tenant.tenant_id.clone(), collection_id.clone());
            RecallProbeLiveState::Wired {
                gate_open: gate.is_open(&scope).await,
            }
        }
        None => RecallProbeLiveState::Unwired,
    };

    let pin_state = state.pin_registry.get(&collection_id);

    let collection_id_for_discovery = collection_id.clone();
    let mut health = build_route_health_with_live_state(
        collection_id,
        engine_str,
        config.dimension,
        distance_metric_str,
        non_negative_stat(stats.vector_count),
        non_negative_stat(stats.data_size_bytes),
        non_negative_stat(stats.index_size_bytes),
        cached_object_economy_status,
        recall_probe_state,
        pin_state,
    );

    // Phase 8 F4a: patch the suspension block with live AXIS state (the index
    // manager is reached via the same global the recall-tune endpoint uses).
    health.suspension = match crate::storage::engines::sst::core::get_sst_axis_manager() {
        Some(axis) => {
            let suspended = axis.is_suspended(&collection_id_for_discovery).await;
            SuspensionHealth {
                observable: true,
                suspended,
                in_memory_index: axis.has_ivf_index(&collection_id_for_discovery).await,
                resumable_from_disk: axis
                    .has_persisted_ivf_index(&collection_id_for_discovery)
                    .await,
                notes: if suspended {
                    "index evicted to free memory; warm-loads from disk on next query"
                } else {
                    "active (or never suspended)"
                },
            }
        }
        None => SuspensionHealth::unobservable(),
    };

    // ADR-023 R3 (c): patch the cold-serving block with live AXIS state — the
    // cold→warm window + per-cluster warm-fill progress for a loaded IVF index.
    // When the `axis` feature is off the builder default
    // (`ColdServingHealth::unobservable()`) is left in place.
    #[cfg(feature = "axis")]
    {
        health.cold_serving = match crate::storage::engines::sst::core::get_sst_axis_manager() {
            Some(axis) => match axis
                .ivf_cold_serving_status(&collection_id_for_discovery)
                .await
            {
                Some((state_str, fetched, total)) => {
                    use crate::index::axis::IvfServingState;
                    let state = match state_str.as_str() {
                        "full_two_stage" => IvfServingState::FullTwoStage,
                        "cold_binary_only" => IvfServingState::ColdBinaryOnly,
                        _ => IvfServingState::ColdBinaryOnly, // Default fallback
                    };
                    ColdServingHealth::from_status(state, fetched, total)
                }
                None => ColdServingHealth::unobservable(),
            },
            None => ColdServingHealth::unobservable(),
        };
    }

    // Patch the recall_drift block when the collection has a
    // `recall_target:<float>` tag. Otherwise the builder default
    // (RecallDriftHealth::unwired()) is correct. When the `axis`
    // feature is off the builder default is left in place.
    #[cfg(feature = "axis")]
    {
        if let Some(recall_target) =
            crate::services::collection::recall_target::parse_recall_target(&config)
        {
            let baseline_n =
                crate::services::collection::recall_target::parse_target_vector_count(&config)
                    .unwrap_or(100_000);
            let current_n = non_negative_stat(stats.vector_count);
            let metric = match config
                .distance_metric
                .and_then(|v| crate::proto::proximadb_v1::DistanceMetric::try_from(v).ok())
            {
                Some(crate::proto::proximadb_v1::DistanceMetric::Cosine) => {
                    crate::compute::distance_computation::DistanceMetric::Cosine
                }
                Some(crate::proto::proximadb_v1::DistanceMetric::Euclidean) => {
                    crate::compute::distance_computation::DistanceMetric::Euclidean
                }
                Some(crate::proto::proximadb_v1::DistanceMetric::DotProduct) => {
                    crate::compute::distance_computation::DistanceMetric::DotProduct
                }
                _ => crate::compute::distance_computation::DistanceMetric::Cosine,
            };
            let top_k = crate::services::collection::recall_target::resolve_top_k(&config);
            let max_ef_search =
                crate::services::collection::recall_target::parse_max_ef_search(&config);
            let report = crate::index::axis::management::detect_recall_drift(
                crate::index::axis::management::RecallDriftInput {
                    baseline_n,
                    current_n,
                    recall_target,
                    top_k,
                    dimension: config.dimension,
                    distance_metric: metric,
                    max_ef_search,
                },
            );
            let kind: &'static str = match report.drift_kind {
                crate::index::axis::management::DriftKind::None => "none",
                crate::index::axis::management::DriftKind::EfSearchOnly => "ef_search_only",
                crate::index::axis::management::DriftKind::EfConstructionOrM => "rebuild_required",
                // IVF variants — when the route-health handler grows
                // an IVF dispatch path (P2 commit 4) these branches
                // will fire on `detect_ivf_recall_drift` output. For
                // now the HNSW-only handler can't produce them.
                crate::index::axis::management::DriftKind::NprobeOnly => "ef_search_only",
                crate::index::axis::management::DriftKind::NlistOrQuantizer => "rebuild_required",
            };
            let baseline_params = Some(RecallAdvisedParams {
                m: report.baseline_params.m,
                ef_construction: report.baseline_params.ef_construction,
                ef_search: report.baseline_params.ef_search,
            });
            let current_params = Some(RecallAdvisedParams {
                m: report.current_params.m,
                ef_construction: report.current_params.ef_construction,
                ef_search: report.current_params.ef_search,
            });
            let clamped = report.current_params.clamped_by_max_ef;
            let projected = report.current_params.projected_recall_if_clamped;
            health.recall_drift = RecallDriftHealth {
                wired: true,
                recall_target: Some(recall_target),
                baseline_vector_count: Some(baseline_n),
                current_vector_count: Some(current_n),
                kind,
                needs_rebuild: report.needs_rebuild(),
                hot_swap_possible: report.hot_swap_possible(),
                summary: report.summary,
                baseline_params,
                current_params,
                recommended_action: recommended_action_for(kind, clamped),
                max_ef_search,
                clamped_by_max_ef: clamped,
                projected_recall_at_clamped_ef: projected,
                // P2.4: dispatch on the collection's active algorithm.
                // HNSW path runs the HNSW drift detector above; IVF
                // collections report "ivf" so the recall-tune /
                // recluster handlers route correctly. The drift block
                // params above are the HNSW-shape report — IVF route-
                // health surface gets the IVF-specific drift block in
                // a follow-up commit; for now the algorithm literal
                // is the key dispatch signal.
                algorithm: active_algorithm_for(&config),
            };
            crate::metrics::recall_drift_metrics::record_recall_drift_observation(
                &collection_id_for_discovery,
                kind,
            );
        } else {
            // No recall_target tag → emit the unwired one-hot so the
            // gauge still surfaces this collection's state on dashboards.
            crate::metrics::recall_drift_metrics::record_recall_drift_observation(
                &collection_id_for_discovery,
                "unwired",
            );
        }
    }

    // Phase 8 (F1): patch the discovery block with live snapshot-coordinator
    // state (the discovery_active projection's freshness + lineage).
    health.discovery = match &state.discovery_service {
        Some(svc) => match svc
            .coordinator()
            .active_projection(&collection_id_for_discovery)
            .await
        {
            Ok(Some(projection)) => DiscoveryHealth::wired(
                Some(format!("{:?}", projection.freshness_state).to_lowercase()),
                projection.source_range,
            ),
            _ => DiscoveryHealth::wired(None, None),
        },
        None => DiscoveryHealth::unwired(),
    };

    Ok(Json(health))
}

/// Operator-permission gate shared by the AXIS mutation endpoints
/// (`/recall-tune`, `/recluster`). Mirrors
/// `crate::network::rest::canonical::primary_pod::authorize_operator`: a
/// caller must present a [`UnifiedUserContext`] with either
/// `SystemAdmin` or `ConfigureSystem` in their effective
/// permissions. Failure surfaces as `ApiError::Unauthorized` (401)
/// when no context was attached and `ApiError::Forbidden` (403)
/// when the context exists but lacks the needed permission — same
/// envelope as the rest of the v2 surface.
///
/// `endpoint` is the operator-facing label used in the error
/// message + audit log so the caller can tell which mutation was
/// rejected.
fn require_recall_admin(
    user_context: Option<&crate::security::rbac_service::UnifiedUserContext>,
    endpoint: &'static str,
) -> ApiResult<String> {
    use crate::security::rbac_service::UnifiedPermission;

    let Some(ctx) = user_context else {
        // No auth context injected. The unified-port REST server
        // doesn't attach the auth_middleware_unified layer (only the
        // legacy multi-port path does — see
        // `RestServer::build_router_for_unified` vs
        // `RestServer::start_with_security`). In that mode, allow
        // the request through with a stable "dev" user_id. This
        // matches the behaviour operators get from the rest of the
        // surface (route-health, list, create) under the same
        // unified-port config, and keeps the endpoint usable in
        // dev / single-node deployments. When the unified port
        // gains auth wiring, the `None` arm will stop being
        // reachable in production-style configs.
        return Ok("dev:unified-port-no-auth".to_string());
    };
    let allowed = ctx
        .effective_permissions
        .contains(&UnifiedPermission::SystemAdmin)
        || ctx
            .effective_permissions
            .contains(&UnifiedPermission::ConfigureSystem);
    if allowed {
        Ok(ctx.user_id.clone())
    } else {
        Err(ApiError::Forbidden(format!(
            "{endpoint}: requires SystemAdmin or ConfigureSystem permission"
        )))
    }
}

/// Response body for `POST /api/v2/_diagnostics/collections/:id/recall-tune`.
///
/// The handler always returns the underlying drift `report` (so the
/// caller sees the same numbers the route-health endpoint would
/// surface) and a separate `action` block that says what actually
/// happened — applied a hot-swap, declined because no drift,
/// declined because the resolution requires a full rebuild, or
/// declined because the collection has no `recall_target:` tag.
#[derive(Debug, serde::Serialize, PartialEq)]
pub struct RecallTuneResponse {
    /// Stability of this surface — experimental until the
    /// route-health contract stabilizes.
    pub stability: &'static str,
    /// Collection ID the tune ran against.
    pub collection_id: String,
    /// Mirrors the route-health block so callers don't need a second
    /// GET. `wired = false` means the collection has no
    /// `recall_target:` tag and there was nothing for the advisor to
    /// drift from.
    pub report: RecallDriftHealth,
    /// What the handler actually did. One of:
    /// * `"applied_hot_swap"` — `ef_search` updated in-place.
    /// * `"no_drift"` — params already match the advisor.
    /// * `"rebuild_required"` — `m` / `ef_construction` need a
    ///   full rebuild; the operator must call the recluster path.
    /// * `"not_wired"` — collection has no `recall_target:` tag.
    pub action: &'static str,
    /// When `action = "applied_hot_swap"`, the per-spec before/after
    /// records. Empty otherwise.
    pub applied_changes: Vec<RecallTuneEfChange>,
}

#[derive(Debug, serde::Serialize, PartialEq)]
pub struct RecallTuneEfChange {
    pub index_name: Option<String>,
    pub previous_ef_search: u32,
    pub new_ef_search: u32,
}

/// Adaptive recall tune handler.
///
/// This endpoint is gated on the `axis` feature — the whole handler is
/// elided (and its route is not registered) when `axis` is off.
///
/// Reads `recall_target:` from the collection's tags, runs
/// `detect_recall_drift`, then:
///
/// * `DriftKind::None` → returns `action = "no_drift"`.
/// * `DriftKind::EfSearchOnly` → calls
///   `AxisManager::apply_hnsw_ef_hot_swap` with the advisor's
///   current `ef_search`; returns the change list.
/// * `DriftKind::EfConstructionOrM` → returns
///   `action = "rebuild_required"`. The operator must drive the
///   recluster (separate slice).
///
/// Returns `404` if the collection doesn't exist, `400` if the
/// collection_id is empty, `401`/`403` if the caller lacks
/// `SystemAdmin` or `ConfigureSystem` permission. The auth gate
/// matches `primary_pod::authorize_operator` because this endpoint
/// **mutates** the live AXIS strategy.
#[cfg(feature = "axis")]
pub async fn post_collection_recall_tune_v2(
    Path(collection_id): Path<String>,
    Extension(tenant): Extension<TenantContext>,
    user_context: Option<Extension<crate::security::rbac_service::UnifiedUserContext>>,
    State(state): State<AppState>,
) -> ApiResult<Json<RecallTuneResponse>> {
    debug!(
        "V2 API: recall-tune for collection '{}' (experimental)",
        collection_id
    );

    if collection_id.is_empty() {
        return Err(ApiError::InvalidArgument(
            "Collection ID is required".to_string(),
        ));
    }

    require_recall_admin(user_context.as_ref().map(|e| &e.0), "recall-tune")?;

    let request = CollectionRequest {
        operation: CollectionOperation::CollectionGet as i32,
        collection_id: Some(collection_id.clone()),
        collection_config: None,
        query_params: Default::default(),
        options: Default::default(),
        migration_config: Default::default(),
    };

    let resp = state
        .api_handlers
        .handle_collection_operation_for_tenant(request, Some(&tenant.tenant_id))
        .await
        .map_err(|e| {
            if e.to_string().contains("not found") {
                ApiError::CollectionNotFound(collection_id.clone())
            } else {
                ApiError::Internal(format!("Failed to get collection: {}", e))
            }
        })?;

    let collection = resp.collection.unwrap_or_default();
    let config = collection.config.unwrap_or_default();
    let stats = collection.stats.unwrap_or_default();

    // No recall_target → no advisor baseline → nothing to do.
    let Some(recall_target) =
        crate::services::collection::recall_target::parse_recall_target(&config)
    else {
        return Ok(Json(RecallTuneResponse {
            stability: "experimental",
            collection_id,
            report: RecallDriftHealth::unwired(),
            action: "not_wired",
            applied_changes: Vec::new(),
        }));
    };

    let baseline_n = crate::services::collection::recall_target::parse_target_vector_count(&config)
        .unwrap_or(100_000);
    let current_n = non_negative_stat(stats.vector_count);
    let metric = match config
        .distance_metric
        .and_then(|v| crate::proto::proximadb_v1::DistanceMetric::try_from(v).ok())
    {
        Some(crate::proto::proximadb_v1::DistanceMetric::Cosine) => {
            crate::compute::distance_computation::DistanceMetric::Cosine
        }
        Some(crate::proto::proximadb_v1::DistanceMetric::Euclidean) => {
            crate::compute::distance_computation::DistanceMetric::Euclidean
        }
        Some(crate::proto::proximadb_v1::DistanceMetric::DotProduct) => {
            crate::compute::distance_computation::DistanceMetric::DotProduct
        }
        _ => crate::compute::distance_computation::DistanceMetric::Cosine,
    };

    let top_k = crate::services::collection::recall_target::resolve_top_k(&config);
    let max_ef_search = crate::services::collection::recall_target::parse_max_ef_search(&config);
    // P2.4: also parse the algorithm-agnostic budgets — the IVF
    // dispatch arm below feeds them to the IVF advisor.
    let max_query_latency_ms =
        crate::services::collection::recall_target::parse_max_query_latency_ms(&config);
    let max_memory_mb = crate::services::collection::recall_target::parse_max_memory_mb(&config);
    let drift = crate::index::axis::management::detect_recall_drift(
        crate::index::axis::management::RecallDriftInput {
            baseline_n,
            current_n,
            recall_target,
            top_k,
            dimension: config.dimension,
            distance_metric: metric,
            max_ef_search,
        },
    );

    let kind_str: &'static str = match drift.drift_kind {
        crate::index::axis::management::DriftKind::None => "none",
        crate::index::axis::management::DriftKind::EfSearchOnly => "ef_search_only",
        crate::index::axis::management::DriftKind::EfConstructionOrM => "rebuild_required",
        crate::index::axis::management::DriftKind::NprobeOnly => "ef_search_only",
        crate::index::axis::management::DriftKind::NlistOrQuantizer => "rebuild_required",
    };

    let clamped = drift.current_params.clamped_by_max_ef;
    let projected = drift.current_params.projected_recall_if_clamped;
    let report = RecallDriftHealth {
        wired: true,
        recall_target: Some(recall_target),
        baseline_vector_count: Some(baseline_n),
        current_vector_count: Some(current_n),
        kind: kind_str,
        needs_rebuild: drift.needs_rebuild(),
        hot_swap_possible: drift.hot_swap_possible(),
        summary: drift.summary.clone(),
        baseline_params: Some(RecallAdvisedParams {
            m: drift.baseline_params.m,
            ef_construction: drift.baseline_params.ef_construction,
            ef_search: drift.baseline_params.ef_search,
        }),
        current_params: Some(RecallAdvisedParams {
            m: drift.current_params.m,
            ef_construction: drift.current_params.ef_construction,
            ef_search: drift.current_params.ef_search,
        }),
        recommended_action: recommended_action_for(kind_str, clamped),
        max_ef_search,
        clamped_by_max_ef: clamped,
        projected_recall_at_clamped_ef: projected,
        algorithm: active_algorithm_for(&config),
    };
    crate::metrics::recall_drift_metrics::record_recall_drift_observation(&collection_id, kind_str);

    // No drift → confirm + exit.
    if matches!(
        drift.drift_kind,
        crate::index::axis::management::DriftKind::None
    ) {
        return Ok(Json(RecallTuneResponse {
            stability: "experimental",
            collection_id,
            report,
            action: "no_drift",
            applied_changes: Vec::new(),
        }));
    }

    // Rebuild required → the in-place tune can't fix it; the caller
    // must drive the recluster.
    if drift.needs_rebuild() {
        return Ok(Json(RecallTuneResponse {
            stability: "experimental",
            collection_id,
            report,
            action: "rebuild_required",
            applied_changes: Vec::new(),
        }));
    }

    // EfSearchOnly → apply the hot-swap.
    let Some(axis_manager) = crate::storage::engines::sst::core::get_sst_axis_manager() else {
        // No AXIS manager registered (e.g., HELIX-only deployment).
        // Report drift but flag that the surface isn't actionable.
        return Ok(Json(RecallTuneResponse {
            stability: "experimental",
            collection_id,
            report,
            action: "not_wired",
            applied_changes: Vec::new(),
        }));
    };

    // P2.4: dispatch on the collection's active algorithm. HNSW
    // collections hot-swap ef_search; IVF collections hot-swap
    // nprobe. Both return the same HotSwapOutcome shape so the
    // response-building code below stays algorithm-agnostic.
    let active_algo = active_algorithm_for(&config);
    let outcome = match active_algo {
        "ivf" => {
            // For IVF, the live-tunable knob is nprobe. Compute the
            // current advised nprobe from the IVF advisor (the
            // HNSW-shape `drift.current_params` doesn't carry IVF
            // sizing; ask the advisor fresh).
            let advisor = crate::index::axis::management::IvfIndexAdvisor::new();
            let metric = match config
                .distance_metric
                .and_then(|v| crate::proto::proximadb_v1::DistanceMetric::try_from(v).ok())
            {
                Some(crate::proto::proximadb_v1::DistanceMetric::Cosine) => {
                    crate::compute::distance_computation::DistanceMetric::Cosine
                }
                Some(crate::proto::proximadb_v1::DistanceMetric::Euclidean) => {
                    crate::compute::distance_computation::DistanceMetric::Euclidean
                }
                Some(crate::proto::proximadb_v1::DistanceMetric::DotProduct) => {
                    crate::compute::distance_computation::DistanceMetric::DotProduct
                }
                _ => crate::compute::distance_computation::DistanceMetric::Cosine,
            };
            let binary_rerank =
                crate::services::collection::recall_target::parse_binary_rerank_allowed(&config);
            let Some(advised) = advisor.advise(&crate::index::axis::management::AnnAdvisorInput {
                vector_count: current_n,
                top_k,
                recall_target,
                dimension: config.dimension,
                distance_metric: metric,
                max_query_latency_ms,
                max_memory_mb,
                binary_rerank_allowed: binary_rerank,
                modalities: Vec::new(),
            }) else {
                // IVF advisor declined — nothing to tune.
                return Ok(Json(RecallTuneResponse {
                    stability: "experimental",
                    collection_id,
                    report,
                    action: "not_wired",
                    applied_changes: Vec::new(),
                }));
            };
            let target_nprobe = match advised.algorithm {
                crate::index::axis::types::IndexAlgorithm::IVF { nprobe, .. } => nprobe,
                _ => {
                    return Ok(Json(RecallTuneResponse {
                        stability: "experimental",
                        collection_id,
                        report,
                        action: "not_wired",
                        applied_changes: Vec::new(),
                    }));
                }
            };
            axis_manager
                .apply_ivf_nprobe_hot_swap(&collection_id, target_nprobe)
                .await
                .map_err(|e| ApiError::Internal(format!("IVF hot-swap failed: {}", e)))?
        }
        _ => axis_manager
            .apply_hnsw_ef_hot_swap(&collection_id, drift.current_params.ef_search)
            .await
            .map_err(|e| ApiError::Internal(format!("HNSW hot-swap failed: {}", e)))?,
    };

    let (action, applied_changes) = if let Some(applied) = outcome.get("Applied") {
        crate::metrics::recall_drift_metrics::record_recall_drift_hot_swap_applied(
            &collection_id,
            crate::metrics::recall_drift_metrics::HOT_SWAP_TRIGGER_OPERATOR,
        );
        let changes: Vec<crate::index::axis::management::HotSwapEfChange> =
            serde_json::from_value(applied["changes"].clone()).unwrap_or_default();
        (
            "applied_hot_swap",
            changes
                .into_iter()
                .map(|c| RecallTuneEfChange {
                    index_name: c.index_name,
                    previous_ef_search: c.previous_ef_search,
                    new_ef_search: c.new_ef_search,
                })
                .collect(),
        )
    } else {
        ("not_wired", Vec::new())
    };

    Ok(Json(RecallTuneResponse {
        stability: "experimental",
        collection_id,
        report,
        action,
        applied_changes,
    }))
}

/// Response body for `POST /api/v2/_diagnostics/collections/:id/recluster`.
///
/// The handler always returns `applied = bool` plus a structured
/// `sized` block carrying the advisor's chosen (m, ef_construction,
/// ef_search) and rationale string. When `applied = false`,
/// `reason` explains why — typically "no recall_target tag",
/// "no records" (empty collection), or "axis manager not wired".
#[derive(Debug, serde::Serialize, PartialEq)]
pub struct RecallReclusterResponse {
    pub stability: &'static str,
    pub collection_id: String,
    pub applied: bool,
    /// Human-readable reason populated when `applied = false`, or
    /// the rebuild's summary line when `applied = true`. Empty
    /// string when there's nothing useful to say.
    pub reason: String,
    /// Number of records the rebuild ingested. `None` when the
    /// rebuild didn't run (no recall_target, no axis, etc.).
    pub rebuilt_vector_count: Option<u64>,
    /// The advisor's sizing decision for the rebuilt graph. `None`
    /// when the rebuild didn't run.
    pub sized: Option<RecallReclusterSized>,
}

#[derive(Debug, serde::Serialize, PartialEq)]
pub struct RecallReclusterSized {
    pub recall_target: f32,
    /// Stable algorithm discriminator — `"hnsw"` or `"ivf"` (P3
    /// adds `"hmgi"`). Matches `SupportedAlgorithm::label()` so
    /// dashboard filters can switch consistently across surfaces.
    pub algorithm: &'static str,

    // ─── HNSW fields ─────────────────────────────────────────
    /// HNSW graph degree. `None` for IVF rebuilds.
    pub m: Option<u32>,
    /// HNSW build-time candidate set. `None` for IVF rebuilds.
    pub ef_construction: Option<u32>,
    /// HNSW search-time beam. `None` for IVF rebuilds.
    pub ef_search: Option<u32>,

    // ─── IVF fields ──────────────────────────────────────────
    /// IVF cluster count. `None` for HNSW rebuilds.
    pub nlist: Option<u32>,
    /// IVF probe count per query. `None` for HNSW rebuilds.
    pub nprobe: Option<u32>,
    /// `true` if the rebuild stamped a PQ quantizer alongside IVF
    /// (binary rerank tier). `false` for raw IVF; `None` for HNSW.
    pub pq_enabled: Option<bool>,

    /// Advisor's free-text rationale string — for operator dashboards.
    pub rationale: String,
}

// ─── Phase 8 F4a (TD-094): collection suspend / resume ──────────────────────

/// Response for `POST /api/v2/collections/:id/suspend`.
#[derive(Debug, Serialize)]
pub struct SuspendResponse {
    pub collection_id: String,
    pub suspended: bool,
}

/// Response for `POST /api/v2/collections/:id/resume`.
#[derive(Debug, Serialize)]
pub struct ResumeResponse {
    pub collection_id: String,
    pub resumed: bool,
    pub in_memory_index: bool,
}

/// 404 a missing collection (tenant-scoped) before mutating index state.
async fn ensure_collection_exists(
    state: &AppState,
    collection_id: &str,
    tenant: &TenantContext,
) -> ApiResult<()> {
    let request = CollectionRequest {
        operation: CollectionOperation::CollectionGet as i32,
        collection_id: Some(collection_id.to_string()),
        collection_config: None,
        query_params: Default::default(),
        options: Default::default(),
        migration_config: Default::default(),
    };
    state
        .api_handlers
        .handle_collection_operation_for_tenant(request, Some(&tenant.tenant_id))
        .await
        .map_err(|e| {
            if e.to_string().contains("not found") {
                ApiError::CollectionNotFound(collection_id.to_string())
            } else {
                ApiError::Internal(format!("Failed to get collection: {e}"))
            }
        })?;
    Ok(())
}

/// `POST /api/v2/collections/:id/suspend` — Phase 8 F4a. Evicts the collection's
/// in-memory IVF index to free memory (the persisted `ivf.bin` + catalog
/// metadata stay); the next query (or `/resume`) warm-loads it. Admin-gated.
pub async fn post_collection_suspend_v2(
    Path(collection_id): Path<String>,
    Extension(tenant): Extension<TenantContext>,
    user_context: Option<Extension<crate::security::rbac_service::UnifiedUserContext>>,
    State(state): State<AppState>,
) -> ApiResult<Json<SuspendResponse>> {
    if collection_id.is_empty() {
        return Err(ApiError::InvalidArgument(
            "Collection ID is required".to_string(),
        ));
    }
    require_recall_admin(user_context.as_ref().map(|e| &e.0), "suspend")?;
    ensure_collection_exists(&state, &collection_id, &tenant).await?;

    let Some(axis) = crate::storage::engines::sst::core::get_sst_axis_manager() else {
        return Err(ApiError::NotImplemented(
            "suspend requires the AXIS index manager (not available in this deployment)"
                .to_string(),
        ));
    };
    axis.suspend_collection(&collection_id)
        .await
        .map_err(|e| ApiError::Internal(format!("suspend '{collection_id}': {e:#}")))?;
    info!("V2 API: suspended collection '{}'", collection_id);
    Ok(Json(SuspendResponse {
        collection_id,
        suspended: true,
    }))
}

/// `POST /api/v2/collections/:id/resume` — Phase 8 F4a. Eagerly warm-loads a
/// suspended collection's IVF index from disk now (lazy resume also happens on
/// the next query). Admin-gated.
pub async fn post_collection_resume_v2(
    Path(collection_id): Path<String>,
    Extension(tenant): Extension<TenantContext>,
    user_context: Option<Extension<crate::security::rbac_service::UnifiedUserContext>>,
    State(state): State<AppState>,
) -> ApiResult<Json<ResumeResponse>> {
    if collection_id.is_empty() {
        return Err(ApiError::InvalidArgument(
            "Collection ID is required".to_string(),
        ));
    }
    require_recall_admin(user_context.as_ref().map(|e| &e.0), "resume")?;
    ensure_collection_exists(&state, &collection_id, &tenant).await?;

    let Some(axis) = crate::storage::engines::sst::core::get_sst_axis_manager() else {
        return Err(ApiError::NotImplemented(
            "resume requires the AXIS index manager (not available in this deployment)".to_string(),
        ));
    };
    let resumed = axis
        .resume_collection(&collection_id)
        .await
        .map_err(|e| ApiError::Internal(format!("resume '{collection_id}': {e:#}")))?;
    info!(
        "V2 API: resumed collection '{}' (served={})",
        collection_id, resumed
    );
    Ok(Json(ResumeResponse {
        collection_id,
        resumed,
        in_memory_index: resumed,
    }))
}

/// Recall-aware HNSW rebuild handler.
///
/// Resolves the `DriftKind::EfConstructionOrM` arm of recall drift:
///
/// 1. Reads the collection — short-circuits if the collection has
///    no `recall_target:<float>` tag (caller never opted in).
/// 2. Reads every record via
///    `VectorOperationsService::list_all_records_with_tenant_context`
///    (same path the recluster + dedup discovery passes use).
/// 3. Calls
///    `AxisManager::rebuild_and_swap_hnsw_index_for_recall_target`
///    with the records + recall_target — atomic swap, also updates
///    the live strategy so post-rebuild queries pick up the new
///    `ef_search`.
/// 4. Returns the advisor's sizing decision so the operator sees
///    exactly what was built.
///
/// Returns `404` for missing collection, `400` for empty
/// collection_id, `401`/`403` for missing/insufficient operator
/// permission. Recluster reads the entire collection and rebuilds
/// the HNSW graph — non-trivial CPU + memory — so it sits behind
/// the same `SystemAdmin` / `ConfigureSystem` gate as
/// `primary_pod`.
/// When the `axis` feature is off the rebuild branch is elided and this
/// handler short-circuits with a 501; the `tenant` / `state` extractors are
/// then unused, so silence that one lint in a no-axis build only.
#[cfg_attr(not(feature = "axis"), allow(unused_variables))]
pub async fn post_collection_recluster_v2(
    Path(collection_id): Path<String>,
    Extension(tenant): Extension<TenantContext>,
    user_context: Option<Extension<crate::security::rbac_service::UnifiedUserContext>>,
    State(state): State<AppState>,
) -> ApiResult<Json<RecallReclusterResponse>> {
    debug!(
        "V2 API: recluster for collection '{}' (experimental, recall-aware)",
        collection_id
    );

    if collection_id.is_empty() {
        return Err(ApiError::InvalidArgument(
            "Collection ID is required".to_string(),
        ));
    }

    require_recall_admin(user_context.as_ref().map(|e| &e.0), "recluster")?;

    // The recall-aware rebuild reads the live collection and downcasts to the
    // concrete AXIS index manager to drive an advisor-sized graph swap. The
    // whole sequence is gated on the `axis` feature; when it is off there is
    // no manager to downcast, so surface a 501 rather than attempt the rebuild
    // (the empty-id + admin gates above still run).
    #[cfg(not(feature = "axis"))]
    return Err(ApiError::NotImplemented(
        "recluster requires the AXIS feature, which is not enabled in this build".to_string(),
    ));

    #[cfg(feature = "axis")]
    {
        // (1) Fetch collection config to read the recall_target tag.
        let request = CollectionRequest {
            operation: CollectionOperation::CollectionGet as i32,
            collection_id: Some(collection_id.clone()),
            collection_config: None,
            query_params: Default::default(),
            options: Default::default(),
            migration_config: Default::default(),
        };
        let resp = state
            .api_handlers
            .handle_collection_operation_for_tenant(request, Some(&tenant.tenant_id))
            .await
            .map_err(|e| {
                if e.to_string().contains("not found") {
                    ApiError::CollectionNotFound(collection_id.clone())
                } else {
                    ApiError::Internal(format!("Failed to get collection: {}", e))
                }
            })?;

        let collection = resp.collection.unwrap_or_default();
        let config = collection.config.unwrap_or_default();

        let Some(recall_target) =
            crate::services::collection::recall_target::parse_recall_target(&config)
        else {
            return Ok(Json(RecallReclusterResponse {
                stability: "experimental",
                collection_id,
                applied: false,
                reason: "collection has no recall_target: tag — nothing to size against"
                    .to_string(),
                rebuilt_vector_count: None,
                sized: None,
            }));
        };

        let Some(axis_manager) = crate::storage::engines::sst::core::get_sst_axis_manager() else {
            return Ok(Json(RecallReclusterResponse {
                stability: "experimental",
                collection_id,
                applied: false,
                reason: "AXIS manager not registered for this deployment".to_string(),
                rebuilt_vector_count: None,
                sized: None,
            }));
        };

        // (2) Read every record. Resolves the user-facing name to the
        // canonical internal id (same as the discovery recluster pass).
        let vector_ops = &state.vector_operations_service;
        let internal_id = vector_ops
            .resolve_collection_object_id(&collection_id)
            .await
            .map_err(|e| ApiError::NotFound(e.to_string()))?
            .to_string();
        let records = vector_ops
            .list_all_records_with_tenant_context(internal_id.as_str(), None)
            .await
            .map_err(|e| ApiError::Internal(format!("Failed to list records: {}", e)))?;

        if records.is_empty() {
            return Ok(Json(RecallReclusterResponse {
                stability: "experimental",
                collection_id,
                applied: false,
                reason: "collection has no records to rebuild".to_string(),
                rebuilt_vector_count: Some(0),
                sized: None,
            }));
        }

        let count = records.len() as u64;
        let top_k = crate::services::collection::recall_target::resolve_top_k(&config);
        let max_ef_search =
            crate::services::collection::recall_target::parse_max_ef_search(&config);
        let max_query_latency_ms =
            crate::services::collection::recall_target::parse_max_query_latency_ms(&config);
        let max_memory_mb =
            crate::services::collection::recall_target::parse_max_memory_mb(&config);
        let binary_rerank =
            crate::services::collection::recall_target::parse_binary_rerank_allowed(&config);

        // (3) Dispatch the rebuild on the collection's active
        // algorithm. HNSW path stays untouched (existing behavior);
        // IVF path calls the new advisor-aware rebuild and
        // normalises the response shape via the `algorithm` literal
        // and per-algorithm sized fields.
        let active_algo = active_algorithm_for(&config);
        let sized: Option<RecallReclusterSized> = match active_algo {
            "ivf" => {
                // Downcast to concrete AxisManager for recall-target advisor methods
                let axis_mgr = axis_manager
                    .as_any()
                    .downcast_ref::<crate::index::axis::management::AxisManager>()
                    .ok_or_else(|| {
                        ApiError::Internal("AXIS manager downcast failed".to_string())
                    })?;
                let advised = axis_mgr
                    .rebuild_and_swap_ivf_index_for_recall_target(
                        internal_id.as_str(),
                        &records,
                        recall_target,
                        top_k,
                        max_query_latency_ms,
                        max_memory_mb,
                        binary_rerank,
                    )
                    .await
                    .map_err(|e| ApiError::Internal(format!("IVF rebuild failed: {}", e)))?;
                let Some(advised) = advised else {
                    return Ok(Json(RecallReclusterResponse {
                        stability: "experimental",
                        collection_id,
                        applied: false,
                        reason:
                            "IVF advisor declined or no usable embeddings — recall_target may exceed IVF ceiling"
                                .to_string(),
                        rebuilt_vector_count: Some(count),
                        sized: None,
                    }));
                };
                let (nlist, nprobe, pq_enabled) = match &advised.algorithm {
                    crate::index::axis::types::IndexAlgorithm::IVF {
                        nlist,
                        nprobe,
                        quantizer,
                    } => (*nlist, *nprobe, quantizer.is_some()),
                    other => {
                        unreachable!("IVF rebuild returned non-IVF algorithm spec: {:?}", other)
                    }
                };
                Some(RecallReclusterSized {
                    recall_target,
                    algorithm: "ivf",
                    m: None,
                    ef_construction: None,
                    ef_search: None,
                    nlist: Some(nlist),
                    nprobe: Some(nprobe),
                    pq_enabled: Some(pq_enabled),
                    rationale: advised.rationale,
                })
            }
            _ => {
                // Downcast to concrete AxisManager for recall-target advisor methods
                let axis_mgr = axis_manager
                    .as_any()
                    .downcast_ref::<crate::index::axis::management::AxisManager>()
                    .ok_or_else(|| {
                        ApiError::Internal("AXIS manager downcast failed".to_string())
                    })?;
                let advised = axis_mgr
                    .rebuild_and_swap_hnsw_index_for_recall_target(
                        internal_id.as_str(),
                        &records,
                        recall_target,
                        top_k,
                        max_ef_search,
                    )
                    .await
                    .map_err(|e| ApiError::Internal(format!("HNSW rebuild failed: {}", e)))?;
                let Some(advised) = advised else {
                    return Ok(Json(RecallReclusterResponse {
                        stability: "experimental",
                        collection_id,
                        applied: false,
                        reason: "no usable embeddings in record set".to_string(),
                        rebuilt_vector_count: Some(count),
                        sized: None,
                    }));
                };
                Some(RecallReclusterSized {
                    recall_target,
                    algorithm: "hnsw",
                    m: Some(advised.m),
                    ef_construction: Some(advised.ef_construction),
                    ef_search: Some(advised.ef_search),
                    nlist: None,
                    nprobe: None,
                    pq_enabled: None,
                    rationale: advised.rationale,
                })
            }
        };

        // After a rebuild the recall-drift state collapses to "none"
        // for this collection (the new graph is sized exactly to the
        // current advised params). Reflect that on the gauge so
        // dashboards / alerts clear immediately rather than waiting
        // for the next route-health GET or sweep tick.
        crate::metrics::recall_drift_metrics::record_recall_drift_observation(
            &collection_id,
            "none",
        );

        let reason = sized
            .as_ref()
            .map(|s| s.rationale.clone())
            .unwrap_or_default();
        Ok(Json(RecallReclusterResponse {
            stability: "experimental",
            collection_id,
            applied: sized.is_some(),
            reason,
            rebuilt_vector_count: Some(count),
            sized,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// "Unknown" and "empty" must not look the same on the wire.
    ///
    /// The counter is maintained by `MetadataStore::update_stats`, which the
    /// record write path never calls, so a populated collection routinely has no
    /// stats block. Coercing that to `0` gives a confident wrong answer that is
    /// indistinguishable from a genuinely empty collection — a downstream project
    /// read it as data loss and filed a durability bug against us (#1527).
    #[test]
    fn record_count_is_none_when_the_server_has_no_stats() {
        assert_eq!(
            record_count_for_wire(None),
            None,
            "an absent stats block means unknown; reporting 0 claims the collection is empty"
        );
    }

    #[test]
    fn record_count_reports_a_known_value_including_a_real_zero() {
        let populated = crate::proto::proximadb_v1::CollectionStats {
            vector_count: 1_000,
            ..Default::default()
        };
        assert_eq!(record_count_for_wire(Some(&populated)), Some(1_000));

        // A written zero is a real answer and must survive as Some(0) — the point
        // is to distinguish "known to be empty" from "not known".
        let genuinely_empty = crate::proto::proximadb_v1::CollectionStats {
            vector_count: 0,
            ..Default::default()
        };
        assert_eq!(record_count_for_wire(Some(&genuinely_empty)), Some(0));
    }

    /// A negative count is corruption, not a value to forward.
    #[test]
    fn record_count_clamps_a_negative_count_rather_than_wrapping() {
        let corrupt = crate::proto::proximadb_v1::CollectionStats {
            vector_count: -5,
            ..Default::default()
        };
        assert_eq!(record_count_for_wire(Some(&corrupt)), Some(0));
    }

    #[test]
    fn active_algorithm_defaults_to_hnsw_for_legacy_config() {
        // No tags + no index_configs → default literal "hnsw". The
        // route-health response leans on this for every collection
        // pre-dating the IVF / HMGI work.
        let cfg = crate::proto::proximadb_v1::CollectionConfig {
            name: "c".to_string(),
            dimension: 128,
            ..Default::default()
        };
        assert_eq!(active_algorithm_for(&cfg), "hnsw");
    }

    #[test]
    fn active_algorithm_detects_ivf_from_index_config() {
        // An IVF entry in index_configs flips the literal — drives
        // the recall-tune handler to call apply_ivf_nprobe_hot_swap.
        let cfg = crate::proto::proximadb_v1::CollectionConfig {
            name: "c".to_string(),
            dimension: 128,
            index_configs: vec![crate::proto::proximadb_v1::IndexConfig {
                index_name: "ivf_primary".to_string(),
                algorithm: crate::proto::proximadb_v1::IndexingAlgorithm::Ivf as i32,
                ..Default::default()
            }],
            ..Default::default()
        };
        assert_eq!(active_algorithm_for(&cfg), "ivf");
    }

    #[test]
    fn active_algorithm_detects_hmgi_from_modalities_tag() {
        // ≥ 2 modalities → HMGI literal. Wins over IVF index_config
        // (modalities is the operator's explicit multi-modal signal).
        let cfg = crate::proto::proximadb_v1::CollectionConfig {
            name: "c".to_string(),
            dimension: 128,
            tags: vec!["modalities:text,image".to_string()],
            ..Default::default()
        };
        assert_eq!(active_algorithm_for(&cfg), "hmgi");
    }

    #[test]
    fn active_algorithm_single_modality_falls_through_to_hnsw() {
        // 1 modality declared → HMGI declines (needs ≥ 2). Falls
        // through to the IVF / HNSW default.
        let cfg = crate::proto::proximadb_v1::CollectionConfig {
            name: "c".to_string(),
            dimension: 128,
            tags: vec!["modalities:text".to_string()],
            ..Default::default()
        };
        assert_eq!(active_algorithm_for(&cfg), "hnsw");
    }

    #[test]
    fn create_collection_failure_maps_collection_exists_to_conflict() {
        match collection_create_failure_error("symbols", Some("COLLECTION_EXISTS")) {
            ApiError::AlreadyExists(message) => {
                assert!(message.contains("symbols"));
            }
            other => panic!("expected AlreadyExists, got {other:?}"),
        }

        match collection_create_failure_error("symbols", Some("collection already exists")) {
            ApiError::AlreadyExists(message) => {
                assert!(message.contains("symbols"));
            }
            other => panic!("expected AlreadyExists, got {other:?}"),
        }
    }

    #[test]
    fn create_collection_failure_maps_other_errors_to_internal() {
        match collection_create_failure_error("symbols", Some("catalog write failed")) {
            ApiError::Internal(message) => {
                assert!(message.contains("symbols"));
                assert!(message.contains("catalog write failed"));
            }
            other => panic!("expected Internal, got {other:?}"),
        }

        match collection_create_failure_error("symbols", None) {
            ApiError::Internal(message) => {
                assert!(message.contains("unknown error"));
            }
            other => panic!("expected Internal, got {other:?}"),
        }
    }

    #[test]
    fn test_create_request_deserialization() {
        let json = r#"{
            "name": "products",
            "dimension": 768,
            "engine": "sst",
            "schema": {
                "columns": [
                    {"name": "category", "data_type": "text", "indexed": true},
                    {"name": "price", "data_type": "float", "filterable": true}
                ],
                "enforcement": "hybrid",
                "allow_additional_fields": true
            },
            "enable_proxima_record": true
        }"#;

        let request: CreateCollectionV2Request = serde_json::from_str(json).unwrap();
        assert_eq!(request.name, "products");
        assert_eq!(request.dimension, 768);
        assert_eq!(request.engine, Some("sst".to_string()));
        assert!(request.enable_proxima_record.unwrap());

        let schema = request.schema.unwrap();
        assert_eq!(schema.columns.len(), 2);
        assert_eq!(schema.enforcement, Some("hybrid".to_string()));
    }

    #[test]
    fn test_schema_definition_serialization() {
        let schema = SchemaDefinition {
            columns: vec![RestColumnDefinition {
                name: "title".to_string(),
                data_type: "text".to_string(),
                nullable: Some(false),
                indexed: Some(true),
                filterable: Some(true),
                max_length: Some(255),
                precision: None,
                scale: None,
                vector_dimension: None,
            }],
            enforcement: Some("strict".to_string()),
            allow_additional_fields: Some(false),
        };

        let json = serde_json::to_string(&schema).unwrap();
        assert!(json.contains("\"title\""));
        assert!(json.contains("\"text\""));
        assert!(json.contains("\"strict\""));
    }

    #[test]
    fn test_decimal_column_validation() {
        let json = r#"{
            "name": "price",
            "data_type": "decimal",
            "precision": 10,
            "scale": 2
        }"#;

        let column: RestColumnDefinition = serde_json::from_str(json).unwrap();
        assert_eq!(column.precision, Some(10));
        assert_eq!(column.scale, Some(2));
    }

    #[test]
    fn test_v2_storage_engine_mapping_uses_proto_enum_values() {
        assert_eq!(
            crate::core::conversions::parse_storage_engine("sst").unwrap() as i32,
            crate::proto::proximadb_v1::StorageEngine::Sst as i32
        );
        assert_eq!(
            crate::core::conversions::parse_storage_engine("tst").unwrap() as i32,
            crate::proto::proximadb_v1::StorageEngine::Tst as i32
        );
        assert_eq!(
            collection_storage_engine_label(Some(
                crate::proto::proximadb_v1::StorageEngine::Viper as i32
            )),
            "viper"
        );
        assert_eq!(
            collection_storage_engine_label(Some(
                crate::proto::proximadb_v1::StorageEngine::Sst as i32
            )),
            "sst"
        );
        assert_eq!(collection_storage_engine_label(None), "auto");
    }

    #[test]
    fn test_v2_distance_metric_mapping_uses_proto_enum_values() {
        assert_eq!(
            crate::core::conversions::parse_distance_metric("dot_product").unwrap() as i32,
            crate::proto::proximadb_v1::DistanceMetric::DotProduct as i32
        );
        assert_eq!(
            collection_distance_metric_label(Some(
                crate::proto::proximadb_v1::DistanceMetric::Cosine as i32
            )),
            "cosine"
        );
        assert_eq!(
            collection_distance_metric_label(Some(
                crate::proto::proximadb_v1::DistanceMetric::Euclidean as i32
            )),
            "euclidean"
        );
        assert_eq!(collection_distance_metric_label(None), "cosine");
    }

    #[test]
    fn test_v2_stats_do_not_wrap_negative_proto_values() {
        assert_eq!(non_negative_stat(12), 12);
        assert_eq!(non_negative_stat(-1), 0);
    }

    // ------------------------------------------------------------------
    // Route-health builder tests. These lock the contract shape so future
    // capability flips are deliberate edits to both the code and the test,
    // not silent drifts.
    // ------------------------------------------------------------------

    #[test]
    fn route_health_builder_reports_resolved_collection_facts() {
        let h = build_route_health(
            "products".to_string(),
            "sst".to_string(),
            768,
            "cosine".to_string(),
            42,
            4096,
            1024,
        );
        assert_eq!(h.collection_id, "products");
        assert_eq!(h.engine, "sst");
        assert_eq!(h.dimension, 768);
        assert_eq!(h.distance_metric, "cosine");
        assert_eq!(h.record_count, 42);
        assert_eq!(h.storage_size_bytes, 4096);
        assert_eq!(h.index_size_bytes, 1024);
        assert_eq!(h.schema_version, "v1");
        assert_eq!(h.stability, "experimental");
    }

    #[test]
    fn route_health_filtered_ann_reports_td_064_state() {
        let h = build_route_health(
            "c".to_string(),
            "sst".to_string(),
            8,
            "cosine".to_string(),
            0,
            0,
            0,
        );
        assert!(h.filtered_ann.id_predicate_supported);
        assert!(
            h.filtered_ann.record_aware_predicates,
            "AxisManager query path evaluates ProximaRecord metadata during predicate traversal"
        );
        // Shortfall path is genuinely wired: predicate_diagnostics::scope in
        // REST records.rs + gRPC record_service.rs, captured shortfall set
        // via mark_predicate_shortfall on SearchPlanTrace, metrics counter
        // axis_predicate_shortfall_total fires on every event.
        assert!(
            h.filtered_ann.post_filter_shortfall_disclosure,
            "TD-064 shortfall path is wired (diagnostics + trace + metrics)"
        );
        assert_eq!(
            h.filtered_ann.td_064_status,
            "record_predicate_and_shortfall_wired"
        );
    }

    #[test]
    fn route_health_writes_match_current_v2_surface() {
        let h = build_route_health(
            "c".to_string(),
            "sst".to_string(),
            8,
            "cosine".to_string(),
            0,
            0,
            0,
        );
        assert!(h.writes.insert);
        assert!(h.writes.upsert);
        assert!(h.writes.update);
        assert!(h.writes.delete);
        // Unwired today — flip these only when REST/gRPC actually accepts them.
        assert!(!h.writes.conditional_write);
        assert!(!h.writes.filter_write);
        assert!(!h.writes.patch);
    }

    #[test]
    fn route_health_freshness_search_request_modes_all_wired() {
        // All three VectorFreshnessMode variants are honored by
        // should_scan_delta_with_time. The bounded_stale_time_bound_check
        // flag specifically requires the watermark_ns threading wired in
        // commit e34a06225 (Phase 5 slice 5.10). Flipping any of these to
        // false requires the corresponding mode being removed from the
        // enum or its branch being disconnected from the search path.
        let h = build_route_health(
            "c".to_string(),
            "sst".to_string(),
            8,
            "cosine".to_string(),
            0,
            0,
            0,
        );
        assert!(h.freshness.search_request_modes.strong);
        assert!(h.freshness.search_request_modes.bounded_stale);
        assert!(h.freshness.search_request_modes.stale_ok);
        assert!(
            h.freshness
                .search_request_modes
                .bounded_stale_time_bound_check
        );
        // Collection-default modes still not stored on the catalog —
        // separate slice. The reason stays in degraded_reasons.
        assert!(!h.freshness.collection_level_modes_wired);
        assert!(
            h.degraded_reasons
                .contains(&DegradedReason::FreshnessModesNotCollectionLevel),
            "search-request modes being wired does NOT imply collection-default modes"
        );
    }

    #[test]
    fn route_health_object_economy_is_explicitly_sst_only() {
        let sst = build_route_health(
            "c".to_string(),
            "sst".to_string(),
            8,
            "cosine".to_string(),
            0,
            0,
            0,
        );
        assert!(sst.object_economy.eligible);
        assert!(sst.object_economy.directory_format_present);
        assert_eq!(sst.object_economy.live_status, "not_checked");
        assert_eq!(sst.object_economy.route_hint, Some("object_economy"));

        let viper = build_route_health(
            "c".to_string(),
            "viper".to_string(),
            8,
            "cosine".to_string(),
            0,
            0,
            0,
        );
        assert!(!viper.object_economy.eligible);
        assert!(!viper.object_economy.directory_format_present);
        assert_eq!(viper.object_economy.live_status, "not_applicable");
        assert_eq!(viper.object_economy.route_hint, None);
    }

    #[test]
    fn route_health_suspension_block_defaults_unobservable_in_builder() {
        // The pure builder has no AXIS manager handle, so the suspension block
        // defaults to unobservable; the async handler patches it with live state.
        let h = build_route_health(
            "c".to_string(),
            "sst".to_string(),
            8,
            "cosine".to_string(),
            0,
            0,
            0,
        );
        assert!(!h.suspension.observable);
        assert!(!h.suspension.suspended);
        assert!(!h.suspension.in_memory_index);
        // ADR-023 R3 (c): the builder default for cold-serving is unobservable
        // (the async handler patches it from the AXIS manager).
        assert!(!h.cold_serving.observable);
        assert_eq!(h.cold_serving.serving_state, None);
        assert!(!h.cold_serving.serving_reduced_recall);
        // Serializes as a nested object on the route-health contract.
        let json = serde_json::to_value(&h).unwrap();
        assert!(
            json.get("suspension").is_some(),
            "suspension block present in contract"
        );
        assert!(
            json.get("cold_serving").is_some(),
            "cold_serving block present in contract"
        );
    }

    #[test]
    fn cold_serving_health_from_status_classifies_recall() {
        use crate::index::axis::IvfServingState;
        // Cold window: ColdBinaryOnly with fewer warm clusters fetched than total
        // ⇒ reduced-recall serving.
        let cold = ColdServingHealth::from_status(IvfServingState::ColdBinaryOnly, 1, 4);
        assert!(cold.observable);
        assert_eq!(cold.serving_state, Some("ColdBinaryOnly"));
        assert!(cold.serving_reduced_recall);
        assert_eq!(cold.warm_clusters_fetched, 1);
        assert_eq!(cold.warm_clusters_total, 4);
        // All warm clusters present (even if still ColdBinaryOnly) ⇒ full recall.
        let warm = ColdServingHealth::from_status(IvfServingState::ColdBinaryOnly, 4, 4);
        assert!(!warm.serving_reduced_recall);
        // FullTwoStage ⇒ full recall regardless of counters.
        let full = ColdServingHealth::from_status(IvfServingState::FullTwoStage, 0, 0);
        assert_eq!(full.serving_state, Some("FullTwoStage"));
        assert!(!full.serving_reduced_recall);
    }

    #[test]
    fn route_health_degraded_reasons_serialize_as_screaming_snake() {
        let reasons = vec![
            DegradedReason::FilteredAnnRecordPredicateBridgePartial,
            DegradedReason::ObjectEconomyDirectoryDegraded,
            DegradedReason::RecallProbeNotWired,
            DegradedReason::ConditionalWritesUnsupported,
        ];
        let s = serde_json::to_string(&reasons).unwrap();
        assert!(s.contains("FILTERED_ANN_RECORD_PREDICATE_BRIDGE_PARTIAL"));
        assert!(s.contains("OBJECT_ECONOMY_DIRECTORY_DEGRADED"));
        assert!(s.contains("RECALL_PROBE_NOT_WIRED"));
        assert!(s.contains("CONDITIONAL_WRITES_UNSUPPORTED"));
    }

    #[test]
    fn route_health_degraded_reasons_are_the_expected_set() {
        // Snapshot of the reasons set. Adding/removing a reason without updating
        // this assertion would silently change the contract. RecallProbeNotWired
        // dropped out once TD-075 wired the gate into the AXIS query path.
        let h = build_route_health(
            "c".to_string(),
            "sst".to_string(),
            8,
            "cosine".to_string(),
            0,
            0,
            0,
        );
        let expected = vec![
            DegradedReason::ObjectEconomyLiveStatusNotReachable,
            DegradedReason::FreshnessModesNotCollectionLevel,
            DegradedReason::ConditionalWritesUnsupported,
            DegradedReason::FilterWritesUnsupported,
        ];
        assert_eq!(h.degraded_reasons, expected);
    }

    #[test]
    fn route_health_json_shape_snapshot() {
        // Snapshot of the top-level JSON keys. Locks the contract so a
        // field rename surfaces as a test diff. Field *values* are
        // covered by the other tests; this one is shape-only.
        let h = build_route_health(
            "c".to_string(),
            "sst".to_string(),
            8,
            "cosine".to_string(),
            0,
            0,
            0,
        );
        let v: serde_json::Value = serde_json::to_value(&h).unwrap();
        let obj = v.as_object().expect("response is a JSON object");
        let mut keys: Vec<&str> = obj.keys().map(|k| k.as_str()).collect();
        keys.sort();
        assert_eq!(
            keys,
            vec![
                "cold_serving",
                "collection_id",
                "degraded_reasons",
                "dimension",
                "discovery",
                "distance_metric",
                "engine",
                "filtered_ann",
                "freshness",
                "index_size_bytes",
                "object_economy",
                "pinning",
                "recall_drift",
                "recall_probe",
                "record_count",
                "schema_version",
                "stability",
                "storage_size_bytes",
                "suspension",
                "writes",
            ]
        );
    }

    #[test]
    fn route_health_recall_drift_unwired_by_default() {
        // Without the live handler patching recall_drift (no
        // recall_target: tag), the builder returns the unwired
        // sentinel.
        let h = build_route_health(
            "c".to_string(),
            "sst".to_string(),
            128,
            "cosine".to_string(),
            0,
            0,
            0,
        );
        assert!(!h.recall_drift.wired);
        assert_eq!(h.recall_drift.kind, "unwired");
        assert!(!h.recall_drift.needs_rebuild);
        assert!(!h.recall_drift.hot_swap_possible);
        assert!(h.recall_drift.recall_target.is_none());
        assert!(h.recall_drift.summary.is_empty());
    }

    #[test]
    fn recall_tune_response_serializes_action_strings() {
        // Pin the action enum-string mapping so dashboards / scripts
        // can rely on the literals.
        let actions: &[&'static str] = &[
            "applied_hot_swap",
            "no_drift",
            "rebuild_required",
            "not_wired",
        ];
        for action in actions {
            let resp = RecallTuneResponse {
                stability: "experimental",
                collection_id: "c1".to_string(),
                report: RecallDriftHealth::unwired(),
                action,
                applied_changes: Vec::new(),
            };
            let v = serde_json::to_value(&resp).unwrap();
            assert_eq!(v["action"], *action);
            assert_eq!(v["stability"], "experimental");
            assert_eq!(v["collection_id"], "c1");
            assert!(v["report"].is_object());
            assert!(v["applied_changes"].is_array());
        }
    }

    fn ctx_with_permissions(
        perms: Vec<crate::security::rbac_service::UnifiedPermission>,
    ) -> crate::security::rbac_service::UnifiedUserContext {
        use chrono::Utc;
        crate::security::rbac_service::UnifiedUserContext {
            user_id: "test_user".to_string(),
            tenant_id: None,
            roles: Vec::new(),
            effective_permissions: perms.into_iter().collect(),
            auth_method: crate::security::rbac_service::UnifiedAuthMethod::Internal,
            session_id: "test_session".to_string(),
            expires_at: None,
            created_at: Utc::now(),
            metadata: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn require_recall_admin_allows_when_no_context_in_unified_port_mode() {
        // The unified-port REST server doesn't attach
        // auth_middleware_unified, so Option<Extension<...>> resolves
        // to None even when security is enabled in config. To keep
        // recall-tune / recluster reachable under that config the
        // helper returns Ok with a stable "dev:..." user_id; audit
        // logs see the surrogate identity, and operator tooling that
        // explicitly relies on the multi-port + auth path still
        // gets the real `ctx.user_id` from the Some arm below.
        let res = require_recall_admin(None, "recall-tune")
            .expect("None ctx must pass in unified-port no-auth mode");
        assert!(
            res.starts_with("dev:"),
            "surrogate user_id must be the `dev:` prefix; got {}",
            res
        );
    }

    #[test]
    fn require_recall_admin_rejects_insufficient_permissions() {
        // A user with TenantRead only must be 403'd — the AXIS
        // mutation endpoints are explicitly operator-scoped, not
        // tenant-scoped, even when the caller has tenant read.
        let ctx = ctx_with_permissions(vec![
            crate::security::rbac_service::UnifiedPermission::TenantRead,
            crate::security::rbac_service::UnifiedPermission::ListCollections,
        ]);
        let res = require_recall_admin(Some(&ctx), "recluster");
        let err = res.expect_err("tenant-read-only ctx must be rejected");
        match err {
            ApiError::Forbidden(msg) => {
                assert!(msg.contains("recluster"));
                assert!(msg.contains("SystemAdmin"));
                assert!(msg.contains("ConfigureSystem"));
            }
            other => panic!("expected Forbidden, got {:?}", other),
        }
    }

    #[test]
    fn require_recall_admin_accepts_system_admin() {
        let ctx = ctx_with_permissions(vec![
            crate::security::rbac_service::UnifiedPermission::SystemAdmin,
        ]);
        let user_id = require_recall_admin(Some(&ctx), "recluster").expect("SystemAdmin must pass");
        assert_eq!(user_id, "test_user", "returned id is audit-log friendly");
    }

    #[test]
    fn require_recall_admin_accepts_configure_system() {
        // ConfigureSystem alone is also sufficient — matches the
        // primary_pod operator gate.
        let ctx = ctx_with_permissions(vec![
            crate::security::rbac_service::UnifiedPermission::ConfigureSystem,
        ]);
        let user_id =
            require_recall_admin(Some(&ctx), "recall-tune").expect("ConfigureSystem must pass");
        assert_eq!(user_id, "test_user");
    }

    #[test]
    fn require_recall_admin_rejects_tenant_admin() {
        // TenantAdmin is *not* sufficient — AXIS mutations are
        // cross-tenant infrastructure decisions. A tenant admin
        // can't reshape another tenant's index. Same boundary as
        // primary_pod operator endpoints.
        let ctx = ctx_with_permissions(vec![
            crate::security::rbac_service::UnifiedPermission::TenantAdmin,
        ]);
        let res = require_recall_admin(Some(&ctx), "recluster");
        assert!(
            matches!(res, Err(ApiError::Forbidden(_))),
            "TenantAdmin must NOT bypass operator gate"
        );
    }

    #[test]
    fn recluster_response_applied_serializes_sizing() {
        let resp = RecallReclusterResponse {
            stability: "experimental",
            collection_id: "products".to_string(),
            applied: true,
            reason: "tier r=0.95 → m=32 ...".to_string(),
            rebuilt_vector_count: Some(123_456),
            sized: Some(RecallReclusterSized {
                recall_target: 0.95,
                algorithm: "hnsw",
                m: Some(32),
                ef_construction: Some(256),
                ef_search: Some(409),
                nlist: None,
                nprobe: None,
                pq_enabled: None,
                rationale: "tier r=0.95 → m=32 ...".to_string(),
            }),
        };
        let v = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["applied"], true);
        assert_eq!(v["collection_id"], "products");
        assert_eq!(v["stability"], "experimental");
        assert_eq!(v["rebuilt_vector_count"], 123_456);
        // f32 → JSON Number → ~0.949999...; compare within 1e-3
        // for the float field, exact for the integer fields.
        let rt = v["sized"]["recall_target"].as_f64().unwrap();
        assert!((rt - 0.95).abs() < 1e-3, "recall_target ≈ 0.95, got {}", rt);
        assert_eq!(v["sized"]["m"], 32);
        assert_eq!(v["sized"]["ef_construction"], 256);
        assert_eq!(v["sized"]["ef_search"], 409);
        assert!(
            v["sized"]["rationale"]
                .as_str()
                .unwrap()
                .contains("tier r=0.95"),
            "rationale must surface advisor's tier label"
        );
    }

    #[test]
    fn recluster_response_not_applied_omits_sizing() {
        let resp = RecallReclusterResponse {
            stability: "experimental",
            collection_id: "c1".to_string(),
            applied: false,
            reason: "collection has no recall_target: tag".to_string(),
            rebuilt_vector_count: None,
            sized: None,
        };
        let v = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["applied"], false);
        assert!(v["sized"].is_null());
        assert!(v["rebuilt_vector_count"].is_null());
        assert!(v["reason"].as_str().unwrap().contains("recall_target"));
    }

    #[test]
    fn recall_tune_response_applied_changes_serialize_with_ef_fields() {
        let resp = RecallTuneResponse {
            stability: "experimental",
            collection_id: "c1".to_string(),
            report: RecallDriftHealth::unwired(),
            action: "applied_hot_swap",
            applied_changes: vec![RecallTuneEfChange {
                index_name: Some("primary".to_string()),
                previous_ef_search: 100,
                new_ef_search: 400,
            }],
        };
        let v = serde_json::to_value(&resp).unwrap();
        let changes = v["applied_changes"].as_array().unwrap();
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0]["index_name"], "primary");
        assert_eq!(changes[0]["previous_ef_search"], 100);
        assert_eq!(changes[0]["new_ef_search"], 400);
    }

    #[test]
    fn recall_drift_unwired_omits_advised_params() {
        // unwired() must leave baseline_params + current_params as
        // None so the JSON serializes the absence (no misleading
        // zeros).
        let h = RecallDriftHealth::unwired();
        assert!(h.baseline_params.is_none());
        assert!(h.current_params.is_none());
        let v = serde_json::to_value(&h).unwrap();
        assert!(v["baseline_params"].is_null());
        assert!(v["current_params"].is_null());
    }

    #[test]
    fn recall_drift_wired_serializes_advised_params() {
        let h = RecallDriftHealth {
            wired: true,
            recall_target: Some(0.95),
            baseline_vector_count: Some(100_000),
            current_vector_count: Some(250_000),
            kind: "ef_search_only",
            needs_rebuild: false,
            hot_swap_possible: true,
            summary: "test".to_string(),
            baseline_params: Some(RecallAdvisedParams {
                m: 32,
                ef_construction: 256,
                ef_search: 409,
            }),
            current_params: Some(RecallAdvisedParams {
                m: 32,
                ef_construction: 256,
                ef_search: 622,
            }),
            recommended_action: "call_recall_tune",
            max_ef_search: None,
            clamped_by_max_ef: false,
            projected_recall_at_clamped_ef: None,
            algorithm: "hnsw",
        };
        let v = serde_json::to_value(&h).unwrap();
        assert_eq!(v["baseline_params"]["m"], 32);
        assert_eq!(v["baseline_params"]["ef_construction"], 256);
        assert_eq!(v["baseline_params"]["ef_search"], 409);
        assert_eq!(v["current_params"]["m"], 32);
        assert_eq!(v["current_params"]["ef_search"], 622);
        // The ef_search delta is the actionable signal — operators
        // can compute it inline (409 → 622) without a second call.
    }

    #[test]
    fn recommended_action_pins_kind_to_next_step() {
        // Stable mapping — dashboard / runbook templates can
        // switch on these literals. `clamped=false` here means the
        // pure kind→action mapping.
        assert_eq!(recommended_action_for("none", false), "none");
        assert_eq!(
            recommended_action_for("ef_search_only", false),
            "call_recall_tune"
        );
        assert_eq!(
            recommended_action_for("rebuild_required", false),
            "call_recluster"
        );
        // Anything else (including "unwired") points operators
        // back to the entry-point — set the tag.
        assert_eq!(
            recommended_action_for("unwired", false),
            "set_recall_target_tag"
        );
        assert_eq!(
            recommended_action_for("garbage_value", false),
            "set_recall_target_tag"
        );
    }

    #[test]
    fn recommended_action_clamp_overrides_every_kind() {
        // The latency-budget clamp signal wins over kind — even
        // "none" / "rebuild_required" — because the operator's real
        // choice is "raise the cap or bump m", not the drift state.
        for kind in ["none", "ef_search_only", "rebuild_required", "unwired"] {
            assert_eq!(
                recommended_action_for(kind, true),
                ACTION_RAISE_MAX_EF_OR_BUMP_M,
                "clamped=true must override kind={}",
                kind
            );
        }
    }

    #[test]
    fn recall_drift_unwired_omits_clamp_fields() {
        let h = RecallDriftHealth::unwired();
        assert!(h.max_ef_search.is_none());
        assert!(!h.clamped_by_max_ef);
        assert!(h.projected_recall_at_clamped_ef.is_none());
        let v = serde_json::to_value(&h).unwrap();
        assert!(v["max_ef_search"].is_null());
        assert_eq!(v["clamped_by_max_ef"], false);
        assert!(v["projected_recall_at_clamped_ef"].is_null());
    }

    #[test]
    fn recall_drift_clamped_serializes_max_ef_fields() {
        let h = RecallDriftHealth {
            wired: true,
            recall_target: Some(0.95),
            baseline_vector_count: Some(100_000),
            current_vector_count: Some(100_000),
            kind: "none",
            needs_rebuild: false,
            hot_swap_possible: false,
            summary: "ef clamped to 300".to_string(),
            baseline_params: Some(RecallAdvisedParams {
                m: 32,
                ef_construction: 256,
                ef_search: 300,
            }),
            current_params: Some(RecallAdvisedParams {
                m: 32,
                ef_construction: 256,
                ef_search: 300,
            }),
            recommended_action: ACTION_RAISE_MAX_EF_OR_BUMP_M,
            max_ef_search: Some(300),
            clamped_by_max_ef: true,
            projected_recall_at_clamped_ef: Some(0.93),
            algorithm: "hnsw",
        };
        let v = serde_json::to_value(&h).unwrap();
        assert_eq!(v["max_ef_search"], 300);
        assert_eq!(v["clamped_by_max_ef"], true);
        let projected = v["projected_recall_at_clamped_ef"].as_f64().unwrap();
        assert!((projected - 0.93).abs() < 1e-3);
        assert_eq!(v["recommended_action"], ACTION_RAISE_MAX_EF_OR_BUMP_M);
    }

    #[test]
    fn recall_drift_health_carries_algorithm_field() {
        // The route-health surface admits a stable `algorithm:`
        // literal so dashboards can switch on HNSW vs IVF without
        // parsing `current_params`. P1 always reports "hnsw"
        // because the drift detector covers HNSW only; the field
        // exists so the IVF surface (P2) doesn't break the shape.
        let unwired = RecallDriftHealth::unwired();
        assert_eq!(unwired.algorithm, "hnsw");
        let v = serde_json::to_value(&unwired).unwrap();
        assert_eq!(v["algorithm"], "hnsw");

        // Allowed values are the SupportedAlgorithm labels.
        // Pinned via the ann_advisor module's label() function so
        // both surfaces stay in lockstep.
        use crate::index::axis::management::SupportedAlgorithm;
        assert_eq!(SupportedAlgorithm::Hnsw.label(), "hnsw");
        assert_eq!(SupportedAlgorithm::Ivf.label(), "ivf");
    }

    #[test]
    fn recall_drift_unwired_recommends_setting_tag() {
        // Default unwired() must carry the "set the tag" hint
        // so freshly-created collections without a recall_target:
        // tag get a clear next step on /route-health.
        let h = RecallDriftHealth::unwired();
        assert_eq!(h.recommended_action, "set_recall_target_tag");
        let v = serde_json::to_value(&h).unwrap();
        assert_eq!(v["recommended_action"], "set_recall_target_tag");
    }

    #[test]
    fn route_health_recall_drift_kind_strings_are_stable() {
        // Wire enum-string mapping pinned so dashboards / SIEM
        // filters can rely on the literals.
        let kinds = vec![
            ("unwired", false, false),
            ("none", false, false),
            ("ef_search_only", false, true),
            ("rebuild_required", true, false),
        ];
        for (kind, needs_rebuild, hot_swap) in kinds {
            let h = RecallDriftHealth {
                wired: true,
                recall_target: Some(0.95),
                baseline_vector_count: Some(100_000),
                current_vector_count: Some(250_000),
                kind,
                needs_rebuild,
                hot_swap_possible: hot_swap,
                summary: "test".to_string(),
                baseline_params: None,
                current_params: None,
                recommended_action: recommended_action_for(kind, false),
                max_ef_search: None,
                clamped_by_max_ef: false,
                projected_recall_at_clamped_ef: None,
                algorithm: "hnsw",
            };
            let v = serde_json::to_value(&h).unwrap();
            assert_eq!(v["kind"], kind);
            assert_eq!(v["needs_rebuild"], needs_rebuild);
            assert_eq!(v["hot_swap_possible"], hot_swap);
        }
    }

    #[test]
    fn route_health_object_economy_reports_cached_loaded_status() {
        let h = build_route_health_with_live_state(
            "c".to_string(),
            "sst".to_string(),
            8,
            "cosine".to_string(),
            0,
            0,
            0,
            Some("loaded"),
            RecallProbeLiveState::Unwired,
            None,
        );

        assert!(h.object_economy.live_status_in_app_state);
        assert_eq!(h.object_economy.live_status, "loaded");
        assert!(
            !h.degraded_reasons
                .contains(&DegradedReason::ObjectEconomyLiveStatusNotReachable)
        );
        assert!(
            !h.degraded_reasons
                .contains(&DegradedReason::ObjectEconomyDirectoryDegraded)
        );
    }

    #[test]
    fn route_health_object_economy_reports_cached_degraded_status() {
        let h = build_route_health_with_live_state(
            "c".to_string(),
            "sst".to_string(),
            8,
            "cosine".to_string(),
            0,
            0,
            0,
            Some("missing"),
            RecallProbeLiveState::Unwired,
            None,
        );

        assert!(h.object_economy.live_status_in_app_state);
        assert_eq!(h.object_economy.live_status, "missing");
        assert!(
            !h.degraded_reasons
                .contains(&DegradedReason::ObjectEconomyLiveStatusNotReachable)
        );
        assert!(
            h.degraded_reasons
                .contains(&DegradedReason::ObjectEconomyDirectoryDegraded)
        );
    }

    // ------------------------------------------------------------------
    // RecallProbeLiveState branch coverage — exercises both the Unwired
    // path (default, no AppState slot) and the Wired path (gate reachable
    // from AppState, per-scope `gate_open` resolved). Since TD-075 wired the
    // gate into the AXIS IVF query path, `wired_to_query_path` is now `true`
    // and `RecallProbeNotWired` no longer fires — independent of AppState
    // reachability (`live_state_in_app_state`), which only governs `gate_open`.
    // ------------------------------------------------------------------

    #[test]
    fn route_health_recall_probe_unwired_reports_none_gate_open() {
        let h = build_route_health(
            "c".to_string(),
            "sst".to_string(),
            8,
            "cosine".to_string(),
            0,
            0,
            0,
        );
        assert!(!h.recall_probe.live_state_in_app_state);
        assert_eq!(h.recall_probe.gate_open, None);
        // TD-075: the AXIS query path consults the gate, so this is wired
        // regardless of whether the gate is reachable from this AppState.
        assert!(h.recall_probe.wired_to_query_path);
        assert!(
            !h.degraded_reasons
                .contains(&DegradedReason::RecallProbeNotWired),
            "RecallProbeNotWired must not fire now that the search path \
             consults the gate (TD-075)"
        );
    }

    #[test]
    fn route_health_recall_probe_wired_reports_gate_open_true() {
        let h = build_route_health_with_live_state(
            "c".to_string(),
            "sst".to_string(),
            8,
            "cosine".to_string(),
            0,
            0,
            0,
            None,
            RecallProbeLiveState::Wired { gate_open: true },
            None,
        );
        assert!(h.recall_probe.live_state_in_app_state);
        assert_eq!(h.recall_probe.gate_open, Some(true));
        // TD-075: gate consulted by the AXIS query path.
        assert!(h.recall_probe.wired_to_query_path);
        assert!(
            !h.degraded_reasons
                .contains(&DegradedReason::RecallProbeNotWired),
            "RecallProbeNotWired must not fire once the gate is query-path wired"
        );
    }

    // ------------------------------------------------------------------
    // Pinning state — Phase 6 CollectionPinRegistry surfaced via AppState.
    // The registry is always reachable (defaults to empty in
    // AppState::new); the per-collection pin is Some only when an
    // operator has explicitly pinned the collection.
    // ------------------------------------------------------------------

    #[test]
    fn object_economy_status_label_covers_all_directory_load_status_variants() {
        // Closed-enum mapping canary: if a future DirectoryLoadStatus variant
        // lands without an arm here, the route-health response would silently
        // misrender the status. Pin each variant → label pair.
        use crate::storage::engines::sst::object_economy_directory::DirectoryLoadStatus;
        assert_eq!(
            object_economy_status_label(&DirectoryLoadStatus::Loaded),
            "loaded"
        );
        assert_eq!(
            object_economy_status_label(&DirectoryLoadStatus::Missing),
            "missing"
        );
        assert_eq!(
            object_economy_status_label(&DirectoryLoadStatus::Corrupt("bad header".to_string())),
            "corrupt"
        );
        assert_eq!(
            object_economy_status_label(&DirectoryLoadStatus::Mismatch {
                expected_collection: "products".to_string(),
                found_collection: "orders".to_string(),
            }),
            "mismatch"
        );
    }

    #[test]
    fn route_health_pinning_reports_no_pin_by_default() {
        let h = build_route_health(
            "c".to_string(),
            "sst".to_string(),
            8,
            "cosine".to_string(),
            0,
            0,
            0,
        );
        assert!(h.pinning.registry_in_app_state);
        assert_eq!(h.pinning.pin, None);
    }

    #[test]
    fn route_health_pinning_renders_pin_details_when_set() {
        use crate::storage::collection_pinning::{CollectionPinTarget, PinState};
        let pin = PinState {
            target: CollectionPinTarget::NvmeSsd,
            replicas: 3,
            pinned_at_ns: 1_700_000_000_000_000_000,
        };
        let h = build_route_health_with_live_state(
            "c".to_string(),
            "sst".to_string(),
            8,
            "cosine".to_string(),
            0,
            0,
            0,
            None,
            RecallProbeLiveState::Unwired,
            Some(pin),
        );
        assert!(h.pinning.registry_in_app_state);
        let details = h.pinning.pin.as_ref().expect("pin should be Some");
        // Use the stable lowercase label, not the enum variant name,
        // so the JSON contract doesn't track internal renames.
        assert_eq!(details.target, "nvme_ssd");
        assert_eq!(details.replicas, 3);
        assert_eq!(details.pinned_at_ns, 1_700_000_000_000_000_000);
    }

    #[test]
    fn route_health_recall_probe_wired_reports_gate_open_false() {
        let h = build_route_health_with_live_state(
            "c".to_string(),
            "sst".to_string(),
            8,
            "cosine".to_string(),
            0,
            0,
            0,
            None,
            RecallProbeLiveState::Wired { gate_open: false },
            None,
        );
        assert!(h.recall_probe.live_state_in_app_state);
        assert_eq!(h.recall_probe.gate_open, Some(false));
    }

    // ------------------------------------------------------------------
    // compute_degraded_reasons branch coverage — exercises BOTH sides of
    // every `if !flag` so coverage isn't skewed by the v1 hardcoded
    // constants. These tests treat the function as the source of truth
    // for the flag → reason mapping; build_route_health is then a thin
    // assembler whose v1 set is checked by
    // `route_health_v1_degraded_reasons_are_the_expected_five`.
    // ------------------------------------------------------------------

    fn all_wired_filtered_ann() -> FilteredAnnHealth {
        FilteredAnnHealth {
            id_predicate_supported: true,
            record_aware_predicates: true,
            predicate_pushdown_infrastructure_present: true,
            predicate_pushdown_default_wired: true,
            post_filter_shortfall_disclosure: true,
            td_064_status: "resolved",
        }
    }
    fn all_wired_writes() -> WriteContractHealth {
        WriteContractHealth {
            insert: true,
            upsert: true,
            update: true,
            delete: true,
            conditional_write: true,
            filter_write: true,
            patch: true,
        }
    }
    fn all_wired_freshness() -> FreshnessHealth {
        FreshnessHealth {
            scope: "collection_level",
            collection_level_modes_wired: true,
            search_request_modes: SearchFreshnessModes {
                strong: true,
                bounded_stale: true,
                stale_ok: true,
                bounded_stale_time_bound_check: true,
            },
            notes: "",
        }
    }
    fn all_wired_object_economy() -> ObjectEconomyHealth {
        ObjectEconomyHealth {
            eligible: true,
            directory_format_present: true,
            live_status_in_app_state: true,
            live_status: "loaded",
            route_hint: Some("object_economy"),
            notes: "",
        }
    }
    fn all_wired_recall_probe() -> RecallProbeHealth {
        RecallProbeHealth {
            implementation_present: true,
            wired_to_query_path: true,
            live_state_in_app_state: true,
            gate_open: Some(true),
            notes: "",
        }
    }

    #[test]
    fn compute_degraded_reasons_returns_empty_when_everything_wired() {
        let reasons = compute_degraded_reasons(
            &all_wired_filtered_ann(),
            &all_wired_writes(),
            &all_wired_freshness(),
            &all_wired_object_economy(),
            &all_wired_recall_probe(),
        );
        assert!(
            reasons.is_empty(),
            "fully-wired capability set must produce no degraded reasons; got {reasons:?}"
        );
    }

    #[test]
    fn compute_degraded_reasons_flips_record_aware_predicate_reason() {
        let mut fa = all_wired_filtered_ann();
        fa.record_aware_predicates = false;
        let reasons = compute_degraded_reasons(
            &fa,
            &all_wired_writes(),
            &all_wired_freshness(),
            &all_wired_object_economy(),
            &all_wired_recall_probe(),
        );
        assert_eq!(
            reasons,
            vec![DegradedReason::FilteredAnnRecordPredicateBridgePartial]
        );
    }

    #[test]
    fn compute_degraded_reasons_flips_post_filter_shortfall_reason() {
        let mut fa = all_wired_filtered_ann();
        fa.post_filter_shortfall_disclosure = false;
        let reasons = compute_degraded_reasons(
            &fa,
            &all_wired_writes(),
            &all_wired_freshness(),
            &all_wired_object_economy(),
            &all_wired_recall_probe(),
        );
        assert_eq!(
            reasons,
            vec![DegradedReason::PostFilterShortfallNotDisclosed]
        );
    }

    #[test]
    fn compute_degraded_reasons_flips_object_economy_reason() {
        let mut oe = all_wired_object_economy();
        oe.live_status_in_app_state = false;
        let reasons = compute_degraded_reasons(
            &all_wired_filtered_ann(),
            &all_wired_writes(),
            &all_wired_freshness(),
            &oe,
            &all_wired_recall_probe(),
        );
        assert_eq!(
            reasons,
            vec![DegradedReason::ObjectEconomyLiveStatusNotReachable]
        );
    }

    #[test]
    fn compute_degraded_reasons_flips_object_economy_directory_degraded_reason() {
        let mut oe = all_wired_object_economy();
        oe.live_status = "missing";
        let reasons = compute_degraded_reasons(
            &all_wired_filtered_ann(),
            &all_wired_writes(),
            &all_wired_freshness(),
            &oe,
            &all_wired_recall_probe(),
        );
        assert_eq!(
            reasons,
            vec![DegradedReason::ObjectEconomyDirectoryDegraded]
        );
    }

    #[test]
    fn compute_degraded_reasons_flips_recall_probe_reason() {
        let mut rp = all_wired_recall_probe();
        rp.wired_to_query_path = false;
        let reasons = compute_degraded_reasons(
            &all_wired_filtered_ann(),
            &all_wired_writes(),
            &all_wired_freshness(),
            &all_wired_object_economy(),
            &rp,
        );
        assert_eq!(reasons, vec![DegradedReason::RecallProbeNotWired]);
    }

    #[test]
    fn compute_degraded_reasons_flips_freshness_reason() {
        let mut f = all_wired_freshness();
        f.collection_level_modes_wired = false;
        let reasons = compute_degraded_reasons(
            &all_wired_filtered_ann(),
            &all_wired_writes(),
            &f,
            &all_wired_object_economy(),
            &all_wired_recall_probe(),
        );
        assert_eq!(
            reasons,
            vec![DegradedReason::FreshnessModesNotCollectionLevel]
        );
    }

    #[test]
    fn compute_degraded_reasons_flips_conditional_write_reason() {
        let mut w = all_wired_writes();
        w.conditional_write = false;
        let reasons = compute_degraded_reasons(
            &all_wired_filtered_ann(),
            &w,
            &all_wired_freshness(),
            &all_wired_object_economy(),
            &all_wired_recall_probe(),
        );
        assert_eq!(reasons, vec![DegradedReason::ConditionalWritesUnsupported]);
    }

    #[test]
    fn compute_degraded_reasons_flips_filter_write_reason() {
        let mut w = all_wired_writes();
        w.filter_write = false;
        let reasons = compute_degraded_reasons(
            &all_wired_filtered_ann(),
            &w,
            &all_wired_freshness(),
            &all_wired_object_economy(),
            &all_wired_recall_probe(),
        );
        assert_eq!(reasons, vec![DegradedReason::FilterWritesUnsupported]);
    }

    #[test]
    fn compute_degraded_reasons_preserves_order_when_all_unwired() {
        // Order must match the v1 set exactly. Reordering the if-chain
        // in compute_degraded_reasons would silently reshuffle JSON
        // output for clients that iterate the array.
        let fa = FilteredAnnHealth {
            id_predicate_supported: false,
            record_aware_predicates: false,
            predicate_pushdown_infrastructure_present: false,
            predicate_pushdown_default_wired: false,
            post_filter_shortfall_disclosure: false,
            td_064_status: "open",
        };
        let writes = WriteContractHealth {
            insert: false,
            upsert: false,
            update: false,
            delete: false,
            conditional_write: false,
            filter_write: false,
            patch: false,
        };
        let freshness = FreshnessHealth {
            scope: "",
            collection_level_modes_wired: false,
            search_request_modes: SearchFreshnessModes {
                strong: false,
                bounded_stale: false,
                stale_ok: false,
                bounded_stale_time_bound_check: false,
            },
            notes: "",
        };
        let oe = ObjectEconomyHealth {
            eligible: false,
            directory_format_present: false,
            live_status_in_app_state: false,
            live_status: "not_applicable",
            route_hint: None,
            notes: "",
        };
        let rp = RecallProbeHealth {
            implementation_present: false,
            wired_to_query_path: false,
            live_state_in_app_state: false,
            gate_open: None,
            notes: "",
        };
        let reasons = compute_degraded_reasons(&fa, &writes, &freshness, &oe, &rp);
        assert_eq!(
            reasons,
            vec![
                DegradedReason::FilteredAnnRecordPredicateBridgePartial,
                DegradedReason::PostFilterShortfallNotDisclosed,
                DegradedReason::ObjectEconomyLiveStatusNotReachable,
                DegradedReason::RecallProbeNotWired,
                DegradedReason::FreshnessModesNotCollectionLevel,
                DegradedReason::ConditionalWritesUnsupported,
                DegradedReason::FilterWritesUnsupported,
            ]
        );
    }

    #[test]
    fn each_degraded_reason_variant_serializes_to_unique_screaming_snake() {
        // Per-variant rename check. If a variant is added without a
        // serde rename matching SCREAMING_SNAKE_CASE convention, this
        // fails before the contract leaks to clients.
        let cases = [
            (
                DegradedReason::FilteredAnnRecordPredicateBridgePartial,
                "\"FILTERED_ANN_RECORD_PREDICATE_BRIDGE_PARTIAL\"",
            ),
            (
                DegradedReason::PostFilterShortfallNotDisclosed,
                "\"POST_FILTER_SHORTFALL_NOT_DISCLOSED\"",
            ),
            (
                DegradedReason::ObjectEconomyLiveStatusNotReachable,
                "\"OBJECT_ECONOMY_LIVE_STATUS_NOT_REACHABLE\"",
            ),
            (
                DegradedReason::ObjectEconomyDirectoryDegraded,
                "\"OBJECT_ECONOMY_DIRECTORY_DEGRADED\"",
            ),
            (
                DegradedReason::RecallProbeNotWired,
                "\"RECALL_PROBE_NOT_WIRED\"",
            ),
            (
                DegradedReason::FreshnessModesNotCollectionLevel,
                "\"FRESHNESS_MODES_NOT_COLLECTION_LEVEL\"",
            ),
            (
                DegradedReason::ConditionalWritesUnsupported,
                "\"CONDITIONAL_WRITES_UNSUPPORTED\"",
            ),
            (
                DegradedReason::FilterWritesUnsupported,
                "\"FILTER_WRITES_UNSUPPORTED\"",
            ),
        ];
        for (variant, expected) in cases {
            assert_eq!(
                serde_json::to_string(&variant).unwrap(),
                expected,
                "variant {variant:?} did not serialize to {expected}"
            );
        }
    }

    #[test]
    fn build_route_health_passes_through_engine_and_distance_labels() {
        // Engine/distance strings flow through unchanged — no canonicalization,
        // no casing change. This pins the contract with `get_collection_v2`
        // which uses the same enum→string mapping.
        for (engine, distance) in [
            ("sst", "cosine"),
            ("helix", "euclidean"),
            ("viper", "dot_product"),
            ("auto", "cosine"),
        ] {
            let h = build_route_health(
                "k".to_string(),
                engine.to_string(),
                384,
                distance.to_string(),
                7,
                11,
                13,
            );
            assert_eq!(h.engine, engine);
            assert_eq!(h.distance_metric, distance);
            assert_eq!(h.dimension, 384);
            assert_eq!(h.record_count, 7);
            assert_eq!(h.storage_size_bytes, 11);
            assert_eq!(h.index_size_bytes, 13);
        }
    }

    fn rest_col(data_type: &str) -> RestColumnDefinition {
        RestColumnDefinition {
            name: "c".to_string(),
            data_type: data_type.to_string(),
            nullable: None,
            indexed: None,
            filterable: None,
            max_length: None,
            precision: None,
            scale: None,
            vector_dimension: None,
        }
    }

    #[test]
    fn rest_data_type_vocabulary_maps_to_canonical_proxima_type() {
        use proximadb_data_model::{ProximaType, TimeUnit, VectorElement};
        // Representative vocabulary → canonical type (ADR-024 Step 5).
        assert_eq!(
            parse_rest_data_type(&rest_col("text")).unwrap(),
            ProximaType::String
        );
        assert_eq!(
            parse_rest_data_type(&rest_col("text_large")).unwrap(),
            ProximaType::String
        );
        assert_eq!(
            parse_rest_data_type(&rest_col("integer")).unwrap(),
            ProximaType::Int64
        );
        assert_eq!(
            parse_rest_data_type(&rest_col("float")).unwrap(),
            ProximaType::Float64
        );
        assert_eq!(
            parse_rest_data_type(&rest_col("boolean")).unwrap(),
            ProximaType::Boolean
        );
        assert_eq!(
            parse_rest_data_type(&rest_col("timestamp_tz")).unwrap(),
            ProximaType::TimestampTz(TimeUnit::Nanosecond)
        );
        assert_eq!(
            parse_rest_data_type(&rest_col("uuid")).unwrap(),
            ProximaType::Uuid
        );
        assert_eq!(
            parse_rest_data_type(&rest_col("array_integer")).unwrap(),
            ProximaType::Array(Box::new(ProximaType::Int64))
        );
        assert_eq!(
            parse_rest_data_type(&rest_col("geo_point")).unwrap(),
            ProximaType::Point
        );

        // vector requires a dimension.
        assert!(parse_rest_data_type(&rest_col("vector")).is_err());
        let mut v = rest_col("vector");
        v.vector_dimension = Some(384);
        assert_eq!(
            parse_rest_data_type(&v).unwrap(),
            ProximaType::DenseVector {
                element: VectorElement::Float32,
                dim: 384
            }
        );

        // decimal requires precision + scale, validated inline.
        assert!(parse_rest_data_type(&rest_col("decimal")).is_err());
        let mut d = rest_col("decimal");
        d.precision = Some(18);
        d.scale = Some(4);
        assert_eq!(
            parse_rest_data_type(&d).unwrap(),
            ProximaType::Decimal {
                precision: 18,
                scale: 4
            }
        );

        // unknown type is rejected (the vocabulary is exactly the ProximaType-mappable set).
        assert!(parse_rest_data_type(&rest_col("not_a_type")).is_err());
    }

    /// TD-122 parity: the REST create request parses per-index `is_primary` and
    /// a top-level `quantization` block (gRPC-v2 `V2IndexSpec`/`V2QuantizationConfig`).
    #[test]
    fn create_request_parses_is_primary_and_quantization() {
        let json = r#"{
            "name": "c",
            "dimension": 128,
            "index_configs": [
                {"algorithm": "hnsw", "hnsw_config": {"m": 24, "ef_construction": 150}, "is_primary": true}
            ],
            "quantization": {"enabled": true, "strategy": "aggressive"}
        }"#;
        let req: CreateCollectionV2Request = serde_json::from_str(json).expect("parse");
        let ics = req.index_configs.expect("index_configs");
        assert_eq!(ics.len(), 1);
        assert_eq!(ics[0].is_primary, Some(true));
        assert_eq!(ics[0].hnsw_config.as_ref().and_then(|h| h.m), Some(24));
        let q = req.quantization.expect("quantization");
        assert_eq!(q.enabled, Some(true));
        assert_eq!(q.strategy.as_deref(), Some("aggressive"));
    }

    /// TD-122 parity: the REST get response serializes `index_specs` (with
    /// HNSW params + is_primary) and `quantization` so a read-after-create
    /// echoes what was set.
    #[test]
    fn get_response_serializes_index_specs_and_quantization() {
        let resp = CollectionV2Response {
            collection_id: "c".into(),
            name: "c".into(),
            dimension: 128,
            engine: "sst".into(),
            distance_metric: "cosine".into(),
            proxima_record_enabled: false,
            canonical_embedding_precision: None,
            schema: None,
            stats: CollectionStatsV2 {
                // Some(0), not 0: the field is Option<u64> now so "unknown" and
                // "known to be empty" are distinguishable. This fixture is about
                // index_specs/quantization, so it just needs a known value.
                record_count: Some(0),
                storage_size_bytes: 0,
                indexed_fields: 0,
                text_field_count: 0,
            },
            content_revision: 7,
            content_revision_token: "epoch:7".to_string(),
            index_specs: vec![IndexSpecOutput {
                algorithm: "hnsw".into(),
                hnsw: Some(HnswConfigOutput {
                    m: Some(24),
                    ef_construction: Some(150),
                    ef_search: Some(64),
                }),
                ivf: None,
                is_primary: true,
            }],
            quantization: Some(QuantizationConfigOutput {
                enabled: true,
                strategy: "aggressive".into(),
            }),
            created_at: "now".into(),
            updated_at: None,
        };
        let v: serde_json::Value = serde_json::to_value(&resp).expect("serialize");
        assert_eq!(v["index_specs"][0]["hnsw"]["m"], 24);
        assert_eq!(v["index_specs"][0]["is_primary"], true);
        assert_eq!(v["quantization"]["enabled"], true);
        assert_eq!(v["quantization"]["strategy"], "aggressive");
    }

    // ---- ADR-028 index_policy validation ----

    fn policy_input(mode: &str) -> IndexPolicyInput {
        IndexPolicyInput {
            mode: Some(mode.to_string()),
            rehydrate: None,
            byte_budget: None,
            nprobe: None,
        }
    }

    #[test]
    fn index_policy_defaults_to_auto() {
        let p = build_index_policy(IndexPolicyInput {
            mode: None,
            rehydrate: None,
            byte_budget: None,
            nprobe: None,
        })
        .expect("auto policy is valid");
        assert_eq!(p.mode, "auto");
        assert_eq!(p.rehydrate, "auto");
        assert_eq!(p.byte_budget, 0);
        assert_eq!(p.nprobe, 0);
    }

    #[test]
    fn index_policy_normalizes_and_accepts_valid_modes() {
        for (raw, expect) in [
            ("Exact", "exact"),
            (" HELIX ", "helix"),
            ("ivf", "ivf"),
            ("HNSW", "hnsw"),
        ] {
            let p = build_index_policy(policy_input(raw)).expect("valid mode");
            assert_eq!(p.mode, expect);
        }
    }

    #[test]
    fn index_policy_rejects_unknown_mode() {
        let err = build_index_policy(policy_input("magic")).unwrap_err();
        assert!(matches!(err, ApiError::InvalidArgument(_)));
    }

    #[test]
    fn index_policy_rejects_rehydrate_with_exact() {
        let err = build_index_policy(IndexPolicyInput {
            mode: Some("exact".into()),
            rehydrate: Some("eager".into()),
            byte_budget: None,
            nprobe: None,
        })
        .unwrap_err();
        assert!(matches!(err, ApiError::InvalidArgument(_)));
    }

    #[test]
    fn index_policy_rejects_nprobe_with_exact() {
        let err = build_index_policy(IndexPolicyInput {
            mode: Some("exact".into()),
            rehydrate: None,
            byte_budget: None,
            nprobe: Some(16),
        })
        .unwrap_err();
        assert!(matches!(err, ApiError::InvalidArgument(_)));
    }

    #[test]
    fn index_policy_exact_with_byte_budget_is_allowed() {
        // byte_budget is meaningful for exact (it caps the auto brute-force gate),
        // so it must NOT be rejected.
        let p = build_index_policy(IndexPolicyInput {
            mode: Some("exact".into()),
            rehydrate: None,
            byte_budget: Some(8 * 1024 * 1024),
            nprobe: None,
        })
        .expect("exact + byte_budget is valid");
        assert_eq!(p.mode, "exact");
        assert_eq!(p.byte_budget, 8 * 1024 * 1024);
    }

    #[test]
    fn index_policy_rejects_unknown_rehydrate() {
        let err = build_index_policy(IndexPolicyInput {
            mode: Some("ivf".into()),
            rehydrate: Some("whenever".into()),
            byte_budget: None,
            nprobe: None,
        })
        .unwrap_err();
        assert!(matches!(err, ApiError::InvalidArgument(_)));
    }
}
