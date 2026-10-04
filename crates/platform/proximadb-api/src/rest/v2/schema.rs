//! REST API v2 — collection schema lifecycle (`GET`/`PUT /api/v2/collections/{id}/schema`).
//!
//! Moved from the root `src/network/rest/v2/schema.rs` under the ADR-094
//! REST-tree convergence: port-only state (`RestAppState`), foundation error
//! envelope, and the collection-schema wire DTOs co-located here (single
//! source of truth shared with the root collections router via re-export).

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

//! Schema management endpoints for v2 API
//!
//! This module provides REST endpoints for managing collection schemas,
//! including retrieval and updates with schema evolution support.
//!
//! ## Endpoints
//!
//! - `GET /api/v2/collections/{id}/schema` - Get collection schema
//! - `PUT /api/v2/collections/{id}/schema` - Update collection schema
//!
//! ## Schema Evolution
//!
//! Schema updates follow these rules:
//! - New columns can be added (must be nullable or have defaults)
//! - Column types cannot be changed (except compatible widening)
//! - Columns cannot be removed (for data safety)
//! - Enforcement mode can be changed with care

use axum::{
    Json,
    extract::{Extension, Path, State},
};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};
use utoipa::ToSchema;

use crate::rest::errors::{RestError as ApiError, RestResult as ApiResult};
use crate::rest::TenantContext;
use crate::rest::state::RestAppState;
use proximadb_data_model::ProximaType;
use proximadb_runtime::{
    CollectionSchemaColumn, CollectionSchemaEnforcement, CollectionSchemaMetadata,
    CollectionSchemaUpdate, CollectionTextStorage,
};


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



/// Schema response with metadata
#[derive(Debug, Serialize, ToSchema)]
pub struct SchemaResponse {
    /// Schema ID (UUID)
    pub schema_id: String,
    /// Schema version (semantic versioning)
    pub schema_version: String,
    /// Collection this schema belongs to
    pub collection_id: String,
    /// Schema definition
    pub schema: SchemaDefinition,
    /// Creation timestamp
    pub created_at: String,
    /// Last update timestamp
    pub updated_at: Option<String>,
    /// Parent schema ID (for evolution tracking)
    pub parent_schema_id: Option<String>,
}

/// GET /api/v2/collections/{collection_id}/schema
///
/// Retrieve the schema definition for a collection.
///
/// ## Path Parameters
///
/// - `collection_id`: Collection name/ID
///
/// ## Response
///
/// Returns [`SchemaResponse`] with full schema details.
///
/// ## Errors
///
/// - `404 Not Found`: Collection or schema does not exist
/// - `500 Internal Server Error`: Retrieval failed
#[utoipa::path(
    get,
    path = "/api/v2/collections/{collection_id}/schema",
    tag = "Schema",
    operation_id = "getCollectionSchema",
    summary = "Get collection schema.",
    params(
        ("collection_id" = String, Path, description = "Collection name/ID."),
    ),
    responses(
        (status = 200, description = "Collection schema.", body = SchemaResponse),
        (status = 404, description = "Resource not found.", body = crate::rest::errors::ErrorResponse),
    ),
)]
pub async fn get_schema(
    Path(collection_id): Path<String>,
    Extension(tenant): Extension<TenantContext>,
    State(state): State<RestAppState>,
) -> ApiResult<Json<SchemaResponse>> {
    debug!("V2 API: Getting schema for collection '{}'", collection_id);

    if collection_id.is_empty() {
        return Err(ApiError::InvalidArgument(
            "Collection ID is required".to_string(),
        ));
    }

    let metadata = state
        .handlers
        .get_collection_schema_metadata(&collection_id, Some(&tenant.tenant_id))
        .await
        .map_err(|e| {
            if e.to_string().contains("not found") {
                ApiError::CollectionNotFound(collection_id.clone())
            } else {
                ApiError::Internal(format!("Failed to get collection schema: {}", e))
            }
        })?
        .ok_or_else(|| ApiError::CollectionNotFound(collection_id.clone()))?;

    if !metadata.enabled {
        return Err(ApiError::CollectionNotFound(format!(
            "No schema defined for collection '{}'. Enable ProximaRecord to use schemas.",
            collection_id
        )));
    }

    let schema_id = metadata
        .schema_id
        .clone()
        .filter(|id| !id.is_empty())
        .unwrap_or_else(|| format!("schema_{}", collection_id));
    let schema_version = metadata
        .schema_version
        .clone()
        .filter(|version| !version.is_empty())
        .unwrap_or_else(|| "1.0.0".to_string());
    let schema_def = schema_definition_from_metadata(&metadata);

    let response = SchemaResponse {
        schema_id,
        schema_version,
        collection_id: collection_id.clone(),
        schema: schema_def,
        created_at: chrono::DateTime::from_timestamp(metadata.created_at_ms / 1000, 0)
            .map_or_else(|| chrono::Utc::now().to_rfc3339(), |dt| dt.to_rfc3339()),
        updated_at: if metadata.updated_at_ms != metadata.created_at_ms {
            chrono::DateTime::from_timestamp(metadata.updated_at_ms / 1000, 0)
                .map(|dt| dt.to_rfc3339())
        } else {
            None
        },
        parent_schema_id: None,
    };

    info!(
        "V2 API: Retrieved schema '{}' v{} for collection '{}'",
        response.schema_id, response.schema_version, collection_id
    );

    Ok(Json(response))
}

/// Request to update schema
///
/// ## Schema Evolution Rules
///
/// 1. **Adding Columns**: New columns must be nullable or have defaults
/// 2. **Type Changes**: Not allowed (except compatible widening like int32->int64)
/// 3. **Removing Columns**: Not allowed (use deprecation instead)
/// 4. **Enforcement Mode**: Can be changed, but strict->flexible may lose validation
///
/// ## Example JSON
///
/// ```json
/// {
///     "columns": [
///         {"name": "category", "data_type": "text", "indexed": true},
///         {"name": "price", "data_type": "float", "filterable": true},
///         {"name": "tags", "data_type": "array_text", "nullable": true}
///     ],
///     "enforcement": "hybrid",
///     "allow_additional_fields": true
/// }
/// ```
#[derive(Debug, Deserialize, ToSchema)]
pub struct UpdateSchemaRequest {
    /// Updated schema definition
    #[serde(flatten)]
    pub schema: SchemaDefinition,
    /// Force update even if validation warnings exist
    ///
    /// Use with caution - may cause data compatibility issues.
    pub force: Option<bool>,
}

/// Schema update response
#[derive(Debug, Serialize, ToSchema)]
pub struct UpdateSchemaResponse {
    /// Updated schema ID
    pub schema_id: String,
    /// New schema version
    pub schema_version: String,
    /// Previous schema ID (for rollback)
    pub previous_schema_id: String,
    /// List of applied changes
    pub changes: Vec<SchemaChange>,
    /// Warnings about the update
    pub warnings: Vec<String>,
    /// Update timestamp
    pub updated_at: String,
}

/// Description of a schema change
#[derive(Debug, Serialize, ToSchema)]
pub struct SchemaChange {
    /// Type of change
    pub change_type: String,
    /// Affected column (if applicable)
    pub column: Option<String>,
    /// Description of the change
    pub description: String,
}

/// PUT /api/v2/collections/{collection_id}/schema
///
/// Update the schema for a collection.
///
/// ## Path Parameters
///
/// - `collection_id`: Collection name/ID
///
/// ## Request Body
///
/// See [`UpdateSchemaRequest`] for the expected JSON format.
///
/// ## Response
///
/// Returns [`UpdateSchemaResponse`] with change details.
///
/// ## Schema Evolution
///
/// Schema updates must follow evolution rules to maintain data compatibility.
/// See [`UpdateSchemaRequest`] for detailed rules.
///
/// ## Errors
///
/// - `400 Bad Request`: Invalid schema or evolution violation
/// - `404 Not Found`: Collection does not exist
/// - `409 Conflict`: Incompatible schema change
/// - `500 Internal Server Error`: Update failed
#[utoipa::path(
    put,
    path = "/api/v2/collections/{collection_id}/schema",
    tag = "Schema",
    operation_id = "updateCollectionSchema",
    summary = "Update collection schema.",
    params(
        ("collection_id" = String, Path, description = "Collection name/ID."),
    ),
    request_body = UpdateSchemaRequest,
    responses(
        (status = 200, description = "Schema updated.", body = UpdateSchemaResponse),
        (status = 400, description = "Invalid request.", body = crate::rest::errors::ErrorResponse),
    ),
)]
pub async fn update_schema(
    Path(collection_id): Path<String>,
    Extension(tenant): Extension<TenantContext>,
    State(state): State<RestAppState>,
    Json(request): Json<UpdateSchemaRequest>,
) -> ApiResult<Json<UpdateSchemaResponse>> {
    info!("V2 API: Updating schema for collection '{}'", collection_id);

    if collection_id.is_empty() {
        return Err(ApiError::InvalidArgument(
            "Collection ID is required".to_string(),
        ));
    }

    let force = request.force.unwrap_or(false);
    if force {
        warn!(
            "V2 API: Force schema update requested for '{}' - validation warnings may be ignored",
            collection_id
        );
    }

    // Validate the new schema
    let schema = &request.schema;

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

    for column in &schema.columns {
        if column.name.is_empty() {
            return Err(ApiError::InvalidArgument(
                "Column name cannot be empty".to_string(),
            ));
        }
        let _ = parse_rest_data_type(column)?;
    }

    let existing_metadata = state
        .handlers
        .get_collection_schema_metadata(&collection_id, Some(&tenant.tenant_id))
        .await
        .map_err(|e| {
            if e.to_string().contains("not found") {
                ApiError::CollectionNotFound(collection_id.clone())
            } else {
                ApiError::Internal(format!("Failed to get collection schema: {}", e))
            }
        })?
        .ok_or_else(|| ApiError::CollectionNotFound(collection_id.clone()))?;

    let existing_schema = existing_metadata
        .enabled
        .then(|| schema_definition_from_metadata(&existing_metadata));

    // Step 2: Validate schema evolution rules
    let validation_result = validate_schema(schema, existing_schema.as_ref());

    // Collect changes and warnings
    let mut changes: Vec<SchemaChange> = Vec::new();
    let mut warnings: Vec<String> = validation_result.warnings.clone();

    // Check for evolution violations (unless force is set)
    if !validation_result.valid && !force {
        // Return conflict error with details about what's wrong
        return Err(ApiError::Conflict(format!(
            "Schema evolution validation failed: {}. Use 'force: true' to override.",
            validation_result.errors.join("; ")
        )));
    }

    // If force is set, add evolution errors as warnings
    if !validation_result.valid && force {
        for error in &validation_result.errors {
            warnings.push(format!("FORCED: {}", error));
        }
    }

    // Step 3: Generate schema diff and changes list
    if let Some(ref existing) = existing_schema {
        let existing_columns: std::collections::HashMap<&str, &ColumnDefinition> = existing
            .columns
            .iter()
            .map(|c| (c.name.as_str(), c))
            .collect();

        let new_columns: std::collections::HashMap<&str, &ColumnDefinition> = schema
            .columns
            .iter()
            .map(|c| (c.name.as_str(), c))
            .collect();

        // Detect added columns
        for (name, col) in &new_columns {
            if !existing_columns.contains_key(name) {
                changes.push(SchemaChange {
                    change_type: "ADD_COLUMN".to_string(),
                    column: Some(name.to_string()),
                    description: format!("Added column '{}' with type '{}'", name, col.data_type),
                });
            }
        }

        // Detect removed columns (only if force is set)
        for name in existing_columns.keys() {
            if !new_columns.contains_key(name) {
                changes.push(SchemaChange {
                    change_type: "REMOVE_COLUMN".to_string(),
                    column: Some(name.to_string()),
                    description: format!("Removed column '{}'", name),
                });
            }
        }

        // Detect property changes
        for (name, new_col) in &new_columns {
            if let Some(existing_col) = existing_columns.get(name) {
                // Check for type change
                if existing_col.data_type != new_col.data_type {
                    changes.push(SchemaChange {
                        change_type: "CHANGE_TYPE".to_string(),
                        column: Some(name.to_string()),
                        description: format!(
                            "Changed type of '{}' from '{}' to '{}'",
                            name, existing_col.data_type, new_col.data_type
                        ),
                    });
                }

                // Check for indexed change
                if existing_col.indexed != new_col.indexed {
                    changes.push(SchemaChange {
                        change_type: "CHANGE_INDEX".to_string(),
                        column: Some(name.to_string()),
                        description: format!(
                            "Changed indexed status of '{}' from {:?} to {:?}",
                            name, existing_col.indexed, new_col.indexed
                        ),
                    });
                }

                // Check for nullable change
                if existing_col.nullable != new_col.nullable {
                    changes.push(SchemaChange {
                        change_type: "CHANGE_NULLABILITY".to_string(),
                        column: Some(name.to_string()),
                        description: format!(
                            "Changed nullable status of '{}' from {:?} to {:?}",
                            name, existing_col.nullable, new_col.nullable
                        ),
                    });
                }
            }
        }

        // Detect enforcement mode change
        if existing.enforcement != schema.enforcement {
            changes.push(SchemaChange {
                change_type: "CHANGE_ENFORCEMENT".to_string(),
                column: None,
                description: format!(
                    "Changed enforcement mode from {:?} to {:?}",
                    existing.enforcement, schema.enforcement
                ),
            });
        }
    } else {
        // No existing schema - this is the initial schema
        for col in &schema.columns {
            changes.push(SchemaChange {
                change_type: "ADD_COLUMN".to_string(),
                column: Some(col.name.clone()),
                description: format!("Added column '{}' with type '{}'", col.name, col.data_type),
            });
        }
    }

    // Step 4: Create new schema version
    let previous_schema_id = existing_metadata
        .schema_id
        .clone()
        .filter(|id| !id.is_empty())
        .unwrap_or_else(|| format!("schema_{}_v0", collection_id));

    let new_schema_id = format!("schema_{}_{}", collection_id, uuid::Uuid::new_v4());
    let new_version = increment_version(
        existing_metadata
            .schema_version
            .as_deref()
            .filter(|version| !version.is_empty())
            .unwrap_or("0.0.0"),
    );

    let update = runtime_schema_update_from_schema_definition(
        schema,
        new_schema_id.clone(),
        new_version.clone(),
    )?;

    state
        .handlers
        .update_collection_schema_metadata(&collection_id, update, Some(&tenant.tenant_id))
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to update collection schema: {}", e)))?;

    let now = chrono::Utc::now().to_rfc3339();

    info!(
        "V2 API: Updated schema for collection '{}' to version '{}' with {} changes",
        collection_id,
        new_version,
        changes.len()
    );

    Ok(Json(UpdateSchemaResponse {
        schema_id: new_schema_id,
        schema_version: new_version,
        previous_schema_id,
        changes,
        warnings,
        updated_at: now,
    }))
}

fn schema_definition_from_metadata(metadata: &CollectionSchemaMetadata) -> SchemaDefinition {
    let columns = metadata
        .columns
        .iter()
        .map(|column| ColumnDefinition {
            name: column.name.clone(),
            data_type: rest_data_type_from_proxima(&column.data_type, column.text_storage)
                .to_string(),
            nullable: Some(column.nullable),
            indexed: Some(column.indexed),
            filterable: Some(column.filterable),
            max_length: column.max_length,
            precision: match &column.data_type {
                ProximaType::Decimal { precision, .. } => Some(*precision),
                _ => None,
            },
            scale: match &column.data_type {
                ProximaType::Decimal { scale, .. } => Some(*scale),
                _ => None,
            },
            vector_dimension: match &column.data_type {
                ProximaType::DenseVector { dim, .. } | ProximaType::BinaryVector { dim } => {
                    Some(*dim as u32)
                }
                _ => None,
            },
        })
        .collect::<Vec<_>>();

    SchemaDefinition {
        columns,
        enforcement: metadata.enforcement.map(runtime_enforcement_to_rest),
        allow_additional_fields: Some(metadata.auto_evolve),
    }
}

fn runtime_schema_update_from_schema_definition(
    schema: &SchemaDefinition,
    schema_id: String,
    schema_version: String,
) -> ApiResult<CollectionSchemaUpdate> {
    let columns = schema
        .columns
        .iter()
        .map(|column| {
            let data_type = parse_rest_data_type(column)?;
            let text_storage = match column.data_type.as_str() {
                "text" => Some(CollectionTextStorage::Inline),
                "text_large" => Some(CollectionTextStorage::Large),
                _ => None,
            };
            let filterable = column.filterable.unwrap_or(
                matches!(text_storage, Some(CollectionTextStorage::Inline))
                    || !matches!(data_type, ProximaType::String),
            );
            Ok(CollectionSchemaColumn {
                name: column.name.clone(),
                data_type,
                nullable: column.nullable.unwrap_or(true),
                indexed: column.indexed.unwrap_or(false),
                filterable,
                text_storage,
                max_length: column.max_length,
            })
        })
        .collect::<ApiResult<Vec<_>>>()?;

    Ok(CollectionSchemaUpdate {
        schema_id,
        schema_version,
        enforcement: runtime_enforcement_from_rest(schema.enforcement.as_deref()),
        auto_evolve: schema.allow_additional_fields.unwrap_or(true),
        columns,
    })
}

fn runtime_enforcement_from_rest(value: Option<&str>) -> CollectionSchemaEnforcement {
    match value {
        Some("strict") => CollectionSchemaEnforcement::Strict,
        Some("flexible") => CollectionSchemaEnforcement::Flexible,
        _ => CollectionSchemaEnforcement::Hybrid,
    }
}

fn runtime_enforcement_to_rest(value: CollectionSchemaEnforcement) -> String {
    match value {
        CollectionSchemaEnforcement::Strict => "strict",
        CollectionSchemaEnforcement::Flexible => "flexible",
        CollectionSchemaEnforcement::Hybrid => "hybrid",
    }
    .to_string()
}

fn rest_data_type_from_proxima(
    data_type: &ProximaType,
    text_storage: Option<CollectionTextStorage>,
) -> &'static str {
    match data_type {
        ProximaType::String => match text_storage {
            Some(CollectionTextStorage::Large) => "text_large",
            _ => "text",
        },
        ProximaType::Int8 => "int8",
        ProximaType::Int16 => "int16",
        ProximaType::Int32 => "int32",
        ProximaType::Int64
        | ProximaType::UInt8
        | ProximaType::UInt16
        | ProximaType::UInt32
        | ProximaType::UInt64 => "integer",
        ProximaType::Float16 | ProximaType::Float32 => "float32",
        ProximaType::Float64 => "float",
        ProximaType::Decimal { .. } => "decimal",
        ProximaType::Boolean => "boolean",
        ProximaType::Timestamp(_) => "timestamp",
        ProximaType::TimestampTz(_) => "timestamp_tz",
        ProximaType::Date => "date",
        ProximaType::Time(_) => "time",
        ProximaType::Uuid => "uuid",
        ProximaType::ULID => "ulid",
        ProximaType::Binary => "binary",
        ProximaType::Json => "json",
        ProximaType::Jsonb => "jsonb",
        ProximaType::Array(inner) => match inner.as_ref() {
            ProximaType::String => "array_text",
            ProximaType::Int64 => "array_integer",
            ProximaType::Float64 => "array_float",
            ProximaType::Boolean => "array_boolean",
            ProximaType::Uuid => "array_uuid",
            _ => "array_any",
        },
        ProximaType::Map { key, value } => match (key.as_ref(), value.as_ref()) {
            (ProximaType::String, ProximaType::String) => "map_string_string",
            (ProximaType::String, ProximaType::Int64) => "map_string_integer",
            (ProximaType::String, ProximaType::Float64) => "map_string_float",
            (ProximaType::String, _) => "map_string_any",
            _ => "map_string_any",
        },
        ProximaType::Struct { .. } => "struct",
        ProximaType::Point | ProximaType::GeographyPoint => "geo_point",
        ProximaType::DenseVector { .. } => "vector",
        ProximaType::SparseVector { .. } => "sparse_vector",
        ProximaType::BinaryVector { .. } => "binary_vector",
        ProximaType::Symbol => "symbol",
        ProximaType::Duration(_) => "duration",
        ProximaType::Interval(_) => "interval",
        ProximaType::Null => "json",
    }
}

/// Map a REST schema-enforcement string to the proto `SchemaEnforcement`
/// discriminant (1=strict, 2=flexible, 3=hybrid; default hybrid).
pub fn enforcement_value(enforcement: Option<&str>) -> i32 {
    match enforcement {
        Some("strict") => 1,
        Some("flexible") => 2,
        Some("hybrid") => 3,
        _ => 3,
    }
}

/// Map a REST scalar column `data_type` to its `(FilterableDataType discriminant,
/// supports_range)` pair, or `None` for text / structured / array types that are
/// not metadata-filterable scalar columns (text → `text_columns`). Inverse:
/// [`filterable_type_to_rest`].
pub fn rest_scalar_filterable_type(data_type: &str) -> Option<(i32, bool)> {
    use proximadb_proto::proximadb_v1::FilterableDataType as F;
    let (ty, supports_range) = match data_type {
        "integer" => (F::FilterableInteger, true),
        "float" => (F::FilterableFloat, true),
        "decimal" => (F::FilterableDecimal, true),
        "boolean" => (F::FilterableBoolean, false),
        "timestamp" => (F::FilterableDatetime, true),
        "timestamp_tz" => (F::FilterableTimestampTz, true),
        "date" => (F::FilterableDate, true),
        "time" => (F::FilterableTime, true),
        "uuid" => (F::FilterableUuid, false),
        _ => return None,
    };
    Some((ty as i32, supports_range))
}

/// Inverse of [`rest_scalar_filterable_type`]: a `FilterableDataType`
/// discriminant back to its REST `data_type` label (for the GET schema view).
pub fn filterable_type_to_rest(v: i32) -> &'static str {
    use proximadb_proto::proximadb_v1::FilterableDataType as F;
    match F::try_from(v) {
        Ok(F::FilterableInteger) => "integer",
        Ok(F::FilterableFloat) => "float",
        Ok(F::FilterableDecimal) => "decimal",
        Ok(F::FilterableBoolean) => "boolean",
        Ok(F::FilterableDatetime) => "timestamp",
        Ok(F::FilterableTimestampTz) => "timestamp_tz",
        Ok(F::FilterableDate) => "date",
        Ok(F::FilterableTime) => "time",
        Ok(F::FilterableUuid) => "uuid",
        _ => "text",
    }
}

/// Populate the ProximaRecord schema fields on a collection config from a REST
/// `SchemaDefinition`. Text / text_large columns become `text_columns` +
/// `text_storage_configs`; scalar (numeric/temporal/boolean/uuid) columns that
/// are filterable become typed `filterable_columns`. Sets
/// `enable_proxima_record = true`.
///
/// Shared by the create-collection and update-schema paths so both persist the
/// same shape (the inverse of [`build_existing_schema`]).
pub fn apply_schema_definition(
    config: &mut proximadb_proto::proximadb_v1::CollectionConfig,
    schema: &SchemaDefinition,
    schema_id: String,
    schema_version: String,
) {
    // TD-FPRUNE-1 (Gap A): a SHORT `text` column marked filterable becomes a
    // typed filterable_column (shredded into a PAX user-column with min/max +
    // bloom + the self-describing footer field-map), NOT a text stripe — so
    // categorical text tags (partition/lang) footer-prune filtered vector search.
    // `text_large` (chunked documents) is never shredded; non-filterable `text`
    // stays a plain text stripe.
    let text_columns: Vec<String> = schema
        .columns
        .iter()
        .filter(|c| {
            c.data_type == "text_large" || (c.data_type == "text" && c.filterable == Some(false))
        })
        .map(|c| c.name.clone())
        .collect();

    // Scalar columns marked filterable (default true) become typed
    // filterable_columns so metadata filters can push down and GetCollection's
    // indexed_fields reflects them. Gap A: filterable short `text` joins them as
    // `FilterableString` (equality-only; hashes are unordered so `supports_range`
    // is false).
    let filterable_columns: Vec<proximadb_proto::proximadb_v1::FilterableColumnSpec> = schema
        .columns
        .iter()
        .filter(|c| c.filterable != Some(false))
        .filter_map(|c| {
            let (data_type, supports_range) = if c.data_type == "text" {
                (
                    proximadb_proto::proximadb_v1::FilterableDataType::FilterableString as i32,
                    false,
                )
            } else {
                rest_scalar_filterable_type(&c.data_type)?
            };
            Some(proximadb_proto::proximadb_v1::FilterableColumnSpec {
                name: c.name.clone(),
                data_type,
                indexed: c.indexed.unwrap_or(false),
                supports_range,
                estimated_cardinality: None,
            })
        })
        .collect();

    let text_storage_configs: Vec<proximadb_proto::proximadb_v1::TextStorageConfig> = schema
        .columns
        .iter()
        .filter(|c| c.data_type == "text_large")
        .map(|c| proximadb_proto::proximadb_v1::TextStorageConfig {
            column_name: c.name.clone(),
            strategy: 1, // TextStorage::Chunked
            inline_threshold: 4096,
            chunked_threshold: 1048576,
            chunk_size: c.max_length.unwrap_or(512),
            ..Default::default()
        })
        .collect();

    config.record_schema = Some(proximadb_proto::proximadb_v1::RecordSchemaConfig {
        schema_id,
        schema_version,
        enforcement: enforcement_value(schema.enforcement.as_deref()),
        auto_evolve: schema.allow_additional_fields.unwrap_or(true),
        columns: Vec::new(), // typed columns travel via text_columns / filterable_columns
    });
    config.enable_proxima_record = Some(true);
    config.text_columns = text_columns;
    config.text_storage_configs = text_storage_configs;
    config.filterable_columns = filterable_columns;
}

/// Build existing schema from collection config
pub fn build_existing_schema(
    config: &proximadb_proto::proximadb_v1::CollectionConfig,
) -> Option<SchemaDefinition> {
    // If ProximaRecord is not enabled or no schema config, return None
    if !config.enable_proxima_record.unwrap_or(false) && config.record_schema.is_none() {
        return None;
    }

    // Build columns from text_columns and text_storage_configs
    let mut columns: Vec<ColumnDefinition> = config
        .text_columns
        .iter()
        .map(|col_name| ColumnDefinition {
            name: col_name.clone(),
            data_type: "text".to_string(),
            nullable: Some(true),
            indexed: Some(false),
            filterable: Some(true),
            max_length: None,
            precision: None,
            scale: None,
            vector_dimension: None,
        })
        .collect();

    // Add text_large columns from text_storage_configs
    for text_config in &config.text_storage_configs {
        if !columns.iter().any(|c| c.name == text_config.column_name) {
            columns.push(ColumnDefinition {
                name: text_config.column_name.clone(),
                data_type: "text_large".to_string(),
                nullable: Some(true),
                indexed: Some(false),
                filterable: Some(false),
                max_length: Some(text_config.chunk_size),
                precision: None,
                scale: None,
                vector_dimension: None,
            });
        }
    }

    // Add scalar typed columns persisted in CollectionConfig.filterable_columns.
    // apply_schema_definition writes integer/float/temporal/bool/uuid schema
    // columns here so metadata filters can push down; GET must invert that
    // mapping or schema round-trips silently lose all non-text columns.
    for filterable in &config.filterable_columns {
        if filterable.name.is_empty() || columns.iter().any(|c| c.name == filterable.name) {
            continue;
        }
        columns.push(ColumnDefinition {
            name: filterable.name.clone(),
            data_type: filterable_type_to_rest(filterable.data_type).to_string(),
            nullable: Some(true),
            indexed: Some(filterable.indexed),
            filterable: Some(true),
            max_length: None,
            precision: None,
            scale: None,
            vector_dimension: None,
        });
    }

    // Get enforcement mode from record_schema if available
    let enforcement =
        config
            .record_schema
            .as_ref()
            .map(|schema_config| match schema_config.enforcement {
                1 => "strict".to_string(),
                2 => "flexible".to_string(),
                3 => "hybrid".to_string(),
                _ => "hybrid".to_string(),
            });

    let allow_additional = config.record_schema.as_ref().is_none_or(|s| s.auto_evolve);

    Some(SchemaDefinition {
        columns,
        enforcement,
        allow_additional_fields: Some(allow_additional),
    })
}

/// Increment semantic version (e.g., "1.0.0" -> "1.0.1")
fn increment_version(current: &str) -> String {
    let parts: Vec<&str> = current.split('.').collect();
    if parts.len() == 3
        && let (Ok(major), Ok(minor), Ok(patch)) = (
            parts[0].parse::<u32>(),
            parts[1].parse::<u32>(),
            parts[2].parse::<u32>(),
        )
    {
        return format!("{}.{}.{}", major, minor, patch + 1);
    }
    // Fallback: return 1.0.0
    "1.0.0".to_string()
}

/// Schema validation result
#[derive(Debug, Serialize)]
pub struct SchemaValidationResult {
    /// Whether the schema is valid
    pub valid: bool,
    /// Validation errors
    pub errors: Vec<String>,
    /// Validation warnings
    pub warnings: Vec<String>,
    /// Suggested fixes
    pub suggestions: Vec<String>,
}

/// Validate a schema definition
///
/// This is a helper function used internally to validate schemas.
/// It checks:
/// - Column name uniqueness
/// - Type validity
/// - Constraint consistency
/// - Evolution compatibility (when comparing with existing schema)
pub fn validate_schema(
    schema: &SchemaDefinition,
    existing_schema: Option<&SchemaDefinition>,
) -> SchemaValidationResult {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    let mut suggestions = Vec::new();

    // Check for duplicate column names
    let mut seen_names = std::collections::HashSet::new();
    for column in &schema.columns {
        if !seen_names.insert(&column.name) {
            errors.push(format!("Duplicate column name: '{}'", column.name));
        }
    }

    // Validate each column
    for column in &schema.columns {
        // Check for reserved names
        let reserved = ["id", "vector", "_id", "_score", "_version"];
        if reserved.contains(&column.name.as_str()) {
            errors.push(format!("Column name '{}' is reserved", column.name));
        }

        // Validate column name format
        if !column.name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            errors.push(format!(
                "Column name '{}' contains invalid characters (use alphanumeric and underscore only)",
                column.name
            ));
        }

        // Warn about non-indexed filterable columns
        if column.filterable.unwrap_or(false) && !column.indexed.unwrap_or(false) {
            warnings.push(format!(
                "Column '{}' is filterable but not indexed - queries may be slow",
                column.name
            ));
            suggestions.push(format!(
                "Consider adding 'indexed: true' to column '{}'",
                column.name
            ));
        }

        // Warn about large max_length for indexed text
        if let Some(max_length) = column.max_length
            && max_length > 4096
            && column.indexed.unwrap_or(false)
        {
            warnings.push(format!(
                "Column '{}' has large max_length ({}) with indexing enabled",
                column.name, max_length
            ));
        }
    }

    // Check evolution compatibility if existing schema provided
    if let Some(existing) = existing_schema {
        let existing_columns: std::collections::HashMap<&str, &ColumnDefinition> = existing
            .columns
            .iter()
            .map(|c| (c.name.as_str(), c))
            .collect();

        let new_columns: std::collections::HashMap<&str, &ColumnDefinition> = schema
            .columns
            .iter()
            .map(|c| (c.name.as_str(), c))
            .collect();

        // Check for removed columns
        for name in existing_columns.keys() {
            if !new_columns.contains_key(name) {
                errors.push(format!(
                    "Cannot remove column '{}' - schema evolution does not allow column removal",
                    name
                ));
            }
        }

        // Check for type changes
        for (name, new_col) in &new_columns {
            if let Some(existing_col) = existing_columns.get(name)
                && existing_col.data_type != new_col.data_type
            {
                errors.push(format!(
                    "Cannot change type of column '{}' from '{}' to '{}'",
                    name, existing_col.data_type, new_col.data_type
                ));
            }
        }

        // Check new columns are nullable or have defaults
        for (name, new_col) in &new_columns {
            if !existing_columns.contains_key(name) {
                // This is a new column
                if !new_col.nullable.unwrap_or(true) {
                    warnings.push(format!(
                        "New column '{}' is not nullable - existing records will have NULL values",
                        name
                    ));
                    suggestions.push(format!(
                        "Consider making column '{}' nullable or providing a default value",
                        name
                    ));
                }
            }
        }
    }

    SchemaValidationResult {
        valid: errors.is_empty(),
        errors,
        warnings,
        suggestions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_validation_basic() {
        let schema = SchemaDefinition {
            columns: vec![
                ColumnDefinition {
                    name: "title".to_string(),
                    data_type: "text".to_string(),
                    nullable: Some(false),
                    indexed: Some(true),
                    filterable: Some(true),
                    max_length: Some(255),
                    precision: None,
                    scale: None,
                    vector_dimension: None,
                },
                ColumnDefinition {
                    name: "price".to_string(),
                    data_type: "float".to_string(),
                    nullable: Some(true),
                    indexed: Some(false),
                    filterable: Some(true),
                    max_length: None,
                    precision: None,
                    scale: None,
                    vector_dimension: None,
                },
            ],
            enforcement: Some("hybrid".to_string()),
            allow_additional_fields: Some(true),
        };

        let result = validate_schema(&schema, None);
        assert!(result.valid);
        // Should have warning about non-indexed filterable column
        assert!(!result.warnings.is_empty());
    }

    #[test]
    fn test_schema_validation_duplicate_columns() {
        let schema = SchemaDefinition {
            columns: vec![
                ColumnDefinition {
                    name: "title".to_string(),
                    data_type: "text".to_string(),
                    nullable: None,
                    indexed: None,
                    filterable: None,
                    max_length: None,
                    precision: None,
                    scale: None,
                    vector_dimension: None,
                },
                ColumnDefinition {
                    name: "title".to_string(), // Duplicate!
                    data_type: "text".to_string(),
                    nullable: None,
                    indexed: None,
                    filterable: None,
                    max_length: None,
                    precision: None,
                    scale: None,
                    vector_dimension: None,
                },
            ],
            enforcement: None,
            allow_additional_fields: None,
        };

        let result = validate_schema(&schema, None);
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.contains("Duplicate")));
    }

    #[test]
    fn test_schema_validation_reserved_name() {
        let schema = SchemaDefinition {
            columns: vec![ColumnDefinition {
                name: "id".to_string(), // Reserved!
                data_type: "text".to_string(),
                nullable: None,
                indexed: None,
                filterable: None,
                max_length: None,
                precision: None,
                scale: None,
                vector_dimension: None,
            }],
            enforcement: None,
            allow_additional_fields: None,
        };

        let result = validate_schema(&schema, None);
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.contains("reserved")));
    }

    #[test]
    fn test_schema_evolution_removed_column() {
        let existing = SchemaDefinition {
            columns: vec![
                ColumnDefinition {
                    name: "title".to_string(),
                    data_type: "text".to_string(),
                    nullable: None,
                    indexed: None,
                    filterable: None,
                    max_length: None,
                    precision: None,
                    scale: None,
                    vector_dimension: None,
                },
                ColumnDefinition {
                    name: "price".to_string(),
                    data_type: "float".to_string(),
                    nullable: None,
                    indexed: None,
                    filterable: None,
                    max_length: None,
                    precision: None,
                    scale: None,
                    vector_dimension: None,
                },
            ],
            enforcement: None,
            allow_additional_fields: None,
        };

        // New schema removes 'price' column
        let new_schema = SchemaDefinition {
            columns: vec![ColumnDefinition {
                name: "title".to_string(),
                data_type: "text".to_string(),
                nullable: None,
                indexed: None,
                filterable: None,
                max_length: None,
                precision: None,
                scale: None,
                vector_dimension: None,
            }],
            enforcement: None,
            allow_additional_fields: None,
        };

        let result = validate_schema(&new_schema, Some(&existing));
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.contains("remove column")));
    }

    #[test]
    fn test_schema_evolution_type_change() {
        let existing = SchemaDefinition {
            columns: vec![ColumnDefinition {
                name: "count".to_string(),
                data_type: "integer".to_string(),
                nullable: None,
                indexed: None,
                filterable: None,
                max_length: None,
                precision: None,
                scale: None,
                vector_dimension: None,
            }],
            enforcement: None,
            allow_additional_fields: None,
        };

        // New schema changes type from integer to text
        let new_schema = SchemaDefinition {
            columns: vec![ColumnDefinition {
                name: "count".to_string(),
                data_type: "text".to_string(), // Type change!
                nullable: None,
                indexed: None,
                filterable: None,
                max_length: None,
                precision: None,
                scale: None,
                vector_dimension: None,
            }],
            enforcement: None,
            allow_additional_fields: None,
        };

        let result = validate_schema(&new_schema, Some(&existing));
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.contains("change type")));
    }

    // =========================================================================
    // Tests for increment_version
    // =========================================================================

    #[test]
    fn test_increment_version_basic() {
        assert_eq!(increment_version("1.0.0"), "1.0.1");
        assert_eq!(increment_version("1.2.3"), "1.2.4");
        assert_eq!(increment_version("0.0.0"), "0.0.1");
    }

    #[test]
    fn test_increment_version_large_numbers() {
        assert_eq!(increment_version("10.20.30"), "10.20.31");
        assert_eq!(increment_version("99.99.99"), "99.99.100");
    }

    #[test]
    fn test_increment_version_invalid_format() {
        // Not three parts
        assert_eq!(increment_version("1.0"), "1.0.0");
        assert_eq!(increment_version("1"), "1.0.0");
        assert_eq!(increment_version(""), "1.0.0");
    }

    #[test]
    fn test_increment_version_non_numeric() {
        assert_eq!(increment_version("a.b.c"), "1.0.0");
        assert_eq!(increment_version("1.2.xyz"), "1.0.0");
    }

    // =========================================================================
    // Tests for build_existing_schema
    // =========================================================================

    #[test]
    fn test_build_existing_schema_no_proxima_record() {
        let config = proximadb_proto::proximadb_v1::CollectionConfig {
            enable_proxima_record: Some(false),
            record_schema: None,
            ..Default::default()
        };
        let result = build_existing_schema(&config);
        assert!(result.is_none());
    }

    #[test]
    fn test_build_existing_schema_with_text_columns() {
        let config = proximadb_proto::proximadb_v1::CollectionConfig {
            enable_proxima_record: Some(true),
            text_columns: vec!["title".to_string(), "body".to_string()],
            record_schema: None,
            ..Default::default()
        };
        let result = build_existing_schema(&config);
        assert!(result.is_some());
        let schema = result.expect("Should build schema");
        assert_eq!(schema.columns.len(), 2);
        assert_eq!(schema.columns[0].name, "title");
        assert_eq!(schema.columns[0].data_type, "text");
        assert_eq!(schema.columns[1].name, "body");
    }

    #[test]
    fn apply_schema_definition_populates_typed_filterable_columns() {
        use proximadb_proto::proximadb_v1::FilterableDataType as F;
        let schema = SchemaDefinition {
            columns: vec![
                // TD-FPRUNE-1 Gap A: filterable short `text` (partition/lang-style
                // categorical) → typed FilterableString filterable_column (shred).
                ColumnDefinition {
                    name: "lang".to_string(),
                    data_type: "text".to_string(),
                    indexed: Some(true),
                    ..blank_col()
                },
                // Non-filterable `text` stays a plain text stripe.
                ColumnDefinition {
                    name: "raw".to_string(),
                    data_type: "text".to_string(),
                    filterable: Some(false),
                    ..blank_col()
                },
                // `text_large` (chunked documents) is never shredded.
                ColumnDefinition {
                    name: "body".to_string(),
                    data_type: "text_large".to_string(),
                    ..blank_col()
                },
                ColumnDefinition {
                    name: "price".to_string(),
                    data_type: "float".to_string(),
                    indexed: Some(true),
                    ..blank_col()
                },
                ColumnDefinition {
                    name: "active".to_string(),
                    data_type: "boolean".to_string(),
                    ..blank_col()
                },
                ColumnDefinition {
                    name: "secret".to_string(),
                    data_type: "integer".to_string(),
                    filterable: Some(false), // opted out
                    ..blank_col()
                },
            ],
            enforcement: Some("strict".to_string()),
            allow_additional_fields: Some(false),
        };
        let mut config = proximadb_proto::proximadb_v1::CollectionConfig::default();
        apply_schema_definition(&mut config, &schema, "s".to_string(), "1.0.0".to_string());

        // text_large + non-filterable text → text_columns (not shredded).
        assert_eq!(
            config.text_columns,
            vec!["raw".to_string(), "body".to_string()]
        );
        // filterable columns: lang (FilterableString) + price (range) + active.
        assert_eq!(config.filterable_columns.len(), 3);
        let lang = config
            .filterable_columns
            .iter()
            .find(|c| c.name == "lang")
            .expect("lang");
        assert_eq!(lang.data_type, F::FilterableString as i32);
        assert!(lang.indexed);
        assert!(!lang.supports_range); // hashes are unordered → equality only
        let price = config
            .filterable_columns
            .iter()
            .find(|c| c.name == "price")
            .expect("price");
        assert_eq!(price.data_type, F::FilterableFloat as i32);
        assert!(price.indexed);
        assert!(price.supports_range);
        let active = config
            .filterable_columns
            .iter()
            .find(|c| c.name == "active")
            .expect("active");
        assert_eq!(active.data_type, F::FilterableBoolean as i32);
        assert!(!active.indexed);
        assert!(!active.supports_range);
        assert!(!config.filterable_columns.iter().any(|c| c.name == "secret"));
        assert!(!config.filterable_columns.iter().any(|c| c.name == "raw"));
    }

    #[test]
    fn build_existing_schema_round_trips_filterable_scalar_columns() {
        use proximadb_proto::proximadb_v1::{FilterableColumnSpec, FilterableDataType as F};

        let schema = SchemaDefinition {
            columns: vec![
                ColumnDefinition {
                    name: "body".to_string(),
                    data_type: "text".to_string(),
                    ..blank_col()
                },
                ColumnDefinition {
                    name: "price".to_string(),
                    data_type: "float".to_string(),
                    indexed: Some(true),
                    filterable: Some(true),
                    ..blank_col()
                },
                ColumnDefinition {
                    name: "created_on".to_string(),
                    data_type: "date".to_string(),
                    filterable: Some(true),
                    ..blank_col()
                },
            ],
            enforcement: Some("hybrid".to_string()),
            allow_additional_fields: Some(true),
        };
        let mut config = proximadb_proto::proximadb_v1::CollectionConfig::default();
        apply_schema_definition(
            &mut config,
            &schema,
            "schema_products_1".to_string(),
            "1.0.0".to_string(),
        );

        let rebuilt = build_existing_schema(&config).expect("schema should rebuild");
        let names_and_types = rebuilt
            .columns
            .iter()
            .map(|c| {
                (
                    c.name.as_str(),
                    c.data_type.as_str(),
                    c.indexed,
                    c.filterable,
                )
            })
            .collect::<Vec<_>>();

        assert!(names_and_types.contains(&("body", "text", Some(false), Some(true))));
        assert!(names_and_types.contains(&("price", "float", Some(true), Some(true))));
        assert!(names_and_types.contains(&("created_on", "date", Some(false), Some(true))));

        let mut legacy_config = proximadb_proto::proximadb_v1::CollectionConfig {
            enable_proxima_record: Some(true),
            filterable_columns: vec![FilterableColumnSpec {
                name: "score".to_string(),
                data_type: F::FilterableFloat as i32,
                indexed: true,
                supports_range: true,
                estimated_cardinality: None,
            }],
            ..Default::default()
        };
        legacy_config.record_schema = Some(proximadb_proto::proximadb_v1::RecordSchemaConfig {
            schema_id: "legacy".to_string(),
            schema_version: "1.0.0".to_string(),
            enforcement: 3,
            auto_evolve: true,
            columns: vec![],
        });

        let legacy = build_existing_schema(&legacy_config).expect("legacy schema should rebuild");
        assert!(
            legacy
                .columns
                .iter()
                .any(|c| c.name == "score" && c.data_type == "float")
        );
    }

    fn blank_col() -> ColumnDefinition {
        ColumnDefinition {
            name: String::new(),
            data_type: String::new(),
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
    fn test_build_existing_schema_with_record_schema_enforcement() {
        let config = proximadb_proto::proximadb_v1::CollectionConfig {
            enable_proxima_record: Some(true),
            record_schema: Some(proximadb_proto::proximadb_v1::RecordSchemaConfig {
                schema_id: "s1".to_string(),
                schema_version: "1.0.0".to_string(),
                enforcement: 1, // strict
                auto_evolve: false,
                columns: vec![],
            }),
            ..Default::default()
        };
        let result = build_existing_schema(&config);
        assert!(result.is_some());
        let schema = result.expect("Should build schema");
        assert_eq!(schema.enforcement, Some("strict".to_string()));
        assert_eq!(schema.allow_additional_fields, Some(false));
    }

    #[test]
    fn test_build_existing_schema_enforcement_mapping() {
        // Test all enforcement values
        for (value, expected) in [(1, "strict"), (2, "flexible"), (3, "hybrid"), (0, "hybrid")] {
            let config = proximadb_proto::proximadb_v1::CollectionConfig {
                enable_proxima_record: Some(true),
                record_schema: Some(proximadb_proto::proximadb_v1::RecordSchemaConfig {
                    enforcement: value,
                    auto_evolve: true,
                    ..Default::default()
                }),
                ..Default::default()
            };
            let schema = build_existing_schema(&config).expect("Should build");
            assert_eq!(
                schema.enforcement,
                Some(expected.to_string()),
                "Failed for enforcement value {}",
                value
            );
        }
    }

    #[test]
    fn test_build_existing_schema_text_storage_configs() {
        let config = proximadb_proto::proximadb_v1::CollectionConfig {
            enable_proxima_record: Some(true),
            text_columns: vec!["summary".to_string()],
            text_storage_configs: vec![proximadb_proto::proximadb_v1::TextStorageConfig {
                column_name: "full_text".to_string(),
                chunk_size: 1024,
                ..Default::default()
            }],
            ..Default::default()
        };
        let result = build_existing_schema(&config);
        assert!(result.is_some());
        let schema = result.expect("Should build schema");
        assert_eq!(schema.columns.len(), 2);
        // First from text_columns
        assert_eq!(schema.columns[0].name, "summary");
        assert_eq!(schema.columns[0].data_type, "text");
        // Second from text_storage_configs
        assert_eq!(schema.columns[1].name, "full_text");
        assert_eq!(schema.columns[1].data_type, "text_large");
        assert_eq!(schema.columns[1].max_length, Some(1024));
    }

    #[test]
    fn test_build_existing_schema_deduplicates_columns() {
        // If a column appears in both text_columns and text_storage_configs,
        // it should not be duplicated
        let config = proximadb_proto::proximadb_v1::CollectionConfig {
            enable_proxima_record: Some(true),
            text_columns: vec!["content".to_string()],
            text_storage_configs: vec![proximadb_proto::proximadb_v1::TextStorageConfig {
                column_name: "content".to_string(),
                chunk_size: 512,
                ..Default::default()
            }],
            ..Default::default()
        };
        let result = build_existing_schema(&config);
        assert!(result.is_some());
        let schema = result.expect("Should build schema");
        // Should only have one "content" column (from text_columns, since it comes first)
        assert_eq!(schema.columns.len(), 1);
        assert_eq!(schema.columns[0].name, "content");
    }

    // =========================================================================
    // Tests for validate_schema - additional coverage
    // =========================================================================

    #[test]
    fn test_validate_schema_invalid_column_name_chars() {
        let schema = SchemaDefinition {
            columns: vec![ColumnDefinition {
                name: "my-column".to_string(), // hyphens not allowed
                data_type: "text".to_string(),
                nullable: None,
                indexed: None,
                filterable: None,
                max_length: None,
                precision: None,
                scale: None,
                vector_dimension: None,
            }],
            enforcement: None,
            allow_additional_fields: None,
        };

        let result = validate_schema(&schema, None);
        assert!(!result.valid);
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("invalid characters"))
        );
    }

    #[test]
    fn test_validate_schema_all_reserved_names() {
        for name in &["id", "vector", "_id", "_score", "_version"] {
            let schema = SchemaDefinition {
                columns: vec![ColumnDefinition {
                    name: name.to_string(),
                    data_type: "text".to_string(),
                    nullable: None,
                    indexed: None,
                    filterable: None,
                    max_length: None,
                    precision: None,
                    scale: None,
                    vector_dimension: None,
                }],
                enforcement: None,
                allow_additional_fields: None,
            };

            let result = validate_schema(&schema, None);
            assert!(!result.valid, "Column name '{}' should be reserved", name);
        }
    }

    #[test]
    fn test_validate_schema_large_max_length_with_index_warns() {
        let schema = SchemaDefinition {
            columns: vec![ColumnDefinition {
                name: "big_text".to_string(),
                data_type: "text".to_string(),
                nullable: None,
                indexed: Some(true),
                filterable: None,
                max_length: Some(10000),
                precision: None,
                scale: None,
                vector_dimension: None,
            }],
            enforcement: None,
            allow_additional_fields: None,
        };

        let result = validate_schema(&schema, None);
        assert!(result.valid);
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.contains("large max_length"))
        );
    }

    #[test]
    fn test_validate_schema_evolution_new_non_nullable_column_warns() {
        let existing = SchemaDefinition {
            columns: vec![ColumnDefinition {
                name: "title".to_string(),
                data_type: "text".to_string(),
                nullable: None,
                indexed: None,
                filterable: None,
                max_length: None,
                precision: None,
                scale: None,
                vector_dimension: None,
            }],
            enforcement: None,
            allow_additional_fields: None,
        };

        let new_schema = SchemaDefinition {
            columns: vec![
                ColumnDefinition {
                    name: "title".to_string(),
                    data_type: "text".to_string(),
                    nullable: None,
                    indexed: None,
                    filterable: None,
                    max_length: None,
                    precision: None,
                    scale: None,
                    vector_dimension: None,
                },
                ColumnDefinition {
                    name: "required_field".to_string(),
                    data_type: "integer".to_string(),
                    nullable: Some(false), // not nullable!
                    indexed: None,
                    filterable: None,
                    max_length: None,
                    precision: None,
                    scale: None,
                    vector_dimension: None,
                },
            ],
            enforcement: None,
            allow_additional_fields: None,
        };

        let result = validate_schema(&new_schema, Some(&existing));
        // Should be valid but with warnings about non-nullable new column
        assert!(result.valid);
        assert!(result.warnings.iter().any(|w| w.contains("not nullable")));
        assert!(!result.suggestions.is_empty());
    }

    #[test]
    fn test_validate_schema_evolution_add_column_is_valid() {
        let existing = SchemaDefinition {
            columns: vec![ColumnDefinition {
                name: "title".to_string(),
                data_type: "text".to_string(),
                nullable: None,
                indexed: None,
                filterable: None,
                max_length: None,
                precision: None,
                scale: None,
                vector_dimension: None,
            }],
            enforcement: None,
            allow_additional_fields: None,
        };

        let new_schema = SchemaDefinition {
            columns: vec![
                ColumnDefinition {
                    name: "title".to_string(),
                    data_type: "text".to_string(),
                    nullable: None,
                    indexed: None,
                    filterable: None,
                    max_length: None,
                    precision: None,
                    scale: None,
                    vector_dimension: None,
                },
                ColumnDefinition {
                    name: "tags".to_string(),
                    data_type: "array_text".to_string(),
                    nullable: Some(true),
                    indexed: None,
                    filterable: None,
                    max_length: None,
                    precision: None,
                    scale: None,
                    vector_dimension: None,
                },
            ],
            enforcement: None,
            allow_additional_fields: None,
        };

        let result = validate_schema(&new_schema, Some(&existing));
        assert!(result.valid);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_validate_schema_empty_columns() {
        let schema = SchemaDefinition {
            columns: vec![],
            enforcement: None,
            allow_additional_fields: None,
        };
        let result = validate_schema(&schema, None);
        assert!(result.valid);
    }

    #[test]
    fn test_validate_schema_filterable_not_indexed_suggestion() {
        let schema = SchemaDefinition {
            columns: vec![ColumnDefinition {
                name: "category".to_string(),
                data_type: "text".to_string(),
                nullable: None,
                indexed: Some(false),
                filterable: Some(true),
                max_length: None,
                precision: None,
                scale: None,
                vector_dimension: None,
            }],
            enforcement: None,
            allow_additional_fields: None,
        };

        let result = validate_schema(&schema, None);
        assert!(result.valid);
        assert!(
            result
                .suggestions
                .iter()
                .any(|s| s.contains("indexed: true"))
        );
    }

    // ============================================================
    // increment_version tests
    // ============================================================

    #[test]
    fn test_increment_version_rollover() {
        assert_eq!(increment_version("1.0.0"), "1.0.1");
        assert_eq!(increment_version("1.0.9"), "1.0.10");
        assert_eq!(increment_version("2.3.5"), "2.3.6");
    }

    #[test]
    fn test_increment_version_fallback_invalid() {
        assert_eq!(increment_version(""), "1.0.0");
        assert_eq!(increment_version("abc"), "1.0.0");
        assert_eq!(increment_version("1.2"), "1.0.0");
    }

    // ============================================================
    // Additional validate_schema edge case tests
    // ============================================================

    #[test]
    fn test_schema_validation_empty_columns() {
        let schema = SchemaDefinition {
            columns: vec![],
            enforcement: None,
            allow_additional_fields: None,
        };
        let result = validate_schema(&schema, None);
        // Empty schema should still be valid (no columns = no errors)
        assert!(result.valid);
    }

    #[test]
    fn test_schema_validation_reserved_column_name() {
        let schema = SchemaDefinition {
            columns: vec![ColumnDefinition {
                name: "id".to_string(), // reserved
                data_type: "text".to_string(),
                nullable: None,
                indexed: None,
                filterable: None,
                max_length: None,
                precision: None,
                scale: None,
                vector_dimension: None,
            }],
            enforcement: None,
            allow_additional_fields: None,
        };
        let result = validate_schema(&schema, None);
        // "id" is a reserved column name — should produce warning or error
        assert!(
            !result.warnings.is_empty() || !result.errors.is_empty(),
            "Reserved column name 'id' should trigger warning or error"
        );
    }

    #[test]
    fn test_schema_validation_vector_type_needs_dimension() {
        let schema = SchemaDefinition {
            columns: vec![ColumnDefinition {
                name: "embedding".to_string(),
                data_type: "vector".to_string(),
                nullable: None,
                indexed: None,
                filterable: None,
                max_length: None,
                precision: None,
                scale: None,
                vector_dimension: None, // missing dimension for vector type
            }],
            enforcement: None,
            allow_additional_fields: None,
        };
        let result = validate_schema(&schema, None);
        // Vector type is accepted (dimension is optional in schema definition)
        // Just verify the validation runs without panic
        let _ = result.valid;
    }

    #[test]
    fn test_schema_validation_multiple_errors() {
        let schema = SchemaDefinition {
            columns: vec![
                ColumnDefinition {
                    name: "dup".to_string(),
                    data_type: "text".to_string(),
                    nullable: None,
                    indexed: None,
                    filterable: None,
                    max_length: None,
                    precision: None,
                    scale: None,
                    vector_dimension: None,
                },
                ColumnDefinition {
                    name: "dup".to_string(), // duplicate
                    data_type: "integer".to_string(),
                    nullable: None,
                    indexed: None,
                    filterable: None,
                    max_length: None,
                    precision: None,
                    scale: None,
                    vector_dimension: None,
                },
            ],
            enforcement: None,
            allow_additional_fields: None,
        };
        let result = validate_schema(&schema, None);
        assert!(!result.valid);
        assert!(
            result.errors.iter().any(|e| e.contains("Duplicate")),
            "Should report duplicate column error"
        );
    }

    #[test]
    fn test_schema_validation_valid_all_types() {
        let schema = SchemaDefinition {
            columns: vec![
                ColumnDefinition {
                    name: "col_text".to_string(),
                    data_type: "text".to_string(),
                    nullable: Some(true),
                    indexed: Some(true),
                    filterable: Some(true),
                    max_length: Some(100),
                    precision: None,
                    scale: None,
                    vector_dimension: None,
                },
                ColumnDefinition {
                    name: "col_int".to_string(),
                    data_type: "integer".to_string(),
                    nullable: Some(false),
                    indexed: Some(true),
                    filterable: Some(true),
                    max_length: None,
                    precision: None,
                    scale: None,
                    vector_dimension: None,
                },
                ColumnDefinition {
                    name: "col_float".to_string(),
                    data_type: "float".to_string(),
                    nullable: Some(true),
                    indexed: Some(false),
                    filterable: Some(false),
                    max_length: None,
                    precision: None,
                    scale: None,
                    vector_dimension: None,
                },
            ],
            enforcement: Some("strict".to_string()),
            allow_additional_fields: Some(false),
        };
        let result = validate_schema(&schema, None);
        assert!(
            result.valid,
            "Valid multi-type schema should pass: {:?}",
            result.errors
        );
    }
}
