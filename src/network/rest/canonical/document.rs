// Document API REST handlers
//
// REST API for MongoDB-like document operations:
// - CRUD operations on JSON documents
// - Document queries with JSON path filtering
// - Index management

use axum::{
    Router,
    extract::{Json, Path, Query, State},
    response::Json as JsonResponse,
    routing::{delete, get, patch, post},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info};

use crate::errors::{ApiError, ApiResult};
use crate::proto::proximadb_v1::{
    AggregationStage, DocIndexType, DocumentCollectionConfig, DocumentUpdate, GroupStage,
    IndexDefinition, LimitStage, LookupStage, MatchStage, ProjectStage, SkipStage, SortStage,
    SqlObject, SqlValue, UnwindStage, UpdateOperation,
};
use crate::storage::document::{
    DocumentQueryParams as ServiceQueryParams, DocumentRecord, DocumentService,
};

/// Document API state
#[derive(Clone)]
pub struct DocumentApiState {
    /// Document service
    pub document_service: Arc<DocumentService>,
}

/// Query parameters for document operations
#[derive(Debug, Deserialize)]
pub struct DocumentQueryParams {
    /// Fields to return (projection)
    #[serde(default)]
    pub projection: Option<String>,
    /// Maximum results
    #[serde(default = "default_limit")]
    pub limit: u32,
    /// JSON filter string (parsed as DocumentFilter)
    #[serde(default)]
    pub filter: Option<String>,
}

fn default_limit() -> u32 {
    100
}

/// Create document request
#[derive(Debug, Deserialize)]
pub struct CreateDocumentRequest {
    /// Document ID (optional, will be generated if not provided)
    pub id: Option<String>,
    /// Document content
    pub document: serde_json::Value,
}

/// REST request body for document-collection creation (legacy root-crate copy).
///
/// Mirrors `proximadb_api::rest::canonical::document::CreateDocumentCollectionRequestBody`
/// — Phase 9 will delete this file. The `…Body` suffix distinguishes the
/// REST shape from the proto-generated `crate::proto::v1::CreateDocumentCollectionRequest`
/// (the gRPC/wire type) already used in `crate::network::rest::canonical::handlers`.
#[derive(Debug, Deserialize)]
pub struct CreateDocumentCollectionRequestBody {
    /// Collection name
    pub name: String,
    /// Initial indexes to create
    #[serde(default)]
    pub indexes: Vec<IndexDefinitionRequest>,
}

/// Index definition request
#[derive(Debug, Clone, Deserialize)]
pub struct IndexDefinitionRequest {
    /// Index name
    pub name: Option<String>,
    /// Path to index
    pub path: String,
    /// Index type (btree, hash, inverted, fulltext)
    #[serde(default = "default_index_type")]
    pub index_type: String,
    /// Unique constraint
    #[serde(default)]
    pub unique: bool,
    /// Sparse index (skip null values)
    #[serde(default)]
    pub sparse: bool,
}

fn default_index_type() -> String {
    "btree".to_string()
}

/// Document response
#[derive(Debug, Serialize)]
pub struct DocumentResponse {
    /// Document ID
    pub id: String,
    /// Document content
    pub document: serde_json::Value,
    /// Version
    pub version: u64,
    /// Updated timestamp (ns)
    pub updated_at_ns: i64,
}

/// Query response
#[derive(Debug, Serialize)]
pub struct QueryResponse {
    /// Documents matching the query
    pub documents: Vec<DocumentResponse>,
    /// Total count (if available)
    pub total_count: Option<u64>,
    /// Whether there are more results
    pub has_more: bool,
}

/// Update document request
#[derive(Debug, Deserialize)]
pub struct UpdateDocumentRequest {
    /// Update operations to apply
    pub updates: Vec<serde_json::Value>,
    /// Expected version for optimistic locking (optional)
    pub expected_version: Option<u64>,
}

/// Batch insert request
#[derive(Debug, Deserialize)]
pub struct BatchInsertRequest {
    /// Documents to insert
    pub documents: Vec<CreateDocumentRequest>,
}

/// Batch insert response
#[derive(Debug, Serialize)]
pub struct BatchInsertResponse {
    /// Number of documents successfully inserted
    pub inserted: u64,
    /// Number of documents that failed
    pub failed: u64,
}

/// Aggregate request
#[derive(Debug, Deserialize)]
pub struct AggregateRequest {
    /// Aggregation pipeline stages
    pub pipeline: Vec<serde_json::Value>,
}

/// Aggregate response
#[derive(Debug, Serialize)]
pub struct AggregateResponse {
    /// Aggregated results
    pub results: Vec<serde_json::Value>,
    /// Query execution time in milliseconds
    pub query_time_ms: u64,
}

/// Create document collection router
pub fn create_document_router() -> Router<DocumentApiState> {
    Router::new()
        // Collection operations
        .route("/collections", post(create_collection))
        .route("/collections", get(list_collections))
        .route("/collections/{collection}", get(get_collection_info))
        .route("/collections/{collection}", delete(delete_collection))
        // Document CRUD
        .route("/collections/{collection}/documents", post(insert_document))
        .route("/collections/{collection}/documents", get(query_documents))
        .route(
            "/collections/{collection}/documents/{id}",
            get(get_document),
        )
        .route(
            "/collections/{collection}/documents/{id}",
            delete(delete_document),
        )
        .route(
            "/collections/{collection}/documents/{id}",
            patch(update_document),
        )
        // Batch and aggregate operations
        .route(
            "/collections/{collection}/documents/_batch",
            post(batch_insert_documents),
        )
        .route(
            "/collections/{collection}/documents/_aggregate",
            post(aggregate_documents),
        )
        // Index operations
        .route("/collections/{collection}/indexes", post(create_index))
        .route("/collections/{collection}/indexes", get(list_indexes))
}

/// Create a document collection
async fn create_collection(
    State(state): State<DocumentApiState>,
    Json(request): Json<CreateDocumentCollectionRequestBody>,
) -> ApiResult<JsonResponse<serde_json::Value>> {
    info!("Creating document collection: {}", request.name);

    // Create indexes from request
    let index_definitions: Vec<IndexDefinition> = request
        .indexes
        .into_iter()
        .map(|idx| IndexDefinition {
            name: idx.name,
            path: idx.path,
            index_type: match idx.index_type.to_lowercase().as_str() {
                "btree" => DocIndexType::Btree as i32,
                "hash" => DocIndexType::Hash as i32,
                "inverted" => DocIndexType::Inverted as i32,
                "fulltext" => DocIndexType::Fulltext as i32,
                "geo" => DocIndexType::Geo as i32,
                _ => DocIndexType::Btree as i32,
            },
            unique: idx.unique,
            sparse: idx.sparse,
        })
        .collect();

    // Create collection config
    let config = DocumentCollectionConfig {
        name: request.name.clone(),
        json_schema: None,
        indexes: index_definitions,
        enable_fulltext: false,
        fulltext_paths: Vec::new(),
        ttl_seconds: 0, // No expiry by default
        compression: None,
        use_jsonb: false,
    };

    state
        .document_service
        .create_collection(&request.name, config)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to create collection: {}", e)))?;

    Ok(JsonResponse(serde_json::json!({
        "success": true,
        "collection": request.name
    })))
}

/// List document collections
async fn list_collections(
    State(state): State<DocumentApiState>,
) -> ApiResult<JsonResponse<serde_json::Value>> {
    let collections = state
        .document_service
        .list_collections()
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to list collections: {}", e)))?;

    // Convert to serializable format
    let collection_info: Vec<serde_json::Value> = collections
        .iter()
        .map(|c| {
            serde_json::json!({
                "name": c.name,
                "document_count": c.document_count,
                "storage_size_bytes": c.storage_size_bytes
            })
        })
        .collect();

    Ok(JsonResponse(serde_json::json!({
        "collections": collection_info
    })))
}

/// Get collection info
async fn get_collection_info(
    State(state): State<DocumentApiState>,
    Path(collection): Path<String>,
) -> ApiResult<JsonResponse<serde_json::Value>> {
    let info = state
        .document_service
        .get_collection(&collection)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to get collection: {}", e)))?
        .ok_or_else(|| ApiError::CollectionNotFound(collection.clone()))?;

    Ok(JsonResponse(serde_json::json!({
        "name": info.name,
        "document_count": info.document_count,
        "indexes": info.indexes.iter().map(|i| serde_json::json!({
            "name": i.name,
            "path": i.path,
            "unique": i.unique
        })).collect::<Vec<_>>()
    })))
}

/// Delete a collection
async fn delete_collection(
    State(state): State<DocumentApiState>,
    Path(collection): Path<String>,
) -> ApiResult<JsonResponse<serde_json::Value>> {
    info!("Deleting document collection: {}", collection);

    state
        .document_service
        .delete_collection(&collection)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to delete collection: {}", e)))?;

    Ok(JsonResponse(serde_json::json!({
        "success": true
    })))
}

/// Insert a document
async fn insert_document(
    State(state): State<DocumentApiState>,
    Path(collection): Path<String>,
    Json(request): Json<CreateDocumentRequest>,
) -> ApiResult<JsonResponse<DocumentResponse>> {
    debug!("Inserting document into collection: {}", collection);

    // Convert JSON to SqlObject
    let sql_object = json_to_sql_object(&request.document)?;

    // Pass id as Option<&str>
    let id_ref = request.id.as_deref();

    let record = state
        .document_service
        .insert_document(&collection, id_ref, sql_object)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to insert document: {}", e)))?;

    Ok(JsonResponse(document_record_to_response(&record)?))
}

/// Get a document by ID
async fn get_document(
    State(state): State<DocumentApiState>,
    Path((collection, id)): Path<(String, String)>,
    Query(params): Query<DocumentQueryParams>,
) -> ApiResult<JsonResponse<DocumentResponse>> {
    debug!("Getting document: {}/{}", collection, id);

    let projection = params.projection.map(|p| {
        p.split(',')
            .map(|s| s.trim().to_string())
            .collect::<Vec<_>>()
    });

    let record = state
        .document_service
        .get_document(&collection, &id, projection)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to get document: {}", e)))?
        .ok_or_else(|| ApiError::CollectionNotFound(format!("Document {} not found", id)))?;

    Ok(JsonResponse(document_record_to_response(&record)?))
}

/// Delete a document
async fn delete_document(
    State(state): State<DocumentApiState>,
    Path((collection, id)): Path<(String, String)>,
) -> ApiResult<JsonResponse<serde_json::Value>> {
    debug!("Deleting document: {}/{}", collection, id);

    state
        .document_service
        .delete_document(&collection, &id)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to delete document: {}", e)))?;

    Ok(JsonResponse(serde_json::json!({
        "success": true,
        "id": id
    })))
}

/// Query documents
async fn query_documents(
    State(state): State<DocumentApiState>,
    Path(collection): Path<String>,
    Query(params): Query<DocumentQueryParams>,
) -> ApiResult<JsonResponse<QueryResponse>> {
    debug!("Querying documents in collection: {}", collection);

    // Build service query params
    let projection = params
        .projection
        .map(|p| {
            p.split(',')
                .map(|s| s.trim().to_string())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    // Parse filter from JSON string if present
    let filter = match &params.filter {
        Some(filter_str) => {
            let parsed: crate::proto::proximadb_v1::DocumentFilter =
                serde_json::from_str(filter_str).map_err(|e| {
                    ApiError::InvalidArgument(format!("Invalid filter JSON: {}", e))
                })?;
            Some(parsed)
        }
        None => None,
    };

    let query_params = ServiceQueryParams {
        filter,
        projection,
        sort: Vec::new(),
        limit: params.limit,
        offset: 0,
        include_count: true,
    };

    let result = state
        .document_service
        .query_documents(&collection, query_params)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to query documents: {}", e)))?;

    let documents: Vec<DocumentResponse> = result
        .documents
        .iter()
        .map(document_record_to_response)
        .collect::<Result<Vec<_>, _>>()?;

    let has_more = documents.len() >= params.limit as usize;

    Ok(JsonResponse(QueryResponse {
        documents,
        total_count: result.total_count,
        has_more,
    }))
}

/// Update a document by ID
async fn update_document(
    State(state): State<DocumentApiState>,
    Path((collection, id)): Path<(String, String)>,
    Json(request): Json<UpdateDocumentRequest>,
) -> ApiResult<JsonResponse<DocumentResponse>> {
    debug!("Updating document: {}/{}", collection, id);

    // Convert JSON update operations to DocumentUpdate proto objects
    let updates: Vec<DocumentUpdate> = request
        .updates
        .iter()
        .map(|v| {
            // Each update value should be an object with operation, path, value fields
            let operation = v
                .get("operation")
                .and_then(|o| o.as_str())
                .map(|s| match s.to_lowercase().as_str() {
                    "set" => UpdateOperation::Set as i32,
                    "unset" => UpdateOperation::Unset as i32,
                    "inc" => UpdateOperation::Inc as i32,
                    "push" => UpdateOperation::Push as i32,
                    "pull" => UpdateOperation::Pull as i32,
                    _ => UpdateOperation::Set as i32,
                })
                .unwrap_or(UpdateOperation::Set as i32);

            let path = v
                .get("path")
                .and_then(|p| p.as_str())
                .unwrap_or("")
                .to_string();

            let value = v.get("value").map(json_to_sql_value);

            DocumentUpdate {
                operation,
                path,
                value,
            }
        })
        .collect();

    let record = state
        .document_service
        .update_document(&collection, &id, updates, request.expected_version)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to update document: {}", e)))?;

    Ok(JsonResponse(document_record_to_response(&record)?))
}

/// Batch insert documents
async fn batch_insert_documents(
    State(state): State<DocumentApiState>,
    Path(collection): Path<String>,
    Json(request): Json<BatchInsertRequest>,
) -> ApiResult<JsonResponse<BatchInsertResponse>> {
    debug!(
        "Batch inserting {} documents into collection: {}",
        request.documents.len(),
        collection
    );

    let mut inserted: u64 = 0;
    let mut failed: u64 = 0;

    for doc in &request.documents {
        let sql_object = match json_to_sql_object(&doc.document) {
            Ok(obj) => obj,
            Err(_) => {
                failed += 1;
                continue;
            }
        };

        let id_ref = doc.id.as_deref();
        match state
            .document_service
            .insert_document(&collection, id_ref, sql_object)
            .await
        {
            Ok(_) => inserted += 1,
            Err(_) => failed += 1,
        }
    }

    Ok(JsonResponse(BatchInsertResponse { inserted, failed }))
}

/// Aggregate documents
async fn aggregate_documents(
    State(state): State<DocumentApiState>,
    Path(collection): Path<String>,
    Json(request): Json<AggregateRequest>,
) -> ApiResult<JsonResponse<AggregateResponse>> {
    debug!("Aggregating documents in collection: {}", collection);

    // Convert JSON pipeline stages to AggregationStage proto objects.
    // AggregationStage is a prost oneof without serde derives, so we parse manually
    // based on the stage type key in each JSON object.
    let pipeline: Vec<AggregationStage> = request
        .pipeline
        .iter()
        .filter_map(|v| json_to_aggregation_stage(v).ok())
        .collect();

    let result = state
        .document_service
        .aggregate_documents(&collection, None, pipeline)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to aggregate documents: {}", e)))?;

    let results: Vec<serde_json::Value> = result.results.iter().map(sql_object_to_json).collect();

    Ok(JsonResponse(AggregateResponse {
        results,
        query_time_ms: result.query_time_ms,
    }))
}

/// Create an index
/// Note: Indexes are created during collection creation via the config.
/// This endpoint is for adding indexes to existing collections (deferred: implement)
async fn create_index(
    State(_state): State<DocumentApiState>,
    Path(collection): Path<String>,
    Json(request): Json<IndexDefinitionRequest>,
) -> ApiResult<JsonResponse<serde_json::Value>> {
    info!("Creating index on {}: {:?}", collection, request.path);

    // Deferred: Implement adding indexes to existing collections
    // For now, indexes must be specified at collection creation time
    Err(ApiError::InvalidArgument(
        "Creating indexes on existing collections is not yet supported. Specify indexes when creating the collection.".to_string()
    ))
}

/// List indexes
async fn list_indexes(
    State(state): State<DocumentApiState>,
    Path(collection): Path<String>,
) -> ApiResult<JsonResponse<serde_json::Value>> {
    let info = state
        .document_service
        .get_collection(&collection)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to get collection: {}", e)))?
        .ok_or_else(|| ApiError::CollectionNotFound(collection.clone()))?;

    Ok(JsonResponse(serde_json::json!({
        "indexes": info.indexes.iter().map(|i| serde_json::json!({
            "name": i.name,
            "path": i.path,
            "unique": i.unique
        })).collect::<Vec<_>>()
    })))
}

// Helper functions

/// Convert JSON Value to SqlObject
fn json_to_sql_object(value: &serde_json::Value) -> ApiResult<SqlObject> {
    match value {
        serde_json::Value::Object(map) => {
            let fields: HashMap<String, SqlValue> = map
                .iter()
                .map(|(k, v)| (k.clone(), json_to_sql_value(v)))
                .collect();
            Ok(SqlObject { fields })
        }
        _ => Err(ApiError::InvalidArgument(
            "Document must be a JSON object".to_string(),
        )),
    }
}

/// Convert JSON Value to SqlValue
fn json_to_sql_value(value: &serde_json::Value) -> SqlValue {
    use crate::proto::proximadb_v1::SqlArray;
    use crate::proto::proximadb_v1::sql_value::Value;

    let inner = match value {
        serde_json::Value::Null => Some(Value::NullValue(0)),
        serde_json::Value::Bool(b) => Some(Value::BoolValue(*b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(Value::Int64Value(i))
            } else if let Some(f) = n.as_f64() {
                Some(Value::NumberValue(f))
            } else {
                Some(Value::StringValue(n.to_string()))
            }
        }
        serde_json::Value::String(s) => Some(Value::StringValue(s.clone())),
        serde_json::Value::Array(arr) => {
            let values: Vec<SqlValue> = arr.iter().map(json_to_sql_value).collect();
            Some(Value::ArrayValue(SqlArray { values }))
        }
        serde_json::Value::Object(map) => {
            let fields: HashMap<String, SqlValue> = map
                .iter()
                .map(|(k, v)| (k.clone(), json_to_sql_value(v)))
                .collect();
            Some(Value::ObjectValue(SqlObject { fields }))
        }
    };

    SqlValue { value: inner }
}

/// Convert SqlValue to JSON
fn sql_value_to_json(value: &SqlValue) -> serde_json::Value {
    proximadb_records::conversions::sql_value_to_json(value)
}

/// Convert SqlObject to JSON
fn sql_object_to_json(obj: &SqlObject) -> serde_json::Value {
    let map: serde_json::Map<String, serde_json::Value> = obj
        .fields
        .iter()
        .map(|(k, v)| (k.clone(), sql_value_to_json(v)))
        .collect();
    serde_json::Value::Object(map)
}

/// Convert a JSON value into an AggregationStage.
///
/// Expected format: `{ "$match": {...} }` or `{ "$group": {...} }` etc.
fn json_to_aggregation_stage(value: &serde_json::Value) -> ApiResult<AggregationStage> {
    use crate::proto::proximadb_v1::aggregation_stage::Stage;

    let obj = value
        .as_object()
        .ok_or_else(|| ApiError::InvalidArgument("Pipeline stage must be a JSON object".into()))?;

    // Determine stage type from the first key
    if let Some(match_val) = obj.get("$match").or_else(|| obj.get("match")) {
        let stage: MatchStage = serde_json::from_value(match_val.clone())
            .map_err(|e| ApiError::InvalidArgument(format!("Invalid $match stage: {}", e)))?;
        Ok(AggregationStage {
            stage: Some(Stage::Match(stage)),
        })
    } else if let Some(group_val) = obj.get("$group").or_else(|| obj.get("group")) {
        let stage: GroupStage = serde_json::from_value(group_val.clone())
            .map_err(|e| ApiError::InvalidArgument(format!("Invalid $group stage: {}", e)))?;
        Ok(AggregationStage {
            stage: Some(Stage::Group(stage)),
        })
    } else if let Some(project_val) = obj.get("$project").or_else(|| obj.get("project")) {
        let stage: ProjectStage = serde_json::from_value(project_val.clone())
            .map_err(|e| ApiError::InvalidArgument(format!("Invalid $project stage: {}", e)))?;
        Ok(AggregationStage {
            stage: Some(Stage::Project(stage)),
        })
    } else if let Some(sort_val) = obj.get("$sort").or_else(|| obj.get("sort")) {
        let stage: SortStage = serde_json::from_value(sort_val.clone())
            .map_err(|e| ApiError::InvalidArgument(format!("Invalid $sort stage: {}", e)))?;
        Ok(AggregationStage {
            stage: Some(Stage::Sort(stage)),
        })
    } else if let Some(limit_val) = obj.get("$limit").or_else(|| obj.get("limit")) {
        // $limit can be a plain number: { "$limit": 10 }
        let limit = limit_val.as_u64().unwrap_or(100) as u32;
        Ok(AggregationStage {
            stage: Some(Stage::Limit(LimitStage { limit })),
        })
    } else if let Some(skip_val) = obj.get("$skip").or_else(|| obj.get("skip")) {
        let skip = skip_val.as_u64().unwrap_or(0) as u32;
        Ok(AggregationStage {
            stage: Some(Stage::Skip(SkipStage { skip })),
        })
    } else if let Some(unwind_val) = obj.get("$unwind").or_else(|| obj.get("unwind")) {
        let stage: UnwindStage = if unwind_val.is_string() {
            UnwindStage {
                path: unwind_val.as_str().unwrap_or("").to_string(),
                preserve_null: false,
            }
        } else {
            serde_json::from_value(unwind_val.clone())
                .map_err(|e| ApiError::InvalidArgument(format!("Invalid $unwind stage: {}", e)))?
        };
        Ok(AggregationStage {
            stage: Some(Stage::Unwind(stage)),
        })
    } else if let Some(lookup_val) = obj.get("$lookup").or_else(|| obj.get("lookup")) {
        let stage: LookupStage = serde_json::from_value(lookup_val.clone())
            .map_err(|e| ApiError::InvalidArgument(format!("Invalid $lookup stage: {}", e)))?;
        Ok(AggregationStage {
            stage: Some(Stage::Lookup(stage)),
        })
    } else {
        Err(ApiError::InvalidArgument(format!(
            "Unknown pipeline stage: {}",
            value
        )))
    }
}

/// Convert DocumentRecord to response
fn document_record_to_response(record: &DocumentRecord) -> ApiResult<DocumentResponse> {
    Ok(DocumentResponse {
        id: record.id.clone(),
        document: sql_object_to_json(&crate::storage::document::proxima_tree_to_sql_object(
            &record.props,
        )),
        version: record.version,
        updated_at_ns: record.updated_at_ns,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_to_sql_value() {
        let json = serde_json::json!({"name": "test", "count": 42, "active": true});
        let obj = json_to_sql_object(&json).unwrap();
        assert!(obj.fields.contains_key("name"));
        assert!(obj.fields.contains_key("count"));
        assert!(obj.fields.contains_key("active"));
    }
}
