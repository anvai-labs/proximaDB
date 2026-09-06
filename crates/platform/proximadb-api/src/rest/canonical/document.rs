//! # Document REST Handlers
//!
//! REST endpoints for MongoDB-like document operations.  All handlers delegate
//! to `DocumentPort` so this module compiles without any dependency on root-crate
//! concrete service types.

use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    Router,
    extract::{Json, Path, Query, State},
    response::Json as JsonResponse,
    routing::{get, post},
};
use proximadb_proto::v1::{
    AggregateDocumentsRequest, AggregationStage, CreateDocumentCollectionRequest,
    DeleteDocumentCollectionRequest, DeleteDocumentRequest, DocIndexType, DocumentCollectionConfig,
    DocumentFilter, DocumentUpdate, GetDocumentRequest, GroupStage, IndexDefinition,
    InsertDocumentRequest, LimitStage, ListDocumentCollectionsRequest, LookupStage, MatchStage,
    ProjectStage, QueryDocumentsRequest, SkipStage, SortStage, SqlArray, SqlObject, SqlValue,
    UnwindStage, UpdateDocumentRequest, UpdateOperation, aggregation_stage::Stage as AggStage,
};
use proximadb_runtime::DocumentPort;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::rest::errors::{RestError, RestResult};

// ── State ─────────────────────────────────────────────────────────────────────

/// Axum state for document REST endpoints.
///
/// Held behind `Arc<dyn DocumentPort>` so no root-crate concrete type crosses
/// the crate boundary.
#[derive(Clone)]
pub struct DocumentRestState {
    pub document_port: Arc<dyn DocumentPort>,
}

// ── Legacy stub types kept for re-export compatibility ────────────────────────

/// Document handler stub.
pub struct DocumentHandler;

impl DocumentHandler {
    pub fn new() -> Self {
        Self
    }
}

impl Default for DocumentHandler {
    fn default() -> Self {
        Self::new()
    }
}

/// Document query handler stub.
pub struct DocumentQueryHandler;

impl DocumentQueryHandler {
    pub fn new() -> Self {
        Self
    }
}

impl Default for DocumentQueryHandler {
    fn default() -> Self {
        Self::new()
    }
}

// ── Request / Response types ──────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct DocumentQueryParams {
    #[serde(default)]
    pub projection: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: u32,
    #[serde(default)]
    pub filter: Option<String>,
}

fn default_limit() -> u32 {
    100
}

#[derive(Debug, Deserialize)]
pub struct CreateDocumentRequest {
    pub id: Option<String>,
    pub document: serde_json::Value,
}

/// REST request body for document-collection creation.
///
/// The `…Body` suffix distinguishes the local REST shape from the
/// proto-generated `proximadb_proto::v1::CreateDocumentCollectionRequest`
/// (the gRPC/wire type the handler converts into) — imported at the top
/// of this module.
#[derive(Debug, Deserialize)]
pub struct CreateDocumentCollectionRequestBody {
    pub name: String,
    #[serde(default)]
    pub indexes: Vec<IndexDefinitionRequest>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IndexDefinitionRequest {
    pub name: Option<String>,
    pub path: String,
    #[serde(default = "default_index_type")]
    pub index_type: String,
    #[serde(default)]
    pub unique: bool,
    #[serde(default)]
    pub sparse: bool,
}

fn default_index_type() -> String {
    "btree".to_string()
}

#[derive(Debug, Serialize)]
pub struct DocumentResponse {
    pub id: String,
    pub document: serde_json::Value,
    pub version: u64,
}

#[derive(Debug, Serialize)]
pub struct QueryResponse {
    pub documents: Vec<DocumentResponse>,
    pub total_count: Option<u64>,
    pub has_more: bool,
}

#[derive(Debug, Deserialize)]
pub struct UpdateDocumentBody {
    pub updates: Vec<serde_json::Value>,
    pub expected_version: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct BatchInsertRequest {
    pub documents: Vec<CreateDocumentRequest>,
}

#[derive(Debug, Serialize)]
pub struct BatchInsertResponse {
    pub inserted: u64,
    pub failed: u64,
}

#[derive(Debug, Deserialize)]
pub struct AggregateRequest {
    pub pipeline: Vec<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct AggregateResponse {
    pub results: Vec<serde_json::Value>,
    pub query_time_ms: u64,
}

// ── Router ────────────────────────────────────────────────────────────────────

pub fn create_document_router() -> Router<DocumentRestState> {
    super::with_v1_compatibility_headers(
        Router::new()
            // Mounted at `/api/v2/document-collections` (see handlers.rs). The
            // resource word is supplied by the mount prefix, so internal routes
            // start at the collection id — no `documents/collections/.../documents`
            // doubling, and no Elasticsearch `_batch`/`_aggregate` underscores.
            .route("/", post(create_collection).get(list_collections))
            .route(
                "/{collection}",
                get(get_collection_info).delete(delete_collection),
            )
            .route(
                "/{collection}/documents",
                post(insert_document).get(query_documents),
            )
            .route(
                "/{collection}/documents/{id}",
                get(get_document)
                    .delete(delete_document)
                    .patch(update_document),
            )
            .route(
                "/{collection}/documents/batch",
                post(batch_insert_documents),
            )
            .route(
                "/{collection}/documents/aggregate",
                post(aggregate_documents),
            )
            .route(
                "/{collection}/indexes",
                post(create_index).get(list_indexes),
            ),
    )
}

// ── Handlers ──────────────────────────────────────────────────────────────────

async fn create_collection(
    State(state): State<DocumentRestState>,
    Json(request): Json<CreateDocumentCollectionRequestBody>,
) -> RestResult<JsonResponse<serde_json::Value>> {
    info!("Creating document collection: {}", request.name);

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

    let config = DocumentCollectionConfig {
        name: request.name.clone(),
        json_schema: None,
        indexes: index_definitions,
        enable_fulltext: false,
        fulltext_paths: Vec::new(),
        ttl_seconds: 0,
        compression: None,
        use_jsonb: false,
    };

    state
        .document_port
        .create_collection(CreateDocumentCollectionRequest {
            config: Some(config),
        })
        .await
        .map_err(|e| RestError::Internal(format!("Failed to create collection: {}", e)))?;

    Ok(JsonResponse(serde_json::json!({
        "success": true,
        "collection": request.name
    })))
}

async fn list_collections(
    State(state): State<DocumentRestState>,
) -> RestResult<JsonResponse<serde_json::Value>> {
    let resp = state
        .document_port
        .list_collections(ListDocumentCollectionsRequest {})
        .await
        .map_err(|e| RestError::Internal(format!("Failed to list collections: {}", e)))?;

    let collections: Vec<serde_json::Value> = resp
        .collections
        .iter()
        .map(|c| {
            serde_json::json!({
                "name": c.name,
                "document_count": c.document_count,
                "storage_size_bytes": c.storage_size_bytes
            })
        })
        .collect();

    Ok(JsonResponse(
        serde_json::json!({ "collections": collections }),
    ))
}

async fn get_collection_info(
    State(state): State<DocumentRestState>,
    Path(collection): Path<String>,
) -> RestResult<JsonResponse<serde_json::Value>> {
    let resp = state
        .document_port
        .list_collections(ListDocumentCollectionsRequest {})
        .await
        .map_err(|e| RestError::Internal(format!("Failed to list collections: {}", e)))?;

    let info = resp
        .collections
        .into_iter()
        .find(|c| c.name == collection)
        .ok_or_else(|| RestError::CollectionNotFound(collection.clone()))?;

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

async fn delete_collection(
    State(state): State<DocumentRestState>,
    Path(collection): Path<String>,
) -> RestResult<JsonResponse<serde_json::Value>> {
    info!("Deleting document collection: {}", collection);

    state
        .document_port
        .delete_collection(DeleteDocumentCollectionRequest { collection })
        .await
        .map_err(|e| RestError::Internal(format!("Failed to delete collection: {}", e)))?;

    Ok(JsonResponse(serde_json::json!({ "success": true })))
}

async fn insert_document(
    State(state): State<DocumentRestState>,
    Path(collection): Path<String>,
    Json(request): Json<CreateDocumentRequest>,
) -> RestResult<JsonResponse<DocumentResponse>> {
    debug!("Inserting document into collection: {}", collection);

    let sql_object = json_to_sql_object(&request.document)?;

    let resp = state
        .document_port
        .insert_document(InsertDocumentRequest {
            collection,
            id: request.id,
            document: Some(sql_object),
        })
        .await
        .map_err(|e| RestError::Internal(format!("Failed to insert document: {}", e)))?;

    Ok(JsonResponse(DocumentResponse {
        id: resp.id,
        document: request.document,
        version: resp.version,
    }))
}

async fn get_document(
    State(state): State<DocumentRestState>,
    Path((collection, id)): Path<(String, String)>,
    Query(params): Query<DocumentQueryParams>,
) -> RestResult<JsonResponse<DocumentResponse>> {
    debug!("Getting document: {}/{}", collection, id);

    let projection: Vec<String> = params
        .projection
        .map(|p| p.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();

    let resp = state
        .document_port
        .get_document(GetDocumentRequest {
            collection,
            id: id.clone(),
            projection,
        })
        .await
        .map_err(|e| RestError::Internal(format!("Failed to get document: {}", e)))?;

    if !resp.found {
        return Err(RestError::NotFound(format!("Document {} not found", id)));
    }

    let document = resp
        .document
        .map(|d| sql_object_to_json(&d))
        .unwrap_or(serde_json::Value::Null);

    Ok(JsonResponse(DocumentResponse {
        id,
        document,
        version: resp.version,
    }))
}

async fn delete_document(
    State(state): State<DocumentRestState>,
    Path((collection, id)): Path<(String, String)>,
) -> RestResult<JsonResponse<serde_json::Value>> {
    debug!("Deleting document: {}/{}", collection, id);

    state
        .document_port
        .delete_document(DeleteDocumentRequest {
            collection,
            id: id.clone(),
        })
        .await
        .map_err(|e| RestError::Internal(format!("Failed to delete document: {}", e)))?;

    Ok(JsonResponse(
        serde_json::json!({ "success": true, "id": id }),
    ))
}

async fn query_documents(
    State(state): State<DocumentRestState>,
    Path(collection): Path<String>,
    Query(params): Query<DocumentQueryParams>,
) -> RestResult<JsonResponse<QueryResponse>> {
    debug!("Querying documents in collection: {}", collection);

    let projection: Vec<String> = params
        .projection
        .map(|p| p.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();

    let filter: Option<DocumentFilter> = match &params.filter {
        Some(filter_str) => Some(
            serde_json::from_str(filter_str)
                .map_err(|e| RestError::InvalidArgument(format!("Invalid filter JSON: {}", e)))?,
        ),
        None => None,
    };

    let limit = params.limit;

    let resp = state
        .document_port
        .query_documents(QueryDocumentsRequest {
            collection,
            filter,
            projection,
            sort: Vec::new(),
            limit,
            offset: 0,
            include_count: true,
        })
        .await
        .map_err(|e| RestError::Internal(format!("Failed to query documents: {}", e)))?;

    let has_more = resp.documents.len() >= limit as usize;

    let documents: Vec<DocumentResponse> = resp
        .documents
        .into_iter()
        .map(|r| DocumentResponse {
            id: r.id,
            document: r
                .document
                .map(|d| sql_object_to_json(&d))
                .unwrap_or(serde_json::Value::Null),
            version: r.version,
        })
        .collect();

    Ok(JsonResponse(QueryResponse {
        documents,
        total_count: resp.total_count,
        has_more,
    }))
}

async fn update_document(
    State(state): State<DocumentRestState>,
    Path((collection, id)): Path<(String, String)>,
    Json(request): Json<UpdateDocumentBody>,
) -> RestResult<JsonResponse<serde_json::Value>> {
    debug!("Updating document: {}/{}", collection, id);

    let updates: Vec<DocumentUpdate> = request
        .updates
        .iter()
        .map(|v| {
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

    let resp = state
        .document_port
        .update_document(UpdateDocumentRequest {
            collection,
            id: id.clone(),
            updates,
            expected_version: request.expected_version,
        })
        .await
        .map_err(|e| RestError::Internal(format!("Failed to update document: {}", e)))?;

    Ok(JsonResponse(serde_json::json!({
        "success": resp.success,
        "id": id,
        "new_version": resp.new_version
    })))
}

async fn batch_insert_documents(
    State(state): State<DocumentRestState>,
    Path(collection): Path<String>,
    Json(request): Json<BatchInsertRequest>,
) -> RestResult<JsonResponse<BatchInsertResponse>> {
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

        match state
            .document_port
            .insert_document(InsertDocumentRequest {
                collection: collection.clone(),
                id: doc.id.clone(),
                document: Some(sql_object),
            })
            .await
        {
            Ok(_) => inserted += 1,
            Err(_) => failed += 1,
        }
    }

    Ok(JsonResponse(BatchInsertResponse { inserted, failed }))
}

async fn aggregate_documents(
    State(state): State<DocumentRestState>,
    Path(collection): Path<String>,
    Json(request): Json<AggregateRequest>,
) -> RestResult<JsonResponse<AggregateResponse>> {
    debug!("Aggregating documents in collection: {}", collection);

    let pipeline: Vec<AggregationStage> = request
        .pipeline
        .iter()
        .filter_map(|v| json_to_aggregation_stage(v).ok())
        .collect();

    let resp = state
        .document_port
        .aggregate_documents(AggregateDocumentsRequest {
            collection,
            filter: None,
            pipeline,
        })
        .await
        .map_err(|e| RestError::Internal(format!("Failed to aggregate documents: {}", e)))?;

    let results: Vec<serde_json::Value> = resp.results.iter().map(sql_object_to_json).collect();

    Ok(JsonResponse(AggregateResponse {
        results,
        query_time_ms: resp.query_time_ms,
    }))
}

async fn create_index(
    State(_state): State<DocumentRestState>,
    Path(collection): Path<String>,
    Json(request): Json<IndexDefinitionRequest>,
) -> RestResult<JsonResponse<serde_json::Value>> {
    info!("Creating index on {}: {:?}", collection, request.path);
    Err(RestError::InvalidArgument(
        "Creating indexes on existing collections is not yet supported; \
         specify indexes when creating the collection."
            .to_string(),
    ))
}

async fn list_indexes(
    State(state): State<DocumentRestState>,
    Path(collection): Path<String>,
) -> RestResult<JsonResponse<serde_json::Value>> {
    let resp = state
        .document_port
        .list_collections(ListDocumentCollectionsRequest {})
        .await
        .map_err(|e| RestError::Internal(format!("Failed to list collections: {}", e)))?;

    let info = resp
        .collections
        .into_iter()
        .find(|c| c.name == collection)
        .ok_or_else(|| RestError::CollectionNotFound(collection.clone()))?;

    Ok(JsonResponse(serde_json::json!({
        "indexes": info.indexes.iter().map(|i| serde_json::json!({
            "name": i.name,
            "path": i.path,
            "unique": i.unique
        })).collect::<Vec<_>>()
    })))
}

// ── JSON ↔ proto conversion helpers ──────────────────────────────────────────

pub fn json_to_sql_object(value: &serde_json::Value) -> RestResult<SqlObject> {
    match value {
        serde_json::Value::Object(map) => {
            let fields: HashMap<String, SqlValue> = map
                .iter()
                .map(|(k, v)| (k.clone(), json_to_sql_value(v)))
                .collect();
            Ok(SqlObject { fields })
        }
        _ => Err(RestError::InvalidArgument(
            "Document must be a JSON object".to_string(),
        )),
    }
}

pub fn json_to_sql_value(value: &serde_json::Value) -> SqlValue {
    use proximadb_proto::v1::sql_value::Value;

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

pub fn sql_value_to_json(value: &SqlValue) -> serde_json::Value {
    proximadb_records::conversions::sql_value_to_json(value)
}

pub fn sql_object_to_json(obj: &SqlObject) -> serde_json::Value {
    let map: serde_json::Map<String, serde_json::Value> = obj
        .fields
        .iter()
        .map(|(k, v)| (k.clone(), sql_value_to_json(v)))
        .collect();
    serde_json::Value::Object(map)
}

fn json_to_aggregation_stage(value: &serde_json::Value) -> RestResult<AggregationStage> {
    let obj = value
        .as_object()
        .ok_or_else(|| RestError::InvalidArgument("Pipeline stage must be a JSON object".into()))?;

    if let Some(match_val) = obj.get("$match").or_else(|| obj.get("match")) {
        let stage: MatchStage = serde_json::from_value(match_val.clone())
            .map_err(|e| RestError::InvalidArgument(format!("Invalid $match stage: {}", e)))?;
        Ok(AggregationStage {
            stage: Some(AggStage::Match(stage)),
        })
    } else if let Some(group_val) = obj.get("$group").or_else(|| obj.get("group")) {
        let stage: GroupStage = serde_json::from_value(group_val.clone())
            .map_err(|e| RestError::InvalidArgument(format!("Invalid $group stage: {}", e)))?;
        Ok(AggregationStage {
            stage: Some(AggStage::Group(stage)),
        })
    } else if let Some(project_val) = obj.get("$project").or_else(|| obj.get("project")) {
        let stage: ProjectStage = serde_json::from_value(project_val.clone())
            .map_err(|e| RestError::InvalidArgument(format!("Invalid $project stage: {}", e)))?;
        Ok(AggregationStage {
            stage: Some(AggStage::Project(stage)),
        })
    } else if let Some(sort_val) = obj.get("$sort").or_else(|| obj.get("sort")) {
        let stage: SortStage = serde_json::from_value(sort_val.clone())
            .map_err(|e| RestError::InvalidArgument(format!("Invalid $sort stage: {}", e)))?;
        Ok(AggregationStage {
            stage: Some(AggStage::Sort(stage)),
        })
    } else if let Some(limit_val) = obj.get("$limit").or_else(|| obj.get("limit")) {
        let limit = limit_val.as_u64().unwrap_or(100) as u32;
        Ok(AggregationStage {
            stage: Some(AggStage::Limit(LimitStage { limit })),
        })
    } else if let Some(skip_val) = obj.get("$skip").or_else(|| obj.get("skip")) {
        let skip = skip_val.as_u64().unwrap_or(0) as u32;
        Ok(AggregationStage {
            stage: Some(AggStage::Skip(SkipStage { skip })),
        })
    } else if let Some(unwind_val) = obj.get("$unwind").or_else(|| obj.get("unwind")) {
        let stage = if unwind_val.is_string() {
            UnwindStage {
                path: unwind_val.as_str().unwrap_or("").to_string(),
                preserve_null: false,
            }
        } else {
            serde_json::from_value(unwind_val.clone())
                .map_err(|e| RestError::InvalidArgument(format!("Invalid $unwind stage: {}", e)))?
        };
        Ok(AggregationStage {
            stage: Some(AggStage::Unwind(stage)),
        })
    } else if let Some(lookup_val) = obj.get("$lookup").or_else(|| obj.get("lookup")) {
        let stage: LookupStage = serde_json::from_value(lookup_val.clone())
            .map_err(|e| RestError::InvalidArgument(format!("Invalid $lookup stage: {}", e)))?;
        Ok(AggregationStage {
            stage: Some(AggStage::Lookup(stage)),
        })
    } else {
        Err(RestError::InvalidArgument(format!(
            "Unknown pipeline stage: {}",
            value
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use async_trait::async_trait;
    use proximadb_proto::v1::{
        AggregateDocumentsResponse, CreateDocumentCollectionResponse,
        DeleteDocumentCollectionResponse, DeleteDocumentResponse, DocumentCollectionInfo,
        DocumentResult, GetDocumentResponse, InsertDocumentResponse,
        ListDocumentCollectionsResponse, QueryDocumentsResponse, UpdateDocumentResponse,
    };

    struct MockDocumentPort;

    fn sample_object() -> SqlObject {
        json_to_sql_object(&serde_json::json!({"name": "doc", "count": 1})).unwrap()
    }

    #[async_trait]
    impl DocumentPort for MockDocumentPort {
        async fn create_collection(
            &self,
            _request: CreateDocumentCollectionRequest,
        ) -> Result<CreateDocumentCollectionResponse> {
            Ok(CreateDocumentCollectionResponse {
                collection_id: "docs".to_string(),
                success: true,
            })
        }

        async fn list_collections(
            &self,
            _request: ListDocumentCollectionsRequest,
        ) -> Result<ListDocumentCollectionsResponse> {
            Ok(ListDocumentCollectionsResponse {
                collections: vec![DocumentCollectionInfo {
                    name: "docs".to_string(),
                    document_count: 2,
                    storage_size_bytes: 256,
                    indexes: vec![IndexDefinition {
                        name: Some("by_name".to_string()),
                        path: "name".to_string(),
                        index_type: DocIndexType::Btree as i32,
                        unique: false,
                        sparse: true,
                    }],
                }],
            })
        }

        async fn delete_collection(
            &self,
            _request: DeleteDocumentCollectionRequest,
        ) -> Result<DeleteDocumentCollectionResponse> {
            Ok(DeleteDocumentCollectionResponse { success: true })
        }

        async fn insert_document(
            &self,
            request: InsertDocumentRequest,
        ) -> Result<InsertDocumentResponse> {
            Ok(InsertDocumentResponse {
                id: request.id.unwrap_or_else(|| "generated".to_string()),
                version: 2,
            })
        }

        async fn get_document(&self, _request: GetDocumentRequest) -> Result<GetDocumentResponse> {
            Ok(GetDocumentResponse {
                document: Some(sample_object()),
                version: 3,
                found: true,
            })
        }

        async fn update_document(
            &self,
            _request: UpdateDocumentRequest,
        ) -> Result<UpdateDocumentResponse> {
            Ok(UpdateDocumentResponse {
                new_version: 4,
                success: true,
            })
        }

        async fn delete_document(
            &self,
            _request: DeleteDocumentRequest,
        ) -> Result<DeleteDocumentResponse> {
            Ok(DeleteDocumentResponse { deleted: true })
        }

        async fn query_documents(
            &self,
            _request: QueryDocumentsRequest,
        ) -> Result<QueryDocumentsResponse> {
            Ok(QueryDocumentsResponse {
                documents: vec![DocumentResult {
                    id: "doc-1".to_string(),
                    document: Some(sample_object()),
                    version: 5,
                    score: Some(0.9),
                }],
                total_count: Some(1),
                query_time_ms: 7,
            })
        }

        async fn aggregate_documents(
            &self,
            _request: AggregateDocumentsRequest,
        ) -> Result<AggregateDocumentsResponse> {
            Ok(AggregateDocumentsResponse {
                results: vec![sample_object()],
                query_time_ms: 8,
            })
        }
    }

    fn state() -> State<DocumentRestState> {
        State(DocumentRestState {
            document_port: Arc::new(MockDocumentPort),
        })
    }

    #[test]
    fn test_json_to_sql_value() {
        let json = serde_json::json!({"name": "test", "count": 42, "active": true});
        let obj = json_to_sql_object(&json).unwrap();
        assert!(obj.fields.contains_key("name"));
        assert!(obj.fields.contains_key("count"));
        assert!(obj.fields.contains_key("active"));
    }

    #[test]
    fn test_sql_object_roundtrip() {
        let original = serde_json::json!({"x": 1, "y": "hello", "z": true});
        let sql_obj = json_to_sql_object(&original).unwrap();
        let roundtripped = sql_object_to_json(&sql_obj);
        assert_eq!(original["x"], roundtripped["x"]);
        assert_eq!(original["y"], roundtripped["y"]);
        assert_eq!(original["z"], roundtripped["z"]);
    }

    #[test]
    fn json_sql_value_conversions_cover_nested_and_binary_values() {
        let value = serde_json::json!({
            "null": null,
            "array": [1, "two", false],
            "object": {"nested": true}
        });
        let object = json_to_sql_object(&value).unwrap();
        let restored = sql_object_to_json(&object);
        assert_eq!(restored["null"], serde_json::Value::Null);
        assert_eq!(restored["array"][1], "two");
        assert_eq!(restored["object"]["nested"], true);

        let bytes = SqlValue {
            value: Some(proximadb_proto::v1::sql_value::Value::BytesValue(vec![
                0xab, 0xcd,
            ])),
        };
        // Canonical rendering — bytes are base64 (was bare hex).
        assert_eq!(sql_value_to_json(&bytes), serde_json::json!("q80="));
        assert!(matches!(
            json_to_sql_object(&serde_json::json!(["not", "object"])),
            Err(RestError::InvalidArgument(_))
        ));
    }

    #[test]
    fn aggregation_stage_parser_accepts_supported_stage_aliases() {
        assert!(matches!(
            json_to_aggregation_stage(&serde_json::json!({"$limit": 10}))
                .unwrap()
                .stage,
            Some(AggStage::Limit(LimitStage { limit: 10 }))
        ));
        assert!(matches!(
            json_to_aggregation_stage(&serde_json::json!({"skip": 3}))
                .unwrap()
                .stage,
            Some(AggStage::Skip(SkipStage { skip: 3 }))
        ));
        assert!(matches!(
            json_to_aggregation_stage(&serde_json::json!({"$unwind": "tags"}))
                .unwrap()
                .stage,
            Some(AggStage::Unwind(UnwindStage { ref path, preserve_null: false })) if path == "tags"
        ));
        assert!(matches!(
            json_to_aggregation_stage(&serde_json::json!({
                "$lookup": {
                    "from_collection": "other",
                    "local_field": "id",
                    "foreign_field": "doc_id",
                    "as_field": "matches"
                }
            }))
            .unwrap()
            .stage,
            Some(AggStage::Lookup(LookupStage { ref from_collection, .. })) if from_collection == "other"
        ));
        assert!(matches!(
            json_to_aggregation_stage(&serde_json::json!("bad")),
            Err(RestError::InvalidArgument(_))
        ));
        assert!(matches!(
            json_to_aggregation_stage(&serde_json::json!({"unknown": {}})),
            Err(RestError::InvalidArgument(_))
        ));
    }

    #[tokio::test]
    async fn document_collection_handlers_route_through_document_port() {
        let JsonResponse(created) = create_collection(
            state(),
            Json(CreateDocumentCollectionRequestBody {
                name: "docs".to_string(),
                indexes: vec![
                    IndexDefinitionRequest {
                        name: Some("hash".to_string()),
                        path: "id".to_string(),
                        index_type: "hash".to_string(),
                        unique: true,
                        sparse: false,
                    },
                    IndexDefinitionRequest {
                        name: Some("unknown".to_string()),
                        path: "fallback".to_string(),
                        index_type: "unknown".to_string(),
                        unique: false,
                        sparse: false,
                    },
                ],
            }),
        )
        .await
        .unwrap();
        assert_eq!(created["collection"], "docs");

        let JsonResponse(listed) = list_collections(state()).await.unwrap();
        assert_eq!(listed["collections"][0]["name"], "docs");

        let JsonResponse(info) = get_collection_info(state(), Path("docs".to_string()))
            .await
            .unwrap();
        assert_eq!(info["document_count"], 2);

        let JsonResponse(indexes) = list_indexes(state(), Path("docs".to_string()))
            .await
            .unwrap();
        assert_eq!(indexes["indexes"][0]["path"], "name");

        let JsonResponse(deleted) = delete_collection(state(), Path("docs".to_string()))
            .await
            .unwrap();
        assert_eq!(deleted["success"], true);

        let err = create_index(
            state(),
            Path("docs".to_string()),
            Json(IndexDefinitionRequest {
                name: None,
                path: "field".to_string(),
                index_type: "btree".to_string(),
                unique: false,
                sparse: false,
            }),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, RestError::InvalidArgument(_)));
    }

    #[tokio::test]
    async fn document_crud_query_batch_and_aggregate_handlers_return_expected_shapes() {
        let JsonResponse(inserted) = insert_document(
            state(),
            Path("docs".to_string()),
            Json(CreateDocumentRequest {
                id: Some("doc-1".to_string()),
                document: serde_json::json!({"name": "doc"}),
            }),
        )
        .await
        .unwrap();
        assert_eq!(inserted.id, "doc-1");
        assert_eq!(inserted.version, 2);

        let JsonResponse(got) = get_document(
            state(),
            Path(("docs".to_string(), "doc-1".to_string())),
            Query(DocumentQueryParams {
                projection: Some("name,count".to_string()),
                limit: 100,
                filter: None,
            }),
        )
        .await
        .unwrap();
        assert_eq!(got.version, 3);
        assert_eq!(got.document["name"], "doc");

        let JsonResponse(query) = query_documents(
            state(),
            Path("docs".to_string()),
            Query(DocumentQueryParams {
                projection: Some("name".to_string()),
                limit: 1,
                filter: None,
            }),
        )
        .await
        .unwrap();
        assert!(query.has_more);
        assert_eq!(query.total_count, Some(1));

        let JsonResponse(updated) = update_document(
            state(),
            Path(("docs".to_string(), "doc-1".to_string())),
            Json(UpdateDocumentBody {
                updates: vec![
                    serde_json::json!({"operation": "set", "path": "name", "value": "new"}),
                    serde_json::json!({"operation": "unset", "path": "old"}),
                    serde_json::json!({"operation": "inc", "path": "count", "value": 1}),
                    serde_json::json!({"operation": "push", "path": "tags", "value": "a"}),
                    serde_json::json!({"operation": "pull", "path": "tags", "value": "b"}),
                    serde_json::json!({"operation": "unknown", "path": "fallback"}),
                ],
                expected_version: Some(3),
            }),
        )
        .await
        .unwrap();
        assert_eq!(updated["new_version"], 4);

        let JsonResponse(deleted) =
            delete_document(state(), Path(("docs".to_string(), "doc-1".to_string())))
                .await
                .unwrap();
        assert_eq!(deleted["id"], "doc-1");

        let JsonResponse(batch) = batch_insert_documents(
            state(),
            Path("docs".to_string()),
            Json(BatchInsertRequest {
                documents: vec![
                    CreateDocumentRequest {
                        id: Some("doc-1".to_string()),
                        document: serde_json::json!({"ok": true}),
                    },
                    CreateDocumentRequest {
                        id: Some("bad".to_string()),
                        document: serde_json::json!(["not", "object"]),
                    },
                ],
            }),
        )
        .await
        .unwrap();
        assert_eq!(batch.inserted, 1);
        assert_eq!(batch.failed, 1);

        let JsonResponse(aggregate) = aggregate_documents(
            state(),
            Path("docs".to_string()),
            Json(AggregateRequest {
                pipeline: vec![
                    serde_json::json!({"$limit": 10}),
                    serde_json::json!({"unknown": {}}),
                ],
            }),
        )
        .await
        .unwrap();
        assert_eq!(aggregate.results[0]["name"], "doc");

        let _router = create_document_router();
        let _handler = DocumentHandler;
        let _query_handler = DocumentQueryHandler::new();
        assert_eq!(default_limit(), 100);
        assert_eq!(default_index_type(), "btree");
    }
}
