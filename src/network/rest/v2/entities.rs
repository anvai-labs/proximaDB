//! REST v2 entity endpoints — a thin JSON adapter over the shared
//! [`EntityOrchestrator`] (`src/services/entity_orchestrator.rs`).
//!
//! An "entity" is a graph node + optional embeddings + optional provenance +
//! optional relations; the orchestration (and fusion-delegating search) lives in
//! the orchestrator, shared with the gRPC facade. Per
//! `SEARCH_SURFACE_CONTRACT_2026_06_24.adoc`: retrieval delegates to the fusion
//! seam; this facade owns no ranking. Tenant isolation is structural — the request
//! tenant is threaded to the orchestrator, which scopes each leg (graph via
//! `for_tenant`, vector via `TenantContext`, document via the scoped collection key);
//! collection names stay tenant-clean (never a `{tenant}::name` fold).

use std::collections::HashMap;

use axum::{
    Extension, Json,
    extract::{Path, State},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use utoipa::ToSchema;

use crate::errors::{ApiError, ApiResult};
use crate::graph::{PropertyValue, property_value::Value as GraphValue};
use crate::network::middleware::tenant::TenantContext;
use crate::network::rest::canonical::handlers::AppState;
use crate::network::rest::openapi::ErrorResponse;
use crate::services::entity_orchestrator::{
    EntityEmbedding, EntityOrchestrator, EntityProvenance, EntityRelation, EntityUpsert,
};

// ---------------------------------------------------------------------------
// DTOs
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, ToSchema)]
pub struct EntityEmbeddingInput {
    pub model_id: String,
    #[serde(default)]
    pub modality: Option<String>,
    pub vector: Vec<f32>,
    #[serde(default)]
    pub dimension: u32,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct EntityProvenanceInput {
    #[serde(default)]
    pub source_id: String,
    #[serde(default)]
    pub chunk_id: String,
    #[serde(default)]
    pub chunk_position: u32,
    #[serde(default)]
    pub extraction_method: String,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct EntityRelationInput {
    pub source_entity_id: String,
    pub target_entity_id: String,
    pub relation_type: String,
    #[serde(default)]
    pub weight: f32,
    #[serde(default)]
    pub properties: HashMap<String, String>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct UpsertEntityRequest {
    /// Empty ⇒ the server generates a UUID.
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub flexible_metadata: HashMap<String, Value>,
    #[serde(default)]
    pub embeddings: Vec<EntityEmbeddingInput>,
    #[serde(default)]
    pub provenance: Option<EntityProvenanceInput>,
    #[serde(default)]
    pub relations: Vec<EntityRelationInput>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct UpsertEntityResponse {
    pub success: bool,
    pub entity_id: String,
    pub message: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct EntityDto {
    pub id: String,
    pub collection_id: String,
    pub flexible_metadata: HashMap<String, Value>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct SearchEntitiesRequest {
    /// Query embedding for vector similarity search. Omit for metadata-only search.
    #[serde(default)]
    pub query_vector: Vec<f32>,
    /// Equality metadata filters as a `{field: value}` JSON object.
    #[serde(default)]
    pub filters: HashMap<String, Value>,
    #[serde(default = "default_top_k")]
    pub top_k: u32,
}

fn default_top_k() -> u32 {
    10
}

#[derive(Debug, Serialize, ToSchema)]
pub struct EntitySearchResult {
    pub entity: EntityDto,
    pub score: f32,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct SearchEntitiesResponse {
    pub results: Vec<EntitySearchResult>,
    pub total: u32,
}

// ---------------------------------------------------------------------------
// JSON ↔ graph PropertyValue conversion (REST-specific)
// ---------------------------------------------------------------------------

fn json_to_property_value(v: &Value) -> Option<PropertyValue> {
    let value = match v {
        Value::String(s) => GraphValue::StringValue(s.clone()),
        Value::Bool(b) => GraphValue::BoolValue(*b),
        Value::Number(n) if n.is_i64() => GraphValue::IntValue(n.as_i64()?),
        Value::Number(n) if n.is_f64() => GraphValue::DoubleValue(n.as_f64()?),
        // Null / arrays / objects are not scalar node props; coerce to string.
        Value::Number(n) => GraphValue::StringValue(n.to_string()),
        other => GraphValue::StringValue(other.to_string()),
    };
    Some(PropertyValue { value: Some(value) })
}

fn property_value_to_json(pv: &PropertyValue) -> Value {
    match &pv.value {
        Some(GraphValue::StringValue(s)) => Value::String(s.clone()),
        Some(GraphValue::IntValue(i)) => Value::Number((*i).into()),
        Some(GraphValue::DoubleValue(f)) => serde_json::Number::from_f64(*f)
            .map(Value::Number)
            .unwrap_or_else(|| Value::Null),
        Some(GraphValue::BoolValue(b)) => Value::Bool(*b),
        Some(GraphValue::BytesValue(b)) => Value::String(format!("<{} bytes>", b.len())),
        _ => Value::Null,
    }
}

/// Build an orchestrator from the app state's backing services (cheap Arc clones).
fn orchestrator(state: &AppState) -> EntityOrchestrator {
    EntityOrchestrator::new(
        state.graph_operations_service.clone(),
        state.vector_operations_service.clone(),
        state.fusion_service.clone(),
        state.document_service.clone(),
    )
}

fn node_to_dto(node: &crate::graph::Node, collection: &str) -> EntityDto {
    let mut flexible_metadata = HashMap::new();
    for (k, v) in &node.properties {
        if !k.starts_with('_') {
            flexible_metadata.insert(k.clone(), property_value_to_json(v));
        }
    }
    let prefix = format!("entity:{collection}:");
    let id = node
        .id
        .strip_prefix(&prefix)
        .unwrap_or(&node.id)
        .to_string();
    EntityDto {
        id,
        collection_id: collection.to_string(),
        flexible_metadata,
    }
}

fn err_internal(operation: &str, e: impl std::fmt::Display) -> ApiError {
    let msg = e.to_string();
    let lower = msg.to_lowercase();
    if lower.contains("not found") {
        ApiError::NotFound(msg)
    } else if lower.contains("invalid") || lower.contains("required") || lower.contains("empty") {
        ApiError::InvalidArgument(msg)
    } else {
        ApiError::Internal(format!("{operation}: {msg}"))
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

#[utoipa::path(
    post,
    path = "/api/v2/collections/{collection_id}/entities",
    params(
        ("collection_id" = String, Path, description = "Collection backing the entities"),
    ),
    request_body = UpsertEntityRequest,
    responses(
        (status = StatusCode::OK, description = "Entity upserted", body = UpsertEntityResponse),
        (status = StatusCode::BAD_REQUEST, description = "Invalid request", body = ErrorResponse),
        (status = StatusCode::INTERNAL_SERVER_ERROR, description = "Upsert failed", body = ErrorResponse),
    ),
    tag = "Entities",
)]
pub async fn upsert_entity_v2(
    Path(collection_id): Path<String>,
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    Json(request): Json<UpsertEntityRequest>,
) -> ApiResult<Json<UpsertEntityResponse>> {
    // Tenant-CLEAN collection name; the orchestrator scopes each leg structurally by the tenant.
    let mut metadata = HashMap::new();
    for (k, v) in &request.flexible_metadata {
        if let Some(pv) = json_to_property_value(v) {
            metadata.insert(k.clone(), pv);
        }
    }
    let embeddings = request
        .embeddings
        .into_iter()
        .map(|e| EntityEmbedding {
            model_id: e.model_id,
            modality: e.modality.unwrap_or_else(|| "text".to_string()),
            vector: e.vector,
            dimension: e.dimension,
        })
        .collect::<Vec<_>>();
    let provenance = request.provenance.map(|p| EntityProvenance {
        source_id: p.source_id,
        chunk_id: p.chunk_id,
        chunk_position: p.chunk_position,
        extraction_method: p.extraction_method,
        metadata: p.metadata,
    });
    let relations = request
        .relations
        .into_iter()
        .map(|r| EntityRelation {
            source_entity_id: r.source_entity_id,
            target_entity_id: r.target_entity_id,
            relation_type: r.relation_type,
            weight: r.weight,
            properties: r.properties,
        })
        .collect::<Vec<_>>();

    let input = EntityUpsert {
        entity_id: request.id,
        metadata,
        embeddings,
        provenance,
        relations,
    };

    let entity_id = orchestrator(&state)
        .upsert(&collection_id, &tenant.tenant_id, input)
        .await
        .map_err(|e| err_internal("upsert entity", e))?;

    Ok(Json(UpsertEntityResponse {
        success: true,
        entity_id,
        message: "Entity upserted successfully".to_string(),
    }))
}

#[utoipa::path(
    get,
    path = "/api/v2/collections/{collection_id}/entities/{entity_id}",
    params(
        ("collection_id" = String, Path, description = "Collection backing the entities"),
        ("entity_id" = String, Path, description = "Entity ID"),
    ),
    responses(
        (status = StatusCode::OK, description = "Entity found", body = EntityDto),
        (status = StatusCode::NOT_FOUND, description = "Entity not found", body = ErrorResponse),
        (status = StatusCode::INTERNAL_SERVER_ERROR, description = "Fetch failed", body = ErrorResponse),
    ),
    tag = "Entities",
)]
pub async fn get_entity_v2(
    Path((collection_id, entity_id)): Path<(String, String)>,
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
) -> ApiResult<Json<EntityDto>> {
    let node = orchestrator(&state)
        .get(&collection_id, &tenant.tenant_id, &entity_id)
        .await
        .map_err(|e| err_internal("get entity", e))?
        .ok_or_else(|| ApiError::NotFound(format!("Entity '{entity_id}' not found")))?;
    Ok(Json(node_to_dto(&node, &collection_id)))
}

#[utoipa::path(
    delete,
    path = "/api/v2/collections/{collection_id}/entities/{entity_id}",
    params(
        ("collection_id" = String, Path, description = "Collection backing the entities"),
        ("entity_id" = String, Path, description = "Entity ID"),
    ),
    responses(
        (status = StatusCode::OK, description = "Entity deleted", body = UpsertEntityResponse),
        (status = StatusCode::NOT_FOUND, description = "Entity not found", body = ErrorResponse),
        (status = StatusCode::INTERNAL_SERVER_ERROR, description = "Delete failed", body = ErrorResponse),
    ),
    tag = "Entities",
)]
pub async fn delete_entity_v2(
    Path((collection_id, entity_id)): Path<(String, String)>,
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
) -> ApiResult<Json<UpsertEntityResponse>> {
    let deleted = orchestrator(&state)
        .delete(&collection_id, &tenant.tenant_id, &entity_id)
        .await
        .map_err(|e| err_internal("delete entity", e))?;
    Ok(Json(UpsertEntityResponse {
        success: deleted,
        entity_id,
        message: if deleted {
            "Entity deleted successfully".to_string()
        } else {
            "Entity not found".to_string()
        },
    }))
}

#[utoipa::path(
    post,
    path = "/api/v2/collections/{collection_id}/entities/search",
    params(
        ("collection_id" = String, Path, description = "Collection backing the entities"),
    ),
    request_body = SearchEntitiesRequest,
    responses(
        (status = StatusCode::OK, description = "Search results", body = SearchEntitiesResponse),
        (status = StatusCode::BAD_REQUEST, description = "Invalid request", body = ErrorResponse),
        (status = StatusCode::INTERNAL_SERVER_ERROR, description = "Search failed", body = ErrorResponse),
    ),
    tag = "Entities",
)]
pub async fn search_entities_v2(
    Path(collection_id): Path<String>,
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    Json(request): Json<SearchEntitiesRequest>,
) -> ApiResult<Json<SearchEntitiesResponse>> {
    // Shared server top_k cap (TD-SECSCAN-1 §5).
    proximadb_search_types::top_k_limit::check_top_k(u64::from(request.top_k))
        .map_err(|e| ApiError::InvalidArgument(e.to_string()))?;
    let query_vector = if request.query_vector.is_empty() {
        None
    } else {
        Some(request.query_vector)
    };
    // Equality filters: {field: value} → PropertyFilter (Equals = 1).
    let filters = request
        .filters
        .iter()
        .filter_map(|(k, v)| {
            json_to_property_value(v).map(|pv| crate::graph::PropertyFilter {
                key: k.clone(),
                operator: 1, // Equals
                value: Some(pv),
            })
        })
        .collect::<Vec<_>>();

    let hits = orchestrator(&state)
        .search(
            &collection_id,
            &tenant.tenant_id,
            query_vector,
            filters,
            request.top_k as usize,
        )
        .await
        .map_err(|e| err_internal("search entities", e))?;

    let results = hits
        .into_iter()
        .map(|hit| EntitySearchResult {
            entity: node_to_dto(&hit.node, &collection_id),
            score: hit.score,
        })
        .collect::<Vec<_>>();
    let total = results.len() as u32;

    Ok(Json(SearchEntitiesResponse { results, total }))
}
