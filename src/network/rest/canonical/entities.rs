/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! REST API handlers for Entity operations in SKS

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Json},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, error, info, warn};

use crate::proto::proximadb_v1::{Entity, EntityResult, MetadataFilter};
use crate::storage::entity_store::{EntityStore, ProximaEntityStore};

/// REST API state containing the entity store
#[derive(Clone)]
pub struct EntityApiState {
    /// Shared entity store for SKS operations
    pub store: Arc<ProximaEntityStore>,
}

/// Request body for entity upsert
#[derive(Debug, serde::Deserialize)]
pub struct UpsertEntityRequest {
    /// Entity data to upsert
    pub entity: Entity,
    /// Automatically create the collection if it does not exist
    pub create_collection_if_missing: Option<bool>,
}

/// Response for entity upsert
#[derive(Debug, Serialize)]
pub struct UpsertEntityResponse {
    /// Whether the upsert succeeded
    pub success: bool,
    /// ID of the upserted entity
    pub entity_id: String,
    /// Status message
    pub message: String,
}

/// Query parameters for entity retrieval
#[derive(Debug, Deserialize)]
pub struct GetEntityQuery {
    /// Include vector embeddings in the response
    pub include_embeddings: Option<bool>,
    /// Include entity relationships in the response
    pub include_relations: Option<bool>,
}

/// Request body for entity search
#[derive(Debug, Deserialize)]
pub struct SearchEntitiesRequest {
    /// Query vector for similarity search
    pub query_vector: Option<Vec<f32>>,
    /// Text query for semantic or keyword search
    pub query_text: Option<String>,
    /// Metadata filters (converted to MetadataFilter internally)
    pub filters: Option<serde_json::Value>,
    /// Maximum number of results to return
    pub top_k: Option<usize>,
    /// Enable progressive multi-stage search
    pub progressive: Option<bool>,
}

/// Response for entity search
#[derive(Debug, Serialize)]
pub struct SearchEntitiesResponse {
    /// Matching entity results
    pub results: Vec<EntityResult>,
    /// Total number of matches found
    pub total: u32,
}

/// Error response
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    /// Error message
    pub error: String,
    /// Additional error details
    pub details: Option<String>,
}

/// Upsert an entity
pub async fn upsert_entity(
    Path(collection_id): Path<String>,
    State(state): State<EntityApiState>,
    Json(request): Json<UpsertEntityRequest>,
) -> impl IntoResponse {
    info!("REST: Upserting entity in collection: {}", collection_id);

    // Validate collection_id
    if collection_id.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Invalid collection_id".to_string(),
                details: Some("collection_id cannot be empty".to_string()),
            }),
        ));
    }

    // Validate entity has at least one embedding
    if request.entity.embeddings.is_empty() {
        warn!("Entity has no embeddings, this may affect search capabilities");
    }

    // Store entity
    match state
        .store
        .upsert_entity(&collection_id, request.entity)
        .await
    {
        Ok(entity_id) => {
            info!("Successfully upserted entity: {}", entity_id);
            Ok(Json(UpsertEntityResponse {
                success: true,
                entity_id: entity_id.clone(),
                message: format!("Entity {} upserted successfully", entity_id),
            }))
        }
        Err(e) => {
            error!("Failed to upsert entity: {}", e);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Failed to upsert entity".to_string(),
                    details: Some(e.to_string()),
                }),
            ))
        }
    }
}

/// Get an entity by ID
pub async fn get_entity(
    Path((collection_id, entity_id)): Path<(String, String)>,
    Query(params): Query<GetEntityQuery>,
    State(state): State<EntityApiState>,
) -> impl IntoResponse {
    debug!(
        "REST: Getting entity {} from collection {}",
        entity_id, collection_id
    );

    // Validate parameters
    if collection_id.is_empty() || entity_id.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Invalid parameters".to_string(),
                details: Some("collection_id and entity_id are required".to_string()),
            }),
        ));
    }

    let include_embeddings = params.include_embeddings.unwrap_or(false);
    let include_relations = params.include_relations.unwrap_or(false);

    // Retrieve entity
    match state
        .store
        .get_entity(
            &collection_id,
            &entity_id,
            include_embeddings,
            include_relations,
        )
        .await
    {
        Ok(Some(entity)) => {
            debug!("Found entity: {}", entity_id);
            Ok(Json(entity))
        }
        Ok(None) => {
            debug!("Entity not found: {}", entity_id);
            Err((
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: "Entity not found".to_string(),
                    details: Some(format!(
                        "Entity {} not found in collection {}",
                        entity_id, collection_id
                    )),
                }),
            ))
        }
        Err(e) => {
            error!("Failed to get entity: {}", e);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Failed to get entity".to_string(),
                    details: Some(e.to_string()),
                }),
            ))
        }
    }
}

/// Delete an entity
pub async fn delete_entity(
    Path((collection_id, entity_id)): Path<(String, String)>,
    Query(params): Query<DeleteEntityQuery>,
    State(state): State<EntityApiState>,
) -> impl IntoResponse {
    info!(
        "REST: Deleting entity {} from collection {} (hard_delete: {})",
        entity_id,
        collection_id,
        params.hard_delete.unwrap_or(false)
    );

    // Validate parameters
    if collection_id.is_empty() || entity_id.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Invalid parameters".to_string(),
                details: Some("collection_id and entity_id are required".to_string()),
            }),
        ));
    }

    let hard_delete = params.hard_delete.unwrap_or(false);

    // Delete entity
    match state
        .store
        .delete_entity(&collection_id, &entity_id, hard_delete)
        .await
    {
        Ok(success) => {
            if success {
                info!("Successfully deleted entity: {}", entity_id);
                Ok(StatusCode::NO_CONTENT)
            } else {
                warn!("Entity not found for deletion: {}", entity_id);
                Err((
                    StatusCode::NOT_FOUND,
                    Json(ErrorResponse {
                        error: "Entity not found".to_string(),
                        details: Some(format!("Entity {} not found", entity_id)),
                    }),
                ))
            }
        }
        Err(e) => {
            error!("Failed to delete entity: {}", e);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Failed to delete entity".to_string(),
                    details: Some(e.to_string()),
                }),
            ))
        }
    }
}

/// Query parameters for entity deletion
#[derive(Debug, Deserialize)]
pub struct DeleteEntityQuery {
    /// Permanently delete instead of soft-delete
    pub hard_delete: Option<bool>,
}

/// Search for entities
pub async fn search_entities(
    Path(collection_id): Path<String>,
    State(state): State<EntityApiState>,
    tenant: Option<axum::Extension<crate::network::middleware::tenant::TenantContext>>,
    Json(request): Json<SearchEntitiesRequest>,
) -> impl IntoResponse {
    // The backing collection is resolved within the request tenant (absent ⇒
    // the default tenant).
    let request_tenant = tenant
        .map(|axum::Extension(t)| t.tenant_id)
        .unwrap_or_else(|| proximadb_tenant::DEFAULT_TENANT.to_string());
    let top_k = request.top_k.unwrap_or(10);

    info!(
        "REST: Searching entities in collection {} (top_k: {})",
        collection_id, top_k
    );

    // Validate parameters
    if collection_id.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Invalid collection_id".to_string(),
                details: Some("collection_id cannot be empty".to_string()),
            }),
        ));
    }

    // top_k=0 stays rejected here (this surface's existing contract, same as
    // REST v2 record search); the UPPER bound is the shared server cap so
    // PROXIMADB_SEARCH_MAX_TOP_K governs it like every other search surface
    // (TD-SECSCAN-1 §5).
    if top_k == 0 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Invalid top_k".to_string(),
                details: Some("top_k must be greater than 0".to_string()),
            }),
        ));
    }
    if let Err(e) = proximadb_search_types::top_k_limit::check_top_k(top_k as u64) {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Invalid top_k".to_string(),
                details: Some(e.to_string()),
            }),
        ));
    }

    // 2026-05-26: text_query field used to silently fall back to query_vector
    // when set; that misled callers into thinking text-to-embedding was happening.
    // The honest contract for v0.3 is: this endpoint accepts pre-computed vectors
    // only. Reject text_query with 400 until an embedding service is wired in.
    if request.query_text.is_some() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "text_query embedding not configured".to_string(),
                details: Some(
                    "This server has no text-to-embedding service wired into the entity search path. \
                     Submit `query_vector` directly (precomputed embedding), or use a future build \
                     where LLMIntegrationEngine is wired. The previous silent fallback to \
                     `query_vector` was misleading and has been removed."
                        .to_string(),
                ),
            }),
        ));
    }
    let query_vector = request.query_vector;

    // Convert filters from JSON Value to MetadataFilter if present
    let metadata_filter = match request.filters {
        Some(filter_json) => {
            // Try to convert JSON Value to MetadataFilter
            match serde_json::from_value::<MetadataFilter>(filter_json) {
                Ok(filter) => Some(filter),
                Err(_) => {
                    warn!("Failed to parse filters, ignoring");
                    None
                }
            }
        }
        None => None,
    };

    // Perform search. (Note: the entity_store trait has no `temporal_filter`
    // parameter — earlier code had a stale `// Deferred: temporal_filter`
    // comment here that suggested the API silently dropped a temporal filter.
    // It did not; the field was never accepted on the REST surface.)
    match state
        .store
        .search_entities_for_tenant(
            &collection_id,
            Some(&request_tenant),
            query_vector,
            metadata_filter,
            top_k,
        )
        .await
    {
        Ok(results) => {
            let entity_results: Vec<EntityResult> = results
                .into_iter()
                .map(|(entity, score)| EntityResult {
                    entity: Some(entity),
                    score,
                    debug_info: Default::default(),
                })
                .collect();

            let total = entity_results.len() as u32;

            info!("Search returned {} results", total);

            Ok(Json(SearchEntitiesResponse {
                results: entity_results,
                total,
            }))
        }
        Err(e) => {
            if let Some(missing) = crate::services::operations::vectors::collection_not_found(&e) {
                return Err((
                    StatusCode::NOT_FOUND,
                    Json(ErrorResponse {
                        error: "Collection not found".to_string(),
                        details: Some(missing.to_string()),
                    }),
                ));
            }
            // Wrong-dimension / empty / non-finite query vector or over-cap
            // top_k: the client's argument (TD-SECSCAN-1 §5), not a server fault.
            if let Some(message) = crate::services::operations::vectors::client_argument_message(&e)
            {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: "Invalid search argument".to_string(),
                        details: Some(message),
                    }),
                ));
            }
            error!("Failed to search entities: {}", e);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Failed to search entities".to_string(),
                    details: Some(e.to_string()),
                }),
            ))
        }
    }
}

/// List entities in a collection
pub async fn list_entities(
    Path(collection_id): Path<String>,
    Query(params): Query<ListEntitiesQuery>,
    State(state): State<EntityApiState>,
) -> impl IntoResponse {
    let offset = params.offset.unwrap_or(0);
    let limit = params.limit.unwrap_or(100).min(1000); // Cap at 1000

    debug!(
        "REST: Listing entities in collection {} (offset: {}, limit: {})",
        collection_id, offset, limit
    );

    // Validate parameters
    if collection_id.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Invalid collection_id".to_string(),
                details: Some("collection_id cannot be empty".to_string()),
            }),
        ));
    }

    // List entities
    match state
        .store
        .list_entities(&collection_id, offset, limit)
        .await
    {
        Ok(entities) => {
            debug!("Listed {} entities", entities.len());
            Ok(Json(entities))
        }
        Err(e) => {
            error!("Failed to list entities: {}", e);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Failed to list entities".to_string(),
                    details: Some(e.to_string()),
                }),
            ))
        }
    }
}

/// Query parameters for entity listing
#[derive(Debug, Deserialize)]
pub struct ListEntitiesQuery {
    /// Pagination offset
    pub offset: Option<usize>,
    /// Maximum number of entities to return
    pub limit: Option<usize>,
}

/// Configure REST API routes for entities
pub fn configure_routes() -> axum::Router<EntityApiState> {
    use axum::routing::{get, post};

    axum::Router::new()
        .route(
            "/v1/collections/{collection_id}/entities",
            post(upsert_entity).get(list_entities),
        )
        .route(
            "/v1/collections/{collection_id}/entities/{entity_id}",
            get(get_entity).delete(delete_entity),
        )
        .route(
            "/v1/collections/{collection_id}/entities/search",
            post(search_entities),
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deferred: Add integration tests for REST API handlers

    #[test]
    fn test_error_response_serialization() {
        let error = ErrorResponse {
            error: "Test error".to_string(),
            details: Some("Test details".to_string()),
        };

        let json = serde_json::to_string(&error).unwrap();
        assert!(json.contains("Test error"));
        assert!(json.contains("Test details"));
    }
}
