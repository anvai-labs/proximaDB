//! Analytics REST endpoints (TD-043 sub-2).
//!
//! Currently exposes:
//!
//! - `POST /api/v1/analytics/entanglement` — compute the
//!   [Entanglement Index](crate::analytics::entanglement) over a
//!   caller-supplied set of `(chunk_id, topic, embedding)` triples.
//!
//! - `GET /api/v1/collections/{id}/entanglement?topic_field=…` — compute
//!   the EI for an existing collection by loading its records.

use axum::{
    Router,
    extract::{Path, Query, State},
    response::Json,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::debug;

use crate::analytics::entanglement::{self, ChunkEmbedding, EntanglementError, EntanglementReport};
use crate::errors::{ApiError, ApiResult};
use crate::services::VectorOperationsService;

/// State for the analytics router.
#[derive(Clone)]
pub struct AnalyticsApiState {
    /// Service for vector operations. Optional to support unit testing
    /// of stateless endpoints without full service instantiation.
    pub vector_ops: Option<Arc<VectorOperationsService>>,
}

impl AnalyticsApiState {
    /// Create a new analytics API state.
    pub fn new(vector_ops: Option<Arc<VectorOperationsService>>) -> Self {
        Self { vector_ops }
    }
}

/// Wire the analytics endpoints under a parent router.
pub fn create_router() -> Router<AnalyticsApiState> {
    Router::new()
        .route("/entanglement", post(compute_entanglement))
        .route(
            "/collections/{collection_id}/entanglement",
            get(get_collection_entanglement),
        )
}

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

/// One chunk in the request. Mirrors
/// [`crate::analytics::entanglement::ChunkEmbedding`] but uses owned strings
/// for clean JSON deserialization.
#[derive(Debug, Deserialize)]
pub struct ChunkInput {
    /// Stable chunk identifier (returned in error messages on validation
    /// failure; not used by the EI computation itself).
    pub chunk_id: String,
    /// Topic label. Two chunks share a topic iff this field is equal.
    pub topic: String,
    /// Embedding vector. All chunks must share the same length.
    pub embedding: Vec<f32>,
}

/// Request body for `POST /api/v1/analytics/entanglement`.
#[derive(Debug, Deserialize)]
pub struct EntanglementRequest {
    /// Chunks to analyze. Empty input returns `EI = 0` with zero counts.
    pub chunks: Vec<ChunkInput>,
}

/// Query parameters for collection-aware EI.
#[derive(Debug, Deserialize)]
pub struct CollectionEiParams {
    /// Field in metadata to use as the topic label.
    pub topic_field: String,
    /// Maximum number of records to analyze (default: 1000).
    pub limit: Option<usize>,
}

/// Response body. JSON-serializable mirror of [`EntanglementReport`].
#[derive(Debug, Serialize)]
pub struct EntanglementResponse {
    /// Mean entanglement across analyzed chunks, in `[0.0, 1.0]`.
    pub overall_ei: f64,
    /// Per-topic mean entanglement.
    pub per_topic_ei: HashMap<String, f64>,
    /// Number of chunks that contributed an `entangled(x)` measurement.
    pub chunks_analyzed: usize,
    /// Number of distinct topics with at least one analyzed chunk.
    pub topics_analyzed: usize,
    /// Chunks skipped because they were the only member of their topic.
    pub skipped_singletons: usize,
}

impl From<EntanglementReport> for EntanglementResponse {
    fn from(r: EntanglementReport) -> Self {
        Self {
            overall_ei: r.overall_ei,
            per_topic_ei: r.per_topic_ei,
            chunks_analyzed: r.chunks_analyzed,
            topics_analyzed: r.topics_analyzed,
            skipped_singletons: r.skipped_singletons,
        }
    }
}

impl From<ChunkInput> for ChunkEmbedding {
    fn from(c: ChunkInput) -> Self {
        Self {
            chunk_id: c.chunk_id,
            topic: c.topic,
            embedding: c.embedding,
        }
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// Map an EI library error to a structured 400 response.
fn entanglement_error_to_api(err: EntanglementError) -> ApiError {
    ApiError::InvalidArgument(err.to_string())
}

/// Compute the Entanglement Index over the supplied chunks.
async fn compute_entanglement(
    State(_): State<AnalyticsApiState>,
    Json(request): Json<EntanglementRequest>,
) -> ApiResult<Json<EntanglementResponse>> {
    debug!("EI request received with {} chunks", request.chunks.len());

    let chunks: Vec<ChunkEmbedding> = request
        .chunks
        .into_iter()
        .map(ChunkEmbedding::from)
        .collect();

    let report = entanglement::entanglement_index(&chunks).map_err(entanglement_error_to_api)?;

    Ok(Json(EntanglementResponse::from(report)))
}

/// Compute EI for a collection by loading its records.
async fn get_collection_entanglement(
    State(state): State<AnalyticsApiState>,
    tenant: Option<axum::Extension<crate::network::middleware::tenant::TenantContext>>,
    Path(collection_id): Path<String>,
    Query(params): Query<CollectionEiParams>,
) -> ApiResult<Json<EntanglementResponse>> {
    let vector_ops = state.vector_ops.ok_or_else(|| {
        ApiError::Internal("VectorOperationsService not available in this context".to_string())
    })?;

    let limit = params.limit.unwrap_or(1000);
    // Shared server top_k cap (TD-SECSCAN-1 §5).
    proximadb_search_types::top_k_limit::check_top_k(limit as u64)
        .map_err(|e| ApiError::InvalidArgument(format!("limit: {e}")))?;

    // Load records from the collection, resolved within the request tenant
    // (absent ⇒ the default tenant).
    let request_tenant = tenant
        .map(|axum::Extension(t)| t.tenant_id)
        .unwrap_or_else(|| proximadb_tenant::DEFAULT_TENANT.to_string());
    let tenant_context =
        crate::storage::tenant::context::TenantContext::for_tenant_id(&request_tenant);
    let records = vector_ops
        .unified_search_native_with_tenant_context(
            &collection_id,
            vec![],
            limit,
            None,
            None,
            Some(&tenant_context),
            #[cfg(feature = "abac-policy")]
            &proximadb_abac::ReadContext::system(
                proximadb_abac::SystemReadReason::Statistics,
                "rest::analytics [CLIENT-PLACEHOLDER]",
            ),
        )
        .await
        .map_err(
            |e| match crate::services::operations::vectors::collection_not_found(&e) {
                Some(_) => ApiError::CollectionNotFound(collection_id.clone()),
                None => ApiError::Internal(e.to_string()),
            },
        )?;

    let chunks: Vec<ChunkEmbedding> = records
        .into_iter()
        .filter_map(|r| {
            let topic = match r.metadata.get(&params.topic_field)? {
                proximadb_data_model::ProximaValue::String(s)
                | proximadb_data_model::ProximaValue::Symbol(s) => s.clone(),
                _ => return None,
            };
            let embedding = r.vector.as_ref().map(|arc| (**arc).clone())?;
            Some(ChunkEmbedding {
                chunk_id: r.id,
                topic,
                embedding,
            })
        })
        .collect();

    if chunks.is_empty() {
        return Err(ApiError::InvalidArgument(format!(
            "No records in collection '{}' have a string field '{}'",
            collection_id, params.topic_field
        )));
    }

    let report = entanglement::entanglement_index(&chunks).map_err(entanglement_error_to_api)?;

    Ok(Json(EntanglementResponse::from(report)))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::body::to_bytes;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    fn router() -> Router {
        create_router().with_state(AnalyticsApiState::new(None))
    }

    async fn post_json(body: serde_json::Value) -> (StatusCode, serde_json::Value) {
        let app = router();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/entanglement")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        let status = response.status();
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = if bytes.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
        };
        (status, json)
    }

    #[tokio::test]
    async fn empty_chunks_returns_zero_ei() {
        let (status, body) = post_json(serde_json::json!({ "chunks": [] })).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["overall_ei"], 0.0);
    }
}
