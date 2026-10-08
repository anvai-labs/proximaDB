//! REST API v2 — time-series surface. Moved from the root v2 tree under ADR-094.

// Copyright (C) 2025 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! v2 REST time-series surface (TD-TS-1).
//!
//! Matches the contract the Python SDK already expects
//! (`clients/python/src/proximadb_sdk/adapters/rest_adapter.py`):
//! `POST /api/v2/timeseries/collections` (create), `.../{c}/ingest`, `.../{c}/query`,
//! `.../{c}/aggregate`, `GET /api/v2/timeseries/collections` (list),
//! `DELETE .../{c}` (delete). Backed by the process-global timeseries service (via the ADR-094 port) over
//! the native TST engine — never the stubbed vector-shaped trait methods. Tenant
//! isolation is structural: the request tenant selects a per-tenant engine in the
//! service (physically separate storage), and the collection name stays tenant-clean.

use axum::extract::Path;
use axum::{Extension, Json};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use proximadb_runtime::timeseries_port::{TimeseriesOpsPort, TsCollectionConfig, TsPoint, timeseries_port};

use crate::rest::errors::{RestError as ApiError, RestResult as ApiResult};
use crate::rest::TenantContext;

fn service() -> ApiResult<Arc<dyn TimeseriesOpsPort>> {
    timeseries_port().ok_or_else(|| {
        ApiError::Internal("timeseries service is not available".to_string())
    })
}

#[derive(Debug, Deserialize)]
pub struct IngestRequest {
    pub points: Vec<TsPoint>,
}

#[derive(Debug, Serialize)]
pub struct IngestResponse {
    pub ingested: usize,
}

#[derive(Debug, Deserialize)]
pub struct QueryRequest {
    pub start_time: i64,
    pub end_time: i64,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct QueryResponse {
    pub points: Vec<TsPoint>,
}

#[derive(Debug, Deserialize)]
pub struct AggregateRequest {
    pub start_time: i64,
    pub end_time: i64,
    #[serde(default = "default_aggregation")]
    pub aggregation: String,
    #[serde(default = "default_bucket_ms")]
    pub bucket_ms: i64,
}

fn default_aggregation() -> String {
    "avg".to_string()
}
fn default_bucket_ms() -> i64 {
    60_000
}

#[derive(Debug, Serialize)]
pub struct AggregateResponse {
    pub buckets: Vec<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct CreateResponse {
    pub name: String,
}

#[derive(Debug, Serialize)]
pub struct ListResponse {
    pub collections: Vec<TsCollectionConfig>,
}

#[derive(Debug, Serialize)]
pub struct DeleteResponse {
    pub success: bool,
}

/// `POST /api/v2/timeseries/collections`
pub async fn create_timeseries_collection(
    Extension(tenant): Extension<TenantContext>,
    Json(config): Json<TsCollectionConfig>,
) -> ApiResult<Json<CreateResponse>> {
    // Tenant-clean logical name: the tenant scopes the collection structurally in the service
    // (a per-tenant engine), so the name the caller sees carries no `{tenant}::` prefix.
    let name = config.name.clone();
    service()?
        .create_collection(&tenant.tenant_id, config)
        .await
        .map_err(|e| ApiError::Internal(format!("create timeseries collection: {e}")))?;
    Ok(Json(CreateResponse { name }))
}

/// `POST /api/v2/timeseries/collections/{collection_id}/ingest`
pub async fn ingest_timeseries(
    Path(collection_id): Path<String>,
    Extension(tenant): Extension<TenantContext>,
    Json(request): Json<IngestRequest>,
) -> ApiResult<Json<IngestResponse>> {
    let ingested = service()?
        .ingest(&tenant.tenant_id, &collection_id, request.points)
        .await
        .map_err(|e| ApiError::Internal(format!("ingest timeseries: {e}")))?;
    Ok(Json(IngestResponse { ingested }))
}

/// `POST /api/v2/timeseries/collections/{collection_id}/query`
pub async fn query_timeseries(
    Path(collection_id): Path<String>,
    Extension(tenant): Extension<TenantContext>,
    Json(request): Json<QueryRequest>,
) -> ApiResult<Json<QueryResponse>> {
    let points = service()?
        .query(
            &tenant.tenant_id,
            &collection_id,
            request.start_time,
            request.end_time,
            request.limit,
        )
        .await
        .map_err(|e| ApiError::Internal(format!("query timeseries: {e}")))?;
    Ok(Json(QueryResponse { points }))
}

/// `POST /api/v2/timeseries/collections/{collection_id}/aggregate`
pub async fn aggregate_timeseries(
    Path(collection_id): Path<String>,
    Extension(tenant): Extension<TenantContext>,
    Json(request): Json<AggregateRequest>,
) -> ApiResult<Json<AggregateResponse>> {
    let buckets = service()?
        .aggregate(
            &tenant.tenant_id,
            &collection_id,
            request.start_time,
            request.end_time,
            &request.aggregation,
            request.bucket_ms,
        )
        .await
        .map_err(|e| ApiError::Internal(format!("aggregate timeseries: {e}")))?;
    Ok(Json(AggregateResponse { buckets }))
}

/// `GET /api/v2/timeseries/collections`
pub async fn list_timeseries_collections(
    Extension(tenant): Extension<TenantContext>,
) -> ApiResult<Json<ListResponse>> {
    // Tenant-scoped: list only this tenant's collections (was a cross-tenant leak).
    let collections = service()?.list_collections(&tenant.tenant_id).await;
    Ok(Json(ListResponse { collections }))
}

/// `DELETE /api/v2/timeseries/collections/{collection_id}`
pub async fn delete_timeseries_collection(
    Path(collection_id): Path<String>,
    Extension(tenant): Extension<TenantContext>,
) -> ApiResult<Json<DeleteResponse>> {
    let success = service()?
        .delete_collection(&tenant.tenant_id, &collection_id)
        .await;
    Ok(Json(DeleteResponse { success }))
}
