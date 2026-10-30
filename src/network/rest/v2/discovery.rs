/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! REST v2 Continuous Discovery endpoints (Phase 8 F1).
//!
//! - `POST /api/v2/collections/:collection_id/discovery-jobs` — create+schedule a job
//! - `GET  /api/v2/collections/:collection_id/discovery-jobs` — list jobs (newest first)
//! - `GET  /api/v2/collections/:collection_id/discovery-jobs/:job_id` — get one job
//!
//! Experimental, v2-only. Handlers reach the `DiscoveryService` (and the
//! background executor's registry) via `AppState`. The executor pins a
//! snapshot, runs the refinement pass, and atomically republishes — see
//! `src/services/discovery/`.

use axum::{
    Json,
    extract::{Path, State},
};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::errors::{ApiError, ApiResult};
use crate::network::rest::canonical::handlers::AppState;
use crate::services::discovery::{DiscoveryJob, DiscoveryJobKind, DiscoveryService};

/// Request body for creating a discovery job. `kind` defaults to `dedup`.
#[derive(Debug, Deserialize)]
pub struct CreateDiscoveryJobRequest {
    #[serde(default = "default_kind")]
    pub kind: DiscoveryJobKind,
}

fn default_kind() -> DiscoveryJobKind {
    DiscoveryJobKind::Dedup
}

/// Single-job response.
#[derive(Debug, Serialize)]
pub struct DiscoveryJobResponse {
    pub job: DiscoveryJob,
}

/// Job-list response for a collection.
#[derive(Debug, Serialize)]
pub struct DiscoveryJobListResponse {
    pub collection_id: String,
    pub jobs: Vec<DiscoveryJob>,
}

/// Resolve the job's collection within the request tenant: another tenant's
/// collection is answered exactly like a missing one, so no job is created,
/// listed or read for it.
async fn require_collection(
    state: &AppState,
    tenant: &crate::network::middleware::tenant::TenantContext,
    collection_id: &str,
) -> ApiResult<()> {
    match state
        .collection_service
        .collection_for_tenant(collection_id, Some(&tenant.tenant_id))
        .await
    {
        Ok(Some(_)) => Ok(()),
        Ok(None) => Err(ApiError::CollectionNotFound(collection_id.to_string())),
        Err(e) => Err(ApiError::Internal(format!("collection lookup failed: {e}"))),
    }
}

fn service(state: &AppState) -> ApiResult<&std::sync::Arc<DiscoveryService>> {
    state.discovery_service.as_ref().ok_or_else(|| {
        ApiError::NotImplemented("Continuous Discovery service is not enabled".to_string())
    })
}

/// `POST /api/v2/collections/:collection_id/discovery-jobs`
pub async fn create_discovery_job_v2(
    Path(collection_id): Path<String>,
    State(state): State<AppState>,
    axum::Extension(tenant): axum::Extension<crate::network::middleware::tenant::TenantContext>,
    Json(req): Json<CreateDiscoveryJobRequest>,
) -> ApiResult<Json<DiscoveryJobResponse>> {
    if collection_id.is_empty() {
        return Err(ApiError::InvalidArgument(
            "Collection ID is required".to_string(),
        ));
    }
    require_collection(&state, &tenant, &collection_id).await?;
    let job = service(&state)?.create_job(collection_id.clone(), req.kind);
    info!(
        "V2 API: scheduled discovery job {} ({:?}) for collection '{}'",
        job.job_id, req.kind, collection_id
    );
    Ok(Json(DiscoveryJobResponse { job }))
}

/// `GET /api/v2/collections/:collection_id/discovery-jobs`
pub async fn list_discovery_jobs_v2(
    Path(collection_id): Path<String>,
    State(state): State<AppState>,
    axum::Extension(tenant): axum::Extension<crate::network::middleware::tenant::TenantContext>,
) -> ApiResult<Json<DiscoveryJobListResponse>> {
    require_collection(&state, &tenant, &collection_id).await?;
    let jobs = service(&state)?.list_jobs(&collection_id);
    Ok(Json(DiscoveryJobListResponse {
        collection_id,
        jobs,
    }))
}

/// `GET /api/v2/collections/:collection_id/discovery-jobs/:job_id`
pub async fn get_discovery_job_v2(
    Path((collection_id, job_id)): Path<(String, String)>,
    State(state): State<AppState>,
    axum::Extension(tenant): axum::Extension<crate::network::middleware::tenant::TenantContext>,
) -> ApiResult<Json<DiscoveryJobResponse>> {
    require_collection(&state, &tenant, &collection_id).await?;
    let job = service(&state)?
        .get_job(&job_id)
        .filter(|j| j.collection_id == collection_id)
        .ok_or_else(|| {
            ApiError::NotFound(format!(
                "discovery job '{job_id}' not found in collection '{collection_id}'"
            ))
        })?;
    Ok(Json(DiscoveryJobResponse { job }))
}
