//! REST API v2 — model-registry surface. Moved from the root v2 tree under ADR-094.

//! Native REST facade for the tenant-scoped xCatalog embedding-model registry.
//!
//! The handlers own HTTP extraction and error mapping only. Lifecycle rules,
//! optimistic concurrency, persistence, and alias resolution remain in the
//! shared [`CatalogModelRegistryService`].

use axum::{
    Json,
    extract::{Extension, Path, State},
};
use proximadb_catalog::mlops::{
    CatalogEmbeddingModelRegistry, CatalogEmbeddingModelVersion, CatalogModelContractError,
    CatalogModelRegistryMutation, CatalogModelUsePolicy,
};
use proximadb_catalog::model_registry_service::{
    CatalogModelRegistryRecord, ModelRegistryServiceError,
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::rest::errors::{RestError as ApiError, RestResult as ApiResult};
use crate::rest::TenantContext;
use crate::rest::state::RestAppState;

#[derive(Debug, Deserialize, ToSchema)]
pub struct CreateModelRegistryRequest {
    /// Tenant-local registry name. Must be one traversal-free catalog segment.
    pub name: String,
}

/// REST uses a decimal string for stable catalog IDs so JavaScript clients do
/// not lose precision. gRPC exposes the same value as `uint64`.
#[derive(Debug, Serialize, ToSchema)]
pub struct ModelRegistryRecordResponse {
    pub tenant_id: String,
    pub asset_id: String,
    pub registry: CatalogEmbeddingModelRegistry,
}

impl From<CatalogModelRegistryRecord> for ModelRegistryRecordResponse {
    fn from(value: CatalogModelRegistryRecord) -> Self {
        Self {
            tenant_id: value.tenant_id,
            asset_id: value.asset_id.to_string(),
            registry: value.registry,
        }
    }
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ListModelRegistriesResponse {
    pub registries: Vec<ModelRegistryRecordResponse>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct ApplyModelRegistryMutationRequest {
    /// Optimistic concurrency token returned by the previous read or mutation.
    pub expected_revision: u64,
    /// Command-shaped mutation; whole-registry replacement is intentionally unsupported.
    pub mutation: CatalogModelRegistryMutation,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct ResolveModelAliasRequest {
    pub alias: String,
    pub dimension: u32,
    /// Runtime implementation that will execute the immutable contract.
    pub runtime: Option<String>,
    /// Require the latest append-only decision to approve the selected version.
    #[serde(default)]
    pub require_approved: bool,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ResolvedEmbeddingModelResponse {
    pub asset_id: String,
    pub registry_name: String,
    pub registry_revision: u64,
    pub contract_sha256: String,
    pub model: CatalogEmbeddingModelVersion,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ResolvedModelBindingResponse {
    pub model: String,
    pub dimension: u32,
    pub model_asset_id: String,
    pub model_version: u64,
    pub contract_sha256: String,
    pub snapshot: ResolvedEmbeddingModelResponse,
}

fn service_error(error: ModelRegistryServiceError) -> ApiError {
    match error {
        ModelRegistryServiceError::InvalidTenant { .. }
        | ModelRegistryServiceError::InvalidName { .. } => {
            ApiError::InvalidArgument(error.to_string())
        }
        ModelRegistryServiceError::AlreadyExists { .. } => {
            ApiError::AlreadyExists(error.to_string())
        }
        ModelRegistryServiceError::NotFound { .. } => ApiError::NotFound(error.to_string()),
        ModelRegistryServiceError::Contract(CatalogModelContractError::RevisionConflict {
            ..
        }) => ApiError::Conflict(error.to_string()),
        ModelRegistryServiceError::Contract(_) => ApiError::InvalidArgument(error.to_string()),
        ModelRegistryServiceError::TenantIdentityUnavailable { .. }
        | ModelRegistryServiceError::MissingAssetId { .. }
        | ModelRegistryServiceError::NotModelRegistry { .. }
        | ModelRegistryServiceError::Catalog(_) => ApiError::Internal(error.to_string()),
    }
}

#[utoipa::path(
    post,
    path = "/api/v2/model-registries",
    tag = "Model Registry",
    operation_id = "createModelRegistry",
    request_body = CreateModelRegistryRequest,
    responses(
        (status = 200, description = "Created model registry.", body = ModelRegistryRecordResponse),
        (status = 400, description = "Invalid registry name.", body = crate::rest::errors::ErrorResponse),
        (status = 409, description = "Registry already exists.", body = crate::rest::errors::ErrorResponse),
    ),
)]
pub async fn create_model_registry(
    Extension(tenant): Extension<TenantContext>,
    State(state): State<RestAppState>,
    Json(request): Json<CreateModelRegistryRequest>,
) -> ApiResult<Json<ModelRegistryRecordResponse>> {
    state.model_registry()?
        .create_registry(&tenant.tenant_id, &request.name)
        .await
        .map(ModelRegistryRecordResponse::from)
        .map(Json)
        .map_err(service_error)
}

#[utoipa::path(
    get,
    path = "/api/v2/model-registries",
    tag = "Model Registry",
    operation_id = "listModelRegistries",
    responses(
        (status = 200, description = "Tenant-scoped registries in deterministic name order.", body = ListModelRegistriesResponse),
    ),
)]
pub async fn list_model_registries(
    Extension(tenant): Extension<TenantContext>,
    State(state): State<RestAppState>,
) -> ApiResult<Json<ListModelRegistriesResponse>> {
    let registries = state.model_registry()?
        .list_registries(&tenant.tenant_id)
        .await
        .map_err(service_error)?
        .into_iter()
        .map(ModelRegistryRecordResponse::from)
        .collect();
    Ok(Json(ListModelRegistriesResponse { registries }))
}

#[utoipa::path(
    get,
    path = "/api/v2/model-registries/{name}",
    tag = "Model Registry",
    operation_id = "getModelRegistry",
    params(("name" = String, Path, description = "Tenant-local registry name.")),
    responses(
        (status = 200, description = "Model registry.", body = ModelRegistryRecordResponse),
        (status = 404, description = "Registry not found.", body = crate::rest::errors::ErrorResponse),
    ),
)]
pub async fn get_model_registry(
    Path(name): Path<String>,
    Extension(tenant): Extension<TenantContext>,
    State(state): State<RestAppState>,
) -> ApiResult<Json<ModelRegistryRecordResponse>> {
    state.model_registry()?
        .get_registry(&tenant.tenant_id, &name)
        .await
        .map(ModelRegistryRecordResponse::from)
        .map(Json)
        .map_err(service_error)
}

#[utoipa::path(
    post,
    path = "/api/v2/model-registries/{name}/mutations",
    tag = "Model Registry",
    operation_id = "applyModelRegistryMutation",
    params(("name" = String, Path, description = "Tenant-local registry name.")),
    request_body = ApplyModelRegistryMutationRequest,
    responses(
        (status = 200, description = "Updated registry.", body = ModelRegistryRecordResponse),
        (status = 400, description = "Invalid lifecycle command.", body = crate::rest::errors::ErrorResponse),
        (status = 404, description = "Registry not found.", body = crate::rest::errors::ErrorResponse),
        (status = 409, description = "Revision conflict.", body = crate::rest::errors::ErrorResponse),
    ),
)]
pub async fn apply_model_registry_mutation(
    Path(name): Path<String>,
    Extension(tenant): Extension<TenantContext>,
    State(state): State<RestAppState>,
    Json(request): Json<ApplyModelRegistryMutationRequest>,
) -> ApiResult<Json<ModelRegistryRecordResponse>> {
    state.model_registry()?
        .apply_mutation(
            &tenant.tenant_id,
            &name,
            request.expected_revision,
            request.mutation,
        )
        .await
        .map(ModelRegistryRecordResponse::from)
        .map(Json)
        .map_err(service_error)
}

#[utoipa::path(
    post,
    path = "/api/v2/model-registries/{name}/resolve",
    tag = "Model Registry",
    operation_id = "resolveModelAlias",
    params(("name" = String, Path, description = "Tenant-local registry name.")),
    request_body = ResolveModelAliasRequest,
    responses(
        (status = 200, description = "Immutable version and digest binding.", body = ResolvedModelBindingResponse),
        (status = 400, description = "Alias, dimension, runtime, or approval policy rejected.", body = crate::rest::errors::ErrorResponse),
        (status = 404, description = "Registry not found.", body = crate::rest::errors::ErrorResponse),
    ),
)]
pub async fn resolve_model_alias(
    Path(name): Path<String>,
    Extension(tenant): Extension<TenantContext>,
    State(state): State<RestAppState>,
    Json(request): Json<ResolveModelAliasRequest>,
) -> ApiResult<Json<ResolvedModelBindingResponse>> {
    let policy = CatalogModelUsePolicy {
        runtime: request.runtime,
        require_approved: request.require_approved,
    };
    let resolved = state.model_registry()?
        .resolve_alias_binding(
            &tenant.tenant_id,
            &name,
            &request.alias,
            request.dimension,
            &policy,
        )
        .await
        .map_err(service_error)?;
    let binding = resolved.binding;
    let response = ResolvedModelBindingResponse {
        model: binding.model,
        dimension: binding.dimension,
        model_asset_id: binding
            .model_asset_id
            .ok_or_else(|| {
                ApiError::Internal("resolved binding has no model asset id".to_string())
            })?
            .to_string(),
        model_version: binding.model_version.ok_or_else(|| {
            ApiError::Internal("resolved binding has no model version".to_string())
        })?,
        contract_sha256: binding.contract_sha256.ok_or_else(|| {
            ApiError::Internal("resolved binding has no contract digest".to_string())
        })?,
        snapshot: ResolvedEmbeddingModelResponse {
            asset_id: resolved.snapshot.asset_id.to_string(),
            registry_name: resolved.snapshot.registry_name,
            registry_revision: resolved.snapshot.registry_revision,
            contract_sha256: resolved.snapshot.contract_sha256,
            model: resolved.snapshot.model,
        },
    };
    Ok(Json(response))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn revision_conflicts_are_retryable_http_conflicts() {
        let error = service_error(ModelRegistryServiceError::Contract(
            CatalogModelContractError::RevisionConflict {
                expected: 2,
                current: 3,
            },
        ));
        assert!(matches!(error, ApiError::Conflict(_)));
    }

    #[test]
    fn stable_asset_ids_are_decimal_strings_in_rest() {
        let registry = CatalogEmbeddingModelRegistry::new("embed").unwrap();
        let response = ModelRegistryRecordResponse::from(CatalogModelRegistryRecord {
            tenant_id: "tenant-a".to_string(),
            asset_id: u64::MAX,
            registry,
        });
        assert_eq!(response.asset_id, "18446744073709551615");
    }
}
