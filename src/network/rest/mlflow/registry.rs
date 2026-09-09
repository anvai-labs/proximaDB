//! MLflow registry wire (TD-MLOPS-1 slice 3) — the model-registry half of
//! the compatibility surface, lowering to `CatalogModelRegistryService`
//! commands (the xCatalog authority; never a second metadata database).
//!
//! Honest-surface rules (adapter over authority):
//! * Registered models map to xCatalog registries (create/get/search);
//!   aliases map to catalog alias transactions (set/get).
//! * `model-versions/create` is REJECTED: an xCatalog version is a typed
//!   immutable executable contract (artifact digest + input/output
//!   contracts + governance) that the MLflow wire cannot carry — version
//!   registration goes through the native lifecycle API. Get/search still
//!   project native versions as MLflow model-version objects.
//! * `transition-stage` is REJECTED with an alias pointer (no built-in
//!   staging/production state machine; MLflow ≥2.5 aliases are the path).
//! * Tags/descriptions on the MLflow registry wire are not persisted (the
//!   registry's annotation facet ships with the native lifecycle surface);
//!   supplying them is an explicit error, never fabricated success.

use axum::extract::State;
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use proximadb_catalog::mlops::CatalogModelRegistryMutation;
use proximadb_catalog::model_registry_service::{
    CatalogModelRegistryRecord, ModelRegistryServiceError,
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::network::middleware::tenant::TenantContext;

use super::{MlflowError, MlflowRead, MlflowResult, MlflowState};

pub fn registry_routes() -> Router<MlflowState> {
    Router::new()
        .route("/registered-models/create", post(registered_models_create))
        .route(
            "/registered-models/get",
            get(registered_models_get).post(registered_models_get),
        )
        .route(
            "/registered-models/get-by-name",
            get(registered_models_get_by_name).post(registered_models_get_by_name),
        )
        .route(
            "/registered-models/search",
            get(registered_models_search).post(registered_models_search),
        )
        .route(
            "/registered-models/alias",
            post(registered_models_alias_set)
                .get(registered_models_alias_get)
                .delete(registered_models_alias_delete),
        )
        .route(
            "/model-versions/get",
            get(model_versions_get).post(model_versions_get),
        )
        .route(
            "/model-versions/search",
            get(model_versions_search).post(model_versions_search),
        )
        .route("/model-versions/create", post(model_versions_create))
        .route(
            "/model-versions/transition-stage",
            post(model_versions_transition_stage),
        )
}

// ---------------------------------------------------------------------------
// DTOs
// ---------------------------------------------------------------------------

#[derive(Default, Deserialize)]
pub(super) struct RegisteredModelCreateRequest {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub tags: Option<Value>,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Default, Deserialize)]
pub struct ModelNameRequest {
    #[serde(default)]
    pub name: String,
}

#[derive(Default, Deserialize)]
pub struct RegisteredModelsSearchRequest {
    #[serde(default)]
    pub filter: Option<String>,
    #[serde(default)]
    pub max_results: Option<u32>,
}

#[derive(Default, Deserialize)]
pub struct AliasRequest {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub alias: String,
    #[serde(default)]
    pub version: Option<String>,
}

#[derive(Default, Deserialize)]
pub struct ModelVersionRequest {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub version: String,
}

#[derive(Default, Deserialize)]
pub struct ModelVersionsSearchRequest {
    #[serde(default)]
    pub name: String,
    /// Real clients pass the model name inside a filter string
    /// (`name='x'`), not as a bare field.
    #[serde(default)]
    pub filter: Option<String>,
    #[serde(default)]
    pub max_results: Option<u32>,
}

#[derive(Default, Deserialize)]
pub struct TransitionStageRequest {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub stage: String,
}

// ---------------------------------------------------------------------------
// Error lowering — the registry service's typed errors to MLflow codes
// ---------------------------------------------------------------------------

impl From<ModelRegistryServiceError> for MlflowError {
    fn from(e: ModelRegistryServiceError) -> Self {
        match &e {
            ModelRegistryServiceError::NotFound { name, .. } => {
                MlflowError::not_found(format!("Could not find registered model '{name}'"))
            }
            ModelRegistryServiceError::AlreadyExists { name, .. } => {
                MlflowError::exists(format!("Registered model '{name}' already exists"))
            }
            ModelRegistryServiceError::InvalidName { reason }
            | ModelRegistryServiceError::InvalidTenant { reason } => {
                MlflowError::invalid(reason.clone())
            }
            other => MlflowError::internal(other.to_string()),
        }
    }
}

fn contract_error(e: ModelRegistryServiceError) -> MlflowError {
    // Contract failures (duplicate version, unknown alias) surface as typed
    // MLflow errors, not opaque internals.
    if let ModelRegistryServiceError::Contract(inner) = &e {
        return match inner {
            proximadb_catalog::mlops::CatalogModelContractError::DuplicateVersion { version } => {
                MlflowError::exists(format!(
                    "Model version {version} already exists (versions are immutable)"
                ))
            }
            proximadb_catalog::mlops::CatalogModelContractError::UnknownAlias { alias } => {
                MlflowError::not_found(format!("Alias '{alias}' is not registered"))
            }
            other_contract => MlflowError::invalid(other_contract.to_string()),
        };
    }
    e.into()
}

// ---------------------------------------------------------------------------
// Response shaping — xCatalog records to MLflow JSON
// ---------------------------------------------------------------------------

fn registered_model_out(record: &CatalogModelRegistryRecord) -> Value {
    let registry = &record.registry;
    // MLflow's RegisteredModel.aliases is repeated RegisteredModelAlias
    // {alias, version} — bare strings parse into EMPTY alias messages
    // client-side (verified against mlflow 2.14.3).
    let aliases: Vec<Value> = registry
        .aliases
        .iter()
        .map(|(alias, version)| json!({"alias": alias, "version": version.to_string()}))
        .collect();
    let latest_versions: Vec<Value> = registry
        .versions
        .keys()
        .filter_map(|v| model_version_out(record, &v.to_string()))
        .collect();
    json!({
        "name": registry.name,
        "aliases": aliases,
        // The proto field is `latest_versions`; a `versions` key is silently
        // dropped by the client.
        "latest_versions": latest_versions,
        "creation_timestamp": 0i64,
        "last_updated_timestamp": 0i64,
    })
}

fn model_version_out(record: &CatalogModelRegistryRecord, version: &str) -> Option<Value> {
    let registry = &record.registry;
    let key: u64 = version.parse().ok()?;
    let entry = registry.versions.get(&key)?;
    let aliases: Vec<&String> = registry
        .aliases
        .iter()
        .filter(|(_, v)| **v == key)
        .map(|(a, _)| a)
        .collect();
    Some(json!({
        "name": registry.name,
        "version": key.to_string(),
        // The artifact URI is the closest MLflow `source` analogue; the
        // digest identity stays on the native contract.
        "source": entry.artifact.uri,
        "run_id": entry.source_run_id.clone().unwrap_or_default(),
        "status": "READY",
        "current_stage": "None",
        "aliases": aliases,
        "creation_timestamp": entry.created_at_ms,
        "last_updated_timestamp": entry.created_at_ms,
    }))
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn registered_models_create(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<RegisteredModelCreateRequest>,
) -> MlflowResult<Json<Value>> {
    if req.name.trim().is_empty() {
        return Err(MlflowError::invalid("name must not be empty"));
    }
    if req.tags.is_some() || req.description.is_some() {
        return Err(MlflowError::invalid(
            "MLflow registry tags/descriptions are not persisted by the compatibility \
             adapter (the annotation facet ships with the native lifecycle API at \
             /api/v2/abac/../model-registry); omit them here rather than lose them",
        ));
    }
    let record = state
        .registry
        .create_registry(&tenant.tenant_id, &req.name)
        .await?;
    Ok(Json(json!({
        "registered_model": registered_model_out(&record),
    })))
}

async fn registered_models_get(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    MlflowRead(req): MlflowRead<ModelNameRequest>,
) -> MlflowResult<Json<Value>> {
    let record = state
        .registry
        .get_registry(&tenant.tenant_id, &req.name)
        .await?;
    Ok(Json(json!({
        "registered_model": registered_model_out(&record),
    })))
}

async fn registered_models_get_by_name(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    MlflowRead(req): MlflowRead<ModelNameRequest>,
) -> MlflowResult<Json<Value>> {
    registered_models_get(State(state), Extension(tenant), MlflowRead(req)).await
}

async fn registered_models_search(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    MlflowRead(req): MlflowRead<RegisteredModelsSearchRequest>,
) -> MlflowResult<Json<Value>> {
    // Subset: `name LIKE '%x%'` / `name = 'x'`; anything else is an explicit
    // error (never silently ignored).
    let mut name_eq: Option<String> = None;
    let mut name_like: Option<String> = None;
    // Tag predicates: the registry has no tag facet yet (the annotation
    // facet ships with the native lifecycle surface), so = never matches
    // and != matches everything — the absent-value semantics the MLflow
    // 3.x client relies on: it appends
    // `tag.`mlflow.prompt.is_prompt` != 'true'` to EVERY search to hide
    // prompt models, and that clause must pass cleanly.
    let mut tag_clauses: Vec<(String, String, bool)> = Vec::new();
    if let Some(filter) = &req.filter {
        for clause in super::split_filter_clauses(filter)? {
            let clause = clause.trim();
            let (field, op, value) = super::parse_filter_parts(clause, true)?;
            let field_lower = field.to_ascii_lowercase();
            let tag_key = field_lower
                .strip_prefix("tag.")
                .or_else(|| field_lower.strip_prefix("tags."))
                .map(|k| k.trim_matches('`').to_string());
            if let Some(key) = tag_key {
                if key.is_empty() {
                    return Err(MlflowError::invalid("tag filter needs a key"));
                }
                match op.as_str() {
                    "=" => tag_clauses.push((key, value, true)),
                    "!=" => tag_clauses.push((key, value, false)),
                    other => {
                        return Err(MlflowError::invalid(format!(
                            "tag filter supports = / !=, got '{other}'"
                        )));
                    }
                }
            } else {
                match field_lower.as_str() {
                    "name" | "attributes.name" => match op.as_str() {
                        "=" => name_eq = Some(value),
                        "LIKE" => name_like = Some(value),
                        other => {
                            return Err(MlflowError::invalid(format!(
                                "name filter supports = / LIKE, got '{other}'"
                            )));
                        }
                    },
                    _ => {
                        return Err(MlflowError::invalid(format!(
                            "unsupported registered-model filter clause '{clause}'"
                        )));
                    }
                }
            }
        }
    }
    let records = state.registry.list_registries(&tenant.tenant_id).await?;
    let mut models: Vec<Value> = records
        .iter()
        .filter(|r| {
            name_eq.as_ref().is_none_or(|v| &r.registry.name == v)
                && name_like
                    .as_ref()
                    .is_none_or(|pat| super::like_match(&r.registry.name, pat))
                && tag_clauses.iter().all(|(_key, value, is_eq)| {
                    // Registry annotation facet not yet present — the
                    // port's absent-value semantics decide.
                    proximadb_catalog::run_store::tag_clause_matches(*is_eq, None, value)
                })
        })
        .map(registered_model_out)
        .collect();
    if let Some(limit) = req.max_results {
        models.truncate(limit as usize);
    }
    Ok(Json(json!({ "registered_models": models })))
}

async fn registered_models_alias_set(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<AliasRequest>,
) -> MlflowResult<Json<Value>> {
    let version = req
        .version
        .as_deref()
        .ok_or_else(|| MlflowError::invalid("version is required (string)"))?;
    let version: u64 = version
        .parse()
        .map_err(|_| MlflowError::invalid(format!("invalid version '{version}'")))?;
    let record = state
        .registry
        .get_registry(&tenant.tenant_id, &req.name)
        .await?;
    state
        .registry
        .apply_mutation(
            &tenant.tenant_id,
            &req.name,
            record.registry.revision,
            CatalogModelRegistryMutation::set_alias(req.alias.clone(), version),
        )
        .await
        .map_err(contract_error)?;
    Ok(Json(json!({})))
}

async fn registered_models_alias_get(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    MlflowRead(req): MlflowRead<AliasRequest>,
) -> MlflowResult<Json<Value>> {
    let record = state
        .registry
        .get_registry(&tenant.tenant_id, &req.name)
        .await?;
    let version = record.registry.aliases.get(&req.alias).ok_or_else(|| {
        MlflowError::not_found(format!(
            "Alias '{}' is not set on registered model '{}'",
            req.alias, req.name
        ))
    })?;
    let version_out = model_version_out(&record, &version.to_string()).ok_or_else(|| {
        MlflowError::internal(format!(
            "alias '{alias}' points at missing version {version}",
            alias = req.alias
        ))
    })?;
    Ok(Json(json!({ "model_version": version_out })))
}

async fn registered_models_alias_delete(
    State(state): State<MlflowState>,
    Extension(_tenant): Extension<TenantContext>,
    Json(req): Json<AliasRequest>,
) -> MlflowResult<Json<Value>> {
    // The authority has set-only alias mutations today; removal needs a
    // service-side RemoveAlias command (tracked with the native lifecycle
    // surface). Honest rejection, never a silent no-op.
    let _ = state;
    Err(MlflowError::invalid(format!(
        "alias deletion is not yet supported by the model-registry authority \
             (set-only mutations); re-point '{}' via the alias set endpoint",
        req.alias
    )))
}

async fn model_versions_get(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    MlflowRead(req): MlflowRead<ModelVersionRequest>,
) -> MlflowResult<Json<Value>> {
    let record = state
        .registry
        .get_registry(&tenant.tenant_id, &req.name)
        .await?;
    let version = model_version_out(&record, &req.version).ok_or_else(|| {
        MlflowError::not_found(format!(
            "Could not find model version '{}/{}'",
            req.name, req.version
        ))
    })?;
    Ok(Json(json!({ "model_version": version })))
}

async fn model_versions_search(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    MlflowRead(req): MlflowRead<ModelVersionsSearchRequest>,
) -> MlflowResult<Json<Value>> {
    let mut name = req.name.trim().to_string();
    if name.is_empty()
        && let Some(filter) = &req.filter
    {
        for clause in super::split_filter_clauses(filter)? {
            let (field, op, value) = super::parse_filter_parts(clause.trim(), true)?;
            if field.eq_ignore_ascii_case("name") && op == "=" {
                name = value;
            } else {
                return Err(MlflowError::invalid(format!(
                    "unsupported model-version filter clause '{}'",
                    clause.trim()
                )));
            }
        }
    }
    if name.is_empty() {
        return Err(MlflowError::invalid(
            "name (or name='...' filter) is required",
        ));
    }
    let record = state
        .registry
        .get_registry(&tenant.tenant_id, &name)
        .await?;
    let mut versions: Vec<Value> = record
        .registry
        .versions
        .keys()
        .filter_map(|v| model_version_out(&record, &v.to_string()))
        .collect();
    versions.reverse(); // newest first, matching MLflow's default ordering
    if let Some(limit) = req.max_results {
        versions.truncate(limit as usize);
    }
    Ok(Json(json!({ "model_versions": versions })))
}

async fn model_versions_create(
    State(_state): State<MlflowState>,
    Extension(_tenant): Extension<TenantContext>,
    Json(_req): Json<Value>,
) -> MlflowResult<Json<Value>> {
    // xCatalog versions are typed immutable executable contracts (artifact
    // digest + tokenizer/rendering input contract + dimension/pooling output
    // contract + governance) — none of which the MLflow wire can carry.
    // Registering a version without them would fabricate an unexecutable
    // model; the honest answer names the native path.
    Err(MlflowError::invalid(
        "create-model-version is not supported through the MLflow adapter: an xCatalog \
         model version is a typed executable contract (artifact digest, input/output \
         contracts, governance). Register versions through the native model-registry \
         lifecycle API; the MLflow surface projects them read-only",
    ))
}

async fn model_versions_transition_stage(
    State(_state): State<MlflowState>,
    Extension(_tenant): Extension<TenantContext>,
    Json(req): Json<TransitionStageRequest>,
) -> MlflowResult<Json<Value>> {
    // No built-in staging/production state machine (design authority);
    // aliases are the supported promotion path.
    Err(MlflowError::invalid(format!(
        "stage transitions are not supported: the registry has no staging/production \
             state machine. Point a '{}' alias at this version instead \
             (registered-models/alias), and resolve deployments by alias",
        req.stage
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_catalog::model_registry_service::CatalogModelRegistryService;
    use std::sync::Arc;

    use crate::network::middleware::tenant::{TenantContext, TenantIdSource};
    use crate::storage::document::DocumentService;
    use crate::storage::engines::sst::SstEngine;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    async fn test_router() -> (Router, Arc<CatalogModelRegistryService>) {
        let engine = Arc::new(SstEngine::new().await.unwrap());
        let document = Arc::new(DocumentService::new(engine));
        let manager = Arc::new(crate::catalog::CatalogManager::new());
        // A bare manager has no default catalog; the service's first call
        // would fail "No default catalog configured". Attach a throwaway
        // native catalog over a unique tempdir (the first registered
        // catalog becomes the default).
        let temp_dir =
            std::env::temp_dir().join(format!("mlflow_registry_test_{}", std::process::id()));
        let _ = tokio::fs::remove_dir_all(&temp_dir).await;
        manager
            .create_native_catalog("mlflow-test", &format!("file://{}", temp_dir.display()))
            .await
            .expect("test catalog");
        let registry = Arc::new(CatalogModelRegistryService::new(manager));
        let run_store =
            Arc::new(crate::services::mlflow_run_store::SubstrateRunStoreFactory::new(document));
        let router = registry_routes()
            .with_state(MlflowState::new(
                run_store,
                registry.clone(),
                std::env::temp_dir(),
            ))
            .layer(axum::Extension(TenantContext::new(
                "default",
                TenantIdSource::Default,
            )));
        (router, registry)
    }

    async fn post(router: &mut Router, path: &str, body: Value) -> (axum::http::StatusCode, Value) {
        let response = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(path)
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        (
            status,
            if bytes.is_empty() {
                Value::Null
            } else {
                serde_json::from_slice(&bytes).unwrap_or(Value::Null)
            },
        )
    }

    #[tokio::test]
    async fn registry_workflow_over_the_wire() {
        let (mut router, service) = test_router().await;

        // Create -> duplicate is RESOURCE_ALREADY_EXISTS.
        let (status, body) = post(
            &mut router,
            "/registered-models/create",
            json!({"name": "text-embedder"}),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK, "{body}");
        assert_eq!(body["registered_model"]["name"], "text-embedder");

        let (status, body) = post(
            &mut router,
            "/registered-models/create",
            json!({"name": "text-embedder"}),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(body["error_code"], "RESOURCE_ALREADY_EXISTS");

        // Tags on create are an explicit rejection (never fabricated success).
        let (status, body) = post(
            &mut router,
            "/registered-models/create",
            json!({"name": "other", "tags": [{"key": "k", "value": "v"}]}),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(body["error_code"], "INVALID_PARAMETER_VALUE");

        // Get + get-by-name.
        let (status, body) = post(
            &mut router,
            "/registered-models/get",
            json!({"name": "text-embedder"}),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK, "{body}");
        assert_eq!(body["registered_model"]["name"], "text-embedder");

        // Search: name equality + LIKE + non-matching empty + garbage clause.
        let (_, body) = post(
            &mut router,
            "/registered-models/search",
            json!({"filter": "name = 'text-embedder'"}),
        )
        .await;
        assert_eq!(body["registered_models"].as_array().unwrap().len(), 1);

        let (_, body) = post(
            &mut router,
            "/registered-models/search",
            json!({"filter": "name LIKE '%embed%'"}),
        )
        .await;
        assert_eq!(body["registered_models"].as_array().unwrap().len(), 1);

        let (_, body) = post(
            &mut router,
            "/registered-models/search",
            json!({"filter": "name = 'nope'"}),
        )
        .await;
        assert_eq!(body["registered_models"].as_array().unwrap().len(), 0);

        let (_, body) = post(
            &mut router,
            "/registered-models/search",
            json!({"filter": "bogus = 1"}),
        )
        .await;
        assert_eq!(body["error_code"], "INVALID_PARAMETER_VALUE");

        // Alias set on a registry with no versions: the authority rejects
        // (unknown version) with a typed error, not an internal one.
        let (status, body) = post(
            &mut router,
            "/registered-models/alias",
            json!({"name": "text-embedder", "alias": "champion", "version": "1"}),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["error_code"], "INVALID_PARAMETER_VALUE");

        // model-versions/create and transition-stage: documented rejections
        // with pointers to the native path / aliases.
        let (status, body) = post(
            &mut router,
            "/model-versions/create",
            json!({"name": "text-embedder", "source": "s3://x"}),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert!(body["message"].as_str().unwrap().contains("lifecycle API"));

        let (status, body) = post(
            &mut router,
            "/model-versions/transition-stage",
            json!({"name": "text-embedder", "version": "1", "stage": "Production"}),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert!(body["message"].as_str().unwrap().contains("alias"));

        // Model versions of a registry with none: empty list, not an error.
        let (status, body) = post(
            &mut router,
            "/model-versions/search",
            json!({"name": "text-embedder"}),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK, "{body}");
        assert_eq!(body["model_versions"].as_array().unwrap().len(), 0);

        // Register a version through the AUTHORITY (the native path the
        // rejection message points at), then exercise alias set/get and the
        // object-shaped alias projection the real client parses.
        service
            .create_registry("default", "versioned")
            .await
            .unwrap();
        let version = proximadb_catalog::mlops::CatalogEmbeddingModelVersion {
            version: 1,
            provider_model_id: "bge-small".to_string(),
            artifact: proximadb_catalog::mlops::CatalogArtifactDescriptor::new(
                "s3://bucket/model.bin",
                "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                1024,
                "application/octet-stream",
            )
            .expect("descriptor"),
            input: minimally_valid_input_contract(),
            output: minimally_valid_output_contract(),
            governance: Default::default(),
            lineage: Default::default(),
            created_at_ms: 1_000,
            source_run_id: Some("run-0001".to_string()),
        };
        let record = service
            .apply_mutation(
                "default",
                "versioned",
                0,
                CatalogModelRegistryMutation::register_version(version),
            )
            .await
            .expect("register");

        let (status, body) = post(
            &mut router,
            "/registered-models/alias",
            json!({"name": "versioned", "alias": "champion", "version": "1"}),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK, "{body}");

        let (status, body) = post(
            &mut router,
            "/registered-models/get",
            json!({"name": "versioned"}),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK, "{body}");
        // MLflow's RegisteredModel.aliases is repeated {alias, version} —
        // bare strings parse into EMPTY alias messages client-side.
        let aliases = body["registered_model"]["aliases"].as_array().unwrap();
        assert_eq!(aliases.len(), 1, "{body}");
        assert_eq!(aliases[0]["alias"], "champion");
        assert_eq!(aliases[0]["version"], "1");
        // The proto field is latest_versions and carries the projection.
        let latest = body["registered_model"]["latest_versions"]
            .as_array()
            .unwrap();
        assert_eq!(latest.len(), 1);
        assert_eq!(latest[0]["version"], "1");
        assert_eq!(latest[0]["source"], "s3://bucket/model.bin");
        assert_eq!(latest[0]["run_id"], "run-0001");
        assert_eq!(record.registry.versions.len(), 1);
    }

    fn minimally_valid_input_contract() -> proximadb_catalog::mlops::CatalogEmbeddingInputContract {
        proximadb_catalog::mlops::CatalogEmbeddingInputContract {
            model_revision: "main".to_string(),
            tokenizer_id: "bge-small".to_string(),
            tokenizer_revision: "main".to_string(),
            tokenizer_fingerprint: format!("sha256:{}", "b".repeat(64)),
            declared_context_limit: 512,
            effective_context_limit: 512,
            special_token_count: 0,
            document_template: "{text}".to_string(),
            query_template: "{text}".to_string(),
            document_parameters: Default::default(),
            query_parameters: Default::default(),
        }
    }

    fn minimally_valid_output_contract() -> proximadb_catalog::mlops::CatalogEmbeddingOutputContract
    {
        proximadb_catalog::mlops::CatalogEmbeddingOutputContract {
            native_dimension: 384,
            dimension_policy: proximadb_catalog::mlops::CatalogDimensionPolicy::Fixed,
            supported_dimensions: vec![384],
            normalized: true,
            pooling: "mean".to_string(),
        }
    }
}
