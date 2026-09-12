//! MLflow 3.x LoggedModel wire (TD-MLOPS-2).
//!
//! The logged-model family lives under the EXISTING `/api/2.0/mlflow`
//! router (`logged-models*` paths, merged into [`super::mlflow_routes`])
//! — the client's dispatch sends every logged-model RPC through the v2
//! map; the `/api/3.0`-annotated duplicates in the proto have no OSS
//! client path.
//!
//! `artifact_uri` is minted as `mlflow-artifacts:/{exp}/models/{id}/
//! artifacts`, mirroring the reference server — the client takes it
//! verbatim and builds an `MLflowArtifactsRepository` against the tracking
//! host, so model artifact uploads/listings ride the EXISTING artifacts
//! proxy with no new plumbing.

use std::collections::BTreeMap;

use axum::extract::{Path, State};
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use proximadb_catalog::run_store::{LoggedModelRecord, LoggedModelStatus, RunLifecycle};
use serde::{Deserialize, Serialize};

use crate::network::middleware::tenant::TenantContext;
use proximadb_catalog::run_store::RunStore;

use super::dto::{KeyValue, KeyValueOut, ParamOut, serialize_metric_value};
use super::{MlflowError, MlflowResult, MlflowState, store_for};

// The model artifact URI lives in `artifacts.rs` beside its siblings
// (`run_artifact_uri`, `experiment_artifact_location`) — one home for the
// URI family.

// ---------------------------------------------------------------------------
// DTOs
// ---------------------------------------------------------------------------

#[derive(Default, Deserialize)]
pub(crate) struct LoggedModelParamIn {
    #[serde(default)]
    pub(crate) key: String,
    #[serde(default)]
    pub(crate) value: String,
}

#[derive(Default, Deserialize)]
pub(crate) struct CreateLoggedModelRequest {
    #[serde(default)]
    pub(crate) experiment_id: String,
    #[serde(default)]
    pub(crate) name: String,
    #[serde(default)]
    pub(crate) model_type: Option<String>,
    #[serde(default)]
    pub(crate) source_run_id: Option<String>,
    #[serde(default)]
    pub(crate) params: Vec<LoggedModelParamIn>,
    #[serde(default)]
    pub(crate) tags: Vec<KeyValue>,
}

#[derive(Default, Deserialize)]
pub(crate) struct SearchLoggedModelsRequest {
    #[serde(default)]
    pub(crate) experiment_ids: Vec<String>,
    #[serde(default)]
    pub(crate) filter: Option<String>,
    /// Metric-dataset scoping the client can send — NOT silently dropped
    /// (the module's own never-ignore law): a non-empty scope is an honest
    /// INVALID_PARAMETER_VALUE rather than an over-broad result set.
    #[serde(default)]
    pub(crate) datasets: Vec<serde_json::Value>,
    #[serde(default)]
    pub(crate) max_results: Option<u32>,
    #[serde(default)]
    pub(crate) order_by: Vec<LoggedModelOrderByIn>,
    #[serde(default)]
    pub(crate) page_token: Option<String>,
}

#[derive(Default, Deserialize)]
pub(crate) struct LoggedModelOrderByIn {
    #[serde(default)]
    pub(crate) field_name: Option<String>,
    #[serde(default)]
    pub(crate) ascending: Option<bool>,
}

#[derive(Default, Deserialize)]
pub(crate) struct FinalizeLoggedModelRequest {
    #[serde(default)]
    pub(crate) status: Option<String>,
}

#[derive(Default, Deserialize)]
pub(crate) struct LogLoggedModelParamsRequest {
    #[serde(default)]
    pub(crate) params: Vec<LoggedModelParamIn>,
}

#[derive(Default, Deserialize)]
pub(crate) struct SetLoggedModelTagsRequest {
    #[serde(default)]
    pub(crate) tags: Vec<KeyValue>,
}

#[derive(Default, Deserialize)]
pub(crate) struct GetLoggedModelRequest {
    #[serde(default)]
    pub(crate) allow_deleted: Option<bool>,
}

#[derive(Default, Deserialize)]
pub(crate) struct ListLoggedModelArtifactsRequest {
    #[serde(default)]
    pub(crate) artifact_directory_path: Option<String>,
    #[serde(default)]
    pub(crate) page_token: Option<String>,
}

/// proto3-JSON enum names carry the `LOGGED_MODEL_` prefix — the bare
/// words ("PENDING"/"READY") are NOT the wire spellings.
fn status_from_json(raw: &str) -> MlflowResult<LoggedModelStatus> {
    match raw {
        "LOGGED_MODEL_PENDING" => Ok(LoggedModelStatus::Pending),
        "LOGGED_MODEL_READY" => Ok(LoggedModelStatus::Ready),
        "LOGGED_MODEL_UPLOAD_FAILED" => Ok(LoggedModelStatus::UploadFailed),
        other => Err(MlflowError::invalid(format!(
            "invalid logged-model status '{other}' (LOGGED_MODEL_PENDING | LOGGED_MODEL_READY | LOGGED_MODEL_UPLOAD_FAILED)"
        ))),
    }
}

fn status_name(status: LoggedModelStatus) -> &'static str {
    match status {
        LoggedModelStatus::Pending => "LOGGED_MODEL_PENDING",
        LoggedModelStatus::Ready => "LOGGED_MODEL_READY",
        LoggedModelStatus::UploadFailed => "LOGGED_MODEL_UPLOAD_FAILED",
    }
}

fn checked_params(params: &[LoggedModelParamIn]) -> MlflowResult<BTreeMap<String, String>> {
    let mut collected = BTreeMap::new();
    for param in params {
        if let Some(existing) = collected.get(&param.key)
            && existing != &param.value
        {
            return Err(MlflowError::invalid(format!(
                "param '{}' logged twice with different values in one request",
                param.key
            )));
        }
        collected.insert(param.key.clone(), param.value.clone());
    }
    Ok(collected)
}

#[derive(Serialize)]
pub(crate) struct LoggedModelInfoOut {
    model_id: String,
    experiment_id: String,
    name: String,
    creation_timestamp_ms: i64,
    last_updated_timestamp_ms: i64,
    artifact_uri: String,
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_run_id: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tags: Vec<KeyValueOut>,
}

#[derive(Serialize)]
pub(crate) struct ModelMetricOut {
    key: String,
    #[serde(serialize_with = "serialize_metric_value")]
    value: f64,
    timestamp: i64,
    step: i64,
    model_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    run_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dataset_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dataset_digest: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct LoggedModelDataOut {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    params: Vec<ParamOut>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    metrics: Vec<ModelMetricOut>,
}

#[derive(Serialize)]
pub(crate) struct LoggedModelOut {
    info: LoggedModelInfoOut,
    data: LoggedModelDataOut,
}

fn latest_model_metrics(
    samples: Vec<proximadb_catalog::run_store::ModelMetricPoint>,
) -> Vec<proximadb_catalog::run_store::ModelMetricPoint> {
    let mut latest = BTreeMap::new();
    for sample in samples {
        let identity = (
            sample.point.key.clone(),
            sample.dataset_name.clone(),
            sample.dataset_digest.clone(),
        );
        let replace = latest.get(&identity).is_none_or(
            |previous: &proximadb_catalog::run_store::ModelMetricPoint| {
                (sample.point.timestamp_ms, sample.point.step)
                    > (previous.point.timestamp_ms, previous.point.step)
            },
        );
        if replace {
            latest.insert(identity, sample);
        }
    }
    latest.into_values().collect()
}

/// Assemble the wire shape with MLflow's latest sample for each
/// (metric-key, dataset-name, dataset-digest) identity.
pub(crate) async fn logged_model_out(
    store: &std::sync::Arc<dyn RunStore>,
    record: &LoggedModelRecord,
) -> MlflowResult<LoggedModelOut> {
    let samples = latest_model_metrics(
        store
            .model_metrics(&record.model_id)
            .await
            .map_err(MlflowError::from)?,
    );
    Ok(LoggedModelOut {
        info: LoggedModelInfoOut {
            model_id: record.model_id.clone(),
            experiment_id: record.experiment_id.to_string(),
            name: record.name.clone(),
            creation_timestamp_ms: record.creation_time_ms,
            last_updated_timestamp_ms: record.last_updated_time_ms,
            artifact_uri: record.artifact_uri.clone(),
            status: status_name(record.status),
            model_type: record.model_type.clone(),
            source_run_id: record.source_run_id.clone(),
            tags: record
                .tags
                .iter()
                .map(|(k, v)| KeyValueOut {
                    key: k.clone(),
                    value: v.clone(),
                })
                .collect(),
        },
        data: LoggedModelDataOut {
            params: record
                .params
                .iter()
                .map(|(k, v)| ParamOut {
                    key: k.clone(),
                    value: v.clone(),
                })
                .collect(),
            metrics: samples
                .iter()
                .map(|s| ModelMetricOut {
                    key: s.point.key.clone(),
                    value: s.point.value,
                    timestamp: s.point.timestamp_ms,
                    step: s.point.step,
                    model_id: record.model_id.clone(),
                    run_id: s.run_id.clone(),
                    dataset_name: s.dataset_name.clone(),
                    dataset_digest: s.dataset_digest.clone(),
                })
                .collect(),
        },
    })
}

/// The logged-model router, merged into [`super::mlflow_routes`] (2.0
/// family — see module docs).
pub fn logged_model_routes() -> Router<MlflowState> {
    Router::new()
        .route("/logged-models", post(logged_models_create))
        .route("/logged-models/search", post(logged_models_search))
        .route(
            "/logged-models/{model_id}",
            get(logged_models_get)
                .patch(logged_models_finalize)
                .delete(logged_models_delete),
        )
        .route(
            "/logged-models/{model_id}/params",
            post(logged_models_log_params),
        )
        .route(
            "/logged-models/{model_id}/tags",
            axum::routing::patch(logged_models_set_tags),
        )
        .route(
            "/logged-models/{model_id}/tags/{tag_key}",
            axum::routing::delete(logged_models_delete_tag),
        )
        .route(
            "/logged-models/{model_id}/artifacts/directories",
            get(logged_models_list_artifacts),
        )
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn logged_models_create(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<CreateLoggedModelRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    if req.name.is_empty() {
        return Err(MlflowError::invalid("name must not be empty"));
    }
    let experiment_id = super::parse_id(&req.experiment_id, "experiment")?;
    let store = store_for(&tenant, &state)?;
    let now_ms = chrono::Utc::now().timestamp_millis();
    // Server-minted id (`m-<32hex>` — the reference format; the client
    // performs no format validation).
    let model_id = format!("m-{}", super::uuid_like_id());
    let params = checked_params(&req.params)?;
    let record = LoggedModelRecord {
        model_id,
        experiment_id,
        name: req.name,
        model_type: req.model_type,
        source_run_id: req.source_run_id,
        artifact_uri: String::new(),
        status: LoggedModelStatus::Pending,
        lifecycle: RunLifecycle::Active,
        params,
        tags: req
            .tags
            .iter()
            .map(|t| (t.key.clone(), t.value.clone()))
            .collect(),
        creation_time_ms: now_ms,
        last_updated_time_ms: now_ms,
    };
    // URI minted after the id exists; stored verbatim by the port.
    let mut record = record;
    record.artifact_uri = super::artifacts::model_artifact_uri(experiment_id, &record.model_id);
    let record = store.create_logged_model(record).await?;
    let model = logged_model_out(&store, &record).await?;
    Ok(Json(serde_json::json!({ "model": model })))
}

async fn logged_models_search(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<SearchLoggedModelsRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    if req.experiment_ids.is_empty() {
        return Err(MlflowError::invalid(
            "experiment_ids must list at least one experiment",
        ));
    }
    if !req.datasets.is_empty() {
        return Err(MlflowError::invalid(
            "dataset-scoped logged-model search is not supported (drop the datasets scope)",
        ));
    }
    let clauses = req
        .filter
        .as_deref()
        .map(super::filter::parse_logged_model_filter)
        .transpose()?
        .unwrap_or_default();
    let mut order_by = Vec::new();
    for order in &req.order_by {
        let ascending = order.ascending.unwrap_or(false);
        match order.field_name.as_deref().unwrap_or_default() {
            "name" => order_by.push(("name", ascending)),
            "creation_timestamp_ms" => order_by.push(("creation_timestamp_ms", ascending)),
            other => {
                return Err(MlflowError::invalid(format!(
                    "unsupported order_by field '{other}' (name | creation_timestamp_ms)"
                )));
            }
        }
    }
    if order_by.is_empty() {
        order_by.push(("creation_timestamp_ms", false));
    }
    let mut experiment_ids = Vec::new();
    for experiment_id in &req.experiment_ids {
        let experiment_id = super::parse_id(experiment_id, "experiment")?;
        if !experiment_ids.contains(&experiment_id) {
            experiment_ids.push(experiment_id);
        }
    }
    let store = store_for(&tenant, &state)?;
    let mut records = Vec::new();
    for experiment_id in experiment_ids {
        records.extend(store.list_logged_models(experiment_id, false).await?);
    }
    records.retain(|record| clauses.iter().all(|clause| clause.matches(record)));
    records.sort_by(|a, b| {
        for (field, ascending) in &order_by {
            let order = match *field {
                "name" => a.name.cmp(&b.name),
                "creation_timestamp_ms" => a.creation_time_ms.cmp(&b.creation_time_ms),
                _ => std::cmp::Ordering::Equal,
            };
            let order = if *ascending { order } else { order.reverse() };
            if order != std::cmp::Ordering::Equal {
                return order;
            }
        }
        a.model_id.cmp(&b.model_id)
    });
    let mut models = Vec::new();
    for record in &records {
        models.push(logged_model_out(&store, record).await?);
    }
    let (models, next_page_token) =
        super::paginate(models, req.max_results, req.page_token.as_deref())?;
    let mut body = serde_json::json!({ "models": models });
    if let Some(token) = next_page_token {
        body["next_page_token"] = serde_json::Value::String(token);
    }
    Ok(Json(body))
}

async fn logged_models_get(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Path(model_id): Path<String>,
    super::MlflowRead(req): super::MlflowRead<GetLoggedModelRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    let store = store_for(&tenant, &state)?;
    let record = store
        .get_logged_model(&model_id, req.allow_deleted.unwrap_or(false))
        .await?;
    let model = logged_model_out(&store, &record).await?;
    Ok(Json(serde_json::json!({ "model": model })))
}

async fn logged_models_finalize(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Path(model_id): Path<String>,
    Json(req): Json<FinalizeLoggedModelRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    let status = req
        .status
        .as_deref()
        .map(status_from_json)
        .transpose()?
        .ok_or_else(|| MlflowError::invalid("status is required"))?;
    let store = store_for(&tenant, &state)?;
    let now_ms = chrono::Utc::now().timestamp_millis();
    let record = store
        .finalize_logged_model(&model_id, status, now_ms)
        .await?;
    let model = logged_model_out(&store, &record).await?;
    Ok(Json(serde_json::json!({ "model": model })))
}

async fn logged_models_delete(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Path(model_id): Path<String>,
) -> MlflowResult<Json<serde_json::Value>> {
    store_for(&tenant, &state)?
        .delete_logged_model(&model_id)
        .await?;
    Ok(Json(serde_json::json!({})))
}

async fn logged_models_log_params(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Path(model_id): Path<String>,
    Json(req): Json<LogLoggedModelParamsRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    let params = checked_params(&req.params)?;
    store_for(&tenant, &state)?
        .log_logged_model_params(&model_id, &params)
        .await?;
    Ok(Json(serde_json::json!({})))
}

async fn logged_models_set_tags(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Path(model_id): Path<String>,
    Json(req): Json<SetLoggedModelTagsRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    let store = store_for(&tenant, &state)?;
    let tags: BTreeMap<String, String> = req.tags.into_iter().map(|t| (t.key, t.value)).collect();
    let record = store.set_logged_model_tags(&model_id, &tags).await?;
    let model = logged_model_out(&store, &record).await?;
    Ok(Json(serde_json::json!({ "model": model })))
}

async fn logged_models_delete_tag(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Path((model_id, tag_key)): Path<(String, String)>,
) -> MlflowResult<Json<serde_json::Value>> {
    store_for(&tenant, &state)?
        .delete_logged_model_tag(&model_id, &tag_key)
        .await?;
    Ok(Json(serde_json::json!({})))
}

async fn logged_models_list_artifacts(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Path(model_id): Path<String>,
    super::MlflowRead(req): super::MlflowRead<ListLoggedModelArtifactsRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    let store = store_for(&tenant, &state)?;
    let record = store.get_logged_model(&model_id, false).await?;
    // The model's artifact repo path: the minted URI's path portion, plus
    // the requested subdirectory. The artifact module resolves it beneath
    // the tenant capability without exposing a host path here.
    let sub = req.artifact_directory_path.unwrap_or_default();
    let mut path = record
        .artifact_uri
        .trim_start_matches("mlflow-artifacts:")
        .trim_matches('/')
        .to_string();
    if !sub.is_empty() {
        for segment in sub.split('/') {
            if segment.is_empty() || segment == "." {
                continue;
            }
            if segment == ".." || segment.contains('\\') || segment.contains('\0') {
                return Err(MlflowError::invalid(format!(
                    "invalid artifact directory segment '{segment}'"
                )));
            }
            path.push('/');
            path.push_str(segment);
        }
    }
    let files = super::artifacts::list_directory(&state, &tenant, &path, &sub).await?;
    let (files, next_page_token) = super::paginate(files, None, req.page_token.as_deref())?;
    let mut body = serde_json::json!({
        "root_uri": record.artifact_uri,
        "files": files,
    });
    if let Some(token) = next_page_token {
        body["next_page_token"] = serde_json::Value::String(token);
    }
    Ok(Json(body))
}
