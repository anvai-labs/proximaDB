//! MLflow-compatible tracking wire (TD-MLOPS-1 slice 2).
//!
//! A thin codec over the [`RunStore`] port: MLflow's REST API 2.x is the
//! external contract (pgwire-class compatibility — deliberately NOT our
//! OpenAPI), every handler lowers to `SubstrateRunStore::for_tenant` built
//! per request from the SHARED tenant plane (`Extension<TenantContext>` —
//! never MLflow body fields) and the shared `DocumentService`. Tenant
//! isolation is structural: a foreign tenant's experiment/run ids are simply
//! absent, so every cross-tenant probe gets the same
//! `RESOURCE_DOES_NOT_EXIST` as a missing id.
//!
//! Default OFF: the router mounts only when `PROXIMADB_MLFLOW_COMPAT_ENABLE`
//! is set (opt-in tier, ENV_GATE_REGISTRY); unset ⇒ every
//! `/api/2.0/mlflow/*` route falls through to the platform 404.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::network::middleware::tenant::TenantContext;
use axum::extract::State;
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use proximadb_catalog::run_store::{
    ExperimentRecord, ExperimentStage, MetricPoint, RunLifecycle, RunRecord, RunStatus, RunStore,
};
use serde::Deserialize;

/// Minimal state: the tracking wire touches nothing but the document
/// substrate. Built once at mount time from the canonical `AppState`.
#[derive(Clone)]
pub struct MlflowState {
    run_store: std::sync::Arc<dyn proximadb_catalog::run_store::RunStoreFactory>,
    registry: Arc<proximadb_catalog::model_registry_service::CatalogModelRegistryService>,
    pub(crate) data_dir: std::path::PathBuf,
}

impl MlflowState {
    pub fn new(
        run_store: std::sync::Arc<dyn proximadb_catalog::run_store::RunStoreFactory>,
        registry: Arc<proximadb_catalog::model_registry_service::CatalogModelRegistryService>,
        data_dir: std::path::PathBuf,
    ) -> Self {
        Self {
            run_store,
            registry,
            data_dir,
        }
    }
}

/// Gate: presence-style opt-in accepting `1|true|on|yes` (unset = the
/// routes never mount; registry row in ENV_GATE_REGISTRY.adoc).
pub fn enabled() -> bool {
    is_enabled(
        std::env::var("PROXIMADB_MLFLOW_COMPAT_ENABLE")
            .ok()
            .as_deref(),
    )
}

/// Pure predicate so the gate semantics are testable without env mutation
/// (edition 2024 makes set_var/remove_var unsafe).
fn is_enabled(value: Option<&str>) -> bool {
    match value {
        Some(v) => matches!(v.trim(), "1" | "true" | "on" | "yes"),
        None => false,
    }
}

pub mod artifacts;
pub mod dto;
pub mod error;
pub(crate) use dto::*;
pub mod filter;
pub(crate) use error::{MlflowError, MlflowResult};
use filter::ExperimentFilter;
pub mod logged_models;
pub mod registry;
pub mod traces;
pub mod ui;

/// The artifacts PROXY lives at /api/2.0/mlflow-artifacts (a sibling of
/// /api/2.0/mlflow, not under it) — mounted separately in server.rs.
pub fn artifacts_router() -> Router<MlflowState> {
    artifacts::artifacts_routes()
}

pub fn artifacts_router_at(prefix: &str) -> Router<MlflowState> {
    artifacts::artifacts_routes_at(prefix)
}

pub fn artifacts_router_relative() -> Router<MlflowState> {
    artifacts::artifacts_routes_relative()
}

pub fn mlflow_routes() -> Router<MlflowState> {
    Router::new()
        .route("/experiments/create", post(experiments_create))
        .route(
            "/experiments/get",
            get(experiments_get).post(experiments_get),
        )
        .route(
            "/experiments/get-by-name",
            get(experiments_get_by_name).post(experiments_get_by_name),
        )
        .route("/experiments/search", post(experiments_search))
        .route("/experiments/delete", post(experiments_delete))
        .route("/experiments/restore", post(experiments_restore))
        .route("/runs/create", post(runs_create))
        .route("/runs/get", get(runs_get).post(runs_get))
        .route("/runs/update", post(runs_update))
        .route("/runs/search", post(runs_search))
        .route("/runs/delete", post(runs_delete))
        .route("/runs/restore", post(runs_restore))
        .route("/runs/log-parameter", post(runs_log_parameter))
        .route("/runs/log-metric", post(runs_log_metric))
        .route("/runs/log-batch", post(runs_log_batch))
        .route("/runs/set-tag", post(runs_set_tag))
        .route("/runs/delete-tag", post(runs_delete_tag))
        .route("/runs/outputs", post(runs_outputs))
        .route("/runs/log-model", post(runs_log_model))
        .route("/runs/log-inputs", post(runs_log_inputs))
        .route(
            "/metrics/get-history",
            get(metrics_get_history).post(metrics_get_history),
        )
        // Traces v3 stragglers the client hard-codes to the 2.0 router
        // (everything else traces lives at /api/3.0 — traces.rs).
        .route("/traces/delete-traces", post(traces_delete_traces))
        .route(
            "/traces/{trace_id}/tags",
            axum::routing::patch(traces_set_tag).delete(traces_delete_tag),
        )
        .merge(registry::registry_routes())
        .merge(logged_models::logged_model_routes())
}

/// MLflow read endpoints are dual-shaped on the wire: the proto HTTP
/// annotations map them to **GET with query-string parameters** in current
/// clients, while older clients POST a JSON body. Accept both.
pub(crate) struct MlflowRead<T>(pub(crate) T);

impl<S, T> axum::extract::FromRequest<S> for MlflowRead<T>
where
    S: Send + Sync,
    T: for<'de> Deserialize<'de>,
{
    type Rejection = MlflowError;

    async fn from_request(
        req: axum::http::Request<axum::body::Body>,
        state: &S,
    ) -> Result<Self, Self::Rejection> {
        if req.method() == axum::http::Method::GET {
            let query = req.uri().query().unwrap_or_default();
            let value = serde_urlencoded::from_str::<T>(query).map_err(|e| {
                MlflowError::invalid(format!("invalid query parameters '{query}': {e}"))
            })?;
            Ok(MlflowRead(value))
        } else {
            match axum::extract::Json::<T>::from_request(req, state).await {
                Ok(axum::Json(value)) => Ok(MlflowRead(value)),
                Err(rejection) => Err(MlflowError::invalid(format!(
                    "invalid JSON body: {rejection}"
                ))),
            }
        }
    }
}

pub(crate) fn store_for(
    tenant: &TenantContext,
    state: &MlflowState,
) -> MlflowResult<std::sync::Arc<dyn RunStore>> {
    state
        .run_store
        .store_for(&tenant.tenant_id)
        .map_err(|e| MlflowError::internal(e.to_string()))
}

fn experiment_out(record: &ExperimentRecord) -> ExperimentOut {
    ExperimentOut {
        experiment_id: record.experiment_id.to_string(),
        name: record.name.clone(),
        artifact_uri: record
            .artifact_location
            .clone()
            .unwrap_or_else(|| artifacts::experiment_artifact_location(record.experiment_id)),
        lifecycle_stage: match record.stage {
            ExperimentStage::Active => "active",
            ExperimentStage::Deleted => "deleted",
        },
        creation_time: record.creation_time_ms,
        last_update_time: record.last_update_time_ms,
        tags: record
            .tags
            .iter()
            .map(|(k, v)| KeyValueOut {
                key: k.clone(),
                value: v.clone(),
            })
            .collect(),
    }
}

fn run_out(record: &RunRecord) -> RunOut {
    run_out_with(record, None)
}

/// `dataset_inputs` ride a separate port collection: embedded only in
/// runs/get (one extra call), never in runs/search (per-run N+1).
fn run_out_with(record: &RunRecord, dataset_inputs: Option<Vec<DatasetInputOut>>) -> RunOut {
    let mut tags: Vec<KeyValueOut> = record
        .tags
        .iter()
        .map(|(k, v)| KeyValueOut {
            key: k.clone(),
            value: v.clone(),
        })
        .collect();
    if let Some(name) = &record.run_name {
        tags.push(KeyValueOut {
            key: "mlflow.runName".to_string(),
            value: name.clone(),
        });
    }
    let inputs = if dataset_inputs.is_some() || !record.model_inputs.is_empty() {
        Some(RunInputsOut {
            dataset_inputs: dataset_inputs.unwrap_or_default(),
            model_inputs: record
                .model_inputs
                .iter()
                .map(|id| ModelInputOut {
                    model_id: id.clone(),
                })
                .collect(),
        })
    } else {
        None
    };
    let outputs = (!record.model_outputs.is_empty()).then(|| RunOutputsOut {
        model_outputs: record
            .model_outputs
            .iter()
            .map(|m| ModelOutputOut {
                model_id: m.model_id.clone(),
                step: m.step,
            })
            .collect(),
    });
    RunOut {
        info: RunInfo {
            run_id: record.run_id.clone(),
            experiment_id: record.experiment_id.to_string(),
            status: match record.status {
                RunStatus::Running => "RUNNING",
                RunStatus::Finished => "FINISHED",
                RunStatus::Failed => "FAILED",
                RunStatus::Killed => "KILLED",
            },
            start_time: record.start_time_ms,
            end_time: record.end_time_ms,
            lifecycle_stage: match record.lifecycle {
                RunLifecycle::Active => "active",
                RunLifecycle::Deleted => "deleted",
            },
            artifact_uri: artifacts::run_artifact_uri(record.experiment_id, &record.run_id),
            run_name: record.run_name.clone(),
        },
        data: RunData {
            metrics: record
                .latest_metrics
                .values()
                .map(|p| MetricOut {
                    key: p.key.clone(),
                    value: p.value,
                    timestamp: p.timestamp_ms,
                    step: p.step,
                })
                .collect(),
            params: record
                .params
                .iter()
                .map(|(k, v)| ParamOut {
                    key: k.clone(),
                    value: v.clone(),
                })
                .collect(),
            tags,
        },
        inputs,
        outputs,
    }
}

pub(crate) fn parse_id(id: &str, what: &str) -> MlflowResult<u64> {
    id.trim()
        .parse::<u64>()
        .map_err(|_| MlflowError::invalid(format!("invalid {what} id '{id}'")))
}

pub(crate) fn uuid_like_id() -> String {
    // Server-generated opaque ids: 32 hex chars (mlflow uses UUID4 without
    // dashes; any unique opaque string satisfies the client contract).
    use rand::RngCore;
    let mut bytes = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut bytes);
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub(crate) fn paginate<T>(
    items: Vec<T>,
    max_results: Option<u32>,
    page_token: Option<&str>,
) -> MlflowResult<(Vec<T>, Option<String>)> {
    let limit = max_results.unwrap_or(1_000) as usize;
    if limit == 0 {
        return Err(MlflowError::invalid(
            "max_results must be greater than zero",
        ));
    }
    let offset = match page_token {
        None | Some("") => 0,
        Some(token) => token
            .parse::<usize>()
            .map_err(|_| MlflowError::invalid("invalid page_token"))?,
    };
    if offset > items.len() {
        return Err(MlflowError::invalid("page_token is past the result set"));
    }
    let end = offset.saturating_add(limit).min(items.len());
    let next_page_token = (end < items.len()).then(|| end.to_string());
    Ok((
        items.into_iter().skip(offset).take(limit).collect(),
        next_page_token,
    ))
}

fn tags_map(tags: &[KeyValue]) -> BTreeMap<String, String> {
    tags.iter()
        .map(|t| (t.key.clone(), t.value.clone()))
        .collect()
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn experiments_create(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<ExperimentsCreateRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    if req.name.is_empty() {
        return Err(MlflowError::invalid("experiment name must not be empty"));
    }
    let store = store_for(&tenant, &state)?;
    let record = store
        .create_experiment(
            &req.name,
            req.artifact_location.as_deref(),
            tags_map(&req.tags),
        )
        .await?;
    Ok(Json(serde_json::json!({
        "experiment_id": record.experiment_id.to_string(),
    })))
}

async fn experiments_get(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    MlflowRead(req): MlflowRead<ExperimentIdRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    let id = parse_id(&req.experiment_id, "experiment")?;
    let store = store_for(&tenant, &state)?;
    let record = store.get_experiment(id).await?;
    Ok(Json(serde_json::json!({
        "experiment": experiment_out(&record),
    })))
}

async fn experiments_get_by_name(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    MlflowRead(req): MlflowRead<ExperimentNameRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    if req.name.is_empty() {
        return Err(MlflowError::invalid("experiment_name must not be empty"));
    }
    let store = store_for(&tenant, &state)?;
    let record = store
        .list_experiments(true)
        .await?
        .into_iter()
        .find(|e| e.name == req.name)
        .ok_or_else(|| {
            MlflowError::not_found(format!(
                "Could not find experiment with name '{name}'",
                name = req.name
            ))
        })?;
    Ok(Json(serde_json::json!({
        "experiment": experiment_out(&record),
    })))
}

async fn experiments_search(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<ExperimentsSearchRequest>,
) -> MlflowResult<Json<ExperimentsSearchResponse>> {
    let lifecycle = match req.view_type.as_deref() {
        None | Some("ACTIVE_ONLY") => Some(ExperimentStage::Active),
        Some("DELETED_ONLY") => Some(ExperimentStage::Deleted),
        Some("ALL") => None,
        Some(other) => {
            return Err(MlflowError::invalid(format!(
                "invalid view_type '{other}' (ACTIVE_ONLY | DELETED_ONLY | ALL)"
            )));
        }
    };
    let filters = req
        .filter
        .as_deref()
        .map(filter::parse_experiment_filter)
        .transpose()?
        .unwrap_or_default();
    let include_deleted = lifecycle != Some(ExperimentStage::Active)
        || filters.iter().any(|filter| {
            matches!(
                filter,
                ExperimentFilter::Lifecycle(ExperimentStage::Deleted)
            )
        });
    let store = store_for(&tenant, &state)?;
    let experiments: Vec<ExperimentOut> = store
        .list_experiments(include_deleted)
        .await?
        .iter()
        .filter(|record| lifecycle.is_none_or(|stage| record.stage == stage))
        .filter(|record| filters.iter().all(|predicate| predicate.matches(record)))
        .map(experiment_out)
        .collect();
    let (experiments, next_page_token) =
        paginate(experiments, req.max_results, req.page_token.as_deref())?;
    Ok(Json(ExperimentsSearchResponse {
        experiments,
        next_page_token,
    }))
}

async fn experiments_delete(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<ExperimentIdRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    let id = parse_id(&req.experiment_id, "experiment")?;
    store_for(&tenant, &state)?.delete_experiment(id).await?;
    Ok(Json(serde_json::json!({})))
}

async fn experiments_restore(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<ExperimentIdRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    let id = parse_id(&req.experiment_id, "experiment")?;
    store_for(&tenant, &state)?.restore_experiment(id).await?;
    Ok(Json(serde_json::json!({})))
}

async fn runs_create(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<RunsCreateRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    let experiment_id = parse_id(&req.experiment_id, "experiment")?;
    let run_id = uuid_like_id();
    let store = store_for(&tenant, &state)?;
    let record = store
        .create_run(
            experiment_id,
            &run_id,
            req.run_name.as_deref(),
            None,
            tags_map(&req.tags),
            req.start_time
                .unwrap_or_else(|| chrono::Utc::now().timestamp_millis()),
        )
        .await?;
    Ok(Json(serde_json::json!({ "run": run_out(&record) })))
}

async fn runs_get(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    MlflowRead(req): MlflowRead<RunIdRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    if req.run_id.is_empty() {
        return Err(MlflowError::invalid("run_id must not be empty"));
    }
    let store = store_for(&tenant, &state)?;
    let record = store.get_run(&req.run_id).await?;
    // runs/get embeds dataset inputs (one extra port call); runs/search
    // does not (per-run N+1 — documented in TD-MLOPS-2).
    let datasets = store.dataset_inputs(&req.run_id).await?;
    let dataset_inputs = datasets
        .iter()
        .map(|d| DatasetInputOut {
            dataset: DatasetOut {
                name: d.dataset_name.clone(),
                digest: d.digest.clone(),
            },
            tags: Vec::new(),
        })
        .collect();
    Ok(Json(
        serde_json::json!({ "run": run_out_with(&record, Some(dataset_inputs)) }),
    ))
}

async fn runs_update(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<RunsUpdateRequest>,
) -> MlflowResult<Json<RunInfoResponse>> {
    if req.run_id.is_empty() {
        return Err(MlflowError::invalid("run_id must not be empty"));
    }
    let store = store_for(&tenant, &state)?;
    let transition = match req.status.as_deref() {
        None => None,
        Some("RUNNING") => Some(None),
        Some("FINISHED") => Some(Some(RunStatus::Finished)),
        Some("FAILED") => Some(Some(RunStatus::Failed)),
        Some("KILLED") => Some(Some(RunStatus::Killed)),
        Some(other) => {
            return Err(MlflowError::invalid(format!(
                "unsupported run status '{other}' (slice 2: RUNNING | FINISHED | FAILED | KILLED)"
            )));
        }
    };
    match transition {
        Some(None) => store.reopen_run(&req.run_id).await?,
        Some(Some(status)) => {
            let end_time = req
                .end_time
                .unwrap_or_else(|| chrono::Utc::now().timestamp_millis());
            store.finish_run(&req.run_id, status, end_time).await?;
        }
        None => {}
    }
    let record = store.get_run(&req.run_id).await?;
    Ok(Json(RunInfoResponse {
        run_info: run_out(&record).info,
    }))
}

async fn runs_log_parameter(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<LogParameterRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    store_for(&tenant, &state)?
        .log_param(&req.run_id, &req.key, &req.value)
        .await?;
    Ok(Json(serde_json::json!({})))
}

async fn runs_log_metric(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<LogMetricRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    // MLflow 3.x model-owned metrics ride the SAME endpoint with a
    // model_id: they never touch the run projection or the finished-run
    // freeze (owner-explicit port method, never the run's log_metric).
    if let Some(model_id) = req.model_id.as_deref().filter(|id| !id.is_empty()) {
        if req.run_id.is_empty() {
            return Err(MlflowError::invalid(
                "run_id must not be empty for a model metric",
            ));
        }
        if req.dataset_name.is_some() != req.dataset_digest.is_some() {
            return Err(MlflowError::invalid(
                "dataset_name and dataset_digest must be provided together",
            ));
        }
        let store = store_for(&tenant, &state)?;
        store.get_run(&req.run_id).await?;
        let sample = proximadb_catalog::run_store::ModelMetricPoint {
            point: MetricPoint {
                key: req.key,
                value: req.value,
                timestamp_ms: req.timestamp,
                step: req.step,
            },
            run_id: Some(req.run_id),
            dataset_name: req.dataset_name,
            dataset_digest: req.dataset_digest,
        };
        store.log_model_metric(model_id, sample).await?;
        return Ok(Json(serde_json::json!({})));
    }
    if req.run_id.is_empty() {
        return Err(MlflowError::invalid("run_id must not be empty"));
    }
    let point = MetricPoint {
        key: req.key,
        value: req.value,
        timestamp_ms: req.timestamp,
        step: req.step,
    };
    store_for(&tenant, &state)?
        .log_metric(&req.run_id, point)
        .await?;
    Ok(Json(serde_json::json!({})))
}

/// Run -> logged-model output links (`RunOutputs.model_outputs`).
async fn runs_outputs(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<RunOutputsRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    if req.run_id.is_empty() {
        return Err(MlflowError::invalid("run_id must not be empty"));
    }
    let store = store_for(&tenant, &state)?;
    let mut outputs = Vec::new();
    for model in &req.models {
        if model.model_id.is_empty() {
            return Err(MlflowError::invalid("model_id must not be empty"));
        }
        // Unknown models are a typed 404 — never a silent orphan link.
        store.get_logged_model(&model.model_id, false).await?;
        outputs.push(proximadb_catalog::run_store::ModelOutputRef {
            model_id: model.model_id.clone(),
            step: model.step,
        });
    }
    store.log_run_outputs(&req.run_id, outputs).await?;
    Ok(Json(serde_json::json!({})))
}

/// Legacy model memento (`LogModel`): appended to the
/// `mlflow.logModel.history` tag as a JSON array element — the reference
/// server's client-visible behavior.
async fn runs_log_model(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<RunLogModelRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    if req.run_id.is_empty() {
        return Err(MlflowError::invalid("run_id must not be empty"));
    }
    let memento: serde_json::Value = serde_json::from_str(&req.model_json).map_err(|_| {
        MlflowError::invalid("model_json must be a serialized model memento (JSON)")
    })?;
    let store = store_for(&tenant, &state)?;
    let run = store.get_run(&req.run_id).await?;
    let mut history = run
        .tags
        .get("mlflow.logModel.history")
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
        .and_then(|value| value.as_array().cloned())
        .unwrap_or_default();
    history.push(memento);
    store
        .set_tag(
            &req.run_id,
            "mlflow.logModel.history",
            &serde_json::to_string(&history)
                .map_err(|e| MlflowError::internal(format!("serialize history tag: {e}")))?,
        )
        .await?;
    Ok(Json(serde_json::json!({})))
}

/// Run input links: datasets lower to the existing dataset collection;
/// models to the run's model_inputs.
async fn runs_log_inputs(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<RunLogInputsRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    if req.run_id.is_empty() {
        return Err(MlflowError::invalid("run_id must not be empty"));
    }
    let store = store_for(&tenant, &state)?;
    for dataset in &req.datasets {
        let Some(ds) = &dataset.dataset else {
            return Err(MlflowError::invalid("dataset input requires 'dataset'"));
        };
        if ds.name.is_empty() || ds.digest.is_empty() {
            return Err(MlflowError::invalid(
                "dataset name and digest must not be empty",
            ));
        }
        store
            .log_dataset_input(
                &req.run_id,
                proximadb_catalog::run_store::RunDatasetInput {
                    dataset_name: ds.name.clone(),
                    digest: ds.digest.clone(),
                },
            )
            .await?;
    }
    let model_ids: Vec<String> = req
        .models
        .iter()
        .map(|m| m.model_id.clone())
        .filter(|id| !id.is_empty())
        .collect();
    if !model_ids.is_empty() {
        store.log_run_model_inputs(&req.run_id, model_ids).await?;
    }
    Ok(Json(serde_json::json!({})))
}

// Traces v3 stragglers the client hard-codes to the 2.0 router.

async fn traces_delete_traces(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<DeleteTracesRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    let experiment_id = parse_id(&req.experiment_id, "experiment")?;
    let by_timestamp = req.max_timestamp_millis.is_some();
    let by_ids = !req.request_ids.is_empty();
    if by_timestamp == by_ids {
        return Err(MlflowError::invalid(
            "exactly one of max_timestamp_millis or request_ids is required",
        ));
    }
    if by_ids && req.max_traces.is_some() {
        return Err(MlflowError::invalid(
            "max_traces cannot be used with request_ids",
        ));
    }
    if req.max_traces == Some(0) {
        return Err(MlflowError::invalid("max_traces must be greater than zero"));
    }

    let store = store_for(&tenant, &state)?;
    let mut selected = store
        .list_traces(experiment_id)
        .await?
        .into_iter()
        .filter(|trace| {
            if by_ids {
                req.request_ids.contains(&trace.trace_id)
            } else {
                req.max_timestamp_millis
                    .is_some_and(|timestamp| trace.request_time_ms <= timestamp)
            }
        })
        .collect::<Vec<_>>();
    selected.sort_by(|left, right| {
        (left.request_time_ms, &left.trace_id).cmp(&(right.request_time_ms, &right.trace_id))
    });
    if let Some(max_traces) = req.max_traces {
        selected.truncate(max_traces as usize);
    }

    // Remove artifacts first. If capability-safe I/O fails, metadata remains
    // retryable instead of claiming a trace is gone while exposing its spans.
    for trace in &selected {
        artifacts::delete_directory(
            &state,
            &tenant,
            &format!("{experiment_id}/traces/{}/artifacts", trace.trace_id),
        )
        .await?;
    }
    let selected_ids = selected
        .iter()
        .map(|trace| trace.trace_id.clone())
        .collect::<Vec<_>>();
    let deleted = store
        .delete_traces(experiment_id, None, None, &selected_ids)
        .await?;
    Ok(Json(serde_json::json!({ "traces_deleted": deleted })))
}

async fn traces_set_tag(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    axum::extract::Path(trace_id): axum::extract::Path<String>,
    Json(req): Json<TraceTagRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    if req.key.is_empty() {
        return Err(MlflowError::invalid("key must not be empty"));
    }
    if req.key == traces::ARTIFACT_LOCATION_TAG {
        return Err(MlflowError::invalid(
            "mlflow.artifactLocation is server-owned and cannot be changed",
        ));
    }
    store_for(&tenant, &state)?
        .set_trace_tag(&trace_id, &req.key, &req.value)
        .await?;
    Ok(Json(serde_json::json!({})))
}

async fn traces_delete_tag(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    axum::extract::Path(trace_id): axum::extract::Path<String>,
    Json(req): Json<TraceTagRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    if req.key.is_empty() {
        return Err(MlflowError::invalid("key must not be empty"));
    }
    if req.key == traces::ARTIFACT_LOCATION_TAG {
        return Err(MlflowError::invalid(
            "mlflow.artifactLocation is server-owned and cannot be deleted",
        ));
    }
    store_for(&tenant, &state)?
        .delete_trace_tag(&trace_id, &req.key)
        .await?;
    Ok(Json(serde_json::json!({})))
}

async fn runs_log_batch(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<LogBatchRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    if req.metrics.len() > 1_000 || req.params.len() > 100 || req.tags.len() > 100 {
        return Err(MlflowError::invalid(
            "log-batch caps: 1000 metrics, 100 params, 100 tags",
        ));
    }
    let store = store_for(&tenant, &state)?;
    for p in &req.params {
        store.log_param(&req.run_id, &p.key, &p.value).await?;
    }
    for m in &req.metrics {
        store
            .log_metric(
                &req.run_id,
                MetricPoint {
                    key: m.key.clone(),
                    value: m.value,
                    timestamp_ms: m.timestamp,
                    step: m.step,
                },
            )
            .await?;
    }
    for t in &req.tags {
        store.set_tag(&req.run_id, &t.key, &t.value).await?;
    }
    Ok(Json(serde_json::json!({})))
}

async fn runs_set_tag(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<SetTagRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    store_for(&tenant, &state)?
        .set_tag(&req.run_id, &req.key, &req.value)
        .await?;
    Ok(Json(serde_json::json!({})))
}

async fn metrics_get_history(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    MlflowRead(req): MlflowRead<MetricHistoryRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    if req.run_id.is_empty() || req.key.is_empty() {
        return Err(MlflowError::invalid(
            "run_id and metric_key must not be empty",
        ));
    }
    let store = store_for(&tenant, &state)?;
    let mut history = store.metric_history(&req.run_id, &req.key).await?;
    history.sort_by_key(|p| (p.step, p.timestamp_ms));
    let metrics: Vec<MetricOut> = history
        .iter()
        .map(|p| MetricOut {
            key: p.key.clone(),
            value: p.value,
            timestamp: p.timestamp_ms,
            step: p.step,
        })
        .collect();
    Ok(Json(serde_json::json!({ "metrics": metrics })))
}

async fn runs_delete_tag(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<SetTagRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    store_for(&tenant, &state)?
        .delete_tag(&req.run_id, &req.key)
        .await?;
    Ok(Json(serde_json::json!({})))
}

async fn runs_delete(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<RunIdRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    store_for(&tenant, &state)?.delete_run(&req.run_id).await?;
    Ok(Json(serde_json::json!({})))
}

async fn runs_restore(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<RunIdRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    store_for(&tenant, &state)?.restore_run(&req.run_id).await?;
    Ok(Json(serde_json::json!({})))
}

async fn runs_search(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<RunsSearchRequest>,
) -> MlflowResult<Json<RunsSearchResponse>> {
    if req.experiment_ids.is_empty() {
        return Err(MlflowError::invalid(
            "experiment_ids must list at least one experiment",
        ));
    }
    let lifecycle = match req.run_view_type.as_deref() {
        None | Some("ACTIVE_ONLY") => Some(RunLifecycle::Active),
        Some("DELETED_ONLY") => Some(RunLifecycle::Deleted),
        Some("ALL") => None,
        Some(other) => {
            return Err(MlflowError::invalid(format!(
                "invalid run_view_type '{other}' (ACTIVE_ONLY | DELETED_ONLY | ALL)"
            )));
        }
    };
    let filter = match &req.filter {
        None => None,
        Some(f) => Some(filter::parse_run_filter(f)?),
    };
    // Slice 2 orders by start_time. Any other order_by key is an explicit
    // INVALID_PARAMETER_VALUE — never silently ignored.
    let mut descending = true;
    for clause in &req.order_by {
        let clause = clause.trim();
        let (field, dir) = clause
            .rsplit_once(' ')
            .ok_or_else(|| MlflowError::invalid(format!("invalid order_by '{clause}'")))?;
        match (field.trim(), dir.trim().to_ascii_uppercase().as_str()) {
            ("attributes.start_time", "ASC") => descending = false,
            ("attributes.start_time", "DESC") => descending = true,
            _ => {
                return Err(MlflowError::invalid(format!(
                    "unsupported order_by '{clause}' (slice 2: attributes.start_time ASC|DESC)"
                )));
            }
        }
    }
    let store = store_for(&tenant, &state)?;
    let mut runs: Vec<RunOut> = Vec::new();
    for exp in &req.experiment_ids {
        let id = parse_id(exp, "experiment")?;
        for record in store
            .list_runs(id, lifecycle != Some(RunLifecycle::Active))
            .await?
        {
            if lifecycle.is_some_and(|stage| record.lifecycle != stage) {
                continue;
            }
            if filter
                .as_ref()
                .is_some_and(|clauses| !clauses.iter().all(|clause| clause.matches(&record)))
            {
                continue;
            }
            runs.push(run_out(&record));
        }
    }
    runs.sort_by(|left, right| {
        let time_order = if descending {
            right.info.start_time.cmp(&left.info.start_time)
        } else {
            left.info.start_time.cmp(&right.info.start_time)
        };
        time_order.then_with(|| left.info.run_id.cmp(&right.info.run_id))
    });
    let (runs, next_page_token) = paginate(runs, req.max_results, req.page_token.as_deref())?;
    Ok(Json(RunsSearchResponse {
        runs,
        next_page_token,
    }))
}

// ---------------------------------------------------------------------------
// Run-search filter subset: params.`k` (= | !=) 'v', tags.`k` LIKE '%v%',
// metrics.`k` (< | <= | > | >= | = | !=) number — AND semantics.
// Unparseable filters are INVALID_PARAMETER_VALUE, never silently ignored.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::middleware::tenant::{TenantContext, TenantIdSource};
    use crate::storage::engines::sst::SstEngine;
    use axum::body::Body;
    use axum::http::Request;
    use axum::http::StatusCode;
    use serde_json::Value;
    use tower::ServiceExt;

    fn tenant_ctx(name: &str) -> TenantContext {
        TenantContext::new(name, TenantIdSource::Default)
    }

    async fn test_router(tenant: &str) -> Router {
        // DIP payoff: the wire tests run on the in-memory double — no
        // substrate bootstrap. (The registry side still needs the real
        // catalog service; its tests live in registry.rs.)
        use proximadb_catalog::run_store::conformance_tests::InMemoryRunStoreFactory;
        mlflow_routes()
            .with_state(MlflowState::new(
                Arc::new(InMemoryRunStoreFactory::new()),
                Arc::new(
                    proximadb_catalog::model_registry_service::CatalogModelRegistryService::new(
                        Arc::new(crate::catalog::CatalogManager::new()),
                    ),
                ),
                std::env::temp_dir(),
            ))
            .layer(axum::Extension(tenant_ctx(tenant)))
    }

    async fn post_json(router: &mut Router, path: &str, body: Value) -> (StatusCode, Value) {
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
        let json = if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap_or(Value::Null)
        };
        (status, json)
    }

    async fn get_json(router: &mut Router, path: &str) -> (StatusCode, Value) {
        let response = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(path)
                    .body(Body::empty())
                    .expect("request should build"),
            )
            .await
            .expect("router should answer");
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body should buffer");
        let json = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        (status, json)
    }

    #[tokio::test]
    async fn mlflow_workflow_over_the_wire() {
        let mut router = test_router("default").await;

        // Experiment create -> string id (JavaScript-safe).
        let (status, body) = post_json(
            &mut router,
            "/experiments/create",
            serde_json::json!({"name": "iris", "tags": [{"key": "team", "value": "ml"}]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let experiment_id = body["experiment_id"].as_str().unwrap().to_string();

        // Experiment filters must actually filter, and malformed filters
        // must fail closed instead of silently selecting every active row.
        let (status, other_body) = post_json(
            &mut router,
            "/experiments/create",
            serde_json::json!({"name": "other"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let other_experiment_id = other_body["experiment_id"]
            .as_str()
            .expect("other experiment id")
            .to_string();
        let (status, body) = post_json(
            &mut router,
            "/experiments/search",
            serde_json::json!({"filter": "name = 'iris'"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["experiments"].as_array().unwrap().len(), 1);
        let (status, body) = post_json(
            &mut router,
            "/experiments/search",
            serde_json::json!({"filter": "bogus = 'iris'"}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["error_code"], "INVALID_PARAMETER_VALUE");

        // Search pagination is stable and never silently drops the tail.
        let (status, first_page) = post_json(
            &mut router,
            "/experiments/search",
            serde_json::json!({"max_results": 1}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{first_page}");
        assert_eq!(first_page["experiments"].as_array().unwrap().len(), 1);
        let page_token = first_page["next_page_token"]
            .as_str()
            .expect("first page must advertise the remaining experiment");
        let (status, second_page) = post_json(
            &mut router,
            "/experiments/search",
            serde_json::json!({"max_results": 1, "page_token": page_token}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{second_page}");
        assert_eq!(second_page["experiments"].as_array().unwrap().len(), 1);
        assert_ne!(
            first_page["experiments"][0]["experiment_id"],
            second_page["experiments"][0]["experiment_id"]
        );
        assert!(
            [experiment_id.as_str(), other_experiment_id.as_str()].contains(
                &second_page["experiments"][0]["experiment_id"]
                    .as_str()
                    .expect("experiment id")
            )
        );

        // Run create -> MLflow run shape (info/data split, mlflow.runName).
        let (status, body) = post_json(
            &mut router,
            "/runs/create",
            serde_json::json!({"experiment_id": experiment_id, "run_name": "baseline", "start_time": 1000}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let run_id = body["run"]["info"]["run_id"].as_str().unwrap().to_string();
        assert_eq!(body["run"]["info"]["status"], "RUNNING");
        assert_eq!(body["run"]["info"]["lifecycle_stage"], "active");
        let has_run_name_tag = body["run"]["data"]["tags"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["key"] == "mlflow.runName" && t["value"] == "baseline");
        assert!(has_run_name_tag, "mlflow.runName tag must round-trip");

        let (status, newer_body) = post_json(
            &mut router,
            "/runs/create",
            serde_json::json!({"experiment_id": experiment_id, "run_name": "newer", "start_time": 2000}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{newer_body}");
        let newer_run_id = newer_body["run"]["info"]["run_id"]
            .as_str()
            .expect("newer run id")
            .to_string();

        // Params + metrics + batch.
        post_json(
            &mut router,
            "/runs/log-parameter",
            serde_json::json!({"run_id": run_id, "key": "lr", "value": "0.01"}),
        )
        .await;
        post_json(
            &mut router,
            "/runs/log-metric",
            serde_json::json!({"run_id": run_id, "key": "rmse", "value": 0.9, "timestamp": 1000, "step": 0}),
        )
        .await;
        post_json(
            &mut router,
            "/runs/log-parameter",
            serde_json::json!({"run_id": run_id, "key": "model class", "value": "linear"}),
        )
        .await;
        let (status, _) = post_json(
            &mut router,
            "/runs/log-batch",
            serde_json::json!({"run_id": run_id, "metrics": [{"key": "rmse", "value": 0.7, "timestamp": 2000, "step": 1}], "params": [{"key": "batch-param", "value": "ok"}], "tags": [{"key": "phase", "value": "tune"}]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        post_json(
            &mut router,
            "/runs/log-metric",
            serde_json::json!({"run_id": run_id, "key": "rmse", "value": 0.1, "timestamp": 1500, "step": 2}),
        )
        .await;
        post_json(
            &mut router,
            "/runs/log-metric",
            serde_json::json!({"run_id": run_id, "key": "rmse", "value": 0.8, "timestamp": 2000, "step": 3}),
        )
        .await;

        // Get run: latest metric + params + tags present.
        let (status, body) = get_json(&mut router, &format!("/runs/get?run_id={run_id}")).await;
        assert_eq!(status, StatusCode::OK);
        let metrics = body["run"]["data"]["metrics"].as_array().unwrap();
        assert_eq!(metrics.len(), 1);
        assert_eq!(
            metrics[0]["value"],
            serde_json::json!(0.8),
            "latest projection uses timestamp, then maximum value on a tie"
        );

        // Default run order is newest-first and pagination reaches the tail.
        let (status, first_page) = post_json(
            &mut router,
            "/runs/search",
            serde_json::json!({"experiment_ids": [experiment_id], "max_results": 1}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{first_page}");
        assert_eq!(first_page["runs"][0]["info"]["run_id"], newer_run_id);
        let page_token = first_page["next_page_token"]
            .as_str()
            .expect("first run page must advertise the remaining run");
        let (status, second_page) = post_json(
            &mut router,
            "/runs/search",
            serde_json::json!({"experiment_ids": [experiment_id], "max_results": 1, "page_token": page_token}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{second_page}");
        assert_eq!(second_page["runs"][0]["info"]["run_id"], run_id);

        // Search: matching filter finds the run; NON-MATCHING returns empty
        // (the negative control — an ignored filter must not pass silently).
        let matching = serde_json::json!({
            "experiment_ids": [experiment_id],
            "filter": "params.lr = '0.01' AND metrics.rmse < 0.95"
        });
        let (status, body) = post_json(&mut router, "/runs/search", matching).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["runs"].as_array().unwrap().len(), 1);

        // The documented MLflow filter grammar accepts quoted identifiers
        // and lowercase AND (the form emitted by real clients).
        let quoted_key = serde_json::json!({
            "experiment_ids": [experiment_id],
            "filter": "params.`model class` = 'linear' and tags.phase = 'tune'"
        });
        let (status, body) = post_json(&mut router, "/runs/search", quoted_key).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["runs"].as_array().unwrap().len(), 1);

        let non_matching = serde_json::json!({
            "experiment_ids": [experiment_id],
            "filter": "params.lr = '9.9'"
        });
        let (_, body) = post_json(&mut router, "/runs/search", non_matching).await;
        assert_eq!(
            body["runs"].as_array().unwrap().len(),
            0,
            "non-matching filter must return EMPTY"
        );

        // Unparseable filter -> INVALID_PARAMETER_VALUE (never ignored).
        let (_, body) = post_json(
            &mut router,
            "/runs/search",
            serde_json::json!({"experiment_ids": [experiment_id], "filter": "bogus_field = 1"}),
        )
        .await;
        assert_eq!(body["error_code"], "INVALID_PARAMETER_VALUE");

        // Finish via update; further writes are INVALID_STATE.
        post_json(
            &mut router,
            "/runs/update",
            serde_json::json!({"run_id": run_id, "status": "FINISHED", "end_time": 9000}),
        )
        .await;
        let (status, body) = post_json(
            &mut router,
            "/runs/log-metric",
            serde_json::json!({"run_id": run_id, "key": "rmse", "value": 0.1, "timestamp": 9500, "step": 2}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error_code"], "INVALID_STATE");

        // Terminal states must round-trip instead of all being rewritten as
        // FINISHED by the substrate.
        let (status, body) = post_json(
            &mut router,
            "/runs/update",
            serde_json::json!({"run_id": newer_run_id, "status": "FAILED", "end_time": 9100}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["run_info"]["status"], "FAILED");

        // Reopening is an explicit MLflow transition. Non-finite metric
        // values use proto3 JSON strings and must survive the wire round-trip.
        let (status, body) = post_json(
            &mut router,
            "/runs/update",
            serde_json::json!({"run_id": newer_run_id, "status": "RUNNING"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let (status, body) = post_json(
            &mut router,
            "/runs/log-metric",
            serde_json::json!({"run_id": newer_run_id, "key": "diverged", "value": "Infinity", "timestamp": 9800, "step": 0}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let (status, body) =
            get_json(&mut router, &format!("/runs/get?run_id={newer_run_id}")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["run"]["data"]["metrics"][0]["value"], "Infinity");

        // Duplicate experiment name -> RESOURCE_ALREADY_EXISTS (MLflow code).
        let (status, body) = post_json(
            &mut router,
            "/experiments/create",
            serde_json::json!({"name": "iris"}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error_code"], "RESOURCE_ALREADY_EXISTS");
    }

    #[tokio::test]
    async fn foreign_tenant_probe_is_uniform_not_found() {
        let mut alice = test_router("alice").await;
        let mut bob = test_router("bob").await;

        let (_, body) = post_json(
            &mut alice,
            "/experiments/create",
            serde_json::json!({"name": "private"}),
        )
        .await;
        let experiment_id = body["experiment_id"].as_str().unwrap().to_string();

        // Cross-tenant probe: RESOURCE_DOES_NOT_EXIST — identical to a
        // missing id (no distinguishing 403 that confirms existence).
        let (status, body) = get_json(
            &mut bob,
            &format!("/experiments/get?experiment_id={experiment_id}"),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error_code"], "RESOURCE_DOES_NOT_EXIST");

        let (missing_status, missing_body) =
            get_json(&mut bob, "/experiments/get?experiment_id=999999").await;
        // Uniform in status + error_code (the message echoes the requester's
        // own id, which leaks nothing — the attacker supplied it).
        assert_eq!(missing_status, status);
        assert_eq!(missing_body["error_code"], body["error_code"]);

        // Same name is legal in the other tenant.
        let (status, _) = post_json(
            &mut bob,
            "/experiments/create",
            serde_json::json!({"name": "private"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    #[test]
    fn gate_defaults_off_and_accepts_truthy() {
        assert!(!is_enabled(None), "unset gate must be OFF");
        assert!(is_enabled(Some("1")));
        assert!(is_enabled(Some("true")));
        assert!(is_enabled(Some(" on ")));
        assert!(!is_enabled(Some("false")), "explicit false stays OFF");
        assert!(!is_enabled(Some("")), "empty stays OFF");
    }

    async fn request_json(
        router: &mut Router,
        method: &str,
        path: &str,
        body: Value,
    ) -> (StatusCode, Value) {
        let response = router
            .clone()
            .oneshot(
                Request::builder()
                    .method(method)
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
        let json = if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap_or(Value::Null)
        };
        (status, json)
    }

    #[tokio::test]
    async fn traces_v3_workflow_over_the_wire() {
        let artifact_dir = tempfile::tempdir().unwrap();
        // Both families, mirroring server.rs.
        let mut router = {
            use proximadb_catalog::run_store::conformance_tests::InMemoryRunStoreFactory;
            let state = MlflowState::new(
                Arc::new(InMemoryRunStoreFactory::new()),
                Arc::new(
                    proximadb_catalog::model_registry_service::CatalogModelRegistryService::new(
                        Arc::new(crate::catalog::CatalogManager::new()),
                    ),
                ),
                artifact_dir.path().to_path_buf(),
            );
            let tenant_layer = axum::Extension(tenant_ctx("default"));
            Router::new()
                .nest("/api/2.0/mlflow", mlflow_routes())
                .nest("/api/2.0/mlflow-artifacts", artifacts_router_relative())
                .nest("/api/3.0/mlflow", traces::traces_routes())
                .with_state(state)
                .layer(tenant_layer)
        };

        let (status, body) = post_json(
            &mut router,
            "/api/2.0/mlflow/experiments/create",
            serde_json::json!({"name": "genai-traces"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "create body: {body}");
        let experiment_id = body["experiment_id"].as_str().unwrap().to_string();

        // StartTraceV3: client shape — trace.trace_info only, spans [].
        let (status, body) = post_json(
            &mut router,
            "/api/3.0/mlflow/traces",
            serde_json::json!({
                "trace": {
                    "trace_info": {
                        "trace_id": "tr-abc123def456",
                        "trace_location": {
                            "type": "MLFLOW_EXPERIMENT",
                            "mlflow_experiment": {"experiment_id": experiment_id},
                        },
                        "state": "OK",
                        "request_time": "2026-09-11T10:00:00.500Z",
                        "execution_duration": "1.500s",
                        "request_preview": "q",
                        "response_preview": "a",
                        "trace_metadata": {"mlflow.sourceRun": "run-9"},
                        "tags": {"team": "genai"},
                    },
                    "spans": [],
                }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "start-trace body: {body}");
        // The load-bearing artifactLocation tag must be present.
        assert_eq!(
            body["trace"]["trace_info"]["tags"]["mlflow.artifactLocation"],
            format!("mlflow-artifacts:/{experiment_id}/traces/tr-abc123def456/artifacts"),
        );
        assert_eq!(body["trace"]["trace_info"]["state"], "OK");
        assert_eq!(
            body["trace"]["trace_info"]["execution_duration"], "1.500s",
            "duration round-trips at ms precision"
        );
        assert_eq!(body["trace"]["spans"].as_array().unwrap().len(), 0);

        // The MLflow exporter retries starts after ambiguous transport
        // failures. An equivalent replay is idempotent; a divergent collision
        // with the same caller-owned trace id is still a conflict.
        let (status, retry) = post_json(
            &mut router,
            "/api/3.0/mlflow/traces",
            serde_json::json!({
                "trace": {
                    "trace_info": {
                        "trace_id": "tr-abc123def456",
                        "trace_location": {
                            "type": "MLFLOW_EXPERIMENT",
                            "mlflow_experiment": {"experiment_id": experiment_id},
                        },
                        "state": "OK",
                        "request_time": "2026-09-11T10:00:00.500Z",
                        "execution_duration": "1.500s",
                        "request_preview": "q",
                        "response_preview": "a",
                        "trace_metadata": {"mlflow.sourceRun": "run-9"},
                        "tags": {"team": "genai"},
                    },
                    "spans": [],
                }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "idempotent retry: {retry}");
        assert_eq!(retry["trace"]["trace_info"]["trace_id"], "tr-abc123def456");
        let (status, _) = post_json(
            &mut router,
            "/api/3.0/mlflow/traces",
            serde_json::json!({
                "trace": {"trace_info": {
                    "trace_id": "tr-abc123def456",
                    "trace_location": {
                        "type": "MLFLOW_EXPERIMENT",
                        "mlflow_experiment": {"experiment_id": experiment_id},
                    },
                    "state": "ERROR",
                    "request_time": "2026-09-11T10:00:00.500Z",
                    "execution_duration": "1.500s",
                    "request_preview": "q",
                    "response_preview": "a",
                    "trace_metadata": {"mlflow.sourceRun": "run-9"},
                    "tags": {"team": "genai"},
                }, "spans": []}
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT);

        let trace_artifact_path = format!(
            "/api/2.0/mlflow-artifacts/artifacts/{experiment_id}/traces/tr-abc123def456/artifacts/spans.json"
        );
        let (status, _) = request_json(
            &mut router,
            "PUT",
            &trace_artifact_path,
            serde_json::json!({"spans": [{"name": "root"}]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        // GET by path id (the UI's form).
        let (status, body) = get_json(&mut router, "/api/3.0/mlflow/traces/tr-abc123def456").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body["trace"]["trace_info"]["trace_metadata"]["mlflow.sourceRun"],
            "run-9",
        );

        // GET /traces/get (query-string form).
        let (status, body) = get_json(
            &mut router,
            "/api/3.0/mlflow/traces/get?trace_id=tr-abc123def456",
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["trace"]["trace_info"]["state"], "OK");

        // Unknown trace: 404 with OUR phrasing (never the reference
        // server's "Trace with ID ... not found" the client pattern-matches).
        let (status, body) =
            get_json(&mut router, "/api/3.0/mlflow/traces/get?trace_id=nope").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error_code"], "RESOURCE_DOES_NOT_EXIST");
        assert_ne!(
            body["message"].as_str().unwrap(),
            "Trace with ID 'nope' not found"
        );

        // Search with the state filter.
        let (status, body) = post_json(
            &mut router,
            "/api/3.0/mlflow/traces/search",
            serde_json::json!({
                "locations": [{
                    "type": "MLFLOW_EXPERIMENT",
                    "mlflow_experiment": {"experiment_id": experiment_id},
                }],
                "filter": "status = 'OK' and metadata.`mlflow.sourceRun` = 'run-9'",
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "search body: {body}");
        assert_eq!(body["traces"].as_array().unwrap().len(), 1);

        // Negative filter.
        let (_, body) = post_json(
            &mut router,
            "/api/3.0/mlflow/traces/search",
            serde_json::json!({
                "locations": [{
                    "type": "MLFLOW_EXPERIMENT",
                    "mlflow_experiment": {"experiment_id": experiment_id},
                }],
                "filter": "status = 'ERROR'",
            }),
        )
        .await;
        assert_eq!(body["traces"].as_array().unwrap().len(), 0);

        // order_by is repeated STRING on the wire ("timestamp DESC" — the
        // exact form the client and the UI bundle send).
        let (status, body) = post_json(
            &mut router,
            "/api/3.0/mlflow/traces/search",
            serde_json::json!({
                "locations": [{
                    "type": "MLFLOW_EXPERIMENT",
                    "mlflow_experiment": {"experiment_id": experiment_id},
                }],
                "order_by": ["timestamp DESC"],
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "string order_by body: {body}");
        assert_eq!(body["traces"].as_array().unwrap().len(), 1);
        let (status, _) = post_json(
            &mut router,
            "/api/3.0/mlflow/traces/search",
            serde_json::json!({
                "locations": [{
                    "type": "MLFLOW_EXPERIMENT",
                    "mlflow_experiment": {"experiment_id": experiment_id},
                }],
                "order_by": ["bogus DESC"],
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        // Unknown filter field fails closed.
        let (status, _) = post_json(
            &mut router,
            "/api/3.0/mlflow/traces/search",
            serde_json::json!({
                "locations": [{
                    "type": "MLFLOW_EXPERIMENT",
                    "mlflow_experiment": {"experiment_id": experiment_id},
                }],
                "filter": "bogus_field = 'x'",
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        // A feedback is a oneof: carrying both value and error must fail
        // closed instead of silently choosing one and dropping the other.
        let (status, _) = post_json(
            &mut router,
            "/api/3.0/mlflow/traces/tr-abc123def456/assessments",
            serde_json::json!({
                "assessment": {
                    "assessment_name": "ambiguous",
                    "feedback": {
                        "value": 5,
                        "error": {"error_code": "E", "error_message": "failed"},
                    },
                }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        // Assessments: create (feedback value passthrough), get, PATCH
        // with mask, delete.
        let (status, body) = post_json(
            &mut router,
            "/api/3.0/mlflow/traces/tr-abc123def456/assessments",
            serde_json::json!({
                "assessment": {
                    "assessment_name": "correctness",
                    "source": {"type": "HUMAN", "source_id": "tester"},
                    "feedback": {"value": {"rating": 5}},
                    "rationale": "grounded",
                    "metadata": {"keep": "original", "replace": "old"},
                }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "assessment body: {body}");
        let assessment_id = body["assessment"]["assessment_id"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(assessment_id.starts_with("as-"));
        assert_eq!(
            body["assessment"]["feedback"]["value"]["rating"],
            serde_json::json!(5),
        );

        // MLflow 3.16 protobuf JSON renders FieldMask as a comma-separated
        // string. Metadata merges; unmasked immutable fields are ignored.
        let (status, body) = request_json(
            &mut router,
            "PATCH",
            &format!("/api/3.0/mlflow/traces/tr-abc123def456/assessments/{assessment_id}"),
            serde_json::json!({
                "assessment": {
                    "assessment_name": "correctness-v2",
                    "rationale": "more evidence",
                    "metadata": {"replace": "new", "added": "yes"},
                    "span_id": "must-not-replace",
                    "source": {"type": "CODE", "source_id": "must-not-replace"},
                },
                "update_mask": "assessment_name,rationale,metadata",
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "assessment patch body: {body}");
        assert_eq!(body["assessment"]["assessment_name"], "correctness-v2");
        assert_eq!(body["assessment"]["rationale"], "more evidence");
        assert_eq!(body["assessment"]["metadata"]["keep"], "original");
        assert_eq!(body["assessment"]["metadata"]["replace"], "new");
        assert_eq!(body["assessment"]["metadata"]["added"], "yes");
        assert_eq!(body["assessment"]["span_id"], Value::Null);
        assert_eq!(body["assessment"]["source"]["type"], "HUMAN");
        assert_eq!(
            body["assessment"]["feedback"]["value"]["rating"],
            serde_json::json!(5),
            "masked PATCH must not drop unmasked fields",
        );

        // Identity/source/span and the assessment value kind are immutable.
        for immutable_path in ["span_id", "source", "trace_id", "assessment_id"] {
            let (status, _) = request_json(
                &mut router,
                "PATCH",
                &format!("/api/3.0/mlflow/traces/tr-abc123def456/assessments/{assessment_id}"),
                serde_json::json!({
                    "assessment": {"span_id": "other"},
                    "update_mask": immutable_path,
                }),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "path {immutable_path}");
        }
        let (status, _) = request_json(
            &mut router,
            "PATCH",
            &format!("/api/3.0/mlflow/traces/tr-abc123def456/assessments/{assessment_id}"),
            serde_json::json!({
                "assessment": {"expectation": {"value": "different kind"}},
                "update_mask": "expectation",
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let (status, _) = request_json(
            &mut router,
            "PATCH",
            &format!("/api/3.0/mlflow/traces/tr-abc123def456/assessments/{assessment_id}"),
            serde_json::json!({"assessment": {"assessment_name": "unmasked"}}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        let (status, _) = request_json(
            &mut router,
            "PATCH",
            &format!("/api/3.0/mlflow/traces/tr-abc123def456/assessments/{assessment_id}"),
            serde_json::json!({
                "assessment": {
                    "feedback": {
                        "value": 1,
                        "error": {"error_code": "E", "error_message": "failed"},
                    },
                },
                "update_mask": "feedback",
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        // Trace tags: the 2.0-prefixed stragglers.
        let (status, _) = request_json(
            &mut router,
            "PATCH",
            "/api/2.0/mlflow/traces/tr-abc123def456/tags",
            serde_json::json!({"key": "reviewed", "value": "yes"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let (_, body) = get_json(&mut router, "/api/3.0/mlflow/traces/tr-abc123def456").await;
        assert_eq!(body["trace"]["trace_info"]["tags"]["reviewed"], "yes");

        // mlflow.artifactLocation is server-owned and load-bearing. Neither
        // mutation endpoint may redirect or remove it.
        let (status, _) = request_json(
            &mut router,
            "PATCH",
            "/api/2.0/mlflow/traces/tr-abc123def456/tags",
            serde_json::json!({"key": "mlflow.artifactLocation", "value": "file:///tmp/redirect"}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let (status, _) = request_json(
            &mut router,
            "DELETE",
            "/api/2.0/mlflow/traces/tr-abc123def456/tags",
            serde_json::json!({"key": "mlflow.artifactLocation"}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let (_, body) = get_json(&mut router, "/api/3.0/mlflow/traces/tr-abc123def456").await;
        assert_eq!(
            body["trace"]["trace_info"]["tags"]["mlflow.artifactLocation"],
            format!("mlflow-artifacts:/{experiment_id}/traces/tr-abc123def456/artifacts"),
        );

        let (status, _) = request_json(
            &mut router,
            "DELETE",
            "/api/2.0/mlflow/traces/tr-abc123def456/tags",
            serde_json::json!({"key": "reviewed"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let (_, body) = get_json(&mut router, "/api/3.0/mlflow/traces/tr-abc123def456").await;
        assert!(
            body["trace"]["trace_info"]["tags"]
                .get("reviewed")
                .is_none()
        );

        // Delete requires exactly one selector. Limits are meaningful only
        // for timestamp deletion and must be positive.
        for invalid in [
            serde_json::json!({"experiment_id": experiment_id}),
            serde_json::json!({
                "experiment_id": experiment_id,
                "max_timestamp_millis": 2_000_000_000_000_i64,
                "request_ids": ["tr-abc123def456"],
            }),
            serde_json::json!({
                "experiment_id": experiment_id,
                "request_ids": ["tr-abc123def456"],
                "max_traces": 1,
            }),
            serde_json::json!({
                "experiment_id": experiment_id,
                "max_timestamp_millis": 2_000_000_000_000_i64,
                "max_traces": 0,
            }),
        ] {
            let (status, _) =
                post_json(&mut router, "/api/2.0/mlflow/traces/delete-traces", invalid).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
        }

        // Delete by explicit ids removes metadata and the owner-derived trace
        // artifact directory (including persisted spans).
        let (status, body) = post_json(
            &mut router,
            "/api/2.0/mlflow/traces/delete-traces",
            serde_json::json!({
                "experiment_id": experiment_id,
                "request_ids": ["tr-abc123def456"],
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["traces_deleted"], serde_json::json!(1));
        let (status, _) = get_json(&mut router, "/api/3.0/mlflow/traces/tr-abc123def456").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let (status, _) = get_json(&mut router, &trace_artifact_path).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn logged_models_workflow_over_the_wire() {
        let artifact_dir = tempfile::tempdir().unwrap();
        let mut router = {
            use proximadb_catalog::run_store::conformance_tests::InMemoryRunStoreFactory;
            let state = MlflowState::new(
                Arc::new(InMemoryRunStoreFactory::new()),
                Arc::new(
                    proximadb_catalog::model_registry_service::CatalogModelRegistryService::new(
                        Arc::new(crate::catalog::CatalogManager::new()),
                    ),
                ),
                artifact_dir.path().to_path_buf(),
            );
            Router::new()
                .nest("/api/2.0/mlflow", mlflow_routes())
                .nest("/api/2.0/mlflow-artifacts", artifacts_router_relative())
                .with_state(state)
                .layer(axum::Extension(tenant_ctx("default")))
        };

        let (_, body) = post_json(
            &mut router,
            "/api/2.0/mlflow/experiments/create",
            serde_json::json!({"name": "genai-models"}),
        )
        .await;
        let experiment_id = body["experiment_id"].as_str().unwrap().to_string();
        let (_, body) = post_json(
            &mut router,
            "/api/2.0/mlflow/runs/create",
            serde_json::json!({"experiment_id": experiment_id, "run_name": "train"}),
        )
        .await;
        let run_id = body["run"]["info"]["run_id"].as_str().unwrap().to_string();

        // Divergent duplicates in one create request must be rejected before
        // conversion to a map can silently retain the last value.
        let (status, _) = post_json(
            &mut router,
            "/api/2.0/mlflow/logged-models",
            serde_json::json!({
                "experiment_id": experiment_id,
                "name": "duplicate-param-probe",
                "params": [
                    {"key": "lr", "value": "0.1"},
                    {"key": "lr", "value": "0.2"},
                ],
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        // Create: server-minted m-<hex> id + artifact_uri.
        let (status, body) = post_json(
            &mut router,
            "/api/2.0/mlflow/logged-models",
            serde_json::json!({
                "experiment_id": experiment_id,
                "name": "greet-classifier",
                "model_type": "python_function",
                "source_run_id": run_id,
                "params": [{"key": "lr", "value": "0.1"}],
                "tags": [{"key": "stage", "value": "dev"}],
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "create body: {body}");
        let model_id = body["model"]["info"]["model_id"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(model_id.starts_with("m-"), "model id: {model_id}");
        assert_eq!(body["model"]["info"]["status"], "LOGGED_MODEL_PENDING");
        assert_eq!(
            body["model"]["info"]["artifact_uri"],
            format!("mlflow-artifacts:/{experiment_id}/models/{model_id}/artifacts"),
        );
        assert_eq!(body["model"]["data"]["params"][0]["key"], "lr");

        // Logged-model artifact listings use the same capability-safe,
        // tenant-relative backend as the artifact proxy.
        let artifact_path = format!(
            "/api/2.0/mlflow-artifacts/artifacts/{experiment_id}/models/{model_id}/artifacts/model.bin"
        );
        let (status, _) = request_json(
            &mut router,
            "PUT",
            &artifact_path,
            serde_json::json!({"weights": [1, 2, 3]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let (status, listing) = get_json(
            &mut router,
            &format!("/api/2.0/mlflow/logged-models/{model_id}/artifacts/directories"),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "artifact listing: {listing}");
        assert_eq!(listing["files"][0]["path"], "model.bin");
        assert!(listing["files"][0]["file_size"].as_u64().unwrap() > 0);

        // Search comparators are applied in caller order. A time-only DESC
        // request must not retain the old implicit name-primary ordering.
        let (_, older) = post_json(
            &mut router,
            "/api/2.0/mlflow/logged-models",
            serde_json::json!({"experiment_id": experiment_id, "name": "z-order-probe"}),
        )
        .await;
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        let (_, newer) = post_json(
            &mut router,
            "/api/2.0/mlflow/logged-models",
            serde_json::json!({"experiment_id": experiment_id, "name": "a-order-probe"}),
        )
        .await;
        let (status, ordered) = post_json(
            &mut router,
            "/api/2.0/mlflow/logged-models/search",
            serde_json::json!({
                "experiment_ids": [experiment_id],
                "filter": "name LIKE '%-order-probe'",
                "order_by": [{"field_name": "creation_timestamp_ms", "ascending": false}],
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "ordered search: {ordered}");
        assert_eq!(
            ordered["models"][0]["info"]["model_id"], newer["model"]["info"]["model_id"],
            "creation time must be the primary requested comparator",
        );
        assert_eq!(
            ordered["models"][1]["info"]["model_id"],
            older["model"]["info"]["model_id"],
        );
        for probe in [&older, &newer] {
            let probe_id = probe["model"]["info"]["model_id"].as_str().unwrap();
            let (status, _) = request_json(
                &mut router,
                "DELETE",
                &format!("/api/2.0/mlflow/logged-models/{probe_id}"),
                serde_json::json!({}),
            )
            .await;
            assert_eq!(status, StatusCode::OK);
        }

        // Model metrics retain producing-run lineage. The model response is
        // latest-per-(key,dataset identity), never the full append history.
        for (value, timestamp, dataset_name, dataset_digest) in [
            (0.91, 1_000, "eval", "sha-eval"),
            (0.95, 2_000, "eval", "sha-eval"),
            (0.50, 3_000, "train", "sha-train"),
        ] {
            let (status, _) = post_json(
                &mut router,
                "/api/2.0/mlflow/runs/log-metric",
                serde_json::json!({
                    "run_id": run_id,
                    "model_id": model_id,
                    "key": "accuracy",
                    "value": value,
                    "timestamp": timestamp,
                    "step": 0,
                    "dataset_name": dataset_name,
                    "dataset_digest": dataset_digest,
                }),
            )
            .await;
            assert_eq!(status, StatusCode::OK);
        }
        let (_, body) = get_json(
            &mut router,
            &format!("/api/2.0/mlflow/logged-models/{model_id}?allow_deleted=false"),
        )
        .await;
        let metrics = body["model"]["data"]["metrics"].as_array().unwrap();
        assert_eq!(metrics.len(), 2, "history is projected by dataset identity");
        let eval = metrics
            .iter()
            .find(|metric| metric["dataset_name"] == "eval")
            .unwrap();
        assert_eq!(eval["value"], serde_json::json!(0.95));
        assert_eq!(eval["run_id"], serde_json::json!(run_id));

        // The RUN must NOT have the model metric in its projection (the
        // field is omitted entirely when empty — proto3-JSON).
        let (_, body) = get_json(
            &mut router,
            &format!("/api/2.0/mlflow/runs/get?run_id={run_id}"),
        )
        .await;
        assert!(
            body["run"]["data"]["metrics"]
                .as_array()
                .map_or(true, Vec::is_empty)
        );

        // Additional params + tags.
        let (status, _) = post_json(
            &mut router,
            &format!("/api/2.0/mlflow/logged-models/{model_id}/params"),
            serde_json::json!({"params": [{"key": "layers", "value": "3"}]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let (status, body) = request_json(
            &mut router,
            "PATCH",
            &format!("/api/2.0/mlflow/logged-models/{model_id}/tags"),
            serde_json::json!({"tags": [{"key": "owner", "value": "genai"}]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "set tags body: {body}");
        assert_eq!(body["model"]["info"]["tags"][0]["key"], "owner");

        // Param immutability is honest.
        let (status, body) = post_json(
            &mut router,
            &format!("/api/2.0/mlflow/logged-models/{model_id}/params"),
            serde_json::json!({"params": [{"key": "lr", "value": "0.9"}]}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "diverged param: {body}");

        // delete-tag on an ABSENT key is a 404.
        let (status, _) = request_json(
            &mut router,
            "DELETE",
            &format!("/api/2.0/mlflow/logged-models/{model_id}/tags/absent"),
            serde_json::json!({}),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);

        // Run output link + embedding in runs/get.
        let (status, _) = post_json(
            &mut router,
            "/api/2.0/mlflow/runs/outputs",
            serde_json::json!({"run_id": run_id, "models": [{"model_id": model_id, "step": 0}]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let (_, body) = get_json(
            &mut router,
            &format!("/api/2.0/mlflow/runs/get?run_id={run_id}"),
        )
        .await;
        assert_eq!(
            body["run"]["outputs"]["model_outputs"][0]["model_id"],
            serde_json::json!(model_id),
        );

        // Unknown model in outputs is a typed 404, never a silent link.
        let (status, _) = post_json(
            &mut router,
            "/api/2.0/mlflow/runs/outputs",
            serde_json::json!({"run_id": run_id, "models": [{"model_id": "m-missing", "step": 0}]}),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);

        // Search with filter + finalize.
        let (status, body) = post_json(
            &mut router,
            "/api/2.0/mlflow/logged-models/search",
            serde_json::json!({
                "experiment_ids": [experiment_id],
                "filter": "name LIKE 'greet-%' and tags.`owner` = 'genai'",
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "search body: {body}");
        assert_eq!(body["models"].as_array().unwrap().len(), 1);

        let (status, body) = request_json(
            &mut router,
            "PATCH",
            &format!("/api/2.0/mlflow/logged-models/{model_id}"),
            serde_json::json!({"status": "LOGGED_MODEL_READY"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "finalize body: {body}");
        assert_eq!(body["model"]["info"]["status"], "LOGGED_MODEL_READY");

        // Finalize does NOT freeze params (reference semantics).
        let (status, _) = post_json(
            &mut router,
            &format!("/api/2.0/mlflow/logged-models/{model_id}/params"),
            serde_json::json!({"params": [{"key": "epochs", "value": "10"}]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        // Soft delete: hidden from get+search, visible with allow_deleted.
        let (status, _) = request_json(
            &mut router,
            "DELETE",
            &format!("/api/2.0/mlflow/logged-models/{model_id}"),
            serde_json::json!({}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let (status, _) = get_json(
            &mut router,
            &format!("/api/2.0/mlflow/logged-models/{model_id}"),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let (status, body) = get_json(
            &mut router,
            &format!("/api/2.0/mlflow/logged-models/{model_id}?allow_deleted=true"),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["model"]["info"]["status"], "LOGGED_MODEL_READY");
        let (_, body) = post_json(
            &mut router,
            "/api/2.0/mlflow/logged-models/search",
            serde_json::json!({"experiment_ids": [experiment_id]}),
        )
        .await;
        assert_eq!(body["models"].as_array().unwrap().len(), 0);
    }
}
