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
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use proximadb_catalog::run_store::{
    ExperimentRecord, ExperimentStage, MetricPoint, RunLifecycle, RunRecord, RunStatus, RunStore,
    RunStoreError,
};
use serde::{Deserialize, Serialize};

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
pub mod registry;

/// The artifacts PROXY lives at /api/2.0/mlflow-artifacts (a sibling of
/// /api/2.0/mlflow, not under it) — mounted separately in server.rs.
pub fn artifacts_router() -> Router<MlflowState> {
    artifacts::artifacts_routes()
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
        .route(
            "/metrics/get-history",
            get(metrics_get_history).post(metrics_get_history),
        )
        .merge(registry::registry_routes())
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

fn store_for(
    tenant: &TenantContext,
    state: &MlflowState,
) -> MlflowResult<std::sync::Arc<dyn RunStore>> {
    state
        .run_store
        .store_for(&tenant.tenant_id)
        .map_err(|e| MlflowError::internal(e.to_string()))
}

// ---------------------------------------------------------------------------
// MLflow REST DTOs (serde snake_case, string ids — JavaScript-safe)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct KeyValue {
    key: String,
    value: String,
}

#[derive(Serialize)]
struct KeyValueOut {
    key: String,
    value: String,
}

#[derive(Default, Deserialize)]
struct ExperimentsCreateRequest {
    name: String,
    #[serde(default)]
    artifact_location: Option<String>,
    #[serde(default)]
    tags: Vec<KeyValue>,
}

#[derive(Serialize)]
struct ExperimentOut {
    experiment_id: String,
    name: String,
    artifact_uri: String,
    lifecycle_stage: &'static str,
    creation_time: i64,
    last_update_time: i64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tags: Vec<KeyValueOut>,
}

#[derive(Default, Deserialize)]
struct ExperimentNameRequest {
    /// Proto field name is `experiment_name`; older JSON bodies use `name`.
    #[serde(default, alias = "experiment_name")]
    name: String,
}

#[derive(Default, Deserialize)]
struct ExperimentIdRequest {
    #[serde(default)]
    experiment_id: String,
}

#[derive(Default, Deserialize)]
struct ExperimentsSearchRequest {
    #[serde(default)]
    max_results: Option<u32>,
    #[serde(default)]
    filter: Option<String>,
    #[serde(default)]
    view_type: Option<String>,
    #[serde(default)]
    page_token: Option<String>,
}

#[derive(Serialize)]
struct ExperimentsSearchResponse {
    experiments: Vec<ExperimentOut>,
    #[serde(skip_serializing_if = "Option::is_none")]
    next_page_token: Option<String>,
}

#[derive(Default, Deserialize)]
struct RunsCreateRequest {
    experiment_id: String,
    #[serde(default)]
    start_time: Option<i64>,
    #[serde(default)]
    run_name: Option<String>,
    #[serde(default)]
    tags: Vec<KeyValue>,
}

#[derive(Serialize)]
struct MetricOut {
    key: String,
    #[serde(serialize_with = "serialize_metric_value")]
    value: f64,
    timestamp: i64,
    step: i64,
}

fn serialize_metric_value<S: serde::Serializer>(
    value: &f64,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    if value.is_nan() {
        serializer.serialize_str("NaN")
    } else if *value == f64::INFINITY {
        serializer.serialize_str("Infinity")
    } else if *value == f64::NEG_INFINITY {
        serializer.serialize_str("-Infinity")
    } else {
        serializer.serialize_f64(*value)
    }
}

#[derive(Serialize)]
struct ParamOut {
    key: String,
    value: String,
}

#[derive(Serialize)]
struct RunData {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    metrics: Vec<MetricOut>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    params: Vec<ParamOut>,
    tags: Vec<KeyValueOut>,
}

#[derive(Serialize)]
struct RunInfo {
    run_id: String,
    experiment_id: String,
    status: &'static str,
    start_time: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    end_time: Option<i64>,
    lifecycle_stage: &'static str,
    artifact_uri: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    run_name: Option<String>,
}

#[derive(Serialize)]
struct RunOut {
    info: RunInfo,
    data: RunData,
}

#[derive(Default, Deserialize)]
struct RunIdRequest {
    #[serde(default)]
    run_id: String,
}

#[derive(Default, Deserialize)]
struct RunsUpdateRequest {
    #[serde(default)]
    run_id: String,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    end_time: Option<i64>,
}

#[derive(Serialize)]
struct RunInfoResponse {
    run_info: RunInfo,
}

#[derive(Default, Deserialize)]
struct LogParameterRequest {
    #[serde(default)]
    run_id: String,
    #[serde(default)]
    key: String,
    #[serde(default)]
    value: String,
}

#[derive(Default, Deserialize)]
struct MetricInput {
    #[serde(default)]
    key: String,
    #[serde(default, deserialize_with = "deserialize_metric_value")]
    value: f64,
    #[serde(default)]
    timestamp: i64,
    #[serde(default)]
    step: i64,
}

#[derive(Default, Deserialize)]
struct LogMetricRequest {
    #[serde(default)]
    run_id: String,
    #[serde(default)]
    key: String,
    #[serde(default, deserialize_with = "deserialize_metric_value")]
    value: f64,
    #[serde(default)]
    timestamp: i64,
    #[serde(default)]
    step: i64,
}

fn deserialize_metric_value<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<f64, D::Error> {
    use serde::de::Error as _;
    let raw = serde_json::Value::deserialize(deserializer)?;
    match raw {
        serde_json::Value::Number(number) => number
            .as_f64()
            .ok_or_else(|| D::Error::custom("metric value out of f64 range")),
        serde_json::Value::String(value) => match value.as_str() {
            "NaN" => Ok(f64::NAN),
            "Infinity" => Ok(f64::INFINITY),
            "-Infinity" => Ok(f64::NEG_INFINITY),
            other => other
                .parse::<f64>()
                .map_err(|_| D::Error::custom(format!("invalid metric value '{other}'"))),
        },
        other => Err(D::Error::custom(format!(
            "metric value must be a number or numeric string, got {other}"
        ))),
    }
}

#[derive(Default, Deserialize)]
struct LogBatchRequest {
    #[serde(default)]
    run_id: String,
    #[serde(default)]
    metrics: Vec<MetricInput>,
    #[serde(default)]
    params: Vec<LogParameterRequest>,
    #[serde(default)]
    tags: Vec<KeyValue>,
}

#[derive(Default, Deserialize)]
struct SetTagRequest {
    #[serde(default)]
    run_id: String,
    #[serde(default)]
    key: String,
    #[serde(default)]
    value: String,
}

#[derive(Default, Deserialize)]
struct RunsSearchRequest {
    #[serde(default)]
    experiment_ids: Vec<String>,
    #[serde(default)]
    filter: Option<String>,
    #[serde(default)]
    run_view_type: Option<String>,
    #[serde(default)]
    max_results: Option<u32>,
    #[serde(default)]
    order_by: Vec<String>,
    #[serde(default)]
    page_token: Option<String>,
}

#[derive(Serialize)]
struct RunsSearchResponse {
    runs: Vec<RunOut>,
    #[serde(skip_serializing_if = "Option::is_none")]
    next_page_token: Option<String>,
}

// ---------------------------------------------------------------------------
// Error envelope — MLflow native
// ---------------------------------------------------------------------------

pub(crate) struct MlflowError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl MlflowError {
    pub(crate) fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            code: "RESOURCE_DOES_NOT_EXIST",
            message: message.into(),
        }
    }

    pub(crate) fn exists(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "RESOURCE_ALREADY_EXISTS",
            message: message.into(),
        }
    }

    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "INVALID_PARAMETER_VALUE",
            message: message.into(),
        }
    }

    pub(crate) fn invalid_state(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "INVALID_STATE",
            message: message.into(),
        }
    }

    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "INTERNAL_ERROR",
            message: message.into(),
        }
    }
}

impl IntoResponse for MlflowError {
    fn into_response(self) -> Response {
        #[derive(Serialize)]
        struct Body {
            error_code: &'static str,
            message: String,
        }
        (
            self.status,
            Json(Body {
                error_code: self.code,
                message: self.message,
            }),
        )
            .into_response()
    }
}

impl From<RunStoreError> for MlflowError {
    fn from(e: RunStoreError) -> Self {
        match e {
            RunStoreError::UnknownExperiment { experiment_id } => MlflowError::not_found(format!(
                "Could not find experiment with ID '{experiment_id}'"
            )),
            RunStoreError::UnknownRun { run_id } => {
                MlflowError::not_found(format!("Could not find run with ID '{run_id}'"))
            }
            RunStoreError::ExperimentNameConflict { name } => {
                MlflowError::exists(format!("Experiment '{name}' already exists"))
            }
            RunStoreError::RunIdConflict { run_id } => {
                MlflowError::exists(format!("Run '{run_id}' already exists"))
            }
            RunStoreError::ParamImmutable { key, run_id } => MlflowError::invalid(format!(
                "Param '{key}' on run '{run_id}' is locked to its first value (MLflow params are immutable; re-logging a different value is rejected)"
            )),
            RunStoreError::RunFinished { run_id } => MlflowError::invalid_state(format!(
                "Run '{run_id}' is finished and no longer accepts param/metric writes"
            )),
            RunStoreError::ExperimentDeleted { experiment_id } => MlflowError::invalid_state(
                format!("Experiment '{experiment_id}' is deleted (restore it first)"),
            ),
            RunStoreError::Empty { field } => {
                MlflowError::invalid(format!("{field} must not be empty"))
            }
            RunStoreError::NotTerminal => {
                MlflowError::internal("tracking store requires a terminal run status")
            }
            RunStoreError::Internal { message } => MlflowError::internal(message),
        }
    }
}

pub(crate) type MlflowResult<T> = Result<T, MlflowError>;

// ---------------------------------------------------------------------------
// Lowering helpers
// ---------------------------------------------------------------------------

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
    }
}

fn parse_id(id: &str, what: &str) -> MlflowResult<u64> {
    id.trim()
        .parse::<u64>()
        .map_err(|_| MlflowError::invalid(format!("invalid {what} id '{id}'")))
}

fn tags_map(tags: &[KeyValue]) -> BTreeMap<String, String> {
    tags.iter()
        .map(|t| (t.key.clone(), t.value.clone()))
        .collect()
}

fn paginate<T>(
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
        .map(parse_experiment_filter)
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

enum ExperimentFilter {
    NameEq(String),
    NameLike(String),
    TagEq(String, String),
    Lifecycle(ExperimentStage),
}

impl ExperimentFilter {
    fn matches(&self, experiment: &ExperimentRecord) -> bool {
        match self {
            Self::NameEq(name) => experiment.name == *name,
            Self::NameLike(pattern) => like_match(&experiment.name, pattern),
            Self::TagEq(key, value) => experiment.tags.get(key) == Some(value),
            Self::Lifecycle(stage) => experiment.stage == *stage,
        }
    }
}

fn parse_experiment_filter(filter: &str) -> MlflowResult<Vec<ExperimentFilter>> {
    let mut predicates = Vec::new();
    for clause in split_filter_clauses(filter)? {
        let clause = clause.trim();
        if let Some(rest) = clause.strip_prefix("tags.") {
            let (key, op, value) = parse_filter_parts(rest, true)?;
            if op != "=" {
                return Err(MlflowError::invalid(
                    "experiment tag filter supports = only",
                ));
            }
            predicates.push(ExperimentFilter::TagEq(key, value));
            continue;
        }
        let (field, op, value) = parse_filter_parts(clause, true)?;
        match field.to_ascii_lowercase().as_str() {
            "name" | "attributes.name" => match op.as_str() {
                "=" => predicates.push(ExperimentFilter::NameEq(value)),
                "LIKE" => predicates.push(ExperimentFilter::NameLike(value)),
                _ => {
                    return Err(MlflowError::invalid(
                        "experiment name filter supports = / LIKE",
                    ));
                }
            },
            "attributes.lifecycle_stage" if op == "=" => {
                let stage = match value.to_ascii_lowercase().as_str() {
                    "active" => ExperimentStage::Active,
                    "deleted" => ExperimentStage::Deleted,
                    _ => {
                        return Err(MlflowError::invalid(format!(
                            "invalid experiment lifecycle_stage '{value}'"
                        )));
                    }
                };
                predicates.push(ExperimentFilter::Lifecycle(stage));
            }
            other => {
                return Err(MlflowError::invalid(format!(
                    "unsupported experiment filter field '{other}'"
                )));
            }
        }
    }
    Ok(predicates)
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

fn uuid_like_id() -> String {
    // Server-generated opaque ids: 32 hex chars (mlflow uses UUID4 without
    // dashes; any unique opaque string satisfies the client contract).
    use rand::RngCore;
    let mut bytes = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut bytes);
    bytes.iter().map(|b| format!("{b:02x}")).collect()
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
    Ok(Json(serde_json::json!({ "run": run_out(&record) })))
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

#[derive(Default, Deserialize)]
struct MetricHistoryRequest {
    #[serde(default)]
    run_id: String,
    #[serde(default, alias = "metric_key")]
    key: String,
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
        Some(f) => Some(parse_run_filter(f)?),
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

enum FieldFilter {
    ParamEq(String, String),
    ParamNe(String, String),
    ParamLike(String, String),
    TagEq(String, String),
    TagNe(String, String),
    TagLike(String, String),
    MetricCmp(String, f64, fn(f64, f64) -> bool),
}

impl FieldFilter {
    fn matches(&self, run: &RunRecord) -> bool {
        match self {
            FieldFilter::ParamEq(k, v) => run.params.get(k) == Some(v),
            FieldFilter::ParamNe(k, v) => run.params.get(k) != Some(v),
            FieldFilter::ParamLike(k, pattern) => run
                .params
                .get(k)
                .is_some_and(|actual| like_match(actual, pattern)),
            FieldFilter::TagEq(k, v) => run.tags.get(k) == Some(v),
            FieldFilter::TagNe(k, v) => run.tags.get(k) != Some(v),
            FieldFilter::TagLike(k, pattern) => match run.tags.get(k) {
                Some(actual) => like_match(actual, pattern),
                None => false,
            },
            FieldFilter::MetricCmp(k, v, cmp) => run
                .latest_metrics
                .get(k)
                .is_some_and(|point| cmp(point.value, *v)),
        }
    }
}

pub(crate) fn like_match(value: &str, pattern: &str) -> bool {
    // SQL LIKE: % = any run, _ = one char. Case-sensitive (MLflow is).
    let mut regex = String::from("^");
    for c in pattern.chars() {
        match c {
            '%' => regex.push_str(".*"),
            '_' => regex.push('.'),
            c => regex.push_str(&regex::escape(&c.to_string())),
        }
    }
    regex.push('$');
    regex::Regex::new(&regex)
        .map(|re| re.is_match(value))
        .unwrap_or(false)
}

fn parse_run_filter(filter: &str) -> MlflowResult<Vec<FieldFilter>> {
    let mut out = Vec::new();
    for clause in split_filter_clauses(filter)? {
        let clause = clause.trim();
        let field = if let Some(rest) = clause.strip_prefix("metrics.") {
            let (key, op, value) = parse_filter_parts(rest, false)?;
            let number: f64 = value.parse().map_err(|_| {
                MlflowError::invalid(format!("metric filter needs a number, got {value}"))
            })?;
            if !number.is_finite() {
                return Err(MlflowError::invalid("metric filter value must be finite"));
            }
            let cmp = cmp_fn(&op)?;
            FieldFilter::MetricCmp(key, number, cmp)
        } else if let Some(rest) = clause.strip_prefix("params.") {
            let (key, op, value) = parse_filter_parts(rest, true)?;
            match op.as_str() {
                "=" => FieldFilter::ParamEq(key, value),
                "!=" => FieldFilter::ParamNe(key, value),
                "LIKE" => FieldFilter::ParamLike(key, value),
                other => {
                    return Err(MlflowError::invalid(format!(
                        "params filter supports = / != / LIKE, got '{other}'"
                    )));
                }
            }
        } else if let Some(rest) = clause.strip_prefix("tags.") {
            let (key, op, value) = parse_filter_parts(rest, true)?;
            match op.as_str() {
                "=" => FieldFilter::TagEq(key, value),
                "!=" => FieldFilter::TagNe(key, value),
                "LIKE" => FieldFilter::TagLike(key, value),
                other => {
                    return Err(MlflowError::invalid(format!(
                        "tags filter supports = / != / LIKE, got '{other}'"
                    )));
                }
            }
        } else {
            return Err(MlflowError::invalid(format!(
                "unsupported filter clause '{clause}' (slice 2: params. / metrics. / tags. clauses ANDed together)"
            )));
        };
        out.push(field);
    }
    if out.is_empty() {
        return Err(MlflowError::invalid("empty filter"));
    }
    Ok(out)
}

pub(crate) fn split_filter_clauses(filter: &str) -> MlflowResult<Vec<&str>> {
    let bytes = filter.as_bytes();
    let mut clauses = Vec::new();
    let mut quote = None;
    let mut start = 0;
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if let Some(active_quote) = quote {
            if byte == active_quote {
                if bytes.get(index + 1) == Some(&active_quote) {
                    index += 2;
                    continue;
                }
                quote = None;
            }
            index += 1;
            continue;
        }
        if matches!(byte, b'\'' | b'`' | b'"') {
            quote = Some(byte);
            index += 1;
            continue;
        }
        let is_and = bytes
            .get(index..index.saturating_add(3))
            .is_some_and(|word| word.eq_ignore_ascii_case(b"and"));
        let left_space = index > start && bytes[index - 1].is_ascii_whitespace();
        let right_space = bytes
            .get(index + 3)
            .is_some_and(|byte| byte.is_ascii_whitespace());
        if is_and && left_space && right_space {
            clauses.push(filter[start..index].trim());
            index += 3;
            start = index;
            continue;
        }
        index += 1;
    }
    if quote.is_some() {
        return Err(MlflowError::invalid("unterminated quote in filter"));
    }
    clauses.push(filter[start..].trim());
    if clauses.iter().any(|clause| clause.is_empty()) {
        return Err(MlflowError::invalid("empty filter clause"));
    }
    Ok(clauses)
}

pub(crate) fn parse_filter_parts(
    rest: &str,
    string_value: bool,
) -> MlflowResult<(String, String, String)> {
    let identifier = r#"(?:`(?:``|[^`])+`|"(?:""|[^"])+"|[^\s!<>=]+)"#;
    let pattern = if string_value {
        format!(r#"(?i)^\s*({identifier})\s*(!=|=|LIKE)\s*'((?:''|[^'])*)'\s*$"#)
    } else {
        format!(r#"(?i)^\s*({identifier})\s*(!=|<=|>=|<|>|=)\s*([^\s]+)\s*$"#)
    };
    let regex = regex::Regex::new(&pattern).map_err(|error| {
        MlflowError::internal(format!("invalid built-in filter regex: {error}"))
    })?;
    let captures = regex
        .captures(rest)
        .ok_or_else(|| MlflowError::invalid(format!("invalid filter clause '{rest}'")))?;
    let key = captures
        .get(1)
        .map(|capture| decode_identifier(capture.as_str()))
        .ok_or_else(|| MlflowError::invalid("filter is missing an identifier"))?;
    let op = captures
        .get(2)
        .map(|capture| capture.as_str().to_ascii_uppercase())
        .ok_or_else(|| MlflowError::invalid("filter is missing an operator"))?;
    let value = captures
        .get(3)
        .map(|capture| capture.as_str().replace("''", "'"))
        .ok_or_else(|| MlflowError::invalid("filter is missing a value"))?;
    Ok((key, op, value))
}

fn decode_identifier(raw: &str) -> String {
    if let Some(inner) = raw
        .strip_prefix('`')
        .and_then(|value| value.strip_suffix('`'))
    {
        inner.replace("``", "`")
    } else if let Some(inner) = raw
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
    {
        inner.replace("\"\"", "\"")
    } else {
        raw.to_string()
    }
}

fn cmp_fn(op: &str) -> MlflowResult<fn(f64, f64) -> bool> {
    Ok(match op {
        "<" => |a, b| a < b,
        "<=" => |a, b| a <= b,
        ">" => |a, b| a > b,
        ">=" => |a, b| a >= b,
        "=" => |a, b| a == b,
        "!=" => |a, b| a != b,
        other => {
            return Err(MlflowError::invalid(format!(
                "unsupported comparison '{other}'"
            )));
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::middleware::tenant::{TenantContext, TenantIdSource};
    use crate::storage::engines::sst::SstEngine;
    use axum::body::Body;
    use axum::http::Request;
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
}
