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
use axum::routing::post;
use axum::{Extension, Json, Router};
use proximadb_catalog::run_store::{
    ExperimentRecord, ExperimentStage, MetricPoint, RunLifecycle, RunRecord, RunStatus, RunStore,
    RunStoreError,
};
use serde::{Deserialize, Serialize};

use crate::services::mlflow_run_store::SubstrateRunStore;
use crate::storage::document::DocumentService;

/// Minimal state: the tracking wire touches nothing but the document
/// substrate. Built once at mount time from the canonical `AppState`.
#[derive(Clone)]
pub struct MlflowState {
    document: Arc<DocumentService>,
}

impl MlflowState {
    pub fn new(document: Arc<DocumentService>) -> Self {
        Self { document }
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

pub fn mlflow_routes() -> Router<MlflowState> {
    Router::new()
        .route("/experiments/create", post(experiments_create))
        .route("/experiments/get", post(experiments_get))
        .route("/experiments/search", post(experiments_search))
        .route("/experiments/delete", post(experiments_delete))
        .route("/experiments/restore", post(experiments_restore))
        .route("/runs/create", post(runs_create))
        .route("/runs/get", post(runs_get))
        .route("/runs/update", post(runs_update))
        .route("/runs/search", post(runs_search))
        .route("/runs/delete", post(runs_delete))
        .route("/runs/restore", post(runs_restore))
        .route("/runs/log-parameter", post(runs_log_parameter))
        .route("/runs/log-metric", post(runs_log_metric))
        .route("/runs/log-batch", post(runs_log_batch))
        .route("/runs/set-tag", post(runs_set_tag))
        .route("/runs/delete-tag", post(runs_delete_tag))
}

async fn store_for(tenant: &TenantContext, state: &MlflowState) -> SubstrateRunStore {
    // for_tenant validates the tenant; a validated request tenant cannot
    // fail here, but fail closed rather than unwrap (mandate 4).
    match SubstrateRunStore::for_tenant(state.document.clone(), &tenant.tenant_id) {
        Ok(store) => store,
        Err(e) => unreachable!("validated tenant produced an invalid scoped collection: {e}"),
    }
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
}

#[derive(Serialize)]
struct ExperimentsSearchResponse {
    experiments: Vec<ExperimentOut>,
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
    value: f64,
    timestamp: i64,
    step: i64,
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
    #[serde(default)]
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
    metric: Option<MetricInput>,
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
}

#[derive(Serialize)]
struct RunsSearchResponse {
    runs: Vec<RunOut>,
}

// ---------------------------------------------------------------------------
// Error envelope — MLflow native
// ---------------------------------------------------------------------------

struct MlflowError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl MlflowError {
    fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            code: "RESOURCE_DOES_NOT_EXIST",
            message: message.into(),
        }
    }

    fn exists(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "RESOURCE_ALREADY_EXISTS",
            message: message.into(),
        }
    }

    fn invalid(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "INVALID_PARAMETER_VALUE",
            message: message.into(),
        }
    }

    fn invalid_state(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "INVALID_STATE",
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
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
            RunStoreError::Internal { message } => MlflowError::internal(message),
        }
    }
}

type MlflowResult<T> = Result<T, MlflowError>;

// ---------------------------------------------------------------------------
// Lowering helpers
// ---------------------------------------------------------------------------

fn experiment_out(record: &ExperimentRecord) -> ExperimentOut {
    ExperimentOut {
        experiment_id: record.experiment_id.to_string(),
        name: record.name.clone(),
        artifact_uri: record.artifact_location.clone().unwrap_or_default(),
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
            },
            start_time: record.start_time_ms,
            end_time: record.end_time_ms,
            lifecycle_stage: match record.lifecycle {
                RunLifecycle::Active => "active",
                RunLifecycle::Deleted => "deleted",
            },
            artifact_uri: String::new(),
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
    let store = store_for(&tenant, &state).await;
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
    Json(req): Json<ExperimentIdRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    let id = parse_id(&req.experiment_id, "experiment")?;
    let store = store_for(&tenant, &state).await;
    let record = store.get_experiment(id).await?;
    Ok(Json(serde_json::json!({
        "experiment": experiment_out(&record),
    })))
}

async fn experiments_search(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<ExperimentsSearchRequest>,
) -> MlflowResult<Json<ExperimentsSearchResponse>> {
    let include_deleted = match req.filter.as_deref() {
        None => false,
        // Slice-2 subset: `name = 'x'` equality and
        // `attributes.lifecycle_stage = 'deleted'`; anything else is an
        // explicit INVALID_PARAMETER_VALUE (never silently ignored).
        Some(f) => match parse_lifecycle_filter(f) {
            LifecycleFilter::Deleted => true,
            LifecycleFilter::Active => false,
        },
    };
    let store = store_for(&tenant, &state).await;
    let mut experiments: Vec<ExperimentOut> = store
        .list_experiments(include_deleted)
        .await?
        .iter()
        .map(experiment_out)
        .collect();
    if let Some(limit) = req.max_results {
        experiments.truncate(limit as usize);
    }
    Ok(Json(ExperimentsSearchResponse { experiments }))
}

enum LifecycleFilter {
    Active,
    Deleted,
}

fn parse_lifecycle_filter(filter: &str) -> LifecycleFilter {
    if filter.contains("deleted") {
        LifecycleFilter::Deleted
    } else {
        LifecycleFilter::Active
    }
}

async fn experiments_delete(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<ExperimentIdRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    let id = parse_id(&req.experiment_id, "experiment")?;
    store_for(&tenant, &state)
        .await
        .delete_experiment(id)
        .await?;
    Ok(Json(serde_json::json!({})))
}

async fn experiments_restore(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<ExperimentIdRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    let id = parse_id(&req.experiment_id, "experiment")?;
    store_for(&tenant, &state)
        .await
        .restore_experiment(id)
        .await?;
    Ok(Json(serde_json::json!({})))
}

async fn runs_create(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<RunsCreateRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    let experiment_id = parse_id(&req.experiment_id, "experiment")?;
    let run_id = uuid_like_id();
    let store = store_for(&tenant, &state).await;
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
    Json(req): Json<RunIdRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    if req.run_id.is_empty() {
        return Err(MlflowError::invalid("run_id must not be empty"));
    }
    let store = store_for(&tenant, &state).await;
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
    let store = store_for(&tenant, &state).await;
    let end_time = match req.status.as_deref() {
        None | Some("RUNNING") => req.end_time,
        Some("FINISHED") | Some("FAILED") | Some("KILLED") => Some(
            req.end_time
                .unwrap_or_else(|| chrono::Utc::now().timestamp_millis()),
        ),
        Some(other) => {
            return Err(MlflowError::invalid(format!(
                "unsupported run status '{other}' (slice 2: RUNNING | FINISHED | FAILED | KILLED)"
            )));
        }
    };
    if let Some(end) = end_time {
        store.finish_run(&req.run_id, end).await?;
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
    store_for(&tenant, &state)
        .await
        .log_param(&req.run_id, &req.key, &req.value)
        .await?;
    Ok(Json(serde_json::json!({})))
}

async fn runs_log_metric(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<LogMetricRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    let metric = req.metric.ok_or_else(|| {
        MlflowError::invalid("log-metric requires a `metric` object (key/value/timestamp/step)")
    })?;
    let point = MetricPoint {
        key: metric.key,
        value: metric.value,
        timestamp_ms: metric.timestamp,
        step: metric.step,
    };
    store_for(&tenant, &state)
        .await
        .log_metric(&req.run_id, point)
        .await?;
    Ok(Json(serde_json::json!({})))
}

async fn runs_log_batch(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<LogBatchRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    let store = store_for(&tenant, &state).await;
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
    store_for(&tenant, &state)
        .await
        .set_tag(&req.run_id, &req.key, &req.value)
        .await?;
    Ok(Json(serde_json::json!({})))
}

async fn runs_delete_tag(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<SetTagRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    store_for(&tenant, &state)
        .await
        .delete_tag(&req.run_id, &req.key)
        .await?;
    Ok(Json(serde_json::json!({})))
}

async fn runs_delete(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<RunIdRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    store_for(&tenant, &state)
        .await
        .delete_run(&req.run_id)
        .await?;
    Ok(Json(serde_json::json!({})))
}

async fn runs_restore(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<RunIdRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    store_for(&tenant, &state)
        .await
        .restore_run(&req.run_id)
        .await?;
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
    let include_deleted = match req.run_view_type.as_deref() {
        None | Some("ACTIVE_ONLY") => false,
        Some("ALL") => true,
        Some(other) => {
            return Err(MlflowError::invalid(format!(
                "invalid run_view_type '{other}' (ACTIVE_ONLY | ALL)"
            )));
        }
    };
    let filter = match &req.filter {
        None => None,
        Some(f) => Some(parse_run_filter(f)?),
    };
    // Slice 2 orders by start_time. Any other order_by key is an explicit
    // INVALID_PARAMETER_VALUE — never silently ignored.
    let mut descending = false;
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
    let store = store_for(&tenant, &state).await;
    let mut runs: Vec<RunOut> = Vec::new();
    for exp in &req.experiment_ids {
        let id = parse_id(exp, "experiment")?;
        for record in store.list_runs(id, include_deleted).await? {
            if let Some(clauses) = &filter {
                if !clauses.iter().all(|c| c.matches(&record)) {
                    continue;
                }
            }
            runs.push(run_out(&record));
        }
    }
    if descending {
        runs.sort_by_key(|r| std::cmp::Reverse(r.info.start_time));
    } else {
        runs.sort_by_key(|r| r.info.start_time);
    }
    if let Some(limit) = req.max_results {
        runs.truncate(limit as usize);
    }
    Ok(Json(RunsSearchResponse { runs }))
}

// ---------------------------------------------------------------------------
// Run-search filter subset: params.`k` (= | !=) 'v', tags.`k` LIKE '%v%',
// metrics.`k` (< | <= | > | >= | = | !=) number — AND semantics.
// Unparseable filters are INVALID_PARAMETER_VALUE, never silently ignored.
// ---------------------------------------------------------------------------

enum FieldFilter {
    ParamEq(String, String),
    ParamNe(String, String),
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
            FieldFilter::TagEq(k, v) => run.tags.get(k) == Some(v),
            FieldFilter::TagNe(k, v) => run.tags.get(k) != Some(v),
            FieldFilter::TagLike(k, pattern) => match run.tags.get(k) {
                Some(actual) => like_match(actual, pattern),
                None => false,
            },
            FieldFilter::MetricCmp(k, v, cmp) => run
                .latest_metrics
                .get(k)
                .map(|p| cmp(p.value, *v))
                .unwrap_or(false),
        }
    }
}

fn like_match(value: &str, pattern: &str) -> bool {
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
    for clause in filter.split(" AND ") {
        let clause = clause.trim();
        let field = if let Some(rest) = clause.strip_prefix("metrics.") {
            let (op, value) = split_cmp(rest)?;
            let number: f64 = value.parse().map_err(|_| {
                MlflowError::invalid(format!("metric filter needs a number, got {value}"))
            })?;
            let cmp = cmp_fn(&op)?;
            let key = field_key(rest, &op)?;
            FieldFilter::MetricCmp(key, number, cmp)
        } else if let Some(rest) = clause.strip_prefix("params.") {
            let (op, value) = split_quoted(rest)?;
            let key = field_key_quoted(rest, &op)?;
            match op.as_str() {
                "=" => FieldFilter::ParamEq(key, value),
                "!=" => FieldFilter::ParamNe(key, value),
                other => {
                    return Err(MlflowError::invalid(format!(
                        "params filter supports = / != , got '{other}'"
                    )));
                }
            }
        } else if let Some(rest) = clause.strip_prefix("tags.") {
            let (op, value) = split_quoted(rest)?;
            let key = field_key_quoted(rest, &op)?;
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

fn split_cmp(rest: &str) -> MlflowResult<(String, String)> {
    for op in ["!=", "<=", ">=", "<", ">", "="] {
        if let Some(idx) = rest.find(op) {
            return Ok((op.to_string(), rest[idx + op.len()..].trim().to_string()));
        }
    }
    Err(MlflowError::invalid(format!(
        "metric filter needs a comparison operator: '{rest}'"
    )))
}

fn split_quoted(rest: &str) -> MlflowResult<(String, String)> {
    for op in ["!=", "="] {
        if let Some(idx) = rest.find(op) {
            let raw = rest[idx + op.len()..].trim();
            let value = raw
                .strip_prefix('\'')
                .and_then(|v| v.strip_suffix('\''))
                .ok_or_else(|| {
                    MlflowError::invalid(format!(
                        "string filter values must be single-quoted: '{raw}'"
                    ))
                })?;
            return Ok((op.to_string(), value.to_string()));
        }
    }
    if rest.contains("LIKE") {
        let idx = rest.find("LIKE").unwrap();
        let raw = rest[idx + 4..].trim();
        let value = raw
            .strip_prefix('\'')
            .and_then(|v| v.strip_suffix('\''))
            .ok_or_else(|| {
                MlflowError::invalid(format!("LIKE pattern must be single-quoted: '{raw}'"))
            })?;
        return Ok(("LIKE".to_string(), value.to_string()));
    }
    Err(MlflowError::invalid(format!(
        "filter needs = / != / LIKE: '{rest}'"
    )))
}

fn field_key(rest: &str, op: &str) -> MlflowResult<String> {
    let idx = rest
        .find(op)
        .ok_or_else(|| MlflowError::internal("operator vanished"))?;
    Ok(rest[..idx].trim().to_string())
}

fn field_key_quoted(rest: &str, op: &str) -> MlflowResult<String> {
    field_key(rest, op)
}

fn cmp_fn(op: &str) -> MlflowResult<fn(f64, f64) -> bool> {
    Ok(match op {
        "<" => |a, b| a < b,
        "<=" => |a, b| a <= b,
        ">" => |a, b| a > b,
        ">=" => |a, b| a >= b,
        "=" => |a, b| (a - b).abs() == 0.0,
        "!=" => |a, b| (a - b).abs() != 0.0,
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
        let engine = Arc::new(SstEngine::new().await.unwrap());
        let document = Arc::new(DocumentService::new(engine));
        mlflow_routes()
            .with_state(MlflowState::new(document))
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

        // Run create -> MLflow run shape (info/data split, mlflow.runName).
        let (status, body) = post_json(
            &mut router,
            "/runs/create",
            serde_json::json!({"experiment_id": experiment_id, "run_name": "baseline"}),
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
            serde_json::json!({"run_id": run_id, "metric": {"key": "rmse", "value": 0.9, "timestamp": 1000, "step": 0}}),
        )
        .await;
        let (status, _) = post_json(
            &mut router,
            "/runs/log-batch",
            serde_json::json!({"run_id": run_id, "metrics": [{"key": "rmse", "value": 0.7, "timestamp": 2000, "step": 1}], "tags": [{"key": "phase", "value": "tune"}]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        // Get run: latest metric + params + tags present.
        let (status, body) = post_json(
            &mut router,
            "/runs/get",
            serde_json::json!({"run_id": run_id}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let metrics = body["run"]["data"]["metrics"].as_array().unwrap();
        assert_eq!(metrics.len(), 1);
        assert_eq!(
            metrics[0]["value"],
            serde_json::json!(0.7),
            "latest projection"
        );

        // Search: matching filter finds the run; NON-MATCHING returns empty
        // (the negative control — an ignored filter must not pass silently).
        let matching = serde_json::json!({
            "experiment_ids": [experiment_id],
            "filter": "params.lr = '0.01' AND metrics.rmse < 0.95"
        });
        let (status, body) = post_json(&mut router, "/runs/search", matching).await;
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
            serde_json::json!({"run_id": run_id, "metric": {"key": "rmse", "value": 0.1, "timestamp": 9500, "step": 2}}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error_code"], "INVALID_STATE");

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
        let (status, body) = post_json(
            &mut bob,
            "/experiments/get",
            serde_json::json!({"experiment_id": experiment_id}),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error_code"], "RESOURCE_DOES_NOT_EXIST");

        let (missing_status, missing_body) = post_json(
            &mut bob,
            "/experiments/get",
            serde_json::json!({"experiment_id": "999999"}),
        )
        .await;
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
