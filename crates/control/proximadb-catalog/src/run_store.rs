//! MLflow-compatible tracking store port (TD-MLOPS-1 slice 1).
//!
//! Experiments, runs, params, metrics and tags are tenant-scoped substrate
//! records — NOT catalog assets and NOT a private metadata database. This
//! module defines the port (types + trait); the first implementation persists
//! through the platform's storage seams and lives beside its wiring. The
//! MLflow wire adapter (slice 2) is a codec over this port only.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// MLflow experiment lifecycle marker. Deletes are soft — deleted experiments
/// are hidden from search and default listing but restorable and directly
/// gettable, matching MLflow client expectations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentStage {
    Active,
    Deleted,
}

/// MLflow run lifecycle stage — orthogonal to [`RunStatus`] exactly as
/// MLflow models it (`lifecycle_stage` x `status`): deleting a finished run
/// and restoring it must yield a finished run again.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunLifecycle {
    Active,
    Deleted,
}

/// MLflow run status. `Running -> Finished` is one-way; finished runs
/// reject further param/metric writes (params are immutable even while
/// running).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Running,
    Finished,
    Failed,
    Killed,
}

impl RunStatus {
    pub fn is_terminal(self) -> bool {
        !matches!(self, RunStatus::Running)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentRecord {
    pub experiment_id: u64,
    pub name: String,
    pub artifact_location: Option<String>,
    pub tags: BTreeMap<String, String>,
    pub stage: ExperimentStage,
    pub creation_time_ms: i64,
    pub last_update_time_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunRecord {
    pub run_id: String,
    pub experiment_id: u64,
    pub run_name: Option<String>,
    pub user_id: Option<String>,
    pub lifecycle: RunLifecycle,
    pub status: RunStatus,
    pub start_time_ms: i64,
    pub end_time_ms: Option<i64>,
    /// Immutable after first write for a key (MLflow param semantics).
    pub params: BTreeMap<String, String>,
    /// Latest value per metric key; full history is append-only.
    pub latest_metrics: BTreeMap<String, MetricPoint>,
    pub tags: BTreeMap<String, String>,
    /// Logged models this run CONSUMED (`RunInputs.model_inputs`); the
    /// default keeps documents written before the field existed readable.
    #[serde(default)]
    pub model_inputs: Vec<String>,
    /// Logged models this run PRODUCED (`RunOutputs.model_outputs`).
    #[serde(default)]
    pub model_outputs: Vec<ModelOutputRef>,
}

/// One append-only metric sample. History preserves insertion (timestamp,
/// step) order per key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricPoint {
    pub key: String,
    /// Proto3-JSON convention both ways: serde_json writes non-finite f64
    /// as `null` (unreadable on the way back), so non-finite values
    /// serialize as the strings "NaN" / "Infinity" / "-Infinity" and the
    /// deserializer accepts numbers or those strings. Without this, one
    /// diverged-loss metric corrupts the whole run document.
    #[serde(with = "metric_value_lossy")]
    pub value: f64,
    pub timestamp_ms: i64,
    pub step: i64,
}

mod metric_value_lossy {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(v: &f64, s: S) -> Result<S::Ok, S::Error> {
        if v.is_nan() {
            "NaN".serialize(s)
        } else if *v == f64::INFINITY {
            "Infinity".serialize(s)
        } else if *v == f64::NEG_INFINITY {
            "-Infinity".serialize(s)
        } else {
            s.serialize_f64(*v)
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
        let raw = serde_json::Value::deserialize(d)?;
        match raw {
            serde_json::Value::Number(n) => n
                .as_f64()
                .ok_or_else(|| serde::de::Error::custom("metric value out of f64 range")),
            serde_json::Value::String(s) => match s.as_str() {
                "NaN" => Ok(f64::NAN),
                "Infinity" => Ok(f64::INFINITY),
                "-Infinity" => Ok(f64::NEG_INFINITY),
                other => other.parse::<f64>().map_err(|_| {
                    serde::de::Error::custom(format!("invalid metric value '{other}'"))
                }),
            },
            other => Err(serde::de::Error::custom(format!(
                "metric value must be a number or numeric string, got {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunDatasetInput {
    /// MLflow dataset name — an alias, never a resolution authority.
    pub dataset_name: String,
    /// Content digest pinning the exact input (reproducibility contract).
    pub digest: String,
}

/// MLflow 3.x trace state (`TraceInfoV3.State` proto enum; the wire speaks
/// the NAME strings OK/ERROR/IN_PROGRESS).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceState {
    Ok,
    Error,
    InProgress,
}

/// How an assessment was produced (`AssessmentSource.SourceType` proto enum).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssessmentSource {
    Human,
    LlmJudge,
    Code,
}

/// Structured failure carried by a feedback assessment instead of a value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssessmentError {
    pub error_code: String,
    pub error_message: String,
    pub stack_trace: Option<String>,
}

/// The assessment payload oneof: exactly one of feedback value / feedback
/// error / expectation / issue (the client's `Assessment.from_proto` raises
/// when a response carries none or several).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AssessmentValue {
    /// `feedback.value` — a `google.protobuf.Value`, arbitrary JSON
    /// passthrough both ways (we never interpret it).
    FeedbackValue(serde_json::Value),
    /// `feedback.error` — the judge/code failed before producing a value.
    FeedbackError(AssessmentError),
    /// `expectation.value` — arbitrary JSON passthrough.
    ExpectationValue(serde_json::Value),
    /// `issue.issue_name`.
    Issue { issue_name: String },
}

/// One evaluation attached to a trace (`mlflow.assessments`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssessmentRecord {
    pub assessment_id: String,
    pub assessment_name: String,
    /// Owning span within the trace (None = trace-level).
    pub span_id: Option<String>,
    pub source: AssessmentSource,
    pub source_id: Option<String>,
    pub create_time_ms: i64,
    pub last_update_time_ms: i64,
    pub value: AssessmentValue,
    pub rationale: Option<String>,
    pub metadata: BTreeMap<String, String>,
    pub valid: bool,
}

/// MLflow 3.x trace record (`TraceInfoV3` minus spans). Spans NEVER live
/// here: the client uploads them as trace artifacts (`traces.json`) through
/// the artifacts proxy and downloads them itself, so the server's Trace
/// responses always carry `spans: []`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceRecord {
    pub trace_id: String,
    pub experiment_id: u64,
    pub client_request_id: Option<String>,
    pub state: TraceState,
    /// Truncated previews only — the client never sends full payloads on
    /// the wire (request/response stay client-side).
    pub request_preview: Option<String>,
    pub response_preview: Option<String>,
    pub request_time_ms: i64,
    pub execution_duration_ms: i64,
    pub metadata: BTreeMap<String, String>,
    pub tags: BTreeMap<String, String>,
    pub assessments: Vec<AssessmentRecord>,
}

/// Logged-model lifecycle (`LoggedModelStatus` proto enum; the wire speaks
/// the NAME strings LOGGED_MODEL_PENDING/READY/UPLOAD_FAILED).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoggedModelStatus {
    Pending,
    Ready,
    UploadFailed,
}

/// MLflow 3.x logged model (`LoggedModel`). Params are immutable after
/// first write (run-param semantics); status transitions via finalize;
/// deletes are soft (hidden from search, directly gettable with
/// `allow_deleted`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoggedModelRecord {
    pub model_id: String,
    pub experiment_id: u64,
    pub name: String,
    pub model_type: Option<String>,
    pub source_run_id: Option<String>,
    pub artifact_uri: String,
    pub status: LoggedModelStatus,
    pub lifecycle: RunLifecycle,
    pub params: BTreeMap<String, String>,
    pub tags: BTreeMap<String, String>,
    pub creation_time_ms: i64,
    pub last_updated_time_ms: i64,
}

/// One model-owned metric sample: a [`MetricPoint`] plus the dataset
/// context the wire echoes in `LoggedModel.data.metrics`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelMetricPoint {
    pub point: MetricPoint,
    /// Run that produced this model-owned sample. Optional only for reading
    /// records written before the lineage field was introduced.
    #[serde(default)]
    pub run_id: Option<String>,
    pub dataset_name: Option<String>,
    pub dataset_digest: Option<String>,
}

/// A run's output link to a logged model (`RunOutputs.model_outputs`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelOutputRef {
    pub model_id: String,
    pub step: i64,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RunStoreError {
    #[error("experiment {experiment_id} not found")]
    UnknownExperiment { experiment_id: u64 },
    #[error("experiment name '{name}' already exists")]
    ExperimentNameConflict { name: String },
    #[error("run '{run_id}' not found")]
    UnknownRun { run_id: String },
    #[error("run '{run_id}' already exists")]
    RunIdConflict { run_id: String },
    #[error("param '{key}' is immutable once logged on '{run_id}' (run or logged model)")]
    ParamImmutable { key: String, run_id: String },
    #[error("run '{run_id}' is finished; metric/param writes are rejected")]
    RunFinished { run_id: String },
    #[error("finish_run requires a terminal status, got Running")]
    NotTerminal,
    #[error("experiment {experiment_id} is deleted; run creation is rejected")]
    ExperimentDeleted { experiment_id: u64 },
    #[error("trace '{trace_id}' not found")]
    UnknownTrace { trace_id: String },
    #[error("trace '{trace_id}' already exists")]
    TraceIdConflict { trace_id: String },
    #[error("trace id '{trace_id}' has invalid characters (allowed: A-Za-z0-9._~-)")]
    InvalidTraceId { trace_id: String },
    #[error("assessment '{assessment_id}' not found on trace '{trace_id}'")]
    UnknownAssessment {
        trace_id: String,
        assessment_id: String,
    },
    #[error("logged model '{model_id}' not found")]
    UnknownLoggedModel { model_id: String },
    #[error("logged model '{model_id}' already exists")]
    LoggedModelIdConflict { model_id: String },
    #[error("status '{status}' is not a valid logged-model status")]
    InvalidModelStatus { status: String },
    #[error("tag '{key}' not found on logged model '{model_id}'")]
    UnknownLoggedModelTag { model_id: String, key: String },
    #[error("names and ids must be non-empty")]
    Empty { field: &'static str },
    #[error("tracking store internal error: {message}")]
    Internal { message: String },
}

/// Result of a metric append — the caller decides whether the latest-value
/// projection advanced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetricAppend {
    pub history_len: u64,
}

/// Tenant-scoped experiment/run tracking port. Implementations MUST enforce
/// tenant isolation structurally (the caller threads tenant identity into the
/// store's construction or per-call context — never inside record payloads).
#[async_trait::async_trait]
pub trait RunStore: Send + Sync {
    /// Create an experiment with a caller-chosen unique name.
    async fn create_experiment(
        &self,
        name: &str,
        artifact_location: Option<&str>,
        tags: BTreeMap<String, String>,
    ) -> Result<ExperimentRecord, RunStoreError>;

    async fn get_experiment(&self, experiment_id: u64) -> Result<ExperimentRecord, RunStoreError>;

    /// Deterministic listing (by id) of active experiments unless
    /// `include_deleted` is set.
    async fn list_experiments(
        &self,
        include_deleted: bool,
    ) -> Result<Vec<ExperimentRecord>, RunStoreError>;

    /// Soft-delete; a deleted experiment can be restored.
    async fn delete_experiment(&self, experiment_id: u64) -> Result<(), RunStoreError>;

    async fn restore_experiment(&self, experiment_id: u64) -> Result<(), RunStoreError>;

    /// Create a run under an experiment with a server-unique caller-supplied
    /// id (MLflow allows client-chosen run ids; conflicts fail loudly).
    async fn create_run(
        &self,
        experiment_id: u64,
        run_id: &str,
        run_name: Option<&str>,
        user_id: Option<&str>,
        tags: BTreeMap<String, String>,
        start_time_ms: i64,
    ) -> Result<RunRecord, RunStoreError>;

    async fn get_run(&self, run_id: &str) -> Result<RunRecord, RunStoreError>;

    /// Runs of one experiment (active only unless `include_deleted`),
    /// ordered by creation.
    async fn list_runs(
        &self,
        experiment_id: u64,
        include_deleted: bool,
    ) -> Result<Vec<RunRecord>, RunStoreError>;

    /// Terminal transition with the MLflow status (Finished/Failed/Killed);
    /// any terminal status freezes param/metric writes.
    async fn finish_run(
        &self,
        run_id: &str,
        status: RunStatus,
        end_time_ms: i64,
    ) -> Result<(), RunStoreError>;

    /// Reopen a terminal run (MLflow UpdateRun -> RUNNING).
    async fn reopen_run(&self, run_id: &str) -> Result<(), RunStoreError>;

    /// Soft-delete a run (hides from listing; direct get still works).
    async fn delete_run(&self, run_id: &str) -> Result<(), RunStoreError>;

    async fn restore_run(&self, run_id: &str) -> Result<(), RunStoreError>;

    /// Log one param. Second write with a DIFFERENT value is an error; the
    /// same value is idempotent (MLflow log-batch retries must succeed).
    async fn log_param(&self, run_id: &str, key: &str, value: &str) -> Result<(), RunStoreError>;

    /// Append a metric sample to the per-key history and advance the
    /// latest-value projection. Rejected on finished runs.
    async fn log_metric(
        &self,
        run_id: &str,
        point: MetricPoint,
    ) -> Result<MetricAppend, RunStoreError>;

    /// Full per-key history in append order.
    async fn metric_history(
        &self,
        run_id: &str,
        key: &str,
    ) -> Result<Vec<MetricPoint>, RunStoreError>;

    /// Set (or overwrite) / delete a tag.
    async fn set_tag(&self, run_id: &str, key: &str, value: &str) -> Result<(), RunStoreError>;

    async fn delete_tag(&self, run_id: &str, key: &str) -> Result<(), RunStoreError>;

    /// Record a typed dataset input (name + digest lineage).
    async fn log_dataset_input(
        &self,
        run_id: &str,
        input: RunDatasetInput,
    ) -> Result<(), RunStoreError>;

    async fn dataset_inputs(&self, run_id: &str) -> Result<Vec<RunDatasetInput>, RunStoreError>;
    /// Create a trace with a client-chosen id (`tr-<32hex>`). Conflicts fail
    /// loudly; the id lands in document ids and artifact path segments so
    /// implementations validate a conservative charset.
    async fn start_trace(&self, trace: TraceRecord) -> Result<TraceRecord, RunStoreError>;

    /// Full trace record including embedded assessments.
    async fn get_trace(&self, trace_id: &str) -> Result<TraceRecord, RunStoreError>;

    /// Traces of one experiment in creation order (the wire sorts
    /// newest-first for search).
    async fn list_traces(&self, experiment_id: u64) -> Result<Vec<TraceRecord>, RunStoreError>;

    /// Bulk delete. With `request_ids`: exactly those traces (that must
    /// belong to the experiment). Otherwise: traces with
    /// `request_time_ms <= max_timestamp_ms` (when set), up to `max_traces`
    /// (when set). Returns the count deleted; repeat calls delete 0.
    async fn delete_traces(
        &self,
        experiment_id: u64,
        max_timestamp_ms: Option<i64>,
        max_traces: Option<u64>,
        request_ids: &[String],
    ) -> Result<u64, RunStoreError>;

    async fn set_trace_tag(
        &self,
        trace_id: &str,
        key: &str,
        value: &str,
    ) -> Result<(), RunStoreError>;

    async fn delete_trace_tag(&self, trace_id: &str, key: &str) -> Result<(), RunStoreError>;

    /// Create or fully replace an assessment (PATCH-with-mask is resolved
    /// by the wire reading the current record first).
    async fn upsert_assessment(
        &self,
        trace_id: &str,
        assessment: AssessmentRecord,
    ) -> Result<AssessmentRecord, RunStoreError>;

    async fn get_assessment(
        &self,
        trace_id: &str,
        assessment_id: &str,
    ) -> Result<AssessmentRecord, RunStoreError>;

    async fn delete_assessment(
        &self,
        trace_id: &str,
        assessment_id: &str,
    ) -> Result<(), RunStoreError>;

    /// Create a logged model with a server-minted caller-supplied id
    /// (`m-<32hex>`); the artifact_uri is minted by the wire (URI shape is
    /// a wire concern) and stored verbatim.
    async fn create_logged_model(
        &self,
        model: LoggedModelRecord,
    ) -> Result<LoggedModelRecord, RunStoreError>;

    /// Direct get; deleted models 404 unless `allow_deleted`.
    async fn get_logged_model(
        &self,
        model_id: &str,
        allow_deleted: bool,
    ) -> Result<LoggedModelRecord, RunStoreError>;

    async fn list_logged_models(
        &self,
        experiment_id: u64,
        include_deleted: bool,
    ) -> Result<Vec<LoggedModelRecord>, RunStoreError>;

    /// Terminal/any status transition + last_updated bump. Params stay
    /// writable (the reference server has no freeze here).
    async fn finalize_logged_model(
        &self,
        model_id: &str,
        status: LoggedModelStatus,
        now_ms: i64,
    ) -> Result<LoggedModelRecord, RunStoreError>;

    /// Soft-delete (hidden from search+default get).
    async fn delete_logged_model(&self, model_id: &str) -> Result<(), RunStoreError>;

    /// Append params with run-param immutability semantics (same value
    /// idempotent, different value rejected).
    async fn log_logged_model_params(
        &self,
        model_id: &str,
        params: &BTreeMap<String, String>,
    ) -> Result<(), RunStoreError>;

    async fn set_logged_model_tags(
        &self,
        model_id: &str,
        tags: &BTreeMap<String, String>,
    ) -> Result<LoggedModelRecord, RunStoreError>;

    /// Deleting an ABSENT tag is an error (reference server: 404).
    async fn delete_logged_model_tag(&self, model_id: &str, key: &str)
    -> Result<(), RunStoreError>;

    /// Append a model-owned metric sample. Model metrics never touch the
    /// run latest-value projection or the finished-run freeze.
    async fn log_model_metric(
        &self,
        model_id: &str,
        sample: ModelMetricPoint,
    ) -> Result<MetricAppend, RunStoreError>;

    async fn model_metric_history(
        &self,
        model_id: &str,
        key: &str,
    ) -> Result<Vec<ModelMetricPoint>, RunStoreError>;

    /// ALL metric samples of a model across keys, in append order — the
    /// wire embeds `LoggedModel.data.metrics` with ONE call (no per-key
    /// N+1).
    async fn model_metrics(&self, model_id: &str) -> Result<Vec<ModelMetricPoint>, RunStoreError>;

    /// Append output links (run PRODUCED these models).
    async fn log_run_outputs(
        &self,
        run_id: &str,
        outputs: Vec<ModelOutputRef>,
    ) -> Result<(), RunStoreError>;

    /// Append input links (run CONSUMED these models).
    async fn log_run_model_inputs(
        &self,
        run_id: &str,
        model_ids: Vec<String>,
    ) -> Result<(), RunStoreError>;
}

/// Conservative trace-id charset: the id lands in substrate document ids
/// AND artifact path segments, so anything outside the unreserved URI set
/// is rejected up front (fail closed). Client-generated ids are
/// `tr-<32hex>`, well inside this set.
pub fn valid_trace_id(trace_id: &str) -> bool {
    !trace_id.is_empty()
        && !trace_id.trim_matches('.').is_empty()
        && trace_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '~' | '-'))
}

/// Model-registry tag evaluation with MLflow absent-value semantics:
/// `=` never matches a missing tag, `!=` matches everything. Lives at the
/// port so every adapter (wire, future CLI) shares one truth; when the
/// registry's annotation facet lands, implementations override with real
/// lookups and this default becomes the in-memory/test truth only.
pub fn tag_clause_matches(is_eq: bool, present_value: Option<&String>, expected: &str) -> bool {
    match (is_eq, present_value) {
        (true, Some(v)) => v == expected,
        (true, None) => false,
        (false, Some(v)) => v != expected,
        (false, None) => true,
    }
}

/// Tenant-scoped factory over [`RunStore`] implementations — the seam the
/// wire depends on (DIP): transports receive a factory, never a concrete
/// store, so a second implementation (test double, alternative substrate)
/// injects without touching handler code.
pub trait RunStoreFactory: Send + Sync {
    /// Returns the tenant-scoped store. The tenant id MUST already be a
    /// validated request tenant (implementations may re-validate cheaply).
    fn store_for(
        &self,
        tenant_id: &str,
    ) -> std::result::Result<std::sync::Arc<dyn RunStore>, RunStoreError>;
}

/// Executable port semantics shared by every implementation.
pub mod conformance_tests {
    use super::*;

    /// Test double factory: one independent in-memory store per tenant —
    /// the same structural isolation the substrate implementation provides.
    pub struct InMemoryRunStoreFactory {
        stores:
            std::sync::Mutex<std::collections::HashMap<String, std::sync::Arc<InMemoryRunStore>>>,
    }

    impl Default for InMemoryRunStoreFactory {
        fn default() -> Self {
            Self::new()
        }
    }

    impl InMemoryRunStoreFactory {
        pub fn new() -> Self {
            Self {
                stores: std::sync::Mutex::new(std::collections::HashMap::new()),
            }
        }
    }

    impl super::RunStoreFactory for InMemoryRunStoreFactory {
        fn store_for(
            &self,
            tenant_id: &str,
        ) -> std::result::Result<std::sync::Arc<dyn RunStore>, RunStoreError> {
            let mut stores = self.stores.lock().unwrap_or_else(|p| p.into_inner());
            Ok(stores
                .entry(tenant_id.to_string())
                .or_insert_with(|| std::sync::Arc::new(InMemoryRunStore::new()))
                .clone())
        }
    }

    /// Reference in-memory implementation: executable semantics for the port.
    /// The substrate-backed store (TD-MLOPS-1 slice 1) must pass the same
    /// conformance battery; the MLflow wire adapter's unit tests use this
    /// double without a substrate.
    pub struct InMemoryRunStore {
        experiments: std::sync::Mutex<Vec<ExperimentRecord>>,
        runs: std::sync::Mutex<Vec<RunRecord>>,
        history: std::sync::Mutex<Vec<(String, MetricPoint)>>,
        datasets: std::sync::Mutex<Vec<(String, RunDatasetInput)>>,
        traces: std::sync::Mutex<Vec<TraceRecord>>,
        logged_models: std::sync::Mutex<Vec<LoggedModelRecord>>,
        model_history: std::sync::Mutex<Vec<(String, ModelMetricPoint)>>,
    }

    impl Default for InMemoryRunStore {
        fn default() -> Self {
            Self::new()
        }
    }

    impl InMemoryRunStore {
        pub fn new() -> Self {
            Self {
                experiments: std::sync::Mutex::new(Vec::new()),
                runs: std::sync::Mutex::new(Vec::new()),
                history: std::sync::Mutex::new(Vec::new()),
                datasets: std::sync::Mutex::new(Vec::new()),
                traces: std::sync::Mutex::new(Vec::new()),
                logged_models: std::sync::Mutex::new(Vec::new()),
                model_history: std::sync::Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait::async_trait]
    impl RunStore for InMemoryRunStore {
        async fn create_experiment(
            &self,
            name: &str,
            artifact_location: Option<&str>,
            tags: BTreeMap<String, String>,
        ) -> Result<ExperimentRecord, RunStoreError> {
            if name.is_empty() {
                return Err(RunStoreError::Empty { field: "name" });
            }
            let mut guard = self.experiments.lock().unwrap();
            if guard.iter().any(|e| e.name == name) {
                return Err(RunStoreError::ExperimentNameConflict {
                    name: name.to_string(),
                });
            }
            let id = guard.len() as u64;
            let record = ExperimentRecord {
                experiment_id: id,
                name: name.to_string(),
                artifact_location: artifact_location.map(str::to_string),
                tags,
                stage: ExperimentStage::Active,
                creation_time_ms: 1_000,
                last_update_time_ms: 1_000,
            };
            guard.push(record.clone());
            Ok(record)
        }

        async fn get_experiment(
            &self,
            experiment_id: u64,
        ) -> Result<ExperimentRecord, RunStoreError> {
            self.experiments
                .lock()
                .unwrap()
                .iter()
                .find(|e| e.experiment_id == experiment_id)
                .cloned()
                .ok_or(RunStoreError::UnknownExperiment { experiment_id })
        }

        async fn list_experiments(
            &self,
            include_deleted: bool,
        ) -> Result<Vec<ExperimentRecord>, RunStoreError> {
            Ok(self
                .experiments
                .lock()
                .unwrap()
                .iter()
                .filter(|e| include_deleted || e.stage == ExperimentStage::Active)
                .cloned()
                .collect())
        }

        async fn delete_experiment(&self, experiment_id: u64) -> Result<(), RunStoreError> {
            let mut guard = self.experiments.lock().unwrap();
            let record = guard
                .iter_mut()
                .find(|e| e.experiment_id == experiment_id)
                .ok_or(RunStoreError::UnknownExperiment { experiment_id })?;
            record.stage = ExperimentStage::Deleted;
            Ok(())
        }

        async fn restore_experiment(&self, experiment_id: u64) -> Result<(), RunStoreError> {
            let mut guard = self.experiments.lock().unwrap();
            let record = guard
                .iter_mut()
                .find(|e| e.experiment_id == experiment_id)
                .ok_or(RunStoreError::UnknownExperiment { experiment_id })?;
            record.stage = ExperimentStage::Active;
            Ok(())
        }

        async fn create_run(
            &self,
            experiment_id: u64,
            run_id: &str,
            run_name: Option<&str>,
            user_id: Option<&str>,
            tags: BTreeMap<String, String>,
            start_time_ms: i64,
        ) -> Result<RunRecord, RunStoreError> {
            if run_id.is_empty() {
                return Err(RunStoreError::Empty { field: "run_id" });
            }
            self.get_experiment(experiment_id).await?;
            {
                let runs = self.runs.lock().unwrap();
                if runs.iter().any(|r| r.run_id == run_id) {
                    return Err(RunStoreError::RunIdConflict {
                        run_id: run_id.to_string(),
                    });
                }
            }
            {
                let experiments = self.experiments.lock().unwrap();
                let stage = experiments
                    .iter()
                    .find(|e| e.experiment_id == experiment_id)
                    .map(|e| e.stage);
                if stage == Some(ExperimentStage::Deleted) {
                    return Err(RunStoreError::ExperimentDeleted { experiment_id });
                }
            }
            let record = RunRecord {
                run_id: run_id.to_string(),
                experiment_id,
                run_name: run_name.map(str::to_string),
                user_id: user_id.map(str::to_string),
                lifecycle: RunLifecycle::Active,
                status: RunStatus::Running,
                start_time_ms,
                end_time_ms: None,
                params: BTreeMap::new(),
                latest_metrics: BTreeMap::new(),
                tags,
                model_inputs: Vec::new(),
                model_outputs: Vec::new(),
            };
            self.runs.lock().unwrap().push(record.clone());
            Ok(record)
        }

        async fn get_run(&self, run_id: &str) -> Result<RunRecord, RunStoreError> {
            self.runs
                .lock()
                .unwrap()
                .iter()
                .find(|r| r.run_id == run_id)
                .cloned()
                .ok_or_else(|| RunStoreError::UnknownRun {
                    run_id: run_id.to_string(),
                })
        }

        async fn list_runs(
            &self,
            experiment_id: u64,
            include_deleted: bool,
        ) -> Result<Vec<RunRecord>, RunStoreError> {
            self.get_experiment(experiment_id).await?;
            Ok(self
                .runs
                .lock()
                .unwrap()
                .iter()
                .filter(|r| {
                    r.experiment_id == experiment_id
                        && (include_deleted || r.lifecycle != RunLifecycle::Deleted)
                })
                .cloned()
                .collect())
        }

        async fn finish_run(
            &self,
            run_id: &str,
            status: RunStatus,
            end_time_ms: i64,
        ) -> Result<(), RunStoreError> {
            if !status.is_terminal() {
                return Err(RunStoreError::NotTerminal);
            }
            let mut runs = self.runs.lock().unwrap();
            let run = runs
                .iter_mut()
                .find(|r| r.run_id == run_id)
                .ok_or_else(|| RunStoreError::UnknownRun {
                    run_id: run_id.to_string(),
                })?;
            run.status = status;
            run.end_time_ms = Some(end_time_ms);
            Ok(())
        }

        async fn reopen_run(&self, run_id: &str) -> Result<(), RunStoreError> {
            let mut runs = self.runs.lock().unwrap();
            let run = runs
                .iter_mut()
                .find(|r| r.run_id == run_id)
                .ok_or_else(|| RunStoreError::UnknownRun {
                    run_id: run_id.to_string(),
                })?;
            run.status = RunStatus::Running;
            run.end_time_ms = None;
            Ok(())
        }

        async fn delete_run(&self, run_id: &str) -> Result<(), RunStoreError> {
            let mut runs = self.runs.lock().unwrap();
            let run = runs
                .iter_mut()
                .find(|r| r.run_id == run_id)
                .ok_or_else(|| RunStoreError::UnknownRun {
                    run_id: run_id.to_string(),
                })?;
            run.lifecycle = RunLifecycle::Deleted;
            Ok(())
        }

        async fn restore_run(&self, run_id: &str) -> Result<(), RunStoreError> {
            let mut runs = self.runs.lock().unwrap();
            let run = runs
                .iter_mut()
                .find(|r| r.run_id == run_id)
                .ok_or_else(|| RunStoreError::UnknownRun {
                    run_id: run_id.to_string(),
                })?;
            run.lifecycle = RunLifecycle::Active;
            Ok(())
        }

        async fn log_param(
            &self,
            run_id: &str,
            key: &str,
            value: &str,
        ) -> Result<(), RunStoreError> {
            let mut runs = self.runs.lock().unwrap();
            {
                let run = runs.iter().find(|r| r.run_id == run_id).ok_or_else(|| {
                    RunStoreError::UnknownRun {
                        run_id: run_id.to_string(),
                    }
                })?;
                if run.status.is_terminal() {
                    return Err(RunStoreError::RunFinished {
                        run_id: run_id.to_string(),
                    });
                }
            }
            let run = runs
                .iter_mut()
                .find(|r| r.run_id == run_id)
                .ok_or_else(|| RunStoreError::UnknownRun {
                    run_id: run_id.to_string(),
                })?;
            match run.params.get(key) {
                Some(existing) if existing == value => Ok(()),
                Some(_) => Err(RunStoreError::ParamImmutable {
                    key: key.to_string(),
                    run_id: run_id.to_string(),
                }),
                None => {
                    run.params.insert(key.to_string(), value.to_string());
                    Ok(())
                }
            }
        }

        async fn log_metric(
            &self,
            run_id: &str,
            point: MetricPoint,
        ) -> Result<MetricAppend, RunStoreError> {
            {
                let runs = self.runs.lock().unwrap();
                let run = runs.iter().find(|r| r.run_id == run_id).ok_or_else(|| {
                    RunStoreError::UnknownRun {
                        run_id: run_id.to_string(),
                    }
                })?;
                if run.status.is_terminal() {
                    return Err(RunStoreError::RunFinished {
                        run_id: run_id.to_string(),
                    });
                }
            }
            {
                let mut history = self.history.lock().unwrap();
                history.push((run_id.to_string(), point.clone()));
            }
            let history_len;
            {
                let mut runs = self.runs.lock().unwrap();
                let run = runs
                    .iter_mut()
                    .find(|r| r.run_id == run_id)
                    .ok_or_else(|| RunStoreError::UnknownRun {
                        run_id: run_id.to_string(),
                    })?;
                let advances_projection =
                    run.latest_metrics.get(&point.key).is_none_or(|current| {
                        point.timestamp_ms > current.timestamp_ms
                            || (point.timestamp_ms == current.timestamp_ms
                                && point.value > current.value)
                    });
                if advances_projection {
                    run.latest_metrics.insert(point.key.clone(), point.clone());
                }
                history_len = self
                    .history
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|(id, p)| id == run_id && p.key == point.key)
                    .count() as u64;
            }
            Ok(MetricAppend { history_len })
        }

        async fn metric_history(
            &self,
            run_id: &str,
            key: &str,
        ) -> Result<Vec<MetricPoint>, RunStoreError> {
            self.get_run(run_id).await?;
            Ok(self
                .history
                .lock()
                .unwrap()
                .iter()
                .filter(|(id, p)| id == run_id && p.key == key)
                .map(|(_, p)| p.clone())
                .collect())
        }

        async fn set_tag(&self, run_id: &str, key: &str, value: &str) -> Result<(), RunStoreError> {
            let mut runs = self.runs.lock().unwrap();
            let run = runs
                .iter_mut()
                .find(|r| r.run_id == run_id)
                .ok_or_else(|| RunStoreError::UnknownRun {
                    run_id: run_id.to_string(),
                })?;
            run.tags.insert(key.to_string(), value.to_string());
            Ok(())
        }

        async fn delete_tag(&self, run_id: &str, key: &str) -> Result<(), RunStoreError> {
            let mut runs = self.runs.lock().unwrap();
            let run = runs
                .iter_mut()
                .find(|r| r.run_id == run_id)
                .ok_or_else(|| RunStoreError::UnknownRun {
                    run_id: run_id.to_string(),
                })?;
            run.tags.remove(key);
            Ok(())
        }

        async fn log_dataset_input(
            &self,
            run_id: &str,
            input: RunDatasetInput,
        ) -> Result<(), RunStoreError> {
            self.get_run(run_id).await?;
            self.datasets
                .lock()
                .unwrap()
                .push((run_id.to_string(), input));
            Ok(())
        }

        async fn dataset_inputs(
            &self,
            run_id: &str,
        ) -> Result<Vec<RunDatasetInput>, RunStoreError> {
            self.get_run(run_id).await?;
            Ok(self
                .datasets
                .lock()
                .unwrap()
                .iter()
                .filter(|(id, _)| id == run_id)
                .map(|(_, input)| input.clone())
                .collect())
        }

        async fn start_trace(&self, trace: TraceRecord) -> Result<TraceRecord, RunStoreError> {
            if trace.trace_id.is_empty() {
                return Err(RunStoreError::Empty { field: "trace_id" });
            }
            if !valid_trace_id(&trace.trace_id) {
                return Err(RunStoreError::InvalidTraceId {
                    trace_id: trace.trace_id,
                });
            }
            self.get_experiment(trace.experiment_id).await?;
            {
                let experiments = self.experiments.lock().unwrap();
                let stage = experiments
                    .iter()
                    .find(|e| e.experiment_id == trace.experiment_id)
                    .map(|e| e.stage);
                if stage == Some(ExperimentStage::Deleted) {
                    return Err(RunStoreError::ExperimentDeleted {
                        experiment_id: trace.experiment_id,
                    });
                }
            }
            let mut traces = self.traces.lock().unwrap();
            if traces.iter().any(|t| t.trace_id == trace.trace_id) {
                return Err(RunStoreError::TraceIdConflict {
                    trace_id: trace.trace_id,
                });
            }
            traces.push(trace.clone());
            Ok(trace)
        }

        async fn get_trace(&self, trace_id: &str) -> Result<TraceRecord, RunStoreError> {
            self.traces
                .lock()
                .unwrap()
                .iter()
                .find(|t| t.trace_id == trace_id)
                .cloned()
                .ok_or_else(|| RunStoreError::UnknownTrace {
                    trace_id: trace_id.to_string(),
                })
        }

        async fn list_traces(&self, experiment_id: u64) -> Result<Vec<TraceRecord>, RunStoreError> {
            self.get_experiment(experiment_id).await?;
            Ok(self
                .traces
                .lock()
                .unwrap()
                .iter()
                .filter(|t| t.experiment_id == experiment_id)
                .cloned()
                .collect())
        }

        async fn delete_traces(
            &self,
            experiment_id: u64,
            max_timestamp_ms: Option<i64>,
            max_traces: Option<u64>,
            request_ids: &[String],
        ) -> Result<u64, RunStoreError> {
            self.get_experiment(experiment_id).await?;
            let mut traces = self.traces.lock().unwrap();
            let selected: Vec<String> = if !request_ids.is_empty() {
                traces
                    .iter()
                    .filter(|t| {
                        t.experiment_id == experiment_id && request_ids.contains(&t.trace_id)
                    })
                    .map(|t| t.trace_id.clone())
                    .collect()
            } else {
                // Candidate order matches the substrate twin (sorted by
                // (request_time, trace_id), not insertion order) so
                // max_traces truncation deletes the SAME set in both.
                let mut candidates: Vec<(i64, String)> = traces
                    .iter()
                    .filter(|t| {
                        t.experiment_id == experiment_id
                            && max_timestamp_ms.is_none_or(|ts| t.request_time_ms <= ts)
                    })
                    .map(|t| (t.request_time_ms, t.trace_id.clone()))
                    .collect();
                candidates.sort();
                if let Some(max) = max_traces {
                    candidates.truncate(max as usize);
                }
                candidates.into_iter().map(|(_, id)| id).collect()
            };
            let deleted = selected.len() as u64;
            traces.retain(|t| !selected.contains(&t.trace_id));
            Ok(deleted)
        }

        async fn set_trace_tag(
            &self,
            trace_id: &str,
            key: &str,
            value: &str,
        ) -> Result<(), RunStoreError> {
            let mut traces = self.traces.lock().unwrap();
            let trace = traces
                .iter_mut()
                .find(|t| t.trace_id == trace_id)
                .ok_or_else(|| RunStoreError::UnknownTrace {
                    trace_id: trace_id.to_string(),
                })?;
            trace.tags.insert(key.to_string(), value.to_string());
            Ok(())
        }

        async fn delete_trace_tag(&self, trace_id: &str, key: &str) -> Result<(), RunStoreError> {
            let mut traces = self.traces.lock().unwrap();
            let trace = traces
                .iter_mut()
                .find(|t| t.trace_id == trace_id)
                .ok_or_else(|| RunStoreError::UnknownTrace {
                    trace_id: trace_id.to_string(),
                })?;
            trace.tags.remove(key);
            Ok(())
        }

        async fn upsert_assessment(
            &self,
            trace_id: &str,
            assessment: AssessmentRecord,
        ) -> Result<AssessmentRecord, RunStoreError> {
            let mut traces = self.traces.lock().unwrap();
            let trace = traces
                .iter_mut()
                .find(|t| t.trace_id == trace_id)
                .ok_or_else(|| RunStoreError::UnknownTrace {
                    trace_id: trace_id.to_string(),
                })?;
            match trace
                .assessments
                .iter_mut()
                .find(|a| a.assessment_id == assessment.assessment_id)
            {
                Some(existing) => *existing = assessment.clone(),
                None => trace.assessments.push(assessment.clone()),
            }
            Ok(assessment)
        }

        async fn get_assessment(
            &self,
            trace_id: &str,
            assessment_id: &str,
        ) -> Result<AssessmentRecord, RunStoreError> {
            self.traces
                .lock()
                .unwrap()
                .iter()
                .find(|t| t.trace_id == trace_id)
                .ok_or_else(|| RunStoreError::UnknownTrace {
                    trace_id: trace_id.to_string(),
                })?
                .assessments
                .iter()
                .find(|a| a.assessment_id == assessment_id)
                .cloned()
                .ok_or_else(|| RunStoreError::UnknownAssessment {
                    trace_id: trace_id.to_string(),
                    assessment_id: assessment_id.to_string(),
                })
        }

        async fn delete_assessment(
            &self,
            trace_id: &str,
            assessment_id: &str,
        ) -> Result<(), RunStoreError> {
            let mut traces = self.traces.lock().unwrap();
            let trace = traces
                .iter_mut()
                .find(|t| t.trace_id == trace_id)
                .ok_or_else(|| RunStoreError::UnknownTrace {
                    trace_id: trace_id.to_string(),
                })?;
            let before = trace.assessments.len();
            trace
                .assessments
                .retain(|a| a.assessment_id != assessment_id);
            if trace.assessments.len() == before {
                return Err(RunStoreError::UnknownAssessment {
                    trace_id: trace_id.to_string(),
                    assessment_id: assessment_id.to_string(),
                });
            }
            Ok(())
        }

        async fn create_logged_model(
            &self,
            model: LoggedModelRecord,
        ) -> Result<LoggedModelRecord, RunStoreError> {
            if model.name.is_empty() {
                return Err(RunStoreError::Empty { field: "name" });
            }
            if model.model_id.is_empty() {
                return Err(RunStoreError::Empty { field: "model_id" });
            }
            self.get_experiment(model.experiment_id).await?;
            {
                let experiments = self.experiments.lock().unwrap();
                let stage = experiments
                    .iter()
                    .find(|e| e.experiment_id == model.experiment_id)
                    .map(|e| e.stage);
                if stage == Some(ExperimentStage::Deleted) {
                    return Err(RunStoreError::ExperimentDeleted {
                        experiment_id: model.experiment_id,
                    });
                }
            }
            let mut models = self.logged_models.lock().unwrap();
            if models.iter().any(|m| m.model_id == model.model_id) {
                return Err(RunStoreError::LoggedModelIdConflict {
                    model_id: model.model_id,
                });
            }
            models.push(model.clone());
            Ok(model)
        }

        async fn get_logged_model(
            &self,
            model_id: &str,
            allow_deleted: bool,
        ) -> Result<LoggedModelRecord, RunStoreError> {
            self.logged_models
                .lock()
                .unwrap()
                .iter()
                .find(|m| m.model_id == model_id)
                .cloned()
                .map(|m| {
                    if !allow_deleted && m.lifecycle == RunLifecycle::Deleted {
                        Err(RunStoreError::UnknownLoggedModel {
                            model_id: model_id.to_string(),
                        })
                    } else {
                        Ok(m)
                    }
                })
                .transpose()?
                .ok_or_else(|| RunStoreError::UnknownLoggedModel {
                    model_id: model_id.to_string(),
                })
        }

        async fn list_logged_models(
            &self,
            experiment_id: u64,
            include_deleted: bool,
        ) -> Result<Vec<LoggedModelRecord>, RunStoreError> {
            self.get_experiment(experiment_id).await?;
            Ok(self
                .logged_models
                .lock()
                .unwrap()
                .iter()
                .filter(|m| {
                    m.experiment_id == experiment_id
                        && (include_deleted || m.lifecycle != RunLifecycle::Deleted)
                })
                .cloned()
                .collect())
        }

        async fn finalize_logged_model(
            &self,
            model_id: &str,
            status: LoggedModelStatus,
            now_ms: i64,
        ) -> Result<LoggedModelRecord, RunStoreError> {
            let mut models = self.logged_models.lock().unwrap();
            let model = models
                .iter_mut()
                .find(|m| m.model_id == model_id)
                .ok_or_else(|| RunStoreError::UnknownLoggedModel {
                    model_id: model_id.to_string(),
                })?;
            model.status = status;
            model.last_updated_time_ms = now_ms;
            Ok(model.clone())
        }

        async fn delete_logged_model(&self, model_id: &str) -> Result<(), RunStoreError> {
            let mut models = self.logged_models.lock().unwrap();
            let model = models
                .iter_mut()
                .find(|m| m.model_id == model_id)
                .ok_or_else(|| RunStoreError::UnknownLoggedModel {
                    model_id: model_id.to_string(),
                })?;
            model.lifecycle = RunLifecycle::Deleted;
            Ok(())
        }

        async fn log_logged_model_params(
            &self,
            model_id: &str,
            params: &BTreeMap<String, String>,
        ) -> Result<(), RunStoreError> {
            let mut models = self.logged_models.lock().unwrap();
            let model = models
                .iter_mut()
                .find(|m| m.model_id == model_id)
                .ok_or_else(|| RunStoreError::UnknownLoggedModel {
                    model_id: model_id.to_string(),
                })?;
            for (key, value) in params {
                match model.params.get(key) {
                    Some(existing) if existing == value => {}
                    Some(_) => {
                        return Err(RunStoreError::ParamImmutable {
                            key: key.clone(),
                            run_id: model_id.to_string(),
                        });
                    }
                    None => {
                        model.params.insert(key.clone(), value.clone());
                    }
                }
            }
            Ok(())
        }

        async fn set_logged_model_tags(
            &self,
            model_id: &str,
            tags: &BTreeMap<String, String>,
        ) -> Result<LoggedModelRecord, RunStoreError> {
            let mut models = self.logged_models.lock().unwrap();
            let model = models
                .iter_mut()
                .find(|m| m.model_id == model_id)
                .ok_or_else(|| RunStoreError::UnknownLoggedModel {
                    model_id: model_id.to_string(),
                })?;
            for (key, value) in tags {
                model.tags.insert(key.clone(), value.clone());
            }
            model.last_updated_time_ms += 1;
            Ok(model.clone())
        }

        async fn delete_logged_model_tag(
            &self,
            model_id: &str,
            key: &str,
        ) -> Result<(), RunStoreError> {
            let mut models = self.logged_models.lock().unwrap();
            let model = models
                .iter_mut()
                .find(|m| m.model_id == model_id)
                .ok_or_else(|| RunStoreError::UnknownLoggedModel {
                    model_id: model_id.to_string(),
                })?;
            if model.tags.remove(key).is_none() {
                return Err(RunStoreError::UnknownLoggedModelTag {
                    model_id: model_id.to_string(),
                    key: key.to_string(),
                });
            }
            Ok(())
        }

        async fn log_model_metric(
            &self,
            model_id: &str,
            sample: ModelMetricPoint,
        ) -> Result<MetricAppend, RunStoreError> {
            self.get_logged_model(model_id, false).await?;
            self.model_history
                .lock()
                .unwrap()
                .push((model_id.to_string(), sample.clone()));
            let history_len = self
                .model_history
                .lock()
                .unwrap()
                .iter()
                .filter(|(id, s)| id == model_id && s.point.key == sample.point.key)
                .count() as u64;
            Ok(MetricAppend { history_len })
        }

        async fn model_metric_history(
            &self,
            model_id: &str,
            key: &str,
        ) -> Result<Vec<ModelMetricPoint>, RunStoreError> {
            self.get_logged_model(model_id, true).await?;
            Ok(self
                .model_history
                .lock()
                .unwrap()
                .iter()
                .filter(|(id, s)| id == model_id && s.point.key == key)
                .map(|(_, s)| s.clone())
                .collect())
        }

        async fn model_metrics(
            &self,
            model_id: &str,
        ) -> Result<Vec<ModelMetricPoint>, RunStoreError> {
            self.get_logged_model(model_id, true).await?;
            Ok(self
                .model_history
                .lock()
                .unwrap()
                .iter()
                .filter(|(id, _)| id == model_id)
                .map(|(_, s)| s.clone())
                .collect())
        }

        async fn log_run_outputs(
            &self,
            run_id: &str,
            outputs: Vec<ModelOutputRef>,
        ) -> Result<(), RunStoreError> {
            let mut runs = self.runs.lock().unwrap();
            let run = runs
                .iter_mut()
                .find(|r| r.run_id == run_id)
                .ok_or_else(|| RunStoreError::UnknownRun {
                    run_id: run_id.to_string(),
                })?;
            run.model_outputs.extend(outputs);
            Ok(())
        }

        async fn log_run_model_inputs(
            &self,
            run_id: &str,
            model_ids: Vec<String>,
        ) -> Result<(), RunStoreError> {
            let mut runs = self.runs.lock().unwrap();
            let run = runs
                .iter_mut()
                .find(|r| r.run_id == run_id)
                .ok_or_else(|| RunStoreError::UnknownRun {
                    run_id: run_id.to_string(),
                })?;
            run.model_inputs.extend(model_ids);
            Ok(())
        }
    }

    /// Port conformance battery — substrate implementations re-run this
    /// against themselves (same semantics, different durability).
    pub async fn port_conformance<S: RunStore>(store: &S) {
        // Experiment lifecycle: create conflicts on duplicate names, soft
        // delete hides from listing, restore revives, direct get always works.
        let exp = store
            .create_experiment("classify-iris", None, BTreeMap::new())
            .await
            .expect("create experiment");
        assert_eq!(exp.name, "classify-iris");
        assert_eq!(exp.stage, ExperimentStage::Active);

        let dup = store
            .create_experiment("classify-iris", None, BTreeMap::new())
            .await
            .unwrap_err();
        assert_eq!(
            dup,
            RunStoreError::ExperimentNameConflict {
                name: "classify-iris".to_string()
            }
        );

        assert_eq!(store.list_experiments(false).await.unwrap().len(), 1);
        store.delete_experiment(exp.experiment_id).await.unwrap();
        assert_eq!(store.list_experiments(false).await.unwrap().len(), 0);
        assert_eq!(store.list_experiments(true).await.unwrap().len(), 1);
        assert_eq!(
            store.get_experiment(exp.experiment_id).await.unwrap().stage,
            ExperimentStage::Deleted
        );
        store.restore_experiment(exp.experiment_id).await.unwrap();
        assert_eq!(store.list_experiments(false).await.unwrap().len(), 1);

        // Run lifecycle.
        let run = store
            .create_run(
                exp.experiment_id,
                "run-0001",
                Some("baseline"),
                Some("tester"),
                BTreeMap::new(),
                1_000,
            )
            .await
            .expect("create run");
        assert_eq!(run.status, RunStatus::Running);

        let unknown_exp = store
            .create_run(999, "run-x", None, None, BTreeMap::new(), 0)
            .await
            .unwrap_err();
        assert_eq!(
            unknown_exp,
            RunStoreError::UnknownExperiment { experiment_id: 999 }
        );

        let dup_run = store
            .create_run(
                exp.experiment_id,
                "run-0001",
                None,
                None,
                BTreeMap::new(),
                0,
            )
            .await
            .unwrap_err();
        assert_eq!(
            dup_run,
            RunStoreError::RunIdConflict {
                run_id: "run-0001".to_string()
            }
        );

        // Params: immutable after first write, idempotent on same value.
        store.log_param("run-0001", "lr", "0.01").await.unwrap();
        store.log_param("run-0001", "lr", "0.01").await.unwrap();
        assert_eq!(
            store.log_param("run-0001", "lr", "0.02").await.unwrap_err(),
            RunStoreError::ParamImmutable {
                key: "lr".to_string(),
                run_id: "run-0001".to_string()
            }
        );

        // Metrics: append-only history in order, latest projection advances.
        let p1 = MetricPoint {
            key: "rmse".to_string(),
            value: 0.9,
            timestamp_ms: 1_000,
            step: 0,
        };
        let p2 = MetricPoint {
            key: "rmse".to_string(),
            value: 0.7,
            timestamp_ms: 2_000,
            step: 1,
        };
        let older = MetricPoint {
            key: "rmse".to_string(),
            value: 0.1,
            timestamp_ms: 1_500,
            step: 2,
        };
        let same_time_higher = MetricPoint {
            key: "rmse".to_string(),
            value: 0.8,
            timestamp_ms: 2_000,
            step: 3,
        };
        store.log_metric("run-0001", p1.clone()).await.unwrap();
        let append = store.log_metric("run-0001", p2.clone()).await.unwrap();
        assert_eq!(append.history_len, 2);
        store.log_metric("run-0001", older.clone()).await.unwrap();
        store
            .log_metric("run-0001", same_time_higher.clone())
            .await
            .unwrap();
        assert_eq!(
            store.metric_history("run-0001", "rmse").await.unwrap(),
            vec![p1.clone(), p2, older, same_time_higher.clone()]
        );
        assert_eq!(
            store.get_run("run-0001").await.unwrap().latest_metrics["rmse"],
            same_time_higher
        );

        // Tags are mutable.
        store.set_tag("run-0001", "team", "search").await.unwrap();
        store.set_tag("run-0001", "team", "vector").await.unwrap();
        assert_eq!(
            store.get_run("run-0001").await.unwrap().tags["team"],
            "vector"
        );
        store.delete_tag("run-0001", "team").await.unwrap();
        assert!(
            !store
                .get_run("run-0001")
                .await
                .unwrap()
                .tags
                .contains_key("team")
        );

        // Mutations on an ABSENT run are UnknownRun — never an internal
        // error (the wire adapter maps this to RESOURCE_DOES_NOT_EXIST).
        assert_eq!(
            store.set_tag("no-such-run", "k", "v").await.unwrap_err(),
            RunStoreError::UnknownRun {
                run_id: "no-such-run".to_string()
            }
        );
        assert_eq!(
            store
                .finish_run("no-such-run", RunStatus::Finished, 1)
                .await
                .unwrap_err(),
            RunStoreError::UnknownRun {
                run_id: "no-such-run".to_string()
            }
        );

        // Counter preservation across unrelated rewrites: tag writes must
        // not disturb the metric seq (a lost counter would make the NEXT
        // append overwrite an existing history doc).
        store.set_tag("run-0001", "unrelated", "x").await.unwrap();
        store.delete_tag("run-0001", "unrelated").await.unwrap();
        store
            .log_metric(
                "run-0001",
                MetricPoint {
                    key: "rmse".to_string(),
                    value: 0.8,
                    timestamp_ms: 3_000,
                    step: 3,
                },
            )
            .await
            .unwrap();
        assert_eq!(
            store
                .metric_history("run-0001", "rmse")
                .await
                .unwrap()
                .len(),
            5,
            "seq must survive tag rewrites — history docs must never be overwritten"
        );

        // Dataset inputs are name+digest lineage records.
        let ds = RunDatasetInput {
            dataset_name: "sift1m".to_string(),
            digest: "sha256:abc".to_string(),
        };
        store
            .log_dataset_input("run-0001", ds.clone())
            .await
            .unwrap();
        assert_eq!(store.dataset_inputs("run-0001").await.unwrap(), vec![ds]);

        // Finish is one-way and freezes param writes too (port contract).
        store
            .finish_run("run-0001", RunStatus::Finished, 9_000)
            .await
            .unwrap();
        assert_eq!(
            store.log_param("run-0001", "late", "1").await.unwrap_err(),
            RunStoreError::RunFinished {
                run_id: "run-0001".to_string()
            }
        );

        // Delete -> restore must PRESERVE the finished status (MLflow
        // separates lifecycle_stage from status).
        store.delete_run("run-0001").await.unwrap();
        store.restore_run("run-0001").await.unwrap();
        let restored = store.get_run("run-0001").await.unwrap();
        assert_eq!(restored.lifecycle, RunLifecycle::Active);
        assert_eq!(restored.status, RunStatus::Finished);

        // Reopen (MLflow UpdateRun -> RUNNING): writes accepted again, end
        // time cleared; re-finish with a FAILED terminal status.
        store.reopen_run("run-0001").await.unwrap();
        let reopened = store.get_run("run-0001").await.unwrap();
        assert_eq!(reopened.status, RunStatus::Running);
        assert_eq!(reopened.end_time_ms, None);
        store
            .log_metric(
                "run-0001",
                MetricPoint {
                    key: "rmse".to_string(),
                    value: 0.05,
                    timestamp_ms: 9_600,
                    step: 4,
                },
            )
            .await
            .unwrap();
        store
            .finish_run("run-0001", RunStatus::Failed, 9_700)
            .await
            .unwrap();
        assert_eq!(
            store.get_run("run-0001").await.unwrap().status,
            RunStatus::Failed
        );
        assert_eq!(
            store
                .finish_run("run-0001", RunStatus::Running, 9_800)
                .await
                .unwrap_err(),
            RunStoreError::NotTerminal
        );
        store.reopen_run("run-0001").await.unwrap();
        store
            .finish_run("run-0001", RunStatus::Finished, 9_000)
            .await
            .unwrap();
        assert_eq!(
            store
                .log_metric(
                    "run-0001",
                    MetricPoint {
                        key: "rmse".to_string(),
                        value: 0.1,
                        timestamp_ms: 9_500,
                        step: 2,
                    }
                )
                .await
                .unwrap_err(),
            RunStoreError::RunFinished {
                run_id: "run-0001".to_string()
            }
        );

        // Cross-run metric isolation: a second run logging the same key has
        // its own history; the first run's is untouched.
        store
            .create_run(
                exp.experiment_id,
                "run-0002",
                None,
                None,
                BTreeMap::new(),
                2_000,
            )
            .await
            .unwrap();
        store
            .log_metric(
                "run-0002",
                MetricPoint {
                    key: "rmse".to_string(),
                    value: 0.5,
                    timestamp_ms: 2_500,
                    step: 0,
                },
            )
            .await
            .unwrap();
        assert_eq!(
            store
                .metric_history("run-0001", "rmse")
                .await
                .unwrap()
                .len(),
            6,
            "run-0001 history: four early points + post-tag-rewrite + post-reopen appends"
        );
        assert_eq!(
            store
                .metric_history("run-0002", "rmse")
                .await
                .unwrap()
                .len(),
            1
        );

        // Second metric key on the SAME run: per-key seq both start at 1 —
        // the doc id must disambiguate (MAJOR-2, run side).
        store
            .log_metric(
                "run-0002",
                MetricPoint {
                    key: "loss".to_string(),
                    value: 0.4,
                    timestamp_ms: 2_600,
                    step: 0,
                },
            )
            .await
            .unwrap();
        assert_eq!(
            store
                .metric_history("run-0002", "rmse")
                .await
                .unwrap()
                .len(),
            1,
            "a second metric key must never clobber the first key's history"
        );
        assert_eq!(
            store
                .metric_history("run-0002", "loss")
                .await
                .unwrap()
                .len(),
            1
        );

        // Run creation in a deleted experiment is rejected.
        store.delete_experiment(exp.experiment_id).await.unwrap();
        assert_eq!(
            store
                .create_run(
                    exp.experiment_id,
                    "run-0003",
                    None,
                    None,
                    BTreeMap::new(),
                    0
                )
                .await
                .unwrap_err(),
            RunStoreError::ExperimentDeleted {
                experiment_id: exp.experiment_id
            }
        );
        store.restore_experiment(exp.experiment_id).await.unwrap();
        store.delete_run("run-0001").await.unwrap();
        assert_eq!(
            store.get_run("run-0001").await.unwrap().status,
            RunStatus::Finished
        );
        assert_eq!(
            store.get_run("run-0001").await.unwrap().end_time_ms,
            Some(9_000)
        );

        // Listing after the delete above: run-0001 hidden (lifecycle=deleted),
        // run-0002 active; include_deleted shows both.
        assert_eq!(
            store
                .list_runs(exp.experiment_id, false)
                .await
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            store
                .list_runs(exp.experiment_id, true)
                .await
                .unwrap()
                .len(),
            2
        );

        // ---- Traces (TD-MLOPS-2): lifecycle, tags, search, delete. ----
        let trace_exp = store
            .create_experiment("traces-models", None, BTreeMap::new())
            .await
            .expect("create traces experiment");

        let trace = store
            .start_trace(TraceRecord {
                trace_id: "tr-abc123".to_string(),
                experiment_id: trace_exp.experiment_id,
                client_request_id: Some("req-1".to_string()),
                state: TraceState::Ok,
                request_preview: Some("what is proxima?".to_string()),
                response_preview: Some("a database".to_string()),
                request_time_ms: 10_000,
                execution_duration_ms: 42,
                metadata: BTreeMap::from([(
                    "mlflow.sourceRun".to_string(),
                    "run-0002".to_string(),
                )]),
                tags: BTreeMap::from([("team".to_string(), "search".to_string())]),
                assessments: Vec::new(),
            })
            .await
            .expect("start trace");
        assert_eq!(trace.trace_id, "tr-abc123");

        let dup_trace = store
            .start_trace(TraceRecord {
                trace_id: "tr-abc123".to_string(),
                experiment_id: trace_exp.experiment_id,
                client_request_id: None,
                state: TraceState::InProgress,
                request_preview: None,
                response_preview: None,
                request_time_ms: 0,
                execution_duration_ms: 0,
                metadata: BTreeMap::new(),
                tags: BTreeMap::new(),
                assessments: Vec::new(),
            })
            .await
            .unwrap_err();
        assert_eq!(
            dup_trace,
            RunStoreError::TraceIdConflict {
                trace_id: "tr-abc123".to_string()
            }
        );

        // Charset validation: ids with path separators are rejected (the id
        // reaches document ids + artifact path segments).
        assert_eq!(
            store
                .start_trace(TraceRecord {
                    trace_id: "tr/../escape".to_string(),
                    experiment_id: trace_exp.experiment_id,
                    client_request_id: None,
                    state: TraceState::InProgress,
                    request_preview: None,
                    response_preview: None,
                    request_time_ms: 0,
                    execution_duration_ms: 0,
                    metadata: BTreeMap::new(),
                    tags: BTreeMap::new(),
                    assessments: Vec::new(),
                })
                .await
                .unwrap_err(),
            RunStoreError::InvalidTraceId {
                trace_id: "tr/../escape".to_string()
            }
        );

        // Mutations on an absent trace are UnknownTrace — never internal.
        assert_eq!(
            store.set_trace_tag("tr-none", "k", "v").await.unwrap_err(),
            RunStoreError::UnknownTrace {
                trace_id: "tr-none".to_string()
            }
        );
        assert_eq!(
            store.get_trace("tr-none").await.unwrap_err(),
            RunStoreError::UnknownTrace {
                trace_id: "tr-none".to_string()
            }
        );

        // Trace tags are mutable and PRESERVE assessments (tag writes are
        // record rewrites — the payload must ride along).
        store
            .upsert_assessment(
                "tr-abc123",
                AssessmentRecord {
                    assessment_id: "as-1".to_string(),
                    assessment_name: "correctness".to_string(),
                    span_id: None,
                    source: AssessmentSource::Human,
                    source_id: Some("tester".to_string()),
                    create_time_ms: 10_100,
                    last_update_time_ms: 10_100,
                    value: AssessmentValue::FeedbackValue(serde_json::json!({"rating": 5})),
                    rationale: Some("grounded".to_string()),
                    metadata: BTreeMap::new(),
                    valid: true,
                },
            )
            .await
            .unwrap();
        store
            .set_trace_tag("tr-abc123", "team", "vector")
            .await
            .unwrap();
        let tagged = store.get_trace("tr-abc123").await.unwrap();
        assert_eq!(tagged.tags["team"], "vector");
        assert_eq!(
            tagged.assessments.len(),
            1,
            "tag rewrite must not drop assessments"
        );
        assert_eq!(
            tagged.assessments[0].value,
            AssessmentValue::FeedbackValue(serde_json::json!({"rating": 5}))
        );

        // Assessment get/delete + unknown ids typed.
        assert_eq!(
            store
                .get_assessment("tr-abc123", "as-none")
                .await
                .unwrap_err(),
            RunStoreError::UnknownAssessment {
                trace_id: "tr-abc123".to_string(),
                assessment_id: "as-none".to_string()
            }
        );
        let fetched = store.get_assessment("tr-abc123", "as-1").await.unwrap();
        assert_eq!(fetched.assessment_name, "correctness");
        store.delete_assessment("tr-abc123", "as-1").await.unwrap();
        assert_eq!(
            store
                .delete_assessment("tr-abc123", "as-1")
                .await
                .unwrap_err(),
            RunStoreError::UnknownAssessment {
                trace_id: "tr-abc123".to_string(),
                assessment_id: "as-1".to_string()
            }
        );

        // A second trace exercises deterministic per-experiment listing.
        store
            .start_trace(TraceRecord {
                trace_id: "tr-err456".to_string(),
                experiment_id: trace_exp.experiment_id,
                client_request_id: None,
                state: TraceState::Error,
                request_preview: None,
                response_preview: None,
                request_time_ms: 20_000,
                execution_duration_ms: 5_000,
                metadata: BTreeMap::from([(
                    "mlflow.sourceRun".to_string(),
                    "run-0002".to_string(),
                )]),
                tags: BTreeMap::new(),
                assessments: Vec::new(),
            })
            .await
            .unwrap();

        let listed = store.list_traces(trace_exp.experiment_id).await.unwrap();
        assert_eq!(listed.len(), 2);
        assert!(listed.iter().any(|trace| trace.trace_id == "tr-err456"));

        // delete_traces by explicit ids; repeats delete 0.
        let deleted = store
            .delete_traces(
                trace_exp.experiment_id,
                None,
                None,
                &["tr-abc123".to_string()],
            )
            .await
            .unwrap();
        assert_eq!(deleted, 1);
        assert_eq!(
            store.get_trace("tr-abc123").await.unwrap_err(),
            RunStoreError::UnknownTrace {
                trace_id: "tr-abc123".to_string()
            }
        );
        assert_eq!(
            store
                .delete_traces(
                    trace_exp.experiment_id,
                    None,
                    None,
                    &["tr-abc123".to_string()],
                )
                .await
                .unwrap(),
            0
        );

        // ---- Logged models (TD-MLOPS-2): lifecycle, params, metrics. ----
        let model = store
            .create_logged_model(LoggedModelRecord {
                model_id: "m-1000".to_string(),
                experiment_id: trace_exp.experiment_id,
                name: "greet-classifier".to_string(),
                model_type: Some("python_function".to_string()),
                source_run_id: Some("run-0002".to_string()),
                artifact_uri: format!(
                    "mlflow-artifacts:/{}/models/m-1000/artifacts",
                    trace_exp.experiment_id
                ),
                status: LoggedModelStatus::Pending,
                lifecycle: RunLifecycle::Active,
                params: BTreeMap::new(),
                tags: BTreeMap::from([("stage".to_string(), "dev".to_string())]),
                creation_time_ms: 30_000,
                last_updated_time_ms: 30_000,
            })
            .await
            .expect("create logged model");

        assert_eq!(
            store.create_logged_model(model.clone()).await.unwrap_err(),
            RunStoreError::LoggedModelIdConflict {
                model_id: "m-1000".to_string()
            }
        );
        assert_eq!(
            store.get_logged_model("m-none", false).await.unwrap_err(),
            RunStoreError::UnknownLoggedModel {
                model_id: "m-none".to_string()
            }
        );

        // Params: run-param immutability semantics.
        let mut new_params = BTreeMap::new();
        new_params.insert("lr".to_string(), "0.1".to_string());
        new_params.insert("layers".to_string(), "3".to_string());
        store
            .log_logged_model_params("m-1000", &new_params)
            .await
            .unwrap();
        store
            .log_logged_model_params("m-1000", &new_params)
            .await
            .unwrap();
        let mut diverged = BTreeMap::new();
        diverged.insert("lr".to_string(), "0.2".to_string());
        assert_eq!(
            store
                .log_logged_model_params("m-1000", &diverged)
                .await
                .unwrap_err(),
            RunStoreError::ParamImmutable {
                key: "lr".to_string(),
                run_id: "m-1000".to_string()
            }
        );

        // Tags: set + delete-absent typed error.
        let mut tag_batch = BTreeMap::new();
        tag_batch.insert("owner".to_string(), "genai".to_string());
        let tagged_model = store
            .set_logged_model_tags("m-1000", &tag_batch)
            .await
            .unwrap();
        assert_eq!(tagged_model.tags["owner"], "genai");
        assert_eq!(
            store
                .delete_logged_model_tag("m-1000", "absent")
                .await
                .unwrap_err(),
            RunStoreError::UnknownLoggedModelTag {
                model_id: "m-1000".to_string(),
                key: "absent".to_string()
            }
        );
        store
            .delete_logged_model_tag("m-1000", "stage")
            .await
            .unwrap();

        // Model metrics: append-only, isolated from run metrics and from
        // the finished-run freeze.
        store
            .log_model_metric(
                "m-1000",
                ModelMetricPoint {
                    point: MetricPoint {
                        key: "accuracy".to_string(),
                        value: 0.91,
                        timestamp_ms: 31_000,
                        step: 0,
                    },
                    run_id: Some("run-0002".to_string()),
                    dataset_name: None,
                    dataset_digest: None,
                },
            )
            .await
            .unwrap();
        let second = store
            .log_model_metric(
                "m-1000",
                ModelMetricPoint {
                    point: MetricPoint {
                        key: "accuracy".to_string(),
                        value: 0.95,
                        timestamp_ms: 32_000,
                        step: 1,
                    },
                    run_id: Some("run-0002".to_string()),
                    dataset_name: Some("holdout".to_string()),
                    dataset_digest: Some("sha256:def".to_string()),
                },
            )
            .await
            .unwrap();
        assert_eq!(second.history_len, 2);
        let history = store
            .model_metric_history("m-1000", "accuracy")
            .await
            .unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history[1].dataset_name.as_deref(), Some("holdout"));
        // Same key on a RUN is a separate history.
        assert_eq!(
            store
                .metric_history("run-0002", "accuracy")
                .await
                .unwrap()
                .len(),
            0
        );

        // Tag rewrites must not disturb model metric history (the substrate
        // rewrite carries durable counters — the in-memory truth here).
        store
            .set_logged_model_tags(
                "m-1000",
                &BTreeMap::from([("unrelated".to_string(), "x".to_string())]),
            )
            .await
            .unwrap();
        store
            .log_model_metric(
                "m-1000",
                ModelMetricPoint {
                    point: MetricPoint {
                        key: "accuracy".to_string(),
                        value: 0.97,
                        timestamp_ms: 33_000,
                        step: 2,
                    },
                    run_id: Some("run-0002".to_string()),
                    dataset_name: None,
                    dataset_digest: None,
                },
            )
            .await
            .unwrap();
        assert_eq!(
            store
                .model_metric_history("m-1000", "accuracy")
                .await
                .unwrap()
                .len(),
            3,
            "model metric history must survive tag rewrites"
        );

        // SECOND metric key on the same model: per-key counters both start
        // at 1, so a key-less doc id would silently overwrite (MAJOR-2).
        store
            .log_model_metric(
                "m-1000",
                ModelMetricPoint {
                    point: MetricPoint {
                        key: "f1".to_string(),
                        value: 0.88,
                        timestamp_ms: 31_500,
                        step: 0,
                    },
                    run_id: Some("run-0002".to_string()),
                    dataset_name: None,
                    dataset_digest: None,
                },
            )
            .await
            .unwrap();
        assert_eq!(
            store
                .model_metric_history("m-1000", "accuracy")
                .await
                .unwrap()
                .len(),
            3,
            "a second metric key must never clobber the first key's history"
        );
        assert_eq!(
            store
                .model_metric_history("m-1000", "f1")
                .await
                .unwrap()
                .len(),
            1
        );
        assert_eq!(store.model_metrics("m-1000").await.unwrap().len(), 4);

        // Finalize: status + last_updated transition; params stay writable.
        let finalized = store
            .finalize_logged_model("m-1000", LoggedModelStatus::Ready, 34_000)
            .await
            .unwrap();
        assert_eq!(finalized.status, LoggedModelStatus::Ready);
        assert_eq!(finalized.last_updated_time_ms, 34_000);
        assert!(finalized.last_updated_time_ms > finalized.creation_time_ms);
        store
            .log_logged_model_params(
                "m-1000",
                &BTreeMap::from([("epochs".to_string(), "10".to_string())]),
            )
            .await
            .unwrap();

        let listed = store
            .list_logged_models(trace_exp.experiment_id, false)
            .await
            .unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].model_id, "m-1000");

        // Soft delete: hidden from search + default get, visible with
        // allow_deleted.
        store.delete_logged_model("m-1000").await.unwrap();
        assert_eq!(
            store
                .list_logged_models(trace_exp.experiment_id, false)
                .await
                .unwrap()
                .len(),
            0
        );
        assert_eq!(
            store.get_logged_model("m-1000", false).await.unwrap_err(),
            RunStoreError::UnknownLoggedModel {
                model_id: "m-1000".to_string()
            }
        );
        assert_eq!(
            store.get_logged_model("m-1000", true).await.unwrap().status,
            LoggedModelStatus::Ready
        );

        // ---- Run <-> model links (TD-MLOPS-2): outputs + inputs embed. ----
        store
            .log_run_outputs(
                "run-0002",
                vec![ModelOutputRef {
                    model_id: "m-1000".to_string(),
                    step: 0,
                }],
            )
            .await
            .unwrap();
        store
            .log_run_model_inputs("run-0002", vec!["m-1000".to_string()])
            .await
            .unwrap();
        let linked = store.get_run("run-0002").await.unwrap();
        assert_eq!(linked.model_outputs.len(), 1);
        assert_eq!(linked.model_outputs[0].model_id, "m-1000");
        assert_eq!(linked.model_inputs, vec!["m-1000".to_string()]);
        assert_eq!(
            store
                .log_run_outputs("no-such-run", Vec::new())
                .await
                .unwrap_err(),
            RunStoreError::UnknownRun {
                run_id: "no-such-run".to_string()
            }
        );
    }
}

#[cfg(test)]
mod tests {
    use super::conformance_tests::{InMemoryRunStore, port_conformance};

    #[tokio::test]
    async fn in_memory_reference_passes_port_conformance() {
        let store = InMemoryRunStore::new();
        port_conformance(&store).await;
    }
}
