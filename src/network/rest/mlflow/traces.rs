//! MLflow 3.x Traces wire (TD-MLOPS-2).
//!
//! The traces family lives at `/api/3.0/mlflow/traces*` (the client's
//! `_V3_TRACE_REST_API_PATH_PREFIX`), EXCEPT two stragglers the client
//! hard-codes to the 2.0 router (`traces/delete-traces` and trace-tag
//! PATCH/DELETE) — those live in [`super::mlflow_routes`].
//!
//! Span policy: we never serve `/version` or the OTLP `/v1/traces` ingest,
//! so the 3.16 client permanently uses its artifact-upload fallback — spans
//! ride `PUT /api/2.0/mlflow-artifacts/artifacts/{exp}/traces/{id}/
//! artifacts/traces.json` through the existing artifacts proxy and the
//! client downloads them itself. Our Trace responses therefore always carry
//! `spans: []` (the server never interprets span payloads).
//!
//! The `mlflow.artifactLocation` tag is load-bearing: the client reads it
//! from the StartTraceV3 response to build that artifact repo and raises
//! `MlflowTraceDataCorrupted` when absent. The wire derives the canonical URI
//! on every response and reserves the tag from PATCH/DELETE, so stored or
//! caller-supplied tags cannot redirect artifact access.

use std::collections::BTreeMap;

use axum::extract::{Path, State};
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use proximadb_catalog::run_store::{
    AssessmentRecord, AssessmentSource, AssessmentValue, RunStoreError, TraceRecord, TraceState,
};
use serde::{Deserialize, Serialize};

use crate::network::middleware::tenant::TenantContext;

use super::dto::serialize_metric_value;
use super::{MlflowError, MlflowResult, MlflowState, store_for};

pub(crate) const ARTIFACT_LOCATION_TAG: &str = "mlflow.artifactLocation";

// ---------------------------------------------------------------------------
// Timestamp / duration / enum proto3-JSON helpers (shared with the
// logged-model family via dto.rs re-exports where useful).
// ---------------------------------------------------------------------------

/// Parse a proto3-JSON `google.protobuf.Timestamp` (RFC3339 string; the
/// client always sends `Timestamp.ToJsonString()` output) to epoch ms.
pub(crate) fn ms_from_rfc3339(raw: &str) -> MlflowResult<i64> {
    chrono::DateTime::parse_from_rfc3339(raw)
        .map(|dt| dt.timestamp_millis())
        .map_err(|_| MlflowError::invalid(format!("invalid Timestamp '{raw}' (RFC 3339 expected)")))
}

/// Emit a proto3-JSON `google.protobuf.Timestamp` at millisecond precision.
pub(crate) fn rfc3339_from_ms(ms: i64) -> String {
    chrono::DateTime::from_timestamp_millis(ms)
        .unwrap_or_default()
        .to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

/// Parse a proto3-JSON `google.protobuf.Duration` ("3.5s" / "0s") to ms.
pub(crate) fn ms_from_duration(raw: &str) -> MlflowResult<i64> {
    let seconds = raw
        .strip_suffix('s')
        .ok_or_else(|| MlflowError::invalid(format!("invalid Duration '{raw}' (*s expected)")))?;
    let value: f64 = seconds
        .parse()
        .map_err(|_| MlflowError::invalid(format!("invalid Duration '{raw}'")))?;
    if !value.is_finite() || value < 0.0 {
        return Err(MlflowError::invalid(format!("invalid Duration '{raw}'")));
    }
    Ok((value * 1000.0).round() as i64)
}

/// Emit a proto3-JSON `google.protobuf.Duration` at millisecond precision.
pub(crate) fn duration_from_ms(ms: i64) -> String {
    format!("{:.3}s", ms as f64 / 1000.0)
}

fn trace_state_from_json(raw: &str) -> MlflowResult<TraceState> {
    match raw {
        "OK" => Ok(TraceState::Ok),
        "ERROR" => Ok(TraceState::Error),
        "IN_PROGRESS" => Ok(TraceState::InProgress),
        other => Err(MlflowError::invalid(format!(
            "invalid trace state '{other}' (OK | ERROR | IN_PROGRESS)"
        ))),
    }
}

fn trace_state_name(state: TraceState) -> &'static str {
    match state {
        TraceState::Ok => "OK",
        TraceState::Error => "ERROR",
        TraceState::InProgress => "IN_PROGRESS",
    }
}

/// The tag carrying the trace artifact root (client contract — see module
/// docs). Mirrors the reference server's
/// `append_to_uri_path(artifact_location, "traces", id, "artifacts")`.
pub(super) fn trace_artifact_location(experiment_id: u64, trace_id: &str) -> String {
    format!("mlflow-artifacts:/{experiment_id}/traces/{trace_id}/artifacts")
}

// ---------------------------------------------------------------------------
// DTOs
// ---------------------------------------------------------------------------

#[derive(Default, Deserialize)]
pub(crate) struct TraceLocationIn {
    #[serde(default, rename = "type")]
    pub(crate) location_type: Option<serde_json::Value>,
    #[serde(default)]
    pub(crate) mlflow_experiment: Option<MlflowExperimentRef>,
}

#[derive(Default, Deserialize)]
pub(crate) struct MlflowExperimentRef {
    #[serde(default)]
    pub(crate) experiment_id: String,
}

impl TraceLocationIn {
    /// Accept the enum as a NAME string ("MLFLOW_EXPERIMENT") or its proto
    /// number; anything else (or the only other variant, inference tables)
    /// is rejected honestly — we only store MLflow-experiment traces.
    fn experiment_id(&self) -> MlflowResult<u64> {
        match &self.location_type {
            None => {}
            Some(serde_json::Value::String(name))
                if name != "MLFLOW_EXPERIMENT" && !name.is_empty() =>
            {
                return Err(MlflowError::invalid(format!(
                    "unsupported trace location type '{name}' (MLFLOW_EXPERIMENT only)"
                )));
            }
            Some(serde_json::Value::String(_)) => {}
            Some(serde_json::Value::Number(n)) if n.as_i64() != Some(1) => {
                return Err(MlflowError::invalid(format!(
                    "unsupported trace location type {n} (MLFLOW_EXPERIMENT = 1)"
                )));
            }
            Some(serde_json::Value::Number(_)) => {}
            Some(other) => {
                return Err(MlflowError::invalid(format!(
                    "invalid trace location type {other}"
                )));
            }
        }
        let id = self
            .mlflow_experiment
            .as_ref()
            .map(|e| e.experiment_id.as_str());
        match id {
            Some(raw) if !raw.is_empty() => raw
                .parse::<u64>()
                .map_err(|_| MlflowError::invalid(format!("invalid experiment id '{raw}'"))),
            _ => Err(MlflowError::invalid(
                "trace_location.mlflow_experiment.experiment_id is required",
            )),
        }
    }
}

#[derive(Default, Deserialize)]
pub(crate) struct AssessmentSourceIn {
    #[serde(default, rename = "type")]
    pub(crate) source_type: Option<serde_json::Value>,
    #[serde(default)]
    pub(crate) source_id: Option<String>,
}

impl AssessmentSourceIn {
    fn parse(&self) -> MlflowResult<(AssessmentSource, Option<String>)> {
        let source = match &self.source_type {
            None => Ok(AssessmentSource::Human),
            Some(serde_json::Value::String(name)) => match name.as_str() {
                "" | "HUMAN" => Ok(AssessmentSource::Human),
                "LLM_JUDGE" => Ok(AssessmentSource::LlmJudge),
                "CODE" => Ok(AssessmentSource::Code),
                other => Err(MlflowError::invalid(format!(
                    "invalid assessment source type '{other}' (HUMAN | LLM_JUDGE | CODE)"
                ))),
            },
            Some(serde_json::Value::Number(n)) => match n.as_i64() {
                Some(1) => Ok(AssessmentSource::Human),
                Some(2) => Ok(AssessmentSource::LlmJudge),
                Some(3) => Ok(AssessmentSource::Code),
                other => Err(MlflowError::invalid(format!(
                    "invalid assessment source type {other:?}"
                ))),
            },
            Some(other) => Err(MlflowError::invalid(format!(
                "invalid assessment source type {other}"
            ))),
        }?;
        Ok((source, self.source_id.clone()))
    }
}

#[derive(Default, Deserialize)]
pub(crate) struct AssessmentFeedbackIn {
    #[serde(default)]
    pub(crate) value: Option<serde_json::Value>,
    #[serde(default)]
    pub(crate) error: Option<AssessmentErrorIn>,
}

#[derive(Default, Deserialize, Serialize)]
pub(crate) struct AssessmentErrorIn {
    #[serde(default)]
    pub(crate) error_code: String,
    #[serde(default)]
    pub(crate) error_message: String,
    #[serde(default)]
    pub(crate) stack_trace: Option<String>,
}

#[derive(Default, Deserialize)]
pub(crate) struct AssessmentExpectationIn {
    #[serde(default)]
    pub(crate) value: Option<serde_json::Value>,
}

#[derive(Default, Deserialize)]
pub(crate) struct AssessmentIssueIn {
    #[serde(default)]
    pub(crate) issue_name: String,
}

#[derive(Default, Deserialize)]
pub(crate) struct AssessmentIn {
    #[serde(default)]
    pub(crate) assessment_id: Option<String>,
    #[serde(default)]
    pub(crate) assessment_name: String,
    #[serde(default)]
    pub(crate) span_id: Option<String>,
    #[serde(default)]
    pub(crate) source: Option<AssessmentSourceIn>,
    #[serde(default)]
    pub(crate) feedback: Option<AssessmentFeedbackIn>,
    #[serde(default)]
    pub(crate) expectation: Option<AssessmentExpectationIn>,
    #[serde(default)]
    pub(crate) issue: Option<AssessmentIssueIn>,
    #[serde(default)]
    pub(crate) rationale: Option<String>,
    #[serde(default)]
    pub(crate) metadata: BTreeMap<String, String>,
    #[serde(default)]
    pub(crate) valid: Option<bool>,
}

impl AssessmentIn {
    /// Build the port record. Exactly one payload variant must be present —
    /// the client's `Assessment.from_proto` raises on a response with none
    /// or several, so we reject the same shape on the way in.
    fn into_record(self, _trace_id: &str, now_ms: i64) -> MlflowResult<AssessmentRecord> {
        if self.assessment_name.is_empty() {
            return Err(MlflowError::invalid("assessment_name must not be empty"));
        }
        let (source, source_id) = self
            .source
            .as_ref()
            .map(AssessmentSourceIn::parse)
            .transpose()?
            .unwrap_or((AssessmentSource::Human, None));
        let mut variants = Vec::new();
        if let Some(feedback) = &self.feedback
            && (feedback.error.is_some() || feedback.value.is_some())
        {
            variants.push(feedback_variant(feedback)?);
        }
        if let Some(expectation) = &self.expectation
            && let Some(value) = &expectation.value
        {
            variants.push(AssessmentValue::ExpectationValue(value.clone()));
        }
        if self
            .issue
            .as_ref()
            .is_some_and(|issue| !issue.issue_name.is_empty())
        {
            variants.push(AssessmentValue::Issue {
                issue_name: self
                    .issue
                    .as_ref()
                    .map(|i| i.issue_name.clone())
                    .unwrap_or_default(),
            });
        }
        if variants.len() != 1 {
            return Err(MlflowError::invalid(format!(
                "assessment must carry exactly one of feedback/expectation/issue (got {})",
                variants.len()
            )));
        }
        let assessment_id = self
            .assessment_id
            .filter(|id| !id.is_empty())
            .unwrap_or_else(|| format!("as-{}", super::uuid_like_id()));
        Ok(AssessmentRecord {
            assessment_id,
            assessment_name: self.assessment_name,
            span_id: self.span_id,
            source,
            source_id,
            create_time_ms: now_ms,
            last_update_time_ms: now_ms,
            value: variants
                .pop()
                .unwrap_or(AssessmentValue::FeedbackValue(serde_json::Value::Null)),
            rationale: self.rationale,
            metadata: self.metadata,
            valid: self.valid.unwrap_or(true),
        })
        // trace_id rides the record's owner (the port call carries it),
        // not the payload.
    }
}

/// Protobuf's JSON serializer camelCases FieldMask segments, so the 3.16
/// client sends "assessmentName" for the proto field `assessment_name`.
/// Accept both spellings (review MAJOR-1: snake-only matching rejected
/// EVERY real `mlflow.update_assessment` call).
fn normalize_mask_path(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    for c in path.chars() {
        if c.is_ascii_uppercase() {
            out.push('_');
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

#[derive(Serialize)]
pub(crate) struct AssessmentSourceOut {
    #[serde(rename = "type")]
    source_type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_id: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct AssessmentOut {
    assessment_id: String,
    assessment_name: String,
    trace_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    span_id: Option<String>,
    source: AssessmentSourceOut,
    create_time: String,
    last_update_time: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    feedback: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expectation: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    issue: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rationale: Option<String>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    metadata: BTreeMap<String, String>,
    valid: bool,
}

fn assessment_out(trace_id: &str, record: &AssessmentRecord) -> AssessmentOut {
    let source_type = match record.source {
        AssessmentSource::Human => "HUMAN",
        AssessmentSource::LlmJudge => "LLM_JUDGE",
        AssessmentSource::Code => "CODE",
    };
    let (feedback, expectation, issue) = match &record.value {
        AssessmentValue::FeedbackValue(value) => {
            (Some(serde_json::json!({ "value": value })), None, None)
        }
        AssessmentValue::FeedbackError(error) => (
            Some(serde_json::json!({
                "error": {
                    "error_code": error.error_code,
                    "error_message": error.error_message,
                    "stack_trace": error.stack_trace,
                }
            })),
            None,
            None,
        ),
        AssessmentValue::ExpectationValue(value) => {
            (None, Some(serde_json::json!({ "value": value })), None)
        }
        AssessmentValue::Issue { issue_name } => (
            None,
            None,
            Some(serde_json::json!({ "issue_name": issue_name })),
        ),
    };
    AssessmentOut {
        assessment_id: record.assessment_id.clone(),
        assessment_name: record.assessment_name.clone(),
        trace_id: trace_id.to_string(),
        span_id: record.span_id.clone(),
        source: AssessmentSourceOut {
            source_type,
            source_id: record.source_id.clone(),
        },
        create_time: rfc3339_from_ms(record.create_time_ms),
        last_update_time: rfc3339_from_ms(record.last_update_time_ms),
        feedback,
        expectation,
        issue,
        rationale: record.rationale.clone(),
        metadata: record.metadata.clone(),
        valid: record.valid,
    }
}

#[derive(Default, Deserialize)]
pub(crate) struct TraceInfoIn {
    #[serde(default)]
    pub(crate) trace_id: String,
    #[serde(default)]
    pub(crate) client_request_id: Option<String>,
    #[serde(default)]
    pub(crate) trace_location: Option<TraceLocationIn>,
    #[serde(default)]
    pub(crate) request_preview: Option<String>,
    #[serde(default)]
    pub(crate) response_preview: Option<String>,
    #[serde(default)]
    pub(crate) request_time: Option<String>,
    #[serde(default)]
    pub(crate) execution_duration: Option<String>,
    #[serde(default)]
    pub(crate) state: Option<String>,
    #[serde(default)]
    pub(crate) trace_metadata: BTreeMap<String, String>,
    #[serde(default)]
    pub(crate) tags: BTreeMap<String, String>,
}

#[derive(Default, Deserialize)]
pub(crate) struct StartTraceRequest {
    #[serde(default)]
    pub(crate) trace: Option<TraceIn>,
}

#[derive(Default, Deserialize)]
pub(crate) struct TraceIn {
    #[serde(default)]
    pub(crate) trace_info: Option<TraceInfoIn>,
    /// Accepted and ignored: the OSS client sends `spans: []` here (span
    /// data rides artifacts — see module docs). A non-empty array is
    /// rejected honestly rather than silently dropped.
    #[serde(default)]
    pub(crate) spans: Vec<serde_json::Value>,
}

#[derive(Serialize)]
pub(crate) struct TraceLocationOut {
    #[serde(rename = "type")]
    location_type: &'static str,
    mlflow_experiment: MlflowExperimentOut,
}

#[derive(Serialize)]
pub(crate) struct MlflowExperimentOut {
    experiment_id: String,
}

#[derive(Serialize)]
pub(crate) struct TraceInfoOut {
    trace_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    client_request_id: Option<String>,
    trace_location: TraceLocationOut,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_preview: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_preview: Option<String>,
    request_time: String,
    execution_duration: String,
    state: &'static str,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    trace_metadata: BTreeMap<String, String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    assessments: Vec<AssessmentOut>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    tags: BTreeMap<String, String>,
}

#[derive(Serialize)]
pub(crate) struct TraceOut {
    trace_info: TraceInfoOut,
    /// Always empty server-side (spans live in the artifact repo; the
    /// client fetches them itself).
    spans: Vec<serde_json::Value>,
}

pub(crate) fn trace_info_out(record: &TraceRecord) -> TraceInfoOut {
    let mut tags = record.tags.clone();
    // Derive this capability from immutable owner fields on every read. This
    // repairs any legacy/corrupt stored value and prevents client-controlled
    // redirects even if a non-wire store implementation accepted one.
    tags.insert(
        ARTIFACT_LOCATION_TAG.to_string(),
        trace_artifact_location(record.experiment_id, &record.trace_id),
    );
    TraceInfoOut {
        trace_id: record.trace_id.clone(),
        client_request_id: record.client_request_id.clone(),
        trace_location: TraceLocationOut {
            location_type: "MLFLOW_EXPERIMENT",
            mlflow_experiment: MlflowExperimentOut {
                experiment_id: record.experiment_id.to_string(),
            },
        },
        request_preview: record.request_preview.clone(),
        response_preview: record.response_preview.clone(),
        request_time: rfc3339_from_ms(record.request_time_ms),
        execution_duration: duration_from_ms(record.execution_duration_ms),
        state: trace_state_name(record.state),
        trace_metadata: record.metadata.clone(),
        assessments: record
            .assessments
            .iter()
            .map(|a| assessment_out(&record.trace_id, a))
            .collect(),
        tags,
    }
}

fn trace_out(record: &TraceRecord) -> TraceOut {
    TraceOut {
        trace_info: trace_info_out(record),
        spans: Vec::new(),
    }
}

// Batch reads intentionally omit missing trace ids, matching MLflow, while
// preserving every other store failure. This helper keeps that distinction
// independently testable without a substrate-specific mock.
fn batch_trace_result(
    result: Result<TraceRecord, proximadb_catalog::run_store::RunStoreError>,
) -> Result<Option<TraceRecord>, proximadb_catalog::run_store::RunStoreError> {
    match result {
        Ok(record) => Ok(Some(record)),
        Err(proximadb_catalog::run_store::RunStoreError::UnknownTrace { .. }) => Ok(None),
        Err(error) => Err(error),
    }
}

#[derive(Default, Deserialize)]
pub(crate) struct GetTraceRequest {
    #[serde(default)]
    pub(crate) trace_id: String,
    /// Accepted for wire compatibility: our traces carry no server-side
    /// spans (the client downloads them from the artifact repo), so there
    /// is no partial-trace state to allow.
    #[serde(default)]
    #[allow(dead_code)]
    pub(crate) allow_partial: Option<bool>,
}

#[derive(Default, Deserialize)]
pub(crate) struct BatchGetTracesRequest {
    #[serde(default)]
    pub(crate) trace_ids: Vec<String>,
}

#[derive(Default, Deserialize)]
pub(crate) struct SearchTracesRequest {
    #[serde(default)]
    pub(crate) locations: Vec<TraceLocationIn>,
    #[serde(default)]
    pub(crate) filter: Option<String>,
    #[serde(default)]
    pub(crate) max_results: Option<u32>,
    /// Repeated STRING (`"timestamp DESC"` — the client and the UI bundle
    /// both send exactly that), not the object form SearchLoggedModels
    /// uses.
    #[serde(default)]
    pub(crate) order_by: Vec<String>,
    #[serde(default)]
    pub(crate) page_token: Option<String>,
}

#[derive(Default, Deserialize)]
pub(crate) struct CreateAssessmentRequest {
    #[serde(default)]
    pub(crate) assessment: Option<AssessmentIn>,
}

impl CreateAssessmentRequest {
    /// The caller-supplied assessment id, when the request carried one
    /// (server-minted ids are always fresh by construction).
    fn supplied_id(&self) -> Option<&str> {
        self.assessment
            .as_ref()
            .and_then(|a| a.assessment_id.as_deref())
            .filter(|id| !id.is_empty())
    }
}

#[derive(Default, Deserialize)]
pub(crate) struct UpdateAssessmentRequest {
    #[serde(default)]
    pub(crate) assessment: Option<AssessmentIn>,
    #[serde(default)]
    pub(crate) update_mask: Option<String>,
}

/// The v3 traces router — nested at `/api/3.0/mlflow` (all paths carry the
/// family's `/traces` prefix; static segments like `/traces/get` win over
/// the `/{trace_id}` parameter route in matchit).
pub fn traces_routes() -> Router<MlflowState> {
    Router::new()
        .route("/traces", post(traces_create))
        .route("/traces/get", get(traces_get))
        .route("/traces/batchGet", get(traces_batch_get))
        .route("/traces/batchGetInfos", post(traces_batch_get_infos))
        .route("/traces/search", post(traces_search))
        .route("/traces/{trace_id}", get(traces_get_by_id))
        .route("/traces/{trace_id}/assessments", post(assessments_create))
        .route(
            "/traces/{trace_id}/assessments/{assessment_id}",
            get(assessments_get)
                .patch(assessments_update)
                .delete(assessments_delete),
        )
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn traces_create(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<StartTraceRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    let Some(trace) = req.trace else {
        return Err(MlflowError::invalid("start-trace requires 'trace'"));
    };
    if !trace.spans.is_empty() {
        return Err(MlflowError::invalid(
            "spans cannot be sent inline (this server stores trace data as artifacts; spans must ride the artifact upload)",
        ));
    }
    let Some(info) = trace.trace_info else {
        return Err(MlflowError::invalid(
            "start-trace requires 'trace.trace_info'",
        ));
    };
    if info.trace_id.is_empty() {
        return Err(MlflowError::invalid(
            "trace_info.trace_id must not be empty",
        ));
    }
    let experiment_id = info
        .trace_location
        .as_ref()
        .map(TraceLocationIn::experiment_id)
        .transpose()?
        .ok_or_else(|| MlflowError::invalid("trace_info.trace_location is required"))?;
    let now_ms = chrono::Utc::now().timestamp_millis();
    let record = TraceRecord {
        trace_id: info.trace_id.clone(),
        experiment_id,
        client_request_id: info.client_request_id,
        state: match info.state.as_deref() {
            None => TraceState::InProgress,
            Some("") => TraceState::InProgress,
            Some(raw) => trace_state_from_json(raw)?,
        },
        request_preview: info.request_preview,
        response_preview: info.response_preview,
        request_time_ms: match info.request_time.as_deref() {
            None | Some("") => now_ms,
            Some(raw) => ms_from_rfc3339(raw)?,
        },
        execution_duration_ms: match info.execution_duration.as_deref() {
            None | Some("") => 0,
            Some(raw) => ms_from_duration(raw)?,
        },
        metadata: info.trace_metadata,
        tags: {
            let mut tags = info.tags;
            // Server-owned, load-bearing (module docs): the client builds
            // its trace artifact repo from this tag.
            tags.insert(
                ARTIFACT_LOCATION_TAG.to_string(),
                trace_artifact_location(experiment_id, &info.trace_id),
            );
            tags
        },
        assessments: Vec::new(),
    };
    let store = store_for(&tenant, &state)?;
    let record = match store.start_trace(record.clone()).await {
        Ok(saved) => saved,
        Err(conflict @ RunStoreError::TraceIdConflict { .. }) => {
            // A concurrent delete can race the re-read — the conflict is
            // the honest answer either way, never a 404.
            match store.get_trace(&record.trace_id).await {
                Ok(existing) if trace_retry_equivalent(&existing, &record) => existing,
                _ => return Err(conflict.into()),
            }
        }
        Err(error) => return Err(error.into()),
    };
    Ok(Json(serde_json::json!({ "trace": trace_out(&record) })))
}

/// Exporters retry StartTrace after ambiguous transport failures. Treat the
/// same immutable trace envelope as an idempotent replay while preserving any
/// assessments or extra tags attached after the original create.
fn trace_retry_equivalent(existing: &TraceRecord, incoming: &TraceRecord) -> bool {
    existing.trace_id == incoming.trace_id
        && existing.experiment_id == incoming.experiment_id
        && existing.client_request_id == incoming.client_request_id
        && existing.state == incoming.state
        && existing.request_preview == incoming.request_preview
        && existing.response_preview == incoming.response_preview
        && existing.request_time_ms == incoming.request_time_ms
        && existing.execution_duration_ms == incoming.execution_duration_ms
        && existing.metadata == incoming.metadata
        && incoming
            .tags
            .iter()
            .all(|(key, value)| existing.tags.get(key) == Some(value))
}

async fn traces_get(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    super::MlflowRead(req): super::MlflowRead<GetTraceRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    if req.trace_id.is_empty() {
        return Err(MlflowError::invalid("trace_id must not be empty"));
    }
    let record = store_for(&tenant, &state)?.get_trace(&req.trace_id).await?;
    Ok(Json(serde_json::json!({ "trace": trace_out(&record) })))
}

async fn traces_get_by_id(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Path(trace_id): Path<String>,
) -> MlflowResult<Json<serde_json::Value>> {
    let record = store_for(&tenant, &state)?.get_trace(&trace_id).await?;
    Ok(Json(serde_json::json!({ "trace": trace_out(&record) })))
}

async fn traces_batch_get(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    axum::extract::RawQuery(query): axum::extract::RawQuery,
) -> MlflowResult<Json<serde_json::Value>> {
    // The client sends repeated keys (trace_ids=a&trace_ids=b) which
    // serde_urlencoded cannot deserialize into a Vec (review MAJOR-2) —
    // collect every occurrence manually, tolerating one bare value.
    let query = query.unwrap_or_default();
    let mut trace_ids: Vec<String> = Vec::new();
    for (key, value) in url::form_urlencoded::parse(query.as_bytes()) {
        if key == "trace_ids" && !value.is_empty() {
            trace_ids.push(value.into_owned());
        }
    }
    let req = BatchGetTracesRequest { trace_ids };
    let store = store_for(&tenant, &state)?;
    let mut traces = Vec::new();
    // MLflow semantics: absent ids are skipped, not an error.
    for trace_id in &req.trace_ids {
        if let Some(record) = batch_trace_result(store.get_trace(trace_id).await)? {
            traces.push(trace_out(&record));
        }
    }
    Ok(Json(serde_json::json!({ "traces": traces })))
}

async fn traces_batch_get_infos(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<BatchGetTracesRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    let store = store_for(&tenant, &state)?;
    let mut infos = Vec::new();
    for trace_id in &req.trace_ids {
        if let Some(record) = batch_trace_result(store.get_trace(trace_id).await)? {
            infos.push(trace_info_out(&record));
        }
    }
    Ok(Json(serde_json::json!({ "trace_infos": infos })))
}

async fn traces_search(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Json(req): Json<SearchTracesRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    if req.locations.is_empty() {
        return Err(MlflowError::invalid(
            "locations must list at least one MLFLOW_EXPERIMENT",
        ));
    }
    let mut experiment_ids = Vec::new();
    for location in &req.locations {
        let experiment_id = location.experiment_id()?;
        if !experiment_ids.contains(&experiment_id) {
            experiment_ids.push(experiment_id);
        }
    }
    let clauses = req
        .filter
        .as_deref()
        .map(super::filter::parse_trace_filter)
        .transpose()?
        .unwrap_or_default();
    // Default order is newest-first; the only supported explicit order is
    // timestamp ASC/DESC — anything else is an honest INVALID_PARAMETER_VALUE.
    let mut ascending = false;
    for order in &req.order_by {
        let clause = order.trim();
        let (field, dir) = clause
            .rsplit_once(' ')
            .ok_or_else(|| MlflowError::invalid(format!("invalid order_by '{clause}'")))?;
        match (field.trim(), dir.trim().to_ascii_uppercase().as_str()) {
            ("timestamp", "ASC") => ascending = true,
            ("timestamp", "DESC") => ascending = false,
            _ => {
                return Err(MlflowError::invalid(format!(
                    "unsupported order_by '{clause}' (timestamp ASC|DESC only)"
                )));
            }
        }
    }
    let store = store_for(&tenant, &state)?;
    let mut records = Vec::new();
    for experiment_id in experiment_ids {
        records.extend(store.list_traces(experiment_id).await?);
    }
    records.retain(|record| clauses.iter().all(|clause| clause.matches(record)));
    records.sort_by(|a, b| {
        let order = if ascending {
            a.request_time_ms.cmp(&b.request_time_ms)
        } else {
            b.request_time_ms.cmp(&a.request_time_ms)
        };
        order.then_with(|| a.trace_id.cmp(&b.trace_id))
    });
    let infos: Vec<TraceInfoOut> = records.iter().map(trace_info_out).collect();
    let (infos, next_page_token) =
        super::paginate(infos, req.max_results, req.page_token.as_deref())?;
    let mut body = serde_json::json!({ "traces": infos });
    if let Some(token) = next_page_token {
        body["next_page_token"] = serde_json::Value::String(token);
    }
    Ok(Json(body))
}

async fn assessments_create(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Path(trace_id): Path<String>,
    Json(req): Json<CreateAssessmentRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    let supplied_id = req.supplied_id().map(str::to_string);
    let Some(assessment) = req.assessment else {
        return Err(MlflowError::invalid("assessment is required"));
    };
    let now_ms = chrono::Utc::now().timestamp_millis();
    let record = assessment.into_record(&trace_id, now_ms)?;
    let store = store_for(&tenant, &state)?;
    // A caller-supplied id that already exists is a create conflict (the
    // reference errors; upsert semantics belong to PATCH — review NIT-3).
    if let Some(id) = supplied_id.as_deref()
        && store.get_assessment(&trace_id, id).await.is_ok()
    {
        return Err(MlflowError::conflict(format!(
            "Assessment '{id}' already exists on trace '{trace_id}'"
        )));
    }
    // The trace must exist first (typed UnknownTrace, never a silent
    // orphan).
    let saved = store.upsert_assessment(&trace_id, record).await?;
    Ok(Json(
        serde_json::json!({ "assessment": assessment_out(&trace_id, &saved) }),
    ))
}

async fn assessments_get(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Path((trace_id, assessment_id)): Path<(String, String)>,
) -> MlflowResult<Json<serde_json::Value>> {
    let record = store_for(&tenant, &state)?
        .get_assessment(&trace_id, &assessment_id)
        .await?;
    Ok(Json(
        serde_json::json!({ "assessment": assessment_out(&trace_id, &record) }),
    ))
}

async fn assessments_update(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Path((trace_id, assessment_id)): Path<(String, String)>,
    Json(req): Json<UpdateAssessmentRequest>,
) -> MlflowResult<Json<serde_json::Value>> {
    let Some(incoming) = req.assessment else {
        return Err(MlflowError::invalid("assessment is required"));
    };
    let store = store_for(&tenant, &state)?;
    let mut current = store.get_assessment(&trace_id, &assessment_id).await?;
    let now_ms = chrono::Utc::now().timestamp_millis();
    // Protobuf JSON renders FieldMask as one comma-separated string. MLflow
    // requires it for assessment PATCH and permits only mutable assessment
    // fields; identity, owner, source, span and value kind stay immutable.
    let Some(mask) = req.update_mask else {
        return Err(MlflowError::invalid("update_mask is required"));
    };
    let mask = mask
        .split(',')
        .map(|path| normalize_mask_path(path.trim()))
        .filter(|path| !path.is_empty())
        .collect::<Vec<_>>();
    if mask.is_empty() {
        return Err(MlflowError::invalid("update_mask must not be empty"));
    }
    for path in &mask {
        match path.as_str() {
            "assessment_name" => {
                if incoming.assessment_name.is_empty() {
                    return Err(MlflowError::invalid("assessment_name must not be empty"));
                }
                current.assessment_name = incoming.assessment_name.clone();
            }
            "feedback" => {
                if !matches!(
                    current.value,
                    AssessmentValue::FeedbackValue(_) | AssessmentValue::FeedbackError(_)
                ) {
                    return Err(MlflowError::invalid("assessment value kind is immutable"));
                }
                let Some(feedback) = &incoming.feedback else {
                    return Err(MlflowError::invalid(
                        "update_mask path 'feedback' requires feedback in the payload",
                    ));
                };
                current.value = feedback_variant(feedback)?;
            }
            "expectation" => {
                if !matches!(current.value, AssessmentValue::ExpectationValue(_)) {
                    return Err(MlflowError::invalid("assessment value kind is immutable"));
                }
                let Some(value) = incoming.expectation.as_ref().and_then(|e| e.value.clone())
                else {
                    return Err(MlflowError::invalid(
                        "update_mask path 'expectation' requires expectation.value",
                    ));
                };
                current.value = AssessmentValue::ExpectationValue(value);
            }
            "issue" => {
                if !matches!(current.value, AssessmentValue::Issue { .. }) {
                    return Err(MlflowError::invalid("assessment value kind is immutable"));
                }
                let Some(issue_name) = incoming
                    .issue
                    .as_ref()
                    .map(|i| i.issue_name.clone())
                    .filter(|name| !name.is_empty())
                else {
                    return Err(MlflowError::invalid(
                        "update_mask path 'issue' requires issue.issue_name",
                    ));
                };
                current.value = AssessmentValue::Issue { issue_name };
            }
            "rationale" => current.rationale = incoming.rationale.clone(),
            "metadata" => current.metadata.extend(incoming.metadata.clone()),
            "valid" => {
                current.valid = incoming.valid.ok_or_else(|| {
                    MlflowError::invalid("update_mask path 'valid' requires valid")
                })?;
            }
            other => {
                return Err(MlflowError::invalid(format!(
                    "unsupported update_mask path '{other}'"
                )));
            }
        }
    }
    current.last_update_time_ms = now_ms;
    let saved = store.upsert_assessment(&trace_id, current).await?;
    Ok(Json(
        serde_json::json!({ "assessment": assessment_out(&trace_id, &saved) }),
    ))
}

/// Extract the feedback oneof variant from an incoming feedback object
/// (value XOR error — a feedback with neither is rejected).
fn feedback_variant(feedback: &AssessmentFeedbackIn) -> MlflowResult<AssessmentValue> {
    match (&feedback.value, &feedback.error) {
        (Some(_), Some(_)) => Err(MlflowError::invalid(
            "feedback must carry exactly one of value or error",
        )),
        (None, Some(error)) => Ok(AssessmentValue::FeedbackError(
            proximadb_catalog::run_store::AssessmentError {
                error_code: error.error_code.clone(),
                error_message: error.error_message.clone(),
                stack_trace: error.stack_trace.clone(),
            },
        )),
        (Some(value), None) => Ok(AssessmentValue::FeedbackValue(value.clone())),
        (None, None) => Err(MlflowError::invalid(
            "feedback must carry exactly one of value or error",
        )),
    }
}

async fn assessments_delete(
    State(state): State<MlflowState>,
    Extension(tenant): Extension<TenantContext>,
    Path((trace_id, assessment_id)): Path<(String, String)>,
) -> MlflowResult<Json<serde_json::Value>> {
    store_for(&tenant, &state)?
        .delete_assessment(&trace_id, &assessment_id)
        .await?;
    Ok(Json(serde_json::json!({})))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_catalog::run_store::RunStoreError;

    #[test]
    fn batch_reads_propagate_internal_store_failures() {
        let result = batch_trace_result(Err(RunStoreError::Internal {
            message: "injected substrate failure".to_string(),
        }));
        assert!(matches!(result, Err(RunStoreError::Internal { .. })));
    }

    #[test]
    fn batch_reads_omit_only_unknown_traces() {
        let result = batch_trace_result(Err(RunStoreError::UnknownTrace {
            trace_id: "missing".to_string(),
        }))
        .expect("unknown trace is an intentional omission");
        assert!(result.is_none());
    }
}

// The metric-value serializer is shared; re-export for the logged-model
// DTOs in the sibling module.
#[allow(unused_imports)]
use serialize_metric_value as _serialize_metric_value;
