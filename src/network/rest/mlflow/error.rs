//! The MLflow wire error envelope — one home for the typed transport
//! errors and their translations (audit #5: extracted from the god-file).

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

use proximadb_catalog::model_registry_service::ModelRegistryServiceError;
use proximadb_catalog::run_store::RunStoreError;

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

    /// Id-existence conflicts (duplicate trace/model ids): the MLflow error
    /// code with HTTP 409 — the resource state, not a bad parameter.
    pub(crate) fn conflict(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::CONFLICT,
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
            // Deliberately NOT the reference server's "Trace with ID 'x'
            // not found" phrasing — the 3.x client pattern-matches exactly
            // that literal on /traces/get 404s and converts it to
            // NotImplemented; ours must stay a plain RESOURCE_DOES_NOT_EXIST.
            RunStoreError::UnknownTrace { trace_id } => {
                MlflowError::not_found(format!("Could not find trace with ID '{trace_id}'"))
            }
            RunStoreError::TraceIdConflict { trace_id } => {
                MlflowError::conflict(format!("Trace '{trace_id}' already exists"))
            }
            RunStoreError::InvalidTraceId { trace_id } => MlflowError::invalid(format!(
                "trace id '{trace_id}' contains invalid characters (allowed: A-Za-z0-9._~-)"
            )),
            RunStoreError::UnknownAssessment {
                trace_id,
                assessment_id,
            } => MlflowError::not_found(format!(
                "Could not find assessment '{assessment_id}' on trace '{trace_id}'"
            )),
            RunStoreError::UnknownLoggedModel { model_id } => {
                MlflowError::not_found(format!("Logged model with ID '{model_id}' not found"))
            }
            RunStoreError::LoggedModelIdConflict { model_id } => {
                MlflowError::conflict(format!("Logged model '{model_id}' already exists"))
            }
            RunStoreError::InvalidModelStatus { status } => MlflowError::invalid(format!(
                "status '{status}' is not a valid logged-model status"
            )),
            RunStoreError::UnknownLoggedModelTag { model_id, key } => MlflowError::not_found(
                format!("Tag '{key}' not found on logged model '{model_id}'"),
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

pub(crate) type MlflowResult<T> = Result<T, MlflowError>;
