//! MLflow REST DTOs (serde snake_case, string ids — JavaScript-safe).
//! Extracted from the god-file (audit #5); handlers stay in mod.rs.

use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub(crate) struct KeyValue {
    pub(crate) key: String,
    pub(crate) value: String,
}

#[derive(Serialize)]
pub(crate) struct KeyValueOut {
    pub(crate) key: String,
    pub(crate) value: String,
}

#[derive(Default, Deserialize)]
pub(crate) struct ExperimentsCreateRequest {
    pub(crate) name: String,
    #[serde(default)]
    pub(crate) artifact_location: Option<String>,
    #[serde(default)]
    pub(crate) tags: Vec<KeyValue>,
}

#[derive(Serialize)]
pub(crate) struct ExperimentOut {
    pub(crate) experiment_id: String,
    pub(crate) name: String,
    pub(crate) artifact_uri: String,
    pub(crate) lifecycle_stage: &'static str,
    pub(crate) creation_time: i64,
    pub(crate) last_update_time: i64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) tags: Vec<KeyValueOut>,
}

#[derive(Default, Deserialize)]
pub(crate) struct ExperimentIdRequest {
    #[serde(default)]
    pub(crate) experiment_id: String,
}

#[derive(Default, Deserialize)]
pub(crate) struct ExperimentNameRequest {
    /// Proto field name is `experiment_name`; older JSON bodies use `name`.
    #[serde(default, alias = "experiment_name")]
    pub(crate) name: String,
}

#[derive(Default, Deserialize)]
pub(crate) struct ExperimentsSearchRequest {
    #[serde(default)]
    pub(crate) max_results: Option<u32>,
    #[serde(default)]
    pub(crate) filter: Option<String>,
    #[serde(default)]
    pub(crate) view_type: Option<String>,
    #[serde(default)]
    pub(crate) page_token: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct ExperimentsSearchResponse {
    pub(crate) experiments: Vec<ExperimentOut>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) next_page_token: Option<String>,
}

#[derive(Default, Deserialize)]
pub(crate) struct RunsCreateRequest {
    pub(crate) experiment_id: String,
    #[serde(default)]
    pub(crate) start_time: Option<i64>,
    #[serde(default)]
    pub(crate) run_name: Option<String>,
    #[serde(default)]
    pub(crate) tags: Vec<KeyValue>,
}

#[derive(Serialize)]
pub(crate) struct MetricOut {
    pub(crate) key: String,
    #[serde(serialize_with = "serialize_metric_value")]
    pub(crate) value: f64,
    pub(crate) timestamp: i64,
    pub(crate) step: i64,
}

#[derive(Serialize)]
pub(crate) struct ParamOut {
    pub(crate) key: String,
    pub(crate) value: String,
}

#[derive(Serialize)]
pub(crate) struct RunData {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) metrics: Vec<MetricOut>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) params: Vec<ParamOut>,
    pub(crate) tags: Vec<KeyValueOut>,
}

#[derive(Serialize)]
pub(crate) struct RunInfo {
    pub(crate) run_id: String,
    pub(crate) experiment_id: String,
    pub(crate) status: &'static str,
    pub(crate) start_time: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) end_time: Option<i64>,
    pub(crate) lifecycle_stage: &'static str,
    pub(crate) artifact_uri: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) run_name: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct RunOut {
    pub(crate) info: RunInfo,
    pub(crate) data: RunData,
}

#[derive(Default, Deserialize)]
pub(crate) struct RunIdRequest {
    #[serde(default)]
    pub(crate) run_id: String,
}

#[derive(Default, Deserialize)]
pub(crate) struct RunsUpdateRequest {
    #[serde(default)]
    pub(crate) run_id: String,
    #[serde(default)]
    pub(crate) status: Option<String>,
    #[serde(default)]
    pub(crate) end_time: Option<i64>,
}

#[derive(Serialize)]
pub(crate) struct RunInfoResponse {
    pub(crate) run_info: RunInfo,
}

#[derive(Default, Deserialize)]
pub(crate) struct LogParameterRequest {
    #[serde(default)]
    pub(crate) run_id: String,
    #[serde(default)]
    pub(crate) key: String,
    #[serde(default)]
    pub(crate) value: String,
}

#[derive(Default, Deserialize)]
pub(crate) struct MetricInput {
    #[serde(default)]
    pub(crate) key: String,
    #[serde(default, deserialize_with = "deserialize_metric_value")]
    pub(crate) value: f64,
    #[serde(default)]
    pub(crate) timestamp: i64,
    #[serde(default)]
    pub(crate) step: i64,
}

#[derive(Default, Deserialize)]
pub(crate) struct LogMetricRequest {
    #[serde(default)]
    pub(crate) run_id: String,
    #[serde(default)]
    pub(crate) key: String,
    #[serde(default, deserialize_with = "deserialize_metric_value")]
    pub(crate) value: f64,
    #[serde(default)]
    pub(crate) timestamp: i64,
    #[serde(default)]
    pub(crate) step: i64,
}

#[derive(Default, Deserialize)]
pub(crate) struct LogBatchRequest {
    #[serde(default)]
    pub(crate) run_id: String,
    #[serde(default)]
    pub(crate) metrics: Vec<MetricInput>,
    #[serde(default)]
    pub(crate) params: Vec<LogParameterRequest>,
    #[serde(default)]
    pub(crate) tags: Vec<KeyValue>,
}

#[derive(Default, Deserialize)]
pub(crate) struct SetTagRequest {
    #[serde(default)]
    pub(crate) run_id: String,
    #[serde(default)]
    pub(crate) key: String,
    #[serde(default)]
    pub(crate) value: String,
}

#[derive(Default, Deserialize)]
pub(crate) struct RunsSearchRequest {
    #[serde(default)]
    pub(crate) experiment_ids: Vec<String>,
    #[serde(default)]
    pub(crate) filter: Option<String>,
    #[serde(default)]
    pub(crate) run_view_type: Option<String>,
    #[serde(default)]
    pub(crate) max_results: Option<u32>,
    #[serde(default)]
    pub(crate) order_by: Vec<String>,
    #[serde(default)]
    pub(crate) page_token: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct RunsSearchResponse {
    pub(crate) runs: Vec<RunOut>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) next_page_token: Option<String>,
}

#[derive(Default, Deserialize)]
pub(crate) struct MetricHistoryRequest {
    #[serde(default)]
    pub(crate) run_id: String,
    #[serde(default, alias = "metric_key")]
    pub(crate) key: String,
}

pub(crate) fn serialize_metric_value<S: serde::Serializer>(
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

pub(crate) fn deserialize_metric_value<'de, D: serde::Deserializer<'de>>(
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
