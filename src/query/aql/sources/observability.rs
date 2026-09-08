//! AQL Source implementation for Observability data model.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::observability::{LogQueryParams, ObservabilityService};
use crate::proto::proximadb_v1::sql_value::Value as SqlValueData;
use crate::query::aql::{
    AqlFrom, AqlQuery, AqlResult, AqlSource, AqlValue, AuditContext, AuditFrame, AuditOp,
    DataModel, Result,
};

pub struct ObservabilityAqlSource {
    obs_svc: Arc<ObservabilityService>,
}

impl ObservabilityAqlSource {
    pub fn new(obs_svc: Arc<ObservabilityService>) -> Self {
        Self { obs_svc }
    }

    fn extract_obs_params(&self, query: &AqlQuery) -> String {
        if let AqlFrom::Source { name, .. } = &query.from {
            name.clone()
        } else {
            "default".to_string()
        }
    }

    fn sql_data_to_aql(value: SqlValueData) -> AqlValue {
        match value {
            SqlValueData::StringValue(s) => AqlValue::String(s),
            SqlValueData::Int64Value(i) => AqlValue::Int(i),
            SqlValueData::NumberValue(f) => AqlValue::Float(f),
            SqlValueData::BoolValue(b) => AqlValue::Bool(b),
            // Same JSON heuristic as the document/vector/graph siblings
            // (deliberate for legacy producers — one policy per crate, and
            // a cross-source predicate sees the same spelling everywhere;
            // the base64 alternative is the tracked AQL family item).
            SqlValueData::BytesValue(b) => {
                if let Ok(json) = serde_json::from_slice(&b) {
                    AqlValue::Jsonb(json)
                } else {
                    AqlValue::Null
                }
            }
            // types.proto tag 9: JSONB by declaration — same JSON heuristic as
            // BytesValue so a JSONB field is not silently nulled by the `_`
            // wildcard.
            // TD-PROTO-2: tag 9 is JSONB by declaration — canonical MessagePack decode
            // (the one sanctioned JSONB->JSON representation), never a raw-JSON
            // heuristic that diverges from the API-facing surfaces.
            SqlValueData::JsonbValue(bytes) => AqlValue::Jsonb(
                proximadb_data_model::ProximaValue::jsonb_to_json_lossy(&bytes),
            ),
            SqlValueData::ObjectValue(obj) => {
                let mut map = serde_json::Map::new();
                for (key, value) in obj.fields {
                    if let Some(inner) = value.value {
                        map.insert(key, Self::aql_to_json_value(Self::sql_data_to_aql(inner)));
                    }
                }
                AqlValue::Jsonb(serde_json::Value::Object(map))
            }
            SqlValueData::ArrayValue(arr) => {
                let values = arr
                    .values
                    .into_iter()
                    .map(|value| {
                        value
                            .value
                            .map(Self::sql_data_to_aql)
                            .map(Self::aql_to_json_value)
                            .unwrap_or(serde_json::Value::Null)
                    })
                    .collect();
                AqlValue::Jsonb(serde_json::Value::Array(values))
            }
            _ => AqlValue::Null,
        }
    }

    fn aql_to_json_value(value: AqlValue) -> serde_json::Value {
        match value {
            AqlValue::String(s) => serde_json::Value::String(s),
            AqlValue::Int(i) => serde_json::json!(i),
            AqlValue::Float(f) => serde_json::json!(f),
            AqlValue::Bool(b) => serde_json::Value::Bool(b),
            AqlValue::Json(j) | AqlValue::Jsonb(j) => j,
            _ => serde_json::Value::Null,
        }
    }
}

#[async_trait]
impl AqlSource for ObservabilityAqlSource {
    fn model(&self) -> DataModel {
        DataModel::Observability
    }

    async fn execute(&self, query: &AqlQuery, ctx: &mut AuditContext) -> Result<AqlResult> {
        let namespace = self.extract_obs_params(query);
        let start = Instant::now();

        let now_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
        let params = LogQueryParams {
            start_time_ns: now_ns - (3600 * 1_000_000_000),
            end_time_ns: now_ns,
            query: None,
            severities: Vec::new(),
            services: Vec::new(),
            sources: Vec::new(),
            limit: 100,
            cursor: None,
        };

        let log_result = self
            .obs_svc
            .query_logs(&namespace, params)
            .await
            .map_err(|e| {
                // Not an SST engine error — wraps a downstream observability
                // service failure. Use Internal until a dedicated query-
                // source-error variant exists.
                proximadb_kernel::error::ProximaDBError::Internal(format!(
                    "observability query source failed: {e}"
                ))
            })?;

        let wall_time_us = start.elapsed().as_micros() as u64;

        let mut rows = Vec::new();
        for log in log_result.logs {
            let mut row = HashMap::new();
            row.insert("timestamp".to_string(), AqlValue::Int(log.timestamp_ns));
            row.insert("severity".to_string(), AqlValue::Int(log.severity.into()));
            row.insert("message".to_string(), AqlValue::String(log.message.clone()));
            if let Some(svc) = log.service {
                row.insert("service".to_string(), AqlValue::String(svc));
            }

            for (key, value) in log.fields {
                if let Some(inner) = value.value {
                    row.insert(key, Self::sql_data_to_aql(inner));
                }
            }
            rows.push(row);
        }

        let frame = AuditFrame {
            frame_id: 0,
            source: self.model(),
            op: AuditOp::DocumentQuery {
                // Reuse DocumentQuery for now or add LogQuery to AuditOp.
                collection: namespace.clone(),
            },
            filters_pushed: Vec::new(),
            filters_post: Vec::new(),
            records_scanned: rows.len() as u64,
            records_returned: rows.len() as u64,
            wall_time_us,
            error: None,
            redaction_count: 0,
        };

        let frame_id = ctx.push_frame(frame);

        Ok(AqlResult { rows, frame_id })
    }
}

#[cfg(test)]
mod tests {
    use super::{AqlValue, ObservabilityAqlSource, SqlValueData};

    #[test]
    fn raw_bytes_follow_the_aql_family_json_heuristic() {
        // Re-pinned round 38: the arm reverted to the sibling JSON-heuristic
        // policy (one spelling per crate — the base64 alternative is the
        // tracked AQL-family item). Non-JSON bytes null; JSON bytes decode.
        let raw = ObservabilityAqlSource::sql_data_to_aql(SqlValueData::BytesValue(vec![0, 255]));
        assert!(matches!(raw, AqlValue::Null));
        let json = ObservabilityAqlSource::sql_data_to_aql(SqlValueData::BytesValue(
            b"{\"k\":1}".to_vec(),
        ));
        assert!(matches!(json, AqlValue::Jsonb(_)));
    }
}
