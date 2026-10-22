//! AQL Source implementation for Observability data model.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::observability::{LogQueryParams, ObservabilityService};
use crate::query::aql::{
    AqlFrom, AqlQuery, AqlResult, AqlSource, AqlValue, AuditContext, AuditOp, DataModel, Result,
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
}

#[async_trait]
impl AqlSource for ObservabilityAqlSource {
    fn model(&self) -> DataModel {
        DataModel::Observability
    }

    async fn execute(&self, query: &AqlQuery, ctx: &mut AuditContext) -> Result<AqlResult> {
        let namespace = self.extract_obs_params(query);
        // Fail closed when the execution was never given a scope.
        ctx.resolution_tenant(&namespace)?;
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
                    row.insert(key, super::sql_value_to_aql(&inner));
                }
            }
            rows.push(row);
        }

        // Emit audit frame
        Ok(super::audited_ok(
            ctx,
            self.model(),
            AuditOp::DocumentQuery {
                // Reuse DocumentQuery for now or add LogQuery to AuditOp.
                collection: namespace.clone(),
            },
            rows,
            wall_time_us,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::AqlValue;
    use crate::proto::proximadb_v1::sql_value::Value as SqlValueData;
    use crate::query::aql::sources::sql_value_to_aql;

    #[test]
    fn raw_bytes_follow_the_aql_family_json_heuristic() {
        // Re-pinned round 38: the arm reverted to the sibling JSON-heuristic
        // policy (one spelling per crate — the base64 alternative is the
        // tracked AQL-family item). Non-JSON bytes null; JSON bytes decode.
        let raw = sql_value_to_aql(&SqlValueData::BytesValue(vec![0, 255]));
        assert!(matches!(raw, AqlValue::Null));
        let json = sql_value_to_aql(&SqlValueData::BytesValue(b"{\"k\":1}".to_vec()));
        assert!(matches!(json, AqlValue::Jsonb(_)));
    }
}
