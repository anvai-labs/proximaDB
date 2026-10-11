//! AQL Source implementation for Vector data model.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::proto::proximadb_v1::sql_value::Value as SqlValueData;
use crate::query::aql::{
    AqlFrom, AqlPredicate, AqlQuery, AqlResult, AqlSource, AqlValue, AuditContext, AuditFrame,
    AuditOp, DataModel, Result,
};
use crate::services::VectorOperationsService;

pub struct VectorAqlSource {
    vector_ops: Arc<VectorOperationsService>,
}

impl VectorAqlSource {
    pub fn new(vector_ops: Arc<VectorOperationsService>) -> Self {
        Self { vector_ops }
    }

    fn extract_vector_search_params(&self, query: &AqlQuery) -> (String, Vec<f32>, u32) {
        // Default parameters
        let mut collection = "default".to_string();
        let query_vector = Vec::new();
        let mut top_k = 10;

        // Extract from FROM clause
        if let AqlFrom::Source { name, .. } = &query.from {
            collection = name.clone();
        }

        // Extract from WHERE clause
        if let Some(AqlPredicate::SemanticMatch {
            query: _q_text,
            threshold: _,
            top_k: k,
            ..
        }) = &query.where_clause.predicate
        {
            // In a real implementation, we'd call an embedding service for q_text.
            // For now, we assume query_vector is provided in the request or handled by caller.
            top_k = *k;
        }

        (collection, query_vector, top_k)
    }

    pub(crate) fn sql_data_to_aql(val: &SqlValueData) -> AqlValue {
        match val {
            SqlValueData::StringValue(s) => AqlValue::String(s.clone()),
            SqlValueData::Int64Value(i) => AqlValue::Int(*i),
            SqlValueData::NumberValue(f) => AqlValue::Float(*f),
            SqlValueData::BoolValue(b) => AqlValue::Bool(*b),
            SqlValueData::BytesValue(b) => {
                // Generic bytes may carry raw JSON from legacy producers — the
                // heuristic stays; JSONB-declared fields (tag 9) are canonical
                // MessagePack and decode through jsonb_to_json_lossy below.
                if let Ok(json) = serde_json::from_slice(b) {
                    AqlValue::Jsonb(json)
                } else {
                    AqlValue::Null
                }
            }
            // types.proto tag 9: JSONB by declaration — same JSON heuristic as
            // BytesValue so a JSONB field is not silently nulled by the `_`
            // wildcard.
            SqlValueData::JsonbValue(b) => {
                // TD-PROTO-2: canonical MessagePack JSONB decode (the one
                // sanctioned JSONB->JSON representation) — not a raw-JSON
                // heuristic that diverges from the API-facing surfaces.
                AqlValue::Jsonb(proximadb_data_model::ProximaValue::jsonb_to_json_lossy(b))
            }
            SqlValueData::ObjectValue(obj) => {
                let mut map = serde_json::Map::new();
                for (k, v) in &obj.fields {
                    if let Some(inner_val) = &v.value {
                        map.insert(
                            k.clone(),
                            Self::aql_to_json_value(&Self::sql_data_to_aql(inner_val)),
                        );
                    }
                }
                AqlValue::Jsonb(serde_json::Value::Object(map))
            }
            SqlValueData::ArrayValue(arr) => {
                let values = arr
                    .values
                    .iter()
                    .map(|value| {
                        value
                            .value
                            .as_ref()
                            .map(Self::sql_data_to_aql)
                            .map(|value| Self::aql_to_json_value(&value))
                            .unwrap_or(serde_json::Value::Null)
                    })
                    .collect();
                AqlValue::Jsonb(serde_json::Value::Array(values))
            }
            _ => AqlValue::Null,
        }
    }

    fn aql_to_json_value(aql: &AqlValue) -> serde_json::Value {
        match aql {
            AqlValue::String(s) => serde_json::Value::String(s.clone()),
            AqlValue::Int(i) => serde_json::json!(i),
            AqlValue::Float(f) => serde_json::json!(f),
            AqlValue::Bool(b) => serde_json::Value::Bool(*b),
            AqlValue::Json(j) | AqlValue::Jsonb(j) => j.clone(),
            _ => serde_json::Value::Null,
        }
    }
}

#[async_trait]
impl AqlSource for VectorAqlSource {
    fn model(&self) -> DataModel {
        DataModel::Vector
    }

    async fn execute(&self, query: &AqlQuery, ctx: &mut AuditContext) -> Result<AqlResult> {
        let (collection, vector, top_k) = self.extract_vector_search_params(query);
        let start = Instant::now();

        // Perform vector search
        // For AQL, we use the vector_ops directly.
        let search_results = self
            .vector_ops
            .unified_search_v1(
                &collection,
                vector,
                top_k as usize,
                None, // No filter pushdown implemented in this skeleton
                None, // No specific search config
                None,
                None,
                proximadb_tenant::AuthClass::Anonymous,
            )
            .await
            .map_err(|e| {
                // Not an SST engine error — wraps a downstream vector
                // service failure. Map to the query subsystem error.
                proximadb_kernel::error::ProximaDBError::Query(
                    proximadb_kernel::error::QueryError::VectorSearch(e.to_string()),
                )
            })?;

        let wall_time_us = start.elapsed().as_micros() as u64;

        // search_results is Vec<SearchResult>, let's get the first one's results
        let mut rows = Vec::new();
        let mut total_records = 0;

        if let Some(res_batch) = search_results.first() {
            total_records = res_batch.results.len() as u64;
            for res in &res_batch.results {
                let mut row = HashMap::new();
                row.insert("id".to_string(), AqlValue::String(res.id.clone()));
                row.insert("score".to_string(), AqlValue::Float(res.score));

                for (k, v) in &res.metadata {
                    if let Some(val) = &v.value {
                        let aql_val = Self::sql_data_to_aql(val);
                        row.insert(k.clone(), aql_val.clone());
                    }
                }
                rows.push(row);
            }
        }

        // Emit audit frame
        let frame = AuditFrame {
            frame_id: 0, // Set by ctx.push_frame
            source: self.model(),
            op: AuditOp::VectorSearch {
                collection: collection.clone(),
                top_k,
                metric: "Cosine".to_string(), // Default
            },
            filters_pushed: Vec::new(), // TODO: Serialize predicates
            filters_post: Vec::new(),
            records_scanned: total_records, // Approximated
            records_returned: rows.len() as u64,
            wall_time_us,
            error: None,
            redaction_count: 0,
        };

        let frame_id = ctx.push_frame(frame);

        Ok(AqlResult { rows, frame_id })
    }
}
