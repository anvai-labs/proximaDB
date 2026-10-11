//! AQL Source implementation for Document data model.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::proto::proximadb_v1::sql_value::Value as SqlValueData;
use crate::query::aql::{
    AqlFrom, AqlQuery, AqlResult, AqlSource, AqlValue, AuditContext, AuditFrame, AuditOp,
    DataModel, Result,
};
use crate::storage::document::{DocumentQueryParams, DocumentService};

pub struct DocumentAqlSource {
    doc_svc: Arc<DocumentService>,
}

impl DocumentAqlSource {
    pub fn new(doc_svc: Arc<DocumentService>) -> Self {
        Self { doc_svc }
    }

    fn extract_doc_params(&self, query: &AqlQuery) -> String {
        if let AqlFrom::Source { name, .. } = &query.from {
            name.clone()
        } else {
            "default".to_string()
        }
    }

    fn sql_data_to_aql(val: &SqlValueData) -> AqlValue {
        match val {
            SqlValueData::StringValue(s) => AqlValue::String(s.clone()),
            SqlValueData::Int64Value(i) => AqlValue::Int(*i),
            SqlValueData::NumberValue(f) => AqlValue::Float(*f),
            SqlValueData::BoolValue(b) => AqlValue::Bool(*b),
            SqlValueData::BytesValue(b) => {
                // Generic bytes may carry raw JSON from legacy producers — the
                // heuristic stays; JSONB-declared fields (tag 9) decode through
                // jsonb_to_json_lossy below.
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
impl AqlSource for DocumentAqlSource {
    fn model(&self) -> DataModel {
        DataModel::Document
    }

    async fn execute(&self, query: &AqlQuery, ctx: &mut AuditContext) -> Result<AqlResult> {
        let collection = self.extract_doc_params(query);
        let start = Instant::now();

        // Perform document query
        let params = DocumentQueryParams {
            limit: 100,
            ..Default::default()
        };

        let query_result = self
            .doc_svc
            .query_documents(&collection, params)
            .await
            .map_err(|e| {
                // Not an SST engine error — wraps a downstream document
                // service failure. Use Internal until a dedicated query-
                // source-error variant exists.
                proximadb_kernel::error::ProximaDBError::Internal(format!(
                    "document query source failed: {e}"
                ))
            })?;

        let wall_time_us = start.elapsed().as_micros() as u64;

        // Convert to AQL rows
        let mut rows = Vec::new();
        for doc in query_result.documents {
            let mut row = HashMap::new();
            row.insert("_id".to_string(), AqlValue::String(doc.id.clone()));

            for (k, v) in crate::storage::document::proxima_tree_to_sql_object(&doc.props).fields {
                if let Some(val) = v.value {
                    let aql_val = Self::sql_data_to_aql(&val);
                    row.insert(k.clone(), aql_val.clone());
                }
            }
            rows.push(row);
        }

        // Emit audit frame
        let frame = AuditFrame {
            frame_id: 0,
            source: self.model(),
            op: AuditOp::DocumentQuery {
                collection: collection.clone(),
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
