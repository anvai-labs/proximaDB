//! AQL Source implementation for Graph data model.

use async_trait::async_trait;
use proximadb_graph_query::service::GraphExecutionService;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::proto::proximadb_v1::property_value::Value as PropValueData;
use crate::query::aql::{
    AqlFrom, AqlQuery, AqlResult, AqlSource, AqlValue, AuditContext, AuditFrame, AuditOp,
    DataModel, Result,
};

pub struct GraphAqlSource {
    graph_svc: Arc<dyn GraphExecutionService>,
}

impl GraphAqlSource {
    pub fn new(graph_svc: Arc<dyn GraphExecutionService>) -> Self {
        Self { graph_svc }
    }

    fn extract_graph_params(&self, query: &AqlQuery) -> (String, u32) {
        let mut graph_id = "default".to_string();
        let max_depth = 2; // Default

        if let AqlFrom::Source { name, .. } = &query.from {
            graph_id = name.clone();
        }

        (graph_id, max_depth)
    }

    fn prop_data_to_aql(val: &PropValueData) -> AqlValue {
        match val {
            PropValueData::StringValue(s) => AqlValue::String(s.clone()),
            PropValueData::IntValue(i) => AqlValue::Int(*i),
            PropValueData::DoubleValue(f) => AqlValue::Float(*f),
            PropValueData::BoolValue(b) => AqlValue::Bool(*b),
            PropValueData::BytesValue(b) => {
                // Generic bytes may carry raw JSON from legacy producers — the
                // heuristic stays; JSONB-declared fields (tag 9) decode through
                // jsonb_to_json_lossy where declared.
                if let Ok(json) = serde_json::from_slice(b) {
                    AqlValue::Jsonb(json)
                } else {
                    AqlValue::Null
                }
            }
            PropValueData::ObjectValue(obj) => {
                let mut map = serde_json::Map::new();
                for (k, v) in &obj.fields {
                    if let Some(inner_val) = &v.value {
                        map.insert(
                            k.clone(),
                            Self::aql_to_json_value(&Self::prop_data_to_aql(inner_val)),
                        );
                    }
                }
                AqlValue::Json(serde_json::Value::Object(map))
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
            AqlValue::Json(j) => j.clone(),
            AqlValue::Jsonb(j) => j.clone(),
            _ => serde_json::Value::Null,
        }
    }
}

#[async_trait]
impl AqlSource for GraphAqlSource {
    fn model(&self) -> DataModel {
        DataModel::Graph
    }

    async fn execute(&self, query: &AqlQuery, ctx: &mut AuditContext) -> Result<AqlResult> {
        let (graph_id, depth) = self.extract_graph_params(query);
        let start = Instant::now();

        let traversal_request = crate::proto::proximadb_v1::TraversalRequest {
            graph_id: graph_id.clone(),
            start_node_id: "root".to_string(),
            max_depth: depth,
            edge_types: Vec::new(),
            node_labels: Vec::new(),
            filters: Vec::new(),
            algorithm: crate::proto::proximadb_v1::TraversalAlgorithm::Bfs as i32,
            limit: Some(100),
            timeout_ms: None,
            max_frontier: None,
        };

        let traversal_result = self
            .graph_svc
            .traverse(&graph_id, traversal_request)
            .await
            .map_err(|e| {
                // Not an SST engine error — wraps a downstream graph
                // service failure. Use Internal until a dedicated query-
                // source-error variant exists.
                proximadb_kernel::error::ProximaDBError::Internal(format!(
                    "graph query source failed: {e}"
                ))
            })?;

        let wall_time_us = start.elapsed().as_micros() as u64;

        let mut rows = Vec::new();
        for node in traversal_result.nodes {
            let mut row = HashMap::new();
            row.insert("id".to_string(), AqlValue::String(node.id.clone()));
            row.insert(
                "labels".to_string(),
                AqlValue::String(node.labels.join(",")),
            );

            for (key, property) in node.properties {
                if let Some(value) = property.value {
                    let aql_val = Self::prop_data_to_aql(&value);
                    row.insert(key.clone(), aql_val.clone());

                    if key == "memory_type" {
                        row.insert("memory_type".to_string(), aql_val);
                    }
                }
            }

            rows.push(row);
        }

        let frame = AuditFrame {
            frame_id: 0,
            source: self.model(),
            op: AuditOp::GraphTraversal {
                graph_id: graph_id.clone(),
                depth,
                algorithm: "BFS".to_string(),
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
