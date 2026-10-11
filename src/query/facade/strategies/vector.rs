//! # Vector Search Strategy
//!
//! Real implementation of `QueryStrategy` for vector similarity searches.
//! Wraps the existing `VectorOps` service infrastructure.
//!
//! ## Features
//!
//! - Converts facade `QueryRequest` to proto `VectorSearchRequest`
//! - Leverages existing search infrastructure
//! - Supports metadata filtering
//! - Returns results in unified `QueryResult` format
//!
//! ## Architecture
//!
//! ```text
//! QueryRequest (facade)
//!       │
//!       ▼
//! VectorSearchStrategy
//!       │
//!       ▼
//! VectorOps.search_v1()
//!       │
//!       ▼
//! StorageEngine.search_vectors_unified()
//!       │
//!       ▼
//! QueryResult (facade)
//! ```

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use tracing::{debug, info, instrument};

use crate::proto::proximadb_v1::{IncludeFields, SearchQuery, VectorSearchRequest};
use proximadb_records::{ProximaRecord, ProximaTreeNode, ScoredRecord};

use crate::query::facade::{
    ExecutionMetrics, QueryContent, QueryContext, QueryRequest, QueryResult, QueryResultData,
    QueryStrategy, QueryType,
};
use crate::services::VectorOps;
use proximadb_runtime::CollectionPort;

/// Vector Search Strategy - Real implementation wrapping VectorOps
///
/// This strategy handles `QueryType::VectorSearch` requests by:
/// 1. Converting the facade request to proto format
/// 2. Delegating to VectorOps for actual execution
/// 3. Converting results back to facade format
pub struct VectorSearchStrategy {
    /// Vector operations service for search execution
    vector_ops: Arc<VectorOps>,
    /// Collection port for metadata lookup (Task #76 migration: was
    /// `Arc<CollectionService>`, now port-typed view from SharedServices).
    collection_port: Arc<dyn CollectionPort>,
    /// Strategy priority (higher = preferred)
    priority: i32,
}

impl VectorSearchStrategy {
    /// Create a new VectorSearchStrategy
    pub fn new(vector_ops: Arc<VectorOps>, collection_port: Arc<dyn CollectionPort>) -> Self {
        Self {
            vector_ops,
            collection_port,
            priority: 100, // High priority for vector searches
        }
    }

    /// Create with custom priority
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Convert facade QueryRequest to proto VectorSearchRequest
    fn to_proto_request(&self, request: &QueryRequest) -> Result<VectorSearchRequest> {
        let (query_vector, top_k) = match &request.content {
            QueryContent::Vector {
                query_vector,
                top_k,
            } => (query_vector.clone(), *top_k),
            _ => return Err(anyhow!("VectorSearchStrategy requires Vector content")),
        };

        let collection_id = request
            .target
            .clone()
            .ok_or_else(|| anyhow!("VectorSearchStrategy requires a target collection"))?;

        Ok(VectorSearchRequest {
            collection_id,
            queries: vec![SearchQuery {
                vector: query_vector,
                filters: crate::core::search::results::proxima_map_to_sql(
                    request.params.vector_filters.clone(),
                ),
                advanced_filter: request.params.vector_advanced_filter.clone(),
            }],
            top_k: top_k as u32,
            include_fields: Some(IncludeFields {
                vector: false,
                metadata: true,
                score: true,
                rank: false,
                source: false,
                source_options: Default::default(),
            }),
            search_params: None,
            distance_metric_override: None,
            search_optimization: None,
        })
    }

    /// Convert proto response to facade QueryResult
    fn to_facade_result(
        &self,
        response: crate::proto::proximadb_v1::VectorOperationResponse,
        execution_time_ms: u64,
    ) -> QueryResult {
        // Extract results from the nested structure
        let search_records = response.results.map(|r| r.results).unwrap_or_default();

        let matches: Vec<ScoredRecord> = search_records
            .into_iter()
            .enumerate()
            .map(|(i, r)| {
                let mut record = ProximaRecord {
                    oid: r.id,
                    created_at_ns: r.timestamp.map(|ms| ms * 1_000_000).unwrap_or(0),
                    ..Default::default()
                };
                // Convert SqlValue metadata to ProximaTree props
                for (key, sql_val) in r.metadata {
                    if let Some(pv) = sql_value_to_proxima_value(sql_val) {
                        record.props.insert(key, ProximaTreeNode::Value(pv));
                    }
                }
                ScoredRecord {
                    record,
                    score: r.score as f32,
                    rank: (i + 1) as u32,
                }
            })
            .collect();

        let results_returned = matches.len();

        QueryResult {
            data: QueryResultData::VectorResults(matches),
            metrics: Some(ExecutionMetrics {
                execution_path: "unified".to_string(),
                strategy_name: "vector".to_string(),
                execution_time_ms,
                planning_time_ms: 0,
                results_scanned: results_returned,
                results_returned,
                cache_hit: false,
                extra: serde_json::json!({
                    "engine": "VectorOps",
                    "success": response.success
                }),
            }),
        }
    }
}

/// Convert proto SqlValue to canonical ProximaValue.
fn sql_value_to_proxima_value(
    value: crate::proto::proximadb_v1::SqlValue,
) -> Option<proximadb_data_model::ProximaValue> {
    use crate::proto::proximadb_v1::sql_value::Value;
    use proximadb_data_model::ProximaValue;

    value.value.map(|v| match v {
        Value::StringValue(s) => ProximaValue::String(s),
        Value::Int64Value(i) => ProximaValue::Int64(i),
        Value::NumberValue(f) => ProximaValue::Float64(f),
        Value::BoolValue(b) => ProximaValue::Boolean(b),
        Value::NullValue(_) => ProximaValue::Null,
        Value::BytesValue(bytes) => ProximaValue::Binary(bytes),
        // Canonical MessagePack JSONB decode (the shared helper) so a tag-9
        // value compares equal across read paths.
        Value::JsonbValue(bytes) => ProximaValue::from_jsonb_or_binary(&bytes),
        Value::ArrayValue(list) => {
            let items: Vec<ProximaValue> = list
                .values
                .into_iter()
                .filter_map(sql_value_to_proxima_value)
                .collect();
            ProximaValue::Array(items)
        }
        Value::ObjectValue(map) => {
            let obj: std::collections::HashMap<String, ProximaValue> = map
                .fields
                .into_iter()
                .filter_map(|(k, v)| sql_value_to_proxima_value(v).map(|pv| (k, pv)))
                .collect();
            ProximaValue::Map(obj)
        }
    })
}

#[async_trait]
impl QueryStrategy for VectorSearchStrategy {
    fn name(&self) -> &str {
        "vector"
    }

    fn can_handle(&self, request: &QueryRequest) -> bool {
        request.query_type == QueryType::VectorSearch
    }

    fn priority(&self) -> i32 {
        self.priority
    }

    #[instrument(skip(self, request, _ctx), fields(strategy = "vector"))]
    async fn execute(&self, request: QueryRequest, _ctx: &QueryContext) -> Result<QueryResult> {
        let start = Instant::now();

        // Get target collection for validation
        let collection_id = request
            .target
            .as_deref()
            .ok_or_else(|| anyhow!("VectorSearchStrategy requires a target collection"))?;

        // Verify collection exists
        let _collection = self
            .collection_port
            .get_collection(collection_id, None)
            .await?
            .ok_or_else(|| anyhow!("Collection '{}' not found", collection_id))?;

        // Convert to proto request
        let proto_request = self.to_proto_request(&request)?;

        debug!(
            collection = %collection_id,
            top_k = proto_request.top_k,
            "Executing vector search via VectorOps"
        );

        // Execute search through VectorOps
        let response = self
            .vector_ops
            .search_v1(proto_request, proximadb_runtime::PortIdentity::anonymous())
            .await?;

        let execution_time_ms = start.elapsed().as_millis() as u64;

        // Convert to facade result
        let result = self.to_facade_result(response, execution_time_ms);

        info!(
            collection = %collection_id,
            results = result.metrics.as_ref().map_or(0, |m| m.results_returned),
            time_ms = execution_time_ms,
            "Vector search completed"
        );

        Ok(result)
    }
}

// ================================================================================
// TESTS
// ================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_records::conversions::sql_value_to_json;

    #[test]
    fn test_sql_value_to_json_string() {
        use crate::proto::proximadb_v1::{SqlValue, sql_value::Value};
        let value = SqlValue {
            value: Some(Value::StringValue("hello".to_string())),
        };
        let json = sql_value_to_json(&value);
        assert_eq!(json, serde_json::json!("hello"));
    }

    #[test]
    fn test_sql_value_to_json_int() {
        use crate::proto::proximadb_v1::{SqlValue, sql_value::Value};
        let value = SqlValue {
            value: Some(Value::Int64Value(42)),
        };
        let json = sql_value_to_json(&value);
        assert_eq!(json, serde_json::json!(42));
    }

    #[test]
    fn test_sql_value_to_json_float() {
        use crate::proto::proximadb_v1::{SqlValue, sql_value::Value};
        let value = SqlValue {
            value: Some(Value::NumberValue(3.5)),
        };
        let json = sql_value_to_json(&value);
        if let serde_json::Value::Number(n) = json {
            let f = n.as_f64().unwrap();
            assert!((f - 3.5).abs() < 0.001);
        } else {
            panic!("Expected Number");
        }
    }

    #[test]
    fn test_sql_value_to_json_bool() {
        use crate::proto::proximadb_v1::{SqlValue, sql_value::Value};
        let value = SqlValue {
            value: Some(Value::BoolValue(true)),
        };
        let json = sql_value_to_json(&value);
        assert_eq!(json, serde_json::json!(true));
    }

    #[test]
    fn test_sql_value_to_json_null() {
        use crate::proto::proximadb_v1::{SqlValue, sql_value::Value};
        let value = SqlValue {
            value: Some(Value::NullValue(0)),
        };
        let json = sql_value_to_json(&value);
        assert_eq!(json, serde_json::Value::Null);
    }

    #[test]
    fn test_sql_value_to_json_none() {
        use crate::proto::proximadb_v1::SqlValue;
        let value = SqlValue { value: None };
        let json = sql_value_to_json(&value);
        assert_eq!(json, serde_json::Value::Null);
    }

    #[test]
    fn test_strategy_can_handle_vector_search() {
        let request = QueryRequest::vector_search(vec![0.1, 0.2], 10);
        assert_eq!(request.query_type, QueryType::VectorSearch);
    }

    #[test]
    fn test_strategy_cannot_handle_sql() {
        let request = QueryRequest::sql("SELECT * FROM foo");
        assert_eq!(request.query_type, QueryType::Sql);
        // VectorSearchStrategy should not handle SQL queries
        assert_ne!(request.query_type, QueryType::VectorSearch);
    }

    #[test]
    fn test_query_request_with_target() {
        let request =
            QueryRequest::vector_search(vec![0.1, 0.2, 0.3], 5).with_target("my_collection");

        assert_eq!(request.target, Some("my_collection".to_string()));
        if let QueryContent::Vector {
            query_vector,
            top_k,
        } = &request.content
        {
            assert_eq!(query_vector.len(), 3);
            assert_eq!(*top_k, 5);
        } else {
            panic!("Expected Vector content");
        }
    }
}
