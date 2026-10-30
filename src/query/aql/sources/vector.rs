//! AQL Source implementation for Vector data model.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::query::aql::{
    AqlFrom, AqlPredicate, AqlQuery, AqlResult, AqlSource, AqlValue, AuditContext, AuditFrame,
    AuditOp, DataModel, Result,
};
use crate::services::VectorOperationsService;
use crate::services::operations::vectors::is_client_argument_error;
use proximadb_kernel::error::{ProximaDBError, QueryError};

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
        let mut query_vector = Vec::new();
        let mut top_k = 10;

        // Extract from FROM clause
        if let AqlFrom::Source { name, .. } = &query.from {
            collection = name.clone();
        }

        // Extract from WHERE clause: `SemanticMatch` carries `top_k` (its text
        // is not embedded here — no embedding service on this source), and the
        // query vector is an `Equals { embedding | vector | query_vector }`
        // literal, both possibly nested under `And`.
        if let Some(pred) = &query.where_clause.predicate {
            Self::walk_predicate(pred, &mut query_vector, &mut top_k);
        }

        (collection, query_vector, top_k)
    }

    fn walk_predicate(pred: &AqlPredicate, query_vector: &mut Vec<f32>, top_k: &mut u32) {
        match pred {
            AqlPredicate::SemanticMatch { top_k: k, .. } => *top_k = *k,
            AqlPredicate::Equals {
                field,
                value: AqlValue::Vector(v),
            } if crate::query::aql::is_query_vector_field(field) => *query_vector = v.clone(),
            AqlPredicate::And { lhs, rhs } => {
                Self::walk_predicate(lhs, query_vector, top_k);
                Self::walk_predicate(rhs, query_vector, top_k);
            }
            _ => {}
        }
    }

    /// Map a `unified_search_v1` failure to the AQL error space. A client
    /// argument error — `top_k` above the cap (message names
    /// `PROXIMADB_SEARCH_MAX_TOP_K`) or an empty / wrong-dimension query
    /// vector (TD-SECSCAN-1) — becomes `InvalidInput` (HTTP 400 on
    /// `POST /api/v2/aql/execute`); anything else is a vector-search failure.
    pub(crate) fn vector_search_error(e: anyhow::Error) -> ProximaDBError {
        if is_client_argument_error(&e) {
            ProximaDBError::InvalidInput(e.to_string())
        } else {
            ProximaDBError::Query(QueryError::VectorSearch(e.to_string()))
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
        // Fail closed on a missing query vector: an empty vector would reach
        // the distance kernel's dimension assertion and panic the handler.
        if vector.is_empty() {
            return Err(ProximaDBError::InvalidInput(
                "AQL vector search requires a query vector: add \
                 Equals { field: \"vector\", value: Vector([...]) } to the WHERE clause"
                    .to_string(),
            ));
        }
        let tenant = ctx.resolution_tenant(&collection)?;
        let start = Instant::now();

        // Perform vector search
        // For AQL, we use the vector_ops directly.
        let search_results = self
            .vector_ops
            .unified_search_v1_for_tenant(
                &collection,
                tenant,
                vector,
                top_k as usize,
                None, // No filter pushdown implemented in this skeleton
                None, // No specific search config
                None,
                None,
                proximadb_tenant::AuthClass::Anonymous,
            )
            .await
            .map_err(Self::vector_search_error)?;

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
                        let aql_val = super::sql_value_to_aql(val);
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
