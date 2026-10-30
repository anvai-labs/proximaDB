//! AQL Source implementation for Document data model.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::query::aql::{
    AqlFrom, AqlQuery, AqlResult, AqlSource, AqlValue, AuditContext, AuditOp, DataModel, Result,
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
}

#[async_trait]
impl AqlSource for DocumentAqlSource {
    fn model(&self) -> DataModel {
        DataModel::Document
    }

    async fn execute(&self, query: &AqlQuery, ctx: &mut AuditContext) -> Result<AqlResult> {
        let collection = self.extract_doc_params(query);
        // The storage key is composed within the request tenant (the same key the
        // document surfaces use); internal execution reads the name as given.
        let storage_key = match ctx.resolution_tenant(&collection)? {
            Some(tenant) => {
                crate::storage::document::service::scoped_document_collection(tenant, &collection)
                    .map_err(|_| crate::query::aql::collection_not_found(&collection))?
            }
            None => collection.clone(),
        };
        let start = Instant::now();

        // Perform document query
        let params = DocumentQueryParams {
            limit: 100,
            ..Default::default()
        };

        let query_result = self
            .doc_svc
            .query_documents(&storage_key, params)
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
                    let aql_val = super::sql_value_to_aql(&val);
                    row.insert(k.clone(), aql_val.clone());
                }
            }
            rows.push(row);
        }

        // Emit audit frame
        Ok(super::audited_ok(
            ctx,
            self.model(),
            AuditOp::DocumentQuery {
                collection: collection.clone(),
            },
            rows,
            wall_time_us,
        ))
    }
}
