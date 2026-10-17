// Cold tier document retrieval
//
// This module provides abstractions for retrieving documents from cold storage
// (documents that have been flushed to the storage engine but evicted from the hot cache).
//
// Design follows SOLID principles:
// - Single Responsibility: Filter building is separate from search execution
// - Open/Closed: ColdTierRetriever trait allows for different storage backend implementations
// - Dependency Inversion: Uses traits for storage access, not concrete implementations

use anyhow::Result;
use std::sync::Arc;

use crate::core::search::SearchParams;
use crate::proto::proximadb_v1::Collection;
use crate::storage::document::DocumentRecord;
use crate::storage::traits::{StorageQueryContext, UnifiedStorageFormat};
use proximadb_filter_expression::{ComparisonOperator, FilterExpression};

/// Most ids one cold-tier engine lookup may carry: the engine's result limit
/// (`SearchParams::top_k`) is a `u16`.
const COLD_TIER_MAX_IDS_PER_LOOKUP: usize = u16::MAX as usize;

// =============================================================================
// TRAIT: ColdTierRetriever (Dependency Inversion Principle)
// =============================================================================

/// Trait for retrieving documents from cold storage tier.
///
/// This trait abstracts the cold tier retrieval mechanism, allowing for
/// different implementations based on the underlying storage backend.
///
/// Implementations can be:
/// - Storage engine based (SST, VIPER, etc.)
/// - External storage (S3, GCS, etc.)
/// - Hybrid approaches
#[async_trait::async_trait]
pub trait ColdTierRetriever: Send + Sync {
    /// Retrieve documents by their IDs from cold storage.
    ///
    /// # Arguments
    /// * `collection` - The collection name/ID
    /// * `ids` - List of document IDs to retrieve
    ///
    /// # Returns
    /// Vector of found documents (may be fewer than requested if some IDs not found)
    async fn retrieve_documents(
        &self,
        collection: &str,
        ids: &[&str],
    ) -> Result<Vec<DocumentRecord>>;

    /// Check if documents exist in cold storage.
    ///
    /// # Arguments
    /// * `collection` - The collection name/ID
    /// * `ids` - List of document IDs to check
    ///
    /// # Returns
    /// Vector of IDs that exist in cold storage
    async fn check_existence(&self, collection: &str, ids: &[&str]) -> Result<Vec<String>>;
}

// =============================================================================
// STRUCT: DocumentMetadataFilterBuilder (Single Responsibility Principle)
// =============================================================================

/// Builder for creating metadata filters to find documents in storage.
///
/// This struct is responsible solely for constructing the filter expressions
/// needed to locate documents in the storage engine.
///
/// Documents are stored with these metadata fields:
/// - `_type`: "document" (type marker)
/// - `_collection`: collection name (routing)
/// - `_document`: serialized JSON document content
/// - `_version`: document version number
pub struct DocumentMetadataFilterBuilder;

impl DocumentMetadataFilterBuilder {
    /// Build a filter expression to find documents by collection and IDs.
    ///
    /// Creates a filter that matches:
    /// - `_type == "document"` AND
    /// - `_collection == collection_name` AND
    /// - `id IN [id1, id2, ...]` (storage IDs are prefixed: "{collection}::{doc_id}")
    ///
    /// # Arguments
    /// * `collection` - The collection name
    /// * `ids` - Document IDs to search for
    ///
    /// # Returns
    /// A FilterExpression that can be used with the storage engine
    pub fn build_document_retrieval_filter(collection: &str, ids: &[&str]) -> FilterExpression {
        // Build the storage-prefixed IDs that documents are stored with
        let storage_ids: Vec<serde_json::Value> = ids
            .iter()
            .map(|id| serde_json::Value::String(format!("{}::{}", collection, id)))
            .collect();

        // Create filter conditions
        let type_filter = FilterExpression::Comparison {
            field: "_type".to_string(),
            operator: ComparisonOperator::Equals,
            value: serde_json::Value::String("document".to_string()),
        };

        let collection_filter = FilterExpression::Comparison {
            field: "_collection".to_string(),
            operator: ComparisonOperator::Equals,
            value: serde_json::Value::String(collection.to_string()),
        };

        // For ID filtering, we use the vector record ID field (not metadata)
        // The storage engine stores documents with ID format: "{collection}::{doc_id}"
        let id_filter = FilterExpression::Comparison {
            field: "id".to_string(),
            operator: ComparisonOperator::In,
            value: serde_json::Value::Array(storage_ids),
        };

        // Combine all filters with AND
        FilterExpression::And(vec![type_filter, collection_filter, id_filter])
    }

    /// Build a filter expression to find all documents in a collection.
    ///
    /// Creates a filter that matches:
    /// - `_type == "document"` AND
    /// - `_collection == collection_name`
    ///
    /// # Arguments
    /// * `collection` - The collection name
    ///
    /// # Returns
    /// A FilterExpression that can be used with the storage engine
    pub fn build_collection_documents_filter(collection: &str) -> FilterExpression {
        let type_filter = FilterExpression::Comparison {
            field: "_type".to_string(),
            operator: ComparisonOperator::Equals,
            value: serde_json::Value::String("document".to_string()),
        };

        let collection_filter = FilterExpression::Comparison {
            field: "_collection".to_string(),
            operator: ComparisonOperator::Equals,
            value: serde_json::Value::String(collection.to_string()),
        };

        FilterExpression::And(vec![type_filter, collection_filter])
    }
}

// =============================================================================
// STRUCT: StorageEngineColdTierRetriever (Open/Closed Principle)
// =============================================================================

/// Implementation of ColdTierRetriever using the UnifiedStorageFormat.
///
/// This implementation uses the storage engine's search capability with
/// metadata filtering to retrieve documents from cold storage.
pub struct StorageEngineColdTierRetriever {
    storage_engine: Arc<dyn UnifiedStorageFormat>,
}

impl StorageEngineColdTierRetriever {
    /// Create a new retriever with the given storage engine.
    pub fn new(storage_engine: Arc<dyn UnifiedStorageFormat>) -> Self {
        Self { storage_engine }
    }

    /// Create a minimal Collection config for document retrieval.
    ///
    /// Documents don't require vector operations, so we create a minimal
    /// config with dimension 1 (for the placeholder [0.0] vector).
    fn create_document_collection_config(collection: &str) -> Arc<Collection> {
        use crate::proto::proximadb_v1::{CollectionConfig, StorageAssignment};

        Arc::new(Collection {
            id: format!("_documents_{}", collection),
            config: Some(CollectionConfig {
                name: format!("_documents_{}", collection),
                dimension: 1, // Documents use placeholder vectors
                ..Default::default()
            }),
            storage_assignment: Some(StorageAssignment {
                base_location: "./data".to_string(), // Default location
                ..Default::default()
            }),
            ..Default::default()
        })
    }

    /// Convert an OptimizedSearchRecord to a DocumentRecord.
    ///
    /// Extracts the document data from the metadata fields and reconstructs
    /// the DocumentRecord.
    fn search_record_to_document(
        &self,
        record: &crate::core::search::results::OptimizedSearchRecord,
    ) -> Option<DocumentRecord> {
        use proximadb_data_model::ProximaValue;

        // Check if this is a document record
        let type_value = record.metadata.get("_type")?;
        match type_value {
            ProximaValue::String(t) if t == "document" => {}
            _ => return None,
        }

        // Get collection
        let collection = record.metadata.get("_collection")?;
        let collection_name = match collection {
            ProximaValue::String(c) => c.clone(),
            _ => return None,
        };

        // Get document JSON
        let doc_value = record.metadata.get("_document")?;
        let doc_json = match doc_value {
            ProximaValue::String(j) => j.clone(),
            ProximaValue::Json(j) | ProximaValue::Jsonb(j) => j.to_string(),
            _ => return None,
        };

        // Deserialize document
        let document: crate::proto::proximadb_v1::SqlObject = match serde_json::from_str(&doc_json)
        {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!("Failed to deserialize document from storage: {}", e);
                return None;
            }
        };

        // Extract original ID (remove collection prefix)
        let original_id = record
            .id
            .strip_prefix(&format!("{}::", collection_name))
            .unwrap_or(&record.id)
            .to_string();

        // Get version
        let version = record
            .metadata
            .get("_version")
            .and_then(|v| match v {
                ProximaValue::Int64(i) => Some(*i as u64),
                _ => None,
            })
            .unwrap_or(0);

        let props = crate::storage::document::sql_object_to_proxima_tree(&document);
        Some(DocumentRecord {
            id: original_id,
            props,
            collection_id: collection_name,
            version,
            updated_at_ns: record.updated_at.unwrap_or(0) * 1_000_000, // Convert ms to ns
            schema_id: None,
            document_type: None,
        })
    }
}

#[async_trait::async_trait]
impl ColdTierRetriever for StorageEngineColdTierRetriever {
    async fn retrieve_documents(
        &self,
        collection: &str,
        ids: &[&str],
    ) -> Result<Vec<DocumentRecord>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        tracing::debug!(
            "Cold tier: Retrieving {} documents from collection '{}'",
            ids.len(),
            collection
        );

        // `SearchParams::top_k` is a `u16` and here it is the RESULT LIMIT: a
        // batch larger than `u16::MAX` ids used to wrap (`ids.len() as u16`:
        // 65,536 → 0) and silently drop documents. Look the ids up in batches
        // the `u16` can carry so every id is fetched (TD-SECSCAN-1 §5).
        let mut documents = Vec::new();
        for batch in ids.chunks(COLD_TIER_MAX_IDS_PER_LOOKUP) {
            let filter =
                DocumentMetadataFilterBuilder::build_document_retrieval_filter(collection, batch);
            let top_k = u16::try_from(batch.len()).map_err(|_| {
                anyhow::anyhow!("cold-tier batch of {} ids exceeds u16", batch.len())
            })?;

            // Create search parameters with the filter
            // We use a placeholder vector since documents don't have real vectors
            let search_params = Arc::new(SearchParams {
                vector: Some(vec![0.0]), // Placeholder vector for document retrieval
                top_k: Some(top_k),      // At most as many results as ids in this batch
                filter_expression: Some(filter),
                include_expired: Some(false),
                ..Default::default()
            });

            let collection_config = Self::create_document_collection_config(collection);
            let ctx = StorageQueryContext::new(search_params, collection_config);
            let results = self.storage_engine.search_vectors_unified(&ctx).await?;

            tracing::debug!(
                "Cold tier: Retrieved {} results from storage for {} requested IDs",
                results.len(),
                batch.len()
            );

            documents.extend(
                results
                    .iter()
                    .filter_map(|r| self.search_record_to_document(r)),
            );
        }

        Ok(documents)
    }

    async fn check_existence(&self, collection: &str, ids: &[&str]) -> Result<Vec<String>> {
        // For existence check, we retrieve documents and return their IDs
        let documents = self.retrieve_documents(collection, ids).await?;
        Ok(documents.into_iter().map(|d| d.id).collect())
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_filter_builder_single_id() {
        let filter = DocumentMetadataFilterBuilder::build_document_retrieval_filter(
            "test_collection",
            &["doc1"],
        );

        // Should be an AND of 3 conditions
        match filter {
            FilterExpression::And(conditions) => {
                assert_eq!(conditions.len(), 3, "Expected 3 filter conditions");
            }
            _ => panic!("Expected AND filter expression"),
        }
    }

    #[test]
    fn test_document_filter_builder_multiple_ids() {
        let filter = DocumentMetadataFilterBuilder::build_document_retrieval_filter(
            "test_collection",
            &["doc1", "doc2", "doc3"],
        );

        // Verify the filter structure
        match filter {
            FilterExpression::And(conditions) => {
                assert_eq!(conditions.len(), 3, "Expected 3 filter conditions");

                // Find the ID filter
                let id_filter = conditions.iter().find(
                    |c| matches!(c, FilterExpression::Comparison { field, .. } if field == "id"),
                );

                assert!(id_filter.is_some(), "Expected ID filter");

                if let Some(FilterExpression::Comparison {
                    operator, value, ..
                }) = id_filter
                {
                    assert!(
                        matches!(operator, ComparisonOperator::In),
                        "Expected In operator"
                    );

                    if let serde_json::Value::Array(ids) = value {
                        assert_eq!(ids.len(), 3, "Expected 3 IDs in the filter");
                    } else {
                        panic!("Expected array value for In operator");
                    }
                }
            }
            _ => panic!("Expected AND filter expression"),
        }
    }

    #[test]
    fn test_collection_documents_filter() {
        let filter =
            DocumentMetadataFilterBuilder::build_collection_documents_filter("test_collection");

        // Should be an AND of 2 conditions
        match filter {
            FilterExpression::And(conditions) => {
                assert_eq!(conditions.len(), 2, "Expected 2 filter conditions");
            }
            _ => panic!("Expected AND filter expression"),
        }
    }

    // ---------------------------------------------------------------------
    // TD-SECSCAN-1 §5 residual: `SearchParams::top_k` is a `u16`. The id-batch
    // lookup used `ids.len() as u16`, so a batch of 65,536 ids asked the engine
    // for 0 results (70,000 → 4,464) and silently DROPPED documents. Every id
    // must be looked up: the batch is split so each engine call's `top_k`
    // covers exactly the ids it filters on.
    // ---------------------------------------------------------------------

    use crate::storage::traits::{
        CompactionParameters, CompactionResult, FlushParameters, FlushResult, StorageFormatStrategy,
    };
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// Records the `top_k` of every `search_vectors_unified` call.
    #[derive(Default)]
    struct TopKRecordingEngine {
        top_ks: Mutex<Vec<Option<u16>>>,
    }

    #[async_trait::async_trait]
    impl UnifiedStorageFormat for TopKRecordingEngine {
        fn engine_name(&self) -> &'static str {
            "TopKRecordingEngine"
        }
        fn engine_version(&self) -> &'static str {
            "1.0.0"
        }
        fn strategy(&self) -> StorageFormatStrategy {
            StorageFormatStrategy::Sst
        }
        async fn do_flush(&self, _params: &FlushParameters) -> Result<FlushResult> {
            anyhow::bail!("not used")
        }
        async fn do_compact(&self, _params: &CompactionParameters) -> Result<CompactionResult> {
            anyhow::bail!("not used")
        }
        async fn collect_engine_metrics(&self) -> Result<HashMap<String, serde_json::Value>> {
            Ok(HashMap::new())
        }
        async fn vector_by_id(
            &self,
            _collection_id: &str,
            _base_path: &str,
            _vector_id: &str,
        ) -> Result<Option<proximadb_records::ProximaRecord>> {
            Ok(None)
        }
        async fn search_vectors_unified(
            &self,
            ctx: &StorageQueryContext,
        ) -> Result<Vec<crate::core::search::results::OptimizedSearchRecord>> {
            self.top_ks
                .lock()
                .map_err(|_| anyhow::anyhow!("poisoned"))?
                .push(ctx.search_params.top_k);
            Ok(Vec::new())
        }
    }

    async fn requested_top_ks(n: usize) -> Vec<Option<u16>> {
        let engine = Arc::new(TopKRecordingEngine::default());
        let retriever = StorageEngineColdTierRetriever::new(engine.clone());
        let owned: Vec<String> = (0..n).map(|i| format!("doc-{i}")).collect();
        let ids: Vec<&str> = owned.iter().map(String::as_str).collect();
        retriever
            .retrieve_documents("docs", &ids)
            .await
            .expect("retrieve");
        engine.top_ks.lock().map(|v| v.clone()).unwrap_or_default()
    }

    #[tokio::test]
    async fn every_id_is_looked_up_across_the_u16_boundary() {
        for n in [65_535usize, 65_536, 70_000] {
            let top_ks = requested_top_ks(n).await;
            let covered: usize = top_ks.iter().map(|k| k.map_or(0, usize::from)).sum();
            assert_eq!(
                covered, n,
                "{n} ids: engine calls asked for {top_ks:?} results in total — ids were dropped"
            );
            assert!(
                top_ks.iter().all(|k| k.is_some_and(|k| k > 0)),
                "{n} ids: no lookup may ask for zero results: {top_ks:?}"
            );
        }
    }
}
