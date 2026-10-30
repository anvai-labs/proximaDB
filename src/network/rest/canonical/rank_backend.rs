//! Production `HybridSearchBackend` impl that fans out to the existing
//! per-collection BM25 index and the shared `VectorOpsPort`.
//!
//! Commit 2 of 5 in the R-7c production `RankServices` wiring. The trait
//! it implements lives in `rest::canonical::rank` and is consumed by
//! `HybridCoordinatorAdapter`. By implementing it over the two surfaces
//! we already construct in `SharedServices` (the shared
//! `HybridFullTextIndexMap` and the runtime `VectorOpsPort`), we avoid
//! introducing any new storage infrastructure for ranking — the
//! adapter just routes through the same primitives REST hybrid search
//! and the gRPC hybrid path already use.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use proximadb_rank_core::{RankError, RankResult};
use proximadb_search_types::search_params::LexicalQueryMode;

use crate::core::search::hybrid::{BM25Result, TextHighlight, VectorResult};
use crate::network::hybrid_search::HybridFullTextIndexMap;
use crate::network::rest::canonical::rank::HybridSearchBackend;
use crate::proto::proximadb_v1::{SearchQuery, VectorSearchRequest};

/// Recall pool size used when the underlying coordinator does not pass a
/// per-side cap. The fused result is trimmed by `HybridCoordinator::with_top_k`
/// and then further by the rank profile's first-phase heap.
const DEFAULT_RECALL_POOL: usize = 1_000;

/// Production-grade `HybridSearchBackend`.
///
/// Both sides are best-effort: a missing per-collection BM25 index or an
/// empty `query` string returns no BM25 candidates, and an empty query
/// vector returns no vector candidates. Surface-level errors bubble out as
/// `RankError::ModelInference` so the rank pipeline reports them as
/// retrieval-side failures rather than profile-side failures.
pub struct ProductionHybridBackend {
    vector_ops: Arc<dyn proximadb_runtime::VectorOpsPort>,
    fulltext_indexes: HybridFullTextIndexMap,
    recall_pool: usize,
    /// Resolves the target collection within the request tenant
    /// (`scope_collection`). `None` (tests) skips the check.
    collection_port: Option<Arc<dyn proximadb_runtime::CollectionPort>>,
}

impl ProductionHybridBackend {
    /// Build a backend around the runtime `VectorOpsPort` and the in-process
    /// `HybridFullTextIndexMap` already maintained by the BM25 ingestion path.
    pub fn new(
        vector_ops: Arc<dyn proximadb_runtime::VectorOpsPort>,
        fulltext_indexes: HybridFullTextIndexMap,
    ) -> Self {
        Self {
            vector_ops,
            fulltext_indexes,
            recall_pool: DEFAULT_RECALL_POOL,
            collection_port: None,
        }
    }

    /// Resolve the target collection within the request tenant before either
    /// leg runs (`HybridSearchBackend::scope_collection`).
    pub fn with_collection_port(
        mut self,
        collection_port: Arc<dyn proximadb_runtime::CollectionPort>,
    ) -> Self {
        self.collection_port = Some(collection_port);
        self
    }

    /// Override the recall pool size used per side. Useful for tests that
    /// want to bound result counts deterministically.
    pub fn with_recall_pool(mut self, recall_pool: usize) -> Self {
        self.recall_pool = recall_pool.max(1);
        self
    }
}

#[async_trait]
impl HybridSearchBackend for ProductionHybridBackend {
    async fn scope_collection(&self, collection: &str, tenant: &str) -> RankResult<()> {
        // Without a collection port nothing can be resolved within a tenant.
        let Some(collection_port) = &self.collection_port else {
            return Err(RankError::CollectionNotFound(collection.to_string()));
        };
        match collection_port
            .get_collection(collection, Some(tenant))
            .await
        {
            Ok(Some(_)) => Ok(()),
            Ok(None) => Err(RankError::CollectionNotFound(collection.to_string())),
            Err(err) => Err(RankError::ModelInference {
                model_id: "production_hybrid_backend.collection".to_string(),
                reason: format!("collection lookup failed for '{collection}': {err}"),
            }),
        }
    }

    async fn bm25_search(&self, collection: &str, query: &str) -> RankResult<Vec<BM25Result>> {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            return Ok(Vec::new());
        }
        let indexes = self
            .fulltext_indexes
            .read()
            .map_err(|err| RankError::ModelInference {
                model_id: "production_hybrid_backend.bm25".to_string(),
                reason: format!("bm25 index map poisoned for '{collection}': {err}"),
            })?;
        let Some(index) = indexes.get(collection) else {
            return Ok(Vec::new());
        };
        let hits = index
            .search_with_mode(trimmed, self.recall_pool, LexicalQueryMode::NaturalLanguage)
            .map_err(|err| RankError::ModelInference {
                model_id: "production_hybrid_backend.bm25".to_string(),
                reason: format!("bm25 search failed for '{collection}': {err}"),
            })?;
        Ok(hits
            .into_iter()
            .map(|hit| BM25Result {
                doc_id: hit.doc_id,
                score: hit.score,
                highlights: Some(
                    hit.matched_terms
                        .into_iter()
                        .map(|term| TextHighlight {
                            field: "content".to_string(),
                            start_offset: 0,
                            end_offset: term.len(),
                            text: term,
                        })
                        .collect(),
                ),
                metadata: HashMap::new(),
            })
            .collect())
    }

    async fn vector_search(
        &self,
        collection: &str,
        vector: &[f32],
    ) -> RankResult<Vec<VectorResult>> {
        if vector.is_empty() {
            return Ok(Vec::new());
        }
        let request = VectorSearchRequest {
            collection_id: collection.to_string(),
            queries: vec![SearchQuery {
                vector: vector.to_vec(),
                filters: HashMap::new(),
                advanced_filter: None,
            }],
            top_k: self.recall_pool as u32,
            include_fields: None,
            search_params: None,
            distance_metric_override: None,
            search_optimization: None,
        };

        let response = self
            .vector_ops
            .search(request, proximadb_runtime::PortIdentity::anonymous())
            .await
            .map_err(|err| {
                // A wrong-dimension / empty / non-finite query vector or
                // over-cap top_k is the client's argument (TD-SECSCAN-1 §5).
                match crate::services::operations::vectors::client_argument_message(&err) {
                    Some(message) => RankError::InvalidArgument(message),
                    None => RankError::ModelInference {
                        model_id: "production_hybrid_backend.vector".to_string(),
                        reason: format!("vector search failed for '{collection}': {err}"),
                    },
                }
            })?;

        let results = response
            .results
            .map(|r| r.results)
            .unwrap_or_default()
            .into_iter()
            .map(|rec| VectorResult {
                doc_id: rec.id,
                score: rec.score,
                distance: (1.0 - rec.score).max(0.0),
                metadata: HashMap::new(),
            })
            .collect();
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::hybrid_search::HybridFullTextIndexMap;
    use crate::proto::proximadb_v1::{SearchResult, SearchVectorRecord, VectorOperationResponse};
    use crate::storage::engines::core::formats::columnar::fulltext_index::{
        FullTextIndex, TokenizerConfig,
    };
    use std::sync::{Mutex, RwLock};

    /// Capture the last search request the backend issued and return canned
    /// records so we can assert on doc_id round-trip + score mapping.
    struct CapturingVectorPort {
        last_request: Mutex<Option<VectorSearchRequest>>,
        results: Mutex<Vec<SearchVectorRecord>>,
        fail_with: Mutex<Option<String>>,
        /// Fail with a typed client-argument error instead of a plain one.
        fail_typed: bool,
    }

    impl CapturingVectorPort {
        fn with_results(results: Vec<SearchVectorRecord>) -> Self {
            Self {
                last_request: Mutex::new(None),
                results: Mutex::new(results),
                fail_with: Mutex::new(None),
                fail_typed: false,
            }
        }

        fn failing(message: impl Into<String>) -> Self {
            Self {
                last_request: Mutex::new(None),
                results: Mutex::new(Vec::new()),
                fail_with: Mutex::new(Some(message.into())),
                fail_typed: false,
            }
        }
    }

    #[async_trait]
    impl proximadb_runtime::VectorOpsPort for CapturingVectorPort {
        async fn search(
            &self,
            request: VectorSearchRequest,
            _identity: proximadb_runtime::PortIdentity<'_>,
        ) -> anyhow::Result<VectorOperationResponse> {
            *self.last_request.lock().unwrap() = Some(request.clone());
            if let Some(reason) = self.fail_with.lock().unwrap().clone() {
                if self.fail_typed {
                    return Err(
                        crate::services::operations::vectors::InvalidQueryVector::other(reason)
                            .into(),
                    );
                }
                return Err(anyhow::anyhow!(reason));
            }
            let records = self.results.lock().unwrap().clone();
            Ok(VectorOperationResponse {
                success: true,
                operation: 0,
                metrics: None,
                results: Some(SearchResult {
                    results: records,
                    total_found: 0,
                    collection_id: Some(request.collection_id.clone()),
                }),
                vector_ids: Vec::new(),
                error_message: None,
                error_code: None,
            })
        }

        async fn batch_upsert(
            &self,
            _request: crate::proto::proximadb_v1::VectorBatchRequest,
            _tenant_id: Option<&str>,
        ) -> anyhow::Result<VectorOperationResponse> {
            unimplemented!("CapturingVectorPort only supports search")
        }

        async fn get_vector(
            &self,
            _collection_id: &str,
            _vector_id: &str,
            _include_vector: bool,
            _include_metadata: bool,
            _tenant_id: Option<&str>,
        ) -> anyhow::Result<VectorOperationResponse> {
            unimplemented!("CapturingVectorPort only supports search")
        }

        async fn flush_all(&self) -> anyhow::Result<()> {
            Ok(())
        }

        async fn metrics(&self) -> anyhow::Result<serde_json::Value> {
            Ok(serde_json::Value::Null)
        }
    }

    fn empty_indexes() -> HybridFullTextIndexMap {
        Arc::new(RwLock::new(HashMap::new()))
    }

    fn indexes_with_documents(collection: &str, docs: Vec<(&str, &str)>) -> HybridFullTextIndexMap {
        let mut map = HashMap::new();
        let mut index = FullTextIndex::new(TokenizerConfig::for_keyword_search());
        for (id, text) in docs {
            index.add_document(id, text).expect("index doc");
        }
        map.insert(collection.to_string(), index);
        Arc::new(RwLock::new(map))
    }

    fn vec_record(id: &str, score: f64) -> SearchVectorRecord {
        SearchVectorRecord {
            id: id.to_string(),
            score,
            vector: Vec::new(),
            metadata: HashMap::new(),
            version: None,
            similarity: None,
            timestamp: None,
            source: None,
            expanded_context: Vec::new(),
            semantic_similarity: None,
            quantization_info: None,
            engine_stats: HashMap::new(),
            index_path: None,
        }
    }

    #[tokio::test]
    async fn bm25_search_returns_empty_for_blank_query() {
        let backend = ProductionHybridBackend::new(
            Arc::new(CapturingVectorPort::with_results(Vec::new())),
            empty_indexes(),
        );
        let out = backend.bm25_search("docs", "   ").await.unwrap();
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn bm25_search_returns_empty_when_collection_has_no_index() {
        let backend = ProductionHybridBackend::new(
            Arc::new(CapturingVectorPort::with_results(Vec::new())),
            empty_indexes(),
        );
        let out = backend.bm25_search("docs", "search me").await.unwrap();
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn bm25_search_hits_per_collection_index() {
        let indexes = indexes_with_documents(
            "docs",
            vec![
                ("d1", "the quick brown fox"),
                ("d2", "lazy dog sleeps"),
                ("d3", "another fox jumps"),
            ],
        );
        let backend = ProductionHybridBackend::new(
            Arc::new(CapturingVectorPort::with_results(Vec::new())),
            indexes,
        );
        let out = backend.bm25_search("docs", "fox").await.unwrap();
        let ids: Vec<_> = out.iter().map(|r| r.doc_id.as_str()).collect();
        assert!(ids.contains(&"d1"));
        assert!(ids.contains(&"d3"));
        assert!(!ids.contains(&"d2"));
    }

    #[tokio::test]
    async fn vector_search_returns_empty_for_empty_vector() {
        let port = Arc::new(CapturingVectorPort::with_results(vec![vec_record(
            "v1", 0.9,
        )]));
        let backend = ProductionHybridBackend::new(port.clone(), empty_indexes());
        let out = backend.vector_search("docs", &[]).await.unwrap();
        assert!(out.is_empty());
        assert!(
            port.last_request.lock().unwrap().is_none(),
            "should not call vector port when query is empty"
        );
    }

    #[tokio::test]
    async fn vector_search_forwards_query_and_maps_results() {
        let port = Arc::new(CapturingVectorPort::with_results(vec![
            vec_record("v1", 0.9),
            vec_record("v2", 0.4),
        ]));
        let backend = ProductionHybridBackend::new(port.clone(), empty_indexes());
        let out = backend
            .vector_search("docs", &[0.1, 0.2, 0.3])
            .await
            .unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].doc_id, "v1");
        assert!((out[0].score - 0.9).abs() < 1e-6);
        assert!((out[0].distance - (1.0 - 0.9)).abs() < 1e-6);

        let captured = port.last_request.lock().unwrap().clone().unwrap();
        assert_eq!(captured.collection_id, "docs");
        assert_eq!(captured.queries.len(), 1);
        assert_eq!(captured.queries[0].vector, vec![0.1, 0.2, 0.3]);
    }

    #[tokio::test]
    async fn vector_search_propagates_port_errors_as_rank_inference_error() {
        let backend = ProductionHybridBackend::new(
            Arc::new(CapturingVectorPort::failing("synthetic upstream failure")),
            empty_indexes(),
        );
        let err = backend
            .vector_search("docs", &[0.1, 0.2])
            .await
            .expect_err("upstream failure must surface");
        match err {
            RankError::ModelInference { model_id, reason } => {
                assert_eq!(model_id, "production_hybrid_backend.vector");
                assert!(reason.contains("synthetic upstream failure"));
            }
            other => panic!("expected ModelInference, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn with_recall_pool_caps_request_top_k() {
        let port = Arc::new(CapturingVectorPort::with_results(Vec::new()));
        let backend =
            ProductionHybridBackend::new(port.clone(), empty_indexes()).with_recall_pool(7);
        backend.vector_search("docs", &[0.1, 0.2]).await.unwrap();
        let captured = port.last_request.lock().unwrap().clone().unwrap();
        assert_eq!(captured.top_k, 7);
    }

    /// TD-SECSCAN-1 §5: the vector leg's typed client-argument error becomes
    /// `RankError::InvalidArgument` (HTTP 400 at the rank route) instead of
    /// being flattened into a `ModelInference` failure (HTTP 500); a plain
    /// upstream failure stays `ModelInference`.
    #[tokio::test]
    async fn vector_search_keeps_client_argument_errors_distinct() {
        let mut port = CapturingVectorPort::failing(
            "Query vector has dimension 3 but collection 'docs' expects dimension 4",
        );
        port.fail_typed = true;
        let backend = ProductionHybridBackend::new(Arc::new(port), empty_indexes());
        match backend.vector_search("docs", &[1.0, 2.0, 3.0]).await {
            Err(RankError::InvalidArgument(message)) => {
                assert!(message.contains("dimension 3"), "{message}")
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
        let backend = ProductionHybridBackend::new(
            Arc::new(CapturingVectorPort::failing("synthetic upstream failure")),
            empty_indexes(),
        );
        assert!(matches!(
            backend.vector_search("docs", &[1.0]).await,
            Err(RankError::ModelInference { .. })
        ));
    }
}
