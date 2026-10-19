//! BM25 full-text indexing port for `proximadb-runtime`.
//!
//! `BM25IndexPort` is the stable contract that the REST `hybrid_index` handler
//! uses to add documents to a per-collection BM25 index without importing
//! root-crate concrete types.

use anyhow::Result;
use async_trait::async_trait;

/// A single document to be indexed for full-text (BM25) search.
#[derive(Debug, Clone)]
pub struct BM25Document {
    pub id: String,
    pub text: String,
}

/// Result of a BM25 index operation.
#[derive(Debug, Clone)]
pub struct BM25IndexResult {
    pub collection: String,
    pub documents_indexed: usize,
    pub total_documents: usize,
}

/// Port for BM25 full-text indexing operations.
///
/// Implemented by the root-crate `Bm25IndexPortImpl`, which wraps
/// the in-memory `HybridFullTextIndexMap`.  When absent the REST
/// handler returns `501 Not Implemented`.
#[async_trait]
pub trait BM25IndexPort: Send + Sync {
    async fn index_documents(
        &self,
        collection: String,
        documents: Vec<BM25Document>,
    ) -> Result<BM25IndexResult>;

    /// Index for a request acting as `request_tenant`: the collection is
    /// resolved within that tenant first, and one it may not resolve is
    /// answered exactly like a missing one. Default: no scoping.
    async fn index_documents_for_tenant(
        &self,
        collection: String,
        documents: Vec<BM25Document>,
        _request_tenant: &str,
    ) -> Result<BM25IndexResult> {
        self.index_documents(collection, documents).await
    }
}
