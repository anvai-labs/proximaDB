//! Hybrid search coordinator
//!
//! Orchestrates parallel BM25 and vector search execution.

use super::{BM25Result, FusedSearchResult, FusionStrategy, HybridFusionEngine, VectorResult};

/// Hybrid search coordinator
///
/// Executes BM25 and vector searches in parallel, then fuses results.
pub struct HybridCoordinator {
    fusion_engine: HybridFusionEngine,
}

impl HybridCoordinator {
    /// Create a new coordinator
    pub fn new(fusion_strategy: FusionStrategy) -> Self {
        Self {
            fusion_engine: HybridFusionEngine::new(fusion_strategy),
        }
    }

    /// Create a new coordinator with custom top_k
    pub fn with_top_k(fusion_strategy: FusionStrategy, top_k: usize) -> Self {
        Self {
            fusion_engine: HybridFusionEngine::new(fusion_strategy).with_top_k(top_k),
        }
    }

    /// Execute hybrid search with parallel execution
    ///
    /// # Arguments
    /// * `bm25_search_fn` - Function to execute BM25 search
    /// * `vector_search_fn` - Function to execute vector search
    /// * `query` - Search query string
    /// * `vector` - Query vector
    ///
    /// # Returns
    /// Fused and sorted results
    ///
    /// # Example
    /// ```ignore
    /// use proximadb::core::search::hybrid::{HybridCoordinator, FusionStrategy};
    ///
    /// let coordinator = HybridCoordinator::new(FusionStrategy::ReciprocalRank { k: 60 });
    /// let results = coordinator.execute_hybrid_search(
    ///     |q| bm25_search(q),
    ///     |v| vector_search(v),
    ///     "search query",
    ///     vec![0.1, 0.2, 0.3],
    /// ).await?;
    /// ```
    pub async fn execute_hybrid_search<F1, F2, Fut1, Fut2>(
        &self,
        bm25_search_fn: F1,
        vector_search_fn: F2,
        query: &str,
        vector: &[f32],
    ) -> anyhow::Result<Vec<FusedSearchResult>>
    where
        F1: FnOnce(String) -> Fut1,
        F2: FnOnce(Vec<f32>) -> Fut2,
        Fut1: std::future::Future<Output = anyhow::Result<Vec<BM25Result>>>,
        Fut2: std::future::Future<Output = anyhow::Result<Vec<VectorResult>>>,
    {
        let query = query.to_string();
        let vector = vector.to_vec();

        // Execute searches in parallel
        let (bm25_result, vector_result) =
            tokio::join!(async move { bm25_search_fn(query).await }, async move {
                vector_search_fn(vector).await
            });

        let bm25_results = bm25_result?;
        let vector_results = vector_result?;

        // Fuse results using fusion engine
        self.fusion_engine
            .fuse(bm25_results, vector_results)
            .map_err(|e| anyhow::anyhow!("Fusion error: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_coordinator_creation() {
        let _coordinator = HybridCoordinator::new(FusionStrategy::ReciprocalRank { k: 60 });
        // Coordinator created successfully
    }

    #[tokio::test]
    async fn test_coordinator_with_custom_top_k() {
        let _coordinator =
            HybridCoordinator::with_top_k(FusionStrategy::ReciprocalRank { k: 60 }, 20);
        // Coordinator created successfully with custom top_k
    }

    #[tokio::test]
    async fn test_hybrid_search_execution() {
        let coordinator = HybridCoordinator::new(FusionStrategy::ReciprocalRank { k: 60 });

        // Mock BM25 search function
        let bm25_search = |_query: String| async move {
            Ok::<Vec<BM25Result>, anyhow::Error>(vec![
                BM25Result {
                    doc_id: "doc1".to_string(),
                    score: 2.5,
                    highlights: None,
                    metadata: HashMap::new(),
                },
                BM25Result {
                    doc_id: "doc2".to_string(),
                    score: 1.8,
                    highlights: None,
                    metadata: HashMap::new(),
                },
            ])
        };

        // Mock vector search function
        let vector_search = |_vector: Vec<f32>| async move {
            Ok::<Vec<VectorResult>, anyhow::Error>(vec![
                VectorResult {
                    doc_id: "doc1".to_string(),
                    score: 0.95,
                    distance: 0.15,
                    metadata: HashMap::new(),
                },
                VectorResult {
                    doc_id: "doc3".to_string(),
                    score: 0.88,
                    distance: 0.22,
                    metadata: HashMap::new(),
                },
            ])
        };

        let results = coordinator
            .execute_hybrid_search(bm25_search, vector_search, "test query", &[0.1, 0.2])
            .await
            .unwrap();

        // Should have 3 unique documents
        assert_eq!(results.len(), 3);

        // doc1 should appear in both BM25 and vector results
        let doc1_result = results.iter().find(|r| r.doc_id == "doc1").unwrap();
        assert_eq!(doc1_result.bm25_score, 2.5);
        assert_eq!(doc1_result.vector_score, 0.95);
    }

    #[tokio::test]
    async fn test_hybrid_search_parallel_execution() {
        let coordinator = HybridCoordinator::new(FusionStrategy::ReciprocalRank { k: 60 });

        let bm25_called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let vector_called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

        let bm25_called_clone = bm25_called.clone();
        let vector_called_clone = vector_called.clone();

        let bm25_search = move |_query: String| async move {
            bm25_called_clone.store(true, std::sync::atomic::Ordering::SeqCst);
            Ok::<Vec<BM25Result>, anyhow::Error>(vec![])
        };

        let vector_search = move |_vector: Vec<f32>| async move {
            vector_called_clone.store(true, std::sync::atomic::Ordering::SeqCst);
            Ok::<Vec<VectorResult>, anyhow::Error>(vec![])
        };

        let _ = coordinator
            .execute_hybrid_search(bm25_search, vector_search, "test", &[0.1])
            .await;

        // Both should have been called (parallel execution)
        assert!(bm25_called.load(std::sync::atomic::Ordering::SeqCst));
        assert!(vector_called.load(std::sync::atomic::Ordering::SeqCst));
    }
}
