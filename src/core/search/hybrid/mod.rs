//! Hybrid Search - BM25 + Vector Similarity Fusion
//!
//! This module implements fusion strategies for combining full-text
//! BM25 search with vector similarity search.
//!
//! # Architecture
//!
//! ## Fusion Strategies
//!
//! ### Reciprocal Rank Fusion (RRF)
//! - Formula: `score = 1/(k + rank_bm25) + 1/(k + rank_vector)`
//! - Robust to score scale differences
//! - Recommended k: 60 (default)
//!
//! ### Weighted Linear Fusion
//! - Formula: `score = alpha * bm25 + (1-alpha) * vector`
//! - Requires score normalization
//! - alpha=0.5 gives equal weight
//!
//! ### Rank Biased Precision (RBP)
//! - Formula: `score = (1-p) * p^(rank-1)`
//! - Emphasizes top ranks
//! - Higher persistence = more emphasis on early ranks
//!
//! # Usage
//!
//! ```ignore
//! use proximadb::core::search::hybrid::{
//!     FusionStrategy, HybridFusionEngine, BM25Result, VectorResult,
//! };
//!
//! let bm25_results = vec![/* ... */];
//! let vector_results = vec![/* ... */];
//!
//! let engine = HybridFusionEngine::new(FusionStrategy::ReciprocalRank { k: 60 });
//! let fused = engine.fuse(bm25_results, vector_results)?;
//! ```

pub mod builder; // Filtered hybrid query builder (Issue #39, SB-09)
pub mod coordinator;
pub mod fusion;
pub mod reranker;
/// Utility-aware scorer (LLD 8) - linear blend + pluggable UAE artifact path.
pub use proximadb_utility_scorer::utility_scorer;

// Export fusion engine, error, and coordinator
pub use builder::{HybridExecutionStrategy, HybridQuery, HybridQueryBuilder, HybridQueryResult};
pub use coordinator::HybridCoordinator;
pub use fusion::{FusionError, HybridFusionEngine};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Fusion strategies for combining BM25 and vector scores
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FusionStrategy {
    /// Reciprocal Rank Fusion
    ///
    /// Formula: `score = 1/(k + rank_bm25) + 1/(k + rank_vector)`
    ///
    /// # Arguments
    /// * `k` - Ranking constant (default: 60)
    ///
    /// # Example
    /// ```ignore
    /// use proximadb::core::search::hybrid::FusionStrategy;
    ///
    /// let strategy = FusionStrategy::ReciprocalRank { k: 60 };
    /// ```
    ReciprocalRank {
        /// RRF parameter k (higher = more uniform weighting, default 60)
        k: usize,
    },

    /// Weighted linear combination
    ///
    /// Formula: `score = alpha * bm25 + (1-alpha) * vector`
    ///
    /// # Arguments
    /// * `alpha` - Weight for BM25 (0.0 to 1.0)
    /// * `bm25_normalize` - Whether to normalize BM25 scores to [0,1]
    /// * `vector_normalize` - Whether to normalize vector scores to [0,1]
    ///
    /// # Example
    /// ```ignore
    /// // Equal weight: 50% BM25, 50% vector
    /// let strategy = FusionStrategy::WeightedLinear {
    ///     alpha: 0.5,
    ///     bm25_normalize: true,
    ///     vector_normalize: true,
    /// };
    /// ```
    ///
    /// # Deprecation (T4.2)
    ///
    /// **This strategy is deprecated.** Use [`ReciprocalRank`](Self::ReciprocalRank) for most
    /// use cases, or migrate to [`crate::core::search::cross_modal_fusion::Fuser`] for the
    /// modern cross-modal fusion seam that supports vector, graph, document, and relational sources.
    ///
    /// Migration path:
    /// - Weighted linear fusion → `ReciprocalRank { k: 60 }` (similar weighted blending)
    /// - Multi-modal fusion → `Fuser::new(FusionPolicy::default())` with vector/graph/doc sources
    #[deprecated(
        since = "0.2.0",
        note = "Use FusionStrategy::ReciprocalRank or crate::core::search::cross_modal_fusion::Fuser instead."
    )]
    WeightedLinear {
        /// Weight for BM25 score (0.0 to 1.0, vector gets 1-alpha)
        alpha: f64,
        /// Whether to normalize BM25 scores to [0,1]
        bm25_normalize: bool,
        /// Whether to normalize vector scores to [0,1]
        vector_normalize: bool,
    },

    /// Rank Biased Precision (RBP)
    ///
    /// Formula: `score = (1-p) * p^(rank-1)`
    ///
    /// # Arguments
    /// * `persistence` - Persistence parameter (0.0 to 1.0)
    ///   - Higher values emphasize top ranks more
    ///   - Typical values: 0.8 to 0.99
    ///
    /// # Example
    /// ```ignore
    /// // Strong emphasis on top ranks
    /// let strategy = FusionStrategy::RankBiasedPrecision {
    ///     persistence: 0.95,
    /// };
    /// ```
    ///
    /// # Deprecation (T4.2)
    ///
    /// **This strategy is deprecated.** Use [`ReciprocalRank`](Self::ReciprocalRank) or migrate to
    /// [`crate::core::search::cross_modal_fusion::Fuser`] for the modern cross-modal fusion seam.
    #[deprecated(
        since = "0.2.0",
        note = "Use FusionStrategy::ReciprocalRank or crate::core::search::cross_modal_fusion::Fuser instead."
    )]
    RankBiasedPrecision {
        /// Persistence parameter p controlling rank emphasis (0.8 to 0.99)
        persistence: f64,
    },

    /// Conditional Normalization
    ///
    /// Normalizes both scores to [0,1] and averages them
    ///
    /// # Deprecation (T4.2)
    ///
    /// **This strategy is deprecated.** Use [`ReciprocalRank`](Self::ReciprocalRank) or migrate to
    /// [`crate::core::search::cross_modal_fusion::Fuser`] for the modern cross-modal fusion seam.
    #[deprecated(
        since = "0.2.0",
        note = "Use FusionStrategy::ReciprocalRank or crate::core::search::cross_modal_fusion::Fuser instead."
    )]
    ConditionalNormalization,

    /// Borda Count
    ///
    /// Rank-based voting method where each document gets points based on its rank
    /// Formula: `score = (N - rank_bm25) + (N - rank_vector)`
    /// where N is the total number of documents in both result sets
    ///
    /// # Example
    /// ```ignore
    /// let strategy = FusionStrategy::BordaCount;
    /// ```
    ///
    /// # Deprecation (T4.2)
    ///
    /// **This strategy is deprecated.** Use [`ReciprocalRank`](Self::ReciprocalRank) or migrate to
    /// [`crate::core::search::cross_modal_fusion::Fuser`] for the modern cross-modal fusion seam.
    #[deprecated(
        since = "0.2.0",
        note = "Use FusionStrategy::ReciprocalRank or crate::core::search::cross_modal_fusion::Fuser instead."
    )]
    BordaCount,

    /// CombSUM
    ///
    /// Simple summation of normalized scores
    /// Formula: `score = bm25_normalized + vector_normalized`
    ///
    /// # Example
    /// ```ignore
    /// let strategy = FusionStrategy::CombSum;
    /// ```
    ///
    /// # Deprecation (T4.2)
    ///
    /// **This strategy is deprecated.** Use [`ReciprocalRank`](Self::ReciprocalRank) or migrate to
    /// [`crate::core::search::cross_modal_fusion::Fuser`] for the modern cross-modal fusion seam.
    #[deprecated(
        since = "0.2.0",
        note = "Use FusionStrategy::ReciprocalRank or crate::core::search::cross_modal_fusion::Fuser instead."
    )]
    CombSum,

    /// CombMIN
    ///
    /// Minimum score selection (pessimistic)
    /// Formula: `score = min(bm25_normalized, vector_normalized)`
    ///
    /// # Example
    /// ```ignore
    /// let strategy = FusionStrategy::CombMin;
    /// ```
    ///
    /// # Deprecation (T4.2)
    ///
    /// **This strategy is deprecated.** Use [`ReciprocalRank`](Self::ReciprocalRank) or migrate to
    /// [`crate::core::search::cross_modal_fusion::Fuser`] for the modern cross-modal fusion seam.
    #[deprecated(
        since = "0.2.0",
        note = "Use FusionStrategy::ReciprocalRank or crate::core::search::cross_modal_fusion::Fuser instead."
    )]
    CombMin,

    /// CombMAX
    ///
    /// Maximum score selection (optimistic)
    /// Formula: `score = max(bm25_normalized, vector_normalized)`
    ///
    /// # Example
    /// ```ignore
    /// let strategy = FusionStrategy::CombMax;
    /// ```
    ///
    /// # Deprecation (T4.2)
    ///
    /// **This strategy is deprecated.** Use [`ReciprocalRank`](Self::ReciprocalRank) or migrate to
    /// [`crate::core::search::cross_modal_fusion::Fuser`] for the modern cross-modal fusion seam.
    #[deprecated(
        since = "0.2.0",
        note = "Use FusionStrategy::ReciprocalRank or crate::core::search::cross_modal_fusion::Fuser instead."
    )]
    CombMax,

    /// Condorcet Fusion
    ///
    /// Pairwise comparison method - document wins if it outranks the other in both lists
    /// Formula: Binary wins, losses summed across all comparisons
    ///
    /// # Example
    /// ```ignore
    /// let strategy = FusionStrategy::Condorcet;
    /// ```
    ///
    /// # Deprecation (T4.2)
    ///
    /// **This strategy is deprecated.** Use [`ReciprocalRank`](Self::ReciprocalRank) or migrate to
    /// [`crate::core::search::cross_modal_fusion::Fuser`] for the modern cross-modal fusion seam.
    #[deprecated(
        since = "0.2.0",
        note = "Use FusionStrategy::ReciprocalRank or crate::core::search::cross_modal_fusion::Fuser instead."
    )]
    Condorcet,

    /// Dempster-Shafer
    ///
    /// Evidence theory combination - treats scores as belief functions
    /// Formula: Combines evidence using Dempster's rule of combination
    ///
    /// # Arguments
    /// * `alpha` - Weighting parameter (0.0 to 1.0)
    ///
    /// # Example
    /// ```ignore
    /// let strategy = FusionStrategy::DempsterShafer { alpha: 0.5 };
    /// ```
    ///
    /// # Deprecation (T4.2)
    ///
    /// **This strategy is deprecated.** Use [`ReciprocalRank`](Self::ReciprocalRank) or migrate to
    /// [`crate::core::search::cross_modal_fusion::Fuser`] for the modern cross-modal fusion seam.
    #[deprecated(
        since = "0.2.0",
        note = "Use FusionStrategy::ReciprocalRank or crate::core::search::cross_modal_fusion::Fuser instead."
    )]
    DempsterShafer {
        /// Belief mass weighting parameter (0.0 to 1.0)
        alpha: f64,
    },

    /// Adaptive Fusion
    ///
    /// Dynamically selects strategy based on query characteristics and result overlap
    /// Chooses between RRF, WeightedLinear, and CombSum based on:
    /// - Result set overlap (Jaccard similarity)
    /// - Score variance
    /// - Rank correlation
    ///
    /// # Example
    /// ```ignore
    /// let strategy = FusionStrategy::Adaptive;
    /// ```
    ///
    /// # Deprecation (T4.2)
    ///
    /// **This strategy is deprecated.** Use [`ReciprocalRank`](Self::ReciprocalRank) or migrate to
    /// [`crate::core::search::cross_modal_fusion::Fuser`] for the modern cross-modal fusion seam.
    #[deprecated(
        since = "0.2.0",
        note = "Use FusionStrategy::ReciprocalRank or crate::core::search::cross_modal_fusion::Fuser instead."
    )]
    Adaptive,

    /// Projection-Based Fusion (B5)
    ///
    /// Formula: `score = project([s_bm25, s_vector], [cos(theta), sin(theta)])`
    /// where theta = alpha * (pi/2)
    ///
    /// # Arguments
    /// * `alpha` - Balance parameter (0.0 to 1.0)
    ///
    /// # Example
    /// ```ignore
    /// let strategy = FusionStrategy::Projection { alpha: 0.5 };
    /// ```
    ///
    /// # Deprecation (T4.2)
    ///
    /// **This strategy is deprecated.** Use [`ReciprocalRank`](Self::ReciprocalRank) or migrate to
    /// [`crate::core::search::cross_modal_fusion::Fuser`] for the modern cross-modal fusion seam.
    #[deprecated(
        since = "0.2.0",
        note = "Use FusionStrategy::ReciprocalRank or crate::core::search::cross_modal_fusion::Fuser instead."
    )]
    Projection {
        /// Balance parameter alpha (0.0 to 1.0)
        alpha: f64,
    },
}

#[allow(deprecated)]
impl std::fmt::Display for FusionStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FusionStrategy::ReciprocalRank { k } => {
                write!(f, "RRF(k={})", k)
            }
            FusionStrategy::WeightedLinear {
                alpha,
                bm25_normalize,
                vector_normalize,
            } => {
                write!(
                    f,
                    "Weighted(alpha={:.2}, bm25_norm={}, vector_norm={})",
                    alpha, bm25_normalize, vector_normalize
                )
            }
            FusionStrategy::RankBiasedPrecision { persistence } => {
                write!(f, "RBP(p={:.2})", persistence)
            }
            FusionStrategy::ConditionalNormalization => {
                write!(f, "CCF")
            }
            FusionStrategy::BordaCount => {
                write!(f, "BordaCount")
            }
            FusionStrategy::CombSum => {
                write!(f, "CombSUM")
            }
            FusionStrategy::CombMin => {
                write!(f, "CombMIN")
            }
            FusionStrategy::CombMax => {
                write!(f, "CombMAX")
            }
            FusionStrategy::Condorcet => {
                write!(f, "Condorcet")
            }
            FusionStrategy::DempsterShafer { alpha } => {
                write!(f, "DempsterShafer(alpha={:.2})", alpha)
            }
            FusionStrategy::Adaptive => {
                write!(f, "Adaptive")
            }
            FusionStrategy::Projection { alpha } => {
                write!(f, "Projection(alpha={:.2})", alpha)
            }
        }
    }
}

/// BM25 search result with text highlighting
#[derive(Debug, Clone, PartialEq)]
pub struct BM25Result {
    /// Document ID
    pub doc_id: String,

    /// BM25 score (higher = better)
    pub score: f64,

    /// Text highlights (if available)
    pub highlights: Option<Vec<TextHighlight>>,

    /// Document metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Vector search result
#[derive(Debug, Clone, PartialEq)]
pub struct VectorResult {
    /// Document ID
    pub doc_id: String,

    /// Similarity score (higher = better)
    pub score: f64,

    /// Distance metric (lower = closer)
    pub distance: f64,

    /// Document metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Fused search result from both BM25 and vector search
#[derive(Debug, Clone)]
pub struct FusedSearchResult {
    /// Document ID
    pub doc_id: String,

    /// BM25 score
    pub bm25_score: f64,

    /// Vector similarity score
    pub vector_score: f64,

    /// Fused/reranked score
    pub fused_score: f64,

    /// Rank in BM25 results (1-based)
    pub bm25_rank: usize,

    /// Rank in vector results (1-based)
    pub vector_rank: usize,

    /// Text highlights (from BM25)
    pub highlights: Option<Vec<TextHighlight>>,

    /// Combined metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Text highlight for search result display
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextHighlight {
    /// Field name containing the highlight
    pub field: String,

    /// Highlighted text
    pub text: String,

    /// Start offset in original text
    pub start_offset: usize,

    /// End offset in original text
    pub end_offset: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[expect(deprecated, reason = "Checks display for legacy fusion variants")]
    fn test_fusion_strategy_display() {
        let strategies = vec![
            FusionStrategy::ReciprocalRank { k: 60 },
            FusionStrategy::WeightedLinear {
                alpha: 0.5,
                bm25_normalize: true,
                vector_normalize: false,
            },
            FusionStrategy::RankBiasedPrecision { persistence: 0.9 },
            FusionStrategy::ConditionalNormalization,
        ];

        for strategy in strategies {
            let display = format!("{}", strategy);
            println!("Strategy: {}", display);
            assert!(!display.is_empty());
        }
    }

    #[test]
    fn test_bm25_result_creation() {
        let result = BM25Result {
            doc_id: "doc1".to_string(),
            score: 2.5,
            highlights: None,
            metadata: HashMap::new(),
        };

        assert_eq!(result.doc_id, "doc1");
        assert_eq!(result.score, 2.5);
    }

    #[test]
    fn test_vector_result_creation() {
        let result = VectorResult {
            doc_id: "doc2".to_string(),
            score: 0.95,
            distance: 0.15,
            metadata: HashMap::new(),
        };

        assert_eq!(result.doc_id, "doc2");
        assert_eq!(result.score, 0.95);
        assert_eq!(result.distance, 0.15);
    }

    #[test]
    fn test_fused_result_creation() {
        let result = FusedSearchResult {
            doc_id: "doc3".to_string(),
            bm25_score: 2.0,
            vector_score: 0.8,
            fused_score: 1.4,
            bm25_rank: 1,
            vector_rank: 2,
            highlights: None,
            metadata: HashMap::new(),
        };

        assert_eq!(result.doc_id, "doc3");
        assert_eq!(result.fused_score, 1.4);
    }

    #[test]
    fn test_text_highlight_creation() {
        let highlight = TextHighlight {
            field: "content".to_string(),
            text: "machine learning".to_string(),
            start_offset: 10,
            end_offset: 26,
        };

        assert_eq!(highlight.field, "content");
        assert_eq!(highlight.text, "machine learning");
    }

    // Additional tests from standalone test files

    #[test]
    fn test_fusion_engine_creation() {
        let _engine = HybridFusionEngine::new(FusionStrategy::ReciprocalRank { k: 60 });

        // Engine created successfully - private fields can't be accessed directly
        assert!(true);
    }

    #[test]
    fn test_empty_fusion() {
        let engine = HybridFusionEngine::new(FusionStrategy::ReciprocalRank { k: 60 });

        let bm25_results: Vec<BM25Result> = vec![];
        let vector_results: Vec<VectorResult> = vec![];

        let fused = engine.fuse(bm25_results, vector_results).unwrap();

        // Should handle empty results
        assert_eq!(fused.len(), 0);
    }

    #[test]
    fn test_fusion_basic_disjoint() {
        let engine = HybridFusionEngine::new(FusionStrategy::ReciprocalRank { k: 60 });

        let bm25_results = vec![BM25Result {
            doc_id: "doc1".to_string(),
            score: 2.5,
            highlights: None,
            metadata: HashMap::new(),
        }];

        let vector_results = vec![VectorResult {
            doc_id: "doc2".to_string(),
            score: 0.95,
            distance: 0.15,
            metadata: HashMap::new(),
        }];

        let fused = engine.fuse(bm25_results, vector_results).unwrap();

        // Should have 2 documents
        assert_eq!(fused.len(), 2);

        // Results should be sorted by fused score
        for i in 1..fused.len() {
            assert!(
                fused[i - 1].fused_score >= fused[i].fused_score,
                "Results not sorted"
            );
        }
    }

    #[test]
    fn test_rrf_score_calculation() {
        // Test RRF score calculation
        // doc1: rank 1 in BM25 → 1/(60+1) = 0.0164
        let engine = HybridFusionEngine::new(FusionStrategy::ReciprocalRank { k: 60 });

        let bm25_results = vec![BM25Result {
            doc_id: "doc1".to_string(),
            score: 2.5,
            highlights: None,
            metadata: HashMap::new(),
        }];

        let vector_results = vec![];

        let fused = engine.fuse(bm25_results, vector_results).unwrap();

        // Verify RRF score calculation: 1/(60+1) = 0.016393...
        let expected_score = 1.0 / 61.0;
        assert!((fused[0].fused_score - expected_score).abs() < 0.0001);
    }

    #[test]
    #[expect(
        deprecated,
        reason = "Checks scoring for the legacy weighted fusion variant"
    )]
    fn test_weighted_linear_fusion() {
        let engine = HybridFusionEngine::new(FusionStrategy::WeightedLinear {
            alpha: 0.5,
            bm25_normalize: false,
            vector_normalize: false,
        });

        let bm25_results = vec![BM25Result {
            doc_id: "doc1".to_string(),
            score: 0.8,
            highlights: None,
            metadata: HashMap::new(),
        }];

        let vector_results = vec![VectorResult {
            doc_id: "doc1".to_string(),
            score: 0.6,
            distance: 0.3,
            metadata: HashMap::new(),
        }];

        let fused = engine.fuse(bm25_results, vector_results).unwrap();

        // Weighted average: 0.5 * 0.8 + 0.5 * 0.6 = 0.7
        assert!((fused[0].fused_score - 0.7).abs() < 0.01);
    }
}
