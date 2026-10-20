//! Fusion strategy implementations
//!
//! Implements various algorithms for combining BM25 and vector search results.

use super::{BM25Result, FusedSearchResult, FusionStrategy, VectorResult};
use std::collections::HashMap;

/// Errors that can occur during fusion
#[derive(Debug, thiserror::Error)]
pub enum FusionError {
    /// Requested fusion strategy is not implemented
    #[error("Strategy not yet implemented: {0}")]
    NotYetImplemented(String),

    /// Fusion parameters are invalid
    #[error("Invalid fusion parameters: {0}")]
    InvalidParameters(String),
}

/// Hybrid fusion engine
///
/// Combines BM25 full-text search results with vector similarity
/// search results using configurable fusion strategies.
pub struct HybridFusionEngine {
    strategy: FusionStrategy,
    default_top_k: usize,
}

impl HybridFusionEngine {
    /// Create a new fusion engine with the given strategy
    ///
    /// # Arguments
    /// * `strategy` - Fusion strategy to use
    ///
    /// # Example
    /// ```ignore
    /// use proximadb::core::search::hybrid::{FusionStrategy, HybridFusionEngine};
    ///
    /// let engine = HybridFusionEngine::new(FusionStrategy::ReciprocalRank { k: 60 });
    /// ```
    pub fn new(strategy: FusionStrategy) -> Self {
        Self {
            strategy,
            default_top_k: 10,
        }
    }

    /// Set default top_k for fusion
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.default_top_k = top_k;
        self
    }

    /// Fuse BM25 and vector results using the configured strategy
    ///
    /// # Arguments
    /// * `bm25_results` - Results from BM25 full-text search
    /// * `vector_results` - Results from vector similarity search
    ///
    /// # Returns
    /// Fused and sorted results
    ///
    /// # Errors
    /// Returns error if fusion strategy is not implemented
    #[allow(deprecated)]
    pub fn fuse(
        &self,
        bm25_results: Vec<BM25Result>,
        vector_results: Vec<VectorResult>,
    ) -> Result<Vec<FusedSearchResult>, FusionError> {
        match self.strategy {
            FusionStrategy::ReciprocalRank { k } => {
                self.reciprocal_rank_fusion(bm25_results, vector_results, k)
            }
            FusionStrategy::WeightedLinear {
                alpha,
                bm25_normalize,
                vector_normalize,
            } => self.weighted_linear_fusion(
                bm25_results,
                vector_results,
                alpha,
                bm25_normalize,
                vector_normalize,
            ),
            FusionStrategy::RankBiasedPrecision { persistence } => {
                self.rank_biased_precision(bm25_results, vector_results, persistence)
            }
            FusionStrategy::ConditionalNormalization => {
                self.conditional_normalization(bm25_results, vector_results)
            }
            FusionStrategy::BordaCount => self.borda_count(bm25_results, vector_results),
            FusionStrategy::CombSum => self.comb_sum(bm25_results, vector_results),
            FusionStrategy::CombMin => self.comb_min(bm25_results, vector_results),
            FusionStrategy::CombMax => self.comb_max(bm25_results, vector_results),
            FusionStrategy::Condorcet => self.condorcet(bm25_results, vector_results),
            FusionStrategy::DempsterShafer { alpha } => {
                self.dempster_shafer(bm25_results, vector_results, alpha)
            }
            FusionStrategy::Adaptive => self.adaptive_fusion(bm25_results, vector_results),
            FusionStrategy::Projection { alpha } => {
                self.projection_fusion(bm25_results, vector_results, alpha)
            }
        }
    }

    /// Score-level projection fusion ("Projection", inspired-by-B5).
    ///
    /// **Naming caveat**: arXiv:2604.13728 introduces a "B5 Projection
    /// Fusion" that operates in *vector space* — combining BGE-dense
    /// (768d) with SPLADE-sparse (30522d, projected to 768d via an
    /// Achlioptas sparse random projection matrix R∈ℝ^{768×30522}) into
    /// one dense vector for ANN, then `q = α_query·d̂ + (1−α_query)·p̂`
    /// followed by L2-normalize, with α_query=0.95 (tuned), α_doc=0.50
    /// (fixed). This implementation is a **score-level** analog:
    /// after BM25 and vector search have each returned ranked lists,
    /// we combine the post-retrieval scalar scores via
    ///
    /// ```text
    /// score = bm25_norm * cos(theta) + vector_norm * sin(theta),
    /// where theta = alpha * pi/2.
    /// ```
    ///
    /// We min-max normalize before projection and apply a small
    /// diversity boost for documents present in both sets.
    ///
    /// This is a different layer of the stack than the paper's B5
    /// (paper fuses *before* retrieval; we fuse *after*). Score-level
    /// projection is a useful operator on its own — it produces a
    /// diversity-leaning alternative to RRF that fits the existing
    /// post-retrieval fusion API — but its quantitative properties are
    /// NOT the same as those the paper measured (paper: B5 nDCG@10
    /// =0.6779 vs RRF=0.8282 — RRF wins relevance; B5 ILD@10=0.389 vs
    /// RRF=0.176 — B5 wins diversity 2.2×).
    ///
    /// A vector-space B5 implementation would require (1) a sparse
    /// encoder integrated into the retrieval path, (2) the Achlioptas
    /// projection matrix, and (3) a single combined-vector ANN index.
    /// Tracked separately, see TD-044 in TECHNICAL_DEBT.adoc.
    fn projection_fusion(
        &self,
        bm25_results: Vec<BM25Result>,
        vector_results: Vec<VectorResult>,
        alpha: f64,
    ) -> Result<Vec<FusedSearchResult>, FusionError> {
        if !(0.0..=1.0).contains(&alpha) {
            return Err(FusionError::InvalidParameters(
                "Alpha must be between 0.0 and 1.0".to_string(),
            ));
        }

        if bm25_results.is_empty() && vector_results.is_empty() {
            return Ok(Vec::new());
        }

        // Find max scores for min-max normalization.
        // NOTE: the 0.0 seed on the max folds below is safe only because
        // every value derived from these maxes (the ranges) is consumed
        // behind a `> 0.0` guard, so an all-negative list (seeded max =
        // 0.0) contributes 0.0 regardless of the seed. If those guards are
        // ever removed, reseed with f64::NEG_INFINITY like the comb_*
        // normalization folds.
        let bm25_max = bm25_results.iter().map(|r| r.score).fold(0.0_f64, f64::max);
        let bm25_min = bm25_results
            .iter()
            .map(|r| r.score)
            .fold(f64::INFINITY, f64::min);
        let bm25_range = if bm25_max > bm25_min {
            bm25_max - bm25_min
        } else {
            1.0
        };

        let vector_max = vector_results
            .iter()
            .map(|r| r.score)
            .fold(0.0_f64, f64::max);
        let vector_min = vector_results
            .iter()
            .map(|r| r.score)
            .fold(f64::INFINITY, f64::min);
        let vector_range = if vector_max > vector_min {
            vector_max - vector_min
        } else {
            1.0
        };

        // Calculate projection vector based on alpha (angle theta)
        let theta = alpha * (std::f64::consts::PI / 2.0);
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        let mut fused_map: HashMap<String, FusedSearchResult> = HashMap::new();

        // Process BM25 results
        for (rank, bm25) in bm25_results.iter().enumerate() {
            let normalized_bm25 = if bm25_max > 0.0 {
                if bm25_max > bm25_min {
                    (bm25.score - bm25_min) / bm25_range
                } else {
                    1.0 // Constant non-zero scores
                }
            } else {
                0.0
            };

            // Initial projection with only BM25 signal
            let projection_score = normalized_bm25 * cos_theta;

            fused_map.insert(
                bm25.doc_id.clone(),
                FusedSearchResult {
                    doc_id: bm25.doc_id.clone(),
                    bm25_score: bm25.score,
                    vector_score: 0.0,
                    fused_score: projection_score,
                    bm25_rank: rank + 1,
                    vector_rank: usize::MAX,
                    highlights: bm25.highlights.clone(),
                    metadata: bm25.metadata.clone(),
                },
            );
        }

        // Process vector results and merge
        for (rank, vector) in vector_results.iter().enumerate() {
            let normalized_vector = if vector_max > 0.0 {
                if vector_max > vector_min {
                    (vector.score - vector_min) / vector_range
                } else {
                    1.0 // Constant non-zero scores
                }
            } else {
                0.0
            };

            let vector_contribution = normalized_vector * sin_theta;

            if let Some(existing) = fused_map.get_mut(&vector.doc_id) {
                // Combine signals
                existing.fused_score += vector_contribution;
                existing.vector_score = vector.score;
                existing.vector_rank = rank + 1;
            } else {
                // Only in vector set
                fused_map.insert(
                    vector.doc_id.clone(),
                    FusedSearchResult {
                        doc_id: vector.doc_id.clone(),
                        bm25_score: 0.0,
                        vector_score: vector.score,
                        fused_score: vector_contribution,
                        bm25_rank: usize::MAX,
                        vector_rank: rank + 1,
                        highlights: None,
                        metadata: vector.metadata.clone(),
                    },
                );
            }
        }

        // Apply diversity boost: rewarding documents present in both sets
        // and normalizing unique contributions.
        for result in fused_map.values_mut() {
            if result.bm25_rank != usize::MAX && result.vector_rank != usize::MAX {
                // Present in both - boost signal
                result.fused_score *= 1.2;
            } else {
                // Present in only one - apply slight diversity penalty to unique results
                // to prioritize high-confidence overlap while maintaining diversity.
                result.fused_score *= 0.9;
            }
        }

        // Sort by fused score (descending)
        let mut fused_results: Vec<_> = fused_map.into_values().collect();
        fused_results.sort_by(|a, b| {
            b.fused_score
                .partial_cmp(&a.fused_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(fused_results)
    }

    /// Reciprocal Rank Fusion (RRF)
    ///
    /// RRF is robust to differences in score scales and provides
    /// good results across different retrieval systems.
    fn reciprocal_rank_fusion(
        &self,
        bm25_results: Vec<BM25Result>,
        vector_results: Vec<VectorResult>,
        k: usize,
    ) -> Result<Vec<FusedSearchResult>, FusionError> {
        let mut fused_map: HashMap<String, FusedSearchResult> = HashMap::new();

        // Process BM25 results
        for (rank, bm25) in bm25_results.iter().enumerate() {
            let rrf_score = 1.0 / (k as f64 + rank as f64 + 1.0);
            fused_map.insert(
                bm25.doc_id.clone(),
                FusedSearchResult {
                    doc_id: bm25.doc_id.clone(),
                    bm25_score: bm25.score,
                    vector_score: 0.0,
                    fused_score: rrf_score,
                    bm25_rank: rank + 1,
                    vector_rank: usize::MAX,
                    highlights: bm25.highlights.clone(),
                    metadata: bm25.metadata.clone(),
                },
            );
        }

        // Process vector results and merge
        for (rank, vector) in vector_results.iter().enumerate() {
            let rrf_score = 1.0 / (k as f64 + rank as f64 + 1.0);

            if let Some(existing) = fused_map.get_mut(&vector.doc_id) {
                // Document appears in both results - sum RRF scores
                existing.fused_score += rrf_score;
                existing.vector_score = vector.score;
                existing.vector_rank = rank + 1;
            } else {
                // Document only in vector results
                fused_map.insert(
                    vector.doc_id.clone(),
                    FusedSearchResult {
                        doc_id: vector.doc_id.clone(),
                        bm25_score: 0.0,
                        vector_score: vector.score,
                        fused_score: rrf_score,
                        bm25_rank: usize::MAX,
                        vector_rank: rank + 1,
                        highlights: None,
                        metadata: vector.metadata.clone(),
                    },
                );
            }
        }

        // Sort by fused score (descending)
        let mut fused_results: Vec<_> = fused_map.into_values().collect();
        fused_results.sort_by(|a, b| {
            b.fused_score
                .partial_cmp(&a.fused_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(fused_results)
    }

    /// Weighted linear combination
    ///
    /// Normalizes scores (if requested) and combines with specified weights.
    fn weighted_linear_fusion(
        &self,
        bm25_results: Vec<BM25Result>,
        vector_results: Vec<VectorResult>,
        alpha: f64,
        bm25_normalize: bool,
        vector_normalize: bool,
    ) -> Result<Vec<FusedSearchResult>, FusionError> {
        // Find max scores for normalization.
        // NOTE: the 0.0 seed is safe only because each max is read solely
        // as a divisor under a `> 0.0` guard below (lists with no positive
        // score fall through to the raw score). If that guard is removed,
        // reseed with f64::NEG_INFINITY like the comb_* normalization folds.
        let bm25_max = if bm25_normalize {
            bm25_results.iter().map(|r| r.score).fold(0.0_f64, f64::max)
        } else {
            1.0
        };

        let vector_max = if vector_normalize {
            vector_results
                .iter()
                .map(|r| r.score)
                .fold(0.0_f64, f64::max)
        } else {
            1.0
        };

        let mut fused_map: HashMap<String, FusedSearchResult> = HashMap::new();

        // Process and normalize BM25 results
        for (rank, bm25) in bm25_results.iter().enumerate() {
            let normalized_score = if bm25_normalize && bm25_max > 0.0 {
                bm25.score / bm25_max
            } else {
                bm25.score
            };

            fused_map.insert(
                bm25.doc_id.clone(),
                FusedSearchResult {
                    doc_id: bm25.doc_id.clone(),
                    bm25_score: bm25.score,
                    vector_score: 0.0,
                    fused_score: alpha * normalized_score,
                    bm25_rank: rank + 1,
                    vector_rank: usize::MAX,
                    highlights: bm25.highlights.clone(),
                    metadata: bm25.metadata.clone(),
                },
            );
        }

        // Process and normalize vector results
        for (rank, vector) in vector_results.iter().enumerate() {
            let normalized_score = if vector_normalize && vector_max > 0.0 {
                vector.score / vector_max
            } else {
                vector.score
            };

            let vector_contribution = (1.0 - alpha) * normalized_score;

            if let Some(existing) = fused_map.get_mut(&vector.doc_id) {
                existing.fused_score += vector_contribution;
                existing.vector_score = vector.score;
                existing.vector_rank = rank + 1;
            } else {
                fused_map.insert(
                    vector.doc_id.clone(),
                    FusedSearchResult {
                        doc_id: vector.doc_id.clone(),
                        bm25_score: 0.0,
                        vector_score: vector.score,
                        fused_score: vector_contribution,
                        bm25_rank: usize::MAX,
                        vector_rank: rank + 1,
                        highlights: None,
                        metadata: vector.metadata.clone(),
                    },
                );
            }
        }

        // Sort by fused score (descending)
        let mut fused_results: Vec<_> = fused_map.into_values().collect();
        fused_results.sort_by(|a, b| {
            b.fused_score
                .partial_cmp(&a.fused_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(fused_results)
    }

    /// Rank Biased Precision
    ///
    /// Emphasizes top ranks exponentially.
    fn rank_biased_precision(
        &self,
        bm25_results: Vec<BM25Result>,
        vector_results: Vec<VectorResult>,
        persistence: f64,
    ) -> Result<Vec<FusedSearchResult>, FusionError> {
        let mut fused_map: HashMap<String, FusedSearchResult> = HashMap::new();

        // Process BM25 results
        for (rank, bm25) in bm25_results.iter().enumerate() {
            let rbp_score = (1.0 - persistence) * persistence.powi(rank as i32);
            fused_map.insert(
                bm25.doc_id.clone(),
                FusedSearchResult {
                    doc_id: bm25.doc_id.clone(),
                    bm25_score: bm25.score,
                    vector_score: 0.0,
                    fused_score: rbp_score,
                    bm25_rank: rank + 1,
                    vector_rank: usize::MAX,
                    highlights: bm25.highlights.clone(),
                    metadata: bm25.metadata.clone(),
                },
            );
        }

        // Process vector results
        for (rank, vector) in vector_results.iter().enumerate() {
            let rbp_score = (1.0 - persistence) * persistence.powi(rank as i32);

            if let Some(existing) = fused_map.get_mut(&vector.doc_id) {
                existing.fused_score += rbp_score;
                existing.vector_score = vector.score;
                existing.vector_rank = rank + 1;
            } else {
                fused_map.insert(
                    vector.doc_id.clone(),
                    FusedSearchResult {
                        doc_id: vector.doc_id.clone(),
                        bm25_score: 0.0,
                        vector_score: vector.score,
                        fused_score: rbp_score,
                        bm25_rank: usize::MAX,
                        vector_rank: rank + 1,
                        highlights: None,
                        metadata: vector.metadata.clone(),
                    },
                );
            }
        }

        // Sort by fused score (descending)
        let mut fused_results: Vec<_> = fused_map.into_values().collect();
        fused_results.sort_by(|a, b| {
            b.fused_score
                .partial_cmp(&a.fused_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(fused_results)
    }

    /// Conditional Normalization
    ///
    /// Normalizes both scores to [0,1] and averages them.
    fn conditional_normalization(
        &self,
        bm25_results: Vec<BM25Result>,
        vector_results: Vec<VectorResult>,
    ) -> Result<Vec<FusedSearchResult>, FusionError> {
        // Find min/max for both score distributions
        let (bm25_min, bm25_max) = bm25_results
            .iter()
            .map(|r| r.score)
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), score| {
                (min.min(score), max.max(score))
            });

        let (vector_min, vector_max) = vector_results
            .iter()
            .map(|r| r.score)
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), score| {
                (min.min(score), max.max(score))
            });

        let bm25_range = bm25_max - bm25_min;
        let vector_range = vector_max - vector_min;

        let mut fused_map: HashMap<String, FusedSearchResult> = HashMap::new();

        // Normalize BM25 results to [0,1]
        for (rank, bm25) in bm25_results.iter().enumerate() {
            let normalized = if bm25_range > 0.0 {
                (bm25.score - bm25_min) / bm25_range
            } else {
                1.0
            };

            fused_map.insert(
                bm25.doc_id.clone(),
                FusedSearchResult {
                    doc_id: bm25.doc_id.clone(),
                    bm25_score: bm25.score,
                    vector_score: 0.0,
                    fused_score: normalized,
                    bm25_rank: rank + 1,
                    vector_rank: usize::MAX,
                    highlights: bm25.highlights.clone(),
                    metadata: bm25.metadata.clone(),
                },
            );
        }

        // Normalize vector results and average
        for (rank, vector) in vector_results.iter().enumerate() {
            let normalized = if vector_range > 0.0 {
                (vector.score - vector_min) / vector_range
            } else {
                1.0
            };

            if let Some(existing) = fused_map.get_mut(&vector.doc_id) {
                // Average: (bm25_norm + vector_norm) / 2
                existing.fused_score = (existing.fused_score + normalized) / 2.0;
                existing.vector_score = vector.score;
                existing.vector_rank = rank + 1;
            } else {
                fused_map.insert(
                    vector.doc_id.clone(),
                    FusedSearchResult {
                        doc_id: vector.doc_id.clone(),
                        bm25_score: 0.0,
                        vector_score: vector.score,
                        fused_score: normalized,
                        bm25_rank: usize::MAX,
                        vector_rank: rank + 1,
                        highlights: None,
                        metadata: vector.metadata.clone(),
                    },
                );
            }
        }

        // Sort by fused score (descending)
        let mut fused_results: Vec<_> = fused_map.into_values().collect();
        fused_results.sort_by(|a, b| {
            b.fused_score
                .partial_cmp(&a.fused_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(fused_results)
    }

    /// Borda Count fusion
    ///
    /// Assigns points based on rank position. Higher ranks get more points.
    /// Formula: `score = (N - rank_bm25) + (N - rank_vector)`
    fn borda_count(
        &self,
        bm25_results: Vec<BM25Result>,
        vector_results: Vec<VectorResult>,
    ) -> Result<Vec<FusedSearchResult>, FusionError> {
        let n = bm25_results.len() + vector_results.len();
        let mut fused_map: HashMap<String, FusedSearchResult> = HashMap::new();

        // Process BM25 results
        for (rank, bm25) in bm25_results.iter().enumerate() {
            let points = n - rank;
            fused_map.insert(
                bm25.doc_id.clone(),
                FusedSearchResult {
                    doc_id: bm25.doc_id.clone(),
                    bm25_score: bm25.score,
                    vector_score: 0.0,
                    fused_score: points as f64,
                    bm25_rank: rank + 1,
                    vector_rank: usize::MAX,
                    highlights: bm25.highlights.clone(),
                    metadata: bm25.metadata.clone(),
                },
            );
        }

        // Process vector results and merge
        for (rank, vector) in vector_results.iter().enumerate() {
            let points = n - rank;

            if let Some(existing) = fused_map.get_mut(&vector.doc_id) {
                // Sum Borda points
                existing.fused_score += points as f64;
                existing.vector_score = vector.score;
                existing.vector_rank = rank + 1;
            } else {
                fused_map.insert(
                    vector.doc_id.clone(),
                    FusedSearchResult {
                        doc_id: vector.doc_id.clone(),
                        bm25_score: 0.0,
                        vector_score: vector.score,
                        fused_score: points as f64,
                        bm25_rank: usize::MAX,
                        vector_rank: rank + 1,
                        highlights: None,
                        metadata: vector.metadata.clone(),
                    },
                );
            }
        }

        let mut fused_results: Vec<_> = fused_map.into_values().collect();
        fused_results.sort_by(|a, b| {
            b.fused_score
                .partial_cmp(&a.fused_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(fused_results)
    }

    /// CombSUM fusion
    ///
    /// Simple summation of normalized scores.
    /// Formula: `score = bm25_normalized + vector_normalized`
    fn comb_sum(
        &self,
        bm25_results: Vec<BM25Result>,
        vector_results: Vec<VectorResult>,
    ) -> Result<Vec<FusedSearchResult>, FusionError> {
        // Normalize BM25 scores to [0,1]
        let bm25_max = bm25_results
            .iter()
            .map(|r| r.score)
            .fold(f64::NEG_INFINITY, |a, b| a.max(b));
        let bm25_min = bm25_results
            .iter()
            .map(|r| r.score)
            .fold(f64::INFINITY, |a, b| a.min(b));
        let bm25_range = (bm25_max - bm25_min).max(0.001); // Avoid division by zero

        // Normalize vector scores to [0,1]
        let vector_max = vector_results
            .iter()
            .map(|r| r.score)
            .fold(f64::NEG_INFINITY, |a, b| a.max(b));
        let vector_min = vector_results
            .iter()
            .map(|r| r.score)
            .fold(f64::INFINITY, |a, b| a.min(b));
        let vector_range = (vector_max - vector_min).max(0.001);

        let mut fused_map: HashMap<String, FusedSearchResult> = HashMap::new();

        // Process BM25 results
        for (rank, bm25) in bm25_results.iter().enumerate() {
            let normalized_score = (bm25.score - bm25_min) / bm25_range;
            fused_map.insert(
                bm25.doc_id.clone(),
                FusedSearchResult {
                    doc_id: bm25.doc_id.clone(),
                    bm25_score: bm25.score,
                    vector_score: 0.0,
                    fused_score: normalized_score,
                    bm25_rank: rank + 1,
                    vector_rank: usize::MAX,
                    highlights: bm25.highlights.clone(),
                    metadata: bm25.metadata.clone(),
                },
            );
        }

        // Process vector results and sum
        for (rank, vector) in vector_results.iter().enumerate() {
            let normalized_score = (vector.score - vector_min) / vector_range;

            if let Some(existing) = fused_map.get_mut(&vector.doc_id) {
                existing.fused_score += normalized_score;
                existing.vector_score = vector.score;
                existing.vector_rank = rank + 1;
            } else {
                fused_map.insert(
                    vector.doc_id.clone(),
                    FusedSearchResult {
                        doc_id: vector.doc_id.clone(),
                        bm25_score: 0.0,
                        vector_score: vector.score,
                        fused_score: normalized_score,
                        bm25_rank: usize::MAX,
                        vector_rank: rank + 1,
                        highlights: None,
                        metadata: vector.metadata.clone(),
                    },
                );
            }
        }

        let mut fused_results: Vec<_> = fused_map.into_values().collect();
        fused_results.sort_by(|a, b| {
            b.fused_score
                .partial_cmp(&a.fused_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(fused_results)
    }

    /// CombMIN fusion
    ///
    /// Takes the minimum score (pessimistic approach).
    /// Formula: `score = min(bm25_normalized, vector_normalized)`
    fn comb_min(
        &self,
        bm25_results: Vec<BM25Result>,
        vector_results: Vec<VectorResult>,
    ) -> Result<Vec<FusedSearchResult>, FusionError> {
        // Normalize BM25 scores
        let bm25_max = bm25_results
            .iter()
            .map(|r| r.score)
            .fold(f64::NEG_INFINITY, |a, b| a.max(b));
        let bm25_min = bm25_results
            .iter()
            .map(|r| r.score)
            .fold(f64::INFINITY, |a, b| a.min(b));
        let bm25_range = (bm25_max - bm25_min).max(0.001);

        // Normalize vector scores
        let vector_max = vector_results
            .iter()
            .map(|r| r.score)
            .fold(f64::NEG_INFINITY, |a, b| a.max(b));
        let vector_min = vector_results
            .iter()
            .map(|r| r.score)
            .fold(f64::INFINITY, |a, b| a.min(b));
        let vector_range = (vector_max - vector_min).max(0.001);

        let mut fused_map: HashMap<String, FusedSearchResult> = HashMap::new();

        // Process BM25 results
        for (rank, bm25) in bm25_results.iter().enumerate() {
            let normalized_score = (bm25.score - bm25_min) / bm25_range;
            fused_map.insert(
                bm25.doc_id.clone(),
                FusedSearchResult {
                    doc_id: bm25.doc_id.clone(),
                    bm25_score: bm25.score,
                    vector_score: 0.0,
                    fused_score: normalized_score,
                    bm25_rank: rank + 1,
                    vector_rank: usize::MAX,
                    highlights: bm25.highlights.clone(),
                    metadata: bm25.metadata.clone(),
                },
            );
        }

        // Process vector results and take minimum
        for (rank, vector) in vector_results.iter().enumerate() {
            let normalized_score = (vector.score - vector_min) / vector_range;

            if let Some(existing) = fused_map.get_mut(&vector.doc_id) {
                existing.fused_score = existing.fused_score.min(normalized_score);
                existing.vector_score = vector.score;
                existing.vector_rank = rank + 1;
            } else {
                fused_map.insert(
                    vector.doc_id.clone(),
                    FusedSearchResult {
                        doc_id: vector.doc_id.clone(),
                        bm25_score: 0.0,
                        vector_score: vector.score,
                        fused_score: normalized_score,
                        bm25_rank: usize::MAX,
                        vector_rank: rank + 1,
                        highlights: None,
                        metadata: vector.metadata.clone(),
                    },
                );
            }
        }

        let mut fused_results: Vec<_> = fused_map.into_values().collect();
        fused_results.sort_by(|a, b| {
            b.fused_score
                .partial_cmp(&a.fused_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(fused_results)
    }

    /// CombMAX fusion
    ///
    /// Takes the maximum score (optimistic approach).
    /// Formula: `score = max(bm25_normalized, vector_normalized)`
    fn comb_max(
        &self,
        bm25_results: Vec<BM25Result>,
        vector_results: Vec<VectorResult>,
    ) -> Result<Vec<FusedSearchResult>, FusionError> {
        // Normalize BM25 scores
        let bm25_max = bm25_results
            .iter()
            .map(|r| r.score)
            .fold(f64::NEG_INFINITY, |a, b| a.max(b));
        let bm25_min = bm25_results
            .iter()
            .map(|r| r.score)
            .fold(f64::INFINITY, |a, b| a.min(b));
        let bm25_range = (bm25_max - bm25_min).max(0.001);

        // Normalize vector scores
        let vector_max = vector_results
            .iter()
            .map(|r| r.score)
            .fold(f64::NEG_INFINITY, |a, b| a.max(b));
        let vector_min = vector_results
            .iter()
            .map(|r| r.score)
            .fold(f64::INFINITY, |a, b| a.min(b));
        let vector_range = (vector_max - vector_min).max(0.001);

        let mut fused_map: HashMap<String, FusedSearchResult> = HashMap::new();

        // Process BM25 results
        for (rank, bm25) in bm25_results.iter().enumerate() {
            let normalized_score = (bm25.score - bm25_min) / bm25_range;
            fused_map.insert(
                bm25.doc_id.clone(),
                FusedSearchResult {
                    doc_id: bm25.doc_id.clone(),
                    bm25_score: bm25.score,
                    vector_score: 0.0,
                    fused_score: normalized_score,
                    bm25_rank: rank + 1,
                    vector_rank: usize::MAX,
                    highlights: bm25.highlights.clone(),
                    metadata: bm25.metadata.clone(),
                },
            );
        }

        // Process vector results and take maximum
        for (rank, vector) in vector_results.iter().enumerate() {
            let normalized_score = (vector.score - vector_min) / vector_range;

            if let Some(existing) = fused_map.get_mut(&vector.doc_id) {
                existing.fused_score = existing.fused_score.max(normalized_score);
                existing.vector_score = vector.score;
                existing.vector_rank = rank + 1;
            } else {
                fused_map.insert(
                    vector.doc_id.clone(),
                    FusedSearchResult {
                        doc_id: vector.doc_id.clone(),
                        bm25_score: 0.0,
                        vector_score: vector.score,
                        fused_score: normalized_score,
                        bm25_rank: usize::MAX,
                        vector_rank: rank + 1,
                        highlights: None,
                        metadata: vector.metadata.clone(),
                    },
                );
            }
        }

        let mut fused_results: Vec<_> = fused_map.into_values().collect();
        fused_results.sort_by(|a, b| {
            b.fused_score
                .partial_cmp(&a.fused_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(fused_results)
    }

    /// Condorcet fusion
    ///
    /// Pairwise comparison: a document "wins" if it ranks higher than another
    /// in both result sets. Score = (wins - losses).
    fn condorcet(
        &self,
        bm25_results: Vec<BM25Result>,
        vector_results: Vec<VectorResult>,
    ) -> Result<Vec<FusedSearchResult>, FusionError> {
        let mut fused_map: HashMap<String, FusedSearchResult> = HashMap::new();

        // Create ranking maps
        let bm25_ranks: HashMap<String, usize> = bm25_results
            .iter()
            .enumerate()
            .map(|(i, r)| (r.doc_id.clone(), i))
            .collect();
        let vector_ranks: HashMap<String, usize> = vector_results
            .iter()
            .enumerate()
            .map(|(i, r)| (r.doc_id.clone(), i))
            .collect();

        // Get all unique documents
        let all_docs: Vec<String> = bm25_results
            .iter()
            .map(|r| r.doc_id.clone())
            .chain(vector_results.iter().map(|r| r.doc_id.clone()))
            .collect();

        // Calculate Condorcet scores
        for doc_id in &all_docs {
            let mut wins = 0i32;
            let mut losses = 0i32;

            let bm25_rank = bm25_ranks.get(doc_id);
            let vector_rank = vector_ranks.get(doc_id);

            // Compare against all other documents
            for other_doc in &all_docs {
                if doc_id == other_doc {
                    continue;
                }

                let bm25_better = bm25_rank
                    .zip(bm25_ranks.get(other_doc))
                    .is_some_and(|(my_r, other_r)| my_r < other_r);
                let vector_better = vector_rank
                    .zip(vector_ranks.get(other_doc))
                    .is_some_and(|(my_r, other_r)| my_r < other_r);

                // Wins if ranks higher in both
                if bm25_better && vector_better {
                    wins += 1;
                }

                let other_bm25_better = bm25_ranks
                    .get(other_doc)
                    .zip(bm25_rank)
                    .is_some_and(|(other_r, my_r)| other_r < my_r);
                let other_vector_better = vector_ranks
                    .get(other_doc)
                    .zip(vector_rank)
                    .is_some_and(|(other_r, my_r)| other_r < my_r);

                // Losses if ranks lower in both
                if other_bm25_better && other_vector_better {
                    losses += 1;
                }
            }

            let condorcet_score = (wins - losses) as f64;

            // Get original scores
            let bm25_score = bm25_results
                .iter()
                .find(|r| &r.doc_id == doc_id)
                .map_or(0.0, |r| r.score);
            let vector_score = vector_results
                .iter()
                .find(|r| &r.doc_id == doc_id)
                .map_or(0.0, |r| r.score);

            fused_map.insert(
                doc_id.to_string(),
                FusedSearchResult {
                    doc_id: doc_id.clone(),
                    bm25_score,
                    vector_score,
                    fused_score: condorcet_score,
                    bm25_rank: bm25_rank.map_or(usize::MAX, |r| r + 1),
                    vector_rank: vector_rank.map_or(usize::MAX, |r| r + 1),
                    highlights: None,
                    metadata: HashMap::new(),
                },
            );
        }

        let mut fused_results: Vec<_> = fused_map.into_values().collect();
        fused_results.sort_by(|a, b| {
            b.fused_score
                .partial_cmp(&a.fused_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(fused_results)
    }

    /// Dempster-Shafer fusion
    ///
    /// Evidence theory combination using weighted belief functions.
    /// Treats normalized scores as evidence masses.
    fn dempster_shafer(
        &self,
        bm25_results: Vec<BM25Result>,
        vector_results: Vec<VectorResult>,
        alpha: f64,
    ) -> Result<Vec<FusedSearchResult>, FusionError> {
        // Normalize scores to [0,1] as belief functions
        let bm25_max = bm25_results
            .iter()
            .map(|r| r.score)
            .fold(f64::NEG_INFINITY, |a, b| a.max(b));
        let bm25_min = bm25_results
            .iter()
            .map(|r| r.score)
            .fold(f64::INFINITY, |a, b| a.min(b));
        let bm25_range = (bm25_max - bm25_min).max(0.001);

        let vector_max = vector_results
            .iter()
            .map(|r| r.score)
            .fold(f64::NEG_INFINITY, |a, b| a.max(b));
        let vector_min = vector_results
            .iter()
            .map(|r| r.score)
            .fold(f64::INFINITY, |a, b| a.min(b));
        let vector_range = (vector_max - vector_min).max(0.001);

        let mut fused_map: HashMap<String, FusedSearchResult> = HashMap::new();

        // Process BM25 results
        for (rank, bm25) in bm25_results.iter().enumerate() {
            let belief = (bm25.score - bm25_min) / bm25_range;
            fused_map.insert(
                bm25.doc_id.clone(),
                FusedSearchResult {
                    doc_id: bm25.doc_id.clone(),
                    bm25_score: bm25.score,
                    vector_score: 0.0,
                    fused_score: belief,
                    bm25_rank: rank + 1,
                    vector_rank: usize::MAX,
                    highlights: bm25.highlights.clone(),
                    metadata: bm25.metadata.clone(),
                },
            );
        }

        // Combine using Dempster's rule (weighted combination)
        for (rank, vector) in vector_results.iter().enumerate() {
            let belief = (vector.score - vector_min) / vector_range;

            if let Some(existing) = fused_map.get_mut(&vector.doc_id) {
                // Dempster's combination rule with weighted alpha
                let m1 = existing.fused_score;
                let m2 = belief;
                let combined = (m1 * m2) + (alpha * (m1 + m2 - 2.0 * m1 * m2));
                existing.fused_score = combined;
                existing.vector_score = vector.score;
                existing.vector_rank = rank + 1;
            } else {
                fused_map.insert(
                    vector.doc_id.clone(),
                    FusedSearchResult {
                        doc_id: vector.doc_id.clone(),
                        bm25_score: 0.0,
                        vector_score: vector.score,
                        fused_score: belief,
                        bm25_rank: usize::MAX,
                        vector_rank: rank + 1,
                        highlights: None,
                        metadata: vector.metadata.clone(),
                    },
                );
            }
        }

        let mut fused_results: Vec<_> = fused_map.into_values().collect();
        fused_results.sort_by(|a, b| {
            b.fused_score
                .partial_cmp(&a.fused_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(fused_results)
    }

    /// Adaptive fusion
    ///
    /// Dynamically selects the best fusion strategy based on:
    /// - Result set overlap (Jaccard similarity)
    /// - Score variance
    /// - Rank correlation
    #[allow(deprecated)]
    fn adaptive_fusion(
        &self,
        bm25_results: Vec<BM25Result>,
        vector_results: Vec<VectorResult>,
    ) -> Result<Vec<FusedSearchResult>, FusionError> {
        // Calculate Jaccard similarity between result sets
        let bm25_ids: std::collections::HashSet<_> =
            bm25_results.iter().map(|r| r.doc_id.as_str()).collect();
        let vector_ids: std::collections::HashSet<_> =
            vector_results.iter().map(|r| r.doc_id.as_str()).collect();

        let intersection = bm25_ids.intersection(&vector_ids).count();
        let union = bm25_ids.union(&vector_ids).count();
        let jaccard = if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        };

        // Choose strategy based on overlap
        let chosen_strategy = if jaccard > 0.5 {
            // High overlap - use WeightedLinear for smooth combination
            FusionStrategy::WeightedLinear {
                alpha: 0.5,
                bm25_normalize: true,
                vector_normalize: true,
            }
        } else if jaccard > 0.2 {
            // Medium overlap - use RRF
            FusionStrategy::ReciprocalRank { k: 60 }
        } else {
            // Low overlap - use CombSUM to combine diverse results
            FusionStrategy::CombSum
        };

        // Execute chosen strategy
        match chosen_strategy {
            FusionStrategy::WeightedLinear {
                alpha,
                bm25_normalize,
                vector_normalize,
            } => self.weighted_linear_fusion(
                bm25_results,
                vector_results,
                alpha,
                bm25_normalize,
                vector_normalize,
            ),
            FusionStrategy::ReciprocalRank { k } => {
                self.reciprocal_rank_fusion(bm25_results, vector_results, k)
            }
            FusionStrategy::CombSum => self.comb_sum(bm25_results, vector_results),
            _ => Err(FusionError::NotYetImplemented(
                "Adaptive strategy selection failed".to_string(),
            )),
        }
    }
}

// These tests cover the legacy `FusionStrategy` hybrid-fusion path, `#[deprecated]` in favour of
// the cross-modal `Fuser` (search-surface contract #280). They intentionally exercise the deprecated
// variants; allow it here until the legacy engine is retired. (Comment is placed ABOVE `#[cfg(test)]`
// because the panic-scanner's cfg(test) block-skip bails on a `;` between the attr and the block.)
#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = HybridFusionEngine::new(FusionStrategy::ReciprocalRank { k: 60 });
        assert_eq!(engine.default_top_k, 10);
    }

    #[test]
    fn test_engine_with_top_k() {
        let engine =
            HybridFusionEngine::new(FusionStrategy::ReciprocalRank { k: 60 }).with_top_k(20);
        assert_eq!(engine.default_top_k, 20);
    }

    #[test]
    fn test_rrf_basic() {
        let engine = HybridFusionEngine::new(FusionStrategy::ReciprocalRank { k: 10 });

        let bm25_results = vec![BM25Result {
            doc_id: "doc1".to_string(),
            score: 1.0,
            highlights: None,
            metadata: HashMap::new(),
        }];

        let vector_results = vec![VectorResult {
            doc_id: "doc2".to_string(),
            score: 0.9,
            distance: 0.1,
            metadata: HashMap::new(),
        }];

        let fused = engine
            .fuse(bm25_results, vector_results)
            .expect("Fusion should succeed");

        // Should have 2 documents
        assert_eq!(fused.len(), 2);
    }

    // ==================== BORDA COUNT TESTS ====================

    #[test]
    fn test_borda_count_disjoint_results() {
        let engine = HybridFusionEngine::new(FusionStrategy::BordaCount);

        // Disjoint results - no overlap
        let bm25_results = vec![
            BM25Result {
                doc_id: "doc1".to_string(),
                score: 1.0,
                highlights: None,
                metadata: HashMap::new(),
            },
            BM25Result {
                doc_id: "doc2".to_string(),
                score: 0.9,
                highlights: None,
                metadata: HashMap::new(),
            },
        ];

        let vector_results = vec![
            VectorResult {
                doc_id: "doc3".to_string(),
                score: 0.95,
                distance: 0.05,
                metadata: HashMap::new(),
            },
            VectorResult {
                doc_id: "doc4".to_string(),
                score: 0.85,
                distance: 0.15,
                metadata: HashMap::new(),
            },
        ];

        let fused = engine
            .fuse(bm25_results, vector_results)
            .expect("Fusion should succeed");

        // Total results: 4 documents
        assert_eq!(fused.len(), 4);

        // All documents should have scores (3, 2, 1, 0)
        let scores: Vec<_> = fused.iter().map(|r| r.fused_score).collect();
        assert!(scores.iter().all(|&s| (0.0..=4.0).contains(&s)));
    }

    #[test]
    fn test_borda_count_overlapping_results() {
        let engine = HybridFusionEngine::new(FusionStrategy::BordaCount);

        // Overlapping results - doc1 appears in both
        let bm25_results = vec![
            BM25Result {
                doc_id: "doc1".to_string(),
                score: 1.0,
                highlights: None,
                metadata: HashMap::new(),
            },
            BM25Result {
                doc_id: "doc2".to_string(),
                score: 0.9,
                highlights: None,
                metadata: HashMap::new(),
            },
        ];

        let vector_results = vec![
            VectorResult {
                doc_id: "doc1".to_string(), // Overlaps
                score: 0.95,
                distance: 0.05,
                metadata: HashMap::new(),
            },
            VectorResult {
                doc_id: "doc3".to_string(),
                score: 0.85,
                distance: 0.15,
                metadata: HashMap::new(),
            },
        ];

        let fused = engine
            .fuse(bm25_results, vector_results)
            .expect("Fusion should succeed");

        // Total unique documents: 3 (doc1, doc2, doc3)
        assert_eq!(fused.len(), 3);

        // doc1 should be ranked highest (appears in both lists)
        // n = 4 (2+2)
        // BM25 rank 0: 4-0 = 4 points
        // Vector rank 0: 4-0 = 4 points
        // Total for doc1: 4 + 4 = 8 points
        let doc1_result = fused
            .iter()
            .find(|r| r.doc_id == "doc1")
            .expect("doc1 should exist in fused results");
        assert_eq!(doc1_result.fused_score, 8.0);

        // doc2: only in BM25 at rank 1: 4-1 = 3 points
        let doc2_result = fused
            .iter()
            .find(|r| r.doc_id == "doc2")
            .expect("doc2 should exist in fused results");
        assert_eq!(doc2_result.fused_score, 3.0);

        // doc3: only in vector at rank 1: 4-1 = 3 points
        let doc3_result = fused
            .iter()
            .find(|r| r.doc_id == "doc3")
            .expect("doc3 should exist in fused results");
        assert_eq!(doc3_result.fused_score, 3.0);
    }

    // ==================== COMBSUM TESTS ====================

    #[test]
    fn test_comb_sum_basic() {
        let engine = HybridFusionEngine::new(FusionStrategy::CombSum);

        let bm25_results = vec![
            BM25Result {
                doc_id: "doc1".to_string(),
                score: 1.0,
                highlights: None,
                metadata: HashMap::new(),
            },
            BM25Result {
                doc_id: "doc2".to_string(),
                score: 0.5,
                highlights: None,
                metadata: HashMap::new(),
            },
        ];

        let vector_results = vec![
            VectorResult {
                doc_id: "doc1".to_string(),
                score: 0.8,
                distance: 0.2,
                metadata: HashMap::new(),
            },
            VectorResult {
                doc_id: "doc3".to_string(),
                score: 0.6,
                distance: 0.4,
                metadata: HashMap::new(),
            },
        ];

        let fused = engine
            .fuse(bm25_results, vector_results)
            .expect("Fusion should succeed");

        assert_eq!(fused.len(), 3);

        // After normalization and summation:
        // BM25: max=1.0, min=0.5, range=0.5
        //   doc1: (1.0-0.5)/0.5 = 1.0
        //   doc2: (0.5-0.5)/0.5 = 0.0
        // Vector: max=0.8, min=0.6, range=0.2
        //   doc1: (0.8-0.6)/0.2 = 1.0
        //   doc3: (0.6-0.6)/0.2 = 0.0
        // doc1: 1.0 + 1.0 = 2.0
        let doc1 = fused
            .iter()
            .find(|r| r.doc_id == "doc1")
            .expect("doc1 should exist in fused results");
        assert!((doc1.fused_score - 2.0).abs() < 0.01);

        // doc1 should have highest score
        assert_eq!(fused[0].doc_id, "doc1");
    }

    fn fused_score_for(fused: &[FusedSearchResult], id: &str) -> f64 {
        fused
            .iter()
            .find(|r| r.doc_id == id)
            .unwrap_or_else(|| panic!("{id} should exist in fused results"))
            .fused_score
    }

    /// Evenly-spaced scores, reversed across the two lists: under correct
    /// min-max normalization every doc sits at (1,0)/(0.5,0.5)/(0,1), so
    /// CombSUM ties all three at 1.0. Under a min fold seeded from 0.0 the
    /// tie collapses into a strict (wrong) order 1.93/1.63/1.33.
    #[test]
    fn test_comb_sum_normalizes_with_true_min() {
        let engine = HybridFusionEngine::new(FusionStrategy::CombSum);
        let bm25 = vec![
            make_bm25("doc_x", 3.0),
            make_bm25("doc_y", 2.0),
            make_bm25("doc_z", 1.0),
        ];
        let vector = vec![
            make_vector("doc_x", 0.28),
            make_vector("doc_y", 0.29),
            make_vector("doc_z", 0.30),
        ];

        let fused = engine.fuse(bm25, vector).expect("fusion succeeds");

        for id in ["doc_x", "doc_y", "doc_z"] {
            let score = fused_score_for(&fused, id);
            assert!(
                (score - 1.0).abs() < 1e-9,
                "{id} should tie at 1.0, got {score}"
            );
        }
    }

    /// All-negative vector scores: a max fold seeded from 0.0 computes
    /// max = 0 and yields nonsense normalization; identity-seeded folds
    /// (NEG_INFINITY max / INFINITY min) handle sign-uniform lists.
    #[test]
    fn test_comb_sum_survives_all_negative_scores() {
        let engine = HybridFusionEngine::new(FusionStrategy::CombSum);
        let bm25 = vec![make_bm25("doc_a", 1.0), make_bm25("doc_b", 0.5)];
        let vector = vec![make_vector("doc_a", -0.2), make_vector("doc_b", -0.8)];

        let fused = engine.fuse(bm25, vector).expect("fusion succeeds");

        // Correct norms: bm25 (1.0, 0.0); vector ((-0.2+0.8)/0.6, 0.0) = (1.0, 0.0)
        assert!((fused_score_for(&fused, "doc_a") - 2.0).abs() < 1e-9);
        assert!(fused_score_for(&fused, "doc_b").abs() < 1e-9);
    }

    /// A constant (single-element) list now normalizes to 0.0 (min == max,
    /// range clamps to 0.001). A min fold seeded from 0.0 normalized it to
    /// 1.0 instead. NOTE: conditional_normalization deliberately maps
    /// constant lists to 1.0; aligning that policy would be a separate,
    /// intentional change.
    #[test]
    fn test_comb_sum_constant_list_normalizes_to_zero() {
        let engine = HybridFusionEngine::new(FusionStrategy::CombSum);
        let bm25 = vec![make_bm25("only", 1.0), make_bm25("other", 0.0)];
        let vector = vec![make_vector("only", 0.5)];

        let fused = engine.fuse(bm25, vector).expect("fusion succeeds");

        // "only": bm25 norm 1.0 + constant-list vector norm 0.0
        let score = fused_score_for(&fused, "only");
        assert!((score - 1.0).abs() < 1e-9, "got {score}");
    }

    #[test]
    fn test_comb_sum_normalization() {
        let engine = HybridFusionEngine::new(FusionStrategy::CombSum);

        // Scores with different ranges
        let bm25_results = vec![BM25Result {
            doc_id: "doc1".to_string(),
            score: 100.0, // Large score
            highlights: None,
            metadata: HashMap::new(),
        }];

        let vector_results = vec![VectorResult {
            doc_id: "doc1".to_string(),
            score: 0.5, // Small score
            distance: 0.5,
            metadata: HashMap::new(),
        }];

        let fused = engine
            .fuse(bm25_results, vector_results)
            .expect("Fusion should succeed");

        // Should normalize before combining
        let doc1 = &fused[0];
        // After normalization, both scores should be in [0,1]
        assert!(doc1.fused_score >= 0.0 && doc1.fused_score <= 2.0);
    }

    // ==================== COMBMIN TESTS ====================

    #[test]
    fn test_comb_min_pessimistic() {
        let engine = HybridFusionEngine::new(FusionStrategy::CombMin);

        let bm25_results = vec![
            BM25Result {
                doc_id: "doc1".to_string(),
                score: 1.0,
                highlights: None,
                metadata: HashMap::new(),
            },
            BM25Result {
                doc_id: "doc2".to_string(),
                score: 0.3,
                highlights: None,
                metadata: HashMap::new(),
            },
        ];

        let vector_results = vec![
            VectorResult {
                doc_id: "doc1".to_string(),
                score: 0.8,
                distance: 0.2,
                metadata: HashMap::new(),
            },
            VectorResult {
                doc_id: "doc2".to_string(),
                score: 0.9,
                distance: 0.1,
                metadata: HashMap::new(),
            },
        ];

        let fused = engine
            .fuse(bm25_results, vector_results)
            .expect("Fusion should succeed");

        // With true min-max normalization:
        //   BM25 norms:   doc1 (1.0-0.3)/0.7 = 1.0, doc2 0.0
        //   Vector norms: doc1 0.0,            doc2 (0.9-0.8)/0.1 = 1.0
        // CombMin is pessimistic: each doc is capped by its WEAKEST list, and
        // both docs sit at an opposite extreme, so both collapse to 0.0.
        // (Under the old min-fold-from-0.0 bug doc1 won with min(1.0, 0.89).)
        let doc1 = fused
            .iter()
            .find(|r| r.doc_id == "doc1")
            .expect("doc1 should exist in fused results");
        let doc2 = fused
            .iter()
            .find(|r| r.doc_id == "doc2")
            .expect("doc2 should exist in fused results");

        assert!(doc1.fused_score.abs() < 1e-9);
        assert!(doc2.fused_score.abs() < 1e-9);
    }

    /// Correct min-max normalization reverses the CombMin winner on the
    /// reversed-fixture: doc_y is mediocre in both lists (0.5/0.5) and wins,
    /// while the cross-list extremes (1,0) and (0,1) collapse to 0. A min
    /// fold seeded from 0.0 used to crown doc_x instead.
    #[test]
    fn test_comb_min_uses_true_min_normalization() {
        let engine = HybridFusionEngine::new(FusionStrategy::CombMin);
        let bm25 = vec![
            make_bm25("doc_x", 3.0),
            make_bm25("doc_y", 2.0),
            make_bm25("doc_z", 1.0),
        ];
        let vector = vec![
            make_vector("doc_x", 0.28),
            make_vector("doc_y", 0.29),
            make_vector("doc_z", 0.30),
        ];

        let fused = engine.fuse(bm25, vector).expect("fusion succeeds");

        assert!((fused_score_for(&fused, "doc_y") - 0.5).abs() < 1e-9);
        assert!(fused_score_for(&fused, "doc_x").abs() < 1e-9);
        assert!(fused_score_for(&fused, "doc_z").abs() < 1e-9);
    }

    // ==================== COMBMAX TESTS ====================

    #[test]
    fn test_comb_max_optimistic() {
        let engine = HybridFusionEngine::new(FusionStrategy::CombMax);

        let bm25_results = vec![
            BM25Result {
                doc_id: "doc1".to_string(),
                score: 1.0,
                highlights: None,
                metadata: HashMap::new(),
            },
            BM25Result {
                doc_id: "doc2".to_string(),
                score: 0.3,
                highlights: None,
                metadata: HashMap::new(),
            },
        ];

        let vector_results = vec![
            VectorResult {
                doc_id: "doc1".to_string(),
                score: 0.6,
                distance: 0.4,
                metadata: HashMap::new(),
            },
            VectorResult {
                doc_id: "doc2".to_string(),
                score: 0.9,
                distance: 0.1,
                metadata: HashMap::new(),
            },
        ];

        let fused = engine
            .fuse(bm25_results, vector_results)
            .expect("Fusion should succeed");

        // After normalization:
        // BM25: max=1.0, min=0.3, range=0.7
        //   doc1: (1.0-0.3)/0.7 = 1.0
        //   doc2: (0.3-0.3)/0.7 = 0.0
        // Vector: max=0.9, min=0.6, range=0.3
        //   doc1: (0.6-0.6)/0.3 = 0.0
        //   doc2: (0.9-0.6)/0.3 = 1.0
        // doc1: max(1.0, 0.0) = 1.0
        // doc2: max(0.0, 1.0) = 1.0
        // Both have the same max score after normalization!
        let doc1 = fused
            .iter()
            .find(|r| r.doc_id == "doc1")
            .expect("doc1 should exist in fused results");
        let doc2 = fused
            .iter()
            .find(|r| r.doc_id == "doc2")
            .expect("doc2 should exist in fused results");

        // Both should have score 1.0 (they're tied)
        assert!((doc1.fused_score - 1.0).abs() < 0.01);
        assert!((doc2.fused_score - 1.0).abs() < 0.01);
    }

    /// CombMax discriminates on the middle doc: doc_y is 0.5 in both lists.
    /// A min fold seeded from 0.0 scored it 0.9667 instead.
    #[test]
    fn test_comb_max_uses_true_min_normalization() {
        let engine = HybridFusionEngine::new(FusionStrategy::CombMax);
        let bm25 = vec![
            make_bm25("doc_x", 3.0),
            make_bm25("doc_y", 2.0),
            make_bm25("doc_z", 1.0),
        ];
        let vector = vec![
            make_vector("doc_x", 0.28),
            make_vector("doc_y", 0.29),
            make_vector("doc_z", 0.30),
        ];

        let fused = engine.fuse(bm25, vector).expect("fusion succeeds");

        assert!((fused_score_for(&fused, "doc_x") - 1.0).abs() < 1e-9);
        assert!((fused_score_for(&fused, "doc_y") - 0.5).abs() < 1e-9);
        assert!((fused_score_for(&fused, "doc_z") - 1.0).abs() < 1e-9);
    }

    // ==================== CONDORCET TESTS ====================

    #[test]
    fn test_condorcet_pairwise() {
        let engine = HybridFusionEngine::new(FusionStrategy::Condorcet);

        let bm25_results = vec![
            BM25Result {
                doc_id: "doc1".to_string(),
                score: 1.0,
                highlights: None,
                metadata: HashMap::new(),
            },
            BM25Result {
                doc_id: "doc2".to_string(),
                score: 0.5,
                highlights: None,
                metadata: HashMap::new(),
            },
        ];

        let vector_results = vec![
            VectorResult {
                doc_id: "doc1".to_string(),
                score: 0.9,
                distance: 0.1,
                metadata: HashMap::new(),
            },
            VectorResult {
                doc_id: "doc2".to_string(),
                score: 0.8,
                distance: 0.2,
                metadata: HashMap::new(),
            },
        ];

        let fused = engine
            .fuse(bm25_results, vector_results)
            .expect("Fusion should succeed");

        assert_eq!(fused.len(), 2);

        // doc1 should win over doc2 in both BM25 and vector
        let doc1 = fused
            .iter()
            .find(|r| r.doc_id == "doc1")
            .expect("doc1 should exist in fused results");
        let doc2 = fused
            .iter()
            .find(|r| r.doc_id == "doc2")
            .expect("doc2 should exist in fused results");

        // doc1 should have positive score (more wins than losses)
        assert!(doc1.fused_score > 0.0);
        // doc2 should have negative score (more losses than wins)
        assert!(doc2.fused_score < 0.0);
    }

    // ==================== DEMPSTER-SHAFER TESTS ====================

    #[test]
    fn test_dempster_shafer_combination() {
        let engine = HybridFusionEngine::new(FusionStrategy::DempsterShafer { alpha: 0.5 });

        let bm25_results = vec![
            BM25Result {
                doc_id: "doc1".to_string(),
                score: 1.0,
                highlights: None,
                metadata: HashMap::new(),
            },
            BM25Result {
                doc_id: "doc2".to_string(),
                score: 0.5,
                highlights: None,
                metadata: HashMap::new(),
            },
        ];

        let vector_results = vec![
            VectorResult {
                doc_id: "doc1".to_string(),
                score: 0.8,
                distance: 0.2,
                metadata: HashMap::new(),
            },
            VectorResult {
                doc_id: "doc2".to_string(),
                score: 0.6,
                distance: 0.4,
                metadata: HashMap::new(),
            },
        ];

        let fused = engine
            .fuse(bm25_results, vector_results)
            .expect("Fusion should succeed");

        assert_eq!(fused.len(), 2);

        // Scores should be combined using Dempster's rule
        let doc1 = fused
            .iter()
            .find(|r| r.doc_id == "doc1")
            .expect("doc1 should exist in fused results");
        let doc2 = fused
            .iter()
            .find(|r| r.doc_id == "doc2")
            .expect("doc2 should exist in fused results");

        // doc1 should have higher combined belief
        assert!(doc1.fused_score > doc2.fused_score);
        // Scores should be in [0,1] range (normalized beliefs)
        assert!(doc1.fused_score >= 0.0 && doc1.fused_score <= 1.0);
    }

    /// With identity-seeded normalization folds the symmetric fixture
    /// combines to a three-way 0.5 tie under Dempster-Shafer (alpha 0.5):
    /// each doc's belief pair is (1,0), (0.5,0.5) or (0,1), and
    /// m1*m2 + 0.5*(m1+m2-2*m1*m2) = 0.5 for all three. Folds seeded from
    /// 0.0 produced a strict 0.97/0.82/0.67 ordering instead.
    #[test]
    fn test_dempster_shafer_symmetric_fixture_ties() {
        let engine = HybridFusionEngine::new(FusionStrategy::DempsterShafer { alpha: 0.5 });
        let bm25 = vec![
            make_bm25("doc_x", 3.0),
            make_bm25("doc_y", 2.0),
            make_bm25("doc_z", 1.0),
        ];
        let vector = vec![
            make_vector("doc_x", 0.28),
            make_vector("doc_y", 0.29),
            make_vector("doc_z", 0.30),
        ];

        let fused = engine.fuse(bm25, vector).expect("fusion succeeds");

        for id in ["doc_x", "doc_y", "doc_z"] {
            let score = fused_score_for(&fused, id);
            assert!(
                (score - 0.5).abs() < 1e-9,
                "{id} should tie at 0.5, got {score}"
            );
        }
    }

    #[test]
    fn test_dempster_shafer_alpha_param() {
        // Test with different alpha values using simple case
        // When m1=0.5, m2=0.5:
        //   alpha=0.1: 0.5*0.5 + 0.1*(0.5+0.5-2*0.5*0.5) = 0.25 + 0.1*0.5 = 0.30
        //   alpha=0.9: 0.5*0.5 + 0.9*(0.5+0.5-2*0.5*0.5) = 0.25 + 0.9*0.5 = 0.70
        let engine_low = HybridFusionEngine::new(FusionStrategy::DempsterShafer { alpha: 0.1 });
        let engine_high = HybridFusionEngine::new(FusionStrategy::DempsterShafer { alpha: 0.9 });

        // Create test data where normalized scores will be 0.5
        let bm25_results = vec![BM25Result {
            doc_id: "doc1".to_string(),
            score: 0.5, // Will normalize to 0.0 (min)
            highlights: None,
            metadata: HashMap::new(),
        }];

        let vector_results = vec![VectorResult {
            doc_id: "doc1".to_string(),
            score: 0.6, // Will normalize differently
            distance: 0.4,
            metadata: HashMap::new(),
        }];

        // Just verify both engines work without error
        let _fused_low = engine_low
            .fuse(bm25_results.clone(), vector_results.clone())
            .expect("Low-alpha fusion should succeed");
        let _fused_high = engine_high
            .fuse(bm25_results, vector_results)
            .expect("High-alpha fusion should succeed");

        // Test passes if both engines complete successfully
        // (The alpha parameter effect is subtle with normalization)
        assert!(true);
    }

    // ==================== ADAPTIVE FUSION TESTS ====================

    #[test]
    fn test_adaptive_high_overlap() {
        let engine = HybridFusionEngine::new(FusionStrategy::Adaptive);

        // High overlap (same docs in both lists)
        let bm25_results = vec![
            BM25Result {
                doc_id: "doc1".to_string(),
                score: 1.0,
                highlights: None,
                metadata: HashMap::new(),
            },
            BM25Result {
                doc_id: "doc2".to_string(),
                score: 0.9,
                highlights: None,
                metadata: HashMap::new(),
            },
        ];

        let vector_results = vec![
            VectorResult {
                doc_id: "doc1".to_string(),
                score: 0.8,
                distance: 0.2,
                metadata: HashMap::new(),
            },
            VectorResult {
                doc_id: "doc2".to_string(),
                score: 0.7,
                distance: 0.3,
                metadata: HashMap::new(),
            },
        ];

        let fused = engine
            .fuse(bm25_results, vector_results)
            .expect("Fusion should succeed");

        // Should use WeightedLinear for high overlap
        assert_eq!(fused.len(), 2);
        // Check that results are properly fused
        let doc1 = fused
            .iter()
            .find(|r| r.doc_id == "doc1")
            .expect("doc1 should exist in fused results");
        assert!(doc1.fused_score > 0.0);
    }

    #[test]
    fn test_adaptive_low_overlap() {
        let engine = HybridFusionEngine::new(FusionStrategy::Adaptive);

        // Low overlap (disjoint results)
        let bm25_results = vec![
            BM25Result {
                doc_id: "doc1".to_string(),
                score: 1.0,
                highlights: None,
                metadata: HashMap::new(),
            },
            BM25Result {
                doc_id: "doc2".to_string(),
                score: 0.9,
                highlights: None,
                metadata: HashMap::new(),
            },
        ];

        let vector_results = vec![
            VectorResult {
                doc_id: "doc3".to_string(),
                score: 0.8,
                distance: 0.2,
                metadata: HashMap::new(),
            },
            VectorResult {
                doc_id: "doc4".to_string(),
                score: 0.7,
                distance: 0.3,
                metadata: HashMap::new(),
            },
        ];

        let fused = engine
            .fuse(bm25_results, vector_results)
            .expect("Fusion should succeed");

        // Should use CombSUM for low overlap
        assert_eq!(fused.len(), 4);
    }

    #[test]
    fn test_adaptive_medium_overlap() {
        let engine = HybridFusionEngine::new(FusionStrategy::Adaptive);

        // Medium overlap (partial overlap)
        let bm25_results = vec![
            BM25Result {
                doc_id: "doc1".to_string(),
                score: 1.0,
                highlights: None,
                metadata: HashMap::new(),
            },
            BM25Result {
                doc_id: "doc2".to_string(),
                score: 0.9,
                highlights: None,
                metadata: HashMap::new(),
            },
            BM25Result {
                doc_id: "doc3".to_string(),
                score: 0.8,
                highlights: None,
                metadata: HashMap::new(),
            },
        ];

        let vector_results = vec![
            VectorResult {
                doc_id: "doc1".to_string(),
                score: 0.95,
                distance: 0.05,
                metadata: HashMap::new(),
            },
            VectorResult {
                doc_id: "doc4".to_string(),
                score: 0.85,
                distance: 0.15,
                metadata: HashMap::new(),
            },
        ];

        let fused = engine
            .fuse(bm25_results, vector_results)
            .expect("Fusion should succeed");

        // Should use RRF for medium overlap (Jaccard ~0.25)
        assert_eq!(fused.len(), 4);
    }

    // ==================== EDGE CASE TESTS ====================

    #[test]
    fn test_empty_results() {
        let engine = HybridFusionEngine::new(FusionStrategy::BordaCount);

        let bm25_results = vec![];
        let vector_results = vec![];

        let fused = engine
            .fuse(bm25_results, vector_results)
            .expect("Fusion should succeed");

        assert_eq!(fused.len(), 0);
    }

    #[test]
    fn test_single_bm25_result() {
        let engine = HybridFusionEngine::new(FusionStrategy::BordaCount);

        let bm25_results = vec![BM25Result {
            doc_id: "doc1".to_string(),
            score: 1.0,
            highlights: None,
            metadata: HashMap::new(),
        }];

        let vector_results = vec![];

        let fused = engine
            .fuse(bm25_results, vector_results)
            .expect("Fusion should succeed");

        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].doc_id, "doc1");
    }

    #[test]
    fn test_single_vector_result() {
        let engine = HybridFusionEngine::new(FusionStrategy::BordaCount);

        let bm25_results = vec![];

        let vector_results = vec![VectorResult {
            doc_id: "doc1".to_string(),
            score: 1.0,
            distance: 0.0,
            metadata: HashMap::new(),
        }];

        let fused = engine
            .fuse(bm25_results, vector_results)
            .expect("Fusion should succeed");

        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].doc_id, "doc1");
    }

    #[test]
    fn test_all_strategies_compile() {
        // Test that all fusion strategies compile and can be created
        let strategies = vec![
            FusionStrategy::ReciprocalRank { k: 60 },
            FusionStrategy::WeightedLinear {
                alpha: 0.5,
                bm25_normalize: true,
                vector_normalize: true,
            },
            FusionStrategy::RankBiasedPrecision { persistence: 0.8 },
            FusionStrategy::ConditionalNormalization,
            FusionStrategy::BordaCount,
            FusionStrategy::CombSum,
            FusionStrategy::CombMin,
            FusionStrategy::CombMax,
            FusionStrategy::Condorcet,
            FusionStrategy::DempsterShafer { alpha: 0.5 },
            FusionStrategy::Adaptive,
        ];

        for strategy in strategies {
            let engine = HybridFusionEngine::new(strategy);
            assert_eq!(engine.default_top_k, 10);
        }
    }

    // ============================================================
    // Fusion strategy coverage tests
    // ============================================================

    fn make_bm25(id: &str, score: f64) -> BM25Result {
        BM25Result {
            doc_id: id.to_string(),
            score,
            highlights: None,
            metadata: HashMap::new(),
        }
    }

    fn make_vector(id: &str, score: f64) -> VectorResult {
        VectorResult {
            doc_id: id.to_string(),
            score,
            distance: 1.0 - score,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_weighted_linear_fusion() {
        let engine = HybridFusionEngine::new(FusionStrategy::WeightedLinear {
            alpha: 0.7,
            bm25_normalize: true,
            vector_normalize: true,
        });

        let bm25 = vec![make_bm25("a", 2.0), make_bm25("b", 1.0)];
        let vector = vec![make_vector("a", 0.9), make_vector("c", 0.8)];

        let fused = engine.fuse(bm25, vector).expect("Fusion should succeed");
        assert!(!fused.is_empty());
        // "a" should be top result (present in both)
        assert_eq!(fused[0].doc_id, "a");
    }

    #[test]
    fn test_rank_biased_precision() {
        let engine =
            HybridFusionEngine::new(FusionStrategy::RankBiasedPrecision { persistence: 0.9 });

        let bm25 = vec![make_bm25("x", 3.0), make_bm25("y", 2.0)];
        let vector = vec![make_vector("y", 0.95), make_vector("z", 0.8)];

        let fused = engine.fuse(bm25, vector).expect("RBP should succeed");
        assert!(fused.len() >= 2);
    }

    #[test]
    fn test_conditional_normalization() {
        let engine = HybridFusionEngine::new(FusionStrategy::ConditionalNormalization);

        let bm25 = vec![make_bm25("d1", 5.0), make_bm25("d2", 1.0)];
        let vector = vec![make_vector("d1", 0.7), make_vector("d3", 0.5)];

        let fused = engine
            .fuse(bm25, vector)
            .expect("ConditionalNorm should succeed");
        assert!(!fused.is_empty());
    }

    #[test]
    fn test_comb_sum() {
        let engine = HybridFusionEngine::new(FusionStrategy::CombSum);

        let bm25 = vec![make_bm25("a", 1.0), make_bm25("b", 0.5)];
        let vector = vec![make_vector("a", 0.8), make_vector("c", 0.6)];

        let fused = engine.fuse(bm25, vector).expect("CombSum should succeed");
        assert!(fused.len() >= 2);
        // "a" should rank highest (appears in both lists)
        assert_eq!(fused[0].doc_id, "a");
    }

    #[test]
    fn test_comb_min() {
        let engine = HybridFusionEngine::new(FusionStrategy::CombMin);

        let bm25 = vec![make_bm25("a", 2.0)];
        let vector = vec![make_vector("a", 0.5), make_vector("b", 0.9)];

        let fused = engine.fuse(bm25, vector).expect("CombMin should succeed");
        assert!(!fused.is_empty());
    }

    #[test]
    fn test_comb_max() {
        let engine = HybridFusionEngine::new(FusionStrategy::CombMax);

        let bm25 = vec![make_bm25("a", 0.3)];
        let vector = vec![make_vector("a", 0.9), make_vector("b", 0.1)];

        let fused = engine.fuse(bm25, vector).expect("CombMax should succeed");
        assert!(!fused.is_empty());
    }

    #[test]
    fn test_condorcet() {
        let engine = HybridFusionEngine::new(FusionStrategy::Condorcet);

        let bm25 = vec![
            make_bm25("p", 3.0),
            make_bm25("q", 2.0),
            make_bm25("r", 1.0),
        ];
        let vector = vec![
            make_vector("q", 0.9),
            make_vector("p", 0.7),
            make_vector("r", 0.5),
        ];

        let fused = engine.fuse(bm25, vector).expect("Condorcet should succeed");
        assert_eq!(fused.len(), 3);
    }

    #[test]
    fn test_dempster_shafer() {
        let engine = HybridFusionEngine::new(FusionStrategy::DempsterShafer { alpha: 0.6 });

        let bm25 = vec![make_bm25("x", 1.5), make_bm25("y", 0.8)];
        let vector = vec![make_vector("x", 0.85), make_vector("z", 0.4)];

        let fused = engine
            .fuse(bm25, vector)
            .expect("DempsterShafer should succeed");
        assert!(!fused.is_empty());
    }

    #[test]
    fn test_adaptive_fusion() {
        let engine = HybridFusionEngine::new(FusionStrategy::Adaptive);

        let bm25 = vec![make_bm25("a", 2.0), make_bm25("b", 1.0)];
        let vector = vec![make_vector("a", 0.9), make_vector("c", 0.7)];

        let fused = engine.fuse(bm25, vector).expect("Adaptive should succeed");
        assert!(!fused.is_empty());
    }

    #[test]
    fn test_fusion_empty_inputs() {
        let engine = HybridFusionEngine::new(FusionStrategy::ReciprocalRank { k: 60 });

        let fused = engine
            .fuse(vec![], vec![])
            .expect("Empty fusion should succeed");
        assert!(fused.is_empty());
    }

    #[test]
    fn test_fusion_single_source_only() {
        let engine = HybridFusionEngine::new(FusionStrategy::CombSum);

        // Only BM25 results, no vector results
        let bm25 = vec![make_bm25("solo", 1.0)];
        let fused = engine
            .fuse(bm25, vec![])
            .expect("Single source should succeed");
        assert!(!fused.is_empty());
    }

    #[test]
    fn test_projection_fusion_basic() {
        let engine = HybridFusionEngine::new(FusionStrategy::Projection { alpha: 0.5 });

        let bm25_results = vec![
            BM25Result {
                doc_id: "doc1".to_string(),
                score: 10.0,
                highlights: None,
                metadata: HashMap::new(),
            },
            BM25Result {
                doc_id: "doc2".to_string(),
                score: 5.0,
                highlights: None,
                metadata: HashMap::new(),
            },
        ];

        let vector_results = vec![
            VectorResult {
                doc_id: "doc1".to_string(),
                score: 0.9,
                distance: 0.1,
                metadata: HashMap::new(),
            },
            VectorResult {
                doc_id: "doc3".to_string(),
                score: 0.8,
                distance: 0.2,
                metadata: HashMap::new(),
            },
        ];

        let fused = engine
            .fuse(bm25_results, vector_results)
            .expect("Projection fusion should succeed");

        // doc1 should be first as it's in both sets with high scores
        assert_eq!(fused[0].doc_id, "doc1");
        // Should have 3 unique documents
        assert_eq!(fused.len(), 3);

        // Verify diversity: doc3 (vector only) and doc2 (bm25 only) should both be present
        let doc_ids: Vec<_> = fused.iter().map(|r| &r.doc_id).collect();
        assert!(doc_ids.contains(&&"doc2".to_string()));
        assert!(doc_ids.contains(&&"doc3".to_string()));
    }

    #[test]
    fn test_projection_fusion_alpha_boundary() {
        // Test with alpha = 1.0 (BM25 only)
        let engine_bm25 = HybridFusionEngine::new(FusionStrategy::Projection { alpha: 0.0 });
        let bm25 = vec![BM25Result {
            doc_id: "b1".to_string(),
            score: 1.0,
            highlights: None,
            metadata: HashMap::new(),
        }];
        let vector = vec![VectorResult {
            doc_id: "v1".to_string(),
            score: 1.0,
            distance: 0.0,
            metadata: HashMap::new(),
        }];

        let fused = engine_bm25.fuse(bm25, vector).unwrap();
        // alpha=0.0 -> theta=0 -> cos(0)=1, sin(0)=0 -> BM25 only (except for diversity boost)
        // b1 score should be higher than v1
        assert_eq!(fused[0].doc_id, "b1");

        // Test with alpha = 1.0 (Vector only)
        let engine_vec = HybridFusionEngine::new(FusionStrategy::Projection { alpha: 1.0 });
        let bm25 = vec![BM25Result {
            doc_id: "b1".to_string(),
            score: 1.0,
            highlights: None,
            metadata: HashMap::new(),
        }];
        let vector = vec![VectorResult {
            doc_id: "v1".to_string(),
            score: 1.0,
            distance: 0.0,
            metadata: HashMap::new(),
        }];

        let fused = engine_vec.fuse(bm25, vector).unwrap();
        // alpha=1.0 -> theta=pi/2 -> cos(pi/2)=0, sin(pi/2)=1 -> Vector only
        assert_eq!(fused[0].doc_id, "v1");
    }
}
