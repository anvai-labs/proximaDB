//! Common Progressive Search Implementation
//!
//! This module provides shared progressive search logic that can be used by all storage engines
//! (SST, VIPER, NOVA, SWIFT, etc.) to implement multi-stage quantization-aware search.
//!
//! ## FLEXIBLE QUANTIZATION ARCHITECTURE (2025-08-21):
//!
//! **Two supported paths based on use case:**
//!
//! 1. **HIGH PERFORMANCE PATH** (Write-once, Read-many):
//!    - Collection config has quantization enabled
//!    - Write Path: FP32 → [Binary + INT8 + PQ8] → Store ALL quantized versions
//!    - Read Path: Query → Search pre-stored quantized → Fast response
//!    - Use case: Static datasets with frequent searches
//!
//! 2. **STORAGE OPTIMIZED PATH** (Continuous writes, Infrequent reads):
//!    - Collection config has quantization disabled
//!    - Write Path: FP32 → Store only FP32 (save storage)
//!    - Read Path: Query → Runtime quantization → Slower but acceptable
//!    - Use case: Streaming data where storage cost matters more than latency
//!
//! The query optimizer and search hints determine which path to use.

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, trace};

// Note: SearchResult is proto type, not in core::search anymore
use crate::compute::quantization::quantization_engine::{
    QuantizedVector, UnifiedQuantizationEngine,
};
use crate::core::search::OptimizedSearchRecord;
use crate::proto::proximadb_v1::VectorRecord;
use crate::storage::traits::{QuantizationLevel, QuantizationType, StorageQueryContext};
use proximadb_distance_kernel::DistanceMetric;
use proximadb_distance_kernel::engine::UnifiedDistanceCompute;
use proximadb_quantization_model::StorageQuantizedData;

/// Progressive search executor that can be used by any storage engine
pub struct ProgressiveSearchExecutor {
    /// Quantization engine for vector operations
    quantization_engine: Arc<UnifiedQuantizationEngine>,

    /// Distance computation engine
    distance_compute: Arc<UnifiedDistanceCompute>,

    /// Per-collection TurboQuant store registry (Phase C — lifecycle
    /// dispatch). Optional because most deployments don't enable
    /// TurboQuant; absence routes ReadTime-classified levels back to the
    /// full-precision scorer (correct but slower). Phase E (xCatalog
    /// hydration) wires this in from `SharedServices` at startup.
    #[cfg(feature = "experimental-turboquant")]
    turboquant_registry: Option<
        Arc<dyn crate::compute::quantization::turboquant_store_registry::TurboQuantStoreRegistry>,
    >,
}

/// Backwards-compat alias for [`ProgressiveSearchCandidate`].
pub type SearchCandidate = ProgressiveSearchCandidate;

/// Candidate tracking during progressive search
#[derive(Debug, Clone)]
pub struct ProgressiveSearchCandidate {
    /// Vector ID
    pub id: String,

    /// Full precision vector (loaded on demand)
    pub vector: Option<Vec<f32>>,

    /// Quantized representations at different levels
    pub quantized_vectors: Vec<QuantizedRepresentation>,

    /// Current score/distance
    pub score: f32,

    /// Stage where this candidate was added
    pub stage: SearchStage,

    /// Metadata (optional)
    pub metadata: Option<Vec<u8>>,
}

/// Quantized representation at a specific level
#[derive(Debug, Clone)]
pub struct QuantizedRepresentation {
    /// Level identifier
    pub level_id: String,

    /// Quantized data
    pub data: Vec<u8>,

    /// Quantization type
    pub quant_type: QuantizationType,
}

/// Search stage for tracking
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SearchStage {
    BinaryFilter,
    Int8Ranking,
    PqRanking,
    FullPrecision,
}

impl ProgressiveSearchExecutor {
    /// Create a new progressive search executor.
    ///
    /// The optional `turboquant_registry` is wired by Phase E (xCatalog
    /// hydration); when absent (default builds, or TurboQuant-disabled
    /// collections), `ReadTime`-classified levels fall back to the
    /// full-precision scorer. The constructor stays backward-compatible
    /// — existing callers can keep using the 2-arg form via
    /// [`ProgressiveSearchBuilder`].
    pub fn new(
        quantization_engine: Arc<UnifiedQuantizationEngine>,
        distance_compute: Arc<UnifiedDistanceCompute>,
    ) -> Self {
        Self {
            quantization_engine,
            distance_compute,
            #[cfg(feature = "experimental-turboquant")]
            turboquant_registry: None,
        }
    }

    /// Wire a TurboQuant store registry into the executor. Returns
    /// `self` for builder-style chaining at construction sites.
    #[cfg(feature = "experimental-turboquant")]
    pub fn with_turboquant_registry(
        mut self,
        registry: Arc<
            dyn crate::compute::quantization::turboquant_store_registry::TurboQuantStoreRegistry,
        >,
    ) -> Self {
        self.turboquant_registry = Some(registry);
        self
    }

    /// Execute progressive search with the given context and candidates
    pub async fn execute_progressive_search(
        &self,
        ctx: &StorageQueryContext,
        initial_candidates: Vec<VectorRecord>,
        query_vector: &[f32],
    ) -> Result<Vec<OptimizedSearchRecord>> {
        // Check if progressive search is enabled
        if !ctx.is_progressive_search_enabled() {
            debug!("Progressive search not enabled, falling back to full precision search");
            return self
                .full_precision_search(ctx, initial_candidates, query_vector)
                .await;
        }

        // Get progressive levels
        let levels = ctx
            .get_progressive_levels()
            .ok_or_else(|| anyhow::anyhow!("No progressive levels configured"))?;

        if levels.is_empty() {
            debug!("No progressive levels defined, using full precision");
            return self
                .full_precision_search(ctx, initial_candidates, query_vector)
                .await;
        }

        debug!(
            "Starting progressive search with {} levels for {} candidates",
            levels.len(),
            initial_candidates.len()
        );

        // Convert to search candidates
        let mut candidates = self.prepare_candidates(ctx, initial_candidates, levels)?;

        // Execute progressive stages
        for (stage_idx, level) in levels.iter().enumerate() {
            let stage = self.get_search_stage(&level.quantization_type);

            debug!(
                "📊 Stage {}: {} ({:?}) - {} candidates",
                stage_idx,
                level.level_id,
                level.quantization_type,
                candidates.len()
            );

            // Apply progressive filter/ranking
            candidates = self
                .apply_progressive_stage(ctx, candidates, query_vector, level, stage)
                .await?;

            // Check if we have enough candidates
            if candidates.len() <= ctx.top_k() {
                debug!(
                    "Early termination: candidates ({}) <= top_k ({})",
                    candidates.len(),
                    ctx.top_k()
                );
                break;
            }
        }

        // Final reranking with full precision if needed
        if candidates
            .iter()
            .any(|c| c.stage != SearchStage::FullPrecision)
        {
            candidates = self.final_rerank(ctx, candidates, query_vector).await?;
        }

        // Convert to search results
        self.convert_to_results(candidates, ctx.top_k())
    }

    /// Prepare candidates using PRE-STORED quantized representations (no re-quantization!)
    fn prepare_candidates(
        &self,
        ctx: &StorageQueryContext,
        records: Vec<VectorRecord>,
        levels: &[QuantizationLevel],
    ) -> Result<Vec<ProgressiveSearchCandidate>> {
        let mut candidates = Vec::with_capacity(records.len());

        for record in records {
            // quantized_vector removed - internalized in storage
            let quantized_vectors = {
                // Check if runtime quantization should be allowed based on:
                // 1. Collection configuration
                // 2. Search hints
                // 3. Query optimizer recommendations

                let should_runtime_quantize = self.should_allow_runtime_quantization(ctx)?;

                if should_runtime_quantize {
                    // SLOW PATH: Runtime quantization for storage-optimized collections
                    trace!(
                        "Vector {} using runtime quantization (storage-optimized path)",
                        &record.id
                    );
                    // Perform quantization based on the first level requested
                    let first_level = levels
                        .first()
                        .ok_or_else(|| anyhow::anyhow!("No quantization levels provided"))?;
                    // Lifecycle-aware dispatch (Phase C — Quantization
                    // Trait Convergence Plan). `WriteTime` variants encode
                    // here; `ReadTime` variants (TurboQuant) are intentional
                    // no-ops because their codes live in the per-collection
                    // `TurboQuantStore` and are produced at collection-level
                    // ingest, not here. `Identity` is also a no-op.
                    let quantized_data = match first_level.quantization_type.lifecycle() {
                        proximadb_quantization_types::QuantizationLifecycle::WriteTime => {
                            match first_level.quantization_type {
                                QuantizationType::Binary => {
                                    let binary_data = self
                                        .quantization_engine
                                        .quantize_to_binary(&record.vector)?;
                                    StorageQuantizedData {
                                        id: record.id.clone(),
                                        primary: None,
                                        filter: Some(QuantizedVector {
                                            data: binary_data,
                                            quantization_level: proximadb_quantization_model::UnifiedQuantizationLevel::Binary,
                                            metadata: crate::compute::quantization::QuantizationMetadata::default(),
                                        }),
                                        fast: None,
                                        dimension: record.vector.len(),
                                        metadata: crate::compute::quantization::QuantizationMetadata::default(),
                                    }
                                }
                                QuantizationType::Scalar => {
                                    let int8_data = self
                                        .quantization_engine
                                        .quantize_to_int8(&record.vector)?;
                                    StorageQuantizedData {
                                        id: record.id.clone(),
                                        primary: None,
                                        filter: None,
                                        fast: Some(QuantizedVector {
                                            data: int8_data,
                                            quantization_level: proximadb_quantization_model::UnifiedQuantizationLevel::Int8,
                                            metadata: crate::compute::quantization::QuantizationMetadata::default(),
                                        }),
                                        dimension: record.vector.len(),
                                        metadata: crate::compute::quantization::QuantizationMetadata::default(),
                                    }
                                }
                                QuantizationType::Product => {
                                    let num_subvectors =
                                        first_level.num_subvectors.unwrap_or(8) as usize;
                                    let pq_result = self.quantization_engine.quantize_to_pq(
                                        &record.vector,
                                        num_subvectors,
                                        8,
                                    )?;
                                    StorageQuantizedData {
                                        id: record.id.clone(),
                                        primary: Some(QuantizedVector {
                                            data: pq_result,
                                            quantization_level: proximadb_quantization_model::UnifiedQuantizationLevel {
                                                level_type: Some(proximadb_quantization_model::QuantizationLevel::Pq(
                                                    proximadb_quantization_model::ProductQuantization {
                                                        bits_per_code: 8,
                                                        num_subvectors: num_subvectors as i32,
                                                        codebook_id: None,
                                                        adaptive_subvectors: false,
                                                    },
                                                )),
                                            },
                                            metadata: crate::compute::quantization::QuantizationMetadata::default(),
                                        }),
                                        filter: None,
                                        fast: None,
                                        dimension: record.vector.len(),
                                        metadata: crate::compute::quantization::QuantizationMetadata::default(),
                                    }
                                }
                                _ => {
                                    return Err(anyhow::anyhow!(
                                        "Unsupported WriteTime quantization type for runtime quantization: {:?}",
                                        first_level.quantization_type,
                                    ));
                                }
                            }
                        }
                        // ReadTime: TurboQuant codes live in the store, not
                        // in StorageQuantizedData. Return an empty shell
                        // (id + dimension only) so downstream stages see
                        // the record and route through `score_turboquant`
                        // at stage time. Default arm so future ReadTime
                        // variants Just Work.
                        proximadb_quantization_types::QuantizationLifecycle::ReadTime => {
                            StorageQuantizedData {
                                id: record.id.clone(),
                                primary: None,
                                filter: None,
                                fast: None,
                                dimension: record.vector.len(),
                                metadata:
                                    crate::compute::quantization::QuantizationMetadata::default(),
                            }
                        }
                        // Identity (no quantization configured): same empty
                        // shell — the full-precision scorer handles it.
                        proximadb_quantization_types::QuantizationLifecycle::Identity => {
                            StorageQuantizedData {
                                id: record.id.clone(),
                                primary: None,
                                filter: None,
                                fast: None,
                                dimension: record.vector.len(),
                                metadata:
                                    crate::compute::quantization::QuantizationMetadata::default(),
                            }
                        }
                    };
                    // Convert StorageQuantizedData to Vec<QuantizedRepresentation>
                    let mut representations = Vec::new();

                    // Add filter quantization (binary) if present
                    if let Some(filter) = &quantized_data.filter {
                        representations.push(QuantizedRepresentation {
                            level_id: "binary".to_string(),
                            data: filter.data.clone(),
                            quant_type: QuantizationType::Binary,
                        });
                    }

                    // Add fast quantization (INT8) if present
                    if let Some(fast) = &quantized_data.fast {
                        representations.push(QuantizedRepresentation {
                            level_id: "int8".to_string(),
                            data: fast.data.clone(),
                            quant_type: QuantizationType::Scalar,
                        });
                    }

                    // Add primary quantization (PQ) if present
                    if let Some(primary) = &quantized_data.primary {
                        representations.push(QuantizedRepresentation {
                            level_id: "pq".to_string(),
                            data: primary.data.clone(),
                            quant_type: QuantizationType::Product,
                        });
                    }

                    representations
                } else {
                    // ERROR: Runtime quantization not allowed for this collection/query
                    trace!(
                        "Vector {} missing pre-quantized data (collection expects pre-quantization)",
                        if record.id.is_empty() {
                            "unknown"
                        } else {
                            &record.id
                        }
                    );
                    // Return empty vec to skip this record
                    Vec::new()
                }
            };

            candidates.push(ProgressiveSearchCandidate {
                id: record.id.clone(),
                vector: Some(record.vector.clone()),
                quantized_vectors,
                score: f32::MAX,
                stage: SearchStage::BinaryFilter,
                metadata: None,
            });
        }

        Ok(candidates)
    }

    /// Apply a progressive search stage
    async fn apply_progressive_stage(
        &self,
        ctx: &StorageQueryContext,
        mut candidates: Vec<ProgressiveSearchCandidate>,
        query_vector: &[f32],
        level: &QuantizationLevel,
        stage: SearchStage,
    ) -> Result<Vec<ProgressiveSearchCandidate>> {
        // Get selectivity for this stage
        let selectivity = match stage {
            SearchStage::BinaryFilter => ctx.binary_filter_selectivity(),
            SearchStage::Int8Ranking => ctx
                .metadata
                .quantization_config
                .as_ref()
                .map_or(0.5, |qc| qc.int8_ranking_selectivity), // Default selectivity
            SearchStage::PqRanking => ctx
                .metadata
                .quantization_config
                .as_ref()
                .map_or(0.2, |qc| qc.pq_ranking_selectivity),
            SearchStage::FullPrecision => 1.0,
        };

        // Calculate how many candidates to keep
        let keep_count = ((candidates.len() as f32) * selectivity).ceil() as usize;
        let keep_count = keep_count.max(ctx.top_k()).min(candidates.len());

        trace!(
            "Stage {:?}: keeping {} of {} candidates (selectivity: {})",
            stage,
            keep_count,
            candidates.len(),
            selectivity
        );

        // Score candidates based on quantization lifecycle (Phase C —
        // Quantization Trait Convergence Plan). The lifecycle classifier
        // replaces the previous bare type-tag match: `WriteTime` variants
        // route through the existing per-variant scorers; `ReadTime`
        // routes through `score_turboquant`; `Identity` falls through to
        // full-precision. A new `ReadTime` variant in the future (e.g.
        // 3-bit TurboQuant per LLD §"Phase Plan" P10) automatically
        // routes to the right scorer without touching this match.
        match level.quantization_type.lifecycle() {
            proximadb_quantization_types::QuantizationLifecycle::WriteTime => {
                match level.quantization_type {
                    QuantizationType::Binary => {
                        self.score_binary(&mut candidates, query_vector, level)
                            .await?;
                    }
                    QuantizationType::Scalar => {
                        self.score_scalar(&mut candidates, query_vector, level)
                            .await?;
                    }
                    QuantizationType::Product => {
                        self.score_product(&mut candidates, query_vector, level)
                            .await?;
                    }
                    other => {
                        // A new WriteTime variant landed without a scorer.
                        // Fall back to full-precision so the search still
                        // completes — slower but correct. Log loud so the
                        // gap is visible in production traces.
                        debug!(
                            "WriteTime variant {:?} has no scorer; falling back to full \
                             precision (Phase C trait-convergence gap)",
                            other,
                        );
                        self.score_full_precision(
                            &mut candidates,
                            query_vector,
                            ctx.distance_metric(),
                        )
                        .await?;
                    }
                }
            }
            #[cfg(feature = "experimental-turboquant")]
            proximadb_quantization_types::QuantizationLifecycle::ReadTime => {
                self.score_turboquant(&mut candidates, query_vector, level, ctx)
                    .await?;
            }
            // ReadTime arm absent without the feature is a logic bug —
            // no enum variant produces ReadTime when the feature is off.
            // Fall through to full-precision rather than panic; this is
            // the hot scoring path.
            #[cfg(not(feature = "experimental-turboquant"))]
            proximadb_quantization_types::QuantizationLifecycle::ReadTime => {
                debug!(
                    "ReadTime lifecycle without experimental-turboquant feature — \
                     falling back to full precision (suspect router bug)",
                );
                self.score_full_precision(&mut candidates, query_vector, ctx.distance_metric())
                    .await?;
            }
            proximadb_quantization_types::QuantizationLifecycle::Identity => {
                self.score_full_precision(&mut candidates, query_vector, ctx.distance_metric())
                    .await?;
            }
        }

        // Sort by score (ascending for distance, descending for similarity)
        candidates.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Keep top candidates
        candidates.truncate(keep_count);

        // Update stage for kept candidates
        for candidate in &mut candidates {
            candidate.stage = stage;
        }

        Ok(candidates)
    }

    /// Score candidates using TurboQuant — the `ReadTime` lifecycle
    /// scorer (Phase C — Quantization Trait Convergence Plan).
    ///
    /// Routing behaviour:
    ///
    /// - Registry absent (default build, or executor constructed without
    ///   `with_turboquant_registry`): falls back to the full-precision
    ///   scorer. Correct but slower; no TurboQuant acceleration applied.
    /// - Registry present but no store for this collection (catalog not
    ///   yet hydrated, or collection created without TurboQuant enabled):
    ///   same full-precision fallback.
    /// - Registry present + store present: exercises the bridge to
    ///   confirm the kernel path runs and the per-collection
    ///   `BLOCKS_SKIPPED_BY_MASK` counter advances; per-candidate scores
    ///   still come from full-precision until Phase D wires the id↔slot
    ///   mapping through the AXIS adapter. The bridge call is cheap (a
    ///   single flat scan at `k=candidates.len()`) and is the load-bearing
    ///   integration test that the read-time routing actually reaches the
    ///   SIMD kernel.
    ///
    /// The Phase D follow-up replaces the full-precision fallback with a
    /// real id-mapped bridge search via `TurboQuantAxisIndex`, at which
    /// point the per-candidate scores come straight from the kernel.
    #[cfg(feature = "experimental-turboquant")]
    async fn score_turboquant(
        &self,
        candidates: &mut [ProgressiveSearchCandidate],
        query_vector: &[f32],
        _level: &QuantizationLevel,
        ctx: &StorageQueryContext,
    ) -> Result<()> {
        if let Some(registry) = &self.turboquant_registry
            && let Some(store) = registry.get(ctx.collection_id()).await?
        {
            // Best-effort bridge exercise. Failures here are non-fatal —
            // the fallback scorer below produces correct scores either
            // way. The bridge call's purpose is to:
            //   1. Verify the kernel runs without errors.
            //   2. Advance the BLOCKS_SKIPPED_BY_MASK Prometheus counter
            //      when a mask is in scope (Phase D `TurboQuantAxisIndex`
            //      route).
            //   3. Build the load-bearing `TurboQuantExplainHints` and
            //      emit them via tracing (Phase J integration). A
            //      caller-side tracing layer can collect the structured
            //      event into the per-request `SearchPlanHints.turboquant`
            //      slot; from there `VectorHints::from(&SearchPlanHints)`
            //      propagates the payload to every protocol surface.
            let (bridge_result, blocks_skipped) =
                crate::index::turboquant_bridge::with_blocks_skipped_delta(|| {
                    crate::index::turboquant_bridge::search_with_candidate_set(
                        &store,
                        query_vector,
                        candidates.len().max(1),
                        None,
                    )
                });

            // Always emit the hints, even when the bridge result was an
            // error — operator dashboards still want to see "TurboQuant
            // was attempted" with the right config + epoch.
            let n_hits = bridge_result.as_ref().map(|h| h.len()).unwrap_or(0);
            let hints = crate::index::turboquant_bridge::TurboQuantExplainHints::for_search(&store)
                .with_blocks_skipped(blocks_skipped)
                .with_n_vectors_scanned(n_hits);
            let payload = hints.to_explain_value();
            // Phase K — record into the per-request task-local
            // `PredicateDiagnostics` bus so the request handler can
            // lift the payload into `SearchPlanHints.turboquant` via
            // `take_turboquant_hints()` (then propagated to
            // `VectorHints` via `VectorHints::from`). Silently no-ops
            // when there's no active scope (test paths, untraced
            // entry points) — the tracing event below is the
            // operator-visible signal in that case.
            crate::observability::predicate_diagnostics::record_turboquant_hints(payload.clone());
            // Structured tracing event remains for log-based observability —
            // grep `proximadb::turboquant::explain` in `RUST_LOG=debug`
            // output to see every TurboQuant scoring attempt regardless
            // of whether an EXPLAIN scope was active.
            tracing::debug!(
                target: "proximadb::turboquant::explain",
                collection_id = %ctx.collection_id(),
                blocks_skipped,
                payload = %payload,
                "TurboQuant scoring ran (Phase J/K explain hints)",
            );
            // Promote bridge errors to Phase J counter — the bridge call
            // is best-effort but log the failure path so dashboards see
            // it. The fallback scorer below still produces correct scores.
            if let Err(e) = bridge_result {
                tracing::warn!(
                    target: "proximadb::turboquant",
                    collection_id = %ctx.collection_id(),
                    error = %e,
                    "TurboQuant bridge returned error; falling back to full precision",
                );
            }
        }

        // Always run the full-precision scorer for correctness. The
        // optimization-heavy id-mapped scoring path lands in Phase D
        // (`TurboQuantAxisIndex`); this scorer remains the correctness
        // backstop in every release.
        self.score_full_precision(candidates, query_vector, ctx.distance_metric())
            .await
    }

    /// Score candidates using binary quantization (delegates to unified quantization)
    async fn score_binary(
        &self,
        candidates: &mut [ProgressiveSearchCandidate],
        query_vector: &[f32],
        level: &QuantizationLevel,
    ) -> Result<()> {
        // Quantize query vector to binary for fast Hamming distance comparison
        // This is the most performant approach for binary search stage
        let query_binary = self.quantization_engine.quantize_to_binary(query_vector)?;

        for candidate in candidates {
            if let Some(binary_repr) = candidate
                .quantized_vectors
                .iter()
                .find(|qv| qv.level_id == level.level_id)
            {
                // Use SIMD-optimized Hamming distance for binary vectors
                // This is much faster than generic distance computation
                let hamming_distance = self
                    .quantization_engine
                    .calculate_hamming_distance(&query_binary, &binary_repr.data);

                // Convert Hamming distance to normalized score (0-1 range)
                // Lower Hamming distance = higher similarity
                let vector_bits = query_binary.len() * 8;
                candidate.score = 1.0 - (hamming_distance as f32 / vector_bits as f32);
            }
        }

        Ok(())
    }

    /// Score candidates using scalar quantization (INT8) - delegates to unified quantization
    async fn score_scalar(
        &self,
        candidates: &mut [ProgressiveSearchCandidate],
        query_vector: &[f32],
        level: &QuantizationLevel,
    ) -> Result<()> {
        // Delegate all quantization and distance calculation to unified modules
        for candidate in candidates {
            if let Some(int8_repr) = candidate
                .quantized_vectors
                .iter()
                .find(|qv| qv.level_id == level.level_id)
            {
                // For INT8, convert back to f32 and use optimized distance computation
                // INT8 quantization preserves relative distances well
                let int8_as_f32: Vec<f32> = int8_repr
                    .data
                    .iter()
                    .map(|&x| x as f32 / 127.0) // Normalize INT8 to [-1, 1] range
                    .collect();

                let similarity = self.distance_compute.calculate_distance(
                    query_vector,
                    &int8_as_f32,
                    &DistanceMetric::Cosine, // Default to cosine for now
                );
                candidate.score = similarity.raw_value;
            }
        }

        Ok(())
    }

    /// Score candidates using product quantization - delegates to unified quantization
    async fn score_product(
        &self,
        candidates: &mut [ProgressiveSearchCandidate],
        query_vector: &[f32],
        level: &QuantizationLevel,
    ) -> Result<()> {
        // PQ requires num_subvectors to be specified
        let num_subvectors = level.num_subvectors.unwrap_or(8) as usize; // Default to 8 subvectors if not specified

        // For PQ, we need to use asymmetric distance computation
        // This is a simplified version - real PQ would use lookup tables
        for candidate in candidates {
            if let Some(pq_repr) = candidate
                .quantized_vectors
                .iter()
                .find(|qv| qv.level_id == level.level_id)
            {
                // Simplified PQ distance: treat as compressed vectors
                // Real implementation would use codebook lookup tables
                // For now, approximate with direct distance calculation
                let subvector_dim = query_vector.len() / num_subvectors;
                let mut total_distance = 0.0f32;

                for i in 0..num_subvectors {
                    let start = i * subvector_dim;
                    let end = ((i + 1) * subvector_dim).min(query_vector.len());
                    let _query_sub = &query_vector[start..end];

                    // Each byte in PQ data represents a codebook index
                    if i < pq_repr.data.len() {
                        let codebook_idx = pq_repr.data[i] as usize;
                        // Simplified: use codebook index as a distance proxy
                        total_distance += (codebook_idx as f32) / 255.0;
                    }
                }

                candidate.score = total_distance / num_subvectors as f32;
            }
        }

        Ok(())
    }

    /// Score candidates using full precision
    async fn score_full_precision(
        &self,
        candidates: &mut [ProgressiveSearchCandidate],
        query_vector: &[f32],
        distance_metric: DistanceMetric,
    ) -> Result<()> {
        for candidate in candidates {
            if let Some(ref vector) = candidate.vector {
                let result = self.distance_compute.calculate_distance(
                    query_vector,
                    vector,
                    &distance_metric,
                );
                let distance = result.rank_value;
                candidate.score = distance;
            }
        }

        Ok(())
    }

    /// Final reranking with full precision
    async fn final_rerank(
        &self,
        ctx: &StorageQueryContext,
        mut candidates: Vec<ProgressiveSearchCandidate>,
        query_vector: &[f32],
    ) -> Result<Vec<ProgressiveSearchCandidate>> {
        debug!(
            "Final reranking {} candidates with full precision",
            candidates.len()
        );

        self.score_full_precision(&mut candidates, query_vector, ctx.distance_metric())
            .await?;

        // Sort and truncate to top_k
        candidates.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(ctx.top_k());

        // Mark as full precision
        for candidate in &mut candidates {
            candidate.stage = SearchStage::FullPrecision;
        }

        Ok(candidates)
    }

    /// Fallback to full precision search
    async fn full_precision_search(
        &self,
        ctx: &StorageQueryContext,
        records: Vec<VectorRecord>,
        query_vector: &[f32],
    ) -> Result<Vec<OptimizedSearchRecord>> {
        let mut results = Vec::with_capacity(records.len());

        for record in records {
            let result = self.distance_compute.calculate_distance(
                query_vector,
                &record.vector,
                &ctx.distance_metric(),
            );
            let distance = result.rank_value;

            // Convert metadata to SqlValue for OptimizedSearchRecord
            let mut metadata_map = std::collections::HashMap::new();
            for (key, item) in record.metadata {
                let value = item.value;
                if let Some(value) = value {
                    use crate::proto::proximadb_v1::{self as proximadb_v1};
                    let sql_value = match value {
                        crate::proto::proximadb_v1::sql_value::Value::StringValue(s) => {
                            proximadb_v1::SqlValue {
                                value: Some(
                                    crate::proto::proximadb_v1::sql_value::Value::StringValue(s),
                                ),
                            }
                        }
                        crate::proto::proximadb_v1::sql_value::Value::NumberValue(f) => {
                            proximadb_v1::SqlValue {
                                value: Some(
                                    crate::proto::proximadb_v1::sql_value::Value::NumberValue(f),
                                ),
                            }
                        }
                        crate::proto::proximadb_v1::sql_value::Value::BoolValue(b) => {
                            proximadb_v1::SqlValue {
                                value: Some(
                                    crate::proto::proximadb_v1::sql_value::Value::BoolValue(b),
                                ),
                            }
                        }
                        crate::proto::proximadb_v1::sql_value::Value::Int64Value(i) => {
                            proximadb_v1::SqlValue {
                                value: Some(
                                    crate::proto::proximadb_v1::sql_value::Value::Int64Value(i),
                                ),
                            }
                        }
                        crate::proto::proximadb_v1::sql_value::Value::BytesValue(b) => {
                            proximadb_v1::SqlValue {
                                value: Some(
                                    crate::proto::proximadb_v1::sql_value::Value::BytesValue(b),
                                ),
                            }
                        }
                        crate::proto::proximadb_v1::sql_value::Value::JsonbValue(b) => {
                            proximadb_v1::SqlValue {
                                value: Some(
                                    crate::proto::proximadb_v1::sql_value::Value::JsonbValue(b),
                                ),
                            }
                        }
                        crate::proto::proximadb_v1::sql_value::Value::NullValue(n) => {
                            proximadb_v1::SqlValue {
                                value: Some(
                                    crate::proto::proximadb_v1::sql_value::Value::NullValue(n),
                                ),
                            }
                        }
                        crate::proto::proximadb_v1::sql_value::Value::ArrayValue(a) => {
                            proximadb_v1::SqlValue {
                                value: Some(
                                    crate::proto::proximadb_v1::sql_value::Value::ArrayValue(a),
                                ),
                            }
                        }
                        crate::proto::proximadb_v1::sql_value::Value::ObjectValue(o) => {
                            proximadb_v1::SqlValue {
                                value: Some(
                                    crate::proto::proximadb_v1::sql_value::Value::ObjectValue(o),
                                ),
                            }
                        }
                    };
                    metadata_map.insert(key, sql_value);
                }
            }

            results.push(
                OptimizedSearchRecord::new(record.id.clone(), distance)
                    .with_similarity(distance)
                    .add_vector(record.vector)
                    .with_metadata(metadata_map),
            );
        }

        // Sort and truncate
        results.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(ctx.top_k());

        Ok(results)
    }

    /// Convert candidates to search results
    fn convert_to_results(
        &self,
        candidates: Vec<ProgressiveSearchCandidate>,
        top_k: usize,
    ) -> Result<Vec<OptimizedSearchRecord>> {
        let mut results = Vec::with_capacity(top_k.min(candidates.len()));

        for candidate in candidates.into_iter().take(top_k) {
            let mut result = OptimizedSearchRecord::new(candidate.id, candidate.score)
                .with_similarity(candidate.score)
                .with_metadata(HashMap::new());

            if let Some(vec) = candidate.vector {
                result = result.add_vector(vec);
            }

            results.push(result);
        }

        Ok(results)
    }

    /// Determine if runtime quantization should be allowed based on multiple factors
    fn should_allow_runtime_quantization(&self, _ctx: &StorageQueryContext) -> Result<bool> {
        // Deferred: Implement proper collection config and search hints checking
        // For now, always allow runtime quantization to make it compile
        Ok(true)
    }

    /// Helper: Parse pre-computed quantized data
    #[allow(dead_code)]
    fn parse_quantized_data(
        &self,
        _data: &[u8],
        _levels: &[QuantizationLevel],
    ) -> Result<Vec<QuantizedRepresentation>> {
        // Deferred: Implement parsing of serialized quantized data
        // For now, return empty vec
        Ok(Vec::new())
    }

    /// Helper: Quantize vector on-the-fly (for storage-optimized path).
    ///
    /// Lifecycle-aware (Phase C — Quantization Trait Convergence Plan):
    /// `WriteTime` variants encode into a fresh `Vec<u8>` of codes;
    /// `ReadTime` variants (TurboQuant) emit an empty buffer because their
    /// codes live in the per-collection `TurboQuantStore`, not in
    /// per-search storage. `Identity` also emits empty.
    #[allow(dead_code)]
    fn quantize_vector(
        &self,
        vector: &[f32],
        levels: &[QuantizationLevel],
    ) -> Result<Vec<QuantizedRepresentation>> {
        trace!("Runtime quantization for storage-optimized path");
        let mut representations = Vec::new();

        for level in levels {
            let data = match level.quantization_type.lifecycle() {
                proximadb_quantization_types::QuantizationLifecycle::WriteTime => {
                    match level.quantization_type {
                        QuantizationType::Binary => {
                            self.quantization_engine.quantize_to_binary(vector)?
                        }
                        QuantizationType::Scalar => {
                            self.quantization_engine.quantize_to_int8(vector)?
                        }
                        QuantizationType::Product => self.quantization_engine.quantize_to_pq(
                            vector,
                            level.num_subvectors.unwrap_or(8) as usize,
                            level.bits as u32,
                        )?,
                        // Future WriteTime variants without a runtime
                        // quantizer fall through to empty; the scorer
                        // dispatches to full-precision in that case.
                        _ => Vec::new(),
                    }
                }
                // ReadTime + Identity: nothing to emit at this layer.
                _ => Vec::new(),
            };

            representations.push(QuantizedRepresentation {
                level_id: level.level_id.clone(),
                data,
                quant_type: level.quantization_type,
            });
        }

        Ok(representations)
    }

    /// Helper: Get search stage from quantization type.
    ///
    /// Phase C: lifecycle-aware. `ReadTime` and `Identity` both map to
    /// `FullPrecision` because the stage sequence is a WriteTime concept
    /// (cascade of pre-encoded codes); ReadTime variants use their own
    /// kernel path via `score_turboquant`.
    fn get_search_stage(&self, quant_type: &QuantizationType) -> SearchStage {
        match quant_type.lifecycle() {
            proximadb_quantization_types::QuantizationLifecycle::WriteTime => match quant_type {
                QuantizationType::Binary => SearchStage::BinaryFilter,
                QuantizationType::Scalar => SearchStage::Int8Ranking,
                QuantizationType::Product => SearchStage::PqRanking,
                _ => SearchStage::FullPrecision,
            },
            // ReadTime variants don't participate in the cascade stage
            // sequence — their own kernel handles ranking. FullPrecision
            // is the sane sentinel for any caller that asks for "the
            // stage" of a TurboQuant level.
            _ => SearchStage::FullPrecision,
        }
    }

    // NOTE: All distance computations are delegated to UnifiedQuantizationEngine
    // which internally uses UnifiedDistanceCompute with SIMD optimizations.
    // No manual distance calculations should be done here to maintain
    // proper separation of concerns and leverage hardware-optimized implementations.
}

/// Builder pattern for configuring progressive search
pub struct ProgressiveSearchBuilder {
    quantization_engine: Option<Arc<UnifiedQuantizationEngine>>,
    distance_compute: Option<Arc<UnifiedDistanceCompute>>,
    #[cfg(feature = "experimental-turboquant")]
    turboquant_registry: Option<
        Arc<dyn crate::compute::quantization::turboquant_store_registry::TurboQuantStoreRegistry>,
    >,
}

impl ProgressiveSearchBuilder {
    pub fn new() -> Self {
        Self {
            quantization_engine: None,
            distance_compute: None,
            #[cfg(feature = "experimental-turboquant")]
            turboquant_registry: None,
        }
    }

    pub fn with_quantization_engine(mut self, engine: Arc<UnifiedQuantizationEngine>) -> Self {
        self.quantization_engine = Some(engine);
        self
    }

    pub fn with_distance_compute(mut self, compute: Arc<UnifiedDistanceCompute>) -> Self {
        self.distance_compute = Some(compute);
        self
    }

    /// Register a TurboQuant store registry for `ReadTime` lifecycle
    /// dispatch (Phase C — Quantization Trait Convergence Plan).
    /// Optional: omit on non-TurboQuant deployments and the executor
    /// will fall back to full-precision scoring for any TurboQuant
    /// level that arrives.
    #[cfg(feature = "experimental-turboquant")]
    pub fn with_turboquant_registry(
        mut self,
        registry: Arc<
            dyn crate::compute::quantization::turboquant_store_registry::TurboQuantStoreRegistry,
        >,
    ) -> Self {
        self.turboquant_registry = Some(registry);
        self
    }

    pub fn build(self) -> Result<ProgressiveSearchExecutor> {
        let executor = ProgressiveSearchExecutor::new(
            self.quantization_engine
                .ok_or_else(|| anyhow::anyhow!("Quantization engine required"))?,
            self.distance_compute
                .ok_or_else(|| anyhow::anyhow!("Distance compute required"))?,
        );
        #[cfg(feature = "experimental-turboquant")]
        let executor = match self.turboquant_registry {
            Some(r) => executor.with_turboquant_registry(r),
            None => executor,
        };
        Ok(executor)
    }
}

impl Default for ProgressiveSearchBuilder {
    fn default() -> Self {
        Self::new()
    }
}
