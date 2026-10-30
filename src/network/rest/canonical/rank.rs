//! REST DTOs + handler logic for the multi-phase ranking pipeline.
//!
//! Realises the wire shape documented in spec §4.9.1 — `rank_profile` +
//! `rank_overrides` on the request, `score_vector` + `match_features` +
//! `summary_features` + `phase_truncated` + `rank_profile_version` on
//! the response.
//!
//! R-7b ships the DTOs and a handler that drives the rank pipeline end-
//! to-end against an injected candidate provider. The handler is *not*
//! yet wired into the production REST router — that needs collection-
//! storage state to source real candidates, which is R-7c's job. The
//! handler function is callable directly (and tested that way) so the
//! integration point is small when R-7c lands.
//!
//! See `roadmap/RANKING_FRAMEWORK_SPEC_2026_05_23.md` (R-7b).

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::errors::{ApiError, ApiResult};

use crate::core::search::rank::CrossModalGlobalScorer;
use proximadb_kernel::{ScoreComponent, ScoreVector};
use proximadb_query::reranking::RerankConfig;
use proximadb_rank_core::{
    BlueprintFactory, DocHandle, FeatureArena, GlobalScorer, NoopAttributeAccess,
    NoopCandidateData, NoopMetricsSink, NoopModelCache, PhaseOutcome, QueryContext, RankError,
    RankResult, ScoreCtx, ScoredHit, SecondPhaseScorer,
};
use proximadb_rank_features::register_builtins;
use proximadb_rank_profile::{CompiledRankProfile, ProfileRegistry};

// =========================================================================
// Request DTOs
// =========================================================================

/// REST request for the rank pipeline.
///
/// Matches the spec §4.9.1 documented shape. `query_vector` is the
/// post-embedding-service vector (caller computed it). `rank_overrides`
/// nests per-phase tweaks the planner / user applies on top of the
/// resolved profile.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RankSearchRequest {
    /// Target collection.
    pub collection: String,
    /// Embedding for retrieval. Not used by R-7b's mock candidate
    /// provider but required so the wire shape is the production one.
    #[serde(default)]
    pub query_vector: Vec<f32>,
    /// Optional query text for the BM25 / full-text side of hybrid
    /// retrieval (R-7c.3.1). The `HybridCoordinatorAdapter` forwards
    /// it to the BM25 backend; absent / empty text means "vector-only"
    /// mode (the BM25 closure still fires but with an empty query, so
    /// the backend can return nothing or apply its own fallback).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_text: Option<String>,
    /// Result count after the global phase.
    #[serde(default = "default_top_k")]
    pub k: usize,
    /// Name of a rank profile in the [`ProfileRegistry`]. When `None`,
    /// the handler returns retrieval-only output (mirrors today's
    /// existing search behavior).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rank_profile: Option<String>,
    /// Per-phase overrides applied on top of the profile.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rank_overrides: Option<RankOverrides>,
    /// What the request acts as (set by the route from the request context,
    /// never from the wire). Candidate providers resolve the collection within
    /// it; an unset scope denies.
    #[serde(skip)]
    pub scope: RankScope,
}

/// Which collections a rank request may resolve. There is no implicit
/// "unscoped": a request that was never given a scope is denied.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum RankScope {
    /// No scope was set; the collection reads as not found.
    #[default]
    Unset,
    /// A request acting as this tenant.
    Tenant(String),
    /// Explicit internal (system) execution, unscoped by tenant.
    Internal,
}

impl RankScope {
    /// The scope of a resolution tenant from `request_scope`: `Some` ⇒ that
    /// tenant, `None` ⇒ explicit internal execution.
    pub fn from_resolution_tenant(tenant: Option<String>) -> Self {
        match tenant {
            Some(tenant) => Self::Tenant(tenant),
            None => Self::Internal,
        }
    }
}

fn default_top_k() -> usize {
    10
}

/// Per-phase override knobs. Optional; when fields are `None`, the
/// profile-resolved values stand.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct RankOverrides {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub second_phase: Option<PhaseOverride>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_phase: Option<PhaseOverride>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct PhaseOverride {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rerank_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_size: Option<u32>,
}

// =========================================================================
// Response DTOs
// =========================================================================

/// One scored hit on the wire.
///
/// `score_vector` is `None` when no profile was attached — preserves
/// the NFR-9 zero-cost-when-unused contract. `match_features` and
/// `summary_features` are emitted only when the profile declares them.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct ScoredHitDto {
    pub id: String,
    pub score: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score_vector: Option<ScoreVectorDto>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub match_features: HashMap<String, f64>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub summary_features: HashMap<String, f64>,
}

/// Wire-form ScoreVector. Mirrors the kernel `ScoreVector` but with
/// JSON-friendly `phase` as `u8` (not `PhaseId` newtype) and components
/// inlined rather than `Arc<[ScoreComponent]>`.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct ScoreVectorDto {
    pub primary: f32,
    pub phase: u8,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub components: Vec<ScoreComponent>,
}

impl From<&ScoreVector> for ScoreVectorDto {
    fn from(sv: &ScoreVector) -> Self {
        Self {
            primary: sv.primary,
            phase: sv.phase.0,
            components: sv.components.as_ref().to_vec(),
        }
    }
}

/// REST response for the rank pipeline.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct RankSearchResponse {
    pub hits: Vec<ScoredHitDto>,
    pub phase_truncated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rank_profile: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rank_profile_version: Option<u32>,
}

// =========================================================================
// Handler — pure-function form, tested directly. R-7c wires it into
// the axum router with real collection storage state.
// =========================================================================

/// Source of candidate `DocHandle`s for a query. Production
/// implementation ([`HybridCoordinatorAdapter`], R-7c.3) wraps the
/// hybrid coordinator; tests pass mock impls so the rank pipeline can
/// be exercised independently of retrieval.
///
/// Async (R-7c.3): the production backend is async (parallel BM25 +
/// vector search via `HybridCoordinator`), so the trait surface is
/// async. Callers `.await` outside `spawn_blocking` — the produced
/// `CandidateBatch.docs` is then moved into the blocking closure for
/// the arena-bearing rank phases; the optional `original_ids` map
/// stays in the outer async scope and is consulted at response-build
/// time (R-7c.3.2).
#[async_trait::async_trait]
pub trait CandidateProvider: Send + Sync {
    async fn candidates(&self, request: &RankSearchRequest) -> RankResult<CandidateBatch>;
}

/// Output of a [`CandidateProvider`] — the set of candidate
/// `DocHandle`s plus an optional translation table for backends that
/// use string ids (R-7c.3.2).
///
/// `docs` is the canonical list the rank pipeline iterates over.
/// `original_ids`, when `Some`, maps each handle back to the backend's
/// original string id; the dispatcher consults it at response-build
/// time so clients see their backend ids round-tripped exactly.
/// When `None`, the dispatcher falls back to stringifying `DocHandle.0`
/// (legacy behavior — covers numeric-id backends + tests).
#[derive(Debug, Clone, Default)]
pub struct CandidateBatch {
    pub docs: Vec<DocHandle>,
    pub original_ids: Option<HashMap<DocHandle, std::sync::Arc<str>>>,
}

impl CandidateBatch {
    /// Build from a `Vec<DocHandle>` with no id translation (typical
    /// for numeric or test backends).
    pub fn from_docs(docs: Vec<DocHandle>) -> Self {
        Self {
            docs,
            original_ids: None,
        }
    }

    /// Build from arbitrary string ids — assigns sequential synthetic
    /// `DocHandle`s (1..=N) and stashes the original strings so the
    /// response can round-trip them exactly.
    pub fn from_string_ids<I, S>(ids: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<std::sync::Arc<str>>,
    {
        let mut docs = Vec::new();
        let mut map = HashMap::new();
        for (i, s) in ids.into_iter().enumerate() {
            let handle = DocHandle((i + 1) as u32);
            docs.push(handle);
            map.insert(handle, s.into());
        }
        Self {
            docs,
            original_ids: Some(map),
        }
    }
}

/// Execute a [`RankSearchRequest`] against the registry and candidate
/// provider, returning the wire response.
///
/// `second_phase_scorer` — optional per-request scorer to fire if the
/// profile has a `second_phase` configured. When `None` and the
/// profile *does* have a second phase, the dispatcher passes through
/// (matches `RankPipeline::run_second_phase`'s no-scorer contract).
/// The HTTP route resolves this via `RankServices::second_phase_scorer`
/// from the profile name; tests pass directly.
pub async fn handle_rank_search(
    req: RankSearchRequest,
    registry: &ProfileRegistry,
    candidates: &dyn CandidateProvider,
    factory: Arc<BlueprintFactory>,
    second_phase_scorer: Option<Arc<dyn SecondPhaseScorer>>,
) -> RankResult<RankSearchResponse> {
    handle_rank_search_with_config(
        req,
        registry,
        candidates,
        factory,
        second_phase_scorer,
        default_rerank_config(),
    )
    .await
}

/// Like [`handle_rank_search`] but with an explicit cross-modal reranker
/// config. The HTTP dispatch path passes the server's `[query.reranking]`
/// config here; the plain wrapper keeps the built-in default for direct
/// callers (tests, mock scaffolds).
pub async fn handle_rank_search_with_config(
    req: RankSearchRequest,
    registry: &ProfileRegistry,
    candidates: &dyn CandidateProvider,
    factory: Arc<BlueprintFactory>,
    second_phase_scorer: Option<Arc<dyn SecondPhaseScorer>>,
    rerank_config: RerankConfig,
) -> RankResult<RankSearchResponse> {
    handle_rank_search_with_metrics(
        req,
        registry,
        candidates,
        factory,
        second_phase_scorer,
        rerank_config,
        None,
    )
    .await
}

/// Like [`handle_rank_search`] but with an optional metrics handle
/// so per-feature observability fires through the spec's Prometheus
/// surface. The original entry point delegates to this with
/// `metrics = None` so existing callers (tests, mock REST scaffolds)
/// stay unchanged.
#[allow(clippy::too_many_arguments)]
pub async fn handle_rank_search_with_metrics(
    req: RankSearchRequest,
    registry: &ProfileRegistry,
    candidates: &dyn CandidateProvider,
    factory: Arc<BlueprintFactory>,
    second_phase_scorer: Option<Arc<dyn SecondPhaseScorer>>,
    rerank_config: RerankConfig,
    metrics: Option<Arc<crate::observability::rank_metrics::RankPipelineMetrics>>,
) -> RankResult<RankSearchResponse> {
    let batch = candidates.candidates(&req).await?;
    let candidate_docs = batch.docs;
    // R-7c.3.2: optional backend-id translation table. Stays in the
    // outer async scope; consulted at response-build time.
    let original_ids = batch.original_ids;
    let render_id = |doc: DocHandle| -> String {
        original_ids
            .as_ref()
            .and_then(|m| m.get(&doc))
            .map(|s| s.to_string())
            .unwrap_or_else(|| doc.0.to_string())
    };
    let qctx = QueryContext {
        query_vector: if req.query_vector.is_empty() {
            None
        } else {
            Some(req.query_vector.clone())
        },
        // R-5b.1.3: seed `query_text` from the request so the
        // tokenized cross-encoder extractor reads it per-request via
        // the QueryContext rather than from interior mutability.
        query_text: req.query_text.as_deref().map(std::sync::Arc::<str>::from),
        ..Default::default()
    };

    // No profile attached → retrieval-only path: return the candidate
    // order unchanged with score = 0 and no score_vector. This mirrors
    // the spec's NFR-9 zero-cost contract for the unmodified search
    // path.
    let Some(profile_name) = req.rank_profile.as_deref() else {
        let hits = candidate_docs
            .iter()
            .take(req.k)
            .map(|&doc| ScoredHitDto {
                id: render_id(doc),
                score: 0.0,
                score_vector: None,
                match_features: HashMap::new(),
                summary_features: HashMap::new(),
            })
            .collect();
        return Ok(RankSearchResponse {
            hits,
            phase_truncated: false,
            rank_profile: None,
            rank_profile_version: None,
        });
    };

    // Resolve the profile.
    let compiled = registry
        .get(profile_name)
        .ok_or_else(|| RankError::ProfileNotFound(profile_name.to_string()))?;
    let _ = factory; // The compiled profile already carries its own factory;
    // the parameter exists for future extension.

    // Global scorer: cross-modal reranker if the profile asked for it.
    // Selected BEFORE spawn_blocking because it crosses the async boundary.
    let global: Option<Arc<dyn GlobalScorer>> = if compiled
        .spec
        .global_phase
        .as_ref()
        .map(|g| g.strategy == "cross_modal")
        .unwrap_or(false)
    {
        Some(Arc::new(CrossModalGlobalScorer::new(rerank_config)))
    } else {
        None
    };

    let topk = req
        .rank_overrides
        .as_ref()
        .and_then(|o| o.global_phase.as_ref())
        .and_then(|g| g.rerank_count)
        .map(|k| k as usize)
        .unwrap_or(req.k);

    // R-7c.1: run the arena-bearing first-phase work on a blocking thread
    // so the outer future is Send. `bumpalo::Bump` is !Sync internally,
    // and ScoreCtx holds &FeatureArena, so holding ScoreCtx across the
    // async `global.score(...).await` below would make the future
    // !Send and axum's tokio multi-threaded runtime reject the handler.
    //
    // Inside the closure we own the materialised pipeline, construct a
    // local arena + Noop context fixtures, run the first phase, and
    // return owned PhaseOutcome. The async global scorer (Send-friendly)
    // runs outside.
    let qctx_for_block = qctx.clone();
    let compiled_for_block = compiled.clone();
    let candidate_docs_for_block = candidate_docs.clone();
    let scorer_for_block = second_phase_scorer.clone();
    let metrics_for_block = metrics.clone().map(|m| (m, compiled.spec.name.clone()));
    let phase_outcome: PhaseOutcome = tokio::task::spawn_blocking(move || {
        let metrics_ref = metrics_for_block
            .as_ref()
            .map(|(m, name)| (m.clone(), name.as_str()));
        run_phases_blocking(
            &compiled_for_block,
            &qctx_for_block,
            &candidate_docs_for_block,
            scorer_for_block.as_deref(),
            metrics_ref,
        )
    })
    .await
    .map_err(|join_err| RankError::ModelInference {
        model_id: "rank_search:phases".into(),
        reason: format!("phase task panicked: {join_err}"),
    })??;

    // Global phase — async, no arena reference in scope so the future is Send.
    let final_hits: Vec<ScoredHit> = match global {
        Some(g) => g.score(phase_outcome.hits.clone(), topk).await?,
        None => {
            let mut h = phase_outcome.hits.clone();
            h.truncate(topk);
            h
        }
    };

    let hits: Vec<ScoredHitDto> = final_hits
        .into_iter()
        .map(|h| {
            let sv = ScoreVector::from_primary(h.score, h.phase);
            // R-7c.5 + R-7c.5b: lift both per-doc Arc snapshots into the
            // wire `HashMap<String, f64>` maps. Both stay empty when
            // the active profile didn't declare the corresponding
            // category (the common case), preserving the existing
            // zero-allocation path.
            let match_features = arc_features_to_wire_map(h.features.as_ref());
            let summary_features = arc_features_to_wire_map(h.summary.as_ref());
            ScoredHitDto {
                id: render_id(h.doc),
                score: h.score,
                score_vector: Some(ScoreVectorDto::from(&sv)),
                match_features,
                summary_features,
            }
        })
        .collect();

    Ok(RankSearchResponse {
        hits,
        phase_truncated: phase_outcome.truncated,
        rank_profile: Some(compiled.spec.name.clone()),
        rank_profile_version: Some(compiled.spec.version),
    })
}

/// Owned-state phase runner: materialises the pipeline, runs first
/// phase, then runs second phase if a scorer is supplied. Designed to
/// be called inside `tokio::task::spawn_blocking` — inputs are owned /
/// Arc-shared so the closure is `Send + 'static`, and the return type
/// is plain data with no arena lifetime. All arena-bearing references
/// are confined to this stack frame.
///
/// When `second_phase_scorer` is `None`, only the first phase runs.
/// When it's `Some(_)` AND the profile has a `second_phase`
/// configured, `RankPipeline::run_second_phase` consumes it. When
/// `Some(_)` BUT the profile has no second phase, the scorer is
/// silently ignored (matches the pipeline's pass-through contract —
/// don't pay for what you didn't ask for).
fn run_phases_blocking(
    compiled: &CompiledRankProfile,
    qctx: &QueryContext,
    candidate_docs: &[DocHandle],
    second_phase_scorer: Option<&dyn SecondPhaseScorer>,
    metrics: Option<(
        Arc<crate::observability::rank_metrics::RankPipelineMetrics>,
        &str,
    )>,
) -> RankResult<PhaseOutcome> {
    use proximadb_kernel::PhaseId;
    let mut pipeline = compiled.materialize(qctx)?;
    let arena = FeatureArena::new();
    let attr = NoopAttributeAccess;
    let cand = NoopCandidateData;
    let models = NoopModelCache;

    // R-7c.4d metrics wiring: when RankServices supplied a
    // Prometheus handle, build a per-phase sink and emit through
    // the spec's `{profile, phase, feature}` label set. Without a
    // handle, keep NoopMetricsSink so the NFR-9 zero-cost-when-
    // unused contract holds.
    let prom_sink: Option<crate::observability::rank_metrics::PrometheusRankSink> =
        metrics.as_ref().map(|(m, profile)| {
            crate::observability::rank_metrics::PrometheusRankSink::new(
                m.clone(),
                *profile,
                PhaseId::FIRST,
            )
        });
    let noop_sink = NoopMetricsSink;
    let metrics_ref: &dyn proximadb_rank_core::RankMetricsSink = match prom_sink.as_ref() {
        Some(s) => s,
        None => &noop_sink,
    };
    let mut ctx = ScoreCtx::new(qctx, &arena, &attr, &cand, &models, metrics_ref);
    let first_outcome = pipeline.run_first_phase(candidate_docs, &mut ctx)?;

    // Drop the arena-bearing context before second phase — the scorer
    // doesn't need it (it carries its own model session) and dropping
    // here makes the lifetime story easier for any future caller that
    // wants to keep the pipeline alive past this function. Neither
    // type currently impls `Drop`, but `drop(_)` still ends the
    // lexical borrow so subsequent code can re-borrow whatever they
    // were carrying.
    #[allow(clippy::drop_non_drop)]
    {
        drop(ctx);
        drop(arena);
    }

    // R-7c.4d follow-up: emit per-phase wall-clock latency (spec
    // §4.10 rank_phase_latency_us histogram) for first phase from
    // the outcome's elapsed_us. Done here at the orchestrator so
    // rank-core stays single-responsibility (it only emits
    // per-feature latency via the metrics sink).
    if let Some((m, profile)) = metrics.as_ref() {
        let phase_label = crate::observability::rank_metrics::RankPipelineMetrics::phase_label_for(
            PhaseId::FIRST.0,
        );
        m.observe_phase_latency_us(profile, phase_label, first_outcome.elapsed_us as f64);
    }

    match second_phase_scorer {
        Some(scorer) => {
            // R-7c.4d follow-up: time the second-phase rescore
            // call here at the orchestrator layer rather than
            // inside rank-core (the SecondPhaseScorer trait
            // doesn't take a metrics ref, and extending it would
            // ripple through every concrete scorer impl). The
            // scorer's identity surfaces as the `feature` label —
            // the trait doesn't expose a name today, so we use the
            // canonical `"second_phase"` placeholder. A future
            // slice can promote a real model_id when SecondPhaseScorer
            // grows a `name()` accessor.
            let t0 = std::time::Instant::now();
            let result = pipeline.run_second_phase(first_outcome, scorer, qctx);
            if let Some((m, profile)) = metrics.as_ref() {
                let elapsed_ns = t0.elapsed().as_nanos() as u64;
                let elapsed_us = elapsed_ns as f64 / 1_000.0;
                let phase_label =
                    crate::observability::rank_metrics::RankPipelineMetrics::phase_label_for(
                        PhaseId::SECOND.0,
                    );
                m.observe_feature_latency_us(profile, phase_label, "second_phase", elapsed_us);
                // Per-phase wall-clock latency for second phase
                // (spec §4.10 schema). Same elapsed value as
                // feature=second_phase because the wrapper-level
                // observation IS the per-phase wall clock — the
                // distinction only matters once a real per-doc
                // feature observation lands.
                m.observe_phase_latency_us(profile, phase_label, elapsed_us);
                if let Ok(ref outcome) = result
                    && outcome.truncated
                {
                    // The second phase preserves the first phase's
                    // truncated flag; if it's still set post-rescore,
                    // surface a second-phase truncation event too so
                    // dashboards attribute correctly (the first-phase
                    // record_phase_truncated emit at the rank-core
                    // call site already fired with "budget"/"deadline").
                    m.inc_phase_truncated(profile, phase_label, "carried_forward");
                }
            }
            result
        }
        None => Ok(first_outcome),
    }
}

// =========================================================================
// Production wiring (R-7c) — RankServices singleton + axum route.
// =========================================================================

/// Bundles every singleton the rank pipeline needs at request time. One
/// instance per process, constructed at server startup and injected into
/// [`crate::network::rest::canonical::handlers::AppState`] via
/// [`AppState::with_rank_services`].
pub struct RankServices {
    pub profile_registry: Arc<ProfileRegistry>,
    pub blueprint_factory: Arc<BlueprintFactory>,
    pub candidate_provider: Arc<dyn CandidateProvider>,
    /// Per-profile second-phase scorer registry (R-7c.2). Keyed by
    /// profile name. Lookup on every request — when present + the
    /// profile has a `second_phase` configured, the dispatcher runs
    /// `RankPipeline::run_second_phase` inside the spawn_blocking
    /// closure. When absent + the profile has a second phase
    /// configured, the dispatcher passes through with `PhaseId::FIRST`
    /// tags preserved (matches `RankPipeline::run_second_phase`'s
    /// no-second-phase contract).
    ///
    /// Concurrent access via `DashMap` so registrations can happen
    /// after `RankServices` is already shared via `Arc` — useful when
    /// a control-plane RPC installs / swaps scorers at runtime
    /// (R-7c.2.1 follow-up).
    pub second_phase_scorers: dashmap::DashMap<String, Arc<dyn SecondPhaseScorer>>,
    /// Cross-modal reranker config for the global phase. Loaded from
    /// `[query.reranking]` in the server config and threaded into the
    /// `CrossModalReranker` construction — before this field existed the
    /// handler hardcoded `default_rerank_config()`, which made the config
    /// file's `rerank_top_k` inert in production (the "config field read by
    /// nobody" defect class). Defaults preserve the previous behavior.
    pub rerank_config: RerankConfig,
    /// Optional Prometheus metric handles for the rank pipeline.
    /// When `Some`, `run_phases_blocking` builds a
    /// `PrometheusRankSink` and emits the spec's
    /// `proximadb_rank_feature_latency_us` / `_phase_truncated_total`
    /// metrics. When `None` (the default), the pipeline keeps using
    /// `NoopMetricsSink` so the zero-cost-when-unused contract
    /// (NFR-9) holds.
    pub metrics: Option<Arc<crate::observability::rank_metrics::RankPipelineMetrics>>,
}

impl RankServices {
    /// Convenience constructor: empty registry + factory pre-populated with
    /// the R-2 built-in features (attribute / closeness / bm25 / freshness /
    /// decay) + supplied candidate provider. Production callers register
    /// profiles via [`ProfileRegistry::install`] after construction.
    pub fn new(candidate_provider: Arc<dyn CandidateProvider>) -> Self {
        let factory = Arc::new(BlueprintFactory::new());
        register_builtins(&factory);
        Self {
            profile_registry: Arc::new(ProfileRegistry::new()),
            blueprint_factory: factory,
            candidate_provider,
            second_phase_scorers: dashmap::DashMap::new(),
            rerank_config: default_rerank_config(),
            metrics: None,
        }
    }

    /// Override the cross-modal reranker config from the server's
    /// `[query.reranking]` section. Call at server startup, before the
    /// services are shared via `Arc`.
    pub fn with_rerank_config(mut self, config: RerankConfig) -> Self {
        self.rerank_config = config;
        self
    }

    /// Wire a `RankPipelineMetrics` handle so per-feature latency +
    /// phase-truncation observations are emitted via Prometheus
    /// instead of the noop default. Call once at server startup
    /// after `RankPipelineMetrics::register(&registry)`.
    pub fn with_metrics(
        mut self,
        metrics: Arc<crate::observability::rank_metrics::RankPipelineMetrics>,
    ) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Register a second-phase scorer against a profile name. Used at
    /// server startup after instantiating concrete scorers (e.g.
    /// `OnnxSecondPhaseScorer` from `proximadb-rank-onnx`) and binding
    /// each to the profile that should fire it.
    ///
    /// Re-registering the same name replaces the prior scorer (last-
    /// write-wins, matches the profile registry's hot-reload contract).
    pub fn register_second_phase_scorer(
        &self,
        profile_name: impl Into<String>,
        scorer: Arc<dyn SecondPhaseScorer>,
    ) {
        self.second_phase_scorers
            .insert(profile_name.into(), scorer);
    }

    /// Look up the second-phase scorer for the named profile. Returns
    /// `None` if no scorer is registered (the dispatcher will then
    /// pass-through the second phase even if the profile has one
    /// configured).
    pub fn second_phase_scorer(&self, profile_name: &str) -> Option<Arc<dyn SecondPhaseScorer>> {
        self.second_phase_scorers
            .get(profile_name)
            .map(|r| r.value().clone())
    }

    /// Install (or hot-reload) a compiled profile into the registry and
    /// fire `proximadb_rank_profile_reload_total{profile, outcome}` per
    /// spec §4.10. The registry's install is infallible so the outcome
    /// label is always `"ok"` here; the `error` variant is reserved for
    /// higher-level install paths (DSL parse failure, validator
    /// rejection) that decide *not* to call into the registry. Callers
    /// that go through the validator should bump the counter with
    /// `outcome="error"` on rejection.
    pub fn install_profile(&self, profile: CompiledRankProfile) {
        let name = profile.spec.name.clone();
        self.profile_registry.install(profile);
        if let Some(metrics) = &self.metrics {
            metrics.inc_profile_reload(&name, "ok");
        }
    }

    /// Record a failed profile install attempt (validation /
    /// compilation rejected the profile, so it never reached the
    /// registry). Bumps the spec §4.10
    /// `proximadb_rank_profile_reload_total{profile, outcome="error"}`
    /// counter when metrics are wired.
    pub fn record_profile_reload_error(&self, profile_name: &str) {
        if let Some(metrics) = &self.metrics {
            metrics.inc_profile_reload(profile_name, "error");
        }
    }
}

/// Mock candidate provider that returns a fixed range of `DocHandle`s for
/// any request. Useful for R-7c smoke tests and as a deployment fallback
/// before the real `HybridCoordinator` adapter lands in R-7c.1.
pub struct MockRangeCandidateProvider {
    pub count: u32,
}

impl Default for MockRangeCandidateProvider {
    fn default() -> Self {
        Self { count: 20 }
    }
}

#[async_trait::async_trait]
impl CandidateProvider for MockRangeCandidateProvider {
    async fn candidates(&self, _request: &RankSearchRequest) -> RankResult<CandidateBatch> {
        Ok(CandidateBatch::from_docs(
            (1..=self.count).map(DocHandle).collect(),
        ))
    }
}

// =========================================================================
// HybridCoordinator integration (R-7c.3)
// =========================================================================

/// Abstract backend the [`HybridCoordinatorAdapter`] calls into. One
/// instance per process — wraps whatever real BM25 + vector services
/// the deployment uses. Tests use mock impls.
///
/// The two methods are `async` because the real services (Tantivy +
/// `UnifiedSearchInterface`) are async. The methods don't take the
/// collection name — that comes from the request and is bound in the
/// closure passed to `HybridCoordinator::execute_hybrid_search`. v1
/// passes the collection via thread-local-style stash; R-7c.3.1 will
/// add it as an explicit arg once we settle on the production
/// signatures of the underlying services.
#[async_trait::async_trait]
pub trait HybridSearchBackend: Send + Sync {
    async fn bm25_search(
        &self,
        collection: &str,
        query: &str,
    ) -> RankResult<Vec<crate::core::search::hybrid::BM25Result>>;
    async fn vector_search(
        &self,
        collection: &str,
        vector: &[f32],
    ) -> RankResult<Vec<crate::core::search::hybrid::VectorResult>>;

    /// Resolve `collection` within `request_tenant` before either leg runs:
    /// `Err(RankError::CollectionNotFound)` when the tenant may not resolve it
    /// (answered exactly like a missing collection). Every backend decides;
    /// there is no unscoped default.
    async fn scope_collection(&self, collection: &str, request_tenant: &str) -> RankResult<()>;
}

/// `CandidateProvider` that delegates to a real
/// [`crate::core::search::hybrid::HybridCoordinator`]. Production
/// constructs one of these at server startup, registers it on
/// `RankServices`, and the rank route routes through it
/// automatically.
///
/// The doc-id contract: `BM25Result.doc_id` and `VectorResult.doc_id`
/// are arbitrary strings; this adapter parses them as decimal `u32`
/// for `DocHandle`. Backends that use non-numeric ids will return
/// `DocHandle(0)` for those rows — the rank pipeline still works but
/// the output ids round-trip wrong. R-7c.3.1 will widen `DocHandle`
/// to a string-aware variant or add an explicit `doc_id_to_handle`
/// trait method.
pub struct HybridCoordinatorAdapter {
    coordinator: crate::core::search::hybrid::HybridCoordinator,
    backend: Arc<dyn HybridSearchBackend>,
}

impl HybridCoordinatorAdapter {
    pub fn new(
        fusion: crate::core::search::hybrid::FusionStrategy,
        backend: Arc<dyn HybridSearchBackend>,
    ) -> Self {
        Self {
            coordinator: crate::core::search::hybrid::HybridCoordinator::new(fusion),
            backend,
        }
    }

    pub fn with_top_k(
        fusion: crate::core::search::hybrid::FusionStrategy,
        top_k: usize,
        backend: Arc<dyn HybridSearchBackend>,
    ) -> Self {
        Self {
            coordinator: crate::core::search::hybrid::HybridCoordinator::with_top_k(fusion, top_k),
            backend,
        }
    }
}

/// Parse a backend doc_id string into a DocHandle. Returns `None` for
/// non-numeric ids — the caller decides whether to drop or use a
/// sentinel. R-7c.3.1 should replace this with a string-aware
/// DocHandle variant.
///
/// Currently unused at production call sites — kept (and
/// `#[allow(dead_code)]`-d) for the R-7c.3.1 follow-up that wires it
/// into the hybrid candidate path.
#[allow(dead_code)]
fn doc_id_to_handle(doc_id: &str) -> Option<DocHandle> {
    doc_id.parse::<u32>().ok().map(DocHandle)
}

#[async_trait::async_trait]
impl CandidateProvider for HybridCoordinatorAdapter {
    async fn candidates(&self, request: &RankSearchRequest) -> RankResult<CandidateBatch> {
        // The hybrid coordinator's BM25 + vector closures need their
        // own clones of the backend Arc — each closure consumes its
        // captured state by move.
        let backend_bm25 = self.backend.clone();
        let backend_vec = self.backend.clone();
        let collection_for_bm25 = request.collection.clone();
        let collection_for_vec = request.collection.clone();

        // R-7c.3.1: query_text now flows from the request. Empty
        // string is the documented "vector-only" sentinel — backends
        // can short-circuit on empty input or apply their own
        // fallback (e.g. broad recall).
        let query_text = request.query_text.clone().unwrap_or_default();

        // Resolve the collection within the request tenant before either leg.
        match &request.scope {
            RankScope::Tenant(tenant) if !tenant.is_empty() => {
                self.backend
                    .scope_collection(&request.collection, tenant)
                    .await?;
            }
            RankScope::Internal => {}
            RankScope::Tenant(_) | RankScope::Unset => {
                return Err(RankError::CollectionNotFound(request.collection.clone()));
            }
        }

        let search = self.coordinator.execute_hybrid_search(
            move |q: String| async move {
                backend_bm25
                    .bm25_search(&collection_for_bm25, &q)
                    .await
                    .map_err(|e| anyhow::anyhow!("{e}"))
            },
            move |v: Vec<f32>| async move {
                backend_vec
                    .vector_search(&collection_for_vec, &v)
                    .await
                    // Keep the typed `RankError` in the chain so a
                    // client-argument error survives the coordinator.
                    .map_err(anyhow::Error::new)
            },
            &query_text,
            &request.query_vector,
        );
        // Both legs run in the request's scope, so a tenant-less identity
        // inside a backend resolves within the same tenant the check used.
        let results = match &request.scope {
            RankScope::Tenant(tenant) => {
                crate::query::request_scope::with_request_tenant(tenant.clone(), search).await
            }
            _ => crate::query::request_scope::with_internal_scope("rank internal", search).await,
        };
        let results = results.map_err(|e| {
            let client_argument = e.chain().find_map(|cause| match cause.downcast_ref() {
                Some(RankError::InvalidArgument(message)) => Some(message.clone()),
                _ => None,
            });
            match client_argument {
                Some(message) => RankError::InvalidArgument(message),
                None => RankError::ModelInference {
                    model_id: "hybrid_coordinator".into(),
                    reason: format!("hybrid search failed: {e}"),
                },
            }
        })?;

        // R-7c.3.2: build CandidateBatch with sequential synthetic
        // DocHandles + the original string ids in `original_ids`. This
        // preserves the backend's ids exactly for the wire response,
        // regardless of whether they're numeric.
        let mut docs = Vec::with_capacity(results.len());
        let mut original_ids: HashMap<DocHandle, std::sync::Arc<str>> =
            HashMap::with_capacity(results.len());
        for (i, r) in results.into_iter().enumerate() {
            let handle = DocHandle((i + 1) as u32);
            docs.push(handle);
            original_ids.insert(handle, std::sync::Arc::from(r.doc_id.as_str()));
        }
        Ok(CandidateBatch {
            docs,
            original_ids: Some(original_ids),
        })
    }
}

/// Plain-Rust route dispatcher (no axum extractors).
///
/// Reads `rank_services` off [`crate::network::rest::canonical::handlers::AppState`]
/// and routes through [`handle_rank_search`]. Maps `RankError` to
/// `ApiError`. The thin axum wrapper lives in `handlers.rs` next to
/// the router registration — this avoids cross-module trait-resolution
/// trouble where the dep graph holds both axum 0.6 and 0.8 (tonic
/// transitively pulls 0.8) and a cross-module handler ends up
/// satisfying only 0.8's `Handler` blanket.
pub async fn rank_search_dispatch(
    app_state: crate::network::rest::canonical::handlers::AppState,
    req: RankSearchRequest,
) -> ApiResult<RankSearchResponse> {
    let services = app_state.rank_services.as_ref().ok_or_else(|| {
        ApiError::NotImplemented(
            "rank services not configured — server started without RankServices injection".into(),
        )
    })?;

    // Resolve the per-profile second-phase scorer up-front. Done in the
    // async outer scope (cheap DashMap lookup) so the inner
    // spawn_blocking closure receives an owned Option<Arc<…>>.
    let second_phase_scorer = req
        .rank_profile
        .as_deref()
        .and_then(|name| services.second_phase_scorer(name));

    handle_rank_search_with_config(
        req,
        services.profile_registry.as_ref(),
        services.candidate_provider.as_ref(),
        services.blueprint_factory.clone(),
        second_phase_scorer,
        services.rerank_config.clone(),
    )
    .await
    .map_err(rank_error_to_api)
}

/// The rank route's error contract: profile-not-found → 404, a client's own
/// argument (an invalid profile, or e.g. a wrong-dimension query vector,
/// TD-SECSCAN-1 §5) → 400, anything else → 500.
fn rank_error_to_api(error: RankError) -> ApiError {
    match error {
        RankError::ProfileNotFound(name) => {
            ApiError::NotFound(format!("rank profile not found: {name}"))
        }
        RankError::InvalidProfile(msg) => ApiError::InvalidArgument(msg),
        RankError::InvalidArgument(msg) => ApiError::InvalidArgument(msg),
        RankError::CollectionNotFound(collection) => ApiError::CollectionNotFound(collection),
        other => ApiError::Internal(format!("rank pipeline failed: {other}")),
    }
}

/// Lift an optional `Arc<[(Arc<str>, f32)]>` feature snapshot into the
/// wire-form `HashMap<String, f64>`. Used for both `match_features`
/// (R-7c.5) and `summary_features` (R-7c.5b) on each `ScoredHitDto`.
/// Empty Arc → empty map (the wire DTO's `skip_serializing_if` then
/// omits the field from JSON; the proto wire encodes a zero-length
/// map field which costs one tag byte).
fn arc_features_to_wire_map(
    snapshot: Option<&std::sync::Arc<[(std::sync::Arc<str>, f32)]>>,
) -> HashMap<String, f64> {
    snapshot
        .map(|arr| {
            arr.iter()
                .map(|(name, value)| (name.to_string(), *value as f64))
                .collect::<HashMap<String, f64>>()
        })
        .unwrap_or_default()
}

/// The production cross-modal default: ENABLED with score-preserving
/// missing-score policy. Deliberately different from `RerankConfig::default()`
/// (disabled) — that one is the "no heuristic policy unless configured" default
/// for config-file construction, this one is what the rank pipeline runs when a
/// profile asks for `strategy = "cross_modal"`. Public so server startup can
/// use it as the fallback when `[query.reranking]` is absent.
pub fn default_rerank_config() -> RerankConfig {
    use proximadb_query::reranking::{MissingScorePolicy, ModelWeightConfig};
    RerankConfig {
        enabled: true,
        semantic_rerank: false,
        diversity_optimization: false,
        context_aware: false,
        missing_score: MissingScorePolicy::Preserve,
        model_weights: ModelWeightConfig::default(),
        ..RerankConfig::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_kernel::PhaseId;
    use proximadb_rank_core::{Blueprint, FeatureExecutor, FeatureLookup, OutputSpec, PhaseConfig};
    use proximadb_rank_features::register_builtins;
    use proximadb_rank_profile::{CompiledRankProfile, PhaseSpec, RankProfileSpec};

    // ---------------- DTO round-trip tests ----------------

    #[test]
    fn request_round_trips_through_json() {
        let req = RankSearchRequest {
            scope: RankScope::Internal,
            collection: "docs".into(),
            query_vector: vec![0.1, 0.2, 0.3],
            query_text: None,
            k: 50,
            rank_profile: Some("semantic_plus_ce".into()),
            rank_overrides: Some(RankOverrides {
                second_phase: Some(PhaseOverride {
                    rerank_count: Some(200),
                    batch_size: Some(32),
                }),
                global_phase: None,
            }),
        };
        let j = serde_json::to_string(&req).unwrap();
        let back: RankSearchRequest = serde_json::from_str(&j).unwrap();
        assert_eq!(back.collection, req.collection);
        assert_eq!(back.k, 50);
        assert_eq!(back.rank_profile.as_deref(), Some("semantic_plus_ce"));
        assert_eq!(
            back.rank_overrides
                .as_ref()
                .unwrap()
                .second_phase
                .as_ref()
                .unwrap()
                .rerank_count,
            Some(200)
        );
    }

    #[test]
    fn request_omits_optionals_from_json_when_unset() {
        let req = RankSearchRequest {
            scope: RankScope::Internal,
            collection: "docs".into(),
            query_vector: vec![],
            query_text: None,
            k: 10,
            rank_profile: None,
            rank_overrides: None,
        };
        let j = serde_json::to_string(&req).unwrap();
        assert!(!j.contains("rank_profile"));
        assert!(!j.contains("rank_overrides"));
    }

    #[test]
    fn response_round_trips_through_json() {
        let resp = RankSearchResponse {
            hits: vec![ScoredHitDto {
                id: "doc_42".into(),
                score: 0.876,
                score_vector: Some(ScoreVectorDto {
                    primary: 0.876,
                    phase: 2,
                    components: vec![ScoreComponent {
                        name: "bm25(title)".into(),
                        value: 12.4,
                        weight: 0.4,
                        contribution: 4.96,
                    }],
                }),
                match_features: HashMap::new(),
                summary_features: HashMap::new(),
            }],
            phase_truncated: false,
            rank_profile: Some("semantic_plus_ce".into()),
            rank_profile_version: Some(7),
        };
        let j = serde_json::to_string(&resp).unwrap();
        let back: RankSearchResponse = serde_json::from_str(&j).unwrap();
        assert_eq!(back, resp);
    }

    #[test]
    fn response_omits_score_vector_when_none() {
        let resp = RankSearchResponse {
            hits: vec![ScoredHitDto {
                id: "doc_1".into(),
                score: 0.5,
                score_vector: None,
                match_features: HashMap::new(),
                summary_features: HashMap::new(),
            }],
            phase_truncated: false,
            rank_profile: None,
            rank_profile_version: None,
        };
        let j = serde_json::to_string(&resp).unwrap();
        assert!(
            !j.contains("score_vector"),
            "score_vector must be omitted when None: {j}"
        );
    }

    #[test]
    fn score_vector_dto_from_kernel() {
        let sv = ScoreVector::new(
            0.5,
            PhaseId::GLOBAL,
            vec![ScoreComponent {
                name: "x".into(),
                value: 1.0,
                weight: 1.0,
                contribution: 1.0,
            }],
        );
        let dto = ScoreVectorDto::from(&sv);
        assert_eq!(dto.primary, 0.5);
        assert_eq!(dto.phase, 2); // GLOBAL = 2
        assert_eq!(dto.components.len(), 1);
    }

    // ---------------- Handler tests ----------------

    struct DocIdExec;
    impl FeatureExecutor for DocIdExec {
        fn execute(
            &mut self,
            doc: DocHandle,
            _lookup: &mut dyn FeatureLookup,
            _ctx: &mut ScoreCtx<'_>,
        ) -> f32 {
            doc.0 as f32
        }
    }

    /// Test-only blueprint: returns the doc id as the score. Used in
    /// place of `bm25(...)` etc. so the handler can compile a profile
    /// without needing real candidate data.
    struct DocIdBp;
    impl Blueprint for DocIdBp {
        fn name(&self) -> &str {
            "docid"
        }
        fn declared_outputs(&self) -> &[OutputSpec] {
            &[]
        }
        fn build_executor(
            &self,
            _cfg: &PhaseConfig,
            _q: &QueryContext,
        ) -> RankResult<Box<dyn FeatureExecutor>> {
            Ok(Box::new(DocIdExec))
        }
    }

    /// Mock candidate provider returning a fixed range of DocHandles.
    struct FixedCandidates(Vec<DocHandle>);
    #[async_trait::async_trait]
    impl CandidateProvider for FixedCandidates {
        async fn candidates(&self, _request: &RankSearchRequest) -> RankResult<CandidateBatch> {
            Ok(CandidateBatch::from_docs(self.0.clone()))
        }
    }

    fn factory_with_docid() -> Arc<BlueprintFactory> {
        let f = Arc::new(BlueprintFactory::new());
        register_builtins(&f);
        f.register(Arc::new(DocIdBp));
        f
    }

    fn install_profile(reg: &ProfileRegistry, factory: Arc<BlueprintFactory>, name: &str) {
        let mut spec = RankProfileSpec::new(name);
        spec.first_phase = Some(PhaseSpec {
            expression: "docid()".into(),
            heap_size: Some(50),
            rerank_count: None,
            batch_size: None,
        });
        spec.version = 1;
        let compiled = CompiledRankProfile::compile(spec, factory).unwrap();
        reg.install(compiled);
    }

    // ---------------- R-7c.5: match_features capture in REST DTO ----------------

    fn install_profile_with_match_features(
        reg: &ProfileRegistry,
        factory: Arc<BlueprintFactory>,
        name: &str,
        match_features: Vec<&str>,
    ) {
        let mut spec = RankProfileSpec::new(name);
        spec.first_phase = Some(PhaseSpec {
            expression: "docid()".into(),
            heap_size: Some(50),
            rerank_count: None,
            batch_size: None,
        });
        spec.match_features = match_features.into_iter().map(String::from).collect();
        spec.version = 1;
        let compiled = CompiledRankProfile::compile(spec, factory).unwrap();
        reg.install(compiled);
    }

    #[tokio::test]
    async fn handler_populates_match_features_when_profile_declares_them() {
        // Profile declares 2 distinct match_features expressions. Each
        // hit's wire `match_features` map carries both, keyed by the
        // expression string (Vespa-style — declared expression IS the
        // name). Two declarations of the same expression would collapse
        // to one wire key because HashMap dedupes by key — production
        // profiles should use unique expressions for distinct columns.
        let registry = ProfileRegistry::new();
        let factory = factory_with_docid();
        install_profile_with_match_features(
            &registry,
            factory.clone(),
            "with_mf",
            // Two different built-in expressions, both pure-function so
            // they resolve without per-doc attribute data.
            vec!["docid()", "1.0"],
        );

        let candidates = FixedCandidates(vec![DocHandle(5), DocHandle(2)]);
        let req = RankSearchRequest {
            scope: RankScope::Internal,
            collection: "docs".into(),
            query_vector: vec![],
            query_text: None,
            k: 2,
            rank_profile: Some("with_mf".into()),
            rank_overrides: None,
        };
        let resp = handle_rank_search(req, &registry, &candidates, factory, None)
            .await
            .unwrap();
        assert_eq!(resp.hits.len(), 2);
        for h in &resp.hits {
            assert_eq!(
                h.match_features.len(),
                2,
                "each hit must carry both declared match_features"
            );
            assert!(h.match_features.contains_key("docid()"));
            assert!(h.match_features.contains_key("1.0"));
        }
        // Per-doc value: docid() returns doc id as f32 → doc 5 → 5.0,
        // doc 2 → 2.0. Constant "1.0" is the same on every doc.
        assert_eq!(resp.hits[0].id, "5");
        assert!((resp.hits[0].match_features["docid()"] - 5.0).abs() < 1e-5);
        assert!((resp.hits[0].match_features["1.0"] - 1.0).abs() < 1e-5);
        assert!((resp.hits[1].match_features["docid()"] - 2.0).abs() < 1e-5);
        assert!((resp.hits[1].match_features["1.0"] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn rank_services_install_profile_increments_reload_counter() {
        // R-7c.4d follow-up: RankServices::install_profile is the
        // hot-reload entry point per spec §4.10. Verify the counter
        // fires with `outcome="ok"` for each successful install and
        // `record_profile_reload_error` bumps the error variant.
        use crate::observability::rank_metrics::RankPipelineMetrics;
        let prom_registry = prometheus::Registry::new();
        let metrics = Arc::new(RankPipelineMetrics::register(&prom_registry).unwrap());
        let candidates: Arc<dyn CandidateProvider> =
            Arc::new(MockRangeCandidateProvider::default());
        let services = RankServices::new(candidates).with_metrics(metrics.clone());

        let factory = factory_with_docid();
        let mut spec = RankProfileSpec::new("p1");
        spec.first_phase = Some(PhaseSpec {
            expression: "docid()".into(),
            heap_size: Some(10),
            rerank_count: None,
            batch_size: None,
        });
        spec.version = 1;
        let compiled = CompiledRankProfile::compile(spec, factory).unwrap();

        // First install → ok counter increments to 1.
        services.install_profile(compiled.clone());
        let ok_cnt = prom_registry
            .gather()
            .into_iter()
            .filter(|mf| mf.name() == "proximadb_rank_profile_reload_total")
            .flat_map(|mf| mf.get_metric().to_vec())
            .filter(|m| {
                m.get_label()
                    .iter()
                    .any(|l| l.name() == "profile" && l.value() == "p1")
                    && m.get_label()
                        .iter()
                        .any(|l| l.name() == "outcome" && l.value() == "ok")
            })
            .map(|m| m.get_counter().value())
            .sum::<f64>();
        assert!((ok_cnt - 1.0).abs() < f64::EPSILON);

        // Reload (same name) → ok counter increments to 2 — verifies
        // the hot-reload path counts the swap.
        services.install_profile(compiled);
        let ok_cnt2 = prom_registry
            .gather()
            .into_iter()
            .filter(|mf| mf.name() == "proximadb_rank_profile_reload_total")
            .flat_map(|mf| mf.get_metric().to_vec())
            .filter(|m| {
                m.get_label()
                    .iter()
                    .any(|l| l.name() == "profile" && l.value() == "p1")
                    && m.get_label()
                        .iter()
                        .any(|l| l.name() == "outcome" && l.value() == "ok")
            })
            .map(|m| m.get_counter().value())
            .sum::<f64>();
        assert!((ok_cnt2 - 2.0).abs() < f64::EPSILON);

        // Error path: caller surfaces a rejected install.
        services.record_profile_reload_error("p1");
        let err_cnt = prom_registry
            .gather()
            .into_iter()
            .filter(|mf| mf.name() == "proximadb_rank_profile_reload_total")
            .flat_map(|mf| mf.get_metric().to_vec())
            .filter(|m| {
                m.get_label()
                    .iter()
                    .any(|l| l.name() == "profile" && l.value() == "p1")
                    && m.get_label()
                        .iter()
                        .any(|l| l.name() == "outcome" && l.value() == "error")
            })
            .map(|m| m.get_counter().value())
            .sum::<f64>();
        assert!((err_cnt - 1.0).abs() < f64::EPSILON);
    }

    fn install_profile_with_summary_features(
        reg: &ProfileRegistry,
        factory: Arc<BlueprintFactory>,
        name: &str,
        summary_features: Vec<&str>,
    ) {
        let mut spec = RankProfileSpec::new(name);
        spec.first_phase = Some(PhaseSpec {
            expression: "docid()".into(),
            heap_size: Some(50),
            rerank_count: None,
            batch_size: None,
        });
        spec.summary_features = summary_features.into_iter().map(String::from).collect();
        spec.version = 1;
        let compiled = CompiledRankProfile::compile(spec, factory).unwrap();
        reg.install(compiled);
    }

    #[tokio::test]
    async fn handler_populates_summary_features_when_profile_declares_them() {
        // R-7c.5b: profile declares summary_features; the per-doc values
        // surface on the wire under ScoredHitDto.summary_features
        // (distinct from match_features even though the lowering
        // machinery is shared).
        let registry = ProfileRegistry::new();
        let factory = factory_with_docid();
        install_profile_with_summary_features(
            &registry,
            factory.clone(),
            "with_sf",
            vec!["docid()", "1.0"],
        );
        let candidates = FixedCandidates(vec![DocHandle(7), DocHandle(4)]);
        let req = RankSearchRequest {
            scope: RankScope::Internal,
            collection: "docs".into(),
            query_vector: vec![],
            query_text: None,
            k: 2,
            rank_profile: Some("with_sf".into()),
            rank_overrides: None,
        };
        let resp = handle_rank_search(req, &registry, &candidates, factory, None)
            .await
            .unwrap();
        assert_eq!(resp.hits.len(), 2);
        for h in &resp.hits {
            assert!(h.match_features.is_empty(), "no match_features declared");
            assert_eq!(h.summary_features.len(), 2);
            assert!(h.summary_features.contains_key("docid()"));
            assert!(h.summary_features.contains_key("1.0"));
        }
        assert_eq!(resp.hits[0].id, "7");
        assert!((resp.hits[0].summary_features["docid()"] - 7.0).abs() < 1e-5);
        assert!((resp.hits[1].summary_features["docid()"] - 4.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn handler_with_metrics_emits_per_feature_histogram_observations() {
        // R-7c.4d end-to-end: with metrics wired, a single-feature
        // profile (`docid()`) + 3 candidates must produce 3
        // histogram observations on (profile=metrics_test,
        // phase=first, feature=docid()) — one per
        // capture_feature_snapshot call on the cache-miss path.
        // (Cache hits are deliberately not timed so the histogram
        // reflects real executor work, not memoization touches.)
        use crate::observability::rank_metrics::RankPipelineMetrics;
        let registry = ProfileRegistry::new();
        let factory = factory_with_docid();
        let mut spec = RankProfileSpec::new("metrics_test");
        spec.first_phase = Some(PhaseSpec {
            expression: "docid()".into(),
            heap_size: Some(10),
            rerank_count: None,
            batch_size: None,
        });
        spec.match_features = vec!["docid()".into()];
        spec.version = 1;
        let compiled = CompiledRankProfile::compile(spec, factory.clone()).unwrap();
        registry.install(compiled);

        let prom_registry = prometheus::Registry::new();
        let metrics = std::sync::Arc::new(RankPipelineMetrics::register(&prom_registry).unwrap());

        let candidates = FixedCandidates(vec![DocHandle(1), DocHandle(2), DocHandle(3)]);
        let req = RankSearchRequest {
            scope: RankScope::Internal,
            collection: "docs".into(),
            query_vector: vec![],
            query_text: None,
            k: 3,
            rank_profile: Some("metrics_test".into()),
            rank_overrides: None,
        };
        let resp = handle_rank_search_with_metrics(
            req,
            &registry,
            &candidates,
            factory,
            None,
            default_rerank_config(),
            Some(metrics.clone()),
        )
        .await
        .unwrap();
        assert_eq!(resp.hits.len(), 3);
        for h in &resp.hits {
            assert_eq!(h.match_features.len(), 1);
            assert!(h.match_features.contains_key("docid()"));
        }

        // Real observation assertion now that rank-core wires the
        // per-feature timing emit. Score-expression timing is a
        // separate concern (the score idx has no symbolic name yet);
        // the spec's `feature=<expr>` label promise only covers
        // declared match/summary features today.
        let observed = prom_registry
            .gather()
            .into_iter()
            .filter(|mf| mf.name() == "proximadb_rank_feature_latency_us")
            .flat_map(|mf| mf.get_metric().to_vec())
            .filter(|m| {
                m.get_label()
                    .iter()
                    .any(|l| l.name() == "profile" && l.value() == "metrics_test")
                    && m.get_label()
                        .iter()
                        .any(|l| l.name() == "phase" && l.value() == "first")
                    && m.get_label()
                        .iter()
                        .any(|l| l.name() == "feature" && l.value() == "docid()")
            })
            .map(|m| m.get_histogram().get_sample_count())
            .sum::<u64>();
        assert!(
            observed >= 3,
            "expected ≥3 histogram observations for first-phase docid() across 3 candidates, got {observed}",
        );
    }

    #[tokio::test]
    async fn handler_with_metrics_emits_second_phase_observation_when_scorer_runs() {
        // R-7c.4d follow-up: when a SecondPhaseScorer is registered
        // AND the profile has a `second_phase` configured AND
        // metrics are wired, the orchestrator must emit one
        // observation on (profile, phase=second, feature=second_phase)
        // for the rescore call. The bridge can't reach inside the
        // scorer (it'd need a trait change), so this is a single
        // wrapper-level observation per rescore — sufficient for
        // dashboards to alert on second-phase latency.
        use crate::observability::rank_metrics::RankPipelineMetrics;
        let registry = ProfileRegistry::new();
        let factory = factory_with_docid();
        let mut spec = RankProfileSpec::new("metrics_test_2p");
        spec.first_phase = Some(PhaseSpec {
            expression: "docid()".into(),
            heap_size: Some(10),
            rerank_count: Some(5),
            batch_size: None,
        });
        spec.second_phase = Some(PhaseSpec {
            expression: "docid()".into(),
            heap_size: Some(5),
            rerank_count: None,
            batch_size: None,
        });
        spec.version = 1;
        let compiled = CompiledRankProfile::compile(spec, factory.clone()).unwrap();
        registry.install(compiled);

        let prom_registry = prometheus::Registry::new();
        let metrics = std::sync::Arc::new(RankPipelineMetrics::register(&prom_registry).unwrap());
        let scorer: Arc<dyn SecondPhaseScorer> =
            Arc::new(proximadb_rank_core::PassthroughSecondPhaseScorer);

        let candidates = FixedCandidates(vec![DocHandle(1), DocHandle(2)]);
        let req = RankSearchRequest {
            scope: RankScope::Internal,
            collection: "docs".into(),
            query_vector: vec![],
            query_text: None,
            k: 2,
            rank_profile: Some("metrics_test_2p".into()),
            rank_overrides: None,
        };
        let resp = handle_rank_search_with_metrics(
            req,
            &registry,
            &candidates,
            factory,
            Some(scorer),
            default_rerank_config(),
            Some(metrics.clone()),
        )
        .await
        .unwrap();
        assert_eq!(resp.hits.len(), 2);

        let observed = prom_registry
            .gather()
            .into_iter()
            .filter(|mf| mf.name() == "proximadb_rank_feature_latency_us")
            .flat_map(|mf| mf.get_metric().to_vec())
            .filter(|m| {
                m.get_label()
                    .iter()
                    .any(|l| l.name() == "profile" && l.value() == "metrics_test_2p")
                    && m.get_label()
                        .iter()
                        .any(|l| l.name() == "phase" && l.value() == "second")
                    && m.get_label()
                        .iter()
                        .any(|l| l.name() == "feature" && l.value() == "second_phase")
            })
            .map(|m| m.get_histogram().get_sample_count())
            .sum::<u64>();
        assert_eq!(
            observed, 1,
            "expected exactly 1 second-phase observation per rescore call, got {observed}",
        );
    }

    #[tokio::test]
    async fn handler_populates_both_match_and_summary_features_independently() {
        // Both declared on the same profile → both wire maps populated,
        // each keyed by its own expression. The lowering machinery is
        // shared but the wire shape (and downstream semantics — Vespa
        // separates match from summary deliberately) keeps them
        // distinct.
        let registry = ProfileRegistry::new();
        let factory = factory_with_docid();
        let mut spec = RankProfileSpec::new("both");
        spec.first_phase = Some(PhaseSpec {
            expression: "docid()".into(),
            heap_size: Some(50),
            rerank_count: None,
            batch_size: None,
        });
        spec.match_features = vec!["docid()".into()];
        spec.summary_features = vec!["1.0".into()];
        spec.version = 1;
        let compiled = CompiledRankProfile::compile(spec, factory.clone()).unwrap();
        registry.install(compiled);

        let candidates = FixedCandidates(vec![DocHandle(11)]);
        let req = RankSearchRequest {
            scope: RankScope::Internal,
            collection: "docs".into(),
            query_vector: vec![],
            query_text: None,
            k: 1,
            rank_profile: Some("both".into()),
            rank_overrides: None,
        };
        let resp = handle_rank_search(req, &registry, &candidates, factory, None)
            .await
            .unwrap();
        let h = &resp.hits[0];
        assert_eq!(h.match_features.len(), 1);
        assert!((h.match_features["docid()"] - 11.0).abs() < 1e-5);
        assert_eq!(h.summary_features.len(), 1);
        assert!((h.summary_features["1.0"] - 1.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn handler_with_no_match_features_emits_empty_map() {
        // NFR-9 fast path: profile without match_features → hits carry
        // an empty `match_features` map (not None — wire shape is HashMap
        // which serializes to `{}` or is skipped via skip_serializing_if).
        let registry = ProfileRegistry::new();
        let factory = factory_with_docid();
        install_profile(&registry, factory.clone(), "no_mf");

        let candidates = FixedCandidates(vec![DocHandle(1)]);
        let req = RankSearchRequest {
            scope: RankScope::Internal,
            collection: "docs".into(),
            query_vector: vec![],
            query_text: None,
            k: 1,
            rank_profile: Some("no_mf".into()),
            rank_overrides: None,
        };
        let resp = handle_rank_search(req, &registry, &candidates, factory, None)
            .await
            .unwrap();
        assert_eq!(resp.hits.len(), 1);
        assert!(
            resp.hits[0].match_features.is_empty(),
            "profile without match_features must emit empty wire map"
        );
    }

    #[tokio::test]
    async fn handler_with_no_profile_returns_retrieval_only() {
        let registry = ProfileRegistry::new();
        let candidates = FixedCandidates(vec![DocHandle(1), DocHandle(2), DocHandle(3)]);
        let factory = factory_with_docid();
        let req = RankSearchRequest {
            scope: RankScope::Internal,
            collection: "docs".into(),
            query_vector: vec![],
            query_text: None,
            k: 10,
            rank_profile: None,
            rank_overrides: None,
        };
        let resp = handle_rank_search(req, &registry, &candidates, factory, None)
            .await
            .unwrap();
        assert_eq!(resp.hits.len(), 3);
        assert!(resp.rank_profile.is_none());
        for h in &resp.hits {
            assert!(h.score_vector.is_none());
        }
    }

    #[tokio::test]
    async fn handler_with_unknown_profile_errors_with_profile_not_found() {
        let registry = ProfileRegistry::new();
        let candidates = FixedCandidates(vec![DocHandle(1)]);
        let factory = factory_with_docid();
        let req = RankSearchRequest {
            scope: RankScope::Internal,
            collection: "docs".into(),
            query_vector: vec![],
            query_text: None,
            k: 10,
            rank_profile: Some("ghost".into()),
            rank_overrides: None,
        };
        match handle_rank_search(req, &registry, &candidates, factory, None).await {
            Err(RankError::ProfileNotFound(name)) => assert_eq!(name, "ghost"),
            Err(_) => panic!("expected ProfileNotFound, got a different RankError"),
            Ok(_) => panic!("expected error"),
        }
    }

    #[tokio::test]
    async fn handler_with_profile_returns_ranked_hits_with_score_vector() {
        let registry = ProfileRegistry::new();
        let factory = factory_with_docid();
        install_profile(&registry, factory.clone(), "test");

        let candidates = FixedCandidates(vec![DocHandle(1), DocHandle(2), DocHandle(3)]);
        let req = RankSearchRequest {
            scope: RankScope::Internal,
            collection: "docs".into(),
            query_vector: vec![],
            query_text: None,
            k: 3,
            rank_profile: Some("test".into()),
            rank_overrides: None,
        };
        let resp = handle_rank_search(req, &registry, &candidates, factory, None)
            .await
            .unwrap();
        assert_eq!(resp.hits.len(), 3);
        assert_eq!(resp.rank_profile.as_deref(), Some("test"));
        assert_eq!(resp.rank_profile_version, Some(1));
        // docid scorer: top → doc 3 → 2 → 1
        assert_eq!(resp.hits[0].id, "3");
        assert!((resp.hits[0].score - 3.0).abs() < 1e-5);
        // Every hit carries a score_vector when a profile is attached.
        for h in &resp.hits {
            assert!(h.score_vector.is_some());
        }
    }

    #[tokio::test]
    async fn handler_truncates_to_k() {
        let registry = ProfileRegistry::new();
        let factory = factory_with_docid();
        install_profile(&registry, factory.clone(), "test");

        let candidates = FixedCandidates((1..=20).map(DocHandle).collect());
        let req = RankSearchRequest {
            scope: RankScope::Internal,
            collection: "docs".into(),
            query_vector: vec![],
            query_text: None,
            k: 5,
            rank_profile: Some("test".into()),
            rank_overrides: None,
        };
        let resp = handle_rank_search(req, &registry, &candidates, factory, None)
            .await
            .unwrap();
        assert_eq!(resp.hits.len(), 5);
    }

    #[tokio::test]
    async fn handler_round_trips_query_vector_into_qctx() {
        // The vector itself isn't used by the docid scorer, but the
        // handler must propagate it so downstream features (closeness,
        // cosine — R-7c) can consume it. Verify the request shape
        // accepts and threads the vector through.
        let registry = ProfileRegistry::new();
        let factory = factory_with_docid();
        install_profile(&registry, factory.clone(), "test");

        let candidates = FixedCandidates(vec![DocHandle(1)]);
        let req = RankSearchRequest {
            scope: RankScope::Internal,
            collection: "docs".into(),
            query_vector: vec![0.5; 384],
            query_text: None,
            k: 1,
            rank_profile: Some("test".into()),
            rank_overrides: None,
        };
        let resp = handle_rank_search(req, &registry, &candidates, factory, None)
            .await
            .unwrap();
        assert_eq!(resp.hits.len(), 1);
    }

    // ---------------- Production-wiring tests (R-7c) ----------------

    #[test]
    fn rank_services_new_pre_populates_builtins() {
        let services = RankServices::new(Arc::new(MockRangeCandidateProvider::default()));
        // R-2 features must be available on a freshly-constructed
        // RankServices so callers don't have to remember
        // register_builtins() at injection time.
        for name in ["attribute", "closeness", "bm25", "freshness", "decay"] {
            assert!(
                services.blueprint_factory.lookup(name).is_some(),
                "expected built-in '{name}' to be registered"
            );
        }
        assert!(services.profile_registry.is_empty());
    }

    #[tokio::test]
    async fn mock_range_candidate_provider_returns_configured_count() {
        let p = MockRangeCandidateProvider { count: 7 };
        let req = RankSearchRequest {
            scope: RankScope::Internal,
            collection: "x".into(),
            query_vector: vec![],
            query_text: None,
            k: 5,
            rank_profile: None,
            rank_overrides: None,
        };
        let docs = p.candidates(&req).await.unwrap().docs;
        assert_eq!(docs.len(), 7);
        assert_eq!(docs[0], DocHandle(1));
        assert_eq!(docs[6], DocHandle(7));
    }

    #[test]
    fn mock_range_default_is_twenty() {
        let p = MockRangeCandidateProvider::default();
        assert_eq!(p.count, 20);
    }

    // NOTE: an axum-level integration test for `rank_search_route` would
    // need a full `AppState`, which requires `SharedServices` construction
    // (storage, catalog, graph, queue, …). That setup is heavyweight and
    // duplicated of the route fixture already exercised by
    // `tests/r7c_route_smoke.rs` (R-7c.1 follow-up: stand up that
    // fixture against the real router and assert 200/404/503 over HTTP).
    // The direct `handle_rank_search` tests above cover the dispatch
    // logic; the axum binding is the trivial `Json(req)` →
    // `handle_rank_search` plumbing in `rank_search_route` plus error
    // mapping, which clippy + the build itself verify.

    #[tokio::test]
    async fn handler_global_phase_k_override_widens_result() {
        // Override the response size beyond the request k via
        // rank_overrides.global_phase.rerank_count. Verifies the
        // override surface is honored.
        let registry = ProfileRegistry::new();
        let factory = factory_with_docid();
        install_profile(&registry, factory.clone(), "test");

        let candidates = FixedCandidates((1..=20).map(DocHandle).collect());
        let req = RankSearchRequest {
            scope: RankScope::Internal,
            collection: "docs".into(),
            query_vector: vec![],
            query_text: None,
            k: 5,
            rank_profile: Some("test".into()),
            rank_overrides: Some(RankOverrides {
                second_phase: None,
                global_phase: Some(PhaseOverride {
                    rerank_count: Some(12),
                    batch_size: None,
                }),
            }),
        };
        let resp = handle_rank_search(req, &registry, &candidates, factory, None)
            .await
            .unwrap();
        // Without global scorer this profile has, override flows
        // through to the orchestrator's topk arg → 12 hits returned.
        assert_eq!(resp.hits.len(), 12);
    }

    // ---------------- R-7c.2: second-phase wiring tests ----------------

    fn install_profile_with_second_phase(
        reg: &ProfileRegistry,
        factory: Arc<BlueprintFactory>,
        name: &str,
    ) {
        let mut spec = RankProfileSpec::new(name);
        spec.first_phase = Some(PhaseSpec {
            expression: "docid()".into(),
            heap_size: Some(50),
            rerank_count: Some(3),
            batch_size: None,
        });
        spec.second_phase = Some(PhaseSpec {
            // The expression here isn't actually parsed by the scorer
            // (the scorer is supplied externally per R-7c.2); it just
            // marks "this profile has a second phase configured".
            expression: "1.0".into(),
            heap_size: None,
            rerank_count: Some(3),
            batch_size: None,
        });
        spec.version = 1;
        let compiled = CompiledRankProfile::compile(spec, factory).unwrap();
        reg.install(compiled);
    }

    #[tokio::test]
    async fn handler_with_no_scorer_passes_through_second_phase() {
        // Profile has second_phase, but no scorer is supplied → first-phase
        // hits flow through unchanged with PhaseId::FIRST preserved.
        let registry = ProfileRegistry::new();
        let factory = factory_with_docid();
        install_profile_with_second_phase(&registry, factory.clone(), "two_phase");

        let candidates = FixedCandidates((1..=5).map(DocHandle).collect());
        let req = RankSearchRequest {
            scope: RankScope::Internal,
            collection: "docs".into(),
            query_vector: vec![],
            query_text: None,
            k: 5,
            rank_profile: Some("two_phase".into()),
            rank_overrides: None,
        };
        let resp = handle_rank_search(req, &registry, &candidates, factory, None)
            .await
            .unwrap();
        assert_eq!(resp.hits.len(), 5);
        // No second-phase scorer ran → score_vector tags FIRST.
        for h in &resp.hits {
            let sv = h.score_vector.as_ref().unwrap();
            assert_eq!(sv.phase, 0, "expected PhaseId::FIRST (0)");
        }
    }

    fn install_profile_with_cross_modal(
        reg: &ProfileRegistry,
        factory: Arc<BlueprintFactory>,
        name: &str,
    ) {
        use proximadb_rank_profile::GlobalPhaseSpec;
        let mut spec = RankProfileSpec::new(name);
        spec.first_phase = Some(PhaseSpec {
            expression: "docid()".into(),
            heap_size: Some(50),
            rerank_count: Some(5),
            batch_size: None,
        });
        spec.global_phase = Some(GlobalPhaseSpec {
            strategy: "cross_modal".into(),
            rerank_count: None,
            config: serde_json::Value::Null,
        });
        spec.version = 1;
        let compiled = CompiledRankProfile::compile(spec, factory).unwrap();
        reg.install(compiled);
    }

    /// TD-SELECTOR-1 gate 3 config wiring: `handle_rank_search_with_config`
    /// must reach the constructed `CrossModalReranker` — before this plumbing,
    /// the handler hardcoded `default_rerank_config()` and `[query.reranking]`
    /// from the server config was inert. With `rerank_top_k = 2` and `k = 5`:
    /// the head of 2 is re-scored (docid ordering preserved here since the
    /// cross-modal base score is the input score), the tail of 3 survives in
    /// input order, and the response carries all 5 hits.
    #[tokio::test]
    async fn handler_respects_rerank_config_head_bound_and_preserves_tail() {
        let registry = ProfileRegistry::new();
        let factory = factory_with_docid();
        install_profile_with_cross_modal(&registry, factory.clone(), "cross_modal_cfg");

        let candidates = FixedCandidates((1..=5).map(DocHandle).collect());
        let req = RankSearchRequest {
            scope: RankScope::Internal,
            collection: "docs".into(),
            query_vector: vec![],
            query_text: None,
            k: 5,
            rank_profile: Some("cross_modal_cfg".into()),
            rank_overrides: None,
        };
        let rerank_config = RerankConfig {
            enabled: true,
            rerank_top_k: 2,
            semantic_rerank: false,
            diversity_optimization: false,
            context_aware: false,
            missing_score: proximadb_query::reranking::MissingScorePolicy::Preserve,
            ..RerankConfig::default()
        };
        let resp = handle_rank_search_with_config(
            req,
            &registry,
            &candidates,
            factory,
            None,
            rerank_config,
        )
        .await
        .unwrap();

        // k = 5 honored through the tail-preserving reranker.
        assert_eq!(resp.hits.len(), 5, "tail must survive to the k bound");
        let ids: Vec<&str> = resp.hits.iter().map(|h| h.id.as_str()).collect();
        assert_eq!(ids, vec!["5", "4", "3", "2", "1"]);
        // Head (2) re-scored → GLOBAL; tail (3) keeps its input phase.
        let phases: Vec<u8> = resp
            .hits
            .iter()
            .map(|h| h.score_vector.as_ref().map(|sv| sv.phase).unwrap_or(0))
            .collect();
        assert_eq!(phases, vec![2, 2, 0, 0, 0]);
    }

    #[tokio::test]
    async fn handler_with_scorer_runs_second_phase_and_rerank_top_k() {
        // Pass a ConstantMultiplier(0.1) — top-3 first-phase scores
        // (5, 4, 3) become (0.5, 0.4, 0.3); tail (2, 1) keeps first-
        // phase scores 2.0, 1.0. The rescored prefix remains ahead of
        // the tail because first/second score scales are incomparable.
        use proximadb_rank_core::ConstantMultiplierSecondPhaseScorer;

        let registry = ProfileRegistry::new();
        let factory = factory_with_docid();
        install_profile_with_second_phase(&registry, factory.clone(), "two_phase");

        let scorer: Arc<dyn SecondPhaseScorer> =
            Arc::new(ConstantMultiplierSecondPhaseScorer { factor: 0.1 });
        let candidates = FixedCandidates((1..=5).map(DocHandle).collect());
        let req = RankSearchRequest {
            scope: RankScope::Internal,
            collection: "docs".into(),
            query_vector: vec![],
            query_text: None,
            k: 5,
            rank_profile: Some("two_phase".into()),
            rank_overrides: None,
        };
        let resp = handle_rank_search(req, &registry, &candidates, factory, Some(scorer))
            .await
            .unwrap();
        assert_eq!(resp.hits.len(), 5);
        assert_eq!(resp.hits[0].id, "5");
        assert!((resp.hits[0].score - 0.5).abs() < 1e-5);
        assert_eq!(resp.hits[1].id, "4");
        assert_eq!(resp.hits[2].id, "3");
        assert_eq!(resp.hits[3].id, "2");
        assert_eq!(resp.hits[4].id, "1");
        // The reranked prefix carries PhaseId::SECOND; the appended tail keeps FIRST.
        let svs: Vec<u8> = resp
            .hits
            .iter()
            .map(|h| h.score_vector.as_ref().unwrap().phase)
            .collect();
        assert_eq!(svs, vec![1, 1, 1, 0, 0]);
    }

    #[test]
    fn rank_services_register_second_phase_scorer_round_trip() {
        use proximadb_rank_core::PassthroughSecondPhaseScorer;
        let services = RankServices::new(Arc::new(MockRangeCandidateProvider::default()));
        assert!(services.second_phase_scorer("nope").is_none());
        services.register_second_phase_scorer("p1", Arc::new(PassthroughSecondPhaseScorer));
        assert!(services.second_phase_scorer("p1").is_some());
        assert!(services.second_phase_scorer("nope").is_none());
    }

    #[test]
    fn rank_services_register_second_phase_scorer_last_write_wins() {
        use proximadb_rank_core::{
            ConstantMultiplierSecondPhaseScorer, PassthroughSecondPhaseScorer,
        };
        let services = RankServices::new(Arc::new(MockRangeCandidateProvider::default()));
        services.register_second_phase_scorer("p", Arc::new(PassthroughSecondPhaseScorer));
        services.register_second_phase_scorer(
            "p",
            Arc::new(ConstantMultiplierSecondPhaseScorer { factor: 2.0 }),
        );
        assert_eq!(services.second_phase_scorers.len(), 1);
        // Verify the active scorer is the multiplier (scales scores) not
        // passthrough (preserves scores).
        let s = services.second_phase_scorer("p").unwrap();
        let out = s
            .rescore(
                vec![ScoredHit::bare(
                    DocHandle(1),
                    5.0,
                    proximadb_kernel::PhaseId::FIRST,
                )],
                &QueryContext::default(),
            )
            .unwrap();
        assert_eq!(out[0].score, 10.0);
    }

    // ---------------- R-7c.3: HybridCoordinatorAdapter tests ----------------

    use crate::core::search::hybrid::{BM25Result, FusionStrategy, VectorResult};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct MockHybridBackend {
        bm25_calls: AtomicUsize,
        vector_calls: AtomicUsize,
        doc_ids: Vec<&'static str>,
    }

    impl MockHybridBackend {
        fn new(doc_ids: Vec<&'static str>) -> Self {
            Self {
                bm25_calls: AtomicUsize::new(0),
                vector_calls: AtomicUsize::new(0),
                doc_ids,
            }
        }
    }

    #[async_trait::async_trait]
    impl HybridSearchBackend for MockHybridBackend {
        async fn scope_collection(&self, _c: &str, _t: &str) -> RankResult<()> {
            Ok(())
        }
        async fn bm25_search(
            &self,
            _collection: &str,
            _query: &str,
        ) -> RankResult<Vec<BM25Result>> {
            self.bm25_calls.fetch_add(1, Ordering::SeqCst);
            Ok(self
                .doc_ids
                .iter()
                .enumerate()
                .map(|(i, id)| BM25Result {
                    doc_id: id.to_string(),
                    score: 1.0 / (i as f64 + 1.0),
                    highlights: None,
                    metadata: HashMap::new(),
                })
                .collect())
        }

        async fn vector_search(
            &self,
            _collection: &str,
            _vector: &[f32],
        ) -> RankResult<Vec<VectorResult>> {
            self.vector_calls.fetch_add(1, Ordering::SeqCst);
            Ok(self
                .doc_ids
                .iter()
                .enumerate()
                .map(|(i, id)| VectorResult {
                    doc_id: id.to_string(),
                    score: 1.0 - (i as f64 * 0.1),
                    distance: i as f64 * 0.1,
                    metadata: HashMap::new(),
                })
                .collect())
        }
    }

    fn rank_req(collection: &str, query_vector: Vec<f32>, k: usize) -> RankSearchRequest {
        RankSearchRequest {
            scope: RankScope::Internal,
            collection: collection.into(),
            query_vector,
            query_text: None,
            k,
            rank_profile: None,
            rank_overrides: None,
        }
    }

    #[tokio::test]
    async fn hybrid_adapter_runs_both_searches_in_parallel() {
        let backend = Arc::new(MockHybridBackend::new(vec!["1", "2", "3"]));
        let adapter = HybridCoordinatorAdapter::new(
            FusionStrategy::ReciprocalRank { k: 60 },
            backend.clone(),
        );
        let req = rank_req("docs", vec![0.1, 0.2, 0.3], 10);
        let docs = adapter.candidates(&req).await.unwrap().docs;
        // Both searches fired exactly once.
        assert_eq!(backend.bm25_calls.load(Ordering::SeqCst), 1);
        assert_eq!(backend.vector_calls.load(Ordering::SeqCst), 1);
        // RRF over 3 docs returned by each side should produce 3 unique handles.
        assert_eq!(docs.len(), 3);
    }

    #[tokio::test]
    async fn hybrid_adapter_preserves_numeric_doc_ids_in_original_map() {
        // R-7c.3.2: doc handles are now sequential synthetic indices
        // (1, 2, 3 …). The original backend strings are preserved in
        // CandidateBatch.original_ids so the response round-trips them
        // exactly. This works for both numeric and arbitrary-string ids.
        let backend = Arc::new(MockHybridBackend::new(vec!["42", "7", "99"]));
        let adapter =
            HybridCoordinatorAdapter::new(FusionStrategy::ReciprocalRank { k: 60 }, backend);
        let req = rank_req("docs", vec![0.5], 10);
        let batch = adapter.candidates(&req).await.unwrap();
        let map = batch.original_ids.expect("adapter must emit original_ids");
        let strings: std::collections::HashSet<String> =
            map.values().map(|s| s.to_string()).collect();
        assert!(strings.contains("42"));
        assert!(strings.contains("7"));
        assert!(strings.contains("99"));
    }

    #[tokio::test]
    async fn hybrid_adapter_preserves_non_numeric_doc_ids() {
        // R-7c.3.2: arbitrary-string backend ids no longer get dropped.
        // All four candidate ids survive the adapter into original_ids.
        let backend = Arc::new(MockHybridBackend::new(vec!["1", "abc", "2", "def"]));
        let adapter =
            HybridCoordinatorAdapter::new(FusionStrategy::ReciprocalRank { k: 60 }, backend);
        let req = rank_req("docs", vec![0.5], 10);
        let batch = adapter.candidates(&req).await.unwrap();
        assert_eq!(batch.docs.len(), 4);
        let strings: std::collections::HashSet<String> = batch
            .original_ids
            .expect("adapter must emit original_ids")
            .values()
            .map(|s| s.to_string())
            .collect();
        for expected in ["1", "abc", "2", "def"] {
            assert!(strings.contains(expected), "missing {expected:?}");
        }
    }

    #[tokio::test]
    async fn hybrid_adapter_propagates_backend_error_as_model_inference() {
        struct BrokenBackend;
        #[async_trait::async_trait]
        impl HybridSearchBackend for BrokenBackend {
            async fn scope_collection(&self, _c: &str, _t: &str) -> RankResult<()> {
                Ok(())
            }
            async fn bm25_search(&self, _c: &str, _q: &str) -> RankResult<Vec<BM25Result>> {
                Err(RankError::ModelInference {
                    model_id: "bm25".into(),
                    reason: "service unavailable".into(),
                })
            }
            async fn vector_search(&self, _c: &str, _v: &[f32]) -> RankResult<Vec<VectorResult>> {
                Ok(Vec::new())
            }
        }
        let adapter = HybridCoordinatorAdapter::new(
            FusionStrategy::ReciprocalRank { k: 60 },
            Arc::new(BrokenBackend),
        );
        let req = rank_req("docs", vec![0.5], 10);
        match adapter.candidates(&req).await {
            Err(RankError::ModelInference { reason, .. }) => {
                assert!(reason.contains("hybrid search failed"));
            }
            Err(_) => panic!("expected ModelInference"),
            Ok(_) => panic!("expected error"),
        }
    }

    /// TD-SECSCAN-1 §5: a client-argument error from the vector leg survives
    /// the hybrid coordinator (it is found anywhere in the error chain) and the
    /// rank route answers it as HTTP 400, not a 500 pipeline failure.
    #[tokio::test]
    async fn hybrid_adapter_maps_vector_leg_client_argument_to_400() {
        struct WrongDimensionBackend;
        #[async_trait::async_trait]
        impl HybridSearchBackend for WrongDimensionBackend {
            async fn scope_collection(&self, _c: &str, _t: &str) -> RankResult<()> {
                Ok(())
            }
            async fn bm25_search(&self, _c: &str, _q: &str) -> RankResult<Vec<BM25Result>> {
                Ok(Vec::new())
            }
            async fn vector_search(&self, _c: &str, _v: &[f32]) -> RankResult<Vec<VectorResult>> {
                Err(RankError::InvalidArgument(
                    "Query vector has dimension 1 but collection 'docs' expects dimension 4".into(),
                ))
            }
        }
        let adapter = HybridCoordinatorAdapter::new(
            FusionStrategy::ReciprocalRank { k: 60 },
            Arc::new(WrongDimensionBackend),
        );
        let err = match adapter.candidates(&rank_req("docs", vec![0.5], 10)).await {
            Err(err) => err,
            Ok(_) => panic!("expected an error"),
        };
        assert!(
            matches!(&err, RankError::InvalidArgument(m) if m.contains("dimension 1")),
            "expected InvalidArgument, got {err:?}"
        );
        assert!(matches!(
            rank_error_to_api(err),
            ApiError::InvalidArgument(m) if m.contains("dimension 1")
        ));
        // A server-side failure stays a 500.
        assert!(matches!(
            rank_error_to_api(RankError::ModelInference {
                model_id: "m".into(),
                reason: "down".into(),
            }),
            ApiError::Internal(_)
        ));
    }

    #[tokio::test]
    async fn hybrid_adapter_round_trips_through_rank_services() {
        // End-to-end: RankServices with HybridCoordinatorAdapter as
        // CandidateProvider; verify the dispatcher gets the adapter's
        // output rather than the mock-range fallback.
        let backend = Arc::new(MockHybridBackend::new(vec!["1", "2", "3", "4", "5"]));
        let adapter =
            HybridCoordinatorAdapter::new(FusionStrategy::ReciprocalRank { k: 60 }, backend);
        let services = RankServices::new(Arc::new(adapter));
        // Profile-free path: candidates pass through as-is with score=0.
        let req = rank_req("docs", vec![0.1, 0.2], 3);
        let registry = services.profile_registry.clone();
        let factory = services.blueprint_factory.clone();
        let resp = handle_rank_search(
            req,
            registry.as_ref(),
            services.candidate_provider.as_ref(),
            factory,
            None,
        )
        .await
        .unwrap();
        // Top-3 returned (out of 5 candidates the backend produced).
        assert_eq!(resp.hits.len(), 3);
    }

    #[test]
    fn doc_id_to_handle_parses_decimal() {
        assert_eq!(doc_id_to_handle("123"), Some(DocHandle(123)));
        assert_eq!(doc_id_to_handle("0"), Some(DocHandle(0)));
        assert!(doc_id_to_handle("abc").is_none());
        assert!(doc_id_to_handle("-5").is_none()); // u32 rejects negatives
        assert!(doc_id_to_handle("").is_none());
    }

    // ---------------- R-7c.3.1: query_text plumbing tests ----------------

    #[tokio::test]
    async fn hybrid_adapter_forwards_query_text_to_bm25() {
        use std::sync::Mutex;
        struct CapturingBackend {
            last_bm25_query: Mutex<Option<String>>,
        }
        #[async_trait::async_trait]
        impl HybridSearchBackend for CapturingBackend {
            async fn scope_collection(&self, _c: &str, _t: &str) -> RankResult<()> {
                Ok(())
            }
            async fn bm25_search(&self, _c: &str, query: &str) -> RankResult<Vec<BM25Result>> {
                *self.last_bm25_query.lock().unwrap() = Some(query.to_string());
                Ok(vec![BM25Result {
                    doc_id: "1".into(),
                    score: 0.5,
                    highlights: None,
                    metadata: HashMap::new(),
                }])
            }
            async fn vector_search(&self, _c: &str, _v: &[f32]) -> RankResult<Vec<VectorResult>> {
                Ok(vec![VectorResult {
                    doc_id: "1".into(),
                    score: 0.8,
                    distance: 0.2,
                    metadata: HashMap::new(),
                }])
            }
        }
        let backend = Arc::new(CapturingBackend {
            last_bm25_query: Mutex::new(None),
        });
        let adapter = HybridCoordinatorAdapter::new(
            FusionStrategy::ReciprocalRank { k: 60 },
            backend.clone(),
        );
        let mut req = rank_req("docs", vec![0.1], 5);
        req.query_text = Some("laptop computer".into());
        let _ = adapter.candidates(&req).await.unwrap();
        let captured = backend.last_bm25_query.lock().unwrap().clone();
        assert_eq!(captured.as_deref(), Some("laptop computer"));
    }

    #[tokio::test]
    async fn hybrid_adapter_uses_empty_string_when_query_text_absent() {
        // Contract: missing query_text → empty string sentinel.
        use std::sync::Mutex;
        struct CapturingBackend {
            last_bm25_query: Mutex<Option<String>>,
        }
        #[async_trait::async_trait]
        impl HybridSearchBackend for CapturingBackend {
            async fn scope_collection(&self, _c: &str, _t: &str) -> RankResult<()> {
                Ok(())
            }
            async fn bm25_search(&self, _c: &str, query: &str) -> RankResult<Vec<BM25Result>> {
                *self.last_bm25_query.lock().unwrap() = Some(query.to_string());
                Ok(Vec::new())
            }
            async fn vector_search(&self, _c: &str, _v: &[f32]) -> RankResult<Vec<VectorResult>> {
                Ok(Vec::new())
            }
        }
        let backend = Arc::new(CapturingBackend {
            last_bm25_query: Mutex::new(None),
        });
        let adapter = HybridCoordinatorAdapter::new(
            FusionStrategy::ReciprocalRank { k: 60 },
            backend.clone(),
        );
        let req = rank_req("docs", vec![0.5], 5); // query_text omitted by rank_req()
        let _ = adapter.candidates(&req).await.unwrap();
        let captured = backend.last_bm25_query.lock().unwrap().clone();
        assert_eq!(captured.as_deref(), Some(""));
    }

    #[test]
    fn request_query_text_round_trips_through_json() {
        let req = RankSearchRequest {
            scope: RankScope::Internal,
            collection: "docs".into(),
            query_vector: vec![0.1],
            query_text: Some("hello world".into()),
            k: 5,
            rank_profile: None,
            rank_overrides: None,
        };
        let j = serde_json::to_string(&req).unwrap();
        assert!(j.contains("query_text"));
        let back: RankSearchRequest = serde_json::from_str(&j).unwrap();
        assert_eq!(back.query_text.as_deref(), Some("hello world"));
    }

    #[test]
    fn request_query_text_omitted_from_json_when_none() {
        let req = RankSearchRequest {
            scope: RankScope::Internal,
            collection: "docs".into(),
            query_vector: vec![0.1],
            query_text: None,
            k: 5,
            rank_profile: None,
            rank_overrides: None,
        };
        let j = serde_json::to_string(&req).unwrap();
        assert!(!j.contains("query_text"));
    }

    // ---------------- R-7c.3.2: CandidateBatch helpers + e2e round-trip ----------------

    #[test]
    fn candidate_batch_from_docs_has_no_id_map() {
        let b = CandidateBatch::from_docs(vec![DocHandle(1), DocHandle(2)]);
        assert_eq!(b.docs.len(), 2);
        assert!(b.original_ids.is_none());
    }

    #[test]
    fn candidate_batch_from_string_ids_assigns_sequential_handles() {
        let b = CandidateBatch::from_string_ids(vec!["alpha", "beta", "gamma"]);
        assert_eq!(b.docs.len(), 3);
        assert_eq!(b.docs[0], DocHandle(1));
        assert_eq!(b.docs[1], DocHandle(2));
        assert_eq!(b.docs[2], DocHandle(3));
        let m = b.original_ids.unwrap();
        assert_eq!(m.get(&DocHandle(1)).unwrap().as_ref(), "alpha");
        assert_eq!(m.get(&DocHandle(2)).unwrap().as_ref(), "beta");
        assert_eq!(m.get(&DocHandle(3)).unwrap().as_ref(), "gamma");
    }

    #[tokio::test]
    async fn handler_with_string_id_backend_round_trips_ids_in_response() {
        // End-to-end: a CandidateProvider that emits arbitrary string
        // ids round-trips them through handle_rank_search into
        // ScoredHitDto.id (rather than the synthetic DocHandle number).
        struct StringIdProvider(Vec<&'static str>);
        #[async_trait::async_trait]
        impl CandidateProvider for StringIdProvider {
            async fn candidates(&self, _request: &RankSearchRequest) -> RankResult<CandidateBatch> {
                Ok(CandidateBatch::from_string_ids(self.0.clone()))
            }
        }
        let registry = ProfileRegistry::new();
        let factory = factory_with_docid();
        let candidates = StringIdProvider(vec!["doc:abc", "doc:xyz", "doc:lmn"]);
        let req = rank_req("docs", vec![], 3);
        let resp = handle_rank_search(req, &registry, &candidates, factory, None)
            .await
            .unwrap();
        let ids: std::collections::HashSet<String> =
            resp.hits.iter().map(|h| h.id.clone()).collect();
        assert!(ids.contains("doc:abc"));
        assert!(ids.contains("doc:xyz"));
        assert!(ids.contains("doc:lmn"));
        for h in &resp.hits {
            assert!(
                h.id.parse::<u32>().is_err(),
                "response id should be the original string, not the synthetic handle: {}",
                h.id
            );
        }
    }

    #[tokio::test]
    async fn handler_falls_back_to_handle_number_when_no_id_map() {
        // Inverse: when CandidateProvider returns CandidateBatch::from_docs
        // (no original_ids), response ids are DocHandle.0.to_string().
        let registry = ProfileRegistry::new();
        let factory = factory_with_docid();
        let candidates = FixedCandidates(vec![DocHandle(7), DocHandle(42)]);
        let req = rank_req("docs", vec![], 5);
        let resp = handle_rank_search(req, &registry, &candidates, factory, None)
            .await
            .unwrap();
        let ids: std::collections::HashSet<String> =
            resp.hits.iter().map(|h| h.id.clone()).collect();
        assert!(ids.contains("7"));
        assert!(ids.contains("42"));
    }
}
