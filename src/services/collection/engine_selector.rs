// Copyright 2025 Vijaykumar Singh.
// Licensed under the Apache License, Version 2.0.

//! Heuristic-based storage engine selection for newly-created collections.
//!
//! # Why
//!
//! ProximaDB ships multiple storage engines with different recall/latency
//! trade-offs (see `src/storage/engines/` and the per-engine docstrings):
//!
//! * **SST** — write-optimized LSM. Blocks are oid-sorted (lexicographic),
//!   so flat-mode centroid pruning collapses on diffuse vector data
//!   (~5% recall at 100K — see TD-096). SST's *intended* vector workflow is
//!   **with an AXIS index** (HNSW/IVF/PQ) attached, or **with progressive
//!   quantization** (fp8/int8 candidate scan + fp32 rerank). The index /
//!   quantization handles ANN; SST handles the OLTP `oid → record` lookup.
//!
//! * **HELIX** — locality-optimized engine that sorts blocks by
//!   PCA-projected Hilbert curve at flush time
//!   (`src/storage/engines/helix/mod.rs:1279` "SORTED FLUSH OPTIMIZATION").
//!   Block centroids reflect real spatial clusters, so the built-in
//!   zone-map pruning achieves ANN-quality recall *without* a separate
//!   index. The right choice for a pure-vector collection that doesn't
//!   want the build/memory cost of an AXIS HNSW.
//!
//! An earlier version routed vector collections without an index to HELIX.
//! That put the default path on a Beta engine with open recall defects
//! (2% recall@10 measured), so TD-ENGSEL-1 returned them to SST, whose PAX
//! segment cascade (IVF cells → RaBitQ → SQ8 rerank) is the index.
//!
//! # Heuristic
//!
//! Given a `CollectionConfig` at create-time, infer the engine via:
//!
//! 1. If `storage_engine` is explicitly set (not Unspecified/None), the
//!    caller wins. The heuristic only fires when the engine is unset.
//! 2. If `auto_index_selection == false`, defer to the SST default and
//!    leave engine choice to the caller (operator opted out of auto-routing).
//! 3. If a `recall_target:<float>` tag is set on the collection, choose
//!    **SST**. The recall-target adaptive stack (advisor, drift
//!    detector, hot-swap, /recluster) lives exclusively on SST+HNSW;
//!    HELIX's Hilbert-sorted block pruning has no ef/m tuning surface,
//!    so a recall_target on HELIX would be a no-op promise.
//! 4. If the collection has `index_configs` populated OR
//!    `quantization.enabled = true`, choose **SST**. Indexes + progressive
//!    quantization compensate for SST's block-pruning limitation.
//! 5. Otherwise (vector collection with no index and no quantization
//!    and no recall_target) choose **SST** (TD-ENGSEL-1). HELIX is an
//!    explicit opt-in: `storage_engine = HELIX` or `index_policy.mode = "helix"`.
//!
//! # Observability
//!
//! Each call returns the chosen engine **and a short reason string** so
//! the caller can log/emit-metric for operator visibility:
//!
//! ```ignore
//! let (engine, reason) = infer_storage_engine(&config);
//! tracing::info!(
//!     target: "collection.engine_selector",
//!     collection = %config.name,
//!     chosen_engine = ?engine,
//!     reason = reason,
//!     "auto-selected storage engine"
//! );
//! ```
//!
//! # Future
//!
//! This is the static heuristic baseline. The RL planner will eventually
//! observe per-collection latency / recall outcomes and learn which
//! (engine, index, quantization) tuple actually performs best for a given
//! workload shape — at that point the heuristic becomes a fallback for
//! cold-start before the RL model has enough signal.

use crate::proto::proximadb_v1::{CollectionConfig, StorageEngine};

/// Static reason codes returned alongside the engine choice. Kept as
/// short literals so they're cheap to log and easy to filter in
/// Prometheus / SIEM pipelines.
pub mod reasons {
    pub const EXPLICIT_OVERRIDE: &str = "explicit_caller_override";
    pub const AUTO_SELECT_DISABLED: &str = "auto_select_disabled_default_sst";
    pub const HAS_INDEX: &str = "has_index_config_route_sst";
    pub const HAS_QUANTIZATION: &str = "has_quantization_route_sst";
    pub const HAS_BOTH: &str = "has_index_and_quantization_route_sst";
    pub const RECALL_TARGET_SET: &str = "recall_target_route_sst";
    /// TD-ENGSEL-1: a vector collection with no index, quantization, recall
    /// target or policy lands on SST, the supported default. HELIX (Beta) is an
    /// explicit opt-in (`storage_engine` or `index_policy.mode = "helix"`).
    pub const VECTOR_NO_INDEX: &str = "vector_no_index_route_sst";
    pub const NON_VECTOR_DEFAULT_SST: &str = "non_vector_default_sst";
    /// `index_policy.mode == "helix"` — owner forced the HELIX engine (ADR-028).
    pub const POLICY_FORCE_HELIX: &str = "index_policy_mode_helix";
    /// `index_policy.mode == "ivf"|"hnsw"` — owner forced an AXIS index, which
    /// lives on SST (ADR-028).
    pub const POLICY_FORCE_SST_INDEX: &str = "index_policy_mode_index_route_sst";
    /// `index_policy.mode == "auto"` + quantization, and the
    /// `PROXIMADB_ROUTER_QUANT_HELIX` bake flag is on — route the quantized
    /// vector collection to HELIX (quantization-native, rebuild-free cold)
    /// instead of the SST cold path (ADR-028 auto-router improvement).
    pub const AUTOROUTE_QUANT_HELIX: &str = "autoroute_quant_route_helix";
}

/// Env flag gating the ADR-028 auto-router improvement: under `mode=auto`, route
/// quantized vector collections to HELIX rather than the SST cold path. Default-OFF
/// (bake-in discipline; the SST `SearchMode` exact gate already corrects SST cold
/// recall, so this is an optimization, not a correctness fix). Owners can always
/// force HELIX explicitly via `index_policy.mode = "helix"`.
fn autoroute_quant_to_helix_enabled() -> bool {
    std::env::var_os("PROXIMADB_ROUTER_QUANT_HELIX").is_some()
}

/// Map an `index_policy.mode` string to a forced engine, if the mode names one.
/// `"auto"`/`""`/`"exact"` return `None` (exact is a *search route*, honored at the
/// search layer on whatever engine the heuristic picks — it does not force an engine).
fn policy_forced_engine(mode: &str) -> Option<(StorageEngine, &'static str)> {
    match mode.trim().to_ascii_lowercase().as_str() {
        "helix" => Some((StorageEngine::Helix, reasons::POLICY_FORCE_HELIX)),
        // IVF / HNSW are AXIS indexes that live on the SST engine.
        "ivf" | "hnsw" => Some((StorageEngine::Sst, reasons::POLICY_FORCE_SST_INDEX)),
        _ => None,
    }
}

/// Decide which storage engine to use for a newly-created collection.
///
/// Returns `(engine, reason)`. The reason is one of the constants in
/// [`reasons`] so callers can attach it to structured logs / metrics
/// without doing string parsing.
///
/// See the module docs for the full decision tree.
pub fn infer_storage_engine(config: &CollectionConfig) -> (StorageEngine, &'static str) {
    // (1) Caller-explicit choice always wins.
    let explicit = config
        .storage_engine
        .and_then(|v| StorageEngine::try_from(v).ok())
        .filter(|e| !matches!(e, StorageEngine::Unspecified));
    if let Some(engine) = explicit {
        return (engine, reasons::EXPLICIT_OVERRIDE);
    }

    // (1b) Per-collection index_policy (ADR-028). When the owner forced an
    // engine-bearing mode (`helix` / `ivf` / `hnsw`), honor it — it is a more
    // specific routing directive than the auto_index_selection opt-out below.
    // `auto`/`exact`/unset fall through; `exact` is a search route honored at the
    // search layer, not an engine choice.
    if let Some(policy) = config.index_policy.as_ref()
        && let Some((engine, reason)) = policy_forced_engine(&policy.mode)
    {
        return (engine, reason);
    }

    // (2) Operator opted out of auto-selection — keep SST as the
    // historical default. The proto default is `auto_index_selection = true`
    // so a None here means "the field was unset", which we treat as
    // "auto-select is on" to preserve the heuristic's reach.
    if let Some(false) = config.auto_index_selection {
        return (StorageEngine::Sst, reasons::AUTO_SELECT_DISABLED);
    }

    // (3a) Vector collections that committed to a recall_target tag
    // are opting into the adaptive HNSW path — that path lives
    // exclusively on SST (HELIX's Hilbert-sorted block pruning has
    // no ef/m tuning surface). Route them to SST even when the
    // caller didn't explicitly attach an HNSW IndexConfig — the
    // recall_target wiring downstream synthesizes one from the
    // advisor's recommendation.
    if super::recall_target::parse_recall_target(config).is_some() {
        return (StorageEngine::Sst, reasons::RECALL_TARGET_SET);
    }

    // (3b) Vector collections that opted into an index or quantization
    // get SST. The index/quantization handles ANN; SST handles
    // oid-keyed point lookups efficiently via bloom + binary search.
    let has_index = !config.index_configs.is_empty();
    let has_quant = config
        .quantization
        .as_ref()
        .and_then(|q| q.enabled)
        .unwrap_or(false);
    match (has_index, has_quant) {
        (true, true) => return (StorageEngine::Sst, reasons::HAS_BOTH),
        (true, false) => return (StorageEngine::Sst, reasons::HAS_INDEX),
        (false, true) => {
            // ADR-028 auto-router improvement: a quantization-only vector
            // collection (the OpenAI-embedding shape) was historically pinned to
            // the SST cold path, whose AXIS HNSW is rebuilt every cold start and
            // can exclude the true NN (TD-165). HELIX is quantization-native
            // (persisted PCA, Hilbert-sorted, rebuild-free cold), so under
            // mode=auto we prefer it once the bake flag is on. Default-OFF: the
            // SST SearchMode exact gate already corrects SST cold recall, so this
            // is an optimization. dimension>0 guards against a non-vector
            // collection that merely tagged quantization.
            if autoroute_quant_to_helix_enabled() && config.dimension > 0 {
                return (StorageEngine::Helix, reasons::AUTOROUTE_QUANT_HELIX);
            }
            return (StorageEngine::Sst, reasons::HAS_QUANTIZATION);
        }
        (false, false) => {}
    }

    // (4) No index, no quantization: SST either way (TD-ENGSEL-1, owner
    //     decision 2026-10-17). A vector collection is served by SST's PAX
    //     cascade — the segment is the index — which the support contract and
    //     the MVP corridor promise; HELIX (Beta, open recall TDs) measured 2%
    //     recall@10 on this default path and is now an explicit opt-in.
    if config.dimension > 0 {
        (StorageEngine::Sst, reasons::VECTOR_NO_INDEX)
    } else {
        (StorageEngine::Sst, reasons::NON_VECTOR_DEFAULT_SST)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::proximadb_v1::{IndexConfig, QuantizationConfig};

    fn vec_config(name: &str) -> CollectionConfig {
        CollectionConfig {
            name: name.to_string(),
            dimension: 128,
            ..Default::default()
        }
    }

    #[test]
    fn explicit_engine_wins_over_heuristic() {
        let mut cfg = vec_config("t1");
        cfg.storage_engine = Some(StorageEngine::Viper as i32);
        let (engine, reason) = infer_storage_engine(&cfg);
        assert_eq!(engine, StorageEngine::Viper);
        assert_eq!(reason, reasons::EXPLICIT_OVERRIDE);
    }

    #[test]
    fn unspecified_engine_treated_as_unset() {
        let mut cfg = vec_config("t2");
        cfg.storage_engine = Some(StorageEngine::Unspecified as i32);
        let (engine, reason) = infer_storage_engine(&cfg);
        // Falls through to heuristic — vector collection, no index → HELIX.
        assert_eq!(engine, StorageEngine::Sst);
        assert_eq!(reason, reasons::VECTOR_NO_INDEX);
    }

    #[test]
    fn auto_select_false_picks_sst() {
        let mut cfg = vec_config("t3");
        cfg.auto_index_selection = Some(false);
        let (engine, reason) = infer_storage_engine(&cfg);
        assert_eq!(engine, StorageEngine::Sst);
        assert_eq!(reason, reasons::AUTO_SELECT_DISABLED);
    }

    #[test]
    fn vector_with_index_picks_sst() {
        let mut cfg = vec_config("t4");
        cfg.index_configs.push(IndexConfig::default());
        let (engine, reason) = infer_storage_engine(&cfg);
        assert_eq!(engine, StorageEngine::Sst);
        assert_eq!(reason, reasons::HAS_INDEX);
    }

    #[test]
    fn vector_with_quantization_picks_sst() {
        let mut cfg = vec_config("t5");
        cfg.quantization = Some(QuantizationConfig {
            enabled: Some(true),
            ..Default::default()
        });
        let (engine, reason) = infer_storage_engine(&cfg);
        assert_eq!(engine, StorageEngine::Sst);
        assert_eq!(reason, reasons::HAS_QUANTIZATION);
    }

    #[test]
    fn vector_with_index_and_quantization_picks_sst() {
        let mut cfg = vec_config("t6");
        cfg.index_configs.push(IndexConfig::default());
        cfg.quantization = Some(QuantizationConfig {
            enabled: Some(true),
            ..Default::default()
        });
        let (engine, reason) = infer_storage_engine(&cfg);
        assert_eq!(engine, StorageEngine::Sst);
        assert_eq!(reason, reasons::HAS_BOTH);
    }

    #[test]
    fn vector_no_index_no_quantization_picks_sst() {
        let cfg = vec_config("t7");
        let (engine, reason) = infer_storage_engine(&cfg);
        assert_eq!(engine, StorageEngine::Sst);
        assert_eq!(reason, reasons::VECTOR_NO_INDEX);
    }

    #[test]
    fn recall_target_tag_routes_to_sst_without_explicit_index() {
        // A recall_target tag alone, no index_configs, no
        // quantization, vector collection: the previous default
        // would have been HELIX (vector-no-index rule), but the
        // recall-target adaptive stack only works on SST+HNSW,
        // so the override fires.
        let mut cfg = vec_config("t10");
        cfg.tags = vec!["recall_target:0.95".to_string()];
        let (engine, reason) = infer_storage_engine(&cfg);
        assert_eq!(engine, StorageEngine::Sst);
        assert_eq!(reason, reasons::RECALL_TARGET_SET);
    }

    #[test]
    fn recall_target_tag_still_loses_to_explicit_engine() {
        // The explicit-override rule from step (1) takes precedence
        // — operators who pin storage_engine know what they want.
        let mut cfg = vec_config("t11");
        cfg.tags = vec!["recall_target:0.95".to_string()];
        cfg.storage_engine = Some(StorageEngine::Helix as i32);
        let (engine, reason) = infer_storage_engine(&cfg);
        assert_eq!(engine, StorageEngine::Helix);
        assert_eq!(reason, reasons::EXPLICIT_OVERRIDE);
    }

    #[test]
    fn quantization_disabled_does_not_route_to_sst() {
        let mut cfg = vec_config("t8");
        cfg.quantization = Some(QuantizationConfig {
            enabled: Some(false),
            ..Default::default()
        });
        let (engine, reason) = infer_storage_engine(&cfg);
        assert_eq!(engine, StorageEngine::Sst);
        assert_eq!(reason, reasons::VECTOR_NO_INDEX);
    }

    #[test]
    fn non_vector_collection_picks_sst() {
        let mut cfg = vec_config("t9");
        cfg.dimension = 0;
        let (engine, reason) = infer_storage_engine(&cfg);
        assert_eq!(engine, StorageEngine::Sst);
        assert_eq!(reason, reasons::NON_VECTOR_DEFAULT_SST);
    }

    // ---- ADR-028 index_policy routing ----

    use crate::proto::proximadb_v1::IndexPolicy;

    fn policy(mode: &str) -> Option<IndexPolicy> {
        Some(IndexPolicy {
            mode: mode.to_string(),
            ..Default::default()
        })
    }

    #[test]
    fn policy_forced_engine_mapping_is_pure() {
        assert_eq!(
            policy_forced_engine("helix").map(|(e, _)| e),
            Some(StorageEngine::Helix)
        );
        assert_eq!(
            policy_forced_engine("HNSW").map(|(e, _)| e),
            Some(StorageEngine::Sst)
        );
        assert_eq!(
            policy_forced_engine(" ivf ").map(|(e, _)| e),
            Some(StorageEngine::Sst)
        );
        // auto / exact / unset name no engine — exact is a search route.
        assert!(policy_forced_engine("auto").is_none());
        assert!(policy_forced_engine("exact").is_none());
        assert!(policy_forced_engine("").is_none());
    }

    #[test]
    fn policy_mode_helix_forces_helix() {
        let mut cfg = vec_config("p1");
        cfg.index_policy = policy("helix");
        let (engine, reason) = infer_storage_engine(&cfg);
        assert_eq!(engine, StorageEngine::Helix);
        assert_eq!(reason, reasons::POLICY_FORCE_HELIX);
    }

    #[test]
    fn policy_mode_ivf_forces_sst_even_without_index() {
        // No index_configs, no quantization — the heuristic alone would pick
        // HELIX, but the explicit ivf policy forces the SST/AXIS route.
        let mut cfg = vec_config("p2");
        cfg.index_policy = policy("ivf");
        let (engine, reason) = infer_storage_engine(&cfg);
        assert_eq!(engine, StorageEngine::Sst);
        assert_eq!(reason, reasons::POLICY_FORCE_SST_INDEX);
    }

    #[test]
    fn policy_mode_ivf_beats_auto_select_disabled() {
        // The explicit ivf policy is more specific than the auto_index_selection
        // opt-out (which would otherwise force the SST *default* with a
        // different reason). Both land on SST here, but via the policy branch.
        let mut cfg = vec_config("p3");
        cfg.auto_index_selection = Some(false);
        cfg.index_policy = policy("ivf");
        let (engine, reason) = infer_storage_engine(&cfg);
        assert_eq!(engine, StorageEngine::Sst);
        assert_eq!(reason, reasons::POLICY_FORCE_SST_INDEX);
    }

    #[test]
    fn policy_mode_exact_falls_through_to_heuristic() {
        // exact does not force an engine; a no-index vector collection still
        // routes to HELIX, and the search layer honors exact there.
        let mut cfg = vec_config("p4");
        cfg.index_policy = policy("exact");
        let (engine, reason) = infer_storage_engine(&cfg);
        assert_eq!(engine, StorageEngine::Sst);
        assert_eq!(reason, reasons::VECTOR_NO_INDEX);
    }

    #[test]
    fn explicit_storage_engine_still_beats_policy() {
        // A pinned storage_engine is the ultimate override (step 1), ahead of
        // the index_policy routing layer.
        let mut cfg = vec_config("p5");
        cfg.storage_engine = Some(StorageEngine::Viper as i32);
        cfg.index_policy = policy("helix");
        let (engine, reason) = infer_storage_engine(&cfg);
        assert_eq!(engine, StorageEngine::Viper);
        assert_eq!(reason, reasons::EXPLICIT_OVERRIDE);
    }
}
