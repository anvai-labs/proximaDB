//! A validated profile bound to a [`BlueprintFactory`], ready to
//! materialize fresh [`RankPipeline`] instances per query.

use crate::spec::RankProfileSpec;
use crate::validator::validate;
use proximadb_rank_core::{
    BlueprintFactory, ExecutorIdx, PhaseBudget, QueryContext, RankError, RankPipeline, RankProgram,
    RankResult,
};
use proximadb_rank_expr::ExprBlueprint;
use std::sync::Arc;

/// Mapping from feature/summary name to the executor slot that
/// produces its value. Aliased to satisfy `clippy::type_complexity`
/// across three signature sites in this module.
pub type FeatureExecutorMap = Arc<[(Arc<str>, ExecutorIdx)]>;

/// A validated profile + the factory it resolved against. Cheap to clone
/// (Arc-shared internals); held in a [`crate::registry::ProfileRegistry`]
/// behind an `ArcSwap` for lock-free hot-reload.
#[derive(Clone)]
pub struct CompiledRankProfile {
    pub spec: Arc<RankProfileSpec>,
    pub factory: Arc<BlueprintFactory>,
}

impl CompiledRankProfile {
    /// Validate and bind a profile to a factory. Returns `Err` if the
    /// profile fails validation; on `Ok`, the profile is ready to
    /// materialize pipelines.
    ///
    /// The profile should already have had its inheritance chain
    /// resolved (call [`crate::validator::resolve_inheritance`] first).
    pub fn compile(spec: RankProfileSpec, factory: Arc<BlueprintFactory>) -> RankResult<Self> {
        validate(&spec, &factory)?;
        Ok(Self {
            spec: Arc::new(spec),
            factory,
        })
    }

    /// Build a fresh [`RankPipeline`] for one query. Per-query cost is
    /// parse + lowering of each phase's expression; this is microseconds
    /// for typical expressions.
    pub fn materialize(&self, qctx: &QueryContext) -> RankResult<RankPipeline> {
        let bp = ExprBlueprint::new(self.factory.clone());

        let first_spec = self.spec.first_phase.as_ref().ok_or_else(|| {
            RankError::InvalidProfile(format!(
                "profile '{}': cannot materialize without a first_phase",
                self.spec.name
            ))
        })?;
        // R-7c.5 + R-7c.5b: the first-phase program now hosts the
        // score executor, the match_features executors, AND the
        // summary_features executors — all sharing the same program
        // so memoization across overlapping sub-expressions works
        // (a match_feature that names the same bm25(...) the score
        // uses runs once per doc, not twice).
        let (first, match_features, summary_features) = build_first_program_with_features(
            &bp,
            &first_spec.expression,
            &self.spec.match_features,
            &self.spec.summary_features,
            qctx,
        )?;

        let second = match &self.spec.second_phase {
            Some(p) => Some(single_executor_program(
                bp.compile_str(&p.expression, qctx)?,
            )?),
            None => None,
        };

        let heap_size = first_spec.heap_size.unwrap_or(100) as usize;
        let rerank_count = self
            .spec
            .second_phase
            .as_ref()
            .and_then(|p| p.rerank_count)
            .unwrap_or(heap_size as u32) as usize;

        // Global phase: R-6 will wire the cross_modal / llm_listwise
        // adapters here. R-4 materializes the pipeline without a global
        // scorer so first/second-only profiles work end-to-end today.
        let global = None;

        Ok(RankPipeline {
            profile_id: self.spec.name.clone(),
            first,
            second,
            global,
            budget: PhaseBudget {
                first_max_us: self.spec.budget.first_max_us,
                second_max_us: self.spec.budget.second_max_us,
                global_max_us: self.spec.budget.global_max_us,
            },
            heap_size,
            rerank_count,
            match_features,
            summary_features,
        })
    }
}

/// Compile the first-phase score expression plus any declared
/// `match_features` and `summary_features` into a single `RankProgram`.
/// The score executor is at `score_idx`; match-feature executors take
/// the next M slots; summary-feature executors take the M+1.. slots.
/// Returns the program along with the two resolved
/// `(name, executor_idx)` mappings the pipeline walks per doc to
/// populate `ScoredHit.features` (R-7c.5) and `ScoredHit.summary`
/// (R-7c.5b).
fn build_first_program_with_features(
    bp: &ExprBlueprint,
    score_expr: &str,
    match_features: &[String],
    summary_features: &[String],
    qctx: &QueryContext,
) -> RankResult<(RankProgram, FeatureExecutorMap, FeatureExecutorMap)> {
    let mut b = RankProgram::builder();
    let score_idx = b.add(bp.compile_str(score_expr, qctx)?);
    b.set_score(score_idx);

    let mut resolved_match: Vec<(Arc<str>, ExecutorIdx)> = Vec::with_capacity(match_features.len());
    for expr in match_features {
        let idx = b.add(bp.compile_str(expr, qctx)?);
        resolved_match.push((Arc::<str>::from(expr.as_str()), idx));
    }
    let mut resolved_summary: Vec<(Arc<str>, ExecutorIdx)> =
        Vec::with_capacity(summary_features.len());
    for expr in summary_features {
        let idx = b.add(bp.compile_str(expr, qctx)?);
        resolved_summary.push((Arc::<str>::from(expr.as_str()), idx));
    }
    Ok((
        b.build()?,
        Arc::<[(Arc<str>, ExecutorIdx)]>::from(resolved_match),
        Arc::<[(Arc<str>, ExecutorIdx)]>::from(resolved_summary),
    ))
}

fn single_executor_program(
    exec: Box<dyn proximadb_rank_core::FeatureExecutor>,
) -> RankResult<RankProgram> {
    let mut b = RankProgram::builder();
    let idx = b.add(exec);
    b.set_score(idx);
    b.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{PhaseSpec, RankProfileSpec};
    use proximadb_rank_core::test_support::MapAttributeAccess;
    use proximadb_rank_core::{
        DocHandle, FeatureArena, NoopCandidateData, NoopMetricsSink, NoopModelCache, ScoreCtx,
    };
    use proximadb_rank_features::register_builtins;
    use std::collections::HashMap;

    fn factory() -> Arc<BlueprintFactory> {
        let f = Arc::new(BlueprintFactory::new());
        register_builtins(&f);
        f
    }

    #[test]
    fn compile_rejects_invalid_spec() {
        let f = factory();
        let bad = RankProfileSpec::new("bad"); // no first_phase
        match CompiledRankProfile::compile(bad, f) {
            Err(RankError::InvalidProfile(_)) => {}
            Err(_) => panic!("expected InvalidProfile, got a different RankError"),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    #[test]
    fn materialize_runs_through_pipeline_and_scores_doc() {
        let f = factory();
        let mut spec = RankProfileSpec::new("e2e");
        spec.first_phase = Some(PhaseSpec {
            expression: "attribute(\"score\") * 2 + 1".into(),
            heap_size: Some(50),
            rerank_count: None,
            batch_size: None,
        });
        let compiled = CompiledRankProfile::compile(spec, f).unwrap();

        let qctx = QueryContext::default();
        let mut pipe = compiled.materialize(&qctx).unwrap();

        let attrs = MapAttributeAccess {
            values: HashMap::from([
                ((0, "score".into()), 3.0),
                ((1, "score".into()), 5.0),
                ((2, "score".into()), 1.0),
            ]),
        };
        let arena = FeatureArena::new();
        let cands = NoopCandidateData;
        let m = NoopModelCache;
        let met = NoopMetricsSink;
        let mut ctx = ScoreCtx::new(&qctx, &arena, &attrs, &cands, &m, &met);
        let candidates = vec![DocHandle(0), DocHandle(1), DocHandle(2)];
        let outcome = pipe.run_first_phase(&candidates, &mut ctx).unwrap();
        // Expected scores: 3*2+1=7, 5*2+1=11, 1*2+1=3 → sorted desc → 11, 7, 3
        assert_eq!(outcome.hits.len(), 3);
        assert!((outcome.hits[0].score - 11.0).abs() < 1e-5);
        assert_eq!(outcome.hits[0].doc, DocHandle(1));
        assert!((outcome.hits[1].score - 7.0).abs() < 1e-5);
        assert!((outcome.hits[2].score - 3.0).abs() < 1e-5);
    }

    #[test]
    fn materialize_threads_budget_through() {
        let f = factory();
        let mut spec = RankProfileSpec::new("with_budget");
        spec.first_phase = Some(PhaseSpec {
            expression: "1.0".into(),
            heap_size: Some(10),
            rerank_count: None,
            batch_size: None,
        });
        spec.budget.first_max_us = Some(12345);
        let compiled = CompiledRankProfile::compile(spec, f).unwrap();
        let qctx = QueryContext::default();
        let pipe = compiled.materialize(&qctx).unwrap();
        assert_eq!(pipe.budget.first_max_us, Some(12345));
        assert_eq!(pipe.heap_size, 10);
    }

    #[test]
    fn materialize_lowers_match_features_into_pipeline_mapping() {
        // R-7c.5: spec.match_features expressions must compile as
        // additional executors in the same first-phase RankProgram, with
        // the (name, ExecutorIdx) mapping surfacing on RankPipeline.
        let f = factory();
        let mut spec = RankProfileSpec::new("with_matchf");
        spec.first_phase = Some(PhaseSpec {
            expression: "1.0".into(),
            heap_size: Some(10),
            rerank_count: None,
            batch_size: None,
        });
        // Two declared match_features expressions — both must lower.
        spec.match_features = vec!["1.0".into(), "2.0".into()];
        let compiled = CompiledRankProfile::compile(spec, f).unwrap();
        let qctx = QueryContext::default();
        let pipe = compiled.materialize(&qctx).unwrap();
        assert_eq!(
            pipe.match_features.len(),
            2,
            "match_features mapping must include both declared expressions"
        );
        assert_eq!(pipe.match_features[0].0.as_ref(), "1.0");
        assert_eq!(pipe.match_features[1].0.as_ref(), "2.0");
        // Score node is at idx 0 → the two match_features executors take
        // 1 and 2.
        assert_eq!(pipe.match_features[0].1.0, 1);
        assert_eq!(pipe.match_features[1].1.0, 2);
        // The RankProgram now has 3 executors total (score + 2 features).
        assert_eq!(pipe.first.num_executors(), 3);
    }

    #[test]
    fn materialize_lowers_summary_features_into_pipeline_mapping() {
        // R-7c.5b: spec.summary_features expressions lower into
        // additional executors in the same first-phase RankProgram.
        // The (name, ExecutorIdx) mapping surfaces on
        // RankPipeline.summary_features, parallel to
        // RankPipeline.match_features (R-7c.5).
        let f = factory();
        let mut spec = RankProfileSpec::new("with_sumf");
        spec.first_phase = Some(PhaseSpec {
            expression: "1.0".into(),
            heap_size: Some(10),
            rerank_count: None,
            batch_size: None,
        });
        spec.summary_features = vec!["1.0".into(), "2.0".into(), "3.0".into()];
        let compiled = CompiledRankProfile::compile(spec, f).unwrap();
        let qctx = QueryContext::default();
        let pipe = compiled.materialize(&qctx).unwrap();
        assert_eq!(pipe.summary_features.len(), 3);
        assert_eq!(pipe.summary_features[0].0.as_ref(), "1.0");
        assert_eq!(pipe.summary_features[1].0.as_ref(), "2.0");
        assert_eq!(pipe.summary_features[2].0.as_ref(), "3.0");
        // Score node is idx 0; no match_features → summary_features
        // take 1, 2, 3.
        assert_eq!(pipe.summary_features[0].1.0, 1);
        assert_eq!(pipe.summary_features[1].1.0, 2);
        assert_eq!(pipe.summary_features[2].1.0, 3);
        // RankProgram = score + 3 summary executors = 4 total.
        assert_eq!(pipe.first.num_executors(), 4);
        // match_features mapping stays empty.
        assert!(pipe.match_features.is_empty());
    }

    #[test]
    fn materialize_lowers_match_then_summary_features_in_separate_groups() {
        // Both groups declared. match_features take the executor
        // slots right after the score (idx 1..M+1); summary_features
        // take the slots after that (idx M+1..M+S+1). This stable
        // layout is what the pipeline relies on per `last_output(idx)`.
        let f = factory();
        let mut spec = RankProfileSpec::new("both");
        spec.first_phase = Some(PhaseSpec {
            expression: "1.0".into(),
            heap_size: Some(10),
            rerank_count: None,
            batch_size: None,
        });
        spec.match_features = vec!["1.0".into(), "2.0".into()];
        spec.summary_features = vec!["3.0".into()];
        let compiled = CompiledRankProfile::compile(spec, f).unwrap();
        let qctx = QueryContext::default();
        let pipe = compiled.materialize(&qctx).unwrap();
        assert_eq!(pipe.match_features.len(), 2);
        assert_eq!(pipe.summary_features.len(), 1);
        // match_features at 1, 2; summary at 3 (after the two match).
        assert_eq!(pipe.match_features[0].1.0, 1);
        assert_eq!(pipe.match_features[1].1.0, 2);
        assert_eq!(pipe.summary_features[0].1.0, 3);
        assert_eq!(pipe.first.num_executors(), 4);
    }

    #[test]
    fn materialize_with_no_summary_features_emits_empty_mapping() {
        let f = factory();
        let mut spec = RankProfileSpec::new("no_sumf");
        spec.first_phase = Some(PhaseSpec {
            expression: "1.0".into(),
            heap_size: Some(10),
            rerank_count: None,
            batch_size: None,
        });
        let compiled = CompiledRankProfile::compile(spec, f).unwrap();
        let qctx = QueryContext::default();
        let pipe = compiled.materialize(&qctx).unwrap();
        assert!(pipe.summary_features.is_empty());
    }

    #[test]
    fn materialize_with_no_match_features_emits_empty_mapping() {
        let f = factory();
        let mut spec = RankProfileSpec::new("no_matchf");
        spec.first_phase = Some(PhaseSpec {
            expression: "1.0".into(),
            heap_size: Some(10),
            rerank_count: None,
            batch_size: None,
        });
        let compiled = CompiledRankProfile::compile(spec, f).unwrap();
        let qctx = QueryContext::default();
        let pipe = compiled.materialize(&qctx).unwrap();
        assert!(
            pipe.match_features.is_empty(),
            "profiles without match_features must produce an empty mapping (NFR-9)"
        );
    }

    #[test]
    fn materialize_with_second_phase_pipes_rerank_count() {
        let f = factory();
        let mut spec = RankProfileSpec::new("two_phase");
        spec.first_phase = Some(PhaseSpec {
            expression: "1.0".into(),
            heap_size: Some(1000),
            rerank_count: None,
            batch_size: None,
        });
        spec.second_phase = Some(PhaseSpec {
            expression: "2.0".into(),
            heap_size: None,
            rerank_count: Some(50),
            batch_size: Some(8),
        });
        let compiled = CompiledRankProfile::compile(spec, f).unwrap();
        let qctx = QueryContext::default();
        let pipe = compiled.materialize(&qctx).unwrap();
        assert_eq!(pipe.heap_size, 1000);
        assert_eq!(pipe.rerank_count, 50);
        assert!(pipe.second.is_some());
    }
}
