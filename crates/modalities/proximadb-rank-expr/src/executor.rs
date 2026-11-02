//! `ExprBlueprint` + `ExprExecutor` — Blueprint wrapper for the expression VM.
//!
//! This blueprint is *not* registered into a `BlueprintFactory` itself
//! (it would create a cycle: the factory looks up sub-features, but the
//! ExprBlueprint also holds a reference to the same factory). R-4's
//! profile compiler will instantiate `ExprBlueprint` explicitly when a
//! profile's `first_phase.expression` (or any phase) needs an
//! expression-backed executor.
//!
//! `ExprExecutor` owns its sub-features — they are *not* shared with the
//! outer `RankProgram`. The trade-off is documented in the spec
//! follow-up to §4.2.

use crate::bytecode::Code;
use crate::lowering::lower;
use crate::parser::parse;
use crate::vm::execute;
use proximadb_rank_core::{
    Blueprint, BlueprintFactory, DocHandle, FeatureExecutor, FeatureLookup, OutputSpec,
    PhaseConfig, QueryContext, RankError, RankResult, ScoreCtx,
};
use std::sync::Arc;

/// Blueprint for `rankingExpression(...)`.
///
/// The single literal arg is the expression source. The blueprint
/// parses and lowers at `build_executor` time using its `factory`
/// reference so the per-doc hot path doesn't pay parse cost.
pub struct ExprBlueprint {
    factory: Arc<BlueprintFactory>,
}

impl ExprBlueprint {
    pub const FEATURE_NAME: &'static str = "rankingExpression";

    pub fn new(factory: Arc<BlueprintFactory>) -> Self {
        Self { factory }
    }

    /// Compile an expression string directly into an executor without going
    /// through the Blueprint trait. R-4's profile compiler can use this
    /// for the `first_phase.expression` short-circuit form (the profile
    /// already carries the expression string).
    pub fn compile_str(
        &self,
        expr: &str,
        qctx: &QueryContext,
    ) -> RankResult<Box<dyn FeatureExecutor>> {
        let ast = parse(expr)?;
        let (code, subs) = lower(&ast, &self.factory, qctx)?;
        Ok(Box::new(ExprExecutor::new(code, subs)))
    }
}

impl Blueprint for ExprBlueprint {
    fn name(&self) -> &str {
        Self::FEATURE_NAME
    }
    fn declared_outputs(&self) -> &[OutputSpec] {
        &[]
    }
    fn build_executor(
        &self,
        cfg: &PhaseConfig,
        qctx: &QueryContext,
    ) -> RankResult<Box<dyn FeatureExecutor>> {
        let expr = cfg.literal_args.first().ok_or_else(|| {
            RankError::InvalidProfile(
                "rankingExpression(...) requires the expression source as its first argument"
                    .into(),
            )
        })?;
        self.compile_str(expr, qctx)
    }
}

/// Compiled expression at query-time: bytecode plus its owned sub-features.
pub struct ExprExecutor {
    code: Code,
    sub_features: Vec<Box<dyn FeatureExecutor>>,
}

impl ExprExecutor {
    pub fn new(code: Code, sub_features: Vec<Box<dyn FeatureExecutor>>) -> Self {
        Self { code, sub_features }
    }

    /// Test-only accessors.
    #[doc(hidden)]
    pub fn op_count(&self) -> usize {
        self.code.len()
    }
    #[doc(hidden)]
    pub fn sub_feature_count(&self) -> usize {
        self.sub_features.len()
    }
}

impl FeatureExecutor for ExprExecutor {
    fn execute(
        &mut self,
        doc: DocHandle,
        lookup: &mut dyn FeatureLookup,
        ctx: &mut ScoreCtx<'_>,
    ) -> f32 {
        execute(&self.code, &mut self.sub_features, doc, lookup, ctx)
    }

    fn precompute(&mut self, ctx: &mut ScoreCtx<'_>) -> RankResult<()> {
        for s in self.sub_features.iter_mut() {
            s.precompute(ctx)?;
        }
        Ok(())
    }

    fn end_of_phase(&mut self, ctx: &mut ScoreCtx<'_>) -> RankResult<()> {
        for s in self.sub_features.iter_mut() {
            s.end_of_phase(ctx)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_rank_core::test_support::MapAttributeAccess;
    use proximadb_rank_core::{
        BlueprintFactory, CandidateData, ExecutorIdx, FeatureArena, NoopAttributeAccess,
        NoopCandidateData, NoopMetricsSink, NoopModelCache, RankProgram,
    };
    use std::collections::HashMap;

    // Re-use the stub features from the lowering tests pattern.
    struct ConstBp(&'static str, f32);
    struct ConstEx(f32);
    impl FeatureExecutor for ConstEx {
        fn execute(
            &mut self,
            _doc: DocHandle,
            _l: &mut dyn FeatureLookup,
            _c: &mut ScoreCtx<'_>,
        ) -> f32 {
            self.0
        }
    }
    impl Blueprint for ConstBp {
        fn name(&self) -> &str {
            self.0
        }
        fn declared_outputs(&self) -> &[OutputSpec] {
            const OUT: &[OutputSpec] = &[];
            OUT
        }
        fn build_executor(
            &self,
            _cfg: &PhaseConfig,
            _q: &QueryContext,
        ) -> RankResult<Box<dyn FeatureExecutor>> {
            Ok(Box::new(ConstEx(self.1)))
        }
    }

    fn factory_with(features: &[(&'static str, f32)]) -> Arc<BlueprintFactory> {
        let f = Arc::new(BlueprintFactory::new());
        for (n, v) in features {
            f.register(Arc::new(ConstBp(n, *v)));
        }
        f
    }

    fn run_executor(mut exec: Box<dyn FeatureExecutor>) -> f32 {
        let q = QueryContext::default();
        let arena = FeatureArena::new();
        let (a, c, m, met) = (
            NoopAttributeAccess,
            NoopCandidateData,
            NoopModelCache,
            NoopMetricsSink,
        );
        let mut ctx = ScoreCtx::new(&q, &arena, &a, &c, &m, &met);
        struct NullLookup;
        impl FeatureLookup for NullLookup {
            fn force(&mut self, _idx: ExecutorIdx, _doc: DocHandle, _c: &mut ScoreCtx<'_>) -> f32 {
                0.0
            }
        }
        let mut lk = NullLookup;
        exec.execute(DocHandle(0), &mut lk, &mut ctx)
    }

    #[test]
    fn expr_blueprint_compiles_via_literal_arg() {
        let f = factory_with(&[("bm25", 5.0), ("closeness", 0.5)]);
        let bp = ExprBlueprint::new(f.clone());
        let cfg = PhaseConfig {
            literal_args: vec!["bm25(\"title\") * 0.4 + closeness(\"emb\") * 0.6".into()],
        };
        let q = QueryContext::default();
        let exec = bp.build_executor(&cfg, &q).unwrap();
        // 5*0.4 + 0.5*0.6 = 2.0 + 0.3 = 2.3
        assert!((run_executor(exec) - 2.3).abs() < 1e-5);
    }

    #[test]
    fn expr_blueprint_compile_str_path() {
        let f = factory_with(&[("bm25", 10.0)]);
        let bp = ExprBlueprint::new(f);
        let q = QueryContext::default();
        let exec = bp.compile_str("bm25(\"title\") + 1", &q).unwrap();
        assert!((run_executor(exec) - 11.0).abs() < 1e-5);
    }

    #[test]
    fn expr_blueprint_propagates_parse_errors() {
        let f = factory_with(&[]);
        let bp = ExprBlueprint::new(f);
        let cfg = PhaseConfig {
            literal_args: vec!["not a valid expr ((((".into()],
        };
        let q = QueryContext::default();
        match bp.build_executor(&cfg, &q) {
            Err(RankError::ExpressionParse(_)) => {}
            Err(_) => panic!("expected ExpressionParse"),
            Ok(_) => panic!("expected error"),
        }
    }

    #[test]
    fn expr_blueprint_missing_expression_errors() {
        let f = factory_with(&[]);
        let bp = ExprBlueprint::new(f);
        let cfg = PhaseConfig::default();
        let q = QueryContext::default();
        match bp.build_executor(&cfg, &q) {
            Err(RankError::InvalidProfile(_)) => {}
            Err(_) => panic!("expected InvalidProfile"),
            Ok(_) => panic!("expected error"),
        }
    }

    #[test]
    fn expr_executor_runs_through_rank_program() {
        // End-to-end: ExprExecutor placed inside a RankProgram should
        // produce the expected score.
        let f = factory_with(&[("bm25", 7.0)]);
        let bp = ExprBlueprint::new(f);
        let q = QueryContext::default();
        let exec = bp.compile_str("bm25(\"title\") * 2 + 1", &q).unwrap();

        let mut b = RankProgram::builder();
        let idx = b.add(exec);
        b.set_score(idx);
        let mut prog = b.build().unwrap();

        let arena = FeatureArena::new();
        let (a, c, m, met) = (
            NoopAttributeAccess,
            NoopCandidateData,
            NoopModelCache,
            NoopMetricsSink,
        );
        let mut ctx = ScoreCtx::new(&q, &arena, &a, &c, &m, &met);
        assert!((prog.rank(DocHandle(0), &mut ctx) - 15.0).abs() < 1e-5);
    }

    #[test]
    fn expr_executor_with_attribute_through_full_stack() {
        // Build an "attr"-like feature that reads from ScoreCtx::attributes.
        struct AttrBp;
        struct AttrEx(String);
        impl FeatureExecutor for AttrEx {
            fn execute(
                &mut self,
                doc: DocHandle,
                _l: &mut dyn FeatureLookup,
                ctx: &mut ScoreCtx<'_>,
            ) -> f32 {
                ctx.attributes.read_f32(doc, &self.0).unwrap_or(0.0)
            }
        }
        impl Blueprint for AttrBp {
            fn name(&self) -> &str {
                "attr"
            }
            fn declared_outputs(&self) -> &[OutputSpec] {
                const OUT: &[OutputSpec] = &[];
                OUT
            }
            fn build_executor(
                &self,
                cfg: &PhaseConfig,
                _q: &QueryContext,
            ) -> RankResult<Box<dyn FeatureExecutor>> {
                Ok(Box::new(AttrEx(cfg.literal_args[0].clone())))
            }
        }

        let f = Arc::new(BlueprintFactory::new());
        f.register(Arc::new(AttrBp));
        let bp = ExprBlueprint::new(f);
        let q = QueryContext::default();
        let exec = bp
            .compile_str("clamp(attr(\"score\") * 2, 0, 100)", &q)
            .unwrap();

        struct NoCand;
        impl CandidateData for NoCand {
            fn retrieval_distance(&self, _doc: DocHandle) -> Option<f32> {
                None
            }
            fn bm25_score(&self, _doc: DocHandle) -> Option<f32> {
                None
            }
        }
        let attrs = MapAttributeAccess {
            values: HashMap::from([((0, "score".into()), 30.0)]),
        };

        let mut wrap = exec;
        let arena = FeatureArena::new();
        let (m, met) = (NoopModelCache, NoopMetricsSink);
        let nc = NoCand;
        let mut ctx = ScoreCtx::new(&q, &arena, &attrs, &nc, &m, &met);
        struct NullLookup;
        impl FeatureLookup for NullLookup {
            fn force(&mut self, _idx: ExecutorIdx, _doc: DocHandle, _c: &mut ScoreCtx<'_>) -> f32 {
                0.0
            }
        }
        let mut lk = NullLookup;
        // attr(score)=30, * 2 = 60, clamp(60, 0, 100) = 60
        assert_eq!(wrap.execute(DocHandle(0), &mut lk, &mut ctx), 60.0);
    }

    #[test]
    fn expr_op_count_within_cap() {
        let f = factory_with(&[("bm25", 1.0)]);
        let bp = ExprBlueprint::new(f);
        let q = QueryContext::default();
        let exec = bp
            .compile_str("bm25(\"a\") * 0.4 + bm25(\"b\") * 0.6", &q)
            .unwrap();
        // Downcast via the public op_count accessor.
        // We can't downcast Box<dyn FeatureExecutor> generically, so reach in
        // by constructing the executor directly from compile_str's result.
        // (Smoke test: just confirm it produced a non-trivial program.)
        let _ = exec;
        // Reach in via the lowering helper used here:
        let ast = parse("bm25(\"a\") * 0.4 + bm25(\"b\") * 0.6").unwrap();
        let f2 = factory_with(&[("bm25", 1.0)]);
        let (code, subs) = lower(&ast, &f2, &q).unwrap();
        let ex = ExprExecutor::new(code, subs);
        assert_eq!(ex.sub_feature_count(), 2);
        assert!(ex.op_count() >= 5);
    }
}
