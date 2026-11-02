//! `attribute(field)` — exposes a raw scalar column value as a ranking feature.
//!
//! The simplest non-trivial feature. The blueprint is configured at compile
//! time with the column name (PhaseConfig.literal_args[0]); the executor
//! looks the value up per-doc via `ctx.attributes.read_f32(doc, field)`.
//!
//! Missing values default to `0.0`. Callers that need a different sentinel
//! should wrap with `if(attribute(x) > 0, ..., ...)` once the expression
//! VM lands in R-3.

use proximadb_rank_core::{
    Blueprint, DocHandle, FeatureExecutor, FeatureLookup, InputSpec, OutputSpec, PhaseConfig,
    RankError, RankResult, ScoreCtx,
};

pub struct AttributeBlueprint;

impl AttributeBlueprint {
    pub const FEATURE_NAME: &'static str = "attribute";
}

impl Blueprint for AttributeBlueprint {
    fn name(&self) -> &str {
        Self::FEATURE_NAME
    }
    fn declared_inputs(&self) -> &[InputSpec] {
        &[]
    }
    fn declared_outputs(&self) -> &[OutputSpec] {
        // Single scalar output; name "out" matches the v1 convention.
        const OUT: &[OutputSpec] = &[];
        OUT
    }
    fn build_executor(
        &self,
        cfg: &PhaseConfig,
        _qctx: &proximadb_rank_core::QueryContext,
    ) -> RankResult<Box<dyn FeatureExecutor>> {
        let field = cfg
            .literal_args
            .first()
            .ok_or_else(|| {
                RankError::InvalidProfile(
                    "attribute(...) requires a field name as its first argument".into(),
                )
            })?
            .clone();
        Ok(Box::new(AttributeExecutor { field }))
    }
}

pub struct AttributeExecutor {
    pub field: String,
}

impl FeatureExecutor for AttributeExecutor {
    fn execute(
        &mut self,
        doc: DocHandle,
        _lookup: &mut dyn FeatureLookup,
        ctx: &mut ScoreCtx<'_>,
    ) -> f32 {
        ctx.attributes.read_f32(doc, &self.field).unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_rank_core::test_support::MapAttributeAccess;
    use proximadb_rank_core::{
        FeatureArena, NoopCandidateData, NoopMetricsSink, NoopModelCache, QueryContext, RankProgram,
    };
    use std::collections::HashMap;

    fn build_program(field: &str) -> RankProgram {
        let bp = AttributeBlueprint;
        let cfg = PhaseConfig {
            literal_args: vec![field.into()],
        };
        let q = QueryContext::default();
        let exec = bp.build_executor(&cfg, &q).unwrap();
        let mut b = RankProgram::builder();
        let idx = b.add(exec);
        b.set_score(idx);
        b.build().unwrap()
    }

    #[test]
    fn attribute_blueprint_reads_column() {
        let mut prog = build_program("price");
        let attrs = MapAttributeAccess {
            values: HashMap::from([((1, "price".into()), 42.0)]),
        };
        let q = QueryContext::default();
        let arena = FeatureArena::new();
        let (c, m, met) = (NoopCandidateData, NoopModelCache, NoopMetricsSink);
        let mut ctx = ScoreCtx::new(&q, &arena, &attrs, &c, &m, &met);
        assert_eq!(prog.rank(DocHandle(1), &mut ctx), 42.0);
    }

    #[test]
    fn attribute_missing_value_returns_zero() {
        let mut prog = build_program("price");
        let attrs = MapAttributeAccess {
            values: HashMap::new(),
        };
        let q = QueryContext::default();
        let arena = FeatureArena::new();
        let (c, m, met) = (NoopCandidateData, NoopModelCache, NoopMetricsSink);
        let mut ctx = ScoreCtx::new(&q, &arena, &attrs, &c, &m, &met);
        assert_eq!(prog.rank(DocHandle(99), &mut ctx), 0.0);
    }

    #[test]
    fn attribute_build_without_field_arg_errors() {
        let bp = AttributeBlueprint;
        let cfg = PhaseConfig::default();
        let q = QueryContext::default();
        match bp.build_executor(&cfg, &q) {
            Err(RankError::InvalidProfile(msg)) => assert!(msg.contains("attribute")),
            Err(_) => panic!("expected InvalidProfile, got a different RankError"),
            Ok(_) => panic!("expected error"),
        }
    }
}
