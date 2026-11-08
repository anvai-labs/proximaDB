//! `decay(field, half_life)` — generic exponential decay over a numeric
//! attribute.
//!
//! `score = exp(- ln(2) * value / half_life)`
//!
//! Unlike `freshness(...)`, `decay(...)` doesn't subtract from "now" —
//! it treats the attribute value as the distance/age/cost directly. Useful
//! for ranking by price, hop count, distance-from-user, etc. where smaller
//! values should score higher.

use proximadb_rank_core::{
    Blueprint, DocHandle, FeatureExecutor, FeatureLookup, OutputSpec, PhaseConfig, QueryContext,
    RankError, RankResult, ScoreCtx,
};

pub struct DecayBlueprint;

impl DecayBlueprint {
    pub const FEATURE_NAME: &'static str = "decay";
}

impl Blueprint for DecayBlueprint {
    fn name(&self) -> &str {
        Self::FEATURE_NAME
    }
    fn declared_outputs(&self) -> &[OutputSpec] {
        &[]
    }
    fn build_executor(
        &self,
        cfg: &PhaseConfig,
        _qctx: &QueryContext,
    ) -> RankResult<Box<dyn FeatureExecutor>> {
        let field = cfg.literal_args.first().cloned().ok_or_else(|| {
            RankError::InvalidProfile(
                "decay(...) requires a field name as its first argument".into(),
            )
        })?;
        let half_life = cfg
            .literal_args
            .get(1)
            .ok_or_else(|| {
                RankError::InvalidProfile(
                    "decay(...) requires a half_life as its second argument".into(),
                )
            })?
            .parse::<f64>()
            .map_err(|_| {
                RankError::InvalidProfile("decay(...) half_life must be a positive number".into())
            })?;
        if !(half_life.is_finite() && half_life > 0.0) {
            return Err(RankError::InvalidProfile(format!(
                "decay(...) half_life must be positive and finite, got: {half_life}"
            )));
        }
        Ok(Box::new(DecayExecutor { field, half_life }))
    }
}

pub struct DecayExecutor {
    pub field: String,
    pub half_life: f64,
}

impl FeatureExecutor for DecayExecutor {
    fn execute(
        &mut self,
        doc: DocHandle,
        _lookup: &mut dyn FeatureLookup,
        ctx: &mut ScoreCtx<'_>,
    ) -> f32 {
        let v = match ctx.attributes.read_f32(doc, &self.field) {
            None => return 0.0,
            Some(x) => (x as f64).max(0.0),
        };
        let lambda = std::f64::consts::LN_2 / self.half_life;
        (-lambda * v).exp() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_rank_core::test_support::MapAttributeAccess;
    use proximadb_rank_core::{
        FeatureArena, NoopCandidateData, NoopMetricsSink, NoopModelCache, RankProgram,
    };
    use std::collections::HashMap;

    fn build(field: &str, half_life: &str) -> RankProgram {
        let bp = DecayBlueprint;
        let cfg = PhaseConfig {
            literal_args: vec![field.into(), half_life.into()],
        };
        let q = QueryContext::default();
        let exec = bp.build_executor(&cfg, &q).unwrap();
        let mut b = RankProgram::builder();
        let idx = b.add(exec);
        b.set_score(idx);
        b.build().unwrap()
    }

    fn run(prog: &mut RankProgram, attrs: &MapAttributeAccess, doc: u32) -> f32 {
        let q = QueryContext::default();
        let arena = FeatureArena::new();
        let (c, m, met) = (NoopCandidateData, NoopModelCache, NoopMetricsSink);
        let mut ctx = ScoreCtx::new(&q, &arena, attrs, &c, &m, &met);
        prog.rank(DocHandle(doc), &mut ctx)
    }

    #[test]
    fn decay_at_zero_value_is_one() {
        let mut prog = build("hops", "5");
        let attrs = MapAttributeAccess {
            values: HashMap::from([((1, "hops".into()), 0.0)]),
        };
        assert!((run(&mut prog, &attrs, 1) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn decay_at_half_life_value_is_half() {
        let mut prog = build("hops", "5");
        let attrs = MapAttributeAccess {
            values: HashMap::from([((1, "hops".into()), 5.0)]),
        };
        assert!((run(&mut prog, &attrs, 1) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn decay_at_double_half_life_is_quarter() {
        let mut prog = build("hops", "5");
        let attrs = MapAttributeAccess {
            values: HashMap::from([((1, "hops".into()), 10.0)]),
        };
        assert!((run(&mut prog, &attrs, 1) - 0.25).abs() < 1e-5);
    }

    #[test]
    fn decay_missing_returns_zero() {
        let mut prog = build("hops", "5");
        let attrs = MapAttributeAccess {
            values: HashMap::new(),
        };
        assert_eq!(run(&mut prog, &attrs, 1), 0.0);
    }

    #[test]
    fn decay_negative_value_clamped_to_zero() {
        // Negative attribute → treat as zero (best score) rather than
        // returning > 1.0 which would break the [0,1] contract.
        let mut prog = build("hops", "5");
        let attrs = MapAttributeAccess {
            values: HashMap::from([((1, "hops".into()), -3.0)]),
        };
        assert!((run(&mut prog, &attrs, 1) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn build_rejects_missing_half_life() {
        let bp = DecayBlueprint;
        let q = QueryContext::default();
        let cfg = PhaseConfig {
            literal_args: vec!["x".into()],
        };
        match bp.build_executor(&cfg, &q) {
            Err(RankError::InvalidProfile(_)) => {}
            Err(_) => panic!("expected InvalidProfile, got a different RankError"),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }
}
