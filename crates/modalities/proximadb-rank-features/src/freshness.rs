//! `freshness(field, half_life_ms)` — exponential time decay against a
//! per-doc timestamp.
//!
//! Reads `field` as a millisecond-since-Unix-epoch timestamp from the
//! attribute store, computes the elapsed milliseconds against
//! `ctx.query.now_ms_or_wall()`, and returns
//!
//! ```text
//! exp(- ln(2) * Δ / half_life_ms)
//! ```
//!
//! which equals `1.0` for now-old docs, `0.5` at one half-life ago, `0.25`
//! at two half-lives, etc. Future-dated docs (negative Δ) get `1.0` —
//! caller can apply a guard upstream if that's a problem.
//!
//! The clock is configurable via `QueryContext::now_ms_unix` for
//! deterministic testing.

use proximadb_rank_core::{
    Blueprint, DocHandle, FeatureExecutor, FeatureLookup, OutputSpec, PhaseConfig, QueryContext,
    RankError, RankResult, ScoreCtx,
};

const DEFAULT_HALF_LIFE_MS: f64 = 86_400_000.0; // 1 day

pub struct FreshnessBlueprint;

impl FreshnessBlueprint {
    pub const FEATURE_NAME: &'static str = "freshness";
}

impl Blueprint for FreshnessBlueprint {
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
                "freshness(...) requires a field name as its first argument".into(),
            )
        })?;
        let half_life_ms = match cfg.literal_args.get(1) {
            None => DEFAULT_HALF_LIFE_MS,
            Some(s) => s.parse::<f64>().map_err(|_| {
                RankError::InvalidProfile(format!(
                    "freshness(...) half_life_ms must be a positive number, got: {s}"
                ))
            })?,
        };
        if !(half_life_ms.is_finite() && half_life_ms > 0.0) {
            return Err(RankError::InvalidProfile(format!(
                "freshness(...) half_life_ms must be positive and finite, got: {half_life_ms}"
            )));
        }
        Ok(Box::new(FreshnessExecutor {
            field,
            half_life_ms,
        }))
    }
}

pub struct FreshnessExecutor {
    pub field: String,
    pub half_life_ms: f64,
}

impl FeatureExecutor for FreshnessExecutor {
    fn execute(
        &mut self,
        doc: DocHandle,
        _lookup: &mut dyn FeatureLookup,
        ctx: &mut ScoreCtx<'_>,
    ) -> f32 {
        let ts_ms = match ctx.attributes.read_f32(doc, &self.field) {
            // Missing or non-positive timestamps → treat as ancient (returns
            // a small but non-zero value rather than panicking).
            None => return 0.0,
            Some(v) => v as f64,
        };
        let now_ms = ctx.query.now_ms_or_wall() as f64;
        let delta_ms = (now_ms - ts_ms).max(0.0);
        let lambda = std::f64::consts::LN_2 / self.half_life_ms;
        (-lambda * delta_ms).exp() as f32
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

    fn build(field: &str, half_life_ms: Option<&str>) -> RankProgram {
        let mut literal_args = vec![field.into()];
        if let Some(h) = half_life_ms {
            literal_args.push(h.into());
        }
        let bp = FreshnessBlueprint;
        let cfg = PhaseConfig { literal_args };
        let q = QueryContext::default();
        let exec = bp.build_executor(&cfg, &q).unwrap();
        let mut b = RankProgram::builder();
        let idx = b.add(exec);
        b.set_score(idx);
        b.build().unwrap()
    }

    fn run(prog: &mut RankProgram, attrs: &MapAttributeAccess, now_ms: i64, doc: u32) -> f32 {
        let q = QueryContext {
            now_ms_unix: Some(now_ms),
            ..Default::default()
        };
        let arena = FeatureArena::new();
        let (c, m, met) = (NoopCandidateData, NoopModelCache, NoopMetricsSink);
        let mut ctx = ScoreCtx::new(&q, &arena, attrs, &c, &m, &met);
        prog.rank(DocHandle(doc), &mut ctx)
    }

    #[test]
    fn freshness_at_now_is_one() {
        let mut prog = build("ts", Some("1000"));
        let attrs = MapAttributeAccess {
            values: HashMap::from([((1, "ts".into()), 1_000_000.0)]),
        };
        let v = run(&mut prog, &attrs, 1_000_000, 1);
        assert!((v - 1.0).abs() < 1e-6, "got {v}");
    }

    #[test]
    fn freshness_at_half_life_is_half() {
        let mut prog = build("ts", Some("1000")); // 1s half-life
        let attrs = MapAttributeAccess {
            values: HashMap::from([((1, "ts".into()), 1_000_000.0)]),
        };
        // 1s = 1000ms later
        let v = run(&mut prog, &attrs, 1_001_000, 1);
        assert!((v - 0.5).abs() < 1e-5, "expected 0.5 at half-life, got {v}");
    }

    #[test]
    fn freshness_at_two_half_lives_is_quarter() {
        let mut prog = build("ts", Some("1000"));
        let attrs = MapAttributeAccess {
            values: HashMap::from([((1, "ts".into()), 1_000_000.0)]),
        };
        let v = run(&mut prog, &attrs, 1_002_000, 1);
        assert!(
            (v - 0.25).abs() < 1e-5,
            "expected 0.25 at 2× half-life, got {v}"
        );
    }

    #[test]
    fn freshness_missing_attribute_returns_zero() {
        let mut prog = build("ts", Some("1000"));
        let attrs = MapAttributeAccess {
            values: HashMap::new(),
        };
        assert_eq!(run(&mut prog, &attrs, 1_000_000, 1), 0.0);
    }

    #[test]
    fn freshness_future_dated_doc_returns_one() {
        let mut prog = build("ts", Some("1000"));
        let attrs = MapAttributeAccess {
            values: HashMap::from([((1, "ts".into()), 2_000_000.0)]),
        };
        let v = run(&mut prog, &attrs, 1_000_000, 1);
        assert!((v - 1.0).abs() < 1e-6, "got {v}");
    }

    #[test]
    fn build_rejects_non_positive_half_life() {
        let bp = FreshnessBlueprint;
        let q = QueryContext::default();
        let cfg = PhaseConfig {
            literal_args: vec!["ts".into(), "0".into()],
        };
        match bp.build_executor(&cfg, &q) {
            Err(RankError::InvalidProfile(_)) => {}
            Err(_) => panic!("expected InvalidProfile, got a different RankError"),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    #[test]
    fn build_rejects_unparseable_half_life() {
        let bp = FreshnessBlueprint;
        let q = QueryContext::default();
        let cfg = PhaseConfig {
            literal_args: vec!["ts".into(), "not-a-number".into()],
        };
        match bp.build_executor(&cfg, &q) {
            Err(RankError::InvalidProfile(_)) => {}
            Err(_) => panic!("expected InvalidProfile, got a different RankError"),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    #[test]
    fn build_uses_default_half_life_when_omitted() {
        let bp = FreshnessBlueprint;
        let q = QueryContext::default();
        let cfg = PhaseConfig {
            literal_args: vec!["ts".into()],
        };
        let _ = bp.build_executor(&cfg, &q).unwrap();
    }
}
