//! Built-in ranking features for ProximaDB.
//!
//! Each module is one feature: a [`Blueprint`] implementation plus its
//! [`FeatureExecutor`]. Call [`register_builtins`] at server startup to
//! populate a shared [`BlueprintFactory`] with everything in this crate.
//!
//! ```ignore
//! use proximadb_rank_core::BlueprintFactory;
//! use proximadb_rank_features::register_builtins;
//!
//! let factory = BlueprintFactory::new();
//! register_builtins(&factory);
//! assert!(factory.lookup("bm25").is_some());
//! ```
//!
//! Features in R-2:
//! - `attribute(field)` — raw scalar column value.
//! - `closeness(field)` — `1 / (1 + retrieval_distance)`; Vespa convention.
//! - `bm25(field)` — surfaces upstream BM25 score.
//! - `freshness(field, half_life_ms?)` — exp decay against query-time clock.
//! - `decay(field, half_life)` — generic exp decay over a numeric attribute.
//!
//! Features deferred to later phases:
//! - `cosine(field, query_vec)` — needs tensor-typed columns (R-9).
//! - `model(model_id, ...)` — needs ONNX scorer (R-5).
//! - `rankingExpression(expr)` — needs expression VM (R-3).
//! - `cross_modal_score(strategy)` — needs adapter to existing reranker (R-6).
//!
//! See `roadmap/RANKING_FRAMEWORK_SPEC_2026_05_23.md` §4.4.

pub mod attribute;
pub mod bm25;
pub mod closeness;
pub mod decay;
pub mod freshness;

pub use attribute::{AttributeBlueprint, AttributeExecutor};
pub use bm25::{Bm25Blueprint, Bm25Executor};
pub use closeness::{ClosenessBlueprint, ClosenessExecutor};
pub use decay::{DecayBlueprint, DecayExecutor};
pub use freshness::{FreshnessBlueprint, FreshnessExecutor};

use proximadb_rank_core::BlueprintFactory;
use std::sync::Arc;

/// Register every R-2 built-in feature into the factory. Idempotent: the
/// underlying factory is last-write-wins, so re-calling with the same set
/// just replaces the registrations.
pub fn register_builtins(factory: &BlueprintFactory) {
    factory.register(Arc::new(AttributeBlueprint));
    factory.register(Arc::new(BlueprintWrapper {
        name: ClosenessBlueprint::FEATURE_NAME,
        inner: Arc::new(ClosenessBlueprint),
    }));
    factory.register(Arc::new(Bm25Blueprint));
    factory.register(Arc::new(FreshnessBlueprint));
    factory.register(Arc::new(DecayBlueprint));
}

// NOTE: BlueprintWrapper exists only so the closeness blueprint can have
// a static `&'static str` name on the struct without `name()` returning a
// borrow into self with a non-'static lifetime. The cleaner solution is to
// have every Blueprint store its own `&'static str` name; refactor in R-4
// when the resolver lands.
struct BlueprintWrapper {
    name: &'static str,
    inner: Arc<dyn proximadb_rank_core::Blueprint>,
}

impl proximadb_rank_core::Blueprint for BlueprintWrapper {
    fn name(&self) -> &str {
        self.name
    }
    fn declared_outputs(&self) -> &[proximadb_rank_core::OutputSpec] {
        self.inner.declared_outputs()
    }
    fn build_executor(
        &self,
        cfg: &proximadb_rank_core::PhaseConfig,
        qctx: &proximadb_rank_core::QueryContext,
    ) -> proximadb_rank_core::RankResult<Box<dyn proximadb_rank_core::FeatureExecutor>> {
        self.inner.build_executor(cfg, qctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_rank_core::test_support::MapAttributeAccess;
    use proximadb_rank_core::{
        BlueprintFactory, CandidateData, DocHandle, FeatureArena, NoopMetricsSink, NoopModelCache,
        PhaseConfig, QueryContext, RankProgram, ScoreCtx,
    };
    use std::collections::HashMap;

    struct AllCands {
        d: HashMap<u32, f32>,
        b: HashMap<u32, f32>,
    }
    impl CandidateData for AllCands {
        fn retrieval_distance(&self, doc: DocHandle) -> Option<f32> {
            self.d.get(&doc.0).copied()
        }
        fn bm25_score(&self, doc: DocHandle) -> Option<f32> {
            self.b.get(&doc.0).copied()
        }
    }

    #[test]
    fn register_builtins_populates_all_features() {
        let f = BlueprintFactory::new();
        register_builtins(&f);
        for name in ["attribute", "closeness", "bm25", "freshness", "decay"] {
            assert!(
                f.lookup(name).is_some(),
                "expected '{name}' to be registered"
            );
        }
    }

    #[test]
    fn end_to_end_attribute_via_factory() {
        // End-to-end: register builtins, look up `attribute`, build an
        // executor, run a RankProgram, and verify the score.
        let f = BlueprintFactory::new();
        register_builtins(&f);
        let bp = f.lookup("attribute").unwrap();
        let cfg = PhaseConfig {
            literal_args: vec!["price".into()],
        };
        let q = QueryContext::default();
        let exec = bp.build_executor(&cfg, &q).unwrap();

        let mut b = RankProgram::builder();
        let idx = b.add(exec);
        b.set_score(idx);
        let mut prog = b.build().unwrap();

        let attrs = MapAttributeAccess {
            values: HashMap::from([((42, "price".into()), 99.5)]),
        };
        let cands = AllCands {
            d: HashMap::new(),
            b: HashMap::new(),
        };
        let arena = FeatureArena::new();
        let met = NoopMetricsSink;
        let m = NoopModelCache;
        let mut ctx = ScoreCtx::new(&q, &arena, &attrs, &cands, &m, &met);
        assert_eq!(prog.rank(DocHandle(42), &mut ctx), 99.5);
    }

    #[test]
    fn end_to_end_closeness_via_factory() {
        let f = BlueprintFactory::new();
        register_builtins(&f);
        let bp = f.lookup("closeness").unwrap();
        let cfg = PhaseConfig {
            literal_args: vec!["embedding".into()],
        };
        let q = QueryContext::default();
        let exec = bp.build_executor(&cfg, &q).unwrap();

        let mut b = RankProgram::builder();
        let idx = b.add(exec);
        b.set_score(idx);
        let mut prog = b.build().unwrap();

        let attrs = MapAttributeAccess {
            values: HashMap::new(),
        };
        let cands = AllCands {
            d: HashMap::from([(7, 1.0)]),
            b: HashMap::new(),
        };
        let arena = FeatureArena::new();
        let met = NoopMetricsSink;
        let m = NoopModelCache;
        let mut ctx = ScoreCtx::new(&q, &arena, &attrs, &cands, &m, &met);
        // closeness at d=1.0 → 1/(1+1) = 0.5
        assert!((prog.rank(DocHandle(7), &mut ctx) - 0.5).abs() < 1e-6);
    }
}
