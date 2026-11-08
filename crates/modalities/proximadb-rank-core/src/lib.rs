//! ProximaDB multi-phase ranking framework — core types.
//!
//! Architecture: see `roadmap/RANKING_FRAMEWORK_SPEC_2026_05_23.md` (R-1).
//!
//! The framework follows the Vespa pattern:
//! `Blueprint` (schema-time prototype) → `BlueprintFactory` (registry)
//! → `FeatureExecutor` (query-time instance) → `RankProgram` (per-phase DAG)
//! → `RankPipeline` (first/second/global phase orchestrator).
//!
//! Canonical score types (`PhaseId`, `ScoreComponent`, `ScoreVector`) live
//! in `proximadb-kernel` and are re-exported here for convenience.

pub mod arena;
pub mod blueprint;
pub mod context;
pub mod error;
pub mod executor;
pub mod pipeline;
pub mod program;
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
pub mod types;

pub use arena::FeatureArena;
pub use blueprint::{Blueprint, BlueprintFactory, InputSpec, OutputSpec, PhaseConfig, ValueKind};
pub use context::{
    AttributeAccess, BatchSlot, CandidateData, ModelCache, NoopAttributeAccess, NoopCandidateData,
    NoopMetricsSink, NoopModelCache, QueryContext, RankMetricsSink, ScoreCtx,
};
pub use error::{RankError, RankResult};
pub use executor::{FeatureExecutor, FeatureLookup};
pub use pipeline::{
    ConstantMultiplierSecondPhaseScorer, GlobalScorer, IdentityGlobalScorer,
    PassthroughSecondPhaseScorer, PhaseBudget, PhaseOutcome, RankPipeline, ScoredHit,
    SecondPhaseScorer,
};
pub use program::RankProgram;
pub use proximadb_kernel::{PhaseId, ScoreComponent, ScoreVector};
pub use types::{DocHandle, ExecutorIdx, FeatureRef};
