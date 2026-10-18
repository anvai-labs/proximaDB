//! Error type for the ranking framework.
//!
//! See `roadmap/RANKING_FRAMEWORK_SPEC_2026_05_23.md` §4.1.1.

use crate::types::ExecutorIdx;
use proximadb_kernel::PhaseId;

#[derive(Debug, thiserror::Error)]
pub enum RankError {
    #[error("rank profile not found: {0}")]
    ProfileNotFound(String),

    #[error("rank profile validation failed: {0}")]
    InvalidProfile(String),

    #[error("feature not registered: {0}")]
    UnknownFeature(String),

    #[error("feature dependency cycle detected (executor {executor:?} would be entered twice)")]
    DependencyCycle { executor: ExecutorIdx },

    #[error("feature dependency depth exceeded max {max}")]
    DependencyTooDeep { max: usize },

    #[error("expression parse error: {0}")]
    ExpressionParse(String),

    #[error("expression type error: {0}")]
    ExpressionType(String),

    #[error("phase budget exceeded: {phase:?} after {elapsed_us}us (budget {budget_us}us)")]
    PhaseBudgetExceeded {
        phase: PhaseId,
        elapsed_us: u64,
        budget_us: u64,
    },

    #[error("model load failed: {model_id}: {reason}")]
    ModelLoad { model_id: String, reason: String },

    #[error("model inference failed: {model_id}: {reason}")]
    ModelInference { model_id: String, reason: String },

    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// The request's own argument is invalid (e.g. a wrong-dimension query
    /// vector reaching a retrieval backend). Request boundaries answer it as
    /// a client error (HTTP 400), not a pipeline failure (TD-SECSCAN-1 §5).
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

pub type RankResult<T> = Result<T, RankError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_messages_are_actionable() {
        let e = RankError::ProfileNotFound("hot_rerank".into());
        assert_eq!(e.to_string(), "rank profile not found: hot_rerank");

        let e = RankError::PhaseBudgetExceeded {
            phase: PhaseId::SECOND,
            elapsed_us: 75_000,
            budget_us: 50_000,
        };
        let msg = e.to_string();
        assert!(msg.contains("75000"));
        assert!(msg.contains("50000"));
        assert!(msg.contains("SECOND") || msg.contains("PhaseId(1)"));
    }

    #[test]
    fn io_error_converts_via_from() {
        let io_err = std::io::Error::other("boom");
        let e: RankError = io_err.into();
        assert!(matches!(e, RankError::Io(_)));
    }
}
