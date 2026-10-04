//! Index management and orchestration

pub mod adaptive_engine;
pub mod analyzer;
#[cfg(feature = "axis")]
pub mod advisor_observations;
pub mod ann_advisor;
pub mod hmgi_param_advisor;
pub mod hnsw_param_advisor;
pub mod ivf_param_advisor;
pub mod manager;
pub mod migration_engine;
pub mod monitor;
pub mod recall_drift;
pub mod strategy;

// Re-export main types
pub use manager::{
    AxisManager, FilterOperator, HotSwapEfChange, HotSwapOutcome, HybridQuery,
    IndexLocationResolver, MetadataFilter, MigrationStatus, QueryResult, ScoredResult, VectorQuery,
};

pub use adaptive_engine::{
    AccessFrequencyMetrics, AdaptiveIndexEngine, CollectionCharacteristics, MetadataComplexity,
    PerformanceMetrics, QueryDistribution, QueryPatternAnalysis, QueryPatternType, TemporalPattern,
};

pub use analyzer::CollectionAnalyzer;
pub use ann_advisor::{
    AnnAdvisorInput, AnnAdvisorOutput, AnnIndexAdvisor, AnnSelector, SupportedAlgorithm,
};
pub use hmgi_param_advisor::HmgiIndexAdvisor;
pub use hnsw_param_advisor::{
    EF_SEARCH_MAX, EF_SEARCH_MIN, HnswIndexAdvisor, HnswSizingInput, HnswSizingOutput,
    advise_hnsw_params,
};
pub use ivf_param_advisor::{IvfIndexAdvisor, nlist_for_n, nprobe_for_recall, recall_for_nprobe};
pub use migration_engine::{
    IndexMigrationEngine, MigrationComplexity, MigrationDecision, MigrationPhase, MigrationPlan,
};
pub use monitor::{AxisMonitor, MonitoringMetrics};
pub use recall_drift::{DriftKind, RecallDriftInput, RecallDriftReport, detect_recall_drift};
pub use strategy::{IndexStrategy, StrategyRecommendation, StrategySelector};
