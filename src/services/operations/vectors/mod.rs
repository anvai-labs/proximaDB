//! Vector Operations Service - Centralized Search Orchestration
//!
//! This module orchestrates all vector search operations across the system.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │           VectorOperationsService (Orchestrator)                 │
//! │  Unified entry point for all vector operations                   │
//! └─────────────────────────────────────────────────────────────────┘
//!                               │
//!         ┌─────────────────────┼─────────────────────┐
//!         ▼                     ▼                     ▼
//! ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
//! │ Search Module │    │ Hybrid Module │    │ Validation    │
//! │ Progressive   │    │ AXIS Builder  │    │ Metadata      │
//! │ Pipeline      │    │               │    │ Enrichment    │
//! └───────────────┘    └──────────────┘    └───────────────┘
//! ```
//!
//! ## Module Organization
//!
//! - **config.rs**: Configuration types (`UnifiedSearchConfig`, `SearchPlanHints`)
//! - **hybrid/**: Hybrid query construction for vector+metadata filtering
//! - **validation/**: Metadata validation and pseudo-query generation
//! - **search/**: Search operations and progressive pipeline utilities
//! - **legacy.rs**: Main service implementation (being decomposed)
//!
//! ## Migration Notes
//!
//! This module is being decomposed from the original 4,269-line vectors.rs file.
//! The main `VectorOperationsService` is currently in legacy.rs and will be
//! further decomposed into focused submodules.

// Configuration types
pub mod config;

// Hybrid query construction
pub mod hybrid;

// Search operations and progressive pipeline
pub mod search;

// Metadata validation and pseudo-query generation
pub mod validation;

// Diagnostics / metrics collaborator (Phase 2.1 decomposition)
mod diagnostics;

// Flush + compaction coordinator (Phase 2.1 decomposition)
mod flush;

// Collection-resolution + engine-resolution collaborator (Phase 2.1 decomposition)
mod resolver;

// WAL-first, per-collection-engine point-read collaborator
mod read;

// Pure input validators for inserts/queries (Phase 2.1 decomposition)
mod input_validation;

// WAL write-primitive coordinator (Phase 2.1 decomposition)
mod write;

// Legacy main service implementation (being decomposed)
mod legacy;

// Public exports
pub use config::{SearchPlanHints, UnifiedSearchConfig};
#[cfg(feature = "axis")]
pub use hybrid::build_axis_hybrid_query;
pub use input_validation::{
    CollectionNotFound, InvalidQueryVector, QueryVectorProblem, client_argument_message,
    collection_not_found, is_client_argument_error, label_client_collection,
};
pub(crate) use legacy::rich_filters_to_filter_expression;
pub use legacy::{
    RichFilterCondition, RichFilterOperator, RichRecordBatchRequest, RichRecordDeleteBatchRequest,
    RichRecordGetRequest, RichRecordGetResponse, RichSearchRequest, RichSearchResponse,
    RichSearchResult, VectorOperationsService,
};
pub use search::{
    executor::{VectorSearchExecutorResult, proto_results_to_vector_records},
    pipeline::{ProgressiveSearchPipeline, StageResult, default_progressive_stages},
};
pub use validation::{
    DefaultPseudoQueryGenerator, PseudoQueryGenerator, apply_pseudo_query_metadata,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_structure() {
        // Verify module exports are accessible
        let _config = UnifiedSearchConfig::default();
        let _hints = SearchPlanHints::default();
        let _generator = DefaultPseudoQueryGenerator;
        let _stages = default_progressive_stages();
    }
}
