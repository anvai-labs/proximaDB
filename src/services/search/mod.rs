//! Search services
//!
//! Various search implementations including streaming, EDR, and batch search

#[cfg(feature = "axis")]
pub mod edr_service;
pub mod streaming;

pub use streaming::{
    SearchMetadata as Metadata, SearchResultBatch as ResultBatch,
    SearchResultStream as ResultStream, StreamingSearchConfig as StreamConfig,
    StreamingSearchService as StreamingSearch, StreamingSearchStats as StreamStats,
};

#[cfg(feature = "axis")]
pub use edr_service::{
    EdrSearchExecution, EdrSearchExecutionRequest, EdrSearchResult, execute_edr_search,
    validate_edr_search_request,
};
