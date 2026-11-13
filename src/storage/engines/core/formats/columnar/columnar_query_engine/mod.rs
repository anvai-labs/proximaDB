//! Modular Parquet Query Engine Components
//!
//! This module provides efficient Parquet querying capabilities for columnar storage engines.
//! It's organized into focused submodules for better maintainability and clarity.

pub mod adaptive_filter_executor;
pub mod column_projector;
pub mod columnar_query_reader;
pub mod columnar_reader;
pub mod filter_pushdown_engine;
pub mod pipeline;
pub mod query_metrics;
pub mod result_cache;
pub mod vectorized_executor;
pub(crate) mod version_columns;

// Re-export main types for convenience with semantic names
pub use adaptive_filter_executor::{BranchedFilterExecutor, FilterPath};
pub use column_projector::{ColumnProjection, ProjectionBuilder};
pub use columnar_reader::{ParquetReader, ReaderBuilder};
pub use filter_pushdown_engine::{FilterPushdown, PredicateBuilder};
pub use pipeline::{
    FilterOperator, PipelineBuilder, PipelineOperator, ScanOperator, ScoreOperator, TopKOperator,
};
pub use query_metrics::{QueryStatistics, StatisticsCollector};
pub use result_cache::{CacheStrategy, QueryCache};
pub use vectorized_executor::{DataChunk, vectorized_filter_batch};

// Re-export unified reader types for compatibility
pub use columnar_query_reader::{
    CollectionContext, FilterValue, PagePruningInfo, PageRange, ReaderConfig, ReadingStrategy,
    ReadingStrategySelector, RowGroupAccessPattern, SchemaMapping, SearchType, SeekRange,
    Stage2Strategy, UnifiedParquetReader, VectorPosition,
};

// Common traits used across query implementations
use crate::proto::proximadb_v1::MetadataFilter;
use anyhow::Result;
use arrow::record_batch::RecordBatch;
use proximadb_records::ProximaRecord;

/// Common trait for all Parquet readers
#[allow(async_fn_in_trait)]
pub trait ParquetQueryEngine: Send + Sync {
    /// Query with metadata filters
    async fn query_with_filters(
        &self,
        file_path: &str,
        filters: &[MetadataFilter],
    ) -> Result<Vec<ProximaRecord>>;

    /// Query by IDs
    async fn query_by_ids(&self, file_path: &str, ids: &[String]) -> Result<Vec<ProximaRecord>>;

    /// Query with projection
    async fn query_with_projection(
        &self,
        file_path: &str,
        columns: &[String],
    ) -> Result<RecordBatch>;

    /// Get query statistics
    fn get_statistics(&self) -> QueryStatistics;
}

/// Configuration for query operations
#[derive(Debug, Clone)]
pub struct QueryConfig {
    /// Enable predicate pushdown
    pub enable_pushdown: bool,

    /// Enable column projection
    pub enable_projection: bool,

    /// Enable statistics-based pruning
    pub enable_statistics: bool,

    /// Cache strategy
    pub cache_strategy: CacheStrategy,

    /// Maximum records to return
    pub limit: Option<usize>,

    /// Enable parallel execution
    pub enable_parallel: bool,

    /// Number of parallel workers
    pub parallel_workers: usize,
}
