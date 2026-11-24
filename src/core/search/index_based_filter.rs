//! Index-Based Filter Evaluation Framework
//!
//! This module implements a sophisticated filtering approach where:
//! 1. Filter evaluation returns qualifying ROW INDICES
//! 2. Top-level orchestrator uses indices for engine-specific optimizations
//! 3. SST: Uses indices to filter already-read block data
//! 4. NOVA: Uses indices to selectively read parquet rows/columns
//! 5. Empty indices = skip entire blocks/row groups (early termination)

use anyhow::Result;
use std::collections::{HashMap, HashSet};
use tracing::{debug, info};

use crate::core::search::FilterExpression;
use proximadb_records::ProximaRecord;

/// Qualifying row indices with metadata about the filtering process
#[derive(Debug, Clone)]
pub struct FilterIndices {
    /// Qualifying row indices within the data source
    pub indices: Vec<usize>,
    /// Total rows evaluated
    pub total_rows: usize,
    /// Engine-specific metadata for optimization decisions
    pub metadata: FilterIndexMetadata,
}

/// Backwards-compat alias for [`FilterIndexMetadata`].
pub type IndexMetadata = FilterIndexMetadata;

/// Metadata about the filter evaluation process for optimization decisions
#[derive(Debug, Clone)]
pub struct FilterIndexMetadata {
    /// Source identifier (block ID, file path, etc.)
    pub source_id: String,
    /// Selectivity ratio (0.0 = no matches, 1.0 = all match)
    pub selectivity: f32,
    /// Filter complexity that was evaluated
    pub filter_complexity: FilterComplexity,
    /// Columns that were actually used in filtering
    pub columns_evaluated: HashSet<String>,
    /// Engine-specific optimization hints
    pub optimization_hints: HashMap<String, serde_json::Value>,
}

/// Classification of filter expression complexity
#[derive(Debug, Clone, PartialEq)]
pub enum FilterComplexity {
    /// Single equality condition
    Simple,
    /// AND/OR with multiple conditions
    Moderate,
    /// Nested expressions with mixed operators
    Complex,
}

impl FilterIndices {
    /// Check if entire block/row group should be skipped
    pub fn should_skip_block(&self) -> bool {
        self.indices.is_empty()
    }

    /// Check if full block should be read (high selectivity)
    pub fn should_read_full_block(&self) -> bool {
        self.metadata.selectivity > 0.9 // If >90% of rows qualify, read everything
    }

    /// Get optimization recommendation for engine
    pub fn get_read_strategy(&self) -> ReadStrategy {
        if self.should_skip_block() {
            ReadStrategy::SkipBlock
        } else if self.should_read_full_block() {
            ReadStrategy::ReadFullBlock
        } else {
            ReadStrategy::SelectiveRead {
                indices: self.indices.clone(),
                estimated_benefit: 1.0 - self.metadata.selectivity,
            }
        }
    }
}

/// Recommended I/O strategy based on filter selectivity
#[derive(Debug, Clone)]
pub enum ReadStrategy {
    /// Skip this block/row group entirely
    SkipBlock,
    /// Read full block (high selectivity makes selective reading inefficient)
    ReadFullBlock,
    /// Selectively read only qualifying rows
    SelectiveRead {
        /// Row indices to read
        indices: Vec<usize>,
        /// Estimated I/O savings fraction vs full block read (0.0 to 1.0)
        estimated_benefit: f32,
    },
}

/// Metadata source abstraction for different engines
pub trait MetadataSource: Send + Sync {
    /// Get row count for this data source
    fn get_row_count(&self) -> usize;

    /// Get column metadata for filtering optimization
    fn get_column_metadata(&self, column_name: &str) -> Option<FilterColumnMetadata>;

    /// Extract metadata value for a specific row and column
    fn get_metadata_value(&self, row_idx: usize, column_name: &str) -> Option<serde_json::Value>;

    /// Get source identifier (file path, block ID, etc.)
    fn get_source_id(&self) -> String;

    /// Check if this source supports selective row reading
    fn supports_selective_reading(&self) -> bool;
}

/// Backwards-compat alias for [`FilterColumnMetadata`].
pub type ColumnMetadata = FilterColumnMetadata;

/// Statistical metadata about a column for filter optimization
#[derive(Debug, Clone)]
pub struct FilterColumnMetadata {
    /// Whether an inverted index exists on this column
    pub has_index: bool,
    /// Approximate number of distinct values
    pub cardinality: Option<u64>,
    /// Minimum value in the column (for range pruning)
    pub min_value: Option<serde_json::Value>,
    /// Maximum value in the column (for range pruning)
    pub max_value: Option<serde_json::Value>,
    /// Number of null values in the column
    pub null_count: Option<u64>,
}

/// Column data type classification for filter evaluation
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnData {
    /// UTF-8 string column
    String,
    /// 64-bit integer column
    Integer,
    /// 64-bit float column
    Float,
    /// Boolean column
    Boolean,
    /// Array/list column
    Array,
}

/// Core trait for index-based filter evaluation
#[async_trait::async_trait]
pub trait FilterIndexEvaluator: Send + Sync {
    /// Evaluate filter expression and return qualifying row indices
    /// This is the key method - returns indices instead of data
    async fn evaluate_filter_indices(
        &self,
        filter_expr: &FilterExpression,
        metadata_source: &(dyn MetadataSource + Send + Sync),
    ) -> Result<FilterIndices>;

    /// Optional: Batch evaluation across multiple sources
    async fn evaluate_batch_indices(
        &self,
        filter_expr: &FilterExpression,
        sources: &[&(dyn MetadataSource + Send + Sync)],
    ) -> Result<Vec<FilterIndices>> {
        let mut results = Vec::new();
        for source in sources {
            let indices = self.evaluate_filter_indices(filter_expr, *source).await?;
            results.push(indices);
        }
        Ok(results)
    }
}

/// Engine-specific data reader that uses indices for optimization
#[async_trait::async_trait]
pub trait IndexBasedDataReader: Send + Sync {
    /// Read data using the provided indices and read strategy
    async fn read_data_by_indices(
        &self,
        source_id: &str,
        read_strategy: &ReadStrategy,
        metadata_source: &(dyn MetadataSource + Send + Sync),
    ) -> Result<Vec<ProximaRecord>>;

    /// Estimate the cost/benefit of selective reading vs full reading
    fn estimate_selective_read_benefit(&self, indices: &[usize], total_rows: usize) -> f32 {
        // Default implementation: benefit = 1 - selectivity
        1.0 - (indices.len() as f32 / total_rows as f32)
    }
}

/// Top-level orchestrator that coordinates index evaluation and data reading
pub struct IndexBasedFilterOrchestrator<E, R>
where
    E: FilterIndexEvaluator,
    R: IndexBasedDataReader,
{
    index_evaluator: E,
    data_reader: R,
}

impl<E, R> IndexBasedFilterOrchestrator<E, R>
where
    E: FilterIndexEvaluator,
    R: IndexBasedDataReader,
{
    /// Create a new orchestrator with the given evaluator and reader
    pub fn new(index_evaluator: E, data_reader: R) -> Self {
        Self {
            index_evaluator,
            data_reader,
        }
    }

    /// Main entry point for index-based filtered search
    pub async fn execute_filtered_search(
        &self,
        filter_expr: &FilterExpression,
        sources: &[&(dyn MetadataSource + Send + Sync)],
    ) -> Result<Vec<ProximaRecord>> {
        info!(
            "Index-based filtering: {} sources to evaluate",
            sources.len()
        );

        // Step 1: Evaluate filter indices across all sources
        let filter_indices_batch = self
            .index_evaluator
            .evaluate_batch_indices(filter_expr, sources)
            .await?;

        // Step 2: Analyze results and plan data reading
        let mut read_plans = Vec::new();
        let mut total_indices = 0;
        let mut skipped_sources = 0;

        for (source, indices) in sources.iter().zip(filter_indices_batch.iter()) {
            let read_strategy = indices.get_read_strategy();

            match &read_strategy {
                ReadStrategy::SkipBlock => {
                    skipped_sources += 1;
                    debug!(
                        "Skipping source {} - no qualifying rows",
                        indices.metadata.source_id
                    );
                    continue;
                }
                ReadStrategy::SelectiveRead {
                    indices,
                    estimated_benefit,
                } => {
                    total_indices += indices.len();
                    debug!(
                        "Selective read: {} indices, {:.1}% benefit",
                        indices.len(),
                        estimated_benefit * 100.0
                    );
                }
                ReadStrategy::ReadFullBlock => {
                    debug!("Full read - high selectivity");
                }
            }

            read_plans.push((source, read_strategy));
        }

        info!(
            "Index evaluation complete: {} indices across {} sources, {} sources skipped",
            total_indices,
            read_plans.len(),
            skipped_sources
        );

        // Step 3: Execute optimized data reading based on indices
        let mut all_results: Vec<ProximaRecord> = Vec::new();

        for (source, read_strategy) in read_plans {
            let source_results = self
                .data_reader
                .read_data_by_indices(&source.get_source_id(), &read_strategy, *source)
                .await?;

            all_results.extend(source_results);
        }

        info!(
            "Index-based filtering complete: {} total results",
            all_results.len()
        );
        Ok(all_results)
    }

    /// Get filtering statistics for performance analysis
    pub async fn get_filter_statistics(
        &self,
        filter_expr: &FilterExpression,
        sources: &[&(dyn MetadataSource + Send + Sync)],
    ) -> Result<FilterStatistics> {
        let filter_indices_batch = self
            .index_evaluator
            .evaluate_batch_indices(filter_expr, sources)
            .await?;

        let mut stats = FilterStatistics::default();

        for indices in filter_indices_batch {
            stats.total_sources += 1;
            stats.total_rows += indices.total_rows;
            stats.qualifying_rows += indices.indices.len();

            if indices.should_skip_block() {
                stats.skipped_sources += 1;
            } else if indices.should_read_full_block() {
                stats.full_read_sources += 1;
            } else {
                stats.selective_read_sources += 1;
            }
        }

        stats.calculate_ratios();
        Ok(stats)
    }
}

/// Aggregate statistics from filter evaluation across multiple data sources
#[derive(Debug, Default)]
pub struct FilterStatistics {
    /// Total number of data sources evaluated
    pub total_sources: usize,
    /// Total rows across all sources
    pub total_rows: usize,
    /// Rows that passed the filter
    pub qualifying_rows: usize,
    /// Sources where all rows were filtered out
    pub skipped_sources: usize,
    /// Sources where all rows qualified
    pub full_read_sources: usize,
    /// Sources where partial rows qualified
    pub selective_read_sources: usize,
    /// Fraction of rows that qualified (0.0 to 1.0)
    pub overall_selectivity: f32,
    /// Fraction of sources that were entirely skipped
    pub skip_ratio: f32,
    /// Fraction of sources that used selective reading
    pub selective_read_ratio: f32,
}

impl FilterStatistics {
    fn calculate_ratios(&mut self) {
        if self.total_rows > 0 {
            self.overall_selectivity = self.qualifying_rows as f32 / self.total_rows as f32;
        }

        if self.total_sources > 0 {
            self.skip_ratio = self.skipped_sources as f32 / self.total_sources as f32;
            self.selective_read_ratio =
                self.selective_read_sources as f32 / self.total_sources as f32;
        }
    }
}

/// Base implementation of filter index evaluator using centralized logic
pub struct BaseFilterIndexEvaluator;

#[async_trait::async_trait]
impl FilterIndexEvaluator for BaseFilterIndexEvaluator {
    async fn evaluate_filter_indices(
        &self,
        filter_expr: &FilterExpression,
        metadata_source: &(dyn MetadataSource + Send + Sync),
    ) -> Result<FilterIndices> {
        let total_rows = metadata_source.get_row_count();
        let source_id = metadata_source.get_source_id();

        debug!(
            "Evaluating filter indices for {}: {} rows",
            source_id, total_rows
        );

        let mut qualifying_indices = Vec::new();
        let mut columns_evaluated = HashSet::new();

        // Extract columns needed for filtering
        let filter_columns =
            crate::core::search::filter_extraction::extract_filter_columns(filter_expr);
        columns_evaluated.extend(filter_columns);

        // Evaluate each row
        for row_idx in 0..total_rows {
            // Build metadata map for this row
            let mut row_metadata = HashMap::new();

            for column_name in &columns_evaluated {
                if let Some(value) = metadata_source.get_metadata_value(row_idx, column_name) {
                    row_metadata.insert(column_name.clone(), value);
                }
            }

            // Use centralized filter evaluation for consistency
            if crate::core::search::json_comparison::evaluate_filter(filter_expr, &row_metadata) {
                qualifying_indices.push(row_idx);
            }
        }

        let selectivity = if total_rows > 0 {
            qualifying_indices.len() as f32 / total_rows as f32
        } else {
            0.0
        };

        debug!(
            "Filter evaluation for {}: {}/{} rows qualify ({:.1}% selectivity)",
            source_id,
            qualifying_indices.len(),
            total_rows,
            selectivity * 100.0
        );

        Ok(FilterIndices {
            indices: qualifying_indices,
            total_rows,
            metadata: FilterIndexMetadata {
                source_id,
                selectivity,
                filter_complexity: FilterComplexity::Simple, // Deferred: Analyze complexity
                columns_evaluated,
                optimization_hints: HashMap::new(),
            },
        })
    }
}
