/*
 * Copyright 2025 Vijaykumar Singh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! ColumnarReadProvider - Unified interface for columnar data access
//!
//! This trait provides a unified abstraction for:
//! - In-memory Arrow RecordBatch access (query result caching, intermediate results)
//! - On-disk Parquet files (NOVA)
//! - On-disk ProximaBlocks (SST engine columnar mode)
//! - Range-pruned I/O with predicate pushdown
//!
//! # Design Principles
//!
//! - **Interface Segregation**: Focused on read operations only
//! - **Open/Closed**: New providers can be added without modifying existing code
//! - **Dependency Inversion**: Higher layers depend on this abstraction
//!
//! # Performance Considerations
//!
//! - Predicate pushdown reduces I/O by pruning at storage level
//! - Projection pushdown reads only required columns
//! - Statistics pruning skips irrelevant row groups/blocks
//! - Streaming mode enables processing data larger than memory

use std::sync::Arc;

use anyhow::Result;
use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;

use crate::core::search::FilterExpression;

/// Statistics about data access for cost-based optimization
#[derive(Debug, Clone, Default)]
pub struct ColumnarAccessStats {
    /// Total rows available
    pub total_rows: u64,
    /// Rows after predicate pruning
    pub rows_after_pruning: u64,
    /// Bytes read from storage
    pub bytes_read: u64,
    /// Row groups/blocks scanned
    pub blocks_scanned: usize,
    /// Row groups/blocks pruned by statistics
    pub blocks_pruned: usize,
    /// Cache hits (for in-memory provider)
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// I/O time in milliseconds
    pub io_time_ms: u64,
    /// Predicate evaluation time in milliseconds
    pub predicate_time_ms: u64,
}

/// Predicate pushdown configuration
#[derive(Debug, Clone)]
pub struct PredicatePushdownConfig {
    /// Enable row group/block pruning using statistics (min/max)
    pub enable_statistics_pruning: bool,
    /// Enable bloom filter checks for ID lookups
    pub enable_bloom_filters: bool,
    /// Enable projection pushdown (read only required columns)
    pub enable_projection: bool,
    /// Columns to project (None = all columns)
    pub projection: Option<Vec<String>>,
    /// Filter expression to push down
    pub filter: Option<FilterExpression>,
    /// Maximum rows to return (limit pushdown)
    pub limit: Option<usize>,
    /// Rows to skip (offset)
    pub offset: Option<usize>,
}

impl Default for PredicatePushdownConfig {
    fn default() -> Self {
        Self {
            enable_statistics_pruning: true,
            enable_bloom_filters: true,
            enable_projection: true,
            projection: None,
            filter: None,
            limit: None,
            offset: None,
        }
    }
}

/// Range specification for targeted I/O
#[derive(Debug, Clone, Default)]
pub struct ColumnarRange {
    /// Row group/block indices to read
    pub block_indices: Option<Vec<usize>>,
    /// Start row within the range
    pub start_row: Option<u64>,
    /// End row within the range (exclusive)
    pub end_row: Option<u64>,
    /// ID range for sorted data
    pub id_range: Option<(String, String)>,
}

/// Capabilities of a columnar read provider
#[derive(Debug, Clone, Default)]
pub struct ColumnarCapabilities {
    /// Supports predicate pushdown to storage level
    pub supports_predicate_pushdown: bool,
    /// Supports projection pushdown (column selection)
    pub supports_projection: bool,
    /// Supports statistics-based row group pruning
    pub supports_statistics_pruning: bool,
    /// Supports bloom filter ID lookups
    pub supports_bloom_filters: bool,
    /// Supports streaming iteration (low memory)
    pub supports_streaming: bool,
    /// Supports parallel block/row group reading
    pub supports_parallel_read: bool,
    /// Data is in-memory (no I/O cost)
    pub is_in_memory: bool,
    /// Supports zero-copy access
    pub supports_zero_copy: bool,
}

/// Unified interface for columnar data access
///
/// This trait abstracts over:
/// - In-memory Arrow RecordBatches (from cache or intermediate results)
/// - On-disk Parquet files (NOVA)
/// - On-disk ProximaBlocks (SST columnar mode)
///
/// # Performance Considerations
///
/// - Predicate pushdown reduces I/O by pruning at storage level
/// - Projection pushdown reads only required columns
/// - Statistics pruning skips irrelevant row groups/blocks
/// - Streaming mode enables processing data larger than memory
#[async_trait]
pub trait ColumnarReadProvider: Send + Sync {
    /// Provider name for logging and debugging
    fn name(&self) -> &str;

    /// Get capabilities of this provider
    fn capabilities(&self) -> ColumnarCapabilities;

    /// Read data as Arrow RecordBatches with optional predicate pushdown
    ///
    /// This is the primary read method. Returns a vector of RecordBatches
    /// that satisfy the given predicates and projections.
    ///
    /// # Parameters
    /// - `config`: Predicate pushdown and projection configuration
    ///
    /// # Returns
    /// - Vector of Arrow RecordBatches
    async fn read_batches(&self, config: PredicatePushdownConfig) -> Result<Vec<RecordBatch>>;

    /// Read specific range of data (for targeted I/O)
    ///
    /// Enables reading specific row groups/blocks without full scan.
    /// Used for AXIS index-driven lookups where row positions are known.
    ///
    /// # Parameters
    /// - `range`: Range specification (row groups, row ranges, ID ranges)
    /// - `config`: Predicate pushdown configuration for additional filtering
    async fn read_range(
        &self,
        range: ColumnarRange,
        config: PredicatePushdownConfig,
    ) -> Result<Vec<RecordBatch>>;

    /// Create a streaming iterator for memory-efficient processing
    ///
    /// Returns an iterator that yields RecordBatches one at a time,
    /// enabling processing of data larger than available memory.
    ///
    /// # Parameters
    /// - `config`: Predicate pushdown configuration
    /// - `batch_size`: Target rows per batch
    async fn stream_batches(
        &self,
        config: PredicatePushdownConfig,
        batch_size: usize,
    ) -> Result<Box<dyn ColumnarBatchStream>>;

    /// Get row count estimate (for cost estimation)
    ///
    /// Returns estimated row count after applying the given predicates.
    /// Used by query planner for cost-based optimization.
    ///
    /// # Parameters
    /// - `filter`: Optional filter to apply for estimation
    async fn estimate_row_count(&self, filter: Option<&FilterExpression>) -> Result<u64>;

    /// Get access statistics from last operation
    fn get_stats(&self) -> ColumnarAccessStats;

    /// Reset statistics counters
    fn reset_stats(&mut self);

    /// Get schema of the data
    fn schema(&self) -> Arc<Schema>;
}

/// Streaming batch iterator for memory-efficient processing
#[async_trait]
pub trait ColumnarBatchStream: Send {
    /// Get next batch, or None if exhausted
    async fn next_batch(&mut self) -> Result<Option<RecordBatch>>;

    /// Get current statistics
    fn stats(&self) -> ColumnarAccessStats;

    /// Cancel the stream (cleanup resources)
    fn cancel(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PredicatePushdownConfig::default();
        assert!(config.enable_statistics_pruning);
        assert!(config.enable_bloom_filters);
        assert!(config.enable_projection);
        assert!(config.projection.is_none());
        assert!(config.filter.is_none());
        assert!(config.limit.is_none());
    }

    #[test]
    fn test_capabilities_default() {
        let caps = ColumnarCapabilities::default();
        assert!(!caps.supports_predicate_pushdown);
        assert!(!caps.is_in_memory);
    }

    #[test]
    fn test_columnar_range_default() {
        let range = ColumnarRange::default();
        assert!(range.block_indices.is_none());
        assert!(range.start_row.is_none());
        assert!(range.end_row.is_none());
        assert!(range.id_range.is_none());
    }
}
