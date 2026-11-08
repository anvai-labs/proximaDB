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

//! ArrowInMemoryProvider - In-memory Arrow RecordBatch provider
//!
//! Provides zero-cost access to cached RecordBatches for:
//! - Query result caching
//! - Intermediate results in multi-stage queries
//! - Materialized CTEs (Common Table Expressions)
//! - Arrow IPC memory-mapped files

use std::sync::{Arc, RwLock};

use anyhow::Result;
use arrow::array::{Array, BooleanArray};
use arrow::compute::filter_record_batch;
use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use tracing::debug;

use crate::core::search::{ComparisonOperator, FilterExpression};

use super::super::provider::{
    ColumnarAccessStats, ColumnarBatchStream, ColumnarCapabilities, ColumnarRange,
    ColumnarReadProvider, PredicatePushdownConfig,
};

/// In-memory provider for cached Arrow RecordBatches
///
/// Use cases:
/// - Query result caching
/// - Intermediate results in multi-stage queries
/// - Materialized CTEs (Common Table Expressions)
/// - Arrow IPC memory-mapped files
pub struct ArrowInMemoryProvider {
    /// Cached record batches
    batches: Vec<RecordBatch>,
    /// Schema reference
    schema: Arc<Schema>,
    /// Collection ID for provenance
    collection_id: String,
    /// Access statistics (mutable through interior mutability)
    stats: RwLock<ColumnarAccessStats>,
}

impl ArrowInMemoryProvider {
    /// Create from existing RecordBatches
    pub fn new(batches: Vec<RecordBatch>, collection_id: String) -> Result<Self> {
        let schema = batches
            .first()
            .map(|b| b.schema())
            .ok_or_else(|| anyhow::anyhow!("Empty batch list"))?;

        let total_rows: u64 = batches.iter().map(|b| b.num_rows() as u64).sum();

        Ok(Self {
            batches,
            schema,
            collection_id,
            stats: RwLock::new(ColumnarAccessStats {
                total_rows,
                ..Default::default()
            }),
        })
    }

    /// Create from a single RecordBatch
    pub fn from_batch(batch: RecordBatch, collection_id: String) -> Result<Self> {
        Self::new(vec![batch], collection_id)
    }

    /// Get collection ID
    pub fn collection_id(&self) -> &str {
        &self.collection_id
    }

    /// Apply filter expression to batch using Arrow compute
    fn apply_filter(batch: &RecordBatch, filter: &FilterExpression) -> Result<RecordBatch> {
        // Convert FilterExpression to Arrow boolean mask
        let mask = Self::filter_to_boolean_array(batch, filter)?;

        // Use arrow::compute::filter to apply
        filter_record_batch(batch, &mask)
            .map_err(|e| anyhow::anyhow!("Filter application failed: {}", e))
    }

    /// Convert FilterExpression to Arrow BooleanArray
    fn filter_to_boolean_array(
        batch: &RecordBatch,
        filter: &FilterExpression,
    ) -> Result<BooleanArray> {
        let num_rows = batch.num_rows();

        match filter {
            FilterExpression::Comparison {
                field,
                operator,
                value,
            } => {
                // Find column by name
                let col_idx = batch
                    .schema()
                    .index_of(field)
                    .map_err(|_| anyhow::anyhow!("Column '{}' not found", field))?;
                let column = batch.column(col_idx);

                // Apply comparison based on operator
                Self::apply_comparison(column, operator, value, num_rows)
            }
            FilterExpression::And(exprs) => {
                // All must be true
                let mut result: Option<BooleanArray> = None;
                for expr in exprs {
                    let mask = Self::filter_to_boolean_array(batch, expr)?;
                    result = Some(match result {
                        Some(existing) => arrow::compute::and(&existing, &mask)
                            .map_err(|e| anyhow::anyhow!("AND operation failed: {}", e))?,
                        None => mask,
                    });
                }
                result.ok_or_else(|| anyhow::anyhow!("Empty AND expression"))
            }
            FilterExpression::Or(exprs) => {
                // Any can be true
                let mut result: Option<BooleanArray> = None;
                for expr in exprs {
                    let mask = Self::filter_to_boolean_array(batch, expr)?;
                    result = Some(match result {
                        Some(existing) => arrow::compute::or(&existing, &mask)
                            .map_err(|e| anyhow::anyhow!("OR operation failed: {}", e))?,
                        None => mask,
                    });
                }
                result.ok_or_else(|| anyhow::anyhow!("Empty OR expression"))
            }
            FilterExpression::Not(expr) => {
                let mask = Self::filter_to_boolean_array(batch, expr)?;
                arrow::compute::not(&mask)
                    .map_err(|e| anyhow::anyhow!("NOT operation failed: {}", e))
            }
        }
    }

    /// Apply comparison operator to column (using serde_json::Value)
    fn apply_comparison(
        column: &Arc<dyn Array>,
        op: &ComparisonOperator,
        value: &serde_json::Value,
        num_rows: usize,
    ) -> Result<BooleanArray> {
        use arrow::array::{Float64Array, Int64Array, StringArray};

        // Presence tests and Regex have no vectorized kernel here. The
        // catch-all below returns an ALL-TRUE mask, which for these
        // operators would be a silent no-op filter — fail closed instead
        // (callers surface the error or fall back to a row-wise path).
        if matches!(
            op,
            ComparisonOperator::Exists | ComparisonOperator::NotExists | ComparisonOperator::Regex
        ) {
            return Err(anyhow::anyhow!(
                "operator {op:?} has no vectorized kernel in the Arrow memory provider"
            ));
        }

        // Handle string comparisons
        if let Some(string_array) = column.as_any().downcast_ref::<StringArray>()
            && let Some(val) = value.as_str()
        {
            return match op {
                ComparisonOperator::Equals => Ok(BooleanArray::from_iter(
                    (0..num_rows).map(|i| Some(string_array.value(i) == val)),
                )),
                ComparisonOperator::NotEquals => Ok(BooleanArray::from_iter(
                    (0..num_rows).map(|i| Some(string_array.value(i) != val)),
                )),
                ComparisonOperator::LessThan => Ok(BooleanArray::from_iter(
                    (0..num_rows).map(|i| Some(string_array.value(i) < val)),
                )),
                ComparisonOperator::LessThanOrEqual => Ok(BooleanArray::from_iter(
                    (0..num_rows).map(|i| Some(string_array.value(i) <= val)),
                )),
                ComparisonOperator::GreaterThan => Ok(BooleanArray::from_iter(
                    (0..num_rows).map(|i| Some(string_array.value(i) > val)),
                )),
                ComparisonOperator::GreaterThanOrEqual => Ok(BooleanArray::from_iter(
                    (0..num_rows).map(|i| Some(string_array.value(i) >= val)),
                )),
                _ => Ok(BooleanArray::from(vec![true; num_rows])),
            };
        }

        // Handle integer comparisons
        if let Some(int_array) = column.as_any().downcast_ref::<Int64Array>()
            && let Some(val) = value.as_i64()
        {
            return match op {
                ComparisonOperator::Equals => Ok(BooleanArray::from_iter(
                    (0..num_rows).map(|i| Some(int_array.value(i) == val)),
                )),
                ComparisonOperator::NotEquals => Ok(BooleanArray::from_iter(
                    (0..num_rows).map(|i| Some(int_array.value(i) != val)),
                )),
                ComparisonOperator::LessThan => Ok(BooleanArray::from_iter(
                    (0..num_rows).map(|i| Some(int_array.value(i) < val)),
                )),
                ComparisonOperator::LessThanOrEqual => Ok(BooleanArray::from_iter(
                    (0..num_rows).map(|i| Some(int_array.value(i) <= val)),
                )),
                ComparisonOperator::GreaterThan => Ok(BooleanArray::from_iter(
                    (0..num_rows).map(|i| Some(int_array.value(i) > val)),
                )),
                ComparisonOperator::GreaterThanOrEqual => Ok(BooleanArray::from_iter(
                    (0..num_rows).map(|i| Some(int_array.value(i) >= val)),
                )),
                _ => Ok(BooleanArray::from(vec![true; num_rows])),
            };
        }

        // Handle float comparisons
        if let Some(float_array) = column.as_any().downcast_ref::<Float64Array>()
            && let Some(val) = value.as_f64()
        {
            return match op {
                ComparisonOperator::Equals => {
                    Ok(BooleanArray::from_iter((0..num_rows).map(|i| {
                        Some((float_array.value(i) - val).abs() < f64::EPSILON)
                    })))
                }
                ComparisonOperator::NotEquals => {
                    Ok(BooleanArray::from_iter((0..num_rows).map(|i| {
                        Some((float_array.value(i) - val).abs() >= f64::EPSILON)
                    })))
                }
                ComparisonOperator::LessThan => Ok(BooleanArray::from_iter(
                    (0..num_rows).map(|i| Some(float_array.value(i) < val)),
                )),
                ComparisonOperator::LessThanOrEqual => Ok(BooleanArray::from_iter(
                    (0..num_rows).map(|i| Some(float_array.value(i) <= val)),
                )),
                ComparisonOperator::GreaterThan => Ok(BooleanArray::from_iter(
                    (0..num_rows).map(|i| Some(float_array.value(i) > val)),
                )),
                ComparisonOperator::GreaterThanOrEqual => Ok(BooleanArray::from_iter(
                    (0..num_rows).map(|i| Some(float_array.value(i) >= val)),
                )),
                _ => Ok(BooleanArray::from(vec![true; num_rows])),
            };
        }

        // Default: return all true (no filtering)
        Ok(BooleanArray::from(vec![true; num_rows]))
    }

    /// Apply projection to batch
    fn apply_projection(batch: &RecordBatch, columns: &[String]) -> Result<RecordBatch> {
        let indices: Vec<usize> = columns
            .iter()
            .filter_map(|name| batch.schema().index_of(name).ok())
            .collect();

        batch
            .project(&indices)
            .map_err(|e| anyhow::anyhow!("Projection failed: {}", e))
    }

    /// Apply limit and offset to batches
    fn apply_limit_offset(
        &self,
        batches: Vec<RecordBatch>,
        offset: Option<usize>,
        limit: Option<usize>,
    ) -> Result<Vec<RecordBatch>> {
        let offset = offset.unwrap_or(0);
        let limit = limit.unwrap_or(usize::MAX);

        let mut result = Vec::new();
        let mut rows_skipped = 0;
        let mut rows_collected = 0;

        for batch in batches {
            let batch_rows = batch.num_rows();

            // Skip rows for offset
            if rows_skipped + batch_rows <= offset {
                rows_skipped += batch_rows;
                continue;
            }

            // Calculate start and end within this batch
            let start = offset.saturating_sub(rows_skipped);

            let remaining = limit.saturating_sub(rows_collected);
            let end = (start + remaining).min(batch_rows);

            if start < end {
                let sliced = batch.slice(start, end - start);
                rows_collected += sliced.num_rows();
                result.push(sliced);
            }

            rows_skipped += batch_rows;

            if rows_collected >= limit {
                break;
            }
        }

        Ok(result)
    }
}

#[async_trait]
impl ColumnarReadProvider for ArrowInMemoryProvider {
    fn name(&self) -> &str {
        "arrow_memory"
    }

    fn capabilities(&self) -> ColumnarCapabilities {
        ColumnarCapabilities {
            supports_predicate_pushdown: true,
            supports_projection: true,
            supports_statistics_pruning: false, // No row groups in memory
            supports_bloom_filters: false,
            supports_streaming: true,
            supports_parallel_read: true,
            is_in_memory: true,
            supports_zero_copy: true,
        }
    }

    async fn read_batches(&self, config: PredicatePushdownConfig) -> Result<Vec<RecordBatch>> {
        let start_time = std::time::Instant::now();
        let mut result = Vec::with_capacity(self.batches.len());
        let mut rows_matched = 0u64;

        for batch in &self.batches {
            let mut processed = batch.clone();

            // Apply filter if present
            if let Some(ref filter) = config.filter {
                processed = Self::apply_filter(&processed, filter)?;
            }

            // Apply projection if present
            if config.enable_projection
                && let Some(ref columns) = config.projection
            {
                processed = Self::apply_projection(&processed, columns)?;
            }

            if processed.num_rows() > 0 {
                rows_matched += processed.num_rows() as u64;
                result.push(processed);
            }
        }

        // Apply limit/offset
        if config.offset.is_some() || config.limit.is_some() {
            result = self.apply_limit_offset(result, config.offset, config.limit)?;
        }

        // Update statistics
        {
            let mut stats = self
                .stats
                .write()
                .map_err(|e| anyhow::anyhow!("Failed to acquire write lock for stats: {}", e))?;
            stats.rows_after_pruning = rows_matched;
            stats.cache_hits += 1;
            stats.predicate_time_ms = start_time.elapsed().as_millis() as u64;
        }

        debug!(
            "ArrowInMemoryProvider: read {} batches with {} rows in {}ms",
            result.len(),
            rows_matched,
            start_time.elapsed().as_millis()
        );

        Ok(result)
    }

    async fn read_range(
        &self,
        range: ColumnarRange,
        config: PredicatePushdownConfig,
    ) -> Result<Vec<RecordBatch>> {
        // For in-memory, ranges are just batch indices
        let batches_to_read: Vec<_> = if let Some(indices) = range.block_indices {
            indices
                .into_iter()
                .filter_map(|i| self.batches.get(i).cloned())
                .collect()
        } else {
            self.batches.clone()
        };

        // Apply row range if specified
        let batches = if range.start_row.is_some() || range.end_row.is_some() {
            let start = range.start_row.unwrap_or(0) as usize;
            let end = range.end_row.unwrap_or(u64::MAX) as usize;

            let mut result = Vec::new();
            let mut current_row = 0;

            for batch in batches_to_read {
                let batch_rows = batch.num_rows();
                let batch_end = current_row + batch_rows;

                if batch_end > start && current_row < end {
                    let slice_start = start.saturating_sub(current_row);
                    let slice_end = (end - current_row).min(batch_rows);
                    if slice_start < slice_end {
                        result.push(batch.slice(slice_start, slice_end - slice_start));
                    }
                }

                current_row = batch_end;
                if current_row >= end {
                    break;
                }
            }
            result
        } else {
            batches_to_read
        };

        // Now apply config filters
        let temp_provider = Self::new(batches, self.collection_id.clone())?;
        temp_provider.read_batches(config).await
    }

    async fn stream_batches(
        &self,
        config: PredicatePushdownConfig,
        _batch_size: usize,
    ) -> Result<Box<dyn ColumnarBatchStream>> {
        // Read all batches and create streaming iterator
        let batches = self.read_batches(config).await?;
        Ok(Box::new(InMemoryBatchStream {
            batches,
            current_index: 0,
            stats: ColumnarAccessStats::default(),
        }))
    }

    async fn estimate_row_count(&self, _filter: Option<&FilterExpression>) -> Result<u64> {
        // For in-memory, we know exact count
        // Deferred: Could apply filter estimation for more accuracy
        let stats = self
            .stats
            .read()
            .map_err(|e| anyhow::anyhow!("Failed to acquire read lock for stats: {}", e))?;
        Ok(stats.total_rows)
    }

    fn get_stats(&self) -> ColumnarAccessStats {
        // In production code, RwLock poisoning is extremely rare and indicates a serious bug
        // Using unwrap here is acceptable as it's a stats accessor in a non-critical path
        self.stats
            .read()
            .map_or_else(|_| ColumnarAccessStats::default(), |stats| stats.clone())
    }

    fn reset_stats(&mut self) {
        let total_rows = self.stats.read().map(|stats| stats.total_rows).unwrap_or(0);
        if let Ok(mut stats) = self.stats.write() {
            *stats = ColumnarAccessStats {
                total_rows,
                ..Default::default()
            };
        }
    }

    fn schema(&self) -> Arc<Schema> {
        Arc::clone(&self.schema)
    }
}

/// Simple in-memory batch stream
struct InMemoryBatchStream {
    batches: Vec<RecordBatch>,
    current_index: usize,
    stats: ColumnarAccessStats,
}

#[async_trait]
impl ColumnarBatchStream for InMemoryBatchStream {
    async fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        if self.current_index >= self.batches.len() {
            return Ok(None);
        }
        let batch = self.batches[self.current_index].clone();
        self.current_index += 1;
        self.stats.blocks_scanned += 1;
        Ok(Some(batch))
    }

    fn stats(&self) -> ColumnarAccessStats {
        self.stats.clone()
    }

    fn cancel(&mut self) {
        self.current_index = self.batches.len();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field};

    fn create_test_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("value", DataType::Int64, false),
        ]));

        let id_array = StringArray::from(vec!["a", "b", "c", "d"]);
        let value_array = Int64Array::from(vec![1, 2, 3, 4]);

        RecordBatch::try_new(schema, vec![Arc::new(id_array), Arc::new(value_array)])
            .expect("Failed to create test batch")
    }

    #[tokio::test]
    async fn test_arrow_memory_provider_basic() {
        let batch = create_test_batch();
        let provider = ArrowInMemoryProvider::new(vec![batch], "test_collection".to_string())
            .expect("Failed to create ArrowInMemoryProvider");

        assert_eq!(provider.name(), "arrow_memory");
        assert!(provider.capabilities().is_in_memory);
        assert!(provider.capabilities().supports_projection);
    }

    #[tokio::test]
    async fn test_read_all_batches() {
        let batch = create_test_batch();
        let provider = ArrowInMemoryProvider::new(vec![batch], "test_collection".to_string())
            .expect("Failed to create ArrowInMemoryProvider");

        let config = PredicatePushdownConfig::default();
        let results = provider
            .read_batches(config)
            .await
            .expect("Failed to read batches");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].num_rows(), 4);
    }

    #[tokio::test]
    async fn test_projection() {
        let batch = create_test_batch();
        let provider = ArrowInMemoryProvider::new(vec![batch], "test_collection".to_string())
            .expect("Failed to create ArrowInMemoryProvider");

        let config = PredicatePushdownConfig {
            projection: Some(vec!["id".to_string()]),
            ..Default::default()
        };
        let results = provider
            .read_batches(config)
            .await
            .expect("Failed to read batches with projection");

        assert_eq!(results[0].num_columns(), 1);
        assert_eq!(results[0].schema().field(0).name(), "id");
    }

    #[tokio::test]
    async fn test_limit_offset() {
        let batch = create_test_batch();
        let provider = ArrowInMemoryProvider::new(vec![batch], "test_collection".to_string())
            .expect("Failed to create ArrowInMemoryProvider");

        let config = PredicatePushdownConfig {
            offset: Some(1),
            limit: Some(2),
            ..Default::default()
        };
        let results = provider
            .read_batches(config)
            .await
            .expect("Failed to read batches with limit/offset");

        let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 2);
    }
}
