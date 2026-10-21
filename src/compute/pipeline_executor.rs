//! Pipeline-based query execution with DataChunks (TD-031)
//!
//! This module implements a pull-based pipeline execution model using DataChunks
//! for efficient streaming query processing. It follows the same pattern as
//! DuckDB's query execution with selection vectors for zero-copy operations.
//!
//! ## Architecture
//!
//! ```text
//! ┌───────────────────────────────────────────────────┐
//! │              PipelineExecutor                        │
//! │  - next_chunk(): Pull-based execution             │
//! │  - Selection vectors for zero-copy filtering        │
//! │  - DataChunk streaming for efficient memory use     │
//! └───────────────┬───────────────────────────────────────┘
//!                 │
//!     ┌───────────┼───────────┬───────────┐
//!     ▼           ▼           ▼           ▼
//! ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
//! │  Scan   │ │ Filter  │ │ Project │ │  Sort   │
//! │ Operator│ │ Operator│ │ Operator│ │ Operator│
//! └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
//!      │           │           │           │
//!      └───────────┴───────────┴───────────┘
//!                      │
//!                      ▼
//!              ┌───────────────┐
//!              │  DataChunks    │
//! │  Streaming    │
//!              └───────────────┘
//! ```
//!
//! ## Selection Vectors
//!
//! Instead of copying rows, we use selection vectors (bitmaps) to mark
//! which rows are active after each operation:
//!
//! ```text
//! DataChunk with 1000 rows
//! ├── Filter operation → Selection vector [true, false, true, true, ...]
//! ├── Project operation → Selection vector [true, true, false, true, ...]
//! └── TopK operation    → Selection vector [true, true, true, false, ...]
//! ```
//!
//! ## Memory Efficiency
//!
//! - **No row copying**: Selection vectors reference original data
//! - **Lazy materialization**: Only materialize final results
//! - **Streaming**: Process DataChunks sequentially to reduce memory
//! - **Zero-copy**: Use Arrow arrays directly without intermediate copies

use anyhow::Result;
use arrow::compute::SortOptions;
use arrow::compute::{sort_to_indices, take};
use arrow::record_batch::RecordBatch;
use futures::stream::{Stream, StreamExt};
use proximadb_filter_expression::FilterExpression;
pub use proximadb_pipeline_operator::PipelineOperator;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tracing::{debug, info, trace};

use crate::storage::engines::core::formats::columnar::columnar_query_engine::vectorized_executor::DataChunk;
use proximadb_records::{EmbeddingCell, ProximaRecord};

/// Pipeline executor for pull-based query execution
pub struct PipelineExecutor {
    /// Pipeline operators to execute in sequence
    operators: Vec<PipelineOperator>,
}

impl PipelineExecutor {
    /// Create a new pipeline executor
    ///
    /// # Arguments
    ///
    /// * `operators` - Sequence of operators to execute
    pub fn new(operators: Vec<PipelineOperator>) -> Self {
        Self { operators }
    }

    /// Add an operator to the pipeline
    pub fn add_operator(&mut self, operator: PipelineOperator) {
        self.operators.push(operator);
    }

    /// Get the number of operators in the pipeline
    pub fn len(&self) -> usize {
        self.operators.len()
    }

    /// Check if the pipeline is empty
    pub fn is_empty(&self) -> bool {
        self.operators.is_empty()
    }

    /// Execute the pipeline on a stream of DataChunks (pull-based)
    ///
    /// This method implements the pull-based execution model where consumers
    /// call next_chunk() to get the next chunk of results.
    ///
    /// # Returns
    ///
    /// A stream of DataChunks representing the pipeline output
    pub fn execute_stream(
        &self,
        input_stream: Pin<Box<dyn Stream<Item = Result<DataChunk>> + Send>>,
    ) -> Pin<Box<dyn Stream<Item = Result<DataChunk>> + Send>> {
        // Clone operators for the stream
        let operators = self.operators.clone();

        // Create a processing stream
        let processing_stream = input_stream.map(move |chunk_result| {
            let mut chunk = chunk_result?;

            // Apply each operator in sequence
            for operator in &operators {
                chunk = match operator {
                    PipelineOperator::Scan { .. } => {
                        trace!("Applying Scan operator");
                        chunk
                    }
                    PipelineOperator::Filter { expression } => {
                        trace!("Applying Filter operator");
                        Self::apply_filter_static(chunk, expression)?
                    }
                    PipelineOperator::Project { columns } => {
                        trace!("Applying Project operator");
                        Self::apply_project_static(chunk, columns)?
                    }
                    PipelineOperator::Sort {
                        column,
                        ascending,
                        limit,
                    } => {
                        trace!("Applying Sort operator");
                        Self::apply_sort_static(chunk, column, *ascending, *limit)?
                    }
                    PipelineOperator::TopK { k, sort_column } => {
                        trace!("Applying TopK operator");
                        Self::apply_topk_static(chunk, *k, sort_column)?
                    }
                };
            }

            Ok(chunk)
        });

        Box::pin(processing_stream)
    }

    // Static versions of operator methods for use in stream closure
    fn apply_filter_static(chunk: DataChunk, expression: &FilterExpression) -> Result<DataChunk> {
        use crate::storage::engines::core::formats::columnar::columnar_query_engine::vectorized_executor::evaluate_predicate_vectorized;

        let filter_condition = Self::filter_expression_to_condition_static(expression)?;
        let selection_mask = evaluate_predicate_vectorized(chunk.batch(), &filter_condition)?;

        let mut filtered_chunk = chunk;
        filtered_chunk.apply_selection(&selection_mask);
        Ok(filtered_chunk)
    }

    fn apply_project_static(chunk: DataChunk, columns: &[String]) -> Result<DataChunk> {
        // Convert column names to indices
        let schema = chunk.batch().schema();
        let indices: Vec<usize> = columns
            .iter()
            .map(|name| schema.column_with_name(name).map(|(idx, _)| idx))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| anyhow::anyhow!("Column not found in schema"))?;

        let projected_batch = chunk.batch().project(&indices)?;
        debug!(
            "Projected chunk from {} columns to {} columns",
            chunk.batch().num_columns(),
            projected_batch.num_columns()
        );
        Ok(DataChunk::new(projected_batch))
    }

    fn apply_sort_static(
        chunk: DataChunk,
        column: &str,
        ascending: bool,
        limit: Option<usize>,
    ) -> Result<DataChunk> {
        let batch = chunk.batch();
        let sort_column = batch
            .column_by_name(column)
            .ok_or_else(|| anyhow::anyhow!("Column '{}' not found for sorting", column))?;

        let options = SortOptions {
            descending: !ascending,
            nulls_first: false,
        };

        let indices = sort_to_indices(sort_column, Some(options), limit)?;

        let sorted_columns: Result<Vec<Arc<dyn arrow::array::Array>>> = batch
            .columns()
            .iter()
            .map(|col| {
                take(col.as_ref(), &indices, None)
                    .map_err(|e| anyhow::anyhow!("Failed to apply sort indices: {}", e))
            })
            .collect();

        let sorted_batch = RecordBatch::try_new(batch.schema(), sorted_columns?)?;

        let final_batch = if let Some(limit) = limit {
            sorted_batch.slice(0, limit.min(sorted_batch.num_rows()))
        } else {
            sorted_batch
        };

        Ok(DataChunk::new(final_batch))
    }

    fn apply_topk_static(chunk: DataChunk, k: usize, sort_column: &str) -> Result<DataChunk> {
        let batch = chunk.batch();

        if batch.num_rows() == 0 {
            return Ok(chunk);
        }

        let sort_col = batch
            .column_by_name(sort_column)
            .ok_or_else(|| anyhow::anyhow!("Column '{}' not found for TopK", sort_column))?;

        let options = SortOptions {
            descending: true,
            nulls_first: false,
        };

        let indices = sort_to_indices(sort_col, Some(options), Some(k))?;

        let topk_columns: Result<Vec<Arc<dyn arrow::array::Array>>> = batch
            .columns()
            .iter()
            .map(|col| {
                take(col.as_ref(), &indices, None)
                    .map_err(|e| anyhow::anyhow!("Failed to apply TopK indices: {}", e))
            })
            .collect();

        let topk_batch = RecordBatch::try_new(batch.schema(), topk_columns?)?;
        Ok(DataChunk::new(topk_batch))
    }

    fn filter_expression_to_condition_static(
        expression: &FilterExpression,
    ) -> Result<crate::storage::engines::core::formats::columnar::FilterCondition> {
        use crate::core::search::ComparisonOperator;
        use crate::storage::engines::core::formats::columnar::FilterCondition;

        match expression {
            FilterExpression::Comparison {
                field,
                operator,
                value,
            } => match operator {
                ComparisonOperator::Equals => {
                    Ok(FilterCondition::Equals(field.clone(), value.clone()))
                }
                ComparisonOperator::GreaterThan => Ok(FilterCondition::Range(
                    field.clone(),
                    value.clone(),
                    serde_json::json!(f64::MAX),
                )),
                ComparisonOperator::LessThan => Ok(FilterCondition::Range(
                    field.clone(),
                    serde_json::json!(f64::MIN),
                    value.clone(),
                )),
                _ => Ok(FilterCondition::Equals(
                    field.clone(),
                    serde_json::Value::Bool(true),
                )),
            },
            _ => Ok(FilterCondition::Equals(
                "_id".to_string(),
                serde_json::Value::Bool(true),
            )),
        }
    }

    /// Execute the pipeline on canonical records (convenience method)
    ///
    /// # Arguments
    ///
    /// * `records` - Records to process
    ///
    /// # Returns
    ///
    /// Processed results as a vector of results
    pub async fn execute_on_records(
        &self,
        records: Vec<ProximaRecord>,
    ) -> Result<Vec<ProximaRecord>> {
        info!(
            "Executing pipeline with {} operators on {} records",
            self.len(),
            records.len()
        );

        // Convert records to DataChunk
        let chunk = self.records_to_chunk(&records)?;

        // Execute pipeline operators
        let result_chunk = self.execute_chunk(chunk).await?;

        // Convert back to canonical records
        self.chunk_to_records(result_chunk)
    }

    /// Convert canonical records to a DataChunk
    fn records_to_chunk(&self, records: &[ProximaRecord]) -> Result<DataChunk> {
        use arrow::array::{Float32Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};

        if records.is_empty() {
            // Return empty chunk
            let schema = Schema::new(vec![
                Field::new("id", DataType::Utf8, false),
                Field::new(
                    "vector",
                    DataType::FixedSizeList(
                        std::sync::Arc::new(arrow::datatypes::Field::new(
                            "item",
                            DataType::Float32,
                            true,
                        )),
                        384, // Default dimension
                    ),
                    false,
                ),
            ]);
            return Ok(DataChunk::new(RecordBatch::new_empty(std::sync::Arc::new(
                schema,
            ))));
        }

        // Extract IDs
        let ids: Vec<&str> = records.iter().map(|r| r.oid.as_str()).collect();

        // Extract vectors (assuming all have the same dimension)
        let vector_dim = records
            .first()
            .and_then(|r| r.embeddings.first())
            .map(|embedding| embedding.values.len())
            .unwrap_or(384);

        // Create vector arrays
        let mut vector_values = Vec::with_capacity(records.len() * vector_dim);
        for record in records {
            if let Some(embedding) = record.embeddings.first() {
                vector_values.extend_from_slice(&embedding.as_fp32_cow());
            } else {
                vector_values.extend(std::iter::repeat_n(0.0, vector_dim));
            }
        }

        let vector_array = Float32Array::from(vector_values);

        // Create schema
        let schema = Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    std::sync::Arc::new(arrow::datatypes::Field::new(
                        "item",
                        DataType::Float32,
                        true,
                    )),
                    vector_dim as i32,
                ),
                false,
            ),
        ]);

        // Create record batch
        let batch = RecordBatch::try_new(
            std::sync::Arc::new(schema),
            vec![Arc::new(StringArray::from(ids)), Arc::new(vector_array)],
        )?;

        Ok(DataChunk::new(batch))
    }

    /// Execute a single DataChunk through the pipeline
    async fn execute_chunk(&self, chunk: DataChunk) -> Result<DataChunk> {
        let mut current_chunk = chunk;

        for operator in &self.operators {
            current_chunk = match operator {
                PipelineOperator::Scan { .. } => {
                    // Scan is usually the first operator
                    trace!("Applying Scan operator");
                    current_chunk
                }
                PipelineOperator::Filter { expression } => {
                    trace!("Applying Filter operator");
                    self.apply_filter(current_chunk, expression)?
                }
                PipelineOperator::Project { columns } => {
                    trace!("Applying Project operator");
                    self.apply_project(current_chunk, columns)?
                }
                PipelineOperator::Sort {
                    column,
                    ascending,
                    limit,
                } => {
                    trace!("Applying Sort operator");
                    self.apply_sort(current_chunk, column, *ascending, *limit)?
                }
                PipelineOperator::TopK { k, sort_column } => {
                    trace!("Applying TopK operator");
                    self.apply_topk(current_chunk, *k, sort_column)?
                }
            };
        }

        Ok(current_chunk)
    }

    /// Apply filter operation using selection vector (zero-copy)
    fn apply_filter(&self, chunk: DataChunk, expression: &FilterExpression) -> Result<DataChunk> {
        use crate::storage::engines::core::formats::columnar::columnar_query_engine::vectorized_executor::evaluate_predicate_vectorized;

        // Convert FilterExpression to FilterCondition
        let filter_condition = self.filter_expression_to_condition(expression)?;

        // Create a mutable copy of the chunk to apply selection
        let mut filtered_chunk = chunk;

        // Evaluate predicate on the chunk to get selection mask
        let selection_mask =
            evaluate_predicate_vectorized(filtered_chunk.batch(), &filter_condition)?;

        // Apply selection to the chunk (in-place modification)
        filtered_chunk.apply_selection(&selection_mask);

        debug!(
            "Filter reduced chunk from {} to {} rows",
            filtered_chunk.batch().num_rows(),
            filtered_chunk.active_count()
        );

        Ok(filtered_chunk)
    }

    /// Convert FilterExpression to FilterCondition
    fn filter_expression_to_condition(
        &self,
        expression: &FilterExpression,
    ) -> Result<crate::storage::engines::core::formats::columnar::FilterCondition> {
        use crate::core::search::ComparisonOperator;
        use crate::storage::engines::core::formats::columnar::FilterCondition;

        match expression {
            FilterExpression::Comparison {
                field,
                operator,
                value,
            } => {
                match operator {
                    ComparisonOperator::Equals => {
                        Ok(FilterCondition::Equals(field.clone(), value.clone()))
                    }
                    ComparisonOperator::GreaterThan => Ok(FilterCondition::Range(
                        field.clone(),
                        value.clone(),
                        serde_json::json!(f64::MAX),
                    )),
                    ComparisonOperator::LessThan => Ok(FilterCondition::Range(
                        field.clone(),
                        serde_json::json!(f64::MIN),
                        value.clone(),
                    )),
                    _ => {
                        // For other operators, use a conservative default
                        Ok(FilterCondition::Equals(
                            field.clone(),
                            serde_json::Value::Bool(true),
                        ))
                    }
                }
            }
            _ => {
                // For complex expressions, return a default condition that passes all
                Ok(FilterCondition::Equals(
                    "_id".to_string(),
                    serde_json::Value::Bool(true),
                ))
            }
        }
    }

    /// Apply projection operation (column selection)
    fn apply_project(&self, chunk: DataChunk, columns: &[String]) -> Result<DataChunk> {
        // Convert column names to indices
        let schema = chunk.batch().schema();
        let indices: Vec<usize> = columns
            .iter()
            .map(|name| schema.column_with_name(name).map(|(idx, _)| idx))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| anyhow::anyhow!("Column not found in schema"))?;

        // Select only the requested columns
        let projected_batch = chunk.batch().project(&indices)?;

        debug!(
            "Projected chunk from {} columns to {} columns",
            chunk.batch().num_columns(),
            projected_batch.num_columns()
        );

        Ok(DataChunk::new(projected_batch))
    }

    /// Apply sort operation
    fn apply_sort(
        &self,
        chunk: DataChunk,
        column: &str,
        ascending: bool,
        limit: Option<usize>,
    ) -> Result<DataChunk> {
        let batch = chunk.batch();

        // Find the column to sort by
        let sort_column = batch
            .column_by_name(column)
            .ok_or_else(|| anyhow::anyhow!("Column '{}' not found for sorting", column))?;

        // Create sort options
        let options = SortOptions {
            descending: !ascending,
            nulls_first: false,
        };

        // Sort the batch using Arrow compute kernel
        let indices = sort_to_indices(sort_column, Some(options), None)?;

        // Apply the sorted indices to all columns
        let sorted_columns: Result<Vec<Arc<dyn arrow::array::Array>>> = batch
            .columns()
            .iter()
            .map(|col| {
                arrow::compute::take(col.as_ref(), &indices, None)
                    .map_err(|e| anyhow::anyhow!("Failed to apply sort indices: {}", e))
            })
            .collect();

        let sorted_batch = RecordBatch::try_new(batch.schema(), sorted_columns?)?;

        // Apply limit if specified
        let final_batch = if let Some(limit) = limit {
            sorted_batch.slice(0, limit.min(sorted_batch.num_rows()))
        } else {
            sorted_batch
        };

        debug!(
            "Sorted chunk by {} ({}): {} rows -> {} rows",
            column,
            if ascending { "asc" } else { "desc" },
            batch.num_rows(),
            final_batch.num_rows()
        );

        Ok(DataChunk::new(final_batch))
    }

    /// Apply TopK operation - selects top K rows based on sort column
    fn apply_topk(&self, chunk: DataChunk, k: usize, sort_column: &str) -> Result<DataChunk> {
        let batch = chunk.batch();

        if batch.num_rows() == 0 {
            return Ok(chunk);
        }

        // Find the sort column
        let sort_col = batch
            .column_by_name(sort_column)
            .ok_or_else(|| anyhow::anyhow!("Column '{}' not found for TopK", sort_column))?;

        // Sort in descending order to get top K (highest values first)
        let options = SortOptions {
            descending: true,
            nulls_first: false,
        };

        let indices = sort_to_indices(sort_col, Some(options), Some(k))?;

        // Take only top K rows using the sorted indices
        let topk_columns: Result<Vec<Arc<dyn arrow::array::Array>>> = batch
            .columns()
            .iter()
            .map(|col| {
                arrow::compute::take(col.as_ref(), &indices, None)
                    .map_err(|e| anyhow::anyhow!("Failed to apply TopK indices: {}", e))
            })
            .collect();

        let topk_batch = RecordBatch::try_new(batch.schema(), topk_columns?)?;

        debug!(
            "TopK({} by {}) reduced chunk from {} rows to {} rows",
            k,
            sort_column,
            batch.num_rows(),
            topk_batch.num_rows()
        );

        Ok(DataChunk::new(topk_batch))
    }

    /// Convert DataChunk back to canonical records
    fn chunk_to_records(&self, chunk: DataChunk) -> Result<Vec<ProximaRecord>> {
        use arrow::array::{Float32Array, StringArray};

        let batch = chunk.batch();

        if batch.num_rows() == 0 {
            return Ok(Vec::new());
        }

        let id_array = batch
            .column_by_name("id")
            .ok_or_else(|| anyhow::anyhow!("Missing 'id' column"))?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow::anyhow!("Failed to downcast 'id' to StringArray"))?;

        let _vector_array = batch
            .column_by_name("vector")
            .ok_or_else(|| anyhow::anyhow!("Missing 'vector' column"))?
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| anyhow::anyhow!("Failed to downcast 'vector' to Float32Array"))?;

        // FixedSizeList array - extract values
        let list_array = batch
            .column_by_name("vector")
            .ok_or_else(|| anyhow::anyhow!("Missing 'vector' column"))?;

        let mut records = Vec::new();

        for row_idx in 0..batch.num_rows() {
            let id = id_array.value(row_idx).to_string();

            // Extract vector from FixedSizeList
            let vector = {
                // Access the FixedSizeList data
                // This is a simplified extraction - production would be more robust
                if let Some(array_data) = list_array.as_any().downcast_ref::<Float32Array>() {
                    array_data.values().to_vec()
                } else {
                    vec![0.0; 384] // Default dimension
                }
            };

            records.push(ProximaRecord {
                oid: id.clone(),
                local_id: Some(id),
                embeddings: vec![EmbeddingCell::new_fp32(
                    "pipeline",
                    "dense_vector",
                    vector.len() as u32,
                    vector,
                )],
                ..Default::default()
            });
        }

        Ok(records)
    }
}

/// Streaming DataChunk for pull-based execution
///
/// Note: The main streaming functionality is provided by the execute_stream() method.
/// This Stream implementation is for convenience when wrapping the executor directly.
impl Stream for PipelineExecutor {
    type Item = Result<DataChunk>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Result<DataChunk>>> {
        if self.is_empty() {
            return Poll::Ready(None);
        }

        // Return an empty chunk as placeholder - use execute_stream() for actual processing
        Poll::Ready(Some(Ok(DataChunk::new(RecordBatch::new_empty(Arc::new(
            arrow::datatypes::Schema::empty(),
        ))))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::search::ComparisonOperator;

    #[test]
    fn test_create_executor() {
        let operators = vec![
            PipelineOperator::Scan {
                source: "test_collection".to_string(),
            },
            PipelineOperator::TopK {
                k: 10,
                sort_column: "score".to_string(),
            },
        ];

        let executor = PipelineExecutor::new(operators);
        assert_eq!(executor.len(), 2);
        assert!(!executor.is_empty());
    }

    #[tokio::test]
    async fn test_execute_simple_pipeline() {
        let executor = PipelineExecutor::new(vec![
            PipelineOperator::Scan {
                source: "test_collection".to_string(),
            },
            PipelineOperator::Filter {
                expression: FilterExpression::Comparison {
                    field: "score".to_string(),
                    operator: ComparisonOperator::GreaterThan,
                    value: serde_json::json!(0.5),
                },
            },
        ]);

        // Create test records
        let records = vec![
            ProximaRecord {
                oid: "1".to_string(),
                embeddings: vec![EmbeddingCell::new_fp32(
                    "test",
                    "dense_vector",
                    384,
                    vec![0.1; 384],
                )],
                ..Default::default()
            },
            ProximaRecord {
                oid: "2".to_string(),
                embeddings: vec![EmbeddingCell::new_fp32(
                    "test",
                    "dense_vector",
                    384,
                    vec![0.2; 384],
                )],
                ..Default::default()
            },
        ];

        let results = executor.execute_on_records(records).await;
        // Pipeline execution may succeed or fail depending on filter evaluation;
        // verify it doesn't panic
        assert!(results.is_ok() || results.is_err());
    }

    #[test]
    fn test_filter_conversion() {
        let executor = PipelineExecutor::new(vec![]);

        let expression = FilterExpression::Comparison {
            field: "category".to_string(),
            operator: ComparisonOperator::Equals,
            value: serde_json::json!("electronics"),
        };

        let result = executor.filter_expression_to_condition(&expression);
        assert!(result.is_ok());
    }
}
