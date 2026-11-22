//! SST Row-Oriented Filtering Optimization
//!
//! Implements fast in-memory filtering on deserialized ProximaRecord data.
//! Optimized for SST's row-based storage where entire blocks are read into memory.

use anyhow::Result;
use std::collections::HashMap;
use tracing::{debug, info};

use proximadb_engine_core::proximablocks::block_structures::ProximaDataBlock;
use proximadb_filter_expression::FilterExpression;
use proximadb_records::ProximaRecord;

/// Fast in-memory filter evaluator for row-oriented SST data
pub struct SSTRowFilterEvaluator {
    /// Cache for converted metadata to avoid repeated conversions
    metadata_cache: HashMap<String, HashMap<String, serde_json::Value>>,
}

impl SSTRowFilterEvaluator {
    pub fn new() -> Self {
        Self {
            metadata_cache: HashMap::new(),
        }
    }

    /// Fast filter evaluation on already-loaded canonical records.
    ///
    /// This is optimal for SST because:
    /// 1. Data is already in memory (row-oriented blocks read in full)
    /// 2. No I/O savings from predicate pushdown
    /// 3. Fast metadata field access from ProximaRecord
    /// 4. Can filter entire blocks in memory efficiently
    pub fn filter_records_fast(
        &mut self,
        records: &[ProximaRecord],
        filter_expr: &FilterExpression,
    ) -> Result<Vec<usize>> {
        let mut qualifying_indices = Vec::new();

        info!(
            "SST Row Filter: Evaluating {} ProximaRecords in mem",
            records.len()
        );

        for (index, record) in records.iter().enumerate() {
            if self.evaluate_filter_fast(filter_expr, &record.props) {
                qualifying_indices.push(index);
            }
        }

        debug!(
            "SST Row Filter: {} out of {} ProximaRecords qualify",
            qualifying_indices.len(),
            records.len()
        );

        Ok(qualifying_indices)
    }

    /// Filter VectorRecord array (common case for SST unified search)
    pub fn filter_vector_records_fast(
        &mut self,
        records: &[ProximaRecord],
        filter_expr: &FilterExpression,
    ) -> Result<Vec<usize>> {
        self.filter_records_fast(records, filter_expr)
    }

    /// Filter entire ProximaDataBlock efficiently
    pub fn filter_data_block(
        &mut self,
        block: &ProximaDataBlock,
        filter_expr: &FilterExpression,
    ) -> Result<Vec<usize>> {
        info!(
            "SST Row Filter: Filtering ProximaDataBlock {} with {} records",
            block.block_id,
            block.records.len()
        );

        self.filter_records_fast(&block.records, filter_expr)
    }

    /// Parallel block filtering for high-performance workloads
    pub async fn filter_blocks_parallel(
        &mut self,
        blocks: &[ProximaDataBlock],
        filter_expr: &FilterExpression,
    ) -> Result<HashMap<u32, Vec<usize>>> {
        let mut block_results = HashMap::new();

        info!("SST Row Filter: Parallel filtering {} blocks", blocks.len());

        // For now, sequential - can be parallelized with rayon
        for block in blocks {
            let indices = self.filter_data_block(block, filter_expr)?;
            if !indices.is_empty() {
                block_results.insert(block.block_id, indices);
            }
        }

        let total_qualifying = block_results.values().map(|v| v.len()).sum::<usize>();
        info!(
            "SST Row Filter: {} blocks have qualifying records, {} total matches",
            block_results.len(),
            total_qualifying
        );

        Ok(block_results)
    }

    /// Get or convert metadata with caching for performance
    #[allow(dead_code)]
    fn get_or_convert_metadata(
        &mut self,
        record_id: &str,
        metadata: &[proximadb_proto::proximadb_v1::MetadataItem],
    ) -> Result<HashMap<String, serde_json::Value>> {
        if let Some(cached) = self.metadata_cache.get(record_id) {
            return Ok(cached.clone());
        }

        // Convert proto metadata to JSON for filtering
        let metadata_map =
            proximadb_storage_common::proto_metadata_helper::proto_metadata_to_json(metadata);

        // Cache if record has stable ID
        if !record_id.is_empty() && !record_id.starts_with("idx_") {
            self.metadata_cache
                .insert(record_id.to_string(), metadata_map.clone());
        }

        Ok(metadata_map)
    }

    /// Fast filter evaluation using unified filter evaluator
    fn evaluate_filter_fast(
        &self,
        expr: &FilterExpression,
        metadata: &proximadb_records::ProximaTree,
    ) -> bool {
        proximadb_search_types::sql_value_filter::evaluate_filter_proxima(expr, metadata)
    }

    /// Clear metadata cache to prevent memory bloat
    pub fn clear_cache(&mut self) {
        self.metadata_cache.clear();
    }

    /// Get cache statistics for monitoring
    pub fn cache_stats(&self) -> (usize, usize) {
        let entries = self.metadata_cache.len();
        let memory_bytes = self
            .metadata_cache
            .iter()
            .map(|(k, v)| k.len() + v.len() * 50) // Rough estimate
            .sum::<usize>();
        (entries, memory_bytes)
    }

    /// Immutable version for Arc-wrapped read operations (no caching)
    pub fn filter_vector_records_immutable(
        &self,
        records: &[ProximaRecord],
        filter_expr: &FilterExpression,
    ) -> Result<Vec<usize>> {
        let mut qualifying_indices = Vec::new();

        debug!(
            "SST Row Filter (immutable): Evaluating {} VectorRecords",
            records.len()
        );

        for (index, record) in records.iter().enumerate() {
            if proximadb_search_types::sql_value_filter::evaluate_filter_proxima(
                filter_expr,
                &record.props,
            ) {
                qualifying_indices.push(index);
            }
        }

        debug!(
            "SST Row Filter (immutable): {} out of {} VectorRecords qualify",
            qualifying_indices.len(),
            records.len()
        );

        Ok(qualifying_indices)
    }
}

impl Default for SSTRowFilterEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimized batch filtering for multiple filter expressions (AND/OR support)
pub struct SSTBatchFilterEvaluator {
    base_evaluator: SSTRowFilterEvaluator,
}

impl SSTBatchFilterEvaluator {
    pub fn new() -> Self {
        Self {
            base_evaluator: SSTRowFilterEvaluator::new(),
        }
    }

    /// Parallel evaluation of AND/OR expressions for high performance
    ///
    /// This implements the parallel thread strategy you mentioned:
    /// - Thread 1: category = 'electronics' → indices [1, 5, 8, 12]  
    /// - Thread 2: price > 100 → indices [2, 5, 7, 8, 10, 12]
    /// - Thread 3: brand = 'Apple' → indices [1, 3, 9]
    /// - AND_result = Thread1 ∩ Thread2 = [5, 8, 12]
    /// - Final_result = AND_result ∪ Thread3 = [1, 3, 5, 8, 9, 12]
    pub async fn evaluate_parallel_filters(
        &mut self,
        records: &[ProximaRecord],
        filter_expr: &FilterExpression,
    ) -> Result<Vec<usize>> {
        match filter_expr {
            FilterExpression::And(sub_exprs) => {
                info!(
                    "SST Parallel Filter: Evaluating {} AND conditions",
                    sub_exprs.len()
                );

                // Evaluate each condition separately (can be parallelized)
                let mut condition_results = Vec::new();
                for expr in sub_exprs {
                    let indices = self
                        .base_evaluator
                        .filter_vector_records_fast(records, expr)?;
                    condition_results.push(indices);
                }

                // Intersection of all results
                let intersection = self.intersect_indices(condition_results);
                info!(
                    "SST Parallel Filter: AND result has {} qualifying indices",
                    intersection.len()
                );
                Ok(intersection)
            }

            FilterExpression::Or(sub_exprs) => {
                info!(
                    "SST Parallel Filter: Evaluating {} OR conditions",
                    sub_exprs.len()
                );

                // Evaluate each condition separately (can be parallelized)
                let mut condition_results = Vec::new();
                for expr in sub_exprs {
                    let indices = self
                        .base_evaluator
                        .filter_vector_records_fast(records, expr)?;
                    condition_results.push(indices);
                }

                // Union of all results
                let union = self.union_indices(condition_results);
                info!(
                    "SST Parallel Filter: OR result has {} qualifying indices",
                    union.len()
                );
                Ok(union)
            }

            FilterExpression::Not(sub_expr) => {
                info!("SST Parallel Filter: Evaluating NOT condition");
                let positive_indices = self
                    .base_evaluator
                    .filter_vector_records_fast(records, sub_expr)?;
                let all_indices: std::collections::HashSet<usize> = (0..records.len()).collect();
                let positive_set: std::collections::HashSet<usize> =
                    positive_indices.into_iter().collect();
                let negated: Vec<usize> = all_indices.difference(&positive_set).cloned().collect();
                Ok(negated)
            }

            _ => {
                // Single condition - use base evaluator
                self.base_evaluator
                    .filter_vector_records_fast(records, filter_expr)
            }
        }
    }

    /// Set intersection for AND operations
    fn intersect_indices(&self, mut index_sets: Vec<Vec<usize>>) -> Vec<usize> {
        if index_sets.is_empty() {
            return Vec::new();
        }

        if index_sets.len() == 1 {
            return index_sets.into_iter().next().unwrap_or_default();
        }

        // Sort by size (smallest first for efficiency)
        index_sets.sort_by_key(|set| set.len());

        let mut result: std::collections::HashSet<usize> = index_sets[0].iter().cloned().collect();

        for set in index_sets.iter().skip(1) {
            let set_hash: std::collections::HashSet<usize> = set.iter().cloned().collect();
            result = result.intersection(&set_hash).cloned().collect();

            // Early termination if intersection becomes empty
            if result.is_empty() {
                break;
            }
        }

        let mut final_result: Vec<usize> = result.into_iter().collect();
        final_result.sort_unstable();
        final_result
    }

    /// Set union for OR operations
    fn union_indices(&self, index_sets: Vec<Vec<usize>>) -> Vec<usize> {
        let mut result: std::collections::HashSet<usize> = std::collections::HashSet::new();

        for set in index_sets {
            result.extend(set);
        }

        let mut final_result: Vec<usize> = result.into_iter().collect();
        final_result.sort_unstable();
        final_result
    }

    /// Immutable version for Arc-wrapped read operations
    pub async fn evaluate_parallel_filters_immutable(
        &self,
        records: &[ProximaRecord],
        filter_expr: &FilterExpression,
    ) -> Result<Vec<usize>> {
        match filter_expr {
            FilterExpression::And(sub_exprs) => {
                debug!(
                    "SST Parallel Filter (immutable): Evaluating {} AND conditions",
                    sub_exprs.len()
                );

                let mut condition_results = Vec::new();
                for expr in sub_exprs {
                    let indices = self
                        .base_evaluator
                        .filter_vector_records_immutable(records, expr)?;
                    condition_results.push(indices);
                }

                let intersection = self.intersect_indices(condition_results);
                Ok(intersection)
            }

            FilterExpression::Or(sub_exprs) => {
                debug!(
                    "SST Parallel Filter (immutable): Evaluating {} OR conditions",
                    sub_exprs.len()
                );

                let mut condition_results = Vec::new();
                for expr in sub_exprs {
                    let indices = self
                        .base_evaluator
                        .filter_vector_records_immutable(records, expr)?;
                    condition_results.push(indices);
                }

                let union = self.union_indices(condition_results);
                Ok(union)
            }

            _ => {
                // Single condition - use base evaluator
                self.base_evaluator
                    .filter_vector_records_immutable(records, filter_expr)
            }
        }
    }
}

impl Default for SSTBatchFilterEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[cfg_attr(test, path = "row_filter_tests.rs")]
#[cfg(test)]
mod tests;
