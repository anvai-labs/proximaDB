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

//! Flush Optimizer
//!
//! Optimizes flush operations for better performance and storage efficiency.
//! Includes multi-batch sorting strategies and compression optimization.

use anyhow::Result;
use std::collections::BinaryHeap;
use tracing::{debug, info};

use proximadb_records::ProximaRecord;

/// Flush optimizer for improving flush performance
pub struct FlushOptimizer;

impl FlushOptimizer {
    /// Create a new flush optimizer
    pub fn new() -> Self {
        Self
    }

    /// Optimize vector sorting for multi-batch flush operations
    ///
    /// For large multi-batch datasets, uses batch-aware sorting with k-way merge
    /// which is more efficient than single O(n log n) sort when k << n.
    pub async fn optimize_multi_batch_sort(
        &self,
        vector_records: Vec<ProximaRecord>,
        batch_ids: &[String],
    ) -> Result<Vec<(String, ProximaRecord)>> {
        let record_count = vector_records.len();
        let batch_count = batch_ids.len();

        debug!(
            "🔍 FlushOptimizer: Processing {} vectors from {} batches",
            record_count, batch_count
        );

        // For small datasets or single batch, use simple Vec + sort
        if record_count < 10000 || batch_count <= 1 {
            debug!("🔍 Using single-sort strategy (small dataset or single batch)");
            return self.simple_sort(vector_records).await;
        }

        // For larger multi-batch datasets, use batch-aware sorting
        debug!(
            "🔍 Using multi-batch sort strategy ({} batches)",
            batch_count
        );
        self.multi_batch_sort(vector_records, batch_count).await
    }

    /// Simple sorting for small datasets
    async fn simple_sort(
        &self,
        vector_records: Vec<ProximaRecord>,
    ) -> Result<Vec<(String, ProximaRecord)>> {
        let mut sorted_records = Vec::with_capacity(vector_records.len());

        // Collect all records into Vec
        for (sequence_number, vector) in vector_records.into_iter().enumerate() {
            let vector_id = if vector.oid.is_empty() {
                "".to_string()
            } else {
                vector.oid.clone()
            };

            // Handle append-only vectors (empty/null IDs) specially
            let key = if vector_id.is_empty() {
                format!("__append_only_seq_{}", sequence_number)
            } else {
                vector_id
            };

            // Keep the optimizer on the canonical record envelope.
            let mut vector_record = vector;
            // Preserve the legacy sequence marker in the canonical version field.
            vector_record.record_version = sequence_number as u64;

            sorted_records.push((key, vector_record));
        }

        // Single efficient sort: O(n log n)
        sorted_records.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(sorted_records)
    }

    /// Multi-batch sorting with k-way merge
    async fn multi_batch_sort(
        &self,
        vector_records: Vec<ProximaRecord>,
        batch_count: usize,
    ) -> Result<Vec<(String, ProximaRecord)>> {
        let record_count = vector_records.len();

        // Group vectors by their order (simulating batch grouping)
        let batch_size = (record_count / batch_count).max(1);
        let mut sorted_batches = Vec::with_capacity(batch_count);

        for (batch_idx, batch_chunk) in vector_records.chunks(batch_size).enumerate() {
            let mut batch_records = Vec::with_capacity(batch_chunk.len());

            for (local_idx, vector) in batch_chunk.iter().enumerate() {
                let sequence_number = batch_idx * batch_size + local_idx;
                let vector_id = if vector.oid.is_empty() {
                    "".to_string()
                } else {
                    vector.oid.clone()
                };

                let key = if vector_id.is_empty() {
                    format!("__append_only_seq_{}", sequence_number)
                } else {
                    vector_id
                };

                // Keep the optimizer on the canonical record envelope.
                let mut vector_record = vector.clone();
                // Preserve the legacy sequence marker in the canonical version field.
                vector_record.record_version = sequence_number as u64;

                batch_records.push((key, vector_record));
            }

            // Sort this batch: O(m log m) where m = batch_size
            batch_records.sort_by(|a, b| a.0.cmp(&b.0));
            sorted_batches.push(batch_records.into_iter());
        }

        // K-way merge of sorted batches: O(n log k) where k = number of batches
        self.k_way_merge(sorted_batches).await
    }

    /// Perform k-way merge of sorted batch iterators
    async fn k_way_merge(
        &self,
        mut sorted_batches: Vec<std::vec::IntoIter<(String, ProximaRecord)>>,
    ) -> Result<Vec<(String, ProximaRecord)>> {
        #[derive(Clone)]
        struct HeapItem {
            key: String,
            record: ProximaRecord,
            batch_idx: usize,
        }

        impl PartialEq for HeapItem {
            fn eq(&self, other: &Self) -> bool {
                self.key == other.key
            }
        }

        impl Eq for HeapItem {}

        impl Ord for HeapItem {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                // Reverse for min-heap behavior
                other.key.cmp(&self.key)
            }
        }

        impl PartialOrd for HeapItem {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut heap = BinaryHeap::new();
        let mut result = Vec::new();

        // Initialize heap with first item from each batch
        for (batch_idx, iter) in sorted_batches.iter_mut().enumerate() {
            if let Some((key, record)) = iter.next() {
                heap.push(HeapItem {
                    key,
                    record,
                    batch_idx,
                });
            }
        }

        // Perform k-way merge
        while let Some(HeapItem {
            key,
            record,
            batch_idx,
        }) = heap.pop()
        {
            result.push((key, record));

            // Add next item from same batch to heap
            if let Some((next_key, next_record)) = sorted_batches[batch_idx].next() {
                heap.push(HeapItem {
                    key: next_key,
                    record: next_record,
                    batch_idx,
                });
            }
        }

        info!(
            "✅ FlushOptimizer: K-way merge completed, {} records processed",
            result.len()
        );
        Ok(result)
    }

    /// Estimate compression improvement from sorting
    pub fn estimate_compression_improvement(&self, record_count: usize) -> f64 {
        // Heuristic: sorting typically improves compression by 10-20%
        // depending on data locality and patterns
        let base_improvement = 0.15; // 15%
        let size_factor = (record_count as f64).log10() / 6.0; // Scale with data size
        (base_improvement + size_factor * 0.05).min(0.25) // Cap at 25%
    }
}

impl Default for FlushOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_records::EmbeddingCell;
    #[tokio::test]
    async fn test_simple_sort() {
        let optimizer = FlushOptimizer::new();

        let vectors = vec![
            create_test_vector("vector_3", vec![3.0, 4.0]),
            create_test_vector("vector_1", vec![1.0, 2.0]),
            create_test_vector("vector_2", vec![2.0, 3.0]),
        ];

        let sorted = optimizer.simple_sort(vectors).await.unwrap();

        // Verify sorting
        assert_eq!(sorted[0].0, "vector_1");
        assert_eq!(sorted[1].0, "vector_2");
        assert_eq!(sorted[2].0, "vector_3");
    }

    #[tokio::test]
    async fn test_multi_batch_optimization() {
        let optimizer = FlushOptimizer::new();

        let vectors = vec![
            create_test_vector("vector_3", vec![3.0, 4.0]),
            create_test_vector("vector_1", vec![1.0, 2.0]),
            create_test_vector("vector_4", vec![4.0, 5.0]),
            create_test_vector("vector_2", vec![2.0, 3.0]),
        ];

        let batch_ids = vec!["batch1".to_string(), "batch2".to_string()];
        let sorted = optimizer
            .optimize_multi_batch_sort(vectors, &batch_ids)
            .await
            .unwrap();

        // Should use simple sort for small dataset
        assert_eq!(sorted.len(), 4);
        assert_eq!(sorted[0].0, "vector_1");
    }

    #[tokio::test]
    async fn test_compression_estimation() {
        let optimizer = FlushOptimizer::new();

        let small_improvement = optimizer.estimate_compression_improvement(100);
        let large_improvement = optimizer.estimate_compression_improvement(100000);

        assert!(small_improvement > 0.0);
        assert!(large_improvement > small_improvement);
        assert!(large_improvement <= 0.25); // Should be capped
    }

    fn create_test_vector(id: &str, vector: Vec<f32>) -> ProximaRecord {
        ProximaRecord {
            oid: id.to_string(),
            created_at_ns: 12_345_000_000,
            updated_at_ns: 12_345_000_000,
            embeddings: vec![EmbeddingCell::new_fp32(
                "test",
                "vector",
                vector.len() as u32,
                vector,
            )],
            ..Default::default()
        }
    }
}
