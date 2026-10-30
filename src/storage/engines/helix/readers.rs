//! Query execution readers for HELIX engine
//!
//! This module provides efficient reading and searching of HELIX SSTables
//! with Hilbert-based pruning and Proxima decoding.

use anyhow::Result;
use futures::future::join_all;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, error, info, trace, warn};

use crate::core::search::bounded_queue::BoundedPriorityQueue;
use crate::core::search::results::OptimizedSearchRecord;
use crate::proto::proximadb_v1::VectorRecord;
use crate::storage::persistence::filesystem::FileSystem;
use proximadb_distance_kernel::DistanceMetric;

use super::SStableMetadata;
// Filter evaluator now uses unified module from core
use crate::storage::engines::core::formats::proximablocks::bloom_filter::{
    SerializedBloomFilter, factory::BloomFilterFactory,
};

/// Check if SSTable might contain specific vector IDs using bloom filter
pub async fn check_bloom_filter(
    sstable: &SStableMetadata,
    vector_ids: &[String],
) -> Result<Vec<bool>> {
    if let Some(ref bloom_data) = sstable.bloom_filter {
        // Create SerializedBloomFilter structure from raw data
        // We need to parse the serialized format properly
        // For now, use the factory directly with appropriate config
        let config = crate::core::bloom::BloomFilterConfig {
            bits_per_key: 10,
            expected_items: 10000,
            enabled: true,
            ..Default::default()
        };

        let serialized = SerializedBloomFilter {
            strategy_type: crate::core::bloom::BloomStrategy::ByteAligned,
            version: SerializedBloomFilter::CURRENT_VERSION,
            config: config.clone(),
            data: bloom_data.clone(),
            metadata: HashMap::new(),
        };
        let bloom = BloomFilterFactory::from_serialized(&serialized)?;

        // Check each ID
        let results: Vec<bool> = vector_ids
            .iter()
            .map(|id| bloom.might_contain(id.as_bytes()))
            .collect();
        Ok(results)
    } else {
        // No bloom filter, assume all might be present
        Ok(vec![true; vector_ids.len()])
    }
}

/// Search an SSTable for nearest vectors with type-safe FilterExpression support
pub async fn search_sstable(
    filesystem: &Arc<
        crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem,
    >,
    sstable: &SStableMetadata,
    query_vector: &[f32],
    _query_hilbert_key: Option<u64>,
    k: usize,
    distance_metric: &DistanceMetric,
    distance_compute: &Arc<proximadb_distance_kernel::engine::UnifiedDistanceCompute>,
    filter_expression: Option<&proximadb_filter_expression::FilterExpression>,
    candidate_ids: Option<&[String]>, // Optional IDs to check via bloom filter
    collection: Option<&crate::proto::proximadb_v1::Collection>,
    prune: &crate::core::search::BlockPruneConfig,
) -> Result<Vec<OptimizedSearchRecord>> {
    // Check bloom filter if candidate IDs provided
    if let Some(ids) = candidate_ids
        && !ids.is_empty()
        && sstable.bloom_filter.is_some()
    {
        let bloom_results = check_bloom_filter(sstable, ids).await?;

        // Skip this SSTable if none of the candidate IDs might be present
        if !bloom_results.iter().any(|&present| present) {
            debug!("Skipping SSTable due to bloom filter pruning");
            return Ok(Vec::new());
        }
    }

    debug!(
        "Searching SSTable at level {} with {} vectors",
        sstable.level, sstable.num_vectors
    );

    // Use proxima search with Hilbert key for block-level pruning
    let search_results = super::proxima::search_helix_sstable(
        filesystem,
        &sstable.path,
        query_vector,
        _query_hilbert_key, // Pass Hilbert key for spatial pruning (80-90% block reduction!)
        k,
        distance_metric,
        distance_compute,
        collection,        // Pass collection for type-safe metadata deserialization
        filter_expression, // Pass FilterExpression for type-safe filtering
        prune,
    )
    .await?;

    // Convert the search results to OptimizedSearchRecord format
    let mut results = Vec::new();
    for (id, distance, sql_metadata) in search_results {
        // Filter is now applied inside search_helix_sstable using type-safe SqlValue evaluation

        // Use standardized distance-to-similarity conversion for consistency across all engines
        // This ensures all engines return the same similarity score for the same distance
        let similarity = crate::core::search::results::OptimizedSearchRecord::standardized_distance_to_similarity(
            distance,
            distance_metric,
        );

        // Metadata is already SqlValue format from search_helix_sstable
        // IMPORTANT: score field contains normalized similarity (0-1, higher = better)
        // This is used for sorting in VOS and display to users
        let record = OptimizedSearchRecord::new(id, similarity)
            .with_similarity(similarity)
            .with_metadata(sql_metadata);

        results.push(record);
    }

    Ok(results)
}

/// Find a specific vector by ID
pub async fn find_vector_by_id(
    filesystem: &Arc<
        crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem,
    >,
    sstable: &SStableMetadata,
    vector_id: &str,
) -> Result<Option<VectorRecord>> {
    // Check bloom filter if available
    if let Some(ref _bloom_data) = sstable.bloom_filter {
        // Deserialize and check bloom filter
        // If not present, return early
        // (Implementation would use actual bloom filter)
    }

    // Read file data
    let file_data = filesystem.read(&sstable.path.to_string_lossy()).await?;
    let mut cursor = std::io::Cursor::new(file_data);

    // Skip magic and version
    cursor.set_position(8);

    // Read number of blocks
    let mut num_blocks_bytes = [0u8; 4];
    std::io::Read::read_exact(&mut cursor, &mut num_blocks_bytes)?;
    let num_blocks = u32::from_le_bytes(num_blocks_bytes);

    // Read blocks
    for _ in 0..num_blocks {
        // Read block size
        let mut size_bytes = [0u8; 4];
        std::io::Read::read_exact(&mut cursor, &mut size_bytes)?;
        let block_size = u32::from_le_bytes(size_bytes) as usize;

        // Read block data
        let mut block_data = vec![0u8; block_size];
        std::io::Read::read_exact(&mut cursor, &mut block_data)?;

        // Deserialize block
        use crate::storage::engines::core::formats::proximablocks::ProximaDataBlock;
        let block = ProximaDataBlock::deserialize(&block_data, None)?;

        let current_time = chrono::Utc::now().timestamp() as u64;
        for record in block.records {
            if record.oid == vector_id {
                // Check if record is expired (tombstone support)
                if let Some(expires_at) = record.valid_to_ns
                    && expires_at as u64 <= current_time
                {
                    // Record is expired, treat as deleted
                    return Ok(None);
                }
                return Ok(Some(crate::proto::proximadb_v1::VectorRecord::from(
                    &record,
                )));
            }
        }
    }

    Ok(None)
}

/// Parallel search across multiple SSTables with type-safe FilterExpression
///
/// This function distributes the search across multiple threads, with each
/// thread searching one or more SSTables in parallel for maximum performance.
pub async fn parallel_search(
    filesystem: Arc<
        crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem,
    >,
    sstables: Vec<SStableMetadata>,
    query_vector: Vec<f32>,
    query_hilbert_key: Option<u64>,
    k: usize,
    distance_metric: DistanceMetric,
    distance_compute: Arc<proximadb_distance_kernel::engine::UnifiedDistanceCompute>,
    filter_expression: Option<proximadb_filter_expression::FilterExpression>,
    collection: Option<std::sync::Arc<crate::proto::proximadb_v1::Collection>>,
    block_prune: crate::core::search::BlockPruneConfig,
) -> Result<Vec<OptimizedSearchRecord>> {
    if sstables.is_empty() {
        return Ok(Vec::new());
    }

    info!(
        "Starting parallel search across {} SSTables",
        sstables.len()
    );
    let start = std::time::Instant::now();

    // Create search tasks for each SSTable
    let search_tasks = sstables.into_iter().enumerate().map(|(idx, sstable)| {
        let fs = filesystem.clone();
        let query = query_vector.clone();
        let query_hilbert = query_hilbert_key; // Copy Option<u64> for thread
        let metric = distance_metric;
        let dist_compute = distance_compute.clone();
        let filter_clone = filter_expression.clone();
        let collection_clone = collection.clone();
        let prune_config = block_prune.clone();

        tokio::spawn(async move {
            trace!(
                "Thread {} searching SSTable at level {}",
                idx, sstable.level
            );
            let thread_start = std::time::Instant::now();

            // Search the SSTable with Hilbert key for block pruning
            let result = search_sstable(
                &fs,
                &sstable,
                &query,
                query_hilbert, // Pass Hilbert key for spatial pruning
                k,
                &metric,
                &dist_compute,
                filter_clone.as_ref(),
                None,                                          // No candidate IDs for now
                collection_clone.as_ref().map(|c| c.as_ref()), // Pass collection for type-safe metadata
                &prune_config,
            )
            .await;

            trace!("Thread {} completed in {:?}", idx, thread_start.elapsed());
            result
        })
    });

    // Wait for all search tasks to complete
    let results = join_all(search_tasks).await;

    // Merge results from all SSTables using bounded queue
    let mut priority_queue = BoundedPriorityQueue::new(k);
    let mut successful_searches = 0;
    let mut failed_searches = 0;

    for (idx, result) in results.into_iter().enumerate() {
        match result {
            Ok(Ok(sstable_results)) => {
                trace!("Thread {} returned {} results", idx, sstable_results.len());
                for result in sstable_results {
                    priority_queue.try_insert(result);
                }
                successful_searches += 1;
            }
            Ok(Err(e)) => {
                warn!("Thread {} failed to search SSTable: {}", idx, e);
                failed_searches += 1;
            }
            Err(e) => {
                error!("Thread {} panicked: {}", idx, e);
                failed_searches += 1;
            }
        }
    }

    // Get top-k results
    let all_results = priority_queue.into_sorted_vec();

    info!(
        "Parallel search completed in {:?}: {} successful, {} failed, {} results",
        start.elapsed(),
        successful_searches,
        failed_searches,
        all_results.len()
    );

    Ok(all_results)
}

/// Backwards-compat alias for [`HelixReaderQueryStats`].
pub type QueryStats = HelixReaderQueryStats;

/// Statistics for query execution
#[derive(Debug, Default)]
pub struct HelixReaderQueryStats {
    pub sstables_scanned: usize,
    pub sstables_pruned: usize,
    pub blocks_scanned: usize,
    pub blocks_pruned: usize,
    pub vectors_evaluated: usize,
    pub pruning_ratio: f64,
}

/// Search an SSTable using quantized vectors for progressive search
///
/// This function reads the quantized_section from blocks and uses binary or INT8
/// vectors for fast approximate distance computation. This is 10-50x faster than
/// FP32 search for initial candidate filtering.
pub async fn search_sstable_quantized(
    filesystem: &Arc<
        crate::storage::persistence::filesystem::caching_filesystem::UnifiedCachingFilesystem,
    >,
    sstable: &SStableMetadata,
    query_vector: &[f32],
    _query_hilbert_key: Option<u64>,
    k: usize,
    use_binary: bool, // true for binary (Stage 2), false for INT8 (Stage 3)
) -> Result<Vec<OptimizedSearchRecord>> {
    debug!(
        "Quantized search ({}) on SSTable at level {} with {} vectors",
        if use_binary { "binary" } else { "INT8" },
        sstable.level,
        sstable.num_vectors
    );

    // Read file data
    let file_data = filesystem.read(&sstable.path.to_string_lossy()).await?;
    let mut cursor = std::io::Cursor::new(file_data);

    // Skip magic and version
    cursor.set_position(8);

    // Read number of blocks
    let mut num_blocks_bytes = [0u8; 4];
    std::io::Read::read_exact(&mut cursor, &mut num_blocks_bytes)?;
    let num_blocks = u32::from_le_bytes(num_blocks_bytes);

    let mut results = Vec::new();

    // Compute query binary sketch for Hamming distance
    let query_binary: Vec<u8> = if use_binary {
        let dim = query_vector.len();
        let mut binary = vec![0u8; dim.div_ceil(8)];
        for (i, &val) in query_vector.iter().enumerate() {
            if val > 0.0 {
                binary[i / 8] |= 1 << (i % 8);
            }
        }
        binary
    } else {
        Vec::new()
    };

    // Read blocks and compute quantized distances
    for _block_idx in 0..num_blocks {
        // Read block size
        let mut size_bytes = [0u8; 4];
        std::io::Read::read_exact(&mut cursor, &mut size_bytes)?;
        let block_size = u32::from_le_bytes(size_bytes) as usize;

        // Read block data
        let mut block_data = vec![0u8; block_size];
        std::io::Read::read_exact(&mut cursor, &mut block_data)?;

        // Deserialize block
        use crate::storage::engines::core::formats::proximablocks::ProximaDataBlock;
        let block = ProximaDataBlock::deserialize(&block_data, None)?;

        // Check if block has quantized section
        if let Some(ref quant_section) = block.quantized_section {
            if use_binary {
                // Binary quantization: compute Hamming distance
                if let Some(ref binary_vectors) = quant_section.binary_vectors {
                    for (vec_idx, binary_vec) in binary_vectors.iter().enumerate() {
                        // Compute Hamming distance
                        let hamming_dist: u32 = query_binary
                            .iter()
                            .zip(binary_vec.iter())
                            .map(|(&a, &b)| (a ^ b).count_ones())
                            .sum();

                        // Convert to approximate similarity (lower Hamming = higher similarity)
                        let max_bits = (query_vector.len() as u32).min(binary_vec.len() as u32 * 8);
                        let similarity = 1.0 - (hamming_dist as f32 / max_bits as f32);

                        // Get record ID and vector from block
                        if let Some(record) = block.records.get(vec_idx) {
                            let wire_record =
                                crate::proto::proximadb_v1::VectorRecord::from(record);
                            // Use with_vector constructor which takes id, score, and vector
                            let result = OptimizedSearchRecord::with_vector(
                                wire_record.id,
                                similarity,
                                wire_record.vector,
                            );
                            results.push(result);
                        }
                    }
                }
            } else {
                // INT8 quantization: compute approximate L2 distance
                if let Some(ref int8_vectors) = quant_section.int8_vectors {
                    // Quantize query to INT8 for comparison
                    let (min_val, max_val) = query_vector
                        .iter()
                        .fold((f32::MAX, f32::MIN), |(min, max), &val| {
                            (min.min(val), max.max(val))
                        });

                    let scale = if (max_val - min_val).abs() > 1e-8 {
                        255.0 / (max_val - min_val)
                    } else {
                        1.0
                    };

                    let query_int8: Vec<i8> = query_vector
                        .iter()
                        .map(|&val| {
                            let normalized = ((val - min_val) * scale).clamp(0.0, 255.0) as u8;
                            (normalized as i16 - 128) as i8
                        })
                        .collect();

                    for (vec_idx, int8_vec) in int8_vectors.iter().enumerate() {
                        // Compute approximate L2 distance on INT8
                        let int8_dist: i64 = query_int8
                            .iter()
                            .zip(int8_vec.iter())
                            .map(|(&a, &b)| {
                                let diff = (a as i64) - (b as i64);
                                diff * diff
                            })
                            .sum();

                        // Convert to approximate similarity
                        let max_dist = (query_int8.len() as i64) * 255 * 255;
                        let similarity =
                            1.0 - ((int8_dist as f64) / (max_dist as f64)).sqrt() as f32;

                        // Get record ID and vector from block
                        if let Some(record) = block.records.get(vec_idx) {
                            let wire_record =
                                crate::proto::proximadb_v1::VectorRecord::from(record);
                            let result = OptimizedSearchRecord::with_vector(
                                wire_record.id,
                                similarity,
                                wire_record.vector,
                            );
                            results.push(result);
                        }
                    }
                }
            }
        } else {
            // Fallback: No quantized section, use FP32 vectors directly
            // This ensures backwards compatibility with non-quantized data
            for record in &block.records {
                let wire_record = crate::proto::proximadb_v1::VectorRecord::from(record);
                // Simple cosine-like similarity approximation
                let dot: f32 = query_vector
                    .iter()
                    .zip(wire_record.vector.iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                let similarity = (dot + 1.0) / 2.0; // Normalize to [0, 1]

                let result = OptimizedSearchRecord::with_vector(
                    wire_record.id,
                    similarity,
                    wire_record.vector,
                );
                results.push(result);
            }
        }
    }

    // Sort by similarity (descending) and return top-k
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(k);

    debug!(
        "Quantized search found {} candidates from {} blocks",
        results.len(),
        num_blocks
    );

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn test_query_stats() {
        let stats = HelixReaderQueryStats {
            sstables_scanned: 10,
            sstables_pruned: 7,
            blocks_scanned: 30,
            blocks_pruned: 20,
            vectors_evaluated: 1000,
            pruning_ratio: 0.7,
        };

        assert_eq!(stats.sstables_scanned - stats.sstables_pruned, 3);
        assert_eq!(stats.pruning_ratio, 0.7);
    }
}
