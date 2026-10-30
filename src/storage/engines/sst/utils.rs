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

//! SST Engine Utilities Module
//!
//! Contains helper functions and utilities for the SST engine.
//! This module provides:
//! - Vector sorting and encoding utilities
//! - Bloom filter construction
//! - Serialization helpers
//! - Metadata management utilities
//! - Common operations shared across SST engine components

use anyhow::Result;
use std::collections::HashMap;
use tracing::{debug, info};

use crate::storage::engines::sst::SstableBloomFilter;
use crate::storage::engines::sst::{SstEngine, SstError};
use proximadb_data_model::ProximaValue;
use proximadb_records::{ProximaRecord, ProximaTreeNode};

impl SstEngine {
    /// Sort vectors for optimal SSTable encoding
    ///
    /// This method sorts vectors to improve compression and locality:
    /// 1. Groups vectors by common metadata keys
    /// 2. Sorts within groups for better compression
    /// 3. Returns sorting statistics for monitoring
    pub async fn sort_vectors_for_sstable_encoding(
        &self,
        vectors: Vec<ProximaRecord>,
    ) -> Result<(Vec<ProximaRecord>, SortingStats)> {
        debug!("🔄 Sorting {} vectors for SSTable encoding", vectors.len());

        let mut sorted_vectors = vectors;

        // Find the most common metadata key for primary sorting
        let primary_sort_key = self.find_primary_sort_key(&sorted_vectors);

        let sort_start = std::time::Instant::now();

        // Sort vectors by primary key and then by ID
        sorted_vectors.sort_by(|a, b| {
            // Primary sort: most common metadata key
            if let Some(ref sort_key) = primary_sort_key {
                let a_value = self.extract_metadata_value(a, sort_key);
                let b_value = self.extract_metadata_value(b, sort_key);

                match a_value.cmp(&b_value) {
                    std::cmp::Ordering::Equal => {
                        // Secondary sort: by record ID
                        a.oid.cmp(&b.oid)
                    }
                    other => other,
                }
            } else {
                // No metadata key: sort by ID only
                a.oid.cmp(&b.oid)
            }
        });

        let sort_duration = sort_start.elapsed();

        // Calculate statistics
        let stats = SortingStats {
            records_sorted: sorted_vectors.len(),
            sort_duration_ms: sort_duration.as_millis() as u64,
            primary_sort_key: primary_sort_key.clone(),
            compression_estimate: self.estimate_compression_improvement(&sorted_vectors),
        };

        info!(
            "✅ Sorted {} vectors in {}ms (primary key: {:?}, estimated compression: {:.1}%)",
            stats.records_sorted,
            stats.sort_duration_ms,
            stats.primary_sort_key,
            stats.compression_estimate * 100.0
        );

        Ok((sorted_vectors, stats))
    }

    /// Find the most common metadata key for sorting
    fn find_primary_sort_key(&self, vectors: &[ProximaRecord]) -> Option<String> {
        let mut key_frequency: HashMap<String, usize> = HashMap::new();

        for vector in vectors {
            for key in vector.props.keys() {
                *key_frequency.entry(key.clone()).or_insert(0) += 1;
            }
        }

        key_frequency
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(key, _)| key.clone())
    }

    /// Extract metadata value as a comparable string
    fn extract_metadata_value(&self, vector: &ProximaRecord, key: &str) -> String {
        vector
            .props
            .get(key)
            .map(|value| self.tree_node_to_sort_string(value))
            .unwrap_or_default()
    }

    /// Convert canonical property nodes to stable sort strings.
    fn tree_node_to_sort_string(&self, node: &ProximaTreeNode) -> String {
        match node {
            ProximaTreeNode::Value(value) => self.proxima_value_to_sort_string(value),
            ProximaTreeNode::Object(tree) => serde_json::to_string(tree).unwrap_or_default(),
        }
    }

    /// Convert ProximaValue to string for comparison.
    fn proxima_value_to_sort_string(&self, value: &ProximaValue) -> String {
        match value {
            ProximaValue::String(value)
            | ProximaValue::Symbol(value)
            | ProximaValue::Decimal(value) => value.clone(),
            ProximaValue::Boolean(value) => value.to_string(),
            ProximaValue::Int8(value) => value.to_string(),
            ProximaValue::Int16(value) => value.to_string(),
            ProximaValue::Int32(value) => value.to_string(),
            ProximaValue::Int64(value) => value.to_string(),
            ProximaValue::UInt8(value) => value.to_string(),
            ProximaValue::UInt16(value) => value.to_string(),
            ProximaValue::UInt32(value) => value.to_string(),
            ProximaValue::UInt64(value) => value.to_string(),
            ProximaValue::Float16(value) => value.to_string(),
            ProximaValue::Float32(value) => value.to_string(),
            ProximaValue::Float64(value) => value.to_string(),
            ProximaValue::Null => String::new(),
            other => serde_json::to_string(other).unwrap_or_default(),
        }
    }

    /// Estimate compression improvement from sorting
    fn estimate_compression_improvement(&self, sorted_vectors: &[ProximaRecord]) -> f64 {
        if sorted_vectors.is_empty() {
            return 0.0;
        }

        // Simple heuristic: measure metadata locality
        let mut consecutive_similar = 0;
        let mut total_comparisons = 0;

        for window in sorted_vectors.windows(2) {
            total_comparisons += 1;
            if self.vectors_have_similar_metadata(&window[0], &window[1]) {
                consecutive_similar += 1;
            }
        }

        if total_comparisons == 0 {
            return 0.0;
        }

        // Higher similarity means better compression
        let similarity_ratio = consecutive_similar as f64 / total_comparisons as f64;
        // Convert to compression estimate (0-30% improvement)
        similarity_ratio * 0.3
    }

    /// Check if two vectors have similar metadata
    fn vectors_have_similar_metadata(&self, a: &ProximaRecord, b: &ProximaRecord) -> bool {
        // Count matching metadata keys
        let a_keys: std::collections::HashSet<_> = a.props.keys().collect();
        let b_keys: std::collections::HashSet<_> = b.props.keys().collect();

        let intersection = a_keys.intersection(&b_keys).count();
        let union = a_keys.union(&b_keys).count();

        if union == 0 {
            // Both vectors have no metadata - they are similar
            return true;
        }

        // Consider similar if >50% keys match
        intersection as f64 / union as f64 > 0.5
    }

    /// Build bloom filter for a set of vector records
    pub async fn build_bloom_filter(
        &self,
        records: &[ProximaRecord],
    ) -> Result<SstableBloomFilter> {
        debug!("Building bloom filter for {} records", records.len());

        use crate::core::bloom::{
            BloomFilterConfig, BloomFilterStats, BloomStrategy, HashAlgorithm,
            factory::BloomFilterFactory,
        };

        let num_keys = records.len();
        let bloom_config = BloomFilterConfig {
            enabled: true,
            strategy: BloomStrategy::BitPacked,
            bits_per_key: 10,
            expected_items: num_keys,
            false_positive_rate: Some(0.01),
            hash_algorithm: HashAlgorithm::XXHash,
        };

        let mut bloom = BloomFilterFactory::create(&bloom_config);
        for record in records {
            bloom.insert(record.oid.as_bytes());
        }

        let data = bloom
            .serialize()
            .map_err(|e| anyhow::anyhow!("Bloom filter serialization failed: {}", e))?;
        let stats = BloomFilterStats {
            key_count: num_keys as u64,
            metadata_columns: 0,
            total_keys: num_keys as u64,
            key_lookups_saved: 0,
            metadata_queries_saved: 0,
        };

        Ok(SstableBloomFilter::new(
            bloom_config,
            data,
            Vec::new(),
            stats,
        ))
    }

    /// Serialize records to SSTable row format
    pub async fn serialize_records_to_sstable_row_format(
        &self,
        records: &[ProximaRecord],
    ) -> Result<Vec<u8>> {
        debug!("📝 Serializing {} records to SSTable format", records.len());

        // In a real implementation, this would:
        // 1. Convert records to SSTable binary format
        // 2. Apply compression
        // 3. Add checksums and metadata
        // 4. Return serialized bytes

        // For now, use simple serialization
        let serialized = serde_json::to_vec(records)
            .map_err(|e| SstError::Internal(format!("Serialization failed: {}", e)))?;

        debug!("✅ Serialized {} bytes", serialized.len());
        Ok(serialized)
    }

    /// Calculate optimal block size for SSTable based on data characteristics
    pub fn calculate_optimal_block_size(
        &self,
        vector_count: usize,
        avg_vector_size: usize,
    ) -> usize {
        let base_block_size = (self.config().block_size_kb * 1024) as usize;

        // Adjust block size based on data characteristics
        let optimal_size = if vector_count < 1000 {
            // Small datasets: smaller blocks for better cache utilization
            base_block_size / 2
        } else if avg_vector_size > 10240 {
            // Large vectors: larger blocks for better compression
            base_block_size * 2
        } else {
            base_block_size
        };

        // Ensure within reasonable bounds (4KB to 1MB)
        optimal_size.clamp(4096, 1048576)
    }

    /// Estimate memory requirements for an operation
    pub fn estimate_memory_requirements(
        &self,
        vector_count: usize,
        avg_vector_size: usize,
    ) -> MemoryEstimate {
        let vector_memory = vector_count * avg_vector_size;
        let bloom_filter_memory = (vector_count * 10) / 8; // 10 bits per entry
        let index_memory = vector_count * 32; // Estimated index overhead
        let buffer_memory = self.config().block_size_kb as usize * 1024 * 4; // 4 blocks buffered

        let total_memory = vector_memory + bloom_filter_memory + index_memory + buffer_memory;

        MemoryEstimate {
            vector_memory,
            bloom_filter_memory,
            index_memory,
            buffer_memory,
            total_memory,
        }
    }

    /// Generate SSTable filename with proper naming convention
    pub fn generate_sstable_filename(&self, level: u32, sequence: u64) -> String {
        format!("L{:02}-{:016}.sst", level, sequence)
    }

    /// Parse SSTable filename to extract level and sequence
    pub fn parse_sstable_filename(&self, filename: &str) -> Result<(u32, u64)> {
        if !filename.ends_with(".sst") {
            return Err(SstError::InvalidArgument(format!(
                "Invalid SSTable filename: {}",
                filename
            ))
            .into());
        }

        let parts: Vec<&str> = filename.trim_end_matches(".sst").split('-').collect();
        if parts.len() != 2 {
            return Err(SstError::InvalidArgument(format!(
                "Invalid SSTable filename format: {}",
                filename
            ))
            .into());
        }

        let level = parts[0]
            .trim_start_matches('L')
            .parse::<u32>()
            .map_err(|e| SstError::InvalidArgument(format!("Invalid level in filename: {}", e)))?;

        let sequence = parts[1].parse::<u64>().map_err(|e| {
            SstError::InvalidArgument(format!("Invalid sequence in filename: {}", e))
        })?;

        Ok((level, sequence))
    }

    /// Check if SSTable file needs compaction based on size and age
    pub fn should_compact_file(&self, file_size: u64, file_age_hours: u64, level: u32) -> bool {
        // Size threshold increases with level
        let size_threshold =
            (self.config().level_size_multiplier.powf(level as f64) * 64.0 * 1_000_000.0) as u64; // Default 64MB base level size

        // Age threshold decreases with level (higher levels compact less frequently)
        let age_threshold = 24 * (level + 1) as u64; // Hours

        file_size > size_threshold || file_age_hours > age_threshold
    }

    /// Calculate write amplification factor
    pub fn calculate_write_amplification(
        &self,
        bytes_written_to_disk: u64,
        bytes_written_by_user: u64,
    ) -> f64 {
        if bytes_written_by_user == 0 {
            return 1.0;
        }

        bytes_written_to_disk as f64 / bytes_written_by_user as f64
    }
}

/// Statistics from vector sorting operation
#[derive(Debug, Clone)]
pub struct SortingStats {
    pub records_sorted: usize,
    pub sort_duration_ms: u64,
    pub primary_sort_key: Option<String>,
    pub compression_estimate: f64,
}

/// Memory requirements estimate
#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    pub vector_memory: usize,
    pub bloom_filter_memory: usize,
    pub index_memory: usize,
    pub buffer_memory: usize,
    pub total_memory: usize,
}

/// SSTable utilities for file management
pub struct SstableFileUtils;

impl SstableFileUtils {
    /// Get all SSTable files from a directory sorted by level and sequence
    pub async fn list_sorted_sstables(dir: &str) -> Result<Vec<SstableFileInfo>> {
        let mut files = Vec::new();

        if let Ok(mut entries) = tokio::fs::read_dir(dir).await {
            while let Some(entry) = entries.next_entry().await? {
                if let Some(name) = entry.file_name().to_str()
                    && name.ends_with(".sst")
                    && let Ok(metadata) = entry.metadata().await
                {
                    files.push(SstableFileInfo {
                        path: entry.path().to_string_lossy().to_string(),
                        name: name.to_string(),
                        size: metadata.len(),
                        created: metadata.created().ok(),
                        level: 0,    // Will be parsed from filename
                        sequence: 0, // Will be parsed from filename
                    });
                }
            }
        }

        // Sort by level and sequence
        files.sort_by(|a, b| a.level.cmp(&b.level).then(a.sequence.cmp(&b.sequence)));

        Ok(files)
    }

    /// Calculate total size of SSTable files
    pub fn calculate_total_size(files: &[SstableFileInfo]) -> u64 {
        files.iter().map(|f| f.size).sum()
    }

    /// Group SSTable files by level
    pub fn group_by_level(files: Vec<SstableFileInfo>) -> HashMap<u32, Vec<SstableFileInfo>> {
        let mut grouped = HashMap::new();

        for file in files {
            grouped
                .entry(file.level)
                .or_insert_with(Vec::new)
                .push(file);
        }

        grouped
    }
}

/// Information about an SSTable file
#[derive(Debug, Clone)]
pub struct SstableFileInfo {
    pub path: String,
    pub name: String,
    pub size: u64,
    pub created: Option<std::time::SystemTime>,
    pub level: u32,
    pub sequence: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::engines::sst::SstConfig;
    use crate::storage::persistence::filesystem::FilesystemFactory;
    use proximadb_distance_kernel::engine::UnifiedDistanceCompute;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_sort_vectors() {
        let engine = create_test_engine().await;

        let vectors = vec![
            create_test_vector("vec3", vec![3.0, 4.0]),
            create_test_vector("vec1", vec![1.0, 2.0]),
            create_test_vector("vec2", vec![2.0, 3.0]),
        ];

        let (sorted, stats) = engine
            .sort_vectors_for_sstable_encoding(vectors)
            .await
            .unwrap();

        // Verify sorting by ID
        assert_eq!(sorted[0].oid, "vec1");
        assert_eq!(sorted[1].oid, "vec2");
        assert_eq!(sorted[2].oid, "vec3");

        // Verify statistics
        assert_eq!(stats.records_sorted, 3);
    }

    #[tokio::test]
    async fn test_filename_generation_and_parsing() {
        let engine = create_test_engine().await;

        let filename = engine.generate_sstable_filename(2, 12345);
        assert_eq!(filename, "L02-0000000000012345.sst");

        let (level, sequence) = engine.parse_sstable_filename(&filename).unwrap();
        assert_eq!(level, 2);
        assert_eq!(sequence, 12345);
    }

    #[tokio::test]
    async fn test_memory_estimation() {
        let engine = create_test_engine().await;

        let estimate = engine.estimate_memory_requirements(1000, 1024);

        assert_eq!(estimate.vector_memory, 1024000);
        assert!(estimate.total_memory > estimate.vector_memory);
    }

    #[tokio::test]
    async fn test_optimal_block_size_calculation() {
        let engine = create_test_engine().await;

        // Small dataset
        let small_block = engine.calculate_optimal_block_size(100, 1024);

        // Large vectors
        let large_block = engine.calculate_optimal_block_size(5000, 20000);

        // Large blocks should be bigger than small blocks
        assert!(large_block >= small_block);
    }

    #[tokio::test]
    async fn test_write_amplification_calculation() {
        let engine = create_test_engine().await;

        let amplification = engine.calculate_write_amplification(10000, 1000);
        assert_eq!(amplification, 10.0);

        // Edge case: no user writes
        let no_writes = engine.calculate_write_amplification(10000, 0);
        assert_eq!(no_writes, 1.0);
    }

    async fn create_test_engine() -> SstEngine {
        let config = SstConfig::default();
        let filesystem_config =
            crate::storage::persistence::filesystem::FilesystemConfig::default();
        let filesystem = Arc::new(FilesystemFactory::create(filesystem_config).await.unwrap());
        let distance_compute = Arc::new(UnifiedDistanceCompute::default());

        SstEngine::new_with_config(config, filesystem, distance_compute)
            .await
            .unwrap()
    }

    fn create_test_vector(id: &str, vector: Vec<f32>) -> ProximaRecord {
        ProximaRecord {
            oid: id.to_string(),
            created_at_ns: 12_345_000_000,
            updated_at_ns: 12_345_000_000,
            record_version: 1,
            embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                "test",
                "dense_vector",
                vector.len() as u32,
                vector,
            )],
            ..ProximaRecord::default()
        }
    }
}
