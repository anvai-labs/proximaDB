//! Adaptive P×K Storage Implementation
//!
//! Implements the refined formula with minimum 10% coverage floor
//! and intelligent compression strategies based on K/D relationship

use super::common::{
    VectorCentroidCompressionMetadata, VectorCentroidMatrix, VectorCentroidStorageStrategy,
};
use super::config::{CompressionStrategy, PxKStrategy};
use crate::compute::quantization::storage_engine::StorageQuantizationEngine;
use crate::proto::proximadb_v1::DistanceMetric;
use anyhow::Result;
use proximadb_distance_kernel::engine::UnifiedDistanceCompute;
use std::collections::HashSet;

/// Selection reason for sparse storage
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum SelectionReason {
    /// Near cluster boundary
    Boundary,
    /// Cluster medoid/representative
    Representative,
    /// Far from centroid
    Outlier,
    /// Maximizes coverage diversity
    Diverse,
    /// Random statistical sample
    Random,
}

/// Vector selection info for sparse storage
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VectorSelection {
    pub vector_idx: u32,
    pub selection_reason: SelectionReason,
    pub importance_score: f32,
}

/// Trait for different P×K storage implementations
pub trait PxKStorageImpl: Send + Sync {
    /// Store distances for a vector
    fn store_distances(&mut self, vector_idx: usize, distances: &[f32]) -> Result<()>;

    /// Get distances for a vector
    fn get_distances(&self, vector_idx: usize) -> Option<Vec<f32>>;

    /// Detect boundary vectors
    fn detect_boundaries(&self, threshold: f32) -> Vec<BoundaryInfo>;

    /// Get memory usage in bytes
    fn memory_usage(&self) -> usize;

    /// Serialize to bytes
    fn serialize(&self) -> Result<Vec<u8>>;
}

/// Boundary information for a vector
#[derive(Debug, Clone)]
pub struct BoundaryInfo {
    pub vector_idx: u32,
    pub primary_cluster: u32,
    pub primary_distance: f32,
    pub secondary_cluster: u32,
    pub secondary_distance: f32,
    pub boundary_ratio: f32,
}

/// Dense full storage (no compression)
pub struct DenseFullStorage {
    matrix: VectorCentroidMatrix,
}

impl PxKStorageImpl for DenseFullStorage {
    fn store_distances(&mut self, vector_idx: usize, distances: &[f32]) -> Result<()> {
        // For Full storage strategy, we store all distances in compressed_data
        // Layout: [v0_c0, v0_c1, ..., v0_ck, v1_c0, v1_c1, ..., v1_ck, ...]
        // Each distance is quantized to u16 for compression

        let num_centroids = self.matrix.num_centroids as usize;
        let range = self.matrix.compression_metadata.global_max_distance
            - self.matrix.compression_metadata.global_min_distance;
        let scale = if range > 0.0 { 65535.0 / range } else { 1.0 };

        // Ensure we have enough space
        let required_size = (vector_idx + 1) * num_centroids * 2; // 2 bytes per u16
        if self.matrix.compressed_data.len() < required_size {
            self.matrix.compressed_data.resize(required_size, 0);
        }

        // Store quantized distances
        for (centroid_idx, &distance) in distances.iter().enumerate() {
            let quantized =
                ((distance - self.matrix.compression_metadata.global_min_distance) * scale) as u16;
            let offset = (vector_idx * num_centroids + centroid_idx) * 2;
            self.matrix.compressed_data[offset] = (quantized & 0xFF) as u8;
            self.matrix.compressed_data[offset + 1] = (quantized >> 8) as u8;
        }

        Ok(())
    }

    fn get_distances(&self, vector_idx: usize) -> Option<Vec<f32>> {
        let num_centroids = self.matrix.num_centroids as usize;
        let num_vectors = self.matrix.num_vectors as usize;

        if vector_idx >= num_vectors {
            return None;
        }

        let mut distances = Vec::with_capacity(num_centroids);

        // Dequantize distances from compressed storage
        for centroid_idx in 0..num_centroids {
            match self.matrix.get_distance(vector_idx, centroid_idx) {
                Ok(distance) => distances.push(distance),
                Err(_) => distances.push(f32::INFINITY), // Missing distance
            }
        }

        Some(distances)
    }

    fn detect_boundaries(&self, threshold: f32) -> Vec<BoundaryInfo> {
        let mut boundaries = Vec::new();
        let num_vectors = self.matrix.num_vectors as usize;
        let num_centroids = self.matrix.num_centroids as usize;

        // Iterate through all vectors to find boundary cases
        for vector_idx in 0..num_vectors {
            // Get all distances for this vector
            let mut distances: Vec<(usize, f32)> = Vec::with_capacity(num_centroids);

            for centroid_idx in 0..num_centroids {
                if let Ok(distance) = self.matrix.get_distance(vector_idx, centroid_idx) {
                    distances.push((centroid_idx, distance));
                }
            }

            if distances.len() < 2 {
                continue;
            }

            // Sort by distance to find closest centroids
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Calculate boundary ratio: closest / second_closest
            // A ratio close to 1.0 means the vector is on a boundary
            let ratio = if distances[1].1 > 0.0 {
                distances[0].1 / distances[1].1
            } else {
                0.0
            };

            // Check if this is a boundary vector (ratio > threshold, typically 0.8)
            if ratio > threshold {
                boundaries.push(BoundaryInfo {
                    vector_idx: vector_idx as u32,
                    primary_cluster: distances[0].0 as u32,
                    primary_distance: distances[0].1,
                    secondary_cluster: distances[1].0 as u32,
                    secondary_distance: distances[1].1,
                    boundary_ratio: ratio,
                });
            }
        }

        boundaries
    }

    fn memory_usage(&self) -> usize {
        // Deferred: Calculate actual memory usage from compressed storage
        // self.matrix.distances.len() * self.matrix.num_clusters * 4
        self.matrix.compressed_data.len()
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        // Serialize matrix to bytes
        bincode::serialize(&self.matrix).map_err(Into::into)
    }
}

/// Sparse coverage storage with compression
pub struct SparseCoverageStorage {
    coverage: f32,
    compression: CompressionStrategy,
    selected_vectors: Vec<VectorSelection>,
    compressed_distances: Vec<Vec<u8>>,
    num_clusters: usize,
    quantization_engine: StorageQuantizationEngine,
}

impl SparseCoverageStorage {
    /// Create new sparse storage
    pub fn new(coverage: f32, compression: CompressionStrategy, num_clusters: usize) -> Self {
        Self {
            coverage,
            compression,
            selected_vectors: Vec::new(),
            compressed_distances: Vec::new(),
            num_clusters,
            // Quantization engine will be initialized when needed
            quantization_engine: {
                let distance_compute =
                    std::sync::Arc::new(UnifiedDistanceCompute::new(DistanceMetric::Cosine));
                let codebook_store: std::sync::Arc<
                    dyn crate::compute::quantization::quantization_engine::CodebookStore,
                > = std::sync::Arc::new(
                    crate::compute::quantization::quantization_engine::InMemoryCodebookStore::new(),
                );
                let unified_engine = std::sync::Arc::new(
                    crate::compute::quantization::quantization_engine::UnifiedQuantizationEngine::new(
                        distance_compute.clone(),
                        codebook_store,
                    ),
                );
                let config = crate::compute::quantization::storage_engine::StorageQuantizationConfig::default();
                StorageQuantizationEngine::new(unified_engine, distance_compute, config)
            },
        }
    }

    /// Select vectors intelligently based on coverage
    pub fn select_vectors(
        &mut self,
        vectors: &[Vec<f32>],
        centroids: &[Vec<f32>],
    ) -> Vec<VectorSelection> {
        let num_select = (vectors.len() as f32 * self.coverage) as usize;
        let mut selected = Vec::with_capacity(num_select);
        let mut selected_indices = HashSet::new();

        // Priority 1: Boundary vectors (40% of budget)
        let boundary_budget = (num_select as f32 * 0.4) as usize;
        let boundaries = self.select_boundary_vectors(vectors, centroids, boundary_budget);
        for b in boundaries {
            selected_indices.insert(b.vector_idx);
            selected.push(b);
        }

        // Priority 2: Representatives (20% of budget)
        let rep_budget = (num_select as f32 * 0.2) as usize;
        let representatives = self.select_representatives(vectors, rep_budget, &selected_indices);
        for r in representatives {
            selected_indices.insert(r.vector_idx);
            selected.push(r);
        }

        // Priority 3: Outliers (20% of budget)
        let outlier_budget = (num_select as f32 * 0.2) as usize;
        let outliers = self.select_outliers(vectors, centroids, outlier_budget, &selected_indices);
        for o in outliers {
            selected_indices.insert(o.vector_idx);
            selected.push(o);
        }

        // Priority 4: Diverse sample (remaining budget)
        let remaining = num_select.saturating_sub(selected.len());
        let diverse = self.select_diverse(vectors, remaining, &selected_indices);
        selected.extend(diverse);

        self.selected_vectors = selected.clone();
        selected
    }

    fn select_boundary_vectors(
        &self,
        vectors: &[Vec<f32>],
        centroids: &[Vec<f32>],
        budget: usize,
    ) -> Vec<VectorSelection> {
        let mut boundaries = Vec::new();
        let distance_compute = UnifiedDistanceCompute::new(DistanceMetric::Cosine);

        for (idx, vector) in vectors.iter().enumerate() {
            // Calculate distances to all centroids
            let mut distances: Vec<f32> = centroids
                .iter()
                .map(|c| {
                    distance_compute
                        .calculate_distance(vector, c, &DistanceMetric::Cosine)
                        .raw_value
                })
                .collect();
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            if distances.len() >= 2 {
                let ratio = distances[0] / distances[1];
                if ratio > 0.8 {
                    boundaries.push(VectorSelection {
                        vector_idx: idx as u32,
                        selection_reason: SelectionReason::Boundary,
                        importance_score: ratio,
                    });
                }
            }
        }

        // Sort by importance and take top budget
        boundaries.sort_by(|a, b| {
            b.importance_score
                .partial_cmp(&a.importance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        boundaries.truncate(budget);
        boundaries
    }

    fn select_representatives(
        &self,
        vectors: &[Vec<f32>],
        budget: usize,
        selected: &HashSet<u32>,
    ) -> Vec<VectorSelection> {
        // Simple k-medoids selection
        let mut representatives = Vec::new();
        let distance_compute = UnifiedDistanceCompute::new(DistanceMetric::Cosine);

        // Find vectors closest to mean
        let dim = vectors[0].len();
        let mut mean = vec![0.0; dim];
        for v in vectors {
            for (i, &val) in v.iter().enumerate() {
                mean[i] += val;
            }
        }
        for val in &mut mean {
            *val /= vectors.len() as f32;
        }

        let mut distances: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .filter(|(idx, _)| !selected.contains(&(*idx as u32)))
            .map(|(idx, v)| {
                (
                    idx,
                    distance_compute
                        .calculate_distance(v, &mean, &DistanceMetric::Cosine)
                        .raw_value,
                )
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for (idx, dist) in distances.iter().take(budget) {
            representatives.push(VectorSelection {
                vector_idx: *idx as u32,
                selection_reason: SelectionReason::Representative,
                importance_score: 1.0 / (1.0 + dist),
            });
        }

        representatives
    }

    fn select_outliers(
        &self,
        vectors: &[Vec<f32>],
        centroids: &[Vec<f32>],
        budget: usize,
        selected: &HashSet<u32>,
    ) -> Vec<VectorSelection> {
        let mut outliers = Vec::new();
        let distance_compute = UnifiedDistanceCompute::new(DistanceMetric::Cosine);

        for (idx, vector) in vectors.iter().enumerate() {
            if selected.contains(&(idx as u32)) {
                continue;
            }

            // Find distance to nearest centroid
            let min_dist = centroids
                .iter()
                .map(|c| {
                    distance_compute
                        .calculate_distance(vector, c, &DistanceMetric::Cosine)
                        .raw_value
                })
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(f32::MAX);

            outliers.push(VectorSelection {
                vector_idx: idx as u32,
                selection_reason: SelectionReason::Outlier,
                importance_score: min_dist,
            });
        }

        // Sort by distance (furthest first) and take top budget
        outliers.sort_by(|a, b| {
            b.importance_score
                .partial_cmp(&a.importance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        outliers.truncate(budget);
        outliers
    }

    fn select_diverse(
        &self,
        vectors: &[Vec<f32>],
        budget: usize,
        selected: &HashSet<u32>,
    ) -> Vec<VectorSelection> {
        let mut diverse = Vec::new();
        let distance_compute = UnifiedDistanceCompute::new(DistanceMetric::Cosine);
        let mut selected_diverse = HashSet::new();

        // Start with random vector
        let available: Vec<usize> = (0..vectors.len())
            .filter(|i| !selected.contains(&(*i as u32)))
            .collect();

        if available.is_empty() {
            return diverse;
        }

        let first = available[0];
        selected_diverse.insert(first);
        diverse.push(VectorSelection {
            vector_idx: first as u32,
            selection_reason: SelectionReason::Diverse,
            importance_score: 1.0,
        });

        // Furthest-first traversal
        while diverse.len() < budget && diverse.len() < available.len() {
            let mut best_idx = None;
            let mut best_min_dist = 0.0;

            for &idx in &available {
                if selected_diverse.contains(&idx) {
                    continue;
                }

                // Find minimum distance to any selected vector
                let min_dist = selected_diverse
                    .iter()
                    .map(|&s| {
                        distance_compute
                            .calculate_distance(&vectors[idx], &vectors[s], &DistanceMetric::Cosine)
                            .raw_value
                    })
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(f32::MAX);

                if min_dist > best_min_dist {
                    best_min_dist = min_dist;
                    best_idx = Some(idx);
                }
            }

            if let Some(idx) = best_idx {
                selected_diverse.insert(idx);
                diverse.push(VectorSelection {
                    vector_idx: idx as u32,
                    selection_reason: SelectionReason::Diverse,
                    importance_score: best_min_dist,
                });
            } else {
                break;
            }
        }

        diverse
    }

    /// Compress distances based on strategy
    fn compress_distances(&self, distances: &[f32]) -> Vec<u8> {
        match self.compression {
            CompressionStrategy::Uncompressed => bincode::serialize(distances).unwrap_or_default(),
            CompressionStrategy::Float16 => {
                // Use the quantization engine for consistent quantization
                let (quantized, min, max) = self.quantization_engine.quantize_to_u16(distances);

                // Store quantization parameters for accurate dequantization
                // Format: [min:4bytes][max:4bytes][quantized_values:2bytes each]
                let mut result = Vec::new();
                result.extend_from_slice(&min.to_le_bytes());
                result.extend_from_slice(&max.to_le_bytes());

                for val in quantized {
                    result.extend_from_slice(&val.to_le_bytes());
                }
                result
            }
            CompressionStrategy::Quantized8 => {
                // Use the quantization engine for consistent quantization
                let (quantized, min, max) = self.quantization_engine.quantize_to_u8(distances);

                // Format: [min:4bytes][max:4bytes][quantized_values:1byte each]
                let mut result = Vec::new();
                result.extend_from_slice(&min.to_le_bytes());
                result.extend_from_slice(&max.to_le_bytes());
                result.extend_from_slice(&quantized);
                result
            }
            CompressionStrategy::Quantized4 => {
                // Use the quantization engine's dedicated 4-bit quantization
                let (packed, min, max, num_values) =
                    self.quantization_engine.quantize_to_u4(distances);

                // Format: [min:4bytes][max:4bytes][num_values:4bytes][packed_values]
                let mut result = Vec::new();
                result.extend_from_slice(&min.to_le_bytes());
                result.extend_from_slice(&max.to_le_bytes());
                result.extend_from_slice(&(num_values as u32).to_le_bytes());
                result.extend_from_slice(&packed);
                result
            }
            CompressionStrategy::Quantized6 => {
                // Use the quantization engine's 6-bit quantization
                let (packed, min, max, num_values) =
                    self.quantization_engine.quantize_to_u6(distances);

                // Format: [min:4bytes][max:4bytes][num_values:4bytes][packed_values]
                let mut result = Vec::new();
                result.extend_from_slice(&min.to_le_bytes());
                result.extend_from_slice(&max.to_le_bytes());
                result.extend_from_slice(&(num_values as u32).to_le_bytes());
                result.extend_from_slice(&packed);
                result
            }
            CompressionStrategy::DeltaEncoded => {
                // Sort and store deltas
                let mut sorted = distances.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let mut result = Vec::new();
                result.extend_from_slice(&sorted[0].to_le_bytes());

                for window in sorted.windows(2) {
                    let delta = ((window[1] - window[0]) * 1000.0) as i16;
                    result.extend_from_slice(&delta.to_le_bytes());
                }
                result
            }
            CompressionStrategy::BitPacked => {
                // Bit packing based on range
                let min = distances.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = distances.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let range = max - min;

                let mut result = Vec::new();
                result.extend_from_slice(&min.to_le_bytes());
                result.extend_from_slice(&range.to_le_bytes());

                // Use 12 bits per value (4096 levels)
                let mut bit_buffer = 0u32;
                let mut bits_in_buffer = 0;

                for &d in distances {
                    let normalized = ((d - min) / range * 4095.0) as u32;
                    bit_buffer |= normalized << bits_in_buffer;
                    bits_in_buffer += 12;

                    while bits_in_buffer >= 8 {
                        result.push((bit_buffer & 0xFF) as u8);
                        bit_buffer >>= 8;
                        bits_in_buffer -= 8;
                    }
                }

                if bits_in_buffer > 0 {
                    result.push((bit_buffer & 0xFF) as u8);
                }

                result
            }
        }
    }
}

impl PxKStorageImpl for SparseCoverageStorage {
    fn store_distances(&mut self, vector_idx: usize, distances: &[f32]) -> Result<()> {
        // Only store if this vector was selected
        if self
            .selected_vectors
            .iter()
            .any(|v| v.vector_idx == vector_idx as u32)
        {
            let compressed = self.compress_distances(distances);
            self.compressed_distances.push(compressed);
        }
        Ok(())
    }

    fn get_distances(&self, vector_idx: usize) -> Option<Vec<f32>> {
        // Find if this vector was selected
        let position = self
            .selected_vectors
            .iter()
            .position(|v| v.vector_idx == vector_idx as u32)?;

        // Decompress distances
        let compressed = &self.compressed_distances[position];

        match self.compression {
            CompressionStrategy::Uncompressed => bincode::deserialize(compressed).ok(),
            CompressionStrategy::Float16 => {
                // Decompress from 16-bit quantized format
                // Format: [min:4bytes][max:4bytes][quantized_values:2bytes each]
                if compressed.len() < 8 {
                    return None;
                }
                let min = f32::from_le_bytes([
                    compressed[0],
                    compressed[1],
                    compressed[2],
                    compressed[3],
                ]);
                let max = f32::from_le_bytes([
                    compressed[4],
                    compressed[5],
                    compressed[6],
                    compressed[7],
                ]);
                let range = if max > min { max - min } else { 1.0 };

                let mut distances = Vec::new();
                for chunk in compressed[8..].as_chunks::<2>().0 {
                    let quantized = u16::from_le_bytes(*chunk);
                    let normalized = quantized as f32 / 65535.0;
                    distances.push(min + normalized * range);
                }
                Some(distances)
            }
            CompressionStrategy::Quantized8 => {
                // Decompress from 8-bit quantized format
                // Format: [min:4bytes][max:4bytes][quantized_values:1byte each]
                if compressed.len() < 8 {
                    return None;
                }
                let min = f32::from_le_bytes([
                    compressed[0],
                    compressed[1],
                    compressed[2],
                    compressed[3],
                ]);
                let max = f32::from_le_bytes([
                    compressed[4],
                    compressed[5],
                    compressed[6],
                    compressed[7],
                ]);
                let range = if max > min { max - min } else { 1.0 };

                let mut distances = Vec::new();
                for &quantized in &compressed[8..] {
                    let normalized = quantized as f32 / 255.0;
                    distances.push(min + normalized * range);
                }
                Some(distances)
            }
            CompressionStrategy::Quantized4 => {
                // Decompress from 4-bit quantized format
                // Format: [min:4bytes][max:4bytes][num_values:4bytes][packed_values]
                if compressed.len() < 12 {
                    return None;
                }
                let min = f32::from_le_bytes([
                    compressed[0],
                    compressed[1],
                    compressed[2],
                    compressed[3],
                ]);
                let max = f32::from_le_bytes([
                    compressed[4],
                    compressed[5],
                    compressed[6],
                    compressed[7],
                ]);
                let num_values = u32::from_le_bytes([
                    compressed[8],
                    compressed[9],
                    compressed[10],
                    compressed[11],
                ]) as usize;

                let packed = &compressed[12..];
                let distances = self
                    .quantization_engine
                    .dequantize_u4(packed, min, max, num_values);
                Some(distances)
            }
            CompressionStrategy::Quantized6 => {
                // Decompress from 6-bit quantized format
                // Format: [min:4bytes][max:4bytes][num_values:4bytes][packed_values]
                if compressed.len() < 12 {
                    return None;
                }
                let min = f32::from_le_bytes([
                    compressed[0],
                    compressed[1],
                    compressed[2],
                    compressed[3],
                ]);
                let max = f32::from_le_bytes([
                    compressed[4],
                    compressed[5],
                    compressed[6],
                    compressed[7],
                ]);
                let num_values = u32::from_le_bytes([
                    compressed[8],
                    compressed[9],
                    compressed[10],
                    compressed[11],
                ]) as usize;

                let packed = &compressed[12..];
                let distances = self
                    .quantization_engine
                    .dequantize_u6(packed, min, max, num_values);
                Some(distances)
            }
            _ => None, // DeltaEncoded and other strategies not implemented yet
        }
    }

    fn detect_boundaries(&self, _threshold: f32) -> Vec<BoundaryInfo> {
        // Only detect boundaries for selected vectors
        Vec::new() // Simplified for now
    }

    fn memory_usage(&self) -> usize {
        self.compressed_distances.iter().map(|d| d.len()).sum()
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        // Serialize all data
        let data = (
            &self.coverage,
            &self.compression,
            &self.selected_vectors,
            &self.compressed_distances,
            &self.num_clusters,
        );
        bincode::serialize(&data).map_err(Into::into)
    }
}

/// Main adaptive P×K storage
pub struct AdaptivePxKStorage {
    pub strategy: PxKStrategy,
    pub storage: Box<dyn PxKStorageImpl>,
}

impl AdaptivePxKStorage {
    /// Create new adaptive storage based on K and D
    pub fn new(k: usize, d: usize, p: usize) -> Self {
        let (strategy, _coverage) = Self::determine_strategy(k, d);
        let storage = Self::create_storage(&strategy, k, p);

        Self { strategy, storage }
    }

    /// Determine optimal strategy based on K/D relationship
    pub fn determine_strategy(k: usize, d: usize) -> (PxKStrategy, f32) {
        let sqrt_d = (d as f32).sqrt() as usize;

        match k {
            // Full storage for small K
            k if k < sqrt_d => (PxKStrategy::DenseFull, 1.0),

            // Compressed full storage
            k if k < d / 4 => (PxKStrategy::DenseCompressed, 1.0),

            // Adaptive sparse storage
            k if k < 10000 => {
                let coverage = Self::calculate_optimal_coverage(k, d);
                let compression = Self::select_compression(k, d, coverage);
                (
                    PxKStrategy::SparseCoverage {
                        coverage,
                        compression,
                    },
                    coverage,
                )
            }

            // Learned index for extreme K
            _ => (PxKStrategy::LearnedIndex, 0.0),
        }
    }

    /// Calculate optimal coverage using refined formula
    fn calculate_optimal_coverage(k: usize, d: usize) -> f32 {
        const MIN_COVERAGE: f32 = 0.10; // 10% floor
        let ratio = k as f32 / d as f32;

        if ratio < 0.25 {
            // k < d/4
            1.0
        } else if ratio < 1.0 {
            // k < d
            // Linear decay from 100% to 50%
            1.0 - 0.5 * (ratio - 0.25) / 0.75
        } else {
            // Logarithmic decay with floor
            let coverage = d as f32 / (k as f32 * (ratio + 2.0).log2());
            coverage.max(MIN_COVERAGE)
        }
    }

    /// Select compression strategy based on K/D ratio and coverage
    fn select_compression(k: usize, d: usize, coverage: f32) -> CompressionStrategy {
        let ratio = k as f32 / d as f32;

        if ratio < 0.25 {
            CompressionStrategy::Uncompressed
        } else if ratio < 1.0 {
            if coverage > 0.8 {
                CompressionStrategy::Float16
            } else {
                CompressionStrategy::Quantized8
            }
        } else {
            if coverage > 0.3 {
                CompressionStrategy::Quantized8
            } else if coverage > 0.15 {
                CompressionStrategy::Quantized4
            } else {
                CompressionStrategy::DeltaEncoded
            }
        }
    }

    /// Create storage implementation based on strategy
    fn create_storage(strategy: &PxKStrategy, k: usize, p: usize) -> Box<dyn PxKStorageImpl> {
        match strategy {
            PxKStrategy::DenseFull | PxKStrategy::DenseCompressed => {
                Box::new(DenseFullStorage {
                    matrix: VectorCentroidMatrix {
                        rowgroup_id: 0, // Will be set when assigned to rowgroup
                        num_vectors: p as u32,
                        num_centroids: k as u32,
                        storage_strategy: VectorCentroidStorageStrategy::Full,
                        compressed_data: Vec::new(),
                        compression_metadata: VectorCentroidCompressionMetadata {
                            centroid_stats: Vec::new(),
                            global_min_distance: 0.0,
                            global_max_distance: 2.0,
                            global_mean_distance: 1.0,
                            centroid_encodings: Vec::new(),
                        },
                        sparse_data: None,
                        hierarchical_data: None,
                    },
                })
            }
            PxKStrategy::SparseCoverage {
                coverage,
                compression,
            } => Box::new(SparseCoverageStorage::new(
                *coverage,
                compression.clone(),
                k,
            )),
            PxKStrategy::LearnedIndex => {
                // Deferred: Implement learned index
                Box::new(DenseFullStorage {
                    matrix: VectorCentroidMatrix {
                        rowgroup_id: 0,
                        num_vectors: 0,
                        num_centroids: k as u32,
                        storage_strategy: VectorCentroidStorageStrategy::Full,
                        compressed_data: Vec::new(),
                        compression_metadata: VectorCentroidCompressionMetadata {
                            centroid_stats: Vec::new(),
                            global_min_distance: 0.0,
                            global_max_distance: 2.0,
                            global_mean_distance: 1.0,
                            centroid_encodings: Vec::new(),
                        },
                        sparse_data: None,
                        hierarchical_data: None,
                    },
                })
            }
        }
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.storage.memory_usage()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_determine_strategy_dense_full_small_k() {
        // k < sqrt(d): DenseFull
        let (strategy, coverage) = AdaptivePxKStorage::determine_strategy(5, 100);
        assert!(matches!(strategy, PxKStrategy::DenseFull));
        assert_eq!(coverage, 1.0);
    }

    #[test]
    fn test_determine_strategy_dense_compressed() {
        // sqrt(d) <= k < d/4: DenseCompressed
        // For d=256, sqrt(d)=16, d/4=64, so k=30 should be DenseCompressed
        let (strategy, coverage) = AdaptivePxKStorage::determine_strategy(30, 256);
        assert!(matches!(strategy, PxKStrategy::DenseCompressed));
        assert_eq!(coverage, 1.0);
    }

    #[test]
    fn test_determine_strategy_sparse_coverage() {
        // d/4 <= k < 10000: SparseCoverage
        // For d=100, d/4=25, so k=100 should be SparseCoverage
        let (strategy, coverage) = AdaptivePxKStorage::determine_strategy(100, 100);
        assert!(matches!(strategy, PxKStrategy::SparseCoverage { .. }));
        assert!(coverage > 0.0 && coverage <= 1.0);
    }

    #[test]
    fn test_determine_strategy_learned_index() {
        // k >= 10000: LearnedIndex
        let (strategy, coverage) = AdaptivePxKStorage::determine_strategy(10_000, 100);
        assert!(matches!(strategy, PxKStrategy::LearnedIndex));
        assert_eq!(coverage, 0.0);
    }

    #[test]
    fn test_calculate_optimal_coverage_low_ratio() {
        // ratio < 0.25: coverage = 1.0
        let coverage = AdaptivePxKStorage::calculate_optimal_coverage(10, 100);
        assert!(
            (coverage - 1.0).abs() < 0.001,
            "Low ratio should give 100% coverage, got {}",
            coverage
        );
    }

    #[test]
    fn test_calculate_optimal_coverage_medium_ratio() {
        // 0.25 <= ratio < 1.0: linear decay from 100% to 50%
        let coverage = AdaptivePxKStorage::calculate_optimal_coverage(50, 100);
        assert!(
            coverage > 0.49 && coverage < 1.01,
            "Medium ratio coverage {} should be between 0.5 and 1.0",
            coverage
        );
    }

    #[test]
    fn test_calculate_optimal_coverage_high_ratio_has_floor() {
        // ratio >= 1.0: logarithmic decay with 10% floor
        let coverage = AdaptivePxKStorage::calculate_optimal_coverage(5000, 100);
        assert!(
            coverage >= 0.10,
            "Coverage should never go below 10% floor, got {}",
            coverage
        );
    }

    #[test]
    fn test_calculate_optimal_coverage_monotonic_decrease() {
        let d = 256;
        let mut prev = 2.0; // Start above max
        for k in [10, 50, 100, 200, 500, 1000, 5000] {
            let coverage = AdaptivePxKStorage::calculate_optimal_coverage(k, d);
            assert!(
                coverage <= prev,
                "Coverage should decrease: k={}, coverage={}, prev={}",
                k,
                coverage,
                prev
            );
            prev = coverage;
        }
    }

    #[test]
    fn test_select_compression_uncompressed_low_ratio() {
        let strategy = AdaptivePxKStorage::select_compression(10, 100, 1.0);
        assert!(matches!(strategy, CompressionStrategy::Uncompressed));
    }

    #[test]
    fn test_select_compression_float16_medium_ratio_high_coverage() {
        let strategy = AdaptivePxKStorage::select_compression(50, 100, 0.9);
        assert!(matches!(strategy, CompressionStrategy::Float16));
    }

    #[test]
    fn test_select_compression_quantized8_medium_ratio_low_coverage() {
        let strategy = AdaptivePxKStorage::select_compression(50, 100, 0.5);
        assert!(matches!(strategy, CompressionStrategy::Quantized8));
    }

    #[test]
    fn test_select_compression_quantized4_high_ratio_low_coverage() {
        let strategy = AdaptivePxKStorage::select_compression(200, 100, 0.2);
        assert!(matches!(strategy, CompressionStrategy::Quantized4));
    }

    #[test]
    fn test_select_compression_delta_encoded_extreme() {
        let strategy = AdaptivePxKStorage::select_compression(1000, 100, 0.1);
        assert!(matches!(strategy, CompressionStrategy::DeltaEncoded));
    }

    #[test]
    fn test_memory_usage_empty_storage() {
        let storage = AdaptivePxKStorage::new(5, 100, 50);
        // Empty storage should have minimal memory usage
        let usage = storage.memory_usage();
        // DenseFull with empty compressed_data should be 0
        assert_eq!(usage, 0);
    }

    #[test]
    fn test_boundary_k_zero() {
        // k=0 should still produce a valid strategy (DenseFull since 0 < sqrt(d))
        let (strategy, _) = AdaptivePxKStorage::determine_strategy(0, 100);
        assert!(matches!(strategy, PxKStrategy::DenseFull));
    }

    #[test]
    fn test_boundary_d_equals_one() {
        // d=1: sqrt(d)=1, d/4=0
        // k=1 >= sqrt(1)=1 and k=1 >= d/4=0, so this depends on exact boundary conditions
        let (strategy, _) = AdaptivePxKStorage::determine_strategy(1, 1);
        // k=1, d=1: k < sqrt(d)=1 is false, k < d/4=0 is false, k < 10000 is true
        assert!(matches!(strategy, PxKStrategy::SparseCoverage { .. }));
    }

    #[test]
    fn test_selection_reason_variants() {
        // Verify all selection reason variants can be constructed
        let reasons = vec![
            SelectionReason::Boundary,
            SelectionReason::Representative,
            SelectionReason::Outlier,
            SelectionReason::Diverse,
            SelectionReason::Random,
        ];
        for reason in reasons {
            let selection = VectorSelection {
                vector_idx: 0,
                selection_reason: reason,
                importance_score: 1.0,
            };
            assert_eq!(selection.vector_idx, 0);
            assert_eq!(selection.importance_score, 1.0);
        }
    }
}
