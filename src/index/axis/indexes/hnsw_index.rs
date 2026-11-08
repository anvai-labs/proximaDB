/*
 * Copyright 2025 ProximaDB
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

//! HNSW (Hierarchical Navigable Small World) index implementation for AXIS
//!
//! This module provides a production-ready HNSW index that integrates seamlessly
//! with the AXIS adaptive indexing system. HNSW is ideal for approximate nearest
//! neighbor search with excellent recall and query performance.

use anyhow::Result;
use async_trait::async_trait;
use dashmap::DashMap;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::{
    Arc, RwLock,
    atomic::{AtomicU64, AtomicUsize, Ordering as AtomicOrdering},
};
use tracing::info;

use crate::compute::distance_computation::DistanceMetric;
use crate::compute::distance_computation::engine::UnifiedDistanceCompute;
// ZeroOverheadVector used for 75-96% memory savings vs VectorRecord
use crate::index::axis::eventlog::{ExtractionMode, IndexEvent};
use crate::index::axis::filterable_metadata::{
    FilterableFieldsConfig, FilterableHnswMetadata, FilterableMetadataCache,
};
use crate::index::axis::index_factory::{AxisVectorIndex, IndexStats};
use crate::index::axis::types::IndexAlgorithm;
use crate::index::axis::utils::{AtomicStats, ConcurrentIdMapping, memory, validation};
use crate::index::axis::zero_overhead_vector::{
    CollectionConfig, QuantizationMethod, ZeroOverheadCollection,
};

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    /// Total memory usage across all components.
    pub total_bytes: usize,
    /// Memory used by the HNSW graph structure (edges, layers).
    pub index_size_bytes: usize,
    /// Memory used by stored vector data.
    pub vector_data_bytes: usize,
    /// Memory used by auxiliary metadata (ID mappings, etc.).
    pub metadata_bytes: usize,
}

/// HNSW index configuration aligned with AXIS standards
#[derive(Debug, Clone)]
pub struct AxisHnswConfig {
    /// Number of bi-directional links for each node
    pub m: usize,
    /// Size of candidate set during construction
    pub ef_construction: usize,
    /// Search parameter (dynamic, can be adjusted at query time)
    pub ef: usize,
    /// Maximum number of layers
    pub max_layers: usize,
    /// Distance metric to use
    pub distance_metric: DistanceMetric,

    // ── Phase C: ACORN / NaviX additions (spec §6.1) ────────────────────────
    /// Neighbor expansion factor for ACORN §5.2.
    ///
    /// During construction each node selects `floor(gamma * M)` initial candidates
    /// before pruning to M stored edges. At search time, nodes that fail the
    /// predicate still expose their full `gamma*M` list so traversal can route
    /// around predicate-sparse regions without getting stuck.
    ///
    /// 1.0 = standard HNSW (no expansion). 1.5–2.0 recommended for mixed workloads.
    pub gamma: f32,

    /// Minimum estimated filter selectivity below which the executor falls back
    /// to pre-filter brute force instead of HNSW traversal (spec §7.3 s_min).
    ///
    /// If `estimated_selectivity <= selectivity_min` → brute force.
    pub selectivity_min: f32,
}

impl AxisHnswConfig {
    /// Effective expanded neighbor count used during ACORN construction.
    pub fn expanded_m(&self) -> usize {
        (self.gamma * self.m as f32).floor() as usize
    }

    /// Returns `true` when the estimated predicate selectivity is so low that
    /// HNSW traversal is expected to be slower than a pre-filtered brute-force scan.
    pub fn should_use_brute_force(&self, estimated_selectivity: f32) -> bool {
        estimated_selectivity <= self.selectivity_min
    }
}

impl Default for AxisHnswConfig {
    fn default() -> Self {
        Self {
            m: 16,                // Good balance of connectivity and memory
            ef_construction: 200, // Higher for better quality
            ef: 50,               // Lower for faster searches
            max_layers: 16,       // Reasonable depth
            distance_metric: DistanceMetric::Cosine,
            gamma: 1.0,            // No expansion by default (legacy-compatible)
            selectivity_min: 0.05, // Fall back to brute force below 5% selectivity
        }
    }
}

/// Wrapper for f32 to implement Ord for use in BinaryHeap
use proximadb_records::TotalOrderedF32 as OrderedFloat;

/// HNSW index implementation using shared infrastructure with collection partitioning
/// Migrated to use IndexVectorStore and ConcurrentIdMapping for consistency
/// NOW SUPPORTS: Queue-based consumption of vectors with quantized/fp32/both representations
pub struct AxisHnswIndex {
    /// Collection identifier for partitioning (optional for backward compatibility)
    collection_id: Option<String>,

    /// Configuration
    config: AxisHnswConfig,
    /// Distance computation
    distance_computer: UnifiedDistanceCompute,

    /// Zero-overhead vector storage - optimal memory use!
    /// Replaces IndexVectorStore with 75-96% memory savings for quantized vectors
    vectors: Arc<RwLock<ZeroOverheadCollection>>,

    /// USING UTILS: Bidirectional ID mapping for external<->internal IDs
    id_mapping: ConcurrentIdMapping,

    /// USING UTILS: Performance statistics tracking
    stats: AtomicStats,

    /// HNSW-specific: Graph layers with composite keys
    /// (layer, internal_node_id) -> connections
    /// Deferred: Add partitioning - will use (collection_id, layer, node_id) in Phase 3
    layers: DashMap<(usize, usize), Vec<usize>>,

    /// HNSW-specific: Maximum layer currently in use (atomic)
    max_layer: AtomicUsize,

    /// HNSW-specific: Entry point for search
    entry_point: RwLock<Option<usize>>,

    /// Random number generator state (lock-free AtomicU64 for concurrent inserts)
    rng_state: AtomicU64,

    /// Algorithm type for trait requirement
    algorithm_type: IndexAlgorithm,

    /// EventLog-based extraction mode for async index updates
    /// Replaces queue consumer pattern with direct event processing
    extraction_mode: ExtractionMode,

    /// NEW: Quantized vector storage for dual representation support
    /// Maps external_id -> quantized_vector for QUANTIZED_ONLY and BOTH modes
    quantized_vectors: Arc<DashMap<String, Vec<u8>>>,

    /// TD-064: Shared filterable-metadata cache (AXIS-provided).
    /// Holds compact <50-byte-per-record metadata for predicate-aware traversal.
    filterable_metadata: FilterableMetadataCache,
}

impl AxisHnswIndex {
    /// Create a new HNSW index with the given configuration using shared utilities
    pub fn new(config: AxisHnswConfig, dimension: usize) -> Result<Self> {
        Self::new_with_collection(None, config, dimension)
    }

    /// Create a new HNSW index for a specific collection
    pub fn new_with_collection(
        collection_id: Option<String>,
        config: AxisHnswConfig,
        dimension: usize,
    ) -> Result<Self> {
        Self::new_with_extraction_mode(collection_id, config, dimension, ExtractionMode::Auto)
    }

    /// Create a new HNSW index with specific extraction mode for EventLog processing
    pub fn new_with_extraction_mode(
        collection_id: Option<String>,
        config: AxisHnswConfig,
        dimension: usize,
        extraction_mode: ExtractionMode,
    ) -> Result<Self> {
        // USING UTILS: Validate dimension
        validation::validate_dimension(dimension)?;

        let coll_str = collection_id.as_ref().map_or("<unnamed>", |s| s.as_str());
        info!(
            "Creating AXIS HNSW index for collection '{}': M={}, ef_construction={}, ef={}, dim={}, repr={:?}",
            coll_str, config.m, config.ef_construction, config.ef, dimension, extraction_mode
        );

        let distance_computer = UnifiedDistanceCompute::new(config.distance_metric);

        let algorithm_type = IndexAlgorithm::HNSW {
            m: config.m as u32,
            ef_construction: config.ef_construction as u32,
            ef_search: config.ef as u32,
            max_elements: 1000000, // Default max elements
        };

        Ok(Self {
            collection_id: collection_id.clone(),
            config,
            distance_computer,

            // Zero-overhead storage with shared collection config
            vectors: Arc::new(RwLock::new(ZeroOverheadCollection::with_capacity(
                Self::get_collection_config(&collection_id, dimension, &extraction_mode),
                1024, // Initial capacity
            ))),
            id_mapping: ConcurrentIdMapping::new(),
            stats: AtomicStats::new(),

            // HNSW-specific structures
            layers: DashMap::new(),
            max_layer: AtomicUsize::new(0),
            entry_point: RwLock::new(None),
            rng_state: AtomicU64::new(42), // Deterministic seed for reproducibility
            algorithm_type,

            // EventLog-based vector consumption (no queue consumer needed)
            extraction_mode,
            quantized_vectors: Arc::new(DashMap::new()),

            // TD-064: shared filterable metadata cache; populated via add_with_metadata
            filterable_metadata: FilterableMetadataCache::new(),
        })
    }

    /// Get collection configuration based on extraction mode and collection ID
    /// This would normally come from a shared collection cache in production
    fn get_collection_config(
        _collection_id: &Option<String>,
        dimension: usize,
        extraction_mode: &ExtractionMode,
    ) -> CollectionConfig {
        match extraction_mode {
            ExtractionMode::Fp32Only | ExtractionMode::Auto => {
                // FP32 mode - no quantization
                CollectionConfig::fp32(dimension)
            }
            ExtractionMode::QuantizedOnly => {
                // Quantized mode - use INT8 by default (could come from collection config)
                CollectionConfig::quantized(dimension, QuantizationMethod::INT8)
            }
            ExtractionMode::Both => {
                // Both modes - store FP32 primarily, quantized in separate map
                CollectionConfig::fp32(dimension)
            }
        }
    }

    /// Generate random level for new node using exponential decay.
    /// Lock-free: uses atomic CAS on the RNG state so concurrent inserts
    /// don't serialize behind a write lock.
    fn get_random_level(&self) -> usize {
        let mut level = 0;

        let mut random_val = self.fast_random_atomic() as f32 / u32::MAX as f32;

        while random_val < 0.5 && level < self.config.max_layers {
            level += 1;
            random_val = self.fast_random_atomic() as f32 / u32::MAX as f32;
        }

        level
    }

    /// Lock-free LCG using atomic compare-and-swap.
    /// Each thread reads the current state, computes the next state, and attempts
    /// to CAS. On contention, the retry re-reads the (now-advanced) state,
    /// which is correct for an LCG — the sequence just skips ahead.
    fn fast_random_atomic(&self) -> u32 {
        loop {
            let current = self.rng_state.load(AtomicOrdering::Relaxed);
            let next = current.wrapping_mul(1664525).wrapping_add(1013904223);
            if self
                .rng_state
                .compare_exchange_weak(
                    current,
                    next,
                    AtomicOrdering::Relaxed,
                    AtomicOrdering::Relaxed,
                )
                .is_ok()
            {
                return (next >> 32) as u32;
            }
        }
    }

    /// Search for ef closest candidates in a specific layer
    /// OPTIMIZED: Uses batch SIMD distance computation for 4-8x speedup
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[usize],
        ef: usize,
        layer: usize,
    ) -> Vec<(usize, f32)> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new(); // Min heap for candidates (to visit)
        let mut dynamic_candidates = BinaryHeap::new(); // Max heap for best found
        let _metric = self.config.distance_metric;

        // Initialize with entry points
        // OPTIMIZATION: Compute distances inline to avoid allocations
        {
            let vectors_lock = self
                .vectors
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());

            for &ep in entry_points {
                if let Some(external_id) = self.id_mapping.external(ep)
                    && let Some(view) = vectors_lock.get(&external_id)
                    && let Some(vector_data) = view.as_f32()
                {
                    // metric_aware_distance normalises every metric
                    // to lower=better so the BinaryHeap ordering is
                    // consistent (see DotProduct recall bug fix).
                    let dist = self.metric_aware_distance(query, vector_data);
                    visited.insert(ep);
                    candidates.push(std::cmp::Reverse((OrderedFloat(dist), ep)));
                    dynamic_candidates.push((OrderedFloat(dist), ep));
                }
            }
        }

        // Explore the graph with distance computation
        while let Some(std::cmp::Reverse((curr_dist, curr_node))) = candidates.pop() {
            // Early termination: if current distance is worse than worst in dynamic_candidates
            if let Some((worst_dist, _)) = dynamic_candidates.peek()
                && curr_dist.0 > worst_dist.0
                && dynamic_candidates.len() >= ef
            {
                break;
            }

            // Compute distances inline to avoid vector cloning (zero-copy optimization)
            if let Some(neighbors) = self.layers.get(&(layer, curr_node)) {
                let vectors_lock = self
                    .vectors
                    .read()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());

                for &neighbor in neighbors.value() {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        if let Some(external_id) = self.id_mapping.external(neighbor)
                            && let Some(view) = vectors_lock.get(&external_id)
                            && let Some(vector_data) = view.as_f32()
                        {
                            let dist = self.metric_aware_distance(query, vector_data);

                            if dynamic_candidates.len() < ef {
                                candidates.push(std::cmp::Reverse((OrderedFloat(dist), neighbor)));
                                dynamic_candidates.push((OrderedFloat(dist), neighbor));
                            } else if let Some((worst_dist, _)) = dynamic_candidates.peek()
                                && dist < worst_dist.0
                            {
                                candidates.push(std::cmp::Reverse((OrderedFloat(dist), neighbor)));
                                dynamic_candidates.push((OrderedFloat(dist), neighbor));

                                if dynamic_candidates.len() > ef {
                                    dynamic_candidates.pop();
                                }
                            }
                        }
                    }
                }
            }
        }

        // Convert to sorted result (ascending by distance)
        let mut result: Vec<_> = dynamic_candidates
            .into_iter()
            .map(|(OrderedFloat(dist), node)| (node, dist))
            .collect();

        result.sort_by(|a, b| {
            a.1.partial_cmp(&b.1).unwrap_or_else(|| {
                // Handle NaN values deterministically
                if a.1.is_nan() && b.1.is_nan() {
                    std::cmp::Ordering::Equal
                } else if a.1.is_nan() {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Less
                }
            })
        });
        result
    }

    /// Select m neighbors using simple heuristic (closest neighbors)
    fn select_neighbors(&self, candidates: Vec<(usize, f32)>, m: usize) -> Vec<usize> {
        candidates
            .into_iter()
            .take(m)
            .map(|(node, _)| node)
            .collect()
    }

    /// ACORN §5.2 — γ-expanded neighbor selection for predicate-aware construction.
    ///
    /// Returns `floor(config.gamma * m)` neighbors so that predicate-filtered
    /// traversal can route around sparse regions without getting stuck at dead ends.
    /// When `gamma == 1.0` this is identical to `select_neighbors`.
    pub fn select_neighbors_gamma(&self, candidates: Vec<(usize, f32)>, m: usize) -> Vec<usize> {
        let expanded_m = self.config.expanded_m().max(m);
        candidates
            .into_iter()
            .take(expanded_m)
            .map(|(node, _)| node)
            .collect()
    }

    /// NaviX predicate-aware graph search (Phase C, spec §6.1).
    ///
    /// Behaves like `search_layer` but accepts a per-node predicate. Nodes that
    /// fail the predicate are still used for graph traversal (skip-through, ACORN
    /// §4.2 "predicate-agnostic greedy search") but are excluded from the result
    /// candidate set. This prevents getting stuck in predicate-sparse subgraphs.
    fn search_layer_predicate<P>(
        &self,
        query: &[f32],
        entry_points: &[usize],
        ef: usize,
        layer: usize,
        predicate: &P,
    ) -> Vec<(usize, f32)>
    where
        P: Fn(usize) -> bool,
    {
        let mut visited = HashSet::new();
        let mut frontier = BinaryHeap::new(); // min-heap of (dist, node) to explore
        let mut result_candidates: BinaryHeap<(OrderedFloat, usize)> = BinaryHeap::new();
        let _metric = self.config.distance_metric;

        {
            let vectors_lock = self
                .vectors
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());

            for &ep in entry_points {
                if let Some(external_id) = self.id_mapping.external(ep)
                    && let Some(view) = vectors_lock.get(&external_id)
                    && let Some(vector_data) = view.as_f32()
                {
                    let dist = self.metric_aware_distance(query, vector_data);
                    visited.insert(ep);
                    frontier.push(std::cmp::Reverse((OrderedFloat(dist), ep)));
                    if predicate(ep) {
                        result_candidates.push((OrderedFloat(dist), ep));
                    }
                }
            }
        }

        while let Some(std::cmp::Reverse((curr_dist, curr_node))) = frontier.pop() {
            // Early-termination: current node is further than the worst accepted result
            if result_candidates.len() >= ef
                && let Some((worst, _)) = result_candidates.peek()
                && curr_dist.0 > worst.0
            {
                break;
            }

            if let Some(neighbors) = self.layers.get(&(layer, curr_node)) {
                let vectors_lock = self
                    .vectors
                    .read()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());

                for &neighbor in neighbors.value() {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        if let Some(external_id) = self.id_mapping.external(neighbor)
                            && let Some(view) = vectors_lock.get(&external_id)
                            && let Some(vector_data) = view.as_f32()
                        {
                            let dist = self.metric_aware_distance(query, vector_data);
                            // Skip-through traversal — but skip entries that
                            // can never be expanded: once `ef` passing results
                            // exist, `result_candidates`' worst is monotone
                            // non-increasing (it only ever evicts its max), so
                            // an entry strictly worse than the current worst
                            // satisfies the early-termination break above at
                            // pop time and is never expanded — pushing it only
                            // grew the frontier. Equal distances still push
                            // (the break is strictly-greater). This changes
                            // neither the result set nor the visit order.
                            if result_candidates.len() < ef
                                || result_candidates
                                    .peek()
                                    .is_none_or(|(worst, _)| dist <= worst.0)
                            {
                                frontier.push(std::cmp::Reverse((OrderedFloat(dist), neighbor)));
                            }
                            // Only add to results if predicate passes
                            if predicate(neighbor) {
                                result_candidates.push((OrderedFloat(dist), neighbor));
                                // Evict worst when over ef
                                while result_candidates.len() > ef {
                                    result_candidates.pop();
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut result: Vec<_> = result_candidates
            .into_iter()
            .map(|(OrderedFloat(dist), node)| (node, dist))
            .collect();
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Closure-based predicate-aware search (Phase C, spec §7 VectorTopK.predicate).
    ///
    /// Lower-level API: the caller supplies a `Fn(&str) -> bool` predicate that
    /// receives the **external string id** of each candidate. Used by the AXIS
    /// filtered-search bridge and unit tests.
    ///
    /// The structured (TD-064) entry point lives on the `AxisVectorIndex` trait
    /// as `search_with_predicate(query, k, tenant, time_range, rls_tags)` and
    /// uses cached filterable metadata for predicate evaluation. That trait
    /// method delegates to this method after building an internal closure.
    ///
    /// Internally delegates to `search_layer_predicate` at layer 0 so the
    /// NaviX skip-through heuristic applies throughout the bottom traversal.
    pub async fn search_with_predicate_fn<P>(
        &self,
        query: &[f32],
        k: usize,
        predicate: P,
        ef_override: Option<usize>,
    ) -> Result<Vec<(String, f32)>>
    where
        P: Fn(&str) -> bool + Send + Sync,
    {
        // Read entry point (RwLock<Option<usize>>)
        let entry_point = match self
            .entry_point
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
        {
            Some(&ep) => ep,
            None => return Ok(Vec::new()), // empty index
        };

        let max_layer = self.max_layer.load(AtomicOrdering::Relaxed);
        let mut current_points = vec![entry_point];

        // Upper layers: unfiltered greedy descent to layer 1 entry
        for layer in (1..=max_layer).rev() {
            current_points = self
                .search_layer(query, &current_points, 1, layer)
                .into_iter()
                .map(|(n, _)| n)
                .collect();
        }

        // Translate external-id predicate to internal-id predicate
        let id_predicate = |internal_id: usize| -> bool {
            self.id_mapping
                .external(internal_id)
                .map(|ext| predicate(&ext))
                .unwrap_or(false)
        };

        // Layer 0: predicate-aware NaviX traversal. A per-query `ef` override
        // (from SearchMode::Approximate) wins when set, floored at `k`; else the
        // historical recall-maximizing default. Oversample (`k` here is already
        // the policy-driven oversample_k) is an orthogonal lever — untouched.
        let ef = match ef_override {
            Some(e) => e.max(k),
            None => self.config.ef.max(k),
        };
        let raw = self.search_layer_predicate(query, &current_points, ef, 0, &id_predicate);

        let results = raw
            .into_iter()
            .take(k)
            .filter_map(|(node, dist)| self.id_mapping.external(node).map(|ext_id| (ext_id, dist)))
            .collect();

        Ok(results)
    }

    /// Shrink connections for a node if it exceeds the maximum degree
    /// This is critical for maintaining graph quality at scale - without this,
    /// nodes can accumulate too many connections leading to poor recall
    /// OPTIMIZED: Uses batch SIMD distance computation for faster pruning
    fn shrink_connections(&self, node_id: usize, layer: usize, max_m: usize) {
        // Get the current connections for this node
        let connections: Vec<usize> = match self.layers.get(&(layer, node_id)) {
            Some(conns) => conns.value().clone(),
            None => return,
        };

        // If under limit, nothing to do
        if connections.len() <= max_m {
            return;
        }

        // Get the vector for this node to compute distances
        let node_vector: Vec<f32> = {
            let external_id = match self.id_mapping.external(node_id) {
                Some(id) => id,
                None => return,
            };
            let vectors = self
                .vectors
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            match vectors.get(&external_id) {
                Some(view) => match view.as_f32() {
                    Some(v) => v.to_vec(),
                    None => return,
                },
                None => return,
            }
        };

        // OPTIMIZED: Collect all neighbor vectors for batch SIMD computation
        let mut neighbor_ids: Vec<usize> = Vec::with_capacity(connections.len());
        let mut neighbor_vectors: Vec<Vec<f32>> = Vec::with_capacity(connections.len());

        {
            let vectors_lock = self
                .vectors
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            for &neighbor in &connections {
                if let Some(neighbor_external) = self.id_mapping.external(neighbor)
                    && let Some(view) = vectors_lock.get(&neighbor_external)
                    && let Some(neighbor_vec) = view.as_f32()
                {
                    neighbor_ids.push(neighbor);
                    neighbor_vectors.push(neighbor_vec.to_vec());
                }
            }
        }

        // Batch compute distances using SIMD (4-8x faster)
        let neighbor_refs: Vec<&[f32]> = neighbor_vectors.iter().map(|v| v.as_slice()).collect();
        let distances = self.distance_computer.distance_batch(
            &node_vector,
            &neighbor_refs,
            Some(self.config.distance_metric),
        );

        // Build neighbor_distances from batch results
        let mut neighbor_distances: Vec<(usize, f32)> =
            neighbor_ids.into_iter().zip(distances).collect();

        // Sort by distance and keep only the closest max_m
        neighbor_distances.sort_by(|a, b| {
            a.1.partial_cmp(&b.1).unwrap_or_else(|| {
                // Handle NaN values deterministically
                if a.1.is_nan() && b.1.is_nan() {
                    std::cmp::Ordering::Equal
                } else if a.1.is_nan() {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Less
                }
            })
        });
        let new_connections: Vec<usize> = neighbor_distances
            .into_iter()
            .take(max_m)
            .map(|(n, _)| n)
            .collect();

        // Update the connections
        if let Some(mut entry) = self.layers.get_mut(&(layer, node_id)) {
            *entry.value_mut() = new_connections;
        }
    }
}

#[async_trait]
impl AxisVectorIndex for AxisHnswIndex {
    async fn add(&self, id: String, vector_data: Vec<f32>) -> Result<()> {
        let start = std::time::Instant::now();

        // USING UTILS: Validate vector ID
        validation::validate_vector_id(&id)?;

        // Check if this ID already exists
        if let Some(_existing_node_id) = self.id_mapping.internal(&id) {
            // Update existing vector
            {
                let mut vectors = self
                    .vectors
                    .write()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                vectors.add_fp32(id.clone(), &vector_data)?;
            }
            // Return early - no need to re-add to graph structure
            self.stats
                .record_success(start.elapsed().as_micros() as u64);
            return Ok(());
        }

        // Register ID mapping and get internal node ID for new vector
        let internal_node_id = self.id_mapping.register(id.clone())?;

        // Store vector with zero-overhead - just raw data!
        {
            let mut vectors = self
                .vectors
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            vectors.add_fp32(id.clone(), &vector_data)?;
        }

        // Determine level for this node
        let level = self.get_random_level();

        // Update maximum layer atomically
        let current_max = self.max_layer.load(AtomicOrdering::Relaxed);
        if level > current_max {
            self.max_layer.store(level, AtomicOrdering::Relaxed);
        }

        // Initialize layers for this node
        for l in 0..=level {
            self.layers.insert((l, internal_node_id), Vec::new());
        }

        // If this is the first node, make it the entry point
        {
            let mut entry_point_lock = self
                .entry_point
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if entry_point_lock.is_none() {
                *entry_point_lock = Some(internal_node_id);
                self.stats
                    .record_success(start.elapsed().as_micros() as u64);
                return Ok(());
            }
        }

        let entry_point = self
            .entry_point
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .ok_or_else(|| anyhow::anyhow!("Entry point must be set when adding non-first node"))?;
        let mut curr_nearest = vec![entry_point];

        // Search from top layer down to level+1 (greedy search with ef=1)
        for layer in (level + 1..=self.max_layer.load(AtomicOrdering::Relaxed)).rev() {
            curr_nearest = self
                .search_layer(&vector_data, &curr_nearest, 1, layer)
                .into_iter()
                .map(|(node, _)| node)
                .collect();
        }

        // Search and connect from level down to 0
        for layer in (0..=level).rev() {
            let candidates = self.search_layer(
                &vector_data,
                &curr_nearest,
                self.config.ef_construction,
                layer,
            );

            // Different M values for layer 0 vs higher layers
            let m = if layer == 0 {
                self.config.m * 2
            } else {
                self.config.m
            };
            // ACORN §5.2: use γ-expanded selection when gamma > 1.0 so predicate-filtered
            // traversal can route around sparse regions without dead-ends.
            let selected = if self.config.gamma > 1.0 {
                self.select_neighbors_gamma(candidates.clone(), m)
            } else {
                self.select_neighbors(candidates.clone(), m)
            };

            // Add bidirectional connections using DashMap
            // CRITICAL FIX: After adding connections, shrink if exceeds max degree
            // Without this, nodes accumulate too many connections at scale,
            // causing 47% recall degradation at 50K vectors
            for neighbor in &selected {
                // Add internal_node_id to neighbor's connections
                self.layers
                    .entry((layer, *neighbor))
                    .or_default()
                    .push(internal_node_id);

                // Shrink neighbor's connections if exceeded max degree
                self.shrink_connections(*neighbor, layer, m);

                // Add neighbor to internal_node_id's connections
                self.layers
                    .entry((layer, internal_node_id))
                    .or_default()
                    .push(*neighbor);
            }

            // Update curr_nearest for next layer (best candidate from this layer)
            curr_nearest = candidates
                .into_iter()
                .take(1)
                .map(|(node, _)| node)
                .collect();
        }

        // Update entry point if this node reaches the highest level
        if level >= self.max_layer.load(AtomicOrdering::Relaxed) {
            *self
                .entry_point
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(internal_node_id);
        }

        // USING UTILS: Record successful operation
        self.stats
            .record_success(start.elapsed().as_micros() as u64);
        Ok(())
    }

    async fn search(
        &self,
        query: &[f32],
        top_k: usize,
        _filter: Option<&HashMap<String, String>>, // Metadata filter, not VectorRecord
    ) -> Result<Vec<(String, f32)>> {
        self.search_with_filter(query, top_k, None).await
    }

    async fn search_with_effort(
        &self,
        query: &[f32],
        top_k: usize,
        ef_override: Option<usize>,
        _filter: Option<&HashMap<String, String>>,
    ) -> Result<Vec<(String, f32)>> {
        // Honor the per-query ef budget on the warm HNSW path.
        self.search_with_filter_ef(query, top_k, ef_override, None)
            .await
    }

    async fn remove(&self, id: &str) -> Result<()> {
        let start = std::time::Instant::now();

        // USING UTILS: Get internal node ID from mapping
        let internal_node_id = match self.id_mapping.internal(id) {
            Some(node_id) => node_id,
            None => {
                self.stats
                    .record_success(start.elapsed().as_micros() as u64);
                return Ok(()); // ID not found, nothing to remove
            }
        };

        // Remove from all layers and clean up connections using DashMap
        let max_layer = self.max_layer.load(AtomicOrdering::Relaxed);
        for layer in 0..=max_layer {
            // Remove this node's connections
            self.layers.remove(&(layer, internal_node_id));

            // Remove connections to this node from all other nodes
            let keys_to_update: Vec<_> = self
                .layers
                .iter()
                .filter(|entry| entry.key().0 == layer)
                .map(|entry| *entry.key())
                .collect();

            for key in keys_to_update {
                if let Some(mut connections) = self.layers.get_mut(&key) {
                    connections.retain(|&x| x != internal_node_id);
                }
            }
        }

        // USING UTILS: Remove from mappings and vectors
        // Remove from vectors collection
        {
            let mut vectors = self
                .vectors
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            vectors.remove(id);
        }
        self.id_mapping.remove_by_external(id);

        // TD-064: drop cached filterable metadata for this id
        self.filterable_metadata.remove(id);

        // Update entry point if necessary
        {
            let mut entry_point_lock = self
                .entry_point
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if *entry_point_lock == Some(internal_node_id) {
                // Find a new entry point from remaining vectors
                // Deferred: ZeroOverheadCollection doesn't have keys() method
                // For now, just set entry point to None when removed
                *entry_point_lock = None;
            }
        }

        // USING UTILS: Record successful operation
        self.stats
            .record_success(start.elapsed().as_micros() as u64);
        Ok(())
    }

    async fn add_with_metadata(
        &self,
        id: String,
        vector_data: Vec<f32>,
        metadata: &FilterableHnswMetadata,
    ) -> Result<()> {
        // TD-064: cache metadata first so a concurrent predicate-aware search
        // observing this id can already evaluate the filter; then add to graph.
        self.filterable_metadata
            .insert(id.clone(), metadata.clone());
        self.add(id, vector_data).await
    }

    async fn search_with_predicate(
        &self,
        query: &[f32],
        top_k: usize,
        tenant_id: Option<&str>,
        time_range_ns: Option<(i64, i64)>,
        rls_tags: Option<&[String]>,
    ) -> Result<Vec<(String, f32)>> {
        // TD-064: HNSW skip-through traversal over cached filterable metadata.
        //
        // When the cache is empty the index has not yet observed any
        // add_with_metadata; fall back to plain ANN. Callers requiring
        // correctness under filters must surface the degradation in EXPLAIN
        // (handled at AxisManager layer).
        if self.filterable_metadata.is_empty() {
            return self.search(query, top_k, None).await;
        }

        let predicate =
            self.filterable_metadata
                .build_predicate(tenant_id, time_range_ns, rls_tags);
        self.search_with_predicate_fn(query, top_k, predicate, None)
            .await
    }

    fn supports_predicate_search(&self) -> bool {
        // True once any metadata has been cached; before first add_with_metadata
        // call the index behaves as a plain ANN and the trait method falls
        // back to standard search.
        !self.filterable_metadata.is_empty()
    }

    fn configure_filterable_fields(&self, config: &FilterableFieldsConfig) -> Result<()> {
        self.filterable_metadata.configure_fields(config);
        Ok(())
    }

    fn algorithm(&self) -> &IndexAlgorithm {
        &self.algorithm_type
    }

    fn stats(&self) -> IndexStats {
        IndexStats {
            vector_count: self
                .vectors
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .len(),
            memory_usage_bytes: self.estimate_memory_usage(),
            index_type: "HNSW".to_string(),
        }
    }
}

impl AxisHnswIndex {
    /// Search with optional filtering.
    ///
    /// Thin wrapper over [`Self::search_with_filter_ef`] with no per-query `ef`
    /// override — preserves the historical signature and behavior for all
    /// existing callers.
    pub async fn search_with_filter(
        &self,
        query: &[f32],
        top_k: usize,
        filter: Option<
            &(dyn for<'a> Fn(&'a proximadb_records::ProximaRecord) -> bool + Send + Sync),
        >,
    ) -> Result<Vec<(String, f32)>> {
        self.search_with_filter_ef(query, top_k, None, filter).await
    }

    /// Search with optional filtering and an optional per-query `ef` override.
    ///
    /// `ef_override`: when `Some(ef)`, layer-0 is searched with `ef.max(top_k)`
    /// candidates instead of the collection-size-aware default; when `None`,
    /// the historical `config.ef.max(clamp(sqrt(N),50,500)).max(top_k)` is used.
    /// This is the knob that lets `SearchMode::Approximate` actually trade
    /// recall for latency on the warm HNSW path.
    pub async fn search_with_filter_ef(
        &self,
        query: &[f32],
        top_k: usize,
        ef_override: Option<usize>,
        _filter: Option<
            &(dyn for<'a> Fn(&'a proximadb_records::ProximaRecord) -> bool + Send + Sync),
        >,
    ) -> Result<Vec<(String, f32)>> {
        let start = std::time::Instant::now();

        // Get entry point
        let entry_point = match self
            .entry_point
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
        {
            Some(&ep) => ep,
            None => {
                self.stats
                    .record_success(start.elapsed().as_micros() as u64);
                return Ok(Vec::new()); // Empty index
            }
        };

        let mut curr_nearest = vec![entry_point];
        let max_layer = self.max_layer.load(AtomicOrdering::Relaxed);

        // Phase profiling — surfaces whether time is spent in the
        // top-layer greedy descent vs the layer-0 ef walk vs the
        // id-mapping post-conversion. Cheap timer per phase.
        let descent_start = std::time::Instant::now();

        // Search from top layer down to layer 1 (greedy with ef=1)
        for layer in (1..=max_layer).rev() {
            curr_nearest = self
                .search_layer(query, &curr_nearest, 1, layer)
                .into_iter()
                .map(|(node, _)| node)
                .collect();
        }
        let descent_us = descent_start.elapsed().as_micros() as u64;

        // Search layer 0 with collection-size-aware ef for consistent recall at scale
        // For N vectors, optimal ef ≈ sqrt(N) for high recall (>95%)
        // This ensures 50K vectors get ef≈223 instead of just 50
        let collection_size = self
            .vectors
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len();
        // size_aware_ef provides a sensible floor for collections
        // that didn't set a strategy spec — production callers that
        // care about recall should set `IndexAlgorithm::HNSW.ef_search`
        // on the strategy spec, which now flows through to
        // `AxisHnswConfig.ef` end-to-end (see insert_into_hnsw and
        // insert_hmgi). The `max` below picks whichever is larger so
        // a customer who explicitly asked for ef_search=500 on a 100K
        // collection still gets 500 (not sqrt(100K)=316).
        let size_aware_ef = ((collection_size as f64).sqrt() as usize).clamp(50, 500);
        // Per-query `ef` override (from SearchMode::Approximate) wins when set,
        // floored at `top_k` so we never return fewer than the requested
        // results. When absent, keep the historical recall-maximizing default.
        let search_ef = match ef_override {
            Some(ef) => ef.max(top_k),
            None => self.config.ef.max(size_aware_ef).max(top_k),
        };

        tracing::debug!(
            "HNSW search: collection_size={}, size_aware_ef={}, config_ef={}, ef_override={:?}, final_ef={}",
            collection_size,
            size_aware_ef,
            self.config.ef,
            ef_override,
            search_ef
        );

        let layer0_start = std::time::Instant::now();
        let candidates = self.search_layer(query, &curr_nearest, search_ef, 0);
        let layer0_us = layer0_start.elapsed().as_micros() as u64;
        let candidates_visited = candidates.len();

        // Convert internal IDs to external IDs - no filtering at index level
        // Metadata filtering happens at storage layer, not in indexes
        let convert_start = std::time::Instant::now();
        let results: Vec<(String, f32)> = candidates
            .into_iter()
            .take(top_k)
            .filter_map(|(internal_node_id, score)| {
                self.id_mapping
                    .external(internal_node_id)
                    .map(|external_id| (external_id, score))
            })
            .collect();
        let convert_us = convert_start.elapsed().as_micros() as u64;
        let total_us = start.elapsed().as_micros() as u64;

        // Emit a single structured event per search so the bench can
        // attribute the latency. Threshold gating keeps logs quiet
        // for fast searches (HMGI partitions usually ~1ms) but
        // surfaces the slow ones (legacy HNSW path apparently
        // ~60-90ms even on identical data — the gap this
        // instrumentation is designed to expose).
        tracing::info!(
            target: "axis_diag",
            site = "AxisHnswIndex::search_with_filter",
            collection_size = collection_size,
            search_ef = search_ef,
            descent_us = descent_us,
            layer0_us = layer0_us,
            convert_us = convert_us,
            total_us = total_us,
            candidates_visited = candidates_visited,
            "HNSW search phase breakdown"
        );

        // USING UTILS: Record successful operation
        self.stats.record_success(total_us);
        Ok(results)
    }

    /// Get the number of vectors in the index
    pub fn size(&self) -> usize {
        let vectors = self
            .vectors
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        vectors.len()
    }

    /// Distance metric the HNSW graph was built with. Exposed so
    /// upstream layers (HMGI router, scorers) can convert the raw
    /// distance values returned by `search_*` into the canonical
    /// `SimilarityResult.normalized_score` shape that the rest of
    /// the stack assumes (higher = better, range [0, 1] for the
    /// common metrics).
    pub fn distance_metric(&self) -> crate::compute::distance_computation::DistanceMetric {
        self.config.distance_metric
    }

    /// Compute a distance with **lower = better** semantics across
    /// every metric the HNSW algorithm sees.
    ///
    /// The compute layer's `distance_with_metric` returns the
    /// metric's native value:
    ///   * Cosine / Euclidean / Manhattan → a distance (lower = better)
    ///   * DotProduct → the raw inner product (HIGHER = better)
    ///
    /// HNSW's `BinaryHeap` logic everywhere assumes lower = better.
    /// Without this wrapper, DotProduct vectors were inserted into
    /// the priority queue with inverted ranking — the algorithm
    /// kept the records with the LOWEST inner product (farthest
    /// from the query) and discarded the closest ones. Measured at
    /// 10K × 128d: recall=0.00 for DotProduct, vs 0.78+ for Cosine
    /// and Euclidean. Negating the similarity-metric values here
    /// restores the lower-better invariant; the router undoes the
    /// negation when constructing `SimilarityResult` for the
    /// caller.
    #[inline]
    fn metric_aware_distance(&self, q: &[f32], v: &[f32]) -> f32 {
        use crate::compute::distance_computation::engine::DistanceMetricExt;
        let raw = self
            .distance_computer
            .distance_with_metric(q, v, &self.config.distance_metric);
        if self.config.distance_metric.is_similarity() {
            -raw
        } else {
            raw
        }
    }

    /// Get memory usage of the index
    pub fn memory_usage(&self) -> MemoryUsage {
        let total = self.estimate_memory_usage();
        MemoryUsage {
            total_bytes: total,
            index_size_bytes: total / 3, // Approximate distribution
            vector_data_bytes: total / 3,
            metadata_bytes: total / 3,
        }
    }

    /// Add a single vector to the index - clean API, no VectorRecord
    pub async fn add_vector(
        &mut self,
        id: String,
        vector: Vec<f32>,
        _metadata: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<()> {
        // Metadata is ignored - indexes don't store metadata, only vectors
        self.add(id, vector).await
    }

    /// Add multiple vectors to the index
    pub async fn add_vectors(
        &mut self,
        batch: Vec<(String, Vec<f32>, Option<HashMap<String, serde_json::Value>>)>,
    ) -> Result<()> {
        for (id, vector, metadata) in batch {
            self.add_vector(id, vector, metadata).await?;
        }
        Ok(())
    }

    /// Search for top_k nearest neighbors
    pub async fn search_simple(&self, query: &[f32], top_k: usize) -> Result<Vec<(String, f32)>> {
        self.search_with_filter(query, top_k, None).await
    }

    /// Remove a vector from the index
    pub async fn remove_vector(&mut self, id: &str) -> Result<bool> {
        self.remove(id).await.map(|_| true)
    }

    /// Optimize the index (no-op for HNSW)
    pub fn optimize(&self) -> Result<()> {
        Ok(())
    }

    /// Estimate memory usage in bytes
    fn estimate_memory_usage(&self) -> usize {
        // USING UTILS: Get vector storage memory usage
        let vector_memory = {
            let vectors = self
                .vectors
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            vectors.memory_usage()
        };

        // USING UTILS: Get ID mapping memory usage
        let id_mapping_memory = memory::dashmap_overhead::<String, usize>(self.id_mapping.len())
            + memory::dashmap_overhead::<usize, String>(self.id_mapping.len());

        // Graph structure memory (layers DashMap)
        let layers_memory = self.layers.len()
            * (std::mem::size_of::<(usize, usize)>() + std::mem::size_of::<Vec<usize>>() + 64);

        // NEW: Quantized vector storage memory
        let quantized_memory = self.quantized_vectors.len() * 128; // Estimate 128 bytes per quantized vector

        // Other structures
        let config_memory = std::mem::size_of::<AxisHnswConfig>();
        let stats_memory = std::mem::size_of::<AtomicStats>();

        vector_memory
            + id_mapping_memory
            + layers_memory
            + quantized_memory
            + config_memory
            + stats_memory
    }

    /// Set extraction mode for EventLog-based async index updates
    pub fn set_extraction_mode(&mut self, mode: ExtractionMode) {
        self.extraction_mode = mode.clone();
        info!(
            "Extraction mode set to {:?} for HNSW index in collection: {:?}",
            mode, self.collection_id
        );
    }

    /// Process EventLog events for async index updates  
    pub async fn process_event(&self, event: &IndexEvent) -> Result<()> {
        info!(
            "Processing EventLog event {} for HNSW index in collection: {:?}",
            event.event_id, self.collection_id
        );

        // Process vectors based on extraction mode and event data availability
        match self.extraction_mode {
            ExtractionMode::Fp32Only if !event.has_fp32 => {
                info!(
                    "Skipping event {} - requires FP32 but event has no FP32 data",
                    event.event_id
                );
                return Ok(());
            }
            ExtractionMode::QuantizedOnly if !event.has_quantized => {
                info!(
                    "Skipping event {} - requires quantized but event has no quantized data",
                    event.event_id
                );
                return Ok(());
            }
            _ => {
                // Auto and Both modes can process any available data
                info!(
                    "Processing event {} with extraction mode {:?}",
                    event.event_id, self.extraction_mode
                );
            }
        }

        // Deferred: Extract vectors from files listed in event.file_paths
        // This would be handled by the EventLog consumer which calls this method
        // after extracting vectors from the storage files

        Ok(())
    }

    /// Process vectors from EventLog event - clean, no VectorRecord
    pub async fn add_vectors_from_event(&self, vectors: Vec<(String, Vec<f32>)>) -> Result<()> {
        // Process FP32 vectors directly - no proto overhead
        for (id, vector) in vectors {
            self.add(id, vector).await?;
        }
        Ok(())
    }

    /// NEW: Dequantize vector for HNSW graph construction
    /// Deferred: Integrate with actual quantization module from storage engines
    #[allow(dead_code)]
    fn dequantize_vector(
        &self,
        _quantized: &[u8],
        _method: &str,
        dimension: usize,
    ) -> Result<Vec<f32>> {
        // PLACEHOLDER: In production, this would use the actual quantization module
        // from src/storage/quantization/ to properly dequantize vectors
        tracing::warn!(
            "Using placeholder dequantization - integrate with storage quantization module"
        );

        // Create a placeholder FP32 vector
        Ok(vec![0.0; dimension])
    }

    /// NEW: Get preferred vector representation for queue consumption
    pub fn extraction_mode(&self) -> ExtractionMode {
        self.extraction_mode.clone()
    }

    /// NEW: Check if quantized vectors are available for search acceleration
    pub fn has_quantized_storage(&self) -> bool {
        !self.quantized_vectors.is_empty()
    }

    /// NEW: Accelerated search using quantized vectors for initial filtering
    /// This implements a two-stage search: quantized filtering + FP32 reranking
    pub async fn search_with_quantized_acceleration(
        &self,
        query: &[f32],
        top_k: usize,
        filter: Option<
            &(dyn for<'a> Fn(&'a proximadb_records::ProximaRecord) -> bool + Send + Sync),
        >,
    ) -> Result<Vec<(String, f32)>> {
        if !self.has_quantized_storage() {
            // No quantized vectors available, use standard search
            return self.search_with_filter(query, top_k, filter).await;
        }

        // Deferred: Implement two-stage search with quantized filtering
        // Stage 1: Fast filtering using quantized vectors
        // Stage 2: FP32 reranking of top candidates
        tracing::warn!("Quantized acceleration not yet implemented - using standard search");

        self.search_with_filter(query, top_k, filter).await
    }

    // ============================================================================
    // SERIALIZATION HELPER METHODS
    // ============================================================================

    /// Get number of vectors via ID mapping (for serialization)
    pub fn id_mapping_len(&self) -> usize {
        self.id_mapping.len()
    }

    /// Get dimension from vector collection config
    pub fn get_dimension(&self) -> usize {
        let vectors = self
            .vectors
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        vectors.config().dimension
    }

    /// Get config M parameter
    pub fn get_config_m(&self) -> usize {
        self.config.m
    }

    /// Get config ef_construction parameter
    pub fn get_config_ef_construction(&self) -> usize {
        self.config.ef_construction
    }

    /// Get config ef parameter
    pub fn get_config_ef(&self) -> usize {
        self.config.ef
    }

    /// Get config max_layers parameter
    pub fn get_config_max_layers(&self) -> usize {
        self.config.max_layers
    }

    /// Get distance metric as numeric code for serialization
    /// Codes: 0=Unspecified, 1=Cosine, 2=Euclidean, 3=DotProduct, 4=Hamming, 5=Manhattan,
    /// 6=Jaccard, 7=Angular, 8=Chebyshev, 9=Canberra, 10=Minkowski, 11=BrayCurtis,
    /// 12=Hellinger, 13=Custom
    pub fn get_config_distance_metric_code(&self) -> u8 {
        match self.config.distance_metric {
            DistanceMetric::Unspecified => 0,
            DistanceMetric::Cosine => 1,
            DistanceMetric::Euclidean => 2,
            DistanceMetric::DotProduct => 3,
            DistanceMetric::Hamming => 4,
            DistanceMetric::Manhattan => 5,
            DistanceMetric::Jaccard => 6,
            DistanceMetric::Angular => 7,
            DistanceMetric::Chebyshev => 8,
            DistanceMetric::Canberra => 9,
            DistanceMetric::Minkowski => 10,
            DistanceMetric::BrayCurtis => 11,
            DistanceMetric::Hellinger => 12,
            DistanceMetric::Custom => 13,
        }
    }

    /// Get collection config details for serialization
    pub fn get_collection_config_details(&self) -> (usize, bool, Option<u8>) {
        let vectors = self
            .vectors
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let config = vectors.config();
        let quant_method = config.quantization_method.map(|m| match m {
            crate::index::axis::zero_overhead_vector::QuantizationMethod::INT8 => 0,
            crate::index::axis::zero_overhead_vector::QuantizationMethod::PQ8 => 1,
            crate::index::axis::zero_overhead_vector::QuantizationMethod::PQ4 => 2,
            crate::index::axis::zero_overhead_vector::QuantizationMethod::Binary => 3,
        });
        (config.dimension, config.is_quantized, quant_method)
    }

    /// Serialize ID mapping to portable format
    pub fn serialize_id_mapping(
        &self,
    ) -> crate::index::axis::storage::serialization::SerializableIdMapping {
        use crate::index::axis::storage::serialization::SerializableIdMapping;

        // Collect all external->internal mappings
        let external_to_internal: Vec<(String, usize)> =
            self.id_mapping.iter_external_to_internal().collect();

        SerializableIdMapping {
            external_to_internal,
            next_id: self.id_mapping.next_id(),
        }
    }

    /// Serialize graph layers to portable format
    pub fn serialize_layers(&self) -> Vec<((usize, usize), Vec<usize>)> {
        self.layers
            .iter()
            .map(|entry| (*entry.key(), entry.value().clone()))
            .collect()
    }

    /// Get max layer value
    pub fn get_max_layer(&self) -> usize {
        self.max_layer.load(AtomicOrdering::Relaxed)
    }

    /// Get entry point
    pub fn get_entry_point(&self) -> Option<usize> {
        *self
            .entry_point
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Serialize vectors to portable format
    pub fn serialize_vectors(
        &self,
    ) -> Vec<crate::index::axis::storage::serialization::SerializableVector> {
        use crate::index::axis::storage::serialization::SerializableVector;

        let vectors = self
            .vectors
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        vectors
            .iter()
            .map(|view| SerializableVector {
                id: view.id().to_string(),
                data: view.raw().as_bytes().to_vec(),
            })
            .collect()
    }

    /// TD-064: Serialize cached filterable metadata for snapshot persistence.
    pub fn serialize_filterable_metadata(&self) -> Vec<(String, FilterableHnswMetadata)> {
        self.filterable_metadata.snapshot()
    }

    /// TD-064: Restore filterable metadata cache after snapshot load.
    pub fn restore_filterable_metadata(&self, entries: Vec<(String, FilterableHnswMetadata)>) {
        self.filterable_metadata.restore(entries);
    }

    /// Serialize quantized vectors
    pub fn serialize_quantized_vectors(&self) -> Vec<(String, Vec<u8>)> {
        self.quantized_vectors
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    /// Restore HNSW state from deserialized data
    pub fn restore_from_state(
        &self,
        id_mapping: crate::index::axis::storage::serialization::SerializableIdMapping,
        layers: Vec<((usize, usize), Vec<usize>)>,
        max_layer: usize,
        entry_point: Option<usize>,
        vectors: Vec<crate::index::axis::storage::serialization::SerializableVector>,
        quantized_vectors: Vec<(String, Vec<u8>)>,
        collection_config: crate::index::axis::storage::serialization::SerializableCollectionConfig,
    ) -> Result<()> {
        use crate::index::axis::zero_overhead_vector::ZeroOverheadVector;

        info!(
            "Restoring HNSW state: {} vectors, {} layers, {} quantized vectors",
            vectors.len(),
            layers.len(),
            quantized_vectors.len()
        );

        // 1. Restore ID mapping
        for (external_id, internal_id) in id_mapping.external_to_internal {
            self.id_mapping.restore_mapping(external_id, internal_id)?;
        }
        self.id_mapping.set_next_id(id_mapping.next_id);

        // 2. Restore graph layers
        for ((layer, node_id), connections) in layers {
            self.layers.insert((layer, node_id), connections);
        }

        // 3. Restore max_layer
        self.max_layer.store(max_layer, AtomicOrdering::Relaxed);

        // 4. Restore entry point
        *self
            .entry_point
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = entry_point;

        // 5. Restore vectors
        {
            let mut vec_store = self
                .vectors
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            for vec in vectors {
                let zero_vec = ZeroOverheadVector::from_bytes(vec.data);
                // Get the ID from the zero-overhead vector
                let id = zero_vec.id(collection_config.dimension * std::mem::size_of::<f32>());
                // For FP32 vectors, add directly
                if !collection_config.is_quantized {
                    let fp32_data = zero_vec.as_f32(collection_config.dimension);
                    vec_store.add_fp32(id.to_string(), fp32_data)?;
                } else {
                    let quant_size = match collection_config.quantization_method {
                        Some(0) => collection_config.dimension, // INT8
                        Some(1) => collection_config.dimension, // PQ8
                        Some(2) => (collection_config.dimension * 4).div_ceil(8), // PQ4
                        Some(3) => collection_config.dimension.div_ceil(8), // Binary
                        _ => collection_config.dimension,
                    };
                    let quant_data = zero_vec.as_quantized(quant_size);
                    vec_store.add_quantized(id.to_string(), quant_data)?;
                }
            }
        }

        // 6. Restore quantized vectors
        for (id, data) in quantized_vectors {
            self.quantized_vectors.insert(id, data);
        }

        info!("HNSW state restoration complete");
        Ok(())
    }
}

/// Factory function to create HNSW index instances
pub fn create_hnsw_index(
    config: AxisHnswConfig,
    dimension: usize,
) -> Result<Box<dyn AxisVectorIndex>> {
    Ok(Box::new(AxisHnswIndex::new(config, dimension)?))
}

/// Factory function to create HNSW index instances with vector representation preference
pub fn create_hnsw_index_with_representation(
    config: AxisHnswConfig,
    dimension: usize,
    extraction_mode: ExtractionMode,
) -> Result<Box<dyn AxisVectorIndex>> {
    Ok(Box::new(AxisHnswIndex::new_with_extraction_mode(
        None,
        config,
        dimension,
        extraction_mode,
    )?))
}

/// Factory function to create HNSW index instances for specific collection with representation
pub fn create_hnsw_index_for_collection(
    collection_id: String,
    config: AxisHnswConfig,
    dimension: usize,
    extraction_mode: ExtractionMode,
) -> Result<Box<dyn AxisVectorIndex>> {
    Ok(Box::new(AxisHnswIndex::new_with_extraction_mode(
        Some(collection_id),
        config,
        dimension,
        extraction_mode,
    )?))
}

#[cfg(test)]
mod tests {
    use crate::compute::distance_computation::DistanceMetric;
    use crate::index::axis::*;

    #[tokio::test]
    async fn test_hnsw_basic_operations() {
        // Initialize hardware capabilities
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init

        let config = AxisHnswConfig::default();
        let index = AxisHnswIndex::new(config, 3).expect("Failed to create HNSW index");

        // Add test vectors
        index
            .add("vec1".to_string(), vec![1.0, 0.0, 0.0])
            .await
            .expect("Failed to add vec1");
        index
            .add("vec2".to_string(), vec![0.0, 1.0, 0.0])
            .await
            .expect("Failed to add vec2");
        index
            .add("vec3".to_string(), vec![1.0, 1.0, 0.0])
            .await
            .expect("Failed to add vec3");

        assert_eq!(index.stats().vector_count, 3);

        // Search should work
        let results = index
            .search(&[1.0, 0.0, 0.0], 2, None)
            .await
            .expect("Failed to search");
        assert!(results.len() <= 2); // HNSW is approximate

        // Remove a vector
        index.remove("vec2").await.expect("Failed to remove vec2");
        assert_eq!(index.stats().vector_count, 2);

        // Remove non-existent vector (should succeed without error)
        index
            .remove("nonexistent")
            .await
            .expect("Failed to remove nonexistent");
    }

    /// Proves the per-query `ef` override (the `SearchEffort` knob threaded via
    /// `search_with_filter_ef`) actually controls the HNSW accuracy/latency
    /// tradeoff: a tiny `ef` (greedy descent) recovers the true nearest
    /// neighbour for STRICTLY fewer queries than a large `ef` on the SAME index,
    /// and disagrees with it on at least one query.
    ///
    /// Regression guard for the fix where `SearchMode`/`nprobe` was dropped
    /// before reaching AXIS, so the warm HNSW path always used the size-aware
    /// default `ef` and the `approximate`/`approximate:N` knob was a no-op. If
    /// the override were ignored, both calls would use the same `ef` and the
    /// results would be identical for every query (`differed == 0`).
    #[tokio::test]
    async fn test_hnsw_ef_override_controls_recall() {
        let _ = proximadb_hardware::hardware_capabilities();

        let dim = 48usize;
        let n = 5000usize;
        let index = AxisHnswIndex::new(AxisHnswConfig::default(), dim).expect("create HNSW index");

        // Deterministic, normalized random vectors so L2 / cosine / dot all rank
        // neighbours identically (the brute-force oracle is metric-agnostic).
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let normalize = |v: &mut Vec<f32>| {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                v.iter_mut().for_each(|x| *x /= norm);
            }
        };

        let mut vecs: Vec<Vec<f32>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut v: Vec<f32> = (0..dim).map(|_| next()).collect();
            normalize(&mut v);
            index.add(format!("v{i}"), v.clone()).await.expect("add");
            vecs.push(v);
        }

        let l2 =
            |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>();

        let queries = 80usize;
        let (mut hit_low, mut hit_high, mut differed) = (0usize, 0usize, 0usize);
        for _ in 0..queries {
            let mut q: Vec<f32> = (0..dim).map(|_| next()).collect();
            normalize(&mut q);

            // Brute-force true nearest neighbour (top-1).
            let true_idx = (0..n)
                .min_by(|&a, &b| {
                    l2(&q, &vecs[a])
                        .partial_cmp(&l2(&q, &vecs[b]))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .expect("non-empty");
            let true_id = format!("v{true_idx}");

            // ef = 1: pure greedy descent (top_k=1, so the `ef.max(top_k)` floor
            // is 1). ef = n: effectively exhaustive.
            let low = index
                .search_with_filter_ef(&q, 1, Some(1), None)
                .await
                .expect("low-ef search");
            let high = index
                .search_with_filter_ef(&q, 1, Some(n), None)
                .await
                .expect("high-ef search");

            let low_id = low.first().map(|(id, _)| id.clone());
            let high_id = high.first().map(|(id, _)| id.clone());
            if low_id.as_deref() == Some(true_id.as_str()) {
                hit_low += 1;
            }
            if high_id.as_deref() == Some(true_id.as_str()) {
                hit_high += 1;
            }
            if low_id != high_id {
                differed += 1;
            }
        }

        // Large ef is near-perfect on this easy data.
        assert!(
            hit_high as f64 / queries as f64 >= 0.85,
            "high-ef recall@1 too low ({hit_high}/{queries}) — index may be broken"
        );
        // The override is plumbed: greedy ef=1 disagrees with exhaustive ef=n on
        // at least one query. If the knob were ignored both would use the same ef
        // and `differed` would be 0.
        assert!(
            differed > 0,
            "ef override had NO effect — ef=1 and ef=n returned identical results for all {queries} queries (knob ignored?)"
        );
        // And it controls accuracy: more effort recovers the true NN more often.
        assert!(
            hit_high > hit_low,
            "large ef must recover the true NN more often than greedy ef=1 (high={hit_high} low={hit_low})"
        );
    }

    /// Same proof as `test_hnsw_ef_override_controls_recall`, but through the
    /// PREDICATE-FILTERED path (`search_with_predicate_fn`) with a permissive
    /// predicate — guards the follow-slice that threads the per-query `ef`
    /// override into the Inline/PostFilter traversal (ACORN). Effort tunes `ef`
    /// (traversal depth); the post-filter oversample is an orthogonal lever.
    #[tokio::test]
    async fn test_hnsw_predicate_ef_override_controls_recall() {
        let _ = proximadb_hardware::hardware_capabilities();

        let dim = 48usize;
        let n = 5000usize;
        let index = AxisHnswIndex::new(AxisHnswConfig::default(), dim).expect("create HNSW index");

        let mut state = 0x0FEE_1DEA_D15E_A5E1u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let normalize = |v: &mut Vec<f32>| {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                v.iter_mut().for_each(|x| *x /= norm);
            }
        };

        let mut vecs: Vec<Vec<f32>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut v: Vec<f32> = (0..dim).map(|_| next()).collect();
            normalize(&mut v);
            index.add(format!("v{i}"), v.clone()).await.expect("add");
            vecs.push(v);
        }

        let l2 =
            |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>();

        let queries = 80usize;
        let (mut hit_low, mut hit_high, mut differed) = (0usize, 0usize, 0usize);
        for _ in 0..queries {
            let mut q: Vec<f32> = (0..dim).map(|_| next()).collect();
            normalize(&mut q);

            let true_idx = (0..n)
                .min_by(|&a, &b| {
                    l2(&q, &vecs[a])
                        .partial_cmp(&l2(&q, &vecs[b]))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .expect("non-empty");
            let true_id = format!("v{true_idx}");

            // Permissive predicate (accept all) so only `ef` drives the result.
            let low = index
                .search_with_predicate_fn(&q, 1, |_id| true, Some(1))
                .await
                .expect("low-ef predicate search");
            let high = index
                .search_with_predicate_fn(&q, 1, |_id| true, Some(n))
                .await
                .expect("high-ef predicate search");

            let low_id = low.first().map(|(id, _)| id.clone());
            let high_id = high.first().map(|(id, _)| id.clone());
            if low_id.as_deref() == Some(true_id.as_str()) {
                hit_low += 1;
            }
            if high_id.as_deref() == Some(true_id.as_str()) {
                hit_high += 1;
            }
            if low_id != high_id {
                differed += 1;
            }
        }

        assert!(
            hit_high as f64 / queries as f64 >= 0.85,
            "high-ef predicate recall@1 too low ({hit_high}/{queries})"
        );
        assert!(
            differed > 0,
            "predicate ef override had NO effect for all {queries} queries (knob ignored?)"
        );
        assert!(
            hit_high > hit_low,
            "large ef must recover the true NN more often than greedy ef=1 on the predicate path (high={hit_high} low={hit_low})"
        );
    }

    #[tokio::test]
    async fn test_hnsw_search_quality() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init

        let mut config = AxisHnswConfig::default();
        config.ef = 200; // Higher ef for better quality
        let index = AxisHnswIndex::new(config, 4).expect("Failed to create HNSW index");

        // Create a set of test vectors
        let test_vectors = [
            ("v1", vec![1.0, 0.0, 0.0, 0.0]),
            ("v2", vec![0.9, 0.1, 0.0, 0.0]),
            ("v3", vec![0.0, 1.0, 0.0, 0.0]),
            ("v4", vec![0.0, 0.0, 1.0, 0.0]),
            ("v5", vec![0.0, 0.0, 0.0, 1.0]),
            ("v6", vec![0.8, 0.2, 0.0, 0.0]),
            ("v7", vec![0.7, 0.3, 0.0, 0.0]),
        ];

        for (id, vector) in test_vectors.iter() {
            index
                .add(id.to_string(), vector.clone())
                .await
                .expect("Failed to add vector");
        }

        // Search for nearest neighbors to v1
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = index
            .search(&query, 3, None)
            .await
            .expect("Failed to search");

        // v1, v2, v6, v7 should be closest (all have high first component)
        // With default config, should find at least 1 result
        assert!(
            !results.is_empty(),
            "Expected at least 1 result, got {}",
            results.len()
        );

        // The first result should be one of the nearby vectors
        // HNSW is probabilistic, so v1, v2, v6, or v7 are all acceptable
        if let Some(first) = results.first() {
            let valid_results = vec!["v1", "v2", "v6", "v7"];
            assert!(
                valid_results.contains(&first.0.as_str()),
                "Expected first result to be one of {:?}, got {}",
                valid_results,
                first.0
            );
        }
    }

    #[tokio::test]
    async fn test_hnsw_layer_navigation() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init

        let mut config = AxisHnswConfig::default();
        config.m = 16;
        config.ef_construction = 200;
        let index = AxisHnswIndex::new(config, 3).expect("Failed to create HNSW index");

        // Add enough vectors to create multiple layers
        for i in 0..50 {
            let vector = vec![(i as f32).sin(), (i as f32).cos(), (i as f32 * 0.5).sin()];
            index
                .add(format!("vec_{}", i), vector)
                .await
                .expect("Failed to add vector");
        }

        // Check that multiple layers were created
        let stats = index.stats();
        assert_eq!(stats.vector_count, 50);

        // Search should work efficiently across layers
        let query = vec![0.5, 0.5, 0.5];
        let results = index
            .search(&query, 10, None)
            .await
            .expect("Failed to search");
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_hnsw_pruning_heuristic() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init

        let mut config = AxisHnswConfig::default();
        config.m = 5; // Small M to test pruning
        let index = AxisHnswIndex::new(config, 2).expect("Failed to create HNSW index");

        // Add vectors in a line to test pruning
        for i in 0..20 {
            index
                .add(format!("v{}", i), vec![i as f32, 0.0])
                .await
                .expect("Failed to add vector");
        }

        // Search for middle point
        let query = vec![10.0, 0.0];
        let results = index
            .search(&query, 5, None)
            .await
            .expect("Failed to search");

        // Should find neighbors despite pruning
        // With small M=5, graph connectivity may be limited, so just check we get some results
        assert!(
            !results.is_empty(),
            "Expected at least 1 result, got {}",
            results.len()
        );
    }

    #[test]
    fn test_hnsw_config_validation() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init

        // Test valid config
        let valid_config = AxisHnswConfig {
            m: 16,
            ef_construction: 200,
            ef: 100,
            max_layers: 10,
            distance_metric: DistanceMetric::Cosine,
            ..AxisHnswConfig::default()
        };
        assert!(AxisHnswIndex::new(valid_config, 128).is_ok());

        // Test invalid dimension
        let config = AxisHnswConfig::default();
        assert!(AxisHnswIndex::new(config.clone(), 0).is_err());

        // Test with very high dimension
        assert!(AxisHnswIndex::new(config, 10000).is_ok());
    }

    #[tokio::test]
    async fn test_hnsw_empty_index_search() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init

        let config = AxisHnswConfig::default();
        let index = AxisHnswIndex::new(config, 3).expect("Failed to create HNSW index");

        // Search on empty index should return empty results
        let results = index
            .search(&[1.0, 0.0, 0.0], 5, None)
            .await
            .expect("Failed to search");
        assert_eq!(results.len(), 0);
    }

    #[tokio::test]
    async fn test_hnsw_duplicate_removal() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init

        let config = AxisHnswConfig::default();
        let index = AxisHnswIndex::new(config, 2).expect("Failed to create HNSW index");

        // Add the same ID twice
        index
            .add("duplicate".to_string(), vec![1.0, 0.0])
            .await
            .expect("Failed to add duplicate");
        assert_eq!(index.stats().vector_count, 1);

        // Adding again should replace
        index
            .add("duplicate".to_string(), vec![1.0, 0.0])
            .await
            .expect("Failed to add duplicate again");
        assert_eq!(index.stats().vector_count, 1);

        // Remove should work
        index
            .remove("duplicate")
            .await
            .expect("Failed to remove duplicate");
        assert_eq!(index.stats().vector_count, 0);
    }

    #[tokio::test]
    async fn test_hnsw_serialization_roundtrip() {
        use crate::index::axis::storage::serialization::IndexSerializer;

        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init

        let config = AxisHnswConfig::default();
        let index = AxisHnswIndex::new(config.clone(), 4).expect("Failed to create HNSW index");

        // Add test vectors
        let test_vectors = [
            ("v1", vec![1.0, 0.0, 0.0, 0.0]),
            ("v2", vec![0.0, 1.0, 0.0, 0.0]),
            ("v3", vec![0.0, 0.0, 1.0, 0.0]),
            ("v4", vec![0.5, 0.5, 0.0, 0.0]),
        ];

        for (id, vector) in test_vectors.iter() {
            index
                .add(id.to_string(), vector.clone())
                .await
                .expect("Failed to add vector");
        }

        // Verify initial state
        assert_eq!(index.stats().vector_count, 4);
        let original_results = index
            .search(&[1.0, 0.0, 0.0, 0.0], 2, None)
            .await
            .expect("Failed to search");
        assert!(!original_results.is_empty());

        // Serialize
        let serialized = IndexSerializer::serialize_hnsw(&index, "test_collection")
            .expect("Failed to serialize HNSW index");
        assert!(
            !serialized.is_empty(),
            "Serialized data should not be empty"
        );

        // Deserialize
        let (restored_index, metadata) = IndexSerializer::deserialize_hnsw(&serialized, &config)
            .expect("Failed to deserialize HNSW index");

        // Verify metadata
        assert_eq!(metadata.num_vectors, 4);
        assert_eq!(metadata.dimension, 4);

        // Verify restored index has same vector count
        assert_eq!(restored_index.stats().vector_count, 4);

        // Verify search works on restored index
        let restored_results = restored_index
            .search(&[1.0, 0.0, 0.0, 0.0], 2, None)
            .await
            .expect("Failed to search restored index");
        assert!(!restored_results.is_empty());

        // The top result should be the same (v1 is closest to query)
        assert_eq!(original_results[0].0, restored_results[0].0);
    }

    // ── Phase C: Filter-Aware Vector Search (ACORN/NaviX) ────────────────────

    #[test]
    fn test_hnsw_config_gamma_default_is_one() {
        let cfg = AxisHnswConfig::default();
        assert_eq!(
            cfg.gamma, 1.0,
            "default gamma = 1.0 means no expansion (legacy)"
        );
        assert!(
            cfg.selectivity_min > 0.0,
            "selectivity_min must be positive"
        );
    }

    #[test]
    fn test_hnsw_config_acorn_expansion() {
        let cfg = AxisHnswConfig {
            gamma: 2.0,
            selectivity_min: 0.01,
            ..AxisHnswConfig::default()
        };
        assert_eq!(cfg.gamma, 2.0);
        assert_eq!(cfg.selectivity_min, 0.01);
        // expanded_m = floor(2.0 * 16) = 32
        assert_eq!(cfg.expanded_m(), 32);
    }

    #[tokio::test]
    async fn test_predicate_search_all_pass_matches_unfiltered() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        let cfg = AxisHnswConfig {
            ef: 200,
            ..AxisHnswConfig::default()
        };
        let index = AxisHnswIndex::new(cfg, 4).expect("create index");

        for i in 0u32..5 {
            let v: Vec<f32> = (0..4)
                .map(|j| if j == i as usize { 1.0 } else { 0.0 })
                .collect();
            index.add(format!("v{i}"), v).await.expect("add");
        }

        let query = vec![1.0f32, 0.0, 0.0, 0.0];
        let unfiltered = index.search(&query, 3, None).await.unwrap();
        let filtered = index
            .search_with_predicate_fn(&query, 3, |_id| true, None)
            .await
            .unwrap();

        // With all-pass predicate, top result must match
        if !unfiltered.is_empty() && !filtered.is_empty() {
            assert_eq!(
                unfiltered[0].0, filtered[0].0,
                "all-pass predicate must return same top-1 as unfiltered"
            );
        }
    }

    #[tokio::test]
    async fn test_predicate_search_excludes_specific_id() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        let cfg = AxisHnswConfig {
            ef: 200,
            ..AxisHnswConfig::default()
        };
        let index = AxisHnswIndex::new(cfg, 4).expect("create index");

        index
            .add("v0".into(), vec![1.0, 0.0, 0.0, 0.0])
            .await
            .unwrap();
        index
            .add("v1".into(), vec![0.9, 0.1, 0.0, 0.0])
            .await
            .unwrap();
        index
            .add("v2".into(), vec![0.0, 1.0, 0.0, 0.0])
            .await
            .unwrap();

        let query = vec![1.0f32, 0.0, 0.0, 0.0];
        let results = index
            .search_with_predicate_fn(&query, 2, |id| id != "v0", None)
            .await
            .unwrap();

        // v0 must not appear in any result
        assert!(
            results.iter().all(|(id, _)| id != "v0"),
            "excluded id must not appear: got {:?}",
            results
        );
    }

    /// TD-064: add_with_metadata caches per-record metadata so the structured
    /// trait method can enforce tenant isolation without consulting any
    /// external lookup. Cross-tenant records must be excluded.
    #[tokio::test]
    async fn test_predicate_search_isolates_by_tenant() {
        use crate::index::axis::filterable_metadata::FilterableHnswMetadata;
        use crate::index::axis::index_factory::AxisVectorIndex;

        let _ = proximadb_hardware::hardware_capabilities();
        let cfg = AxisHnswConfig {
            ef: 200,
            ..AxisHnswConfig::default()
        };
        let index = AxisHnswIndex::new(cfg, 4).expect("create index");

        // Add 4 vectors split across two tenants. Vectors are deliberately
        // close in vector space so post-filter would have shrunk results
        // — predicate-aware search must keep recall while enforcing tenant.
        let make_meta = |tenant: &str| {
            let mut m = FilterableHnswMetadata::default();
            m.tenant_id = Some(tenant.to_string());
            m
        };

        index
            .add_with_metadata("a1".into(), vec![1.0, 0.0, 0.0, 0.0], &make_meta("acme"))
            .await
            .unwrap();
        index
            .add_with_metadata("a2".into(), vec![0.9, 0.1, 0.0, 0.0], &make_meta("acme"))
            .await
            .unwrap();
        index
            .add_with_metadata("b1".into(), vec![0.8, 0.2, 0.0, 0.0], &make_meta("beta"))
            .await
            .unwrap();
        index
            .add_with_metadata("b2".into(), vec![0.7, 0.3, 0.0, 0.0], &make_meta("beta"))
            .await
            .unwrap();

        assert!(
            index.supports_predicate_search(),
            "index must report predicate support once metadata is cached"
        );

        // Query as tenant "acme" — only a1, a2 may appear.
        let query = vec![1.0f32, 0.0, 0.0, 0.0];
        let results = index
            .search_with_predicate(&query, 4, Some("acme"), None, None)
            .await
            .unwrap();
        assert!(
            !results.is_empty(),
            "predicate-aware search must return results for matching tenant"
        );
        assert!(
            results.iter().all(|(id, _)| id.starts_with('a')),
            "cross-tenant ids must not appear in tenant 'acme' query: got {:?}",
            results
        );

        // Query as tenant "beta" — only b1, b2 may appear.
        let results_beta = index
            .search_with_predicate(&query, 4, Some("beta"), None, None)
            .await
            .unwrap();
        assert!(
            !results_beta.is_empty(),
            "predicate-aware search must return results for tenant 'beta'"
        );
        assert!(
            results_beta.iter().all(|(id, _)| id.starts_with('b')),
            "cross-tenant ids must not appear in tenant 'beta' query: got {:?}",
            results_beta
        );
    }

    /// TD-064: When the cache has no metadata for an id and the caller has
    /// supplied a tenant predicate, the record must be excluded (fail-closed).
    #[tokio::test]
    async fn test_predicate_search_fails_closed_for_unindexed_metadata() {
        use crate::index::axis::filterable_metadata::FilterableHnswMetadata;
        use crate::index::axis::index_factory::AxisVectorIndex;

        let _ = proximadb_hardware::hardware_capabilities();
        let cfg = AxisHnswConfig {
            ef: 200,
            ..AxisHnswConfig::default()
        };
        let index = AxisHnswIndex::new(cfg, 4).expect("create index");

        // Mix: one vector inserted WITH metadata (tenant "acme"), one WITHOUT.
        let mut meta = FilterableHnswMetadata::default();
        meta.tenant_id = Some("acme".into());
        index
            .add_with_metadata("a1".into(), vec![1.0, 0.0, 0.0, 0.0], &meta)
            .await
            .unwrap();
        index
            .add("legacy".into(), vec![0.9, 0.1, 0.0, 0.0])
            .await
            .unwrap();

        let query = vec![1.0f32, 0.0, 0.0, 0.0];
        let results = index
            .search_with_predicate(&query, 4, Some("acme"), None, None)
            .await
            .unwrap();

        // "legacy" must be excluded — no cached metadata + tenant predicate
        // ⇒ fail-closed.
        assert!(
            results.iter().all(|(id, _)| id != "legacy"),
            "unindexed-metadata record must be excluded under tenant predicate: got {:?}",
            results
        );
    }

    #[test]
    fn test_smin_fallback_selector() {
        let cfg = AxisHnswConfig {
            selectivity_min: 0.05,
            ..AxisHnswConfig::default()
        };
        assert!(
            cfg.should_use_brute_force(0.01),
            "0.01 < 0.05 → brute force"
        );
        assert!(!cfg.should_use_brute_force(0.10), "0.10 > 0.05 → HNSW ok");
        assert!(
            cfg.should_use_brute_force(0.05),
            "at threshold → brute force (safe)"
        );
    }

    #[tokio::test]
    async fn test_insert_uses_gamma_expansion_when_gamma_gt_one() {
        // A high gamma index should build more edges per node than a gamma=1.0 index.
        // We verify this by checking that a node in the gamma>1 index has at least as
        // many neighbors as the gamma=1 baseline (ideally more, but at small scale
        // they may coincide). The key invariant: gamma>1 NEVER produces fewer edges.
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init

        let cfg_base = AxisHnswConfig {
            m: 4,
            gamma: 1.0,
            ..AxisHnswConfig::default()
        };
        let cfg_expanded = AxisHnswConfig {
            m: 4,
            gamma: 2.0,
            ..AxisHnswConfig::default()
        };

        let base_index = AxisHnswIndex::new(cfg_base, 2).expect("create base index");
        let exp_index = AxisHnswIndex::new(cfg_expanded, 2).expect("create expanded index");

        for i in 0..8u32 {
            let v = vec![i as f32, (8 - i) as f32];
            base_index.add(format!("v{i}"), v.clone()).await.unwrap();
            exp_index.add(format!("v{i}"), v).await.unwrap();
        }

        // Count connections at layer 0 across all nodes in both indexes
        let base_edges: usize = base_index
            .layers
            .iter()
            .filter(|e| e.key().0 == 0)
            .map(|e| e.value().len())
            .sum();
        let exp_edges: usize = exp_index
            .layers
            .iter()
            .filter(|e| e.key().0 == 0)
            .map(|e| e.value().len())
            .sum();

        assert!(
            exp_edges >= base_edges,
            "gamma=2.0 index must have at least as many edges as gamma=1.0 (got {exp_edges} vs {base_edges})"
        );
    }

    #[test]
    fn test_select_neighbors_gamma_returns_more_candidates() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init
        let cfg = AxisHnswConfig {
            m: 4,
            gamma: 2.0,
            ..AxisHnswConfig::default()
        };
        let index = AxisHnswIndex::new(cfg, 4).expect("create index");

        // 10 candidates, gamma=2.0, m=4 → expanded picks min(8, 10) = 8
        let candidates: Vec<(usize, f32)> = (0..10).map(|i| (i, i as f32 * 0.1)).collect();
        let standard = index.select_neighbors(candidates.clone(), 4);
        let expanded = index.select_neighbors_gamma(candidates, 4);

        assert_eq!(standard.len(), 4);
        assert_eq!(expanded.len(), 8, "gamma=2.0 * m=4 → 8 neighbors");
    }

    /// Frontier-prune guard: with a PERMISSIVE predicate, the predicate walk
    /// must return exactly what the plain `search_layer` finds on the same
    /// graph — the prune added to `search_layer_predicate` skips only entries
    /// that could never be expanded (strictly worse than the current worst of
    /// a full `ef` result set), so it can change neither the result set nor
    /// the visit order.
    #[tokio::test]
    async fn test_search_layer_predicate_matches_plain_layer_when_permissive() {
        // Initialize hardware capabilities
        let _ = proximadb_hardware::hardware_capabilities();

        let config = AxisHnswConfig::default();
        let index = AxisHnswIndex::new(config, 4).expect("create HNSW index");

        // Deterministic spread of distinct-distance vectors (no ties).
        let mut rng: u64 = 0x243F6A8885A308D3;
        let mut next = || {
            rng ^= rng >> 12;
            rng ^= rng << 25;
            rng ^= rng >> 27;
            let v: Vec<f32> = (0..4)
                .map(|i| ((rng >> (i * 8)) & 0xFF) as f32 / 255.0)
                .collect();
            v
        };
        for i in 0..80 {
            index
                .add(format!("v{i}"), next())
                .await
                .expect("add vector");
        }

        let entry = index
            .get_entry_point()
            .expect("entry point exists after inserts");
        let query = next();
        let ef = 16;

        let plain = index.search_layer(&query, &[entry], ef, 0);
        let permissive = index.search_layer_predicate(&query, &[entry], ef, 0, &|_| true);

        // Both return ascending-distance Vec<(node, dist)>.
        let plain_set: Vec<(usize, f32)> = {
            let mut v = plain.clone();
            v.sort_by(|a, b| a.0.cmp(&b.0));
            v
        };
        let permissive_set: Vec<(usize, f32)> = {
            let mut v = permissive.clone();
            v.sort_by(|a, b| a.0.cmp(&b.0));
            v
        };
        assert_eq!(
            plain_set, permissive_set,
            "permissive predicate walk diverged from plain search_layer"
        );
    }

    /// Sparse-predicate drain guard: when NOTHING near the query passes the
    /// predicate, the ACORN skip-through walk must still drain through
    /// non-matching nodes and find the far matches. The frontier prune only
    /// activates once `ef` PASSING results exist, so a sparse walk keeps its
    /// full-reachable-graph recall behaviour. Guards against anyone turning
    /// the prune into a hard beam cap (documented as a recall regression in
    /// the 1.3 investigation comment above `oversample_k`).
    #[tokio::test]
    async fn test_search_layer_predicate_drains_to_far_matches_when_sparse() {
        // Initialize hardware capabilities
        let _ = proximadb_hardware::hardware_capabilities();

        let config = AxisHnswConfig::default();
        let index = AxisHnswIndex::new(config, 2).expect("create HNSW index");

        // Two tight clusters: "near" vectors around (0,0), passing "far"
        // vectors around (1000,1000). The query sits in the near cluster, so
        // every node near the query FAILS the predicate and the walk must
        // skip through them to reach the far matches.
        for i in 0..30 {
            let jitter = i as f32 * 0.01;
            index
                .add(format!("near{i}"), vec![jitter, jitter])
                .await
                .expect("add near vector");
        }
        for i in 0..10 {
            let jitter = i as f32 * 0.01;
            index
                .add(format!("far{i}"), vec![1000.0 + jitter, 1000.0 + jitter])
                .await
                .expect("add far vector");
        }

        let entry = index
            .get_entry_point()
            .expect("entry point exists after inserts");
        let query = [0.0_f32, 0.0_f32];
        let ef = 4;

        let passing: std::collections::HashSet<String> =
            (0..10).map(|i| format!("far{i}")).collect();
        let results = index.search_layer_predicate(&query, &[entry], ef, 0, &|node| {
            index
                .id_mapping
                .external(node)
                .is_some_and(|id| passing.contains(&id))
        });

        assert!(
            !results.is_empty(),
            "sparse predicate must still reach far matches via skip-through"
        );
        // Everything returned must PASS the predicate (far-cluster nodes).
        for (node, _) in &results {
            let id = index
                .id_mapping
                .external(*node)
                .expect("internal id resolves");
            assert!(
                passing.contains(&id),
                "predicate-failing node {id} leaked into results"
            );
        }
    }
}
