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

//! Vector Search Algorithms Module
//!
//! This module provides integration with the AXIS adaptive indexing system
//! for sophisticated vector search capabilities.

use anyhow::{Result, anyhow};
#[cfg(feature = "axis")]
use std::sync::Arc;

// Import AXIS indexing components
use crate::compute::distance_computation::DistanceMetric;
#[cfg(feature = "axis")]
use crate::index::axis::{AxisVectorIndex, IndexAlgorithm as AxisIndexAlgorithm, IndexFactory};
use crate::services::operations::vectors::VectorOperationsService;

// Type aliases for compatibility
pub type VectorSearchResult = crate::core::service_types::VectorSearchResult;
pub type SearchParameters = VectorSearchConfig;

// Search query structure
#[derive(Debug, Clone)]
pub struct SearchQuery {
    pub vector: Vec<f32>,
    pub top_k: usize,
    pub distance_metric: DistanceMetric,
}

/// Backwards-compat alias for [`VectorSearchConfig`].
pub type SearchConfig = VectorSearchConfig;

// Search configuration
#[derive(Debug, Clone)]
pub struct VectorSearchConfig {
    pub algorithm: SearchAlgorithm,
    pub timeout_ms: Option<u64>,
}

/// Vector search algorithm types (AXIS-based)
#[derive(Debug, Clone)]
pub enum SearchAlgorithm {
    /// Brute force search (exact)
    BruteForce,
    /// IVF (Inverted File) index with AXIS configuration
    IVF {
        n_clusters: usize,
        n_probe: usize,
        enable_pq: bool,
    },
    /// LSH (Locality Sensitive Hashing) with AXIS configuration
    LSH {
        n_tables: usize,
        n_hashes: usize,
        hash_width: f32,
    },
    /// HNSW (Hierarchical Navigable Small World) - uses AXIS integration
    HNSW {
        m: usize,
        ef_construction: usize,
        ef_search: usize,
    },
}

/// Vector search engine using AXIS indexes (AXIS-gated; no external consumers).
#[cfg(feature = "axis")]
pub struct VectorSearchEngine {
    /// Selected algorithm
    algorithm: SearchAlgorithm,
    /// The AXIS index implementation
    axis_index: Option<Box<dyn AxisVectorIndex>>,
    /// Dimension of vectors
    dimension: usize,
    /// Distance metric
    distance_metric: DistanceMetric,
}

#[cfg(feature = "axis")]
impl VectorSearchEngine {
    /// Create new search engine with specified algorithm
    pub fn new(
        algorithm: SearchAlgorithm,
        dimension: usize,
        distance_metric: DistanceMetric,
    ) -> Self {
        // Don't create index immediately - wait for training if needed
        Self {
            algorithm,
            axis_index: None,
            dimension,
            distance_metric,
        }
    }

    /// Initialize the index (must be called before use, especially for IVF which needs training)
    pub async fn initialize(&mut self, training_data: Option<&[Vec<f32>]>) -> Result<()> {
        let axis_algorithm = match &self.algorithm {
            SearchAlgorithm::IVF {
                n_clusters,
                n_probe,
                enable_pq,
            } => {
                AxisIndexAlgorithm::IVF {
                    nlist: *n_clusters as u32,
                    nprobe: *n_probe as u32,
                    quantizer: if *enable_pq {
                        Some(Box::new(AxisIndexAlgorithm::PQ {
                            m: 8,     // 8 subquantizers
                            nbits: 8, // 8 bits per subquantizer
                            train_size: 10000,
                        }))
                    } else {
                        None
                    },
                }
            }
            SearchAlgorithm::LSH {
                n_tables,
                n_hashes,
                hash_width,
            } => AxisIndexAlgorithm::LSH {
                n_projections: *n_hashes as u32,
                n_hash_tables: *n_tables as u32,
                hash_width: *hash_width,
            },
            SearchAlgorithm::HNSW {
                m,
                ef_construction,
                ef_search,
            } => {
                AxisIndexAlgorithm::HNSW {
                    m: *m as u32,
                    ef_construction: *ef_construction as u32,
                    ef_search: *ef_search as u32,
                    max_elements: 1_000_000, // Default
                }
            }
            SearchAlgorithm::BruteForce => {
                return Ok(()); // No index needed for brute force
            }
        };

        // Create the AXIS index
        self.axis_index = Some(
            IndexFactory::create_trained_index(
                &axis_algorithm,
                self.dimension,
                self.distance_metric,
                training_data,
            )
            .await?,
        );

        Ok(())
    }

    /// Add vectors to the index
    pub async fn add_vectors(
        &self,
        vectors: Vec<(String, Arc<proximadb_records::ProximaRecord>)>,
    ) -> Result<()> {
        if let Some(index) = &self.axis_index {
            for (id, record) in vectors {
                let values = record
                    .embeddings
                    .first()
                    .map(|e| e.values.to_fp32_owned())
                    .unwrap_or_default();
                index.add(id, values).await?;
            }
            Ok(())
        } else if matches!(self.algorithm, SearchAlgorithm::BruteForce) {
            Ok(()) // Brute force doesn't maintain an index
        } else {
            Err(anyhow!("Index not initialized. Call initialize() first."))
        }
    }

    /// Search for k nearest neighbors
    pub async fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&std::collections::HashMap<String, String>>,
    ) -> Result<Vec<(String, f32)>> {
        if let Some(index) = &self.axis_index {
            index.search(query, k, filter).await
        } else if matches!(self.algorithm, SearchAlgorithm::BruteForce) {
            // Brute force search would go through all vectors
            // This should integrate with the storage engine
            Ok(Vec::new())
        } else {
            Err(anyhow!("Index not initialized. Call initialize() first."))
        }
    }

    /// Get algorithm info
    pub fn algorithm_info(&self) -> String {
        match &self.algorithm {
            SearchAlgorithm::BruteForce => "Brute Force (Exact Search)".to_string(),
            SearchAlgorithm::IVF {
                n_clusters,
                n_probe,
                enable_pq,
            } => {
                format!(
                    "AXIS IVF: {} clusters, {} probe, PQ: {}",
                    n_clusters, n_probe, enable_pq
                )
            }
            SearchAlgorithm::LSH {
                n_tables,
                n_hashes,
                hash_width,
            } => {
                format!(
                    "AXIS LSH: {} tables, {} hashes, width: {}",
                    n_tables, n_hashes, hash_width
                )
            }
            SearchAlgorithm::HNSW {
                m,
                ef_construction,
                ef_search,
            } => {
                format!(
                    "AXIS HNSW: M={}, ef_construction={}, ef_search={}",
                    m, ef_construction, ef_search
                )
            }
        }
    }

    /// Get index statistics
    pub fn stats(&self) -> Option<crate::index::axis::IndexStats> {
        self.axis_index.as_ref().map(|index| index.stats())
    }
}

/// Search algorithm factory
pub struct SearchAlgorithmFactory;

impl SearchAlgorithmFactory {
    /// Select best algorithm based on dataset characteristics
    pub fn select_algorithm(
        num_vectors: usize,
        dimension: usize,
        accuracy_target: f32,
    ) -> SearchAlgorithm {
        // Simple heuristics for algorithm selection
        if num_vectors < 10_000 {
            // Small dataset - brute force is fine
            SearchAlgorithm::BruteForce
        } else if dimension > 512 && accuracy_target < 0.95 {
            // High dimension, moderate accuracy - use LSH
            SearchAlgorithm::LSH {
                n_tables: 20,
                n_hashes: 10,
                hash_width: 1.0,
            }
        } else if num_vectors < 1_000_000 {
            // Medium dataset - use IVF
            let n_clusters = (num_vectors as f64).sqrt() as usize;
            SearchAlgorithm::IVF {
                n_clusters: n_clusters.clamp(256, 4096),
                n_probe: 32,
                enable_pq: false,
            }
        } else {
            // Large dataset - use HNSW
            SearchAlgorithm::HNSW {
                m: 16,
                ef_construction: 200,
                ef_search: 64,
            }
        }
    }

    /// Create optimal IVF configuration
    pub fn optimize_ivf_config(num_vectors: usize, recall_target: f32) -> SearchAlgorithm {
        let n_clusters = (num_vectors as f64).sqrt() as usize;
        let n_probe = if recall_target >= 0.99 {
            n_clusters / 4 // Search 25% of clusters for high recall
        } else if recall_target >= 0.95 {
            n_clusters / 10 // Search 10% for good recall
        } else {
            n_clusters / 20 // Search 5% for fast search
        };

        SearchAlgorithm::IVF {
            n_clusters: n_clusters.clamp(256, 4096),
            n_probe: n_probe.clamp(1, 256),
            enable_pq: num_vectors > 10_000_000, // Enable PQ for very large datasets
        }
    }

    /// Create optimal LSH configuration
    pub fn optimize_lsh_config(_dimension: usize, recall_target: f32) -> SearchAlgorithm {
        let (n_tables, n_hashes) = if recall_target >= 0.95 {
            (30, 12) // High recall
        } else if recall_target >= 0.85 {
            (20, 10) // Balanced
        } else {
            (10, 8) // Fast search
        };

        SearchAlgorithm::LSH {
            n_tables,
            n_hashes,
            hash_width: 1.0,
        }
    }
}
/// Execute vector search with given parameters
pub async fn execute_search(
    _vector_service: &VectorOperationsService,
    _params: &VectorSearchConfig,
) -> Result<VectorSearchResult> {
    // Placeholder implementation - delegates to vector service
    Err(anyhow!("Vector search not yet implemented"))
}
