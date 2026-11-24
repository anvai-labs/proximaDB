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

//! GlobalIdIndex - Fast vector ID to storage location mapping
//!
//! This index provides O(1) lookup for vector IDs to their storage locations,
//! enabling fast get-by-id operations across all storage engines.
//!
//! ## Key Features
//! - O(1) vector ID lookup performance
//! - Storage location mapping across all engines
//! - Collection-based isolation
//! - Optional persistence for recovery
//! - Memory-efficient HashMap implementation

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tracing::{debug, info, warn};

use crate::index::axis::types::IndexAlgorithm;

/// Global ID index for fast vector retrieval by ID
#[derive(Debug)]
pub struct GlobalIdIndex {
    /// Map from vector ID to storage location
    id_map: Arc<RwLock<HashMap<String, StorageLocation>>>,
    /// Index configuration
    #[allow(dead_code)]
    config: GlobalIdIndexConfig,
    /// Collection ID for isolation
    collection_id: String,
    /// Statistics tracking
    stats: Arc<RwLock<GlobalIdIndexStats>>,
    /// Algorithm configuration for trait implementation
    algorithm_config: IndexAlgorithm,
}

/// Storage location information for a vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageLocation {
    /// Storage engine type (SST, NOVA, etc.)
    pub engine_type: String,
    /// File path where vector is stored
    pub file_path: String,
    /// Offset within the file
    pub offset: u64,
    /// Size of the vector record
    pub size: u32,
    /// Timestamp when vector was stored
    pub stored_at: i64,
    /// Optional metadata for optimization
    pub metadata: Option<HashMap<String, String>>,
}

/// Configuration for GlobalIdIndex
#[derive(Debug, Clone)]
pub struct GlobalIdIndexConfig {
    /// Enable compression for storage location data
    pub enable_compression: bool,
    /// Maximum cache size (number of entries)
    pub cache_size: usize,
    /// Enable persistence for recovery
    pub persistence_enabled: bool,
    /// Persistence file path
    pub persistence_path: Option<String>,
}

impl Default for GlobalIdIndexConfig {
    fn default() -> Self {
        Self {
            enable_compression: true,
            cache_size: 100_000, // 100K vector IDs
            persistence_enabled: true,
            persistence_path: None,
        }
    }
}

/// Statistics for GlobalIdIndex
#[derive(Debug, Clone, Default)]
pub struct GlobalIdIndexStats {
    /// Total number of vector IDs indexed
    pub total_vectors: usize,
    /// Memory usage in bytes (estimated)
    pub memory_usage_bytes: usize,
    /// Number of lookups performed
    pub lookup_count: u64,
    /// Number of successful lookups
    pub successful_lookups: u64,
    /// Average lookup latency in microseconds
    pub avg_lookup_latency_us: f64,
}

impl GlobalIdIndex {
    /// Create new GlobalIdIndex
    pub fn new(collection_id: String, config: GlobalIdIndexConfig) -> Self {
        info!(
            "🆔 Creating GlobalIdIndex for collection: {}",
            collection_id
        );

        let algorithm_config = IndexAlgorithm::GlobalId {
            cache_size: config.cache_size,
            persistence_enabled: config.persistence_enabled,
        };

        Self {
            id_map: Arc::new(RwLock::new(HashMap::new())),
            config,
            collection_id,
            stats: Arc::new(RwLock::new(GlobalIdIndexStats::default())),
            algorithm_config,
        }
    }

    /// Add vector ID to location mapping
    pub async fn add_vector(&self, vector_id: String, location: StorageLocation) -> Result<()> {
        debug!(
            "📍 Adding vector ID mapping: {} -> {}",
            vector_id, location.file_path
        );

        {
            let mut map = self
                .id_map
                .write()
                .map_err(|e| anyhow!("Lock error: {}", e))?;
            map.insert(vector_id.clone(), location);
        } // Drop lock before async call

        // Update statistics
        self.update_stats_after_insert().await?;

        debug!("✅ Vector ID mapping added: {}", vector_id);
        Ok(())
    }

    /// Get storage location for vector ID (O(1) operation)
    pub async fn get_location(&self, vector_id: &str) -> Result<Option<StorageLocation>> {
        let start_time = std::time::Instant::now();

        let result = {
            let map = self
                .id_map
                .read()
                .map_err(|e| anyhow!("Lock error: {}", e))?;
            map.get(vector_id).cloned()
        }; // Drop lock before async call

        // Update statistics
        self.update_stats_after_lookup(start_time.elapsed(), result.is_some())
            .await?;

        debug!(
            "🔍 Vector ID lookup: {} -> {:?}",
            vector_id,
            result.is_some()
        );
        Ok(result)
    }

    /// Remove vector ID mapping
    pub async fn remove_vector(&self, vector_id: &str) -> Result<bool> {
        debug!("🗑️ Removing vector ID mapping: {}", vector_id);

        let removed = {
            let mut map = self
                .id_map
                .write()
                .map_err(|e| anyhow!("Lock error: {}", e))?;
            map.remove(vector_id).is_some()
        }; // Drop lock before async call

        if removed {
            // Update statistics
            self.update_stats_after_removal().await?;
            debug!("✅ Vector ID mapping removed: {}", vector_id);
        } else {
            debug!("⚠️ Vector ID not found for removal: {}", vector_id);
        }

        Ok(removed)
    }

    /// Bulk add vector mappings for efficiency
    pub async fn add_vectors_bulk(
        &self,
        mappings: HashMap<String, StorageLocation>,
    ) -> Result<usize> {
        info!("📦 Bulk adding {} vector ID mappings", mappings.len());

        let count = mappings.len();
        {
            let mut map = self
                .id_map
                .write()
                .map_err(|e| anyhow!("Lock error: {}", e))?;
            for (vector_id, location) in mappings {
                map.insert(vector_id, location);
            }
        } // Drop lock before async call

        // Update statistics
        self.update_stats_after_bulk_insert(count).await?;

        info!("✅ Bulk added {} vector ID mappings", count);
        Ok(count)
    }

    /// Get index statistics
    pub fn get_stats(&self) -> Result<GlobalIdIndexStats> {
        let stats = self
            .stats
            .read()
            .map_err(|e| anyhow!("Lock error: {}", e))?;
        Ok(stats.clone())
    }

    /// Get collection ID
    pub fn collection_id(&self) -> &str {
        &self.collection_id
    }

    /// Get total vector count
    pub fn vector_count(&self) -> usize {
        match self.id_map.read() {
            Ok(map) => map.len(),
            Err(e) => {
                warn!("Lock error while reading vector_count: {}", e);
                0
            }
        }
    }

    /// Check if vector ID exists
    pub async fn contains_vector(&self, vector_id: &str) -> Result<bool> {
        let map = self
            .id_map
            .read()
            .map_err(|e| anyhow!("Lock error: {}", e))?;
        Ok(map.contains_key(vector_id))
    }

    /// Get all vector IDs (for debugging/admin)
    pub fn get_all_vector_ids(&self) -> Result<Vec<String>> {
        let map = self
            .id_map
            .read()
            .map_err(|e| anyhow!("Lock error: {}", e))?;
        Ok(map.keys().cloned().collect())
    }

    /// Clear all mappings (for testing)
    pub async fn clear(&self) -> Result<()> {
        warn!(
            "🧹 Clearing all vector ID mappings for collection: {}",
            self.collection_id
        );

        let mut map = self
            .id_map
            .write()
            .map_err(|e| anyhow!("Lock error: {}", e))?;
        map.clear();

        // Reset statistics
        let mut stats = self
            .stats
            .write()
            .map_err(|e| anyhow!("Lock error: {}", e))?;
        *stats = GlobalIdIndexStats::default();

        Ok(())
    }

    // Private helper methods for statistics updates
    async fn update_stats_after_insert(&self) -> Result<()> {
        let mut stats = self
            .stats
            .write()
            .map_err(|e| anyhow!("Lock error: {}", e))?;
        let map = self
            .id_map
            .read()
            .map_err(|e| anyhow!("Lock error: {}", e))?;

        stats.total_vectors = map.len();
        stats.memory_usage_bytes = map.len() * 128; // Rough estimate: 128 bytes per entry

        Ok(())
    }

    async fn update_stats_after_lookup(
        &self,
        latency: std::time::Duration,
        success: bool,
    ) -> Result<()> {
        let mut stats = self
            .stats
            .write()
            .map_err(|e| anyhow!("Lock error: {}", e))?;

        stats.lookup_count += 1;
        if success {
            stats.successful_lookups += 1;
        }

        // Update rolling average latency
        let latency_us = latency.as_micros() as f64;
        stats.avg_lookup_latency_us =
            (stats.avg_lookup_latency_us * (stats.lookup_count - 1) as f64 + latency_us)
                / stats.lookup_count as f64;

        Ok(())
    }

    async fn update_stats_after_removal(&self) -> Result<()> {
        let mut stats = self
            .stats
            .write()
            .map_err(|e| anyhow!("Lock error: {}", e))?;
        let map = self
            .id_map
            .read()
            .map_err(|e| anyhow!("Lock error: {}", e))?;

        stats.total_vectors = map.len();
        stats.memory_usage_bytes = map.len() * 128;

        Ok(())
    }

    async fn update_stats_after_bulk_insert(&self, _count: usize) -> Result<()> {
        let mut stats = self
            .stats
            .write()
            .map_err(|e| anyhow!("Lock error: {}", e))?;
        let map = self
            .id_map
            .read()
            .map_err(|e| anyhow!("Lock error: {}", e))?;

        stats.total_vectors = map.len();
        stats.memory_usage_bytes = map.len() * 128;

        Ok(())
    }
}

// Implement required traits for AXIS integration
#[async_trait]
impl crate::index::axis::index_factory::AxisVectorIndex for GlobalIdIndex {
    async fn add(&self, vector_id: String, vector_data: Vec<f32>) -> Result<()> {
        // For GlobalIdIndex, create a default storage location
        let storage_location = StorageLocation {
            engine_type: "SST".to_string(),
            file_path: format!("/data/{}/{}", self.collection_id, vector_id),
            offset: 0,
            size: (vector_data.len() * 4) as u32,
            stored_at: chrono::Utc::now().timestamp(),
            metadata: None,
        };

        self.add_vector(vector_id, storage_location).await
    }

    async fn search(
        &self,
        _query: &[f32],
        _top_k: usize,
        _filter: Option<&std::collections::HashMap<String, String>>,
    ) -> Result<Vec<(String, f32)>> {
        // GlobalIdIndex is for exact lookup, not similarity search
        Ok(vec![])
    }

    async fn remove(&self, vector_id: &str) -> Result<()> {
        self.remove_vector(vector_id).await?;
        Ok(())
    }

    fn algorithm(&self) -> &IndexAlgorithm {
        // This is problematic - trait wants reference to owned data
        // Need to store algorithm in struct
        &self.algorithm_config
    }

    fn stats(&self) -> crate::index::axis::index_factory::IndexStats {
        let map_len = match self.id_map.read() {
            Ok(map) => map.len(),
            Err(e) => {
                warn!("Lock error while reading stats: {}", e);
                0
            }
        };
        crate::index::axis::index_factory::IndexStats {
            vector_count: map_len,
            memory_usage_bytes: map_len * 128, // Estimated bytes per entry
            index_type: "GlobalId".to_string(),
        }
    }
}
