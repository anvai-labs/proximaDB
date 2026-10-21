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

//! Common utilities for AXIS index implementations
//!
//! This module provides reusable, high-performance data structures and utilities
//! that are shared across all AXIS index implementations. The focus is on:
//!
//! - Lock-free concurrent data structures using DashMap
//! - Atomic counters for statistics
//! - Vector storage with consistent patterns
//! - ID mapping utilities
//! - Memory usage estimation helpers

use anyhow::Result;
use dashmap::DashMap;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::proto::proximadb_v1::MetadataItem;
use proximadb_records::ProximaRecord;

/// High-performance concurrent vector storage
/// Used by INDEX implementations to store vector data for search operations
/// NOTE: This is different from cache storage - indexes organize data for search,
/// caches store computed results for fast repeated access.
#[derive(Debug)]
pub struct IndexVectorStore {
    /// Vector data storage: vector_id -> ProximaRecord
    vectors: DashMap<String, Arc<ProximaRecord>>,
    /// Count of stored vectors (atomic for lock-free updates)
    count: AtomicUsize,
    /// Total dimension (for validation and memory estimation)
    dimension: usize,
}

impl IndexVectorStore {
    /// Create a new concurrent vector store
    pub fn new(dimension: usize) -> Self {
        Self {
            vectors: DashMap::new(),
            count: AtomicUsize::new(0),
            dimension,
        }
    }

    /// Insert a vector record
    pub fn insert(&self, id: String, vector: Arc<ProximaRecord>) -> Result<()> {
        let vec_dim = vector.embeddings.first().map_or(0, |e| e.values.len());
        if vec_dim != self.dimension {
            return Err(anyhow::anyhow!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dimension,
                vec_dim
            ));
        }

        let prev = self.vectors.insert(id, vector);
        if prev.is_none() {
            self.count.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Get a vector by ID
    pub fn get(&self, id: &str) -> Option<Arc<ProximaRecord>> {
        self.vectors.get(id).map(|entry| entry.value().clone())
    }

    /// Remove a vector by ID
    pub fn remove(&self, id: &str) -> Option<Arc<ProximaRecord>> {
        if let Some((_, removed)) = self.vectors.remove(id) {
            self.count.fetch_sub(1, Ordering::Relaxed);
            Some(removed)
        } else {
            None
        }
    }

    /// Check if a vector exists
    pub fn contains(&self, id: &str) -> bool {
        self.vectors.contains_key(id)
    }

    /// Get current vector count (lock-free)
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    /// Check if store is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Estimate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let vector_data = self.len() * self.dimension * std::mem::size_of::<f32>();
        let metadata_estimate = self.len() * 200; // Rough estimate for metadata
        let overhead =
            self.vectors.capacity() * std::mem::size_of::<(String, Arc<ProximaRecord>)>();
        vector_data + metadata_estimate + overhead
    }

    /// Get all vector IDs (for iteration)
    pub fn keys(&self) -> Vec<String> {
        self.vectors
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Bidirectional ID mapping for indexes that use internal node IDs
/// Used by HNSW and other graph-based indexes
#[derive(Debug)]
pub struct ConcurrentIdMapping {
    /// External ID -> Internal ID
    external_to_internal: DashMap<String, usize>,
    /// Internal ID -> External ID
    internal_to_external: DashMap<usize, String>,
    /// Next available internal ID
    next_id: AtomicUsize,
}

impl ConcurrentIdMapping {
    /// Create a new ID mapping
    pub fn new() -> Self {
        Self {
            external_to_internal: DashMap::new(),
            internal_to_external: DashMap::new(),
            next_id: AtomicUsize::new(0),
        }
    }

    /// Register a new external ID and get its internal ID
    pub fn register(&self, external_id: String) -> Result<usize> {
        // Check if already exists
        if let Some(entry) = self.external_to_internal.get(&external_id) {
            return Ok(*entry.value());
        }

        // Allocate new internal ID
        let internal_id = self.next_id.fetch_add(1, Ordering::Relaxed);

        // Insert both mappings
        self.external_to_internal
            .insert(external_id.clone(), internal_id);
        self.internal_to_external.insert(internal_id, external_id);

        Ok(internal_id)
    }

    /// Get internal ID for external ID
    pub fn internal(&self, external_id: &str) -> Option<usize> {
        self.external_to_internal
            .get(external_id)
            .map(|entry| *entry.value())
    }

    /// Get external ID for internal ID
    pub fn external(&self, internal_id: usize) -> Option<String> {
        self.internal_to_external
            .get(&internal_id)
            .map(|entry| entry.value().clone())
    }

    /// Remove mapping by external ID
    pub fn remove_by_external(&self, external_id: &str) -> Option<usize> {
        if let Some((_, internal_id)) = self.external_to_internal.remove(external_id) {
            self.internal_to_external.remove(&internal_id);
            Some(internal_id)
        } else {
            None
        }
    }

    /// Remove mapping by internal ID
    pub fn remove_by_internal(&self, internal_id: usize) -> Option<String> {
        if let Some((_, external_id)) = self.internal_to_external.remove(&internal_id) {
            self.external_to_internal.remove(&external_id);
            Some(external_id)
        } else {
            None
        }
    }

    /// Get current count of mappings
    pub fn len(&self) -> usize {
        self.external_to_internal.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.external_to_internal.is_empty()
    }

    // ============================================================================
    // SERIALIZATION HELPER METHODS
    // ============================================================================

    /// Iterate over external->internal mappings (for serialization)
    pub fn iter_external_to_internal(&self) -> impl Iterator<Item = (String, usize)> + '_ {
        self.external_to_internal
            .iter()
            .map(|entry| (entry.key().clone(), *entry.value()))
    }

    /// Get next available ID value (for serialization)
    pub fn next_id(&self) -> usize {
        self.next_id.load(Ordering::Relaxed)
    }

    /// Restore a mapping directly (for deserialization)
    /// This bypasses the auto-increment logic of register()
    pub fn restore_mapping(&self, external_id: String, internal_id: usize) -> Result<()> {
        self.external_to_internal
            .insert(external_id.clone(), internal_id);
        self.internal_to_external.insert(internal_id, external_id);
        Ok(())
    }

    /// Set the next_id counter (for deserialization)
    pub fn set_next_id(&self, next_id: usize) {
        self.next_id.store(next_id, Ordering::Relaxed);
    }
}

impl Default for ConcurrentIdMapping {
    fn default() -> Self {
        Self::new()
    }
}

/// Atomic statistics tracker for index performance monitoring
#[derive(Debug)]
pub struct AtomicStats {
    /// Total number of operations performed
    operations: AtomicUsize,
    /// Number of successful operations
    successful: AtomicUsize,
    /// Number of failed operations
    failed: AtomicUsize,
    /// Cumulative processing time in microseconds
    total_time_us: AtomicUsize,
}

impl AtomicStats {
    /// Create new atomic stats
    pub fn new() -> Self {
        Self {
            operations: AtomicUsize::new(0),
            successful: AtomicUsize::new(0),
            failed: AtomicUsize::new(0),
            total_time_us: AtomicUsize::new(0),
        }
    }

    /// Record a successful operation
    pub fn record_success(&self, duration_us: u64) {
        self.operations.fetch_add(1, Ordering::Relaxed);
        self.successful.fetch_add(1, Ordering::Relaxed);
        self.total_time_us
            .fetch_add(duration_us as usize, Ordering::Relaxed);
    }

    /// Record a failed operation
    pub fn record_failure(&self, duration_us: u64) {
        self.operations.fetch_add(1, Ordering::Relaxed);
        self.failed.fetch_add(1, Ordering::Relaxed);
        self.total_time_us
            .fetch_add(duration_us as usize, Ordering::Relaxed);
    }

    /// Get success rate (0.0 to 1.0)
    pub fn success_rate(&self) -> f64 {
        let total = self.operations.load(Ordering::Relaxed);
        if total == 0 {
            return 1.0;
        }
        self.successful.load(Ordering::Relaxed) as f64 / total as f64
    }

    /// Get average operation time in microseconds
    pub fn avg_time_us(&self) -> f64 {
        let total = self.operations.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        self.total_time_us.load(Ordering::Relaxed) as f64 / total as f64
    }

    /// Get total operations count
    pub fn total_operations(&self) -> usize {
        self.operations.load(Ordering::Relaxed)
    }

    /// Get successful operations count
    pub fn successful_operations(&self) -> usize {
        self.successful.load(Ordering::Relaxed)
    }

    /// Get failed operations count
    pub fn failed_operations(&self) -> usize {
        self.failed.load(Ordering::Relaxed)
    }
}

impl Default for AtomicStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for metadata conversion
pub mod metadata {
    use super::*;

    /// Convert proto MetadataItem vector to JSON HashMap
    pub fn proto_to_json(metadata: Vec<MetadataItem>) -> Option<HashMap<String, JsonValue>> {
        if metadata.is_empty() {
            return None;
        }

        let mut map = HashMap::new();
        for item in metadata {
            if let Some(value) = item.value {
                let json_value = match value {
                    crate::proto::proximadb_v1::metadata_item::Value::StringValue(s) => {
                        JsonValue::String(s)
                    }
                    crate::proto::proximadb_v1::metadata_item::Value::NumberValue(f) => {
                        serde_json::Number::from_f64(f).map_or(JsonValue::Null, JsonValue::Number)
                    }
                    crate::proto::proximadb_v1::metadata_item::Value::BoolValue(b) => {
                        JsonValue::Bool(b)
                    }
                };
                map.insert(item.key, json_value);
            }
        }
        Some(map)
    }

    /// Convert JSON HashMap to proto MetadataItem vector
    pub fn json_to_proto(metadata: Option<&HashMap<String, JsonValue>>) -> Vec<MetadataItem> {
        match metadata {
            Some(map) => map
                .iter()
                .filter_map(|(key, value)| {
                    let proto_value = match value {
                        JsonValue::String(s) => Some(
                            crate::proto::proximadb_v1::metadata_item::Value::StringValue(
                                s.clone(),
                            ),
                        ),
                        JsonValue::Number(n) => n
                            .as_f64()
                            .map(crate::proto::proximadb_v1::metadata_item::Value::NumberValue),
                        JsonValue::Bool(b) => Some(
                            crate::proto::proximadb_v1::metadata_item::Value::BoolValue(*b),
                        ),
                        _ => None,
                    };

                    proto_value.map(|value| MetadataItem {
                        key: key.clone(),
                        value: Some(value),
                    })
                })
                .collect(),
            None => Vec::new(),
        }
    }

    /// Estimate memory usage of metadata in bytes
    pub fn estimate_memory(metadata: &Option<HashMap<String, JsonValue>>) -> usize {
        match metadata {
            Some(map) => {
                map.iter()
                    .map(|(k, v)| {
                        k.len()
                            + match v {
                                JsonValue::String(s) => s.len() + 24, // String overhead
                                JsonValue::Number(_) => 8,
                                JsonValue::Bool(_) => 1,
                                _ => 16, // Estimate for other types
                            }
                    })
                    .sum::<usize>()
                    + map.capacity() * 16 // HashMap overhead
            }
            None => 0,
        }
    }
}

/// Memory estimation utilities
pub mod memory {
    /// Estimate memory for a vector of f32
    pub fn vector_memory(dimension: usize) -> usize {
        dimension * std::mem::size_of::<f32>()
    }

    /// Estimate memory for a DashMap with given capacity
    pub fn dashmap_overhead<K, V>(capacity: usize) -> usize {
        capacity * (std::mem::size_of::<K>() + std::mem::size_of::<V>() + 16) // DashMap overhead
    }

    /// Estimate memory for Vec with capacity
    pub fn vec_memory<T>(capacity: usize) -> usize {
        capacity * std::mem::size_of::<T>()
    }
}

/// Configuration validation utilities
pub mod validation {
    use anyhow::Result;

    /// Validate vector dimension is reasonable
    pub fn validate_dimension(dimension: usize) -> Result<()> {
        if dimension == 0 {
            return Err(anyhow::anyhow!("Vector dimension must be greater than 0"));
        }
        if dimension > 100_000 {
            return Err(anyhow::anyhow!(
                "Vector dimension {} is too large (max 100,000)",
                dimension
            ));
        }
        Ok(())
    }

    /// Validate k parameter for search
    pub fn validate_k(k: usize, max_k: usize) -> Result<usize> {
        if k == 0 {
            return Err(anyhow::anyhow!("k must be greater than 0"));
        }
        Ok(k.min(max_k))
    }

    /// Validate vector ID format
    pub fn validate_vector_id(id: &str) -> Result<()> {
        if id.is_empty() {
            return Err(anyhow::anyhow!("Vector ID cannot be empty"));
        }
        if id.len() > 255 {
            return Err(anyhow::anyhow!("Vector ID too long (max 255 characters)"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concurrent_vector_store() {
        use proximadb_records::{EmbeddingCell, ProximaRecord};

        let store = IndexVectorStore::new(3);

        let vector = Arc::new(ProximaRecord {
            oid: "test1".to_string(),
            embeddings: vec![EmbeddingCell::new_fp32(
                "default",
                "vector",
                3,
                vec![1.0, 2.0, 3.0],
            )],
            ..Default::default()
        });

        // Test insert
        assert!(store.insert("test1".to_string(), vector.clone()).is_ok());
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());

        // Test get
        assert!(store.get("test1").is_some());

        // Test remove
        assert!(store.remove("test1").is_some());
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn test_concurrent_id_mapping() {
        let mapping = ConcurrentIdMapping::new();

        // Test register
        let id1 = mapping.register("external1".to_string()).unwrap();
        let id2 = mapping.register("external2".to_string()).unwrap();

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(mapping.len(), 2);

        // Test lookups
        assert_eq!(mapping.internal("external1"), Some(0));
        assert_eq!(mapping.external(1).as_deref(), Some("external2"));

        // Test remove
        assert_eq!(mapping.remove_by_external("external1"), Some(0));
        assert_eq!(mapping.len(), 1);
    }

    #[test]
    fn test_atomic_stats() {
        let stats = AtomicStats::new();

        stats.record_success(100);
        stats.record_success(200);
        stats.record_failure(150);

        assert_eq!(stats.total_operations(), 3);
        assert_eq!(stats.successful_operations(), 2);
        assert_eq!(stats.failed_operations(), 1);
        assert!((stats.success_rate() - 0.666666).abs() < 0.001);
        assert!((stats.avg_time_us() - 150.0).abs() < 0.001);
    }
}
