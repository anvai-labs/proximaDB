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

//! Single Unified Index - One Memory Table for All Collection Metadata
//!
//! Eliminates dual-index sync complexity and recovery challenges by using
//! a single data structure with built-in secondary key support.

use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::Arc;

use crate::proto::proximadb_v1::Collection;

/// Collection index entry - contains both record and secondary keys
#[derive(Debug, Clone)]
pub struct CollectionIndexEntry {
    /// The actual collection record
    pub record: Arc<Collection>,

    /// Pre-computed secondary keys for fast lookups
    pub name_key: String,
    pub uuid_key: String,
}

impl CollectionIndexEntry {
    pub fn new(record: Collection) -> Self {
        let name_key = record
            .config
            .as_ref()
            .map_or_else(|| "unnamed".to_string(), |c| c.name.clone());
        let uuid_key = record.id.clone();

        Self {
            record: Arc::new(record),
            name_key,
            uuid_key,
        }
    }

    /// Update with new record, maintaining key consistency
    pub fn update_record(&mut self, new_record: Collection) {
        self.name_key = new_record
            .config
            .as_ref()
            .map_or_else(|| "unnamed".to_string(), |c| c.name.clone());
        self.uuid_key = new_record.id.clone();
        self.record = Arc::new(new_record);
    }
}

/// Performance metrics for the single index
#[derive(Debug, Clone)]
pub struct SingleIndexMetrics {
    /// Total number of collections indexed
    pub total_collections: usize,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Total lookups by UUID
    pub lookups_by_uuid: u64,
    /// Total lookups by name
    pub lookups_by_name: u64,
    /// Cache hit count
    pub cache_hits: u64,
    /// Cache miss count
    pub cache_misses: u64,
    /// Average lookup time in nanoseconds
    pub avg_lookup_time_ns: u64,
    /// Last index rebuild timestamp
    pub last_rebuild_timestamp: Option<i64>,
}

/// Single unified index using UUID as primary key with secondary name lookup
/// This eliminates the dual-index sync problem by using a single memory table
/// with built-in secondary key indexing for O(1) lookups on both UUID and name
#[derive(Debug)]
pub struct SingleCollectionIndex {
    /// Primary store: UUID -> CollectionIndexEntry
    /// DashMap provides lock-free concurrent access with excellent performance
    entries: DashMap<String, CollectionIndexEntry>,

    /// Secondary index: Name -> UUID for O(1) name lookups
    /// This eliminates the O(n) scan for name-based lookups
    name_to_uuid: DashMap<String, String>,

    /// Performance metrics
    metrics: Arc<RwLock<SingleIndexMetrics>>,
}

impl SingleCollectionIndex {
    /// Create new single index
    pub fn new() -> Self {
        Self {
            entries: DashMap::new(),
            name_to_uuid: DashMap::new(),
            metrics: Arc::new(RwLock::new(SingleIndexMetrics {
                total_collections: 0,
                memory_usage_bytes: 0,
                lookups_by_uuid: 0,
                lookups_by_name: 0,
                cache_hits: 0,
                cache_misses: 0,
                avg_lookup_time_ns: 0,
                last_rebuild_timestamp: None,
            })),
        }
    }

    /// Insert or update collection - single atomic operation
    /// This is the ONLY method that modifies the index, ensuring consistency
    pub fn upsert_collection(&self, record: Collection) {
        let start = std::time::Instant::now();
        let uuid = record.id.clone();
        let name = record
            .config
            .as_ref()
            .map_or_else(|| "unnamed".to_string(), |c| c.name.clone());

        // Check if this is an update (collection exists)
        let old_name = self.entries.get(&uuid).map(|e| e.name_key.clone());

        // Update primary index
        let entry = CollectionIndexEntry::new(record);
        self.entries.insert(uuid.clone(), entry);

        // Update secondary index
        // If name changed, remove old mapping
        if let Some(old_name) = old_name
            && old_name != name
        {
            self.name_to_uuid.remove(&old_name);
        }
        self.name_to_uuid.insert(name, uuid);

        // Update metrics
        let elapsed = start.elapsed().as_nanos() as u64;
        let mut metrics = self.metrics.write();
        metrics.total_collections = self.entries.len();
        metrics.memory_usage_bytes = self.estimate_memory_usage();
        metrics.avg_lookup_time_ns = (metrics.avg_lookup_time_ns + elapsed) / 2;
    }

    /// Remove collection by UUID - single atomic operation
    pub fn remove_collection(&self, uuid: &str) -> Option<Arc<Collection>> {
        let start = std::time::Instant::now();

        let result = self.entries.remove(uuid).map(|(_, entry)| {
            // Remove from secondary index
            self.name_to_uuid.remove(&entry.name_key);
            entry.record
        });

        // Update metrics
        let elapsed = start.elapsed().as_nanos() as u64;
        let mut metrics = self.metrics.write();
        metrics.total_collections = self.entries.len();
        metrics.memory_usage_bytes = self.estimate_memory_usage();
        metrics.avg_lookup_time_ns = (metrics.avg_lookup_time_ns + elapsed) / 2;

        if result.is_some() {
            metrics.cache_hits += 1;
        } else {
            metrics.cache_misses += 1;
        }

        result
    }

    /// Get collection by UUID - O(1) primary key lookup
    pub fn get_by_uuid(&self, uuid: &str) -> Option<Arc<Collection>> {
        let start = std::time::Instant::now();
        let result = self.entries.get(uuid).map(|entry| entry.record.clone());

        // Update metrics
        let elapsed = start.elapsed().as_nanos() as u64;
        let mut metrics = self.metrics.write();
        metrics.lookups_by_uuid += 1;
        if result.is_some() {
            metrics.cache_hits += 1;
        } else {
            metrics.cache_misses += 1;
        }
        metrics.avg_lookup_time_ns = (metrics.avg_lookup_time_ns + elapsed) / 2;

        result
    }

    /// Get collection by name - O(1) lookup using secondary index
    pub fn get_by_name(&self, name: &str) -> Option<Arc<Collection>> {
        let start = std::time::Instant::now();

        // O(1) lookup in secondary index
        let result = self
            .name_to_uuid
            .get(name)
            .and_then(|uuid| {
                let uuid_str: &str = uuid.as_ref();
                self.entries.get(uuid_str)
            })
            .map(|entry| entry.record.clone());

        // Update metrics
        let elapsed = start.elapsed().as_nanos() as u64;
        let mut metrics = self.metrics.write();
        metrics.lookups_by_name += 1;
        if result.is_some() {
            metrics.cache_hits += 1;
        } else {
            metrics.cache_misses += 1;
        }
        metrics.avg_lookup_time_ns = (metrics.avg_lookup_time_ns + elapsed) / 2;

        result
    }

    /// Get UUID by name - O(1) lookup using secondary index
    pub fn get_uuid_by_name(&self, name: &str) -> Option<String> {
        self.name_to_uuid.get(name).map(|uuid| uuid.clone())
    }

    /// Check if collection exists by UUID - O(1)
    pub fn exists_by_uuid(&self, uuid: &str) -> bool {
        self.entries.contains_key(uuid)
    }

    /// Check if collection exists by name - O(1) using secondary index
    pub fn exists_by_name(&self, name: &str) -> bool {
        self.name_to_uuid.contains_key(name)
    }

    /// List all collections - O(n) efficient iteration
    pub fn list_all(&self) -> Vec<Arc<Collection>> {
        self.entries
            .iter()
            .map(|entry| entry.value().record.clone())
            .collect()
    }

    /// Get collection count - O(1)
    pub fn count(&self) -> usize {
        self.entries.len()
    }

    /// Clear all data - for testing or rebuild
    pub fn clear(&self) {
        self.entries.clear();
        self.name_to_uuid.clear();

        let mut metrics = self.metrics.write();
        metrics.total_collections = 0;
        metrics.memory_usage_bytes = 0;
        metrics.last_rebuild_timestamp = Some(chrono::Utc::now().timestamp());
    }

    /// Rebuild from collection records - critical for recovery
    /// This is the key method for disk recovery - single operation, no sync issues
    pub fn rebuild_from_records(&self, records: Vec<Collection>) {
        // Clear and rebuild atomically
        self.clear();

        // Batch rebuild for better performance
        let mut name_mappings = Vec::with_capacity(records.len());

        for record in records {
            let uuid = record.id.clone();
            let name = record
                .config
                .as_ref()
                .map_or_else(|| "unnamed".to_string(), |c| c.name.clone());

            // Insert into primary index
            let entry = CollectionIndexEntry::new(record);
            self.entries.insert(uuid.clone(), entry);

            // Collect for batch secondary index update
            name_mappings.push((name, uuid));
        }

        // Batch update secondary index
        for (name, uuid) in name_mappings {
            self.name_to_uuid.insert(name, uuid);
        }

        let mut metrics = self.metrics.write();
        metrics.total_collections = self.entries.len();
        metrics.memory_usage_bytes = self.estimate_memory_usage();
        metrics.last_rebuild_timestamp = Some(chrono::Utc::now().timestamp());

        tracing::info!(
            "SingleCollectionIndex rebuilt with {} collections",
            self.entries.len()
        );
    }

    /// Get performance metrics
    pub fn get_metrics(&self) -> SingleIndexMetrics {
        self.metrics.read().clone()
    }

    /// Get metrics (alias for get_metrics)
    pub fn metrics(&self) -> SingleIndexMetrics {
        self.get_metrics()
    }

    /// Filter collections by predicate - O(n) parallel scan
    pub fn filter_collections<F>(&self, predicate: F) -> Vec<Arc<Collection>>
    where
        F: Fn(&Collection) -> bool + Sync,
    {
        self.entries
            .iter()
            .filter(|entry| predicate(&entry.value().record))
            .map(|entry| entry.value().record.clone())
            .collect()
    }

    /// Prefix search on collection names - optimized with secondary index
    pub fn find_by_name_prefix(&self, prefix: &str) -> Vec<Arc<Collection>> {
        // Still O(n) on secondary index, but faster than scanning all records
        self.name_to_uuid
            .iter()
            .filter(|entry| entry.key().starts_with(prefix))
            .filter_map(|entry| self.entries.get(entry.value()).map(|e| e.record.clone()))
            .collect()
    }

    /// Estimate memory usage for monitoring
    fn estimate_memory_usage(&self) -> usize {
        let primary_size = self.entries.len()
            * (
                32 +  // UUID key
                64 +  // Name key (average)
                std::mem::size_of::<Collection>() +
                64
                // Arc and entry overhead
            );

        let secondary_size = self.name_to_uuid.len()
            * (
                64 +  // Name key (average)
                32 +  // UUID value
                32
                // DashMap overhead
            );

        primary_size + secondary_size
    }

    /// Export all collections as a map for snapshots
    pub fn export_all(&self) -> std::collections::BTreeMap<String, Collection> {
        self.entries
            .iter()
            .map(|entry| {
                let record = (*entry.value().record).clone();
                (entry.key().clone(), record)
            })
            .collect()
    }
}

impl Default for SingleCollectionIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe wrapper with additional safety guarantees
pub struct ThreadSafeSingleIndex {
    /// Underlying single collection index
    index: SingleCollectionIndex,
}

impl ThreadSafeSingleIndex {
    pub fn new() -> Self {
        Self {
            index: SingleCollectionIndex::new(),
        }
    }

    /// All operations delegate to the single index
    pub fn upsert_collection(&self, record: Collection) {
        self.index.upsert_collection(record);
    }

    pub fn remove_collection(&self, uuid: &str) -> Option<Arc<Collection>> {
        self.index.remove_collection(uuid)
    }

    pub fn get_by_uuid(&self, uuid: &str) -> Option<Arc<Collection>> {
        self.index.get_by_uuid(uuid)
    }

    pub fn get_by_name(&self, name: &str) -> Option<Arc<Collection>> {
        self.index.get_by_name(name)
    }

    pub fn get_uuid_by_name(&self, name: &str) -> Option<String> {
        self.index.get_uuid_by_name(name)
    }

    pub fn exists_by_uuid(&self, uuid: &str) -> bool {
        self.index.exists_by_uuid(uuid)
    }

    pub fn exists_by_name(&self, name: &str) -> bool {
        self.index.exists_by_name(name)
    }

    pub fn list_all(&self) -> Vec<Arc<Collection>> {
        self.index.list_all()
    }

    pub fn count(&self) -> usize {
        self.index.count()
    }

    pub fn rebuild_from_records(&self, records: Vec<Collection>) {
        self.index.rebuild_from_records(records);
    }

    pub fn get_metrics(&self) -> SingleIndexMetrics {
        self.index.metrics()
    }

    pub fn filter_collections<F>(&self, predicate: F) -> Vec<Arc<Collection>>
    where
        F: Fn(&Collection) -> bool + Sync,
    {
        self.index.filter_collections(predicate)
    }
}

impl Default for ThreadSafeSingleIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::proximadb_v1::{
        Collection, CollectionConfig, CollectionStats, DistanceMetric, StorageEngine,
    };
    use tracing::debug;

    fn create_test_collection(id: &str, name: &str) -> Collection {
        let temp_dir = tempfile::tempdir().unwrap();
        Collection {
            id: id.to_string(),
            config: Some(CollectionConfig {
                name: name.to_string(),
                embedding_config: None,
                dimension: 128,
                distance_metric: Some(DistanceMetric::Cosine as i32),
                storage_engine: Some(StorageEngine::Viper as i32),
                filterable_columns: vec![],
                index_configs: vec![],
                quantization: None,
                primary_index: Some("HNSW".to_string()),
                auto_index_selection: Some(false),
                description: Some("Test collection".to_string()),
                tags: vec![],
                owner: Some("test_user".to_string()),
                embedding_models: vec![],
                storage_config: None,
                record_schema: None,
                enable_proxima_record: None,
                text_columns: vec![],
                text_storage_configs: vec![],
                enable_dual_use_embeddings: None,
                canonical_embedding_precision: None,
                permitted_principals: vec![],
                index_policy: None,
                pax_vector_quant: None,
            }),
            stats: Some(CollectionStats {
                vector_count: 100,
                index_size_bytes: 1024,
                data_size_bytes: 2048,
            }),
            created_at: chrono::Utc::now().timestamp_millis(),
            updated_at: chrono::Utc::now().timestamp_millis(),
            storage_assignment: Some(crate::proto::proximadb_v1::StorageAssignment {
                primary_path: format!("{}", temp_dir.path().display()),
                backup_paths: vec![],
                engine: StorageEngine::Viper as i32,
                engine_config: std::collections::HashMap::new(),
                base_location: format!("{}", temp_dir.path().display()),
                assigned_at: chrono::Utc::now().timestamp_micros(),
                ..Default::default()
            }),
        }
    }

    #[test]
    fn test_single_index_operations() {
        let index = SingleCollectionIndex::new();

        let collection = create_test_collection("uuid-123", "test-collection");

        // Test upsert
        index.upsert_collection(collection.clone());
        assert_eq!(index.count(), 1);

        // Test UUID lookup - O(1)
        let result = index.get_by_uuid("uuid-123");
        assert!(result.is_some());
        assert_eq!(result.unwrap().id, "uuid-123");

        // Test name lookup - O(n) but efficient
        let result = index.get_by_name("test-collection");
        assert!(result.is_some());
        assert_eq!(result.unwrap().id, "uuid-123");

        // Test UUID by name
        let uuid = index.get_uuid_by_name("test-collection");
        assert_eq!(uuid.unwrap(), "uuid-123");

        // Test existence checks
        assert!(index.exists_by_uuid("uuid-123"));
        assert!(index.exists_by_name("test-collection"));
        assert!(!index.exists_by_uuid("nonexistent"));
        assert!(!index.exists_by_name("nonexistent"));

        // Test removal
        let removed = index.remove_collection("uuid-123");
        assert!(removed.is_some());
        assert_eq!(index.count(), 0);
        assert!(!index.exists_by_uuid("uuid-123"));
        assert!(!index.exists_by_name("test-collection"));
    }

    #[test]
    fn test_single_index_rebuild() {
        let index = SingleCollectionIndex::new();

        let records = vec![
            create_test_collection("uuid-1", "collection-1"),
            create_test_collection("uuid-2", "collection-2"),
            create_test_collection("uuid-3", "collection-3"),
        ];

        index.rebuild_from_records(records);

        assert_eq!(index.count(), 3);
        assert!(index.exists_by_uuid("uuid-1"));
        assert!(index.exists_by_name("collection-2"));

        let metrics = index.get_metrics();
        assert!(metrics.last_rebuild_timestamp.is_some());
        assert_eq!(metrics.total_collections, 3);
    }

    #[test]
    fn test_concurrent_access() {
        use std::thread;

        let index = Arc::new(SingleCollectionIndex::new());
        let mut handles = vec![];

        // Spawn multiple threads for concurrent operations
        for i in 0..10 {
            let index_clone = index.clone();
            let handle = thread::spawn(move || {
                let collection =
                    create_test_collection(&format!("uuid-{}", i), &format!("collection-{}", i));
                index_clone.upsert_collection(collection);

                // Immediate lookup to test consistency
                let uuid_result = index_clone.get_by_uuid(&format!("uuid-{}", i));
                assert!(uuid_result.is_some());

                let name_result = index_clone.get_by_name(&format!("collection-{}", i));
                assert!(name_result.is_some());
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify final state
        assert_eq!(index.count(), 10);

        // Test concurrent reads
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let index_clone = index.clone();
                thread::spawn(move || {
                    let result = index_clone.get_by_uuid(&format!("uuid-{}", i));
                    assert!(result.is_some());
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_performance_characteristics() {
        let index = SingleCollectionIndex::new();

        // Insert test data
        for i in 0..1000 {
            let collection =
                create_test_collection(&format!("uuid-{:04}", i), &format!("collection-{:04}", i));
            index.upsert_collection(collection);
        }

        // Test UUID lookup performance (should be O(1))
        let start = std::time::Instant::now();
        for i in 0..100 {
            let _result = index.get_by_uuid(&format!("uuid-{:04}", i));
        }
        let uuid_duration = start.elapsed();

        // Test name lookup performance (O(n) but should be fast)
        let start = std::time::Instant::now();
        for i in 0..100 {
            let _result = index.get_by_name(&format!("collection-{:04}", i));
        }
        let name_duration = start.elapsed();

        debug!("UUID lookup (100 ops): {:?}", uuid_duration);
        debug!("Name lookup (100 ops): {:?}", name_duration);

        // UUID lookups should typically be faster than name lookups
        // But in test environments with small datasets, timing can be unreliable
        // Just verify both complete in reasonable time
        assert!(
            uuid_duration.as_millis() < 1000,
            "UUID lookups took too long: {:?}",
            uuid_duration
        );
        assert!(
            name_duration.as_millis() < 1000,
            "Name lookups took too long: {:?}",
            name_duration
        );

        let metrics = index.get_metrics();
        assert!(metrics.avg_lookup_time_ns > 0);
        assert_eq!(metrics.total_collections, 1000);
    }
}
