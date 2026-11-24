// Filesystem Metadata Cache - Specialized cache for zero-copy filesystem metadata
//
// This cache is specifically designed for filesystem metadata used by the zero-copy
// I/O system. It stores lightweight file metadata with a structured key format.
//
// Key Format: "{filepath}:{collection_id}:{engine_type}"
// Example: "/data/collection1/file.sst:collection1:SST"
//
// This is separate from the general MetadataStore which handles JSON metadata,
// as filesystem metadata has very different requirements:
// - Need for memory-mapped access
// - Bytemuck-compatible fixed-size headers
// - High-frequency access patterns
// - Different eviction strategies

use anyhow::Result;
use dashmap::DashMap;
use std::sync::Arc;

use crate::storage::cache::base::BaseCacheImpl;
use crate::storage::cache::traits::{BaseCache, CacheValue};

// Temporary placeholder for MmappedMetadata
// Deferred: Import from zero_copy_io_system::metadata_cache when circular dependency is resolved
#[derive(Debug, Clone)]
pub struct MmappedMetadata;

impl MmappedMetadata {
    pub fn memory_footprint(&self) -> usize {
        // Placeholder implementation
        std::mem::size_of::<Self>()
    }
}

/// Filesystem metadata entry with zero-copy support
#[derive(Debug, Clone)]
pub struct FilesystemMetadata {
    /// Memory-mapped metadata for zero-copy access
    pub mmap_metadata: Option<Arc<MmappedMetadata>>,

    /// File size in bytes
    pub file_size: u64,

    /// Last modification time
    pub last_modified: u64,

    /// Whether file can be skipped for current query
    pub can_skip: bool,

    /// Selective ranges if partial read is needed
    pub selective_ranges: Option<Vec<(u64, u64)>>,

    /// Collection ID this file belongs to
    pub collection_id: String,

    /// Engine type (SST, NOVA, etc.)
    pub engine_type: String,
}

impl CacheValue for FilesystemMetadata {
    fn size_bytes(&self) -> usize {
        // Base struct size
        let mut size = std::mem::size_of::<Self>();

        // Add string allocations
        size += self.collection_id.len();
        size += self.engine_type.len();

        // Add selective ranges if present
        if let Some(ref ranges) = self.selective_ranges {
            size += ranges.len() * std::mem::size_of::<(u64, u64)>();
        }

        // Add mmap metadata footprint if present
        if let Some(ref mmap) = self.mmap_metadata {
            size += mmap.memory_footprint();
        }

        size
    }
}

/// Specialized filesystem metadata cache that integrates with unified cache
pub struct FilesystemMetadataStore {
    /// Base cache implementation for integration with unified system
    base: BaseCacheImpl<String, FilesystemMetadata>,

    /// Direct access map for hot paths (bypasses base cache overhead)
    /// This is for ultra-low latency access to most frequently used entries
    hot_cache: DashMap<String, Arc<FilesystemMetadata>>,

    /// Maximum entries in hot cache
    max_hot_entries: usize,
}

impl FilesystemMetadataStore {
    pub fn new(max_memory_mb: usize, max_hot_entries: usize) -> Self {
        Self {
            base: BaseCacheImpl::new(max_memory_mb),
            hot_cache: DashMap::new(),
            max_hot_entries,
        }
    }

    /// Generate cache key from components
    pub fn make_key(filepath: &str, collection_id: &str, engine_type: &str) -> String {
        format!("{}:{}:{}", filepath, collection_id, engine_type)
    }

    /// Put filesystem metadata with automatic hot cache promotion
    pub async fn put_metadata(
        &self,
        filepath: &str,
        collection_id: &str,
        engine_type: &str,
        metadata: FilesystemMetadata,
    ) -> Result<()> {
        let key = Self::make_key(filepath, collection_id, engine_type);

        // Always put in base cache for unified management
        BaseCache::put_with_hooks(&self.base, key.clone(), metadata).await;

        // Promote to hot cache if frequently accessed
        // (In real implementation, would track access frequency)
        if self.hot_cache.len() < self.max_hot_entries
            && let Some(entry) = BaseCache::get_with_hooks(&self.base, &key).await
        {
            self.hot_cache.insert(key, Arc::new(entry));
        }

        Ok(())
    }

    /// Get filesystem metadata with hot cache fast path
    pub async fn get_metadata(
        &self,
        filepath: &str,
        collection_id: &str,
        engine_type: &str,
    ) -> Option<Arc<FilesystemMetadata>> {
        let key = Self::make_key(filepath, collection_id, engine_type);

        // Fast path: check hot cache first
        if let Some(entry) = self.hot_cache.get(&key) {
            return Some(Arc::clone(&entry));
        }

        // Slow path: check base cache
        if let Some(metadata) = BaseCache::get_with_hooks(&self.base, &key).await {
            let arc_metadata = Arc::new(metadata);

            // Consider promoting to hot cache
            if self.should_promote_to_hot_cache(&key) {
                self.promote_to_hot_cache(key.clone(), Arc::clone(&arc_metadata));
            }

            return Some(arc_metadata);
        }

        None
    }

    /// Check if file can be skipped for query
    pub async fn can_skip_file(
        &self,
        filepath: &str,
        collection_id: &str,
        engine_type: &str,
    ) -> bool {
        if let Some(metadata) = self
            .get_metadata(filepath, collection_id, engine_type)
            .await
        {
            return metadata.can_skip;
        }
        false
    }

    /// Get selective ranges for partial read
    pub async fn get_selective_ranges(
        &self,
        filepath: &str,
        collection_id: &str,
        engine_type: &str,
    ) -> Option<Vec<(u64, u64)>> {
        if let Some(metadata) = self
            .get_metadata(filepath, collection_id, engine_type)
            .await
        {
            return metadata.selective_ranges.clone();
        }
        None
    }

    /// Invalidate entry in both caches
    pub async fn invalidate(&self, filepath: &str, collection_id: &str, engine_type: &str) -> bool {
        let key = Self::make_key(filepath, collection_id, engine_type);

        // Remove from hot cache
        self.hot_cache.remove(&key);

        // Remove from base cache
        BaseCache::invalidate(&self.base, &key).await
    }

    /// Clear entries for a specific collection
    pub async fn clear_collection(&self, collection_id: &str) {
        // Remove from hot cache
        self.hot_cache
            .retain(|key, _| !key.contains(&format!(":{}", collection_id)));

        // Note: Base cache doesn't support pattern-based removal
        // Would need to track keys separately or enhance base cache
    }

    /// Get cache statistics
    pub async fn stats(&self) -> FilesystemCacheStats {
        // Get metrics snapshot from unified metrics
        let snapshot = self.base.metrics().get_snapshot().await;
        let hit_rate = if snapshot.cache_hits + snapshot.cache_misses > 0 {
            snapshot.cache_hits as f64 / (snapshot.cache_hits + snapshot.cache_misses) as f64
        } else {
            0.0
        };

        FilesystemCacheStats {
            base_entries: self.base.size().await,
            hot_entries: self.hot_cache.len(),
            base_memory_bytes: self.base.memory_usage().await,
            hit_rate,
        }
    }

    // Private helper methods

    fn should_promote_to_hot_cache(&self, _key: &str) -> bool {
        // Simple policy: promote if hot cache isn't full
        self.hot_cache.len() < self.max_hot_entries
    }

    fn promote_to_hot_cache(&self, key: String, metadata: Arc<FilesystemMetadata>) {
        // Evict LRU if needed (simplified - in production would track access times)
        if self.hot_cache.len() >= self.max_hot_entries {
            // Remove random entry (should be LRU in production)
            if let Some(entry) = self.hot_cache.iter().next() {
                self.hot_cache.remove(entry.key());
            }
        }

        self.hot_cache.insert(key, metadata);
    }
}

/// Statistics for filesystem metadata cache
#[derive(Debug, Clone)]
pub struct FilesystemCacheStats {
    pub base_entries: usize,
    pub hot_entries: usize,
    pub base_memory_bytes: usize,
    pub hit_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_filesystem_metadata_cache() {
        let cache = FilesystemMetadataStore::new(100, 10);

        let metadata = FilesystemMetadata {
            mmap_metadata: None,
            file_size: 1024 * 1024,
            last_modified: 1234567890,
            can_skip: false,
            selective_ranges: Some(vec![(0, 1024), (2048, 3072)]),
            collection_id: "test_collection".to_string(),
            engine_type: "SST".to_string(),
        };

        // Test put and get
        cache
            .put_metadata("/data/file.sst", "test_collection", "SST", metadata)
            .await
            .unwrap();

        let retrieved = cache
            .get_metadata("/data/file.sst", "test_collection", "SST")
            .await;

        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.file_size, 1024 * 1024);
        assert_eq!(retrieved.collection_id, "test_collection");

        // Test can_skip_file
        let can_skip = cache
            .can_skip_file("/data/file.sst", "test_collection", "SST")
            .await;
        assert!(!can_skip);

        // Test selective ranges
        let ranges = cache
            .get_selective_ranges("/data/file.sst", "test_collection", "SST")
            .await;
        assert!(ranges.is_some());
        assert_eq!(ranges.unwrap().len(), 2);

        // Test invalidation
        let invalidated = cache
            .invalidate("/data/file.sst", "test_collection", "SST")
            .await;
        assert!(invalidated);

        // Verify it's gone
        let retrieved = cache
            .get_metadata("/data/file.sst", "test_collection", "SST")
            .await;
        assert!(retrieved.is_none());
    }

    #[test]
    fn test_key_generation() {
        let key = FilesystemMetadataStore::make_key("/path/to/file.sst", "collection1", "SST");
        assert_eq!(key, "/path/to/file.sst:collection1:SST");
    }
}
