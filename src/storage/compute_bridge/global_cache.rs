//! Global Quantization Cache
//!
//! Unified cache for quantization codebooks integrated with CrossCacheOrchestrator.
//! Provides collection-partitioned storage for PQ, Binary, and INT8 codebooks
//! with intelligent memory management and access pattern tracking.

use anyhow::Result;
use dashmap::DashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use tracing::{debug, info};

use proximadb_quantization_kernel::quantization_engine::{Codebook, CodebookStore};
use crate::storage::cache::orchestrator::{CacheType, CrossCacheOrchestrator};
use proximadb_kernel::hash::XxHash64;

/// Composite key for global quantization cache
/// Format: "{collection_id}#{quantization_type}#{level_params}"
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QuantizationCacheKey {
    pub collection_id: String,
    pub quantization_type: String, // "pq", "binary", "int8"
    pub level_params: String,      // "8_16" for PQ8 with 16 subvectors, etc.
}

impl QuantizationCacheKey {
    /// Create cache key for PQ quantization
    pub fn pq(collection_id: &str, bits_per_code: u8, num_subvectors: u32) -> Self {
        Self {
            collection_id: collection_id.to_string(),
            quantization_type: "pq".to_string(),
            level_params: format!("{}_{}", bits_per_code, num_subvectors),
        }
    }

    /// Create cache key for binary quantization
    pub fn binary(collection_id: &str) -> Self {
        Self {
            collection_id: collection_id.to_string(),
            quantization_type: "binary".to_string(),
            level_params: "1".to_string(),
        }
    }

    /// Create cache key for INT8 quantization
    pub fn int8(collection_id: &str) -> Self {
        Self {
            collection_id: collection_id.to_string(),
            quantization_type: "int8".to_string(),
            level_params: "8".to_string(),
        }
    }

    /// Parse from string
    pub fn from_string(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split('#').collect();
        if parts.len() == 3 {
            Some(Self {
                collection_id: parts[0].to_string(),
                quantization_type: parts[1].to_string(),
                level_params: parts[2].to_string(),
            })
        } else {
            None
        }
    }
}

impl fmt::Display for QuantizationCacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}#{}#{}",
            self.collection_id, self.quantization_type, self.level_params
        )
    }
}

/// Key for quantized vector cache
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QuantizedVectorKey {
    /// Hash of the original vector
    vector_hash: u64,
    /// Quantization level
    level: String,
    /// Collection ID for partitioning
    collection_id: String,
}

impl QuantizedVectorKey {
    pub fn new(vector: &[f32], level: &str, collection_id: &str) -> Self {
        // Use fast XxHash64 for vector hashing
        let mut hasher = XxHash64::new(0);
        for &v in vector {
            hasher.write_u32(v.to_bits());
        }
        Self {
            vector_hash: hasher.finish(),
            level: level.to_string(),
            collection_id: collection_id.to_string(),
        }
    }
}

/// Cached quantized vector result
#[derive(Clone)]
pub struct CachedQuantizedVector {
    /// Quantized data
    pub data: Arc<Vec<u8>>,
    /// Access count for LRU tracking
    pub access_count: Arc<std::sync::atomic::AtomicUsize>,
    /// Last access timestamp
    pub last_access: Arc<std::sync::atomic::AtomicU64>,
}

/// Global quantization cache integrated with CrossCacheOrchestrator
pub struct GlobalQuantizationCache {
    /// Collection-partitioned codebook storage
    /// Key: QuantizationCacheKey as string
    /// Value: Serialized codebook data
    codebooks: Arc<DashMap<String, Arc<Codebook>>>,

    /// Quantized vector cache with LRU eviction
    /// Key: (vector_hash, level, collection_id)
    /// Value: Quantized vector data
    quantized_vectors: Arc<DashMap<QuantizedVectorKey, CachedQuantizedVector>>,

    /// Maximum number of cached quantized vectors per collection
    max_cached_vectors_per_collection: usize,

    /// CrossCacheOrchestrator for unified memory management
    orchestrator: Option<Arc<CrossCacheOrchestrator>>,

    /// Memory budget allocated for quantization (managed by orchestrator)
    allocated_memory_bytes: std::sync::atomic::AtomicUsize,

    /// Cache hit count
    cache_hits: std::sync::atomic::AtomicUsize,
    /// Cache miss count
    cache_misses: std::sync::atomic::AtomicUsize,
}

impl GlobalQuantizationCache {
    /// Create new global quantization cache
    pub fn new() -> Self {
        Self::with_capacity(10000) // Default to 10K cached vectors per collection
    }

    /// Create with specified capacity
    pub fn with_capacity(max_vectors_per_collection: usize) -> Self {
        Self {
            codebooks: Arc::new(DashMap::new()),
            quantized_vectors: Arc::new(DashMap::new()),
            max_cached_vectors_per_collection: max_vectors_per_collection,
            orchestrator: CrossCacheOrchestrator::global(),
            allocated_memory_bytes: std::sync::atomic::AtomicUsize::new(0),
            cache_hits: std::sync::atomic::AtomicUsize::new(0),
            cache_misses: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Store codebook with CrossCacheOrchestrator integration (internal method)
    pub async fn store_codebook_internal(
        &self,
        key: QuantizationCacheKey,
        codebook: Codebook,
    ) -> Result<()> {
        let key_str = key.to_string();

        // Track access with CrossCacheOrchestrator
        if let Some(ref orchestrator) = self.orchestrator {
            orchestrator.track_access_async(key_str.clone(), CacheType::Quantization);
        }

        // Estimate memory usage
        let estimated_bytes = Self::estimate_codebook_size(&codebook);

        // Store in DashMap
        self.codebooks.insert(key_str.clone(), Arc::new(codebook));

        // Update memory tracking
        self.allocated_memory_bytes
            .fetch_add(estimated_bytes, std::sync::atomic::Ordering::Relaxed);

        info!(
            "📚 Stored quantization codebook: {} (estimated {} bytes)",
            key_str, estimated_bytes
        );

        Ok(())
    }

    /// Retrieve codebook with access tracking
    pub async fn get_codebook(&self, key: &QuantizationCacheKey) -> Option<Arc<Codebook>> {
        let key_str = key.to_string();

        // Track access with CrossCacheOrchestrator
        if let Some(ref orchestrator) = self.orchestrator {
            orchestrator.track_access_async(key_str.clone(), CacheType::Quantization);
        }

        // Get from DashMap
        if let Some(codebook) = self.codebooks.get(&key_str) {
            debug!("🎯 Retrieved quantization codebook: {}", key_str);
            Some(codebook.clone())
        } else {
            debug!("❌ Quantization codebook not found: {}", key_str);
            None
        }
    }

    /// Check if codebook exists
    pub fn has_codebook(&self, key: &QuantizationCacheKey) -> bool {
        self.codebooks.contains_key(&key.to_string())
    }

    /// Remove codebook for a collection (collection cleanup)
    pub async fn remove_collection_codebooks(&self, collection_id: &str) -> Result<usize> {
        let mut removed_count = 0;
        let mut removed_bytes = 0;

        // Find all keys for this collection
        let keys_to_remove: Vec<String> = self
            .codebooks
            .iter()
            .filter_map(|entry| {
                if let Some(parsed_key) = QuantizationCacheKey::from_string(entry.key()) {
                    if parsed_key.collection_id == collection_id {
                        Some(entry.key().clone())
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        // Remove codebooks
        for key_str in keys_to_remove {
            if let Some((_, codebook)) = self.codebooks.remove(&key_str) {
                let estimated_bytes = Self::estimate_codebook_size(&codebook);
                removed_bytes += estimated_bytes;
                removed_count += 1;
            }
        }

        // Update memory tracking
        self.allocated_memory_bytes
            .fetch_sub(removed_bytes, std::sync::atomic::Ordering::Relaxed);

        info!(
            "🧹 Removed {} quantization codebooks for collection '{}' ({} bytes freed)",
            removed_count, collection_id, removed_bytes
        );

        Ok(removed_count)
    }

    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> QuantizationCacheStats {
        let codebook_count = self.codebooks.len();
        let allocated_bytes = self
            .allocated_memory_bytes
            .load(std::sync::atomic::Ordering::Relaxed);

        QuantizationCacheStats {
            codebook_count,
            allocated_bytes,
            collections_count: self.count_unique_collections(),
        }
    }

    /// Count unique collections in cache
    fn count_unique_collections(&self) -> usize {
        let mut collections = std::collections::HashSet::new();

        for entry in self.codebooks.iter() {
            if let Some(parsed_key) = QuantizationCacheKey::from_string(entry.key()) {
                collections.insert(parsed_key.collection_id);
            }
        }

        collections.len()
    }

    /// Estimate codebook memory usage
    fn estimate_codebook_size(codebook: &Codebook) -> usize {
        use crate::compute::quantization::quantization_engine::CodebookData;
        // Rough estimation based on codebook structure
        // This is a simplified calculation - in production you'd want more precise measurement
        match &codebook.data {
            CodebookData::ProductQuantization { centroids, .. } => {
                // Estimate: num_subspaces * num_centroids * centroid_dimension * sizeof(f32)
                centroids.len()
                    * 256
                    * centroids
                        .first()
                        .and_then(|c| c.first())
                        .map_or(0, |c| c.len())
                    * 4
            }
            CodebookData::Binary { thresholds } => {
                // Binary codebooks: dimension * sizeof(f32) for thresholds
                thresholds.len() * 4
            }
            CodebookData::Scalar { scales, offsets } => {
                // Scalar quantization parameters: 2 f32 values * dimensions
                (scales.len() + offsets.len()) * 4
            }
            CodebookData::Custom(_) => {
                // Custom codebooks: estimate 1KB
                1024
            }
        }
    }

    // ========================================================================
    // QUANTIZED VECTOR CACHING METHODS
    // ========================================================================

    /// Get or compute quantized vector
    /// Returns cached result if available, otherwise computes and caches
    pub fn get_or_quantize(
        &self,
        vector: &[f32],
        level: &str,
        collection_id: &str,
        quantize_fn: impl FnOnce() -> Result<Vec<u8>>,
    ) -> Result<Arc<Vec<u8>>> {
        let key = QuantizedVectorKey::new(vector, level, collection_id);

        // Check cache first
        if let Some(cached) = self.quantized_vectors.get(&key) {
            // Update access stats
            cached
                .access_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            cached.last_access.store(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                std::sync::atomic::Ordering::Relaxed,
            );

            self.cache_hits
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            debug!(
                "Quantization cache hit for collection: {}, level: {}, hash: {}",
                collection_id, level, key.vector_hash
            );

            return Ok(Arc::clone(&cached.data));
        }

        // Cache miss - compute quantization
        self.cache_misses
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        debug!(
            "Quantization cache miss for collection: {}, level: {}, hash: {}",
            collection_id, level, key.vector_hash
        );

        let quantized = quantize_fn()?;
        let quantized_arc = Arc::new(quantized);

        // Check if we need to evict old entries for this collection
        self.evict_if_needed(collection_id);

        // Store in cache
        let cached = CachedQuantizedVector {
            data: Arc::clone(&quantized_arc),
            access_count: Arc::new(std::sync::atomic::AtomicUsize::new(1)),
            last_access: Arc::new(std::sync::atomic::AtomicU64::new(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            )),
        };

        self.quantized_vectors.insert(key, cached);

        // Update memory tracking
        let size_estimate = quantized_arc.len();
        self.allocated_memory_bytes
            .fetch_add(size_estimate, std::sync::atomic::Ordering::Relaxed);

        Ok(quantized_arc)
    }

    /// Evict least recently used entries if cache is full for a collection
    fn evict_if_needed(&self, collection_id: &str) {
        // Count vectors for this collection
        let collection_count = self
            .quantized_vectors
            .iter()
            .filter(|entry| entry.key().collection_id == collection_id)
            .count();

        if collection_count >= self.max_cached_vectors_per_collection {
            debug!(
                "Cache full for collection {}, evicting LRU entries",
                collection_id
            );

            // Find and remove least recently used entry
            let mut oldest_key = None;
            let mut oldest_time = u64::MAX;

            for entry in self.quantized_vectors.iter() {
                if entry.key().collection_id == collection_id {
                    let last_access = entry
                        .value()
                        .last_access
                        .load(std::sync::atomic::Ordering::Relaxed);
                    if last_access < oldest_time {
                        oldest_time = last_access;
                        oldest_key = Some(entry.key().clone());
                    }
                }
            }

            if let Some(key) = oldest_key
                && let Some((_, removed)) = self.quantized_vectors.remove(&key)
            {
                let size_estimate = removed.data.len();
                self.allocated_memory_bytes
                    .fetch_sub(size_estimate, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    /// Get cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        let hits = self.cache_hits.load(std::sync::atomic::Ordering::Relaxed);
        let misses = self.cache_misses.load(std::sync::atomic::Ordering::Relaxed);
        let total = hits + misses;

        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    /// Clear quantized vector cache for a collection
    pub fn clear_quantized_vectors(&self, collection_id: &str) {
        let mut removed_size = 0usize;

        self.quantized_vectors.retain(|key, value| {
            if key.collection_id == collection_id {
                removed_size += value.data.len();
                false
            } else {
                true
            }
        });

        self.allocated_memory_bytes
            .fetch_sub(removed_size, std::sync::atomic::Ordering::Relaxed);

        info!(
            "Cleared quantized vector cache for collection: {}, freed {} bytes",
            collection_id, removed_size
        );
    }

    /// Get statistics including quantized vector cache
    pub fn get_extended_stats(&self) -> ExtendedCacheStats {
        ExtendedCacheStats {
            codebook_count: self.codebooks.len(),
            quantized_vector_count: self.quantized_vectors.len(),
            allocated_bytes: self
                .allocated_memory_bytes
                .load(std::sync::atomic::Ordering::Relaxed),
            cache_hits: self.cache_hits.load(std::sync::atomic::Ordering::Relaxed),
            cache_misses: self.cache_misses.load(std::sync::atomic::Ordering::Relaxed),
            hit_rate: self.cache_hit_rate(),
        }
    }
}

impl Default for GlobalQuantizationCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Extended cache statistics
#[derive(Debug, Clone)]
pub struct ExtendedCacheStats {
    pub codebook_count: usize,
    pub quantized_vector_count: usize,
    pub allocated_bytes: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub hit_rate: f64,
}

/// Statistics for quantization cache
#[derive(Debug, Clone)]
pub struct QuantizationCacheStats {
    pub codebook_count: usize,
    pub allocated_bytes: usize,
    pub collections_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    // === QuantizationCacheKey tests ===

    #[test]
    fn test_cache_key_pq() {
        let key = QuantizationCacheKey::pq("my_collection", 8, 16);
        assert_eq!(key.collection_id, "my_collection");
        assert_eq!(key.quantization_type, "pq");
        assert_eq!(key.level_params, "8_16");
    }

    #[test]
    fn test_cache_key_binary() {
        let key = QuantizationCacheKey::binary("coll_1");
        assert_eq!(key.collection_id, "coll_1");
        assert_eq!(key.quantization_type, "binary");
        assert_eq!(key.level_params, "1");
    }

    #[test]
    fn test_cache_key_int8() {
        let key = QuantizationCacheKey::int8("coll_2");
        assert_eq!(key.collection_id, "coll_2");
        assert_eq!(key.quantization_type, "int8");
        assert_eq!(key.level_params, "8");
    }

    #[test]
    fn test_cache_key_to_string() {
        let key = QuantizationCacheKey::pq("test", 4, 32);
        assert_eq!(key.to_string(), "test#pq#4_32");
    }

    #[test]
    fn test_cache_key_from_string_valid() {
        let key = QuantizationCacheKey::from_string("my_coll#binary#1");
        assert!(key.is_some());
        let key = key.unwrap_or_else(|| panic!("Expected Some"));
        assert_eq!(key.collection_id, "my_coll");
        assert_eq!(key.quantization_type, "binary");
        assert_eq!(key.level_params, "1");
    }

    #[test]
    fn test_cache_key_from_string_invalid_too_few_parts() {
        assert!(QuantizationCacheKey::from_string("only_one").is_none());
    }

    #[test]
    fn test_cache_key_from_string_invalid_too_many_parts() {
        assert!(QuantizationCacheKey::from_string("a#b#c#d").is_none());
    }

    #[test]
    fn test_cache_key_from_string_empty() {
        assert!(QuantizationCacheKey::from_string("").is_none());
    }

    #[test]
    fn test_cache_key_roundtrip() {
        let original = QuantizationCacheKey::pq("vectors", 8, 16);
        let serialized = original.to_string();
        let parsed = QuantizationCacheKey::from_string(&serialized);
        assert!(parsed.is_some());
        let parsed = parsed.unwrap_or_else(|| panic!("Expected Some"));
        assert_eq!(parsed, original);
    }

    #[test]
    fn test_cache_key_equality() {
        let a = QuantizationCacheKey::binary("coll");
        let b = QuantizationCacheKey::binary("coll");
        assert_eq!(a, b);
    }

    #[test]
    fn test_cache_key_inequality_different_collection() {
        let a = QuantizationCacheKey::binary("coll_a");
        let b = QuantizationCacheKey::binary("coll_b");
        assert_ne!(a, b);
    }

    #[test]
    fn test_cache_key_inequality_different_type() {
        let a = QuantizationCacheKey::binary("coll");
        let b = QuantizationCacheKey::int8("coll");
        assert_ne!(a, b);
    }

    #[test]
    fn test_cache_key_hash_consistency() {
        use std::collections::HashSet;
        let key1 = QuantizationCacheKey::pq("test", 8, 16);
        let key2 = QuantizationCacheKey::pq("test", 8, 16);
        let mut set = HashSet::new();
        set.insert(key1);
        assert!(set.contains(&key2));
    }

    // === QuantizedVectorKey tests ===

    #[test]
    fn test_quantized_vector_key_new() {
        let vector = vec![1.0f32, 2.0, 3.0];
        let key = QuantizedVectorKey::new(&vector, "int8", "coll");
        assert_eq!(key.level, "int8");
        assert_eq!(key.collection_id, "coll");
        // vector_hash should be deterministic
        let key2 = QuantizedVectorKey::new(&vector, "int8", "coll");
        assert_eq!(key.vector_hash, key2.vector_hash);
    }

    #[test]
    fn test_quantized_vector_key_different_vectors() {
        let v1 = vec![1.0f32, 2.0, 3.0];
        let v2 = vec![4.0f32, 5.0, 6.0];
        let k1 = QuantizedVectorKey::new(&v1, "pq", "c");
        let k2 = QuantizedVectorKey::new(&v2, "pq", "c");
        assert_ne!(k1.vector_hash, k2.vector_hash);
    }

    #[test]
    fn test_quantized_vector_key_different_levels() {
        let v = vec![1.0f32, 2.0];
        let k1 = QuantizedVectorKey::new(&v, "int8", "c");
        let k2 = QuantizedVectorKey::new(&v, "binary", "c");
        // Same hash but different level
        assert_eq!(k1.vector_hash, k2.vector_hash);
        assert_ne!(k1, k2);
    }

    // === ExtendedCacheStats tests ===

    #[test]
    fn test_extended_cache_stats_clone() {
        let stats = ExtendedCacheStats {
            codebook_count: 5,
            quantized_vector_count: 100,
            allocated_bytes: 4096,
            cache_hits: 80,
            cache_misses: 20,
            hit_rate: 0.8,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.codebook_count, 5);
        assert_eq!(cloned.quantized_vector_count, 100);
        assert_eq!(cloned.allocated_bytes, 4096);
        assert_eq!(cloned.cache_hits, 80);
        assert_eq!(cloned.cache_misses, 20);
        assert!((cloned.hit_rate - 0.8).abs() < f64::EPSILON);
    }

    // === QuantizationCacheStats tests ===

    #[test]
    fn test_quantization_cache_stats_clone() {
        let stats = QuantizationCacheStats {
            codebook_count: 3,
            allocated_bytes: 2048,
            collections_count: 2,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.codebook_count, 3);
        assert_eq!(cloned.allocated_bytes, 2048);
        assert_eq!(cloned.collections_count, 2);
    }

    // === GlobalQuantizationCache basic tests ===

    #[test]
    fn test_cache_hit_rate_no_accesses() {
        let cache = GlobalQuantizationCache::with_capacity(10);
        assert!((cache.cache_hit_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_has_codebook_empty_cache() {
        let cache = GlobalQuantizationCache::with_capacity(10);
        let key = QuantizationCacheKey::pq("test", 8, 16);
        assert!(!cache.has_codebook(&key));
    }

    #[test]
    fn test_get_memory_stats_empty() {
        let cache = GlobalQuantizationCache::with_capacity(10);
        let stats = cache.get_memory_stats();
        assert_eq!(stats.codebook_count, 0);
        assert_eq!(stats.allocated_bytes, 0);
        assert_eq!(stats.collections_count, 0);
    }

    #[test]
    fn test_get_extended_stats_empty() {
        let cache = GlobalQuantizationCache::with_capacity(10);
        let stats = cache.get_extended_stats();
        assert_eq!(stats.codebook_count, 0);
        assert_eq!(stats.quantized_vector_count, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
        assert!((stats.hit_rate - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_should_use_hot_storage_default() {
        let cache = GlobalQuantizationCache::with_capacity(10);
        assert!(cache.should_use_hot_storage("any_collection"));
    }
}

/// Implementation of CodebookStore trait for global cache
#[async_trait::async_trait]
impl CodebookStore for GlobalQuantizationCache {
    async fn store_codebook(&self, id: &str, codebook: &Codebook) -> Result<()> {
        // Parse the ID to extract collection and quantization details
        // ID format expected: "collection_id:quantization_details"
        let parts: Vec<&str> = id.split(':').collect();
        if parts.len() != 2 {
            return Err(anyhow::anyhow!("Invalid codebook ID format: {}", id));
        }

        let collection_id = parts[0];
        let quant_details = parts[1];

        // Parse quantization details to create proper cache key
        let cache_key = if quant_details.starts_with("pq") {
            // Format: "pq8_16" -> PQ8 with 16 subvectors
            let params: Vec<&str> = quant_details
                .strip_prefix("pq")
                .unwrap_or("8_16")
                .split('_')
                .collect();
            let bits = params.first().unwrap_or(&"8").parse().unwrap_or(8);
            let subvectors = params.get(1).unwrap_or(&"16").parse().unwrap_or(16);
            QuantizationCacheKey::pq(collection_id, bits, subvectors)
        } else if quant_details == "binary" {
            QuantizationCacheKey::binary(collection_id)
        } else if quant_details == "int8" {
            QuantizationCacheKey::int8(collection_id)
        } else {
            return Err(anyhow::anyhow!(
                "Unknown quantization type: {}",
                quant_details
            ));
        };

        // Use our internal store_codebook method
        self.store_codebook_internal(cache_key, codebook.clone())
            .await
    }

    async fn get_codebook(&self, id: &str) -> Result<Option<Codebook>> {
        // Parse ID and convert to cache key (same logic as store)
        let parts: Vec<&str> = id.split(':').collect();
        if parts.len() != 2 {
            return Ok(None);
        }

        let collection_id = parts[0];
        let quant_details = parts[1];

        let cache_key = if quant_details.starts_with("pq") {
            let params: Vec<&str> = quant_details
                .strip_prefix("pq")
                .unwrap_or("8_16")
                .split('_')
                .collect();
            let bits = params.first().unwrap_or(&"8").parse().unwrap_or(8);
            let subvectors = params.get(1).unwrap_or(&"16").parse().unwrap_or(16);
            QuantizationCacheKey::pq(collection_id, bits, subvectors)
        } else if quant_details == "binary" {
            QuantizationCacheKey::binary(collection_id)
        } else if quant_details == "int8" {
            QuantizationCacheKey::int8(collection_id)
        } else {
            return Ok(None);
        };

        Ok(self
            .get_codebook(&cache_key)
            .await
            .map(|arc| (*arc).clone()))
    }

    async fn list_codebooks(&self) -> Result<Vec<String>> {
        Ok(self
            .codebooks
            .iter()
            .map(|entry| entry.key().clone())
            .collect())
    }
}

/// Global singleton instance
static GLOBAL_QUANTIZATION_CACHE: std::sync::OnceLock<Arc<GlobalQuantizationCache>> =
    std::sync::OnceLock::new();

impl GlobalQuantizationCache {
    /// Get global singleton instance
    pub fn global() -> Arc<GlobalQuantizationCache> {
        GLOBAL_QUANTIZATION_CACHE
            .get_or_init(|| Arc::new(GlobalQuantizationCache::new()))
            .clone()
    }

    /// Get singleton instance (alias for consistency with other components)
    pub fn instance() -> Option<Arc<GlobalQuantizationCache>> {
        Some(Self::global())
    }

    /// Get or create a quantization engine for a specific collection
    /// Uses intelligent caching strategy based on collection size and access patterns
    pub async fn get_or_create_engine(
        &self,
        _collection_id: String,
    ) -> Arc<crate::compute::quantization::quantization_engine::UnifiedQuantizationEngine> {
        // For now, create a simple UnifiedQuantizationEngine with the global cache as codebook store
        // This provides the unified interface expected by the engines
        Arc::new(
            crate::compute::quantization::quantization_engine::UnifiedQuantizationEngine::new(
                Arc::new(proximadb_distance_kernel::engine::UnifiedDistanceCompute::default()),
                Self::global()
                    as Arc<dyn crate::compute::quantization::quantization_engine::CodebookStore>,
            ),
        )
    }

    /// Update collection size for adaptive storage strategy decisions
    pub fn update_collection_size(&self, collection_id: &str, size: usize) {
        // This would be used by the adaptive strategy to decide between hot/cold storage
        // For now, we'll keep the simple DashMap approach
        debug!(
            "Updated collection '{}' size to {} vectors",
            collection_id, size
        );
    }

    /// Check if collection should use hot (in-memory) or cold (persistent) storage
    pub fn should_use_hot_storage(&self, _collection_id: &str) -> bool {
        // Simple heuristic: collections with frequent access use hot storage
        // In practice, this would check access patterns and collection size
        true // Default to hot storage for now
    }
}
