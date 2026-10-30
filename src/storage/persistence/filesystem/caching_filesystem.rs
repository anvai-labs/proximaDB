//! Unified Caching Filesystem
//!
//! A consolidated filesystem implementation that merges the capabilities of
//! IntelligentFilesystem and ZeroCopyFilesystem into a single, efficient layer.
//!
//! ## Key Features
//! - Single metadata cache shared across all operations
//! - Lock-free design for hot paths
//! - Zero-copy operations where possible
//! - Integrated I/O optimization from ZeroCopyIOSystem
//! - Workload-based configuration presets

use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use tracing::debug;

use crate::storage::persistence::filesystem::{
    DirEntry, FileMetadata, FileOptions, FileSystem, FilesystemFile, FsResult,
};
use proximadb_kernel::error::ProximaDBError;

use super::access_tracker::AccessPatternTracker;
use super::cache_config::{UnifiedCacheConfig, WorkloadType};
use super::cache_metrics::CacheMetrics;
use super::disk_cache::DiskCacheManager;
use super::metadata_cache::{CachedMetadata, UnifiedMetadataCache};
use super::metadata_traits::{EngineMetadataSerializer, GenericMetadataSerializer};
use super::prefetch_engine::PrefetchEngine;
use super::range_optimizer::RangeOptimizer;

/// Unified caching filesystem that consolidates all caching layers
pub struct UnifiedCachingFilesystem {
    /// The underlying filesystem (Local, S3, Azure, GCS, etc.)
    underlying_fs: Arc<dyn FileSystem>,

    /// Unified metadata cache (single instance for all metadata)
    metadata_cache: Arc<UnifiedMetadataCache>,

    /// Disk cache manager for local file caching
    disk_cache: Arc<DiskCacheManager>,

    /// I/O optimization components (from ZeroCopyIOSystem)
    range_optimizer: Arc<RangeOptimizer>,

    /// Pattern tracking and learning
    access_tracker: Arc<AccessPatternTracker>,
    prefetch_engine: Arc<PrefetchEngine>,

    /// Configuration
    config: Arc<UnifiedCacheConfig>,

    /// Collection context
    collection_id: String,
    engine_type: String,

    /// Metrics
    metrics: Arc<CacheMetrics>,

    /// Engine-specific metadata serializer (provided by the storage engine)
    #[allow(dead_code)]
    metadata_serializer: Arc<dyn EngineMetadataSerializer>,
}

impl UnifiedCachingFilesystem {
    /// Create a new unified caching filesystem
    pub fn new(
        underlying_fs: Arc<dyn FileSystem>,
        collection_id: String,
        engine_type: String,
    ) -> Self {
        Self::with_config(
            underlying_fs,
            UnifiedCacheConfig::default(),
            collection_id,
            engine_type,
        )
    }

    /// Create with engine-provided metadata serializer
    pub fn with_serializer(
        underlying_fs: Arc<dyn FileSystem>,
        collection_id: String,
        engine_type: String,
        metadata_serializer: Arc<dyn EngineMetadataSerializer>,
    ) -> Self {
        Self::with_config_and_serializer(
            underlying_fs,
            UnifiedCacheConfig::default(),
            collection_id,
            engine_type,
            metadata_serializer,
        )
    }

    /// Create with specific configuration (uses generic serializer)
    pub fn with_config(
        underlying_fs: Arc<dyn FileSystem>,
        config: UnifiedCacheConfig,
        collection_id: String,
        engine_type: String,
    ) -> Self {
        Self::with_config_and_serializer(
            underlying_fs,
            config,
            collection_id,
            engine_type,
            Arc::new(GenericMetadataSerializer),
        )
    }

    /// Create with both configuration and serializer
    pub fn with_config_and_serializer(
        underlying_fs: Arc<dyn FileSystem>,
        config: UnifiedCacheConfig,
        collection_id: String,
        engine_type: String,
        metadata_serializer: Arc<dyn EngineMetadataSerializer>,
    ) -> Self {
        let config = Arc::new(config);

        // Create shared metadata cache
        let metadata_cache = Arc::new(UnifiedMetadataCache::new(
            config.memory.total_budget_mb * config.memory.metadata_percentage as usize / 100,
            config.behavior.default_ttl_secs,
        ));

        // Create disk cache manager
        let disk_cache = Arc::new(DiskCacheManager::new(
            config.disk.path.clone(),
            config.disk.max_size_gb,
        ));

        // Create I/O optimization components
        let range_optimizer = Arc::new(RangeOptimizer::new(
            config.io.range_merge_threshold,
            config.io.range_optimization_threshold_mb,
        ));

        // Create pattern tracking
        let access_tracker = Arc::new(AccessPatternTracker::new());
        let prefetch_engine = Arc::new(PrefetchEngine::new(config.io.enable_prefetching));

        // Create metrics collector
        let metrics = Arc::new(CacheMetrics::new());

        Self {
            underlying_fs,
            metadata_cache,
            disk_cache,
            range_optimizer,
            access_tracker,
            prefetch_engine,
            config,
            collection_id,
            engine_type,
            metrics,
            metadata_serializer,
        }
    }

    /// Create with workload preset
    pub fn for_workload(
        underlying_fs: Arc<dyn FileSystem>,
        workload: WorkloadType,
        collection_id: String,
        engine_type: String,
    ) -> Self {
        let config = UnifiedCacheConfig::from_workload(workload);
        Self::with_config(underlying_fs, config, collection_id, engine_type)
    }

    /// Builder pattern for configuration
    pub fn builder() -> UnifiedFilesystemBuilder {
        UnifiedFilesystemBuilder::new()
    }

    /// Create a new unified caching filesystem with local backend for testing
    ///
    /// This is a convenience method for tests that creates a UnifiedCachingFilesystem
    /// backed by a local filesystem at the specified path.
    #[cfg(test)]
    pub async fn new_local(base_path: &std::path::Path) -> FsResult<Arc<Self>> {
        use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
        use std::sync::Arc;

        // Create local filesystem
        let local_fs = LocalFileSystem::new(LocalConfig {
            root_dir: Some(base_path.to_path_buf()),
            follow_symlinks: true,
            default_permissions: None,
            sync_enabled: false, // Disable sync for tests
        })
        .await?;

        // Create unified caching filesystem
        Ok(Arc::new(Self::new(
            Arc::new(local_fs),
            "test_collection".to_string(),
            "test_engine".to_string(),
        )))
    }

    /// Generate cache key for this filesystem instance
    fn cache_key(&self, path: &str) -> String {
        format!("{}:{}:{}", path, self.collection_id, self.engine_type)
    }

    /// Get filesystem cache metrics
    pub async fn get_metrics(&self) -> FilesystemMetrics {
        let metadata_stats = self.metadata_cache.stats();
        let disk_stats = self.disk_cache.stats();
        let cache_report = self.metrics.get_report().await;

        FilesystemMetrics {
            metadata_cache_size: metadata_stats.memory_usage_bytes,
            metadata_cache_entries: metadata_stats.entries,
            disk_cache_size: disk_stats.bytes_saved as usize,
            disk_cache_entries: disk_stats.entries,
            total_hits: cache_report.total_hits,
            total_misses: cache_report.total_misses,
            hit_rate: cache_report.overall_hit_rate,
        }
    }

    /// Record access for pattern learning
    async fn record_access(&self, path: &str, operation: AccessOperation) {
        self.access_tracker.record(path, operation).await;
        self.metrics.record_access();

        // Trigger prefetching if enabled
        if self.config.io.enable_prefetching {
            self.prefetch_engine
                .maybe_prefetch(path, &self.access_tracker)
                .await;
        }
    }

    /// Check if path should be cached locally
    fn should_cache_locally(&self, path: &str, size: usize) -> bool {
        // Don't cache if disk cache is disabled
        if !self.config.disk.enabled {
            return false;
        }

        // Don't cache very large files
        if size > self.config.disk.max_file_size_mb * 1024 * 1024 {
            return false;
        }

        // Check access frequency
        self.access_tracker.is_hot(path)
    }
}

impl fmt::Debug for UnifiedCachingFilesystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UnifiedCachingFilesystem")
            .field("collection_id", &self.collection_id)
            .field("engine_type", &self.engine_type)
            .field("underlying_fs", &self.underlying_fs.filesystem_type())
            .field("config", &self.config)
            .finish()
    }
}

#[async_trait]
impl FileSystem for UnifiedCachingFilesystem {
    fn filesystem_type(&self) -> &'static str {
        "UnifiedCaching"
    }

    async fn read(&self, path: &str) -> FsResult<Vec<u8>> {
        let cache_key = self.cache_key(path);

        // Record access for pattern learning
        self.record_access(path, AccessOperation::Read).await;

        // Check disk cache first
        if let Some(cached_path) = self.disk_cache.get(path).await {
            debug!("Disk cache hit for {}", path);
            self.metrics.record_cache_hit(CacheType::Disk);
            return self.underlying_fs.read(&cached_path).await;
        }

        // Check if we can optimize with range reads
        if self.config.io.enable_range_optimization
            && let Ok(metadata) = self.metadata(path).await
            && metadata.size > (self.config.io.range_optimization_threshold_mb * 1024 * 1024) as u64
        {
            // Use range optimization for large files
            return self.optimized_range_read(path, &metadata).await;
        }

        // Fall back to regular read
        let data = self.underlying_fs.read(path).await?;

        // Extract and cache metadata based on storage engine
        match self.engine_type.to_lowercase().as_str() {
            "viper" | "nova" => {
                // These engines use Parquet format
                self.metadata_cache
                    .extract_parquet_metadata(&cache_key, &data)
                    .await;
            }
            // Other engines might have different metadata formats
            _ => {
                // For now, just cache the file metadata
            }
        }

        // Cache locally if appropriate
        if self.should_cache_locally(path, data.len()) {
            self.disk_cache.put(path, &data).await;
        }

        Ok(data)
    }

    async fn write(&self, path: &str, data: &[u8], options: Option<FileOptions>) -> FsResult<()> {
        let cache_key = self.cache_key(path);

        // Record access
        self.record_access(path, AccessOperation::Write).await;

        // Invalidate caches
        self.metadata_cache.invalidate(&cache_key).await;
        self.disk_cache.invalidate(path).await;

        // Write through to underlying filesystem
        self.underlying_fs.write(path, data, options).await?;

        // Update disk cache if appropriate
        if self.should_cache_locally(path, data.len()) {
            self.disk_cache.put(path, data).await;
        }

        Ok(())
    }

    fn supports_bounded_local_file_write(&self) -> bool {
        self.underlying_fs.supports_bounded_local_file_write()
    }

    async fn write_local_file(
        &self,
        path: &str,
        local_path: &std::path::Path,
        options: Option<FileOptions>,
    ) -> FsResult<u64> {
        let cache_key = self.cache_key(path);
        self.record_access(path, AccessOperation::Write).await;
        self.metadata_cache.invalidate(&cache_key).await;
        self.disk_cache.invalidate(path).await;
        self.underlying_fs
            .write_local_file(path, local_path, options)
            .await
    }

    async fn write_if_absent(
        &self,
        path: &str,
        data: &[u8],
        options: Option<FileOptions>,
    ) -> FsResult<()> {
        let cache_key = self.cache_key(path);
        self.metadata_cache.invalidate(&cache_key).await;
        self.disk_cache.invalidate(path).await;
        self.underlying_fs
            .write_if_absent(path, data, options)
            .await?;
        if self.should_cache_locally(path, data.len()) {
            self.disk_cache.put(path, data).await;
        }
        Ok(())
    }

    async fn append(&self, path: &str, data: &[u8]) -> FsResult<()> {
        let cache_key = self.cache_key(path);

        // Invalidate caches
        self.metadata_cache.invalidate(&cache_key).await;
        self.disk_cache.invalidate(path).await;

        // Append through to underlying filesystem
        self.underlying_fs.append(path, data).await
    }

    fn supports_append(&self) -> bool {
        self.underlying_fs.supports_append()
    }

    async fn delete(&self, path: &str) -> FsResult<()> {
        let cache_key = self.cache_key(path);

        // Invalidate all caches
        self.metadata_cache.invalidate(&cache_key).await;
        self.disk_cache.invalidate(path).await;

        // Delete from underlying filesystem
        self.underlying_fs.delete(path).await
    }

    async fn move_file(&self, from: &str, to: &str) -> FsResult<()> {
        // Invalidate caches for both paths
        self.metadata_cache.invalidate(&self.cache_key(from)).await;
        self.metadata_cache.invalidate(&self.cache_key(to)).await;
        self.disk_cache.invalidate(from).await;
        self.disk_cache.invalidate(to).await;

        // Move in underlying filesystem
        self.underlying_fs.move_file(from, to).await
    }

    async fn exists(&self, path: &str) -> FsResult<bool> {
        // Try metadata cache first (existence check is just metadata)
        let cache_key = self.cache_key(path);

        if let Some(_metadata) = self.metadata_cache.get(&cache_key).await {
            self.metrics.record_cache_hit(CacheType::Metadata);
            return Ok(true); // If we have metadata, file exists
        }

        // Check underlying filesystem
        let exists = self.underlying_fs.exists(path).await?;

        // Cache negative results too (with shorter TTL)
        if !exists {
            self.metadata_cache
                .put_negative(&cache_key, Duration::from_secs(60))
                .await;
        }

        Ok(exists)
    }

    async fn metadata(&self, path: &str) -> FsResult<FileMetadata> {
        let cache_key = self.cache_key(path);

        // Check cache first
        if let Some(cached) = self.metadata_cache.get(&cache_key).await {
            self.metrics.record_cache_hit(CacheType::Metadata);
            return Ok(cached.metadata.clone());
        }

        // Cache miss - fetch from underlying filesystem
        self.metrics.record_cache_miss(CacheType::Metadata);
        let metadata = self.underlying_fs.metadata(path).await?;

        // Cache the metadata
        let cached_metadata = CachedMetadata {
            metadata: metadata.clone(),
            parquet_footer: None, // Will be populated on first Parquet read
            bloom_filter: None,   // Will be populated if available
            cached_at: Instant::now(),
            ttl: Duration::from_secs(self.config.behavior.default_ttl_secs),
            access_count: 1,
            size_bytes: 0, // Will be calculated by cache
        };

        self.metadata_cache.put(cache_key, cached_metadata).await;

        Ok(metadata)
    }

    async fn list(&self, path: &str) -> FsResult<Vec<DirEntry>> {
        // Directory listings are not cached (too dynamic)
        // But we can prefetch metadata for listed files
        let entries = self.underlying_fs.list(path).await?;

        if self.config.io.enable_prefetching {
            // Prefetch metadata for all listed files in background
            let metadata_cache = self.metadata_cache.clone();
            let underlying_fs = self.underlying_fs.clone();
            let collection_id = self.collection_id.clone();
            let engine_type = self.engine_type.clone();
            let entries_clone = entries.clone();

            tokio::spawn(async move {
                for entry in &entries_clone {
                    if !entry.metadata.is_directory {
                        let cache_key = format!("{}:{}:{}", entry.url, collection_id, engine_type);
                        if metadata_cache.get(&cache_key).await.is_none()
                            && let Ok(metadata) = underlying_fs.metadata(&entry.url).await
                        {
                            let cached = CachedMetadata {
                                metadata,
                                parquet_footer: None,
                                bloom_filter: None,
                                cached_at: Instant::now(),
                                access_count: 0,
                                size_bytes: 0,
                                ttl: Duration::from_secs(300),
                            };
                            metadata_cache.put(cache_key, cached).await;
                        }
                    }
                }
            });
        }

        Ok(entries)
    }

    async fn create_dir(&self, path: &str) -> FsResult<()> {
        self.underlying_fs.create_dir(path).await
    }

    async fn create_dir_all(&self, path: &str) -> FsResult<()> {
        self.underlying_fs.create_dir_all(path).await
    }

    async fn remove_dir_all(&self, path: &str) -> FsResult<()> {
        // Invalidate all cached entries under this directory
        self.metadata_cache.invalidate_prefix(path).await;
        self.disk_cache.invalidate_prefix(path).await;

        self.underlying_fs.remove_dir_all(path).await
    }

    async fn copy(&self, from: &str, to: &str) -> FsResult<()> {
        // Copy in underlying filesystem
        self.underlying_fs.copy(from, to).await?;

        // If source is cached, we can cache the destination too
        if let Some(data) = self.disk_cache.get_data(from).await {
            self.disk_cache.put(to, &data).await;
        }

        Ok(())
    }

    async fn sync(&self) -> FsResult<()> {
        self.underlying_fs.sync().await
    }

    // TD-FSDECOR-1: this cache is write-through (writes invalidate and go
    // straight to `underlying_fs`), so a durability barrier is the underlying
    // filesystem's own. Without these, the trait defaults made both a silent
    // no-op — and this layer wraps the storage engines unconditionally, so e.g.
    // RAPTOR's flush-footer `sync_file` never reached disk.
    async fn sync_file(&self, path: &str) -> FsResult<()> {
        self.underlying_fs.sync_file(path).await
    }

    async fn sync_data(&self, path: &str) -> FsResult<()> {
        self.underlying_fs.sync_data(path).await
    }

    // TD-CACHECAS-1: conditional replace is the underlying filesystem's, and
    // the trait defaults made this layer report it unsupported (so callers
    // that need it — leases, queue offsets — could not run through it). A
    // swap changes the bytes behind the cached copy, so the copy is dropped
    // before the swap (no stale copy is served while it runs) and again after
    // it (a reader that re-cached the old bytes mid-swap does not keep them).
    fn supports_conditional_replace(&self) -> bool {
        self.underlying_fs.supports_conditional_replace()
    }

    async fn compare_exchange(
        &self,
        path: &str,
        expected: Option<&[u8]>,
        replacement: &[u8],
        guards: &[(&str, Option<&[u8]>)],
    ) -> FsResult<bool> {
        let cache_key = self.cache_key(path);
        self.metadata_cache.invalidate(&cache_key).await;
        self.disk_cache.invalidate(path).await;
        let swapped = self
            .underlying_fs
            .compare_exchange(path, expected, replacement, guards)
            .await;
        self.metadata_cache.invalidate(&cache_key).await;
        self.disk_cache.invalidate(path).await;
        swapped
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn open_file(&self, path: &str, create: bool) -> FsResult<Box<dyn FilesystemFile>> {
        self.underlying_fs.open_file(path, create).await
    }

    /// Override default read_range to delegate to underlying filesystem properly
    /// CRITICAL: This must be in the FileSystem trait impl, not the inherent impl,
    /// otherwise the default trait implementation is used when calling through `&dyn FileSystem`.
    async fn read_range(&self, path: &str, offset: u64, length: u64) -> FsResult<Vec<u8>> {
        self.record_access(path, AccessOperation::RangeRead).await;

        if let Some(cached_data) = self.disk_cache.get_data(path).await
            && let Ok(start) = usize::try_from(offset)
            && let Ok(len) = usize::try_from(length)
            && let Some(end) = start.checked_add(len)
            && end <= cached_data.len()
        {
            self.metrics
                .record_cache_hit_with_size(CacheType::Disk, len);
            return Ok(cached_data[start..end].to_vec());
        }

        self.metrics.record_cache_miss(CacheType::Disk);
        self.underlying_fs.read_range(path, offset, length).await
    }

    /// Delegate mmap to underlying filesystem for zero-copy reads
    /// This enables memory-mapped access when the underlying fs is LocalFileSystem
    async fn get_mmap(&self, path: &str) -> FsResult<Option<memmap2::Mmap>> {
        self.underlying_fs.get_mmap(path).await
    }

    /// Check if underlying filesystem supports memory mapping
    fn supports_mmap(&self) -> bool {
        self.underlying_fs.supports_mmap()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum AccessOperation {
    Read,
    Write,
    RangeRead,
    Metadata,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CacheType {
    Metadata,
    Disk,
    Memory,
}

/// Filesystem cache metrics
#[derive(Debug, Clone)]
pub struct FilesystemMetrics {
    pub metadata_cache_size: usize,
    pub metadata_cache_entries: usize,
    pub disk_cache_size: usize,
    pub disk_cache_entries: usize,
    pub total_hits: u64,
    pub total_misses: u64,
    pub hit_rate: f64,
}

impl UnifiedCachingFilesystem {
    async fn optimized_range_read(&self, path: &str, metadata: &FileMetadata) -> FsResult<Vec<u8>> {
        // Use engine-aware range optimization
        let ranges = self
            .range_optimizer
            .optimize_engine_ranges(
                path,
                metadata.size,
                &self.engine_type,
                None, // column_indices
                None, // row_group_indices
            )
            .await;

        if ranges.is_empty() {
            // No optimization possible, read full file
            return self.underlying_fs.read(path).await;
        }

        // Read optimized ranges and combine
        let mut data = Vec::with_capacity(metadata.size as usize);
        for range in ranges {
            // FIX: Third parameter is length, not end offset
            let length = range.end - range.start;
            let chunk = self.read_range(path, range.start, length).await?;
            data.extend_from_slice(&chunk);
        }

        Ok(data)
    }
}

/// Builder for UnifiedCachingFilesystem
pub struct UnifiedFilesystemBuilder {
    underlying_fs: Option<Arc<dyn FileSystem>>,
    config: Option<UnifiedCacheConfig>,
    collection_id: Option<String>,
    engine_type: Option<String>,
}

impl UnifiedFilesystemBuilder {
    pub fn new() -> Self {
        Self {
            underlying_fs: None,
            config: None,
            collection_id: None,
            engine_type: None,
        }
    }

    pub fn with_filesystem(mut self, fs: Arc<dyn FileSystem>) -> Self {
        self.underlying_fs = Some(fs);
        self
    }

    pub fn with_config(mut self, config: UnifiedCacheConfig) -> Self {
        self.config = Some(config);
        self
    }

    pub fn with_workload(mut self, workload: WorkloadType) -> Self {
        self.config = Some(UnifiedCacheConfig::from_workload(workload));
        self
    }

    pub fn with_collection(mut self, collection_id: String) -> Self {
        self.collection_id = Some(collection_id);
        self
    }

    pub fn with_engine(mut self, engine_type: String) -> Self {
        self.engine_type = Some(engine_type);
        self
    }

    pub fn build(self) -> Result<UnifiedCachingFilesystem, ProximaDBError> {
        let underlying_fs = self
            .underlying_fs
            .ok_or_else(|| ProximaDBError::InvalidInput("Filesystem required".into()))?;
        let collection_id = self.collection_id.unwrap_or_else(|| "default".to_string());
        let engine_type = self.engine_type.unwrap_or_else(|| "unknown".to_string());
        let config = self.config.unwrap_or_default();

        Ok(UnifiedCachingFilesystem::with_config(
            underlying_fs,
            config,
            collection_id,
            engine_type,
        ))
    }
}

impl Default for UnifiedFilesystemBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};
    use tempfile::TempDir;

    async fn test_caching_fs() -> (UnifiedCachingFilesystem, TempDir) {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let local = LocalFileSystem::new(LocalConfig {
            root_dir: Some(temp_dir.path().to_path_buf()),
            ..LocalConfig::default()
        })
        .await
        .expect("local filesystem");

        (
            UnifiedCachingFilesystem::new(
                Arc::new(local),
                "test_collection".to_string(),
                "sst".to_string(),
            ),
            temp_dir,
        )
    }

    /// TD-OBJSTORE-4 S2/S3: the caching decorator MUST delegate `write_if_absent`
    /// to the underlying backend's atomic conditional-create, not emulate it with an
    /// overwriting `write` — the crash-window recovery commit primitive rests on it.
    #[tokio::test]
    async fn write_if_absent_delegates_through_caching_wrapper() {
        let (fs, _temp_dir) = test_caching_fs().await;

        fs.write_if_absent("commit.pax", b"first", None)
            .await
            .expect("first conditional create succeeds");

        let err = fs
            .write_if_absent("commit.pax", b"second", None)
            .await
            .expect_err("second conditional create on the same key must fail");
        assert!(
            matches!(
                err,
                crate::storage::persistence::filesystem::FilesystemError::AlreadyExists(_)
            ),
            "caching wrapper must surface the backend's AlreadyExists; got {err:?}"
        );

        assert_eq!(
            fs.read("commit.pax").await.expect("read back"),
            b"first",
            "first value must be preserved (no clobber) through the caching wrapper"
        );
    }

    #[tokio::test]
    async fn read_range_records_access_for_pattern_learning() {
        let (fs, _temp_dir) = test_caching_fs().await;
        fs.write("range-source.bin", b"0123456789", None)
            .await
            .expect("write test file");

        let range = fs
            .read_range("range-source.bin", 2, 4)
            .await
            .expect("range read");

        assert_eq!(range, b"2345");
        assert!(fs.metrics.get_report().await.total_accesses >= 2);
    }

    #[tokio::test]
    async fn read_range_uses_cached_full_file_body_when_available() {
        let (fs, _temp_dir) = test_caching_fs().await;
        fs.disk_cache.put("cached-source.bin", b"abcdefghij").await;

        let range = fs
            .read_range("cached-source.bin", 2, 3)
            .await
            .expect("cached range read");

        assert_eq!(range, b"cde");
        assert!(fs.get_metrics().await.total_hits >= 1);
    }
}

#[cfg(test)]
mod sync_forwarding_tests {
    use std::sync::Arc;

    use proximadb_storage_filesystem_types::testing::SyncRecorder;

    use super::{FileSystem, UnifiedCachingFilesystem};

    /// TD-FSDECOR-1: the always-on caching layer must pass both durability
    /// barriers to the filesystem it caches.
    #[tokio::test]
    async fn caching_filesystem_forwards_durability_barriers() {
        let inner = Arc::new(SyncRecorder::default());
        let cached =
            UnifiedCachingFilesystem::new(inner.clone(), "c".to_string(), "sst".to_string());
        cached.sync_file("seg.pax").await.expect("sync_file");
        cached.sync_data("wal.log").await.expect("sync_data");
        assert_eq!(
            inner.synced(),
            vec!["file seg.pax".to_string(), "data wal.log".to_string()]
        );
    }
}

#[cfg(test)]
mod cas_forwarding_tests {
    use std::sync::Arc;

    use crate::storage::persistence::filesystem::local::{LocalConfig, LocalFileSystem};

    use super::{FileSystem, UnifiedCacheConfig, UnifiedCachingFilesystem};

    /// TD-CACHECAS-1: a conditional replace goes to the cached filesystem, and
    /// a swap drops the cached copy — a read after it must never be served the
    /// pre-swap bytes from the disk cache.
    #[tokio::test]
    async fn caching_filesystem_forwards_compare_exchange_and_invalidates() {
        let data_dir = tempfile::tempdir().expect("data dir");
        let cache_dir = tempfile::tempdir().expect("cache dir");
        let local = LocalFileSystem::new(LocalConfig {
            root_dir: Some(data_dir.path().to_path_buf()),
            follow_symlinks: true,
            default_permissions: None,
            sync_enabled: false,
        })
        .await
        .expect("local fs");
        let mut config = UnifiedCacheConfig::default();
        config.disk.path = cache_dir.path().to_path_buf();
        let cached = UnifiedCachingFilesystem::with_config(
            Arc::new(local),
            config,
            "c".to_string(),
            "sst".to_string(),
        );

        assert!(cached.supports_conditional_replace());
        assert!(
            cached
                .compare_exchange("lease.meta", None, b"first", &[])
                .await
                .expect("create")
        );
        // Read until the path is hot and the disk cache holds a copy.
        for _ in 0..6 {
            assert_eq!(cached.read("lease.meta").await.expect("read"), b"first");
        }
        assert!(
            cached.disk_cache.get("lease.meta").await.is_some(),
            "precondition: reads of the path are served from the disk cache"
        );

        assert!(
            cached
                .compare_exchange("lease.meta", Some(b"first"), b"second", &[])
                .await
                .expect("swap")
        );
        assert!(
            cached.disk_cache.get("lease.meta").await.is_none(),
            "a swap must drop the cached copy"
        );
        assert_eq!(cached.read("lease.meta").await.expect("read"), b"second");

        // A stale expectation does not swap, and the value stands.
        assert!(
            !cached
                .compare_exchange("lease.meta", Some(b"first"), b"bad", &[])
                .await
                .expect("stale swap")
        );
        assert_eq!(cached.read("lease.meta").await.expect("read"), b"second");
    }
}
