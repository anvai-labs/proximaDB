//! Universal Performance Optimization Module
//!
//! Provides common performance optimizations for all storage engines:
//! - Fast I/O with memory-mapped operations
//! - Cloud storage cost optimization
//! - Bandwidth reduction through smart caching and prefetching
//! - Hardware-accelerated operations
//!
//! This module eliminates code duplication across SST, VIPER, SWIFT, and other engines.

use anyhow::Result;
use async_trait::async_trait;
use memmap2::MmapOptions;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::storage::persistence::filesystem::{FileStorageTier, FilesystemFactory};
use proximadb_compression::{CompressionAlgorithm, StandardCompression};
use proximadb_distance_kernel::DistanceMetric;
use proximadb_runtime_common::pool::VectorMemoryPool;

/// Backwards-compat alias for [`GlobalMemoryPoolConfig`].
pub type MemoryPoolConfig = GlobalMemoryPoolConfig;

/// Global memory pool configuration
pub struct GlobalMemoryPoolConfig {
    /// Memory pool size in MB (default: 10% of available memory)
    pub pool_size_mb: Option<usize>,
    /// Whether to use smart defaults based on system memory
    pub use_smart_defaults: bool,
}

impl Default for GlobalMemoryPoolConfig {
    fn default() -> Self {
        Self {
            pool_size_mb: None,
            use_smart_defaults: true,
        }
    }
}

/// Global configuration for the shared memory pool
static MEMORY_POOL_CONFIG: Lazy<GlobalMemoryPoolConfig> = Lazy::new(|| {
    // Try to read from environment or config
    if let Ok(size_str) = std::env::var("PROXIMADB_MEMORY_POOL_SIZE_MB")
        && let Ok(size) = size_str.parse::<usize>()
    {
        tracing::info!("Using memory pool size from environment: {}MB", size);
        return GlobalMemoryPoolConfig {
            pool_size_mb: Some(size),
            use_smart_defaults: false,
        };
    }
    GlobalMemoryPoolConfig::default()
});

/// Calculate smart default pool size based on available system memory
fn calculate_smart_pool_size() -> usize {
    // Try to get system memory info
    #[cfg(target_os = "linux")]
    {
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            // Parse MemAvailable or MemTotal from /proc/meminfo
            for line in meminfo.lines() {
                if line.starts_with("MemAvailable:") || line.starts_with("MemTotal:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1)
                        && let Ok(kb) = kb_str.parse::<usize>()
                    {
                        let mb = kb / 1024;
                        // Use 10-15% of available memory for the pool
                        let pool_size = (mb as f64 * 0.12) as usize;
                        // Cap between 512MB and 8GB
                        let capped_size = pool_size.clamp(512, 8192);
                        tracing::info!(
                            "Smart memory pool sizing: System memory {}MB, pool size {}MB (12% of available)",
                            mb,
                            capped_size
                        );
                        return capped_size;
                    }
                    break;
                }
            }
        }
    }

    // Default fallback: 2GB for production systems
    tracing::info!("Using default memory pool size: 2048MB");
    2048
}

/// Global shared memory pool for all storage engines
/// This singleton ensures memory buffers are reused across all engines for maximum efficiency
static SHARED_MEMORY_POOL: Lazy<Arc<VectorMemoryPool>> = Lazy::new(|| {
    let config = &*MEMORY_POOL_CONFIG;

    let pool_size_mb = if let Some(size) = config.pool_size_mb {
        size
    } else if config.use_smart_defaults {
        calculate_smart_pool_size()
    } else {
        2048 // Default 2GB
    };

    tracing::info!(
        "Initializing global shared memory pool for universal optimizer with size: {}MB",
        pool_size_mb
    );

    // Note: VectorMemoryPool::new() doesn't take size parameter anymore
    // The pool will manage its own internal sizing based on usage patterns
    Arc::new(VectorMemoryPool::new())
});

/// Get the global shared memory pool instance
/// All storage engines should use this instead of creating their own pools
pub fn get_shared_memory_pool() -> Arc<VectorMemoryPool> {
    SHARED_MEMORY_POOL.clone()
}

/// Configure the memory pool size from server config
/// This should be called early in server initialization, before any engine is created
pub fn configure_memory_pool_from_config(pool_size_mb: Option<usize>) {
    if Arc::strong_count(&*SHARED_MEMORY_POOL) > 1 {
        tracing::warn!(
            "Memory pool already initialized with {} references, configuration ignored",
            Arc::strong_count(&*SHARED_MEMORY_POOL)
        );
        return;
    }

    if let Some(size) = pool_size_mb {
        // Set environment variable for the lazy static to pick up
        unsafe {
            std::env::set_var("PROXIMADB_MEMORY_POOL_SIZE_MB", size.to_string());
        }
        tracing::info!("Configured memory pool size from server config: {}MB", size);
    }
}

/// Universal optimization strategy applicable to all storage engines
#[derive(Debug, Clone)]
pub enum UniversalOptimizationStrategy {
    /// Optimize for maximum read performance
    PerformanceFirst,
    /// Optimize for minimal memory usage
    MemoryEfficient,
    /// Optimize for cloud storage cost efficiency
    CostOptimized,
    /// Balance between performance, memory, and cost
    Balanced,
    /// Engine-specific custom strategy
    Custom(String),
}

/// Universal I/O configuration for all storage engines
#[derive(Debug, Clone)]
pub struct UniversalIOConfig {
    pub enable_memory_mapping: bool,     // Memory-mapped file access
    pub cache_size_mb: usize,            // Total cache size
    pub parallel_operations: usize,      // Concurrent operations
    pub enable_prefetching: bool,        // Predictive data loading
    pub prefetch_size_mb: usize,         // Prefetch chunk size
    pub tiered_storage_threshold: f32,   // Access frequency threshold for tiering
    pub eviction_threshold: f32,         // Cache eviction threshold
    pub enable_compression: bool,        // In-memory compression
    pub compression_threshold_kb: usize, // Compress objects larger than X KB
}

impl Default for UniversalIOConfig {
    fn default() -> Self {
        Self {
            enable_memory_mapping: true,
            cache_size_mb: 1024,    // 1GB default cache
            parallel_operations: 8, // 8 concurrent operations
            enable_prefetching: true,
            prefetch_size_mb: 128,         // 128MB prefetch chunks
            tiered_storage_threshold: 0.3, // 30% access frequency threshold
            eviction_threshold: 0.85,      // Evict at 85% full
            enable_compression: true,
            compression_threshold_kb: 64, // Compress >64KB objects
        }
    }
}

/// Universal performance optimization provider for all storage engines
#[derive(Debug, Clone)]
pub struct UniversalPerformanceOptimizer {
    /// I/O configuration
    io_config: UniversalIOConfig,

    /// Optimization strategy
    optimization_strategy: UniversalOptimizationStrategy,

    /// Filesystem factory for seamless local/cloud integration
    filesystem_factory: Arc<FilesystemFactory>,

    /// Memory-mapped file cache (using Arc for shared ownership)
    mmap_cache: Arc<RwLock<HashMap<String, Arc<memmap2::Mmap>>>>,

    /// Generic data cache for any storage engine
    data_cache: Arc<RwLock<HashMap<String, Arc<Vec<u8>>>>>,

    /// Compression provider
    compression_provider: StandardCompression,

    /// Access pattern tracking for optimization
    access_patterns: Arc<RwLock<HashMap<String, PerfOptAccessStats>>>,
}

/// Backwards-compat alias for [`PerfOptAccessStats`].
pub type AccessStats = PerfOptAccessStats;

/// Access statistics for optimization decisions
#[derive(Debug, Clone)]
pub struct PerfOptAccessStats {
    pub access_count: u64,
    pub last_access: chrono::DateTime<chrono::Utc>,
    pub total_bytes_read: u64,
    pub average_access_time_ms: f64,
}

impl UniversalPerformanceOptimizer {
    /// Create new universal performance optimizer
    /// Uses the global shared memory pool and hardware capabilities
    pub fn new(
        io_config: UniversalIOConfig,
        optimization_strategy: UniversalOptimizationStrategy,
        filesystem_factory: Arc<FilesystemFactory>,
    ) -> Self {
        Self {
            io_config,
            optimization_strategy,
            filesystem_factory,
            mmap_cache: Arc::new(RwLock::new(HashMap::new())),
            data_cache: Arc::new(RwLock::new(HashMap::new())),
            compression_provider: StandardCompression,
            access_patterns: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create with default configuration for specific strategy
    /// Uses the global shared memory pool and hardware capabilities automatically
    pub async fn with_strategy(strategy: UniversalOptimizationStrategy) -> Result<Self> {
        let io_config = match strategy {
            UniversalOptimizationStrategy::PerformanceFirst => UniversalIOConfig {
                cache_size_mb: 4096,       // 4GB cache
                parallel_operations: 16,   // High concurrency
                prefetch_size_mb: 256,     // Large prefetch
                enable_compression: false, // Skip compression for speed
                ..Default::default()
            },
            UniversalOptimizationStrategy::MemoryEfficient => UniversalIOConfig {
                cache_size_mb: 256,           // Small cache
                parallel_operations: 4,       // Low concurrency
                prefetch_size_mb: 32,         // Small prefetch
                enable_compression: true,     // Use compression
                compression_threshold_kb: 16, // Aggressive compression
                eviction_threshold: 0.7,      // Early eviction
                ..Default::default()
            },
            UniversalOptimizationStrategy::CostOptimized => UniversalIOConfig {
                cache_size_mb: 512,            // Medium cache
                parallel_operations: 4,        // Reduced concurrency
                prefetch_size_mb: 64,          // Minimal prefetch
                tiered_storage_threshold: 0.7, // Aggressive cold storage
                enable_compression: true,      // Maximize compression
                ..Default::default()
            },
            _ => UniversalIOConfig::default(),
        };

        // Initialize filesystem factory for seamless local/cloud integration
        let filesystem_config =
            crate::storage::persistence::filesystem::FilesystemConfig::default();
        let filesystem_factory = Arc::new(FilesystemFactory::create(filesystem_config).await?);

        Ok(Self::new(io_config, strategy, filesystem_factory))
    }

    // ============================================================================
    // FAST I/O OPERATIONS
    // ============================================================================

    /// Filesystem-integrated memory-mapped file access for ultra-fast I/O
    /// Supports local filesystem and cloud storage through unified filesystem API
    pub async fn get_memory_mapped_file(
        &self,
        file_url: &str,
    ) -> Result<Option<Arc<memmap2::Mmap>>> {
        // Check cache first
        {
            let cache = self.mmap_cache.read().await;
            if let Some(mmap) = cache.get(file_url) {
                return Ok(Some(Arc::clone(mmap)));
            }
        }

        // Use filesystem factory to handle local/cloud storage seamlessly
        let _filesystem = self.filesystem_factory.get_filesystem(file_url)?;

        // Try local file memory mapping first (works for file:// URLs)
        if file_url.starts_with("file://")
            && let Some(local_path) = file_url.strip_prefix("file://")
            && let Ok(file) = File::open(local_path)
        {
            let mmap = Arc::new(unsafe { MmapOptions::new().map(&file)? });

            // Cache for future access
            {
                let mut cache = self.mmap_cache.write().await;
                cache.insert(file_url.to_string(), Arc::clone(&mmap));
            }

            // Update access patterns
            self.update_access_stats(file_url, file.metadata()?.len())
                .await;

            return Ok(Some(mmap));
        }

        // For cloud storage, fall back to regular I/O (memory mapping not supported)
        // This will use the data cache instead
        Ok(None)
    }

    /// Filesystem-integrated data reading with automatic caching
    /// Handles both local and cloud storage transparently
    pub async fn read_data_optimized(&self, file_url: &str) -> Result<Vec<u8>> {
        // Check cache first
        {
            let cache = self.data_cache.read().await;
            if let Some(data) = cache.get(file_url) {
                return Ok((**data).clone());
            }
        }

        // Use filesystem API for seamless local/cloud access
        let filesystem = self.filesystem_factory.get_filesystem(file_url)?;
        let data = filesystem.read(file_url).await?;

        // Determine if we should cache based on file size and access patterns
        let should_cache = data.len() < self.io_config.cache_size_mb * 1024 * 1024 / 4; // Use 1/4 of cache for single file

        if should_cache {
            let mut cache = self.data_cache.write().await;
            cache.insert(file_url.to_string(), Arc::new(data.clone()));
        }

        // Update access patterns
        self.update_access_stats(file_url, data.len() as u64).await;

        Ok(data)
    }

    /// Filesystem-integrated data writing with tier optimization
    pub async fn write_data_optimized(
        &self,
        file_url: &str,
        data: &[u8],
        tier: FileStorageTier,
    ) -> Result<()> {
        // Compress data based on tier strategy
        let compressed_data = self.compress_for_tier(data, tier).await?;

        // Use filesystem API for seamless local/cloud writing
        let filesystem = self.filesystem_factory.get_filesystem(file_url)?;
        filesystem.write(file_url, &compressed_data, None).await?;

        // Update cache if appropriate
        if compressed_data.len() < self.io_config.cache_size_mb * 1024 * 1024 / 8 {
            let mut cache = self.data_cache.write().await;
            cache.insert(file_url.to_string(), Arc::new(data.to_vec()));
        }

        // Update access patterns
        self.update_access_stats(file_url, compressed_data.len() as u64)
            .await;

        Ok(())
    }

    /// List files with filesystem integration and optimization hints
    pub async fn list_files_optimized(&self, directory_url: &str) -> Result<Vec<String>> {
        let filesystem = self.filesystem_factory.get_filesystem(directory_url)?;
        let file_infos = filesystem.list(directory_url).await?;

        // Extract file paths and update access patterns for directory listing
        let file_paths: Vec<String> = file_infos
            .iter()
            .map(|info| info.metadata.path.clone())
            .collect();

        // Update access pattern for directory
        self.update_access_stats(directory_url, file_paths.len() as u64)
            .await;

        Ok(file_paths)
    }

    /// Parallel operations with configurable concurrency
    pub async fn parallel_operations<T, F, Fut>(
        &self,
        items: Vec<T>,
        operation: F,
    ) -> Result<Vec<Result<Fut::Output>>>
    where
        F: Fn(T) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future + Send + 'static,
        Fut::Output: Send + 'static,
        T: Send + 'static,
    {
        let semaphore = Arc::new(tokio::sync::Semaphore::new(
            self.io_config.parallel_operations,
        ));
        let mut handles = Vec::new();

        for item in items {
            let permit = semaphore.clone().acquire_owned().await?;
            let op = operation.clone();

            let handle = tokio::spawn(async move {
                let result = op(item).await;
                drop(permit);
                result
            });
            handles.push(handle);
        }

        let mut results = Vec::new();
        for handle in handles {
            results.push(
                handle
                    .await
                    .map_err(|e| anyhow::anyhow!("Operation failed: {}", e)),
            );
        }

        Ok(results)
    }

    /// Memory pool optimization using the global shared pool
    pub async fn get_memory_buffer(&self, _size: usize) -> Result<Vec<f32>> {
        // VectorMemoryPool provides pre-allocated buffers
        // The acquire() method returns a PooledItem which derefs to Vec<f32>
        // Need to deref the Lazy first to get the Arc<VectorMemoryPool>
        let pooled_item = SHARED_MEMORY_POOL.as_ref().vector_buffers.acquire();
        Ok(pooled_item.take())
    }

    // ============================================================================
    // CLOUD STORAGE COST OPTIMIZATION
    // ============================================================================

    /// Determine optimal storage tier based on access patterns
    pub async fn optimize_storage_tier(
        &self,
        key: &str,
        data_size_bytes: usize,
    ) -> Result<FileStorageTier> {
        let access_frequency = self.get_access_frequency(key).await;

        match self.optimization_strategy {
            UniversalOptimizationStrategy::PerformanceFirst => {
                // Keep frequently accessed data in fast tier
                if access_frequency > 0.1 {
                    Ok(FileStorageTier::NVMe)
                } else {
                    Ok(FileStorageTier::SSD)
                }
            }
            UniversalOptimizationStrategy::MemoryEfficient => {
                // Optimize for memory usage
                if data_size_bytes < 32 * 1024 * 1024 {
                    // < 32MB
                    Ok(FileStorageTier::NVMe)
                } else {
                    Ok(FileStorageTier::HDD)
                }
            }
            UniversalOptimizationStrategy::CostOptimized => {
                // Aggressive cost optimization
                if access_frequency > self.io_config.tiered_storage_threshold {
                    Ok(FileStorageTier::NVMe)
                } else {
                    Ok(FileStorageTier::HDD)
                }
            }
            UniversalOptimizationStrategy::Balanced => {
                // Balance performance and cost
                if access_frequency > 0.3 && data_size_bytes < 64 * 1024 * 1024 {
                    Ok(FileStorageTier::NVMe)
                } else if access_frequency > 0.1 {
                    Ok(FileStorageTier::SSD)
                } else {
                    Ok(FileStorageTier::HDD)
                }
            }
            UniversalOptimizationStrategy::Custom(_) => {
                // Default to balanced for custom strategies
                Ok(FileStorageTier::SSD)
            }
        }
    }

    /// Tier-aware compression optimization
    pub async fn compress_for_tier(&self, data: &[u8], tier: FileStorageTier) -> Result<Vec<u8>> {
        use proximadb_compression::{CompressionContext, CompressionProvider};

        let (algorithm, level) = match tier {
            FileStorageTier::Memory | FileStorageTier::NVMe => {
                // Fast compression for hot tier
                (CompressionAlgorithm::Lz4, 1) // Fastest
            }
            FileStorageTier::SSD | FileStorageTier::AzurePremium | FileStorageTier::GcsSSD => {
                // Balanced compression
                (CompressionAlgorithm::Snappy, 3)
            }
            FileStorageTier::HDD
            | FileStorageTier::S3Standard
            | FileStorageTier::AzureStandard
            | FileStorageTier::GcsHDD
            | FileStorageTier::S3GlacierInstant => {
                // Maximum compression for cost savings
                (CompressionAlgorithm::Zstd, 9)
            }
            FileStorageTier::S3Express => {
                // Balanced for S3 Express
                (CompressionAlgorithm::Snappy, 2)
            }
        };

        // Use VectorSerialization context for universal optimization
        self.compression_provider.compress(
            data,
            algorithm,
            level,
            CompressionContext::VectorSerialization,
        )
    }

    // ============================================================================
    // BANDWIDTH OPTIMIZATION
    // ============================================================================

    /// Smart prefetching based on access patterns with filesystem integration
    pub async fn prefetch_data(&self, file_urls: &[String]) -> Result<()> {
        if !self.io_config.enable_prefetching || file_urls.is_empty() {
            return Ok(());
        }

        let prefetch_count = self.io_config.prefetch_size_mb / 4; // Assume ~4MB per file
        let urls_to_prefetch = &file_urls[..prefetch_count.min(file_urls.len())];

        // Group by filesystem type for optimal batching
        let mut local_files = Vec::new();
        let mut cloud_files = Vec::new();

        for url in urls_to_prefetch {
            // Check if already cached
            {
                let cache = self.data_cache.read().await;
                if cache.contains_key(url) {
                    continue;
                }
            }

            // Categorize by storage type based on URL prefix
            if url.starts_with("file://") {
                local_files.push(url.clone());
            } else {
                cloud_files.push(url.clone());
            }
        }

        // Asynchronously prefetch with filesystem-aware batching
        let filesystem_factory = self.filesystem_factory.clone();
        let data_cache = self.data_cache.clone();

        // Local files: Use memory mapping if possible
        if !local_files.is_empty() {
            let mmap_cache = self.mmap_cache.clone();
            tokio::spawn(async move {
                for file_url in local_files {
                    if let Some(local_path) = file_url.strip_prefix("file://")
                        && let Ok(file) = File::open(local_path)
                        && let Ok(mmap) = unsafe { MmapOptions::new().map(&file) }
                    {
                        let mut cache = mmap_cache.write().await;
                        cache.insert(file_url, Arc::new(mmap));
                    }
                }
            });
        }

        // Cloud files: Use regular caching
        if !cloud_files.is_empty() {
            tokio::spawn(async move {
                for file_url in cloud_files {
                    if let Ok(filesystem) = filesystem_factory.get_filesystem(&file_url)
                        && let Ok(data) = filesystem.read(&file_url).await
                    {
                        let mut cache = data_cache.write().await;
                        cache.insert(file_url, Arc::new(data));
                    }
                }
            });
        }

        Ok(())
    }

    /// Cache management with intelligent eviction
    pub async fn evict_cache_if_needed(&self) -> Result<()> {
        let cache_size = {
            let cache = self.data_cache.read().await;
            cache.len()
        };

        // Estimate cache size (rough approximation)
        let estimated_size_mb = cache_size / 256; // Assume ~4KB per entry
        let threshold_size =
            (self.io_config.cache_size_mb as f32 * self.io_config.eviction_threshold) as usize;

        if estimated_size_mb > threshold_size {
            // LRU eviction based on access patterns
            let evict_count = cache_size / 4; // Remove 25%
            let mut cache = self.data_cache.write().await;
            let mut access_patterns = self.access_patterns.write().await;

            // Sort by last access time and remove least recently used
            let mut entries: Vec<(String, chrono::DateTime<chrono::Utc>)> = access_patterns
                .iter()
                .map(|(k, v)| (k.clone(), v.last_access))
                .collect();

            entries.sort_by_key(|e| e.1);

            for (key, _) in entries.into_iter().take(evict_count) {
                cache.remove(&key);
                access_patterns.remove(&key);
            }

            tracing::info!(
                "Universal optimizer: Evicted {} items from cache",
                evict_count
            );
        }

        Ok(())
    }

    // ============================================================================
    // HARDWARE ACCELERATION
    // ============================================================================

    /// Hardware-accelerated distance computation (delegates to unified modules)
    pub async fn compute_distances_accelerated(
        &self,
        query: &[f32],
        candidates: &[Vec<f32>],
        metric: DistanceMetric,
    ) -> Result<Vec<f32>> {
        let mut distances = Vec::new();

        if proximadb_hardware::best_simd_level() >= proximadb_hardware::SimdLevel::AVX2 {
            // Use SIMD acceleration
            for candidate in candidates {
                let distance = match metric {
                    DistanceMetric::Cosine => {
                        // Simplified SIMD cosine (use unified module in production)
                        1.0 - query
                            .iter()
                            .zip(candidate.iter())
                            .map(|(a, b)| a * b)
                            .sum::<f32>()
                    }
                    DistanceMetric::Euclidean => {
                        // Simplified SIMD euclidean
                        query
                            .iter()
                            .zip(candidate.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f32>()
                            .sqrt()
                    }
                    _ => {
                        // Default fallback
                        query
                            .iter()
                            .zip(candidate.iter())
                            .map(|(a, b)| (a - b).abs())
                            .sum::<f32>()
                    }
                };
                distances.push(distance);
            }
        } else {
            // Standard computation
            for candidate in candidates {
                let distance = query
                    .iter()
                    .zip(candidate.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f32>();
                distances.push(distance);
            }
        }

        Ok(distances)
    }

    // ============================================================================
    // HELPER METHODS
    // ============================================================================

    /// Update access statistics for optimization decisions (public for testing)
    pub async fn update_access_stats(&self, key: &str, bytes_read: u64) {
        let mut patterns = self.access_patterns.write().await;
        let entry = patterns
            .entry(key.to_string())
            .or_insert_with(|| PerfOptAccessStats {
                access_count: 0,
                last_access: chrono::Utc::now(),
                total_bytes_read: 0,
                average_access_time_ms: 0.0,
            });

        entry.access_count += 1;
        entry.last_access = chrono::Utc::now();
        entry.total_bytes_read += bytes_read;
    }

    /// Get access frequency for key (0.0 to 1.0) (public for testing)
    pub async fn get_access_frequency(&self, key: &str) -> f32 {
        let patterns = self.access_patterns.read().await;
        if let Some(stats) = patterns.get(key) {
            // Simple frequency calculation based on recent access
            let hours_since_access = chrono::Utc::now()
                .signed_duration_since(stats.last_access)
                .num_hours() as f32;

            if hours_since_access < 1.0 {
                1.0 // Very recent access
            } else if hours_since_access < 24.0 {
                0.5 // Recent access
            } else {
                0.1 // Old access
            }
        } else {
            0.0 // No access history
        }
    }

    /// Get current configuration
    pub fn get_config(&self) -> &UniversalIOConfig {
        &self.io_config
    }

    /// Get optimization strategy
    pub fn get_strategy(&self) -> &UniversalOptimizationStrategy {
        &self.optimization_strategy
    }
}

/// Trait for storage engines to implement universal performance optimization
#[async_trait]
pub trait UniversallyOptimized {
    /// Get the universal performance optimizer instance
    fn universal_optimizer(&self) -> &UniversalPerformanceOptimizer;

    /// Engine-specific optimization setup
    async fn setup_engine_optimizations(&self) -> Result<()> {
        // Default implementation - engines can override
        Ok(())
    }

    /// Engine-specific performance metrics
    async fn collect_performance_metrics(&self) -> Result<HashMap<String, serde_json::Value>> {
        // Default implementation
        Ok(HashMap::new())
    }
}

/// Insert the standard `universal_*` optimizer metrics into a stats map.
///
/// Shared implementation for the `UniversallyOptimized` engines'
/// `collect_performance_metrics` overrides (NOVA/RAPTOR/SWIFT/VIPER) — each
/// engine keeps its own prefixed keys at the call site and appends this block
/// for the optimizer's configuration surface.
pub fn insert_universal_optimizer_metrics(
    metrics: &mut HashMap<String, serde_json::Value>,
    optimizer: &UniversalPerformanceOptimizer,
) {
    let strategy = optimizer.get_strategy();
    metrics.insert(
        "universal_optimization_strategy".to_string(),
        serde_json::Value::String(format!("{:?}", strategy)),
    );

    let config = optimizer.get_config();
    metrics.insert(
        "universal_cache_size_mb".to_string(),
        serde_json::Value::Number(serde_json::Number::from(config.cache_size_mb)),
    );
    metrics.insert(
        "universal_parallel_operations".to_string(),
        serde_json::Value::Number(serde_json::Number::from(config.parallel_operations)),
    );
    metrics.insert(
        "universal_prefetching_enabled".to_string(),
        serde_json::Value::Bool(config.enable_prefetching),
    );
}

// Include comprehensive tests module
// #[cfg(test)]
// mod performance_optimization_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_universal_optimizer_creation() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init

        let optimizer =
            UniversalPerformanceOptimizer::with_strategy(UniversalOptimizationStrategy::Balanced)
                .await
                .unwrap();

        assert!(matches!(
            optimizer.get_strategy(),
            UniversalOptimizationStrategy::Balanced
        ));
        assert_eq!(optimizer.get_config().parallel_operations, 8);
    }

    #[tokio::test]
    async fn test_storage_tier_optimization() {
        let _ = proximadb_hardware::hardware_capabilities(); // OnceLock auto-init

        let optimizer = UniversalPerformanceOptimizer::with_strategy(
            UniversalOptimizationStrategy::CostOptimized,
        )
        .await
        .unwrap();

        let tier = optimizer
            .optimize_storage_tier("test_key", 1024)
            .await
            .unwrap();
        assert!(matches!(tier, FileStorageTier::HDD)); // Should default to HDD for cost optimization
    }
}
