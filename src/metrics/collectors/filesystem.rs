//! Filesystem metrics collector for unified metrics framework
//! Integrates zero-copy filesystem and cache metrics

use super::{MetricsCollector, MetricsSample};
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Canonical storage engine type (re-exported from `proximadb-storage-common` via
/// `core::types`). The previous local 6-variant copy (a subset missing TST) was
/// consolidated into the single source of truth.
pub use crate::core::types::StorageEngineType;

/// File operation types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FileOperation {
    Read,
    Write,
}

/// Stats for a specific storage engine
#[derive(Debug, Clone)]
pub struct EngineFileStats {
    pub files_read: u64,
    pub files_written: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
}

/// Filesystem metrics collector that integrates with unified framework
pub struct FilesystemMetricsCollector {
    /// Reference to the zero-copy filesystem metrics
    zerocopy_metrics: Arc<ZeroCopyMetrics>,

    /// Reference to general filesystem metrics
    general_metrics: Arc<GeneralFilesystemMetrics>,
}

/// Zero-copy filesystem specific metrics
pub struct ZeroCopyMetrics {
    // Cache metrics
    pub memory_cache_hits: AtomicU64,
    pub memory_cache_misses: AtomicU64,
    pub memory_cache_size_bytes: AtomicU64,
    pub memory_cache_entries: AtomicU64,
    pub memory_cache_evictions: AtomicU64,

    pub disk_cache_hits: AtomicU64,
    pub disk_cache_misses: AtomicU64,
    pub disk_cache_size_bytes: AtomicU64,
    pub disk_cache_entries: AtomicU64,
    pub disk_cache_evictions: AtomicU64,

    // Latency tracking
    pub total_cache_hit_latency_ns: AtomicU64,
    pub total_cache_miss_latency_ns: AtomicU64,
    pub cache_hit_count_for_latency: AtomicU64,
    pub cache_miss_count_for_latency: AtomicU64,

    // Metadata cache metrics
    pub metadata_cache_hits: AtomicU64,
    pub metadata_cache_misses: AtomicU64,
    pub files_skipped: AtomicU64,
    pub bytes_saved_by_skipping: AtomicU64,

    // Download optimization metrics
    pub selective_downloads: AtomicU64,
    pub full_downloads: AtomicU64,
    pub total_bytes_downloaded: AtomicU64,
    pub total_bytes_saved: AtomicU64,

    // Zero-copy operation metrics
    pub zerocopy_operations: AtomicU64,
    pub zerocopy_bytes_transferred: AtomicU64,
    pub sendfile_operations: AtomicU64,
    pub mmap_operations: AtomicU64,
}

/// General filesystem metrics
pub struct GeneralFilesystemMetrics {
    // I/O operations
    pub read_operations: AtomicU64,
    pub write_operations: AtomicU64,
    pub delete_operations: AtomicU64,
    pub list_operations: AtomicU64,

    // Bytes transferred
    pub bytes_read: AtomicU64,
    pub bytes_written: AtomicU64,

    // Error tracking
    pub read_errors: AtomicU64,
    pub write_errors: AtomicU64,
    pub permission_errors: AtomicU64,
    pub not_found_errors: AtomicU64,

    // Cloud storage specific
    pub s3_operations: AtomicU64,
    pub gcs_operations: AtomicU64,
    pub azure_operations: AtomicU64,
    pub local_operations: AtomicU64,

    // Bandwidth optimization
    pub parallel_downloads: AtomicU64,
    pub chunked_uploads: AtomicU64,
    pub multipart_uploads: AtomicU64,

    // Storage engine file type metrics
    pub sst_files_read: AtomicU64,
    pub sst_files_written: AtomicU64,
    pub nova_files_read: AtomicU64,
    pub nova_files_written: AtomicU64,
    pub raptor_files_read: AtomicU64,
    pub raptor_files_written: AtomicU64,
    pub swift_files_read: AtomicU64,
    pub swift_files_written: AtomicU64,
    pub helix_files_read: AtomicU64,
    pub helix_files_written: AtomicU64,

    // Engine-specific byte tracking
    pub sst_bytes_read: AtomicU64,
    pub sst_bytes_written: AtomicU64,
    pub nova_bytes_read: AtomicU64,
    pub nova_bytes_written: AtomicU64,
    pub raptor_bytes_read: AtomicU64,
    pub raptor_bytes_written: AtomicU64,
    pub swift_bytes_read: AtomicU64,
    pub swift_bytes_written: AtomicU64,
    pub helix_bytes_read: AtomicU64,
    pub helix_bytes_written: AtomicU64,
}

impl FilesystemMetricsCollector {
    pub fn new() -> Self {
        Self {
            zerocopy_metrics: Arc::new(ZeroCopyMetrics::new()),
            general_metrics: Arc::new(GeneralFilesystemMetrics::new()),
        }
    }

    /// Get reference to zero-copy metrics for direct updates
    pub fn zerocopy_metrics(&self) -> Arc<ZeroCopyMetrics> {
        self.zerocopy_metrics.clone()
    }

    /// Get reference to general metrics for direct updates
    pub fn general_metrics(&self) -> Arc<GeneralFilesystemMetrics> {
        self.general_metrics.clone()
    }

    /// Track file operation for specific storage engine
    pub fn track_engine_file_operation(
        &self,
        engine_type: StorageEngineType,
        operation: FileOperation,
        bytes: u64,
    ) {
        match (engine_type, operation) {
            (StorageEngineType::SST, FileOperation::Read) => {
                self.general_metrics
                    .sst_files_read
                    .fetch_add(1, Ordering::Relaxed);
                self.general_metrics
                    .sst_bytes_read
                    .fetch_add(bytes, Ordering::Relaxed);
            }
            (StorageEngineType::SST, FileOperation::Write) => {
                self.general_metrics
                    .sst_files_written
                    .fetch_add(1, Ordering::Relaxed);
                self.general_metrics
                    .sst_bytes_written
                    .fetch_add(bytes, Ordering::Relaxed);
            }
            (StorageEngineType::NOVA, FileOperation::Read) => {
                self.general_metrics
                    .nova_files_read
                    .fetch_add(1, Ordering::Relaxed);
                self.general_metrics
                    .nova_bytes_read
                    .fetch_add(bytes, Ordering::Relaxed);
            }
            (StorageEngineType::NOVA, FileOperation::Write) => {
                self.general_metrics
                    .nova_files_written
                    .fetch_add(1, Ordering::Relaxed);
                self.general_metrics
                    .nova_bytes_written
                    .fetch_add(bytes, Ordering::Relaxed);
            }
            (StorageEngineType::RAPTOR, FileOperation::Read) => {
                self.general_metrics
                    .raptor_files_read
                    .fetch_add(1, Ordering::Relaxed);
                self.general_metrics
                    .raptor_bytes_read
                    .fetch_add(bytes, Ordering::Relaxed);
            }
            (StorageEngineType::RAPTOR, FileOperation::Write) => {
                self.general_metrics
                    .raptor_files_written
                    .fetch_add(1, Ordering::Relaxed);
                self.general_metrics
                    .raptor_bytes_written
                    .fetch_add(bytes, Ordering::Relaxed);
            }
            (StorageEngineType::SWIFT, FileOperation::Read) => {
                self.general_metrics
                    .swift_files_read
                    .fetch_add(1, Ordering::Relaxed);
                self.general_metrics
                    .swift_bytes_read
                    .fetch_add(bytes, Ordering::Relaxed);
            }
            (StorageEngineType::SWIFT, FileOperation::Write) => {
                self.general_metrics
                    .swift_files_written
                    .fetch_add(1, Ordering::Relaxed);
                self.general_metrics
                    .swift_bytes_written
                    .fetch_add(bytes, Ordering::Relaxed);
            }
            (StorageEngineType::HELIX, FileOperation::Read) => {
                self.general_metrics
                    .helix_files_read
                    .fetch_add(1, Ordering::Relaxed);
                self.general_metrics
                    .helix_bytes_read
                    .fetch_add(bytes, Ordering::Relaxed);
            }
            (StorageEngineType::HELIX, FileOperation::Write) => {
                self.general_metrics
                    .helix_files_written
                    .fetch_add(1, Ordering::Relaxed);
                self.general_metrics
                    .helix_bytes_written
                    .fetch_add(bytes, Ordering::Relaxed);
            }
            // TST (and any future engine) has no dedicated per-engine counter yet;
            // only the general counters below are updated.
            _ => {}
        }

        // Also update general counters
        match operation {
            FileOperation::Read => {
                self.general_metrics
                    .read_operations
                    .fetch_add(1, Ordering::Relaxed);
                self.general_metrics
                    .bytes_read
                    .fetch_add(bytes, Ordering::Relaxed);
            }
            FileOperation::Write => {
                self.general_metrics
                    .write_operations
                    .fetch_add(1, Ordering::Relaxed);
                self.general_metrics
                    .bytes_written
                    .fetch_add(bytes, Ordering::Relaxed);
            }
        }
    }

    /// Get engine-specific file statistics
    pub fn engine_stats(&self, engine_type: StorageEngineType) -> EngineFileStats {
        match engine_type {
            StorageEngineType::SST => EngineFileStats {
                files_read: self.general_metrics.sst_files_read.load(Ordering::Relaxed),
                files_written: self
                    .general_metrics
                    .sst_files_written
                    .load(Ordering::Relaxed),
                bytes_read: self.general_metrics.sst_bytes_read.load(Ordering::Relaxed),
                bytes_written: self
                    .general_metrics
                    .sst_bytes_written
                    .load(Ordering::Relaxed),
            },
            StorageEngineType::NOVA => EngineFileStats {
                files_read: self.general_metrics.nova_files_read.load(Ordering::Relaxed),
                files_written: self
                    .general_metrics
                    .nova_files_written
                    .load(Ordering::Relaxed),
                bytes_read: self.general_metrics.nova_bytes_read.load(Ordering::Relaxed),
                bytes_written: self
                    .general_metrics
                    .nova_bytes_written
                    .load(Ordering::Relaxed),
            },
            StorageEngineType::RAPTOR => EngineFileStats {
                files_read: self
                    .general_metrics
                    .raptor_files_read
                    .load(Ordering::Relaxed),
                files_written: self
                    .general_metrics
                    .raptor_files_written
                    .load(Ordering::Relaxed),
                bytes_read: self
                    .general_metrics
                    .raptor_bytes_read
                    .load(Ordering::Relaxed),
                bytes_written: self
                    .general_metrics
                    .raptor_bytes_written
                    .load(Ordering::Relaxed),
            },
            StorageEngineType::SWIFT => EngineFileStats {
                files_read: self
                    .general_metrics
                    .swift_files_read
                    .load(Ordering::Relaxed),
                files_written: self
                    .general_metrics
                    .swift_files_written
                    .load(Ordering::Relaxed),
                bytes_read: self
                    .general_metrics
                    .swift_bytes_read
                    .load(Ordering::Relaxed),
                bytes_written: self
                    .general_metrics
                    .swift_bytes_written
                    .load(Ordering::Relaxed),
            },
            StorageEngineType::HELIX => EngineFileStats {
                files_read: self
                    .general_metrics
                    .helix_files_read
                    .load(Ordering::Relaxed),
                files_written: self
                    .general_metrics
                    .helix_files_written
                    .load(Ordering::Relaxed),
                bytes_read: self
                    .general_metrics
                    .helix_bytes_read
                    .load(Ordering::Relaxed),
                bytes_written: self
                    .general_metrics
                    .helix_bytes_written
                    .load(Ordering::Relaxed),
            },
            // TST (and any future engine) has no dedicated per-engine counters yet.
            _ => EngineFileStats {
                files_read: 0,
                files_written: 0,
                bytes_read: 0,
                bytes_written: 0,
            },
        }
    }

    /// Export all metrics to HashMap for unified framework
    async fn export_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();

        // Memory cache metrics
        let mem_hits = self
            .zerocopy_metrics
            .memory_cache_hits
            .load(Ordering::Relaxed);
        let mem_misses = self
            .zerocopy_metrics
            .memory_cache_misses
            .load(Ordering::Relaxed);
        let mem_total = mem_hits + mem_misses;

        metrics.insert("fs.memory_cache.hits".to_string(), mem_hits as f64);
        metrics.insert("fs.memory_cache.misses".to_string(), mem_misses as f64);
        if mem_total > 0 {
            metrics.insert(
                "fs.memory_cache.hit_rate".to_string(),
                mem_hits as f64 / mem_total as f64,
            );
        }
        metrics.insert(
            "fs.memory_cache.size_bytes".to_string(),
            self.zerocopy_metrics
                .memory_cache_size_bytes
                .load(Ordering::Relaxed) as f64,
        );
        metrics.insert(
            "fs.memory_cache.entries".to_string(),
            self.zerocopy_metrics
                .memory_cache_entries
                .load(Ordering::Relaxed) as f64,
        );
        metrics.insert(
            "fs.memory_cache.evictions".to_string(),
            self.zerocopy_metrics
                .memory_cache_evictions
                .load(Ordering::Relaxed) as f64,
        );

        // Disk cache metrics
        let disk_hits = self
            .zerocopy_metrics
            .disk_cache_hits
            .load(Ordering::Relaxed);
        let disk_misses = self
            .zerocopy_metrics
            .disk_cache_misses
            .load(Ordering::Relaxed);
        let disk_total = disk_hits + disk_misses;

        metrics.insert("fs.disk_cache.hits".to_string(), disk_hits as f64);
        metrics.insert("fs.disk_cache.misses".to_string(), disk_misses as f64);
        if disk_total > 0 {
            metrics.insert(
                "fs.disk_cache.hit_rate".to_string(),
                disk_hits as f64 / disk_total as f64,
            );
        }
        metrics.insert(
            "fs.disk_cache.size_bytes".to_string(),
            self.zerocopy_metrics
                .disk_cache_size_bytes
                .load(Ordering::Relaxed) as f64,
        );
        metrics.insert(
            "fs.disk_cache.entries".to_string(),
            self.zerocopy_metrics
                .disk_cache_entries
                .load(Ordering::Relaxed) as f64,
        );
        metrics.insert(
            "fs.disk_cache.evictions".to_string(),
            self.zerocopy_metrics
                .disk_cache_evictions
                .load(Ordering::Relaxed) as f64,
        );

        // Combined cache metrics
        let total_hits = mem_hits + disk_hits;
        let total_misses = mem_misses + disk_misses;
        let total_accesses = total_hits + total_misses;
        if total_accesses > 0 {
            metrics.insert(
                "fs.cache.overall_hit_rate".to_string(),
                total_hits as f64 / total_accesses as f64,
            );
        }

        // Latency metrics
        let hit_count = self
            .zerocopy_metrics
            .cache_hit_count_for_latency
            .load(Ordering::Relaxed);
        let miss_count = self
            .zerocopy_metrics
            .cache_miss_count_for_latency
            .load(Ordering::Relaxed);

        if hit_count > 0 {
            let avg_hit_latency_ns = self
                .zerocopy_metrics
                .total_cache_hit_latency_ns
                .load(Ordering::Relaxed) as f64
                / hit_count as f64;
            metrics.insert(
                "fs.cache.avg_hit_latency_us".to_string(),
                avg_hit_latency_ns / 1000.0,
            );
        }

        if miss_count > 0 {
            let avg_miss_latency_ns = self
                .zerocopy_metrics
                .total_cache_miss_latency_ns
                .load(Ordering::Relaxed) as f64
                / miss_count as f64;
            metrics.insert(
                "fs.cache.avg_miss_latency_us".to_string(),
                avg_miss_latency_ns / 1000.0,
            );
        }

        // Metadata cache metrics
        let meta_hits = self
            .zerocopy_metrics
            .metadata_cache_hits
            .load(Ordering::Relaxed);
        let meta_misses = self
            .zerocopy_metrics
            .metadata_cache_misses
            .load(Ordering::Relaxed);
        let meta_total = meta_hits + meta_misses;

        metrics.insert("fs.metadata_cache.hits".to_string(), meta_hits as f64);
        metrics.insert("fs.metadata_cache.misses".to_string(), meta_misses as f64);
        if meta_total > 0 {
            metrics.insert(
                "fs.metadata_cache.hit_rate".to_string(),
                meta_hits as f64 / meta_total as f64,
            );
        }
        metrics.insert(
            "fs.metadata_cache.files_skipped".to_string(),
            self.zerocopy_metrics.files_skipped.load(Ordering::Relaxed) as f64,
        );
        metrics.insert(
            "fs.metadata_cache.bytes_saved".to_string(),
            self.zerocopy_metrics
                .bytes_saved_by_skipping
                .load(Ordering::Relaxed) as f64,
        );

        // Download optimization metrics
        let selective = self
            .zerocopy_metrics
            .selective_downloads
            .load(Ordering::Relaxed);
        let full = self.zerocopy_metrics.full_downloads.load(Ordering::Relaxed);
        let total_downloads = selective + full;

        metrics.insert("fs.downloads.selective".to_string(), selective as f64);
        metrics.insert("fs.downloads.full".to_string(), full as f64);
        if total_downloads > 0 {
            metrics.insert(
                "fs.downloads.selective_ratio".to_string(),
                selective as f64 / total_downloads as f64,
            );
        }
        metrics.insert(
            "fs.downloads.bytes_saved".to_string(),
            self.zerocopy_metrics
                .total_bytes_saved
                .load(Ordering::Relaxed) as f64,
        );

        // Zero-copy operation metrics
        metrics.insert(
            "fs.zerocopy.operations".to_string(),
            self.zerocopy_metrics
                .zerocopy_operations
                .load(Ordering::Relaxed) as f64,
        );
        metrics.insert(
            "fs.zerocopy.bytes_transferred".to_string(),
            self.zerocopy_metrics
                .zerocopy_bytes_transferred
                .load(Ordering::Relaxed) as f64,
        );
        metrics.insert(
            "fs.zerocopy.sendfile_ops".to_string(),
            self.zerocopy_metrics
                .sendfile_operations
                .load(Ordering::Relaxed) as f64,
        );
        metrics.insert(
            "fs.zerocopy.mmap_ops".to_string(),
            self.zerocopy_metrics
                .mmap_operations
                .load(Ordering::Relaxed) as f64,
        );

        // General I/O metrics
        metrics.insert(
            "fs.io.read_ops".to_string(),
            self.general_metrics.read_operations.load(Ordering::Relaxed) as f64,
        );
        metrics.insert(
            "fs.io.write_ops".to_string(),
            self.general_metrics
                .write_operations
                .load(Ordering::Relaxed) as f64,
        );
        metrics.insert(
            "fs.io.bytes_read".to_string(),
            self.general_metrics.bytes_read.load(Ordering::Relaxed) as f64,
        );
        metrics.insert(
            "fs.io.bytes_written".to_string(),
            self.general_metrics.bytes_written.load(Ordering::Relaxed) as f64,
        );

        // Storage engine file type metrics
        metrics.insert(
            "fs.engine.sst.files_read".to_string(),
            self.general_metrics.sst_files_read.load(Ordering::Relaxed) as f64,
        );
        metrics.insert(
            "fs.engine.sst.files_written".to_string(),
            self.general_metrics
                .sst_files_written
                .load(Ordering::Relaxed) as f64,
        );
        metrics.insert(
            "fs.engine.sst.bytes_read".to_string(),
            self.general_metrics.sst_bytes_read.load(Ordering::Relaxed) as f64,
        );
        metrics.insert(
            "fs.engine.sst.bytes_written".to_string(),
            self.general_metrics
                .sst_bytes_written
                .load(Ordering::Relaxed) as f64,
        );

        metrics.insert(
            "fs.engine.nova.files_read".to_string(),
            self.general_metrics.nova_files_read.load(Ordering::Relaxed) as f64,
        );
        metrics.insert(
            "fs.engine.nova.files_written".to_string(),
            self.general_metrics
                .nova_files_written
                .load(Ordering::Relaxed) as f64,
        );
        metrics.insert(
            "fs.engine.nova.bytes_read".to_string(),
            self.general_metrics.nova_bytes_read.load(Ordering::Relaxed) as f64,
        );
        metrics.insert(
            "fs.engine.nova.bytes_written".to_string(),
            self.general_metrics
                .nova_bytes_written
                .load(Ordering::Relaxed) as f64,
        );

        metrics.insert(
            "fs.engine.raptor.files_read".to_string(),
            self.general_metrics
                .raptor_files_read
                .load(Ordering::Relaxed) as f64,
        );
        metrics.insert(
            "fs.engine.raptor.files_written".to_string(),
            self.general_metrics
                .raptor_files_written
                .load(Ordering::Relaxed) as f64,
        );
        metrics.insert(
            "fs.engine.raptor.bytes_read".to_string(),
            self.general_metrics
                .raptor_bytes_read
                .load(Ordering::Relaxed) as f64,
        );
        metrics.insert(
            "fs.engine.raptor.bytes_written".to_string(),
            self.general_metrics
                .raptor_bytes_written
                .load(Ordering::Relaxed) as f64,
        );

        metrics.insert(
            "fs.engine.swift.files_read".to_string(),
            self.general_metrics
                .swift_files_read
                .load(Ordering::Relaxed) as f64,
        );
        metrics.insert(
            "fs.engine.swift.files_written".to_string(),
            self.general_metrics
                .swift_files_written
                .load(Ordering::Relaxed) as f64,
        );
        metrics.insert(
            "fs.engine.swift.bytes_read".to_string(),
            self.general_metrics
                .swift_bytes_read
                .load(Ordering::Relaxed) as f64,
        );
        metrics.insert(
            "fs.engine.swift.bytes_written".to_string(),
            self.general_metrics
                .swift_bytes_written
                .load(Ordering::Relaxed) as f64,
        );

        metrics.insert(
            "fs.engine.prism.files_read".to_string(),
            self.general_metrics
                .helix_files_read
                .load(Ordering::Relaxed) as f64,
        );
        metrics.insert(
            "fs.engine.prism.files_written".to_string(),
            self.general_metrics
                .helix_files_written
                .load(Ordering::Relaxed) as f64,
        );
        metrics.insert(
            "fs.engine.prism.bytes_read".to_string(),
            self.general_metrics
                .helix_bytes_read
                .load(Ordering::Relaxed) as f64,
        );
        metrics.insert(
            "fs.engine.prism.bytes_written".to_string(),
            self.general_metrics
                .helix_bytes_written
                .load(Ordering::Relaxed) as f64,
        );

        // Error metrics
        let total_errors = self.general_metrics.read_errors.load(Ordering::Relaxed)
            + self.general_metrics.write_errors.load(Ordering::Relaxed)
            + self
                .general_metrics
                .permission_errors
                .load(Ordering::Relaxed)
            + self
                .general_metrics
                .not_found_errors
                .load(Ordering::Relaxed);

        metrics.insert("fs.errors.total".to_string(), total_errors as f64);
        metrics.insert(
            "fs.errors.read".to_string(),
            self.general_metrics.read_errors.load(Ordering::Relaxed) as f64,
        );
        metrics.insert(
            "fs.errors.write".to_string(),
            self.general_metrics.write_errors.load(Ordering::Relaxed) as f64,
        );

        // Cloud storage metrics
        metrics.insert(
            "fs.cloud.s3_ops".to_string(),
            self.general_metrics.s3_operations.load(Ordering::Relaxed) as f64,
        );
        metrics.insert(
            "fs.cloud.gcs_ops".to_string(),
            self.general_metrics.gcs_operations.load(Ordering::Relaxed) as f64,
        );
        metrics.insert(
            "fs.cloud.azure_ops".to_string(),
            self.general_metrics
                .azure_operations
                .load(Ordering::Relaxed) as f64,
        );
        metrics.insert(
            "fs.cloud.local_ops".to_string(),
            self.general_metrics
                .local_operations
                .load(Ordering::Relaxed) as f64,
        );

        metrics
    }
}

impl Default for FilesystemMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl ZeroCopyMetrics {
    pub fn new() -> Self {
        Self {
            memory_cache_hits: AtomicU64::new(0),
            memory_cache_misses: AtomicU64::new(0),
            memory_cache_size_bytes: AtomicU64::new(0),
            memory_cache_entries: AtomicU64::new(0),
            memory_cache_evictions: AtomicU64::new(0),
            disk_cache_hits: AtomicU64::new(0),
            disk_cache_misses: AtomicU64::new(0),
            disk_cache_size_bytes: AtomicU64::new(0),
            disk_cache_entries: AtomicU64::new(0),
            disk_cache_evictions: AtomicU64::new(0),
            total_cache_hit_latency_ns: AtomicU64::new(0),
            total_cache_miss_latency_ns: AtomicU64::new(0),
            cache_hit_count_for_latency: AtomicU64::new(0),
            cache_miss_count_for_latency: AtomicU64::new(0),
            metadata_cache_hits: AtomicU64::new(0),
            metadata_cache_misses: AtomicU64::new(0),
            files_skipped: AtomicU64::new(0),
            bytes_saved_by_skipping: AtomicU64::new(0),
            selective_downloads: AtomicU64::new(0),
            full_downloads: AtomicU64::new(0),
            total_bytes_downloaded: AtomicU64::new(0),
            total_bytes_saved: AtomicU64::new(0),
            zerocopy_operations: AtomicU64::new(0),
            zerocopy_bytes_transferred: AtomicU64::new(0),
            sendfile_operations: AtomicU64::new(0),
            mmap_operations: AtomicU64::new(0),
        }
    }

    /// Record cache hit with timing
    pub fn record_cache_hit(&self, latency_ns: u64) {
        self.memory_cache_hits.fetch_add(1, Ordering::Relaxed);
        self.total_cache_hit_latency_ns
            .fetch_add(latency_ns, Ordering::Relaxed);
        self.cache_hit_count_for_latency
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record cache miss with timing
    pub fn record_cache_miss(&self, latency_ns: u64) {
        self.memory_cache_misses.fetch_add(1, Ordering::Relaxed);
        self.total_cache_miss_latency_ns
            .fetch_add(latency_ns, Ordering::Relaxed);
        self.cache_miss_count_for_latency
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Update cache size metrics
    pub fn update_cache_metrics(
        &self,
        memory_size: u64,
        memory_entries: u64,
        disk_size: u64,
        disk_entries: u64,
    ) {
        self.memory_cache_size_bytes
            .store(memory_size, Ordering::Relaxed);
        self.memory_cache_entries
            .store(memory_entries, Ordering::Relaxed);
        self.disk_cache_size_bytes
            .store(disk_size, Ordering::Relaxed);
        self.disk_cache_entries
            .store(disk_entries, Ordering::Relaxed);
    }
}

impl Default for ZeroCopyMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl GeneralFilesystemMetrics {
    pub fn new() -> Self {
        Self {
            read_operations: AtomicU64::new(0),
            write_operations: AtomicU64::new(0),
            delete_operations: AtomicU64::new(0),
            list_operations: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
            bytes_written: AtomicU64::new(0),
            read_errors: AtomicU64::new(0),
            write_errors: AtomicU64::new(0),
            permission_errors: AtomicU64::new(0),
            not_found_errors: AtomicU64::new(0),
            s3_operations: AtomicU64::new(0),
            gcs_operations: AtomicU64::new(0),
            azure_operations: AtomicU64::new(0),
            local_operations: AtomicU64::new(0),
            parallel_downloads: AtomicU64::new(0),
            chunked_uploads: AtomicU64::new(0),
            multipart_uploads: AtomicU64::new(0),
            // Storage engine file type metrics
            sst_files_read: AtomicU64::new(0),
            sst_files_written: AtomicU64::new(0),
            sst_bytes_read: AtomicU64::new(0),
            sst_bytes_written: AtomicU64::new(0),
            nova_files_read: AtomicU64::new(0),
            nova_files_written: AtomicU64::new(0),
            nova_bytes_read: AtomicU64::new(0),
            nova_bytes_written: AtomicU64::new(0),
            raptor_files_read: AtomicU64::new(0),
            raptor_files_written: AtomicU64::new(0),
            raptor_bytes_read: AtomicU64::new(0),
            raptor_bytes_written: AtomicU64::new(0),
            swift_files_read: AtomicU64::new(0),
            swift_files_written: AtomicU64::new(0),
            swift_bytes_read: AtomicU64::new(0),
            swift_bytes_written: AtomicU64::new(0),
            helix_files_read: AtomicU64::new(0),
            helix_files_written: AtomicU64::new(0),
            helix_bytes_read: AtomicU64::new(0),
            helix_bytes_written: AtomicU64::new(0),
        }
    }
}

impl Default for GeneralFilesystemMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl MetricsCollector for FilesystemMetricsCollector {
    async fn collect(&self) -> Result<MetricsSample> {
        let values = self.export_metrics().await;

        Ok(MetricsSample {
            timestamp: Instant::now(),
            collector: "filesystem".to_string(),
            values,
        })
    }

    fn name(&self) -> &'static str {
        "FilesystemMetrics"
    }

    fn recommended_interval(&self) -> Duration {
        Duration::from_secs(10) // Collect every 10 seconds
    }
}
