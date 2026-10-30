//! Hybrid Parquet Writer Strategy
//!
//! Dynamically chooses between streaming and batch modes based on workload characteristics.
//! Provides optimal performance across different insertion patterns.
//!
//! Features:
//! - Automatic mode selection based on data characteristics
//! - Adaptive buffering for optimal memory usage
//! - Smart row group sizing based on insertion rate
//! - Concurrent write support with lock-free coordination
//! - Metrics and monitoring for mode transitions

use anyhow::Result;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info, trace, warn};

use crate::storage::engines::core::formats::columnar::parquet_write_engine::{
    batch_writer::BatchParquetWriter, streaming_writer::StreamingParquetWriter,
    writer_config::ParquetWriterConfig, writer_statistics::StreamingParquetWriterStats,
};
use proximadb_records::ProximaRecord;

/// Hybrid writer mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WriterMode {
    /// Streaming mode for continuous inserts
    Streaming,
    /// Batch mode for bulk inserts
    Batch,
    /// Adaptive mode that switches dynamically
    Adaptive,
}

/// Insertion pattern detection
#[derive(Debug, Clone)]
pub struct InsertionPattern {
    /// Average records per second
    pub insert_rate: f64,

    /// Average batch size
    pub avg_batch_size: f64,

    /// Batch size variance
    pub batch_size_variance: f64,

    /// Time between batches
    pub avg_inter_batch_time: Duration,

    /// Pattern consistency score (0.0 - 1.0)
    pub consistency_score: f64,

    /// Detected pattern type
    pub pattern_type: PatternType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PatternType {
    /// Steady stream of small inserts
    Streaming,
    /// Large bulk inserts
    Bulk,
    /// Mixed pattern
    Mixed,
    /// Unknown or insufficient data
    Unknown,
}

/// Hybrid writer configuration
#[derive(Debug, Clone)]
pub struct HybridWriterConfig {
    /// Base Parquet writer config
    pub base_config: ParquetWriterConfig,

    /// Initial mode
    pub initial_mode: WriterMode,

    /// Enable automatic mode switching
    pub enable_auto_switch: bool,

    /// Minimum records before switching modes
    pub mode_switch_threshold: usize,

    /// Pattern detection window size
    pub pattern_window_size: usize,

    /// Streaming mode threshold (records/second)
    pub streaming_threshold: f64,

    /// Batch mode threshold (average batch size)
    pub batch_threshold: usize,

    /// Maximum buffer size before forced flush
    pub max_buffer_size: usize,

    /// Buffer time limit before forced flush
    pub buffer_time_limit: Duration,

    /// Enable concurrent writes
    pub enable_concurrent_writes: bool,

    /// Maximum concurrent writers
    pub max_concurrent_writers: usize,

    /// Row group size optimization
    pub optimize_row_group_size: bool,

    /// Minimum row group size
    pub min_row_group_size: usize,

    /// Maximum row group size
    pub max_row_group_size: usize,
}

impl Default for HybridWriterConfig {
    fn default() -> Self {
        // All optimizations ENABLED by default for adaptive performance
        // Users can override any setting if needed
        Self {
            base_config: ParquetWriterConfig::default(), // Inherits all optimizations
            initial_mode: WriterMode::Adaptive,          // DEFAULT: Auto-adapt to workload
            enable_auto_switch: true,                    // DEFAULT ON: Dynamic mode switching
            mode_switch_threshold: 1000,
            pattern_window_size: 100,
            streaming_threshold: 100.0, // 100 records/second
            batch_threshold: 1000,      // 1000 records per batch
            max_buffer_size: 100_000,
            buffer_time_limit: Duration::from_secs(30),
            enable_concurrent_writes: true, // DEFAULT ON: Concurrent write support
            max_concurrent_writers: 4,
            optimize_row_group_size: true, // DEFAULT ON: Smart row group sizing
            min_row_group_size: 1_000,
            max_row_group_size: 100_000,
        }
    }
}

/// Root-side quantization encoder for the streaming writer's port seam
/// (TD-DECOMP-78): the level-specific `QuantizationEnginePort` implemented by
/// the modality kernel's `UnifiedQuantizationEngine`.
pub(crate) fn root_quantization_engine()
-> Option<std::sync::Arc<dyn proximadb_storage_ports::QuantizationEnginePort>> {
    use crate::compute::quantization::quantization_engine::{
        InMemoryCodebookStore, UnifiedQuantizationEngine,
    };
    Some(std::sync::Arc::new(UnifiedQuantizationEngine::new(
        std::sync::Arc::new(proximadb_distance_kernel::engine::UnifiedDistanceCompute::default()),
        std::sync::Arc::new(InMemoryCodebookStore::new()),
    )))
}

/// Root-side batch/level quantization engine for the columnar serializer's port
/// seam (TD-DECOMP-80): the `StorageQuantizationEnginePort` implemented by the
/// modality kernel's `StorageQuantizationEngine`.
pub(crate) fn root_storage_quantization_engine()
-> Option<std::sync::Arc<dyn proximadb_storage_ports::StorageQuantizationEnginePort>> {
    use crate::compute::quantization::quantization_engine::{
        InMemoryCodebookStore, UnifiedQuantizationEngine,
    };
    use crate::compute::quantization::storage_engine::StorageQuantizationEngine;

    let distance_compute =
        std::sync::Arc::new(proximadb_distance_kernel::engine::UnifiedDistanceCompute::default());
    let unified_engine = std::sync::Arc::new(UnifiedQuantizationEngine::new(
        distance_compute.clone(),
        std::sync::Arc::new(InMemoryCodebookStore::new()),
    ));
    Some(std::sync::Arc::new(StorageQuantizationEngine::new(
        unified_engine,
        distance_compute,
        crate::compute::quantization::storage_engine::StorageQuantizationConfig::default(),
    )))
}

pub struct HybridParquetWriter {
    /// Configuration
    config: HybridWriterConfig,

    /// Current mode
    current_mode: Arc<RwLock<WriterMode>>,

    /// Streaming writer (when in streaming mode)
    streaming_writer: Arc<Mutex<Option<StreamingParquetWriter>>>,

    /// Record buffer for batch mode
    buffer: Arc<RwLock<Vec<ProximaRecord>>>,

    /// Insertion history for pattern detection
    insertion_history: Arc<RwLock<VecDeque<InsertionEvent>>>,

    /// Statistics
    stats: Arc<HybridWriterStats>,

    /// File path
    file_path: PathBuf,

    /// Vector dimension
    dimension: usize,

    /// Metadata collector for hierarchical metadata (NOVA)
    metadata_collector: Arc<Mutex<Option<Box<dyn super::metadata_collector::MetadataCollector>>>>,

    /// Filterable columns for metadata extraction (VIPER)
    filterable_columns: Option<Vec<crate::proto::proximadb_v1::FilterableColumnSpec>>,

    /// Last flush time
    last_flush_time: Arc<RwLock<Instant>>,

    /// Background flush task handle
    flush_task: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,

    /// Filesystem factory for I/O operations
    filesystem_factory: Arc<crate::storage::persistence::filesystem::FilesystemFactory>,
}

/// Insertion event for pattern tracking
#[derive(Debug, Clone)]
struct InsertionEvent {
    timestamp: Instant,
    batch_size: usize,
    #[allow(dead_code)]
    mode: WriterMode,
}

/// Hybrid writer statistics
#[derive(Debug)]
pub struct HybridWriterStats {
    total_records: AtomicU64,
    streaming_writes: AtomicU64,
    batch_writes: AtomicU64,
    mode_switches: AtomicUsize,
    buffer_flushes: AtomicUsize,
    forced_flushes: AtomicUsize,
    avg_batch_size: AtomicU64,
    avg_flush_latency_ms: AtomicU64,
}

impl HybridParquetWriter {
    /// Create new hybrid writer
    pub async fn new<P: AsRef<Path>>(
        file_path: P,
        dimension: usize,
        config: HybridWriterConfig,
    ) -> Result<Self> {
        let file_path = file_path.as_ref().to_path_buf();
        trace!(
            "HybridParquetWriter::new - starting, file_path: {:?}, mode: {:?}",
            file_path, config.initial_mode
        );
        info!(
            "Creating hybrid Parquet writer: {:?}, mode: {:?}",
            file_path, config.initial_mode
        );

        // Create a default filesystem factory (for local files only)
        let filesystem_factory = Arc::new(
            crate::storage::persistence::filesystem::FilesystemFactory::create_default().await?,
        );

        let streaming_writer = if config.initial_mode == WriterMode::Streaming {
            trace!("HybridParquetWriter::new - creating StreamingParquetWriter");
            // For local files, use the simple constructor that doesn't need a filesystem factory
            let writer = StreamingParquetWriter::new(
                &file_path,
                dimension,
                config.base_config.clone(),
                None, // Filterable columns will be set via setter method
                filesystem_factory.clone(),
                root_quantization_engine(),
            )
            .await?;
            trace!("HybridParquetWriter::new - StreamingParquetWriter created");
            Some(writer)
        } else {
            trace!("HybridParquetWriter::new - not creating StreamingParquetWriter (batch mode)");
            None
        };

        let stats = Arc::new(HybridWriterStats {
            total_records: AtomicU64::new(0),
            streaming_writes: AtomicU64::new(0),
            batch_writes: AtomicU64::new(0),
            mode_switches: AtomicUsize::new(0),
            buffer_flushes: AtomicUsize::new(0),
            forced_flushes: AtomicUsize::new(0),
            avg_batch_size: AtomicU64::new(0),
            avg_flush_latency_ms: AtomicU64::new(0),
        });

        let writer = Self {
            config: config.clone(),
            current_mode: Arc::new(RwLock::new(config.initial_mode)),
            streaming_writer: Arc::new(Mutex::new(streaming_writer)),
            buffer: Arc::new(RwLock::new(Vec::new())),
            insertion_history: Arc::new(RwLock::new(VecDeque::new())),
            stats,
            file_path,
            dimension,
            metadata_collector: Arc::new(Mutex::new(None)),
            filterable_columns: None,
            last_flush_time: Arc::new(RwLock::new(Instant::now())),
            flush_task: Arc::new(Mutex::new(None)),
            filesystem_factory,
        };

        // Start background flush task
        // Note: Background task will be started when needed

        Ok(writer)
    }

    /// Create new hybrid writer with specific filesystem factory
    pub fn with_filesystem_factory<P: AsRef<Path>>(
        file_path: P,
        dimension: usize,
        config: HybridWriterConfig,
        filesystem_factory: Arc<crate::storage::persistence::filesystem::FilesystemFactory>,
    ) -> Result<Self> {
        let file_path = file_path.as_ref().to_path_buf();
        info!(
            "Creating hybrid Parquet writer with custom filesystem: {:?}, mode: {:?}",
            file_path, config.initial_mode
        );

        let streaming_writer = if config.initial_mode == WriterMode::Streaming {
            Some(StreamingParquetWriter::with_filesystem_factory(
                &file_path,
                dimension,
                config.base_config.clone(),
                None, // Filterable columns will be set via setter method
                filesystem_factory.clone(),
                root_quantization_engine(),
            )?)
        } else {
            None
        };

        let stats = Arc::new(HybridWriterStats {
            total_records: AtomicU64::new(0),
            streaming_writes: AtomicU64::new(0),
            batch_writes: AtomicU64::new(0),
            mode_switches: AtomicUsize::new(0),
            buffer_flushes: AtomicUsize::new(0),
            forced_flushes: AtomicUsize::new(0),
            avg_batch_size: AtomicU64::new(0),
            avg_flush_latency_ms: AtomicU64::new(0),
        });

        let writer = Self {
            config: config.clone(),
            current_mode: Arc::new(RwLock::new(config.initial_mode)),
            streaming_writer: Arc::new(Mutex::new(streaming_writer)),
            buffer: Arc::new(RwLock::new(Vec::new())),
            insertion_history: Arc::new(RwLock::new(VecDeque::new())),
            stats,
            file_path,
            dimension,
            metadata_collector: Arc::new(Mutex::new(None)),
            filterable_columns: None,
            last_flush_time: Arc::new(RwLock::new(Instant::now())),
            flush_task: Arc::new(Mutex::new(None)),
            filesystem_factory,
        };

        Ok(writer)
    }

    /// Set metadata collector for hierarchical metadata (NOVA)
    pub async fn set_metadata_collector(
        &self,
        collector: Box<dyn super::metadata_collector::MetadataCollector>,
    ) {
        let mut metadata_lock = self.metadata_collector.lock().await;
        *metadata_lock = Some(collector);
    }

    /// Set filterable columns for metadata extraction (VIPER)
    pub fn set_filterable_columns(
        &mut self,
        columns: Vec<crate::proto::proximadb_v1::FilterableColumnSpec>,
    ) {
        self.filterable_columns = Some(columns);
    }

    /// Write records with adaptive strategy
    pub async fn write(&self, records: &[ProximaRecord]) -> Result<()> {
        let batch_size = records.len();
        let timestamp = Instant::now();

        // Record insertion event
        self.record_insertion_event(timestamp, batch_size).await;

        // Detect pattern and potentially switch modes
        if self.config.enable_auto_switch {
            self.detect_and_switch_mode().await?;
        }

        // Route to appropriate writer
        let mode = *self.current_mode.read().await;
        match mode {
            WriterMode::Streaming => {
                self.write_streaming(records).await?;
            }
            WriterMode::Batch => {
                self.write_batch(records).await?;
            }
            WriterMode::Adaptive => {
                // Decide based on current batch size
                if batch_size >= self.config.batch_threshold {
                    self.write_batch(records).await?;
                } else {
                    self.write_streaming(records).await?;
                }
            }
        }

        // Update statistics
        self.stats
            .total_records
            .fetch_add(batch_size as u64, Ordering::Relaxed);

        Ok(())
    }

    /// Write in streaming mode
    async fn write_streaming(&self, records: &[ProximaRecord]) -> Result<()> {
        trace!("Writing {} records in streaming mode", records.len());

        let mut writer_lock = self.streaming_writer.lock().await;

        // Create streaming writer if needed
        if writer_lock.is_none() {
            let mut new_writer = StreamingParquetWriter::with_filesystem_factory(
                &self.file_path,
                self.dimension,
                self.config.base_config.clone(),
                self.filterable_columns.as_deref(),
                self.filesystem_factory.clone(),
                root_quantization_engine(),
            )?;

            // Transfer metadata collector to streaming writer if present
            let mut metadata_lock = self.metadata_collector.lock().await;
            if let Some(collector) = metadata_lock.take() {
                new_writer = new_writer.with_metadata_collector(collector);
            }

            *writer_lock = Some(new_writer);
        }

        // Write records
        if let Some(writer) = writer_lock.as_mut() {
            writer.write_batch(records).await?;
        }

        self.stats
            .streaming_writes
            .fetch_add(records.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Write in batch mode (public for compatibility)
    pub async fn write_batch(&self, records: &[ProximaRecord]) -> Result<()> {
        let records_len = records.len();
        trace!("Writing {} records in batch mode", records_len);

        // Add to buffer
        {
            let mut buffer = self.buffer.write().await;
            buffer.extend_from_slice(records);

            // Check if buffer needs flushing
            if buffer.len() >= self.config.max_buffer_size {
                drop(buffer); // Release lock before flushing
                self.flush_buffer(false).await?;
            }
        }

        self.stats
            .batch_writes
            .fetch_add(records_len as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Flush buffer to disk
    async fn flush_buffer(&self, force: bool) -> Result<()> {
        let start_time = Instant::now();

        let records = {
            let mut buffer = self.buffer.write().await;
            if buffer.is_empty() {
                return Ok(());
            }

            if !force && buffer.len() < self.config.min_row_group_size {
                // Don't flush small buffers unless forced
                return Ok(());
            }

            std::mem::take(&mut *buffer)
        };

        if records.is_empty() {
            return Ok(());
        }

        info!("Flushing {} records to disk", records.len());

        // Use batch writer for flush (ports threaded from the hybrid writer's own
        // injected factory/encoder — TD-DECOMP-78 composition-root pattern).
        let mut writer = BatchParquetWriter::new(
            &self.file_path,
            self.dimension,
            self.config.base_config.clone(),
        )
        .with_ports(self.filesystem_factory.clone(), root_quantization_engine());

        writer.write_all(&records).await?;

        // Update statistics
        self.stats.buffer_flushes.fetch_add(1, Ordering::Relaxed);
        if force {
            self.stats.forced_flushes.fetch_add(1, Ordering::Relaxed);
        }

        let flush_latency = start_time.elapsed().as_millis() as u64;
        self.update_avg_flush_latency(flush_latency);

        // Update last flush time
        *self.last_flush_time.write().await = Instant::now();

        debug!("Buffer flush completed in {}ms", flush_latency);
        Ok(())
    }

    /// Record insertion event for pattern detection
    async fn record_insertion_event(&self, timestamp: Instant, batch_size: usize) {
        let mode = *self.current_mode.read().await;
        let event = InsertionEvent {
            timestamp,
            batch_size,
            mode,
        };

        let mut history = self.insertion_history.write().await;
        history.push_back(event);

        // Keep only recent history
        while history.len() > self.config.pattern_window_size {
            history.pop_front();
        }
    }

    /// Detect pattern and switch mode if needed
    async fn detect_and_switch_mode(&self) -> Result<()> {
        let pattern = self.detect_pattern().await;

        if pattern.pattern_type == PatternType::Unknown {
            return Ok(()); // Not enough data
        }

        let current_mode = *self.current_mode.read().await;
        let recommended_mode = self.recommend_mode(&pattern);

        if current_mode != recommended_mode {
            info!(
                "Switching mode from {:?} to {:?} based on pattern: {:?}",
                current_mode, recommended_mode, pattern.pattern_type
            );

            self.switch_mode(recommended_mode).await?;
            self.stats.mode_switches.fetch_add(1, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Detect insertion pattern
    async fn detect_pattern(&self) -> InsertionPattern {
        let history = self.insertion_history.read().await;

        if history.len() < 10 {
            return InsertionPattern {
                insert_rate: 0.0,
                avg_batch_size: 0.0,
                batch_size_variance: 0.0,
                avg_inter_batch_time: Duration::from_secs(0),
                consistency_score: 0.0,
                pattern_type: PatternType::Unknown,
            };
        }

        // Calculate statistics
        let mut batch_sizes = Vec::new();
        let mut inter_batch_times = Vec::new();
        let mut prev_timestamp = None;

        for event in history.iter() {
            batch_sizes.push(event.batch_size as f64);

            if let Some(prev) = prev_timestamp {
                inter_batch_times.push(event.timestamp.duration_since(prev));
            }
            prev_timestamp = Some(event.timestamp);
        }

        let avg_batch_size = batch_sizes.iter().sum::<f64>() / batch_sizes.len() as f64;
        let batch_size_variance = Self::calculate_variance(&batch_sizes, avg_batch_size);

        let avg_inter_batch_time = if !inter_batch_times.is_empty() {
            let total_nanos: u128 = inter_batch_times.iter().map(|d| d.as_nanos()).sum();
            Duration::from_nanos((total_nanos / inter_batch_times.len() as u128) as u64)
        } else {
            Duration::from_secs(1)
        };

        let insert_rate = if avg_inter_batch_time.as_secs_f64() > 0.0 {
            avg_batch_size / avg_inter_batch_time.as_secs_f64()
        } else {
            0.0
        };

        // Calculate consistency score
        let consistency_score = if batch_size_variance > 0.0 {
            1.0 / (1.0 + batch_size_variance.sqrt() / avg_batch_size)
        } else {
            1.0
        };

        // Determine pattern type
        let pattern_type = if avg_batch_size >= self.config.batch_threshold as f64 {
            PatternType::Bulk
        } else if insert_rate >= self.config.streaming_threshold {
            PatternType::Streaming
        } else if consistency_score < 0.5 {
            PatternType::Mixed
        } else {
            PatternType::Unknown
        };

        InsertionPattern {
            insert_rate,
            avg_batch_size,
            batch_size_variance,
            avg_inter_batch_time,
            consistency_score,
            pattern_type,
        }
    }

    /// Calculate variance
    fn calculate_variance(values: &[f64], mean: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let sum_squared_diff: f64 = values.iter().map(|v| (v - mean).powi(2)).sum();

        sum_squared_diff / values.len() as f64
    }

    /// Recommend mode based on pattern
    fn recommend_mode(&self, pattern: &InsertionPattern) -> WriterMode {
        match pattern.pattern_type {
            PatternType::Streaming => WriterMode::Streaming,
            PatternType::Bulk => WriterMode::Batch,
            PatternType::Mixed => WriterMode::Adaptive,
            PatternType::Unknown => self.config.initial_mode,
        }
    }

    /// Switch writer mode
    async fn switch_mode(&self, new_mode: WriterMode) -> Result<()> {
        let current_mode = *self.current_mode.read().await;

        if current_mode == new_mode {
            return Ok(());
        }

        debug!("Switching from {:?} to {:?} mode", current_mode, new_mode);

        // Flush any pending data before switching
        if current_mode == WriterMode::Batch {
            self.flush_buffer(true).await?;
        } else if current_mode == WriterMode::Streaming {
            // Finalize streaming writer
            let mut writer_lock = self.streaming_writer.lock().await;
            if let Some(writer) = writer_lock.take() {
                let (_, data, _) = writer.finalize().await?;
                // Write the data to the file using tokio::fs for local files
                // Since temp files are always local, we can write directly
                tokio::fs::write(&self.file_path, &data).await?;
            }
        }

        // Update mode
        *self.current_mode.write().await = new_mode;

        // Initialize new writer if needed
        if new_mode == WriterMode::Streaming {
            let mut writer_lock = self.streaming_writer.lock().await;
            *writer_lock = Some(StreamingParquetWriter::with_filesystem_factory(
                &self.file_path,
                self.dimension,
                self.config.base_config.clone(),
                self.filterable_columns.as_deref(), // Pass filterable columns!
                self.filesystem_factory.clone(),
                root_quantization_engine(),
            )?);
        }

        Ok(())
    }

    /// Start background flush task
    #[allow(dead_code)]
    async fn start_background_flush_task(&mut self) {
        let buffer = self.buffer.clone();
        let last_flush_time = self.last_flush_time.clone();
        let buffer_time_limit = self.config.buffer_time_limit;
        let flush_fn = {
            let self_clone = self.clone_for_task();
            move || {
                let self_clone = self_clone.clone();
                async move { self_clone.flush_buffer(false).await }
            }
        };

        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(5));

            loop {
                interval.tick().await;

                // Check if buffer needs time-based flush
                let last_flush = *last_flush_time.read().await;
                if Instant::now().duration_since(last_flush) > buffer_time_limit {
                    let buffer_size = buffer.read().await.len();
                    if buffer_size > 0 {
                        debug!(
                            "Time-based buffer flush triggered ({} records)",
                            buffer_size
                        );
                        if let Err(e) = flush_fn().await {
                            warn!("Background flush failed: {}", e);
                        }
                    }
                }
            }
        });

        *self.flush_task.lock().await = Some(handle);
    }

    /// Update average flush latency
    fn update_avg_flush_latency(&self, new_latency: u64) {
        let current = self.stats.avg_flush_latency_ms.load(Ordering::Relaxed);
        let flushes = self.stats.buffer_flushes.load(Ordering::Relaxed);

        if flushes == 0 {
            self.stats
                .avg_flush_latency_ms
                .store(new_latency, Ordering::Relaxed);
        } else {
            let new_avg = (current * (flushes as u64 - 1) + new_latency) / flushes as u64;
            self.stats
                .avg_flush_latency_ms
                .store(new_avg, Ordering::Relaxed);
        }
    }

    /// Clone for background tasks
    #[allow(dead_code)]
    #[allow(clippy::unimplemented)] // Deferred: Implement Arc wrapping for background task cloning
    fn clone_for_task(&self) -> Arc<Self> {
        // This would need proper Arc wrapping in production
        // For now, return a placeholder
        unimplemented!("Clone for task not implemented")
    }

    /// Finalize writer and flush all pending data
    pub async fn finalize(
        self,
    ) -> Result<(
        StreamingParquetWriterStats,
        Option<Box<dyn super::metadata_collector::MetadataCollector>>,
    )> {
        info!("Finalizing hybrid Parquet writer");

        // Stop background task
        if let Some(handle) = self.flush_task.lock().await.take() {
            handle.abort();
        }

        // Flush any remaining buffer
        self.flush_buffer(true).await?;

        // Finalize streaming writer if active and get its stats
        let mut writer_lock = self.streaming_writer.lock().await;
        let result = if let Some(writer) = writer_lock.take() {
            let (stats, data, collector) = writer.finalize().await?;
            // Write the data to the file using tokio::fs for local files
            // Since temp files are always local, we can write directly
            // Ensure parent directory exists
            if let Some(parent) = self.file_path.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            tokio::fs::write(&self.file_path, &data).await?;
            (stats, collector)
        } else {
            // Create default stats if no streaming writer was used
            let stats = StreamingParquetWriterStats {
                // File information
                file_path: String::new(), // No file written in this case
                file_size: 0,
                total_row_groups: 0,
                // Record statistics
                total_records: self.stats.total_records.load(Ordering::Relaxed) as usize,
                unique_ids: 0,
                duplicate_ids: 0,
                uncompressed_size: 0,
                compressed_size: 0,
                vector_data_size: 0,
                metadata_size: 0,
                compression_ratio: 1.0,
                vector_compression_ratio: 0.0,
                metadata_compression_ratio: 0.0,
                row_groups_written: 0,
                avg_row_group_size: 0,
                min_row_group_size: 0,
                max_row_group_size: 0,
                bloom_filter_count: 0,
                bloom_filter_total_size: 0,
                write_duration: Duration::default(),
                compression_duration: Duration::default(),
                index_build_duration: Duration::default(),
                throughput_records_per_sec: 0.0,
                throughput_mb_per_sec: 0.0,
                quantization_enabled: false,
                quantization_levels: Vec::new(),
                quantization_space_saved: 0,
                filterable_columns_count: 0,
                records_with_metadata: 0,
                avg_metadata_fields: 0.0,
                vector_norm_min: None,
                vector_norm_max: None,
                vector_norm_mean: None,
                vector_component_min: None,
                vector_component_max: None,
            };
            // Get metadata collector if not passed to streaming writer
            let mut metadata_lock = self.metadata_collector.lock().await;
            (stats, metadata_lock.take())
        };

        info!(
            "Hybrid writer finalized with {} total records",
            result.0.total_records
        );
        Ok(result)
    }

    /// Get current statistics
    pub async fn get_statistics(&self) -> HybridWriterStatistics {
        HybridWriterStatistics {
            total_records: self.stats.total_records.load(Ordering::Relaxed),
            streaming_writes: self.stats.streaming_writes.load(Ordering::Relaxed),
            batch_writes: self.stats.batch_writes.load(Ordering::Relaxed),
            mode_switches: self.stats.mode_switches.load(Ordering::Relaxed),
            buffer_flushes: self.stats.buffer_flushes.load(Ordering::Relaxed),
            forced_flushes: self.stats.forced_flushes.load(Ordering::Relaxed),
            avg_batch_size: self.stats.avg_batch_size.load(Ordering::Relaxed),
            avg_flush_latency_ms: self.stats.avg_flush_latency_ms.load(Ordering::Relaxed),
        }
    }

    /// Get current mode
    pub async fn get_current_mode(&self) -> WriterMode {
        *self.current_mode.read().await
    }

    /// Force mode change (for testing)
    pub async fn force_mode(&self, mode: WriterMode) -> Result<()> {
        self.switch_mode(mode).await
    }

    /// Write all records (without finalizing - call finalize() separately)
    /// Returns the path where data was written (could be temp file for atomic writes)
    pub async fn write_all(&mut self, records: &[ProximaRecord]) -> Result<PathBuf> {
        if records.is_empty() {
            return Ok(self.file_path.clone());
        }

        // Write all records
        self.write(records).await?;

        // Force flush any buffered data
        if self.config.initial_mode == WriterMode::Batch
            || *self.current_mode.read().await == WriterMode::Batch
        {
            self.flush_buffer(true).await?;
        }

        // Return the path where data was written
        Ok(self.file_path.clone())
    }

    /// Static method to write to temp, finalize, and upload with disk cache
    /// This is the recommended way to use HybridParquetWriter for flush operations
    pub async fn write_with_cache(
        records: &[ProximaRecord],
        dimension: usize,
        config: HybridWriterConfig,
        final_url: &str, // Final destination (s3://bucket/file or /path/file)
        filesystem_factory: &crate::storage::persistence::filesystem::FilesystemFactory,
        filterable_columns: Option<Vec<crate::proto::proximadb_v1::FilterableColumnSpec>>,
        metadata_collector: Option<Box<dyn super::metadata_collector::MetadataCollector>>,
    ) -> Result<(
        StreamingParquetWriterStats,
        Option<Box<dyn super::metadata_collector::MetadataCollector>>,
    )> {
        if records.is_empty() {
            return Ok((
                StreamingParquetWriterStats {
                    file_path: final_url.to_string(),
                    total_records: 0,
                    total_row_groups: 0,
                    file_size: 0,
                    compression_ratio: 1.0,
                    bloom_filter_count: 0,
                    ..Default::default() // Fill in the rest with defaults
                },
                metadata_collector,
            ));
        }

        // Create temp file for writing
        let _temp_dir = tempfile::tempdir()?;
        let temp_filename = format!("hybrid_temp_{}.parquet", uuid::Uuid::new_v4());
        let temp_path = _temp_dir.path().join(temp_filename);

        trace!("write_with_cache - temp_dir: {:?}", _temp_dir.path());
        trace!("write_with_cache - temp_path: {:?}", temp_path);

        // Create writer with temp path - use default filesystem for temp files
        // Temp files are always local, so we don't need the passed filesystem_factory
        // Force BATCH mode to defer StreamingParquetWriter creation until after filterable_columns are set
        let mut config = config;
        config.initial_mode = WriterMode::Batch; // Don't create StreamingParquetWriter in constructor
        trace!("write_with_cache - creating writer with batch mode (will switch to streaming)");
        let mut writer = HybridParquetWriter::new(&temp_path, dimension, config).await?;
        trace!("write_with_cache - writer created");

        // Set filterable columns BEFORE switching to streaming mode
        if let Some(cols) = filterable_columns {
            writer.set_filterable_columns(cols);
        }

        // Now switch to streaming mode - this will create StreamingParquetWriter with filterable_columns
        writer.force_mode(WriterMode::Streaming).await?;
        trace!("write_with_cache - switched to streaming mode with filterable columns");

        // Set metadata collector if provided
        if let Some(collector) = metadata_collector {
            writer.set_metadata_collector(collector).await;
        }

        // Write all records
        trace!("write_with_cache - writing {} records", records.len());
        writer.write(records).await?;
        trace!("write_with_cache - records written");

        // Finalize to get stats and metadata collector
        trace!("write_with_cache - finalizing writer");
        let (stats, collector) = writer.finalize().await?;
        trace!("write_with_cache - writer finalized, checking if temp file exists");
        trace!(
            "write_with_cache - temp_path exists: {}",
            temp_path.exists()
        );

        // Read temp file data (keep _temp_dir alive until after this)
        trace!("write_with_cache - reading temp file: {:?}", temp_path);
        let data = tokio::fs::read(&temp_path).await?;
        trace!("write_with_cache - temp file read, {} bytes", data.len());

        // Get filesystem for final URL
        let fs = filesystem_factory.get_filesystem(final_url)?;

        // Write to final location with directory creation - UnifiedCachingFilesystem will:
        // 1. Create parent directories if needed
        // 2. Upload to cloud (if s3://, azure://, etc)
        // 3. Populate disk cache automatically
        // 4. Future reads will hit cache, avoiding cloud costs
        let write_options = crate::storage::persistence::filesystem::FileOptions {
            create_dirs: true,
            overwrite: true,
            ..Default::default()
        };
        fs.write(final_url, &data, Some(write_options)).await?;

        debug!(
            "HybridParquetWriter: Wrote {} records to {} with disk cache",
            stats.total_records, final_url
        );

        // Clean up temp directory
        drop(_temp_dir);

        Ok((stats, collector))
    }
}

/// Statistics from hybrid writer
#[derive(Debug, Clone)]
pub struct HybridWriterStatistics {
    pub total_records: u64,
    pub streaming_writes: u64,
    pub batch_writes: u64,
    pub mode_switches: usize,
    pub buffer_flushes: usize,
    pub forced_flushes: usize,
    pub avg_batch_size: u64,
    pub avg_flush_latency_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_records::EmbeddingCell;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_hybrid_writer_streaming_mode() {
        let dir = tempdir().expect("Failed to create temp directory for test");
        let file_path = dir.path().join("test_hybrid_streaming.parquet");

        let config = HybridWriterConfig {
            initial_mode: WriterMode::Streaming,
            enable_auto_switch: false,
            ..Default::default()
        };

        let writer = HybridParquetWriter::new(&file_path, 128, config)
            .await
            .expect("Failed to create hybrid writer in streaming mode");

        // Write small batches (streaming pattern)
        for i in 0..10 {
            let values = vec![0.1_f32; 128];
            let dim = values.len() as u32;
            let mut rec = ProximaRecord {
                oid: format!("vec_{i}"),
                ..Default::default()
            };
            rec.embeddings
                .push(EmbeddingCell::new_fp32("default", "vector", dim, values));
            let records = vec![rec];

            writer
                .write(&records)
                .await
                .expect("Failed to write records in streaming mode");
        }

        let (stats, _collector) = writer
            .finalize()
            .await
            .expect("Failed to finalize streaming writer");
        assert_eq!(stats.total_records, 10);
        // Note: These fields are on HybridWriterStats, not StreamingParquetWriterStats
        // The test needs to be rewritten or we need to track these separately
    }

    #[tokio::test]
    async fn test_hybrid_writer_batch_mode() {
        let dir = tempdir().expect("Failed to create temp directory for test");
        let file_path = dir.path().join("test_hybrid_batch.parquet");

        let config = HybridWriterConfig {
            initial_mode: WriterMode::Batch,
            enable_auto_switch: false,
            ..Default::default()
        };

        let writer = HybridParquetWriter::new(&file_path, 128, config)
            .await
            .expect("Failed to create hybrid writer in batch mode");

        // Write large batch (batch pattern)
        let records: Vec<ProximaRecord> = (0..1000)
            .map(|i| {
                let values = vec![0.1_f32; 128];
                let dim = values.len() as u32;
                let mut rec = ProximaRecord {
                    oid: format!("vec_{i}"),
                    ..Default::default()
                };
                rec.embeddings
                    .push(EmbeddingCell::new_fp32("default", "vector", dim, values));
                rec
            })
            .collect();

        writer
            .write(&records)
            .await
            .expect("Failed to write batch records");

        let (stats, _collector) = writer
            .finalize()
            .await
            .expect("Failed to finalize batch writer");
        assert_eq!(stats.total_records, 1000);
        // Note: streaming_writes and batch_writes are internal HybridWriterStats fields
    }

    #[tokio::test]
    async fn test_pattern_detection() {
        let dir = tempdir().expect("Failed to create temp directory for test");
        let file_path = dir.path().join("test_pattern.parquet");

        let config = HybridWriterConfig::default();
        let writer = HybridParquetWriter::new(&file_path, 128, config)
            .await
            .expect("Failed to create hybrid writer for pattern detection test");

        // Simulate streaming pattern
        for _ in 0..20 {
            writer.record_insertion_event(Instant::now(), 10).await;
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        let pattern = writer.detect_pattern().await;
        assert!(pattern.avg_batch_size < 100.0);
        assert_eq!(pattern.pattern_type, PatternType::Streaming);

        // Clear history
        writer.insertion_history.write().await.clear();

        // Simulate bulk pattern - need at least 10 events for detection
        for _ in 0..10 {
            writer.record_insertion_event(Instant::now(), 5000).await;
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        let pattern = writer.detect_pattern().await;
        assert!(pattern.avg_batch_size > 1000.0);
        assert_eq!(pattern.pattern_type, PatternType::Bulk);
    }
}
