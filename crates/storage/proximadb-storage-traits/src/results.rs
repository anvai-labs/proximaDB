//! Result types for storage operations.
//!
//! This module contains the result types returned by storage operations
//! such as flush, compaction, and engine statistics.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use proximadb_kernel::CompactBatchId as BatchId;

/// Unified flush result that accommodates different engine types.
///
/// Note: Default values use u64::MAX to indicate uninitialized state.
/// This allows distinguishing between:
/// - Uninitialized: u64::MAX (default)
/// - Successful operation with zero results: 0
///
/// Backwards-compat alias for [`StorageFlushResult`].
pub type FlushResult = StorageFlushResult;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageFlushResult {
    /// Operation completed successfully
    pub success: bool,

    /// Collections affected by the flush
    pub collections_affected: Vec<String>,

    /// Number of entries flushed
    pub entries_flushed: Option<u64>,

    /// Bytes written to storage
    pub bytes_written: Option<u64>,

    /// Number of files/segments created
    pub files_created: Option<u64>,

    /// Actual file paths created (for AXIS index building)
    pub file_paths: Vec<String>,

    /// Duration of the operation
    pub duration_ms: Option<u64>,

    /// Timestamp when operation completed
    pub completed_at: DateTime<Utc>,

    /// Engine-specific metrics
    pub engine_metrics: HashMap<String, serde_json::Value>,

    /// Whether compaction was triggered as a result
    pub compaction_triggered: bool,

    /// Error message if post-flush compaction failed (for observability and retry scheduling)
    pub compaction_error: Option<String>,

    /// Batch IDs that were successfully flushed (for WAL cleanup coordination)
    pub flushed_batch_ids: Vec<BatchId>,
}

impl Default for StorageFlushResult {
    fn default() -> Self {
        Self {
            success: false,
            collections_affected: Vec::new(),
            entries_flushed: None,
            bytes_written: None,
            files_created: None,
            file_paths: Vec::new(),
            duration_ms: None,
            completed_at: Utc::now(),
            engine_metrics: HashMap::new(),
            compaction_triggered: false,
            compaction_error: None,
            flushed_batch_ids: Vec::new(),
        }
    }
}

/// Unified compaction result that accommodates different engine types.
///
/// Note: Default values use u64::MAX to indicate uninitialized state.
/// This allows distinguishing between:
/// - Uninitialized: u64::MAX (default)
/// - Successful operation with zero results: 0
///
/// Backwards-compat alias for [`StorageCompactionResult`].
pub type CompactionResult = StorageCompactionResult;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageCompactionResult {
    /// Operation completed successfully
    pub success: bool,

    /// Collections affected by the compaction
    pub collections_affected: Vec<String>,

    /// Number of entries processed
    pub entries_processed: Option<u64>,

    /// Number of entries removed (tombstones, duplicates, etc.)
    pub entries_removed: Option<u64>,

    /// Bytes read during compaction
    pub bytes_read: Option<u64>,

    /// Bytes written during compaction
    pub bytes_written: Option<u64>,

    /// Input files/segments processed
    pub input_files: Option<u64>,

    /// Output files/segments created
    pub output_files: Option<u64>,

    /// Duration of the operation
    pub duration_ms: Option<u64>,

    /// Timestamp when operation completed
    pub completed_at: DateTime<Utc>,

    /// Engine-specific metrics (e.g., compression ratio, level info)
    pub engine_metrics: HashMap<String, serde_json::Value>,
}

impl Default for StorageCompactionResult {
    fn default() -> Self {
        Self {
            success: false,
            collections_affected: Vec::new(),
            entries_processed: None,
            entries_removed: None,
            bytes_read: None,
            bytes_written: None,
            input_files: None,
            output_files: None,
            duration_ms: None,
            completed_at: Utc::now(),
            engine_metrics: HashMap::new(),
        }
    }
}

/// The FULL-collapse provenance a `COMPACT ... FULL` result carries
/// (TD-COMPACT-14/15), extracted once so every API surface (REST, Flight,
/// DDL) reports the same keys and cannot drift. Fields the engine did not
/// report (a plain, non-FULL pass) stay `None`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FullCompactionMetrics {
    pub segments_before: Option<usize>,
    pub segments_after: Option<usize>,
    pub passes: Option<usize>,
    pub reached_target: Option<bool>,
    pub stop_reason: Option<String>,
    pub pending_files: Option<usize>,
    pub bytes_read: Option<u64>,
    pub bytes_written: Option<u64>,
    pub duration_ms: Option<u64>,
}

impl StorageCompactionResult {
    /// Extract the FULL provenance from `engine_metrics` (keys written by
    /// the SST FULL branch). The single extraction point for all surfaces.
    pub fn full_compaction_metrics(&self) -> FullCompactionMetrics {
        let metric_u64 = |key: &str| {
            self.engine_metrics
                .get(key)
                .and_then(|value| value.as_u64())
        };
        FullCompactionMetrics {
            segments_before: metric_u64("segments_before").map(|v| v as usize),
            segments_after: metric_u64("segments_after").map(|v| v as usize),
            passes: metric_u64("passes").map(|v| v as usize),
            reached_target: self
                .engine_metrics
                .get("reached_target")
                .and_then(|value| value.as_bool()),
            stop_reason: self
                .engine_metrics
                .get("stop_reason")
                .and_then(|value| value.as_str().map(str::to_string)),
            pending_files: metric_u64("pending_files").map(|v| v as usize),
            // Fall back to the canonical top-level fields: a path that sets
            // the struct fields but not the engine_metrics map (SST's plain
            // non-FULL compaction, other engines) still reports its measured
            // bytes instead of null. Otherwise this "single extraction
            // point" silently loses data it was built to centralize.
            bytes_read: metric_u64("bytes_read").or(self.bytes_read),
            bytes_written: metric_u64("bytes_written").or(self.bytes_written),
            duration_ms: metric_u64("duration_ms").or(self.duration_ms),
        }
    }
}

/// Engine statistics for monitoring and observability.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct EngineStatistics {
    /// Engine name and version
    pub engine_name: String,
    pub engine_version: String,

    /// Total storage size
    pub total_storage_bytes: u64,

    /// Memory usage
    pub memory_usage_bytes: u64,

    /// Number of collections
    pub collection_count: usize,

    /// Last flush time
    pub last_flush: Option<DateTime<Utc>>,

    /// Last compaction time
    pub last_compaction: Option<DateTime<Utc>>,

    /// Pending operations
    pub pending_flushes: u64,
    pub pending_compactions: u64,

    /// Engine-specific metrics
    pub engine_specific: HashMap<String, serde_json::Value>,
}

/// Engine health status for monitoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineHealth {
    /// Overall health status
    pub healthy: bool,

    /// Health status message
    pub status: String,

    /// Last health check time
    pub last_check: DateTime<Utc>,

    /// Response time for health check
    pub response_time_ms: f64,

    /// Error count in recent period
    pub error_count: usize,

    /// Warning messages
    pub warnings: Vec<String>,

    /// Engine-specific health metrics
    pub metrics: HashMap<String, serde_json::Value>,
}

impl Default for EngineHealth {
    fn default() -> Self {
        Self {
            healthy: false,
            status: String::new(),
            last_check: Utc::now(),
            response_time_ms: 0.0,
            error_count: 0,
            warnings: Vec::new(),
            metrics: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flush_result_default() {
        let result = StorageFlushResult::default();
        assert!(!result.success);
        assert!(result.collections_affected.is_empty());
    }

    #[test]
    fn test_compaction_result_default() {
        let result = StorageCompactionResult::default();
        assert!(!result.success);
        assert!(result.collections_affected.is_empty());
    }

    #[test]
    fn test_engine_health() {
        let health = EngineHealth {
            healthy: true,
            status: "OK".to_string(),
            last_check: Utc::now(),
            response_time_ms: 10.0,
            error_count: 0,
            warnings: vec![],
            metrics: HashMap::new(),
        };
        assert!(health.healthy);
        assert_eq!(health.status, "OK");
    }
}
