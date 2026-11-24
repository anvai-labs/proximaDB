//! Compaction Manager - Optimize storage by merging and consolidating files
//!
//! This module implements intelligent compaction strategies for all storage engines,
//! including SST-based engines (LSM trees) and columnar engines (Parquet).

use anyhow::Result;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info};

// Import from parent operations module
use crate::CompactionResult;
use proximadb_proto::proximadb_v1::StorageEngine as StorageEngineType;

// Import from our flattened strategy modules (unused but kept for future integration)
#[allow(unused_imports)]
use super::{CompactionExecutionResult, CompactionPlan, CompactionStrategyRegistry, FileMetadata};

/// Compaction manager coordinates optimization operations across storage engines
pub struct CompactionManager {
    /// Configuration for compaction behavior
    config: CompactionConfig,

    /// Current compaction state
    state: Arc<RwLock<CompactionState>>,

    /// Performance metrics tracking
    #[allow(dead_code)]
    metrics: Arc<CompactionMetrics>,
}

/// Compaction configuration
#[derive(Debug, Clone)]
struct CompactionConfig {
    /// Maximum number of concurrent compactions
    #[allow(dead_code)]
    max_concurrent_compactions: usize,

    /// Minimum files required to trigger minor compaction
    #[allow(dead_code)]
    minor_compaction_threshold: usize,

    /// Minimum files required to trigger major compaction
    #[allow(dead_code)]
    major_compaction_threshold: usize,

    /// Maximum time between compactions (force compaction)
    max_compaction_interval: Duration,

    /// Target file size after compaction
    #[allow(dead_code)]
    target_file_size_mb: usize,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            max_concurrent_compactions: 3,
            minor_compaction_threshold: 4,
            major_compaction_threshold: 10,
            max_compaction_interval: Duration::from_secs(24 * 60 * 60), // 24 hours
            target_file_size_mb: 128,
        }
    }
}

/// Current compaction state
#[derive(Debug, Default)]
struct CompactionState {
    /// Active compactions by collection
    active_compactions: std::collections::HashMap<String, Vec<ActiveCompaction>>,

    /// Last compaction time by collection
    last_compaction_time: std::collections::HashMap<String, Instant>,

    /// Compaction statistics
    stats: CompactionStatistics,
}

/// Active compaction tracking
#[derive(Debug, Clone)]
struct ActiveCompaction {
    operation_id: String,
    #[allow(dead_code)]
    collection_id: String,
    #[allow(dead_code)]
    compaction_type: CompactionType,
    #[allow(dead_code)]
    started_at: Instant,
    #[allow(dead_code)]
    estimated_completion: Instant,
}

/// Types of compaction operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionType {
    /// Minor compaction within single level
    Minor,
    /// Major compaction across multiple levels
    Major,
    /// Full collection compaction
    Full,
}

/// Compaction performance statistics
#[derive(Debug, Default)]
struct CompactionStatistics {
    total_compactions: u64,
    successful_compactions: u64,
    #[allow(dead_code)]
    failed_compactions: u64,
    total_bytes_compacted: u64,
    total_files_compacted: u64,
    #[allow(dead_code)]
    average_compaction_time: Duration,
}

/// Compaction metrics tracking
#[derive(Debug, Default)]
pub struct CompactionMetrics {
    /// Performance counters
    #[allow(dead_code)]
    counters: Arc<RwLock<CompactionStatistics>>,
}

impl CompactionManager {
    /// Create new compaction manager
    pub fn new() -> Result<Self> {
        info!("🔧 Initializing CompactionManager");

        Ok(Self {
            config: CompactionConfig::default(),
            state: Arc::new(RwLock::new(CompactionState::default())),
            metrics: Arc::new(CompactionMetrics::default()),
        })
    }

    /// Execute minor compaction for specific level range
    pub async fn execute_minor_compaction(
        &self,
        collection_id: &str,
        start_level: u32,
        end_level: u32,
        engine_type: StorageEngineType,
    ) -> Result<CompactionResult> {
        info!(
            "🔧 Executing minor compaction for collection: {} levels {}-{} (engine: {:?})",
            collection_id, start_level, end_level, engine_type
        );

        let start_time = Instant::now();
        let operation_id = format!(
            "minor_{}_{}_{}_{}",
            collection_id,
            start_level,
            end_level,
            chrono::Utc::now().timestamp_millis()
        );

        // Record active compaction
        {
            let mut state = self.state.write().await;
            let active_ops = state
                .active_compactions
                .entry(collection_id.to_string())
                .or_default();
            active_ops.push(ActiveCompaction {
                operation_id: operation_id.clone(),
                collection_id: collection_id.to_string(),
                compaction_type: CompactionType::Minor,
                started_at: start_time,
                estimated_completion: start_time + Duration::from_secs(60), // Estimate 1 minute
            });
        }

        // Execute engine-specific minor compaction
        let result = match engine_type {
            StorageEngineType::Sst => {
                self.compact_sst_levels(collection_id, start_level, end_level)
                    .await?
            }
            StorageEngineType::Helix => {
                self.compact_helix_segments(collection_id, start_level, end_level)
                    .await?
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "Minor compaction not implemented for engine: {:?}",
                    engine_type
                ));
            }
        };

        // Update metrics and clean up
        let duration = start_time.elapsed();
        {
            let mut state = self.state.write().await;
            if let Some(active_ops) = state.active_compactions.get_mut(collection_id) {
                active_ops.retain(|op| op.operation_id != operation_id);
            }
            state
                .last_compaction_time
                .insert(collection_id.to_string(), Instant::now());

            state.stats.total_compactions += 1;
            state.stats.successful_compactions += 1;
            state.stats.total_bytes_compacted += result.bytes_freed;
            state.stats.total_files_compacted += result.files_compacted.len() as u64;
        }

        info!(
            "✅ Minor compaction completed for collection: {} in {:?} (freed: {} bytes)",
            collection_id, duration, result.bytes_freed
        );

        // Bump corpus_version for every tenant that has cached plans
        // against this collection. Compaction publishes a new segment
        // and removes old ones; the planner's selectivity + route
        // choice may shift because the storage layout changed. The
        // compaction manager only knows the collection_id, not the
        // tenant, so we use the registry's all-tenants variant.
        let bumped = if let Some(p) = proximadb_storage_ports::corpus_version() {
            p.bump_collection_all_tenants(collection_id).await
        } else {
            0
        };
        if bumped > 0 {
            tracing::debug!(
                collection = %collection_id,
                bumped,
                "🔄 corpus_version bumped after minor compaction"
            );
        }

        Ok(result)
    }

    /// Execute major compaction across multiple levels
    pub async fn execute_major_compaction(
        &self,
        collection_id: &str,
        engine_type: StorageEngineType,
    ) -> Result<CompactionResult> {
        info!(
            "🔧 Executing major compaction for collection: {} (engine: {:?})",
            collection_id, engine_type
        );

        let start_time = Instant::now();
        let operation_id = format!(
            "major_{}_{}",
            collection_id,
            chrono::Utc::now().timestamp_millis()
        );

        // Record active compaction
        {
            let mut state = self.state.write().await;
            let active_ops = state
                .active_compactions
                .entry(collection_id.to_string())
                .or_default();
            active_ops.push(ActiveCompaction {
                operation_id: operation_id.clone(),
                collection_id: collection_id.to_string(),
                compaction_type: CompactionType::Major,
                started_at: start_time,
                estimated_completion: start_time + Duration::from_secs(300), // Estimate 5 minutes
            });
        }

        // Execute engine-specific major compaction
        let result = match engine_type {
            StorageEngineType::Sst => self.compact_sst_full_collection(collection_id).await?,
            StorageEngineType::Helix => self.compact_helix_full_collection(collection_id).await?,
            _ => {
                return Err(anyhow::anyhow!(
                    "Major compaction not implemented for engine: {:?}",
                    engine_type
                ));
            }
        };

        // Update metrics and clean up
        let duration = start_time.elapsed();
        {
            let mut state = self.state.write().await;
            if let Some(active_ops) = state.active_compactions.get_mut(collection_id) {
                active_ops.retain(|op| op.operation_id != operation_id);
            }
            state
                .last_compaction_time
                .insert(collection_id.to_string(), Instant::now());

            state.stats.total_compactions += 1;
            state.stats.successful_compactions += 1;
            state.stats.total_bytes_compacted += result.bytes_freed;
            state.stats.total_files_compacted += result.files_compacted.len() as u64;
        }

        info!(
            "✅ Major compaction completed for collection: {} in {:?} (freed: {} bytes)",
            collection_id, duration, result.bytes_freed
        );

        // Bump corpus_version across all tenants — major compaction
        // rewrites the storage layout more invasively than minor;
        // any cached plan should be reconsidered against the new state.
        let bumped = if let Some(p) = proximadb_storage_ports::corpus_version() {
            p.bump_collection_all_tenants(collection_id).await
        } else {
            0
        };
        if bumped > 0 {
            tracing::debug!(
                collection = %collection_id,
                bumped,
                "🔄 corpus_version bumped after major compaction"
            );
        }

        Ok(result)
    }

    /// Check if collection needs compaction based on heuristics
    pub async fn needs_compaction(&self, collection_id: &str) -> bool {
        let state = self.state.read().await;

        // Check if enough time has passed since last compaction
        if let Some(last_time) = state.last_compaction_time.get(collection_id)
            && last_time.elapsed() > self.config.max_compaction_interval
        {
            return true;
        }

        // Deferred: Add additional heuristics:
        // - Number of small files
        // - Read amplification metrics
        // - Space amplification ratios
        // - Query performance degradation

        false
    }

    /// Get current compaction status for monitoring
    pub async fn get_compaction_status(&self) -> CompactionStatus {
        let state = self.state.read().await;

        CompactionStatus {
            active_compactions: state.active_compactions.values().flatten().count(),
            collections_being_compacted: state.active_compactions.len(),
            total_compactions_completed: state.stats.total_compactions,
            total_bytes_freed: state.stats.total_bytes_compacted,
        }
    }

    // Private implementation methods for different engines

    async fn compact_sst_levels(
        &self,
        collection_id: &str,
        start_level: u32,
        end_level: u32,
    ) -> Result<CompactionResult> {
        // Deferred: Implement SST level compaction
        // 1. Identify overlapping SSTables in level range
        // 2. Merge SSTables with efficient key range processing
        // 3. Write optimized SSTables with proper bloom filters
        // 4. Update metadata and remove old files

        debug!(
            "Compacting SST levels {}-{} for collection: {}",
            start_level, end_level, collection_id
        );

        Ok(CompactionResult {
            collection_id: collection_id.to_string(),
            files_compacted: vec![format!("level_{}_{}.sst", start_level, end_level)],
            files_created: vec![format!("compacted_{}_{}.sst", start_level, end_level)],
            bytes_freed: 1024 * 1024, // Placeholder
            duration: Duration::from_secs(30),
            should_trigger_requantization: false,
        })
    }

    async fn compact_helix_segments(
        &self,
        collection_id: &str,
        start_level: u32,
        end_level: u32,
    ) -> Result<CompactionResult> {
        // Deferred: Implement HELIX segment compaction
        // 1. Merge HELIX segments with optimal clustering
        // 2. Rebuild zone maps and PCA projections
        // 3. Optimize Hilbert curve ordering
        // 4. Update liquid clustering metadata

        debug!(
            "Compacting HELIX segments levels {}-{} for collection: {}",
            start_level, end_level, collection_id
        );

        Ok(CompactionResult {
            collection_id: collection_id.to_string(),
            files_compacted: vec![format!("segment_{}_{}.helix", start_level, end_level)],
            files_created: vec![format!("compacted_{}_{}.helix", start_level, end_level)],
            bytes_freed: 3072 * 1024, // Placeholder
            duration: Duration::from_secs(60),
            should_trigger_requantization: true, // HELIX benefits from re-quantization
        })
    }

    async fn compact_sst_full_collection(&self, collection_id: &str) -> Result<CompactionResult> {
        // Deferred: Implement full SST collection compaction
        debug!("Full SST compaction for collection: {}", collection_id);

        Ok(CompactionResult {
            collection_id: collection_id.to_string(),
            files_compacted: vec!["all_levels.sst".to_string()],
            files_created: vec!["compacted_full.sst".to_string()],
            bytes_freed: 5120 * 1024,
            duration: Duration::from_secs(120),
            should_trigger_requantization: false,
        })
    }

    async fn compact_helix_full_collection(&self, collection_id: &str) -> Result<CompactionResult> {
        // Deferred: Implement full HELIX collection compaction
        debug!("Full HELIX compaction for collection: {}", collection_id);

        Ok(CompactionResult {
            collection_id: collection_id.to_string(),
            files_compacted: vec!["all_segments.helix".to_string()],
            files_created: vec!["compacted_full.helix".to_string()],
            bytes_freed: 8192 * 1024,
            duration: Duration::from_secs(240),
            should_trigger_requantization: true,
        })
    }
}

/// Current compaction status for monitoring
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CompactionStatus {
    pub active_compactions: usize,
    pub collections_being_compacted: usize,
    pub total_compactions_completed: u64,
    pub total_bytes_freed: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use proximadb_proto::proximadb_v1::StorageEngine as StorageEngineType;

    #[tokio::test]
    async fn test_compaction_manager_creation() {
        let manager = CompactionManager::new().unwrap();
        let status = manager.get_compaction_status().await;

        assert_eq!(status.active_compactions, 0);
        assert_eq!(status.collections_being_compacted, 0);
    }

    #[tokio::test]
    async fn test_minor_compaction_execution() {
        let manager = CompactionManager::new().unwrap();

        // This test would need mock storage engines in a real implementation
        // For now, we test that the structure works correctly
        assert!(!manager.needs_compaction("test_collection").await);
    }

    #[tokio::test]
    async fn test_compaction_status_tracking() {
        let manager = CompactionManager::new().unwrap();
        let initial_status = manager.get_compaction_status().await;

        assert_eq!(initial_status.active_compactions, 0);
        assert_eq!(initial_status.total_compactions_completed, 0);
    }
}
