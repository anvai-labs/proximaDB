//! Unified Operations Coordination - Flush, Compaction, and Re-quantization
//!
//! This module implements the critical infrastructure for coordinating background operations
//! across all storage engines (SST, VIPER, RAPTOR, NOVA, SWIFT, PRISM, HELIX).
//!
//! Key responsibilities:
//! - Prevent data corruption through intelligent locking
//! - Maximize concurrent throughput where operations don't conflict
//! - Coordinate flush → compaction → re-quantization workflows
//! - Provide observability and metrics for operation monitoring
//!
//! Extracted from the root crate's `src/storage/operations` as a slice of the root-crate
//! decomposition (see `docs/history/planning/ROOT_CRATE_DECOMPOSITION_PLAN_2026_06_21.adoc`). The
//! root crate re-exports it as `crate::storage::operations` for source compatibility.

// Lint posture inherited from the root crate, under which this code was written
// (src/lib.rs carried the same crate-level allows). Behaviour-neutral move.
#![allow(clippy::type_complexity)] // Needs type-alias extraction
#![allow(clippy::too_many_arguments)] // Needs config-struct refactor
#![allow(clippy::result_large_err)] // Needs error-type refactor

pub mod auto_scheduler;
pub mod compaction;
pub mod flush;
pub mod locks;
pub mod metrics;
pub mod requantization;

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use proximadb_proto::proximadb_v1::StorageEngine as StorageEngineType;

// Import compaction scheduler (strategies are registered automatically)
use compaction::scheduler::CompactionScheduler;

/// Result of a queued operation
#[derive(Debug, Clone)]
pub struct QueuedOperationResult {
    pub operation_id: String,
    pub collection_id: String,
    pub operation_type: OperationType,
    pub status: OperationExecutionStatus,
    pub queued_at: std::time::SystemTime,
    pub estimated_start_time: Option<std::time::SystemTime>,
}

/// Status of an operation execution
#[derive(Debug, Clone)]
pub enum OperationExecutionStatus {
    Queued,
    Running,
    Completed,
    Failed(String),
}

/// Central coordination system for all background operations
///
/// This coordinator ensures data consistency and maximizes performance by:
/// - Coordinating operations across all 7 storage engines
/// - Preventing conflicts through intelligent locking
/// - Optimizing operation scheduling based on cost models
/// - Providing comprehensive metrics and observability
#[derive(Clone)]
pub struct UnifiedOperationCoordinator {
    /// Global lock manager preventing concurrent conflicts
    lock_manager: Arc<locks::GlobalLockManager>,

    /// Flush manager for memtable → storage transitions
    flush_manager: Arc<flush::FlushManager>,

    /// Compaction manager for storage optimization
    compaction_manager: Arc<compaction::CompactionManager>,

    /// Compaction scheduler with priority queue and pluggable strategies
    /// (Leveled for SST/Nova/Raptor, Tiered for Viper/Helix/Swift)
    #[allow(dead_code)]
    compaction_scheduler: Arc<CompactionScheduler>,

    /// Re-quantization manager for codebook updates
    requantization_manager: Arc<requantization::RequantizationManager>,

    /// Operation metrics and monitoring
    metrics: Arc<metrics::OperationMetrics>,

    /// Operation coordination state
    state: Arc<RwLock<CoordinatorState>>,
}

/// Coordinator internal state
#[derive(Debug, Default)]
struct CoordinatorState {
    /// Currently executing operations by collection
    active_operations: HashMap<String, Vec<ActiveOperation>>,

    /// Operation queue by priority
    pending_operations: std::collections::VecDeque<PendingOperation>,

    /// Queued operations for tracking
    #[allow(dead_code)]
    queued_operations: HashMap<String, QueuedOperationResult>,

    /// Performance statistics
    operation_stats: OperationStatistics,
}

/// Active operation tracking
#[derive(Debug, Clone)]
struct ActiveOperation {
    #[allow(dead_code)]
    operation_id: String,
    operation_type: OperationType,
    #[allow(dead_code)]
    collection_id: String,
    #[allow(dead_code)]
    started_at: std::time::Instant,
    #[allow(dead_code)]
    estimated_duration: Duration,
}

/// Pending operation in queue
#[derive(Debug, Clone)]
struct PendingOperation {
    #[allow(dead_code)]
    operation_type: OperationType,
    #[allow(dead_code)]
    collection_id: String,
    priority: OperationPriority,
    #[allow(dead_code)]
    requested_at: std::time::Instant,
    #[allow(dead_code)]
    estimated_cost: f64,
}

/// Types of background operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum OperationType {
    /// Flush memtable to storage
    Flush,
    /// Minor compaction (level optimization)
    MinorCompaction,
    /// Major compaction (cross-level optimization)
    MajorCompaction,
    /// Re-quantization (codebook updates)
    Requantization,
}

/// Operation priority levels
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum OperationPriority {
    /// Critical operations (data consistency)
    Critical = 0,
    /// High priority (performance impact)
    High = 1,
    /// Normal priority (optimization)
    Normal = 2,
    /// Low priority (background cleanup)
    Low = 3,
}

/// Operation execution statistics
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct OperationStatistics {
    pub total_operations: u64,
    pub successful_operations: u64,
    pub failed_operations: u64,
    pub average_duration_ms: f64,
    pub operations_by_type: HashMap<OperationType, u64>,
}

impl UnifiedOperationCoordinator {
    /// Create new operation coordinator with default compaction strategies
    ///
    /// The scheduler automatically registers:
    /// - LeveledCompactionStrategy for SST, Nova, Raptor engines (LSM-tree style)
    /// - TieredCompactionStrategy for Viper, Helix, Swift engines (columnar/size-tiered)
    pub fn new() -> Result<Self> {
        info!("Initializing UnifiedOperationCoordinator");

        // Create compaction scheduler (strategies are registered automatically)
        let compaction_scheduler = Arc::new(CompactionScheduler::new());
        info!("CompactionScheduler initialized with Leveled and Tiered strategies");

        Ok(Self {
            lock_manager: Arc::new(locks::GlobalLockManager::new()?),
            flush_manager: Arc::new(flush::FlushManager::new()?),
            compaction_manager: Arc::new(compaction::CompactionManager::new()?),
            compaction_scheduler,
            requantization_manager: Arc::new(requantization::RequantizationManager::new()?),
            metrics: Arc::new(metrics::OperationMetrics::new()),
            state: Arc::new(RwLock::new(CoordinatorState::default())),
        })
    }

    /// Schedule flush operation with conflict avoidance
    ///
    /// Flushes are critical for data durability and must be prioritized.
    /// This method ensures flushes don't conflict with ongoing compactions.
    pub async fn schedule_flush(
        &self,
        collection_id: &str,
        engine_type: StorageEngineType,
    ) -> Result<OpsFlushResult> {
        info!(
            "Scheduling flush for collection: {} (engine: {:?})",
            collection_id, engine_type
        );

        // 1. Check for conflicts with existing operations
        if self
            .has_conflicting_operations(collection_id, OperationType::Flush)
            .await
        {
            warn!(
                "Flush delayed due to conflicting operations on collection: {}",
                collection_id
            );
            self.queue_flush_operation(collection_id, OperationPriority::Critical)
                .await;
            return Err(anyhow::anyhow!(
                "Flush queued due to conflicts - will execute when possible"
            ));
        }

        // 2. Acquire necessary locks
        let _flush_lock = self.lock_manager.acquire_flush_lock(collection_id).await?;

        // 3. Execute flush with the appropriate engine
        let start_time = std::time::Instant::now();
        let result = match engine_type {
            StorageEngineType::Sst => self.flush_manager.flush_sst(collection_id).await?,
            StorageEngineType::Viper => self.flush_manager.flush_viper(collection_id).await?,
            StorageEngineType::Helix => self.flush_manager.flush_helix(collection_id).await?,
            _ => {
                return Err(anyhow::anyhow!(
                    "Flush not implemented for engine: {:?}",
                    engine_type
                ));
            }
        };

        // 4. Update metrics and release locks
        let duration = start_time.elapsed();
        self.metrics
            .record_operation(OperationType::Flush, duration, true)
            .await;

        info!(
            "Flush completed for collection: {} in {:?}",
            collection_id, duration
        );

        // 5. Trigger automatic compaction if needed
        if result.should_trigger_compaction {
            let coordinator = self.clone();
            let collection_id = collection_id.to_string();
            tokio::spawn(async move {
                if let Err(e) = coordinator.schedule_minor_compaction(&collection_id).await {
                    error!("Failed to trigger automatic compaction: {}", e);
                }
            });
        }

        Ok(result)
    }

    /// Queue a flush operation for later execution
    async fn queue_flush_operation(&self, collection_id: &str, priority: OperationPriority) {
        let mut state = self.state.write().await;

        let pending_op = PendingOperation {
            operation_type: OperationType::Flush,
            collection_id: collection_id.to_string(),
            priority,
            requested_at: std::time::Instant::now(),
            estimated_cost: 1.0,
        };

        state.pending_operations.push_back(pending_op);
        state
            .pending_operations
            .make_contiguous()
            .sort_by_key(|op| op.priority);

        info!(
            "Queued Flush operation for collection: {} (priority: {:?})",
            collection_id, priority
        );
    }

    /// Schedule minor compaction with optimization
    pub async fn schedule_minor_compaction(
        &self,
        collection_id: &str,
    ) -> Result<OpsCompactionResult> {
        info!(
            "Scheduling minor compaction for collection: {}",
            collection_id
        );

        // 1. Check for conflicts with existing operations
        if self
            .has_conflicting_operations(collection_id, OperationType::MinorCompaction)
            .await
        {
            warn!(
                "Minor compaction delayed due to conflicting operations on collection: {}",
                collection_id
            );
            return Err(anyhow::anyhow!("Minor compaction queued due to conflicts"));
        }

        // 2. Acquire compaction lock for levels 0-2 (typical minor compaction range)
        let _compaction_lock = self
            .lock_manager
            .acquire_compaction_lock(collection_id, 0, 2)
            .await?;

        // 3. Execute minor compaction
        let start_time = std::time::Instant::now();
        let result = self
            .compaction_manager
            .execute_minor_compaction(collection_id, 0, 2, StorageEngineType::Sst)
            .await?;

        // 4. Update metrics
        let duration = start_time.elapsed();
        self.metrics
            .record_operation(OperationType::MinorCompaction, duration, true)
            .await;

        info!(
            "Minor compaction completed for collection: {} in {:?}",
            collection_id, duration
        );

        // 5. Check if major compaction or re-quantization is needed
        if result.should_trigger_requantization {
            let coordinator = self.clone();
            let collection_id = collection_id.to_string();
            tokio::spawn(async move {
                if let Err(e) = coordinator.schedule_requantization(&collection_id).await {
                    error!("Failed to trigger automatic re-quantization: {}", e);
                }
            });
        }

        Ok(result)
    }

    /// Schedule major compaction across levels
    pub async fn schedule_major_compaction(
        &self,
        collection_id: &str,
    ) -> Result<OpsCompactionResult> {
        info!(
            "Scheduling major compaction for collection: {}",
            collection_id
        );

        // 1. Check for conflicts with existing operations (major compaction conflicts with flushes)
        if self
            .has_conflicting_operations(collection_id, OperationType::MajorCompaction)
            .await
        {
            warn!(
                "Major compaction delayed due to conflicting operations on collection: {}",
                collection_id
            );
            return Err(anyhow::anyhow!("Major compaction queued due to conflicts"));
        }

        // 2. Acquire exclusive locks across all levels (major compaction)
        let _compaction_lock = self
            .lock_manager
            .acquire_compaction_lock(collection_id, 0, 10) // Full range
            .await?;

        // 3. Execute major compaction
        let start_time = std::time::Instant::now();
        let result = self
            .compaction_manager
            .execute_major_compaction(collection_id, StorageEngineType::Sst)
            .await?;

        // 4. Update metrics
        let duration = start_time.elapsed();
        self.metrics
            .record_operation(OperationType::MajorCompaction, duration, true)
            .await;

        info!(
            "Major compaction completed for collection: {} in {:?} (freed: {} bytes)",
            collection_id, duration, result.bytes_freed
        );

        // 5. Schedule re-quantization if data distribution changed significantly
        if result.should_trigger_requantization {
            let coordinator = self.clone();
            let collection_id = collection_id.to_string();
            tokio::spawn(async move {
                if let Err(e) = coordinator.schedule_requantization(&collection_id).await {
                    error!(
                        "Failed to trigger automatic re-quantization after major compaction: {}",
                        e
                    );
                }
            });
        }

        Ok(result)
    }

    /// Schedule re-quantization when data distribution changes
    pub async fn schedule_requantization(
        &self,
        collection_id: &str,
    ) -> Result<RequantizationResult> {
        info!(
            "Scheduling re-quantization for collection: {}",
            collection_id
        );

        // 1. Analyze if re-quantization is actually needed
        let engine_type = self.get_collection_engine_type(collection_id);
        let needs_requantization = self
            .requantization_manager
            .analyze_collection(collection_id, engine_type)
            .await?;

        if !needs_requantization {
            info!(
                "Re-quantization not needed for collection: {} based on analysis",
                collection_id
            );
            return Ok(RequantizationResult {
                collection_id: collection_id.to_string(),
                codebooks_updated: vec![],
                quality_improvement: 0.0,
                duration: Duration::from_millis(0),
            });
        }

        // 2. Check for conflicts (re-quantization conflicts with all other operations)
        if self
            .has_conflicting_operations(collection_id, OperationType::Requantization)
            .await
        {
            warn!(
                "Re-quantization delayed due to conflicting operations on collection: {}",
                collection_id
            );
            return Err(anyhow::anyhow!("Re-quantization queued due to conflicts"));
        }

        // 3. Acquire global re-quantization lock (exclusive)
        let _requantization_lock = self.lock_manager.acquire_requantization_lock().await?;

        // 4. Execute re-quantization with automatic type detection
        let start_time = std::time::Instant::now();
        let result = self
            .requantization_manager
            .execute_requantization(
                collection_id,
                requantization::QuantizationType::FullRequantization,
                engine_type,
            )
            .await?;

        // 5. Update metrics
        let duration = start_time.elapsed();
        self.metrics
            .record_operation(OperationType::Requantization, duration, true)
            .await;

        info!(
            "Re-quantization completed for collection: {} in {:?} (quality improvement: {:.2}%)",
            collection_id,
            duration,
            result.quality_improvement * 100.0
        );

        Ok(result)
    }

    /// Get the storage engine type for a collection (simplified for now)
    fn get_collection_engine_type(&self, _collection_id: &str) -> StorageEngineType {
        // Default to SST - in production this would query collection metadata
        StorageEngineType::Sst
    }

    /// Check for conflicting operations
    async fn has_conflicting_operations(
        &self,
        collection_id: &str,
        operation: OperationType,
    ) -> bool {
        let state = self.state.read().await;

        if let Some(active_ops) = state.active_operations.get(collection_id) {
            active_ops
                .iter()
                .any(|op| self.operations_conflict(operation, op.operation_type))
        } else {
            false
        }
    }

    /// Determine if two operation types conflict
    fn operations_conflict(&self, op1: OperationType, op2: OperationType) -> bool {
        use OperationType::*;

        match (op1, op2) {
            // Flushes conflict with major compactions
            (Flush, MajorCompaction) | (MajorCompaction, Flush) => true,
            // Re-quantization conflicts with all write operations
            (Requantization, _) | (_, Requantization) => true,
            // Minor compactions can run with flushes on different levels
            (Flush, MinorCompaction) | (MinorCompaction, Flush) => false,
            // Same operation types conflict
            (a, b) if a == b => true,
            // Otherwise, operations can run concurrently
            _ => false,
        }
    }

    /// Estimate operation cost for scheduling optimization
    #[allow(dead_code)]
    async fn estimate_operation_cost(&self, _collection_id: &str, operation: OperationType) -> f64 {
        let base_cost = match operation {
            OperationType::Flush => 1.0,
            OperationType::MinorCompaction => 5.0,
            OperationType::MajorCompaction => 20.0,
            OperationType::Requantization => 15.0,
        };

        // Simplified cost estimation - can be enhanced with collection size, etc.
        debug!("Cost estimation for {:?}: {}", operation, base_cost);
        base_cost
    }

    /// Get current operation status for monitoring
    pub async fn get_operation_status(&self) -> OperationStatus {
        let state = self.state.read().await;

        OperationStatus {
            active_operations: state.active_operations.values().map(|v| v.len()).sum(),
            pending_operations: state.pending_operations.len(),
            total_collections_managed: state.active_operations.len(),
            operation_stats: state.operation_stats.clone(),
        }
    }
}

/// Backwards-compat alias for [`OpsFlushResult`].
pub type FlushResult = OpsFlushResult;

/// Flush operation result
#[derive(Debug, Clone)]
pub struct OpsFlushResult {
    pub collection_id: String,
    pub files_created: Vec<String>,
    pub bytes_written: u64,
    pub duration: Duration,
    pub should_trigger_compaction: bool,
}

/// Backwards-compat alias for [`OpsCompactionResult`].
pub type CompactionResult = OpsCompactionResult;

/// Compaction operation result
#[derive(Debug, Clone)]
pub struct OpsCompactionResult {
    pub collection_id: String,
    pub files_compacted: Vec<String>,
    pub files_created: Vec<String>,
    pub bytes_freed: u64,
    pub duration: Duration,
    pub should_trigger_requantization: bool,
}

/// Re-quantization operation result
#[derive(Debug, Clone)]
pub struct RequantizationResult {
    pub collection_id: String,
    pub codebooks_updated: Vec<String>,
    pub quality_improvement: f64,
    pub duration: Duration,
}

/// Current operation status for monitoring
#[derive(Debug, Clone)]
pub struct OperationStatus {
    pub active_operations: usize,
    pub pending_operations: usize,
    pub total_collections_managed: usize,
    pub operation_stats: OperationStatistics,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_coordinator_creation() {
        let coordinator = UnifiedOperationCoordinator::new().unwrap();
        let status = coordinator.get_operation_status().await;

        assert_eq!(status.active_operations, 0);
        assert_eq!(status.pending_operations, 0);
    }

    #[tokio::test]
    async fn test_operation_conflict_detection() {
        let coordinator = UnifiedOperationCoordinator::new().unwrap();

        // Test conflict matrix
        assert!(
            coordinator.operations_conflict(OperationType::Flush, OperationType::MajorCompaction)
        );
        assert!(
            coordinator.operations_conflict(OperationType::Requantization, OperationType::Flush)
        );
        assert!(
            !coordinator.operations_conflict(OperationType::Flush, OperationType::MinorCompaction)
        );
    }

    #[test]
    fn test_operation_type_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(OperationType::Flush);
        set.insert(OperationType::MinorCompaction);
        assert_eq!(set.len(), 2);
    }
}
