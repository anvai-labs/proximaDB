//! Parameter types for storage operations.
//!
//! This module contains configuration and parameter types used
//! throughout the storage layer for flush, compaction, and other operations.

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use proximadb_proto::proximadb_v1::Collection;
// `BatchId` is the canonical alias for `CompactBatchId` (hoisted to the kernel
// foundation crate). The alias matches the root's
// `crate::storage::persistence::write_ahead_log::BatchId` re-export so the
// field types stay identical for back-compat.
use proximadb_kernel::CompactBatchId as BatchId;

/// Flexible flush parameters that work for all storage engines.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FlushParameters {
    /// Decimal catalog object identity at the legacy storage-trait boundary.
    ///
    /// `None` means a global flush for engines that support it. New scheduling,
    /// admission, and cache code must immediately parse this adapter field to
    /// `CollectionObjectId`; user-facing collection aliases never cross this
    /// boundary.
    pub collection_id: Option<String>,

    /// Force immediate flush regardless of thresholds
    pub force: bool,

    /// Wait for completion before returning
    pub synchronous: bool,

    /// Engine-specific hints
    pub hints: HashMap<String, serde_json::Value>,

    /// Maximum time to wait for operation
    pub timeout_ms: Option<u64>,

    /// Canonical records to flush (provided by FlushCoordinator from WAL).
    /// Protocol adapters (REST, gRPC, Arrow, pgwire) convert to `ProximaRecord` before
    /// inserting into WAL/flush path. `VectorRecord` no longer crosses this boundary.
    pub vector_records: Vec<proximadb_records::ProximaRecord>,

    /// Whether to trigger compaction after flush
    pub trigger_compaction: bool,

    /// Batch IDs involved in this flush operation (for coordination)
    pub batch_ids: Vec<BatchId>,

    /// Collection configuration to avoid redundant lookups
    pub collection_config: Option<Collection>,

    /// Estimated size in bytes for metrics tracking
    pub estimated_size: usize,
}

impl FlushParameters {
    /// Resolve the target collection id from the explicit parameter or cached config.
    pub fn get_collection_id(&self) -> Result<String> {
        self.collection_id
            .clone()
            .or_else(|| {
                self.collection_config
                    .as_ref()
                    .map(|collection| collection.id.clone())
            })
            .ok_or_else(|| anyhow!("No collection_id provided in flush parameters"))
    }

    /// Resolve the native, globally unique catalog object identity used for
    /// admission, scheduling, WAL, and cache keying.
    ///
    /// Aliases and UUID-shaped legacy identifiers must be resolved by the
    /// catalog before they reach this storage boundary. Hashing one here would
    /// create a second, collision-prone identity domain that cannot be joined
    /// reliably with catalog, MVCC, deletion-vector, or compaction state.
    pub fn get_collection_object_id(
        &self,
    ) -> Result<proximadb_kernel::stable_id::CollectionObjectId> {
        let collection_id = self.get_collection_id()?;
        collection_id.parse().map_err(|error| {
            anyhow!(
                "flush collection_id must be a decimal catalog object id, got {collection_id:?}: {error}"
            )
        })
    }

    /// Resolve the composite `CollectionIdentity` from the cached config's
    /// `StorageAssignment` typed triple (ADR-0083 rev2 D3 — always minted).
    /// Returns `CollectionIdentity::default()` (all-zeros) when the typed
    /// triple is absent (legacy/pre-migration — should not happen post-Phase 1
    /// always-mint, but safe).
    pub fn get_collection_identity(&self) -> proximadb_kernel::stable_id::CollectionIdentity {
        self.collection_config
            .as_ref()
            .and_then(|cc| cc.storage_assignment.as_ref())
            .and_then(|sa| {
                match (
                    sa.typed_account_id,
                    sa.typed_namespace_id,
                    sa.typed_collection_id,
                ) {
                    (Some(a), Some(ns), Some(c)) => {
                        Some(proximadb_kernel::stable_id::CollectionIdentity {
                            account_id: a,
                            namespace_id: ns as u16,
                            collection_id: c,
                        })
                    }
                    _ => None,
                }
            })
            .unwrap_or_default()
    }

    /// Resolve the collection data directory from cached config or an engine hint.
    pub fn get_data_dir(&self) -> Result<String> {
        if let Some(collection_config) = &self.collection_config
            && let Some(storage_assignment) = &collection_config.storage_assignment
        {
            return Ok(format!(
                "{}/{}/data",
                storage_assignment.base_location, collection_config.id
            ));
        }

        self.hints
            .get("data_dir")
            .and_then(|value| value.as_str())
            .map(str::to_string)
            .ok_or_else(|| anyhow!("Flush parameters require collection_config or a data_dir hint"))
    }
}

/// Flexible compaction parameters that work for all storage engines.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompactionParameters {
    /// Target collection (None means global compaction for engines that support it)
    pub collection_id: Option<String>,

    /// Force compaction regardless of thresholds
    pub force: bool,

    /// Wait for completion before returning
    pub synchronous: bool,

    /// Engine-specific hints (e.g., target level for LSM, cluster hints for VIPER)
    pub hints: HashMap<String, serde_json::Value>,

    /// Maximum time to wait for operation
    pub timeout_ms: Option<u64>,

    /// Priority level for the operation
    pub priority: OperationPriority,

    /// Collection configuration to avoid redundant lookups
    pub collection_config: Option<Collection>,

    /// Estimated input size in bytes for metrics tracking
    pub estimated_input_size: usize,

    /// Operator-requested cross-level collapse (`COMPACT ... FULL`): merge
    /// toward `CompactionConfig::full_target_segments`, pairing the largest
    /// segments across levels. Engines without a cross-level mode treat it
    /// as an explicit single pass.
    #[serde(default)]
    pub full: bool,
}

impl CompactionParameters {
    /// Resolve the target collection id from the explicit parameter or cached config.
    pub fn get_collection_id(&self) -> Result<String> {
        self.collection_id
            .clone()
            .or_else(|| {
                self.collection_config
                    .as_ref()
                    .map(|collection| collection.id.clone())
            })
            .ok_or_else(|| {
                anyhow!("Compaction parameters require a collection_id or collection_config")
            })
    }

    /// Resolve the native, globally unique catalog object identity.
    ///
    /// Compaction may retire files and cache entries, so accepting a mutable
    /// collection alias at this boundary would make the operation ambiguous.
    pub fn get_collection_object_id(
        &self,
    ) -> Result<proximadb_kernel::stable_id::CollectionObjectId> {
        let collection_id = self.get_collection_id()?;
        collection_id.parse().map_err(|error| {
            anyhow!(
                "compaction collection_id must be a decimal catalog object id, got {collection_id:?}: {error}"
            )
        })
    }

    /// Resolve the collection data directory from cached config or an engine hint.
    pub fn get_data_dir(&self) -> Result<String> {
        if let Some(collection_config) = &self.collection_config
            && let Some(storage_assignment) = &collection_config.storage_assignment
        {
            return Ok(format!(
                "{}/{}/data",
                storage_assignment.base_location, collection_config.id
            ));
        }

        self.hints
            .get("data_dir")
            .and_then(|value| value.as_str())
            .map(str::to_string)
            .ok_or_else(|| {
                anyhow!("Compaction parameters require collection_config or a data_dir hint")
            })
    }
}

/// Operation priority levels for storage operations.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize)]
pub enum OperationPriority {
    Low = 0,
    #[default]
    Medium = 1,
    High = 2,
    Critical = 3,
}

/// Performance tier hint for storage engines.
///
/// ## Purpose:
///
/// PerformanceTier provides hints to storage engines about data temperature,
/// enabling intelligent tiering decisions for optimal cost/performance balance.
///
/// ## Tiering Strategy:
///
/// - **Hot**: Memory/NVMe SSD, uncompressed or lightly compressed
/// - **Warm**: SSD with moderate compression (ZSTD level 3)
/// - **Cold**: HDD/Cloud with heavy compression (ZSTD level 9)
/// - **Archive**: Glacier/Archive with maximum compression (ZSTD level 19)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PerformanceTier {
    /// Hot data - keep in memory/SSD, optimize for latency
    /// Target: <1ms latency, highest cost
    Hot,

    /// Warm data - balance between latency and cost
    /// Target: <10ms latency, moderate cost
    Warm,

    /// Cold data - optimize for cost
    /// Target: <100ms latency, lowest cost
    Cold,

    /// Archive data - lowest cost, longest retrieval time
    /// Target: Minutes to hours, archival storage
    Archive,

    /// Default tier (system decides)
    #[default]
    Auto,
}

/// Storage engine strategy enumeration for polymorphic engine selection
/// (re-export).
///
/// `StorageEngineStrategy` and its `StorageFormatStrategy` alias have been
/// hoisted to the `proximadb-storage-ports` crate alongside the rest of the
/// engine-capability descriptor cluster. These thin re-exports preserve the
/// existing `crate::storage::traits::{StorageEngineStrategy, StorageFormatStrategy}`
/// paths so every caller resolves unchanged.
pub use proximadb_storage_ports::{StorageEngineStrategy, StorageFormatStrategy};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flush_parameters_default() {
        let params = FlushParameters::default();
        assert!(params.collection_id.is_none());
        assert!(!params.force);
        assert!(!params.synchronous);
    }

    #[test]
    fn flush_adapter_accepts_only_catalog_object_identity() {
        let numeric = FlushParameters {
            collection_id: Some("184467".to_string()),
            ..Default::default()
        };
        assert_eq!(numeric.get_collection_object_id().unwrap(), 184467);

        let alias = FlushParameters {
            collection_id: Some("customer-orders".to_string()),
            ..Default::default()
        };
        assert!(
            alias
                .get_collection_object_id()
                .unwrap_err()
                .to_string()
                .contains("decimal catalog object id")
        );
    }

    #[test]
    fn test_compaction_parameters_default() {
        let params = CompactionParameters::default();
        assert!(params.collection_id.is_none());
        assert!(!params.force);
        assert_eq!(params.priority, OperationPriority::Medium);
    }

    #[test]
    fn compaction_adapter_accepts_only_catalog_object_identity() {
        let numeric = CompactionParameters {
            collection_id: Some("184467".to_string()),
            ..Default::default()
        };
        assert_eq!(numeric.get_collection_object_id().unwrap(), 184467);

        let alias = CompactionParameters {
            collection_id: Some("customer-orders".to_string()),
            ..Default::default()
        };
        assert!(
            alias
                .get_collection_object_id()
                .unwrap_err()
                .to_string()
                .contains("decimal catalog object id")
        );
    }

    #[test]
    fn test_operation_priority_ordering() {
        assert!(OperationPriority::Critical > OperationPriority::High);
        assert!(OperationPriority::High > OperationPriority::Medium);
        assert!(OperationPriority::Medium > OperationPriority::Low);
    }

    #[test]
    fn test_performance_tier_default() {
        let tier = PerformanceTier::default();
        assert_eq!(tier, PerformanceTier::Auto);
    }
}
