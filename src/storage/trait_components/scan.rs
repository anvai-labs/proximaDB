//! Storage Engine Scan Trait
//!
//! Defines scan operations for storage engines. Scans provide
//! iterator-based access to storage data with various optimization
//! capabilities.

use anyhow::Result;
use async_trait::async_trait;

use crate::proto::proximadb_v1::Collection;
use crate::storage::scan_strategy::{ScanCapabilities, ScanIterator, ScanStrategy};
use crate::storage::traits::StorageFormatStrategy;

/// Scan operations for storage engines
///
/// This trait provides iterator-based access to storage data with
/// engine-specific optimizations:
///
/// - **SST**: Bloom filters, range scans, block cache
/// - **VIPER**: Predicate pushdown, column projection, row group pruning
/// - **NOVA**: Progressive quantization, zone maps, streaming
/// - **RAPTOR**: Tier-aware scanning, consolidated reading
#[async_trait]
pub trait StorageScan: Send + Sync {
    /// Get the storage strategy for this engine
    fn strategy(&self) -> StorageFormatStrategy;

    /// Create a scan iterator based on the unified scan strategy pattern
    ///
    /// This follows the successful pattern from RAPTOR's scan_vectors_with_strategy
    /// and is implemented differently by each engine:
    ///
    /// - **SST**: Uses modular block readers with bloom filters
    /// - **VIPER**: Uses columnar predicate pushdown
    /// - **NOVA**: Uses progressive quantization stages
    /// - **RAPTOR**: Uses tier-aware consolidated reading
    ///
    /// Default implementation returns an error - engines should override.
    async fn create_scan(
        &self,
        _collection_id: &str,
        _strategy: ScanStrategy,
        _collection_config: Option<&Collection>,
    ) -> Result<Box<dyn ScanIterator>> {
        Err(anyhow::anyhow!(
            "Scan not implemented for this engine. Use search_vectors_unified for now."
        ))
    }

    /// Get scan capabilities for this engine
    ///
    /// Reports what optimizations this engine supports for scans.
    fn scan_capabilities(&self) -> ScanCapabilities {
        match self.strategy() {
            StorageFormatStrategy::Sst => ScanCapabilities {
                supports_bloom_filters: true,
                supports_block_cache: true,
                supports_range_scans: true,
                supports_index_scans: true,
                ..Default::default()
            },
            StorageFormatStrategy::Viper => ScanCapabilities {
                supports_predicate_pushdown: true,
                supports_column_projection: true,
                supports_row_group_pruning: true,
                supports_parallel_column_evaluation: true,
                supports_zone_maps: true,
                supports_streaming: true,
                ..Default::default()
            },
            StorageFormatStrategy::Nova => ScanCapabilities {
                supports_predicate_pushdown: true,
                supports_column_projection: true,
                supports_row_group_pruning: true,
                supports_parallel_column_evaluation: true,
                supports_progressive_quantization: true,
                supports_zone_maps: true,
                supports_streaming: true,
                ..Default::default()
            },
            StorageFormatStrategy::Raptor => ScanCapabilities {
                supports_predicate_pushdown: true,
                supports_column_projection: true,
                supports_row_group_pruning: true,
                supports_bloom_filters: true,
                supports_streaming: true,
                supports_tier_aware_scanning: true,
                supports_consolidated_reading: true,
                ..Default::default()
            },
            StorageFormatStrategy::Swift => ScanCapabilities {
                supports_bloom_filters: true,
                supports_block_cache: true,
                supports_range_scans: true,
                supports_index_scans: true,
                supports_tier_aware_scanning: true,
                ..Default::default()
            },
            StorageFormatStrategy::Helix => ScanCapabilities {
                supports_row_group_pruning: true,
                supports_block_cache: true,
                ..Default::default()
            },
            _ => ScanCapabilities::default(),
        }
    }
}
