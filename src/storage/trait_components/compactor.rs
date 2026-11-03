//! Storage Engine Compactor Trait
//!
//! Defines compaction operations for storage engines. Compaction merges
//! smaller files into larger ones, removes tombstones, and optimizes
//! storage layout for better read performance.

use anyhow::Result;
use async_trait::async_trait;
use chrono::Utc;

use crate::proto::proximadb_v1::Collection;
use crate::storage::traits::{CompactionParameters, CompactionResult};

use super::StorageIdentity;

/// Compaction operations for storage engines
///
/// Compaction is a background maintenance operation that:
/// - Merges small files into larger ones (reduces file count)
/// - Removes tombstones and expired data
/// - Optimizes data layout for read performance
/// - Reclaims storage space
///
/// # Engine-Specific Strategies
///
/// - **SST**: Leveled compaction (LSM-tree style)
/// - **HELIX**: Liquid clustering with PCA + Hilbert curves
/// - **VIPER**: Basic Parquet merge
/// - **NOVA**: Deduplication + ID sorting
/// - **RAPTOR**: Tier-aware compaction
#[async_trait]
pub trait StorageCompactor: StorageIdentity + Send + Sync {
    /// Core compaction operation - engine-specific implementation (required)
    ///
    /// Engines implement their specific compaction strategy here.
    async fn do_compact(&self, params: &CompactionParameters) -> Result<CompactionResult>;

    /// High-level compaction operation with common pre/post processing
    ///
    /// Wraps `do_compact` with:
    /// - Parameter validation
    /// - Timing and metrics
    /// - Logging
    async fn compact(&self, params: CompactionParameters) -> Result<CompactionResult> {
        let start_time = std::time::Instant::now();

        // Common pre-compaction validation
        self.validate_compaction_parameters(&params).await?;

        // Log operation start
        tracing::info!(
            "Starting {} compaction for collection: {:?} (force: {}, priority: {:?})",
            self.engine_name(),
            params.collection_id,
            params.force,
            params.priority
        );

        // Delegate to engine-specific implementation
        let mut result = self.do_compact(&params).await?;

        // Common post-compaction processing
        result.duration_ms = Some(start_time.elapsed().as_millis() as u64);
        result.completed_at = Utc::now();

        // Log operation completion
        tracing::info!(
            "{} compaction completed: {} entries processed, {} removed in {}ms",
            self.engine_name(),
            result.entries_processed.unwrap_or(0),
            result.entries_removed.unwrap_or(0),
            result.duration_ms.unwrap_or(0)
        );

        Ok(result)
    }

    /// Compact a specific collection's data
    ///
    /// Convenience method that creates CompactionParameters and delegates to `do_compact`.
    async fn compact_collection(
        &self,
        collection_id: &str,
        collection_config: Option<&Collection>,
    ) -> Result<CompactionResult> {
        let params = CompactionParameters {
            full: false,
            collection_id: Some(collection_id.to_string()),
            collection_config: collection_config.cloned(),
            force: false,
            synchronous: true,
            ..Default::default()
        };

        self.do_compact(&params).await
    }

    /// Check if compaction is needed with engine-specific heuristics
    async fn should_compact(&self, _collection_id: Option<&str>) -> Result<bool> {
        // Default: no automatic compaction needed
        Ok(false)
    }

    /// Validate compaction parameters
    async fn validate_compaction_parameters(&self, params: &CompactionParameters) -> Result<()> {
        if params.collection_id.is_some() && !self.supports_collection_level_operations() {
            tracing::warn!(
                "{} engine doesn't support collection-level compaction, performing global compaction",
                self.engine_name()
            );
        }

        if let Some(timeout) = params.timeout_ms
            && timeout == 0
        {
            return Err(anyhow::anyhow!("Compaction timeout cannot be zero"));
        }

        Ok(())
    }

    // =========================================================================
    // HELPER METHODS
    // =========================================================================

    /// Extract collection ID from compaction parameters
    fn get_collection_id_from_compaction_params(
        &self,
        params: &CompactionParameters,
    ) -> Result<String> {
        params.get_collection_id()
    }

    /// Construct data directory path from compaction parameters
    fn get_data_dir_from_compaction_params(&self, params: &CompactionParameters) -> Result<String> {
        if let Some(ref collection_config) = params.collection_config {
            let collection_id = &collection_config.id;
            if let Some(ref storage_assignment) = collection_config.storage_assignment {
                let base_location = &storage_assignment.base_location;
                Ok(format!("{}/{}/data", base_location, collection_id))
            } else {
                Err(anyhow::anyhow!(
                    "No storage assignment found in collection config for '{}'",
                    collection_id
                ))
            }
        } else {
            params.get_data_dir()
        }
    }
}
