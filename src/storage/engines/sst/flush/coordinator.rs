/*
 * Copyright 2025 Vijaykumar Singh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! Flush Coordinator
//!
//! Coordinates flush operations across multiple collections and manages
//! the overall flush workflow for the SST engine.

use anyhow::Result;
use std::sync::Arc;
use tracing::{debug, info};

use crate::storage::engines::sst::SstEngine;
use crate::storage::traits::{FlushParameters, FlushResult};

/// Flush coordinator for managing complex flush operations
pub struct FlushCoordinator {
    engine: Arc<SstEngine>,
}

impl FlushCoordinator {
    /// Create a new flush coordinator
    pub fn new(engine: Arc<SstEngine>) -> Self {
        Self { engine }
    }

    /// Coordinate a flush operation
    pub async fn coordinate_flush(&self, params: FlushParameters) -> Result<FlushResult> {
        debug!("🔄 FlushCoordinator: Starting coordinated flush");

        // Pre-flush validation
        self.validate_flush_parameters(&params)?;

        // Tier-migration hook: when tiering is enabled, decide the target
        // tier for this flush BEFORE the bytes land. The decision is
        // currently advisory — physical-path routing is deferred — but
        // the call populates the tiering engine's state so subsequent
        // `evaluate_collection` calls can reason about the new segment.
        let estimated_bytes: u64 = params.estimated_size as u64;
        if let (Some(tiering), Some(coll)) = (
            self.engine.tiering_integration(),
            params.collection_id.as_ref(),
        ) {
            let tier = tiering
                .determine_flush_tier(coll.as_str(), estimated_bytes)
                .await;
            debug!(
                "🪜 FlushCoordinator: tiering decided target tier {:?} for collection {} ({}B)",
                tier, coll, estimated_bytes
            );
        }

        // Execute the flush
        let result = self.engine.flush_implementation(&params).await?;

        // Post-flush operations
        self.post_flush_operations(&params, &result).await?;

        info!("✅ FlushCoordinator: Flush coordination completed successfully");
        Ok(result)
    }

    /// Validate flush parameters
    fn validate_flush_parameters(&self, params: &FlushParameters) -> Result<()> {
        if params.vector_records.is_empty() {
            return Err(anyhow::anyhow!("No vectors provided for flush"));
        }

        if params.collection_id.is_none() {
            return Err(anyhow::anyhow!("Collection ID is required"));
        }

        if params.collection_config.is_none() {
            return Err(anyhow::anyhow!("Collection configuration is required"));
        }

        debug!("✅ FlushCoordinator: Parameters validation passed");
        Ok(())
    }

    /// Post-flush operations
    async fn post_flush_operations(
        &self,
        params: &FlushParameters,
        result: &FlushResult,
    ) -> Result<()> {
        debug!("🔧 FlushCoordinator: Executing post-flush operations");

        // Log flush statistics
        if let (Some(entries), Some(bytes)) = (result.entries_flushed, result.bytes_written) {
            info!("📊 Flush Stats: {} entries, {} bytes", entries, bytes);
        }

        // Tier-migration hook: record the write as an access event so the
        // policy engine sees newly-flushed data as freshly touched.
        // Without this signal, just-flushed segments look as cold as
        // never-accessed ones and would be demoted immediately by an
        // age-based policy.
        if let (Some(tiering), Some(coll)) = (
            self.engine.tiering_integration(),
            params.collection_id.as_ref(),
        ) {
            let bytes_written = result.bytes_written.unwrap_or(0);
            tiering
                .record_access(
                    coll.as_str(),
                    coll.as_str(), // collection-level write event; per-item events come from search path
                    crate::storage::tiering::tracker::AccessType::Write,
                    bytes_written,
                )
                .await;
        }

        // Trigger compaction if needed
        if result.compaction_triggered {
            info!("🔄 Triggering background compaction");
            // Compaction would be triggered here

            // Tier-migration hook: when a compaction is queued, evaluate
            // the collection for pending tier migrations. The compaction
            // is the natural moment to re-tier because the segment-set
            // about to be merged represents the current physical layout
            // — any access-driven tier change should be reflected before
            // the merge rewrites the segments.
            //
            // The returned migration tasks are NOT executed here (data
            // movement is still deferred — see `src/storage/tiering/mod.rs`
            // remaining-integration item #4). What this call does today
            // is feed the policy engine's accounting so operator
            // dashboards see pending migrations and so the next
            // `evaluate_all()` doesn't re-propose the same migration.
            if let (Some(tiering), Some(coll)) = (
                self.engine.tiering_integration(),
                params.collection_id.as_ref(),
            ) {
                match tiering.evaluate_collection(coll.as_str()).await {
                    Ok(tasks) if !tasks.is_empty() => {
                        info!(
                            "🪜 FlushCoordinator: tiering proposed {} migration task(s) for collection {} (execution deferred)",
                            tasks.len(),
                            coll
                        );
                    }
                    Ok(_) => {
                        debug!(
                            "🪜 FlushCoordinator: no migration tasks proposed for collection {}",
                            coll
                        );
                    }
                    Err(e) => {
                        // Tiering evaluation failure should not block the
                        // flush — it's advisory, not on the durability
                        // path. Log and continue.
                        debug!(
                            "🪜 FlushCoordinator: tiering evaluation failed for {}: {} (advisory, flush succeeded)",
                            coll, e
                        );
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::engines::sst::SstConfig;
    use crate::storage::persistence::filesystem::FilesystemFactory;
    use proximadb_distance_kernel::engine::UnifiedDistanceCompute;

    #[tokio::test]
    async fn test_flush_coordinator_validation() {
        let engine = create_test_engine().await;
        let coordinator = FlushCoordinator::new(Arc::new(engine));

        // Test with empty vectors - should fail
        let params = FlushParameters {
            vector_records: vec![],
            batch_ids: vec![],
            collection_id: Some("test".to_string()),
            collection_config: None,
            force: false,
            synchronous: true,
            hints: std::collections::HashMap::new(),
            timeout_ms: None,
            trigger_compaction: false,
            estimated_size: 0,
            collection_properties: std::collections::HashMap::new(),
        };

        assert!(coordinator.validate_flush_parameters(&params).is_err());
    }

    /// Regression: SstEngine MUST NOT emit Vector Object Economy
    /// directory updates by default. The conservative-by-design
    /// constructor-injection model means a freshly-constructed engine
    /// is silent until `with_directory_cache(...)` is called by
    /// SharedServices. Flipping this default would silently turn on
    /// directory writes for every test and bench harness.
    #[tokio::test]
    async fn sst_engine_default_does_not_emit_directory() {
        use crate::storage::engines::sst::object_economy_directory::VectorObjectEconomyDirectoryCache;
        use std::sync::Arc;

        // Default construction: no directory cache configured.
        let engine = create_test_engine().await;
        assert!(
            !engine.directory_cache_configured(),
            "freshly-constructed SstEngine must NOT emit directory updates"
        );

        // Opt in via builder — accessor flips.
        let cache = Arc::new(VectorObjectEconomyDirectoryCache::new());
        let engine_opted_in = create_test_engine().await.with_directory_cache(cache);
        assert!(engine_opted_in.directory_cache_configured());
    }

    async fn create_test_engine() -> SstEngine {
        let config = SstConfig::default();
        let filesystem_config =
            crate::storage::persistence::filesystem::FilesystemConfig::default();
        let filesystem = Arc::new(FilesystemFactory::create(filesystem_config).await.unwrap());
        let distance_compute = Arc::new(UnifiedDistanceCompute::default());

        SstEngine::new_with_config(config, filesystem, distance_compute)
            .await
            .unwrap()
    }
}
