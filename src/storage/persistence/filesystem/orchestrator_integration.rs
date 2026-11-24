//! Integration between UnifiedCachingFilesystem and CrossCacheOrchestrator
//!
//! This module provides the glue between the filesystem-level caching
//! and the application-level cache orchestration.

use std::sync::Arc;

use async_trait::async_trait;
use tracing::{debug, trace};

use crate::storage::cache::backend::CacheTier;
use crate::storage::cache::orchestrator::{
    CacheType as OrchestratorCacheType, CrossCacheOrchestrator,
};
use crate::storage::persistence::filesystem::caching_filesystem::{
    CacheType as FilesystemCacheType, UnifiedCachingFilesystem,
};

/// Extension trait for UnifiedCachingFilesystem to integrate with CrossCacheOrchestrator
#[async_trait]
pub trait OrchestratorIntegration {
    /// Report cache access to the orchestrator
    async fn report_cache_access(
        &self,
        orchestrator: &CrossCacheOrchestrator,
        key: &str,
        cache_type: FilesystemCacheType,
        hit: bool,
    );

    /// Get prefetch suggestions from orchestrator
    async fn get_prefetch_suggestions(
        &self,
        orchestrator: &CrossCacheOrchestrator,
        key: &str,
    ) -> Vec<String>;

    /// Sync cache metrics with orchestrator
    async fn sync_metrics(&self, orchestrator: &CrossCacheOrchestrator);
}

#[async_trait]
impl OrchestratorIntegration for UnifiedCachingFilesystem {
    async fn report_cache_access(
        &self,
        orchestrator: &CrossCacheOrchestrator,
        key: &str,
        cache_type: FilesystemCacheType,
        hit: bool,
    ) {
        // Map filesystem cache type to orchestrator cache type
        let orchestrator_type = match cache_type {
            FilesystemCacheType::Metadata => OrchestratorCacheType::Metadata,
            FilesystemCacheType::Disk => OrchestratorCacheType::VectorData,
            FilesystemCacheType::Memory => OrchestratorCacheType::QueryResult,
        };

        // Track the access asynchronously (clone the type since it doesn't implement Copy)
        orchestrator.track_access_async(key.to_string(), orchestrator_type.clone());

        // Update metrics - map cache type to tier
        let metrics = orchestrator.metrics();
        if hit {
            // Map orchestrator cache type to cache tier
            let tier = match orchestrator_type {
                OrchestratorCacheType::Metadata => CacheTier::L1, // Metadata is hot, L1
                OrchestratorCacheType::QueryResult => CacheTier::L1, // Query results are hot
                OrchestratorCacheType::VectorData => CacheTier::L2, // Vector data in L2
                OrchestratorCacheType::FilterBitmap => CacheTier::L1, // Filter bitmaps are frequently accessed
                OrchestratorCacheType::IndexStructure => CacheTier::L2, // Index structures in L2
                OrchestratorCacheType::QueryPlan => CacheTier::L1,    // Query plans are hot
                OrchestratorCacheType::EntityHeader => CacheTier::L2, // Entity headers in L2
                OrchestratorCacheType::EmbeddingCatalog => CacheTier::L2, // Embedding catalog in L2
                OrchestratorCacheType::GraphNode => CacheTier::L2,    // Graph nodes in L2
                OrchestratorCacheType::GraphEdge => CacheTier::L2,    // Graph edges in L2
                OrchestratorCacheType::GraphAdjacency => CacheTier::L3, // Adjacency lists in L3
                OrchestratorCacheType::GraphPropertyIndex => CacheTier::L3, // Property indexes in L3
                OrchestratorCacheType::DistanceTable => CacheTier::L2,      // Distance tables in L2
                OrchestratorCacheType::MetricsSnapshot => CacheTier::L3, // Metrics snapshots in L3
                OrchestratorCacheType::Quantization => CacheTier::L2, // Quantization codebooks in L2
            };
            metrics.record_hit(tier);
        } else {
            metrics.record_miss();
        }

        trace!(
            "Reported cache {} for key {} to orchestrator",
            if hit { "hit" } else { "miss" },
            key
        );
    }

    async fn get_prefetch_suggestions(
        &self,
        orchestrator: &CrossCacheOrchestrator,
        key: &str,
    ) -> Vec<String> {
        // Get correlated items from pattern tracker
        let _pattern_tracker = orchestrator.pattern_tracker();

        // For now, return empty as the pattern tracker doesn't expose correlation methods directly
        // In a real implementation, we would need to enhance the pattern tracker API
        let suggestions = vec![];

        debug!(
            "Got {} prefetch suggestions for key {} from orchestrator",
            suggestions.len(),
            key
        );

        suggestions
    }

    async fn sync_metrics(&self, orchestrator: &CrossCacheOrchestrator) {
        // Get filesystem cache metrics
        let fs_metrics = self.get_metrics().await;

        // Record metrics in orchestrator's metrics system
        let metrics = orchestrator.metrics();

        // Report current cache sizes and hit rates
        for _ in 0..fs_metrics.total_hits {
            metrics.record_hit(CacheTier::L1); // Use L1 for metadata cache hits
        }
        for _ in 0..fs_metrics.total_misses {
            metrics.record_miss();
        }

        trace!("Synced filesystem cache metrics with orchestrator");
    }
}

/// Builder extension to create UnifiedCachingFilesystem with orchestrator integration
pub struct OrchestratorAwareFilesystemBuilder {
    orchestrator: Option<Arc<CrossCacheOrchestrator>>,
}

impl OrchestratorAwareFilesystemBuilder {
    pub fn new() -> Self {
        Self { orchestrator: None }
    }

    pub fn with_orchestrator(mut self, orchestrator: Arc<CrossCacheOrchestrator>) -> Self {
        self.orchestrator = Some(orchestrator);
        self
    }

    pub async fn build_filesystem(
        self,
        underlying_fs: Arc<dyn crate::storage::persistence::filesystem::FileSystem>,
        collection_id: String,
        engine_type: String,
    ) -> Arc<UnifiedCachingFilesystem> {
        let filesystem = Arc::new(UnifiedCachingFilesystem::new(
            underlying_fs,
            collection_id.clone(),
            engine_type.clone(),
        ));

        // If orchestrator is provided, register the filesystem with it
        if let Some(orchestrator) = self.orchestrator {
            debug!(
                "Registered UnifiedCachingFilesystem for collection {} with orchestrator",
                collection_id
            );

            // Start background metric sync task
            let fs = filesystem.clone();
            let orch = orchestrator.clone();
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
                loop {
                    interval.tick().await;
                    fs.sync_metrics(&orch).await;
                }
            });
        }

        filesystem
    }
}

impl Default for OrchestratorAwareFilesystemBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper to create engine-aware cache key
pub fn create_cache_key(collection_id: &str, engine_type: &str, path: &str) -> String {
    format!("{}:{}:{}", path, collection_id, engine_type)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_generation() {
        let key = create_cache_key("my_collection", "nova", "/data/file.parquet");
        assert_eq!(key, "/data/file.parquet:my_collection:nova");
    }
}
