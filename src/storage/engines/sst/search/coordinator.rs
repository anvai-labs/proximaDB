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

//! Search Coordinator
//!
//! Coordinates complex search operations and manages search strategy selection
//! for the SST engine. Provides intelligent routing between different search
//! approaches based on query characteristics.

use anyhow::Result;
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::core::search::bounded_queue::BoundedPriorityQueue;
use crate::core::search::results::OptimizedSearchRecord;
use crate::storage::engines::sst::SstEngine;
use crate::storage::traits::StorageQueryContext;

/// Search strategy enumeration
#[derive(Debug, Clone)]
pub enum SearchStrategy {
    /// Direct search through SSTable files
    Direct { reason: String, estimated_cost: f64 },
    /// Orchestrated search with advanced optimization
    Orchestrated {
        reason: String,
        estimated_cost: f64,
        use_indexes: bool,
    },
    /// Hybrid approach combining multiple strategies
    Hybrid {
        strategies: Vec<SearchStrategy>,
        estimated_cost: f64,
    },
}

/// Search coordinator for managing complex search operations
pub struct SearchCoordinator {
    engine: Arc<SstEngine>,
}

impl SearchCoordinator {
    /// Create a new search coordinator
    pub fn new(engine: Arc<SstEngine>) -> Self {
        Self { engine }
    }

    /// Coordinate a search operation with intelligent strategy selection
    pub async fn coordinate_search(
        &self,
        ctx: &StorageQueryContext,
    ) -> Result<Vec<OptimizedSearchRecord>> {
        debug!("🎯 SearchCoordinator: Starting search coordination");

        // Analyze query characteristics
        let strategy = self.select_search_strategy(ctx).await?;
        debug!("📋 Selected strategy: {:?}", strategy);

        // Execute search based on selected strategy
        let results = self.execute_search_strategy(ctx, strategy).await?;

        // Post-process results
        let optimized_results = self.post_process_results(results, ctx).await?;

        // Tier-migration hook: record access for every vector returned so
        // the policy engine sees heat for the collection. No-op when the
        // engine has no tiering integration attached (legacy path) or the
        // integration is disabled via config. Each result is recorded as
        // a `Read` access with an approximate byte-size of the vector;
        // collection-level patterns aggregate from these per-item events.
        if let Some(tiering) = self.engine.tiering_integration() {
            let collection_id = ctx.collection_id();
            let vec_bytes = ctx
                .query_vector()
                .map(|v| std::mem::size_of_val(v) as u64)
                .unwrap_or(0);
            for record in &optimized_results {
                tiering
                    .record_access(
                        collection_id,
                        &record.id,
                        crate::storage::tiering::tracker::AccessType::Read,
                        vec_bytes,
                    )
                    .await;
            }
        }

        info!("✅ SearchCoordinator: Search coordination completed successfully");
        Ok(optimized_results)
    }

    /// Select optimal search strategy based on query characteristics
    async fn select_search_strategy(&self, ctx: &StorageQueryContext) -> Result<SearchStrategy> {
        let collection_id = ctx.collection_id();
        let has_filters = ctx.search_params.filter_expression.is_some();
        let vector_dimension = ctx.query_vector().map_or(0, |v| v.len());

        debug!(
            "🔍 Analyzing query: collection={}, has_filters={}, dimensions={}",
            collection_id, has_filters, vector_dimension
        );

        // Simple heuristics for strategy selection
        let strategy = if has_filters && vector_dimension > 512 {
            SearchStrategy::Orchestrated {
                reason: "High-dimensional query with index support".to_string(),
                estimated_cost: 50.0,
                use_indexes: true,
            }
        } else if has_filters {
            SearchStrategy::Direct {
                reason: "Filtered query benefits from bloom filter pipeline".to_string(),
                estimated_cost: 100.0,
            }
        } else {
            SearchStrategy::Direct {
                reason: "Simple query, direct search is optimal".to_string(),
                estimated_cost: 75.0,
            }
        };

        info!("🎯 Selected search strategy: {:?}", strategy);
        Ok(strategy)
    }

    /// Execute search based on the selected strategy
    fn execute_search_strategy<'a>(
        &'a self,
        ctx: &'a StorageQueryContext,
        strategy: SearchStrategy,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Vec<OptimizedSearchRecord>>> + Send + 'a>,
    > {
        Box::pin(async move {
            match strategy {
                SearchStrategy::Direct { reason, .. } => {
                    info!("🔍 Executing direct search: {}", reason);
                    std::sync::Arc::clone(&self.engine)
                        .search_vectors_unified_arc(ctx)
                        .await
                }
                SearchStrategy::Orchestrated {
                    reason,
                    use_indexes,
                    ..
                } => {
                    info!(
                        "🎯 Executing orchestrated search: {}, use_indexes: {}",
                        reason, use_indexes
                    );
                    // For now, fall back to unified search since full orchestration is not yet implemented
                    warn!(
                        "🔄 Orchestrated search not fully implemented, falling back to unified search"
                    );
                    std::sync::Arc::clone(&self.engine)
                        .search_vectors_unified_arc(ctx)
                        .await
                }
                SearchStrategy::Hybrid { strategies, .. } => {
                    info!(
                        "🔀 Executing hybrid search with {} strategies",
                        strategies.len()
                    );
                    // Execute the first strategy for now
                    if let Some(first_strategy) = strategies.into_iter().next() {
                        self.execute_search_strategy(ctx, first_strategy).await
                    } else {
                        std::sync::Arc::clone(&self.engine)
                            .search_vectors_unified_arc(ctx)
                            .await
                    }
                }
            }
        })
    }

    /// Filter tombstones from search results
    ///
    /// Tombstones are identified by:
    /// - Empty vector (None or empty Vec) AND expires_at in the past (or 0)
    ///
    /// Deleted records are marked with empty vectors + expires_at in past during delete.
    /// These should be excluded from search results.
    fn filter_tombstones(&self, results: Vec<OptimizedSearchRecord>) -> Vec<OptimizedSearchRecord> {
        let original_count = results.len();
        let current_time_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        let filtered: Vec<OptimizedSearchRecord> = results
            .into_iter()
            .filter(|r| {
                // Tombstone check: empty vector + expires_at in past
                let is_empty_vector = r.vector.as_ref().is_none_or(|v| v.is_empty());
                let is_expired = r.expires_at.is_some_and(|e| e <= current_time_secs);
                let is_tombstone = is_empty_vector && is_expired;

                // Keep records that are NOT tombstones AND have valid vectors
                !is_tombstone && r.vector.as_ref().is_some_and(|v| !v.is_empty())
            })
            .collect();

        let filtered_count = original_count - filtered.len();
        if filtered_count > 0 {
            debug!(
                "🗑️ Tombstone filter: removed {} deleted records from {} total",
                filtered_count, original_count
            );
        }

        filtered
    }

    /// Post-process search results for optimization
    async fn post_process_results(
        &self,
        results: Vec<OptimizedSearchRecord>,
        ctx: &StorageQueryContext,
    ) -> Result<Vec<OptimizedSearchRecord>> {
        debug!("🔧 Post-processing {} search results", results.len());

        // Filter out tombstones (deleted records) first
        let results = self.filter_tombstones(results);
        debug!("📊 After tombstone filtering: {} results", results.len());

        // Use bounded priority queue for efficient top-k selection
        let k = ctx.top_k();
        let mut priority_queue = BoundedPriorityQueue::new(k);

        // Insert all results into bounded queue
        for result in results {
            priority_queue.try_insert(result);
        }

        // Get sorted results from bounded queue
        let mut results = priority_queue.into_sorted_vec();
        debug!(
            "📊 Selected top-{} results from bounded queue",
            results.len()
        );

        // Apply score normalization if needed
        if !results.is_empty() {
            let max_score = results
                .iter()
                .map(|r| r.score)
                .fold(f32::NEG_INFINITY, f32::max);

            if max_score > 0.0 {
                for result in &mut results {
                    result.score /= max_score;
                }
                debug!("📈 Normalized scores by max score: {}", max_score);
            }
        }

        debug!(
            "✅ Post-processing completed, returning {} results",
            results.len()
        );
        Ok(results)
    }

    /// Estimate search cost for a given strategy
    pub async fn estimate_search_cost(
        &self,
        _ctx: &StorageQueryContext,
        strategy: &SearchStrategy,
    ) -> Result<f64> {
        match strategy {
            SearchStrategy::Direct { estimated_cost, .. } => Ok(*estimated_cost),
            SearchStrategy::Orchestrated { estimated_cost, .. } => Ok(*estimated_cost),
            SearchStrategy::Hybrid { estimated_cost, .. } => Ok(*estimated_cost),
        }
    }

    /// Get search statistics for monitoring
    pub async fn get_search_statistics(&self) -> Result<SearchStatistics> {
        Ok(SearchStatistics {
            total_searches: 0, // Would be tracked in a real implementation
            avg_latency_ms: 0.0,
            cache_hit_rate: 0.0,
            strategy_distribution: std::collections::HashMap::new(),
        })
    }
}

/// Search statistics for monitoring and optimization
#[derive(Debug, Clone)]
pub struct SearchStatistics {
    pub total_searches: u64,
    pub avg_latency_ms: f64,
    pub cache_hit_rate: f64,
    pub strategy_distribution: std::collections::HashMap<String, u64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::proximadb_v1::DistanceMetric;
    use crate::query::query_optimizer::SearchParams;
    use crate::storage::engines::sst::SstConfig;
    use crate::storage::persistence::filesystem::FilesystemFactory;
    use proximadb_distance_kernel::engine::UnifiedDistanceCompute;

    #[tokio::test]
    async fn test_search_strategy_selection() {
        let engine = create_test_engine().await;
        let coordinator = SearchCoordinator::new(Arc::new(engine));

        // Test direct strategy selection
        let ctx = create_test_context(false, false);
        let strategy = coordinator.select_search_strategy(&ctx).await.unwrap();

        match strategy {
            SearchStrategy::Direct { .. } => {
                // Expected for simple query
            }
            _ => panic!("Expected Direct strategy for simple query"),
        }
    }

    #[tokio::test]
    async fn test_cost_estimation() {
        let engine = create_test_engine().await;
        let coordinator = SearchCoordinator::new(Arc::new(engine));

        let strategy = SearchStrategy::Direct {
            reason: "Test".to_string(),
            estimated_cost: 100.0,
        };

        let ctx = create_test_context(false, false);
        let cost = coordinator
            .estimate_search_cost(&ctx, &strategy)
            .await
            .unwrap();
        assert_eq!(cost, 100.0);
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

    fn create_test_context(_use_indexes: bool, _has_quantization: bool) -> StorageQueryContext {
        let search_params = Arc::new(SearchParams {
            query_vectors: None,
            vector: Some(vec![1.0, 2.0, 3.0]),
            top_k: Some(10),
            distance_metric: Some(DistanceMetric::Cosine),
            filter_expression: None,
            filters: None,
            accuracy_threshold: Some(0.95),
            include_expired: Some(false),
            timeout_ms: Some(5000),
            enable_two_stage: Some(true),
            quantization_hint: None,
            enable_clustering_hint: Some(true),
            enable_metadata_filtering_hint: Some(true),
            custom_hints: None,
            requires_ordering: None,
            runtime_hints: None,
            enable_progressive_search: Some(false),
            progressive_recalls: None,
            enable_vectorized_execution: Some(false),
            enable_parallel_morsels: Some(false),
            enable_pipeline_execution: Some(false),
            search_mode: crate::core::search::SearchMode::default(),
            block_prune: crate::core::search::BlockPruneConfig::default(),
            text_query: None,
            hybrid_mode: crate::core::search::HybridSearchMode::default(),
            vector_weight: None,
            freshness_mode: None,
        });

        let collection = Arc::new(crate::proto::proximadb_v1::Collection {
            storage_class_status: None,
            id: "test_collection".to_string(),
            config: Some(crate::proto::proximadb_v1::CollectionConfig {
                name: "test_collection".to_string(),
                dimension: 128,
                distance_metric: Some(crate::proto::proximadb_v1::DistanceMetric::Cosine as i32),
                storage_engine: Some(crate::proto::proximadb_v1::StorageEngine::Sst as i32),
                ..Default::default()
            }),
            stats: Some(crate::proto::proximadb_v1::CollectionStats::default()),
            created_at: 0,
            updated_at: 0,
            storage_assignment: None,
        });

        StorageQueryContext::new(search_params, collection)
    }
}
