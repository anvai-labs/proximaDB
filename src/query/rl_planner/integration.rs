//! RL Planner Integration with UnifiedQueryOptimizer
//!
//! Provides hooks to integrate the RL planner into the query execution pipeline
//! without modifying the core optimizer significantly.

use std::sync::Arc;

use tokio::sync::RwLock;
use tracing::{debug, info, trace};

use super::{
    ExecutionAction, ExecutionLog, ObjectEconomyFeatures, PlannerState, RLPlanner, RLPlannerConfig,
    StageLog, action::QuantizationStage,
};
use crate::query::query_optimizer::{
    ExecutionStep, Index, QueryOptimizerProgressiveStage, SearchAlgorithm, SearchExecutionMethod,
    UnifiedExecutionPlan, UnifiedQueryContext,
};
use crate::storage::engine_capabilities::{
    EngineCapabilities, SearchIndexType, SearchQuantizationLevel, StorageEngine,
};

/// Wrapper that adds RL planning capabilities to query optimization
pub struct RLPlannerIntegration {
    /// The RL planner instance
    planner: Arc<RwLock<RLPlanner>>,
    /// Configuration
    config: RLPlannerConfig,
}

impl RLPlannerIntegration {
    /// Create new RL planner integration
    pub fn new(config: RLPlannerConfig) -> Self {
        Self {
            planner: Arc::new(RwLock::new(RLPlanner::new(config.clone()))),
            config,
        }
    }

    /// Extract object economy features from query context
    ///
    /// Uses VectorObjectEconomyDirectory if available to extract block-level
    /// statistics for cost-aware routing. Returns disabled features if
    /// object economy directory is not available.
    fn extract_object_economy_features(
        &self,
        _context: &UnifiedQueryContext<'_>,
    ) -> ObjectEconomyFeatures {
        // TODO: Extract from context.object_economy_directory when available
        // For now, return disabled features
        // This will be wired in Task 5 when we load the directory at query planning time
        ObjectEconomyFeatures::disabled()
    }

    /// Extract state from query context
    ///
    /// Uses EngineCapabilities to determine available indexes and quantization levels
    /// based on the storage engine, rather than relying solely on collection config.
    pub fn extract_state(&self, context: &UnifiedQueryContext<'_>) -> PlannerState {
        let storage_engine = self.infer_storage_engine(context);
        let has_filter = context.filter_params.is_some();
        let _filter_selectivity = if has_filter { 0.3 } else { 1.0 }; // Conservative estimate

        // Use EngineCapabilities to get supported indexes and quantization
        let capabilities_engine = self.to_capabilities_engine(storage_engine);
        let available_indexes = self.get_indexes_from_capabilities(capabilities_engine, context);
        let available_quantization = self.get_quantization_from_capabilities(capabilities_engine);

        // Extract object economy features
        let object_economy = self.extract_object_economy_features(context);

        PlannerState::builder()
            .query_dimension(
                context
                    .collection
                    .config
                    .as_ref()
                    .map_or(128, |c| c.dimension),
            )
            .top_k(context.search_params.and_then(|p| p.top_k).unwrap_or(10) as u32)
            .collection_size(context.total_vectors as u64)
            .storage_engine(storage_engine)
            .available_indexes(available_indexes)
            .available_quantization(available_quantization)
            .object_economy(object_economy)
            .build()
    }

    /// Convert RL planner's StorageEngineType to EngineCapabilities' StorageEngine
    fn to_capabilities_engine(&self, engine: super::state::StorageEngineType) -> StorageEngine {
        use super::state::StorageEngineType;
        match engine {
            StorageEngineType::SST => StorageEngine::Sst,
            StorageEngineType::HELIX => StorageEngine::Helix,
            StorageEngineType::VIPER => StorageEngine::Viper,
            StorageEngineType::SWIFT => StorageEngine::Swift,
            StorageEngineType::NOVA => StorageEngine::Nova,
            StorageEngineType::RAPTOR => StorageEngine::Raptor,
            // TST (and any unmapped engine) maps to SST capabilities as the
            // safest baseline, consistent with the planner's SST default.
            _ => StorageEngine::Sst,
        }
    }

    /// Get available indexes from EngineCapabilities, merged with collection config
    fn get_indexes_from_capabilities(
        &self,
        engine: StorageEngine,
        context: &UnifiedQueryContext<'_>,
    ) -> Vec<super::state::IndexType> {
        use super::state::IndexType;

        // Start with engine-supported indexes from EngineCapabilities
        let engine_indexes = EngineCapabilities::get_supported_index_types(engine);
        let mut indexes = Vec::new();

        // Map SearchIndexType to state::IndexType
        for idx in engine_indexes {
            let mapped = match idx {
                SearchIndexType::Flat => Some(IndexType::Flat),
                SearchIndexType::HNSW => Some(IndexType::HNSW),
                SearchIndexType::IVF => Some(IndexType::IVF),
                SearchIndexType::LSH => Some(IndexType::LSH),
                SearchIndexType::PQ => Some(IndexType::PQ),
                // Engine-specific indexes map to closest equivalent
                SearchIndexType::HilbertCurve | SearchIndexType::AdaCurve => None,
                SearchIndexType::ZoneMap | SearchIndexType::AdaptiveMatrix => None,
            };
            if let Some(idx_type) = mapped
                && !indexes.contains(&idx_type)
            {
                indexes.push(idx_type);
            }
        }

        // Also check collection config for explicitly enabled indexes
        if let Some(config) = &context.collection.config {
            for index_config in &config.index_configs {
                if index_config.enabled.unwrap_or(true) {
                    use crate::proto::proximadb_v1::IndexingAlgorithm;
                    let idx_type = match index_config.algorithm() {
                        IndexingAlgorithm::Hnsw => Some(IndexType::HNSW),
                        IndexingAlgorithm::Ivf => Some(IndexType::IVF),
                        IndexingAlgorithm::Lsh => Some(IndexType::LSH),
                        IndexingAlgorithm::Annoy => Some(IndexType::Annoy),
                        IndexingAlgorithm::Pq => Some(IndexType::PQ),
                        _ => None,
                    };
                    if let Some(it) = idx_type
                        && !indexes.contains(&it)
                    {
                        indexes.push(it);
                    }
                }
            }
        }

        // Flat is always available
        if !indexes.contains(&IndexType::Flat) {
            indexes.insert(0, IndexType::Flat);
        }

        indexes
    }

    /// Get available quantization levels from EngineCapabilities
    fn get_quantization_from_capabilities(
        &self,
        engine: StorageEngine,
    ) -> Vec<super::state::QuantizationLevel> {
        use super::state::QuantizationLevel;

        let engine_quant = EngineCapabilities::get_supported_quantization_levels(engine);
        let mut levels = Vec::new();

        for q in engine_quant {
            let mapped = match q {
                SearchQuantizationLevel::FP32 => QuantizationLevel::None,
                SearchQuantizationLevel::INT8 => QuantizationLevel::INT8,
                SearchQuantizationLevel::Binary => QuantizationLevel::Binary,
                SearchQuantizationLevel::PQ4 => QuantizationLevel::PQ4,
                SearchQuantizationLevel::PQ8 => QuantizationLevel::PQ8,
            };
            if !levels.contains(&mapped) {
                levels.push(mapped);
            }
        }

        // Ensure None (FP32) is always available
        if !levels.contains(&QuantizationLevel::None) {
            levels.insert(0, QuantizationLevel::None);
        }

        levels
    }

    /// Infer storage engine from collection config
    fn infer_storage_engine(
        &self,
        context: &UnifiedQueryContext<'_>,
    ) -> super::state::StorageEngineType {
        use super::state::StorageEngineType;
        use crate::proto::proximadb_v1::StorageEngine;

        // Try to infer from collection config
        if let Some(config) = &context.collection.config {
            return match config.storage_engine() {
                StorageEngine::Sst => StorageEngineType::SST,
                StorageEngine::Helix => StorageEngineType::HELIX,
                StorageEngine::Viper => StorageEngineType::VIPER,
                StorageEngine::Swift => StorageEngineType::SWIFT,
                StorageEngine::Nova => StorageEngineType::NOVA,
                StorageEngine::Raptor => StorageEngineType::RAPTOR,
                _ => StorageEngineType::SST,
            };
        }

        StorageEngineType::SST // Default
    }

    /// Select action based on current state
    /// Deterministic exploitation: returns the arm with the highest expected value (α/(α+β)).
    /// This is the hot-path method; Thompson Sampling exploration is excluded here.
    pub async fn exploit_best_action(&self, state: &PlannerState) -> ExecutionAction {
        let planner = self.planner.read().await;
        planner.exploit_best_action(state).await
    }

    pub async fn select_action(&self, state: &PlannerState) -> ExecutionAction {
        let planner = self.planner.read().await;
        planner.select_action(state).await
    }

    /// Convert RL action to execution plan modifications
    pub fn apply_action_to_plan(&self, action: &ExecutionAction, plan: &mut UnifiedExecutionPlan) {
        // Modify execution steps based on RL action
        for step in &mut plan.execution_steps {
            if let ExecutionStep::VectorSearch {
                execution_method,
                candidates,
                ..
            } = step
            {
                // Apply index strategy from action
                if let Some(ref strategy) = action.index_strategy {
                    *execution_method = match strategy {
                        super::action::IndexStrategy::HNSW { .. } => {
                            SearchExecutionMethod::IndexBased {
                                index_type: Index::HNSW,
                            }
                        }
                        super::action::IndexStrategy::IVF { .. } => {
                            SearchExecutionMethod::IndexBased {
                                index_type: Index::IVF,
                            }
                        }
                        super::action::IndexStrategy::LSH { .. } => {
                            SearchExecutionMethod::IndexBased {
                                index_type: Index::LSH,
                            }
                        }
                        super::action::IndexStrategy::DirectScan => {
                            SearchExecutionMethod::DirectFP32
                        }
                        _ => execution_method.clone(),
                    };
                }

                // Apply search mode
                if let super::action::SearchModeAction::Approximate { expansion_factor } =
                    &action.search_mode
                {
                    *candidates = (*candidates as f32 * expansion_factor) as usize;
                }

                // Object-economy quantization floor: raise the search method's
                // effective precision to at least the requested floor. The
                // mapping rules are documented on
                // `apply_quantization_floor_to_method`.
                if action.object_economy.enabled
                    && let Some(floor) = action.object_economy.quantization_floor
                {
                    apply_quantization_floor_to_method(execution_method, floor);
                }
            }
        }

        // Apply parallelism settings
        plan.parallelism.use_simd = action.parallelism.enable_simd;

        // Log object economy action if enabled
        if action.object_economy.enabled {
            trace!(
                "Object economy routing applied: centroid={}, zorder={}, max_blocks={:?}",
                action.object_economy.use_centroid_routing,
                action.object_economy.use_zorder_pruning,
                action.object_economy.max_blocks_to_scan,
            );
        }

        trace!(
            "Applied RL action to plan: {} steps, action={}",
            plan.execution_steps.len(),
            action.describe()
        );
    }

    /// Report execution result for learning
    ///
    /// This method:
    /// 1. Calculates reward based on execution metrics
    /// 2. Updates the bandit with the observed state-action-reward tuple
    /// 3. Adds to experience buffer for batch learning
    pub async fn report_execution(
        &self,
        state: &PlannerState,
        action: &ExecutionAction,
        latency_ms: f64,
        recall: f32,
        throughput_qps: f32,
    ) {
        // Calculate reward using the reward calculator
        let reward = {
            let planner = self.planner.read().await;
            planner.calculate_reward(latency_ms, recall, throughput_qps, None)
        };

        // Update planner with observed state-action-reward tuple
        // This updates the bandit distribution and experience buffer
        {
            let planner = self.planner.read().await;
            planner.update(state, action, reward).await;
        }

        debug!(
            "🎯 RL feedback: latency={:.1}ms, recall={:.3}, throughput={:.1}qps, reward={:.3}, action={}",
            latency_ms,
            recall,
            throughput_qps,
            reward,
            action.describe()
        );
    }

    /// Log execution for offline analysis
    pub async fn log_execution(
        &self,
        query_id: &str,
        collection_id: &str,
        state: &PlannerState,
        action: &ExecutionAction,
        latency_ms: f64,
        recall: f32,
        _stages: Vec<StageLog>,
        reward: f32,
    ) {
        let log = ExecutionLog::builder(query_id, collection_id)
            .state(state.clone())
            .action(action.clone())
            .latency_ms(latency_ms)
            .recall(recall)
            .reward(reward)
            .build();

        let planner = self.planner.read().await;
        planner.log_execution(log).await;
    }

    /// Get action statistics for monitoring
    pub async fn get_action_stats(&self) -> std::collections::HashMap<String, (f64, u64)> {
        let planner = self.planner.read().await;
        planner.get_action_stats().await
    }

    /// Save learned policy to file
    pub async fn save_policy(&self, path: &str) -> anyhow::Result<()> {
        let planner = self.planner.read().await;
        planner.save_policy(path).await
    }

    /// Load learned policy from file
    pub async fn load_policy(&self, path: &str) -> anyhow::Result<bool> {
        let planner = self.planner.read().await;
        planner.load_policy(path).await
    }

    /// Total reward updates observed (passthrough for the checkpoint dirty-flag).
    pub async fn total_updates(&self) -> u64 {
        self.planner.read().await.total_updates().await
    }

    /// Check if RL planning is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }
}

impl Default for RLPlannerIntegration {
    fn default() -> Self {
        Self::new(RLPlannerConfig::default())
    }
}

/// Global RL planner integration instance
static RL_PLANNER: once_cell::sync::OnceCell<RLPlannerIntegration> =
    once_cell::sync::OnceCell::new();

/// Initialize global RL planner
pub fn init_rl_planner(config: RLPlannerConfig) {
    if RL_PLANNER.set(RLPlannerIntegration::new(config)).is_err() {
        tracing::warn!("RL planner already initialized");
    } else {
        info!("🎯 RL Query Planner initialized with Thompson Sampling");
    }
}

/// Get global RL planner instance
pub fn get_rl_planner() -> Option<&'static RLPlannerIntegration> {
    RL_PLANNER.get()
}

/// Convenience function to select action if RL planner is enabled
pub async fn rl_select_action(context: &UnifiedQueryContext<'_>) -> Option<ExecutionAction> {
    if let Some(planner) = get_rl_planner()
        && planner.is_enabled()
    {
        let state = planner.extract_state(context);
        return Some(planner.select_action(&state).await);
    }
    None
}

/// Raise a `SearchExecutionMethod` to satisfy a request-side quantization
/// floor coming from the object-economy RL action.
///
/// Mapping rules:
///
/// * [`SearchExecutionMethod::DirectFP32`] always satisfies any floor (FP32
///   is the highest precision).
/// * [`SearchExecutionMethod::IndexBased`] is left unchanged — index-based
///   methods carry their own quality contract and the floor does not apply
///   uniformly across index families.
/// * [`SearchExecutionMethod::QuantizedOnly`] is raised in place to the
///   floor's quantization type when one exists. If the floor is FP16 or
///   FP32 (which have no quantization equivalent) the method is replaced
///   with `DirectFP32`.
/// * [`SearchExecutionMethod::Progressive`] gets an `ExactSearch` final
///   stage appended only when the floor requires exact precision (FP16+).
///   Lower floors are assumed satisfied by any progressive pipeline because
///   the pipeline's last quantized stage is followed by a reranker
///   elsewhere in the planner.
fn apply_quantization_floor_to_method(
    method: &mut SearchExecutionMethod,
    floor: QuantizationStage,
) {
    match method {
        SearchExecutionMethod::DirectFP32 => {
            // FP32 dominates every floor.
        }
        SearchExecutionMethod::IndexBased { .. } => {
            trace!(
                "Object-economy floor ({floor}) skipped for IndexBased method; \
                 index-family quality is governed elsewhere"
            );
        }
        SearchExecutionMethod::QuantizedOnly { quantization_type } => {
            if quantization_type.quality_rank() >= floor.quality_rank() {
                return;
            }
            if let Some(raised) = floor.as_quantization_type() {
                trace!(
                    "Object-economy floor raised QuantizedOnly from {:?} to {:?}",
                    quantization_type, raised
                );
                *quantization_type = raised;
            } else {
                trace!(
                    "Object-economy floor ({floor}) requires exact precision; \
                     replacing QuantizedOnly with DirectFP32"
                );
                *method = SearchExecutionMethod::DirectFP32;
            }
        }
        SearchExecutionMethod::Progressive { stages } => {
            if !floor.requires_exact_final_stage() {
                return;
            }
            let already_exact = stages
                .last()
                .map(|stage| matches!(stage.algorithm, SearchAlgorithm::ExactSearch))
                .unwrap_or(false);
            if already_exact {
                return;
            }
            let final_candidates = stages.last().map(|s| s.candidates).unwrap_or(64);
            trace!(
                "Object-economy floor ({floor}) appending ExactSearch final stage to \
                 Progressive pipeline"
            );
            stages.push(QueryOptimizerProgressiveStage {
                algorithm: SearchAlgorithm::ExactSearch,
                candidates: final_candidates,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::proximadb_v1::Collection;
    use crate::query::query_optimizer::QuantizationType;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_rl_integration_creation() {
        let integration = RLPlannerIntegration::default();
        assert!(integration.is_enabled());
    }

    #[tokio::test]
    async fn test_extract_state() {
        let integration = RLPlannerIntegration::default();

        let collection = Arc::new(Collection {
            id: "test".to_string(),
            config: Some(Default::default()),
            ..Default::default()
        });

        let context = UnifiedQueryContext {
            collection,
            search_params: None,
            filter_params: None,
            optimization_goal: crate::query::query_optimizer::OptimizationGoal::Balanced,
            available_files: vec![],
            total_vectors: 10000,
            total_columns: 5,
            query_vectors: None,
        };

        let state = integration.extract_state(&context);
        assert_eq!(state.collection_size, 10000);
    }

    #[tokio::test]
    async fn test_select_action() {
        let integration = RLPlannerIntegration::default();
        let state = PlannerState::default();

        let action = integration.select_action(&state).await;
        assert!(action.index_strategy.is_some());
    }

    #[tokio::test]
    async fn test_action_stats() {
        let integration = RLPlannerIntegration::default();
        let state = PlannerState::default();

        // Select some actions to generate stats
        for _ in 0..5 {
            let _ = integration.select_action(&state).await;
        }

        let stats = integration.get_action_stats().await;
        // Stats may or may not be populated depending on exploration
        // This test just ensures it doesn't crash
        let _ = stats.len();
    }

    #[tokio::test]
    async fn test_extract_object_economy_features() {
        let integration = RLPlannerIntegration::default();

        let collection = Arc::new(Collection {
            id: "test".to_string(),
            config: Some(Default::default()),
            ..Default::default()
        });

        let context = UnifiedQueryContext {
            collection,
            search_params: None,
            filter_params: None,
            optimization_goal: crate::query::query_optimizer::OptimizationGoal::Balanced,
            available_files: vec![],
            total_vectors: 10000,
            total_columns: 5,
            query_vectors: None,
        };

        let state = integration.extract_state(&context);
        // Object economy features should be disabled since no directory is available yet
        assert!(!state.object_economy.enabled);
        assert_eq!(state.object_economy.block_count, 0);
    }

    // ── Quantization floor mapping tests ────────────────────────────────

    #[test]
    fn floor_no_op_for_direct_fp32() {
        let mut method = SearchExecutionMethod::DirectFP32;
        apply_quantization_floor_to_method(&mut method, QuantizationStage::FP32);
        assert!(matches!(method, SearchExecutionMethod::DirectFP32));
    }

    #[test]
    fn floor_raises_quantized_only_from_binary_to_int8() {
        let mut method = SearchExecutionMethod::QuantizedOnly {
            quantization_type: QuantizationType::Binary,
        };
        apply_quantization_floor_to_method(&mut method, QuantizationStage::INT8);
        match method {
            SearchExecutionMethod::QuantizedOnly { quantization_type } => {
                assert_eq!(quantization_type, QuantizationType::INT8);
            }
            other => panic!("expected QuantizedOnly INT8, got {other:?}"),
        }
    }

    #[test]
    fn floor_keeps_quantized_only_when_current_meets_or_exceeds() {
        let mut method = SearchExecutionMethod::QuantizedOnly {
            quantization_type: QuantizationType::INT8,
        };
        apply_quantization_floor_to_method(&mut method, QuantizationStage::Binary);
        match method {
            SearchExecutionMethod::QuantizedOnly { quantization_type } => {
                assert_eq!(quantization_type, QuantizationType::INT8);
            }
            other => panic!("expected QuantizedOnly INT8, got {other:?}"),
        }
    }

    #[test]
    fn floor_promotes_quantized_only_to_direct_fp32_when_floor_is_fp16() {
        let mut method = SearchExecutionMethod::QuantizedOnly {
            quantization_type: QuantizationType::INT8,
        };
        apply_quantization_floor_to_method(&mut method, QuantizationStage::FP16);
        assert!(matches!(method, SearchExecutionMethod::DirectFP32));
    }

    #[test]
    fn floor_promotes_quantized_only_to_direct_fp32_when_floor_is_fp32() {
        let mut method = SearchExecutionMethod::QuantizedOnly {
            quantization_type: QuantizationType::PQ4,
        };
        apply_quantization_floor_to_method(&mut method, QuantizationStage::FP32);
        assert!(matches!(method, SearchExecutionMethod::DirectFP32));
    }

    #[test]
    fn floor_appends_exact_stage_to_progressive_when_floor_requires_exact() {
        let mut method = SearchExecutionMethod::Progressive {
            stages: vec![QueryOptimizerProgressiveStage {
                algorithm: SearchAlgorithm::QuantizedSearch,
                candidates: 128,
            }],
        };
        apply_quantization_floor_to_method(&mut method, QuantizationStage::FP32);
        match method {
            SearchExecutionMethod::Progressive { stages } => {
                assert_eq!(stages.len(), 2);
                assert!(matches!(
                    stages[0].algorithm,
                    SearchAlgorithm::QuantizedSearch
                ));
                assert!(matches!(stages[1].algorithm, SearchAlgorithm::ExactSearch));
                assert_eq!(stages[1].candidates, 128);
            }
            other => panic!("expected Progressive, got {other:?}"),
        }
    }

    #[test]
    fn floor_does_not_duplicate_exact_stage_when_already_present() {
        let mut method = SearchExecutionMethod::Progressive {
            stages: vec![
                QueryOptimizerProgressiveStage {
                    algorithm: SearchAlgorithm::QuantizedSearch,
                    candidates: 128,
                },
                QueryOptimizerProgressiveStage {
                    algorithm: SearchAlgorithm::ExactSearch,
                    candidates: 32,
                },
            ],
        };
        apply_quantization_floor_to_method(&mut method, QuantizationStage::FP32);
        match method {
            SearchExecutionMethod::Progressive { stages } => {
                assert_eq!(stages.len(), 2);
            }
            other => panic!("expected Progressive, got {other:?}"),
        }
    }

    #[test]
    fn floor_no_op_on_progressive_when_floor_does_not_require_exact() {
        let original = QueryOptimizerProgressiveStage {
            algorithm: SearchAlgorithm::QuantizedSearch,
            candidates: 128,
        };
        let mut method = SearchExecutionMethod::Progressive {
            stages: vec![original.clone()],
        };
        apply_quantization_floor_to_method(&mut method, QuantizationStage::INT8);
        match method {
            SearchExecutionMethod::Progressive { stages } => {
                assert_eq!(stages.len(), 1);
                assert!(matches!(
                    stages[0].algorithm,
                    SearchAlgorithm::QuantizedSearch
                ));
            }
            other => panic!("expected Progressive, got {other:?}"),
        }
    }

    #[test]
    fn floor_leaves_index_based_method_unchanged() {
        let mut method = SearchExecutionMethod::IndexBased {
            index_type: crate::query::query_optimizer::Index::HNSW,
        };
        apply_quantization_floor_to_method(&mut method, QuantizationStage::FP32);
        assert!(matches!(
            method,
            SearchExecutionMethod::IndexBased {
                index_type: crate::query::query_optimizer::Index::HNSW
            }
        ));
    }
}
