//! # Reinforcement Learning Query Planner
//!
//! This module implements an RL-based Adaptive Query Planner that learns optimal
//! execution paths across all storage engines, index algorithms, and quantization strategies.
//!
//! ## Overview
//!
//! The planner uses Contextual Bandits with Thompson Sampling to:
//! 1. **Observe** query characteristics + system state
//! 2. **Explore** all optimization paths systematically
//! 3. **Log** execution results (latency, recall, throughput, memory)
//! 4. **Learn** from outcomes using reinforcement learning
//! 5. **Exploit** learned policies to select optimal paths
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                      RL Query Planner                                │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                      │
//! │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐            │
//! │  │ State        │   │ Action       │   │ Reward       │            │
//! │  │ Encoder      │──▶│ Selector     │──▶│ Calculator   │            │
//! │  └──────────────┘   └──────────────┘   └──────────────┘            │
//! │         │                  │                  │                     │
//! │         ▼                  ▼                  ▼                     │
//! │  ┌──────────────────────────────────────────────────────┐          │
//! │  │              Experience Replay Buffer                 │          │
//! │  │  (state, action, reward, next_state) tuples          │          │
//! │  └──────────────────────────────────────────────────────┘          │
//! │         │                                                           │
//! │         ▼                                                           │
//! │  ┌──────────────────────────────────────────────────────┐          │
//! │  │         Contextual Bandit (Thompson Sampling)         │          │
//! │  │  Learns: state → optimal action mapping               │          │
//! │  └──────────────────────────────────────────────────────┘          │
//! │                                                                      │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Supported Optimization Paths
//!
//! - **Index Types**: HNSW, IVF, LSH, Annoy, PQ, DirectScan
//! - **Search Modes**: Exact, Approximate, Adaptive
//! - **Quantization**: Binary, INT8, PQ4, PQ8, FP32 (Progressive pipelines)
//! - **Block Pruning**: Off, Sqrt, Ratio-based
//! - **Storage Engines**: SST, HELIX, VIPER, SWIFT, NOVA, RAPTOR
//!
//! ## Usage
//!
//! ```rust,ignore
//! use proximadb::query::rl_planner::{RLPlanner, PlannerState, OptimizationGoal};
//!
//! let planner = RLPlanner::new(Default::default());
//! let state = planner.encode_state(&query_context);
//! let action = planner.select_action(&state);
//!
//! // Execute query with selected action
//! let result = execute_with_action(&query, &action);
//!
//! // Update planner with observed reward
//! let reward = planner.calculate_reward(&result, OptimizationGoal::Balanced);
//! planner.update(&state, &action, reward);
//! ```

pub mod action;
pub mod bandit;
pub mod experience;
pub mod integration;
pub mod logging;
pub mod paths;
pub mod rank_choice;
pub use proximadb_rl_reward::reward;
pub mod state;

// Re-export main types
pub use action::{
    BlockPruneConfig, ExecutionAction, IndexStrategy, ParallelismConfig, QuantizationStage,
    SearchModeAction,
};
pub use bandit::ContextualBanditPlanner;
pub use experience::ExperienceBuffer;
pub use logging::{
    ActionHistory, ExecutionLog, ExplainIntegration, RLDecisionContext, RLDecisionLogger, StageLog,
};
pub use reward::{OptimizationGoal, OptimizationTarget, RewardCalculator};
pub use state::{FilterComplexity, ObjectEconomyFeatures, PlannerState};

// Re-export integration utilities
pub use integration::{RLPlannerIntegration, get_rl_planner, init_rl_planner, rl_select_action};

use std::sync::Arc;
use tokio::sync::RwLock;

/// RL-based Query Planner configuration
#[derive(Debug, Clone)]
pub struct RLPlannerConfig {
    /// Enable RL-based planning (false = use static heuristics)
    pub enabled: bool,
    /// Exploration rate for ε-greedy fallback (0.0 - 1.0)
    pub exploration_rate: f32,
    /// Use Thompson Sampling (true) or ε-greedy (false)
    pub thompson_sampling: bool,
    /// Size of experience replay buffer
    pub experience_buffer_size: usize,
    /// Number of experiences before batch update
    pub batch_update_interval: usize,
    /// Log all query executions
    pub log_all_executions: bool,
    /// Path for execution logs (JSONL format)
    pub log_path: Option<String>,
    /// Default optimization goal
    pub default_goal: OptimizationGoal,
}

impl Default for RLPlannerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            exploration_rate: 0.1,
            thompson_sampling: true,
            experience_buffer_size: 10_000,
            batch_update_interval: 100,
            log_all_executions: true,
            log_path: None,
            default_goal: OptimizationGoal::Balanced,
        }
    }
}

/// Main RL Query Planner
///
/// Coordinates state encoding, action selection, reward calculation,
/// and policy updates for optimal query execution path selection.
pub struct RLPlanner {
    /// Configuration
    config: RLPlannerConfig,
    /// Contextual bandit for action selection
    bandit: Arc<RwLock<ContextualBanditPlanner>>,
    /// Experience replay buffer
    experience_buffer: Arc<RwLock<ExperienceBuffer>>,
    /// Execution logger
    logger: Arc<RwLock<logging::ExecutionLogger>>,
    /// Reward calculator
    reward_calculator: RewardCalculator,
}

impl RLPlanner {
    /// Create new RL planner with configuration
    pub fn new(config: RLPlannerConfig) -> Self {
        let bandit =
            ContextualBanditPlanner::new(config.exploration_rate, config.thompson_sampling);
        let experience_buffer = ExperienceBuffer::new(config.experience_buffer_size);
        let logger = logging::ExecutionLogger::new(config.log_path.clone());
        let reward_calculator = RewardCalculator::new(config.default_goal);

        Self {
            config,
            bandit: Arc::new(RwLock::new(bandit)),
            experience_buffer: Arc::new(RwLock::new(experience_buffer)),
            logger: Arc::new(RwLock::new(logger)),
            reward_calculator,
        }
    }

    /// Check if RL planning is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Return the best action by deterministic expected-value exploitation (α/(α+β) per arm).
    ///
    /// This is the **hot-path method**. Thompson Sampling stochastic exploration is
    /// intentionally excluded here; it only runs during the background `batch_update` cycle
    /// so that exploration variance never adds latency to live queries (CAKE arXiv:2602.04181,
    /// Scheduling Decisions arXiv:2501.16256).
    pub async fn exploit_best_action(&self, state: &PlannerState) -> ExecutionAction {
        let bandit = self.bandit.read().await;
        bandit.exploit_best_action(state)
    }

    /// Select optimal action for given state
    pub async fn select_action(&self, state: &PlannerState) -> ExecutionAction {
        let bandit = self.bandit.read().await;
        bandit.select_action(state)
    }

    /// Calculate reward from execution result
    pub fn calculate_reward(
        &self,
        latency_ms: f64,
        recall: f32,
        throughput_qps: f32,
        target: Option<&OptimizationTarget>,
    ) -> f32 {
        self.reward_calculator
            .calculate(latency_ms, recall, throughput_qps, target)
    }

    /// Update planner with observed reward
    pub async fn update(&self, state: &PlannerState, action: &ExecutionAction, reward: f32) {
        // Update bandit
        {
            let mut bandit = self.bandit.write().await;
            bandit.update(state, action, reward);
        }

        // Add to experience buffer
        {
            let mut buffer = self.experience_buffer.write().await;
            buffer.add(state.clone(), action.clone(), reward);
        }

        // Check if batch update needed
        let should_batch_update = {
            let buffer = self.experience_buffer.read().await;
            buffer.len() >= self.config.batch_update_interval
                && buffer.len() % self.config.batch_update_interval == 0
        };

        if should_batch_update {
            self.batch_update().await;
        }
    }

    /// Log execution result
    pub async fn log_execution(&self, log: ExecutionLog) {
        if self.config.log_all_executions {
            let mut logger = self.logger.write().await;
            if let Err(e) = logger.log(&log).await {
                tracing::warn!("Failed to log execution: {}", e);
            }
        }
    }

    /// Perform batch update from experience buffer
    async fn batch_update(&self) {
        let experiences = {
            let buffer = self.experience_buffer.read().await;
            buffer.sample(self.config.batch_update_interval)
        };

        let mut bandit = self.bandit.write().await;
        for (state, action, reward) in experiences {
            bandit.update(&state, &action, reward);
        }
    }

    /// Get statistics about action usage
    pub async fn get_action_stats(&self) -> std::collections::HashMap<String, (f64, u64)> {
        let bandit = self.bandit.read().await;
        bandit.get_action_stats()
    }

    /// Load persisted policy from file
    pub async fn load_policy(&self, path: &str) -> anyhow::Result<bool> {
        let mut bandit = self.bandit.write().await;
        bandit.load_from_file(path).await
    }

    /// Persist current policy to file
    pub async fn save_policy(&self, path: &str) -> anyhow::Result<()> {
        let bandit = self.bandit.read().await;
        bandit.save_to_file(path).await
    }

    /// Total number of reward updates observed since the planner was created.
    /// Used by the checkpoint task to skip no-op rewrites when no learning has
    /// occurred between ticks.
    pub async fn total_updates(&self) -> u64 {
        self.bandit.read().await.total_updates()
    }
}

impl Default for RLPlanner {
    fn default() -> Self {
        Self::new(RLPlannerConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rl_planner_creation() {
        let planner = RLPlanner::default();
        assert!(planner.is_enabled());
    }

    #[tokio::test]
    async fn test_action_selection() {
        let planner = RLPlanner::default();
        let state = PlannerState::default();
        let action = planner.select_action(&state).await;

        // Should return a valid action
        assert!(action.index_strategy.is_some());
    }

    #[tokio::test]
    async fn test_reward_calculation() {
        let planner = RLPlanner::default();

        // Good result: low latency, high recall
        let reward = planner.calculate_reward(5.0, 0.98, 500.0, None);
        assert!(reward > 0.5);

        // Bad result: high latency, low recall
        let reward = planner.calculate_reward(100.0, 0.5, 10.0, None);
        assert!(reward < 0.5);
    }

    #[tokio::test]
    async fn test_update_cycle() {
        let planner = RLPlanner::default();
        let state = PlannerState::default();
        let action = planner.select_action(&state).await;

        // Update with positive reward
        planner.update(&state, &action, 0.9).await;

        // Action stats should be updated
        let stats = planner.get_action_stats().await;
        assert!(!stats.is_empty());
    }
}
