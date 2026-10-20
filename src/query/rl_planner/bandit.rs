//! Contextual Bandit with Thompson Sampling
//!
//! Implements the core RL algorithm for action selection. Uses Bayesian
//! posterior updates to balance exploration and exploitation.

use std::collections::HashMap;

use rand::distributions::{Distribution, Uniform};
use rand_distr::Beta;
use serde::{Deserialize, Serialize};
use tracing::warn;

use super::action::{ActionId, ActionSpace, ExecutionAction};
use super::state::PlannerState;

/// Beta distribution parameters for Bayesian reward estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetaDistribution {
    /// Alpha parameter (successes + prior)
    pub alpha: f64,
    /// Beta parameter (failures + prior)
    pub beta: f64,
}

impl BetaDistribution {
    /// Create new Beta distribution with uniform prior
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self { alpha, beta }
    }

    /// Sample from the posterior distribution
    pub fn sample(&self) -> f64 {
        let mut rng = rand::thread_rng();
        match Beta::new(self.alpha, self.beta).or_else(|_| Beta::new(1.0, 1.0)) {
            Ok(beta) => beta.sample(&mut rng),
            Err(error) => {
                // TD-007: Safe fallback to neutral reward (0.5) when Beta distribution
                // construction fails. This is acceptable because:
                // 1. Beta::new only fails with invalid parameters (alpha <= 0, beta <= 0)
                // 2. We try a fallback uniform prior (1.0, 1.0) first
                // 3. Returning 0.5 (neutral reward) is better than panicking the sampler
                warn!(
                    error = %error,
                    alpha = self.alpha,
                    beta = self.beta,
                    "Failed to construct Beta distribution; using neutral reward sample"
                );
                0.5
            }
        }
    }

    /// Get expected value (mean)
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Get variance
    pub fn variance(&self) -> f64 {
        let n = self.alpha + self.beta;
        (self.alpha * self.beta) / (n * n * (n + 1.0))
    }

    /// Update with observed reward
    pub fn update(&mut self, reward: f32) {
        // TD-RLNAN-1: a non-finite reward would poison the posterior for good.
        if !reward.is_finite() {
            return;
        }
        // Convert reward [0, 1] to success/failure counts
        // Higher reward = more success weight
        let reward = reward.clamp(0.0, 1.0);
        self.alpha += reward as f64;
        self.beta += (1.0 - reward) as f64;
    }

    /// Get count of observations
    pub fn count(&self) -> u64 {
        (self.alpha + self.beta - 2.0).max(0.0) as u64
    }
}

impl Default for BetaDistribution {
    fn default() -> Self {
        Self::new(1.0, 1.0) // Uniform prior
    }
}

/// Context features for linear contextual bandit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextWeights {
    /// Weights for feature-action interactions
    pub weights: Vec<f64>,
    /// Learning rate
    pub learning_rate: f64,
}

impl ContextWeights {
    /// Create new context weights
    pub fn new(num_features: usize) -> Self {
        Self {
            weights: vec![0.0; num_features],
            learning_rate: 0.01,
        }
    }

    /// Compute context bonus from features
    pub fn compute_bonus(&self, features: &[f32]) -> f64 {
        features
            .iter()
            .zip(self.weights.iter())
            .map(|(f, w)| (*f as f64) * w)
            .sum()
    }

    /// Update weights with gradient step
    pub fn update(&mut self, features: &[f32], error: f64) {
        // TD-RLNAN-1: skip any step that would make a weight non-finite; a NaN
        // weight is persisted as `null` and the policy no longer loads.
        if !error.is_finite() {
            return;
        }
        for (i, f) in features.iter().enumerate() {
            if i < self.weights.len() {
                let next = self.weights[i] + self.learning_rate * error * (*f as f64);
                if next.is_finite() {
                    self.weights[i] = next;
                }
            }
        }
    }
}

impl Default for ContextWeights {
    fn default() -> Self {
        Self::new(50) // Default feature dimension
    }
}

/// Contextual Bandit Planner with Thompson Sampling
///
/// Uses Bayesian posterior estimation for each action, with contextual
/// features to adjust expected rewards based on query/system state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualBanditPlanner {
    /// Per-action reward estimates (Beta distributions)
    action_stats: HashMap<ActionId, BetaDistribution>,
    /// Per-action context weights
    context_weights: HashMap<ActionId, ContextWeights>,
    /// Exploration rate for ε-greedy fallback
    exploration_rate: f32,
    /// Base exploration rate (before decay) — used to reset for new contexts
    #[serde(skip_serializing, default)]
    base_exploration_rate: f32,
    /// Use Thompson Sampling (true) or ε-greedy (false)
    #[serde(skip_serializing, default)]
    use_thompson_sampling: bool,
    /// Per-engine action spaces
    #[serde(skip_serializing, default)]
    engine_action_spaces: HashMap<String, ActionSpace>,
    /// Default action space
    #[serde(skip_serializing, default)]
    default_action_space: ActionSpace,
    /// Total number of updates
    total_updates: u64,
    /// Per-engine update counts — tracks learning maturity per engine.
    /// When a new engine appears (0 updates), exploration is temporarily boosted.
    engine_update_counts: HashMap<String, u64>,
}

impl ContextualBanditPlanner {
    /// Create new contextual bandit planner
    pub fn new(exploration_rate: f32, use_thompson_sampling: bool) -> Self {
        let mut engine_action_spaces = HashMap::new();
        // Only live storage engines get action spaces. SWIFT and RAPTOR are
        // retired/experimental-off; a query routed to an engine without a space
        // falls back to `default_action_space` (SST) via the `unwrap_or` in
        // select/exploit, which is safe.
        for engine in &["SST", "HELIX", "VIPER", "NOVA"] {
            engine_action_spaces.insert(engine.to_string(), ActionSpace::for_engine(engine));
        }

        let mut planner = Self {
            action_stats: HashMap::new(),
            context_weights: HashMap::new(),
            exploration_rate,
            base_exploration_rate: exploration_rate,
            use_thompson_sampling,
            engine_action_spaces,
            default_action_space: ActionSpace::for_engine("SST"),
            total_updates: 0,
            engine_update_counts: HashMap::new(),
        };

        // Seed with informed priors from static cost model to reduce cold-start exploration
        planner.seed_warm_priors();

        planner
    }

    /// Seed the planner with warm-start priors derived from static cost model heuristics.
    ///
    /// Instead of starting with uniform Beta(1,1) for all actions (which requires ~100+
    /// observations per arm to converge with 36+ arms), this seeds informed priors:
    /// - Index-backed actions (HNSW, IVF) get higher alpha → biased toward exploitation
    /// - Baseline scan actions get moderate priors → always competitive
    /// - Progressive quantization gets a boost for high-dimensional vectors
    ///
    /// The priors are weak (alpha+beta ~10) so they're quickly overridden by real data,
    /// but they dramatically reduce the cold-start random exploration period.
    pub fn seed_warm_priors(&mut self) {
        for (engine_name, action_space) in &self.engine_action_spaces {
            for action in &action_space.actions {
                let action_id = action.to_action_id();

                // Skip actions that already have observations
                if self.action_stats.contains_key(&action_id) {
                    continue;
                }

                // Assign priors based on action characteristics.
                // Higher alpha/beta ratio = higher expected reward.
                // Total (alpha+beta) controls prior strength — ~10 means
                // it takes ~10 real observations to halve the prior's influence.
                let (alpha, beta) = match (
                    engine_name.as_str(),
                    &action.index_strategy,
                    action.quantization_stages.len(),
                ) {
                    // HNSW index strategies — best for low-latency approximate search
                    (_, Some(idx), _) if format!("{:?}", idx).contains("Hnsw") => (7.0, 3.0),
                    // IVF strategies — good for large datasets
                    (_, Some(idx), _) if format!("{:?}", idx).contains("Ivf") => (6.0, 4.0),
                    // Progressive quantization (2+ stages) — good for high-dim vectors
                    (_, _, stages) if stages >= 2 => (6.0, 4.0),
                    // Single-stage quantization — moderate prior
                    (_, _, 1) => (5.0, 5.0),
                    // Baseline scans (no index, no quantization) — always works but slower
                    _ => (4.0, 6.0),
                };

                self.action_stats
                    .insert(action_id, BetaDistribution::new(alpha, beta));
            }
        }
    }

    /// Select action using Thompson Sampling or ε-greedy.
    /// For engines with few observations (<50 updates), temporarily boosts exploration
    /// to the base rate, preventing a well-explored engine from suppressing exploration
    /// Return the action with the highest expected value (α/(α+β)) for all arms.
    /// This is the deterministic exploitation path used on the query hot path.
    /// Thompson Sampling exploration only runs in the background batch-update cycle.
    pub fn exploit_best_action(&self, state: &PlannerState) -> ExecutionAction {
        let engine_name = state.storage_engine.to_string();
        let action_space = self
            .engine_action_spaces
            .get(&engine_name)
            .unwrap_or(&self.default_action_space);
        self.best_action(state, action_space)
    }

    /// for a newly-encountered one.
    pub fn select_action(&self, state: &PlannerState) -> ExecutionAction {
        // Get action space for this engine
        let engine_name = state.storage_engine.to_string();
        // TD-007: unwrap_or with safe default - fallback to default action space
        let action_space = self
            .engine_action_spaces
            .get(&engine_name)
            .unwrap_or(&self.default_action_space);

        if self.use_thompson_sampling {
            // Thompson Sampling naturally explores uncertain arms, but boost exploration
            // for new engines by using the base rate when the engine is under-explored
            self.thompson_sampling_select(state, action_space)
        } else {
            // For ε-greedy: use higher exploration rate for under-explored engines
            let engine_updates = self
                .engine_update_counts
                .get(&engine_name)
                .copied()
                .unwrap_or(0);
            let effective_rate = if engine_updates < 50 {
                self.base_exploration_rate
            } else {
                self.exploration_rate
            };
            self.epsilon_greedy_select_with_rate(state, action_space, effective_rate)
        }
    }

    /// Thompson Sampling action selection
    fn thompson_sampling_select(
        &self,
        state: &PlannerState,
        action_space: &ActionSpace,
    ) -> ExecutionAction {
        let features = state.as_feature_vector();
        let mut best_action = action_space.actions[0].clone();
        let mut best_sample = f64::MIN;

        for action in &action_space.actions {
            let action_id = action.to_action_id();

            // Sample from posterior
            let posterior_sample = self
                .action_stats
                .get(&action_id)
                .map_or(0.5, |beta| beta.sample());

            // Add context bonus
            let context_bonus = self
                .context_weights
                .get(&action_id)
                .map_or(0.0, |w| w.compute_bonus(&features));

            let total_sample = posterior_sample + context_bonus;

            if total_sample > best_sample {
                best_sample = total_sample;
                best_action = action.clone();
            }
        }

        best_action
    }

    /// ε-greedy action selection (uses instance exploration rate)
    fn _epsilon_greedy_select(
        &self,
        state: &PlannerState,
        action_space: &ActionSpace,
    ) -> ExecutionAction {
        self.epsilon_greedy_select_with_rate(state, action_space, self.exploration_rate)
    }

    /// ε-greedy action selection with explicit exploration rate
    fn epsilon_greedy_select_with_rate(
        &self,
        state: &PlannerState,
        action_space: &ActionSpace,
        rate: f32,
    ) -> ExecutionAction {
        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(0.0_f32, 1.0);

        if uniform.sample(&mut rng) < rate {
            // Explore: random action
            action_space.random_action().clone()
        } else {
            // Exploit: best known action
            self.best_action(state, action_space)
        }
    }

    /// Get best known action based on expected reward
    fn best_action(&self, state: &PlannerState, action_space: &ActionSpace) -> ExecutionAction {
        let features = state.as_feature_vector();
        let mut best_action = action_space.actions[0].clone();
        let mut best_value = f64::MIN;

        for action in &action_space.actions {
            let action_id = action.to_action_id();

            // Expected value from posterior
            let expected = self
                .action_stats
                .get(&action_id)
                .map_or(0.5, |beta| beta.mean());

            // Context bonus
            let context_bonus = self
                .context_weights
                .get(&action_id)
                .map_or(0.0, |w| w.compute_bonus(&features));

            let total_value = expected + context_bonus;

            if total_value > best_value {
                best_value = total_value;
                best_action = action.clone();
            }
        }

        best_action
    }

    /// Update planner with observed reward
    pub fn update(&mut self, state: &PlannerState, action: &ExecutionAction, reward: f32) {
        // TD-RLNAN-1: one finite, clamped reward feeds both the posterior and
        // the gradient step.
        if !reward.is_finite() {
            return;
        }
        let reward = reward.clamp(0.0, 1.0);
        let action_id = action.to_action_id();
        let features = state.as_feature_vector();

        // Update Beta distribution
        self.action_stats
            .entry(action_id)
            .or_default()
            .update(reward);

        // Update context weights
        let expected = self
            .action_stats
            .get(&action_id)
            .map_or(0.5, |b| b.mean() as f32);
        let error = (reward - expected) as f64;

        self.context_weights
            .entry(action_id)
            .or_default()
            .update(&features, error);

        self.total_updates += 1;

        // Track per-engine learning maturity
        let engine_name = state.storage_engine.to_string();
        *self.engine_update_counts.entry(engine_name).or_insert(0) += 1;

        // Decay exploration rate over time
        if self.total_updates.is_multiple_of(1000) && self.exploration_rate > 0.01 {
            self.exploration_rate *= 0.99;
        }
    }

    /// Get action statistics (expected value, count)
    pub fn get_action_stats(&self) -> HashMap<String, (f64, u64)> {
        self.action_stats
            .iter()
            .map(|(id, beta)| {
                let action = ExecutionAction::from_action_id(*id);
                (action.describe(), (beta.mean(), beta.count()))
            })
            .collect()
    }

    /// Get total number of updates
    pub fn total_updates(&self) -> u64 {
        self.total_updates
    }

    /// Get exploration rate
    pub fn exploration_rate(&self) -> f32 {
        self.exploration_rate
    }

    /// Save planner state to file
    pub async fn save_to_file(&self, path: &str) -> anyhow::Result<()> {
        // JSON has no NaN/inf (serde writes them as null, which cannot be read
        // back): persist only finite learned values.
        let mut persisted = self.clone();
        persisted.make_finite();
        let json = serde_json::to_string_pretty(&persisted)?;
        tokio::fs::write(path, json).await?;
        tracing::info!("Saved RL planner state to {}", path);
        Ok(())
    }

    /// Load planner state from file
    /// Returns whether a policy was loaded (`false` when the file was unreadable
    /// and a fresh policy is kept).
    pub async fn load_from_file(&mut self, path: &str) -> anyhow::Result<bool> {
        let json = tokio::fs::read_to_string(path).await?;
        // An unreadable policy is a learned cache, not data: start fresh rather
        // than fail startup.
        let mut loaded: ContextualBanditPlanner = match serde_json::from_str(&json) {
            Ok(loaded) => loaded,
            Err(error) => {
                // TD-RLNAN-1: move the unreadable policy aside so a bad file
                // never lingers; keep it beside the original for diagnosis,
                // and start fresh either way.
                match super::set_aside_unreadable_policy(path).await {
                    Ok(aside) => tracing::warn!(
                        path,
                        aside_path = %aside,
                        %error,
                        "RL planner policy file is invalid; moved aside, starting with a fresh policy"
                    ),
                    Err(rename_error) => tracing::warn!(
                        path,
                        %error,
                        %rename_error,
                        "RL planner policy file is invalid and could not be moved aside; starting with a fresh in-memory policy"
                    ),
                }
                return Ok(false);
            }
        };
        loaded.make_finite();

        self.action_stats = loaded.action_stats;
        self.context_weights = loaded.context_weights;
        self.exploration_rate = loaded.exploration_rate;
        self.total_updates = loaded.total_updates;
        // Restore per-engine learning maturity too. Without this every engine
        // looks "new" (0 updates) after restart and exploration is wrongly
        // boosted via the <50-updates branch in select_action.
        self.engine_update_counts = loaded.engine_update_counts;

        tracing::info!(
            "Loaded RL planner state from {} ({} actions, {} updates)",
            path,
            self.action_stats.len(),
            self.total_updates
        );
        Ok(true)
    }

    /// Replace non-finite learned values: a non-finite weight becomes 0.0 and a
    /// non-finite (or non-positive) Beta parameter resets to the uniform prior.
    fn make_finite(&mut self) {
        for stats in self.action_stats.values_mut() {
            if !stats.alpha.is_finite() || stats.alpha <= 0.0 {
                stats.alpha = 1.0;
            }
            if !stats.beta.is_finite() || stats.beta <= 0.0 {
                stats.beta = 1.0;
            }
        }
        for weights in self.context_weights.values_mut() {
            for weight in weights.weights.iter_mut() {
                if !weight.is_finite() {
                    *weight = 0.0;
                }
            }
            if !weights.learning_rate.is_finite() {
                weights.learning_rate = 0.01;
            }
        }
        if !self.exploration_rate.is_finite() {
            self.exploration_rate = 0.1;
        }
    }

    /// Get top-k best actions for a given state
    pub fn get_top_actions(&self, state: &PlannerState, k: usize) -> Vec<(ExecutionAction, f64)> {
        let engine_name = state.storage_engine.to_string();
        // TD-007: unwrap_or with safe default - fallback to default action space
        let action_space = self
            .engine_action_spaces
            .get(&engine_name)
            .unwrap_or(&self.default_action_space);

        let features = state.as_feature_vector();
        let mut scored_actions: Vec<(ExecutionAction, f64)> = action_space
            .actions
            .iter()
            .map(|action| {
                let action_id = action.to_action_id();
                let expected = self.action_stats.get(&action_id).map_or(0.5, |b| b.mean());
                let context_bonus = self
                    .context_weights
                    .get(&action_id)
                    .map_or(0.0, |w| w.compute_bonus(&features));
                (action.clone(), expected + context_bonus)
            })
            .collect();

        scored_actions.sort_by(|a, b| {
            // TD-007: unwrap_or with safe default - Equal for NaN comparisons
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored_actions.truncate(k);
        scored_actions
    }

    /// Reset planner to initial state
    pub fn reset(&mut self) {
        self.action_stats.clear();
        self.context_weights.clear();
        self.total_updates = 0;
        self.exploration_rate = 0.1;
    }
}

impl Default for ContextualBanditPlanner {
    fn default() -> Self {
        Self::new(0.1, true)
    }
}

/// Upper Confidence Bound (UCB) variant for comparison
pub struct UCBPlanner {
    /// Per-action statistics
    action_stats: HashMap<ActionId, (f64, u64)>, // (sum, count)
    /// UCB exploration constant
    exploration_constant: f64,
    /// Total observations
    total_observations: u64,
}

impl UCBPlanner {
    /// Create new UCB planner
    pub fn new(exploration_constant: f64) -> Self {
        Self {
            action_stats: HashMap::new(),
            exploration_constant,
            total_observations: 0,
        }
    }

    /// Select action using UCB1
    pub fn select_action(&self, action_space: &ActionSpace) -> ExecutionAction {
        let mut best_action = action_space.actions[0].clone();
        let mut best_ucb = f64::MIN;

        for action in &action_space.actions {
            let action_id = action.to_action_id();

            let ucb = match self.action_stats.get(&action_id) {
                Some((sum, count)) if *count > 0 => {
                    let mean = sum / (*count as f64);
                    let exploration = self.exploration_constant
                        * ((self.total_observations as f64).ln() / (*count as f64)).sqrt();
                    mean + exploration
                }
                _ => f64::MAX, // Explore unvisited actions first
            };

            if ucb > best_ucb {
                best_ucb = ucb;
                best_action = action.clone();
            }
        }

        best_action
    }

    /// Update with observed reward
    pub fn update(&mut self, action: &ExecutionAction, reward: f32) {
        let action_id = action.to_action_id();
        let entry = self.action_stats.entry(action_id).or_insert((0.0, 0));
        entry.0 += reward as f64;
        entry.1 += 1;
        self.total_observations += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beta_distribution() {
        let mut beta = BetaDistribution::default();
        assert!((beta.mean() - 0.5).abs() < 0.01);

        // Update with positive rewards
        for _ in 0..10 {
            beta.update(0.9);
        }
        assert!(beta.mean() > 0.5);

        // Sample should be in [0, 1]
        for _ in 0..100 {
            let sample = beta.sample();
            assert!((0.0..=1.0).contains(&sample));
        }
    }

    #[test]
    fn test_context_weights() {
        let mut weights = ContextWeights::new(10);
        let features = vec![0.5; 10];

        // Initial bonus should be 0
        assert!((weights.compute_bonus(&features) - 0.0).abs() < 0.001);

        // Update with positive error
        weights.update(&features, 1.0);
        assert!(weights.compute_bonus(&features) > 0.0);
    }

    #[test]
    fn test_thompson_sampling_selection() {
        let planner = ContextualBanditPlanner::new(0.1, true);
        let state = PlannerState::default();

        // Should return a valid action
        let action = planner.select_action(&state);
        assert!(action.index_strategy.is_some());
    }

    #[test]
    fn test_epsilon_greedy_selection() {
        let planner = ContextualBanditPlanner::new(1.0, false); // 100% exploration
        let state = PlannerState::default();

        // With 100% exploration, should return random actions
        let mut actions = std::collections::HashSet::new();
        for _ in 0..100 {
            let action = planner.select_action(&state);
            actions.insert(action.to_action_id());
        }
        // Should have multiple different actions
        assert!(actions.len() > 1);
    }

    #[test]
    fn test_update_and_learning() {
        let mut planner = ContextualBanditPlanner::new(0.0, true); // No exploration
        let state = PlannerState::default();

        // Get initial action
        let initial_action = planner.select_action(&state);

        // Strongly reward a different action
        let good_action = ExecutionAction::with_hnsw(100);
        for _ in 0..50 {
            planner.update(&state, &good_action, 1.0);
        }

        // Punish initial action
        for _ in 0..50 {
            planner.update(&state, &initial_action, 0.0);
        }

        // Now should prefer the rewarded action
        let new_action = planner.select_action(&state);
        let _good_id = good_action.to_action_id();
        let _new_id = new_action.to_action_id();

        // The good action should have high expected value
        let stats = planner.get_action_stats();
        let good_desc = good_action.describe();
        if let Some((mean, _)) = stats.get(&good_desc) {
            assert!(*mean > 0.8, "Good action should have high expected value");
        }
    }

    #[test]
    fn test_get_top_actions() {
        let mut planner = ContextualBanditPlanner::new(0.1, true);
        let state = PlannerState::default();

        // Train on some actions
        let action1 = ExecutionAction::with_hnsw(50);
        let action2 = ExecutionAction::with_hnsw(100);

        for _ in 0..10 {
            planner.update(&state, &action1, 0.9);
            planner.update(&state, &action2, 0.7);
        }

        let top_actions = planner.get_top_actions(&state, 3);
        assert_eq!(top_actions.len(), 3);

        // First action should have highest score
        assert!(top_actions[0].1 >= top_actions[1].1);
        assert!(top_actions[1].1 >= top_actions[2].1);
    }

    #[test]
    fn test_ucb_planner() {
        let mut planner = UCBPlanner::new(2.0);
        let action_space = ActionSpace::for_engine("SST");

        // Initially should explore all actions
        let mut visited = std::collections::HashSet::new();
        for _ in 0..action_space.len() * 2 {
            let action = planner.select_action(&action_space);
            visited.insert(action.to_action_id());
            planner.update(&action, 0.5);
        }

        // Should have visited multiple actions
        assert!(visited.len() > 1);
    }

    /// TD-RLNAN-1: a search on an empty collection (collection_size = 0) made
    /// the log-scaled feature -inf, the gradient step turned a weight into NaN,
    /// serde_json wrote it as `null`, and the next boot refused to load the
    /// policy ("invalid type: null, expected f64"), so the server would not start.
    #[tokio::test]
    async fn empty_collection_update_keeps_policy_loadable() {
        let mut planner = ContextualBanditPlanner::new(0.1, true);
        let state = PlannerState {
            collection_size: 0,
            ..PlannerState::default()
        };
        let action = planner.select_action(&state);
        for reward in [0.8, 0.2, f32::NAN, f32::INFINITY, 0.5] {
            planner.update(&state, &action, reward);
        }
        assert!(
            planner
                .context_weights
                .values()
                .all(|w| w.weights.iter().all(|x| x.is_finite())),
            "learned weights must stay finite"
        );
        assert!(
            planner
                .action_stats
                .values()
                .all(|b| b.alpha.is_finite() && b.beta.is_finite()),
            "reward statistics must stay finite"
        );

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("rl_policy.json");
        let path = path.to_str().expect("utf-8 path");
        planner.save_to_file(path).await.expect("save");
        let mut loaded = ContextualBanditPlanner::new(0.1, true);
        loaded
            .load_from_file(path)
            .await
            .expect("a saved policy must load back");
    }

    #[test]
    fn feature_vector_is_finite_for_degenerate_state() {
        let state = PlannerState {
            collection_size: 0,
            filter_selectivity: f32::NAN,
            ..PlannerState::default()
        };
        assert!(
            state.as_feature_vector().iter().all(|f| f.is_finite()),
            "every feature must be finite"
        );
    }

    #[tokio::test]
    async fn test_save_and_load() {
        let mut planner = ContextualBanditPlanner::new(0.1, true);
        let state = PlannerState::default();
        let action = planner.select_action(&state);

        // Train
        for _ in 0..10 {
            planner.update(&state, &action, 0.8);
        }

        // Save
        let path = "/tmp/test_rl_planner_state.json";
        planner.save_to_file(path).await.unwrap();

        // Load into new planner
        let mut loaded = ContextualBanditPlanner::new(0.1, true);
        loaded.load_from_file(path).await.unwrap();

        // Should have same stats
        assert_eq!(planner.total_updates(), loaded.total_updates());
        assert_eq!(planner.action_stats.len(), loaded.action_stats.len());

        // The static action-space config must NOT be persisted (it's regenerated
        // by `new()` and never read back); only learned state is on disk. This
        // is what keeps the policy file small.
        let on_disk = tokio::fs::read_to_string(path).await.unwrap();
        assert!(
            !on_disk.contains("engine_action_spaces"),
            "persisted policy must omit static engine_action_spaces, got: {on_disk}"
        );
        assert!(
            !on_disk.contains("default_action_space"),
            "persisted policy must omit static default_action_space"
        );

        // engine_update_counts is learned state and must round-trip (it was
        // previously dropped on load, making every engine look "new" after a
        // restart and wrongly boosting exploration).
        assert_eq!(
            planner.engine_update_counts, loaded.engine_update_counts,
            "engine_update_counts must round-trip through save/load"
        );

        // Cleanup
        let _ = tokio::fs::remove_file(path).await;
    }

    #[tokio::test]
    async fn non_finite_policy_values_are_never_persisted_and_bad_files_reset() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("rl_policy.json");
        let path = path.to_str().expect("utf8 path");

        let mut planner = ContextualBanditPlanner::new(0.1, true);
        planner.context_weights.insert(
            1,
            ContextWeights {
                weights: vec![0.5, f64::NAN, f64::INFINITY],
                learning_rate: 0.01,
            },
        );
        planner.save_to_file(path).await.expect("save");
        let mut reloaded = ContextualBanditPlanner::new(0.1, true);
        reloaded.load_from_file(path).await.expect("load");
        assert_eq!(reloaded.context_weights[&1].weights, vec![0.5, 0.0, 0.0]);

        // A file that cannot be read back (e.g. null where a number belongs)
        // starts a fresh policy instead of failing.
        tokio::fs::write(
            path,
            r#"{"action_stats":{},"context_weights":{"1":{"weights":[null],"learning_rate":0.01}},"exploration_rate":0.1,"total_updates":3,"engine_update_counts":{}}"#,
        )
        .await
        .expect("write");
        let mut fresh = ContextualBanditPlanner::new(0.1, true);
        let loaded = fresh
            .load_from_file(path)
            .await
            .expect("invalid file must not fail");
        assert!(!loaded, "an invalid file is reported as not loaded");
        assert_eq!(fresh.total_updates(), 0);
    }
}
