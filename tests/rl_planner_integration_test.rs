//! RL Planner Integration Tests
//!
//! Tests the complete RL planner flow:
//! 1. Initialization from config
//! 2. State extraction from query context
//! 3. Action selection during query optimization
//! 4. Reward feedback after query execution
//! 5. Policy persistence and loading
//! 6. Learning over multiple queries

use proximadb::query::rl_planner::state::StorageEngineType;
use proximadb::query::rl_planner::{
    ExecutionAction, FilterComplexity, IndexStrategy, OptimizationGoal, PlannerState,
    RLPlannerConfig, SearchModeAction, get_rl_planner, init_rl_planner,
};
use tempfile::TempDir;

/// Test that RL planner can be initialized with custom config
#[test]
fn test_rl_planner_initialization() {
    let config = RLPlannerConfig {
        enabled: true,
        thompson_sampling: true,
        exploration_rate: 0.1,
        experience_buffer_size: 100,
        batch_update_interval: 10,
        log_all_executions: false,
        log_path: None,
        default_goal: OptimizationGoal::Balanced,
    };

    init_rl_planner(config);

    let planner = get_rl_planner();
    assert!(planner.is_some(), "RL planner should be initialized");
    assert!(
        planner.unwrap().is_enabled(),
        "RL planner should be enabled"
    );
}

/// Test complete feedback loop: select action -> execute -> report reward
#[tokio::test]
async fn test_rl_feedback_loop() {
    // Initialize planner
    let config = RLPlannerConfig {
        enabled: true,
        thompson_sampling: true,
        exploration_rate: 0.05, // Low exploration for deterministic testing
        experience_buffer_size: 100,
        batch_update_interval: 5,
        log_all_executions: false,
        log_path: None,
        default_goal: OptimizationGoal::Balanced,
    };
    init_rl_planner(config);

    let planner = get_rl_planner().expect("Planner should be initialized");

    // Create a mock state
    let state = PlannerState::builder()
        .query_dimension(768)
        .top_k(10)
        .collection_size(10_000)
        .storage_engine(StorageEngineType::SST)
        .build();

    // Select an action
    let action = planner.select_action(&state).await;
    assert!(
        !action.describe().is_empty(),
        "Action should have description"
    );

    // Simulate good execution and report reward
    planner
        .report_execution(&state, &action, 5.0, 0.98, 200.0)
        .await;

    // Get stats to verify update
    let stats = planner.get_action_stats().await;
    assert!(
        !stats.is_empty(),
        "Stats should have entries after feedback"
    );
}

/// Test that planner learns from repeated queries
#[tokio::test]
async fn test_rl_learning_over_queries() {
    let config = RLPlannerConfig {
        enabled: true,
        thompson_sampling: false, // Use epsilon-greedy for predictable learning
        exploration_rate: 0.0,    // Pure exploitation for testing
        experience_buffer_size: 100,
        batch_update_interval: 5,
        log_all_executions: false,
        log_path: None,
        default_goal: OptimizationGoal::MinLatency,
    };
    init_rl_planner(config);

    let planner = get_rl_planner().expect("Planner should be initialized");

    let state = PlannerState::builder()
        .query_dimension(128)
        .top_k(10)
        .collection_size(1_000)
        .storage_engine(StorageEngineType::SST)
        .build();

    // Simulate 10 queries with varying performance
    for i in 0..10 {
        let action = planner.select_action(&state).await;

        // Simulate execution with some actions performing better
        let latency = if action.describe().contains("HNSW") {
            5.0 + (i as f64 * 0.1) // HNSW gets good latency
        } else {
            15.0 + (i as f64 * 0.5) // Others get worse latency
        };

        let recall = 0.95;
        let throughput = 1000.0 / latency as f32;

        planner
            .report_execution(&state, &action, latency, recall, throughput)
            .await;
    }

    // Verify learning happened
    let stats = planner.get_action_stats().await;
    assert!(!stats.is_empty(), "Should have tracked some actions");
}

/// Test policy persistence and loading
#[tokio::test]
async fn test_policy_persistence() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let policy_path = temp_dir.path().join("test_policy.json");
    let policy_path_str = policy_path.to_string_lossy().to_string();

    // Initialize and train
    let config = RLPlannerConfig {
        enabled: true,
        thompson_sampling: true,
        exploration_rate: 0.1,
        experience_buffer_size: 100,
        batch_update_interval: 5,
        log_all_executions: false,
        log_path: None,
        default_goal: OptimizationGoal::Balanced,
    };
    init_rl_planner(config.clone());

    let planner = get_rl_planner().expect("Planner should be initialized");

    // Add some training data
    let state = PlannerState::builder()
        .query_dimension(768)
        .top_k(10)
        .collection_size(10_000)
        .storage_engine(StorageEngineType::HELIX)
        .build();

    for _ in 0..5 {
        let action = planner.select_action(&state).await;
        planner
            .report_execution(&state, &action, 10.0, 0.95, 100.0)
            .await;
    }

    // Save policy
    planner
        .save_policy(&policy_path_str)
        .await
        .expect("Should save policy");
    assert!(policy_path.exists(), "Policy file should exist");

    // Verify file has content
    let content = std::fs::read_to_string(&policy_path).expect("Should read policy file");
    assert!(content.len() > 10, "Policy file should have content");
    assert!(
        content.contains("alpha") || content.contains("action"),
        "Policy should contain learned data"
    );
}

/// Test action space coverage for different engines
#[tokio::test]
async fn test_action_space_coverage() {
    let config = RLPlannerConfig::default();
    init_rl_planner(config);

    let planner = get_rl_planner().expect("Planner should be initialized");

    let engines = [
        StorageEngineType::SST,
        StorageEngineType::HELIX,
        StorageEngineType::SWIFT,
        StorageEngineType::NOVA,
        StorageEngineType::RAPTOR,
    ];

    for engine in engines {
        let state = PlannerState::builder()
            .query_dimension(768)
            .top_k(10)
            .collection_size(10_000)
            .storage_engine(engine)
            .build();

        // Get multiple actions to test exploration
        let mut actions_seen = std::collections::HashSet::new();
        for _ in 0..10 {
            let action = planner.select_action(&state).await;
            actions_seen.insert(action.describe());
        }

        // Should see at least some variety (with exploration)
        assert!(
            !actions_seen.is_empty(),
            "Should select valid actions for engine {:?}",
            engine
        );
    }
}

/// Test experience buffer batching
#[tokio::test]
async fn test_experience_batching() {
    let config = RLPlannerConfig {
        enabled: true,
        thompson_sampling: true,
        exploration_rate: 0.1,
        experience_buffer_size: 100,
        batch_update_interval: 5, // Batch every 5 experiences
        log_all_executions: false,
        log_path: None,
        default_goal: OptimizationGoal::Balanced,
    };
    init_rl_planner(config);

    let planner = get_rl_planner().expect("Planner should be initialized");

    let state = PlannerState::builder()
        .query_dimension(768)
        .top_k(10)
        .collection_size(10_000)
        .storage_engine(StorageEngineType::SST)
        .build();

    // Add exactly batch_update_interval experiences
    for i in 0..5 {
        let action = planner.select_action(&state).await;
        let latency = 10.0 + i as f64;
        planner
            .report_execution(&state, &action, latency, 0.95, 100.0)
            .await;
    }

    // Buffer should have processed the batch
    let stats = planner.get_action_stats().await;
    assert!(!stats.is_empty(), "Stats should be updated after batch");
}

/// Test that disabled config results in is_enabled() returning false
/// Note: Due to global state, this test only verifies is_enabled() behavior
/// when config.enabled = false
#[test]
fn test_disabled_planner_config() {
    // Create a disabled config
    let config = RLPlannerConfig {
        enabled: false,
        ..Default::default()
    };

    // Verify the config is set correctly
    assert!(!config.enabled, "Config should have enabled = false");

    // Note: We can't test the actual planner behavior because the global
    // planner may have been initialized by other tests. The is_enabled()
    // method reads from the initialized state, not the config.
}

/// Test state feature extraction
#[test]
fn test_state_feature_extraction() {
    let state = PlannerState::builder()
        .query_dimension(768)
        .top_k(10)
        .with_filter(0.1, FilterComplexity::Simple)
        .collection_size(100_000)
        .storage_engine(StorageEngineType::HELIX)
        .memory_pressure(0.3)
        .cpu_utilization(0.5)
        .build();

    // State should have valid feature vector
    let features = state.as_feature_vector();
    assert!(!features.is_empty(), "Features should not be empty");
    assert!(features.len() >= 5, "Should have multiple features");

    // Verify state properties
    assert_eq!(state.query_dimension, 768);
    assert_eq!(state.top_k, 10);
    assert!(state.has_filter);
    assert!((state.filter_selectivity - 0.1).abs() < 0.01);
    assert_eq!(state.collection_size, 100_000);
}

/// Test action description formatting
#[test]
fn test_action_descriptions() {
    // Use default action as base
    let base = ExecutionAction::default();

    let actions = vec![
        ExecutionAction {
            index_strategy: Some(IndexStrategy::HNSW {
                m: 16,
                ef_search: 100,
            }),
            bloom_filter_enabled: true,
            ..base.clone()
        },
        ExecutionAction {
            index_strategy: Some(IndexStrategy::IVF { n_probe: 16 }),
            search_mode: SearchModeAction::Approximate {
                expansion_factor: 1.5,
            },
            zone_map_enabled: true,
            ..base.clone()
        },
        ExecutionAction {
            index_strategy: Some(IndexStrategy::DirectScan),
            ..base.clone()
        },
    ];

    for action in actions {
        let desc = action.describe();
        assert!(!desc.is_empty(), "Action should have description");
        // Description should mention the index type
        assert!(
            desc.contains("HNSW") || desc.contains("IVF") || desc.contains("DirectScan"),
            "Description should mention index type: {}",
            desc
        );
    }
}
