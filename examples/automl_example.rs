// Copyright 2025 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Example: Using AutoML for automatic database optimization
//!
//! This example demonstrates how to use ProximaDB's AutoML framework to:
//! 1. Monitor workload patterns
//! 2. Predict performance bottlenecks
//! 3. Automatically optimize configurations
//! 4. Tune hyperparameters for best performance

use proximadb::automl::{
    AutoMLConfig, AutoMLCoordinator, HyperparameterTuner, OptimizationGoal, OptimizationPipeline,
    TuningConfig, WorkloadAnalyzer, WorkloadPattern,
    optimization::OptimizationStrategy,
    tuning::{ParameterValue, ProximaDBHyperparameters, TuningAlgorithm},
};
use std::collections::HashMap;
use tokio::time::{Duration, sleep};
use tracing::{info, warn};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    info!("🚀 ProximaDB AutoML Example");

    // Step 1: Configure AutoML
    let automl_config = AutoMLConfig {
        enabled: true,
        min_data_points: 100,
        optimization_interval_secs: 60, // Optimize every minute
        min_improvement_threshold: 5.0, // Require 5% improvement
        max_concurrent_optimizations: 2,
        enable_workload_prediction: true,
        enable_hyperparameter_tuning: true,
        enable_auto_indexing: true,
        enable_quantization_optimization: true,
    };

    // Step 2: Create and start AutoML coordinator
    info!("📊 Starting AutoML Coordinator...");
    let coordinator = AutoMLCoordinator::new(automl_config.clone()).await?;
    coordinator.start().await?;

    // Step 3: Simulate workload and monitor patterns
    info!("🔍 Monitoring workload patterns...");
    let analyzer = WorkloadAnalyzer::new().await?;
    analyzer.start_monitoring().await?;

    // Simulate a changing workload
    simulate_workload_changes(&analyzer).await?;

    // Step 4: Perform manual optimization for demonstration
    info!("⚙️ Running manual optimization...");
    let pipeline = OptimizationPipeline::new(automl_config.clone()).await?;
    pipeline.start().await?;

    // Detect current workload pattern
    let pattern = analyzer.analyze_workload("demo_collection").await?;
    info!("Detected workload pattern: {:?}", pattern);

    // Optimize for the detected pattern
    let optimization_goal = match pattern.clone() {
        WorkloadPattern::ReadHeavy => OptimizationGoal::MinimizeLatency,
        WorkloadPattern::WriteHeavy => OptimizationGoal::MaximizeThroughput,
        WorkloadPattern::Analytics => OptimizationGoal::MaximizeAccuracy,
        _ => OptimizationGoal::Balanced,
    };

    info!("Optimizing for goal: {:?}", optimization_goal);

    let best_config = pipeline
        .optimize(
            "demo_collection",
            optimization_goal,
            OptimizationStrategy::BayesianOptimization { n_iterations: 20 },
            pattern.clone(),
        )
        .await?;

    info!("✅ Optimization complete!");
    info!("  Optimal Index: {}", best_config.index.algorithm);
    info!("  Optimal Quantization: {}", best_config.quantization.level);
    info!("  Optimal Engine: {}", best_config.engine.engine_type);
    info!("  Optimal Cache: {} MB", best_config.cache.cache_size_mb);

    // Step 5: Tune hyperparameters for the selected index
    info!("🎯 Tuning hyperparameters...");
    tune_index_parameters(&best_config.index.algorithm).await?;

    // Step 6: Monitor AutoML metrics
    info!("📈 AutoML Metrics:");
    let status = coordinator.get_status().await;
    info!("  Active optimizations: {}", status.active_optimizations);
    info!("  Total optimizations: {}", status.total_optimizations);
    info!(
        "  Average improvement: {:.2}%",
        status.average_improvement * 100.0
    );

    let metrics = coordinator.get_metrics().await;
    info!("  Predictions made: {}", metrics.predictions_made);
    info!(
        "  Successful optimizations: {}",
        metrics.optimizations_successful
    );

    // Step 7: Demonstrate workload prediction
    info!("🔮 Predicting future workload...");
    let future_pattern = analyzer.predict_pattern("demo_collection", 3600).await?;
    info!("Predicted pattern in 1 hour: {:?}", future_pattern);

    if future_pattern != pattern {
        warn!(
            "⚠️ Workload pattern expected to change from {:?} to {:?}",
            pattern, future_pattern
        );
        info!("Preparing preemptive optimization for future workload...");

        // Preemptively optimize for predicted pattern
        let _future_config = pipeline
            .optimize(
                "demo_collection",
                OptimizationGoal::Balanced,
                OptimizationStrategy::RandomSearch { budget: 10 },
                future_pattern,
            )
            .await?;

        info!("Future-optimized configuration ready for deployment");
    }

    // Let AutoML run for a while
    info!("⏰ AutoML running in background for 10 seconds...");
    sleep(Duration::from_secs(10)).await;

    // Cleanup
    info!("🛑 Stopping AutoML services...");
    coordinator.stop().await?;
    analyzer.stop_monitoring().await?;
    pipeline.stop().await?;

    info!("✨ AutoML example complete!");

    Ok(())
}

/// Simulate changing workload patterns
async fn simulate_workload_changes(
    analyzer: &WorkloadAnalyzer,
) -> Result<(), Box<dyn std::error::Error>> {
    // Phase 1: Read-heavy workload
    info!("Simulating read-heavy workload...");
    for i in 0..30 {
        analyzer
            .record_metric("demo_collection", "reads_per_sec", 1000.0 + i as f64 * 10.0)
            .await?;
        analyzer
            .record_metric("demo_collection", "writes_per_sec", 50.0)
            .await?;
        analyzer
            .record_metric(
                "demo_collection",
                "query_latency_ms",
                5.0 + (i as f64 * 0.1),
            )
            .await?;
        analyzer
            .record_metric("demo_collection", "memory_usage_mb", 512.0)
            .await?;
        analyzer
            .record_metric("demo_collection", "cpu_usage_percent", 45.0)
            .await?;
    }

    // Phase 2: Transition to write-heavy
    info!("Transitioning to write-heavy workload...");
    for i in 0..20 {
        let read_ratio = 1.0 - (i as f64 / 20.0);
        analyzer
            .record_metric("demo_collection", "reads_per_sec", 1000.0 * read_ratio)
            .await?;
        analyzer
            .record_metric(
                "demo_collection",
                "writes_per_sec",
                50.0 + 950.0 * (1.0 - read_ratio),
            )
            .await?;
        analyzer
            .record_metric("demo_collection", "query_latency_ms", 10.0)
            .await?;
        analyzer
            .record_metric("demo_collection", "memory_usage_mb", 768.0)
            .await?;
        analyzer
            .record_metric("demo_collection", "cpu_usage_percent", 60.0)
            .await?;
    }

    // Phase 3: Analytics workload
    info!("Simulating analytics workload...");
    for _ in 0..20 {
        analyzer
            .record_metric("demo_collection", "reads_per_sec", 10.0)
            .await?;
        analyzer
            .record_metric("demo_collection", "writes_per_sec", 5.0)
            .await?;
        analyzer
            .record_metric("demo_collection", "query_latency_ms", 1500.0)
            .await?; // Complex queries
        analyzer
            .record_metric("demo_collection", "memory_usage_mb", 2048.0)
            .await?;
        analyzer
            .record_metric("demo_collection", "cpu_usage_percent", 85.0)
            .await?;
    }

    Ok(())
}

/// Tune hyperparameters for a specific index algorithm
async fn tune_index_parameters(algorithm: &str) -> Result<(), Box<dyn std::error::Error>> {
    let tuner = HyperparameterTuner::new(TuningConfig {
        max_trials: 20,
        timeout_per_trial: 5,
        early_stopping_patience: 5,
        min_improvement: 0.02,
        parallel_trials: true,
        max_parallel_trials: 4,
    })
    .await?;

    // Add parameters based on algorithm
    let params = match algorithm {
        "HNSW" => ProximaDBHyperparameters::hnsw_params(),
        "IVF" => ProximaDBHyperparameters::ivf_params(),
        _ => {
            info!("No specific parameters to tune for {}", algorithm);
            return Ok(());
        }
    };

    for param in params {
        tuner.add_parameter(param).await?;
    }

    // Define objective function (simulated evaluation)
    let objective = |params: HashMap<String, ParameterValue>| async move {
        // In production, this would actually test the configuration
        // Here we simulate an evaluation based on parameter values

        let mut score = 100.0;

        // HNSW scoring
        if let Some(ParameterValue::Integer(m)) = params.get("M") {
            // Prefer moderate M values (16-32)
            score += if (16..=32).contains(m) { 20.0 } else { -10.0 };
        }

        if let Some(ParameterValue::Integer(ef)) = params.get("ef_construction") {
            // Higher ef_construction generally better but more expensive
            score += (*ef as f64).ln() * 5.0;
        }

        // IVF scoring
        if let Some(ParameterValue::Float(nlist)) = params.get("nlist") {
            // Prefer sqrt(n) for nlist where n is dataset size
            let optimal = (10000.0_f64).sqrt(); // Assuming 10k vectors
            score -= (nlist - optimal).abs() / 10.0;
        }

        Ok(score)
    };

    info!("Running hyperparameter tuning with TPE algorithm...");
    let tuner_with_algo = tuner.with_algorithm(TuningAlgorithm::TPE);
    let best_params = tuner_with_algo.tune(objective).await?;

    info!("Best hyperparameters found:");
    for (param_name, param_value) in &best_params {
        match param_value {
            ParameterValue::Integer(v) => info!("  {}: {}", param_name, v),
            ParameterValue::Float(v) => info!("  {}: {:.2}", param_name, v),
            ParameterValue::String(v) => info!("  {}: {}", param_name, v),
            ParameterValue::Boolean(v) => info!("  {}: {}", param_name, v),
        }
    }

    // Get tuning statistics
    let trials = tuner_with_algo.get_trials().await;
    info!("Total trials executed: {}", trials.len());

    if let Some(best_trial) = tuner_with_algo.get_best_trial().await {
        info!("Best trial score: {:.2}", best_trial.score);
        info!("Best trial duration: {} ms", best_trial.duration_ms);
    }

    Ok(())
}

// Example output:
// ```
// 🚀 ProximaDB AutoML Example
// 📊 Starting AutoML Coordinator...
// 🔍 Monitoring workload patterns...
// Simulating read-heavy workload...
// Transitioning to write-heavy workload...
// Simulating analytics workload...
// ⚙️ Running manual optimization...
// Detected workload pattern: Analytics
// Optimizing for goal: MaximizeAccuracy
// ✅ Optimization complete!
//   Optimal Index: HNSW
//   Optimal Quantization: PQ8
//   Optimal Engine: NOVA
//   Optimal Cache: 1024 MB
// 🎯 Tuning hyperparameters...
// Running hyperparameter tuning with TPE algorithm...
// Best hyperparameters found:
//   M: 24
//   ef_construction: 200
//   ef_search: 50
// Total trials executed: 20
// Best trial score: 145.32
// Best trial duration: 12 ms
// 📈 AutoML Metrics:
//   Active optimizations: 0
//   Total optimizations: 1
//   Average improvement: 15.00%
//   Predictions made: 5
//   Successful optimizations: 1
// 🔮 Predicting future workload...
// Predicted pattern in 1 hour: ReadHeavy
// ⚠️ Workload pattern expected to change from Analytics to ReadHeavy
// Preparing preemptive optimization for future workload...
// Future-optimized configuration ready for deployment
// ⏰ AutoML running in background for 10 seconds...
// 🛑 Stopping AutoML services...
// ✨ AutoML example complete!
// ```
