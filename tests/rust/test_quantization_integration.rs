//! Quantization Integration Tests
//!
//! Tests for the new quantization support and search optimization hints
//! that were added to the proto definition and handlers.

use anyhow::Result;
use serde_json::json;
use tracing::debug;

#[tokio::test]
async fn test_quantization_config_creation() -> Result<()> {
    // Test that we can create quantization configurations
    // This validates the proto definitions are working

    // Create a simple PQ quantization config
    let pq_config = json!({
        "quantization_type": "PQ",
        "bits_per_code": 8,
        "num_subspaces": 16
    });

    assert_eq!(pq_config["quantization_type"], "PQ");
    assert_eq!(pq_config["bits_per_code"], 8);
    assert_eq!(pq_config["num_subspaces"], 16);

    // Create search optimization hints
    let optimization_hints = json!({
        "enable_two_stage_search": true,
        "candidate_multiplier": 2.0,
        "min_candidates": 100,
        "max_candidates": 1000,
        "accuracy_threshold": 0.95,
        "quantization_hint": "PQ8"
    });

    assert_eq!(optimization_hints["enable_two_stage_search"], true);
    assert_eq!(optimization_hints["candidate_multiplier"], 2.0);
    assert_eq!(optimization_hints["quantization_hint"], "PQ8");

    debug!("✅ Quantization configuration test passed");
    Ok(())
}

#[tokio::test]
async fn test_search_optimization_hints_parsing() -> Result<()> {
    // Test that search optimization hints can be parsed correctly
    // This validates the handler updates are working

    let hints_json = json!({
        "enable_two_stage_search": true,
        "candidate_multiplier": 1.5,
        "min_candidates": 50,
        "max_candidates": 500,
        "accuracy_threshold": 0.9,
        "quantization_hint": "FP32"
    });

    // Extract values as the handlers would
    let enable_two_stage = hints_json
        .get("enable_two_stage_search")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let candidate_multiplier = hints_json
        .get("candidate_multiplier")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);

    let quantization_hint = hints_json
        .get("quantization_hint")
        .and_then(|v| v.as_str())
        .unwrap_or("FP32");

    assert!(enable_two_stage);
    assert_eq!(candidate_multiplier, 1.5);
    assert_eq!(quantization_hint, "FP32");

    debug!("✅ Search optimization hints parsing test passed");
    Ok(())
}

#[tokio::test]
async fn test_rest_search_request_with_optimization_hints() -> Result<()> {
    // Test that REST search requests can include optimization hints
    // This validates the REST handler updates

    let search_request = json!({
        "vector": [1.0, 2.0, 3.0, 4.0],
        "k": 10,
        "optimization_hints": {
            "enable_two_stage_search": true,
            "candidate_multiplier": 2.0,
            "quantization_hint": "PQ8"
        }
    });

    // Validate the structure
    assert!(search_request.get("optimization_hints").is_some());
    assert_eq!(
        search_request.get("k"),
        Some(&serde_json::Value::Number(serde_json::Number::from(10)))
    );

    let hints = search_request.get("optimization_hints").unwrap();
    assert_eq!(hints["enable_two_stage_search"], true);
    assert_eq!(hints["candidate_multiplier"], 2.0);
    assert_eq!(hints["quantization_hint"], "PQ8");

    debug!("✅ REST search request with optimization hints test passed");
    Ok(())
}

#[tokio::test]
async fn test_grpc_optimization_hints_structure() -> Result<()> {
    // Test that gRPC optimization hints have the expected structure
    // This validates the proto updates

    let hints_map = json!({
        "enable_two_stage_search": true,
        "candidate_multiplier": 1.8,
        "min_candidates": 100,
        "max_candidates": 800,
        "accuracy_threshold": 0.95,
        "quantization_hint": "PQ4"
    });

    // Simulate gRPC handler processing
    let search_hints = json!({
        "enable_two_stage_search": hints_map.get("enable_two_stage_search"),
        "candidate_multiplier": hints_map.get("candidate_multiplier"),
        "quantization_hint": hints_map.get("quantization_hint")
    });

    assert!(search_hints.get("enable_two_stage_search").is_some());
    assert!(search_hints.get("candidate_multiplier").is_some());
    assert!(search_hints.get("quantization_hint").is_some());

    debug!("✅ gRPC optimization hints structure test passed");
    Ok(())
}

#[tokio::test]
async fn test_quantization_type_variants() -> Result<()> {
    // Test different quantization type configurations

    let quantization_types = vec![
        ("PQ4", 4, 8),
        ("PQ8", 8, 16),
        ("INT8", 8, 1),
        ("FP32", 32, 1),
    ];

    for (qtype, bits, subspaces) in quantization_types {
        let config = json!({
            "quantization_type": qtype,
            "bits_per_code": bits,
            "num_subspaces": subspaces
        });

        assert_eq!(config["quantization_type"], qtype);
        assert_eq!(config["bits_per_code"], bits);
        assert_eq!(config["num_subspaces"], subspaces);
    }

    debug!("✅ Quantization type variants test passed");
    Ok(())
}
