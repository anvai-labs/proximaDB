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

//! # Phase 3 Cross-Workstream Integration Tests
//!
//! Comprehensive integration tests verifying interactions between Phase 3 workstreams:
//!
//! - WS-2: External Catalogs (Delta Lake)
//! - WS-3: Streaming Storage Integration
//! - WS-5: mTLS Infrastructure
//! - WS-6: Integration Tests & Documentation
//!
//! Run with: `cargo test --test phase3_integration_test`
//!
//! For feature-gated tests:
//! - `cargo test --test phase3_integration_test --features delta-lake`

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;

// =============================================================================
// WS-3: Streaming Storage Integration Tests
// =============================================================================

mod streaming_flush_tests {
    use super::*;
    use proximadb::streaming::{
        BackpressureLevel, FlushRetryConfig, SessionConfig, StreamConfig, StreamCoordinator,
    };
    use proximadb_records::{EmbeddingCell, EmbeddingValues, ProximaRecord};

    /// Helper to create test vectors
    fn create_test_vectors(count: usize, start_id: usize, dimension: usize) -> Vec<ProximaRecord> {
        (0..count)
            .map(|i| ProximaRecord {
                oid: format!("vec_{}", start_id + i),
                embeddings: vec![EmbeddingCell::new_fp32(
                    "default",
                    String::new(),
                    dimension as u32,
                    vec![0.1 * (i as f32); dimension],
                )],
                ..Default::default()
            })
            .collect()
    }

    /// Helper config for tests with small buffers
    fn small_buffer_config() -> StreamConfig {
        StreamConfig {
            max_streams: 100,
            default_buffer_size: 256,
            global_rate_limit: 10000,
            flush_interval: Duration::from_millis(50),
            session_timeout: Duration::from_secs(60),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_flush_retry_config_defaults() {
        let config = FlushRetryConfig::default();

        assert_eq!(config.max_retries, 3);
        assert!(config.initial_delay.as_millis() >= 50);
        assert!(config.max_delay.as_secs() >= 1);
        assert!(config.backoff_multiplier >= 1.5);
    }

    #[tokio::test]
    async fn test_flush_retry_config_custom() {
        let config = FlushRetryConfig {
            max_retries: 5,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 2.5,
        };

        assert_eq!(config.max_retries, 5);
        assert_eq!(config.initial_delay.as_millis(), 100);
        assert_eq!(config.max_delay.as_secs(), 10);
        assert!((config.backoff_multiplier - 2.5).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_coordinator_stats_initial() {
        let coordinator = StreamCoordinator::new(small_buffer_config());
        let stats = coordinator.stats();

        assert_eq!(stats.active_sessions, 0);
        assert_eq!(stats.total_buffered_records, 0);
    }

    #[tokio::test]
    async fn test_coordinator_stats_after_session_creation() {
        let coordinator = StreamCoordinator::new(small_buffer_config());

        let _session_id = coordinator
            .create_session("test_collection".to_string(), SessionConfig::default())
            .await
            .expect("Failed to create session");

        let stats = coordinator.stats();
        assert_eq!(stats.active_sessions, 1);
        assert!(stats.sessions_created >= 1);
    }

    #[tokio::test]
    async fn test_coordinator_stats_after_push() {
        let coordinator = StreamCoordinator::new(small_buffer_config());

        let session_id = coordinator
            .create_session("test_collection".to_string(), SessionConfig::default())
            .await
            .expect("Failed to create session");

        let vectors = create_test_vectors(50, 0, 64);
        coordinator
            .push_records(&session_id, vectors)
            .await
            .unwrap();

        let stats = coordinator.stats();
        assert_eq!(stats.total_buffered_records, 50);
        assert!(stats.sessions_created >= 1);
    }

    #[tokio::test]
    async fn test_coordinator_stats_multiple_sessions() {
        let coordinator = StreamCoordinator::new(small_buffer_config());

        let session1 = coordinator
            .create_session("collection_1".to_string(), SessionConfig::default())
            .await
            .unwrap();

        let session2 = coordinator
            .create_session("collection_2".to_string(), SessionConfig::default())
            .await
            .unwrap();

        // Push to both sessions
        let vectors1 = create_test_vectors(30, 0, 64);
        let vectors2 = create_test_vectors(20, 100, 64);

        coordinator.push_records(&session1, vectors1).await.unwrap();
        coordinator.push_records(&session2, vectors2).await.unwrap();

        let stats = coordinator.stats();
        assert_eq!(stats.active_sessions, 2);
        assert_eq!(stats.total_buffered_records, 50);
    }

    #[tokio::test]
    async fn test_streaming_backpressure_levels() {
        let config = StreamConfig {
            default_buffer_size: 64, // Small buffer to trigger backpressure
            ..Default::default()
        };
        let coordinator = StreamCoordinator::new(config);

        let session_id = coordinator
            .create_session("backpressure_test".to_string(), SessionConfig::default())
            .await
            .unwrap();

        // Push vectors until we hit backpressure
        let vectors = create_test_vectors(60, 0, 32);
        let result = coordinator
            .push_records(&session_id, vectors)
            .await
            .unwrap();

        // With 60 items in 64-slot buffer, should have high backpressure
        assert!(result.buffer_percent >= 80);
        assert!(result.backpressure >= BackpressureLevel::High);
    }

    #[tokio::test]
    async fn test_streaming_session_drain() {
        let coordinator = StreamCoordinator::new(small_buffer_config());

        let session_id = coordinator
            .create_session("drain_test".to_string(), SessionConfig::default())
            .await
            .unwrap();

        // Push 100 vectors
        let vectors = create_test_vectors(100, 0, 64);
        coordinator
            .push_records(&session_id, vectors)
            .await
            .unwrap();

        // Drain in batches
        let batch1 = coordinator.drain_records(&session_id, 30).unwrap();
        assert_eq!(batch1.len(), 30);

        let batch2 = coordinator.drain_records(&session_id, 50).unwrap();
        assert_eq!(batch2.len(), 50);

        let batch3 = coordinator.drain_records(&session_id, 50).unwrap();
        assert_eq!(batch3.len(), 20); // Only 20 remaining
    }

    #[tokio::test]
    async fn test_streaming_session_close_and_stats() {
        let coordinator = StreamCoordinator::new(small_buffer_config());

        let session_id = coordinator
            .create_session("close_test".to_string(), SessionConfig::default())
            .await
            .unwrap();

        let stats_before = coordinator.stats();
        assert_eq!(stats_before.active_sessions, 1);

        coordinator.close_session(&session_id);

        let stats_after = coordinator.stats();
        assert_eq!(stats_after.active_sessions, 0);
    }

    #[tokio::test]
    async fn test_streaming_rate_limiting() {
        let config = StreamConfig {
            global_rate_limit: 50, // Very low rate for testing
            ..Default::default()
        };
        let coordinator = StreamCoordinator::new(config);

        let session_id = coordinator
            .create_session("rate_limit_test".to_string(), SessionConfig::default())
            .await
            .unwrap();

        // First push should succeed (within burst allowance)
        let vectors1 = create_test_vectors(50, 0, 32);
        let result1 = coordinator.push_records(&session_id, vectors1).await;
        assert!(result1.is_ok());

        // Immediate second push should be rate limited
        let vectors2 = create_test_vectors(50, 100, 32);
        let result2 = coordinator.push_records(&session_id, vectors2).await;
        assert!(result2.is_err());
    }

    #[tokio::test]
    async fn test_streaming_concurrent_sessions() {
        let coordinator = Arc::new(StreamCoordinator::new(StreamConfig {
            default_buffer_size: 4096,
            global_rate_limit: 1_000_000,
            ..Default::default()
        }));

        let mut handles = vec![];

        // Create 5 concurrent sessions
        for i in 0..5 {
            let coord = Arc::clone(&coordinator);
            handles.push(tokio::spawn(async move {
                let session_id = coord
                    .create_session(format!("concurrent_{}", i), SessionConfig::default())
                    .await
                    .unwrap();

                // Push vectors
                let vectors = create_test_vectors(100, i * 1000, 32);
                coord.push_records(&session_id, vectors).await.unwrap();

                session_id
            }));
        }

        let session_ids: Vec<_> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        let stats = coordinator.stats();
        assert_eq!(stats.active_sessions, 5);
        assert_eq!(stats.total_buffered_records, 500);

        // Clean up
        for sid in session_ids {
            coordinator.close_session(&sid);
        }
    }

    #[tokio::test]
    async fn test_streaming_session_info() {
        let coordinator = StreamCoordinator::new(small_buffer_config());

        let session_id = coordinator
            .create_session("info_test".to_string(), SessionConfig::default())
            .await
            .unwrap();

        // Initial info
        let info = coordinator.get_session_info(&session_id).unwrap();
        assert_eq!(info.collection, "info_test");
        assert_eq!(info.buffer_len, 0);
        assert_eq!(info.records_received, 0);

        // Push some vectors
        let vectors = create_test_vectors(25, 0, 32);
        coordinator
            .push_records(&session_id, vectors)
            .await
            .unwrap();

        let info_after = coordinator.get_session_info(&session_id).unwrap();
        assert_eq!(info_after.buffer_len, 25);
        assert_eq!(info_after.records_received, 25);
    }
}

// =============================================================================
// WS-1: Graph WAL Recovery Tests - ORION Engine
// =============================================================================

mod graph_wal_recovery_tests {
    use super::*;
    use proximadb::graph::engines::orion::OrionGraphEngine;
    use proximadb::graph::{Edge, Node, engines::GraphEngine};

    fn create_test_node(id: &str, label: &str) -> Node {
        Node {
            id: id.to_string(),
            labels: vec![label.to_string()],
            properties: HashMap::new(),
            embedding: None,
            created_at_ms: chrono::Utc::now().timestamp_millis(),
            updated_at_ms: chrono::Utc::now().timestamp_millis(),
        }
    }

    fn create_test_edge(id: &str, from: &str, to: &str, edge_type: &str) -> Edge {
        Edge {
            id: id.to_string(),
            from_node_id: from.to_string(),
            to_node_id: to.to_string(),
            edge_type: edge_type.to_string(),
            properties: HashMap::new(),
            weight: Some(1.0),
            created_at_ms: chrono::Utc::now().timestamp_millis(),
            updated_at_ms: chrono::Utc::now().timestamp_millis(),
        }
    }

    #[tokio::test]
    async fn test_orion_node_insert_wal() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let base_url = format!("file://{}", temp_dir.path().display());

        let engine = OrionGraphEngine::with_persistence_for_graph(
            "insert_test".to_string(),
            base_url.clone(),
            true,
            proximadb::graph::unified_wal_factory(),
        )
        .await
        .expect("Failed to create engine");

        // Insert nodes
        let node = create_test_node("test_node_1", "TestLabel");
        let inserted = engine.insert_node(node).await.expect("Insert failed");

        assert_eq!(inserted.id, "test_node_1");

        // Flush WAL
        engine.flush_wal().await.expect("Flush failed");
    }

    #[tokio::test]
    async fn test_orion_edge_insert_wal() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let base_url = format!("file://{}", temp_dir.path().display());

        let engine = OrionGraphEngine::with_persistence_for_graph(
            "edge_test".to_string(),
            base_url.clone(),
            true,
            proximadb::graph::unified_wal_factory(),
        )
        .await
        .expect("Failed to create engine");

        // Insert nodes first
        let node1 = create_test_node("edge_node_1", "Person");
        let node2 = create_test_node("edge_node_2", "Person");
        engine
            .insert_node(node1)
            .await
            .expect("Insert node1 failed");
        engine
            .insert_node(node2)
            .await
            .expect("Insert node2 failed");

        // Insert edge
        let edge = create_test_edge("test_edge_1", "edge_node_1", "edge_node_2", "KNOWS");
        let inserted = engine.insert_edge(edge).await.expect("Insert edge failed");

        assert_eq!(inserted.id, "test_edge_1");

        engine.flush_wal().await.expect("Flush failed");
    }

    #[tokio::test]
    async fn test_orion_update_node_wal() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let base_url = format!("file://{}", temp_dir.path().display());

        let engine = OrionGraphEngine::with_persistence_for_graph(
            "update_test".to_string(),
            base_url.clone(),
            true,
            proximadb::graph::unified_wal_factory(),
        )
        .await
        .expect("Failed to create engine");

        // Insert node
        let node = create_test_node("upd_node", "Original");
        engine.insert_node(node).await.expect("Insert failed");

        // Update node
        let mut updated_node =
            (*engine.get_node(&"upd_node".to_string()).unwrap().unwrap()).clone();
        updated_node.labels.push("Updated".to_string());
        let updated = engine
            .update_node(updated_node)
            .await
            .expect("Update failed");

        assert!(updated.labels.contains(&"Updated".to_string()));

        engine.flush_wal().await.expect("Flush failed");
    }

    #[tokio::test]
    async fn test_orion_delete_node_wal() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let base_url = format!("file://{}", temp_dir.path().display());

        let engine = OrionGraphEngine::with_persistence_for_graph(
            "delete_test".to_string(),
            base_url.clone(),
            true,
            proximadb::graph::unified_wal_factory(),
        )
        .await
        .expect("Failed to create engine");

        // Insert nodes
        let node1 = create_test_node("del_node_1", "ToKeep");
        let node2 = create_test_node("del_node_2", "ToDelete");
        engine
            .insert_node(node1)
            .await
            .expect("Insert node1 failed");
        engine
            .insert_node(node2)
            .await
            .expect("Insert node2 failed");

        // Delete one node
        let deleted = engine
            .delete_node(&"del_node_2".to_string())
            .await
            .expect("Delete failed");
        assert!(deleted.is_some());

        // Verify deletion
        let missing = engine
            .get_node(&"del_node_2".to_string())
            .expect("Get failed");
        assert!(missing.is_none());

        engine.flush_wal().await.expect("Flush failed");
    }

    #[tokio::test]
    async fn test_orion_recovery_from_wal() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let base_url = format!("file://{}", temp_dir.path().display());

        // Phase 1: Insert data and flush
        {
            let engine = OrionGraphEngine::with_persistence_for_graph(
                "recovery_test".to_string(),
                base_url.clone(),
                true,
                proximadb::graph::unified_wal_factory(),
            )
            .await
            .expect("Failed to create engine");

            for i in 0..10 {
                let node = create_test_node(&format!("rec_node_{}", i), "RecoveryTest");
                engine.insert_node(node).await.expect("Insert failed");
            }

            engine.flush_wal().await.expect("Flush failed");
        }

        // Phase 2: Recover and verify
        {
            let engine = OrionGraphEngine::with_persistence_for_graph(
                "recovery_test".to_string(),
                base_url.clone(),
                true,
                proximadb::graph::unified_wal_factory(),
            )
            .await
            .expect("Failed to create engine for recovery");

            engine.recover().await.expect("Recovery failed");

            // Verify nodes were recovered
            for i in 0..10 {
                let node_id = format!("rec_node_{}", i);
                let recovered = engine.get_node(&node_id).expect("Get failed");
                assert!(
                    recovered.is_some(),
                    "Node {} should be recovered after WAL replay",
                    node_id
                );
            }
        }
    }

    #[tokio::test]
    async fn test_orion_recovery_with_updates() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let base_url = format!("file://{}", temp_dir.path().display());

        // Phase 1: Insert and update
        {
            let engine = OrionGraphEngine::with_persistence_for_graph(
                "update_recovery".to_string(),
                base_url.clone(),
                true,
                proximadb::graph::unified_wal_factory(),
            )
            .await
            .expect("Failed to create engine");

            let node = create_test_node("upd_rec_node", "Original");
            engine.insert_node(node).await.expect("Insert failed");

            let mut updated = (*engine
                .get_node(&"upd_rec_node".to_string())
                .unwrap()
                .unwrap())
            .clone();
            updated.labels.push("Modified".to_string());
            engine.update_node(updated).await.expect("Update failed");

            engine.flush_wal().await.expect("Flush failed");
        }

        // Phase 2: Recover and verify update persisted
        {
            let engine = OrionGraphEngine::with_persistence_for_graph(
                "update_recovery".to_string(),
                base_url.clone(),
                true,
                proximadb::graph::unified_wal_factory(),
            )
            .await
            .expect("Failed to create engine for recovery");

            engine.recover().await.expect("Recovery failed");

            let recovered = engine
                .get_node(&"upd_rec_node".to_string())
                .expect("Get failed")
                .expect("Node should exist");

            assert!(
                recovered.labels.contains(&"Modified".to_string()),
                "Update should persist after recovery. Labels: {:?}",
                recovered.labels
            );
        }
    }

    #[tokio::test]
    async fn test_orion_recovery_with_deletes() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let base_url = format!("file://{}", temp_dir.path().display());

        // Phase 1: Insert and delete
        {
            let engine = OrionGraphEngine::with_persistence_for_graph(
                "delete_recovery".to_string(),
                base_url.clone(),
                true,
                proximadb::graph::unified_wal_factory(),
            )
            .await
            .expect("Failed to create engine");

            let node1 = create_test_node("del_rec_keep", "ToKeep");
            let node2 = create_test_node("del_rec_remove", "ToRemove");
            engine.insert_node(node1).await.expect("Insert failed");
            engine.insert_node(node2).await.expect("Insert failed");

            engine
                .delete_node(&"del_rec_remove".to_string())
                .await
                .expect("Delete failed");

            engine.flush_wal().await.expect("Flush failed");
        }

        // Phase 2: Recover and verify delete persisted
        {
            let engine = OrionGraphEngine::with_persistence_for_graph(
                "delete_recovery".to_string(),
                base_url.clone(),
                true,
                proximadb::graph::unified_wal_factory(),
            )
            .await
            .expect("Failed to create engine for recovery");

            engine.recover().await.expect("Recovery failed");

            let kept = engine
                .get_node(&"del_rec_keep".to_string())
                .expect("Get failed");
            let deleted = engine
                .get_node(&"del_rec_remove".to_string())
                .expect("Get failed");

            assert!(kept.is_some(), "Kept node should exist after recovery");
            assert!(
                deleted.is_none(),
                "Deleted node should not exist after recovery"
            );
        }
    }

    #[tokio::test]
    async fn test_orion_edge_recovery() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let base_url = format!("file://{}", temp_dir.path().display());

        // Phase 1: Create graph with edges
        {
            let engine = OrionGraphEngine::with_persistence_for_graph(
                "edge_recovery".to_string(),
                base_url.clone(),
                true,
                proximadb::graph::unified_wal_factory(),
            )
            .await
            .expect("Failed to create engine");

            // Create chain: A -> B -> C
            for c in ['A', 'B', 'C'] {
                let node = create_test_node(&format!("chain_{}", c), "ChainNode");
                engine.insert_node(node).await.expect("Insert failed");
            }

            let edge_ab = create_test_edge("e_ab", "chain_A", "chain_B", "NEXT");
            let edge_bc = create_test_edge("e_bc", "chain_B", "chain_C", "NEXT");
            engine
                .insert_edge(edge_ab)
                .await
                .expect("Insert edge failed");
            engine
                .insert_edge(edge_bc)
                .await
                .expect("Insert edge failed");

            engine.flush_wal().await.expect("Flush failed");
        }

        // Phase 2: Recover and verify edges
        {
            let engine = OrionGraphEngine::with_persistence_for_graph(
                "edge_recovery".to_string(),
                base_url.clone(),
                true,
                proximadb::graph::unified_wal_factory(),
            )
            .await
            .expect("Failed to create engine for recovery");

            engine.recover().await.expect("Recovery failed");

            // Verify edges exist
            let edge_ab = engine
                .get_edge(&"e_ab".to_string())
                .expect("Get edge failed");
            let edge_bc = engine
                .get_edge(&"e_bc".to_string())
                .expect("Get edge failed");

            assert!(edge_ab.is_some(), "Edge A->B should be recovered");
            assert!(edge_bc.is_some(), "Edge B->C should be recovered");
        }
    }

    #[tokio::test]
    async fn test_orion_graph_traversal_after_recovery() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let base_url = format!("file://{}", temp_dir.path().display());

        // Phase 1: Create and persist graph
        {
            let engine = OrionGraphEngine::with_persistence_for_graph(
                "traversal_recovery".to_string(),
                base_url.clone(),
                true,
                proximadb::graph::unified_wal_factory(),
            )
            .await
            .expect("Failed to create engine");

            // Create star pattern: center -> spoke1, spoke2, spoke3
            let center = create_test_node("center", "Hub");
            engine.insert_node(center).await.expect("Insert failed");

            for i in 1..=3 {
                let spoke = create_test_node(&format!("spoke_{}", i), "Spoke");
                engine.insert_node(spoke).await.expect("Insert failed");

                let edge = create_test_edge(
                    &format!("e_center_spoke_{}", i),
                    "center",
                    &format!("spoke_{}", i),
                    "CONNECTS",
                );
                engine.insert_edge(edge).await.expect("Insert edge failed");
            }

            engine.flush_wal().await.expect("Flush failed");
        }

        // Phase 2: Recover and verify traversal works
        {
            let engine = OrionGraphEngine::with_persistence_for_graph(
                "traversal_recovery".to_string(),
                base_url.clone(),
                true,
                proximadb::graph::unified_wal_factory(),
            )
            .await
            .expect("Failed to create engine for recovery");

            engine.recover().await.expect("Recovery failed");

            // Test traversal from center
            let neighbors = engine
                .get_neighbors(&"center".to_string(), None)
                .expect("Get neighbors failed");

            assert_eq!(
                neighbors.len(),
                3,
                "Center should have 3 neighbors after recovery"
            );
        }
    }
}
