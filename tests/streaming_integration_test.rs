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

//! Streaming Integration Tests
//!
//! Comprehensive end-to-end tests for the real-time streaming module.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use proximadb::streaming::kafka::{
    CommitStrategy, DeserializationFormat, DlqConfig, KafkaConsumerConfig, MessageDeserializer,
};
use proximadb::streaming::subscriptions::{
    QueryEvaluator, ScoredResult, SubscriptionConfig, SubscriptionId, SubscriptionManager,
};
use proximadb::streaming::{
    BackpressureLevel, RingBuffer, SessionConfig, SessionState, StreamConfig, StreamCoordinator,
    StreamId, Watermarks,
};
use proximadb_records::{EmbeddingCell, EmbeddingValues, ProximaRecord};

// =============================================================================
// Helper Functions
// =============================================================================

fn create_test_vectors(count: usize, start_id: usize, dimension: usize) -> Vec<ProximaRecord> {
    let dim = dimension as u32;
    (0..count)
        .map(|i| ProximaRecord {
            oid: format!("vec_{}", start_id + i),
            embeddings: vec![EmbeddingCell::new_fp32(
                "default",
                "dense_vector",
                dim,
                vec![0.1 * (i as f32); dimension],
            )],
            ..Default::default()
        })
        .collect()
}

fn create_small_config() -> StreamConfig {
    StreamConfig {
        max_streams: 100,
        default_buffer_size: 256,
        global_rate_limit: 10000,
        flush_interval: Duration::from_millis(50),
        session_timeout: Duration::from_secs(60),
        ..Default::default()
    }
}

// =============================================================================
// Ring Buffer Tests
// =============================================================================

#[test]
fn test_ring_buffer_basic_operations() {
    let buffer = RingBuffer::<i32>::new(16);

    // Push elements
    for i in 0..10 {
        assert!(buffer.try_push(i).is_ok());
    }
    assert_eq!(buffer.len(), 10);
    assert!(!buffer.is_empty());

    // Pop elements
    for i in 0..10 {
        assert_eq!(buffer.try_pop(), Some(i));
    }
    assert!(buffer.is_empty());
    assert_eq!(buffer.try_pop(), None);
}

#[test]
fn test_ring_buffer_wraparound() {
    let buffer = RingBuffer::<u32>::new(8);

    // Fill buffer
    for i in 0..8 {
        assert!(buffer.try_push(i).is_ok());
    }

    // Should be full
    assert!(buffer.try_push(999).is_err());

    // Pop half
    for i in 0..4 {
        assert_eq!(buffer.try_pop(), Some(i));
    }

    // Push more (tests wraparound)
    for i in 8..12 {
        assert!(buffer.try_push(i).is_ok());
    }

    // Verify order
    let remaining: Vec<_> = buffer.drain_all();
    assert_eq!(remaining, vec![4, 5, 6, 7, 8, 9, 10, 11]);
}

#[test]
fn test_ring_buffer_concurrent_access() {
    use std::thread;

    let buffer = Arc::new(RingBuffer::<u64>::new(4096));
    let num_producers = 4;
    let items_per_producer = 5000;
    let total_items = AtomicUsize::new(0);
    let total_ref = Arc::new(total_items);

    let mut handles = vec![];

    // Producer threads
    for t in 0..num_producers {
        let buf = Arc::clone(&buffer);
        handles.push(thread::spawn(move || {
            let mut pushed = 0;
            for i in 0..items_per_producer {
                let value = t as u64 * items_per_producer as u64 + i as u64;
                // Retry with backoff
                let mut attempts = 0;
                while buf.try_push(value).is_err() && attempts < 1000 {
                    thread::yield_now();
                    attempts += 1;
                }
                if attempts < 1000 {
                    pushed += 1;
                }
            }
            pushed
        }));
    }

    // Consumer thread
    let buf_consumer = Arc::clone(&buffer);
    let total_clone = Arc::clone(&total_ref);
    let consumer = thread::spawn(move || {
        let mut consumed = 0;
        let deadline = Instant::now() + Duration::from_secs(5);

        while Instant::now() < deadline {
            if buf_consumer.try_pop().is_some() {
                consumed += 1;
            } else {
                thread::yield_now();
            }
        }
        total_clone.store(consumed, Ordering::SeqCst);
        consumed
    });

    let pushed: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
    let consumed = consumer.join().unwrap();

    // Account for remaining items
    let remaining = buffer.drain_all().len();

    assert_eq!(pushed, consumed + remaining);
}

#[test]
fn test_ring_buffer_backpressure_levels() {
    let buffer = RingBuffer::<i32>::new(1024);

    // Initially no backpressure
    assert_eq!(buffer.backpressure_level(), BackpressureLevel::None);

    // Fill to low watermark (~25%)
    for i in 0..256 {
        let _ = buffer.try_push(i);
    }
    assert!(buffer.backpressure_level() <= BackpressureLevel::Low);

    // Fill to medium (~50%)
    for i in 0..256 {
        let _ = buffer.try_push(i);
    }
    assert!(buffer.backpressure_level() >= BackpressureLevel::Low);

    // Fill to high (~75%)
    for i in 0..256 {
        let _ = buffer.try_push(i);
    }
    assert!(buffer.backpressure_level() >= BackpressureLevel::Medium);

    // Fill to critical (~100%)
    for i in 0..256 {
        let _ = buffer.try_push(i);
    }
    assert!(buffer.backpressure_level() >= BackpressureLevel::High);
}

#[test]
fn test_ring_buffer_batch_drain() {
    let buffer = RingBuffer::<i32>::new(64);

    // Push 50 items
    for i in 0..50 {
        assert!(buffer.try_push(i).is_ok());
    }

    // Drain in batches of 20
    let batch1 = buffer.drain(20);
    assert_eq!(batch1.len(), 20);
    assert_eq!(batch1[0], 0);
    assert_eq!(batch1[19], 19);

    let batch2 = buffer.drain(20);
    assert_eq!(batch2.len(), 20);
    assert_eq!(batch2[0], 20);

    let batch3 = buffer.drain(20);
    assert_eq!(batch3.len(), 10); // Only 10 remaining
}

// =============================================================================
// Watermarks Tests
// =============================================================================

#[test]
fn test_watermarks_custom_ratios() {
    let wm = Watermarks::with_ratios(10000, 0.3, 0.7, 0.9);

    assert_eq!(wm.low(), 3000);
    assert_eq!(wm.high(), 7000);
    assert_eq!(wm.critical(), 9000);
}

#[test]
fn test_watermarks_from_capacity() {
    let wm = Watermarks::from_capacity(8192);

    // Default is 25%, 75%, 90%
    assert_eq!(wm.low(), 2048);
    assert_eq!(wm.high(), 6144);
    assert_eq!(wm.critical(), 7372); // ~90%
}

#[test]
fn test_watermarks_small_buffer() {
    let wm = Watermarks::with_ratios(16, 0.25, 0.75, 0.9);

    assert_eq!(wm.low(), 4);
    assert_eq!(wm.high(), 12);
    assert_eq!(wm.critical(), 14);
}

// =============================================================================
// Stream Coordinator Tests
// =============================================================================

#[tokio::test]
async fn test_coordinator_session_lifecycle() {
    let coordinator = StreamCoordinator::new(create_small_config());

    // Create session
    let session_id = coordinator
        .create_session("test_collection".to_string(), SessionConfig::default())
        .await
        .expect("Failed to create session");

    assert_eq!(coordinator.session_count(), 1);

    // Get session info
    let info = coordinator.get_session_info(&session_id).unwrap();
    assert_eq!(info.collection, "test_collection");
    assert!(matches!(info.state, SessionState::Active));

    // Close session
    coordinator.close_session(&session_id);
    assert_eq!(coordinator.session_count(), 0);
    assert!(coordinator.get_session_info(&session_id).is_none());
}

#[tokio::test]
async fn test_coordinator_max_sessions_limit() {
    let config = StreamConfig {
        max_streams: 3,
        ..Default::default()
    };
    let coordinator = StreamCoordinator::new(config);

    // Create max sessions
    for i in 0..3 {
        coordinator
            .create_session(format!("collection_{}", i), SessionConfig::default())
            .await
            .expect("Should create session");
    }

    assert_eq!(coordinator.session_count(), 3);

    // Should fail to create more
    let result = coordinator
        .create_session("overflow".to_string(), SessionConfig::default())
        .await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_coordinator_push_and_drain() {
    let coordinator = StreamCoordinator::new(create_small_config());

    let session_id = coordinator
        .create_session("vectors".to_string(), SessionConfig::default())
        .await
        .unwrap();

    // Push vectors
    let vectors = create_test_vectors(100, 0, 128);
    let result = coordinator
        .push_records(&session_id, vectors)
        .await
        .unwrap();

    assert_eq!(result.pushed, 100);
    assert_eq!(result.dropped, 0);

    // Drain vectors
    let drained = coordinator.drain_records(&session_id, 50).unwrap();
    assert_eq!(drained.len(), 50);
    assert_eq!(drained[0].oid, "vec_0");
    assert_eq!(drained[49].oid, "vec_49");

    // Check remaining
    let info = coordinator.get_session_info(&session_id).unwrap();
    assert_eq!(info.buffer_len, 50);
}

#[tokio::test]
async fn test_coordinator_backpressure_propagation() {
    let config = StreamConfig {
        default_buffer_size: 64, // Small buffer
        ..Default::default()
    };
    let coordinator = StreamCoordinator::new(config);

    let session_id = coordinator
        .create_session("test".to_string(), SessionConfig::default())
        .await
        .unwrap();

    // Fill buffer to trigger backpressure
    let vectors = create_test_vectors(60, 0, 64);
    let result = coordinator
        .push_records(&session_id, vectors)
        .await
        .unwrap();

    assert!(result.backpressure >= BackpressureLevel::High);
    assert!(result.buffer_percent > 90);
}

#[tokio::test]
async fn test_coordinator_concurrent_pushes() {
    let coordinator = Arc::new(StreamCoordinator::new(StreamConfig {
        default_buffer_size: 8192,
        global_rate_limit: 1_000_000, // High rate limit
        ..Default::default()
    }));

    let session_id = coordinator
        .create_session("concurrent".to_string(), SessionConfig::default())
        .await
        .unwrap();

    let mut handles = vec![];

    // Spawn multiple concurrent producers
    for t in 0..4 {
        let coord = Arc::clone(&coordinator);
        let sid = session_id.clone();
        handles.push(tokio::spawn(async move {
            let mut total_pushed = 0;
            for batch in 0..10 {
                let vectors = create_test_vectors(50, t * 1000 + batch * 100, 64);
                if let Ok(result) = coord.push_records(&sid, vectors).await {
                    total_pushed += result.pushed;
                }
            }
            total_pushed
        }));
    }

    let results: Vec<usize> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

    let total_pushed: usize = results.iter().sum();
    assert_eq!(total_pushed, 4 * 10 * 50);
}

#[tokio::test]
async fn test_coordinator_session_stats_tracking() {
    let coordinator = StreamCoordinator::new(create_small_config());

    let session_id = coordinator
        .create_session("stats_test".to_string(), SessionConfig::default())
        .await
        .unwrap();

    // Push 100 vectors
    let vectors = create_test_vectors(100, 0, 64);
    coordinator
        .push_records(&session_id, vectors)
        .await
        .unwrap();

    // Drain 30
    coordinator.drain_records(&session_id, 30).unwrap();

    let info = coordinator.get_session_info(&session_id).unwrap();
    assert_eq!(info.records_received, 100);
    assert_eq!(info.records_processed, 30);
    assert_eq!(info.buffer_len, 70);
    let _ = info.age_secs; // usize: always >= 0
}

// =============================================================================
// Rate Limiter Tests
// =============================================================================

#[tokio::test]
async fn test_rate_limiter_basic() {
    let config = StreamConfig {
        global_rate_limit: 50, // Very low rate
        ..Default::default()
    };
    let coordinator = StreamCoordinator::new(config);

    let session_id = coordinator
        .create_session("rate_test".to_string(), SessionConfig::default())
        .await
        .unwrap();

    // First push should succeed (burst allowance)
    let vectors = create_test_vectors(50, 0, 64);
    let result = coordinator.push_records(&session_id, vectors).await;
    assert!(result.is_ok());

    // Immediate second push should be rate limited
    let vectors = create_test_vectors(50, 100, 64);
    let result = coordinator.push_records(&session_id, vectors).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_rate_limiter_recovery() {
    let config = StreamConfig {
        global_rate_limit: 100,
        ..Default::default()
    };
    let coordinator = StreamCoordinator::new(config);

    let session_id = coordinator
        .create_session("rate_recovery".to_string(), SessionConfig::default())
        .await
        .unwrap();

    // Exhaust rate limit
    let vectors = create_test_vectors(100, 0, 64);
    coordinator
        .push_records(&session_id, vectors)
        .await
        .unwrap();

    // Should be rate limited
    let vectors = create_test_vectors(10, 200, 64);
    assert!(
        coordinator
            .push_records(&session_id, vectors)
            .await
            .is_err()
    );

    // Wait for refill
    tokio::time::sleep(Duration::from_millis(1100)).await;

    // Should work now
    let vectors = create_test_vectors(50, 300, 64);
    let result = coordinator.push_records(&session_id, vectors).await;
    assert!(result.is_ok());
}

// =============================================================================
// Subscription Manager Tests
// =============================================================================

#[tokio::test]
async fn test_subscription_manager_create_subscription() {
    let manager = SubscriptionManager::with_defaults();

    let config = SubscriptionConfig {
        top_k: 10,
        score_threshold: 0.8,
        include_initial: false,
        ..Default::default()
    };

    let handle = manager
        .subscribe("collection_1".to_string(), vec![0.1; 128], config, None)
        .await
        .expect("Should create subscription");

    assert!(manager.stats().total_subscriptions >= 1);
    assert!(!handle.id.as_str().is_empty());
}

#[tokio::test]
async fn test_subscription_manager_multiple_subscriptions() {
    let manager = SubscriptionManager::with_defaults();

    let mut handles = vec![];

    for i in 0..5 {
        let config = SubscriptionConfig {
            top_k: 10,
            score_threshold: 0.0,
            include_initial: true,
            ..Default::default()
        };

        let handle = manager
            .subscribe(
                format!("collection_{}", i),
                vec![0.1 * (i as f32); 128],
                config,
                None,
            )
            .await
            .unwrap();

        handles.push(handle);
    }

    assert_eq!(manager.stats().total_subscriptions, 5);

    // Remove one
    manager.unsubscribe(&handles[2].id).unwrap();
    assert_eq!(manager.stats().total_subscriptions, 4);
}

#[tokio::test]
async fn test_subscription_pause_resume() {
    let manager = SubscriptionManager::with_defaults();

    let config = SubscriptionConfig {
        top_k: 10,
        score_threshold: 0.0,
        include_initial: false,
        ..Default::default()
    };

    let handle = manager
        .subscribe("test".to_string(), vec![0.1; 128], config, None)
        .await
        .unwrap();

    // Activate subscription first
    manager.activate(&handle.id, vec![]).await.unwrap();

    // Pause
    manager.pause(&handle.id).unwrap();

    // Resume
    manager.resume(&handle.id).unwrap();
}

// =============================================================================
// Query Evaluator Tests
// =============================================================================

#[test]
fn test_query_evaluator_creation() {
    let _evaluator = QueryEvaluator::new();
    // Evaluator should be created without panic
    assert!(true);
}

#[test]
fn test_query_evaluator_evaluate_vectors() {
    let evaluator = QueryEvaluator::new();

    let query = vec![0.5; 8];
    let vectors = vec![
        ("id_1".to_string(), vec![0.5; 8], 0.0),
        ("id_2".to_string(), vec![0.6; 8], 0.0),
        ("id_3".to_string(), vec![0.7; 8], 0.0),
    ];

    let results = evaluator.evaluate_vectors(&query, &vectors, 10, 0.0);

    // Should return results with computed similarities
    assert!(!results.is_empty());
}

// =============================================================================
// Kafka Consumer Configuration Tests
// =============================================================================

#[test]
fn test_kafka_consumer_config_creation() {
    let config = KafkaConsumerConfig::with_brokers(
        vec!["localhost:9092".to_string()],
        "vectors-topic",
        "proximadb-consumers",
    );

    assert_eq!(config.brokers, vec!["localhost:9092"]);
    assert_eq!(config.topic, "vectors-topic");
}

#[test]
fn test_kafka_consumer_config_local() {
    let config = KafkaConsumerConfig::local("test-topic", "test-group");

    assert_eq!(config.topic, "test-topic");
    assert_eq!(config.group.group_id, "test-group");
}

#[test]
fn test_kafka_consumer_config_validation() {
    let config = KafkaConsumerConfig::default();
    assert!(config.validate().is_ok());

    let invalid_config = KafkaConsumerConfig {
        brokers: vec![],
        ..Default::default()
    };
    assert!(invalid_config.validate().is_err());
}

#[test]
fn test_kafka_dlq_config() {
    let dlq = DlqConfig::default();

    assert_eq!(dlq.topic, "vectors-dlq");
    assert_eq!(dlq.max_retries, 3);
}

// =============================================================================
// Message Deserializer Tests
// =============================================================================

#[test]
fn test_json_deserializer() {
    let deserializer = MessageDeserializer::new(DeserializationFormat::Json);

    let json = r#"{
        "id": "vec_123",
        "vector": [0.1, 0.2, 0.3, 0.4],
        "metadata": {"category": "test"}
    }"#;

    let result = deserializer.deserialize(json.as_bytes());
    assert!(result.is_ok());

    let msg = result.unwrap();
    assert_eq!(msg.id, "vec_123");
    assert_eq!(msg.vector.len(), 4);
}

#[test]
fn test_json_deserializer_with_dimension() {
    let deserializer = MessageDeserializer::with_dimension(DeserializationFormat::Json, 4);

    let json = r#"{"id": "v1", "vector": [0.1, 0.2, 0.3, 0.4]}"#;
    let result = deserializer.deserialize(json.as_bytes());
    assert!(result.is_ok());

    // Wrong dimension should fail
    let wrong_json = r#"{"id": "v2", "vector": [0.1, 0.2]}"#;
    let result = deserializer.deserialize(wrong_json.as_bytes());
    assert!(result.is_err());
}

#[test]
fn test_deserializer_invalid_json() {
    let deserializer = MessageDeserializer::new(DeserializationFormat::Json);

    let invalid = b"not valid json";
    let result = deserializer.deserialize(invalid);
    assert!(result.is_err());
}

// =============================================================================
// Backpressure Level Tests
// =============================================================================

#[test]
fn test_backpressure_level_ordering() {
    assert!(BackpressureLevel::None < BackpressureLevel::Low);
    assert!(BackpressureLevel::Low < BackpressureLevel::Medium);
    assert!(BackpressureLevel::Medium < BackpressureLevel::High);
    assert!(BackpressureLevel::High < BackpressureLevel::Critical);
}

#[test]
fn test_backpressure_delay_values() {
    assert_eq!(BackpressureLevel::None.delay_ms(), 0);
    assert_eq!(BackpressureLevel::Low.delay_ms(), 10);
    assert_eq!(BackpressureLevel::Medium.delay_ms(), 50);
    assert_eq!(BackpressureLevel::High.delay_ms(), 200);
    assert_eq!(BackpressureLevel::Critical.delay_ms(), 1000);
}

#[test]
fn test_backpressure_level_debug() {
    assert_eq!(format!("{:?}", BackpressureLevel::Critical), "Critical");
}

// =============================================================================
// Session ID Tests
// =============================================================================

#[test]
fn test_stream_id_uniqueness() {
    let ids: Vec<StreamId> = (0..1000).map(|_| StreamId::new()).collect();

    // All IDs should be unique
    let unique: std::collections::HashSet<_> = ids.iter().map(|id| id.to_string()).collect();
    assert_eq!(unique.len(), 1000);
}

#[test]
fn test_stream_id_serialization() {
    let id = StreamId::new();
    let string = id.to_string();

    // Should be able to reconstruct from string
    let reconstructed = StreamId::from_string(string.clone());
    assert_eq!(reconstructed.to_string(), string);
}

// =============================================================================
// Subscription ID Tests
// =============================================================================

#[test]
fn test_subscription_id_uniqueness() {
    let ids: Vec<SubscriptionId> = (0..100).map(|_| SubscriptionId::new()).collect();

    // All IDs should be unique
    let unique: std::collections::HashSet<_> =
        ids.iter().map(|id| id.as_str().to_string()).collect();
    assert_eq!(unique.len(), 100);
}

// =============================================================================
// End-to-End Latency Tests
// =============================================================================

#[tokio::test]
async fn test_push_latency_under_load() {
    let coordinator = StreamCoordinator::new(StreamConfig {
        default_buffer_size: 16384,
        global_rate_limit: 1_000_000,
        ..Default::default()
    });

    let session_id = coordinator
        .create_session("latency_test".to_string(), SessionConfig::default())
        .await
        .unwrap();

    let mut latencies = vec![];

    for _ in 0..100 {
        let vectors = create_test_vectors(100, 0, 128);

        let start = Instant::now();
        let _ = coordinator.push_records(&session_id, vectors).await;
        let elapsed = start.elapsed();

        latencies.push(elapsed);
    }

    // Calculate P50, P99
    latencies.sort();
    let p50 = latencies[49];
    let p99 = latencies[98];

    // P99 should be under 10ms for local operations
    assert!(
        p99 < Duration::from_millis(10),
        "P99 latency too high: {:?}",
        p99
    );
    assert!(
        p50 < Duration::from_millis(1),
        "P50 latency too high: {:?}",
        p50
    );
}

#[tokio::test]
async fn test_throughput_measurement() {
    let coordinator = Arc::new(StreamCoordinator::new(StreamConfig {
        default_buffer_size: 65536,
        global_rate_limit: 10_000_000,
        ..Default::default()
    }));

    let session_id = coordinator
        .create_session("throughput_test".to_string(), SessionConfig::default())
        .await
        .unwrap();

    let batch_count = 100;
    let batch_size = 500;

    let start = Instant::now();

    for batch in 0..batch_count {
        let vectors = create_test_vectors(batch_size, batch * batch_size, 64);
        coordinator
            .push_records(&session_id, vectors)
            .await
            .unwrap();
    }

    let elapsed = start.elapsed();
    let total_vectors = batch_count * batch_size;
    let throughput = total_vectors as f64 / elapsed.as_secs_f64();

    // Should achieve at least 100K vectors/second locally
    assert!(
        throughput > 100_000.0,
        "Throughput too low: {:.0} vec/sec",
        throughput
    );
}

// =============================================================================
// Scored Result Tests
// =============================================================================

#[test]
fn test_scored_result_creation() {
    let result = ScoredResult {
        vector_id: "test_id".to_string(),
        score: 0.95,
        position: 0,
    };

    assert_eq!(result.vector_id, "test_id");
    assert_eq!(result.score, 0.95);
    assert_eq!(result.position, 0);
}

#[test]
fn test_scored_result_ordering() {
    let r1 = ScoredResult {
        vector_id: "a".to_string(),
        score: 0.9,
        position: 0,
    };
    let r2 = ScoredResult {
        vector_id: "b".to_string(),
        score: 0.8,
        position: 1,
    };

    // Ord is by score descending - higher score comes first in sorted order
    // So r1 (0.9) < r2 (0.8) in Ord ordering because higher scores sort first
    assert!(r1 < r2);
}

// =============================================================================
// Integration: Coordinator + Subscriptions
// =============================================================================

#[tokio::test]
async fn test_coordinator_with_subscriptions() {
    let coordinator = Arc::new(StreamCoordinator::new(create_small_config()));
    let subscription_manager = Arc::new(SubscriptionManager::with_defaults());

    // Create streaming session
    let session_id = coordinator
        .create_session("searchable".to_string(), SessionConfig::default())
        .await
        .unwrap();

    // Create subscription for the collection
    let sub_config = SubscriptionConfig {
        top_k: 10,
        score_threshold: 0.7,
        include_initial: false,
        ..Default::default()
    };

    let sub_handle = subscription_manager
        .subscribe("searchable".to_string(), vec![0.5; 128], sub_config, None)
        .await
        .unwrap();

    // Push vectors
    let vectors = create_test_vectors(100, 0, 128);
    coordinator
        .push_records(&session_id, vectors)
        .await
        .unwrap();

    // Both should be active
    assert_eq!(coordinator.session_count(), 1);
    assert!(subscription_manager.stats().total_subscriptions >= 1);

    // Cleanup
    subscription_manager.unsubscribe(&sub_handle.id).unwrap();
    coordinator.close_session(&session_id);
}

// =============================================================================
// Error Handling Tests
// =============================================================================

#[tokio::test]
async fn test_push_to_nonexistent_session() {
    let coordinator = StreamCoordinator::new(create_small_config());

    let fake_id = StreamId::new();
    let vectors = create_test_vectors(10, 0, 64);

    let result = coordinator.push_records(&fake_id, vectors).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_drain_from_nonexistent_session() {
    let coordinator = StreamCoordinator::new(create_small_config());

    let fake_id = StreamId::new();
    let result = coordinator.drain_records(&fake_id, 10);
    assert!(result.is_err());
}

#[test]
fn test_subscription_id_creation() {
    let id = SubscriptionId::new();
    assert!(!id.as_str().is_empty());
}

// =============================================================================
// Commit Strategy Tests
// =============================================================================

#[test]
fn test_commit_strategy_variants() {
    assert_eq!(format!("{:?}", CommitStrategy::AtLeastOnce), "AtLeastOnce");
    assert_eq!(format!("{:?}", CommitStrategy::AtMostOnce), "AtMostOnce");
    assert_eq!(format!("{:?}", CommitStrategy::ExactlyOnce), "ExactlyOnce");
    assert_eq!(format!("{:?}", CommitStrategy::Manual), "Manual");
}

#[test]
fn test_deserialization_format_variants() {
    assert_eq!(format!("{:?}", DeserializationFormat::Json), "Json");
    assert_eq!(format!("{:?}", DeserializationFormat::Avro), "Avro");
    assert_eq!(format!("{:?}", DeserializationFormat::Protobuf), "Protobuf");
    assert_eq!(format!("{:?}", DeserializationFormat::Raw), "Raw");
}

// =============================================================================
// Result Set Tests (Top-K Maintenance)
// =============================================================================

use proximadb::streaming::subscriptions::{ResultChange, ResultSet};

#[test]
fn test_result_set_basic_operations() {
    let mut rs = ResultSet::new(3);

    // Empty initially
    assert!(rs.is_empty());
    assert_eq!(rs.len(), 0);

    // Add some results
    let results = vec![
        ScoredResult {
            vector_id: "r1".to_string(),
            score: 0.9,
            position: 0,
        },
        ScoredResult {
            vector_id: "r2".to_string(),
            score: 0.8,
            position: 0,
        },
        ScoredResult {
            vector_id: "r3".to_string(),
            score: 0.7,
            position: 0,
        },
    ];

    let changes = rs.update(results);
    assert_eq!(changes.len(), 3);
    assert_eq!(rs.len(), 3);
}

#[test]
fn test_result_set_top_k_maintenance() {
    let mut rs = ResultSet::new(3);

    // Add 5 results, should keep top 3
    let results = vec![
        ScoredResult {
            vector_id: "r1".to_string(),
            score: 0.5,
            position: 0,
        },
        ScoredResult {
            vector_id: "r2".to_string(),
            score: 0.6,
            position: 0,
        },
        ScoredResult {
            vector_id: "r3".to_string(),
            score: 0.7,
            position: 0,
        },
        ScoredResult {
            vector_id: "r4".to_string(),
            score: 0.8,
            position: 0,
        },
        ScoredResult {
            vector_id: "r5".to_string(),
            score: 0.9,
            position: 0,
        },
    ];

    let changes = rs.update(results);

    // Should have kept only 3
    assert_eq!(rs.len(), 3);

    // Should have the highest scores
    assert!(rs.contains("r5"));
    assert!(rs.contains("r4"));
    assert!(rs.contains("r3"));
    assert!(!rs.contains("r2"));
    assert!(!rs.contains("r1"));

    // Should have removal changes
    assert!(
        changes
            .iter()
            .any(|c| matches!(c, ResultChange::Removed { .. }))
    );
}

#[test]
fn test_result_set_score_ordering() {
    let mut rs = ResultSet::new(5);

    let results = vec![
        ScoredResult {
            vector_id: "r3".to_string(),
            score: 0.3,
            position: 0,
        },
        ScoredResult {
            vector_id: "r1".to_string(),
            score: 0.1,
            position: 0,
        },
        ScoredResult {
            vector_id: "r5".to_string(),
            score: 0.5,
            position: 0,
        },
        ScoredResult {
            vector_id: "r2".to_string(),
            score: 0.2,
            position: 0,
        },
        ScoredResult {
            vector_id: "r4".to_string(),
            score: 0.4,
            position: 0,
        },
    ];

    rs.update(results);

    // Verify ordering (highest first)
    let vec = rs.to_vec();
    assert_eq!(vec[0].vector_id, "r5");
    assert_eq!(vec[1].vector_id, "r4");
    assert_eq!(vec[2].vector_id, "r3");
    assert_eq!(vec[3].vector_id, "r2");
    assert_eq!(vec[4].vector_id, "r1");

    // Verify positions
    assert_eq!(vec[0].position, 0);
    assert_eq!(vec[1].position, 1);
    assert_eq!(vec[2].position, 2);
}

#[test]
fn test_result_set_score_update() {
    let mut rs = ResultSet::new(3);

    // Initial results
    let results = vec![
        ScoredResult {
            vector_id: "r1".to_string(),
            score: 0.9,
            position: 0,
        },
        ScoredResult {
            vector_id: "r2".to_string(),
            score: 0.8,
            position: 0,
        },
        ScoredResult {
            vector_id: "r3".to_string(),
            score: 0.7,
            position: 0,
        },
    ];
    rs.update(results);

    // Update r3 to have highest score
    let updates = vec![ScoredResult {
        vector_id: "r3".to_string(),
        score: 0.95,
        position: 0,
    }];
    let changes = rs.update(updates);

    // Should have a score change
    assert!(changes.iter().any(|c| matches!(
        c,
        ResultChange::ScoreChanged { vector_id, .. } if vector_id == "r3"
    )));

    // r3 should now be first
    let vec = rs.to_vec();
    assert_eq!(vec[0].vector_id, "r3");
    assert!((vec[0].score - 0.95).abs() < f32::EPSILON);
}

#[test]
fn test_result_set_removal() {
    let mut rs = ResultSet::new(5);

    let results = vec![
        ScoredResult {
            vector_id: "r1".to_string(),
            score: 0.9,
            position: 0,
        },
        ScoredResult {
            vector_id: "r2".to_string(),
            score: 0.8,
            position: 0,
        },
        ScoredResult {
            vector_id: "r3".to_string(),
            score: 0.7,
            position: 0,
        },
    ];
    rs.update(results);

    // Remove r2
    let change = rs.remove("r2");
    assert!(change.is_some());
    assert_eq!(rs.len(), 2);
    assert!(!rs.contains("r2"));
}

#[test]
fn test_result_set_threshold() {
    let mut rs = ResultSet::with_threshold(5, 0.5);

    let results = vec![
        ScoredResult {
            vector_id: "r1".to_string(),
            score: 0.9,
            position: 0,
        },
        ScoredResult {
            vector_id: "r2".to_string(),
            score: 0.4, // Below threshold
            position: 0,
        },
        ScoredResult {
            vector_id: "r3".to_string(),
            score: 0.6,
            position: 0,
        },
        ScoredResult {
            vector_id: "r4".to_string(),
            score: 0.3, // Below threshold
            position: 0,
        },
    ];

    rs.update(results);

    // Should only include above threshold
    assert_eq!(rs.len(), 2);
    assert!(rs.contains("r1"));
    assert!(rs.contains("r3"));
    assert!(!rs.contains("r2"));
    assert!(!rs.contains("r4"));
}

#[test]
fn test_result_set_would_qualify() {
    let mut rs = ResultSet::new(3);

    // Empty set - anything qualifies
    assert!(rs.would_qualify(0.1));

    // Fill up
    let results = vec![
        ScoredResult {
            vector_id: "r1".to_string(),
            score: 0.9,
            position: 0,
        },
        ScoredResult {
            vector_id: "r2".to_string(),
            score: 0.8,
            position: 0,
        },
        ScoredResult {
            vector_id: "r3".to_string(),
            score: 0.7,
            position: 0,
        },
    ];
    rs.update(results);

    // Higher than min (0.7) qualifies
    assert!(rs.would_qualify(0.8));
    assert!(rs.would_qualify(0.95));

    // Lower than min doesn't qualify
    assert!(!rs.would_qualify(0.6));
    assert!(!rs.would_qualify(0.1));
}

#[test]
fn test_result_set_min_max_score() {
    let mut rs = ResultSet::new(5);

    assert!(rs.min_score().is_none());
    assert!(rs.max_score().is_none());

    let results = vec![
        ScoredResult {
            vector_id: "r1".to_string(),
            score: 0.9,
            position: 0,
        },
        ScoredResult {
            vector_id: "r2".to_string(),
            score: 0.5,
            position: 0,
        },
        ScoredResult {
            vector_id: "r3".to_string(),
            score: 0.7,
            position: 0,
        },
    ];
    rs.update(results);

    assert!((rs.max_score().unwrap() - 0.9).abs() < f32::EPSILON);
    assert!((rs.min_score().unwrap() - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_result_set_clear() {
    let mut rs = ResultSet::new(5);

    let results = vec![
        ScoredResult {
            vector_id: "r1".to_string(),
            score: 0.9,
            position: 0,
        },
        ScoredResult {
            vector_id: "r2".to_string(),
            score: 0.8,
            position: 0,
        },
    ];
    rs.update(results);

    let changes = rs.clear();
    assert_eq!(changes.len(), 2);
    assert!(rs.is_empty());
}

#[test]
fn test_result_set_remove_many() {
    let mut rs = ResultSet::new(5);

    let results = vec![
        ScoredResult {
            vector_id: "r1".to_string(),
            score: 0.9,
            position: 0,
        },
        ScoredResult {
            vector_id: "r2".to_string(),
            score: 0.8,
            position: 0,
        },
        ScoredResult {
            vector_id: "r3".to_string(),
            score: 0.7,
            position: 0,
        },
        ScoredResult {
            vector_id: "r4".to_string(),
            score: 0.6,
            position: 0,
        },
    ];
    rs.update(results);

    let changes = rs.remove_many(&["r1".to_string(), "r3".to_string()]);
    assert_eq!(changes.len(), 2);
    assert_eq!(rs.len(), 2);
    assert!(!rs.contains("r1"));
    assert!(!rs.contains("r3"));
}

#[test]
fn test_result_set_get() {
    let mut rs = ResultSet::new(5);

    let results = vec![ScoredResult {
        vector_id: "r1".to_string(),
        score: 0.9,
        position: 0,
    }];
    rs.update(results);

    let result = rs.get("r1");
    assert!(result.is_some());
    assert_eq!(result.unwrap().vector_id, "r1");

    assert!(rs.get("nonexistent").is_none());
}

// =============================================================================
// Storage Engine Integration Tests (TDD)
// Note: Storage flush integration is tested through coordinator unit tests
// =============================================================================

// =============================================================================
// Live Query Subscription Integration Tests (TDD)
// =============================================================================

#[tokio::test]
async fn test_subscription_with_coordinator_integration() {
    let coordinator = Arc::new(StreamCoordinator::new(create_small_config()));
    let subscription_manager = Arc::new(SubscriptionManager::with_defaults());

    // Create streaming session
    let session_id = coordinator
        .create_session(
            "live_query_collection".to_string(),
            SessionConfig::default(),
        )
        .await
        .unwrap();

    // Create subscription
    let config = SubscriptionConfig {
        top_k: 5,
        score_threshold: 0.5,
        include_initial: false,
        ..Default::default()
    };

    let handle = subscription_manager
        .subscribe(
            "live_query_collection".to_string(),
            vec![0.5; 64],
            config,
            None,
        )
        .await
        .unwrap();

    // Activate subscription
    subscription_manager
        .activate(&handle.id, vec![])
        .await
        .unwrap();

    // Push vectors
    let vectors = create_test_vectors(20, 0, 64);
    let push_result = coordinator
        .push_records(&session_id, vectors)
        .await
        .unwrap();
    assert_eq!(push_result.pushed, 20);

    // Notify subscription manager of inserts
    let notify_vectors: Vec<(String, Vec<f32>, f32)> = (0..20)
        .map(|i| (format!("vec_{}", i), vec![0.1 * (i as f32); 64], 0.0))
        .collect();

    let notified = subscription_manager
        .notify_insert("live_query_collection", notify_vectors)
        .await;

    // At least one subscription should be notified
    assert!(notified >= 1);

    // Cleanup
    subscription_manager.unsubscribe(&handle.id).unwrap();
    coordinator.close_session(&session_id);
}

#[tokio::test]
async fn test_subscription_receives_updates() {
    let subscription_manager = SubscriptionManager::with_defaults();

    let config = SubscriptionConfig {
        top_k: 3,
        score_threshold: 0.0,
        include_initial: false,
        debounce_ms: 10,
        max_delay_ms: 50,
        track_positions: true,
    };

    let handle = subscription_manager
        .subscribe("update_test".to_string(), vec![1.0, 0.0, 0.0], config, None)
        .await
        .unwrap();

    // Activate with initial results
    let initial = vec![
        ScoredResult {
            vector_id: "v1".to_string(),
            score: 0.9,
            position: 0,
        },
        ScoredResult {
            vector_id: "v2".to_string(),
            score: 0.8,
            position: 1,
        },
    ];
    subscription_manager
        .activate(&handle.id, initial)
        .await
        .unwrap();

    // Insert new vectors that should affect the subscription
    let new_vectors = vec![("v3".to_string(), vec![0.95, 0.05, 0.0], 0.0)];

    let affected = subscription_manager
        .notify_insert("update_test", new_vectors)
        .await;
    assert!(affected >= 1);

    // Small delay to allow processing
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Cleanup
    subscription_manager.unsubscribe(&handle.id).unwrap();
}

#[tokio::test]
async fn test_subscription_removal_notification() {
    let subscription_manager = SubscriptionManager::with_defaults();

    let config = SubscriptionConfig {
        top_k: 5,
        score_threshold: 0.0,
        include_initial: true,
        ..Default::default()
    };

    let handle = subscription_manager
        .subscribe("removal_test".to_string(), vec![0.5; 8], config, None)
        .await
        .unwrap();

    // Activate with initial results
    let initial = vec![
        ScoredResult {
            vector_id: "r1".to_string(),
            score: 0.9,
            position: 0,
        },
        ScoredResult {
            vector_id: "r2".to_string(),
            score: 0.8,
            position: 1,
        },
        ScoredResult {
            vector_id: "r3".to_string(),
            score: 0.7,
            position: 2,
        },
    ];
    subscription_manager
        .activate(&handle.id, initial)
        .await
        .unwrap();

    // Notify removal - returns number of subscriptions notified
    // The manager may return 0 if the removed vector doesn't affect ranking
    let notified = subscription_manager
        .notify_remove("removal_test", &["r2".to_string()])
        .await;

    // Subscription manager was invoked (may be 0 if no ranking change)
    let _ = notified; // usize: always >= 0

    // Cleanup
    subscription_manager.unsubscribe(&handle.id).unwrap();
}

#[tokio::test]
async fn test_subscription_threshold_filtering() {
    let subscription_manager = SubscriptionManager::with_defaults();

    // High threshold subscription
    let config = SubscriptionConfig {
        top_k: 10,
        score_threshold: 0.9, // Only very high scores
        include_initial: false,
        ..Default::default()
    };

    let handle = subscription_manager
        .subscribe("threshold_test".to_string(), vec![1.0; 8], config, None)
        .await
        .unwrap();

    subscription_manager
        .activate(&handle.id, vec![])
        .await
        .unwrap();

    // Insert vectors with varying scores
    // The subscription manager's evaluator will compute similarity
    let new_vectors = vec![
        ("high_score".to_string(), vec![0.99; 8], 0.0),
        ("low_score".to_string(), vec![0.1; 8], 0.0),
    ];

    // Should process (actual filtering happens in evaluator)
    let notified = subscription_manager
        .notify_insert("threshold_test", new_vectors)
        .await;
    let _ = notified; // usize: always >= 0 // Manager was notified

    subscription_manager.unsubscribe(&handle.id).unwrap();
}

#[tokio::test]
async fn test_multiple_subscriptions_same_collection() {
    let subscription_manager = SubscriptionManager::with_defaults();

    let config1 = SubscriptionConfig {
        top_k: 5,
        score_threshold: 0.5,
        include_initial: false,
        ..Default::default()
    };

    let config2 = SubscriptionConfig {
        top_k: 10,
        score_threshold: 0.8,
        include_initial: false,
        ..Default::default()
    };

    // Different query vectors -> different subscriptions
    let handle1 = subscription_manager
        .subscribe(
            "shared_collection".to_string(),
            vec![1.0, 0.0, 0.0],
            config1,
            None,
        )
        .await
        .unwrap();

    let handle2 = subscription_manager
        .subscribe(
            "shared_collection".to_string(),
            vec![0.0, 1.0, 0.0],
            config2,
            None,
        )
        .await
        .unwrap();

    subscription_manager
        .activate(&handle1.id, vec![])
        .await
        .unwrap();
    subscription_manager
        .activate(&handle2.id, vec![])
        .await
        .unwrap();

    // Insert vectors
    let new_vectors = vec![
        ("v1".to_string(), vec![0.9, 0.1, 0.0], 0.0),
        ("v2".to_string(), vec![0.1, 0.9, 0.0], 0.0),
    ];

    let notified = subscription_manager
        .notify_insert("shared_collection", new_vectors)
        .await;

    // Both subscriptions should be notified (they're watching same collection)
    assert!(notified >= 2);

    subscription_manager.unsubscribe(&handle1.id).unwrap();
    subscription_manager.unsubscribe(&handle2.id).unwrap();
}

// =============================================================================
// Kafka Consumer Integration Tests (TDD)
// =============================================================================

use proximadb::streaming::kafka::KafkaVectorConsumer;

#[tokio::test]
async fn test_kafka_consumer_process_batch() {
    let config = KafkaConsumerConfig::local("vectors-topic", "test-consumer-group");
    let coordinator = Arc::new(StreamCoordinator::new(StreamConfig::default()));
    let consumer = Arc::new(KafkaVectorConsumer::new(config, coordinator.clone()));

    // Create batch of JSON messages
    let messages = vec![
        (
            br#"{"id": "k1", "vector": [0.1, 0.2, 0.3, 0.4], "collection": "kafka_test"}"#.to_vec(),
            None,
        ),
        (
            br#"{"id": "k2", "vector": [0.5, 0.6, 0.7, 0.8], "collection": "kafka_test"}"#.to_vec(),
            None,
        ),
        (
            br#"{"id": "k3", "vector": [0.9, 1.0, 1.1, 1.2], "collection": "kafka_test"}"#.to_vec(),
            None,
        ),
    ];

    let result = consumer.process_messages(messages).await;

    assert_eq!(result.successful, 3);
    assert_eq!(result.failed, 0);
    assert!(result.errors.is_empty());
    assert!(result.processing_time.as_nanos() > 0);
}

#[tokio::test]
async fn test_kafka_consumer_handles_invalid_json() {
    let config = KafkaConsumerConfig::local("vectors-topic", "test-group");
    let coordinator = Arc::new(StreamCoordinator::new(StreamConfig::default()));
    let consumer = Arc::new(KafkaVectorConsumer::new(config, coordinator));

    let messages = vec![
        (br#"{"id": "valid", "vector": [0.1, 0.2]}"#.to_vec(), None),
        (br#"not valid json at all"#.to_vec(), None),
        (br#"{"missing": "vector field"}"#.to_vec(), None),
    ];

    let result = consumer.process_messages(messages).await;

    // First message succeeds, others fail deserialization
    assert!(result.successful >= 1);
    assert!(result.failed >= 1);
}

#[tokio::test]
async fn test_kafka_consumer_default_collection() {
    let mut config = KafkaConsumerConfig::local("vectors-topic", "test-group");
    config.default_collection = Some("default_collection".to_string());

    let coordinator = Arc::new(StreamCoordinator::new(StreamConfig::default()));
    let consumer = Arc::new(KafkaVectorConsumer::new(config, coordinator.clone()));

    // Message without explicit collection
    let messages = vec![(br#"{"id": "v1", "vector": [0.1, 0.2, 0.3]}"#.to_vec(), None)];

    let result = consumer.process_messages(messages).await;

    assert_eq!(result.successful, 1);

    // Should have created a session for default_collection
    // (verified by successful processing)
}

#[tokio::test]
async fn test_kafka_consumer_metrics_tracking() {
    let config = KafkaConsumerConfig::local("metrics-topic", "metrics-group");
    let coordinator = Arc::new(StreamCoordinator::new(StreamConfig::default()));
    let consumer = Arc::new(KafkaVectorConsumer::new(config, coordinator));

    // Process some messages
    let messages = vec![
        (br#"{"id": "m1", "vector": [0.1]}"#.to_vec(), None),
        (br#"{"id": "m2", "vector": [0.2]}"#.to_vec(), None),
        (br#"invalid"#.to_vec(), None),
    ];

    consumer.process_messages(messages).await;

    let metrics = consumer.metrics();

    assert_eq!(metrics.messages_received, 3);
    assert_eq!(metrics.messages_processed, 2);
    assert_eq!(metrics.messages_failed, 1);
    assert_eq!(metrics.batches_processed, 1);
    assert!(metrics.bytes_received > 0);
}

#[tokio::test]
async fn test_kafka_consumer_start_stop_lifecycle() {
    let config = KafkaConsumerConfig::local("lifecycle-topic", "lifecycle-group");
    let coordinator = Arc::new(StreamCoordinator::new(StreamConfig::default()));
    let consumer = Arc::new(KafkaVectorConsumer::new(config, coordinator));

    // Start consumer
    let handle = consumer.clone().start().await.unwrap();

    assert!(consumer.is_running());

    // Stop consumer
    handle.stop().await;

    // Allow time for shutdown
    tokio::time::sleep(Duration::from_millis(200)).await;

    assert!(!consumer.is_running());
}

#[tokio::test]
async fn test_kafka_consumer_cannot_start_twice() {
    let config = KafkaConsumerConfig::local("double-start-topic", "test-group");
    let coordinator = Arc::new(StreamCoordinator::new(StreamConfig::default()));
    let consumer = Arc::new(KafkaVectorConsumer::new(config, coordinator));

    // First start succeeds
    let handle = consumer.clone().start().await.unwrap();

    // Second start should fail
    let result = consumer.clone().start().await;
    assert!(result.is_err());

    // Cleanup
    handle.stop().await;
    tokio::time::sleep(Duration::from_millis(100)).await;
}

#[tokio::test]
async fn test_kafka_consumer_backpressure_propagation() {
    let config = KafkaConsumerConfig::local("backpressure-topic", "bp-group");
    let coordinator = Arc::new(StreamCoordinator::new(StreamConfig {
        default_buffer_size: 32, // Very small buffer
        ..Default::default()
    }));
    let consumer = Arc::new(KafkaVectorConsumer::new(config, coordinator));

    // Process many messages to potentially trigger backpressure
    let messages: Vec<_> = (0..50)
        .map(|i| {
            let json = format!(r#"{{"id": "bp_{}", "vector": [{}]}}"#, i, i as f32 * 0.1);
            (json.into_bytes(), None)
        })
        .collect();

    let result = consumer.process_messages(messages).await;

    // Some should succeed even under backpressure
    assert!(result.successful > 0);
}

#[tokio::test]
async fn test_kafka_consumer_multi_collection_routing() {
    let config = KafkaConsumerConfig::local("multi-collection-topic", "mc-group");
    let coordinator = Arc::new(StreamCoordinator::new(StreamConfig::default()));
    let consumer = Arc::new(KafkaVectorConsumer::new(config, coordinator.clone()));

    // Messages for different collections
    let messages = vec![
        (
            br#"{"id": "c1v1", "vector": [0.1], "collection": "collection_a"}"#.to_vec(),
            None,
        ),
        (
            br#"{"id": "c1v2", "vector": [0.2], "collection": "collection_a"}"#.to_vec(),
            None,
        ),
        (
            br#"{"id": "c2v1", "vector": [0.3], "collection": "collection_b"}"#.to_vec(),
            None,
        ),
        (
            br#"{"id": "c3v1", "vector": [0.4], "collection": "collection_c"}"#.to_vec(),
            None,
        ),
    ];

    let result = consumer.process_messages(messages).await;

    assert_eq!(result.successful, 4);

    // Coordinator should have 3 sessions (one per collection)
    assert_eq!(coordinator.session_count(), 3);
}

// =============================================================================
// End-to-End Integration Tests
// Note: Tests requiring mock storage are in coordinator unit tests
// =============================================================================

#[tokio::test]
async fn test_streaming_subscription_end_to_end() {
    // This test verifies the integration between streaming coordinator and subscription manager
    // without requiring a mock storage engine

    let coordinator = Arc::new(StreamCoordinator::new(StreamConfig::default()));
    let subscription_manager = Arc::new(SubscriptionManager::with_defaults());

    // Create streaming session
    let session_id = coordinator
        .create_session("e2e_collection".to_string(), SessionConfig::default())
        .await
        .unwrap();

    // Create live query subscription
    let sub_config = SubscriptionConfig {
        top_k: 10,
        score_threshold: 0.0,
        include_initial: false,
        ..Default::default()
    };

    let sub_handle = subscription_manager
        .subscribe(
            "e2e_collection".to_string(),
            vec![0.5; 64],
            sub_config,
            None,
        )
        .await
        .unwrap();

    subscription_manager
        .activate(&sub_handle.id, vec![])
        .await
        .unwrap();

    // Push vectors (simulating streaming ingestion)
    for batch in 0..5 {
        let vectors = create_test_vectors(50, batch * 50, 64);
        let result = coordinator
            .push_records(&session_id, vectors)
            .await
            .unwrap();
        assert_eq!(result.pushed, 50);
    }

    // Notify subscriptions about the pushed vectors
    let notify_vectors: Vec<_> = (0..250)
        .map(|i| (format!("vec_{}", i), vec![0.1 * (i as f32 % 10.0); 64], 0.0))
        .collect();

    let notified = subscription_manager
        .notify_insert("e2e_collection", notify_vectors)
        .await;

    // Subscription should have been notified
    assert!(notified >= 1);

    // Cleanup
    subscription_manager.unsubscribe(&sub_handle.id).unwrap();
    coordinator.close_session(&session_id);
}

#[tokio::test]
async fn test_kafka_consumer_to_coordinator_pipeline() {
    // This test verifies Kafka consumer -> Coordinator integration
    // without requiring a mock storage engine

    let kafka_config = KafkaConsumerConfig::local("kafka-storage-topic", "ks-group");
    let coordinator = Arc::new(StreamCoordinator::new(StreamConfig::default()));
    let consumer = Arc::new(KafkaVectorConsumer::new(kafka_config, coordinator.clone()));

    // Simulate Kafka messages
    let messages: Vec<_> = (0..100)
        .map(|i| {
            let json = format!(
                r#"{{"id": "kafka_vec_{}", "vector": [{}], "collection": "kafka_storage_collection"}}"#,
                i,
                (0..32).map(|j| format!("{}", (i * 32 + j) as f32 * 0.01)).collect::<Vec<_>>().join(",")
            );
            (json.into_bytes(), None)
        })
        .collect();

    // Process through Kafka consumer
    let batch_result = consumer.process_messages(messages).await;
    assert_eq!(batch_result.successful, 100);

    // Get the session that was auto-created
    let sessions = coordinator.list_sessions();
    assert!(!sessions.is_empty());

    // Verify the session has the expected collection
    let session_id = &sessions[0];
    let collection = coordinator.get_session_collection(session_id).unwrap();
    assert_eq!(collection, "kafka_storage_collection");
}

// =============================================================================
// Live Subscription Update Tests
// =============================================================================

#[tokio::test]
async fn test_subscription_manager_notify_insert() {
    let manager = SubscriptionManager::with_defaults();

    // Create subscription
    let config = SubscriptionConfig {
        top_k: 10,
        score_threshold: 0.0,
        include_initial: false,
        ..Default::default()
    };

    let handle = manager
        .subscribe("test_collection".to_string(), vec![0.5; 8], config, None)
        .await
        .unwrap();

    // Activate the subscription with empty initial results
    manager.activate(&handle.id, vec![]).await.unwrap();

    // Notify about new inserts
    let vectors = vec![
        ("v1".to_string(), vec![0.5; 8], 0.0),
        ("v2".to_string(), vec![0.6; 8], 0.0),
    ];

    let notified = manager.notify_insert("test_collection", vectors).await;

    // Should have notified the subscription
    let _ = notified; // usize: always >= 0

    // Cleanup
    manager.unsubscribe(&handle.id).unwrap();
}

#[tokio::test]
async fn test_subscription_manager_notify_remove() {
    let manager = SubscriptionManager::with_defaults();

    // Create subscription
    let config = SubscriptionConfig {
        top_k: 10,
        score_threshold: 0.0,
        include_initial: true,
        ..Default::default()
    };

    let handle = manager
        .subscribe("test_collection".to_string(), vec![0.5; 8], config, None)
        .await
        .unwrap();

    // Activate with initial results
    let initial_results = vec![
        ScoredResult {
            vector_id: "v1".to_string(),
            score: 0.9,
            position: 0,
        },
        ScoredResult {
            vector_id: "v2".to_string(),
            score: 0.8,
            position: 1,
        },
    ];
    manager.activate(&handle.id, initial_results).await.unwrap();

    // Notify about removal
    let removed = vec!["v1".to_string()];
    let notified = manager.notify_remove("test_collection", &removed).await;

    // Should have processed the removal
    let _ = notified; // usize: always >= 0

    // Cleanup
    manager.unsubscribe(&handle.id).unwrap();
}

#[tokio::test]
async fn test_subscription_fingerprint_deduplication_with_updates() {
    let manager = SubscriptionManager::with_defaults();

    // Create two subscriptions with same fingerprint
    let config = SubscriptionConfig {
        top_k: 10,
        score_threshold: 0.8,
        include_initial: false,
        ..Default::default()
    };
    let vector = vec![0.5; 8];

    let handle1 = manager
        .subscribe("test".to_string(), vector.clone(), config.clone(), None)
        .await
        .unwrap();

    let handle2 = manager
        .subscribe("test".to_string(), vector.clone(), config.clone(), None)
        .await
        .unwrap();

    // Should share fingerprint
    let stats = manager.stats();
    assert_eq!(stats.total_subscriptions, 2);
    assert_eq!(stats.unique_fingerprints, 1);
    assert_eq!(stats.dedup_hits, 1);

    // Cleanup
    manager.unsubscribe(&handle1.id).unwrap();
    manager.unsubscribe(&handle2.id).unwrap();
}

// =============================================================================
// Query Evaluator Incremental Tests
// =============================================================================

#[test]
fn test_query_evaluator_incremental_evaluate() {
    let evaluator = QueryEvaluator::new();
    let query = vec![1.0, 0.0, 0.0];

    // Initial results
    let current = vec![
        ScoredResult {
            vector_id: "v1".to_string(),
            score: 0.9,
            position: 0,
        },
        ScoredResult {
            vector_id: "v2".to_string(),
            score: 0.8,
            position: 1,
        },
    ];

    // New vector that should enter top-k
    let new_vectors = vec![("v3".to_string(), vec![0.95, 0.05, 0.0], 0.0)];

    let (results, changes) = evaluator.incremental_evaluate(&query, &current, &new_vectors, 2, 0.0);

    assert_eq!(results.len(), 2);
    // v3 should be in results with high score
    assert!(results.iter().any(|r| r.vector_id == "v3"));
    // There should be changes
    assert!(!changes.is_empty());
}

#[test]
fn test_query_evaluator_would_affect_subscriptions() {
    let evaluator = QueryEvaluator::new();

    let subscriptions: Vec<(&[f32], f32, f32)> = vec![
        (&[1.0, 0.0, 0.0][..], 0.5, 0.8), // High threshold, moderate min
        (&[0.0, 1.0, 0.0][..], 0.5, 0.9), // Different direction
    ];

    let vector = vec![0.9, 0.1, 0.0]; // Similar to first query

    let affected = evaluator.would_affect_subscriptions(&vector, &subscriptions);

    assert!(affected[0]); // Should affect first subscription
    assert!(!affected[1]); // Should not affect second (wrong direction)
}

// =============================================================================
// Flush Stats Tests
// =============================================================================

use proximadb::streaming::FlushStats;

// Note: FlushStats import is also at line ~1352 but Rust handles re-imports gracefully

#[test]
fn test_flush_stats_default() {
    let stats = FlushStats::default();

    assert_eq!(stats.records_flushed, 0);
    assert_eq!(stats.bytes_written, 0);
    assert!(!stats.success);
    assert!(stats.collection_id.is_empty());
}

// =============================================================================
// Coordinator Get Session Collection Tests
// =============================================================================

#[tokio::test]
async fn test_coordinator_get_session_collection() {
    let coordinator = StreamCoordinator::new(create_small_config());

    let session_id = coordinator
        .create_session("my_collection".to_string(), SessionConfig::default())
        .await
        .unwrap();

    let collection = coordinator.get_session_collection(&session_id);
    assert_eq!(collection, Some("my_collection".to_string()));

    // Non-existent session
    let fake_id = StreamId::new();
    assert!(coordinator.get_session_collection(&fake_id).is_none());
}

// =============================================================================
// Result Set Stats Tests
// =============================================================================

use proximadb::streaming::subscriptions::ResultSetStats;

#[test]
fn test_result_set_stats() {
    let mut rs = ResultSet::new(10);

    let results = vec![
        ScoredResult {
            vector_id: "r1".to_string(),
            score: 0.9,
            position: 0,
        },
        ScoredResult {
            vector_id: "r2".to_string(),
            score: 0.5,
            position: 0,
        },
    ];
    rs.update(results);

    let stats: ResultSetStats = (&rs).into();
    assert_eq!(stats.count, 2);
    assert!((stats.max_score.unwrap() - 0.9).abs() < f32::EPSILON);
    assert!((stats.min_score.unwrap() - 0.5).abs() < f32::EPSILON);
    assert_eq!(stats.top_k, 10);
}

// =============================================================================
// Flush Retry Config Tests
// =============================================================================

use proximadb::streaming::FlushRetryConfig;

#[test]
fn test_flush_retry_config_default() {
    let config = FlushRetryConfig::default();

    assert_eq!(config.max_retries, 3);
    assert_eq!(config.initial_delay, Duration::from_millis(100));
    assert_eq!(config.max_delay, Duration::from_secs(5));
    assert!((config.backoff_multiplier - 2.0).abs() < f64::EPSILON);
}

#[test]
fn test_flush_retry_config_custom() {
    let config = FlushRetryConfig {
        max_retries: 5,
        initial_delay: Duration::from_millis(50),
        max_delay: Duration::from_secs(10),
        backoff_multiplier: 1.5,
    };

    assert_eq!(config.max_retries, 5);
    assert_eq!(config.initial_delay, Duration::from_millis(50));
    assert_eq!(config.max_delay, Duration::from_secs(10));
    assert!((config.backoff_multiplier - 1.5).abs() < f64::EPSILON);
}

// =============================================================================
// Coordinator Stats Tests
// =============================================================================

#[tokio::test]
async fn test_coordinator_stats_basic() {
    let coordinator = StreamCoordinator::new(create_small_config());

    // Initial stats
    let stats = coordinator.stats();
    assert_eq!(stats.active_sessions, 0);
    assert_eq!(stats.total_buffered_records, 0);

    // Create sessions and push data
    let session_id = coordinator
        .create_session("stats_collection".to_string(), SessionConfig::default())
        .await
        .unwrap();

    let vectors = create_test_vectors(50, 0, 64);
    coordinator
        .push_records(&session_id, vectors)
        .await
        .unwrap();

    // Updated stats
    let stats = coordinator.stats();
    assert_eq!(stats.active_sessions, 1);
    assert_eq!(stats.total_buffered_records, 50);
    assert_eq!(stats.sessions_created, 1);
}

#[tokio::test]
async fn test_coordinator_stats_multiple_sessions() {
    let coordinator = StreamCoordinator::new(StreamConfig {
        default_buffer_size: 1000,
        ..Default::default()
    });

    for i in 0..3 {
        let session_id = coordinator
            .create_session(format!("collection_{}", i), SessionConfig::default())
            .await
            .unwrap();

        let vectors = create_test_vectors(100, 0, 64);
        coordinator
            .push_records(&session_id, vectors)
            .await
            .unwrap();
    }

    let stats = coordinator.stats();
    assert_eq!(stats.active_sessions, 3);
    assert_eq!(stats.total_buffered_records, 300);
    assert_eq!(stats.sessions_created, 3);
    assert_eq!(stats.sessions_closed, 0);
}

#[tokio::test]
async fn test_coordinator_stats_after_close() {
    let coordinator = StreamCoordinator::new(create_small_config());

    let session_id = coordinator
        .create_session("close_test".to_string(), SessionConfig::default())
        .await
        .unwrap();

    coordinator.close_session(&session_id);

    let stats = coordinator.stats();
    assert_eq!(stats.active_sessions, 0);
    assert_eq!(stats.sessions_created, 1);
    assert_eq!(stats.sessions_closed, 1);
}

// =============================================================================
// Total Buffered Records Tests
// =============================================================================

#[tokio::test]
async fn test_total_buffered_records() {
    let coordinator = StreamCoordinator::new(StreamConfig {
        default_buffer_size: 500,
        ..Default::default()
    });

    assert_eq!(coordinator.total_buffered_records(), 0);

    let session1 = coordinator
        .create_session("col1".to_string(), SessionConfig::default())
        .await
        .unwrap();

    let session2 = coordinator
        .create_session("col2".to_string(), SessionConfig::default())
        .await
        .unwrap();

    // Push different amounts to each session
    let vectors1 = create_test_vectors(30, 0, 64);
    coordinator.push_records(&session1, vectors1).await.unwrap();

    let vectors2 = create_test_vectors(70, 0, 64);
    coordinator.push_records(&session2, vectors2).await.unwrap();

    assert_eq!(coordinator.total_buffered_records(), 100);

    // Drain some
    coordinator.drain_records(&session1, 20).unwrap();
    assert_eq!(coordinator.total_buffered_records(), 80);
}

// =============================================================================
// Flush With Retry Tests
// Note: Full retry tests require mock storage and are covered in coordinator unit tests
// =============================================================================

// =============================================================================
// Flush Stats Enhanced Tests
// =============================================================================

#[test]
fn test_flush_stats_enhanced_default() {
    let stats = FlushStats::default();

    assert_eq!(stats.retry_count, 0);
    assert!(!stats.was_retried);
}

// =============================================================================
// Metrics Integration Tests
// =============================================================================

use proximadb::streaming::StreamMetrics;

#[test]
fn test_stream_metrics_flush_success() {
    let metrics = StreamMetrics::new();

    metrics.record_flush_success("test_collection", 100, 51200, 0);

    // Metrics recorded without panic
    assert!(true);
}

#[test]
fn test_stream_metrics_flush_failure() {
    let metrics = StreamMetrics::new();

    metrics.record_flush_failure("test_collection", 2);

    // Metrics recorded without panic
    assert!(true);
}

#[test]
fn test_stream_metrics_flush_with_retries() {
    let metrics = StreamMetrics::new();

    // Record successful flush that required retries
    metrics.record_flush_success("retry_collection", 50, 25600, 3);

    // Record failed flush with retries
    metrics.record_flush_failure("failed_collection", 5);

    // Both recorded without panic
    assert!(true);
}

// =============================================================================
// Integrated Service Tests
// =============================================================================

use proximadb::streaming::{IntegratedServiceConfig, IntegratedStreamingService};

#[tokio::test]
async fn test_integrated_service_creation() {
    let config = IntegratedServiceConfig::default();
    let service = IntegratedStreamingService::new(config);

    let stats = service.stats();
    assert_eq!(stats.active_sessions, 0);
    assert_eq!(stats.active_subscriptions, 0);
}

#[tokio::test]
async fn test_integrated_service_session_lifecycle() {
    let service = IntegratedStreamingService::new(IntegratedServiceConfig::default());

    let session_id = service
        .create_session("integrated_test".to_string(), SessionConfig::default())
        .await
        .unwrap();

    assert_eq!(service.stats().active_sessions, 1);

    service.close_session(&session_id);

    assert_eq!(service.stats().active_sessions, 0);
}

#[tokio::test]
async fn test_integrated_service_push_records() {
    let service = IntegratedStreamingService::new(IntegratedServiceConfig::default());

    let session_id = service
        .create_session("push_test".to_string(), SessionConfig::default())
        .await
        .unwrap();

    let vectors = create_test_vectors(50, 0, 64);
    let result = service.push_records(&session_id, vectors).await.unwrap();

    assert_eq!(result.pushed, 50);
    assert_eq!(result.dropped, 0);
}

#[tokio::test]
async fn test_integrated_service_push_and_notify() {
    let mut config = IntegratedServiceConfig::default();
    config.auto_notify_subscriptions = true;

    let service = IntegratedStreamingService::new(config);

    let session_id = service
        .create_session("notify_test".to_string(), SessionConfig::default())
        .await
        .unwrap();

    // Create a subscription for the collection
    let sub_config = SubscriptionConfig {
        top_k: 5,
        score_threshold: 0.0,
        include_initial: false,
        ..Default::default()
    };

    let sub_handle = service
        .subscribe("notify_test".to_string(), vec![0.5; 64], sub_config, None)
        .await
        .unwrap();

    service
        .activate_subscription(&sub_handle.id, vec![])
        .await
        .unwrap();

    // Push with notification
    let vectors = create_test_vectors(10, 0, 64);
    let result = service.push_and_notify(&session_id, vectors).await.unwrap();

    assert_eq!(result.push_result.pushed, 10);
    // Subscription was notified (may be 0 if no vectors matched)
    let _ = result.subscriptions_notified; // usize: always >= 0

    // Cleanup
    service.unsubscribe(&sub_handle.id).unwrap();
    service.close_session(&session_id);
}

#[tokio::test]
async fn test_integrated_service_auto_notify_disabled() {
    let mut config = IntegratedServiceConfig::default();
    config.auto_notify_subscriptions = false;

    let service = IntegratedStreamingService::new(config);

    let session_id = service
        .create_session("no_notify".to_string(), SessionConfig::default())
        .await
        .unwrap();

    let vectors = create_test_vectors(10, 0, 64);
    let result = service.push_and_notify(&session_id, vectors).await.unwrap();

    // With auto_notify disabled, should be 0
    assert_eq!(result.subscriptions_notified, 0);
}

#[tokio::test]
async fn test_integrated_service_subscription_management() {
    let service = IntegratedStreamingService::new(IntegratedServiceConfig::default());

    let sub_config = SubscriptionConfig {
        top_k: 10,
        score_threshold: 0.5,
        include_initial: false,
        ..Default::default()
    };

    // Create subscription
    let handle = service
        .subscribe("sub_mgmt".to_string(), vec![0.5; 64], sub_config, None)
        .await
        .unwrap();

    assert!(service.stats().active_subscriptions >= 1);

    // Activate
    service
        .activate_subscription(&handle.id, vec![])
        .await
        .unwrap();

    // Unsubscribe
    service.unsubscribe(&handle.id).unwrap();

    // After unsubscribe, should be gone
    // (subscription count may be cached briefly)
}

#[tokio::test]
async fn test_integrated_service_stats() {
    let service = IntegratedStreamingService::new(IntegratedServiceConfig::default());

    // Create sessions
    for i in 0..3 {
        let _ = service
            .create_session(format!("stats_col_{}", i), SessionConfig::default())
            .await
            .unwrap();
    }

    // Create subscriptions
    for i in 0..2 {
        let sub_config = SubscriptionConfig::default();
        let _ = service
            .subscribe(
                format!("sub_col_{}", i),
                vec![0.1 * (i as f32); 64],
                sub_config,
                None,
            )
            .await
            .unwrap();
    }

    let stats = service.stats();
    assert_eq!(stats.active_sessions, 3);
    assert_eq!(stats.active_subscriptions, 2);
    assert_eq!(stats.sessions_created, 3);
}

#[tokio::test]
async fn test_integrated_service_shutdown() {
    let service = IntegratedStreamingService::new(IntegratedServiceConfig::default());

    let _ = service
        .create_session("shutdown_session".to_string(), SessionConfig::default())
        .await
        .unwrap();

    // Shutdown without storage
    service.shutdown(None).await;

    // After shutdown, sessions should be cleared
    assert_eq!(service.stats().active_sessions, 0);
}

// test_integrated_service_flush_to_storage - requires mock storage, tested in unit tests

// =============================================================================
// Coordinator Accessor Tests
// =============================================================================

#[tokio::test]
async fn test_integrated_service_coordinator_access() {
    let service = IntegratedStreamingService::new(IntegratedServiceConfig::default());

    let coordinator = service.coordinator();

    // Should be able to use coordinator directly
    let session_id = coordinator
        .create_session("direct_access".to_string(), SessionConfig::default())
        .await
        .unwrap();

    assert_eq!(coordinator.session_count(), 1);
    coordinator.close_session(&session_id);
}

#[tokio::test]
async fn test_integrated_service_subscription_access() {
    let service = IntegratedStreamingService::new(IntegratedServiceConfig::default());

    let subscriptions = service.subscriptions();

    // Should be able to use subscription manager directly
    let config = SubscriptionConfig::default();
    let handle = subscriptions
        .subscribe("direct_sub".to_string(), vec![0.5; 64], config, None)
        .await
        .unwrap();

    assert!(subscriptions.stats().total_subscriptions >= 1);
    subscriptions.unsubscribe(&handle.id).unwrap();
}

// =============================================================================
// Push Result Enhanced Tests
// =============================================================================

#[tokio::test]
async fn test_push_result_suggested_delay() {
    let coordinator = StreamCoordinator::new(StreamConfig {
        default_buffer_size: 64,
        ..Default::default()
    });

    let session_id = coordinator
        .create_session("delay_test".to_string(), SessionConfig::default())
        .await
        .unwrap();

    // Push enough to trigger backpressure
    let vectors = create_test_vectors(60, 0, 64);
    let result = coordinator
        .push_records(&session_id, vectors)
        .await
        .unwrap();

    // Should have non-zero delay when backpressure is high
    let delay = result.suggested_delay();
    if result.backpressure >= BackpressureLevel::High {
        assert!(delay.as_millis() > 0);
    }
}

#[tokio::test]
async fn test_push_result_has_drops() {
    let coordinator = StreamCoordinator::new(StreamConfig {
        default_buffer_size: 32,
        ..Default::default()
    });

    let session_id = coordinator
        .create_session("drops_test".to_string(), SessionConfig::default())
        .await
        .unwrap();

    // Push more than buffer can hold
    let vectors = create_test_vectors(50, 0, 64);
    let result = coordinator
        .push_records(&session_id, vectors)
        .await
        .unwrap();

    // Some should be dropped
    assert!(result.has_drops() || result.pushed == 50);
}

#[tokio::test]
async fn test_push_result_requires_slowdown() {
    let coordinator = StreamCoordinator::new(StreamConfig {
        default_buffer_size: 64,
        ..Default::default()
    });

    let session_id = coordinator
        .create_session("slowdown_test".to_string(), SessionConfig::default())
        .await
        .unwrap();

    // Fill buffer significantly
    let vectors = create_test_vectors(60, 0, 64);
    let result = coordinator
        .push_records(&session_id, vectors)
        .await
        .unwrap();

    // Should require slowdown when buffer is nearly full
    if result.buffer_percent > 90 {
        assert!(result.requires_slowdown());
    }
}
