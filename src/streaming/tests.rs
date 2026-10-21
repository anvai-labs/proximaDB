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

//! Integration tests for the streaming module

use super::*;
use std::sync::Arc;
#[tokio::test]
async fn test_end_to_end_streaming() {
    let coordinator = Arc::new(StreamCoordinator::default());

    // Create session
    let session_id = coordinator
        .create_session("vectors".to_string(), SessionConfig::default())
        .await
        .expect("Failed to create session");

    // Push records in batches
    for batch in 0..10 {
        let records: Vec<proximadb_records::ProximaRecord> = (0..100)
            .map(|i| proximadb_records::ProximaRecord {
                oid: format!("vec_{}_{}", batch, i),
                embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                    "default",
                    "vector",
                    128,
                    vec![0.1 * (i as f32); 128],
                )],
                record_version: 1,
                ..Default::default()
            })
            .collect();

        let result = coordinator.push_records(&session_id, records).await;
        assert!(result.is_ok());

        let push_result = result.unwrap();
        assert_eq!(push_result.pushed, 100);
        assert_eq!(push_result.dropped, 0);
    }

    // Drain and verify
    let mut total_drained = 0;
    loop {
        let drained = coordinator
            .drain_records(&session_id, 200)
            .expect("Failed to drain");
        if drained.is_empty() {
            break;
        }
        total_drained += drained.len();
    }

    assert_eq!(total_drained, 1000);

    // Close session
    coordinator.close_session(&session_id);
    assert_eq!(coordinator.session_count(), 0);
}

#[tokio::test]
async fn test_backpressure_signaling() {
    let config = StreamConfig {
        default_buffer_size: 128, // Small buffer to trigger backpressure
        ..Default::default()
    };
    let coordinator = StreamCoordinator::new(config);

    let session_id = coordinator
        .create_session("vectors".to_string(), SessionConfig::default())
        .await
        .expect("Failed to create session");

    // Fill the buffer
    let records: Vec<proximadb_records::ProximaRecord> = (0..128)
        .map(|i| proximadb_records::ProximaRecord {
            oid: format!("vec_{}", i),
            embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                "default",
                "vector",
                128,
                vec![0.1; 128],
            )],
            record_version: 1,
            ..Default::default()
        })
        .collect();

    let result = coordinator
        .push_records(&session_id, records)
        .await
        .unwrap();

    // Should have critical backpressure
    assert!(result.backpressure >= BackpressureLevel::High);
    assert!(result.buffer_percent > 70);
}

#[tokio::test]
async fn test_concurrent_sessions() {
    let coordinator = Arc::new(StreamCoordinator::default());

    // Create multiple sessions concurrently
    let mut handles = vec![];
    for i in 0..10 {
        let coord = coordinator.clone();
        handles.push(tokio::spawn(async move {
            let session_id = coord
                .create_session(format!("collection_{}", i), SessionConfig::default())
                .await
                .expect("Failed to create session");

            // Push some records
            for _ in 0..5 {
                let records: Vec<proximadb_records::ProximaRecord> = (0..50)
                    .map(|j| proximadb_records::ProximaRecord {
                        oid: format!("vec_{}_{}", i, j),
                        embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                            "default",
                            "vector",
                            64,
                            vec![0.1; 64],
                        )],
                        record_version: 1,
                        ..Default::default()
                    })
                    .collect();

                coord.push_records(&session_id, records).await.unwrap();
            }

            session_id
        }));
    }

    let session_ids: Vec<_> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

    assert_eq!(coordinator.session_count(), 10);

    // Close all sessions
    for session_id in session_ids {
        coordinator.close_session(&session_id);
    }

    assert_eq!(coordinator.session_count(), 0);
}

#[tokio::test]
async fn test_session_stats() {
    let coordinator = StreamCoordinator::default();

    let session_id = coordinator
        .create_session("test".to_string(), SessionConfig::default())
        .await
        .unwrap();

    // Push some records
    let records: Vec<proximadb_records::ProximaRecord> = (0..100)
        .map(|i| proximadb_records::ProximaRecord {
            oid: format!("vec_{}", i),
            embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                "default",
                "vector",
                64,
                vec![0.1; 64],
            )],
            record_version: 1,
            ..Default::default()
        })
        .collect();

    coordinator
        .push_records(&session_id, records)
        .await
        .unwrap();

    // Drain some
    coordinator.drain_records(&session_id, 30).unwrap();

    // Check stats
    let stats = coordinator.get_session_info(&session_id).unwrap();
    assert_eq!(stats.collection, "test");
    assert_eq!(stats.records_received, 100);
    assert_eq!(stats.records_processed, 30);
    assert_eq!(stats.buffer_len, 70);
}

#[test]
fn test_ring_buffer_stress() {
    use std::sync::Arc;
    use std::thread;

    let buffer = Arc::new(RingBuffer::<u64>::new(8192));
    let num_threads = 8;
    let items_per_thread = 10000;

    let mut handles = vec![];

    // Producer threads
    for t in 0..num_threads {
        let buf = Arc::clone(&buffer);
        handles.push(thread::spawn(move || {
            let mut pushed = 0;
            for i in 0..items_per_thread {
                let value = t as u64 * items_per_thread as u64 + i as u64;
                while buf.try_push(value).is_err() {
                    thread::yield_now();
                }
                pushed += 1;
            }
            pushed
        }));
    }

    // Consumer thread
    let buf_consumer = Arc::clone(&buffer);
    let consumer = thread::spawn(move || {
        let expected = num_threads * items_per_thread;
        let mut consumed = 0;
        let mut empty_count = 0;

        while consumed < expected {
            match buf_consumer.try_pop() {
                Some(_) => {
                    consumed += 1;
                    empty_count = 0;
                }
                None => {
                    empty_count += 1;
                    if empty_count > 100000 {
                        // Safety: don't hang forever
                        break;
                    }
                    thread::yield_now();
                }
            }
        }
        consumed
    });

    let total_pushed: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
    let total_consumed = consumer.join().unwrap();

    // Account for any remaining in buffer
    let remaining = buffer.drain_all().len();

    assert_eq!(total_pushed, num_threads * items_per_thread);
    assert_eq!(total_consumed + remaining, num_threads * items_per_thread);
}

#[tokio::test]
async fn test_rate_limiter_integration() {
    let config = StreamConfig {
        global_rate_limit: 100, // Low rate for testing
        ..Default::default()
    };
    let coordinator = StreamCoordinator::new(config);

    let session_id = coordinator
        .create_session("test".to_string(), SessionConfig::default())
        .await
        .unwrap();

    // First batch should succeed (within burst capacity)
    let records: Vec<proximadb_records::ProximaRecord> = (0..100)
        .map(|i| proximadb_records::ProximaRecord {
            oid: format!("vec_{}", i),
            embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                "default",
                "vector",
                64,
                vec![0.1; 64],
            )],
            record_version: 1,
            ..Default::default()
        })
        .collect();

    let result = coordinator.push_records(&session_id, records).await;
    assert!(result.is_ok());

    // Second immediate batch should be rate limited
    let records: Vec<proximadb_records::ProximaRecord> = (100..200)
        .map(|i| proximadb_records::ProximaRecord {
            oid: format!("vec_{}", i),
            embeddings: vec![proximadb_records::EmbeddingCell::new_fp32(
                "default",
                "vector",
                64,
                vec![0.1; 64],
            )],
            record_version: 1,
            ..Default::default()
        })
        .collect();

    let result = coordinator.push_records(&session_id, records).await;
    assert!(matches!(result, Err(StreamError::RateLimited { .. })));
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
fn test_watermarks_configuration() {
    let wm = Watermarks::with_ratios(10000, 0.2, 0.6, 0.85);
    assert_eq!(wm.low(), 2000);
    assert_eq!(wm.high(), 6000);
    assert_eq!(wm.critical(), 8500);
    assert_eq!(wm.medium(), 4000);
}
