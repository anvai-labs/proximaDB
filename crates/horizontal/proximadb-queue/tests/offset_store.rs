//! offset_store integration tests — proves Consumer::ack writes the
//! committed offset to disk atomically and concurrently-safe.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use proximadb_queue::{Message, QueueClient, QueueConfig, TopicConfig};
use serde_json::Value;
use tempfile::TempDir;

fn cfg(root: &std::path::Path, name: &str, partition_count: u32) -> QueueConfig {
    let mut topics = HashMap::new();
    topics.insert(
        name.to_string(),
        TopicConfig {
            partition_count,
            memory_capacity: 256,
            ..Default::default()
        },
    );
    QueueConfig {
        root: format!("file://{}", root.display()),
        topics,
        ..QueueConfig::default()
    }
}

/// After acking 7 of 10 messages, the on-disk `offset.meta` for the
/// partition that holds them contains a committed_offset of at least 6
/// (offsets are 0-indexed, so the 7th message has offset 6).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ack_commits_offset_to_disk() {
    let tmp = TempDir::new().expect("tempdir");
    let root = tmp.path();
    let client = QueueClient::open(cfg(root, "t", 1)).await.expect("open");

    let producer = client.producer();
    for i in 0..10u32 {
        producer
            .send(Message::new("t", "tenant-a", vec![i as u8]))
            .await
            .expect("send");
    }

    let consumer = client.consumer("g");
    consumer.subscribe("t", &[0]).await.expect("subscribe");
    let batch = consumer
        .poll(7, Duration::from_millis(200))
        .await
        .expect("poll");
    assert_eq!(batch.len(), 7, "should poll 7 messages");

    let ack_ids: Vec<proximadb_queue::MessageId> = batch
        .iter()
        .map(|delivery| delivery.message_id.clone())
        .collect();
    consumer.ack(&ack_ids).await.expect("ack");

    // offset.meta must now exist on disk.
    let meta_path = root.join("t").join("0").join("g").join("offset.meta");
    assert!(meta_path.exists(), "offset.meta should be written");
    let body = std::fs::read_to_string(&meta_path).expect("read meta");
    let parsed: Value = serde_json::from_str(&body).expect("parse json");
    assert_eq!(parsed["group"], "g");
    let committed = parsed["committed_offset"].as_u64().expect("u64");
    // 7 acks of offsets 0..6 → max = 6.
    assert_eq!(committed, 6, "committed_offset should be 6 after 7 acks");
}

/// 100 concurrent acks against the same partition must not corrupt
/// `offset.meta`. The final on-disk file must be parseable and contain
/// a valid (monotonically reachable) committed_offset.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn offset_commit_is_atomic() {
    let tmp = TempDir::new().expect("tempdir");
    let root = tmp.path();
    let client = QueueClient::open(cfg(root, "t", 1)).await.expect("open");

    let producer = client.producer();
    for i in 0..100u32 {
        producer
            .send(Message::new("t", "tenant-a", vec![i as u8]))
            .await
            .expect("send");
    }

    let consumer = Arc::new(client.consumer("g"));
    consumer.subscribe("t", &[0]).await.expect("subscribe");
    // Drain so the in_flight tracker has all 100.
    let mut delivery_ids = Vec::new();
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while delivery_ids.len() < 100 && std::time::Instant::now() < deadline {
        let batch = consumer
            .poll(32, Duration::from_millis(50))
            .await
            .expect("poll");
        delivery_ids.extend(batch.into_iter().map(|delivery| delivery.message_id));
    }
    assert_eq!(delivery_ids.len(), 100, "should poll all 100 first");

    // 100 ack tasks, one per delivered identity.
    let mut handles = Vec::new();
    for id in delivery_ids {
        let c = consumer.clone();
        handles.push(tokio::spawn(async move { c.ack(&[id]).await }));
    }
    for h in handles {
        h.await.unwrap().expect("ack");
    }

    // Every delivery was ACKed successfully. Scheduling cannot change the
    // final contiguous watermark: it must reach 99, not merely be parseable.
    let meta_path = root.join("t").join("0").join("g").join("offset.meta");
    assert!(meta_path.exists(), "offset.meta should exist");
    let body = std::fs::read_to_string(&meta_path).expect("read meta");
    let parsed: Value = serde_json::from_str(&body).expect("parse json (no corruption)");
    let committed = parsed["committed_offset"].as_u64().expect("u64");
    assert_eq!(
        committed, 99,
        "all successful ACKs must reach the final offset"
    );
    assert_eq!(parsed["group"], "g");
}

/// ADR-079 §Semantics: each consumer group has its OWN committed-offset file,
/// so group A's ack does not clobber group B's cursor (the pub/sub isolation
/// property). RED for slice 2: today there is one shared `offset.meta` per
/// partition, so this test fails until offset paths are keyed by group.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn each_consumer_group_has_an_independent_committed_offset() {
    let tmp = TempDir::new().expect("tempdir");
    let root = tmp.path();
    let client = QueueClient::open(cfg(root, "t", 1)).await.expect("open");
    let producer = client.producer();
    for i in 0..5u32 {
        producer
            .send(Message::new("t", "tenant-a", vec![i as u8]))
            .await
            .expect("send");
    }

    // group A acks all five (committed_offset → 4).
    let a = client.consumer("a");
    a.subscribe("t", &[0]).await.expect("subscribe a");
    let batch = a.poll(5, Duration::from_millis(200)).await.expect("poll a");
    let ids: Vec<proximadb_queue::MessageId> = batch.iter().map(|d| d.message_id.clone()).collect();
    a.ack(&ids).await.expect("ack a");

    // group A has its OWN offset.meta under t/0/a/.
    let a_meta = root.join("t").join("0").join("a").join("offset.meta");
    assert!(a_meta.exists(), "group A writes its own offset.meta");
    let v: Value =
        serde_json::from_str(&std::fs::read_to_string(&a_meta).expect("read")).expect("parse");
    assert_eq!(v["committed_offset"].as_u64(), Some(4));
    assert_eq!(v["group"], "a");

    // group B (never subscribed) has NO offset file — independent of A.
    let b_meta = root.join("t").join("0").join("b").join("offset.meta");
    assert!(!b_meta.exists(), "group A's ack must not clobber group B");
}
