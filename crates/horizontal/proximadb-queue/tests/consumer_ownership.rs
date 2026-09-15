use proximadb_queue::{Message, QueueClient, QueueConfig, TopicConfig};
use std::time::Duration;

async fn setup() -> (tempfile::TempDir, std::sync::Arc<QueueClient>) {
    let dir = tempfile::tempdir().unwrap();
    let config = QueueConfig {
        root: format!("file://{}", dir.path().display()),
        topics: [(
            "t".into(),
            TopicConfig {
                partition_count: 1,
                ..Default::default()
            },
        )]
        .into(),
        ..Default::default()
    };
    let client = QueueClient::open(config).await.unwrap();
    for value in 0..3 {
        client
            .producer()
            .send(Message::new("t", "tenant", vec![value]))
            .await
            .unwrap();
    }
    (dir, client)
}

#[tokio::test]
async fn separate_consumers_on_one_client_cannot_share_ownership() {
    let (_dir, client) = setup().await;
    let first = client.consumer("g");
    first.subscribe("t", &[0]).await.unwrap();
    let second = client.consumer("g");
    assert!(second.subscribe("t", &[0]).await.is_err());
}

#[tokio::test]
async fn ack_does_not_skip_an_unacknowledged_gap() {
    let (dir, client) = setup().await;
    let consumer = client.consumer("g");
    consumer.subscribe("t", &[0]).await.unwrap();
    let batch = consumer.poll(3, Duration::ZERO).await.unwrap();
    consumer.ack(&[batch[2].message_id.clone()]).await.unwrap();
    assert!(
        !dir.path().join("t/0/g/offset.meta").exists(),
        "a later ACK must not make offsets 0 and 1 collectible"
    );
    consumer
        .ack(&[batch[0].message_id.clone(), batch[1].message_id.clone()])
        .await
        .unwrap();
    let bytes = std::fs::read(dir.path().join("t/0/g/offset.meta")).unwrap();
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&bytes).unwrap()["committed_offset"],
        2
    );
}

#[tokio::test]
async fn nack_redelivers_without_committing_progress() {
    let (dir, client) = setup().await;
    let consumer = client.consumer("g");
    consumer.subscribe("t", &[0]).await.unwrap();
    let batch = consumer.poll(1, Duration::ZERO).await.unwrap();
    consumer.nack(&[batch[0].message_id.clone()]).await.unwrap();
    assert!(!dir.path().join("t/0/g/offset.meta").exists());
    let again = consumer.poll(1, Duration::ZERO).await.unwrap();
    assert_eq!(again[0].message_id, batch[0].message_id);
}

#[tokio::test]
async fn corrupt_progress_is_not_a_cold_start() {
    let (dir, client) = setup().await;
    std::fs::create_dir_all(dir.path().join("t/0/g")).unwrap();
    std::fs::write(dir.path().join("t/0/g/offset.meta"), b"corrupt").unwrap();
    assert!(client.consumer("g").subscribe("t", &[0]).await.is_err());
}

#[tokio::test]
async fn group_path_aliases_are_rejected_not_silently_shared() {
    let (_dir, client) = setup().await;
    for group in ["a/b", "a:b", "..", "_g", ""] {
        assert!(
            client.consumer(group).subscribe("t", &[0]).await.is_err(),
            "aliased group {group:?} must be rejected"
        );
    }
}

#[tokio::test]
async fn existing_mixed_case_group_is_preserved_and_case_alias_is_rejected() {
    let (dir, client) = setup().await;
    let consumer = client.consumer("podA");
    consumer.subscribe("t", &[0]).await.unwrap();
    assert!(dir.path().join("t/0/podA/lease.meta").exists());
    let alias = client.consumer("poda");
    assert!(alias.subscribe("t", &[0]).await.is_err());
    client.shutdown().await.unwrap();
}

#[tokio::test]
async fn topic_paths_cannot_escape_queue_root_from_consumer_or_producer() {
    let (dir, client) = setup().await;
    let outside = tempfile::tempdir().unwrap();
    let absolute = outside.path().join("escape");
    for topic in [absolute.to_str().unwrap(), "../escape", "a/../../escape"] {
        assert!(client.consumer("g").subscribe(topic, &[0]).await.is_err());
        assert!(
            client
                .producer()
                .send(Message::new(topic, "tenant", vec![1]))
                .await
                .is_err()
        );
    }
    assert!(!absolute.exists());
    assert!(!dir.path().join("a").exists());
    client.shutdown().await.unwrap();
}

#[tokio::test]
async fn successor_owner_fences_old_ack_and_poll() {
    let (dir, client) = setup().await;
    let old = client.consumer("g");
    old.subscribe("t", &[0]).await.unwrap();
    let batch = old.poll(1, Duration::ZERO).await.unwrap();
    // Deterministically model expiry without wall-clock sleeps.
    let path = dir.path().join("t/0/g/lease.meta");
    let mut lease: proximadb_queue::leases::LeaseMeta =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    lease.expires_at_unix_nanos = 0;
    std::fs::write(&path, serde_json::to_vec(&lease).unwrap()).unwrap();
    let next = client.consumer("g");
    next.subscribe("t", &[0]).await.unwrap();
    assert!(old.ack(&[batch[0].message_id.clone()]).await.is_err());
    assert!(old.poll(1, Duration::ZERO).await.is_err());
    assert!(old.subscribe("t", &[0]).await.is_err());
    old.shutdown().await.unwrap();
    assert!(next.poll(1, Duration::ZERO).await.is_ok());
}

#[tokio::test]
async fn client_shutdown_awaits_consumers_and_prevents_new_subscription() {
    let (dir, client) = setup().await;
    let consumer = client.consumer("g");
    consumer.subscribe("t", &[0]).await.unwrap();
    client.shutdown().await.unwrap();
    let lease: proximadb_queue::leases::LeaseMeta =
        serde_json::from_slice(&std::fs::read(dir.path().join("t/0/g/lease.meta")).unwrap())
            .unwrap();
    assert_eq!(lease.expires_at_unix_nanos, 0);
    assert!(consumer.poll(1, Duration::ZERO).await.is_err());
    assert!(client.consumer("new").subscribe("t", &[0]).await.is_err());
}

#[tokio::test]
async fn consumer_rejects_second_topic_even_after_first_topic_ack_is_drained() {
    let (dir, client) = setup().await;
    let consumer = client.consumer("g");
    consumer.subscribe("t", &[0]).await.unwrap();
    let batch = consumer.poll(3, Duration::ZERO).await.unwrap();
    consumer
        .ack(
            &batch
                .iter()
                .map(|d| d.message_id.clone())
                .collect::<Vec<_>>(),
        )
        .await
        .unwrap();
    assert!(
        consumer.subscribe_all("other").await.is_err(),
        "topic-less MessageId cannot distinguish a late duplicate ACK from another topic"
    );
    assert!(
        !dir.path().join("other").exists(),
        "reject before topic creation"
    );
    consumer.shutdown().await.unwrap();
    client.shutdown().await.unwrap();
}

#[tokio::test]
async fn lease_only_group_blocks_recovery_skipping_fast_group_progress() {
    let (dir, client) = setup().await;
    let fast = client.consumer("fast");
    let slow = client.consumer("slow");
    fast.subscribe("t", &[0]).await.unwrap();
    slow.subscribe("t", &[0]).await.unwrap();
    let batch = fast.poll(3, Duration::ZERO).await.unwrap();
    fast.ack(
        &batch
            .iter()
            .map(|d| d.message_id.clone())
            .collect::<Vec<_>>(),
    )
    .await
    .unwrap();
    let fs = proximadb_queue::fs::LocalFs::new_arc();
    let groups = proximadb_queue::offset_store::read_all_groups(&fs, dir.path(), "t", 0)
        .await
        .unwrap();
    assert!(groups.contains(&("slow".into(), None)));
    assert_eq!(
        groups.iter().map(|(_, offset)| *offset).min().flatten(),
        None
    );
    client.shutdown().await.unwrap();
    let restarted = QueueClient::open(QueueConfig {
        root: format!("file://{}", dir.path().display()),
        topics: [(
            "t".into(),
            TopicConfig {
                partition_count: 1,
                ..Default::default()
            },
        )]
        .into(),
        ..Default::default()
    })
    .await
    .unwrap();
    let slow = restarted.consumer("slow");
    slow.subscribe("t", &[0]).await.unwrap();
    assert_eq!(slow.poll(3, Duration::ZERO).await.unwrap().len(), 3);
    restarted.shutdown().await.unwrap();
}

#[tokio::test]
async fn corrupt_or_misidentified_group_progress_cannot_disappear_from_scan() {
    let (dir, client) = setup().await;
    let consumer = client.consumer("g");
    consumer.subscribe("t", &[0]).await.unwrap();
    let fs = proximadb_queue::fs::LocalFs::new_arc();
    for body in [
        b"corrupt".as_slice(),
        br#"{"group":"wrong","committed_offset":99}"#,
    ] {
        std::fs::write(dir.path().join("t/0/g/offset.meta"), body).unwrap();
        assert!(
            proximadb_queue::offset_store::read_all_groups(&fs, dir.path(), "t", 0)
                .await
                .is_err()
        );
    }
    client.shutdown().await.unwrap();
}
