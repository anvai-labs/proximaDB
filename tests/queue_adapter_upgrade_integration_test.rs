//! Upgrade admission through the public queue/factory adapter boundary.

#![cfg(unix)]

use std::collections::HashMap;
use std::sync::Arc;

use proximadb::services::queue_fs_adapter::FactoryQueueFs;
use proximadb::storage::persistence::filesystem::FilesystemFactory;
use proximadb_queue::{Message, QueueClient, QueueConfig, TopicConfig};

async fn descendant_upgrade_is_rejected(progress_only: bool) {
    let dir = tempfile::tempdir().expect("tempdir");
    let adapter_root = dir.path().join("queue-base");
    let queue_root = adapter_root.join("subqueue");
    // The old mapper concatenated the adapter root and entire queue path,
    // even when that queue was a descendant rather than the adapter root.
    let legacy = adapter_root.join(queue_root.strip_prefix("/").expect("absolute fixture"));
    let topics = HashMap::from([(
        "events".into(),
        TopicConfig {
            partition_count: 1,
            ..Default::default()
        },
    )]);
    let legacy_file = if progress_only {
        let file = legacy.join("events/0/podA/offset.meta");
        std::fs::create_dir_all(file.parent().expect("parent")).expect("legacy group");
        std::fs::write(&file, br#"{"group":"podA","committed_offset":42}"#)
            .expect("legacy progress");
        file
    } else {
        let old = QueueClient::open(QueueConfig {
            root: legacy.display().to_string(),
            topics: topics.clone(),
            ..Default::default()
        })
        .await
        .expect("legacy fixture");
        old.producer()
            .send(Message::new(
                "events",
                "tenant-a",
                b"legacy-child-frame".to_vec(),
            ))
            .await
            .expect("legacy send");
        old.shutdown().await.expect("legacy shutdown");
        drop(old);
        legacy.join("events/0/0000000000.qseg")
    };
    let before = std::fs::read(&legacy_file).expect("legacy bytes");
    let factory = Arc::new(FilesystemFactory::create_default().await.expect("factory"));
    let adapter = FactoryQueueFs::new(factory, format!("file://{}", adapter_root.display()))
        .expect("adapter");
    let error = match QueueClient::open_with_fs(
        QueueConfig {
            root: format!("file://{}", queue_root.display()),
            topics,
            ..Default::default()
        },
        Some(adapter),
    )
    .await
    {
        Err(error) => error,
        Ok(queue) => {
            queue
                .shutdown()
                .await
                .expect("stop incorrectly admitted queue");
            panic!("descendant-root admission must not abandon legacy history");
        }
    };
    assert!(error.to_string().contains("legacy queue layout"), "{error}");
    assert!(error.to_string().contains("migration"), "{error}");
    assert_eq!(
        std::fs::read(legacy_file).expect("preserved history"),
        before
    );
    assert!(!queue_root.exists(), "no canonical state before rejection");
}

#[tokio::test]
async fn descendant_root_rejects_legacy_frames_before_initialization() {
    descendant_upgrade_is_rejected(false).await;
}

#[tokio::test]
async fn descendant_root_rejects_legacy_progress_after_segment_reaping() {
    descendant_upgrade_is_rejected(true).await;
}

#[tokio::test]
async fn clean_descendant_restarts_without_touching_another_legacy_queue() {
    let dir = tempfile::tempdir().expect("tempdir");
    let adapter_root = dir.path().join("queue-base");
    let queue_root = adapter_root.join("clean-subqueue");
    let other_root = adapter_root.join("other");
    let other_legacy = adapter_root.join(other_root.strip_prefix("/").expect("absolute fixture"));
    let other_file = other_legacy.join("events/0/podA/offset.meta");
    std::fs::create_dir_all(other_file.parent().expect("parent")).expect("other legacy group");
    let other_bytes = br#"{"group":"podA","committed_offset":42}"#;
    std::fs::write(&other_file, other_bytes).expect("other legacy checkpoint");
    let config = QueueConfig {
        root: format!("file://{}", queue_root.display()),
        topics: HashMap::from([(
            "events".into(),
            TopicConfig {
                partition_count: 1,
                ..Default::default()
            },
        )]),
        ..Default::default()
    };
    let factory = Arc::new(FilesystemFactory::create_default().await.expect("factory"));
    let adapter = FactoryQueueFs::new(factory, format!("file://{}", adapter_root.display()))
        .expect("adapter");
    let first = QueueClient::open_with_fs(config.clone(), Some(adapter.clone()))
        .await
        .expect("clean descendant open");
    first
        .producer()
        .send(Message::new("events", "tenant-a", b"clean-frame".to_vec()))
        .await
        .expect("send");
    first.shutdown().await.expect("shutdown");
    drop(first);
    let restarted = QueueClient::open_with_fs(config, Some(adapter))
        .await
        .expect("clean descendant restart");
    let consumer = restarted.consumer("new-group");
    consumer.subscribe("events", &[0]).await.expect("subscribe");
    let batch = consumer
        .poll(1, std::time::Duration::ZERO)
        .await
        .expect("poll persisted frame");
    assert_eq!(batch.len(), 1);
    assert_eq!(batch[0].payload, b"clean-frame");
    consumer.shutdown().await.expect("consumer shutdown");
    restarted.shutdown().await.expect("queue shutdown");
    assert_eq!(
        std::fs::read(other_file).expect("other history preserved"),
        other_bytes
    );
}

#[tokio::test]
async fn explicit_relative_file_root_restarts_with_custom_backend_root() {
    use proximadb::storage::persistence::filesystem::{FilesystemConfig, local::LocalConfig};

    let backend = tempfile::tempdir().expect("backend root");
    let fixture = tempfile::tempdir_in(".").expect("working-directory fixture");
    let root = std::path::PathBuf::from(".")
        .join(fixture.path().file_name().expect("fixture name"))
        .join("queue");
    assert!(root.is_relative());
    let root_url = format!("file://{}", root.display());
    let config = QueueConfig {
        root: root_url.clone(),
        topics: HashMap::from([(
            "events".into(),
            TopicConfig {
                partition_count: 1,
                ..Default::default()
            },
        )]),
        ..Default::default()
    };
    let factory = Arc::new(
        FilesystemFactory::create(FilesystemConfig {
            local: Some(LocalConfig {
                root_dir: Some(backend.path().to_path_buf()),
                ..Default::default()
            }),
            ..Default::default()
        })
        .await
        .expect("custom-root factory"),
    );
    let adapter = FactoryQueueFs::new(factory, root_url).expect("explicit-relative adapter");
    let first = QueueClient::open_with_fs(config.clone(), Some(adapter.clone()))
        .await
        .expect("explicit file URL open");
    first
        .producer()
        .send(Message::new(
            "events",
            "tenant-a",
            b"relative-file-frame".to_vec(),
        ))
        .await
        .expect("send");
    first.shutdown().await.expect("shutdown");
    drop(first);
    let restarted = QueueClient::open_with_fs(config, Some(adapter))
        .await
        .expect("explicit file URL restart");
    let consumer = restarted.consumer("readers");
    consumer.subscribe("events", &[0]).await.expect("subscribe");
    let batch = consumer
        .poll(1, std::time::Duration::ZERO)
        .await
        .expect("poll recovered frame");
    assert_eq!(batch.len(), 1);
    assert_eq!(batch[0].payload, b"relative-file-frame");
    consumer.shutdown().await.expect("consumer shutdown");
    restarted.shutdown().await.expect("queue shutdown");
    assert!(root.join("events/0/0000000000.qseg").exists());
    assert!(
        !backend.path().join(&root).exists(),
        "explicit file URLs bypass backend root_dir"
    );
}

#[tokio::test]
async fn dotted_root_relative_queue_preserves_list_caller_coordinates() {
    let fixture = tempfile::tempdir_in(".").expect("fixture");
    let adapter_root =
        std::path::PathBuf::from(fixture.path().file_name().expect("fixture name")).join("queue");
    let queue_path = std::path::PathBuf::from(".").join(&adapter_root);
    assert!(adapter_root.is_relative());
    let factory = Arc::new(FilesystemFactory::create_default().await.expect("factory"));
    let adapter = FactoryQueueFs::new(factory, format!("file://{}", adapter_root.display()))
        .expect("adapter");
    // This spelling is interpreted as an explicit root-relative operation by
    // the existing mapper. LIST must preserve the caller's leading `./` so
    // QueueClient can prove each returned child belongs to its requested parent.
    let config = QueueConfig {
        root: format!("file://{}", queue_path.display()),
        topics: HashMap::from([(
            "events".into(),
            TopicConfig {
                partition_count: 1,
                ..Default::default()
            },
        )]),
        ..Default::default()
    };
    let first = QueueClient::open_with_fs(config.clone(), Some(adapter.clone()))
        .await
        .expect("dotted relative queue open");
    first
        .producer()
        .send(Message::new(
            "events",
            "tenant-a",
            b"caller-coordinates".to_vec(),
        ))
        .await
        .expect("send");
    first.shutdown().await.expect("shutdown");
    drop(first);
    let restarted = QueueClient::open_with_fs(config, Some(adapter))
        .await
        .expect("restart");
    let consumer = restarted.consumer("readers");
    consumer.subscribe("events", &[0]).await.expect("subscribe");
    let batch = consumer
        .poll(1, std::time::Duration::ZERO)
        .await
        .expect("poll");
    assert_eq!(batch.len(), 1);
    assert_eq!(batch[0].payload, b"caller-coordinates");
    consumer.shutdown().await.expect("consumer shutdown");
    restarted.shutdown().await.expect("queue shutdown");
    assert!(
        adapter_root
            .join(&adapter_root)
            .join("events/0/0000000000.qseg")
            .exists()
    );
}
