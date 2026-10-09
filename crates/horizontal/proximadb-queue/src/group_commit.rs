//! Per-partition group-commit coordinator.
//!
//! Mirrors the pattern from
//! `src/storage/persistence/write_ahead_log/batch_sync_coordinator.rs` in
//! the main proximadb crate, scoped to one partition's active segment file.
//!
//! ## Contract
//!
//! Producers call `wait_for_fsync(segment_path)` after appending bytes to
//! the segment. The coordinator collects pending waiters and, after
//! `max_wait` ms OR `max_batch` waiters, runs a single `fsync(segment_path)`
//! call that satisfies all of them. Each waiter receives `Ok(())` (or the
//! batch error) on its oneshot.
//!
//! This is the core of the LSM-style group-commit fsync that lets the
//! queue ack thousands of concurrent producers per fsync call instead of
//! one fsync per producer.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{Mutex, Notify, oneshot};
use tokio::task::JoinHandle;
use tracing::{debug, warn};

use crate::error::QueueError;
use crate::fs::{QueueFs, Result};

/// One waiter waiting for a fsync of `segment_path`.
struct PendingFsync {
    segment_path: PathBuf,
    reply: oneshot::Sender<Result<()>>,
}

#[derive(Debug, Clone)]
pub struct GroupCommitConfig {
    pub max_wait: Duration,
    pub max_batch: usize,
}

impl Default for GroupCommitConfig {
    fn default() -> Self {
        Self {
            max_wait: Duration::from_millis(5),
            max_batch: 64,
        }
    }
}

pub struct GroupCommitCoordinator {
    pending: Arc<Mutex<Vec<PendingFsync>>>,
    notify: Arc<Notify>,
    config: GroupCommitConfig,
    fs: Arc<dyn QueueFs>,
    _drain_task: JoinHandle<()>,
}

impl GroupCommitCoordinator {
    pub fn new(fs: Arc<dyn QueueFs>, config: GroupCommitConfig) -> Arc<Self> {
        let pending = Arc::new(Mutex::new(Vec::<PendingFsync>::new()));
        let notify = Arc::new(Notify::new());
        let drain_pending = pending.clone();
        let drain_notify = notify.clone();
        let drain_fs = fs.clone();
        let drain_max_wait = config.max_wait;
        let drain_max_batch = config.max_batch;
        let drain_task = tokio::spawn(async move {
            loop {
                // Wait for either a wake notification or the max_wait tick.
                tokio::select! {
                    _ = drain_notify.notified() => {}
                    _ = tokio::time::sleep(drain_max_wait) => {}
                }
                let mut batch = drain_pending.lock().await;
                if batch.is_empty() {
                    continue;
                }
                let take_n = batch.len().min(drain_max_batch);
                let to_flush: Vec<PendingFsync> = batch.drain(..take_n).collect();
                drop(batch);

                // Group by segment_path - one fsync per unique segment.
                let mut by_path: std::collections::HashMap<
                    PathBuf,
                    Vec<oneshot::Sender<Result<()>>>,
                > = std::collections::HashMap::new();
                for item in to_flush {
                    by_path
                        .entry(item.segment_path)
                        .or_default()
                        .push(item.reply);
                }
                for (path, replies) in by_path {
                    let batch_size = replies.len();
                    let result = drain_fs.fsync(&path).await;
                    // Group-commit amortization metric: how many waiters
                    // shared this single fsync. Higher is better (more
                    // saturated batching = lower per-message fsync cost).
                    // Extract topic + partition labels from the segment
                    // path layout `{root}/{topic}/{partition}/{segment_id}.qseg`.
                    if let Some((topic, partition)) = topic_partition_from_segment_path(&path) {
                        crate::metrics::QUEUE_FSYNC_BATCH_SIZE
                            .with_label_values(&[&topic, &partition])
                            .observe(batch_size as f64);
                    }
                    // QueueError isn't Clone (anyhow::Error inside Other isn't),
                    // so materialize the error string once and rebuild a fresh
                    // QueueError::Persistence per waiter.
                    let err_str = result.as_ref().err().map(|e| e.to_string());
                    for tx in replies {
                        let waiter_result = match &err_str {
                            Some(s) => Err(QueueError::Persistence(s.clone())),
                            None => Ok(()),
                        };
                        let _ = tx.send(waiter_result);
                    }
                    if let Err(e) = result {
                        warn!(?path, error = %e, "group_commit fsync failed");
                    } else {
                        debug!(?path, "group_commit fsync ok");
                    }
                }
            }
        });

        Arc::new(Self {
            pending,
            notify,
            config,
            fs,
            _drain_task: drain_task,
        })
    }

    /// Register a fsync request for `segment_path`. Returns when the
    /// batch fsync completes (Ok) or fails (Err).
    pub async fn wait_for_fsync(self: &Arc<Self>, segment_path: PathBuf) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        {
            let mut pending = self.pending.lock().await;
            pending.push(PendingFsync {
                segment_path,
                reply: tx,
            });
            // Wake the drain task if we've hit max_batch.
            if pending.len() >= self.config.max_batch {
                self.notify.notify_one();
            }
        }
        rx.await
            .map_err(|_| QueueError::Persistence("group_commit drainer dropped".into()))?
    }

    pub fn config(&self) -> &GroupCommitConfig {
        &self.config
    }

    #[allow(dead_code)]
    pub fn fs(&self) -> &Arc<dyn QueueFs> {
        &self.fs
    }
}

/// Extract `(topic, partition)` from a segment path with the layout
/// `{root}/{topic}/{partition}/{segment_id}.qseg`. Used by the
/// group-commit metric to label fsync batch-size observations.
/// Returns `None` for paths that don't match the expected layout
/// (e.g. test fixtures or partially-formed paths) — the metric is
/// then silently dropped rather than panicking.
fn topic_partition_from_segment_path(path: &Path) -> Option<(String, String)> {
    let partition_dir = path.parent()?;
    let topic_dir = partition_dir.parent()?;
    let partition = partition_dir.file_name()?.to_str()?.to_string();
    let topic = topic_dir.file_name()?.to_str()?.to_string();
    Some((topic, partition))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fs::Metadata;
    use async_trait::async_trait;
    use std::sync::Mutex as StdMutex;

    #[derive(Debug)]
    struct RecordingFs {
        fsyncs: StdMutex<Vec<PathBuf>>,
        fail_fsync: bool,
    }

    impl RecordingFs {
        fn new(fail_fsync: bool) -> Arc<Self> {
            Arc::new(Self {
                fsyncs: StdMutex::new(Vec::new()),
                fail_fsync,
            })
        }
    }

    #[async_trait]
    impl QueueFs for RecordingFs {
        async fn create_dir_all(&self, _path: &std::path::Path) -> Result<()> {
            Ok(())
        }

        async fn append(&self, _path: &std::path::Path, _data: &[u8]) -> Result<()> {
            Ok(())
        }

        async fn fsync(&self, path: &std::path::Path) -> Result<()> {
            self.fsyncs.lock().unwrap().push(path.to_path_buf());
            if self.fail_fsync {
                Err(QueueError::Persistence("disk full".to_string()))
            } else {
                Ok(())
            }
        }

        async fn read(&self, _path: &std::path::Path) -> Result<Vec<u8>> {
            Ok(Vec::new())
        }

        async fn list(&self, _dir: &std::path::Path) -> Result<Vec<PathBuf>> {
            Ok(Vec::new())
        }

        async fn rename(&self, _from: &std::path::Path, _to: &std::path::Path) -> Result<()> {
            Ok(())
        }

        async fn delete(&self, _path: &std::path::Path) -> Result<()> {
            Ok(())
        }

        async fn metadata(&self, _path: &std::path::Path) -> Result<Metadata> {
            Ok(Metadata {
                size_bytes: 0,
                is_directory: false,
            })
        }
    }

    #[test]
    fn default_group_commit_config_matches_queue_defaults() {
        let config = GroupCommitConfig::default();

        assert_eq!(config.max_wait, Duration::from_millis(5));
        assert_eq!(config.max_batch, 64);
    }

    #[tokio::test]
    async fn wait_for_fsync_batches_waiters_for_same_segment() {
        let fs = RecordingFs::new(false);
        let coordinator = GroupCommitCoordinator::new(
            fs.clone(),
            GroupCommitConfig {
                max_wait: Duration::from_secs(60),
                max_batch: 2,
            },
        );
        let segment = PathBuf::from("/tmp/topic/0/0000000000.qseg");

        let (first, second) = tokio::join!(
            coordinator.wait_for_fsync(segment.clone()),
            coordinator.wait_for_fsync(segment.clone())
        );

        first.unwrap();
        second.unwrap();
        assert_eq!(coordinator.config().max_batch, 2);
        assert_eq!(fs.fsyncs.lock().unwrap().as_slice(), &[segment]);
        let _ = coordinator.fs();
    }

    #[tokio::test]
    async fn wait_for_fsync_propagates_persistence_errors() {
        let fs = RecordingFs::new(true);
        let coordinator = GroupCommitCoordinator::new(
            fs.clone(),
            GroupCommitConfig {
                max_wait: Duration::from_secs(60),
                max_batch: 1,
            },
        );

        let error = coordinator
            .wait_for_fsync(PathBuf::from("/tmp/topic/0/0000000000.qseg"))
            .await
            .unwrap_err();

        assert!(error.to_string().contains("disk full"));
        assert_eq!(fs.fsyncs.lock().unwrap().len(), 1);
    }
}
