//! Per-partition disk tier - appends serialized messages to segment files
//! and coordinates fsync via the per-partition `GroupCommitCoordinator`.
//!
//! ## Segment naming
//!
//! Mirrors the WAL convention from
//! `src/storage/persistence/write_ahead_log/disk_manager.rs:127-142`:
//!
//! ```text
//! {queue_root}/{topic}/{partition_id}/{segment_id:010}.qseg
//! ```
//!
//! `segment_id` is a zero-padded monotonic u64 so list ordering is
//! lexicographic = chronological.
//!
//! Active segment is the highest-numbered file in the partition
//! directory. When it crosses `disk_rotation_size_mb`, the writer
//! seals it (atomic by ceasing writes) and opens a new active segment
//! with `segment_id + 1`.
//!
//! ## Write protocol
//!
//! Each appended message is framed as:
//!
//! ```text
//! [4 bytes BE: payload_len] [payload_len bytes: bincode(Message)]
//! ```
//!
//! Bincode is the workspace's default binary format; recovery reads
//! the file in this same framing.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::Mutex;
use tracing::trace;

use crate::config::TopicConfig;
use crate::error::QueueError;
use crate::fs::{QueueFs, Result};
use crate::group_commit::{GroupCommitConfig, GroupCommitCoordinator};
use crate::message::Message;
use crate::topic::PartitionId;

const SEGMENT_EXT: &str = "qseg";

/// File-on-disk record describing one segment.
#[derive(Debug, Clone)]
pub struct Segment {
    pub topic: String,
    pub partition: PartitionId,
    pub segment_id: u64,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub sealed: bool,
}

/// Per-partition disk writer. One instance per `(topic, partition)`.
pub struct PartitionDiskWriter {
    topic: String,
    partition: PartitionId,
    root: PathBuf,
    fs: Arc<dyn QueueFs>,
    group_commit: Arc<GroupCommitCoordinator>,
    config: TopicConfig,
    /// Mutex around (active_segment_id, active_segment_size_bytes).
    /// Locked briefly during append + rotation check.
    state: Mutex<DiskWriterState>,
    /// Per-partition monotonic offset counter. Each `append` reserves
    /// the next value atomically and writes it into the frame's
    /// 8-byte big-endian offset header. After process restart,
    /// `recovery::recover` bumps this past the replayed max via
    /// `set_next_offset` so newly-appended messages don't collide with
    /// recovered ones.
    next_offset: AtomicU64,
}

#[derive(Debug)]
struct DiskWriterState {
    active_segment_id: u64,
    active_segment_path: PathBuf,
    active_segment_size: u64,
}

impl PartitionDiskWriter {
    pub async fn open(
        topic: String,
        partition: PartitionId,
        root: PathBuf,
        fs: Arc<dyn QueueFs>,
        config: TopicConfig,
    ) -> Result<Arc<Self>> {
        let partition_dir = root.join(&topic).join(partition.to_string());
        fs.create_dir_all(&partition_dir).await?;

        // Discover existing segments to resume from. Highest id is the active
        // one (or 0 if none exist).
        let existing = list_segments(&*fs, &partition_dir, &topic, partition).await?;
        let (active_segment_id, active_segment_path, active_segment_size) =
            if let Some(latest) = existing.into_iter().max_by_key(|s| s.segment_id) {
                (latest.segment_id, latest.path, latest.size_bytes)
            } else {
                let id = 0u64;
                let path = segment_path(&partition_dir, id);
                // Touch the file so subsequent appends/lists are consistent.
                fs.append(&path, &[]).await?;
                (id, path, 0)
            };

        let group_commit = GroupCommitCoordinator::new(
            fs.clone(),
            GroupCommitConfig {
                max_wait: config.group_commit_max_wait,
                max_batch: config.group_commit_max_batch,
            },
        );

        Ok(Arc::new(Self {
            topic,
            partition,
            root,
            fs,
            group_commit,
            config,
            state: Mutex::new(DiskWriterState {
                active_segment_id,
                active_segment_path,
                active_segment_size,
            }),
            // Recovery overrides this via `set_next_offset` after
            // scanning existing segments; cold-start leaves it at 0.
            next_offset: AtomicU64::new(0),
        }))
    }

    /// Recovery uses this after replaying disk segments to bump the
    /// next-assigned offset past the highest one observed. Without this,
    /// newly-appended messages on a recovered queue would re-use offsets
    /// that already exist in old segments → consumer dedup confusion.
    pub fn set_next_offset(&self, value: u64) {
        self.next_offset.store(value, Ordering::Relaxed);
    }

    pub fn current_next_offset(&self) -> u64 {
        self.next_offset.load(Ordering::Relaxed)
    }

    /// Recovery may replay an archive while local bootstrap files are empty.
    /// Future writes must use a new segment ID, never create a partial local
    /// replacement that shadows the archived segment on the next restart.
    pub(crate) async fn advance_past_recovered_segment(&self, recovered: u64) -> Result<()> {
        let mut state = self.state.lock().await;
        if state.active_segment_id > recovered {
            return Ok(());
        }
        let id = recovered
            .checked_add(1)
            .ok_or_else(|| QueueError::Persistence("recovered segment ID exhausted".into()))?;
        let path = segment_path(
            &self.root.join(&self.topic).join(self.partition.to_string()),
            id,
        );
        self.fs.append(&path, &[]).await?;
        state.active_segment_id = id;
        state.active_segment_path = path;
        state.active_segment_size = 0;
        Ok(())
    }

    /// The currently-active (still-growing) segment id. Object-tier
    /// upload skips this one and only mirrors sealed segments (those
    /// with id strictly less than this).
    pub async fn active_segment_id(&self) -> u64 {
        self.state.lock().await.active_segment_id
    }

    /// Serialize + append + register fsync. Returns the segment path that
    /// the caller should await on via `wait_for_fsync` (Strict mode) plus
    /// the offset assigned to this message.
    ///
    /// Frame format: `[4 BE: payload_len][8 BE: offset][payload]`. The
    /// offset is durably colocated with the message so recovery can
    /// reconstruct it without a side-channel.
    pub async fn append(self: &Arc<Self>, message: &Message) -> Result<AppendOutcome> {
        let offset = self
            .next_offset
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                next.checked_add(1)
            })
            .map_err(|_| QueueError::Persistence("queue disk offset exhausted".into()))?;
        let bytes = bincode::serialize(message)
            .map_err(|e| QueueError::Persistence(format!("serialize: {e}")))?;
        let mut framed = Vec::with_capacity(4 + 8 + bytes.len());
        framed.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
        framed.extend_from_slice(&offset.to_be_bytes());
        framed.extend_from_slice(&bytes);

        let frame_len = framed.len() as u64;
        let mut state = self.state.lock().await;

        // Rotation check - if appending this frame would push past the
        // rotation threshold AND the segment is non-empty, seal + open new.
        let rotation_bytes = (self.config.disk_rotation_size_mb as u64) * 1024 * 1024;
        if state.active_segment_size > 0 && state.active_segment_size + frame_len > rotation_bytes {
            let new_id = state.active_segment_id + 1;
            let new_path = segment_path(
                &self.root.join(&self.topic).join(self.partition.to_string()),
                new_id,
            );
            self.fs.append(&new_path, &[]).await?;
            state.active_segment_id = new_id;
            state.active_segment_path = new_path;
            state.active_segment_size = 0;
            trace!(
                topic = %self.topic,
                partition = self.partition,
                segment_id = new_id,
                "segment rotated"
            );
        }

        let active_path = state.active_segment_path.clone();
        self.fs.append(&active_path, &framed).await?;
        state.active_segment_size += frame_len;
        drop(state);

        Ok(AppendOutcome {
            segment_path: active_path,
            offset,
        })
    }

    /// Block until the named segment has been fsync'd by the group-commit
    /// coordinator. Strict-mode producers call this after `append`.
    pub async fn wait_for_fsync(self: &Arc<Self>, segment_path: PathBuf) -> Result<()> {
        self.group_commit.wait_for_fsync(segment_path).await
    }

    pub async fn segments(&self) -> Result<Vec<Segment>> {
        let partition_dir = self.root.join(&self.topic).join(self.partition.to_string());
        list_segments(&*self.fs, &partition_dir, &self.topic, self.partition).await
    }
}

#[derive(Debug)]
pub struct AppendOutcome {
    pub segment_path: PathBuf,
    pub offset: u64,
}

/// Build the segment file path for a given partition + segment_id.
fn segment_path(partition_dir: &Path, segment_id: u64) -> PathBuf {
    partition_dir.join(format!("{segment_id:010}.{SEGMENT_EXT}"))
}

async fn list_segments(
    fs: &dyn QueueFs,
    partition_dir: &Path,
    topic: &str,
    partition: PartitionId,
) -> Result<Vec<Segment>> {
    let entries = fs.list(partition_dir).await?;
    let mut segments = Vec::new();
    for path in entries {
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Some(stem) = name.strip_suffix(&format!(".{SEGMENT_EXT}")) else {
            continue;
        };
        let Ok(segment_id) = stem.parse::<u64>() else {
            continue;
        };
        let size_bytes = fs.metadata(&path).await.map(|m| m.size_bytes).unwrap_or(0);
        segments.push(Segment {
            topic: topic.to_string(),
            partition,
            segment_id,
            path,
            size_bytes,
            sealed: false, // sealing is purely logical - active = highest id
        });
    }
    Ok(segments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fs::LocalFs;

    fn rotating_config() -> TopicConfig {
        TopicConfig {
            partition_count: 1,
            memory_capacity: 8,
            disk_rotation_size_mb: 0,
            ..TopicConfig::default()
        }
    }

    #[tokio::test]
    async fn open_touches_initial_segment_and_reports_segment_metadata() {
        let fs = LocalFs::new_arc();
        let root = tempfile::tempdir().unwrap();
        let writer = PartitionDiskWriter::open(
            "orders".to_string(),
            0,
            root.path().to_path_buf(),
            fs,
            TopicConfig::default(),
        )
        .await
        .unwrap();

        let segments = writer.segments().await.unwrap();

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].topic, "orders");
        assert_eq!(segments[0].partition, 0);
        assert_eq!(segments[0].segment_id, 0);
        assert_eq!(segments[0].size_bytes, 0);
        assert!(!segments[0].sealed);
        assert!(segments[0].path.ends_with("0000000000.qseg"));
    }

    #[tokio::test]
    async fn exhausted_offset_does_not_write_or_wrap() {
        let fs = LocalFs::new_arc();
        let root = tempfile::tempdir().unwrap();
        let writer = PartitionDiskWriter::open(
            "orders".into(),
            0,
            root.path().into(),
            fs.clone(),
            rotating_config(),
        )
        .await
        .unwrap();
        writer.set_next_offset(u64::MAX);
        let path = writer.segments().await.unwrap()[0].path.clone();
        let before = fs.read(&path).await.unwrap();
        assert!(
            writer
                .append(&Message::new("orders", "tenant", vec![1]))
                .await
                .is_err()
        );
        assert_eq!(writer.current_next_offset(), u64::MAX);
        assert_eq!(fs.read(&path).await.unwrap(), before);
    }

    #[tokio::test]
    async fn append_frames_messages_and_rotates_when_threshold_is_exceeded() {
        let fs = LocalFs::new_arc();
        let root = tempfile::tempdir().unwrap();
        let writer = PartitionDiskWriter::open(
            "orders".to_string(),
            0,
            root.path().to_path_buf(),
            fs.clone(),
            rotating_config(),
        )
        .await
        .unwrap();

        let first_message = Message::new("orders", "tenant-a", b"first".to_vec());
        let first = writer.append(&first_message).await.unwrap();
        let second = writer
            .append(&Message::new("orders", "tenant-a", b"second".to_vec()))
            .await
            .unwrap();

        assert_ne!(first.segment_path, second.segment_path);
        assert!(first.segment_path.ends_with("0000000000.qseg"));
        assert!(second.segment_path.ends_with("0000000001.qseg"));

        // Frame format: [4 BE: len][8 BE: offset][len bytes: bincode payload]
        let first_bytes = fs.read(&first.segment_path).await.unwrap();
        let len = u32::from_be_bytes(first_bytes[0..4].try_into().unwrap()) as usize;
        let frame_offset = u64::from_be_bytes(first_bytes[4..12].try_into().unwrap());
        let restored: Message = bincode::deserialize(&first_bytes[12..12 + len]).unwrap();
        assert_eq!(restored.topic, "orders");
        assert_eq!(restored.payload, b"first");
        assert_eq!(frame_offset, first.offset);
        assert_eq!(frame_offset, 0, "first message gets offset 0");

        let mut segment_ids: Vec<u64> = writer
            .segments()
            .await
            .unwrap()
            .into_iter()
            .map(|segment| segment.segment_id)
            .collect();
        // Directory iteration order is backend/platform dependent; this test
        // verifies the rotated segment inventory, not an ordering contract.
        segment_ids.sort_unstable();
        assert_eq!(segment_ids, vec![0, 1]);
    }
}
