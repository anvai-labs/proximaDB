//! Startup recovery — replay disk segments past the per-partition committed
//! offset back into the memory tier so consumers resume seamlessly across
//! process restarts.
//!
//! Walks each topic's partition directories, reads framed messages from
//! every `.qseg` file in segment-id order, and pushes them into the
//! topic's `PartitionMemory` ring buffer, skipping any message whose offset
//! is `<=` the per-partition committed offset (`offset_store`) so already-
//! acked work is not redelivered (pinned by `restart_skips_already_acked_
//! messages`). Recovery uses the minimum across discovered groups; a lease-only
//! group with no ACK blocks skipping. Each subscription resumes at its own
//! inclusive committed offset plus one.
//!
//! Crash-safety: each message frame is `[4 BE: payload_len][8 BE: offset][bincode bytes]`.
//! A truncated final frame (process killed mid-fsync before
//! group_commit completed) is detected and silently skipped — the
//! producer never received an ack for that message, so it's safe to drop.

use std::io::{Cursor, Read};
use std::path::Path;
use std::sync::Arc;

use bincode::Options;
use tracing::{debug, info};

use crate::QueueClient;
use crate::error::QueueError;
use crate::fs::QueueFs;
use crate::message::Message;
use crate::topic::PartitionId;

const SEGMENT_EXT: &str = "qseg";

/// Extract `(segment_id, path)` from a directory entry whose filename
/// looks like `0000000123.qseg`. Returns None for any other entry
/// (the marker sidecars, hidden files, etc.).
fn parse_segment_entry(path: std::path::PathBuf) -> Option<(u64, std::path::PathBuf)> {
    let name = path.file_name()?.to_str()?.to_owned();
    let stem = name.strip_suffix(&format!(".{SEGMENT_EXT}"))?;
    let segment_id = stem.parse::<u64>().ok()?;
    Some((segment_id, path))
}

/// Replay persisted segments back into the in-memory tier on startup.
///
/// Walks every registered topic, iterates its declared partitions,
/// reads each partition directory's `.qseg` files in segment-id order,
/// and pushes deserialized messages into the partition's
/// `PartitionMemory`. Auto-created topics (registered lazily after
/// startup) don't have segments to recover, so the lazy path skips
/// recovery — only topics declared in `QueueConfig::topics` go through
/// here.
pub async fn recover(client: &QueueClient) -> crate::Result<usize> {
    let topic_names = client.topic_names().await;
    if topic_names.is_empty() {
        debug!("recovery: no topics registered; nothing to replay");
        return Ok(0);
    }

    // Compute the archive root once (None when not configured).
    let archive_root: Option<std::path::PathBuf> = client
        .config()
        .object_archive
        .as_deref()
        .map(crate::object_tier::resolve_archive_root)
        .transpose()?;

    let mut total_replayed = 0usize;
    for topic in topic_names {
        let Some(state) = client.topic_state(&topic).await else {
            continue;
        };
        for partition_id in 0..state.config.partition_count {
            let partition_dir = client
                .root_path()
                .join(&topic)
                .join(partition_id.to_string());
            let archive_partition_dir = archive_root
                .as_ref()
                .map(|root| root.join(&topic).join(partition_id.to_string()));
            let count = replay_partition(
                client.fs(),
                client.archive_fs(),
                &partition_dir,
                archive_partition_dir.as_deref(),
                &topic,
                partition_id,
                &state,
            )
            .await?;
            total_replayed += count;
        }
    }

    info!(messages_replayed = total_replayed, "recovery complete");
    Ok(total_replayed)
}

async fn replay_partition(
    fs: &Arc<dyn QueueFs>,
    archive_fs: &Arc<dyn QueueFs>,
    partition_dir: &Path,
    archive_partition_dir: Option<&Path>,
    topic: &str,
    partition: PartitionId,
    state: &crate::TopicState,
) -> crate::Result<usize> {
    // Merge segments from local disk + (optionally) the archive. For
    // each segment_id, prefer a nonempty local disk copy; fall back to
    // the archive when the disk copy is missing or an empty bootstrap
    // placeholder. A failed local read is never evidence of absence. This is the
    // fresh-node-rebuild path: ECS pod reschedules onto a new node
    // with empty NVMe; recovery loads segments straight from the
    // archive.
    let mut segments: std::collections::BTreeMap<
        u64,
        (Option<std::path::PathBuf>, Option<std::path::PathBuf>),
    > = std::collections::BTreeMap::new();
    let local_entries = fs.list(partition_dir).await?;
    for path in local_entries {
        if let Some((id, p)) = parse_segment_entry(path) {
            segments.entry(id).or_default().0 = Some(p);
        }
    }
    if let Some(archive_dir) = archive_partition_dir {
        let archive_entries = match archive_fs.list(archive_dir).await {
            Ok(entries) => entries,
            Err(QueueError::NotFound(_)) => Vec::new(),
            Err(error) => return Err(error),
        };
        for path in archive_entries {
            if let Some((id, p)) = parse_segment_entry(path) {
                segments.entry(id).or_default().1 = Some(p);
            }
        }
    }
    let mem = state
        .memory
        .get(partition as usize)
        .ok_or(QueueError::PartitionNotFound {
            topic: topic.to_string(),
            partition,
        })?
        .clone();
    let disk_writer = state
        .disk_writers
        .get(partition as usize)
        .ok_or(QueueError::PartitionNotFound {
            topic: topic.to_string(),
            partition,
        })?
        .clone();

    // Recovery skip uses the MIN committed offset across all consumer groups:
    // a frame is only left out of the replay once EVERY group has acked past
    // it (pub/sub safe — a fast group can't cause recovery to drop frames a
    // slower group still needs). None = no group has committed yet (cold
    // start, or all groups fresh) → replay every persisted frame. Each group
    // resumes at its OWN cursor on `subscribe` (offset_store::read), so this
    // skip is purely a memory-capacity optimization, not a correctness gate.
    let root_path = partition_dir
        .parent()
        .and_then(|topic_dir| topic_dir.parent())
        .ok_or_else(|| {
            QueueError::Persistence(format!(
                "recovery: cannot derive queue root from {partition_dir:?}"
            ))
        })?;
    let groups = crate::offset_store::read_all_groups(fs, root_path, topic, partition).await?;
    let committed = groups.iter().map(|(_, o)| *o).min().flatten();

    let mut replayed = 0usize;
    let mut skipped = 0usize;
    // Reaping can remove every frame while leaving durable group progress.
    // Never reuse those offsets: producer allocation uses the MAX acknowledged
    // offset as a floor, whereas replay skipping above deliberately uses MIN.
    // Reuse the existing progress authority rather than add another checkpoint.
    let mut max_next_offset = groups
        .iter()
        .filter_map(|(_, offset)| *offset)
        .max()
        .map(|offset| {
            offset.checked_add(1).ok_or_else(|| {
                QueueError::Persistence(format!(
                    "recovery: durable progress exhausted offsets for topic={topic} partition={partition}"
                ))
            })
        })
        .transpose()?;
    let mut max_recovered_segment = None;
    // BTreeMap orders the replay by ascending segment ID.
    for (segment_id, (local_path, archive_path)) in segments {
        let (path, bytes) = if let Some(path) = local_path {
            let bytes = fs.read(&path).await?;
            if bytes.is_empty() {
                if let Some(archive_path) = archive_path {
                    let archived = archive_fs.read(&archive_path).await?;
                    (archive_path, archived)
                } else {
                    (path, bytes)
                }
            } else {
                (path, bytes)
            }
        } else if let Some(path) = archive_path {
            let bytes = archive_fs.read(&path).await?;
            (path, bytes)
        } else {
            return Err(QueueError::Persistence(format!(
                "recovery: segment {segment_id} has no local or archive path"
            )));
        };
        if bytes.is_empty() {
            continue;
        }

        let mut cursor = Cursor::new(&bytes[..]);
        loop {
            // Frame: [4 BE len][8 BE offset][len bytes bincode payload].
            let mut len_buf = [0u8; 4];
            if cursor.read_exact(&mut len_buf).is_err() {
                break; // EOF
            }
            let len = u32::from_be_bytes(len_buf) as usize;
            let mut offset_buf = [0u8; 8];
            if cursor.read_exact(&mut offset_buf).is_err() {
                debug!(
                    ?path,
                    "recovery: truncated offset header, stopping segment scan"
                );
                break;
            }
            let frame_offset = u64::from_be_bytes(offset_buf);
            // Defensive: bail if the declared length would overflow the
            // remaining segment bytes (truncated final frame from a
            // crashed producer).
            let remaining = bytes.len().saturating_sub(cursor.position() as usize);
            if len > remaining {
                debug!(
                    ?path,
                    declared = len,
                    available = remaining,
                    "recovery: truncated trailing frame, stopping segment scan"
                );
                break;
            }
            let mut payload = vec![0u8; len];
            if cursor.read_exact(&mut payload).is_err() {
                break;
            }
            // Match the writer's fixed-width bincode encoding, while rejecting
            // trailing payload bytes and allocation claims beyond the frame.
            let message: Message = bincode::DefaultOptions::new()
                .with_fixint_encoding()
                .with_limit(len as u64)
                .reject_trailing_bytes()
                .deserialize(&payload)
                .map_err(|error| QueueError::Persistence(format!(
                    "recovery: malformed complete frame in {path:?} at offset {frame_offset}: {error}"
                )))?;
            let next_offset = frame_offset.checked_add(1).ok_or_else(|| {
                QueueError::Persistence(format!(
                    "recovery: offset overflow in {path:?} at offset {frame_offset}"
                ))
            })?;

            // Track the max frame_offset across ALL replayed frames
            // (including skipped ones) so the disk writer's next_offset
            // resumes past every previously-assigned offset.
            max_next_offset = Some(max_next_offset.map_or(next_offset, |m| m.max(next_offset)));
            max_recovered_segment = Some(segment_id);

            // Skip if already acked by the consumer group.
            if let Some(c) = committed
                && frame_offset <= c
            {
                skipped += 1;
                continue;
            }

            // Re-enqueue with the frame-recorded offset so MessageId is
            // stable across crash/replay.
            if mem
                .enqueue_with_offset(message, frame_offset)
                .await
                .is_err()
            {
                return Err(QueueError::Persistence(format!(
                    "recovery: memory tier full at topic={topic} partition={partition} \
                     segment={segment_id} replayed={replayed} — increase memory_capacity"
                )));
            }
            replayed += 1;
        }
    }

    // Bump the disk writer past the highest observed offset so newly-
    // appended messages don't collide with recovered ones.
    // Also leave recovered segment IDs immutable: a fresh-node append must not
    // create a partial local copy that shadows the corresponding archive.
    if let Some(segment_id) = max_recovered_segment {
        disk_writer
            .advance_past_recovered_segment(segment_id)
            .await?;
    }
    if let Some(next) = max_next_offset {
        disk_writer.set_next_offset(next);
    }

    if replayed > 0 || skipped > 0 {
        debug!(
            topic = topic,
            partition = partition,
            messages_replayed = replayed,
            messages_skipped = skipped,
            committed_offset = ?committed,
            "recovery: partition replayed"
        );
    }
    Ok(replayed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fs::LocalFs;
    use crate::memory_tier::PartitionMemory;
    use crate::{TopicConfig, TopicState};

    async fn topic_state(capacity: usize, root: &Path) -> TopicState {
        // Open a real disk writer per partition since Phase 2C-b's
        // replay_partition bumps the writer's next_offset after replay.
        let fs: Arc<dyn QueueFs> = LocalFs::new_arc();
        let cfg = TopicConfig {
            partition_count: 1,
            memory_capacity: capacity,
            ..TopicConfig::default()
        };
        let writer = crate::disk_tier::PartitionDiskWriter::open(
            "orders".to_string(),
            0,
            root.to_path_buf(),
            fs,
            cfg.clone(),
        )
        .await
        .expect("open writer");
        TopicState {
            config: cfg,
            memory: vec![Arc::new(PartitionMemory::new(0, capacity))],
            disk_writers: vec![writer],
        }
    }

    /// Build a frame in the new format: [4 BE len][8 BE offset][payload].
    /// The offset value here is what the disk writer would have assigned
    /// when this message was originally written; tests pass it explicitly
    /// so the recovery skip / max-offset behavior can be exercised
    /// deterministically.
    fn frame_with_offset(message: &Message, offset: u64) -> Vec<u8> {
        let encoded = bincode::serialize(message).unwrap();
        let mut framed = Vec::with_capacity(4 + 8 + encoded.len());
        framed.extend_from_slice(&(encoded.len() as u32).to_be_bytes());
        framed.extend_from_slice(&offset.to_be_bytes());
        framed.extend_from_slice(&encoded);
        framed
    }

    #[tokio::test]
    async fn replay_partition_rejects_missing_partition_dir() {
        let fs = LocalFs::new_arc();
        let root = tempfile::tempdir().unwrap();
        let state = topic_state(4, root.path()).await;

        let result = replay_partition(
            &fs,
            &fs,
            &root.path().join("missing"),
            None,
            "orders",
            0,
            &state,
        )
        .await;

        assert!(result.is_err());
        assert_eq!(state.memory[0].depth().await, 0);
    }

    #[tokio::test]
    async fn replay_partition_replays_valid_frames_and_stops_at_truncated_tail() {
        let fs = LocalFs::new_arc();
        let root = tempfile::tempdir().unwrap();
        let partition_dir = root.path().join("orders").join("0");
        fs.create_dir_all(&partition_dir).await.unwrap();
        fs.append(&partition_dir.join("ignore.txt"), b"not a segment")
            .await
            .unwrap();

        let mut segment_bytes =
            frame_with_offset(&Message::new("orders", "tenant-a", b"survives".to_vec()), 0);
        // Append a truncated tail: length header claims 99 bytes, only
        // 5 actually follow. Recovery must stop the scan cleanly.
        segment_bytes.extend_from_slice(&99u32.to_be_bytes());
        segment_bytes.extend_from_slice(&0u64.to_be_bytes());
        segment_bytes.extend_from_slice(b"short");
        fs.append(&partition_dir.join("0000000000.qseg"), &segment_bytes)
            .await
            .unwrap();
        let state = topic_state(4, root.path()).await;

        let replayed = replay_partition(&fs, &fs, &partition_dir, None, "orders", 0, &state)
            .await
            .unwrap();

        assert_eq!(replayed, 1);
        let restored = state.memory[0].read_from(0, 10).await;
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].message.payload, b"survives");
    }

    #[tokio::test]
    async fn replay_partition_surfaces_memory_full_as_persistence_error() {
        let fs = LocalFs::new_arc();
        let root = tempfile::tempdir().unwrap();
        let partition_dir = root.path().join("orders").join("0");
        fs.create_dir_all(&partition_dir).await.unwrap();

        let mut segment_bytes =
            frame_with_offset(&Message::new("orders", "tenant-a", b"first".to_vec()), 0);
        segment_bytes.extend_from_slice(&frame_with_offset(
            &Message::new("orders", "tenant-a", b"second".to_vec()),
            1,
        ));
        fs.append(&partition_dir.join("0000000000.qseg"), &segment_bytes)
            .await
            .unwrap();
        let state = topic_state(1, root.path()).await;

        let error = replay_partition(&fs, &fs, &partition_dir, None, "orders", 0, &state)
            .await
            .unwrap_err();

        assert!(error.to_string().contains("memory tier full"));
    }
}
