//! Disk-segment reaper — closes the segment lifecycle.
//!
//! Once a sealed segment is durably mirrored to the object archive
//! AND every consumer group has committed past its last offset, the
//! on-disk copy is redundant — it's the third-tier cache for messages
//! already (a) durable in the archive and (b) consumed past. Deleting
//! it reclaims local disk and keeps `disk_writer.segments()` cheap.
//!
//! ## Eligibility rule
//!
//! A sealed segment (id < active_id) is reapable iff:
//!
//! 1. **Upload condition**: when `object_archive` is configured, the
//!    `{segment_path}.uploaded` sidecar marker exists. If no archive
//!    is configured, this condition is vacuously satisfied (the user
//!    has opted into "local disk only" durability).
//! 2. **Consumer condition**: the minimum inclusive committed offset across
//!    every discovered group (including lease-only groups) is `Some(c)` with
//!    `c >= segment.last_offset`. If no `offset.meta` exists yet, the
//!    segment is NOT reapable (no consumer has acked anything; the
//!    messages may still be needed).
//!
//! Once both hold, the segment file + its `.uploaded` marker are
//! unlinked. The archive copy stays (that's the long-term tier).
//!
//! ## Why scan instead of subscribe
//!
//! A scan keeps the design simple — no extra channels between
//! ack/upload/reap. The default 1-second poll is far shorter than any
//! human-perceivable retention window, and the scan is O(segments)
//! which is small (16 MB rotation → at most ~hundreds even on busy
//! partitions).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use crate::QueueClient;
use crate::fs::QueueFs;
use crate::topic::PartitionId;

const SEGMENT_EXT: &str = "qseg";
const UPLOADED_MARKER_EXT: &str = "uploaded";
const DEFAULT_POLL_INTERVAL: Duration = Duration::from_secs(1);

pub struct Reaper {
    fs: Arc<dyn QueueFs>,
    poll_interval: Duration,
    archive_configured: bool,
}

impl Reaper {
    pub fn new(fs: Arc<dyn QueueFs>, archive_configured: bool) -> Self {
        Self {
            fs,
            poll_interval: DEFAULT_POLL_INTERVAL,
            archive_configured,
        }
    }

    pub fn with_poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    pub fn start(self, client: Arc<QueueClient>) -> (JoinHandle<()>, oneshot::Sender<()>) {
        let (tx, mut rx) = oneshot::channel::<()>();
        let handle = tokio::spawn(async move {
            info!(
                poll_ms = self.poll_interval.as_millis() as u64,
                archive_configured = self.archive_configured,
                "reaper started"
            );
            let mut ticker = tokio::time::interval(self.poll_interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tokio::select! {
                    _ = &mut rx => break,
                    _ = ticker.tick() => {
                        if let Err(e) = self.scan_once(&client).await {
                            warn!(error = %e, "reaper scan failed");
                        }
                    }
                }
            }
            info!("reaper stopped");
        });
        (handle, tx)
    }

    async fn scan_once(&self, client: &QueueClient) -> crate::Result<()> {
        for topic in client.topic_names().await {
            let Some(state) = client.topic_state(&topic).await else {
                continue;
            };
            for partition_id in 0..state.config.partition_count {
                let Some(writer) = state.disk_writers.get(partition_id as usize) else {
                    continue;
                };
                let writer = writer.clone();
                let active_id = writer.active_segment_id().await;
                let segments = writer.segments().await.unwrap_or_default();
                let groups = crate::offset_store::read_all_groups(
                    &self.fs,
                    client.root_path(),
                    &topic,
                    partition_id,
                )
                .await?;
                // A segment is reapable only once EVERY group has consumed
                // past it — take the min committed offset across groups.
                // None ⇒ no group has acked yet ⇒ keep the segment (pub/sub
                // safety: never delete before the slowest group has read).
                let committed = groups.iter().map(|(_, o)| *o).min().flatten();
                for segment in segments {
                    // Skip the active segment — producers are still
                    // appending to it.
                    if segment.segment_id >= active_id {
                        continue;
                    }
                    if let Err(e) = self
                        .maybe_reap(&topic, partition_id, &segment.path, committed)
                        .await
                    {
                        warn!(
                            topic = %topic,
                            partition = partition_id,
                            segment_id = segment.segment_id,
                            error = %e,
                            "reaper iteration failed",
                        );
                    }
                }
            }
        }
        Ok(())
    }

    async fn maybe_reap(
        &self,
        topic: &str,
        partition: PartitionId,
        segment_path: &Path,
        committed: Option<u64>,
    ) -> crate::Result<()> {
        // Condition 1: archive — only relevant when configured.
        let marker = marker_path(segment_path);
        if self.archive_configured && self.fs.metadata(&marker).await.is_err() {
            return Ok(()); // Archive pending; try later.
        }

        // Condition 2: a consumer commit must exist AND span past the
        // segment's last offset. Without `offset.meta`, no consumer has
        // acked anything — keep the segment.
        let committed = match committed {
            Some(c) => c,
            None => return Ok(()),
        };
        let last_offset = last_offset_in_segment(&self.fs, segment_path).await?;
        let Some(last_offset) = last_offset else {
            // Empty segment (shouldn't happen for sealed ones, but
            // defensively delete to clean up).
            self.fs.delete(segment_path).await?;
            if self.archive_configured {
                let _ = self.fs.delete(&marker).await;
            }
            return Ok(());
        };
        if committed < last_offset {
            return Ok(()); // Consumer is behind; keep the segment.
        }

        // Both conditions hold — delete the on-disk segment + marker.
        self.fs.delete(segment_path).await?;
        if self.archive_configured {
            let _ = self.fs.delete(&marker).await;
        }
        debug!(
            topic = %topic,
            partition = partition,
            ?segment_path,
            committed = committed,
            last_offset = last_offset,
            "reaper deleted on-disk segment (archive retains)",
        );
        Ok(())
    }
}

/// Return the highest frame-offset stored in `segment_path`, or `None`
/// if the segment has no parsable frames. Reads the whole file (sealed
/// segments are bounded by `disk_rotation_size_mb`, default 16 MB).
async fn last_offset_in_segment(
    fs: &Arc<dyn QueueFs>,
    segment_path: &Path,
) -> crate::Result<Option<u64>> {
    use std::io::{Cursor, Read};
    let bytes = fs.read(segment_path).await?;
    if bytes.is_empty() {
        return Ok(None);
    }
    let mut cursor = Cursor::new(&bytes[..]);
    let mut last: Option<u64> = None;
    loop {
        let mut len_buf = [0u8; 4];
        if cursor.read_exact(&mut len_buf).is_err() {
            break;
        }
        let len = u32::from_be_bytes(len_buf) as usize;
        let mut offset_buf = [0u8; 8];
        if cursor.read_exact(&mut offset_buf).is_err() {
            break;
        }
        let offset = u64::from_be_bytes(offset_buf);
        let remaining = bytes.len().saturating_sub(cursor.position() as usize);
        if len > remaining {
            break; // truncated tail
        }
        cursor.set_position(cursor.position() + len as u64);
        last = Some(offset);
    }
    Ok(last)
}

fn marker_path(disk_segment: &Path) -> PathBuf {
    let mut p = disk_segment.as_os_str().to_owned();
    p.push(format!(".{UPLOADED_MARKER_EXT}"));
    PathBuf::from(p)
}

// SEGMENT_EXT is exported for the segment listing logic elsewhere.
#[allow(dead_code)]
pub(crate) const fn segment_ext() -> &'static str {
    SEGMENT_EXT
}
