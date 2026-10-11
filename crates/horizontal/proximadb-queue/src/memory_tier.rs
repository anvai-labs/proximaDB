//! Memory tier — retained, offset-indexed per-partition buffer (ADR-079 §Semantics).
//!
//! Each `(topic, partition)` owns one `PartitionMemory`. Producers append
//! messages; **consumer groups read by cursor** (`read_from(offset, max)`)
//! without removing them, so multiple independent groups (pub/sub) each see
//! the whole stream. Messages are retained until trimmed below the low
//! watermark (consumer/reaper). The disk tier writes through transparently:
//! every successful append to memory has already had its bytes appended to
//! the active disk segment, so the segment log is the source of truth.
//!
//! Storage is a `BTreeMap<offset, entry>` so concurrent producers (which
//! receive offsets from the disk writer in commit order but may enqueue into
//! memory out of order) never violate contiguity — inserts are by key.
//! `read_from` walks the **contiguous prefix** from the cursor, stopping at the
//! first gap, so a consumer never sees a hole a lagging producer has not filled
//! yet (Kafka's high-water-mark property). This replaces the earlier pop-once
//! `ArrayQueue`, which was fundamentally single-consumer-per-partition.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::{Mutex, Notify};

use crate::message::{BackpressureHint, Message, MessageId};
use crate::topic::PartitionId;

const SOFT_PCT: f32 = 0.75;
const HARD_PCT: f32 = 0.95;

pub struct PartitionMemory {
    pub(crate) partition: PartitionId,
    capacity: usize,
    buf: Mutex<RetainedBuf>,
    /// Monotonic high-water mark — the next offset a producer will receive.
    next_offset: AtomicU64,
    /// Wake consumers blocked on `poll` when something is appended.
    pub(crate) notify: Notify,
}

struct RetainedBuf {
    entries: BTreeMap<u64, MemoryEntry>,
}

#[derive(Clone)]
pub struct MemoryEntry {
    pub message: Message,
    pub message_id: MessageId,
    pub offset: u64,
}

impl PartitionMemory {
    pub fn new(partition: PartitionId, capacity: usize) -> Self {
        let cap = capacity.max(1);
        Self {
            partition,
            capacity: cap,
            buf: Mutex::new(RetainedBuf {
                entries: BTreeMap::new(),
            }),
            next_offset: AtomicU64::new(0),
            notify: Notify::new(),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub async fn depth(&self) -> usize {
        self.buf.lock().await.entries.len()
    }

    pub async fn depth_pct(&self) -> f32 {
        self.depth().await as f32 / self.capacity as f32
    }

    pub async fn pressure(&self) -> Option<PressureLevel> {
        let pct = self.depth_pct().await;
        if pct >= HARD_PCT {
            Some(PressureLevel::Hard(pct))
        } else if pct >= SOFT_PCT {
            Some(PressureLevel::Soft(pct))
        } else {
            None
        }
    }

    /// The next offset a producer will receive (the high-water mark of the log).
    pub fn next_offset(&self) -> u64 {
        self.next_offset.load(Ordering::Relaxed)
    }

    /// Append a message, assigning the next offset.
    #[allow(clippy::result_large_err)]
    pub async fn try_enqueue(
        self: &Arc<Self>,
        message: Message,
    ) -> std::result::Result<(MessageEntry, Option<BackpressureHint>), Message> {
        let offset =
            match self
                .next_offset
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                    next.checked_add(1)
                }) {
                Ok(offset) => offset,
                Err(_) => return Err(message),
            };
        self.enqueue_at(message, offset).await
    }

    /// Append a message with a pre-assigned offset (recovery / disk-owned
    /// offset assignment). The disk frame's offset is the source of truth
    /// across restart; this advances the internal counter past it.
    #[allow(clippy::result_large_err)]
    pub async fn enqueue_with_offset(
        self: &Arc<Self>,
        message: Message,
        offset: u64,
    ) -> std::result::Result<(MessageEntry, Option<BackpressureHint>), Message> {
        let Some(next) = offset.checked_add(1) else {
            return Err(message);
        };
        self.next_offset.fetch_max(next, Ordering::Relaxed);
        self.enqueue_at(message, offset).await
    }

    async fn enqueue_at(
        self: &Arc<Self>,
        message: Message,
        offset: u64,
    ) -> std::result::Result<(MessageEntry, Option<BackpressureHint>), Message> {
        let id = MessageId::new(self.partition, /* segment_id */ 0, offset);
        let entry = MemoryEntry {
            message: message.clone(),
            message_id: id.clone(),
            offset,
        };
        {
            let mut buf = self.buf.lock().await;
            if buf.entries.len() >= self.capacity {
                return Err(message);
            }
            // Insert by offset key — concurrent producers may enqueue out of
            // order; the BTreeMap tolerates that and `read_from` serves only
            // the contiguous prefix, so gaps from in-flight producers never
            // reach a consumer.
            buf.entries.insert(offset, entry);
        }
        self.notify.notify_waiters();
        let hint = match self.pressure().await {
            Some(PressureLevel::Soft(_)) => Some(BackpressureHint::Soft),
            _ => None,
        };
        Ok((
            MessageEntry {
                message_id: id,
                offset,
            },
            hint,
        ))
    }

    /// Read up to `max` messages starting at `cursor` **without removing them**
    /// — the pub/sub primitive. Walks the **contiguous prefix** from `cursor`:
    /// stops at the first missing offset (a concurrent producer has not filled
    /// it yet) so a consumer never observes a hole. If `cursor` lags behind the
    /// trimmed base, clamps up to the first available offset.
    pub async fn read_from(self: &Arc<Self>, cursor: u64, max: usize) -> Vec<MemoryEntry> {
        let buf = self.buf.lock().await;
        let mut out = Vec::with_capacity(max);
        let mut expected: Option<u64> = None;
        for (&off, entry) in buf.entries.range(cursor..) {
            match expected {
                // First available key — clamp a lagging cursor up to it.
                None => {}
                Some(exp) if off == exp => {}
                Some(_) => break, // gap — stop at the contiguous prefix boundary
            }
            if out.len() >= max {
                break;
            }
            out.push(entry.clone());
            expected = off.checked_add(1);
            if expected.is_none() {
                break;
            }
        }
        out
    }

    /// Drop retained entries whose offset is `< watermark`.
    pub async fn trim_below(self: &Arc<Self>, watermark: u64) {
        let mut buf = self.buf.lock().await;
        // split_off returns the >= watermark half and leaves self with < half;
        // reassigning keeps the >= half, dropping the trimmed prefix.
        buf.entries = buf.entries.split_off(&watermark);
    }
}

#[derive(Debug, Clone, Copy)]
pub enum PressureLevel {
    Soft(f32),
    Hard(f32),
}

impl PressureLevel {
    pub fn pct(self) -> f32 {
        match self {
            Self::Soft(p) | Self::Hard(p) => p,
        }
    }
}

/// Lightweight view returned from enqueue — the caller turns this into a full
/// `MessageReceipt` once disk fsync is confirmed (Strict) or immediately (Lazy).
pub struct MessageEntry {
    pub message_id: MessageId,
    pub offset: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(n: u8) -> Message {
        Message::new("topic", "tenant", vec![n])
    }

    #[tokio::test]
    async fn partition_memory_capacity_is_never_zero() {
        let memory = PartitionMemory::new(3, 0);
        assert_eq!(memory.partition, 3);
        assert_eq!(memory.capacity(), 1);
        assert_eq!(memory.depth().await, 0);
        assert_eq!(memory.depth_pct().await, 0.0);
        assert!(memory.pressure().await.is_none());
    }

    #[tokio::test]
    async fn enqueue_assigns_partition_scoped_ids_and_read_preserves_fifo_order() {
        let memory = Arc::new(PartitionMemory::new(5, 4));

        let (first, hint) = memory.try_enqueue(message(1)).await.unwrap();
        let (second, _) = memory.try_enqueue(message(2)).await.unwrap();

        assert_eq!(first.message_id, MessageId::new(5, 0, 0));
        assert_eq!(first.offset, 0);
        assert_eq!(second.message_id, MessageId::new(5, 0, 1));
        assert!(hint.is_none());
        assert_eq!(memory.depth().await, 2);

        let read = memory.read_from(0, 10).await;
        assert_eq!(read.len(), 2);
        assert_eq!(read[0].message.payload, vec![1]);
        assert_eq!(read[1].message.payload, vec![2]);
        // Non-consuming — depth unchanged (pub/sub).
        assert_eq!(memory.depth().await, 2);
    }

    #[tokio::test]
    async fn pressure_transitions_from_none_to_soft_to_hard() {
        let memory = Arc::new(PartitionMemory::new(1, 4));

        assert!(memory.pressure().await.is_none());
        memory.try_enqueue(message(1)).await.unwrap();
        memory.try_enqueue(message(2)).await.unwrap();
        assert!(memory.pressure().await.is_none());

        let (_, hint) = memory.try_enqueue(message(3)).await.unwrap();
        assert!(matches!(hint, Some(BackpressureHint::Soft)));
        assert!(matches!(
            memory.pressure().await.unwrap(),
            PressureLevel::Soft(_)
        ));

        memory.try_enqueue(message(4)).await.unwrap();
        assert!(matches!(
            memory.pressure().await.unwrap(),
            PressureLevel::Hard(_)
        ));
    }

    #[tokio::test]
    async fn full_window_rejects_append_but_offset_counter_remains_monotonic() {
        let memory = Arc::new(PartitionMemory::new(2, 1));

        let first = memory.try_enqueue(message(1)).await.unwrap().0;
        let rejected = match memory.try_enqueue(message(2)).await {
            Ok(_) => panic!("expected full window to reject append"),
            Err(message) => message,
        };
        assert_eq!(first.offset, 0);
        assert_eq!(rejected.payload, vec![2]);

        memory.trim_below(1).await; // free one slot
        let next = memory.try_enqueue(message(3)).await.unwrap().0;
        assert_eq!(next.offset, 2);
        assert_eq!(next.message_id, MessageId::new(2, 0, 2));
    }

    #[tokio::test]
    async fn read_from_respects_max_and_range_bounds() {
        let memory = Arc::new(PartitionMemory::new(0, 3));
        memory.try_enqueue(message(1)).await.unwrap(); // offset 0
        memory.try_enqueue(message(2)).await.unwrap(); // offset 1
        memory.try_enqueue(message(3)).await.unwrap(); // offset 2

        assert_eq!(memory.read_from(0, 0).await.len(), 0); // max=0 → none
        assert_eq!(memory.read_from(0, 2).await.len(), 2); // max=2 from cursor 0
        assert_eq!(memory.read_from(2, 2).await.len(), 1); // cursor 2 → only offset 2
        assert!(memory.read_from(3, 2).await.is_empty()); // past the end
    }

    // ---- ADR-079 §Semantics: pub/sub (consumer-group) ----

    #[tokio::test]
    async fn read_from_is_non_consuming_so_two_cursors_see_the_same_messages() {
        let memory = Arc::new(PartitionMemory::new(1, 8));
        for n in 1..=4u8 {
            memory.try_enqueue(message(n)).await.unwrap();
        }

        let group_a = memory.read_from(0, 10).await;
        let group_b = memory.read_from(0, 10).await;

        assert_eq!(group_a.len(), 4, "group A sees all messages");
        assert_eq!(
            group_b.len(),
            4,
            "group B sees all messages (not consumed by A)"
        );
        assert_eq!(
            group_a.iter().map(|e| e.offset).collect::<Vec<_>>(),
            group_b.iter().map(|e| e.offset).collect::<Vec<_>>(),
        );
    }

    #[tokio::test]
    async fn read_from_advances_per_call_without_consuming_other_groups() {
        let memory = Arc::new(PartitionMemory::new(2, 8));
        for n in 1..=4u8 {
            memory.try_enqueue(message(n)).await.unwrap();
        }
        let a1 = memory.read_from(0, 2).await;
        let a2 = memory.read_from(2, 2).await;
        assert_eq!(a1.len(), 2);
        assert_eq!(a2.len(), 2);
        assert_eq!(a1[0].offset, 0);
        assert_eq!(a2[0].offset, 2);
        let b = memory.read_from(0, 10).await;
        assert_eq!(b.len(), 4);
    }

    #[tokio::test]
    async fn trim_below_drops_old_entries_but_a_lagging_read_clamps_to_base() {
        let memory = Arc::new(PartitionMemory::new(9, 8));
        for n in 1..=4u8 {
            memory.try_enqueue(message(n)).await.unwrap();
        }
        memory.trim_below(2).await;
        assert_eq!(memory.depth().await, 2);
        let read = memory.read_from(0, 10).await;
        assert_eq!(read.len(), 2);
        assert_eq!(read[0].offset, 2);
    }

    /// A gap (offset not yet inserted by a concurrent producer) truncates the
    /// read to the contiguous prefix — the Kafka high-water-mark property.
    #[tokio::test]
    async fn read_from_stops_at_the_first_gap() {
        let memory = Arc::new(PartitionMemory::new(1, 8));
        // Insert 0,1,2,4 — leave a gap at 3 (simulate an in-flight producer).
        memory.enqueue_with_offset(message(0), 0).await.unwrap();
        memory.enqueue_with_offset(message(1), 1).await.unwrap();
        memory.enqueue_with_offset(message(2), 2).await.unwrap();
        memory.enqueue_with_offset(message(4), 4).await.unwrap();

        let read = memory.read_from(0, 10).await;
        assert_eq!(read.len(), 3, "stops at the gap before offset 3");
        assert_eq!(read[2].offset, 2);
        // Filling the gap extends the contiguous prefix to include 4.
        memory.enqueue_with_offset(message(3), 3).await.unwrap();
        let read = memory.read_from(0, 10).await;
        assert_eq!(read.len(), 5);
        assert_eq!(read[4].offset, 4);
    }
}
