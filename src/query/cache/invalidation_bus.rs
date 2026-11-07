// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! TD-CACHE-10 S2 seam: cross-pod cache-invalidation events over the
//! segment-queue primitive (ADR-079 Q1 consumer groups), default-off.
//!
//! Co-design position (pressure test 2026-08-28): single-binary is the
//! operative deployment — process-local invalidation is complete, and the
//! byte-level segment caches are safe-by-immutability (no coherence needed,
//! ever). This bus exists so that WHEN a multi-pod deployment lands, the
//! invalidation signal is configuration, not new design:
//!
//! * **Publish** (writer pod, after WAL commit / flush visibility / DV
//!   write): one `CacheInvalidationEvent` to the shared `cache-invalidation`
//!   topic. Publish failure is logged and counted — it must never fail the
//!   write (TD-CACHE-10's original constraint).
//! * **Subscribe** (every other pod): consumer group keyed by pod id; each
//!   admitted event drives
//!   `CacheInvalidationCoordinator::invalidate_collection`. (It formerly also
//!   dropped the object-economy directory cache; no server component creates
//!   that cache since TD-OEDIRWM-1.)
//!
//! Ordering/dedup: `GlobalLsnAllocator` LSNs are PER-POD monotonic atomics,
//! so events carry `(pod_id, lsn)` and admission is "lsn strictly greater
//! than this pod's high-water mark" — duplicates and out-of-order replays
//! drop. Cross-pod LSN comparison is meaningless by design; the
//! single-writer-per-collection routing model (PrimaryPodRegistry) is what
//! makes per-pod sub-sequences the correct ordering domain.
//!
//! Default-off: `PROXIMADB_CACHE_INVALIDATION_BUS=1` plus a configured
//! queue root activates the bus. Everything here is exercised in-process —
//! the queue is backed by local dirs in tests.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use proximadb_queue::{Consumer, Message, Producer, QueueClient};

/// The shared topic all cache-invalidation events flow through.
pub const CACHE_INVALIDATION_TOPIC: &str = "cache-invalidation";

/// Wire schema version. Bump on a breaking `CacheInvalidationEvent` change;
/// consumers skip messages whose version they don't understand.
pub const EVENT_SCHEMA_VERSION: u8 = 1;

/// One cross-pod invalidation signal. Serialized as JSON into the queue
/// message payload (opaque to the queue; human-readable in logs; unknown
/// fields are forward-compatible).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheInvalidationEvent {
    pub schema_version: u8,
    /// Emitting pod's identity (queue consumer groups key on this too).
    pub pod_id: String,
    pub tenant: String,
    pub collection: String,
    pub kind: InvalidationKind,
    /// Emitting pod's WAL LSN at the mutation (per-pod monotonic; pair with
    /// `pod_id` for global ordering/dedup — cross-pod comparison is
    /// meaningless by design).
    pub lsn: u64,
}

/// What mutated the collection — receivers currently treat all kinds alike
/// (drop the collection's caches); the kind is carried for future
/// fine-grained (predicate/segment-scoped) invalidation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InvalidationKind {
    Write,
    Ddl,
    Flush,
    Compaction,
    DeleteVector,
}

impl CacheInvalidationEvent {
    pub fn encode(&self) -> serde_json::Result<Vec<u8>> {
        serde_json::to_vec(self)
    }

    pub fn decode(bytes: &[u8]) -> serde_json::Result<Self> {
        serde_json::from_slice(bytes)
    }

    /// The queue dedup/ordering key: `(pod_id, lsn)`.
    pub fn ordering_key(&self) -> (String, u64) {
        (self.pod_id.clone(), self.lsn)
    }
}

/// Per-pod high-water admission for consumed events. Admits an event iff
/// its LSN is strictly greater than the emitting pod's high-water mark —
/// this both deduplicates replays and drops out-of-order older events.
#[derive(Debug, Default)]
pub struct EventDedup {
    high_water: HashMap<String, u64>,
}

impl EventDedup {
    pub fn new() -> Self {
        Self::default()
    }

    /// `true` = admit (first event from this pod, or a strictly newer LSN).
    pub fn admit(&mut self, event: &CacheInvalidationEvent) -> bool {
        let (_, lsn) = event.ordering_key();
        match self.high_water.get(&event.pod_id) {
            Some(&hw) if lsn <= hw => false,
            _ => {
                self.high_water.insert(event.pod_id.clone(), lsn);
                true
            }
        }
    }
}

/// Publish/subscribe handle over the boot-built queue client. Clone-cheap
/// (`Arc` inside `QueueClient`).
#[derive(Clone)]
pub struct InvalidationBus {
    client: Arc<QueueClient>,
    pod_id: String,
}

impl InvalidationBus {
    pub fn new(client: Arc<QueueClient>, pod_id: impl Into<String>) -> Self {
        Self {
            client,
            pod_id: pod_id.into(),
        }
    }

    /// Publish one invalidation event. Errors propagate to the caller, but
    /// the WRITE-path wrapper (`publish_after_commit`-style call sites) must
    /// log-and-continue — a lost invalidation costs a TTL-bounded stale read,
    /// never correctness.
    pub async fn publish(&self, event: &CacheInvalidationEvent) -> anyhow::Result<()> {
        let producer: Producer = self.client.producer();
        producer
            .send(Message::new(
                CACHE_INVALIDATION_TOPIC,
                event.tenant.clone(),
                event.encode()?,
            ))
            .await?;
        Ok(())
    }

    /// A consumer in group `pod_id` subscribed to the invalidation topic.
    /// Each pod uses its OWN pod id as the group so every pod receives every
    /// event (broadcast), while offsets stay per-pod.
    pub fn consumer(&self) -> Consumer {
        self.client.consumer(self.pod_id.clone())
    }
}

/// Poll one batch from the invalidation topic, decode, admit via `dedup`,
/// and drive `handler` per admitted event. Returns the number of events
/// that reached the handler. Malformed/unknown-version payloads are
/// skipped (forward compatibility), never fatal.
pub async fn drive_once(
    consumer: &Consumer,
    dedup: &mut EventDedup,
    handler: &mut dyn FnMut(&CacheInvalidationEvent) -> usize,
) -> anyhow::Result<usize> {
    let deliveries = consumer
        .poll(16, std::time::Duration::from_millis(0))
        .await?;
    let mut driven = 0usize;
    let mut acked: Vec<proximadb_queue::message::MessageId> = Vec::new();
    for delivery in deliveries {
        match CacheInvalidationEvent::decode(&delivery.payload) {
            Ok(event) if event.schema_version == EVENT_SCHEMA_VERSION && dedup.admit(&event) => {
                let _ = handler(&event);
                driven += 1;
            }
            // Unknown version / malformed / duplicate / stale: durable skip.
            _ => {}
        }
        acked.push(delivery.message_id);
    }
    if !acked.is_empty() {
        consumer.ack(&acked).await?;
    }
    Ok(driven)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(pod: &str, tenant: &str, collection: &str, lsn: u64) -> CacheInvalidationEvent {
        CacheInvalidationEvent {
            schema_version: EVENT_SCHEMA_VERSION,
            pod_id: pod.to_string(),
            tenant: tenant.to_string(),
            collection: collection.to_string(),
            kind: InvalidationKind::Write,
            lsn,
        }
    }

    /// Wire round-trip incl. every kind surviving encode/decode.
    #[test]
    fn event_serde_round_trip_all_kinds() {
        for kind in [
            InvalidationKind::Write,
            InvalidationKind::Ddl,
            InvalidationKind::Flush,
            InvalidationKind::Compaction,
            InvalidationKind::DeleteVector,
        ] {
            let e = CacheInvalidationEvent {
                schema_version: EVENT_SCHEMA_VERSION,
                pod_id: "pod-7".into(),
                tenant: "t".into(),
                collection: "coll".into(),
                kind,
                lsn: 42,
            };
            let decoded =
                CacheInvalidationEvent::decode(&e.encode().expect("encode")).expect("decode");
            assert_eq!(decoded, e);
        }
        // Schema-version mismatch is detectable by the consumer.
        let mut future = event("p", "t", "c", 1);
        future.schema_version = EVENT_SCHEMA_VERSION + 1;
        let bytes = future.encode().expect("encode");
        let decoded = CacheInvalidationEvent::decode(&bytes).expect("decodes as v1 shape");
        assert_ne!(decoded.schema_version, EVENT_SCHEMA_VERSION);
    }

    /// Dedup + ordering contract: first event from a pod admits; duplicates
    /// and older-LSN replays drop; a NEWER lsn admits.
    #[test]
    fn event_dedup_admits_strictly_increasing_per_pod() {
        let mut dedup = EventDedup::new();
        assert!(dedup.admit(&event("podA", "t", "c", 10)));
        assert!(
            !dedup.admit(&event("podA", "t", "c", 10)),
            "duplicate dropped"
        );
        assert!(!dedup.admit(&event("podA", "t", "c", 9)), "older dropped");
        assert!(dedup.admit(&event("podA", "t", "c", 11)), "newer admitted");
        // Pods are independent ordering domains.
        assert!(dedup.admit(&event("podB", "t", "c", 3)));
    }

    /// The mechanism proof (in-process multi-pod): two "pods" (distinct
    /// consumer groups over one queue dir); pod A publishes a write event,
    /// pod B's drive_once admits it, drives the handler (invalidation
    /// counter), and a replay of the same event is deduped.
    #[tokio::test]
    async fn bus_publish_consume_drives_handler_with_dedup() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let root = format!("file://{}", dir.path().display());
        let client = QueueClient::open(proximadb_queue::QueueConfig {
            root,
            ..Default::default()
        })
        .await
        .expect("queue open");

        let bus_a = InvalidationBus::new(client.clone(), "podA");
        let bus_b = InvalidationBus::new(client.clone(), "podB");

        // Subscribe BEFORE publishing: a consumer group's offset initializes
        // at the current end, so events produced before the group exists are
        // (correctly) not delivered to it.
        let consumer = bus_b.consumer();
        // subscribe_all: an empty/missing partition list silently receives
        // nothing (TD-CACHE-10 closeout mitigation).
        consumer
            .subscribe_all(CACHE_INVALIDATION_TOPIC)
            .await
            .expect("subscribe");

        let published = event("podA", "t", "coll-x", 7);
        bus_a.publish(&published).await.expect("publish");

        let mut dedup = EventDedup::new();
        let invalidated = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = invalidated.clone();
        let mut handler = |e: &CacheInvalidationEvent| {
            assert_eq!(e.collection, "coll-x");
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            1_usize
        };
        let driven = drive_once(&consumer, &mut dedup, &mut handler)
            .await
            .expect("drive");
        assert_eq!(driven, 1, "one event reached the handler");
        assert_eq!(
            invalidated.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "handler invalidated the collection"
        );

        // Replay the same event (same pod, same lsn): deduped, handler not
        // re-driven — though the delivery is still acknowledged.
        bus_a.publish(&published).await.expect("re-publish");
        let driven2 = drive_once(&consumer, &mut dedup, &mut handler)
            .await
            .expect("drive 2");
        assert_eq!(driven2, 0, "duplicate (pod_id, lsn) must be deduped");
        assert_eq!(
            invalidated.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "handler NOT re-driven for a duplicate"
        );
    }
}
