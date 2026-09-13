/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! Embedding drainer — consumes the `embed-ingest` topic from
//! `proximadb-queue`, embeds via the in-process `EmbeddingService`
//! singleton, and inserts the populated records into the target
//! collection. Phase 2G of the queue plan.
//!
//! ## Replaces the obsolete pending-collection design
//!
//! An earlier scaffold of this module used a pseudo-queue collection
//! (`pending_embed`) as the staging surface for async embedding work.
//! That design was abandoned in favor of the dedicated
//! `proximadb-queue` subsystem (per README locked decision:
//! write-many-read-once messaging belongs in a real queue, not in a
//! ProximaDB collection). This module is now the consumer side of
//! that queue.
//!
//! ## Message contract
//!
//! Producers (REST `/v2/documents?mode=async` in Phase 2H) serialize
//! [`EmbedIngestPayload`] as JSON into the queue `Message.payload`.
//! The drainer deserializes, embeds the text records via
//! `EmbeddingService::embed_sync`, and forwards the populated
//! `ProximaRecord`s to the target collection through the
//! [`DrainerInsertSink`] trait — which production wires to
//! `UnifiedHandlers::handle_record_insert_batch_for_tenant`.
//!
//! ## DrainerInsertSink trait
//!
//! Abstracts the insert target so tests can validate the drain logic
//! without spinning up a full proximadb-server. Production has one
//! impl (wrapping UnifiedHandlers); tests provide their own mock.
//!
//! ## What's still deferred
//!
//! - **SST bulk-load bypass** (Phase 2F): today, the drainer's insert
//!   goes through the normal WAL → memtable path. Locked invariant #5
//!   says async should bulk-load SST segments directly. The
//!   [`DrainerInsertSink`] trait makes this swap-in-place when the
//!   storage-engine refactor lands.
//! - **DLQ on max_attempts**: a failed batch terminates the drainer and
//!   releases its consumer lease so a supervisor-created replacement can
//!   replay from durable progress; no explicit DLQ promotion yet.
//! - **Multi-replica partition assignment**: this drainer subscribes
//!   to ALL partitions by default. Cross-process leases (Phase 2E)
//!   prevent two replicas from competing; assignment via partition
//!   ranges or rendezvous-hash is a follow-up.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use proximadb_embedding::EmbeddingService;
use proximadb_embedding::config::{EmbedRoute, EmbedRouteIdentity};
use proximadb_embedding::scheduler::IngestMode;
use proximadb_embedding::service::{EmbedBatch, EmbedRecord};
use proximadb_queue::QueueClient;
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

/// Default topic name for text-only ingest events the drainer
/// processes. Aligned with the README's topic naming convention.
pub const EMBED_INGEST_TOPIC: &str = "embed-ingest";

/// One pending record's worth of payload as the producer serializes it
/// into the queue message body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedIngestRecord {
    pub oid: String,
    pub text: String,
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, String>,
}

/// The full queue-message payload. One message can carry many records
/// (the v3 REST handler batches records from a single API call into
/// one message, so the drainer can embed them as a batch).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedIngestPayload {
    pub target_collection: String,
    pub tenant_id: String,
    /// Credential-free route identity admitted by the producer. The drainer
    /// resolves fresh credentials but rejects model or geometry drift.
    pub embedding_route_identity: EmbedRouteIdentity,
    /// Collection geometry observed during admission.
    pub expected_dimension: u32,
    pub records: Vec<EmbedIngestRecord>,
}

struct ReadyEmbedIngestPayload {
    payload: EmbedIngestPayload,
    route: EmbedRoute,
    /// Stable queue-message identity (TD-SANDHI-3): the drainer acks only after every route
    /// group succeeds, and a failed batch is redelivered on restart/lease takeover (in-process
    /// redelivery never happens — the read cursor advances on delivery) and re-embeds — this id
    /// is the at-most-once key for the re-emitted usage event.
    message_id: String,
}

/// Insert path the drainer calls once embedding has populated vectors.
/// Production wraps `UnifiedHandlers`; tests provide an in-memory mock.
#[async_trait]
pub trait DrainerInsertSink: Send + Sync {
    async fn insert(
        &self,
        target_collection: &str,
        tenant_id: &str,
        records: Vec<EmbeddedRecord>,
    ) -> anyhow::Result<()>;
}

/// Record handed to the insert sink: text + populated vector + metadata.
/// Tests assert on this shape; the production sink projects it into
/// `ProximaRecord` for the UnifiedHandlers call.
#[derive(Debug, Clone)]
pub struct EmbeddedRecord {
    pub oid: String,
    pub text: String,
    pub vector: Vec<f32>,
    pub vector_dim: u32,
    pub metadata: std::collections::HashMap<String, String>,
    /// Collection's canonical embedding precision. `None` defaults to
    /// `Fp32` (legacy behavior — sink builds an `EmbeddingValues::Fp32`
    /// cell). `Some(target)` instructs the sink to coerce the fp32
    /// vector to `target` via `EmbeddingValues::from_fp32_lossy` before
    /// constructing the `EmbeddingCell`.
    ///
    /// The drainer populates this from a per-collection lookup
    /// (`CanonicalPrecisionResolver::resolve`) once the resolver is
    /// plumbed into the drainer's construction site. Until then, all
    /// records carry `None` and the sink stays on the fp32 path.
    pub target_precision: Option<proximadb_records::EmbeddingScalarType>,
}

#[derive(Debug, Clone)]
pub struct EmbeddingDrainerConfig {
    /// Topic name to consume from. Defaults to `EMBED_INGEST_TOPIC`.
    pub topic: String,
    /// Consumer group identity. Single-group-per-topic per the locked
    /// architectural invariant; default `"embed-drainer"`.
    pub group_id: String,
    /// Max messages drained per poll. The drainer embeds them as one
    /// `EmbedBatch` so larger values amortize inference overhead — but
    /// they also enlarge the at-least-once redelivery window on crash.
    pub batch_size: usize,
    /// Max wait per poll iteration.
    pub poll_wait: Duration,
}

impl Default for EmbeddingDrainerConfig {
    fn default() -> Self {
        Self {
            topic: EMBED_INGEST_TOPIC.to_string(),
            group_id: "embed-drainer".to_string(),
            batch_size: 32,
            poll_wait: Duration::from_millis(100),
        }
    }
}

pub struct EmbeddingDrainer {
    queue: Arc<QueueClient>,
    embed_service: Arc<EmbeddingService>,
    sink: Arc<dyn DrainerInsertSink>,
    config: EmbeddingDrainerConfig,
    /// When set, the drainer calls `resolver.resolve(table_id)` per payload
    /// to populate each `EmbeddedRecord.target_precision` with the
    /// collection's `canonical_embedding_precision`. When `None`, all
    /// records ship with `target_precision: None` and the sink stays on
    /// the legacy fp32 path. Set via `with_precision_resolver` after
    /// construction.
    precision_resolver:
        Option<Arc<proximadb_catalog::canonical_precision::CanonicalPrecisionResolver>>,
}

impl EmbeddingDrainer {
    pub fn new(
        queue: Arc<QueueClient>,
        embed_service: Arc<EmbeddingService>,
        sink: Arc<dyn DrainerInsertSink>,
        config: EmbeddingDrainerConfig,
    ) -> Self {
        Self {
            queue,
            embed_service,
            sink,
            config,
            precision_resolver: None,
        }
    }

    /// Attach a `CanonicalPrecisionResolver` so the drainer can stamp
    /// `EmbeddedRecord.target_precision` from each collection's
    /// `canonical_embedding_precision`. Without this, records flow
    /// through as fp32 regardless of the collection's setting.
    ///
    /// The drainer maps a `target_collection: String` to a
    /// `TableIdentifier` using the same convention as
    /// `CollectionManager::collection_table_identifier`:
    /// `TableIdentifier::parse(name)`, with unqualified names defaulting
    /// to the `"default"` namespace.
    pub fn with_precision_resolver(
        mut self,
        resolver: Arc<proximadb_catalog::canonical_precision::CanonicalPrecisionResolver>,
    ) -> Self {
        self.precision_resolver = Some(resolver);
        self
    }

    /// Map a drainer-side `target_collection` string to the catalog
    /// `TableIdentifier`. Mirrors
    /// `src/services/collection/manager.rs:collection_table_identifier`
    /// so the resolver hits the same key the manager wrote on
    /// collection creation.
    fn collection_to_table_identifier(
        target_collection: &str,
    ) -> proximadb_catalog::TableIdentifier {
        let parsed = proximadb_catalog::TableIdentifier::parse(target_collection);
        if parsed.namespace.is_empty() {
            proximadb_catalog::TableIdentifier::new(vec!["default".to_string()], parsed.name)
        } else {
            parsed
        }
    }

    /// Spawn the drainer onto the current tokio runtime. Returns the
    /// result-bearing JoinHandle plus a shutdown oneshot. Drop the sender (or
    /// send `()`) to stop the loop after the current iteration. Subscription,
    /// polling, batch-processing, and queue-publication failures are terminal
    /// and surface through the handle; a supervisor decides whether and when
    /// to construct a new consumer.
    ///
    /// `partitions` declares which partitions this drainer owns —
    /// typically all of them for a single-replica deployment. With
    /// multi-replica scaleout, each replica passes a disjoint subset
    /// and the cross-process lease (Phase 2E) enforces ownership.
    pub fn start(
        self,
        partitions: Vec<u32>,
    ) -> (JoinHandle<anyhow::Result<()>>, oneshot::Sender<()>) {
        let (tx, mut rx) = oneshot::channel::<()>();
        let handle = tokio::spawn(async move {
            info!(
                topic = %self.config.topic,
                group = %self.config.group_id,
                partitions = ?partitions,
                batch_size = self.config.batch_size,
                "embedding drainer started"
            );
            let consumer = self.queue.consumer(self.config.group_id.clone());
            let outcome = match consumer.subscribe(&self.config.topic, &partitions).await {
                Err(error) => Err(anyhow::anyhow!("drainer subscribe failed: {error}")),
                Ok(()) => loop {
                    tokio::select! {
                        _ = &mut rx => {
                            info!("embedding drainer received shutdown signal");
                            break Ok(());
                        }
                        poll_result = consumer.poll(self.config.batch_size, self.config.poll_wait) => {
                            match poll_result {
                                Ok(messages) if messages.is_empty() => continue,
                                Ok(messages) => {
                                    if let Err(error) = self.process_batch(&consumer, messages).await {
                                        break Err(anyhow::anyhow!("drainer batch failed: {error:#}"));
                                    }
                                }
                                Err(error) => {
                                    break Err(anyhow::anyhow!("drainer poll failed: {error}"));
                                }
                            }
                        }
                    }
                },
            };
            if let Err(task_error) = &outcome {
                error!(error = %task_error, "embedding drainer task failed; releasing consumer ownership");
            }
            let shutdown_result = consumer.shutdown().await;
            if let Err(shutdown_error) = &shutdown_result {
                error!(error = %shutdown_error, "embedding drainer consumer shutdown failed");
            }
            info!("embedding drainer stopped");
            match (outcome, shutdown_result) {
                (Ok(()), Ok(())) => Ok(()),
                (Err(error), Ok(())) => Err(error),
                (Ok(()), Err(shutdown)) => Err(anyhow::anyhow!(
                    "drainer consumer shutdown failed: {shutdown}"
                )),
                (Err(error), Err(shutdown)) => Err(anyhow::anyhow!(
                    "drainer failed: {error:#}; consumer shutdown also failed: {shutdown}"
                )),
            }
        });
        (handle, tx)
    }

    /// Process one polled batch: parse → embed → insert → ack.
    ///
    /// This is the function the SST-bulk-load optimization (deferred
    /// Phase 2F) will rewrite. Today it loops sink.insert per message;
    /// future version groups all batch records, sorts by oid, and
    /// calls a `BulkLoader::ingest_sorted_segment` once.
    async fn process_batch(
        &self,
        consumer: &proximadb_queue::Consumer,
        deliveries: Vec<proximadb_queue::Delivery>,
    ) -> anyhow::Result<()> {
        let delivery_ids: Vec<proximadb_queue::MessageId> = deliveries
            .iter()
            .map(|delivery| delivery.message_id.clone())
            .collect();
        let mut partition_queues: std::collections::BTreeMap<
            proximadb_queue::PartitionId,
            std::collections::VecDeque<ReadyEmbedIngestPayload>,
        > = std::collections::BTreeMap::new();

        for delivery in &deliveries {
            let payload: EmbedIngestPayload = match serde_json::from_slice(&delivery.payload) {
                Ok(p) => p,
                Err(e) => {
                    warn!(error = %e, "drainer: malformed payload; acking to avoid hot-looping");
                    continue;
                }
            };
            if payload.tenant_id != delivery.tenant_id {
                anyhow::bail!(
                    "drainer: envelope tenant '{}' does not match payload tenant '{}'",
                    delivery.tenant_id,
                    payload.tenant_id
                );
            }
            if payload.records.is_empty() {
                anyhow::bail!("drainer: payload contains no records");
            }
            if let Some(record) = payload
                .records
                .iter()
                .find(|record| record.text.trim().is_empty())
            {
                anyhow::bail!("drainer: record '{}' contains no text", record.oid);
            }
            let route = self
                .embed_service
                .resolve_admitted_route(&payload.tenant_id, &payload.embedding_route_identity)
                .ok_or_else(|| {
                    let current = self.embed_service.resolve_route(&payload.tenant_id);
                    anyhow::anyhow!(
                        "drainer: tenant route identity changed after admission: expected {:?}, got {:?}",
                        payload.embedding_route_identity,
                        EmbedRouteIdentity::from(&current)
                    )
                })?;
            let route_dimension = u32::try_from(route.dimension())
                .map_err(|_| anyhow::anyhow!("drainer: route dimension exceeds u32"))?;
            if route_dimension != payload.expected_dimension {
                anyhow::bail!(
                    "drainer: admitted dimension {} does not match route dimension {}",
                    payload.expected_dimension,
                    route_dimension
                );
            }
            let partition = delivery
                .message_id
                .partition()
                .ok_or_else(|| anyhow::anyhow!("drainer: malformed delivery id"))?;
            // MessageId renders as "{partition}:{segment}:{offset}" — the stable identity of
            // this logical ingest across redeliveries (TD-SANDHI-3 idempotency key).
            let message_id = delivery.message_id.0.clone();
            partition_queues
                .entry(partition)
                .or_default()
                .push_back(ReadyEmbedIngestPayload {
                    payload,
                    route,
                    message_id,
                });
        }

        // Preserve FIFO within each queue partition. At each step, batch all
        // compatible jobs currently at independent partition heads; consume
        // consecutive same-route jobs from each selected partition.
        while let Some(route) = partition_queues
            .values()
            .find_map(|queue| queue.front().map(|ready| ready.route.clone()))
        {
            let mut group = Vec::new();
            for queue in partition_queues.values_mut() {
                while queue.front().is_some_and(|ready| ready.route == route) {
                    let Some(ready) = queue.pop_front() else {
                        break;
                    };
                    group.push(ready);
                }
            }
            self.process_route_group(route, group).await?;
        }

        if !delivery_ids.is_empty() {
            consumer
                .ack(&delivery_ids)
                .await
                .map_err(anyhow::Error::new)?;
            debug!(count = delivery_ids.len(), "drainer batch ack'd");
        }
        Ok(())
    }

    async fn process_route_group(
        &self,
        route: EmbedRoute,
        group: Vec<ReadyEmbedIngestPayload>,
    ) -> anyhow::Result<()> {
        let mut batch_records = Vec::new();
        let mut payload_ranges = Vec::with_capacity(group.len());
        for ready in &group {
            let payload = &ready.payload;
            let start = batch_records.len();
            batch_records.extend(payload.records.iter().map(|record| EmbedRecord {
                id: record.oid.clone(),
                text: record.text.clone(),
                tenant_id: payload.tenant_id.clone(),
            }));
            payload_ranges.push(start..batch_records.len());
        }

        let result = self
            .embed_service
            .embed_sync_with_route(
                EmbedBatch {
                    records: batch_records,
                    mode: IngestMode::Async,
                },
                route.clone(),
            )
            .await
            .map_err(|e| anyhow::anyhow!("drainer embed failed: {e}"))?;
        // TD-SANDHI-3: the adapter-measured wall clock (queue wait excluded, measured inside
        // the scheduler worker — the usage-event `duration_ms` contract).
        let provider_duration_ms = result.provider_duration_ms;
        let batch_records_total = batch_record_count(&payload_ranges);
        let dimension = u32::try_from(route.dimension())
            .map_err(|_| anyhow::anyhow!("drainer: route dimension exceeds u32"))?;
        // TD-SANDHI-1: the provider's real token usage is batch-level (all payloads in this
        // route group). Split it across payloads proportionally by record count so each tenant
        // is metered its share of the measured tokens.
        let batch_usage = result.usage;
        let total_records = batch_records_total as u64;

        // Meter EVERY payload before ALL shape validation (the vector-count check below
        // included): the provider already consumed the whole batch's tokens even when the
        // returned vectors are unusable — and each event's idempotency_key (queue message id)
        // makes at-least-once redelivery safe.
        for (i, ready) in group.iter().enumerate() {
            let payload = &ready.payload;
            let payload_records = payload.records.len() as u64;
            let real_input_tokens = batch_usage
                .map(|usage| measured_share(usage.input_tokens, payload_records, total_records));
            record_embedding_consumption(
                payload,
                &route,
                real_input_tokens,
                carrier_duration(i, provider_duration_ms),
                &ready.message_id,
            );
        }

        if result.vectors.len() != batch_records_total {
            anyhow::bail!(
                "drainer: provider returned {} vectors for {} records",
                result.vectors.len(),
                batch_records_total
            );
        }

        for (ready, range) in group.iter().zip(payload_ranges) {
            let payload = &ready.payload;
            let vectors = result
                .vectors
                .get(range)
                .ok_or_else(|| anyhow::anyhow!("drainer: embed result shape mismatch"))?;
            if let Some(vector) = vectors
                .iter()
                .find(|vector| vector.len() != payload.expected_dimension as usize)
            {
                anyhow::bail!(
                    "drainer: provider returned dimension {} for admitted dimension {}",
                    vector.len(),
                    payload.expected_dimension
                );
            }
            let target_precision = self.resolve_target_precision(payload).await;
            let embedded = payload
                .records
                .iter()
                .zip(vectors.iter())
                .map(|(record, vector)| EmbeddedRecord {
                    oid: record.oid.clone(),
                    text: record.text.clone(),
                    vector: vector.clone(),
                    vector_dim: dimension,
                    metadata: record.metadata.clone(),
                    target_precision,
                })
                .collect();
            self.sink
                .insert(&payload.target_collection, &payload.tenant_id, embedded)
                .await
                .map_err(|e| anyhow::anyhow!("drainer sink insert failed: {e}"))?;
        }
        Ok(())
    }

    async fn resolve_target_precision(
        &self,
        payload: &EmbedIngestPayload,
    ) -> Option<proximadb_records::EmbeddingScalarType> {
        let resolver = self.precision_resolver.as_ref()?;
        let table_id = Self::collection_to_table_identifier(&payload.target_collection);
        match resolver.resolve(&table_id).await {
            Ok(precision) => Some(precision),
            Err(e) => {
                warn!(
                    collection = %payload.target_collection,
                    tenant = %payload.tenant_id,
                    error = %e,
                    "drainer: precision resolver failed; falling back to fp32"
                );
                None
            }
        }
    }
}

fn batch_record_count(ranges: &[std::ops::Range<usize>]) -> usize {
    ranges.last().map_or(0, |range| range.end)
}

/// The route group's batch wall clock is stamped on exactly ONE event — the first payload's
/// (TD-SANDHI-3). N events × full duration would inflate any `SUM(duration_ms)` rollup N×
/// (provider utilization >100%); with one carrier, sum-over-events stays exact and the single
/// honest latency sample survives. Split out as a fn so the selection policy is pinned by test
/// (`carrier_duration_is_first_payload_only` fails on a regression to unconditional `Some`).
fn carrier_duration(payload_index: usize, provider_duration_ms: u64) -> Option<u64> {
    (payload_index == 0).then_some(provider_duration_ms)
}

/// A measured batch never attributes a record-carrying payload zero tokens: floor-only could
/// yield 0 (2 real tokens across 3 payloads), which on the wire is indistinguishable from the
/// gateway-bug zeros TD-SANDHI-3 refuses to certify — and would drop 100% of the measurement
/// from KEU in that regime. Clamping the floored share up to 1 over-counts by at most one
/// token per floored payload (≤ `batch_size` ≈ 32 tokens/batch), keeping the sum over payloads
/// within a few tokens of the measured batch total. Preconditions hold at the call site: a
/// measured batch implies `input_tokens > 0` (parser guards) and payloads are non-empty.
fn measured_share(batch_input_tokens: u64, payload_records: u64, total_records: u64) -> u64 {
    split_input_tokens(batch_input_tokens, payload_records, total_records).max(1)
}

/// Split a batch's total input-token count across one payload, proportionally by its record
/// share (TD-SANDHI-1). A route group can batch several tenants' payloads into one provider
/// call, whose usage is reported batch-wide; this attributes each tenant its slice. Returns 0
/// for an empty batch (guards div-by-zero).
fn split_input_tokens(batch_input_tokens: u64, payload_records: u64, total_records: u64) -> u64 {
    if total_records == 0 {
        return 0;
    }
    ((u128::from(batch_input_tokens) * u128::from(payload_records)) / u128::from(total_records))
        as u64
}

/// Meter one payload's embedding consumption (TD-SANDHI-1 / ADR-067).
///
/// `real_input_tokens` carries the provider's **measured** input-token count for this payload
/// (external providers that report `usage`); when `None` the count*512 heuristic is used (local
/// BGE, or BYO whose contract has no usage). `duration_ms` is `Some` only on the route group's
/// designated carrier event (TD-SANDHI-3). `message_id` is the queue-message identity, carried
/// as the event's `idempotency_key` so at-least-once redelivery cannot double-bill. For
/// external routes, also emits the neutral usage event at the egress boundary (default-inert
/// unless `PROXIMADB_EMIT_USAGE_EVENTS`).
fn record_embedding_consumption(
    payload: &EmbedIngestPayload,
    route: &EmbedRoute,
    real_input_tokens: Option<u64>,
    duration_ms: Option<u64>,
    message_id: &str,
) {
    let embedding_count = payload.records.len() as u64;
    let provider_reported = real_input_tokens.is_some();
    let input_tokens = real_input_tokens.unwrap_or_else(|| embedding_count.saturating_mul(512));
    // Output tokens stay a compute proxy (count × dimension) — embeddings emit no completion
    // tokens, so no provider reports them; this is ProximaDB's own KEU storage unit.
    let output_tokens = embedding_count.saturating_mul(route.dimension() as u64);
    let (provider, model, external) = route_labels(route);
    crate::metrics::consumption_metrics::record_keu_units(
        Some(&payload.tenant_id),
        provider,
        &model,
        "embed_batch",
        input_tokens,
        output_tokens,
    );
    // ADR-067 Fix 2: emit the shared neutral usage event at the external-provider boundary.
    // Local BGE is self-hosted (no external egress) → no event.
    if external {
        // TD-SANDHI-3: provider-reported when the provider returned a usage figure, estimated
        // for the count×512 heuristic.
        let basis = if provider_reported {
            crate::metrics::usage_event::UsageBasis::ProviderReported
        } else {
            crate::metrics::usage_event::UsageBasis::Estimated
        };
        build_external_embedding_event(
            payload,
            provider,
            &model,
            input_tokens,
            basis,
            duration_ms,
            message_id,
        )
        .emit();
    }
}

/// Neutral (provider slug, model id, external-egress?) labels for one embedding route.
fn route_labels(route: &EmbedRoute) -> (&'static str, String, bool) {
    match route {
        EmbedRoute::BgeSmall => ("victor", "bge-small-en-v1.5".to_string(), false),
        EmbedRoute::BgeLarge => ("victor", "bge-large-en-v1.5".to_string(), false),
        EmbedRoute::BgeM3 => ("victor", "bge-m3".to_string(), false),
        EmbedRoute::AzureOpenAi { model } => ("azure_openai", format!("azure_{model:?}"), true),
        EmbedRoute::OpenAi { model } => ("openai", format!("openai_{model:?}"), true),
        EmbedRoute::Cohere { model } => ("cohere", format!("cohere_{model:?}"), true),
        EmbedRoute::Byo { url, .. } => ("byo", url.clone(), true),
    }
}

/// Build (without emitting) the neutral usage event for one external embedding payload,
/// stamped with measurement provenance and the queue-message idempotency key (TD-SANDHI-3).
/// Split from [`record_embedding_consumption`] so the stamping is unit-testable without the
/// env-gated emit.
fn build_external_embedding_event(
    payload: &EmbedIngestPayload,
    provider: &'static str,
    model: &str,
    input_tokens: u64,
    usage_basis: crate::metrics::usage_event::UsageBasis,
    duration_ms: Option<u64>,
    message_id: &str,
) -> crate::metrics::usage_event::UsageEvent {
    let mut event = crate::metrics::usage_event::UsageEvent::external_embedding(
        provider,
        model,
        Some(payload.tenant_id.as_str()),
        input_tokens,
        "embed_batch",
    )
    .with_provenance(usage_basis, duration_ms);
    event.idempotency_key = Some(message_id.to_string());
    event
}

// ── tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use proximadb_catalog::Catalog;
    use proximadb_embedding::config::{ByoAuth, EmbeddingConfig};
    use proximadb_embedding::scheduler::EmbedSchedulerConfig;
    use proximadb_queue::{Message, QueueConfig, TopicConfig};
    use std::collections::HashMap;
    use std::io::{Read, Write};
    use std::sync::Arc;
    use tempfile::TempDir;
    use tokio::sync::Mutex;

    use crate::testing::InMemoryTestCatalog;

    #[derive(Default)]
    struct RecordingSink {
        calls: Mutex<Vec<(String, String, Vec<EmbeddedRecord>)>>,
    }

    #[async_trait]
    impl DrainerInsertSink for RecordingSink {
        async fn insert(
            &self,
            target_collection: &str,
            tenant_id: &str,
            records: Vec<EmbeddedRecord>,
        ) -> anyhow::Result<()> {
            self.calls.lock().await.push((
                target_collection.to_string(),
                tenant_id.to_string(),
                records,
            ));
            Ok(())
        }
    }

    struct FailingSink;

    #[async_trait]
    impl DrainerInsertSink for FailingSink {
        async fn insert(
            &self,
            _target_collection: &str,
            _tenant_id: &str,
            _records: Vec<EmbeddedRecord>,
        ) -> anyhow::Result<()> {
            anyhow::bail!("injected sink failure")
        }
    }

    fn ensure_embedding_singleton() {
        if proximadb_embedding::EmbeddingService::try_global().is_some() {
            return;
        }
        let _ = proximadb_embedding::EmbeddingService::initialize(
            EmbeddingConfig {
                route: EmbedRoute::BgeSmall,
            },
            EmbedSchedulerConfig::default(),
        );
    }

    fn queue_cfg(tmp: &std::path::Path) -> QueueConfig {
        let mut topics = HashMap::new();
        topics.insert(
            EMBED_INGEST_TOPIC.to_string(),
            TopicConfig {
                partition_count: 2,
                lease_duration: Duration::from_secs(30),
                ..Default::default()
            },
        );
        QueueConfig {
            root: format!("file://{}", tmp.display()),
            topics,
            ..QueueConfig::default()
        }
    }

    fn start_byo_test_endpoint(vectors: Vec<Vec<f32>>) -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind BYO test server");
        let addr = listener.local_addr().expect("local addr");
        std::thread::spawn(move || {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);
            let body = serde_json::json!({
                "embeddings": vectors,
                "model_version": "test",
            })
            .to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(response.as_bytes());
        });
        format!("http://{}", addr)
    }

    fn with_byo_auth(route: &EmbedRoute, auth: ByoAuth) -> EmbedRoute {
        let EmbedRoute::Byo {
            url,
            declared_dim,
            declared_precision,
            batch_size,
            timeout_ms,
            ..
        } = route
        else {
            panic!("test helper requires a BYO route");
        };
        EmbedRoute::Byo {
            url: url.clone(),
            auth,
            declared_dim: *declared_dim,
            declared_precision: *declared_precision,
            batch_size: *batch_size,
            timeout_ms: *timeout_ms,
        }
    }

    /// Producer sends one well-formed payload; drainer embeds + the
    /// sink receives a record with a non-empty vector.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn drainer_embeds_and_forwards_to_sink() {
        ensure_embedding_singleton();
        let tmp = TempDir::new().expect("tempdir");
        let queue = QueueClient::open(queue_cfg(tmp.path()))
            .await
            .expect("queue open");
        let embed_service = proximadb_embedding::EmbeddingService::global();
        let route = EmbedRoute::Byo {
            url: start_byo_test_endpoint(vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]]),
            auth: ByoAuth::None,
            declared_dim: 3,
            declared_precision: proximadb_records::EmbeddingScalarType::Fp32,
            batch_size: 8,
            timeout_ms: 1_000,
        };
        embed_service.update_tenant_route("tenant-a", route.clone());
        let sink = Arc::new(RecordingSink::default());

        let producer = queue.producer();
        let payload = EmbedIngestPayload {
            target_collection: "knowledge".to_string(),
            tenant_id: "tenant-a".to_string(),
            embedding_route_identity: (&route).into(),
            expected_dimension: 3,
            records: vec![
                EmbedIngestRecord {
                    oid: "doc-1".to_string(),
                    text: "what is rust async runtime".to_string(),
                    metadata: HashMap::new(),
                },
                EmbedIngestRecord {
                    oid: "doc-2".to_string(),
                    text: "tokio mpsc channel backpressure".to_string(),
                    metadata: HashMap::new(),
                },
            ],
        };
        let bytes = serde_json::to_vec(&payload).unwrap();
        producer
            .send(Message::new(EMBED_INGEST_TOPIC, "tenant-a", bytes))
            .await
            .expect("send");

        let sink_for_drainer: Arc<dyn DrainerInsertSink> = sink.clone();
        let drainer = EmbeddingDrainer::new(
            queue.clone(),
            embed_service,
            sink_for_drainer,
            EmbeddingDrainerConfig {
                batch_size: 8,
                poll_wait: Duration::from_millis(50),
                ..Default::default()
            },
        );
        let (handle, shutdown) = drainer.start(vec![0, 1]);

        // Wait up to 3s for the sink to record the insert.
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        loop {
            if !sink.calls.lock().await.is_empty() {
                break;
            }
            if std::time::Instant::now() > deadline {
                panic!("drainer never invoked sink");
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        let calls = sink.calls.lock().await;
        assert_eq!(calls.len(), 1, "exactly one sink call expected");
        let (target, tenant, recs) = &calls[0];
        assert_eq!(target, "knowledge");
        assert_eq!(tenant, "tenant-a");
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].oid, "doc-1");
        assert_eq!(recs[1].oid, "doc-2");
        assert!(!recs[0].vector.is_empty(), "vector must be populated");
        assert_eq!(
            recs[0].vector.len() as u32,
            recs[0].vector_dim,
            "vector_dim matches actual length"
        );
        assert_ne!(
            recs[0].vector, recs[1].vector,
            "distinct texts should produce distinct vectors"
        );
        drop(calls);

        let _ = shutdown.send(());
        handle
            .await
            .expect("drainer task join")
            .expect("clean drainer shutdown");
        queue.shutdown().await.expect("queue shutdown");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn drainer_skips_malformed_payload() {
        ensure_embedding_singleton();
        let tmp = TempDir::new().expect("tempdir");
        let queue = QueueClient::open(queue_cfg(tmp.path())).await.unwrap();
        let embed_service = proximadb_embedding::EmbeddingService::global();
        let sink = Arc::new(RecordingSink::default());

        let producer = queue.producer();
        producer
            .send(Message::new(
                EMBED_INGEST_TOPIC,
                "tenant-a",
                b"not-json".to_vec(),
            ))
            .await
            .expect("send malformed");

        let sink_for_drainer: Arc<dyn DrainerInsertSink> = sink.clone();
        let drainer = EmbeddingDrainer::new(
            queue.clone(),
            embed_service,
            sink_for_drainer,
            EmbeddingDrainerConfig {
                batch_size: 8,
                poll_wait: Duration::from_millis(50),
                ..Default::default()
            },
        );
        let (handle, shutdown) = drainer.start(vec![0, 1]);
        // Run for ~300ms — enough for several poll cycles.
        tokio::time::sleep(Duration::from_millis(300)).await;
        let _ = shutdown.send(());
        handle
            .await
            .expect("drainer task join")
            .expect("clean drainer shutdown");

        assert!(
            sink.calls.lock().await.is_empty(),
            "malformed payload must NOT invoke sink",
        );
        queue.shutdown().await.expect("shutdown");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn terminal_consumer_failure_is_surfaced_to_the_supervisor() {
        ensure_embedding_singleton();
        let tmp = TempDir::new().expect("tempdir");
        let queue = QueueClient::open(queue_cfg(tmp.path()))
            .await
            .expect("queue open");
        let sink: Arc<dyn DrainerInsertSink> = Arc::new(RecordingSink::default());
        let drainer = EmbeddingDrainer::new(
            queue.clone(),
            proximadb_embedding::EmbeddingService::global(),
            sink,
            EmbeddingDrainerConfig {
                poll_wait: Duration::from_millis(10),
                ..Default::default()
            },
        );
        let (handle, shutdown) = drainer.start(vec![0]);
        let lease_path = tmp
            .path()
            .join(EMBED_INGEST_TOPIC)
            .join("0/embed-drainer/lease.meta");
        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        while !lease_path.exists() && std::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let lease_bytes = std::fs::read(&lease_path).expect("drainer acquired lease");
        let mut lease: proximadb_queue::leases::LeaseMeta =
            serde_json::from_slice(&lease_bytes).expect("lease metadata");
        lease.expires_at_unix_nanos = 0;
        let replacement_path = lease_path.with_extension("replacement");
        std::fs::write(
            &replacement_path,
            serde_json::to_vec(&lease).expect("encode expired lease"),
        )
        .expect("write expired lease");
        std::fs::rename(replacement_path, &lease_path).expect("publish expired lease");

        let task_result = tokio::time::timeout(Duration::from_secs(1), handle)
            .await
            .expect("terminal consumer must not remain in a poll-error loop")
            .expect("drainer task join");
        let error = task_result.expect_err("terminal ownership failure must reach supervisor");
        assert!(
            error.to_string().contains("drainer poll failed"),
            "unexpected terminal error: {error:#}"
        );
        drop(shutdown);
        queue.shutdown().await.expect("queue shutdown");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sink_failure_is_surfaced_to_the_supervisor() {
        ensure_embedding_singleton();
        let tmp = TempDir::new().expect("tempdir");
        let queue = QueueClient::open(queue_cfg(tmp.path()))
            .await
            .expect("queue open");
        let embed_service = proximadb_embedding::EmbeddingService::global();
        let route = EmbedRoute::Byo {
            url: start_byo_test_endpoint(vec![vec![0.1, 0.2, 0.3]]),
            auth: ByoAuth::None,
            declared_dim: 3,
            declared_precision: proximadb_records::EmbeddingScalarType::Fp32,
            batch_size: 8,
            timeout_ms: 1_000,
        };
        embed_service.update_tenant_route("tenant-sink-failure", route.clone());
        let payload = EmbedIngestPayload {
            target_collection: "sink-failure-target".to_string(),
            tenant_id: "tenant-sink-failure".to_string(),
            embedding_route_identity: (&route).into(),
            expected_dimension: 3,
            records: vec![EmbedIngestRecord {
                oid: "doc-1".to_string(),
                text: "must remain recoverable".to_string(),
                metadata: HashMap::new(),
            }],
        };
        let receipt = queue
            .producer()
            .send(Message::new(
                EMBED_INGEST_TOPIC,
                "tenant-sink-failure",
                serde_json::to_vec(&payload).expect("serialize payload"),
            ))
            .await
            .expect("send");
        let drainer = EmbeddingDrainer::new(
            queue.clone(),
            embed_service,
            Arc::new(FailingSink),
            EmbeddingDrainerConfig {
                poll_wait: Duration::from_millis(10),
                ..Default::default()
            },
        );
        let (handle, shutdown) = drainer.start(vec![receipt.partition]);

        let task_result = tokio::time::timeout(Duration::from_secs(3), handle)
            .await
            .expect("sink failure must not leave a live consumer renewing its lease")
            .expect("drainer task join");
        let failure = task_result.expect_err("sink failure must reach supervisor");
        assert!(
            failure.to_string().contains("injected sink failure"),
            "unexpected terminal error: {failure:#}"
        );
        drop(shutdown);
        queue.shutdown().await.expect("queue shutdown");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ack_ownership_failure_preserves_queue_error() {
        ensure_embedding_singleton();
        let tmp = TempDir::new().expect("tempdir");
        let queue = QueueClient::open(queue_cfg(tmp.path()))
            .await
            .expect("queue open");
        let receipt = queue
            .producer()
            .send(Message::new(
                EMBED_INGEST_TOPIC,
                "tenant-a",
                b"not-json".to_vec(),
            ))
            .await
            .expect("send malformed payload");
        let consumer = queue.consumer("embed-drainer");
        consumer
            .subscribe(EMBED_INGEST_TOPIC, &[receipt.partition])
            .await
            .expect("subscribe");
        let deliveries = consumer
            .poll(1, Duration::ZERO)
            .await
            .expect("poll malformed payload");

        let lease_path = tmp
            .path()
            .join(EMBED_INGEST_TOPIC)
            .join(receipt.partition.to_string())
            .join("embed-drainer/lease.meta");
        let lease_bytes = std::fs::read(&lease_path).expect("consumer acquired lease");
        let mut lease: proximadb_queue::leases::LeaseMeta =
            serde_json::from_slice(&lease_bytes).expect("lease metadata");
        lease.expires_at_unix_nanos = 0;
        let replacement_path = lease_path.with_extension("replacement");
        std::fs::write(
            &replacement_path,
            serde_json::to_vec(&lease).expect("encode expired lease"),
        )
        .expect("write expired lease");
        std::fs::rename(replacement_path, &lease_path).expect("publish expired lease");

        let drainer = EmbeddingDrainer::new(
            queue.clone(),
            proximadb_embedding::EmbeddingService::global(),
            Arc::new(RecordingSink::default()),
            EmbeddingDrainerConfig::default(),
        );
        let error = drainer
            .process_batch(&consumer, deliveries)
            .await
            .expect_err("ACK without ownership must fail");
        assert!(
            error
                .downcast_ref::<proximadb_queue::QueueError>()
                .is_some(),
            "queue error type must survive anyhow propagation: {error:#}"
        );
        consumer.shutdown().await.expect("consumer shutdown");
        queue.shutdown().await.expect("queue shutdown");
    }

    #[test]
    fn split_input_tokens_attributes_by_record_share() {
        // 300 real batch tokens across 30 records → a 10-record payload gets 100.
        assert_eq!(split_input_tokens(300, 10, 30), 100);
        // Single-tenant batch keeps the whole count.
        assert_eq!(split_input_tokens(1234, 5, 5), 1234);
        // Integer division floors (no panic, no over-attribution).
        assert_eq!(split_input_tokens(100, 1, 3), 33);
        // A tiny measured batch floors a payload's share to 0 — the metering call site clamps
        // that to a minimum of 1 (measured_share) so a measured batch never certifies a zero.
        assert_eq!(split_input_tokens(2, 1, 3), 0);
        // Empty batch → 0, never a div-by-zero.
        assert_eq!(split_input_tokens(100, 0, 0), 0);
        // Large counts do not overflow (u128 intermediate).
        assert_eq!(split_input_tokens(u64::MAX, 1, 1), u64::MAX);
    }

    /// TD-SANDHI-3: a measured share never certifies a zero — floor-only could yield 0 for a
    /// tiny measured batch (2 tokens / 3 payloads), indistinguishable on the wire from the
    /// gateway-bug zeros the parser guards refuse. Clamp to a minimum of 1; over-count is
    /// bounded by one token per floored payload.
    #[test]
    fn measured_share_never_certifies_a_zero() {
        assert_eq!(measured_share(2, 1, 3), 1);
        assert_eq!(measured_share(1, 5, 10), 1);
        assert_eq!(measured_share(10, 1, 12), 1);
        assert_eq!(measured_share(100, 1, 3), 33);
        assert_eq!(measured_share(300, 10, 30), 100);
        assert_eq!(measured_share(1234, 5, 5), 1234);
        assert_eq!(measured_share(u64::MAX, 1, 1), u64::MAX);
    }

    /// TD-SANDHI-3: the batch duration belongs to exactly one carrier event per route group.
    #[test]
    fn carrier_duration_is_first_payload_only() {
        assert_eq!(carrier_duration(0, 4321), Some(4321));
        assert_eq!(carrier_duration(1, 4321), None);
        assert_eq!(carrier_duration(31, 4321), None);
    }

    /// TD-SANDHI-3: the external embedding event carries measurement provenance (basis +
    /// `Final` + `success` outcome, duration only on the designated carrier event of a route
    /// group) and the queue-message `idempotency_key` that makes at-least-once redelivery
    /// safe to consume.
    #[test]
    fn external_embedding_event_stamps_provenance_and_idempotency() {
        let route = EmbedRoute::Byo {
            url: "https://byo.example/v1/embed".to_string(),
            auth: ByoAuth::None,
            declared_dim: 3,
            declared_precision: proximadb_records::EmbeddingScalarType::Fp32,
            batch_size: 8,
            timeout_ms: 1_000,
        };
        let (provider, model, external) = route_labels(&route);
        assert_eq!(provider, "byo");
        assert_eq!(model, "https://byo.example/v1/embed");
        assert!(external);
        // Local BGE is the non-egress counter-case: no event, and victor-labeled KEU.
        let (bge_provider, _, bge_external) = route_labels(&EmbedRoute::BgeSmall);
        assert_eq!(bge_provider, "victor");
        assert!(!bge_external);

        let payload = EmbedIngestPayload {
            target_collection: "knowledge".to_string(),
            tenant_id: "tenant-a".to_string(),
            embedding_route_identity: (&route).into(),
            expected_dimension: 3,
            records: vec![],
        };

        let carrier = build_external_embedding_event(
            &payload,
            provider,
            &model,
            1024,
            crate::metrics::usage_event::UsageBasis::Estimated,
            Some(4321),
            "7:0:3",
        );
        assert_eq!(carrier.idempotency_key.as_deref(), Some("7:0:3"));
        assert_eq!(carrier.duration_ms, Some(4321));
        assert_eq!(
            carrier.usage_basis,
            Some(crate::metrics::usage_event::UsageBasis::Estimated)
        );
        assert_eq!(
            carrier.usage_completeness,
            Some(crate::metrics::usage_event::UsageCompleteness::Final)
        );
        assert_eq!(carrier.outcome.as_deref(), Some("success"));
        assert_eq!(carrier.tokens_in, 1024);

        // A non-carrier batch-mate: same provenance, no duration, its own key.
        let mate = build_external_embedding_event(
            &payload,
            provider,
            &model,
            1024,
            crate::metrics::usage_event::UsageBasis::ProviderReported,
            None,
            "7:0:4",
        );
        assert_eq!(mate.duration_ms, None);
        assert_eq!(mate.idempotency_key.as_deref(), Some("7:0:4"));
        assert_eq!(
            mate.usage_basis,
            Some(crate::metrics::usage_event::UsageBasis::ProviderReported)
        );
    }

    #[test]
    fn payload_requires_canonical_admission_facts() {
        let legacy = serde_json::json!({
            "target_collection": "docs",
            "tenant_id": "tenant-a",
            "records": [{"oid": "doc-1", "text": "body"}],
        });

        assert!(
            serde_json::from_value::<EmbedIngestPayload>(legacy).is_err(),
            "route and admitted dimension are required queue contract fields"
        );
    }

    /// One poll may contain unrelated tenants and model geometries. Each
    /// payload must execute only when its credential-free route identity still
    /// matches, while using freshly resolved credentials and committing every
    /// real delivery id.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn drainer_groups_by_admitted_route_and_commits_delivery_ids() {
        ensure_embedding_singleton();
        let tmp = TempDir::new().expect("tempdir");
        let queue = QueueClient::open(queue_cfg(tmp.path()))
            .await
            .expect("queue open");
        let embed_service = proximadb_embedding::EmbeddingService::global();
        let route_a = EmbedRoute::Byo {
            url: start_byo_test_endpoint(vec![vec![0.1, 0.2]]),
            auth: ByoAuth::Bearer {
                secret_ref: "admission-secret-a".to_string(),
            },
            declared_dim: 2,
            declared_precision: proximadb_records::EmbeddingScalarType::Fp32,
            batch_size: 8,
            timeout_ms: 1_000,
        };
        let route_b = EmbedRoute::Byo {
            url: start_byo_test_endpoint(vec![vec![0.3, 0.4, 0.5]]),
            auth: ByoAuth::Bearer {
                secret_ref: "admission-secret-b".to_string(),
            },
            declared_dim: 3,
            declared_precision: proximadb_records::EmbeddingScalarType::Fp32,
            batch_size: 8,
            timeout_ms: 1_000,
        };
        let producer = queue.producer();
        let mut partitions = Vec::new();
        for (tenant, collection, route, dimension) in [
            ("tenant-route-a", "docs-a", &route_a, 2),
            ("tenant-route-b", "docs-b", &route_b, 3),
        ] {
            let payload = EmbedIngestPayload {
                target_collection: collection.to_string(),
                tenant_id: tenant.to_string(),
                embedding_route_identity: route.into(),
                expected_dimension: dimension,
                records: vec![EmbedIngestRecord {
                    oid: format!("{tenant}-doc"),
                    text: format!("text for {tenant}"),
                    metadata: HashMap::new(),
                }],
            };
            let payload_bytes = serde_json::to_vec(&payload).unwrap();
            assert!(
                !String::from_utf8_lossy(&payload_bytes).contains("admission-secret"),
                "durable queue payload must not contain BYO credentials"
            );
            let receipt = producer
                .send(Message::new(EMBED_INGEST_TOPIC, tenant, payload_bytes))
                .await
                .expect("send");
            partitions.push(receipt.partition);
        }

        embed_service.update_tenant_route(
            "tenant-route-a",
            with_byo_auth(
                &route_a,
                ByoAuth::Bearer {
                    secret_ref: "rotated-secret-a".to_string(),
                },
            ),
        );
        embed_service.update_tenant_route(
            "tenant-route-b",
            with_byo_auth(
                &route_b,
                ByoAuth::Bearer {
                    secret_ref: "rotated-secret-b".to_string(),
                },
            ),
        );

        let sink = Arc::new(RecordingSink::default());
        let sink_for_drainer: Arc<dyn DrainerInsertSink> = sink.clone();
        let drainer = EmbeddingDrainer::new(
            queue.clone(),
            embed_service,
            sink_for_drainer,
            EmbeddingDrainerConfig {
                batch_size: 8,
                poll_wait: Duration::from_millis(50),
                ..Default::default()
            },
        );
        let (handle, shutdown) = drainer.start(vec![0, 1]);

        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        loop {
            let inserts_complete = sink.calls.lock().await.len() == 2;
            let commits_complete = partitions.iter().all(|partition| {
                tmp.path()
                    .join(EMBED_INGEST_TOPIC)
                    .join(partition.to_string())
                    // Q1 (ADR-079): committed offsets are per consumer group —
                    // {partition}/{group}/offset.meta. The drainer's default
                    // group is "embed-drainer" (EmbeddingDrainerConfig::default).
                    .join("embed-drainer")
                    .join("offset.meta")
                    .exists()
            });
            if inserts_complete && commits_complete {
                break;
            }
            if std::time::Instant::now() > deadline {
                panic!("drainer did not insert and commit all admitted jobs");
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        let calls = sink.calls.lock().await;
        let mut dimensions: HashMap<&str, u32> = HashMap::new();
        for (_, tenant, records) in calls.iter() {
            dimensions.insert(tenant.as_str(), records[0].vector_dim);
        }
        assert_eq!(dimensions.get("tenant-route-a"), Some(&2));
        assert_eq!(dimensions.get("tenant-route-b"), Some(&3));
        drop(calls);

        let _ = shutdown.send(());
        handle
            .await
            .expect("drainer task join")
            .expect("clean drainer shutdown");
        queue.shutdown().await.expect("queue shutdown");
    }

    /// With a resolver attached and a fp16 collection in the catalog,
    /// the drainer must stamp `EmbeddedRecord.target_precision = Some(Fp16)`
    /// on every record routed to that collection. The sink (in
    /// production: BulkLoadDrainerSink) then coerces the fp32 vector
    /// to fp16 before constructing the ProximaRecord. End-to-end ingest
    /// for an fp16 collection now produces fp16 records.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn drainer_stamps_target_precision_from_resolver() {
        use proximadb_catalog::cache::CatalogCache;
        use proximadb_catalog::canonical_precision::CanonicalPrecisionResolver;
        use proximadb_catalog::{Catalog, CatalogNamespace, CatalogTableSchema, TableIdentifier};
        use std::collections::HashMap;
        use tokio::sync::RwLock;

        ensure_embedding_singleton();
        let tmp = TempDir::new().expect("tempdir");
        let queue = QueueClient::open(queue_cfg(tmp.path()))
            .await
            .expect("queue open");
        let embed_service = proximadb_embedding::EmbeddingService::global();
        let route = EmbedRoute::Byo {
            url: start_byo_test_endpoint(vec![vec![0.1, 0.2, 0.3]]),
            auth: ByoAuth::None,
            declared_dim: 3,
            declared_precision: proximadb_records::EmbeddingScalarType::Fp32,
            batch_size: 8,
            timeout_ms: 1_000,
        };
        embed_service.update_tenant_route("tenant-fp16", route.clone());

        // Stand up an in-memory catalog with one fp16 collection.
        let cache = Arc::new(CatalogCache::new(1000, 60));
        let cat: Arc<dyn Catalog> = Arc::new(InMemoryTestCatalog::new("drainer-test".to_string()));
        cat.create_namespace(&["default".to_string()], HashMap::new())
            .await
            .unwrap();
        let table_id = TableIdentifier::new(vec!["default".to_string()], "fp16_docs");
        let mut schema = CatalogTableSchema {
            name: "fp16_docs".to_string(),
            ..Default::default()
        };
        schema.canonical_embedding_precision = proximadb_records::EmbeddingScalarType::Fp16;
        cat.create_table(&table_id, schema).await.unwrap();

        let resolver = Arc::new(CanonicalPrecisionResolver::new(
            cat.clone() as Arc<dyn Catalog>,
            cache,
        ));

        let sink = Arc::new(RecordingSink::default());
        let producer = queue.producer();
        let payload = EmbedIngestPayload {
            target_collection: "fp16_docs".to_string(), // unqualified → default namespace
            tenant_id: "tenant-fp16".to_string(),
            embedding_route_identity: (&route).into(),
            expected_dimension: 3,
            records: vec![EmbedIngestRecord {
                oid: "doc-1".to_string(),
                text: "fp16 collection ingest".to_string(),
                metadata: HashMap::new(),
            }],
        };
        producer
            .send(Message::new(
                EMBED_INGEST_TOPIC,
                "tenant-fp16",
                serde_json::to_vec(&payload).unwrap(),
            ))
            .await
            .expect("send");

        let sink_for_drainer: Arc<dyn DrainerInsertSink> = sink.clone();
        let drainer = EmbeddingDrainer::new(
            queue.clone(),
            embed_service,
            sink_for_drainer,
            EmbeddingDrainerConfig {
                batch_size: 8,
                poll_wait: Duration::from_millis(50),
                ..Default::default()
            },
        )
        .with_precision_resolver(resolver);
        let (handle, shutdown) = drainer.start(vec![0, 1]);

        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        loop {
            if !sink.calls.lock().await.is_empty() {
                break;
            }
            if std::time::Instant::now() > deadline {
                panic!("drainer never invoked sink");
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        let calls = sink.calls.lock().await;
        assert_eq!(calls.len(), 1);
        let (_target, _tenant, recs) = &calls[0];
        assert_eq!(recs.len(), 1);
        assert_eq!(
            recs[0].target_precision,
            Some(proximadb_records::EmbeddingScalarType::Fp16),
            "resolver must stamp target_precision=Fp16 from the catalog's \
             canonical_embedding_precision for this collection"
        );
        drop(calls);

        let _ = shutdown.send(());
        handle
            .await
            .expect("drainer task join")
            .expect("clean drainer shutdown");
        queue.shutdown().await.expect("queue shutdown");
    }
}
