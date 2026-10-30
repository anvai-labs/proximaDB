//! # ProximaDB Observability Engine
//!
//! Runtime implementation of ProximaDB's observability surface — the per-query
//! I/O trace substrate (co-design C0), metering-event builders, the trace
//! batcher/digest/fingerprint/sampling/retention pipeline, route-explain,
//! predicate-diagnostics, embedding/rank precision metrics, and the alerting +
//! audit sub-systems.
//!
//! ## Extraction status (root-crate decomposition)
//!
//! Slices 1–2 extracted the foundation-pure leaves (alerting/audit/trace,
//! promql/tantivy/metrics, buffer/parser). **Slice 3** (this revision) landed
//! the *facade* behind the [`ObservabilityStoragePort`] (dependency inversion):
//! `service` (`ObservabilityService` + param/result structs) and the
//! `query`/`ingestion` facades (`ObservabilityQueryEngine`/`ObservabilityIngester`)
//! now live here and hold `Arc<dyn ObservabilityStoragePort>`.
//! The root composition root injects the concrete WAL-backed
//! `ObservabilityStorage` (which impls the port). The format *adapters*
//! (OTLP/Syslog/CEF/… ingress servers) and `storage/` (WAL up-edges) remain in
//! the root `src/observability/`, as does
//! the `impl ObservabilityStorageOperations` (root-local trait).
//!
//! **Slice 2a** extracted `rank_metrics` (rank-pipeline Prometheus metrics —
//! `RankPipelineMetrics`/`PrometheusRankSink`/`ModelCacheMetricsObserver`, deps
//! only `prometheus` + fully-qualified `proximadb_rank_*` trait impls); the root
//! re-exports it so `/metrics/prometheus` and its callers resolve unchanged.
//!
//! The root `crate::observability` module re-exports this crate, so the
//! ~80 inbound callers (`crate::*`) are unchanged.
//!
//! ## Foundation
//!
//! The modules here depend only on foundation types (`proximadb-data-model`,
//! `proximadb-kernel`, `proximadb-records`, `proximadb-tenant`) plus horizontal
//! crates (serde, tokio, prometheus, opentelemetry). No upward edge into the
//! root.

/// Proto re-export so moved modules' `crate::proto::proximadb_v1::*` paths
/// resolve (mirrors the root `src/proto/mod.rs`). Add more re-exports as the
/// extracted surface grows.
pub mod proto {
    pub use proximadb_proto::proximadb_v1;
}

/// Plain-data types shared across the observability storage seam
/// (`TraceSpan`, `TraceSummary`, `ObservabilityStorageStats`).
pub mod model;
/// Dependency-inversion port for the observability storage layer
/// (`ObservabilityStoragePort`) — dissolves the facade→storage up-edge.
pub mod ports;
/// Top-level observability service facade (`ObservabilityService`) + the
/// ingestion/query param/result structs. Holds `Arc<dyn ObservabilityStoragePort>`;
/// the root injects the concrete WAL-backed `ObservabilityStorage`.
pub mod service;

pub use model::{ObservabilityStorageStats, TraceSpan, TraceSummary};
pub use ports::ObservabilityStoragePort;

/// Alerting engine — rule-based evaluation, escalation, persistence, notifications.
pub mod alerting;
/// Audit logging for compliance and security event tracking.
pub mod audit;
/// High-throughput ingestion pipeline — foundation-pure leaves (`buffer`,
/// `parser`). The facade + format adapters remain in the root.
pub mod ingestion;
/// Per-query I/O trace bus (co-design C0 — physical-dimension trace substrate).
pub mod io_trace;
/// Metering event builder — SearchPlanTrace → operator metering-event JSON.
pub mod metering_event;
/// TD-161 external OTLP metering push (ADR-027 dual-sink push half). Feature
/// `otlp-metering` + runtime-gated by `PROXIMADB_OTLP_ENDPOINT`; no-ops otherwise.
pub mod metering_otlp;
/// `ObjectStore` decorator feeding the per-query io_trace (ADR-030/TD-158).
pub mod object_store_trace;
/// Embedding-precision metrics — Prometheus gauges/counters.
pub mod precision_metrics;
/// TD-064 predicate diagnostics bus — recall-shortfall events from deep search.
pub mod predicate_diagnostics;
/// Query engine for logs, metrics, and traces — foundation-pure leaves
/// (`promql`, `tantivy_log_index`, `metrics`). The facade + logs + traces
/// remain in the root.
pub mod query;
/// Rank-pipeline metrics — Prometheus histograms/counters per RANKING_FRAMEWORK_SPEC
/// NFR-8 (R-7c.4d follow-up). Extracted from the root (Slice 2a); deps only
/// `prometheus` + the `proximadb_rank_*` trait impls.
pub mod rank_metrics;
/// Route explain builder — human-readable explanation from a SearchPlanTrace.
pub mod route_explain;
/// SearchPlanTrace — per-query telemetry envelope (KRU billing + learned planner).
pub mod search_plan_trace;
/// Post-execution SearchPlanTrace builder.
pub mod search_plan_trace_builder;
pub mod search_rank_metrics;
/// Tenant Prometheus label resolver — bounded cardinality-safe label resolution.
pub mod tenant_label;
/// Trace batcher — bundles N traces into one POST for the async billing sink.
pub mod trace_batcher;
/// Trace digest — stable FNV-1a hash for billing dedup / idempotency keys.
pub mod trace_digest;
/// TD-TRACE-2 S3 — the durable io_trace record envelope (ADR-066 D1): a
/// homogeneous header + modality-tagged payload, built from an `IoTraceSnapshot`.
pub mod trace_envelope;
/// Trace fingerprint — shape-only hash for incident-triage grouping.
pub mod trace_fingerprint;
/// Trace retention policy — companion to trace_sampling; per-tier age windows.
pub mod trace_retention;
/// Trace sampling policy — LLD-anchored down-sampling by tier + load.
pub mod trace_sampling;
/// Workload mix detector — fingerprint counts → typed summary for tier hints.
pub mod workload_mix;
