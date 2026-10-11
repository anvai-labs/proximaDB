/*
 * Copyright 2026 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! Per-query I/O trace bus (C0 — co-design trace substrate)
//!
//! This is the foundational deliverable of the co-design mandate
//! (`docs/12-design/CODESIGN_DIMENSIONAL_ARCHITECTURE_2026_06_19.adoc`, §4.1):
//! *you cannot co-design against a trace distribution you do not capture.*
//!
//! The existing `consumption_metrics` counters meter four per-tenant
//! **aggregates** (object-store ops, storage byte-seconds, task time, cache
//! stats) — excellent for billing, but they cannot answer the questions
//! co-design actually asks: *for this one query, how many GETs did we pay, how
//! many bytes did we move, did the footer cache hit, and which engine burned
//! the compute?* Those are the quantities the `ComputeScheduler` cost model
//! (§3 of the spec) must minimize, and they are per-query, not per-tenant.
//!
//! `IoTrace` captures them. It is bound to the request future as a
//! [`tokio::task_local!`] — exactly like
//! [`crate::predicate_diagnostics`] — so any depth of the
//! storage/engine call stack can record into it without threading a new
//! parameter through dozens of signatures (the dominant cost of doing this any
//! other way). At the request boundary the handler wraps the query in
//! [`instrument`] (or [`scope`]); downstream I/O sites call the free
//! [`record_op`] / [`record_bytes_read`] / [`record_footer`] helpers, which
//! **silently no-op outside an active scope** (so direct service/test callers
//! keep working and the Prometheus counters remain the operator-visible signal).
//!
//! On completion the snapshot is emitted as a structured `tracing` event under
//! the [`TARGET`] target. Once OpenTelemetry export is wired (§4.4) these
//! events become spans on the trace backend; today they are already
//! grep-able structured logs. This module adds **no new billing authority** —
//! the per-tenant counters stay the source of truth for chargeback; this is the
//! finer, per-query *source* the spec calls for.
//!
//! ## OpenTelemetry export (§4.4 — dependency-gated follow-up)
//!
//! Today the snapshot is emitted as a structured [`tracing`] event under
//! [`TARGET`] and is routable by any subscriber layer. Full export to an OTLP
//! collector (so these become persisted, queryable spans on a trace backend) is
//! a deliberate **dependency decision** not yet taken: the tree has
//! `tracing-subscriber` + `tracing-appender` but none of `opentelemetry`,
//! `opentelemetry_sdk`, `opentelemetry-otlp`, or `tracing-opentelemetry`. When
//! that stack is approved the wiring is small and already enabled here:
//!
//! 1. [`IoTraceSnapshot`] is `serde`-serializable, so it maps directly to OTLP
//!    span attributes (or a JSON sink) with no further plumbing.
//! 2. Attach a `tracing_opentelemetry::layer()` to the subscriber registry in
//!    `src/bin/server.rs` (beside the existing console/file `fmt` layers, at the
//!    `tracing_subscriber::registry()` call), gated on `OTEL_EXPORTER_OTLP_ENDPOINT`.
//! 3. Point the existing `crate::monitoring::opentelemetry` config's
//!    `otlp_endpoint` at the collector.
//!
//! Until then, persistence is available with zero new deps by routing [`TARGET`]
//! to a `tracing-appender` JSON file sink.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// `tracing` target for emitted per-query I/O trace events. Kept distinct so
/// operators can route/sample it independently (and the future OTLP layer can
/// map it to a span).
pub const TARGET: &str = "proximadb::io_trace";

tokio::task_local! {
    /// Active per-query I/O trace for the current task. Bound by [`scope`] /
    /// [`instrument`].
    ///
    /// Held behind an `Arc` so a handle can be cloned out via
    /// [`current_handle`] and captured by components (e.g. `TracingObjectStore`,
    /// `ProximaScanExec`) whose I/O later runs on DataFusion-**spawned** tokio
    /// tasks — a `task_local!` does not cross `tokio::spawn`, so those tasks
    /// must record through a captured handle rather than this task-local
    /// (TD-OLAP-3). All counters are atomic, so concurrent spawned readers
    /// aggregate into the one shared trace correctly.
    static IO_TRACE: Arc<IoTrace>;
}

/// Classification of an object-store operation by its cost shape. The universe
/// prices GET, PUT, LIST and DELETE differently (LIST and GET dominate
/// scan-heavy ANN/OLAP; PUT dominates ingest), so the trace keeps them apart
/// rather than collapsing to a single "ops" count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoOp {
    Get,
    Put,
    List,
    Delete,
}

impl IoOp {
    /// Best-effort classification of the operation strings already passed to
    /// `consumption_metrics::record_object_store_op` (e.g. `"fetch_pax"`,
    /// `"list_parquet"`, `"write_parquet"`) so existing call sites can feed the
    /// trace with a one-line addition. Unknown verbs map to [`IoOp::Get`] (the
    /// conservative read default) — the verb is still preserved by the caller's
    /// Prometheus label.
    pub fn classify(operation: &str) -> Self {
        if operation.starts_with("list") {
            IoOp::List
        } else if operation.starts_with("write")
            || operation.starts_with("put")
            || operation.starts_with("delete")
        {
            if operation.starts_with("delete") {
                IoOp::Delete
            } else {
                IoOp::Put
            }
        } else {
            // read_parquet, fetch_pax, fetch_pax_ranged, get, ...
            IoOp::Get
        }
    }
}

/// Per-query accumulator for the physical-dimension quantities the co-design
/// cost model consumes. Held inside a [`tokio::task_local!`]; all counters are
/// atomic so concurrent segment reads within one query aggregate correctly.
#[derive(Debug, Default)]
pub struct IoTrace {
    get_ops: AtomicU64,
    put_ops: AtomicU64,
    list_ops: AtomicU64,
    delete_ops: AtomicU64,
    /// Bytes fetched from object storage (Dimension 1 — the read term of KRU).
    bytes_read: AtomicU64,
    /// Number of ranged GET requests issued. With `bytes_read` yields the
    /// average GET size — the read-granularity signal (§2.1): coalesced toward
    /// the ~8-16 MiB S3 optimum, or fragmented into per-request-fee-dominated
    /// small GETs?
    range_gets: AtomicU64,
    /// Split-pruning outcome (TD-OLAP-3): candidate row-group splits
    /// considered by scans, and how many were skipped BEFORE fetch.
    /// `splits_pruned / splits_total` is the runtime-filter skip ratio the
    /// promotion gate consumes.
    splits_total: AtomicU64,
    splits_pruned: AtomicU64,
    /// PAX cascade centroid block-prune outcome (TD-RDSTRAT-5 S3): `centroid_pruned_blocks`
    /// of `centroid_total_blocks` skipped by the VOE-directory centroid probe before
    /// scanning. Distinct from `splits_*` (row-group runtime filter) so the ratios
    /// don't cross-contaminate; the recall gate asserts this engaged
    /// (`centroid_pruned_blocks > 0`) rather than silently falling back to a full scan.
    centroid_total_blocks: AtomicU64,
    centroid_pruned_blocks: AtomicU64,
    /// TD-RDSTRAT-8 two-level IVF coarse-probe outcome (durable — lands in the
    /// warehouse `VectorAnnPayload` satellite via `TracePayload::classify`). The
    /// coarse probe ranks `k_c` centroids in RAM and ranged-reads only `nprobe`
    /// cells, cutting the dominant per-query cost term (GET round-trips). These
    /// counters make that cut observable per-query so nprobe/spill tuning is
    /// evidence-led ("trace before you tune"), not asserted.
    /// `ivf_whole_region_fallback` counts segments where the probe was armed but
    /// missed and the read fell back to the whole Region-A scan.
    ivf_cells_total: AtomicU64,
    ivf_cells_probed: AtomicU64,
    ivf_probed_rows: AtomicU64,
    ivf_fetch_rounds: AtomicU64,
    ivf_whole_region_fallback: AtomicU64,
    /// Physical PAX Region-A (RaBitQ) and Region-B (SQ8) bytes fetched by the
    /// coarse probe (TD-RDSTRAT-8 PR-C1). Metadata/A0/footer bytes stay in the
    /// universal `bytes_read`; these are the vector-body bytes the probe paid
    /// for. Cache hits contribute zero.
    ivf_region_a_bytes: AtomicU64,
    ivf_region_b_bytes: AtomicU64,
    /// PAX Stage-F footer-stat fetch prunes (TD-FPRUNE-1 P2): how many D-blocks
    /// the segment footer's `FooterBlockStats` provably ruled out (dropped from
    /// the FETCH plan before any body GET) out of the blocks considered.
    /// Attribution counter — the physical savings already land in the universal
    /// `get_ops`/`bytes_read` as unfetched reads.
    pax_footer_total_blocks: AtomicU64,
    pax_footer_pruned_blocks: AtomicU64,
    /// General bounded-concurrent `read_ranges` metrics (TD-RDSTRAT-12).
    /// Populated by any `read_ranges` or `read_ranges_prefetch` call with
    /// `parallel > 1` / `inflight > 1`. Unlike IVF-specific fields, these
    /// cover ALL ranged-read paths: footer fetches, Region-A/B reads, and
    /// general multi-range reads (not just IVF coarse probe).
    read_ranges_fetch_rounds: AtomicU64,
    read_ranges_max_inflight: AtomicU64,
    /// TD-RDSTRAT-13 C1: per-tier physical-GET attribution for the SST PAX
    /// cascade. Index = the SST `CacheTier` declaration order (fixed contract):
    /// 0=InvariantIndex(IdxA), 1=SearchControl(Ctl), 2=InvariantMeta(Meta),
    /// 3=ProbeIndex(ProbeA), 4=SurvivorPayload(Surv), 5=ResultPayload(OID).
    /// Always-on (like the `ivf_*` fields); recorded 1:1 with physical ranged
    /// GETs at the read sites, so `Σ tier_get_ops == range_gets` for a query
    /// whose GETs all come from the PAX cascade. Makes the "unwaved metadata
    /// GETs" the S3 campaign decomposed observable per query.
    tier_get_ops: [AtomicU64; 6],
    tier_get_bytes: [AtomicU64; 6],
    /// Object-storage metadata ops (`fs.metadata` HEAD) on the search path
    /// (TD-RDSTRAT-13 C3 removes the per-segment HEAD; this counter is how the
    /// removal is proven — `metadata_ops == 0` on the known-size path).
    metadata_ops: AtomicU64,
    /// Coarse-probe rank wall microseconds including any morsel join
    /// (TD-RDSTRAT-13 C2) — the user-visible Compute term of the
    /// waves-not-sums model `T ≈ Σ rounds×RTT + bytes/BW + Compute`.
    ivf_probe_rank_us: AtomicU64,
    /// TD-RDSTRAT-13 PR-B tail attribution: the four uninstrumented segments
    /// of the cold cascade, measured so the ~300 ms disk-p50 tail is bounded
    /// by data before any optimization. All additive (multi-segment sums).
    /// Region-B survivor fetch + SQ8 rerank wall (both branches).
    region_b_rerank_us: AtomicU64,
    /// Region-D top-k OID ranged fetch + decode wall (v4-chunk + legacy arms).
    region_d_fetch_decode_us: AtomicU64,
    /// Top-k rehydration (`read_records_by_positions`) wall — the second
    /// Region-D pass (metadata + header/footer + block fetches + decode).
    rehydrate_us: AtomicU64,
    /// Whole-cascade umbrella wall (`try_pax_cascade` entry to exit) — the
    /// residual after the other timers names the unattributed remainder.
    cascade_total_us: AtomicU64,
    /// Runtime-filter wait outcomes (ADR-056 AQE-S11): how often the probe scan's
    /// `wait_complete()` rendezvous resolved with the filter arrived (pruning
    /// enabled) vs timed out (filterless, conservative), plus the wall ms spent
    /// waiting. The route cost model consumes the arrived/(arrived+timed_out)
    /// ratio to learn per-workload whether the wait budget pays.
    runtime_filter_arrived: AtomicU64,
    runtime_filter_timed_out: AtomicU64,
    runtime_filter_wait_ms: AtomicU64,
    /// Bytes written to object storage (ingest/flush — KIU).
    bytes_written: AtomicU64,
    /// Footer/metadata cache outcomes (Dimension 3 — the highest-ROI cache).
    footer_hits: AtomicU64,
    footer_misses: AtomicU64,
    /// In-process DRAM cache outcomes. These are logical request outcomes,
    /// recorded once per requested immutable range rather than once per
    /// implementation-key probe (exact key plus optional parent key).
    survivor_l1_hits: AtomicU64,
    survivor_l1_misses: AtomicU64,
    /// Persistent local-disk L2 cache probe outcomes (ADR-085 / TD-IOTRACE-4).
    /// Distinct from the in-memory footer cache: an L2 hit serves bytes that
    /// survived process restart/L1 eviction without a billed ranged GET, so a
    /// footer-cold query can still be physically warm — the cache-state signal
    /// downstream priors aggregation needs alongside the footer ratio.
    l2_hits: AtomicU64,
    l2_misses: AtomicU64,
    /// Chargeable egress bytes — moved cross-region / to the internet off the
    /// free same-region path (Dimension 2 — KEU). Recorded only for chargeable
    /// localities; the route cost model's egress weight consumes this.
    egress_bytes: AtomicU64,
    /// PAX RaBitQ cascade **logical** striped-read projection (ADR-057 / TD-RDSTRAT-3):
    /// the bytes + ranged GETs a *selective* striped read WOULD move (Stage-1 codes
    /// stripe + Stage-2 candidate rerank rows, with the real ranked candidate set),
    /// kept DISTINCT from the physical `bytes_read`/`range_gets` (which today reflect
    /// the whole-segment `fs.read`). The pair makes the striped-vs-whole headroom
    /// observable per query on real candidate scatter — the co-design "trace before
    /// you tune" signal the flip gates on. Recording here is projection-only; it
    /// moves no bytes and changes no read path.
    logical_striped_bytes: AtomicU64,
    logical_striped_gets: AtomicU64,
    /// Embedding API calls (Dimension 5 — KEU, Kilo-Embedding-Units). A neutral
    /// counter for code-embedding operations; pricing happens downstream in
    /// AnvaiOps. Distinct from general read KRU.
    embedding_calls: AtomicU64,
    /// Total input tokens consumed by embedding operations.
    embedding_input_tokens: AtomicU64,
    /// Total output tokens (or equivalently, vector count) generated by embedding operations.
    embedding_output_tokens: AtomicU64,
    /// Compute milliseconds attributed by engine (Dimension 4 — KRU/KIU). Kept
    /// in a small map so a single query that touches multiple engines (e.g. a
    /// Volcano point lookup plus a DataFusion aggregate) attributes each.
    compute_ms: Mutex<BTreeMap<String, u64>>,
    /// pgwire relational-pipeline SETUP wall milliseconds (TD-OLAP-4): per-query
    /// pre-execution cost in `try_run_select` BEFORE the engine runs — table-name
    /// collection + xCatalog schema pre-resolution + route classification. Paid
    /// only on the DataFusion (path-2) route, not the native early-return path.
    setup_ms: AtomicU64,
    /// pgwire result EMIT wall milliseconds (TD-OLAP-4): encoding the materialized
    /// rows to `RowDescription` + `DataRow` frames and writing them to the socket,
    /// AFTER execution — the last unmeasured span in the per-query wall floor.
    emit_ms: AtomicU64,
    /// SessionContext build wall milliseconds (TD-OLAP-4): per-query cost of
    /// creating a fresh DataFusion `SessionContext` and re-registering all
    /// UDFs/UDAFs — paid before table open, reusable across queries.
    session_ms: AtomicU64,
    /// Table-OPEN wall milliseconds (TD-OLAP-4): the per-query fixed cost of
    /// discovering + opening the parquet base (LIST + HEAD-per-file + footer
    /// read) BEFORE execution. Distinct from `compute_ms` (execution) — a
    /// footer-only query does ~0 compute yet pays this floor. Drops to ~0 on a
    /// warm table-open cache hit, so it is the direct signal for that lever.
    open_ms: AtomicU64,
    /// Query lowering + logical/physical planning wall milliseconds (TD-OLAP-4),
    /// the other half of the per-query floor separate from execution.
    plan_ms: AtomicU64,
    /// Table-OPEN cache outcomes (TD-OLAP-4): a hit skips the LIST+HEAD+footer
    /// discovery and reuses cached schema/splits/file-sizes.
    table_open_hits: AtomicU64,
    table_open_misses: AtomicU64,
    /// Plan-geometry vector (TD-EXEC-2 Slice 1, observe-only): the pre-execution
    /// geometric summary of the served physical plan — depth, node/leaf counts,
    /// fan-out, blocking-operator count — recorded at the plan→execute seam as
    /// neutral scalars (io_trace never depends on a query-layer type). One
    /// measured vector feeds three resource laws: stack sizing, engine routing,
    /// parallelism. Max semantics so a multi-statement scope keeps its deepest plan.
    plan_depth: AtomicU64,
    plan_nodes: AtomicU64,
    plan_leaves: AtomicU64,
    plan_fanout: AtomicU64,
    plan_blocking: AtomicU64,
    /// Per-operator-kind counts of the served plan (TD-EXEC-2) — the histogram
    /// half of the geometry vector, keyed by neutral op-kind labels.
    plan_ops: Mutex<BTreeMap<String, u64>>,
    /// Measured stack high-water mark (bytes) of the plan-tree recursions
    /// (planner lowering + executor build), sampled via the planner's
    /// `stack_probe` (TD-EXEC-2 Slice 1). Max across recordings — the binding
    /// per-query figure that calibrates `frame_bytes[op_kind]`.
    stack_hwm_bytes: AtomicU64,
    /// The route this query was served on, as `(shape_class, backend_label)`
    /// strings (co-design C4). Stamped by the `ComputeScheduler` so the flush can
    /// feed the trace-driven cost model the cost of *this kind of query on that
    /// engine* — without io_trace depending on any query-layer type. `None` until
    /// a route is chosen.
    route: Mutex<Option<(String, String)>>,
    /// Physical vector access methods completed inside this query. Unlike the
    /// containing compute route, a SQL query may contain multiple vector
    /// sources, so this is an ordered list rather than a last-write-wins slot.
    /// Samples are bounded primitives/labels only — never vectors, predicates,
    /// collection names, or user data.
    vector_accesses: Mutex<Vec<VectorAccessTrace>>,
    /// Query-local proof counter used by a physical engine to distinguish an
    /// ANN path that actually engaged from a non-exact request that fell back
    /// to exact execution. Internal coordination only; the durable fact is the
    /// resulting `VectorAccessTrace`, not this counter.
    vector_ann_proofs: AtomicU64,
    /// The resolved per-collection storage profile (`append_bulk`/`churn`) this
    /// query read under (ADR-061 D6 / TD-WLP-6), stamped at the search boundary
    /// so a query's projection strategy is observable per-tenant. `None` until
    /// stamped; io_trace never depends on the `StorageProfile` type (neutral
    /// label string only).
    storage_profile: Mutex<Option<String>>,
    /// Per-operator execution vector (TD-TRACE-1 Slice 2, observe-only): the
    /// metered actuals of the served physical plan — one entry per operator in
    /// pre-order — `{op, rows_in, rows_out, ms_self, bytes, spill}`. Populated ONLY
    /// on the metered path (`EXPLAIN ANALYZE`), so a normal SELECT never allocates
    /// here; the hot path stays lean by construction, not by feature gate. Recorded
    /// as neutral primitives (io_trace never depends on a query-layer type, exactly
    /// like the geometry vector). Feeds the structured EXPLAIN surface, and is
    /// available on the snapshot for the cost model / billing (per-op detail; the
    /// KRU meter itself still sums `compute_ms`, so this never double-charges).
    exec_ops: Mutex<Vec<ExecOpTrace>>,
    /// Stable per-query id (UUID v4), minted at `instrument()` entry (TD-TRACE-2 /
    /// ADR-066). Identifies this query's record in the durable trace sink and is the
    /// join key for the future warehouse header↔satellite tables. `None` until
    /// stamped (e.g. a raw `IoTrace::new()` outside `instrument`).
    query_id: Mutex<Option<String>>,
    /// TD-CACHE-3 S1: the requesting tenant, stamped at `instrument()` scope
    /// entry. Ambient carrier for engine-side per-tenant consumers (survivor
    /// cache fair-share keys, billing labels) — the same task-local scope that
    /// already wraps every instrumented request, so no per-call threading.
    /// Absent outside an instrumented scope (or across un-propagated spawns —
    /// the same constraint all io_trace metering already has).
    tenant_id: Mutex<Option<String>>,
    /// Catalog-authoritative stable tenant identity for in-process ownership,
    /// cache fair-sharing, and authorization joins. Unlike `tenant_id`, this
    /// stays numeric and is never derived from an alias.
    tenant_stable_id: Mutex<Option<u64>>,
}

/// Neutral primitive tuple carrying one operator's metered actuals into
/// [`record_exec_vector`] — `(op, rows_in, rows_out, ms_self, bytes, spill)`. A
/// type alias so the recording API stays a neutral tuple (no cross-crate type)
/// without tripping `clippy::type_complexity`.
pub type ExecOpSample<'a> = (&'a str, u64, u64, u64, Option<u64>, bool);

/// One physical-plan operator's metered execution actuals (TD-TRACE-1 Slice 2).
/// A neutral, self-contained snapshot type owned by io_trace (NOT the executor's
/// `NodeMetric`) so the modality layer never depends on the query layer. Integer /
/// `Option` fields only, to preserve `IoTraceSnapshot`'s `Eq`.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExecOpTrace {
    /// Operator keyword (pre-order, aligned with `EXPLAIN`'s plan lines).
    pub op: String,
    /// Rows fed into this operator = sum of its direct children's `rows_out`.
    pub rows_in: u64,
    /// Rows this operator emitted.
    pub rows_out: u64,
    /// Exclusive (self) wall-clock milliseconds — inclusive minus children.
    pub ms_self: u64,
    /// Bytes processed, when the engine tracks it. `None` for the row-oriented
    /// native Volcano executor; the DataFusion adapter (Slice 3) fills it.
    pub bytes: Option<u64>,
    /// Whether this operator spilled to disk. Always `false` for native (no
    /// spilling); the DataFusion adapter (Slice 3) sets it.
    pub spill: bool,
}

/// Caller intent presented to a physical vector engine. This is deliberately
/// independent of the query-layer `SearchMode` type so the observability layer
/// remains dependency-inverted and the durable labels stay bounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorSearchIntent {
    Exact,
    Approximate,
    Adaptive,
}

impl VectorSearchIntent {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Approximate => "approximate",
            Self::Adaptive => "adaptive",
        }
    }
}

/// Physical access method that actually served a successful vector operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorAccessPath {
    Exact,
    Ann,
}

impl VectorAccessPath {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Ann => "ann",
        }
    }
}

/// Coarse physical locality of the storage that served a vector access.
///
/// This intentionally describes only what the read path can prove from its
/// storage URL. It does not claim an object access tier (`hot`/`cool`/`cold`):
/// that write-side property is not currently available at vector-read time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorStorageScope {
    /// A local path or `file://` filesystem.
    Local,
    /// A network/object filesystem supported by the canonical filesystem port.
    Remote,
    /// The scheme is absent from the current filesystem vocabulary.
    #[default]
    Unknown,
}

impl VectorStorageScope {
    /// Classify the physical read locality from a canonical storage URL.
    /// Unknown schemes fail closed instead of being priced as local or remote.
    pub fn from_storage_url(storage_url: &str) -> Self {
        let Some((scheme, _)) = storage_url.split_once("://") else {
            return Self::Local;
        };
        match scheme.to_ascii_lowercase().as_str() {
            "file" => Self::Local,
            "s3" | "gcs" | "gs" | "az" | "azure" | "adls" | "abfs" | "hdfs" | "http" | "https" => {
                Self::Remote
            }
            _ => Self::Unknown,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Remote => "remote",
            Self::Unknown => "unknown",
        }
    }
}

/// Query-side cache policy visible before a vector access is routed.
///
/// This records which of SST's two warm-tier dependencies are available. It
/// does not predict whether a particular query will hit them; completed
/// `cold`/`warm`/`mixed`/`bypass` outcomes remain derived from the query-local
/// I/O counters. Keeping policy and outcome separate lets Exact and ANN share a
/// truthful route-time cell even when Exact bypasses a cache that ANN consults.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorCachePolicy {
    Disabled,
    InvariantsOnly,
    SurvivorOnly,
    Full,
    /// Legacy or uninstrumented access. Unknown policy is durable evidence but
    /// must not train an automatic route decision.
    #[default]
    Unknown,
}

impl VectorCachePolicy {
    /// Derive the bounded policy from the two existing SST cache seams.
    pub fn from_tiers(invariants_enabled: bool, survivor_enabled: bool) -> Self {
        match (invariants_enabled, survivor_enabled) {
            (false, false) => Self::Disabled,
            (true, false) => Self::InvariantsOnly,
            (false, true) => Self::SurvivorOnly,
            (true, true) => Self::Full,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::InvariantsOnly => "invariants_only",
            Self::SurvivorOnly => "survivor_only",
            Self::Full => "full",
            Self::Unknown => "unknown",
        }
    }

    pub fn is_admissible(self) -> bool {
        self != Self::Unknown
    }
}

/// One successfully completed physical vector access. Exact numeric geometry
/// is retained for offline analysis; it is not emitted as a metrics label.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct VectorAccessTrace {
    /// Stable physical engine label (for example `sst`).
    pub engine: String,
    pub dimensions: u64,
    pub top_k: u64,
    pub has_filter: bool,
    pub requested_mode: VectorSearchIntent,
    pub actual_path: VectorAccessPath,
    /// Read-time locality, derived from the storage URL at the engine boundary.
    /// Defaulted so traces written before this field remain readable.
    #[serde(default)]
    pub storage_scope: VectorStorageScope,
    /// Cache availability known before execution. Defaulted so traces written
    /// before this additive field remain readable and fail closed for learning.
    #[serde(default)]
    pub cache_policy: VectorCachePolicy,
}

impl IoTrace {
    /// Create an empty trace.
    pub fn new() -> Self {
        Self::default()
    }

    /// Stamp the stable per-query id (TD-TRACE-2). Set once at `instrument()` entry.
    pub fn set_query_id(&self, id: String) {
        *self.query_id.lock().unwrap_or_else(|p| p.into_inner()) = Some(id);
    }

    /// Stamp the requesting tenant for this scope (TD-CACHE-3 S1).
    pub fn set_tenant(&self, tenant: Option<String>) {
        *self.tenant_id.lock().unwrap_or_else(|p| p.into_inner()) = tenant;
    }

    /// Stamp the catalog-authoritative stable tenant id for this scope.
    pub fn set_tenant_stable_id(&self, tenant_stable_id: Option<u64>) {
        *self
            .tenant_stable_id
            .lock()
            .unwrap_or_else(|p| p.into_inner()) = tenant_stable_id;
    }

    /// The stamped tenant, if any.
    pub fn tenant(&self) -> Option<String> {
        self.tenant_id
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// The stamped catalog-authoritative stable tenant id, if resolved.
    pub fn tenant_stable_id(&self) -> Option<u64> {
        *self
            .tenant_stable_id
            .lock()
            .unwrap_or_else(|p| p.into_inner())
    }

    /// The stamped per-query id, if any.
    pub fn query_id(&self) -> Option<String> {
        self.query_id
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// Stamp the route this query is served on (`shape_class`, `backend_label`).
    /// Last write wins (a query is served by one engine). Neutral strings only.
    pub fn record_route(&self, shape_class: &str, backend_label: &str) {
        *self.route.lock().unwrap_or_else(|p| p.into_inner()) =
            Some((shape_class.to_string(), backend_label.to_string()));
    }

    /// Append one successfully completed physical vector operator access.
    pub fn record_vector_access(&self, sample: VectorAccessTrace) {
        self.vector_accesses
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(sample);
    }

    /// Mark one physical ANN mechanism as successfully engaged.
    pub fn record_vector_ann_proof(&self) {
        self.vector_ann_proofs.fetch_add(1, Ordering::Relaxed);
    }

    /// Current number of physical ANN engagement proofs in this query.
    pub fn vector_ann_proofs(&self) -> u64 {
        self.vector_ann_proofs.load(Ordering::Relaxed)
    }

    /// The stamped route, if any.
    pub fn route(&self) -> Option<(String, String)> {
        self.route.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// Stamp the resolved storage profile this query read under (neutral label,
    /// e.g. `append_bulk`/`churn`). Last write wins. TD-WLP-6.
    pub fn record_storage_profile(&self, profile: &str) {
        *self
            .storage_profile
            .lock()
            .unwrap_or_else(|p| p.into_inner()) = Some(profile.to_string());
    }

    /// The stamped storage-profile label, if any.
    pub fn storage_profile(&self) -> Option<String> {
        self.storage_profile
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// Record one classified object-store operation.
    pub fn record_op(&self, op: IoOp) {
        let counter = match op {
            IoOp::Get => &self.get_ops,
            IoOp::Put => &self.put_ops,
            IoOp::List => &self.list_ops,
            IoOp::Delete => &self.delete_ops,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Add to bytes fetched from object storage.
    pub fn record_bytes_read(&self, bytes: u64) {
        self.bytes_read.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Add to the PAX cascade's **logical** striped-read projection (bytes + GETs)
    /// — the selective read a striped path WOULD issue. Kept distinct from the
    /// physical counters (ADR-057 / TD-RDSTRAT-3).
    pub fn record_logical_striped(&self, bytes: u64, gets: u64) {
        self.logical_striped_bytes
            .fetch_add(bytes, Ordering::Relaxed);
        self.logical_striped_gets.fetch_add(gets, Ordering::Relaxed);
    }

    /// Add to the count of ranged GET requests issued.
    pub fn record_range_gets(&self, gets: u64) {
        self.range_gets.fetch_add(gets, Ordering::Relaxed);
    }

    /// Record a split-pruning outcome (TD-OLAP-3): `total` candidate
    /// row-group splits considered by a scan, of which `pruned` were skipped
    /// before fetch.
    pub fn record_splits(&self, total: u64, pruned: u64) {
        self.splits_total.fetch_add(total, Ordering::Relaxed);
        self.splits_pruned.fetch_add(pruned, Ordering::Relaxed);
    }

    /// Record a PAX cascade centroid block-prune outcome (TD-RDSTRAT-5 S3): of
    /// `total` blocks in the segment, `pruned` were skipped by the centroid probe.
    pub fn record_centroid_prune(&self, total: u64, pruned: u64) {
        self.centroid_total_blocks
            .fetch_add(total, Ordering::Relaxed);
        self.centroid_pruned_blocks
            .fetch_add(pruned, Ordering::Relaxed);
    }

    /// Record a TD-RDSTRAT-8 two-level IVF coarse-probe outcome for the active
    /// query: of `cells_total` persisted coarse centroids, `cells_probed` were
    /// ranked in RAM and `probed_rows` rows read across `fetch_rounds` coalesced
    /// Region-A ranged-reads. `whole_region_fallback = true` marks a segment where
    /// the probe was armed but missed and the read fell back to the whole Region-A
    /// scan (the GET budget the probe exists to avoid). Lands in the warehouse
    /// `VectorAnnPayload` via `TracePayload::classify`.
    pub fn record_ivf_coarse_probe(
        &self,
        cells_total: u64,
        cells_probed: u64,
        probed_rows: u64,
        fetch_rounds: u64,
        whole_region_fallback: bool,
    ) {
        self.ivf_cells_total
            .fetch_add(cells_total, Ordering::Relaxed);
        self.ivf_cells_probed
            .fetch_add(cells_probed, Ordering::Relaxed);
        self.ivf_probed_rows
            .fetch_add(probed_rows, Ordering::Relaxed);
        self.ivf_fetch_rounds
            .fetch_add(fetch_rounds, Ordering::Relaxed);
        if whole_region_fallback {
            self.ivf_whole_region_fallback
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record physical bytes fetched from PAX Region A (RaBitQ) and Region B
    /// (SQ8) during the coarse probe. Additive; cache hits contribute zero.
    pub fn record_pax_region_bytes(&self, region_a: u64, region_b: u64) {
        self.ivf_region_a_bytes
            .fetch_add(region_a, Ordering::Relaxed);
        self.ivf_region_b_bytes
            .fetch_add(region_b, Ordering::Relaxed);
    }

    /// Record a PAX Stage-F footer-stat fetch-prune outcome (TD-FPRUNE-1 P2):
    /// `total` D-blocks considered by the filtered allow-set build, `pruned`
    /// dropped by footer-resident min/max before any body GET. `pruned > 0`
    /// proves the zero-GET prune engaged (vs a stats-free conservative fetch).
    pub fn record_pax_footer_prune(&self, total: u64, pruned: u64) {
        self.pax_footer_total_blocks
            .fetch_add(total, Ordering::Relaxed);
        self.pax_footer_pruned_blocks
            .fetch_add(pruned, Ordering::Relaxed);
    }

    /// Record a general bounded-concurrent `read_ranges` outcome (TD-RDSTRAT-12):
    /// `fetch_rounds` = number of parallel rounds issued, `max_inflight` = peak
    /// in-flight reads in any round. Covers ALL ranged-read paths (footer fetches,
    /// Region-A/B reads, and general multi-range reads), not just IVF probe.
    pub fn record_read_ranges(&self, fetch_rounds: u64, max_inflight: u64) {
        self.read_ranges_fetch_rounds
            .fetch_add(fetch_rounds, Ordering::Relaxed);
        self.read_ranges_max_inflight
            .fetch_max(max_inflight, Ordering::Relaxed);
    }

    /// Record one physical ranged GET attributed to PAX cascade tier
    /// `tier_idx` (contract: the SST `CacheTier` declaration order, 0..6 —
    /// see the field docs). Out-of-range indices are dropped: a tier added on
    /// the SST side without widening this array must fail the drift test, not
    /// silently fold into another tier.
    pub fn record_tier_get(&self, tier_idx: usize, bytes: u64) {
        if tier_idx < 6 {
            self.tier_get_ops[tier_idx].fetch_add(1, Ordering::Relaxed);
            self.tier_get_bytes[tier_idx].fetch_add(bytes, Ordering::Relaxed);
        }
    }

    /// Record one object-storage metadata op (HEAD / `fs.metadata`) on the
    /// search path (TD-RDSTRAT-13 C3).
    pub fn record_metadata_op(&self) {
        self.metadata_ops.fetch_add(1, Ordering::Relaxed);
    }

    /// Record the coarse-probe rank wall duration in microseconds, including
    /// any morsel join (TD-RDSTRAT-13 C2). Additive across segments so a
    /// multi-segment query's compute term sums like its I/O terms do.
    pub fn record_ivf_probe_rank_us(&self, us: u64) {
        self.ivf_probe_rank_us.fetch_add(us, Ordering::Relaxed);
    }

    /// Record the Region-B survivor fetch+rerank wall in microseconds
    /// (TD-RDSTRAT-13 PR-B tail attribution). Additive across segments.
    pub fn record_region_b_rerank_us(&self, us: u64) {
        self.region_b_rerank_us.fetch_add(us, Ordering::Relaxed);
    }

    /// Record the Region-D top-k OID fetch+decode wall in microseconds
    /// (TD-RDSTRAT-13 PR-B tail attribution). Additive across segments.
    pub fn record_region_d_fetch_decode_us(&self, us: u64) {
        self.region_d_fetch_decode_us
            .fetch_add(us, Ordering::Relaxed);
    }

    /// Record the top-k rehydration wall in microseconds (TD-RDSTRAT-13 PR-B).
    /// Additive across segments.
    pub fn record_rehydrate_us(&self, us: u64) {
        self.rehydrate_us.fetch_add(us, Ordering::Relaxed);
    }

    /// Record the whole-cascade umbrella wall in microseconds
    /// (TD-RDSTRAT-13 PR-B).
    pub fn record_cascade_total_us(&self, us: u64) {
        self.cascade_total_us.fetch_add(us, Ordering::Relaxed);
    }

    /// Record a runtime-filter wait outcome (ADR-056 AQE-S11): `arrived` = the
    /// filter completed within the wait budget (pruning enabled); otherwise it
    /// timed out (splits read filterless, conservative). `waited_ms` = the wall
    /// ms actually spent at the rendezvous.
    pub fn record_runtime_filter_wait(&self, arrived: bool, waited_ms: u64) {
        if arrived {
            self.runtime_filter_arrived.fetch_add(1, Ordering::Relaxed);
        } else {
            self.runtime_filter_timed_out
                .fetch_add(1, Ordering::Relaxed);
        }
        self.runtime_filter_wait_ms
            .fetch_add(waited_ms, Ordering::Relaxed);
    }

    /// Add to bytes written to object storage.
    pub fn record_bytes_written(&self, bytes: u64) {
        self.bytes_written.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record a footer/metadata cache outcome.
    pub fn record_footer(&self, hit: bool) {
        let counter = if hit {
            &self.footer_hits
        } else {
            &self.footer_misses
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a batch of footer/metadata cache outcomes at once — convenient for
    /// forwarding a `RangedSegmentReader`'s per-open `SegmentReadStats`.
    pub fn record_footers(&self, hits: u64, misses: u64) {
        self.footer_hits.fetch_add(hits, Ordering::Relaxed);
        self.footer_misses.fetch_add(misses, Ordering::Relaxed);
    }

    /// Record logical in-process DRAM cache outcomes.
    pub fn record_survivor_l1s(&self, hits: u64, misses: u64) {
        self.survivor_l1_hits.fetch_add(hits, Ordering::Relaxed);
        self.survivor_l1_misses.fetch_add(misses, Ordering::Relaxed);
    }

    /// Record a batch of persistent-L2 cache probe outcomes (ADR-085 tier).
    pub fn record_l2s(&self, hits: u64, misses: u64) {
        self.l2_hits.fetch_add(hits, Ordering::Relaxed);
        self.l2_misses.fetch_add(misses, Ordering::Relaxed);
    }

    /// Add to chargeable egress bytes (cross-region / internet — KEU).
    pub fn record_egress_bytes(&self, bytes: u64) {
        self.egress_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Attribute compute milliseconds to a named engine.
    pub fn record_compute_ms(&self, engine: &str, ms: u64) {
        let mut g = self.compute_ms.lock().unwrap_or_else(|p| p.into_inner());
        *g.entry(engine.to_string()).or_insert(0) += ms;
    }

    /// Add to the pgwire result-emit wall milliseconds (row encode + socket write).
    pub fn record_emit_ms(&self, ms: u64) {
        self.emit_ms.fetch_add(ms, Ordering::Relaxed);
    }

    /// Add to the pgwire relational-pipeline setup wall milliseconds.
    pub fn record_setup_ms(&self, ms: u64) {
        self.setup_ms.fetch_add(ms, Ordering::Relaxed);
    }

    /// Add to the SessionContext build wall milliseconds.
    pub fn record_session_ms(&self, ms: u64) {
        self.session_ms.fetch_add(ms, Ordering::Relaxed);
    }

    /// Add to the table-OPEN wall milliseconds (discovery + footer open before
    /// execution). Additive so multiple registered tables accumulate.
    pub fn record_open_ms(&self, ms: u64) {
        self.open_ms.fetch_add(ms, Ordering::Relaxed);
    }

    /// Add to the lowering + planning wall milliseconds.
    pub fn record_plan_ms(&self, ms: u64) {
        self.plan_ms.fetch_add(ms, Ordering::Relaxed);
    }

    /// Record the served plan's geometry vector (TD-EXEC-2 Slice 1, observe-only):
    /// neutral scalars measured from the physical plan at the plan→execute seam,
    /// plus the per-op-kind histogram as `(label, count)` pairs. Scalars keep the
    /// max so a multi-statement scope reflects its deepest plan; histogram counts
    /// accumulate.
    pub fn record_plan_geometry(
        &self,
        depth: u64,
        nodes: u64,
        leaves: u64,
        fanout: u64,
        blocking: u64,
        ops: &[(&str, u64)],
    ) {
        self.plan_depth.fetch_max(depth, Ordering::Relaxed);
        self.plan_nodes.fetch_max(nodes, Ordering::Relaxed);
        self.plan_leaves.fetch_max(leaves, Ordering::Relaxed);
        self.plan_fanout.fetch_max(fanout, Ordering::Relaxed);
        self.plan_blocking.fetch_max(blocking, Ordering::Relaxed);
        let mut g = self.plan_ops.lock().unwrap_or_else(|p| p.into_inner());
        for (kind, count) in ops {
            *g.entry((*kind).to_string()).or_insert(0) += count;
        }
    }

    /// Record the metered per-operator execution vector (TD-TRACE-1 Slice 2,
    /// observe-only, `EXPLAIN ANALYZE` path only). Neutral primitive tuples
    /// `(op, rows_in, rows_out, ms_self, bytes, spill)` so io_trace never depends on
    /// the executor's `NodeMetric` — exactly like [`Self::record_plan_geometry`].
    /// Appended in pre-order; a normal (non-metered) query never calls this, so the
    /// hot path allocates nothing here.
    pub fn record_exec_vector(&self, ops: &[ExecOpSample<'_>]) {
        if ops.is_empty() {
            return;
        }
        let mut v = self.exec_ops.lock().unwrap_or_else(|p| p.into_inner());
        v.reserve(ops.len());
        for (op, rows_in, rows_out, ms_self, bytes, spill) in ops {
            v.push(ExecOpTrace {
                op: (*op).to_string(),
                rows_in: *rows_in,
                rows_out: *rows_out,
                ms_self: *ms_self,
                bytes: *bytes,
                spill: *spill,
            });
        }
    }

    /// Record a measured stack high-water mark (bytes) for one of the plan-tree
    /// recursions (TD-EXEC-2 Slice 1). Max semantics — the binding figure wins.
    pub fn record_stack_hwm(&self, bytes: u64) {
        self.stack_hwm_bytes.fetch_max(bytes, Ordering::Relaxed);
    }

    /// Record a table-OPEN cache outcome: `true` = hit (discovery reused, no
    /// LIST/HEAD/footer I/O), `false` = miss (cold open).
    pub fn record_table_open(&self, hit: bool) {
        let counter = if hit {
            &self.table_open_hits
        } else {
            &self.table_open_misses
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an embedding API call (KEU — Kilo-Embedding-Units metering).
    pub fn record_embedding_calls(&self, calls: u64) {
        self.embedding_calls.fetch_add(calls, Ordering::Relaxed);
    }

    /// Record input tokens consumed by embedding operations.
    pub fn record_embedding_input_tokens(&self, tokens: u64) {
        self.embedding_input_tokens
            .fetch_add(tokens, Ordering::Relaxed);
    }

    /// Record output tokens (or vector count) generated by embedding operations.
    pub fn record_embedding_output_tokens(&self, tokens: u64) {
        self.embedding_output_tokens
            .fetch_add(tokens, Ordering::Relaxed);
    }

    /// Record a complete embedding operation with input/output token counts.
    pub fn record_embedding(&self, input_tokens: u64, output_tokens: u64) {
        self.embedding_calls.fetch_add(1, Ordering::Relaxed);
        self.embedding_input_tokens
            .fetch_add(input_tokens, Ordering::Relaxed);
        self.embedding_output_tokens
            .fetch_add(output_tokens, Ordering::Relaxed);
    }

    /// Take a plain-value snapshot for emission/inspection.
    pub fn snapshot(&self) -> IoTraceSnapshot {
        IoTraceSnapshot {
            get_ops: self.get_ops.load(Ordering::Relaxed),
            put_ops: self.put_ops.load(Ordering::Relaxed),
            list_ops: self.list_ops.load(Ordering::Relaxed),
            delete_ops: self.delete_ops.load(Ordering::Relaxed),
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
            range_gets: self.range_gets.load(Ordering::Relaxed),
            logical_striped_bytes: self.logical_striped_bytes.load(Ordering::Relaxed),
            logical_striped_gets: self.logical_striped_gets.load(Ordering::Relaxed),
            splits_total: self.splits_total.load(Ordering::Relaxed),
            splits_pruned: self.splits_pruned.load(Ordering::Relaxed),
            centroid_total_blocks: self.centroid_total_blocks.load(Ordering::Relaxed),
            centroid_pruned_blocks: self.centroid_pruned_blocks.load(Ordering::Relaxed),
            ivf_cells_total: self.ivf_cells_total.load(Ordering::Relaxed),
            ivf_cells_probed: self.ivf_cells_probed.load(Ordering::Relaxed),
            ivf_probed_rows: self.ivf_probed_rows.load(Ordering::Relaxed),
            ivf_fetch_rounds: self.ivf_fetch_rounds.load(Ordering::Relaxed),
            ivf_whole_region_fallback: self.ivf_whole_region_fallback.load(Ordering::Relaxed),
            ivf_region_a_bytes: self.ivf_region_a_bytes.load(Ordering::Relaxed),
            ivf_region_b_bytes: self.ivf_region_b_bytes.load(Ordering::Relaxed),
            pax_footer_total_blocks: self.pax_footer_total_blocks.load(Ordering::Relaxed),
            pax_footer_pruned_blocks: self.pax_footer_pruned_blocks.load(Ordering::Relaxed),
            read_ranges_fetch_rounds: self.read_ranges_fetch_rounds.load(Ordering::Relaxed),
            read_ranges_max_inflight: self.read_ranges_max_inflight.load(Ordering::Relaxed),
            tier_get_ops: std::array::from_fn(|i| self.tier_get_ops[i].load(Ordering::Relaxed)),
            tier_get_bytes: std::array::from_fn(|i| self.tier_get_bytes[i].load(Ordering::Relaxed)),
            metadata_ops: self.metadata_ops.load(Ordering::Relaxed),
            ivf_probe_rank_us: self.ivf_probe_rank_us.load(Ordering::Relaxed),
            region_b_rerank_us: self.region_b_rerank_us.load(Ordering::Relaxed),
            region_d_fetch_decode_us: self.region_d_fetch_decode_us.load(Ordering::Relaxed),
            rehydrate_us: self.rehydrate_us.load(Ordering::Relaxed),
            cascade_total_us: self.cascade_total_us.load(Ordering::Relaxed),
            runtime_filter_arrived: self.runtime_filter_arrived.load(Ordering::Relaxed),
            runtime_filter_timed_out: self.runtime_filter_timed_out.load(Ordering::Relaxed),
            runtime_filter_wait_ms: self.runtime_filter_wait_ms.load(Ordering::Relaxed),
            bytes_written: self.bytes_written.load(Ordering::Relaxed),
            footer_hits: self.footer_hits.load(Ordering::Relaxed),
            footer_misses: self.footer_misses.load(Ordering::Relaxed),
            survivor_l1_hits: self.survivor_l1_hits.load(Ordering::Relaxed),
            survivor_l1_misses: self.survivor_l1_misses.load(Ordering::Relaxed),
            l2_hits: self.l2_hits.load(Ordering::Relaxed),
            l2_misses: self.l2_misses.load(Ordering::Relaxed),
            egress_bytes: self.egress_bytes.load(Ordering::Relaxed),
            embedding_calls: self.embedding_calls.load(Ordering::Relaxed),
            embedding_input_tokens: self.embedding_input_tokens.load(Ordering::Relaxed),
            embedding_output_tokens: self.embedding_output_tokens.load(Ordering::Relaxed),
            compute_ms: self
                .compute_ms
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .clone(),
            setup_ms: self.setup_ms.load(Ordering::Relaxed),
            emit_ms: self.emit_ms.load(Ordering::Relaxed),
            session_ms: self.session_ms.load(Ordering::Relaxed),
            open_ms: self.open_ms.load(Ordering::Relaxed),
            plan_ms: self.plan_ms.load(Ordering::Relaxed),
            table_open_hits: self.table_open_hits.load(Ordering::Relaxed),
            table_open_misses: self.table_open_misses.load(Ordering::Relaxed),
            plan_depth: self.plan_depth.load(Ordering::Relaxed),
            plan_nodes: self.plan_nodes.load(Ordering::Relaxed),
            plan_leaves: self.plan_leaves.load(Ordering::Relaxed),
            plan_fanout: self.plan_fanout.load(Ordering::Relaxed),
            plan_blocking: self.plan_blocking.load(Ordering::Relaxed),
            plan_ops: self
                .plan_ops
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .clone(),
            stack_hwm_bytes: self.stack_hwm_bytes.load(Ordering::Relaxed),
            route: self.route(),
            vector_accesses: self
                .vector_accesses
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .clone(),
            storage_profile: self.storage_profile(),
            exec_ops: self
                .exec_ops
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .clone(),
            query_id: self.query_id(),
        }
    }
}

/// Immutable, plain-value view of an [`IoTrace`] at a point in time — what gets
/// emitted as a `tracing` event and what a future cost-model reader consumes.
///
/// Serde-serializable so an export layer (OTLP span attributes, a JSON sink, or
/// a cost-model reader) can consume the snapshot directly — the data-shape half
/// of §4.4. See the module docs for the OTLP-collector wiring plan.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct IoTraceSnapshot {
    pub get_ops: u64,
    pub put_ops: u64,
    pub list_ops: u64,
    pub delete_ops: u64,
    pub bytes_read: u64,
    pub range_gets: u64,
    /// PAX cascade logical striped-read projection: bytes + ranged GETs a selective
    /// striped read would move (ADR-057 / TD-RDSTRAT-3). With `bytes_read`
    /// (whole-segment physical) this yields the measured striped-vs-whole headroom.
    #[serde(default)]
    pub logical_striped_bytes: u64,
    #[serde(default)]
    pub logical_striped_gets: u64,
    /// Candidate row-group splits considered by scans (TD-OLAP-3).
    #[serde(default)]
    pub splits_total: u64,
    /// Splits skipped before fetch — with `splits_total`, the skip ratio.
    #[serde(default)]
    pub splits_pruned: u64,
    /// PAX cascade centroid block-prune (TD-RDSTRAT-5 S3): total blocks in the
    /// segment and how many the centroid probe skipped. `centroid_pruned_blocks > 0`
    /// proves the prune engaged (vs a silent full-scan fallback).
    #[serde(default)]
    pub centroid_total_blocks: u64,
    #[serde(default)]
    pub centroid_pruned_blocks: u64,
    /// TD-RDSTRAT-8 two-level IVF coarse-probe outcome (durable warehouse
    /// satellite — see `VectorAnnPayload`).
    #[serde(default)]
    pub ivf_cells_total: u64,
    #[serde(default)]
    pub ivf_cells_probed: u64,
    #[serde(default)]
    pub ivf_probed_rows: u64,
    #[serde(default)]
    pub ivf_fetch_rounds: u64,
    #[serde(default)]
    pub ivf_whole_region_fallback: u64,
    /// Physical PAX Region-A (RaBitQ) / Region-B (SQ8) bytes fetched by the
    /// coarse probe (TD-RDSTRAT-8 PR-C1). Cache hits contribute zero.
    #[serde(default)]
    pub ivf_region_a_bytes: u64,
    #[serde(default)]
    pub ivf_region_b_bytes: u64,
    /// PAX Stage-F footer-stat fetch prunes (TD-FPRUNE-1 P2): D-blocks the
    /// footer's `FooterBlockStats` ruled out before any body GET, over the
    /// blocks considered. `pruned > 0` proves the zero-GET prune engaged.
    #[serde(default)]
    pub pax_footer_total_blocks: u64,
    #[serde(default)]
    pub pax_footer_pruned_blocks: u64,
    /// General bounded-concurrent `read_ranges` metrics (TD-RDSTRAT-12).
    /// Populated by any `read_ranges` or `read_ranges_prefetch` call with
    /// `parallel > 1` / `inflight > 1`. Unlike IVF-specific fields, these
    /// cover ALL ranged-read paths: footer fetches, Region-A/B reads, and
    /// general multi-range reads (not just IVF coarse probe).
    #[serde(default)]
    pub read_ranges_fetch_rounds: u64,
    /// TD-RDSTRAT-13 C1: per-tier physical-GET attribution (index contract on
    /// [`IoTrace::tier_get_ops`]).
    #[serde(default)]
    pub tier_get_ops: [u64; 6],
    #[serde(default)]
    pub tier_get_bytes: [u64; 6],
    /// Object-storage metadata ops (HEAD) on the search path (C3).
    #[serde(default)]
    pub metadata_ops: u64,
    /// Coarse-probe rank wall microseconds incl. morsel join (C2).
    #[serde(default)]
    pub ivf_probe_rank_us: u64,
    /// TD-RDSTRAT-13 PR-B tail attribution (all additive, micros).
    #[serde(default)]
    pub region_b_rerank_us: u64,
    #[serde(default)]
    pub region_d_fetch_decode_us: u64,
    #[serde(default)]
    pub rehydrate_us: u64,
    #[serde(default)]
    pub cascade_total_us: u64,
    #[serde(default)]
    pub read_ranges_max_inflight: u64,
    /// Runtime-filter wait outcomes (ADR-056 AQE-S11): arrived vs timed-out +
    /// the wall ms spent waiting. `arrived / (arrived + timed_out)` is the
    /// per-workload signal the route cost model learns to tune the wait budget.
    #[serde(default)]
    pub runtime_filter_arrived: u64,
    #[serde(default)]
    pub runtime_filter_timed_out: u64,
    #[serde(default)]
    pub runtime_filter_wait_ms: u64,
    pub bytes_written: u64,
    pub footer_hits: u64,
    pub footer_misses: u64,
    /// In-process DRAM cache outcomes. `serde(default)` keeps snapshots
    /// serialized before this additive evidence readable.
    #[serde(default)]
    pub survivor_l1_hits: u64,
    #[serde(default)]
    pub survivor_l1_misses: u64,
    /// Persistent local-disk L2 cache probe outcomes (ADR-085 / TD-IOTRACE-4).
    /// `serde(default)` keeps snapshots serialized before this field readable.
    #[serde(default)]
    pub l2_hits: u64,
    #[serde(default)]
    pub l2_misses: u64,
    pub egress_bytes: u64,
    pub embedding_calls: u64,
    pub embedding_input_tokens: u64,
    pub embedding_output_tokens: u64,
    /// Compute wall ms attributed **per engine** — the engine dimension of the
    /// geometry-dependent dispatch (TD-OLAP-4). Keys distinguish `datafusion`,
    /// `native`/`native-vectorized`, and `volcano`, so the route cost model can
    /// learn which engine wins per query shape.
    pub compute_ms: BTreeMap<String, u64>,
    /// The route this query was served on, `(shape_class, backend_label)` — the
    /// top dispatch dimension (which engine). `None` if no route was stamped.
    #[serde(default)]
    pub route: Option<(String, String)>,
    /// Successfully completed vector accesses in execution order. Additive and
    /// defaulted so snapshots emitted before TD-XMODAL-4 S2 remain readable.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vector_accesses: Vec<VectorAccessTrace>,
    /// The resolved per-collection storage profile (`append_bulk`/`churn`) this
    /// query read under (ADR-061 D6 / TD-WLP-6). `None` if unstamped.
    #[serde(default)]
    pub storage_profile: Option<String>,
    /// Stable per-query id (UUID v4), minted at `instrument()` entry (TD-TRACE-2 /
    /// ADR-066) — the durable trace sink's record id + future warehouse join key.
    /// `None` for a raw snapshot taken outside `instrument`.
    #[serde(default)]
    pub query_id: Option<String>,
    /// pgwire relational-pipeline setup wall ms — pre-execution xCatalog schema
    /// resolution + route classification, DataFusion route only (TD-OLAP-4).
    #[serde(default)]
    pub setup_ms: u64,
    /// pgwire result-emit wall ms — row encode + socket write, post-execution
    /// (TD-OLAP-4).
    #[serde(default)]
    pub emit_ms: u64,
    /// SessionContext build wall ms — per-query context+UDF setup (TD-OLAP-4).
    #[serde(default)]
    pub session_ms: u64,
    /// Table-OPEN wall ms — the per-query discovery+footer floor (TD-OLAP-4).
    #[serde(default)]
    pub open_ms: u64,
    /// Lowering + planning wall ms — the other half of the floor (TD-OLAP-4).
    #[serde(default)]
    pub plan_ms: u64,
    /// Table-OPEN cache hits (discovery reused, no LIST/HEAD/footer I/O).
    #[serde(default)]
    pub table_open_hits: u64,
    /// Table-OPEN cache misses (cold open).
    #[serde(default)]
    pub table_open_misses: u64,
    /// Served plan's longest root→leaf path (TD-EXEC-2 geometry; 0 = not recorded).
    #[serde(default)]
    pub plan_depth: u64,
    /// Served plan's total operator count (TD-EXEC-2 geometry).
    #[serde(default)]
    pub plan_nodes: u64,
    /// Served plan's leaf (scan/values) count — source fan-in (TD-EXEC-2 geometry).
    #[serde(default)]
    pub plan_leaves: u64,
    /// Served plan's widest sibling set — parallelism signal (TD-EXEC-2 geometry).
    #[serde(default)]
    pub plan_fanout: u64,
    /// Served plan's pipeline-breaker count (joins+sorts+aggregates) —
    /// memory/spill signal (TD-EXEC-2 geometry).
    #[serde(default)]
    pub plan_blocking: u64,
    /// Served plan's per-operator-kind histogram (TD-EXEC-2 geometry).
    #[serde(default)]
    pub plan_ops: BTreeMap<String, u64>,
    /// Measured stack high-water mark (bytes) of the plan-tree recursions,
    /// via the planner's `stack_probe` (TD-EXEC-2 Slice 1). The calibration
    /// figure that resolves `frame_bytes[op_kind]`; 0 = not measured.
    #[serde(default)]
    pub stack_hwm_bytes: u64,
    /// Per-operator execution actuals in pre-order (TD-TRACE-1 Slice 2), populated
    /// only on the metered `EXPLAIN ANALYZE` path — empty for a normal query.
    #[serde(default)]
    pub exec_ops: Vec<ExecOpTrace>,
}

impl IoTraceSnapshot {
    /// Total object-store operations across all verbs.
    pub fn total_ops(&self) -> u64 {
        self.get_ops + self.put_ops + self.list_ops + self.delete_ops
    }

    /// Footer-cache hit ratio in `[0, 1]`; `None` when the footer cache was not
    /// consulted (no hits or misses recorded).
    pub fn footer_hit_ratio(&self) -> Option<f64> {
        let total = self.footer_hits + self.footer_misses;
        if total == 0 {
            None
        } else {
            Some(self.footer_hits as f64 / total as f64)
        }
    }

    /// Total attributed compute milliseconds across engines.
    pub fn total_compute_ms(&self) -> u64 {
        self.compute_ms.values().copied().sum()
    }

    /// Average bytes per ranged GET — the read-granularity signal (§2.1).
    /// `None` when no ranged GETs were issued. A value far below the ~8-16 MiB
    /// S3 cost-throughput optimum means reads are fragmented and the per-GET fee
    /// dominates; this is the number the storage co-design lever moves.
    pub fn avg_get_bytes(&self) -> Option<f64> {
        if self.range_gets == 0 {
            None
        } else {
            Some(self.bytes_read as f64 / self.range_gets as f64)
        }
    }

    /// `true` when nothing was recorded — used to suppress empty trace events.
    pub fn is_empty(&self) -> bool {
        self.total_ops() == 0
            && self.bytes_read == 0
            && self.range_gets == 0
            && self.bytes_written == 0
            && self.footer_hits == 0
            && self.footer_misses == 0
            && self.survivor_l1_hits == 0
            && self.survivor_l1_misses == 0
            && self.l2_hits == 0
            && self.l2_misses == 0
            && self.egress_bytes == 0
            && self.embedding_calls == 0
            && self.embedding_input_tokens == 0
            && self.embedding_output_tokens == 0
            && self.compute_ms.is_empty()
            && self.vector_accesses.is_empty()
            // TD-RDSTRAT-12 §2: wave-only queries (bounded-concurrent
            // read_ranges with no other metered I/O) are NOT empty — a trace
            // carrying only wave metrics must still reach the billing observer.
            && self.read_ranges_fetch_rounds == 0
    }

    /// Total embedding tokens (input + output).
    pub fn total_embedding_tokens(&self) -> u64 {
        self.embedding_input_tokens + self.embedding_output_tokens
    }

    /// Emit this snapshot as a structured `tracing` event under [`TARGET`].
    /// No-op when empty. `tenant_id` and `route` label the query; all physical
    /// quantities become event fields the OTLP layer (§4.4) maps to a span.
    /// Emit the per-query I/O trace as a structured `tracing` event.
    ///
    /// TD-160: this is the **perf-class emission** — gated behind the `io-trace`
    /// cargo feature (default OFF) so it costs nothing in normal operation. When
    /// the feature is off this is a near-no-op; the cheap core counters and the
    /// route/cache/**billing** observers in [`instrument`] are unaffected and stay
    /// always-on (billing is never gated — ADR-027 non-entanglement).
    pub fn emit(&self, tenant_id: Option<&str>, route: &str) {
        if self.is_empty() {
            return;
        }
        #[cfg(feature = "io-trace")]
        tracing::info!(
            target: TARGET,
            tenant_id = tenant_id.unwrap_or("default"),
            route = route,
            storage_profile = self.storage_profile.as_deref().unwrap_or("unset"),
            get_ops = self.get_ops,
            put_ops = self.put_ops,
            list_ops = self.list_ops,
            delete_ops = self.delete_ops,
            bytes_read = self.bytes_read,
            range_gets = self.range_gets,
            avg_get_bytes = self.avg_get_bytes().unwrap_or(0.0),
            bytes_written = self.bytes_written,
            footer_hits = self.footer_hits,
            footer_misses = self.footer_misses,
            footer_hit_ratio = self.footer_hit_ratio().unwrap_or(f64::NAN),
            survivor_l1_hits = self.survivor_l1_hits,
            survivor_l1_misses = self.survivor_l1_misses,
            l2_hits = self.l2_hits,
            l2_misses = self.l2_misses,
            egress_bytes = self.egress_bytes,
            embedding_calls = self.embedding_calls,
            embedding_input_tokens = self.embedding_input_tokens,
            embedding_output_tokens = self.embedding_output_tokens,
            embedding_tokens_total = self.total_embedding_tokens(),
            compute_ms_total = self.total_compute_ms(),
            compute_ms_by_engine = ?self.compute_ms,
            "per-query I/O trace"
        );
        // Perf emission compiled out (default): consume the args so the signature
        // is stable and no unused-var warning fires in the feature-off build.
        #[cfg(not(feature = "io-trace"))]
        let _ = (tenant_id, route);
    }
}

// ---- Free helpers: record into the active scope, or silently no-op. ----

/// Record one object-store operation into the active query trace. Silently
/// no-ops outside an active [`scope`]/[`instrument`].
pub fn record_op(op: IoOp) {
    let _ = IO_TRACE.try_with(|t| t.record_op(op));
}

/// Classify an operation verb (as used by `consumption_metrics`) and record it.
pub fn record_op_str(operation: &str) {
    record_op(IoOp::classify(operation));
}

/// Add to bytes fetched from object storage for the active query.
pub fn record_bytes_read(bytes: u64) {
    let _ = IO_TRACE.try_with(|t| t.record_bytes_read(bytes));
}

/// Record the PAX cascade's logical striped-read projection (bytes + GETs a
/// selective read would move) into the active query trace — distinct from the
/// physical `bytes_read`/`range_gets` (ADR-057 / TD-RDSTRAT-3). Silently no-ops
/// outside an active scope; a core counter (always-on, like `record_bytes_read`).
pub fn record_logical_striped(bytes: u64, gets: u64) {
    let _ = IO_TRACE.try_with(|t| t.record_logical_striped(bytes, gets));
}

/// Add to the count of ranged GET requests for the active query.
///
/// ADR-030 **core counter — always-on** (deliberately NOT behind `io-trace`,
/// alongside `record_bytes_read`/`record_compute_ms`): the billing observer drains
/// `range_gets` to compute `avg_get_size = bytes_read / range_gets`, and the route
/// cost model prices `per_get`, so this must read real values even when the
/// perf-emission class is compiled out. It is one task-local atomic increment per
/// physical GET — the ~free accumulator core, never the cost the `io-trace` gate
/// exists to remove. Silently no-ops outside an active scope.
pub fn record_range_gets(gets: u64) {
    let _ = IO_TRACE.try_with(|t| t.record_range_gets(gets));
}

/// Record a scan's split-pruning outcome for the active query (TD-OLAP-3).
/// Core counter (always-on, like `record_range_gets`): the runtime-filter
/// promotion gate reads `splits_pruned / splits_total` from the billing
/// snapshot. Silently no-ops outside an active scope.
pub fn record_splits(total: u64, pruned: u64) {
    let _ = IO_TRACE.try_with(|t| t.record_splits(total, pruned));
}

/// Record a PAX cascade centroid block-prune outcome for the active query
/// (TD-RDSTRAT-5 S3): of `total` blocks, `pruned` were skipped by the centroid
/// probe. Silently no-ops outside an active scope.
pub fn record_centroid_prune(total: u64, pruned: u64) {
    let _ = IO_TRACE.try_with(|t| t.record_centroid_prune(total, pruned));
}

/// Record a TD-RDSTRAT-8 two-level IVF coarse-probe outcome for the active
/// query (durable — lands in the warehouse `VectorAnnPayload`). See
/// [`IoTrace::record_ivf_coarse_probe`]. Silently no-ops outside an active scope.
pub fn record_ivf_coarse_probe(
    cells_total: u64,
    cells_probed: u64,
    probed_rows: u64,
    fetch_rounds: u64,
    whole_region_fallback: bool,
) {
    let _ = IO_TRACE.try_with(|t| {
        t.record_ivf_coarse_probe(
            cells_total,
            cells_probed,
            probed_rows,
            fetch_rounds,
            whole_region_fallback,
        )
    });
}

/// Record physical PAX Region-A (RaBitQ) / Region-B (SQ8) bytes fetched by the
/// coarse probe for the active query (TD-RDSTRAT-8 PR-C1). No-ops outside a
/// query scope. See [`IoTrace::record_pax_region_bytes`].
pub fn record_pax_region_bytes(region_a: u64, region_b: u64) {
    let _ = IO_TRACE.try_with(|t| t.record_pax_region_bytes(region_a, region_b));
}

/// Record a PAX Stage-F footer-stat fetch-prune outcome for the active query
/// (TD-FPRUNE-1 P2): `total` D-blocks considered, `pruned` dropped by
/// footer-resident min/max before any body GET. No-ops outside a query scope.
/// See [`IoTrace::record_pax_footer_prune`].
pub fn record_pax_footer_prune(total: u64, pruned: u64) {
    let _ = IO_TRACE.try_with(|t| t.record_pax_footer_prune(total, pruned));
}

/// Record general bounded-concurrent `read_ranges` metrics for the active query
/// (TD-RDSTRAT-12). No-ops outside a query scope. See [`IoTrace::record_read_ranges`].
pub fn record_read_ranges(fetch_rounds: u64, max_inflight: u64) {
    let _ = IO_TRACE.try_with(|t| t.record_read_ranges(fetch_rounds, max_inflight));
}

/// Record one physical ranged GET attributed to PAX cascade tier `tier_idx`
/// (TD-RDSTRAT-13 C1; index contract on [`IoTrace::record_tier_get`]).
/// No-ops outside a query scope.
pub fn record_tier_get(tier_idx: usize, bytes: u64) {
    let _ = IO_TRACE.try_with(|t| t.record_tier_get(tier_idx, bytes));
}

/// Record one object-storage metadata op (HEAD) on the search path
/// (TD-RDSTRAT-13 C3). No-ops outside a query scope.
pub fn record_metadata_op() {
    let _ = IO_TRACE.try_with(|t| t.record_metadata_op());
}

/// Record the coarse-probe rank wall duration in microseconds for the active
/// query (TD-RDSTRAT-13 C2). No-ops outside a query scope.
pub fn record_ivf_probe_rank_us(us: u64) {
    let _ = IO_TRACE.try_with(|t| t.record_ivf_probe_rank_us(us));
}

/// Record the Region-B survivor fetch+rerank wall for the active query
/// (TD-RDSTRAT-13 PR-B). No-ops outside a query scope.
pub fn record_region_b_rerank_us(us: u64) {
    let _ = IO_TRACE.try_with(|t| t.record_region_b_rerank_us(us));
}

/// Record the Region-D top-k OID fetch+decode wall for the active query
/// (TD-RDSTRAT-13 PR-B). No-ops outside a query scope.
pub fn record_region_d_fetch_decode_us(us: u64) {
    let _ = IO_TRACE.try_with(|t| t.record_region_d_fetch_decode_us(us));
}

/// Record the top-k rehydration wall for the active query (TD-RDSTRAT-13
/// PR-B). No-ops outside a query scope.
pub fn record_rehydrate_us(us: u64) {
    let _ = IO_TRACE.try_with(|t| t.record_rehydrate_us(us));
}

/// Record the whole-cascade umbrella wall for the active query
/// (TD-RDSTRAT-13 PR-B). No-ops outside a query scope.
pub fn record_cascade_total_us(us: u64) {
    let _ = IO_TRACE.try_with(|t| t.record_cascade_total_us(us));
}

/// Drain general bounded-concurrent `read_ranges` metrics from the storage layer
/// and forward to the active query trace (TD-RDSTRAT-12).
///
/// This bridges the layer boundary: the storage layer records metrics globally
/// (avoiding upward dependencies), and this function (in the observability layer)
/// drains them and forwards to the per-query io_trace. Call this after operations
/// that use `FileSystem::read_ranges` with parallelism enabled.
///
/// Scope-gated BY DESIGN: outside an active io_trace scope the FS slot is left
/// untouched so no-scope callers (unit tests, background scans) keep the
/// last-wave visibility the raw slot always provided — `record_read_ranges`
/// would no-op anyway, so draining there would only destroy the metrics.
pub fn drain_and_forward_read_ranges_metrics() {
    use proximadb_storage_filesystem_types::drain_read_ranges_metrics;
    let in_scope = IO_TRACE.try_with(|_| true).is_ok();
    if in_scope && let Some(metrics) = drain_read_ranges_metrics() {
        record_read_ranges(metrics.fetch_rounds, metrics.max_inflight);
    }
}

/// Record a runtime-filter wait outcome for the active query (ADR-056 AQE-S11).
pub fn record_runtime_filter_wait(arrived: bool, waited_ms: u64) {
    let _ = IO_TRACE.try_with(|t| t.record_runtime_filter_wait(arrived, waited_ms));
}

/// Add to bytes written to object storage for the active query.
pub fn record_bytes_written(bytes: u64) {
    let _ = IO_TRACE.try_with(|t| t.record_bytes_written(bytes));
}

/// Record a footer/metadata cache outcome for the active query. (TD-160:
/// perf/geometry trace class — compile-time gated.)
#[cfg(feature = "io-trace")]
pub fn record_footer(hit: bool) {
    let _ = IO_TRACE.try_with(|t| t.record_footer(hit));
}
#[cfg(not(feature = "io-trace"))]
#[inline(always)]
pub fn record_footer(_hit: bool) {}

/// Record a batch of footer/metadata cache outcomes for the active query.
/// (TD-160: perf/geometry trace class — compile-time gated.)
#[cfg(feature = "io-trace")]
pub fn record_footers(hits: u64, misses: u64) {
    let _ = IO_TRACE.try_with(|t| t.record_footers(hits, misses));
}
#[cfg(not(feature = "io-trace"))]
#[inline(always)]
pub fn record_footers(_hits: u64, _misses: u64) {}

/// Record logical in-process DRAM cache outcomes for the active query.
/// Compile-time gated with the other performance/geometry trace fields.
#[cfg(feature = "io-trace")]
pub fn record_survivor_l1s(hits: u64, misses: u64) {
    let _ = IO_TRACE.try_with(|t| t.record_survivor_l1s(hits, misses));
}
#[cfg(not(feature = "io-trace"))]
#[inline(always)]
pub fn record_survivor_l1s(_hits: u64, _misses: u64) {}

/// Record a batch of persistent-L2 cache probe outcomes for the active query
/// (ADR-085 / TD-IOTRACE-4). (TD-160: perf/geometry trace class —
/// compile-time gated, same class as the footer-cache outcomes.)
#[cfg(feature = "io-trace")]
pub fn record_l2s(hits: u64, misses: u64) {
    let _ = IO_TRACE.try_with(|t| t.record_l2s(hits, misses));
}
#[cfg(not(feature = "io-trace"))]
#[inline(always)]
pub fn record_l2s(_hits: u64, _misses: u64) {}

/// Record chargeable egress bytes (cross-region / internet — KEU) for the
/// active query.
pub fn record_egress_bytes(bytes: u64) {
    let _ = IO_TRACE.try_with(|t| t.record_egress_bytes(bytes));
}

/// Attribute compute milliseconds to `engine` for the active query.
pub fn record_compute_ms(engine: &str, ms: u64) {
    let _ = IO_TRACE.try_with(|t| t.record_compute_ms(engine, ms));
}

/// Add table-OPEN wall ms (discovery + footer open) to the active query trace.
pub fn record_open_ms(ms: u64) {
    let _ = IO_TRACE.try_with(|t| t.record_open_ms(ms));
}

/// Add pgwire relational-pipeline setup wall ms to the active query trace.
pub fn record_setup_ms(ms: u64) {
    let _ = IO_TRACE.try_with(|t| t.record_setup_ms(ms));
}

/// Record the served plan's geometry vector for the active query (TD-EXEC-2
/// Slice 1, observe-only). Core counter (always-on, like `record_plan_ms`):
/// the route cost model's geometry tier and the stack calibration fit both
/// consume it from the ledger. Silently no-ops outside an active scope.
pub fn record_plan_geometry(
    depth: u64,
    nodes: u64,
    leaves: u64,
    fanout: u64,
    blocking: u64,
    ops: &[(&str, u64)],
) {
    let _ =
        IO_TRACE.try_with(|t| t.record_plan_geometry(depth, nodes, leaves, fanout, blocking, ops));
}

/// Record the metered per-operator execution vector for the active query
/// (TD-TRACE-1 Slice 2, observe-only). Core counter (always-on, never behind the
/// `io-trace` feature — the cost model / billing read per-op detail from the
/// ledger): only the metered `EXPLAIN ANALYZE` path calls this, so a normal query
/// never allocates. Neutral primitive tuples `(op, rows_in, rows_out, ms_self,
/// bytes, spill)` so io_trace never depends on the executor's `NodeMetric`.
/// Silently no-ops outside an active scope.
pub fn record_exec_vector(ops: &[ExecOpSample<'_>]) {
    let _ = IO_TRACE.try_with(|t| t.record_exec_vector(ops));
}

/// Record a measured plan-recursion stack high-water mark (bytes) for the
/// active query (TD-EXEC-2 Slice 1). Silently no-ops outside an active scope.
pub fn record_stack_hwm(bytes: u64) {
    let _ = IO_TRACE.try_with(|t| t.record_stack_hwm(bytes));
}

/// Add pgwire result-emit wall ms (row encode + socket write) to the active trace.
pub fn record_emit_ms(ms: u64) {
    let _ = IO_TRACE.try_with(|t| t.record_emit_ms(ms));
}

/// Add SessionContext build wall ms to the active query trace.
pub fn record_session_ms(ms: u64) {
    let _ = IO_TRACE.try_with(|t| t.record_session_ms(ms));
}

/// Add lowering + planning wall ms to the active query trace.
pub fn record_plan_ms(ms: u64) {
    let _ = IO_TRACE.try_with(|t| t.record_plan_ms(ms));
}

/// Record a table-OPEN cache outcome on the active query trace.
pub fn record_table_open(hit: bool) {
    let _ = IO_TRACE.try_with(|t| t.record_table_open(hit));
}

/// Stamp the route (`shape_class`, `backend_label`) onto the active query trace.
/// Silently no-ops outside an active scope. Neutral strings — io_trace never
/// depends on a query-layer type.
pub fn record_route(shape_class: &str, backend_label: &str) {
    let _ = IO_TRACE.try_with(|t| t.record_route(shape_class, backend_label));
}

/// Append one successfully completed physical vector access to the active
/// query trace. Silently no-ops outside an active scope.
pub fn record_vector_access(sample: VectorAccessTrace) {
    let _ = IO_TRACE.try_with(|t| t.record_vector_access(sample));
}

/// Mark physical ANN engagement in the active query. This is an internal
/// attribution signal; callers emit the durable access sample only after the
/// enclosing vector operation succeeds.
pub fn record_vector_ann_proof() {
    let _ = IO_TRACE.try_with(|t| t.record_vector_ann_proof());
}

/// Read the active query's physical ANN proof count, or zero outside a scope.
pub fn vector_ann_proof_count() -> u64 {
    IO_TRACE
        .try_with(|t| t.vector_ann_proofs())
        .unwrap_or_default()
}

/// Stamp the resolved storage profile (`append_bulk`/`churn`) onto the active
/// query trace (ADR-061 D6 / TD-WLP-6). Silently no-ops outside an active
/// scope. Neutral label string — io_trace never depends on `StorageProfile`.
pub fn record_storage_profile(profile: &str) {
    let _ = IO_TRACE.try_with(|t| t.record_storage_profile(profile));
}

/// Record embedding API calls for the active query (KEU metering).
pub fn record_embedding_calls(calls: u64) {
    let _ = IO_TRACE.try_with(|t| t.record_embedding_calls(calls));
}

/// Record input tokens consumed by embedding operations for the active query.
pub fn record_embedding_input_tokens(tokens: u64) {
    let _ = IO_TRACE.try_with(|t| t.record_embedding_input_tokens(tokens));
}

/// Record output tokens generated by embedding operations for the active query.
pub fn record_embedding_output_tokens(tokens: u64) {
    let _ = IO_TRACE.try_with(|t| t.record_embedding_output_tokens(tokens));
}

/// Record a complete embedding operation (input + output tokens) for the active query.
pub fn record_embedding(input_tokens: u64, output_tokens: u64) {
    let _ = IO_TRACE.try_with(|t| t.record_embedding(input_tokens, output_tokens));
}

/// Snapshot the active query trace, if any.
pub fn snapshot() -> Option<IoTraceSnapshot> {
    IO_TRACE.try_with(|t| t.snapshot()).ok()
}

/// Clone out an `Arc` handle to the active query trace, if any.
///
/// Components whose I/O later runs on DataFusion-spawned tokio tasks (where the
/// `IO_TRACE` task-local is absent) capture this handle **while still in the
/// query scope** — table open / physical planning — and record through it, so
/// their reads attribute to the correct per-query trace (TD-OLAP-3). Returns
/// `None` outside any [`scope`]/[`instrument`].
pub fn current_handle() -> Option<Arc<IoTrace>> {
    IO_TRACE.try_with(|t| t.clone()).ok()
}

/// Observer invoked at trace flush with `(snapshot, shape_class, backend_label)`
/// when the query stamped a route. This is the dependency-inversion seam: the
/// query layer registers a sink that feeds the trace-driven route cost model
/// (C4), so io_trace ingests into routing *without depending on it*.
type RouteObserver = dyn Fn(&IoTraceSnapshot, &str, &str) + Send + Sync;

static ROUTE_OBSERVER: Mutex<Option<Box<RouteObserver>>> = Mutex::new(None);

/// Install (or clear with `None`) the route-trace observer. Called once at
/// startup by the query layer; replaceable in tests.
pub fn set_route_observer(observer: Option<Box<RouteObserver>>) {
    *ROUTE_OBSERVER.lock().unwrap_or_else(|p| p.into_inner()) = observer;
}

/// Feed the registered observer, if any, with a completed query's route trace.
fn notify_route_observer(snap: &IoTraceSnapshot, shape_class: &str, backend_label: &str) {
    if let Some(obs) = ROUTE_OBSERVER
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .as_ref()
    {
        obs(snap, shape_class, backend_label);
    }
}

/// Observer invoked once at query completion when at least one physical vector
/// access was recorded. Separate from `RouteObserver`: the latter is the one
/// containing compute backend, while this dimension may repeat per query.
type VectorAccessObserver = dyn Fn(&IoTraceSnapshot) + Send + Sync;

static VECTOR_ACCESS_OBSERVER: Mutex<Option<Box<VectorAccessObserver>>> = Mutex::new(None);

/// Install or clear the vector-access observer. The query layer uses this
/// dependency-inversion seam to feed observe-only access-path cost cells.
pub fn set_vector_access_observer(observer: Option<Box<VectorAccessObserver>>) {
    *VECTOR_ACCESS_OBSERVER
        .lock()
        .unwrap_or_else(|p| p.into_inner()) = observer;
}

fn notify_vector_access_observer(snap: &IoTraceSnapshot) {
    if let Some(observer) = VECTOR_ACCESS_OBSERVER
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .as_ref()
    {
        observer(snap);
    }
}

/// Observer invoked at trace flush with `snapshot` for cache sizing feedback (T2.2).
/// The cache orchestrator registers to receive footer hit-rate and avg_get_bytes signals.
/// This is the dependency-inversion seam: io_trace feeds cache sizing *without depending on it*.
type CacheObserver = dyn Fn(&IoTraceSnapshot) + Send + Sync;

static CACHE_OBSERVER: Mutex<Option<Box<CacheObserver>>> = Mutex::new(None);

/// Install (or clear with `None`) the cache-trace observer. Called once at startup
/// by the cache orchestrator; replaceable in tests.
pub fn set_cache_observer(observer: Option<Box<CacheObserver>>) {
    *CACHE_OBSERVER.lock().unwrap_or_else(|p| p.into_inner()) = observer;
}

/// Feed the registered cache observer, if any, with a completed query's trace.
/// Called after route observer; both observers can be active simultaneously.
fn notify_cache_observer(snap: &IoTraceSnapshot) {
    if let Some(obs) = CACHE_OBSERVER
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .as_ref()
    {
        obs(snap);
    }
}

/// Observer invoked at trace flush with `snapshot` + `tenant_id` for **billing**
/// (ADR-030). The metering layer registers a sink that emits the always-on
/// per-tenant consumption meters (KRU read-compute, from `compute_ms`) from the
/// same measured snapshot the perf observers read — so cost-model bytes and
/// billed bytes cannot diverge. Dependency-inversion seam: io_trace feeds billing
/// *without depending on it*.
///
/// Unlike the route/cache observers this is the **billing class** — always-on,
/// never wrapped in the perf `io-trace` feature gate (ADR-027 non-entanglement;
/// CI-guarded).
type BillingObserver = dyn Fn(&IoTraceSnapshot, Option<&str>) + Send + Sync;

static BILLING_OBSERVER: Mutex<Option<Box<BillingObserver>>> = Mutex::new(None);

/// Install (or clear with `None`) the billing-trace observer. Called once at
/// startup by the metering layer; replaceable in tests.
pub fn set_billing_observer(observer: Option<Box<BillingObserver>>) {
    *BILLING_OBSERVER.lock().unwrap_or_else(|p| p.into_inner()) = observer;
}

/// Feed the registered billing observer, if any, with a completed query's trace
/// and the owning tenant. Called after the route + cache observers at scope close.
fn notify_billing_observer(snap: &IoTraceSnapshot, tenant_id: Option<&str>) {
    if let Some(obs) = BILLING_OBSERVER
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .as_ref()
    {
        obs(snap, tenant_id);
    }
}

/// Observer invoked at trace flush with `snapshot` + `tenant_id` for the durable
/// **trace ETL sink** (TD-TRACE-2 / ADR-066). The sink layer registers a sink that
/// enqueues the completed per-query snapshot to a bounded spool for background
/// export. Dependency-inversion seam: io_trace feeds the sink *without depending on
/// it*, exactly like the billing observer — but this is a SEPARATE, default-OFF
/// observer, so billing stays always-on / never-gated (ADR-027) and the sink can be
/// gated without touching it. The registered sink MUST only enqueue (no I/O on the
/// query path).
type TraceObserver = dyn Fn(&IoTraceSnapshot, Option<&str>) + Send + Sync;

static TRACE_OBSERVER: Mutex<Option<Box<TraceObserver>>> = Mutex::new(None);

/// Install (or clear with `None`) the trace-sink observer. Called once at startup by
/// the sink layer when the sink is enabled; replaceable in tests. Default: none
/// installed ⇒ zero cost.
pub fn set_trace_observer(observer: Option<Box<TraceObserver>>) {
    *TRACE_OBSERVER.lock().unwrap_or_else(|p| p.into_inner()) = observer;
}

/// Feed the registered trace-sink observer, if any, with a completed query's trace
/// and the owning tenant. Called last in the `instrument()` fan-out.
fn notify_trace_observer(snap: &IoTraceSnapshot, tenant_id: Option<&str>) {
    if let Some(obs) = TRACE_OBSERVER
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .as_ref()
    {
        obs(snap, tenant_id);
    }
}

/// Bind a fresh [`IoTrace`] to `future` and await it. Lower-level than
/// [`instrument`]; use when the caller wants to read the snapshot itself before
/// the scope ends.
pub async fn scope<F: std::future::Future>(future: F) -> F::Output {
    IO_TRACE.scope(Arc::new(IoTrace::new()), future).await
}

/// Bind an existing query trace to a child future. `tokio::task_local!` does
/// not cross `tokio::spawn`; engine schedulers capture [`current_handle`] before
/// spawning and rebind it with this function so child I/O remains attributable
/// to the owning query rather than disappearing from its cost cell.
pub async fn scope_with_handle<F: std::future::Future>(
    trace: Arc<IoTrace>,
    future: F,
) -> F::Output {
    IO_TRACE.scope(trace, future).await
}

/// Wrap a query future in a fresh trace, run it, then emit the captured
/// snapshot as a [`TARGET`] event labelled by `tenant_id`/`route`. This is the
/// one call a request handler adds at the query boundary — co-locate it with
/// the existing `predicate_diagnostics::scope`.
/// TD-CACHE-3 S1: the ambient tenant of the current instrumented request
/// scope, if any. Engine-side per-tenant consumers (survivor-cache fair-share
/// keys) read this instead of threading tenant through every search
/// signature. `None` outside a scope — callers must fall back gracefully.
pub fn current_tenant() -> Option<String> {
    IO_TRACE.try_with(|t| t.tenant()).ok().flatten()
}

/// Catalog-authoritative stable tenant id of the current instrumented request.
/// No alias parsing or hashing fallback is permitted: unresolved means `None`.
pub fn current_tenant_stable_id() -> Option<u64> {
    IO_TRACE.try_with(|t| t.tenant_stable_id()).ok().flatten()
}

pub async fn instrument<F>(
    tenant_id: Option<String>,
    route: impl Into<String>,
    future: F,
) -> F::Output
where
    F: std::future::Future,
{
    instrument_with_stable_tenant(tenant_id, None, route, future).await
}

/// Instrument a query with both its external billing label and its
/// catalog-authoritative numeric tenant identity.
pub async fn instrument_with_stable_tenant<F>(
    tenant_id: Option<String>,
    tenant_stable_id: Option<u64>,
    route: impl Into<String>,
    future: F,
) -> F::Output
where
    F: std::future::Future,
{
    let route = route.into();
    IO_TRACE
        .scope(Arc::new(IoTrace::new()), async move {
            // TD-TRACE-2: mint a stable per-query id at scope entry so the durable
            // trace sink can identify (and later join) every query's record. One
            // UUID + one lock set per query — negligible, and off any row loop.
            let _ = IO_TRACE.try_with(|t| t.set_query_id(uuid::Uuid::new_v4().to_string()));
            // TD-CACHE-3 S1: stamp the tenant into the scope so engine-side
            // consumers (per-tenant cache keys) can read it ambiently.
            let _ = IO_TRACE.try_with(|t| t.set_tenant(tenant_id.clone()));
            let _ = IO_TRACE.try_with(|t| t.set_tenant_stable_id(tenant_stable_id));
            let out = future.await;
            // Still inside the scope: read and emit before the binding drops.
            if let Ok((snap, stamped_route)) = IO_TRACE.try_with(|t| (t.snapshot(), t.route())) {
                snap.emit(tenant_id.as_deref(), &route);
                // C4 ingestion: if a route was stamped, feed the cost model the
                // measured cost of this (shape-class, backend). Skip empty traces.
                if !snap.is_empty()
                    && let Some((shape_class, backend_label)) = stamped_route
                {
                    notify_route_observer(&snap, &shape_class, &backend_label);
                }
                if !snap.vector_accesses.is_empty() {
                    notify_vector_access_observer(&snap);
                }
                // T2.2 ingestion: feed the cache orchestrator for trace-driven sizing.
                // Skip empty traces to avoid wasteful cache budget updates.
                if !snap.is_empty() {
                    notify_cache_observer(&snap);
                }
                // ADR-030 billing fan-out (always-on): emit the per-tenant
                // consumption meters (KRU) from the same snapshot. Tenant + the
                // per-engine compute_ms are both in hand here.
                if !snap.is_empty() {
                    notify_billing_observer(&snap, tenant_id.as_deref());
                }
                // TD-TRACE-2 durable trace sink (separate, default-OFF observer —
                // billing above stays always-on/never-gated). Fires last; only
                // enqueues the completed snapshot for background export.
                if !snap.is_empty() {
                    notify_trace_observer(&snap, tenant_id.as_deref());
                }
            }
            out
        })
        .await
}

#[cfg(all(test, feature = "io-trace"))]
mod tests {
    use super::*;

    /// ADR-030: the always-on billing observer fires at scope close with the
    /// owning tenant and the same accumulated snapshot the perf observers see —
    /// so KRU is emitted from `compute_ms` and cannot diverge from the cost
    /// model's view. Drives the real `instrument` drain path end to end.
    #[tokio::test]
    #[allow(clippy::type_complexity)] // test-only capture cell for the billing observer
    async fn billing_observer_receives_tenant_and_accumulated_compute_ms() {
        use std::sync::{Arc, Mutex};
        let captured: Arc<Mutex<Option<(Option<String>, u64, u64)>>> = Arc::new(Mutex::new(None));
        let sink = captured.clone();
        set_billing_observer(Some(Box::new(move |snap, tenant| {
            *sink.lock().unwrap_or_else(|p| p.into_inner()) = Some((
                tenant.map(String::from),
                snap.total_compute_ms(),
                snap.bytes_read,
            ));
        })));

        instrument(Some("acme".to_string()), "test-route", async {
            record_compute_ms("native", 4);
            record_compute_ms("native", 3);
            record_bytes_read(100);
        })
        .await;

        set_billing_observer(None); // restore global state for other tests

        let got = captured.lock().unwrap_or_else(|p| p.into_inner()).clone();
        assert_eq!(
            got,
            Some((Some("acme".to_string()), 7, 100)),
            "billing observer must receive the tenant + summed compute_ms (4+3) + bytes_read from the same snapshot"
        );
    }

    /// TD-TRACE-2: the durable trace-sink observer fires at scope close (like
    /// billing), with the completed snapshot carrying a minted `query_id` and the
    /// owning tenant. An EMPTY query never fires it (the `!is_empty()` guard).
    #[tokio::test]
    #[allow(clippy::type_complexity)]
    async fn trace_observer_receives_snapshot_with_query_id_and_skips_empty() {
        use std::sync::{Arc, Mutex};
        let captured: Arc<Mutex<Vec<(Option<String>, Option<String>)>>> =
            Arc::new(Mutex::new(Vec::new()));
        let sink = captured.clone();
        set_trace_observer(Some(Box::new(move |snap, tenant| {
            sink.lock()
                .unwrap_or_else(|p| p.into_inner())
                .push((tenant.map(String::from), snap.query_id.clone()));
        })));

        // Non-empty query → observer fires with a minted query_id.
        instrument(Some("acme".to_string()), "test-route", async {
            record_bytes_read(42);
        })
        .await;
        // Empty query → observer must NOT fire (no measurable I/O).
        instrument(Some("acme".to_string()), "test-route", async {}).await;

        set_trace_observer(None); // restore global state for other tests

        let got = captured.lock().unwrap_or_else(|p| p.into_inner()).clone();
        assert_eq!(
            got.len(),
            1,
            "trace observer fires only for the non-empty query"
        );
        assert_eq!(got[0].0.as_deref(), Some("acme"));
        assert!(
            got[0].1.as_deref().is_some_and(|id| !id.is_empty()),
            "the snapshot carries a minted query_id: {got:?}"
        );
    }

    #[test]
    fn classify_maps_known_verbs() {
        assert_eq!(IoOp::classify("list_pax"), IoOp::List);
        assert_eq!(IoOp::classify("list_parquet"), IoOp::List);
        assert_eq!(IoOp::classify("read_parquet"), IoOp::Get);
        assert_eq!(IoOp::classify("fetch_pax_ranged"), IoOp::Get);
        assert_eq!(IoOp::classify("write_parquet"), IoOp::Put);
        assert_eq!(IoOp::classify("delete_segment"), IoOp::Delete);
        // Unknown verb is the conservative read default.
        assert_eq!(IoOp::classify("mystery"), IoOp::Get);
    }

    #[test]
    fn snapshot_aggregates_and_derives() {
        let t = IoTrace::new();
        t.record_op(IoOp::List);
        t.record_op(IoOp::Get);
        t.record_op(IoOp::Get);
        t.record_bytes_read(1_024);
        t.record_bytes_read(512);
        t.record_range_gets(3);
        t.record_footer(true);
        t.record_footer(true);
        t.record_footer(false);
        t.record_egress_bytes(2_048);
        t.record_compute_ms("volcano", 3);
        t.record_compute_ms("volcano", 4);
        t.record_compute_ms("datafusion", 10);

        let s = t.snapshot();
        assert_eq!(s.get_ops, 2);
        assert_eq!(s.list_ops, 1);
        assert_eq!(s.total_ops(), 3);
        assert_eq!(s.bytes_read, 1_536);
        assert_eq!(s.range_gets, 3);
        assert_eq!(s.avg_get_bytes(), Some(1_536.0 / 3.0));
        assert_eq!(s.egress_bytes, 2_048);
        assert_eq!(s.footer_hit_ratio(), Some(2.0 / 3.0));
        assert_eq!(s.total_compute_ms(), 17);
        assert_eq!(s.compute_ms.get("volcano"), Some(&7));
        assert!(!s.is_empty());
    }

    #[test]
    fn ivf_coarse_probe_record_populates_snapshot() {
        // TD-RDSTRAT-8: record_ivf_coarse_probe must populate all five durable
        // counters so the warehouse VectorAnnPayload captures the probe outcome.
        let t = IoTrace::new();
        t.record_ivf_coarse_probe(64, 8, 4096, 3, false);
        t.record_ivf_coarse_probe(0, 0, 0, 0, true); // armed-but-missed fallback
        t.record_pax_region_bytes(200_000, 800_000); // physical PAX tier bytes
        let s = t.snapshot();
        assert_eq!(s.ivf_cells_total, 64);
        assert_eq!(s.ivf_cells_probed, 8);
        assert_eq!(s.ivf_probed_rows, 4096);
        assert_eq!(s.ivf_fetch_rounds, 3);
        assert_eq!(s.ivf_whole_region_fallback, 1);
        assert_eq!(s.ivf_region_a_bytes, 200_000);
        assert_eq!(s.ivf_region_b_bytes, 800_000);
    }

    #[test]
    fn empty_snapshot_is_empty() {
        assert!(IoTrace::new().snapshot().is_empty());
        assert_eq!(IoTraceSnapshot::default().footer_hit_ratio(), None);
        assert_eq!(IoTraceSnapshot::default().avg_get_bytes(), None);
    }

    #[tokio::test]
    async fn free_helpers_record_into_active_scope() {
        let captured = scope(async {
            record_op_str("fetch_pax");
            record_op_str("list_pax");
            record_bytes_read(4_096);
            record_footer(false);
            record_compute_ms("sst", 5);
            snapshot()
        })
        .await;

        let s = captured.expect("snapshot inside scope");
        assert_eq!(s.get_ops, 1);
        assert_eq!(s.list_ops, 1);
        assert_eq!(s.bytes_read, 4_096);
        assert_eq!(s.footer_misses, 1);
        assert_eq!(s.compute_ms.get("sst"), Some(&5));
    }

    #[tokio::test]
    async fn free_helpers_noop_outside_scope() {
        // No active scope: every helper must silently no-op, never panic, and
        // snapshot() returns None.
        record_op(IoOp::Get);
        record_bytes_read(999);
        record_footer(true);
        record_compute_ms("sst", 1);
        assert!(snapshot().is_none());
    }

    #[tokio::test]
    async fn record_footers_batches_outcomes() {
        let s = scope(async {
            record_footers(7, 3);
            record_footer(true); // one more hit on top
            snapshot()
        })
        .await
        .expect("snapshot inside scope");
        assert_eq!(s.footer_hits, 8);
        assert_eq!(s.footer_misses, 3);
        assert_eq!(s.footer_hit_ratio(), Some(8.0 / 11.0));
    }

    #[tokio::test]
    async fn record_route_round_trips_in_scope_and_noops_outside() {
        record_route("olap/parquet", "DataFusionLocal"); // outside scope: no-op
        let r = scope(async {
            record_route("olap/parquet", "DataFusionLocal");
            IO_TRACE.try_with(|t| t.route()).ok().flatten()
        })
        .await;
        assert_eq!(
            r,
            Some(("olap/parquet".to_string(), "DataFusionLocal".to_string()))
        );
    }

    #[tokio::test]
    async fn vector_accesses_preserve_every_physical_operator_in_order() {
        record_vector_access(VectorAccessTrace {
            engine: "sst".to_string(),
            dimensions: 384,
            top_k: 10,
            has_filter: false,
            requested_mode: VectorSearchIntent::Adaptive,
            actual_path: VectorAccessPath::Exact,
            storage_scope: VectorStorageScope::Unknown,
            cache_policy: VectorCachePolicy::Unknown,
        }); // outside scope: no-op

        let snap = scope(async {
            record_vector_access(VectorAccessTrace {
                engine: "sst".to_string(),
                dimensions: 384,
                top_k: 10,
                has_filter: false,
                requested_mode: VectorSearchIntent::Adaptive,
                actual_path: VectorAccessPath::Exact,
                storage_scope: VectorStorageScope::Local,
                cache_policy: VectorCachePolicy::Disabled,
            });
            record_vector_access(VectorAccessTrace {
                engine: "sst".to_string(),
                dimensions: 768,
                top_k: 25,
                has_filter: true,
                requested_mode: VectorSearchIntent::Approximate,
                actual_path: VectorAccessPath::Ann,
                storage_scope: VectorStorageScope::Remote,
                cache_policy: VectorCachePolicy::Full,
            });
            IO_TRACE.try_with(|t| t.snapshot()).unwrap()
        })
        .await;

        assert_eq!(snap.vector_accesses.len(), 2);
        assert_eq!(snap.vector_accesses[0].actual_path, VectorAccessPath::Exact);
        assert_eq!(snap.vector_accesses[1].dimensions, 768);
        assert_eq!(snap.vector_accesses[1].actual_path, VectorAccessPath::Ann);
        assert_eq!(
            snap.vector_accesses[1].cache_policy,
            VectorCachePolicy::Full
        );
        assert!(!snap.is_empty(), "an access-only trace is durable evidence");
    }

    #[test]
    fn vector_storage_scope_classifies_urls_without_claiming_access_tier() {
        assert_eq!(
            VectorStorageScope::from_storage_url("file:///var/lib/proximadb/data"),
            VectorStorageScope::Local
        );
        assert_eq!(
            VectorStorageScope::from_storage_url("/var/lib/proximadb/data"),
            VectorStorageScope::Local
        );
        for url in [
            "s3://bucket/data",
            "gs://bucket/data",
            "az://container/data",
            "azure://container/data",
            "adls://container/data",
            "abfs://container/data",
            "hdfs://namenode/data",
            "https://object-gateway.example/data",
        ] {
            assert_eq!(
                VectorStorageScope::from_storage_url(url),
                VectorStorageScope::Remote,
                "{url}"
            );
        }
        assert_eq!(
            VectorStorageScope::from_storage_url("future-store://bucket/data"),
            VectorStorageScope::Unknown
        );
    }

    #[test]
    fn vector_cache_policy_maps_both_route_time_tier_seams() {
        assert_eq!(
            VectorCachePolicy::from_tiers(false, false),
            VectorCachePolicy::Disabled
        );
        assert_eq!(
            VectorCachePolicy::from_tiers(true, false),
            VectorCachePolicy::InvariantsOnly
        );
        assert_eq!(
            VectorCachePolicy::from_tiers(false, true),
            VectorCachePolicy::SurvivorOnly
        );
        assert_eq!(
            VectorCachePolicy::from_tiers(true, true),
            VectorCachePolicy::Full
        );
        assert!(!VectorCachePolicy::Unknown.is_admissible());
    }

    #[test]
    fn legacy_vector_access_defaults_route_time_dimensions_to_unknown() {
        let legacy = r#"{
            "engine":"sst",
            "dimensions":384,
            "top_k":10,
            "has_filter":false,
            "requested_mode":"exact",
            "actual_path":"exact"
        }"#;
        let access: VectorAccessTrace = serde_json::from_str(legacy).expect("legacy access");
        assert_eq!(access.storage_scope, VectorStorageScope::Unknown);
        assert_eq!(access.cache_policy, VectorCachePolicy::Unknown);
    }

    #[tokio::test]
    async fn flush_feeds_vector_observer_without_a_compute_route_stamp() {
        use std::sync::Arc;

        let seen: Arc<Mutex<Vec<IoTraceSnapshot>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&seen);
        set_vector_access_observer(Some(Box::new(move |snap| {
            sink.lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(snap.clone());
        })));

        instrument(None, "rest.v2.records.search", async {
            record_vector_access(VectorAccessTrace {
                engine: "sst".to_string(),
                dimensions: 384,
                top_k: 10,
                has_filter: false,
                requested_mode: VectorSearchIntent::Exact,
                actual_path: VectorAccessPath::Exact,
                storage_scope: VectorStorageScope::Local,
                cache_policy: VectorCachePolicy::Disabled,
            });
        })
        .await;
        set_vector_access_observer(None);

        let got = seen.lock().unwrap_or_else(|p| p.into_inner());
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].vector_accesses.len(), 1);
    }

    #[tokio::test]
    async fn existing_handle_rebinds_trace_across_spawn() {
        let snap = scope(async {
            let trace = current_handle().expect("active trace handle");
            tokio::spawn(scope_with_handle(trace, async {
                record_range_gets(1);
                record_bytes_read(4096);
                record_vector_ann_proof();
            }))
            .await
            .expect("child task");
            snapshot().expect("outer snapshot")
        })
        .await;

        assert_eq!(snap.range_gets, 1);
        assert_eq!(snap.bytes_read, 4096);
    }

    #[tokio::test]
    async fn floor_fall_through_attributes_route_to_the_floor_not_the_primary() {
        // ADR-064 / TD-TRACE-1: when a primary engine declines and the DataFusion
        // FLOOR serves the query, the dispatch re-stamps the route
        // (relational_pipeline.rs "last write wins") so the cost cell learns the
        // engine that ACTUALLY served — never the primary that declined. This
        // pins that attribution contract against the future TD-ROUTE-3 dispatch
        // refactor (which must preserve it).
        let snap = scope(async {
            record_route("olap/parquet", "Native(Volcano)"); // primary attempt
            record_route("olap/parquet", "DataFusionLocal"); // floor served → re-stamp
            IO_TRACE.try_with(|t| t.snapshot()).unwrap()
        })
        .await;
        assert_eq!(
            snap.route,
            Some(("olap/parquet".to_string(), "DataFusionLocal".to_string())),
            "floor fall-through must attribute to the floor, not the declined primary"
        );
    }

    /// TD-WLP-6: the storage-profile stamp round-trips in scope (and into the
    /// snapshot), no-ops outside a scope.
    #[tokio::test]
    async fn record_storage_profile_round_trips_and_reaches_snapshot() {
        record_storage_profile("churn"); // outside scope: no-op
        let snap = scope(async {
            record_storage_profile("churn");
            IO_TRACE.try_with(|t| t.snapshot()).unwrap()
        })
        .await;
        assert_eq!(snap.storage_profile.as_deref(), Some("churn"));
        // A fresh trace has no profile stamp.
        assert_eq!(IoTrace::new().snapshot().storage_profile, None);
    }

    #[tokio::test]
    async fn record_exec_vector_round_trips_to_snapshot() {
        // Outside a scope: silent no-op (no panic).
        record_exec_vector(&[("Scan", 0, 10, 1, None, false)]);
        let snap = scope(async {
            record_exec_vector(&[
                ("Aggregate", 10, 1, 3, None, false),
                ("Scan", 0, 10, 1, Some(2048), false),
            ]);
            IO_TRACE.try_with(|t| t.snapshot()).unwrap()
        })
        .await;
        assert_eq!(snap.exec_ops.len(), 2);
        assert_eq!(snap.exec_ops[0].op, "Aggregate");
        assert_eq!(snap.exec_ops[0].rows_in, 10);
        assert_eq!(snap.exec_ops[0].rows_out, 1);
        assert_eq!(snap.exec_ops[0].ms_self, 3);
        assert_eq!(snap.exec_ops[1].op, "Scan");
        assert_eq!(snap.exec_ops[1].bytes, Some(2048));
        assert!(!snap.exec_ops[1].spill);
        // A fresh trace records no exec ops (metered path only).
        assert!(IoTrace::new().snapshot().exec_ops.is_empty());
    }

    // Both flush→observer cases live in ONE test: the route observer is a
    // process-global, so running the set/clear cases sequentially here avoids a
    // race with a parallel sibling test. (Other instrument tests stamp no route,
    // so they never invoke the observer regardless of install order.)
    #[tokio::test]
    async fn flush_feeds_route_observer_only_when_route_is_stamped() {
        use std::sync::Arc;
        let seen: Arc<Mutex<Vec<(IoTraceSnapshot, String, String)>>> =
            Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        set_route_observer(Some(Box::new(move |snap, class, backend| {
            sink.lock()
                .unwrap()
                .push((snap.clone(), class.to_string(), backend.to_string()));
        })));

        // (1) Stamped route → observer fires with the measured snapshot.
        instrument(Some("t".to_string()), "pgwire.query", async {
            record_op_str("fetch_pax_ranged");
            record_bytes_read(8 << 20);
            record_range_gets(2);
            record_route("olap/parquet", "DataFusionLocal");
        })
        .await;

        // (2) I/O but NO route stamped → observer must NOT fire.
        instrument(None, "rest.v2.records.search", async {
            record_op_str("fetch_pax");
        })
        .await;

        set_route_observer(None); // reset global so other tests are unaffected

        let got = seen.lock().unwrap();
        assert_eq!(
            got.len(),
            1,
            "observer fired exactly once (only the routed query)"
        );
        let (snap, class, backend) = &got[0];
        assert_eq!(class, "olap/parquet");
        assert_eq!(backend, "DataFusionLocal");
        assert_eq!(snap.range_gets, 2);
        assert_eq!(snap.bytes_read, 8 << 20);
    }

    #[test]
    fn snapshot_json_round_trips_for_export() {
        // §4.4 enabler: the snapshot is the export payload (OTLP attrs / JSON
        // sink / cost-model reader), so it must serde round-trip losslessly.
        let mut compute = BTreeMap::new();
        compute.insert("datafusion".to_string(), 12u64);
        let s = IoTraceSnapshot {
            get_ops: 5,
            list_ops: 1,
            bytes_read: 8_388_608,
            range_gets: 2,
            footer_hits: 3,
            footer_misses: 1,
            egress_bytes: 4_096,
            compute_ms: compute,
            ..Default::default()
        };
        let json = serde_json::to_string(&s).expect("serialize");
        let back: IoTraceSnapshot = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(s, back);
        // Spot-check a derived field survives the data it depends on.
        assert_eq!(back.avg_get_bytes(), Some(8_388_608.0 / 2.0));
    }

    #[tokio::test]
    async fn instrument_runs_and_returns_output() {
        let out = instrument(
            Some("tenant-a".to_string()),
            "rest.v2.records.scan",
            async {
                record_op_str("fetch_pax");
                record_bytes_read(2_048);
                42
            },
        )
        .await;
        assert_eq!(out, 42);
    }

    #[test]
    fn embedding_recording_aggregates_correctly() {
        let t = IoTrace::new();
        t.record_embedding_calls(5);
        t.record_embedding_input_tokens(1000);
        t.record_embedding_output_tokens(2000);

        // Record individual tokens
        t.record_embedding_input_tokens(500);
        t.record_embedding_output_tokens(1000);

        let s = t.snapshot();
        assert_eq!(s.embedding_calls, 5);
        assert_eq!(s.embedding_input_tokens, 1500);
        assert_eq!(s.embedding_output_tokens, 3000);
        assert_eq!(s.total_embedding_tokens(), 4500);
    }

    #[test]
    fn embedding_records_complete_operation() {
        let t = IoTrace::new();
        t.record_embedding(100, 200); // input, output
        t.record_embedding(50, 100);

        let s = t.snapshot();
        assert_eq!(s.embedding_calls, 2);
        assert_eq!(s.embedding_input_tokens, 150);
        assert_eq!(s.embedding_output_tokens, 300);
        assert_eq!(s.total_embedding_tokens(), 450);
    }

    /// TD-IOTRACE-4: persistent-L2 probe outcomes accumulate into the snapshot,
    /// count as trace activity, and a snapshot serialized BEFORE the fields
    /// existed still deserializes with them defaulted (mixed-read-safe).
    #[test]
    fn l2_probes_record_snapshot_and_default_from_legacy_json() {
        let t = IoTrace::new();
        t.record_l2s(3, 1);
        t.record_l2s(1, 0);
        let s = t.snapshot();
        assert_eq!(s.l2_hits, 4);
        assert_eq!(s.l2_misses, 1);
        assert!(!s.is_empty(), "an L2-only trace is not empty");

        let mut v = serde_json::to_value(IoTraceSnapshot::default()).unwrap();
        let obj = v.as_object_mut().unwrap();
        obj.remove("l2_hits");
        obj.remove("l2_misses");
        let legacy: IoTraceSnapshot = serde_json::from_value(v).unwrap();
        assert_eq!((legacy.l2_hits, legacy.l2_misses), (0, 0));
    }

    /// The survivor cache is an in-process L1 above the filesystem. Its
    /// outcomes must remain distinct from persistent L2 and physical GETs,
    /// while old snapshots remain readable after the additive fields land.
    #[test]
    fn l1_probes_record_snapshot_and_default_from_legacy_json() {
        let t = IoTrace::new();
        t.record_survivor_l1s(2, 1);
        t.record_survivor_l1s(3, 0);
        let s = t.snapshot();
        assert_eq!((s.survivor_l1_hits, s.survivor_l1_misses), (5, 1));
        assert!(!s.is_empty(), "an L1-only trace is not empty");

        let mut v = serde_json::to_value(IoTraceSnapshot::default()).unwrap();
        let obj = v.as_object_mut().unwrap();
        obj.remove("survivor_l1_hits");
        obj.remove("survivor_l1_misses");
        let legacy: IoTraceSnapshot = serde_json::from_value(v).unwrap();
        assert_eq!((legacy.survivor_l1_hits, legacy.survivor_l1_misses), (0, 0));
    }

    #[tokio::test]
    async fn free_helpers_record_embedding_in_scope() {
        let captured = scope(async {
            record_embedding_calls(3);
            record_embedding_input_tokens(100);
            record_embedding_output_tokens(200);
            record_embedding(50, 75); // +1 call, +50 input, +75 output
            snapshot()
        })
        .await;

        let s = captured.expect("snapshot inside scope");
        assert_eq!(s.embedding_calls, 4); // 3 + 1
        assert_eq!(s.embedding_input_tokens, 150); // 100 + 50
        assert_eq!(s.embedding_output_tokens, 275); // 200 + 75
        assert_eq!(s.total_embedding_tokens(), 425);
    }

    #[tokio::test]
    async fn empty_snapshot_includes_embedding_fields() {
        let s = IoTrace::new().snapshot();
        assert_eq!(s.embedding_calls, 0);
        assert_eq!(s.embedding_input_tokens, 0);
        assert_eq!(s.embedding_output_tokens, 0);
        assert_eq!(s.total_embedding_tokens(), 0);
    }

    #[tokio::test]
    async fn snapshot_json_includes_embedding_fields() {
        let mut compute = BTreeMap::new();
        compute.insert("datafusion".to_string(), 12u64);
        let s = IoTraceSnapshot {
            get_ops: 5,
            list_ops: 1,
            bytes_read: 8_388_608,
            range_gets: 2,
            footer_hits: 3,
            footer_misses: 1,
            egress_bytes: 4_096,
            embedding_calls: 10,
            embedding_input_tokens: 1000,
            embedding_output_tokens: 2000,
            compute_ms: compute,
            ..Default::default()
        };

        let json = serde_json::to_string(&s).expect("serialize");
        let back: IoTraceSnapshot = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(s, back);
        assert_eq!(back.total_embedding_tokens(), 3000);
    }
}
