// Copyright 2025 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Read-side compute routing — the `ComputeScheduler`.
//!
//! Implements the planner-boundary half of the multi-engine routing contract in
//! `docs/12-design/DATA_WAREHOUSE_AND_ENGINEERING_COURSE_CORRECTION_2026_06_04.adoc`
//! §5 (Intelligent Multi-Engine Routing, Policy→RL). It selects ONE physical
//! execution engine per query plan and materializes the choice as a
//! [`SelectRouteDecision`] for telemetry and (once a SELECT EXPLAIN surface
//! exists) `EXPLAIN`.
//!
//! ## Convergence (Convergence Gate)
//! This is the READ/query-execution analog of the table-WRITE router's
//! [`crate::query::table_write_plan::RoutedExecutionPlan`] /
//! `RouteDecisionMetadata`. It deliberately REUSES the canonical engine enum
//! [`ComputeBackend`] and [`CatalogWorkloadProfile`] instead of introducing a
//! parallel vocabulary — there is one engine taxonomy across read and write.
//! It does not own durable authority.
//!
//! * `authority_mode`: control-plane routing decision; no durable authority.
//! * `policy_boundary`: exactly one engine chosen per query plan at the planner
//!   boundary (pgwire today), never per row.
//! * `freshness`: the static rule keeps strong freshness on
//!   [`ComputeBackend::Native`] — the live Volcano executor over
//!   WAL+`RecordStorage`. Only OLAP-shape queries over Parquet-backed tables
//!   route to `DataFusionLocal` (P1, live behind the default-on
//!   `datafusion-integration` feature).
//!
//! Routing is three tiers behind this one seam: the static shape rule
//! ([`ComputeScheduler::route_select`]), the trace-driven cost-model consult
//! ([`ComputeScheduler::route_select_advised`] — observe-mode advisory on every
//! pgwire relational `SELECT`), and the flag-gated live override
//! (`PROXIMADB_ROUTE_COST_OVERRIDE`, default OFF — explore/exploit over
//! freshness-safe candidates with the TD-170 round-trip hard gate; enablement
//! gated by TD-ROUTE-1's capability-aware candidate fix). The rule engine grows
//! without moving the seam.

use crate::query::read_route::{
    CandidateReadRoute, ReadFreshnessSla, ReadPolicyBoundary, ReadSplitSummary, RoutedReadPlan,
};
use crate::query::table_write_plan::ComputeBackend;
use proximadb_catalog::CatalogAuthorityMode;
use proximadb_catalog::CatalogWorkloadProfile;

/// Estimated cardinality bucket of a query's scan/result — a §5.2 Phase-1 shape
/// input. Bucketed (not raw counts) so it refines the cost-model shape-class
/// without exploding the key space. `Unknown` (the default) keeps the coarse
/// 2-part class, so a planner that cannot estimate rows changes nothing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CardinalityClass {
    /// No estimate available — class stays coarse (backward-compatible).
    #[default]
    Unknown,
    /// Point / very small result (≲ 1k rows) — favors the low-latency row path.
    Small,
    /// Mid-size scan/result (≲ 1M rows).
    Medium,
    /// Large scan/result — favors the vectorized columnar engine.
    Large,
}

impl CardinalityClass {
    /// Bucket an estimated row count; `None` → [`CardinalityClass::Unknown`].
    pub fn from_estimate(rows: Option<u64>) -> Self {
        match rows {
            None => CardinalityClass::Unknown,
            Some(n) if n <= 1_000 => CardinalityClass::Small,
            Some(n) if n <= 1_000_000 => CardinalityClass::Medium,
            Some(_) => CardinalityClass::Large,
        }
    }

    /// Stable shape-class suffix, or `None` when unknown (omitted from the key).
    pub(crate) fn class_suffix(self) -> Option<&'static str> {
        match self {
            CardinalityClass::Unknown => None,
            CardinalityClass::Small => Some("card=s"),
            CardinalityClass::Medium => Some("card=m"),
            CardinalityClass::Large => Some("card=l"),
        }
    }
}

/// Bucketed count of partitions / row-groups / segments a route fans out over —
/// a §5.2 Phase-1 shape input. High fan-out is GET-round-trip-dominated (the
/// dominant cost term), so it discriminates routes the binary signals cannot.
/// `Unknown` (the default) keeps the coarse class.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PartitionFanout {
    /// No partition count available — class stays coarse.
    #[default]
    Unknown,
    /// A single partition / segment.
    Single,
    /// A handful (≤ 8) — bounded fan-out.
    Few,
    /// Many partitions — fan-out / GET-count dominated.
    Many,
}

impl PartitionFanout {
    /// Bucket a partition/segment count; `None` → [`PartitionFanout::Unknown`].
    pub fn from_count(count: Option<u32>) -> Self {
        match count {
            None => PartitionFanout::Unknown,
            Some(0) | Some(1) => PartitionFanout::Single,
            Some(n) if n <= 8 => PartitionFanout::Few,
            Some(_) => PartitionFanout::Many,
        }
    }

    /// Stable shape-class suffix, or `None` when unknown (omitted from the key).
    pub(crate) fn class_suffix(self) -> Option<&'static str> {
        match self {
            PartitionFanout::Unknown => None,
            PartitionFanout::Single => Some("part=1"),
            PartitionFanout::Few => Some("part=f"),
            PartitionFanout::Many => Some("part=m"),
        }
    }
}

/// TD-OLAP-4 operation dimension: the OLAP operation class, so the cost model
/// keys its per-engine samples by *what the query does* (the geometry the shadow
/// ledger showed engines win/lose on), not just cardinality/partition. Default
/// `Unknown` preserves the coarse class + warmed cells.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OperationClass {
    #[default]
    Unknown,
    /// Unfiltered scalar aggregate answerable from footer stats — `COUNT(*)`,
    /// `MIN`/`MAX`. Native wins (metadata elision, flat cost).
    MetadataElidable,
    /// Ungrouped scalar aggregate over column data — `SUM`/`AVG`/filtered `COUNT`.
    /// Native-competitive once vectorized + morsel-parallel (TD-OLAP-12).
    ScalarAggregate,
    /// `GROUP BY` aggregate. High-cardinality → DataFusion (native has no spilling).
    Grouped,
    /// String-heavy predicate/projection — `LIKE`/regex/string funcs. DataFusion
    /// (native has no predicate pushdown / wide-string kernels yet).
    StringHeavy,
    /// Anything else — joins, window, subquery, plain projection.
    Other,
}

impl OperationClass {
    pub(crate) fn class_suffix(self) -> Option<&'static str> {
        match self {
            OperationClass::Unknown => None,
            OperationClass::MetadataElidable => Some("op=meta"),
            OperationClass::ScalarAggregate => Some("op=agg"),
            OperationClass::Grouped => Some("op=grp"),
            OperationClass::StringHeavy => Some("op=str"),
            OperationClass::Other => Some("op=other"),
        }
    }
}

/// Bucketed estimated plan depth — half of the TD-EXEC-2 geometry tier.
///
/// Seed cutoffs: `Shallow` (≤ 6) covers the maximal single-table operator chain
/// (scan→filter→aggregate→having→project→sort — the ClickBench posture native
/// wins per TD-OLAP-15); `Mid` (7–12) a few joins; `Deep` (> 12) join trees /
/// nested subqueries (the TPC-H posture DataFusion wins). The ledger records the
/// *measured* `plan_depth` per query (TD-EXEC-2 Slice 1), so these bands are
/// refittable from evidence, not asserted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthBand {
    /// Estimated depth ≤ 6 — a single-table operator chain.
    Shallow,
    /// Estimated depth 7–12 — a handful of joins / one nesting level.
    Mid,
    /// Estimated depth > 12 — join trees, deep subquery nesting.
    Deep,
}

impl DepthBand {
    /// Bucket an estimated plan depth (operator count on the longest root→leaf path).
    pub fn from_depth(depth: u32) -> Self {
        match depth {
            0..=6 => DepthBand::Shallow,
            7..=12 => DepthBand::Mid,
            _ => DepthBand::Deep,
        }
    }
}

/// Bucketed pipeline-breaker count (joins + sorts + aggregates — the ops
/// `plan_geometry::OpKind::is_blocking` flags) — the other half of the geometry
/// tier. Blocking count is the strongest engine discriminator TD-OLAP-15
/// measured: zero/low = streaming scan shapes (native-competitive), high =
/// join/agg-bound shapes (DataFusion's surface).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockingBand {
    /// No pipeline breakers — pure scan/filter/project/limit, streams through.
    Zero,
    /// 1–2 breakers — a single aggregate and/or sort (scan-heavy analytics).
    Low,
    /// ≥ 3 breakers — join-bearing / multi-breaker plans.
    High,
}

impl BlockingBand {
    /// Bucket an estimated pipeline-breaker count.
    pub fn from_count(count: u32) -> Self {
        match count {
            0 => BlockingBand::Zero,
            1..=2 => BlockingBand::Low,
            _ => BlockingBand::High,
        }
    }
}

/// TD-EXEC-2 Slice 3: the plan-geometry tier of the cost-model shape-class —
/// bucketed estimated depth × pipeline-breaker bands, so the EWMA cost cells
/// accumulate per shape⊕geometry class and `route_select_advised` compares
/// engines on the geometry they measurably win/lose on (TD-OLAP-15). Estimated
/// from the AST *before* lowering (the route decision precedes the physical
/// plan); because `finalize_route` stamps this class into the io_trace, the
/// consult key and the EWMA fold key are the same by construction. Default
/// `Unknown` contributes no suffix — the coarse class and warmed cells are
/// preserved (backward-compatible, like every other signal here).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GeometryClass {
    /// No geometry estimate available — class stays coarse.
    #[default]
    Unknown,
    /// Bucketed estimate: depth band × blocking band.
    Known {
        /// Bucketed estimated plan depth.
        depth: DepthBand,
        /// Bucketed estimated pipeline-breaker count.
        blocking: BlockingBand,
    },
}

impl GeometryClass {
    /// Bucket an estimated (depth, pipeline-breaker count) pair.
    pub fn from_estimate(depth: u32, blocking: u32) -> Self {
        GeometryClass::Known {
            depth: DepthBand::from_depth(depth),
            blocking: BlockingBand::from_count(blocking),
        }
    }

    /// Stable shape-class suffix `geom=<depth-band>x<blocking-band>`, or `None`
    /// when unknown (omitted from the key).
    pub(crate) fn class_suffix(self) -> Option<&'static str> {
        let GeometryClass::Known { depth, blocking } = self else {
            return None;
        };
        Some(match (depth, blocking) {
            (DepthBand::Shallow, BlockingBand::Zero) => "geom=sx0",
            (DepthBand::Shallow, BlockingBand::Low) => "geom=sxlo",
            (DepthBand::Shallow, BlockingBand::High) => "geom=sxhi",
            (DepthBand::Mid, BlockingBand::Zero) => "geom=mx0",
            (DepthBand::Mid, BlockingBand::Low) => "geom=mxlo",
            (DepthBand::Mid, BlockingBand::High) => "geom=mxhi",
            (DepthBand::Deep, BlockingBand::Zero) => "geom=dx0",
            (DepthBand::Deep, BlockingBand::Low) => "geom=dxlo",
            (DepthBand::Deep, BlockingBand::High) => "geom=dxhi",
        })
    }
}

/// Shape signals the scheduler routes on.
///
/// P0 used only `engages_relational` — the join / `GROUP BY` / aggregate / set-op
/// gate (the OLAP-shape signal). P1 added `parquet_backed`. C4 Phase-2b adds the
/// §5.2 Phase-1 inputs — `cardinality` and `partition_fanout` — which refine the
/// cost-model shape-class so routing and exploration discriminate beyond
/// `olap/oltp × native/parquet`. TD-OLAP-4 adds `operation_class`; TD-EXEC-2
/// Slice 3 adds `geometry`. All default to `Unknown`, preserving the coarse
/// class (and warmed cells).
#[derive(Debug, Clone, Copy, Default)]
pub struct QueryShape {
    /// The query lowers against the relational algebra engine for a reason the
    /// single-table legacy path cannot serve (joins, `GROUP BY`, aggregates,
    /// `HAVING`, set-ops). Today this is the OLAP-candidate signal.
    pub engages_relational: bool,
    /// P1: every referenced table is Parquet-backed (object-store / open-format),
    /// so the OLAP arm can route to DataFusion. Set by the planner boundary
    /// (`try_run_select`) only when the `datafusion-integration` feature is
    /// compiled in, so the route is never advertised when the build can't honor it.
    pub parquet_backed: bool,
    /// Estimated scan/result cardinality bucket (§5.2 Phase-1). `Unknown` default.
    pub cardinality: CardinalityClass,
    /// Bucketed partition/row-group fan-out (§5.2 Phase-1). `Unknown` default.
    pub partition_fanout: PartitionFanout,
    /// TD-OLAP-1 slice 2: tables backed by PAX segments (not Parquet).
    ///
    /// Observed and carried on the shape, but it does NOT select a route today:
    /// TD-USUB-9 removed the arm that stamped `DataFusionLocal` for it, because
    /// dispatch requires `parquet_backed` and Volcano actually served. The real
    /// slice-2 dispatch wiring will consume this field again.
    pub pax_backed: bool,
    /// TD-USUB-2: every referenced table can be registered with the DataFusion
    /// destination as a native (non-Parquet) relational table, so the full ANSI SQL
    /// surface is reachable without a manual `ALTER TABLE … MATERIALIZE`.
    ///
    /// This is a genuine *route* input, not a dispatch-time override: DataFusion
    /// really does execute the query, so stamping `DataFusionLocal` is accurate.
    /// Deciding it here keeps decision and dispatch in one place — overriding the
    /// backend at dispatch instead would mis-attribute cost-model cells, the exact
    /// failure TD-USUB-9 tracks. Default-OFF behind
    /// `PROXIMADB_NATIVE_TABLE_PROVIDER`; `false` default.
    pub native_registerable: bool,
    /// TD-OLAP-4: the OLAP operation class (metadata-elidable / scalar-aggregate /
    /// grouped / string-heavy) — the geometry dimension the engines win/lose on.
    pub operation_class: OperationClass,
    /// TD-EXEC-2 Slice 3: bucketed plan-geometry estimate (depth band ×
    /// pipeline-breaker band) refining the cost-model class. `Unknown` default.
    pub geometry: GeometryClass,
    /// TD-ROUTE-1: the plan contains a JOIN anywhere (top-level FROM, derived
    /// table / nested join / CTE, or a subquery that lowers to a semi/anti/left
    /// join) or a set-op body. The Native arm cannot serve these over Parquet —
    /// Volcano has join executors but no external-parquet scan, and the
    /// vectorized native-over-parquet path is single-scan-only — so a flipped
    /// plan would be served by the DataFusion floor while the io_trace stamps
    /// Native (poisoning the cost model's per-backend cells), and
    /// `EXPLAIN ANALYZE` would execute a diverging plan. This is an
    /// *eligibility* bit (ADR-058: eligibility precedes selection), not a cost
    /// input: the cost-model override must never FLIP such a plan onto
    /// `Native`; static routes are unaffected. It is a dedicated bit because
    /// [`OperationClass`] cannot carry it — a `JOIN … GROUP BY` classifies as
    /// `Grouped` (the FROM clause is never inspected there). Computed at the
    /// pgwire boundary (`query_join_bearing`); `false` default.
    pub join_bearing: bool,
}

/// Vector-operator geometry presented to the same scheduler without conflating
/// an operator access method with the containing SQL plan's compute backend.
/// A `vector_search(...)` source may execute inside a DataFusion join, so this
/// is deliberately a sibling of [`QueryShape`], not a boolean embedded in it.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorQueryShape {
    /// Query-vector width. Kept as measured geometry for route telemetry; it is
    /// not bucketed into an unvalidated threshold on the hot path.
    pub dimensions: usize,
    /// Requested result cardinality.
    pub top_k: u32,
    /// Whether the vector operator carries a pushed predicate.
    pub has_filter: bool,
    /// Explicit caller intent. `None` means the correctness-first exact floor.
    pub requested_mode: Option<proximadb_search_types::search_params::SearchMode>,
}

/// Observable vector access-path decision produced at the canonical scheduler
/// seam and consumed by the one native vector-search port.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorRouteDecision {
    /// Vector access methods execute in the specialized native backend.
    pub backend: ComputeBackend,
    /// Catalog workload classification for telemetry and policy.
    pub workload_profile: CatalogWorkloadProfile,
    /// Typed access strategy and execution effort that must be forwarded to
    /// storage unchanged. Exact scan and ANN remain access methods over the
    /// same authority, not peer [`ComputeBackend`] variants.
    pub search_mode: proximadb_search_types::search_params::SearchMode,
    /// Human-readable selection rationale.
    pub reason: String,
    /// What produced the decision. Cost-driven override remains default-OFF;
    /// this slice records the static/explicit selection truthfully.
    pub source: RouteSource,
    has_filter: bool,
}

impl VectorRouteDecision {
    /// Stable low-cardinality label derived from the canonical search mode.
    pub fn strategy_label(&self) -> &'static str {
        use proximadb_search_types::search_params::SearchMode;

        match &self.search_mode {
            SearchMode::Exact => "exact",
            SearchMode::Approximate { .. } => "approximate",
            SearchMode::Adaptive { .. } => "adaptive",
        }
    }

    /// Stable class used by route metrics. Dimension/top-k remain numeric trace
    /// geometry until measured evidence justifies stable buckets; this avoids
    /// fragmenting cost cells around arbitrary thresholds.
    pub fn shape_class(&self) -> String {
        format!(
            "vector/native/mode={}/filter={}",
            self.strategy_label(),
            if self.has_filter { "yes" } else { "no" }
        )
    }

    /// Stable one-line diagnostic for pgwire/UDTF logs and future EXPLAIN.
    pub fn explain_line(&self) -> String {
        format!(
            "Vector Route: Native(Vector) (strategy={}, reason=\"{}\")",
            self.strategy_label(),
            self.reason
        )
    }
}

/// What produced a [`SelectRouteDecision`] — a `route_decisions_total` metric
/// label (co-design C4 operator surface). `Static` is the policy rule; the two
/// override variants only occur when `PROXIMADB_ROUTE_COST_OVERRIDE` is enabled.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RouteSource {
    /// The static shape-rule decision (no cost-model override fired).
    #[default]
    Static,
    /// Live cost-model override to a confidently-cheaper freshness-safe backend.
    OverrideExploit,
    /// Rate-limited exploration flip to a cold freshness-safe backend (warm-up).
    OverrideExplore,
}

impl RouteSource {
    /// Stable metric-label form for `route_decisions_total{source}`.
    pub fn as_str(&self) -> &'static str {
        match self {
            RouteSource::Static => "static",
            RouteSource::OverrideExploit => "override_exploit",
            RouteSource::OverrideExplore => "override_explore",
        }
    }
}

/// Process-level route master switches (TD-ROUTE-3). The env-driven
/// `PROXIMADB_NATIVE_ROUTE` / `PROXIMADB_DUCKDB_ROUTE` gates are read ONCE at the
/// pgwire boundary and threaded in here so the decision core
/// ([`ComputeScheduler::route_select_advised`]) stays free of `std::env` reads
/// and is unit-testable by injecting flags. When a switch is on and the shape is
/// the corresponding primary's eligible shape, the decision IS that engine —
/// authoritative for the thin `match decision.backend` dispatch (no downstream
/// re-decision). Both default-`false` (`RouteFlags::default()`), preserving the
/// static + cost-model route.
#[derive(Debug, Clone, Copy, Default)]
pub struct RouteFlags {
    /// `PROXIMADB_NATIVE_ROUTE`: force the native-over-parquet vectorized engine
    /// as PRIMARY for the ops it is eligible for (metadata-elidable / scalar
    /// aggregate). DataFusion remains the correctness floor (the adapter's own
    /// fine plan-shape gates + fallback).
    pub native_route: bool,
    /// `PROXIMADB_DUCKDB_ROUTE`: force DuckDB-local as PRIMARY for the join/grouped
    /// parquet shapes it is the measured win for. DataFusion remains the floor.
    pub duckdb_route: bool,
}

/// The two parquet-OLAP PRIMARY engines' *shape* eligibility (ADR-058:
/// eligibility precedes selection). Pure over [`QueryShape`] and unit-testable;
/// the fine plan-shape gates (single-scan / no-predicate / width-cap / stats
/// trust) stay inside the native adapter (`try_native_over_parquet`), which needs
/// the lowered physical plan.
///
/// The two sets are **disjoint** — native is the *single-scan* ungrouped
/// scalar-aggregate shape (a join-bearing plan is NEVER native-servable, so it is
/// excluded here; the adapter would decline it anyway), while DuckDB is the
/// join/grouped shape. Disjointness makes the env-force precedence unambiguous and
/// the dispatch `match` arms mutually exclusive: a `(scalar-agg, join-bearing)`
/// query is DuckDB's, never native's — which is exactly where the former dispatch
/// sent it (native attempted, declined the join, fell through to DuckDB).
///
/// Mirrors the former dispatch gates (`native_eligible_op &&` the adapter's
/// single-scan gate / `duckdb_eligible_shape`) so folding them into the decision
/// changes no served engine — only makes `decision.backend` authoritative.
fn parquet_primary_eligibility(shape: &QueryShape) -> (bool, bool) {
    let native_ok = !shape.join_bearing
        && matches!(
            shape.operation_class,
            OperationClass::MetadataElidable | OperationClass::ScalarAggregate
        );
    let duckdb_ok = shape.join_bearing || matches!(shape.operation_class, OperationClass::Grouped);
    (native_ok, duckdb_ok)
}

/// Which of ADR-058's three routing axes a [`RouteReason`] explains. The router
/// decides in this order — a backend must be *eligible* (can serve the shape
/// soundly) before it is *selected* (cheapest by the cost cells), and an
/// external/Parquet route additionally passes a *trust* gate (are the catalog
/// statistics trusted enough to route here, ADR-058 D5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum RouteAxis {
    /// Soundness — this backend *can* serve the shape (ADR-058: eligibility
    /// precedes selection).
    Eligibility,
    /// Cost — why this eligible backend was chosen (static rule or cost-model
    /// override/explore).
    Selection,
    /// Statistics trust — the external/Parquet route trusts the catalog
    /// base-snapshot (ADR-058 D5).
    Trust,
}

/// One structured reason behind a [`SelectRouteDecision`], tagged by the axis it
/// explains. Replaces the single flat `reason` string as the machine-readable
/// record the ADR-064 query trace and `EXPLAIN` consume — the `reason` string is
/// retained as the concatenated human one-liner (ADR-064 Decision 1 /
/// TD-TRACE-1).
#[derive(Debug, Clone, serde::Serialize)]
pub struct RouteReason {
    /// Which routing axis this reason explains.
    pub axis: RouteAxis,
    /// Human-readable detail for the axis.
    pub detail: String,
}

/// A materialized read-route decision: the engine, the workload it was
/// classified as, and a human-readable reason. Emitted to telemetry and, once a
/// SELECT `EXPLAIN` surface exists, surfaced there via [`Self::explain_line`].
#[derive(Debug, Clone)]
pub struct SelectRouteDecision {
    /// Selected physical engine (canonical [`ComputeBackend`]).
    pub backend: ComputeBackend,
    /// Workload classification this route was based on.
    pub workload_profile: CatalogWorkloadProfile,
    /// Human-readable reason for the choice (the concatenation of [`Self::reasons`]
    /// details, kept for the one-line telemetry/`EXPLAIN` form).
    pub reason: String,
    /// Structured, axis-tagged reasons behind the decision (ADR-058
    /// eligibility→selection→trust). The machine-readable record the ADR-064
    /// query trace consumes; never empty for a real decision (a static Selection
    /// reason is always seeded).
    pub reasons: Vec<RouteReason>,
    /// What produced this decision (static rule vs cost-model override/explore).
    pub source: RouteSource,
}

impl SelectRouteDecision {
    /// Append a structured, axis-tagged reason (ADR-064 / TD-TRACE-1). Does not
    /// touch the human `reason` string — callers that also want the one-liner
    /// updated do that separately so the two never silently diverge.
    fn push_reason(&mut self, axis: RouteAxis, detail: impl Into<String>) {
        self.reasons.push(RouteReason {
            axis,
            detail: detail.into(),
        });
    }

    /// Stable one-line `EXPLAIN`/telemetry form, e.g.
    /// `Compute Route: Native(Volcano) (workload=Olap, reason="...")`.
    pub fn explain_line(&self) -> String {
        format!(
            "Compute Route: {} (workload={:?}, reason=\"{}\")",
            backend_label(&self.backend),
            self.workload_profile,
            self.reason
        )
    }

    /// Stable short backend label (e.g. `Native(Volcano)`, `DataFusionLocal`) for the
    /// structured `EXPLAIN` JSON surface. Single source for the engine name.
    pub fn compute_route_label(&self) -> String {
        backend_label(&self.backend)
    }

    /// Materialize this scheduler decision into the typed read-route contract.
    ///
    /// This is intentionally conservative: until split planning is wired, routes
    /// use a whole-collection split summary and leave execution behavior unchanged.
    pub fn routed_read_plan(&self) -> RoutedReadPlan {
        let mut plan = RoutedReadPlan::native_whole_collection(self.workload_profile);
        plan.backend = self.backend.clone();
        plan.authority_mode = authority_mode_for_backend(&self.backend);
        plan.policy_boundary = policy_boundary_for_backend(&self.backend);
        plan.freshness_sla = freshness_for_backend(&self.backend);
        plan.candidate_routes = vec![CandidateReadRoute {
            backend: self.backend.clone(),
            access_method: access_method_for_backend(&self.backend).to_string(),
            reason: self.reason.clone(),
        }];
        plan
    }

    /// Like [`Self::routed_read_plan`] but carries a concrete split inventory
    /// (e.g. the Parquet row groups discovered for a `DataFusionLocal` route)
    /// instead of the conservative whole-collection placeholder, so EXPLAIN
    /// discloses the real partition count the executor fans out over.
    pub fn routed_read_plan_with_splits(&self, split_summary: ReadSplitSummary) -> RoutedReadPlan {
        self.routed_read_plan().with_split_summary(split_summary)
    }
}

/// Short, stable label for a backend in EXPLAIN/telemetry output. Also the
/// canonical key the trace-driven [`crate::query::route_cost_model`] aggregates
/// observations under, so labels and cost-model keys never diverge.
pub(crate) fn backend_label(backend: &ComputeBackend) -> String {
    match backend {
        ComputeBackend::Native => "Native(Volcano)".to_string(),
        ComputeBackend::DataFusionLocal => "DataFusionLocal".to_string(),
        ComputeBackend::DataFusionDistributed => "DataFusionDistributed".to_string(),
        ComputeBackend::PolarsLocal => "PolarsLocal".to_string(),
        ComputeBackend::DuckDbCompat => "DuckDbCompat".to_string(),
        ComputeBackend::ExternalDelegated(name) => format!("ExternalDelegated({name})"),
    }
}

fn authority_mode_for_backend(backend: &ComputeBackend) -> CatalogAuthorityMode {
    match backend {
        ComputeBackend::Native => CatalogAuthorityMode::ProximaAuthoritative,
        ComputeBackend::DataFusionLocal | ComputeBackend::DataFusionDistributed => {
            CatalogAuthorityMode::ProjectionPublication
        }
        ComputeBackend::ExternalDelegated(_) => CatalogAuthorityMode::FederatedRead,
        ComputeBackend::PolarsLocal | ComputeBackend::DuckDbCompat => {
            CatalogAuthorityMode::ProximaAuthoritative
        }
    }
}

fn policy_boundary_for_backend(backend: &ComputeBackend) -> ReadPolicyBoundary {
    match backend {
        ComputeBackend::Native => ReadPolicyBoundary::EngineEnforced,
        ComputeBackend::DataFusionLocal | ComputeBackend::DataFusionDistributed => {
            ReadPolicyBoundary::ConnectorEnforced
        }
        ComputeBackend::ExternalDelegated(_) => ReadPolicyBoundary::ExternalPolicy,
        ComputeBackend::PolarsLocal | ComputeBackend::DuckDbCompat => {
            ReadPolicyBoundary::EngineEnforced
        }
    }
}

fn freshness_for_backend(backend: &ComputeBackend) -> ReadFreshnessSla {
    match backend {
        ComputeBackend::Native => ReadFreshnessSla::Synchronous,
        ComputeBackend::DataFusionLocal | ComputeBackend::DataFusionDistributed => {
            ReadFreshnessSla::CatalogValue("base-snapshot".to_string())
        }
        ComputeBackend::ExternalDelegated(_) => {
            ReadFreshnessSla::CatalogValue("external-snapshot".to_string())
        }
        ComputeBackend::PolarsLocal | ComputeBackend::DuckDbCompat => ReadFreshnessSla::Synchronous,
    }
}

fn access_method_for_backend(backend: &ComputeBackend) -> &'static str {
    match backend {
        ComputeBackend::Native => "canonical-record-scan",
        ComputeBackend::DataFusionLocal => "datafusion-local-scan",
        ComputeBackend::DataFusionDistributed => "ballista-distributed-scan",
        ComputeBackend::PolarsLocal => "polars-local-scan",
        ComputeBackend::DuckDbCompat => "duckdb-compat-scan",
        ComputeBackend::ExternalDelegated(_) => "external-delegated-scan",
    }
}

/// Policy/heuristic read-route scheduler (course correction §5.2, P1 live).
///
/// The static rule routes OLAP-shape queries over Parquet-backed tables to
/// [`ComputeBackend::DataFusionLocal`] and everything else to `Native` (the live
/// Volcano path) in exactly one place ([`Self::route_select`]).
/// [`Self::route_select_advised`] layers the trace-driven cost model on top:
/// observe-mode advisory by default, with a flag-gated live override
/// (`PROXIMADB_ROUTE_COST_OVERRIDE`, default OFF) that only flips between
/// freshness-safe candidates. The rule engine grows
/// (cardinality/partition/point-lookup, then the §5.2 Phase-2 `RLPlanner`
/// learner) without moving this seam.
#[derive(Debug, Default, Clone, Copy)]
pub struct ComputeScheduler;

impl ComputeScheduler {
    /// Construct the (stateless) scheduler.
    pub fn new() -> Self {
        Self
    }

    /// Route a relational `SELECT` plan to a physical engine.
    ///
    /// Public entry: applies the static shape rule, then records the decision
    /// metric and stamps the route onto the io_trace scope via [`finalize_route`].
    pub fn route_select(&self, shape: QueryShape) -> SelectRouteDecision {
        finalize_route(self.route_select_inner(shape), shape)
    }

    /// Select the access strategy for one vector-search operator.
    ///
    /// This shares the scheduler policy boundary and observability with SQL
    /// routing while preserving the architectural axes: the containing plan
    /// still chooses one [`ComputeBackend`] via [`Self::route_select`], and this
    /// method chooses an access method inside its vector source. With no caller
    /// intent the static route remains exact; approximate/adaptive selection is
    /// activated only by explicit typed intent until comparable route-cost
    /// cells and recall ratchets exist.
    pub fn route_vector_search(&self, shape: VectorQueryShape) -> VectorRouteDecision {
        use proximadb_search_types::search_params::SearchMode;

        let (search_mode, intent) = match shape.requested_mode {
            None => (
                SearchMode::Exact,
                "safe default: exact search preserves the SQL correctness floor".to_string(),
            ),
            Some(SearchMode::Exact) => (
                SearchMode::Exact,
                "explicit exact intent preserves the SQL correctness floor".to_string(),
            ),
            Some(mode @ SearchMode::Approximate { nprobe }) => {
                let detail = match nprobe {
                    Some(nprobe) => format!("explicit approximate intent (nprobe={nprobe})"),
                    None => "explicit approximate intent (engine-selected effort)".to_string(),
                };
                (mode, detail)
            }
            Some(mode @ SearchMode::Adaptive { threshold }) => (
                mode,
                format!("explicit adaptive intent (vector threshold={threshold})"),
            ),
        };
        let decision = VectorRouteDecision {
            backend: ComputeBackend::Native,
            workload_profile: CatalogWorkloadProfile::Vector,
            search_mode,
            reason: format!(
                "{intent}; dimensions={}, top_k={}, filter={}",
                shape.dimensions, shape.top_k, shape.has_filter
            ),
            source: RouteSource::Static,
            has_filter: shape.has_filter,
        };
        crate::metrics::route_metrics::record_decision(
            "Native(Vector)",
            &decision.shape_class(),
            decision.source.as_str(),
        );
        decision
    }

    /// The static shape rule ONLY — no metric, no io_trace stamp. The pure
    /// policy core; [`Self::route_select`] and [`Self::route_select_advised`]
    /// wrap it (the latter may override the backend before stamping, so the
    /// stamp reflects the FINAL engine — see [`finalize_route`]).
    fn route_select_inner(&self, shape: QueryShape) -> SelectRouteDecision {
        let mut decision = match (shape.engages_relational, shape.parquet_backed) {
            // P1: OLAP shape over Parquet-backed (object-store) table(s) → DataFusion.
            (true, true) => SelectRouteDecision {
                backend: ComputeBackend::DataFusionLocal,
                workload_profile: CatalogWorkloadProfile::Olap,
                reason: "OLAP shape over Parquet-backed table(s) — DataFusion over object storage"
                    .to_string(),
                reasons: Vec::new(),
                source: RouteSource::Static,
            },
            // OLAP shape on native storage — Volcano serves it from WAL+RecordStorage
            // until the relational base tier is Parquet/Iceberg (course-correction §6 P3).
            (true, false) => {
                // TD-USUB-9 (route-stamp integrity): there used to be an arm here
                // stamping `DataFusionLocal` when `pax_backed && pax_reader_enabled()`.
                // It was REMOVED because it stamped a route that never ran:
                // dispatch requires `parquet_backed`, so Volcano served the query
                // while the cost model recorded Volcano's measured latency and I/O
                // against DataFusion cells — poisoning exactly the evidence ADR-052's
                // observe→ingest→act sequence depends on, and doing so silently.
                //
                // Removing it changes no execution (Volcano served before and after);
                // it only makes the stamp true. The PAX reader itself stays gated and
                // available for the real slice-2 dispatch wiring, which will add an
                // arm here once DataFusion genuinely executes this shape — the bar the
                // `native_registerable` arm below already meets.
                if shape.native_registerable {
                    // TD-USUB-2: the tables are registerable with DataFusion directly,
                    // so the full ANSI surface (CTE / window / join / subquery) is
                    // reachable without a manual MATERIALIZE. DataFusion genuinely
                    // executes here, so this stamp is accurate.
                    SelectRouteDecision {
                        backend: ComputeBackend::DataFusionLocal,
                        workload_profile: CatalogWorkloadProfile::Olap,
                        reason:
                            "OLAP shape on native relational table(s) — DataFusion over registered rows"
                                .to_string(),
                        reasons: Vec::new(),
                        source: RouteSource::Static,
                    }
                } else {
                    SelectRouteDecision {
                        backend: ComputeBackend::Native,
                        workload_profile: CatalogWorkloadProfile::Olap,
                        reason:
                            "OLAP shape (join/group-by/aggregate/set-op) on native storage — Volcano"
                                .to_string(),
                        reasons: Vec::new(),
                        source: RouteSource::Static,
                    }
                }
            }
            (false, _) => SelectRouteDecision {
                backend: ComputeBackend::Native,
                workload_profile: CatalogWorkloadProfile::Oltp,
                reason: "OLTP shape (point/simple select) — Volcano".to_string(),
                reasons: Vec::new(),
                source: RouteSource::Static,
            },
        };
        seed_static_route_reasons(&mut decision, shape);
        decision
    }

    /// Route a relational `SELECT` and materialize the typed read-route plan.
    pub fn route_select_plan(&self, shape: QueryShape) -> RoutedReadPlan {
        self.route_select(shape).routed_read_plan()
    }

    /// Route a `SELECT`, then — if a trace-driven cost model is supplied —
    /// consult its frozen recommendation table (a lock-free `ArcSwap` load +
    /// `HashMap` get, never the learn-path mutex) and either disclose an advisory
    /// (observe-mode) or flip to a freshness-safe challenger (live override,
    /// flag-gated via `PROXIMADB_ROUTE_COST_OVERRIDE`). With override OFF (the
    /// default) or no warmed history, the static decision is returned unchanged.
    ///
    /// This closes the co-design loop (C0 trace → cost model → router) while
    /// keeping the decision lock-free and O(1): freshness-safety is baked into
    /// the frozen table (only `olap/parquet` offers two safe engines), so the
    /// consult can never flip a query onto an engine that would serve it
    /// incorrectly (e.g. a point/OLTP query onto a stale base snapshot).
    pub fn route_select_advised(
        &self,
        shape: QueryShape,
        model: Option<&crate::query::route_cost_model::RouteCostModel>,
        flags: RouteFlags,
    ) -> SelectRouteDecision {
        use crate::query::route_cost_model::RouteConsult;

        let mut decision = self.route_select_inner(shape);

        // TD-ROUTE-3: env master-switch force takes precedence over the cost
        // model — this mirrors the former dispatch order, where the native /
        // DuckDB primary was attempted under `native_route_enabled() || advised`
        // BEFORE the DataFusion floor. Folding it here makes `decision.backend`
        // the engine that will actually be attempted, so dispatch is a thin
        // `match` with no re-decision. The native/duckdb eligible shapes are
        // disjoint, so at most one branch fires. Default-OFF: both flags false →
        // falls straight through to the static + cost-model path unchanged. The
        // adapter still owns the primary→floor fallback, so a forced primary that
        // declines is still served correctly by DataFusion.
        if shape.parquet_backed {
            let (native_ok, duckdb_ok) = parquet_primary_eligibility(&shape);
            if flags.native_route && native_ok {
                let prev = backend_label(&decision.backend);
                decision.backend = ComputeBackend::Native;
                decision.reason = format!(
                    "{} | PROXIMADB_NATIVE_ROUTE forces native-over-parquet PRIMARY \
                     (op eligible; {prev} is the floor)",
                    decision.reason
                );
                decision.push_reason(
                    RouteAxis::Selection,
                    "PROXIMADB_NATIVE_ROUTE forces native-over-parquet PRIMARY (op eligible)",
                );
                return finalize_route(decision, shape);
            }
            if flags.duckdb_route && duckdb_ok {
                let prev = backend_label(&decision.backend);
                decision.backend = ComputeBackend::DuckDbCompat;
                decision.reason = format!(
                    "{} | PROXIMADB_DUCKDB_ROUTE forces duckdb-local PRIMARY \
                     (shape eligible; {prev} is the floor)",
                    decision.reason
                );
                decision.push_reason(
                    RouteAxis::Selection,
                    "PROXIMADB_DUCKDB_ROUTE forces duckdb-local PRIMARY (shape eligible)",
                );
                return finalize_route(decision, shape);
            }
        }

        let Some(model) = model else {
            return finalize_route(decision, shape);
        };
        let class = crate::query::route_cost_model::shape_class(&shape);

        // Lock-free hierarchical consult: the refined class first, then each
        // coarser ancestor — so a key-refinement tier (geom/op/card) never
        // orphans warmed cells (ADR-058 D3 density across generations). `None`
        // at EVERY level → keep the static route unchanged, and count the miss
        // (a hot class that keeps missing after warm-up means the key
        // fragmented faster than cells warm).
        let consulted = model.consult_with_fallback(&class);
        if consulted.is_none() {
            crate::metrics::route_metrics::record_consult_miss(&class);
        }
        let (consult, consult_class) = match consulted {
            Some((rec, level)) => (Some(rec), Some(level)),
            None => (None, None),
        };
        // Disclose when an ancestor's cells served the decision (refined cell
        // still cold) — visible in EXPLAIN/telemetry, zero behavior change.
        let cells_note = match consult_class.as_deref() {
            Some(level) if level != class => format!(" [cells: {level}]"),
            _ => String::new(),
        };

        // TD-ROUTE-1 + TD-ROUTE-3 capability gate (ADR-058: eligibility precedes
        // selection). A cost-model flip may only target a backend that will
        // actually serve the shape as PRIMARY — otherwise the flip stamps that
        // engine while the DataFusion floor serves, poisoning the cost cells (the
        // "last-write-wins" mis-attribution TD-ROUTE-3 removes). The eligible
        // shapes are the SAME sets the former dispatch gates checked
        // (`native_eligible_op` / `duckdb_eligible_shape`), now the single source
        // of truth via [`parquet_primary_eligibility`]:
        //   - Native over Parquet: only the ungrouped scalar-aggregate ops the
        //     native-over-parquet adapter can serve, and never a join-bearing
        //     plan (no multi-table native scan; the single-scan vectorized path).
        //   - DuckDB-local: only the join/grouped shapes it is the measured win
        //     for.
        // Other backends carry no extra primary-eligibility gate here. Static
        // routes are untouched; override OFF (default) = no change.
        // `native_flip_ok` already excludes join-bearing plans (see
        // `parquet_primary_eligibility`), subsuming the former standalone
        // `!join_bearing` Native guard.
        let (native_flip_ok, duckdb_flip_ok) = parquet_primary_eligibility(&shape);
        let flip_eligible = |b: &ComputeBackend| match b {
            ComputeBackend::Native => native_flip_ok,
            ComputeBackend::DuckDbCompat => duckdb_flip_ok,
            _ => true,
        };

        // Per-class staged go-live (TD-EXEC-2 §3): the scope gate reads the
        // DECISION class (the full refined key), so enabling an ancestor prefix
        // enables all of its refinements — consistent with the hierarchical
        // consult, even when an ancestor's cells served the recommendation.
        if model.override_active_for(&class)
            && let Some(consult) = consult.as_ref()
        {
            // Exploration (warm-up) first: rate-limited flip to the baked
            // least-sampled freshness-safe candidate so it accrues history the
            // exploit override can later act on. Bounded + freshness-safe.
            if let Some(target) = consult
                .explore_target()
                // Staleness guard: the frozen target may have warmed since the
                // last recompute; re-check against the frozen ranked samples (a
                // target absent from `ranked` was never observed → still cold).
                .filter(|t| {
                    flip_eligible(t)
                        && backend_label(t) != backend_label(&decision.backend)
                        && consult
                            .find(t)
                            .is_none_or(|c| c.samples < model.min_samples())
                })
                && model
                    .next_exploration_tick()
                    .is_multiple_of(model.exploration_interval())
            {
                let prev = backend_label(&decision.backend);
                decision.reason = format!(
                    "{} | cost-model EXPLORE {prev}→{} (gathering cost history){cells_note}",
                    decision.reason,
                    backend_label(target)
                );
                decision.push_reason(
                    RouteAxis::Selection,
                    format!(
                        "cost-model EXPLORE {prev}→{} (gathering cost history){cells_note}",
                        backend_label(target)
                    ),
                );
                decision.backend = target.clone();
                decision.source = RouteSource::OverrideExplore;
                return finalize_route(decision, shape);
            }

            // Exploit override — TD-170 hard round-trip gate, then the soft
            // min_advantage byte-cost gate — both resolved from frozen data.
            if let Some(static_choice) = consult.find(&decision.backend)
                && static_choice.samples >= model.min_samples()
            {
                // Hard gate: a backend over the round-trip budget loses even
                // when it moves fewer bytes (latency = depth × RTT). Flip to the
                // cheapest freshness-safe candidate WITHIN the budget.
                if let Some(budget) = model.rtt_budget()
                    && static_choice.get_ops > budget
                    && let Some(within) = consult
                        .ranked()
                        .iter()
                        .filter(|c| c.get_ops <= budget && flip_eligible(&c.backend))
                        .min_by(|a, b| {
                            a.score
                                .partial_cmp(&b.score)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .filter(|c| backend_label(&c.backend) != backend_label(&decision.backend))
                {
                    let prev = backend_label(&decision.backend);
                    decision.reason = format!(
                        "{} | cost-model OVERRIDE {prev}→{} (RTT budget {budget:.0} exceeded){cells_note}",
                        decision.reason,
                        backend_label(&within.backend)
                    );
                    decision.push_reason(
                        RouteAxis::Selection,
                        format!(
                            "cost-model OVERRIDE {prev}→{} (RTT budget {budget:.0} exceeded){cells_note}",
                            backend_label(&within.backend)
                        ),
                    );
                    decision.backend = within.backend.clone();
                    decision.source = RouteSource::OverrideExploit;
                    return finalize_route(decision, shape);
                }

                // Soft gate: flip to the cheapest warmed candidate if it beats
                // the static route by at least `min_advantage`.
                if let Some(challenger) = consult
                    .ranked()
                    .iter()
                    .filter(|c| c.samples >= model.min_samples() && flip_eligible(&c.backend))
                    .find(|c| backend_label(&c.backend) != backend_label(&decision.backend))
                    .filter(|c| c.score < static_choice.score * (1.0 - model.min_advantage()))
                {
                    let prev = backend_label(&decision.backend);
                    decision.reason = format!(
                        "{} | cost-model OVERRIDE {prev}→{} (score {:.1} vs {:.1}){cells_note}",
                        decision.reason,
                        backend_label(&challenger.backend),
                        challenger.score,
                        static_choice.score
                    );
                    decision.push_reason(
                        RouteAxis::Selection,
                        format!(
                            "cost-model OVERRIDE {prev}→{} (score {:.1} vs {:.1}){cells_note}",
                            backend_label(&challenger.backend),
                            challenger.score,
                            static_choice.score
                        ),
                    );
                    decision.backend = challenger.backend.clone();
                    decision.source = RouteSource::OverrideExploit;
                    return finalize_route(decision, shape);
                }
            }

            // ADR-058 observability: disclose when *eligibility* removed Native
            // from consideration — the warmed-cheapest candidate is Native but
            // the plan is join-bearing. Phrased as ineligibility (not "a flip
            // was suppressed"): the cost gates (min_samples/min_advantage/RTT)
            // are not re-checked here, so a flip is not necessarily what the
            // exploit path would have chosen.
            if shape.join_bearing
                && let Some(best) = consult
                    .ranked()
                    .iter()
                    .find(|c| c.samples >= model.min_samples())
                && matches!(best.backend, ComputeBackend::Native)
                && backend_label(&best.backend) != backend_label(&decision.backend)
            {
                decision.reason = format!(
                    "{} | cost-model: Native suppressed as flip target — join-bearing plan is \
                     ineligible (no multi-table native-over-parquet scan; TD-ROUTE-1){cells_note}",
                    decision.reason
                );
                decision.push_reason(
                    RouteAxis::Eligibility,
                    format!(
                        "Native suppressed as flip target — join-bearing plan is ineligible \
                         (no multi-table native-over-parquet scan; TD-ROUTE-1){cells_note}"
                    ),
                );
                return finalize_route(decision, shape);
            }
        }

        // Observe-mode advisory (no behavior change): disclose the cheapest
        // warmed freshness-safe candidate from the frozen table.
        if let Some(consult) = consult.as_ref()
            && let Some(rec) = consult
                .ranked()
                .iter()
                .find(|c| c.samples >= model.min_samples())
        {
            let advisory = if backend_label(&rec.backend) == backend_label(&decision.backend) {
                format!("cost-model concurs (score {:.1})", rec.score)
            } else {
                format!(
                    "cost-model would prefer {} (score {:.1}) — observe-mode, route unchanged",
                    backend_label(&rec.backend),
                    rec.score
                )
            };
            decision.reason = format!("{} | {advisory}{cells_note}", decision.reason);
        }

        finalize_route(decision, shape)
    }
}

/// Seed the structured, axis-tagged reasons (ADR-058 eligibility→selection→trust)
/// on a decision produced by the static shape rule (ADR-064 / TD-TRACE-1). Every
/// decision records at least one `Eligibility` + one `Selection` reason; the
/// Parquet/PAX (external, base-snapshot) route additionally records a `Trust`
/// reason. The cost-model advised path appends further reasons when it overrides.
fn seed_static_route_reasons(decision: &mut SelectRouteDecision, shape: QueryShape) {
    // Eligibility — why the chosen backend can serve this shape SOUNDLY.
    let eligibility = match &decision.backend {
        ComputeBackend::DataFusionLocal if shape.parquet_backed => {
            "DataFusionLocal eligible: OLAP shape over Parquet-backed object-store table(s)"
        }
        ComputeBackend::DataFusionLocal => {
            "DataFusionLocal eligible: OLAP shape over PAX-backed segment(s) via PaxSplitReader"
        }
        ComputeBackend::Native if shape.engages_relational => {
            "Native(Volcano) eligible: OLAP shape on native storage (WAL+RecordStorage; \
             no external scan required)"
        }
        ComputeBackend::Native => "Native(Volcano) eligible: point/OLTP shape",
        _ => "route eligible for this shape (ADR-058 soundness gate)",
    };
    decision.push_reason(RouteAxis::Eligibility, eligibility);

    // Selection — the static rule's cost rationale (the human one-liner).
    let selection = decision.reason.clone();
    decision.push_reason(RouteAxis::Selection, selection);

    // Trust — the external Parquet/PAX route trusts the catalog base-snapshot
    // statistics (ADR-058 D5); native storage is synchronous with no trust gate.
    // TD-USUB-9: the PAX clause was dropped with the route arm above. A
    // PAX-backed table that is not parquet-backed is served by Volcano from
    // WAL+RecordStorage, which is synchronous and has no trust gate — claiming it
    // trusts object-store base-snapshot statistics described a route that never ran.
    if shape.parquet_backed {
        decision.push_reason(
            RouteAxis::Trust,
            "trusts object-store catalog base-snapshot statistics (ADR-058 D5)",
        );
    }
}

/// Record the decision metric (and the OLAP-on-Native nudge), then stamp the
/// FINAL route onto the active io_trace scope. Called once per route decision,
/// AFTER any override has settled the backend — so the learn loop attributes a
/// query's measured cost to the engine that actually served it (fixing the prior
/// bug where the static backend was stamped before an override could flip it).
/// Stamping is a no-op when no io_trace scope is active (unit tests / EXPLAIN).
fn finalize_route(decision: SelectRouteDecision, shape: QueryShape) -> SelectRouteDecision {
    let class = crate::query::route_cost_model::shape_class(&shape);
    let backend = backend_label(&decision.backend);
    crate::metrics::route_metrics::record_decision(&backend, &class, decision.source.as_str());
    if decision.backend == ComputeBackend::Native && class.starts_with("olap/") {
        crate::metrics::route_metrics::record_olap_on_native(&class);
    }
    crate::observability::io_trace::record_route(&class, &backend);
    decision
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_route_preserves_safe_default_and_explicit_search_intent() {
        use proximadb_search_types::search_params::SearchMode;

        let scheduler = ComputeScheduler::new();
        let default_route = scheduler.route_vector_search(VectorQueryShape {
            dimensions: 384,
            top_k: 10,
            has_filter: false,
            requested_mode: None,
        });
        assert_eq!(default_route.backend, ComputeBackend::Native);
        assert_eq!(
            default_route.workload_profile,
            CatalogWorkloadProfile::Vector
        );
        assert_eq!(default_route.search_mode, SearchMode::Exact);
        assert_eq!(default_route.strategy_label(), "exact");
        assert!(default_route.reason.contains("safe default"));

        let explicit_exact_route = scheduler.route_vector_search(VectorQueryShape {
            dimensions: 384,
            top_k: 10,
            has_filter: false,
            requested_mode: Some(SearchMode::Exact),
        });
        assert_eq!(explicit_exact_route.search_mode, SearchMode::Exact);
        assert_eq!(explicit_exact_route.strategy_label(), "exact");
        assert!(
            explicit_exact_route
                .reason
                .contains("explicit exact intent")
        );

        let approximate = SearchMode::Approximate { nprobe: Some(12) };
        let approximate_route = scheduler.route_vector_search(VectorQueryShape {
            dimensions: 1_536,
            top_k: 50,
            has_filter: true,
            requested_mode: Some(approximate.clone()),
        });
        assert_eq!(approximate_route.search_mode, approximate);
        assert_eq!(approximate_route.strategy_label(), "approximate");
        assert!(approximate_route.reason.contains("nprobe=12"));
        assert_eq!(
            approximate_route.shape_class(),
            "vector/native/mode=approximate/filter=yes"
        );
    }

    #[test]
    fn vector_access_routing_does_not_change_relational_backend_routing() {
        let scheduler = ComputeScheduler::new();
        let before = scheduler.route_select(QueryShape {
            engages_relational: true,
            parquet_backed: true,
            ..Default::default()
        });

        let _ = scheduler.route_vector_search(VectorQueryShape {
            dimensions: 384,
            top_k: 10,
            has_filter: false,
            requested_mode: None,
        });

        let after = scheduler.route_select(QueryShape {
            engages_relational: true,
            parquet_backed: true,
            ..Default::default()
        });
        assert_eq!(before.backend, ComputeBackend::DataFusionLocal);
        assert_eq!(after.backend, before.backend);
        assert_eq!(after.workload_profile, before.workload_profile);
    }

    #[test]
    fn olap_shape_on_native_storage_stays_on_volcano() {
        let decision = ComputeScheduler::new().route_select(QueryShape {
            engages_relational: true,
            parquet_backed: false,
            ..Default::default()
        });
        // OLAP over native storage stays on Volcano (no Parquet base tier yet).
        assert_eq!(decision.backend, ComputeBackend::Native);
        assert_eq!(decision.workload_profile, CatalogWorkloadProfile::Olap);
        assert!(decision.reason.to_lowercase().contains("olap"));
    }

    #[test]
    fn oltp_shape_classifies_oltp_on_native() {
        let decision = ComputeScheduler::new().route_select(QueryShape {
            engages_relational: false,
            parquet_backed: false,
            ..Default::default()
        });
        assert_eq!(decision.backend, ComputeBackend::Native);
        assert_eq!(decision.workload_profile, CatalogWorkloadProfile::Oltp);
    }

    #[test]
    fn p1_olap_over_parquet_routes_to_datafusion() {
        let d = ComputeScheduler::new().route_select(QueryShape {
            engages_relational: true,
            parquet_backed: true,
            ..Default::default()
        });
        assert_eq!(d.backend, ComputeBackend::DataFusionLocal);
        assert_eq!(d.workload_profile, CatalogWorkloadProfile::Olap);
        assert_eq!(d.compute_route_label(), "DataFusionLocal");
    }

    // ---- TD-ROUTE-3: authoritative decision (shape + RouteFlags → backend) ----

    fn parquet_shape(op: OperationClass, join_bearing: bool) -> QueryShape {
        QueryShape {
            engages_relational: true,
            parquet_backed: true,
            operation_class: op,
            join_bearing,
            ..Default::default()
        }
    }

    #[test]
    fn native_eligible_op_with_native_flag_decides_native() {
        // ScalarAggregate over parquet + PROXIMADB_NATIVE_ROUTE → the decision is
        // authoritative Native (the engine dispatch will attempt as primary), no
        // downstream re-check. `model: None` isolates the env-force fold.
        let d = ComputeScheduler::new().route_select_advised(
            parquet_shape(OperationClass::ScalarAggregate, false),
            None,
            RouteFlags {
                native_route: true,
                duckdb_route: false,
            },
        );
        assert_eq!(d.backend, ComputeBackend::Native);
    }

    #[test]
    fn native_eligible_op_without_flag_or_override_stays_datafusion() {
        // Same shape, flags off, no cost model → the static parquet route
        // (DataFusion floor). Native is NOT engaged just because the op is eligible.
        let d = ComputeScheduler::new().route_select_advised(
            parquet_shape(OperationClass::ScalarAggregate, false),
            None,
            RouteFlags::default(),
        );
        assert_eq!(d.backend, ComputeBackend::DataFusionLocal);
    }

    #[test]
    fn join_bearing_with_duckdb_flag_decides_duckdb() {
        let d = ComputeScheduler::new().route_select_advised(
            parquet_shape(OperationClass::Other, true),
            None,
            RouteFlags {
                native_route: false,
                duckdb_route: true,
            },
        );
        assert_eq!(d.backend, ComputeBackend::DuckDbCompat);
    }

    #[test]
    fn join_bearing_with_native_flag_stays_datafusion_native_ineligible() {
        // A join-bearing plan is NOT native-eligible (no multi-table native scan),
        // so the native master switch cannot force it onto Native — the decision
        // keeps the DataFusion floor. This is the core attribution invariant.
        let d = ComputeScheduler::new().route_select_advised(
            parquet_shape(OperationClass::Other, true),
            None,
            RouteFlags {
                native_route: true,
                duckdb_route: false,
            },
        );
        assert_eq!(d.backend, ComputeBackend::DataFusionLocal);
    }

    #[test]
    fn grouped_without_flags_stays_datafusion() {
        let d = ComputeScheduler::new().route_select_advised(
            parquet_shape(OperationClass::Grouped, false),
            None,
            RouteFlags::default(),
        );
        assert_eq!(d.backend, ComputeBackend::DataFusionLocal);
    }

    #[test]
    fn route_flags_inert_on_non_parquet_shapes() {
        // The env master switches gate only the parquet-OLAP primary arm. An
        // OLAP-on-native shape stays Volcano/Native regardless of the flags.
        let d = ComputeScheduler::new().route_select_advised(
            QueryShape {
                engages_relational: true,
                parquet_backed: false,
                operation_class: OperationClass::ScalarAggregate,
                ..Default::default()
            },
            None,
            RouteFlags {
                native_route: true,
                duckdb_route: true,
            },
        );
        // (true, false) → native storage OLAP → Volcano. Unchanged by flags.
        assert_eq!(d.backend, ComputeBackend::Native);
        assert_eq!(d.workload_profile, CatalogWorkloadProfile::Olap);
    }

    #[test]
    fn native_and_duckdb_primary_eligibility_are_disjoint() {
        // At most one parquet primary is ever eligible for a given query, so the
        // env-force precedence (native-first) is unambiguous and the dispatch
        // `match` arms are mutually exclusive. Guards against a future
        // OperationClass change silently making both fire.
        use OperationClass::*;
        for op in [
            Unknown,
            MetadataElidable,
            ScalarAggregate,
            Grouped,
            StringHeavy,
            Other,
        ] {
            for join_bearing in [false, true] {
                let (native_ok, duckdb_ok) =
                    parquet_primary_eligibility(&parquet_shape(op, join_bearing));
                assert!(
                    !(native_ok && duckdb_ok),
                    "native/duckdb eligibility overlapped for op={op:?} join_bearing={join_bearing}"
                );
            }
        }
    }

    #[test]
    fn decision_records_structured_eligibility_selection_trust_reasons() {
        // ADR-064 / TD-TRACE-1: a route decision records structured, axis-tagged
        // reasons (not just a flat string). The Parquet route exercises all three
        // ADR-058 axes: eligibility (soundness), selection (cost), trust (stats).
        let d = ComputeScheduler::new().route_select(QueryShape {
            engages_relational: true,
            parquet_backed: true,
            ..Default::default()
        });
        let axes: Vec<RouteAxis> = d.reasons.iter().map(|r| r.axis).collect();
        assert!(
            axes.contains(&RouteAxis::Eligibility),
            "missing Eligibility reason: {:?}",
            d.reasons
        );
        assert!(axes.contains(&RouteAxis::Selection));
        assert!(
            axes.contains(&RouteAxis::Trust),
            "Parquet route must record a stats-trust reason (ADR-058 D5)"
        );
        // The Selection reason mirrors the human one-liner (never diverges).
        assert!(
            d.reasons
                .iter()
                .any(|r| r.axis == RouteAxis::Selection && r.detail == d.reason),
        );

        // Native-storage OLAP: eligible + selected, but NO trust gate (synchronous
        // native storage, not an external base-snapshot).
        let n = ComputeScheduler::new().route_select(QueryShape {
            engages_relational: true,
            parquet_backed: false,
            ..Default::default()
        });
        assert_eq!(n.backend, ComputeBackend::Native);
        assert!(n.reasons.iter().any(|r| r.axis == RouteAxis::Eligibility));
        assert!(
            !n.reasons.iter().any(|r| r.axis == RouteAxis::Trust),
            "native storage has no external stats-trust gate: {:?}",
            n.reasons
        );
    }

    #[test]
    fn oltp_never_routes_off_native_even_if_parquet() {
        // Point/simple selects stay on Volcano (strong freshness) regardless of format.
        let d = ComputeScheduler::new().route_select(QueryShape {
            engages_relational: false,
            parquet_backed: true,
            ..Default::default()
        });
        assert_eq!(d.backend, ComputeBackend::Native);
    }

    #[test]
    fn datafusion_route_carries_concrete_row_group_split_inventory() {
        let plan = ComputeScheduler::new()
            .route_select(QueryShape {
                engages_relational: true,
                parquet_backed: true,
                ..Default::default()
            })
            .routed_read_plan_with_splits(ReadSplitSummary::row_groups(
                8,
                Some(50_000),
                Some(1 << 20),
            ));

        assert_eq!(plan.backend, ComputeBackend::DataFusionLocal);
        let explain = plan.route_explanation();
        assert_eq!(explain.selected_backend, "DataFusionLocal");
        // Real row-group inventory, not the whole-collection `1`-partition default.
        assert_eq!(explain.split_strategy, "row_group");
        assert_eq!(explain.partition_count, 8);
        assert_eq!(explain.estimated_rows, Some(50_000));
    }

    #[test]
    fn explain_line_is_stable_and_readable() {
        let line = ComputeScheduler::new()
            .route_select(QueryShape {
                engages_relational: true,
                parquet_backed: false,
                ..Default::default()
            })
            .explain_line();
        assert!(line.starts_with("Compute Route: Native(Volcano) (workload=Olap"));
    }

    fn io_snap(get_ops: u64, bytes_read: u64) -> crate::observability::io_trace::IoTraceSnapshot {
        crate::observability::io_trace::IoTraceSnapshot {
            get_ops,
            bytes_read,
            ..Default::default()
        }
    }

    #[test]
    fn advised_without_model_is_identical_to_static() {
        let shape = QueryShape {
            engages_relational: true,
            parquet_backed: false,
            ..Default::default()
        };
        let s = ComputeScheduler::new();
        let plain = s.route_select(shape);
        let advised = s.route_select_advised(shape, None, RouteFlags::default());
        assert_eq!(advised.backend, plain.backend);
        assert_eq!(advised.reason, plain.reason);
    }

    #[test]
    fn advised_is_observe_mode_never_overrides_backend() {
        use crate::query::route_cost_model::RouteCostModel;
        // OLAP-over-Parquet: static rule => DataFusionLocal. Warm Native as far
        // cheaper (few coalesced GETs vs DataFusion's many) so observe-mode
        // DISCLOSES the preference without flipping the backend. (The only class
        // with two freshness-safe engines — the only one an advisory can prefer.)
        let shape = QueryShape {
            engages_relational: true,
            parquet_backed: true,
            ..Default::default()
        };
        let model = RouteCostModel::new()
            .with_min_samples(1)
            .with_recompute_every(1);
        for _ in 0..4 {
            model.observe(
                "olap/parquet",
                &ComputeBackend::Native,
                &io_snap(3, 16 << 20),
            );
            model.observe(
                "olap/parquet",
                &ComputeBackend::DataFusionLocal,
                &io_snap(300, 16 << 20),
            );
        }
        let advised = ComputeScheduler::new().route_select_advised(
            shape,
            Some(&model),
            RouteFlags::default(),
        );
        // Backend is UNCHANGED (observe-mode) ...
        assert_eq!(advised.backend, ComputeBackend::DataFusionLocal);
        // ... but the divergence is disclosed for EXPLAIN/validation.
        assert!(advised.reason.contains("would prefer Native"));
        assert!(advised.reason.contains("observe-mode"));
    }

    #[test]
    fn advised_notes_when_cost_model_concurs() {
        use crate::query::route_cost_model::RouteCostModel;
        let shape = QueryShape {
            engages_relational: false,
            parquet_backed: false,
            ..Default::default()
        };
        let model = RouteCostModel::new()
            .with_min_samples(1)
            .with_recompute_every(1);
        // Only Native has history for this shape-class → it is the min, concurring
        // with the static OLTP→Native rule.
        for _ in 0..3 {
            model.observe("oltp/native", &ComputeBackend::Native, &io_snap(2, 8192));
        }
        let advised = ComputeScheduler::new().route_select_advised(
            shape,
            Some(&model),
            RouteFlags::default(),
        );
        assert_eq!(advised.backend, ComputeBackend::Native);
        assert!(advised.reason.contains("cost-model concurs"));
    }

    #[test]
    fn override_off_keeps_static_backend_even_when_model_disagrees() {
        use crate::query::route_cost_model::RouteCostModel;
        let shape = QueryShape {
            engages_relational: true,
            parquet_backed: true,
            ..Default::default()
        }; // static => DataFusionLocal
        let model = RouteCostModel::new()
            .with_min_samples(1)
            .with_recompute_every(1); // override OFF (default)
        for _ in 0..5 {
            model.observe(
                "olap/parquet",
                &ComputeBackend::Native,
                &io_snap(3, 16 << 20),
            );
            model.observe(
                "olap/parquet",
                &ComputeBackend::DataFusionLocal,
                &io_snap(300, 16 << 20),
            );
        }
        let d = ComputeScheduler::new().route_select_advised(
            shape,
            Some(&model),
            RouteFlags::default(),
        );
        // Native is far cheaper, but override is off → static DataFusion holds.
        assert_eq!(d.backend, ComputeBackend::DataFusionLocal);
        assert!(d.reason.contains("would prefer Native") && d.reason.contains("observe-mode"));
    }

    #[test]
    fn override_on_flips_olap_parquet_to_the_cheaper_backend() {
        use crate::query::route_cost_model::RouteCostModel;
        // TD-ROUTE-3: a Native flip is only eligible for a native-servable shape
        // (ungrouped scalar aggregate) — an Unknown-op shape can never be served by
        // the native-over-parquet adapter, so flipping the DECISION onto Native
        // there would mis-attribute (the floor serves). The refined `op=agg` class
        // consults the warmed `olap/parquet` ancestor cells via hierarchical
        // fallback.
        let shape = QueryShape {
            engages_relational: true,
            parquet_backed: true,
            operation_class: OperationClass::ScalarAggregate,
            ..Default::default()
        }; // static => DataFusionLocal
        let model = RouteCostModel::new()
            .with_min_samples(1)
            .with_recompute_every(1);
        model.set_override_enabled(true);
        for _ in 0..5 {
            model.observe(
                "olap/parquet",
                &ComputeBackend::Native,
                &io_snap(3, 16 << 20),
            );
            model.observe(
                "olap/parquet",
                &ComputeBackend::DataFusionLocal,
                &io_snap(300, 16 << 20),
            );
        }
        let d = ComputeScheduler::new().route_select_advised(
            shape,
            Some(&model),
            RouteFlags::default(),
        );
        // Override fires: the route is flipped to the measured-cheaper engine.
        assert_eq!(d.backend, ComputeBackend::Native);
        assert!(d.reason.contains("OVERRIDE"));
    }

    #[test]
    fn class_scoped_override_gates_on_the_decision_class() {
        use crate::query::compute_scheduler::{GeometryClass, OperationClass};
        use crate::query::route_cost_model::{OverrideScope, RouteCostModel, shape_class};
        // TD-EXEC-2 per-class staged go-live at the router: two shapes whose
        // refined classes differ only in operation class; the scope enables one
        // prefix. Warm identical Native-cheaper evidence under BOTH refined
        // classes — the scoped-in shape flips, the scoped-out shape stays
        // static (observe-mode advisory only).
        let scoped_in = QueryShape {
            engages_relational: true,
            parquet_backed: true,
            operation_class: OperationClass::ScalarAggregate,
            geometry: GeometryClass::from_estimate(3, 1),
            ..Default::default()
        }; // static => DataFusionLocal
        let scoped_out = QueryShape {
            operation_class: OperationClass::Grouped,
            ..scoped_in
        };
        let model = RouteCostModel::new()
            .with_min_samples(1)
            .with_recompute_every(1);
        for shape in [scoped_in, scoped_out] {
            let class = shape_class(&shape);
            for _ in 0..5 {
                model.observe(&class, &ComputeBackend::Native, &io_snap(3, 16 << 20));
                model.observe(
                    &class,
                    &ComputeBackend::DataFusionLocal,
                    &io_snap(300, 16 << 20),
                );
            }
        }
        model.set_override_scope(OverrideScope::Classes(vec![
            "olap/parquet/op=agg".to_string(),
        ]));
        let d = ComputeScheduler::new().route_select_advised(
            scoped_in,
            Some(&model),
            RouteFlags::default(),
        );
        assert_eq!(d.backend, ComputeBackend::Native);
        assert!(d.reason.contains("OVERRIDE"), "got: {}", d.reason);
        let d = ComputeScheduler::new().route_select_advised(
            scoped_out,
            Some(&model),
            RouteFlags::default(),
        );
        assert_eq!(
            d.backend,
            ComputeBackend::DataFusionLocal,
            "a class outside the override scope must keep its static route"
        );
        assert!(
            !d.reason.contains("OVERRIDE") && !d.reason.contains("EXPLORE"),
            "scoped-out class must stay observe-only, got: {}",
            d.reason
        );
    }

    #[test]
    fn override_never_flips_join_bearing_plan_to_native() {
        use crate::query::route_cost_model::RouteCostModel;
        // Same warmed model as the flip test — Native measured far cheaper —
        // but the plan carries a JOIN, which the Native arm cannot serve over
        // Parquet (TD-ROUTE-1: no multi-table native-over-parquet scan).
        let shape = QueryShape {
            engages_relational: true,
            parquet_backed: true,
            join_bearing: true,
            ..Default::default()
        }; // static => DataFusionLocal
        let model = RouteCostModel::new()
            .with_min_samples(1)
            .with_recompute_every(1);
        model.set_override_enabled(true);
        for _ in 0..5 {
            model.observe(
                "olap/parquet",
                &ComputeBackend::Native,
                &io_snap(3, 16 << 20),
            );
            model.observe(
                "olap/parquet",
                &ComputeBackend::DataFusionLocal,
                &io_snap(300, 16 << 20),
            );
        }
        let d = ComputeScheduler::new().route_select_advised(
            shape,
            Some(&model),
            RouteFlags::default(),
        );
        assert_eq!(
            d.backend,
            ComputeBackend::DataFusionLocal,
            "a join-bearing plan must never be flipped onto Native (no multi-table native-over-parquet scan)"
        );
        assert!(
            d.reason.contains("suppressed"),
            "eligibility veto must be disclosed, got: {}",
            d.reason
        );
    }

    #[test]
    fn exploration_skips_native_for_join_bearing_plan() {
        use crate::query::route_cost_model::RouteCostModel;
        let shape = QueryShape {
            engages_relational: true,
            parquet_backed: true,
            join_bearing: true,
            ..Default::default()
        }; // static => DataFusionLocal
        let model = RouteCostModel::new()
            .with_min_samples(1)
            .with_exploration_interval(1)
            .with_recompute_every(1);
        model.set_override_enabled(true);
        // DataFusion warm, Native cold — exploration would normally pick Native.
        for _ in 0..5 {
            model.observe(
                "olap/parquet",
                &ComputeBackend::DataFusionLocal,
                &io_snap(4, 8192),
            );
        }
        let d = ComputeScheduler::new().route_select_advised(
            shape,
            Some(&model),
            RouteFlags::default(),
        );
        assert_eq!(
            d.backend,
            ComputeBackend::DataFusionLocal,
            "exploration must not route a join-bearing plan onto Native"
        );
        assert!(!d.reason.contains("EXPLORE"), "got: {}", d.reason);
    }

    #[test]
    fn rtt_budget_flip_respects_join_eligibility() {
        use crate::query::route_cost_model::RouteCostModel;
        let shape = QueryShape {
            engages_relational: true,
            parquet_backed: true,
            join_bearing: true,
            ..Default::default()
        }; // static => DataFusionLocal
        let model = RouteCostModel::new()
            .with_min_samples(1)
            .with_recompute_every(1)
            .with_rtt_budget(Some(10.0));
        model.set_override_enabled(true);
        // Static (DataFusion) blows the RTT budget; Native is within it — but
        // the plan is join-bearing, so the hard gate must NOT flip to Native.
        for _ in 0..5 {
            model.observe(
                "olap/parquet",
                &ComputeBackend::Native,
                &io_snap(3, 16 << 20),
            );
            model.observe(
                "olap/parquet",
                &ComputeBackend::DataFusionLocal,
                &io_snap(300, 16 << 20),
            );
        }
        let d = ComputeScheduler::new().route_select_advised(
            shape,
            Some(&model),
            RouteFlags::default(),
        );
        assert_eq!(
            d.backend,
            ComputeBackend::DataFusionLocal,
            "the RTT-budget hard gate must not flip a join-bearing plan onto Native"
        );
    }

    #[test]
    fn override_on_explores_under_explored_olap_parquet_candidate() {
        use crate::query::route_cost_model::RouteCostModel;
        // TD-ROUTE-3: exploration may only target a native-servable shape
        // (native-eligible op); the refined `op=agg` class consults the warmed
        // `olap/parquet` ancestor cells via hierarchical fallback.
        let shape = QueryShape {
            engages_relational: true,
            parquet_backed: true,
            operation_class: OperationClass::ScalarAggregate,
            ..Default::default()
        }; // static => DataFusionLocal
        let model = RouteCostModel::new()
            .with_min_samples(1)
            .with_exploration_interval(1)
            .with_recompute_every(1);
        model.set_override_enabled(true);
        // DataFusion (the static route) is warm; Native is unexplored → explore it.
        for _ in 0..5 {
            model.observe(
                "olap/parquet",
                &ComputeBackend::DataFusionLocal,
                &io_snap(4, 8192),
            );
        }
        let d = ComputeScheduler::new().route_select_advised(
            shape,
            Some(&model),
            RouteFlags::default(),
        );
        assert_eq!(
            d.backend,
            ComputeBackend::Native,
            "exploration routes to the under-explored freshness-safe candidate"
        );
        assert!(d.reason.contains("EXPLORE"));
    }

    #[test]
    fn exploration_never_targets_freshness_critical_oltp() {
        use crate::query::route_cost_model::RouteCostModel;
        let shape = QueryShape {
            engages_relational: false,
            parquet_backed: true,
            ..Default::default()
        }; // static => Native (OLTP)
        let model = RouteCostModel::new()
            .with_min_samples(1)
            .with_exploration_interval(1)
            .with_recompute_every(1);
        model.set_override_enabled(true);
        // Even with DataFusion history for this class, OLTP's freshness-safe set
        // is just [Native], so exploration has nothing to probe.
        for _ in 0..5 {
            model.observe(
                "oltp/parquet",
                &ComputeBackend::DataFusionLocal,
                &io_snap(1, 4096),
            );
        }
        let d = ComputeScheduler::new().route_select_advised(
            shape,
            Some(&model),
            RouteFlags::default(),
        );
        assert_eq!(d.backend, ComputeBackend::Native);
        assert!(!d.reason.contains("EXPLORE"));
    }

    #[test]
    fn override_never_flips_freshness_critical_oltp() {
        use crate::query::route_cost_model::RouteCostModel;
        // Point/OLTP shape: static => Native, MUST stay Native (strong freshness)
        // no matter what the cost model says.
        let shape = QueryShape {
            engages_relational: false,
            parquet_backed: true,
            ..Default::default()
        };
        let model = RouteCostModel::new()
            .with_min_samples(1)
            .with_recompute_every(1);
        model.set_override_enabled(true);
        // Even if DataFusion looks absurdly cheap for this class, it is not a
        // freshness-safe candidate for OLTP, so it can never be chosen.
        for _ in 0..5 {
            model.observe(
                "oltp/parquet",
                &ComputeBackend::Native,
                &io_snap(500, 16 << 20),
            );
            model.observe(
                "oltp/parquet",
                &ComputeBackend::DataFusionLocal,
                &io_snap(1, 4096),
            );
        }
        let d = ComputeScheduler::new().route_select_advised(
            shape,
            Some(&model),
            RouteFlags::default(),
        );
        assert_eq!(
            d.backend,
            ComputeBackend::Native,
            "OLTP must never be overridden off Native"
        );
        assert!(!d.reason.contains("OVERRIDE"));
    }

    #[test]
    fn scheduler_materializes_routed_read_plan() {
        let plan = ComputeScheduler::new().route_select_plan(QueryShape {
            engages_relational: true,
            parquet_backed: true,
            ..Default::default()
        });

        assert_eq!(plan.backend, ComputeBackend::DataFusionLocal);
        assert_eq!(plan.workload_profile, CatalogWorkloadProfile::Olap);
        assert_eq!(plan.route_explanation().selected_backend, "DataFusionLocal");
        assert_eq!(
            plan.route_explanation().policy_boundary,
            "connector-enforced"
        );
    }

    #[test]
    fn cold_model_consult_returns_static_decision() {
        use crate::query::route_cost_model::RouteCostModel;
        // Empty model (no observations) → frozen table empty → consult None →
        // static decision unchanged, no advisory appended.
        let model = RouteCostModel::new();
        let shape = QueryShape {
            engages_relational: true,
            parquet_backed: true,
            ..Default::default()
        };
        let advised = ComputeScheduler::new().route_select_advised(
            shape,
            Some(&model),
            RouteFlags::default(),
        );
        assert_eq!(advised.backend, ComputeBackend::DataFusionLocal);
        assert_eq!(advised.source, RouteSource::Static);
        assert!(!advised.reason.contains("cost-model"));
    }

    #[test]
    fn override_sets_decision_source_label() {
        use crate::query::route_cost_model::RouteCostModel;
        // TD-ROUTE-3: native-eligible shape so the override→Native flip is a valid
        // (servable) route, not a mis-attribution.
        let shape = QueryShape {
            engages_relational: true,
            parquet_backed: true,
            operation_class: OperationClass::ScalarAggregate,
            ..Default::default()
        };
        let model = RouteCostModel::new()
            .with_min_samples(1)
            .with_recompute_every(1);
        model.set_override_enabled(true);
        for _ in 0..5 {
            model.observe(
                "olap/parquet",
                &ComputeBackend::Native,
                &io_snap(3, 16 << 20),
            );
            model.observe(
                "olap/parquet",
                &ComputeBackend::DataFusionLocal,
                &io_snap(300, 16 << 20),
            );
        }
        let d = ComputeScheduler::new().route_select_advised(
            shape,
            Some(&model),
            RouteFlags::default(),
        );
        assert_eq!(d.backend, ComputeBackend::Native);
        assert_eq!(d.source, RouteSource::OverrideExploit);
    }

    /// TD-USUB-9 route-stamp integrity: a PAX-backed (not Parquet-backed) OLAP
    /// shape must stamp the backend that ACTUALLY serves it.
    ///
    /// Dispatch requires `parquet_backed`, so Volcano serves this shape. The old
    /// `pax_reader_enabled()` arm stamped `DataFusionLocal` anyway, which meant
    /// the cost model recorded Volcano's measured latency and I/O against
    /// DataFusion cells — silently corrupting the evidence ADR-052's
    /// observe→ingest→act sequence is built on. The env gate is deliberately not
    /// consulted here any more, so this holds with it set or unset.
    #[test]
    fn pax_backed_olap_stamps_the_backend_that_actually_serves() {
        let shape = QueryShape {
            engages_relational: true,
            parquet_backed: false,
            pax_backed: true,
            operation_class: OperationClass::ScalarAggregate,
            ..Default::default()
        };
        let decision = ComputeScheduler::new().route_select(shape);
        assert_eq!(
            decision.backend,
            ComputeBackend::Native,
            "PAX-backed-but-not-parquet-backed OLAP is served by Volcano, so it must \
             not be stamped as DataFusion — that is what poisons the cost model"
        );

        // The Trust axis must not claim object-store base-snapshot trust either:
        // this route reads WAL+RecordStorage synchronously.
        assert!(
            !decision
                .reasons
                .iter()
                .any(|r| r.detail.contains("base-snapshot statistics")),
            "a Volcano-served route must not claim object-store snapshot trust"
        );
    }

    #[test]
    fn olap_on_native_is_static_with_volcano_backend() {
        // An olap/native route stays on Volcano (single freshness-safe candidate)
        // with a Static source; the ROUTE_OLAP_ON_NATIVE_TOTAL nudge is the
        // operator signal (asserted in route_metrics tests), recorded in finalize_route.
        let shape = QueryShape {
            engages_relational: true,
            parquet_backed: false,
            ..Default::default()
        };
        let d = ComputeScheduler::new().route_select(shape);
        assert_eq!(d.backend, ComputeBackend::Native);
        assert_eq!(d.source, RouteSource::Static);
    }
}
