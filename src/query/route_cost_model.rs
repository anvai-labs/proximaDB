// Copyright 2026 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Trace-driven route cost model (co-design C4 — the keystone of the loop).
//!
//! The `ComputeScheduler` ([`crate::query::compute_scheduler`]) chooses a
//! physical engine from *static* shape heuristics. C4 closes the co-design loop
//! the C0 trace substrate was built for: feed the *measured* per-query
//! [`IoTraceSnapshot`] back so the scheduler can cost candidate routes from what
//! they *actually paid* (object-store GETs, bytes moved, compute-ms) rather than
//! a fixed rule — the Policy→RL evolution named in the course correction §5.2.
//!
//! ## How the loop works
//!
//! 1. A query runs under an `io_trace` scope (C0) and, at completion, yields an
//!    [`IoTraceSnapshot`] of the physical quantities it paid.
//! 2. [`RouteCostModel::observe`] folds that snapshot into a per-(shape-class,
//!    backend) EWMA — the running cost of serving *this kind of query* on *that
//!    engine*.
//! 3. At the next route decision for the same shape-class,
//!    [`RouteCostModel::recommend`] compares the learned cost of each candidate
//!    backend and names the cheapest. The scheduler surfaces this as an
//!    EXPLAIN/telemetry advisory (**observe-mode** — see
//!    `ComputeScheduler::route_select_advised`); flipping live routing onto the
//!    recommendation is a later, flag-gated slice.
//!
//! ## OSS boundary — neutral cost, not pricing
//!
//! The score combines *neutral relative I/O/compute units*, NOT currency. Per
//! the OSS/enterprise boundary, *pricing* (KSU/KRU/KEU $ weights, tenant
//! tiers) belongs to the commercial control plane; *routing* is OSS mechanism.
//! The default [`CostWeights`] only encode the co-design ordering — an
//! object-store round-trip dominates bandwidth, which dominates CPU (P5: the
//! dominant cost term for a cloud DB is I/O round-trips, not compute) — and are
//! tunable. They are deliberately *not* calibrated dollar figures.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use arc_swap::ArcSwap;

use crate::observability::io_trace::{IoTraceSnapshot, VectorAccessTrace, VectorCachePolicy};
use crate::query::compute_scheduler::{QueryShape, backend_label};
use crate::query::table_write_plan::ComputeBackend;

/// Process-global trace-driven route cost model. Fed by completed routed queries
/// (via [`install_route_cost_observer`]) and consulted at the route decision.
pub static GLOBAL_ROUTE_COST_MODEL: LazyLock<RouteCostModel> = LazyLock::new(RouteCostModel::new);

/// Wire the io_trace flush to feed [`GLOBAL_ROUTE_COST_MODEL`]: every completed
/// query that stamped a route (via `ComputeScheduler`) folds its measured
/// snapshot into the model. Call once at startup. This is the query-layer half
/// of the dependency-inversion seam — io_trace defines the observer type and
/// never depends on this module.
pub fn install_route_cost_observer() {
    crate::observability::io_trace::set_route_observer(Some(Box::new(
        |snap, shape_class, backend_label| {
            GLOBAL_ROUTE_COST_MODEL.observe_by_label(shape_class, backend_label, snap);
        },
    )));
    crate::observability::io_trace::set_vector_access_observer(Some(Box::new(|snap| {
        observe_vector_access_snapshot(&GLOBAL_ROUTE_COST_MODEL, snap);
    })));
    // Flag-gated live override (default OFF). PROXIMADB_ROUTE_COST_OVERRIDE is
    // either the original global bool ("1"/"true" → every class) or, for the
    // TD-EXEC-2 staged go-live, a comma-separated list of shape-class prefixes
    // ("olap/parquet/op=agg,olap/parquet/op=cnt") enabling the override only
    // where cells have been warmed and verified — everything else stays
    // observe-only. Unparseable values fail safe to OFF.
    let scope = std::env::var("PROXIMADB_ROUTE_COST_OVERRIDE")
        .map(|v| OverrideScope::parse(&v))
        .unwrap_or(OverrideScope::Off);
    if scope != OverrideScope::Off {
        tracing::info!("route cost model: live override enabled with scope {scope:?}");
    }
    GLOBAL_ROUTE_COST_MODEL.set_override_scope(scope);
}

/// Query-local cache outcome for a completed vector access. This is observed
/// evidence, not a pre-execution prediction: it may partition learned cells,
/// but must not be treated as a route-time cache promise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorCacheOutcome {
    Cold,
    Warm,
    Mixed,
    /// The instrumented read path issued physical reads without consulting a
    /// traced cache. This is a known uncached regime, not missing evidence.
    Bypass,
    Unknown,
}

impl VectorCacheOutcome {
    pub fn from_snapshot(snap: &IoTraceSnapshot) -> Self {
        let hits = snap
            .footer_hits
            .saturating_add(snap.survivor_l1_hits)
            .saturating_add(snap.l2_hits);
        let misses = snap
            .footer_misses
            .saturating_add(snap.survivor_l1_misses)
            .saturating_add(snap.l2_misses);

        // Physical I/O is authoritative. Lower-tier misses followed by an L1
        // or L2 hit are still warm when no object-store GET escaped the cache;
        // any cache hit plus a physical GET is mixed. With io-trace compiled
        // out, absence of cache counters is missing evidence — UNLESS the
        // route-time policy declares the regime uncached: `Disabled` is stamped
        // always-on from the real engine cache handles, so physical GETs under
        // a Disabled policy are a KNOWN uncached regime (bypass) regardless of
        // whether the probe counters were compiled in.
        if snap.get_ops == 0 && hits > 0 {
            Self::Warm
        } else if snap.get_ops > 0 && hits > 0 {
            Self::Mixed
        } else if snap.get_ops > 0 && misses > 0 {
            Self::Cold
        } else if snap.get_ops > 0
            && (Self::policy_declares_uncached(snap) || cfg!(feature = "io-trace"))
        {
            Self::Bypass
        } else {
            Self::Unknown
        }
    }

    /// True when every recorded access ran under an explicitly `Disabled`
    /// cache policy — route-time evidence that no cache tier exists to be
    /// consulted, independent of the `io-trace` feature.
    fn policy_declares_uncached(snap: &IoTraceSnapshot) -> bool {
        !snap.vector_accesses.is_empty()
            && snap
                .vector_accesses
                .iter()
                .all(|a| a.cache_policy == VectorCachePolicy::Disabled)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cold => "cold",
            Self::Warm => "warm",
            Self::Mixed => "mixed",
            Self::Bypass => "bypass",
            Self::Unknown => "unknown",
        }
    }

    /// Whether the cache regime is sufficiently observed to partition an
    /// admission cohort. Unknown remains durable but cannot authorize routing.
    pub fn is_admissible(self) -> bool {
        self != Self::Unknown
    }
}

/// Exact, unbucketed physical geometry for one vector access-path cost cell.
/// The engine, physical storage locality, and route-time cache policy belong in
/// the shape so the same identity can be constructed before execution. Caller
/// intent and completed cache outcome are durable evidence but not cell
/// dimensions: either would prevent a truthful exact-vs-ANN comparison.
pub fn vector_access_shape(access: &VectorAccessTrace) -> String {
    format!(
        "vector/engine={}/dim={}/k={}/filter={}/storage={}/cache_policy={}",
        access.engine,
        access.dimensions,
        access.top_k,
        if access.has_filter { "yes" } else { "no" },
        access.storage_scope.as_str(),
        access.cache_policy.as_str(),
    )
}

/// Fold a completed vector query into the shared measured-cost cell store.
/// Aggregate I/O/compute cannot be apportioned truthfully across multiple
/// vector operators, so multi-access queries remain durable evidence but are
/// excluded from online learning until operator-local costs exist.
fn observe_vector_access_snapshot(model: &RouteCostModel, snap: &IoTraceSnapshot) -> bool {
    let [access] = snap.vector_accesses.as_slice() else {
        return false;
    };
    let cache_outcome = VectorCacheOutcome::from_snapshot(snap);
    if access.storage_scope == crate::observability::io_trace::VectorStorageScope::Unknown
        || !access.cache_policy.is_admissible()
        || !cache_outcome.is_admissible()
    {
        return false;
    }
    model.observe_by_label(
        &vector_access_shape(access),
        access.actual_path.as_str(),
        snap,
    );
    true
}

// ── Override scope: global bool → per-shape-class staged go-live (D4) ───────

/// Scope of the live route override — ADR-058 D4 kept the flip opt-in and
/// per-deployment; TD-EXEC-2 §3 plans the go-live "per shape⊕geometry class,
/// behind the shadow-compare + ratchet". This type carries that staging: a
/// deployment can enable the override only for the classes whose cells it has
/// warmed and verified (e.g. via `cargo ledger-tpc`), leaving every other class
/// observe-only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OverrideScope {
    /// Observe-only (the default): the model never changes a route.
    Off,
    /// Override enabled for every shape-class (the original global bool).
    All,
    /// Override enabled only for shape-classes matching one of these prefixes
    /// on a `/` boundary — `olap/parquet/op=agg` covers all of its geometry
    /// refinements (`olap/parquet/op=agg/geom=sx0`, …) but not the sibling
    /// `olap/parquet/op=grp`.
    Classes(Vec<String>),
}

impl OverrideScope {
    /// Parse the `PROXIMADB_ROUTE_COST_OVERRIDE` value. `"1"`/`"true"` → [`Self::All`];
    /// empty/`"0"`/`"false"` → [`Self::Off`]; otherwise a comma-separated list of
    /// shape-class prefixes (tokens must contain `/` — the minimal well-formed
    /// class key is `workload/base`). A value with no well-formed token fails
    /// safe to [`Self::Off`] — a typo must never silently enable a global flip.
    pub fn parse(raw: &str) -> Self {
        let v = raw.trim();
        if v.is_empty() || v == "0" || v.eq_ignore_ascii_case("false") {
            return Self::Off;
        }
        if v == "1" || v.eq_ignore_ascii_case("true") {
            return Self::All;
        }
        let classes: Vec<String> = v
            .split(',')
            .map(str::trim)
            .filter(|t| t.contains('/'))
            .map(str::to_string)
            .collect();
        if classes.is_empty() {
            Self::Off
        } else {
            Self::Classes(classes)
        }
    }

    /// Whether the override may fire for `shape_class` under this scope.
    /// Prefix matching is on `/` boundaries, mirroring the hierarchical class
    /// key (`consult_with_fallback`): enabling an ancestor enables all of its
    /// refinements, never a lexical near-miss.
    pub fn enables(&self, shape_class: &str) -> bool {
        match self {
            Self::Off => false,
            Self::All => true,
            Self::Classes(prefixes) => prefixes.iter().any(|p| {
                shape_class == p.as_str()
                    || shape_class
                        .strip_prefix(p.as_str())
                        .is_some_and(|rest| rest.starts_with('/'))
            }),
        }
    }
}

// ── Persistence: let the measured cost history survive restarts ─────────────
//
// The trace-driven model learns per (shape-class, backend) from measured I/O.
// In-memory only, it re-learns cold on every boot — so the co-design signal
// ("route from measured quantities") is thrown away at each restart, and any
// future live-override slice would face a cold-start mislearning window.
// Persisting the EWMA cells fixes both: warm advisories across restarts, and a
// warm model when override is eventually enabled. Behavior is unchanged while
// the override stays OFF (default). Mirrors the RL-planner policy persistence
// (`{data_dir}/rl_policy.json`).

/// On-disk snapshot version. Only the EWMA cells are persisted; the frozen
/// consult table is re-derived on load. Bumping this invalidates older files
/// (they load nothing and the model re-learns), so a format change is never
/// mis-read.
const COST_MODEL_PERSIST_VERSION: u32 = 2;

#[derive(serde::Serialize, serde::Deserialize)]
struct PersistedCostModel {
    version: u32,
    cells: Vec<PersistedCell>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct PersistedCell {
    shape_class: String,
    backend: String,
    cell: Cell,
}

/// Path of the persisted global cost model under a server data dir.
pub fn cost_model_persist_path(data_dir: &std::path::Path) -> std::path::PathBuf {
    data_dir.join("route_cost_cells.json")
}

/// Warm the global cost model from `{data_dir}/route_cost_cells.json` at startup
/// if present. Best-effort: a missing / unreadable / older-version file loads
/// nothing and the model re-learns. Routing is unchanged either way (the model
/// stays observe-only until `PROXIMADB_ROUTE_COST_OVERRIDE`).
pub fn load_persisted_cost_model(data_dir: &std::path::Path) {
    let path = cost_model_persist_path(data_dir);
    let loaded = GLOBAL_ROUTE_COST_MODEL.load_from(&path);
    if loaded > 0 {
        tracing::info!(
            "route cost model: warmed {loaded} learned cell(s) from {}",
            path.display()
        );
    }
}

/// Persist the global cost model to `{data_dir}/route_cost_cells.json` at
/// shutdown (best-effort; a write failure is logged, not fatal).
pub fn persist_cost_model(data_dir: &std::path::Path) {
    let path = cost_model_persist_path(data_dir);
    match GLOBAL_ROUTE_COST_MODEL.persist_to(&path) {
        Ok(n) if n > 0 => tracing::info!(
            "route cost model: persisted {n} learned cell(s) to {}",
            path.display()
        ),
        Ok(_) => {} // nothing learned yet — nothing to persist
        Err(e) => tracing::warn!(
            "route cost model: failed to persist to {}: {e}",
            path.display()
        ),
    }
}

impl RouteCostModel {
    /// Persist the learned cells to `path` via an atomic temp+rename. Returns
    /// the number of cells written (only cells with ≥1 sample are persisted).
    fn persist_to(&self, path: &std::path::Path) -> std::io::Result<usize> {
        let snapshot = {
            let cells = self.cells.lock().unwrap_or_else(|p| p.into_inner());
            PersistedCostModel {
                version: COST_MODEL_PERSIST_VERSION,
                cells: cells
                    .iter()
                    .filter(|(_, c)| c.samples > 0)
                    .map(|((class, backend), cell)| PersistedCell {
                        shape_class: class.clone(),
                        backend: backend.clone(),
                        cell: *cell,
                    })
                    .collect(),
            }
        };
        let n = snapshot.cells.len();
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let bytes = serde_json::to_vec_pretty(&snapshot).map_err(std::io::Error::other)?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, &bytes)?;
        std::fs::rename(&tmp, path)?;
        Ok(n)
    }

    /// Replace the learned cells with those persisted at `path`, then warm the
    /// frozen consult table. Returns the number of cells loaded; 0 on a missing,
    /// unreadable, or version-mismatched file (the model re-learns from scratch).
    fn load_from(&self, path: &std::path::Path) -> usize {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(_) => return 0, // no persisted model yet
        };
        let snapshot: PersistedCostModel = match serde_json::from_slice(&bytes) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(
                    "route cost model: ignoring unreadable persisted file {}: {e}",
                    path.display()
                );
                return 0;
            }
        };
        if snapshot.version != COST_MODEL_PERSIST_VERSION {
            tracing::warn!(
                "route cost model: ignoring persisted file {} (version {}, expected {})",
                path.display(),
                snapshot.version,
                COST_MODEL_PERSIST_VERSION
            );
            return 0;
        }
        let n = snapshot.cells.len();
        {
            let mut cells = self.cells.lock().unwrap_or_else(|p| p.into_inner());
            for pc in snapshot.cells {
                cells.insert((pc.shape_class, pc.backend), pc.cell);
            }
        }
        // Re-derive the frozen consult table from the loaded cells so advisories
        // (and a future enabled override) see the warm history immediately.
        self.recompute_locked();
        n
    }
}

// ── Per-Parquet-location shape stats (route-time classification) ────────────

/// One Parquet table's physical shape, warmed for free wherever the table is
/// opened and peeked at route time so the cost-model shape-class discriminates
/// OLAP-over-Parquet queries by fan-out / cardinality WITHOUT a cold footer
/// read on the route path (co-design P5: I/O round-trips, not CPU, dominate).
#[derive(Clone, Copy, Debug)]
pub struct TableShapeStat {
    /// Row-group / split count — the GET-fan-out a scan of this table pays.
    pub row_groups: u32,
    /// Estimated row count from footer statistics, when the footer carries it.
    pub rows: Option<u64>,
}

/// Process-global, bounded cache of per-Parquet-location [`TableShapeStat`]s.
/// Warmed as a free side-effect of opening a table for execution/EXPLAIN (the
/// opener already reads the footer) and consulted at the route decision so the
/// scheduler's [`QueryShape`] carries real `partition_fanout` / `cardinality`
/// for hot tables. A cold (never-scanned) table simply lacks an entry → the
/// shape stays coarse (`Unknown`), so a fresh table changes nothing until its
/// first scan warms the stat. Workload-adaptive, and adds zero I/O to routing.
static GLOBAL_TABLE_SHAPE_STATS: LazyLock<Mutex<HashMap<String, TableShapeStat>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Soft cap on cached locations. Locations are catalog-bounded in practice; this
/// only guards a runaway (e.g. a bug minting synthetic locations). Over-cap the
/// cache clears wholesale — stat data is advisory and re-warms on the next scan.
const TABLE_SHAPE_STAT_CAP: usize = 8192;

/// Record (or refresh) a table's shape stat. Called by the Parquet opener after
/// it has the footer, so this adds no I/O. Best-effort: a poisoned lock or an
/// over-cap cache is silently dropped (classification then falls back coarse).
pub fn record_table_shape_stat(location: &str, row_groups: u32, rows: Option<u64>) {
    let mut guard = GLOBAL_TABLE_SHAPE_STATS
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    if guard.len() >= TABLE_SHAPE_STAT_CAP {
        guard.clear();
    }
    guard.insert(location.to_string(), TableShapeStat { row_groups, rows });
}

/// Aggregate the cached stats for `locations` into cost-model shape buckets.
///
/// Conservative: if *any* referenced location lacks a warmed stat, returns
/// `(Unknown, Unknown)` — never half-classify a multi-table query from partial
/// data, since an unknown table could dominate fan-out / cardinality. An empty
/// slice also yields `Unknown`. Sums row-groups (total GET fan-out) and rows
/// (total scan cardinality) across the tables, then buckets via the scheduler's
/// [`PartitionFanout`] / [`CardinalityClass`].
///
/// [`PartitionFanout`]: crate::query::compute_scheduler::PartitionFanout
/// [`CardinalityClass`]: crate::query::compute_scheduler::CardinalityClass
pub fn classify_table_shapes(
    locations: &[String],
) -> (
    crate::query::compute_scheduler::PartitionFanout,
    crate::query::compute_scheduler::CardinalityClass,
) {
    use crate::query::compute_scheduler::{CardinalityClass, PartitionFanout};
    if locations.is_empty() {
        return (PartitionFanout::Unknown, CardinalityClass::Unknown);
    }
    let guard = GLOBAL_TABLE_SHAPE_STATS
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let mut total_row_groups = 0u32;
    let mut total_rows: Option<u64> = Some(0);
    for loc in locations {
        let Some(stat) = guard.get(loc) else {
            return (PartitionFanout::Unknown, CardinalityClass::Unknown);
        };
        total_row_groups = total_row_groups.saturating_add(stat.row_groups);
        total_rows = match (total_rows, stat.rows) {
            (Some(acc), Some(r)) => Some(acc.saturating_add(r)),
            _ => None,
        };
    }
    (
        PartitionFanout::from_count(Some(total_row_groups)),
        CardinalityClass::from_estimate(total_rows),
    )
}

/// Per-tenant governance **tier-entitlement multiplier** resolver — the final
/// term of the §3 `Cost(q)` objective (Dimension 5, C5). Maps a `tenant_id` to a
/// neutral scalar applied to the *reported* cost.
type TierMultiplierFn = dyn Fn(&str) -> f64 + Send + Sync;

static TIER_MULTIPLIER_RESOLVER: Mutex<Option<Box<TierMultiplierFn>>> = Mutex::new(None);

/// Install (or clear with `None`) the per-tenant tier-multiplier resolver. This
/// is the OSS **mechanism** / DI seam (the `LimitsResolver` pattern): the OSS
/// core ships no resolver (default multiplier `1.0`, fully inert); the commercial
/// control plane (anvaiops) installs one at startup that maps a tenant to its
/// tier entitlement from claims/config. Replaceable in tests. No new authority —
/// the tenant→tier authority stays in the control plane; this only consults it.
pub fn set_tier_multiplier_resolver(resolver: Option<Box<TierMultiplierFn>>) {
    *TIER_MULTIPLIER_RESOLVER
        .lock()
        .unwrap_or_else(|p| p.into_inner()) = resolver;
}

/// The tier-entitlement multiplier for `tenant_id`. Returns `1.0` (inert) when no
/// resolver is installed, the tenant is unknown (`None`), or the resolver returns
/// a non-finite / non-positive value — a non-positive multiplier could invert
/// cost ordering, so it is rejected fail-safe.
pub fn tier_entitlement_multiplier(tenant_id: Option<&str>) -> f64 {
    let Some(id) = tenant_id else {
        return 1.0;
    };
    let guard = TIER_MULTIPLIER_RESOLVER
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    match guard.as_ref() {
        Some(resolve) => {
            let m = resolve(id);
            if m.is_finite() && m > 0.0 { m } else { 1.0 }
        }
        None => 1.0,
    }
}

/// The §3 final per-tenant `Cost(q)`: the neutral base route score scaled by the
/// tenant's tier-entitlement multiplier — the last term of the unified cost
/// objective. This is deliberately **routing-neutral**: a single per-tenant
/// scalar scales every candidate equally, so it can never change *which* engine
/// is cheapest (entitlement that restricts the candidate *set* is the scheduler's
/// `override_candidates`, safe-by-construction). Its role is to make the reported
/// / billed cost the real per-tenant `Cost(q)` for EXPLAIN, chargeback, and a
/// future budget-aware admission decision — completing the objective the C0/C4
/// loop was built to minimize, while keeping pricing/policy out of the OSS core.
pub fn final_cost(base_score: f64, tenant_id: Option<&str>) -> f64 {
    base_score * tier_entitlement_multiplier(tenant_id)
}

/// Stable shape-class key the model aggregates under. The coarse base mirrors the
/// scheduler's binary signals (`engages_relational` × `parquet_backed`); C4
/// Phase-2b refines it with the §5.2 Phase-1 inputs — cardinality and partition
/// fan-out — TD-OLAP-4 with the operation class, and TD-EXEC-2 Slice 3 with the
/// plan-geometry tier (`geom=<depth-band>x<blocking-band>`) — but only when they
/// are *known*. An `Unknown` signal contributes no suffix, so a planner that
/// cannot estimate them yields exactly the original 2-part key (`olap/parquet`,
/// `oltp/native`, …) — backward-compatible with warmed cells and existing
/// EXPLAIN output.
pub fn shape_class(shape: &QueryShape) -> String {
    let workload = if shape.engages_relational {
        "olap"
    } else {
        "oltp"
    };
    let base = if shape.parquet_backed {
        "parquet"
    } else {
        "native"
    };
    let mut class = format!("{workload}/{base}");
    for suffix in [
        shape.cardinality.class_suffix(),
        shape.partition_fanout.class_suffix(),
        shape.operation_class.class_suffix(),
        shape.geometry.class_suffix(),
    ]
    .into_iter()
    .flatten()
    {
        class.push('/');
        class.push_str(suffix);
    }
    class
}

/// Neutral relative weights for the route cost score. NOT dollars (see module
/// docs). They encode only the co-design cost ordering of the §3 unified cost
/// objective `Cost(q)`: an object-store round-trip dominates cross-region egress,
/// which dominates same-region read bandwidth, which dominates a ms of CPU
/// (P5: for a cloud DB the dominant cost term is I/O round-trips + egress, not
/// compute — §2.2).
#[derive(Debug, Clone, Copy)]
pub struct CostWeights {
    /// Per ranged GET — the fixed-latency, per-request term (the dominant cost
    /// for object-store-served reads). §3 `GET_count·get_fee`.
    pub per_get: f64,
    /// Per MiB read — the same-region read-bandwidth term. §3 `bytes_read·bw_cost`.
    pub per_mib_read: f64,
    /// Per MiB moved cross-region / to the internet — the KEU **egress** term
    /// (§2.2, §3 `bytes_moved·locality_cost`). Weighted above same-region read
    /// bandwidth because cross-region (~$0.02/GB) / internet egress
    /// (~$0.09-0.12/GB by cloud) is frequently the dominant TCO term. Fed by the
    /// trace's `egress_bytes`, which is **zero on the free same-region path**, so
    /// this term is inert until a deployment actually moves bytes cross-region.
    pub per_mib_egress: f64,
    /// Per MiB written to object storage — the KIU ingest / storage-write term
    /// (§3 storage side). Zero for read-only SELECT routing, so inert today;
    /// present so a write-route cost (`table_write_plan`) folds in here too.
    pub per_mib_written: f64,
    /// Per compute-millisecond — the CPU term (smallest for I/O-bound DB work).
    /// §3 `engine_ms·rate`.
    pub per_compute_ms: f64,
}

impl Default for CostWeights {
    fn default() -> Self {
        // Illustrative neutral ordering (round-trip ≫ egress ≫ read-bw ≫ CPU),
        // tunable. `per_mib_written` mirrors `per_mib_read` (a PUT's bandwidth).
        Self {
            per_get: 20.0,
            per_mib_read: 5.0,
            per_mib_egress: 25.0,
            per_mib_written: 5.0,
            per_compute_ms: 1.0,
        }
    }
}

/// The cost-bearing physical quantities of one query, extracted from its
/// measured [`IoTraceSnapshot`]. Folded (EWMA) per (shape-class, backend) and
/// scored by [`CostWeights`] into the §3 `Cost(q)`. Keeping them in one struct
/// (rather than positional `f64` args) keeps `fold`/`score` self-documenting as
/// terms are added.
#[derive(Debug, Clone, Copy, Default)]
struct CostQuantities {
    get_ops: f64,
    bytes_read: f64,
    egress_bytes: f64,
    bytes_written: f64,
    compute_ms: f64,
    // Diagnostic observability (ADR-056 AQE-S2): the runtime-filter skip ratio
    // + the wait-outcome ratio. NOT in `score` (pruning already reduced
    // `get_ops`/`bytes_read` — scoring these double-counts). Recorded per
    // (shape-class, backend) so promotion/demotion gates + AQE-S4 (build/probe
    // swap) can read them.
    splits_total: f64,
    splits_pruned: f64,
    runtime_filter_arrived: f64,
    runtime_filter_timed_out: f64,
    runtime_filter_wait_ms: f64,
}

impl CostQuantities {
    fn from_snapshot(snap: &IoTraceSnapshot) -> Self {
        Self {
            get_ops: snap.get_ops as f64,
            bytes_read: snap.bytes_read as f64,
            egress_bytes: snap.egress_bytes as f64,
            bytes_written: snap.bytes_written as f64,
            compute_ms: snap.total_compute_ms() as f64,
            splits_total: snap.splits_total as f64,
            splits_pruned: snap.splits_pruned as f64,
            runtime_filter_arrived: snap.runtime_filter_arrived as f64,
            runtime_filter_timed_out: snap.runtime_filter_timed_out as f64,
            runtime_filter_wait_ms: snap.runtime_filter_wait_ms as f64,
        }
    }
}

/// One (shape-class, backend) cell: EWMA means of the cost-bearing quantities.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
struct Cell {
    get_ops: f64,
    bytes_read: f64,
    egress_bytes: f64,
    bytes_written: f64,
    compute_ms: f64,
    splits_total: f64,
    splits_pruned: f64,
    runtime_filter_arrived: f64,
    runtime_filter_timed_out: f64,
    runtime_filter_wait_ms: f64,
    samples: u64,
}

impl Cell {
    fn fold(&mut self, alpha: f64, q: CostQuantities) {
        if self.samples == 0 {
            self.get_ops = q.get_ops;
            self.bytes_read = q.bytes_read;
            self.egress_bytes = q.egress_bytes;
            self.bytes_written = q.bytes_written;
            self.compute_ms = q.compute_ms;
            self.splits_total = q.splits_total;
            self.splits_pruned = q.splits_pruned;
            self.runtime_filter_arrived = q.runtime_filter_arrived;
            self.runtime_filter_timed_out = q.runtime_filter_timed_out;
            self.runtime_filter_wait_ms = q.runtime_filter_wait_ms;
        } else {
            let blend = |old: f64, new: f64| alpha * new + (1.0 - alpha) * old;
            self.get_ops = blend(self.get_ops, q.get_ops);
            self.bytes_read = blend(self.bytes_read, q.bytes_read);
            self.egress_bytes = blend(self.egress_bytes, q.egress_bytes);
            self.bytes_written = blend(self.bytes_written, q.bytes_written);
            self.compute_ms = blend(self.compute_ms, q.compute_ms);
            self.splits_total = blend(self.splits_total, q.splits_total);
            self.splits_pruned = blend(self.splits_pruned, q.splits_pruned);
            self.runtime_filter_arrived =
                blend(self.runtime_filter_arrived, q.runtime_filter_arrived);
            self.runtime_filter_timed_out =
                blend(self.runtime_filter_timed_out, q.runtime_filter_timed_out);
            self.runtime_filter_wait_ms =
                blend(self.runtime_filter_wait_ms, q.runtime_filter_wait_ms);
        }
        self.samples += 1;
    }

    fn quantities(&self) -> CostQuantities {
        CostQuantities {
            get_ops: self.get_ops,
            bytes_read: self.bytes_read,
            egress_bytes: self.egress_bytes,
            bytes_written: self.bytes_written,
            compute_ms: self.compute_ms,
            splits_total: self.splits_total,
            splits_pruned: self.splits_pruned,
            runtime_filter_arrived: self.runtime_filter_arrived,
            runtime_filter_timed_out: self.runtime_filter_timed_out,
            runtime_filter_wait_ms: self.runtime_filter_wait_ms,
        }
    }
}

/// Learned cost estimate for serving a shape-class on one backend — the EWMA of
/// each §3 `Cost(q)` quantity plus the weighted neutral score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RouteCost {
    pub samples: u64,
    pub get_ops: f64,
    pub bytes_read: f64,
    /// EWMA cross-region / internet egress bytes (KEU) — zero on the free same-region path.
    pub egress_bytes: f64,
    /// EWMA bytes written to object storage (KIU) — zero for read-only routes.
    pub bytes_written: f64,
    pub compute_ms: f64,
    /// Weighted neutral score (lower is cheaper).
    pub score: f64,
}

/// The scheduler-facing recommendation: the cheapest candidate that has enough
/// history, plus the runners-up for EXPLAIN.
#[derive(Debug, Clone)]
pub struct RouteRecommendation {
    pub backend: ComputeBackend,
    pub score: f64,
    pub samples: u64,
    /// `(backend_label, score)` for every candidate with history, cheapest first.
    pub ranked: Vec<(String, f64)>,
}

impl RouteRecommendation {
    /// Compact reason string for the EXPLAIN/telemetry advisory.
    pub fn reason(&self) -> String {
        let ranked = self
            .ranked
            .iter()
            .map(|(b, s)| format!("{b}={s:.1}"))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "min-cost over {} sample(s): {} [{}]",
            self.samples,
            backend_label(&self.backend),
            ranked
        )
    }
}

// ── Frozen recommendation table: the lock-free consult path (co-design C4) ──
//
// The consult half of the cost model. `RouteCostModel::consult` does a single
// `ArcSwap::load` + `HashMap` get — never the learn-path `cells` mutex — so the
// route decision stays O(1) and lock-free even when live override is enabled.
// The EWMA learner is the SOLE writer, recomputing this table debounced on the
// observe path; readers always see a consistent immutable snapshot.

/// One freshness-safe candidate's baked cost summary in the frozen table.
#[derive(Debug, Clone)]
pub struct BackendChoice {
    /// The candidate engine.
    pub backend: ComputeBackend,
    /// Weighted neutral score (lower = cheaper) — the soft byte-cost term.
    pub score: f64,
    /// EWMA sample count at bake time.
    pub samples: u64,
    /// Predicted physical GET count — the TD-170 hard round-trip-budget term.
    pub get_ops: f64,
}

/// Per-shape-class pre-baked recommendation: everything the lock-free consult
/// needs to reproduce the explore / override / advisory decision without a
/// lock. Immutable once published via [`ArcSwap`].
#[derive(Debug, Clone)]
pub struct ClassRecommendation {
    /// Every freshness-safe candidate with ANY history, cheapest-first. Carries
    /// score/samples/get_ops so the consult can apply BOTH the soft
    /// `min_advantage` gate and the hard `rtt_budget` gate from frozen data.
    ranked: Vec<BackendChoice>,
    /// The least-sampled freshness-safe candidate still under `min_samples` (the
    /// exploration warm-up target), or `None` once all are warm. May target a
    /// never-observed safe candidate (0 samples). Resolved against the live
    /// `explore_tick` at consult time.
    explore_target: Option<ComputeBackend>,
}

impl ClassRecommendation {
    /// Ranked freshness-safe candidates with history, cheapest-first.
    pub fn ranked(&self) -> &[BackendChoice] {
        &self.ranked
    }
    /// The baked exploration warm-up target, if any candidate is still cold.
    pub fn explore_target(&self) -> Option<&ComputeBackend> {
        self.explore_target.as_ref()
    }
    /// Find the candidate matching `backend` (by canonical label). Used to
    /// resolve the static route's baked cost at consult time.
    pub fn find(&self, backend: &ComputeBackend) -> Option<&BackendChoice> {
        let want = backend_label(backend);
        self.ranked
            .iter()
            .find(|c| backend_label(&c.backend) == want)
    }
}

/// Immutable snapshot of all per-class recommendations, held in an [`ArcSwap`]
/// inside [`RouteCostModel`]. The consult path loads it once and does a
/// `HashMap` get — fully lock-free.
#[derive(Debug, Clone, Default)]
struct RouteRecommendationTable {
    by_class: HashMap<String, ClassRecommendation>,
}

/// Lock-free consult over the frozen recommendation table (DIP seam). The
/// scheduler depends on this abstraction, not the concrete [`RouteCostModel`];
/// the concrete model implements it backed by its [`ArcSwap`] table, and a test
/// double can satisfy it with canned data. Production consult stays
/// monomorphized over the concrete model (no `&dyn` on the hot path) — the
/// trait exists for testability and a future persisted/remote cost model.
pub trait RouteConsult {
    /// Lock-free consult for one shape-class. `None` when the class has no baked
    /// entry (cold start — the caller keeps the static route).
    fn consult(&self, shape_class: &str) -> Option<ClassRecommendation>;
    /// Hierarchical consult (TD-EXEC-2 Slice 3): the refined class first, then
    /// each progressively coarser ancestor (one `/suffix` stripped at a time)
    /// down to the 2-part `workload/base` key. Returns the recommendation and
    /// the class level that served it. Refining the key must never orphan
    /// warmed learning: a query keyed `olap/parquet/geom=mxhi` is served by the
    /// warmed `olap/parquet` cells until its own refined cell warms via the
    /// stamped fold — so the frozen table stays effectively *dense* across
    /// key-refinement generations (ADR-058 D3), instead of every new tier
    /// resetting the model. A bounded handful of hash gets against the same
    /// frozen table — still lock-free, still consult-O(1).
    fn consult_with_fallback(&self, shape_class: &str) -> Option<(ClassRecommendation, String)> {
        let mut level = shape_class;
        loop {
            if let Some(rec) = self.consult(level) {
                return Some((rec, level.to_string()));
            }
            match level.rfind('/') {
                // Strip the last `/suffix`; the 2-part `workload/base` key is
                // the floor (its prefix has no further `/`).
                Some(idx) if level[..idx].contains('/') => level = &level[..idx],
                _ => return None,
            }
        }
    }
    /// Whether live override is enabled for ANY class (reads the live scope).
    fn override_active(&self) -> bool;
    /// Whether live override may fire for `shape_class` (per-class staged
    /// go-live, TD-EXEC-2 §3). Defaults to the global answer so a test double
    /// that only models the bool keeps working; the concrete model refines it
    /// against its [`OverrideScope`].
    fn override_active_for(&self, shape_class: &str) -> bool {
        let _ = shape_class;
        self.override_active()
    }
    /// Advance and return the exploration rate-limit tick (pre-increment value).
    fn next_exploration_tick(&self) -> u64;
    /// Config surfaced for the consult's gate logic.
    fn min_samples(&self) -> u64;
    fn min_advantage(&self) -> f64;
    fn exploration_interval(&self) -> u64;
    /// TD-170 hard round-trip budget (`None` = disabled).
    fn rtt_budget(&self) -> Option<f64>;
}

/// Inverse of [`backend_label`] — recover a [`ComputeBackend`] from its frozen
/// label. Returns `None` for an unrecognized label so the recomputer skips it
/// instead of panicking (a stale/mistyped label never corrupts the frozen table).
fn parse_backend_label(label: &str) -> Option<ComputeBackend> {
    match label {
        "Native(Volcano)" => Some(ComputeBackend::Native),
        "DataFusionLocal" => Some(ComputeBackend::DataFusionLocal),
        "DataFusionDistributed" => Some(ComputeBackend::DataFusionDistributed),
        "PolarsLocal" => Some(ComputeBackend::PolarsLocal),
        "DuckDbCompat" => Some(ComputeBackend::DuckDbCompat),
        _ => label
            .strip_prefix("ExternalDelegated(")
            .and_then(|s| s.strip_suffix(')'))
            .map(|s| ComputeBackend::ExternalDelegated(s.to_string())),
    }
}

/// The freshness-safe backend SET for a shape-class, derived from its prefix.
/// Only `olap/parquet` has multiple freshness-compatible engines (Native's
/// strong-freshness scan and DataFusion's base-snapshot scan both correctly
/// answer an analytic query); every other class keeps its single static
/// backend, so a baked override can never flip a query onto an engine that
/// would serve it incorrectly.
///
/// NOTE (TD-ROUTE-1): freshness-safety is baked here, but *capability*
/// eligibility is NOT — the class string carries no join signal (a
/// `JOIN … GROUP BY` classes as `op=grp`), so join-bearing plans are kept off
/// Native at decision time in `ComputeScheduler::route_select_advised` via
/// `QueryShape::join_bearing`, which also covers ancestor-fallback consults.
/// DuckDB's residual freshness term — it cannot run the ADR-025 post-snapshot
/// WAL delta merge — is likewise a dispatch-time gate (the pipeline only
/// executes DuckDB when no referenced table carries a pending delta; the
/// DataFusion floor serves otherwise and the route is re-stamped).
fn freshness_safe_backends_from_class(shape_class: &str) -> Vec<ComputeBackend> {
    if shape_class.starts_with("olap/parquet") {
        // DuckDB remains an explicit-route evidence harness until its reader
        // reports comparable GET/byte quantities. Ranking compute-only cells
        // beside fully metered engines would make live overrides prefer an
        // artificially cheap candidate.
        vec![ComputeBackend::Native, ComputeBackend::DataFusionLocal]
    } else {
        vec![ComputeBackend::Native]
    }
}

/// Trace-driven, thread-safe route cost model. One instance is consulted at the
/// route decision and fed by completed-query traces.
#[derive(Debug)]
pub struct RouteCostModel {
    cells: Mutex<HashMap<(String, String), Cell>>,
    alpha: f64,
    weights: CostWeights,
    /// Minimum samples before a backend's estimate is trusted enough to compare.
    min_samples: u64,
    /// Live route override scope (flag-gated; default [`OverrideScope::Off`] →
    /// observe-only). `ArcSwap` so the process-global model can be re-scoped at
    /// startup while consults read it lock-free.
    override_scope: ArcSwap<OverrideScope>,
    /// Minimum fractional cost advantage the recommended backend must beat the
    /// static choice by before a live override fires — guards against flapping
    /// on marginal/noisy differences.
    min_advantage: f64,
    /// Phase-2 exploration: counter rate-limiting how often an under-explored
    /// freshness-safe candidate is sampled (~1 in `exploration_interval`).
    explore_tick: AtomicU64,
    exploration_interval: u64,
    /// TD-170 / ADR-034 P7: a HARD per-query round-trip (predicted `get_ops`)
    /// budget. `None` = disabled (default). When set, the override fires
    /// regardless of the soft `min_advantage` byte-cost gate if the static choice
    /// exceeds the budget and a freshness-safe candidate is within it — because
    /// latency = depth × RTT, so a backend over the round-trip budget loses even
    /// when it moves fewer bytes.
    rtt_budget: Option<f64>,
    /// Frozen recommendation table — the LOCK-FREE consult surface. Read via
    /// [`ArcSwap`] (`load` + `HashMap` get) on the route hot path; written only
    /// by [`Self::recompute_locked`] (debounced on the observe path). Empty at
    /// construction → consult returns `None` → callers keep the static route.
    table: ArcSwap<RouteRecommendationTable>,
    /// Observation counter driving the debounced frozen-table recompute.
    observation_count: AtomicU64,
    /// Recompute the frozen table every Nth observation (debounce). Lower warms
    /// the consult view faster at more recompute cost; min 1 (recompute always).
    recompute_every: u64,
}

/// Parse `PROXIMADB_ROUTE_RTT_BUDGET` (a positive round-trip count) — the
/// production source for the TD-170 round-trip budget. Default `None` (disabled).
fn rtt_budget_from_env() -> Option<f64> {
    std::env::var("PROXIMADB_ROUTE_RTT_BUDGET")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|b| *b > 0.0)
}

impl Default for RouteCostModel {
    fn default() -> Self {
        Self::new()
    }
}

impl RouteCostModel {
    /// EWMA α=0.2, default neutral weights, 3-sample warmup, override OFF,
    /// 15% min advantage before an override fires.
    pub fn new() -> Self {
        Self {
            cells: Mutex::new(HashMap::new()),
            alpha: 0.2,
            weights: CostWeights::default(),
            min_samples: 3,
            override_scope: ArcSwap::from_pointee(OverrideScope::Off),
            min_advantage: 0.15,
            explore_tick: AtomicU64::new(0),
            exploration_interval: 16,
            rtt_budget: rtt_budget_from_env(),
            table: ArcSwap::from_pointee(RouteRecommendationTable::default()),
            observation_count: AtomicU64::new(0),
            recompute_every: 64,
        }
    }

    /// Set the hard round-trip budget (predicted `get_ops`); `None` disables it
    /// (the default). Builder form for tests; production reads
    /// `PROXIMADB_ROUTE_RTT_BUDGET` at construction.
    pub fn with_rtt_budget(mut self, budget: Option<f64>) -> Self {
        self.rtt_budget = budget.filter(|b| *b > 0.0);
        self
    }

    /// Tune how often exploration samples an under-explored candidate (~1 in N
    /// eligible decisions). Lower = warms faster but costs more; min 1.
    pub fn with_exploration_interval(mut self, n: u64) -> Self {
        self.exploration_interval = n.max(1);
        self
    }

    /// Override the warmup threshold (samples required before a cell is trusted).
    pub fn with_min_samples(mut self, n: u64) -> Self {
        self.min_samples = n.max(1);
        self
    }

    /// Tune the minimum cost advantage (fraction in [0,1)) required to override.
    pub fn with_min_advantage(mut self, frac: f64) -> Self {
        self.min_advantage = frac.clamp(0.0, 0.99);
        self
    }

    /// Tune how often the frozen consult table is recomputed (every Nth
    /// observation). `1` rebuilds on every observe (deterministic for tests);
    /// production defaults to `64`. Min 1.
    pub fn with_recompute_every(mut self, n: u64) -> Self {
        self.recompute_every = n.max(1);
        self
    }

    /// Enable/disable live route override for EVERY class (flag-gated; default
    /// OFF). The original global-bool form, kept for tests and callers that
    /// don't stage per-class; equivalent to [`Self::set_override_scope`] with
    /// [`OverrideScope::All`] / [`OverrideScope::Off`].
    pub fn set_override_enabled(&self, on: bool) {
        self.set_override_scope(if on {
            OverrideScope::All
        } else {
            OverrideScope::Off
        });
    }

    /// Set the live-override scope (TD-EXEC-2 per-shape-class staged go-live).
    /// Enabling any scope forces a frozen-table recompute so the very first
    /// override consult reads current history, not a possibly-stale debounced
    /// snapshot.
    pub fn set_override_scope(&self, scope: OverrideScope) {
        let enabled = scope != OverrideScope::Off;
        self.override_scope.store(Arc::new(scope));
        if enabled {
            self.recompute_locked();
        }
    }

    /// Whether live override is enabled for ANY class (scope is not `Off`).
    pub fn override_active(&self) -> bool {
        !matches!(**self.override_scope.load(), OverrideScope::Off)
    }

    /// Whether live override may fire for `shape_class` under the current scope.
    pub fn override_active_for(&self, shape_class: &str) -> bool {
        self.override_scope.load().enables(shape_class)
    }

    /// The §3 unified `Cost(q)` in neutral units: read (GETs + bandwidth) +
    /// egress (KEU) + ingest/storage-write (KIU) + compute. The storage·time
    /// (KSU) term and the tenant tier multiplier are *not* per-query routing
    /// discriminators — KSU is a per-tenant aggregate the consumption meter owns,
    /// and a scalar tier multiplier cancels across candidates (slice 4 applies it
    /// only to the *reported* cost). The cache-hit offset is already captured: a
    /// footer-cache hit shows up as fewer GETs / fewer bytes in the trace.
    fn score(&self, q: CostQuantities) -> f64 {
        let mib = |bytes: f64| bytes / (1024.0 * 1024.0);
        self.weights.per_get * q.get_ops
            + self.weights.per_mib_read * mib(q.bytes_read)
            + self.weights.per_mib_egress * mib(q.egress_bytes)
            + self.weights.per_mib_written * mib(q.bytes_written)
            + self.weights.per_compute_ms * q.compute_ms
    }

    /// Fold a completed query's measured trace into the (shape-class, backend)
    /// cell. `compute_ms` is the snapshot's total across engines (the query was
    /// served by `backend`, so its compute is attributed there).
    pub fn observe(&self, shape_class: &str, backend: &ComputeBackend, snap: &IoTraceSnapshot) {
        self.observe_by_label(shape_class, &backend_label(backend), snap);
    }

    /// Like [`Self::observe`] but keyed by the backend's label string directly —
    /// the form the io_trace ingestion observer uses, since io_trace stamps a
    /// neutral label, not a `ComputeBackend` (no layer-up dependency).
    pub fn observe_by_label(&self, shape_class: &str, backend_label: &str, snap: &IoTraceSnapshot) {
        // TD-EXEC-2 Slice 3 calibration loop, closed at the observe→ingest seam
        // (off the query path): the `geom=` band the route was KEYED on (an AST
        // estimate stamped into the class at decision time) vs the band of the
        // MEASURED plan geometry the trace carries (Slice 1). This makes the
        // estimator's accuracy a live, falsifiable series — divergence rate =
        // fraction of folds where the labels differ — not just an offline
        // ledger analysis. Both planner seams emit measured geometry (the
        // native `plan_instrumented` and the DataFusion adapter's logical-plan
        // walk); skipped only when the trace carries none (legacy paths).
        if snap.plan_depth > 0
            && let Some(estimated) = shape_class.split('/').find(|s| s.starts_with("geom="))
            && let Some(measured) = crate::query::compute_scheduler::GeometryClass::from_estimate(
                snap.plan_depth as u32,
                snap.plan_blocking as u32,
            )
            .class_suffix()
        {
            crate::metrics::route_metrics::record_geometry_estimate(
                estimated.trim_start_matches("geom="),
                measured.trim_start_matches("geom="),
            );
        }
        let key = (shape_class.to_string(), backend_label.to_string());
        {
            let mut cells = self.cells.lock().unwrap_or_else(|p| p.into_inner());
            cells
                .entry(key)
                .or_default()
                .fold(self.alpha, CostQuantities::from_snapshot(snap));
        }
        // Debounced recompute of the frozen consult table. Every Nth observation
        // rebuilds it (O(cells), tiny); the consult path stays lock-free between
        // recomputes. Runs inline on the (less latency-sensitive) query-completion
        // path — no background thread, no shutdown wiring.
        let n = self.observation_count.fetch_add(1, Ordering::Relaxed);
        if (n + 1).is_multiple_of(self.recompute_every) {
            self.recompute_locked();
        }
    }

    /// The `(shape-class, backend-label)` keys that have learned history. Eval/
    /// introspection support: lets a test discover which shape-classes a workload
    /// exercised so it can warm a challenger backend and force the override to
    /// flip a route (see the TD-ROUTE-1 pgwire override eval).
    pub fn learned_cell_keys(&self) -> Vec<(String, String)> {
        let cells = self.cells.lock().unwrap_or_else(|p| p.into_inner());
        cells
            .iter()
            .filter(|(_, c)| c.samples > 0)
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Current learned estimate for one (shape-class, backend), if any history.
    pub fn estimate(&self, shape_class: &str, backend: &ComputeBackend) -> Option<RouteCost> {
        self.estimate_by_label(shape_class, &backend_label(backend))
    }

    /// Current learned estimate for an arbitrary bounded access-path label.
    /// Vector exact/ANN observations share the same persistence and EWMA
    /// machinery without becoming peer `ComputeBackend` variants.
    pub fn estimate_by_label(&self, shape_class: &str, route_label: &str) -> Option<RouteCost> {
        let key = (shape_class.to_string(), route_label.to_string());
        let cells = self.cells.lock().unwrap_or_else(|p| p.into_inner());
        let c = cells.get(&key)?;
        if c.samples == 0 {
            return None;
        }
        Some(RouteCost {
            samples: c.samples,
            get_ops: c.get_ops,
            bytes_read: c.bytes_read,
            egress_bytes: c.egress_bytes,
            bytes_written: c.bytes_written,
            compute_ms: c.compute_ms,
            score: self.score(c.quantities()),
        })
    }

    /// Among `candidates`, recommend the cheapest backend whose cell has reached
    /// the warmup threshold. Returns `None` when no candidate has enough history
    /// (the scheduler then keeps its static decision). Ties keep candidate order.
    pub fn recommend(
        &self,
        shape_class: &str,
        candidates: &[ComputeBackend],
    ) -> Option<RouteRecommendation> {
        self.recommend_filtered(shape_class, candidates, None)
    }

    /// Like [`recommend`] but, when `max_rtt` is `Some`, only considers candidates
    /// whose predicted `get_ops` is within that round-trip budget (TD-170 hard
    /// admission filter) — used to pick the cheapest *within-budget* backend.
    fn recommend_filtered(
        &self,
        shape_class: &str,
        candidates: &[ComputeBackend],
        max_rtt: Option<f64>,
    ) -> Option<RouteRecommendation> {
        let mut scored: Vec<(ComputeBackend, RouteCost)> = candidates
            .iter()
            .filter_map(|b| {
                self.estimate(shape_class, b)
                    .filter(|c| c.samples >= self.min_samples)
                    .filter(|c| max_rtt.is_none_or(|budget| c.get_ops <= budget))
                    .map(|c| (b.clone(), c))
            })
            .collect();
        if scored.is_empty() {
            return None;
        }
        // Stable: sort by score, ties keep input order (sort_by is stable).
        scored.sort_by(|a, b| {
            a.1.score
                .partial_cmp(&b.1.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let ranked = scored
            .iter()
            .map(|(b, c)| (backend_label(b), c.score))
            .collect::<Vec<_>>();
        let (backend, cost) = scored.into_iter().next()?;
        Some(RouteRecommendation {
            backend,
            score: cost.score,
            samples: cost.samples,
            ranked,
        })
    }

    /// Live-override recommendation. Returns `Some(better)` ONLY when ALL hold:
    /// override is enabled; a candidate other than `static_backend` is the
    /// cheapest with warmed history; the static choice itself has warmed history
    /// (so we have evidence to beat, not a cold guess); and the challenger is
    /// cheaper by at least `min_advantage`. Otherwise `None` — the scheduler
    /// keeps its static route. The freshness-safe `candidates` set is chosen by
    /// the scheduler; this method only ranks cost, so it can never propose a
    /// backend the caller didn't deem correct for the query.
    pub fn recommend_override(
        &self,
        shape_class: &str,
        static_backend: &ComputeBackend,
        candidates: &[ComputeBackend],
    ) -> Option<RouteRecommendation> {
        if !self.override_active_for(shape_class) {
            return None;
        }
        let static_cost = self.estimate(shape_class, static_backend)?;
        if static_cost.samples < self.min_samples {
            return None; // not enough evidence about the static route to beat it
        }
        // TD-170: the round-trip budget is a HARD admission constraint, checked
        // BEFORE the soft byte-cost gate — and even when the static route is
        // otherwise the byte-cheapest. If the static choice's predicted round-trips
        // exceed the budget, route to the cheapest freshness-safe candidate WITHIN
        // the budget: a backend over the round-trip budget loses even when it moves
        // fewer bytes (latency = depth × RTT), regardless of `min_advantage`.
        if let Some(budget) = self.rtt_budget
            && static_cost.get_ops > budget
            && let Some(within) = self.recommend_filtered(shape_class, candidates, Some(budget))
            && backend_label(&within.backend) != backend_label(static_backend)
        {
            return Some(within);
        }
        // Soft byte-cost gate: flip only to a candidate beating the static route by
        // at least `min_advantage`.
        let rec = self.recommend(shape_class, candidates)?;
        if backend_label(&rec.backend) == backend_label(static_backend) {
            return None; // the model already agrees with the static choice
        }
        if rec.score < static_cost.score * (1.0 - self.min_advantage) {
            Some(rec)
        } else {
            None // advantage too small — don't flap
        }
    }

    /// Phase-2 exploration pick (warm-up). When override is enabled and a
    /// freshness-safe candidate is still under-explored (`< min_samples`),
    /// occasionally route to the LEAST-sampled candidate so it accrues cost
    /// history — otherwise the static rule always picks one engine, the other
    /// never warms, and `recommend_override` could never fire. Properties:
    ///
    /// * **Bounded** — returns `None` once every candidate is warm (exploration
    ///   then yields to exploitation).
    /// * **Rate-limited** — fires ~1 in `exploration_interval` eligible
    ///   decisions, so the cost of probing a non-optimal engine stays small.
    /// * **Freshness-safe** — the caller passes the freshness-safe candidate set,
    ///   so exploration can never target an engine that would serve the query
    ///   incorrectly (OLTP gets a single-candidate set → always `None`).
    /// * **Flag-gated** — `None` unless override is enabled.
    pub fn exploration_choice(
        &self,
        shape_class: &str,
        candidates: &[ComputeBackend],
    ) -> Option<ComputeBackend> {
        if !self.override_active_for(shape_class) || candidates.len() < 2 {
            return None;
        }
        let (cand, samples) = candidates
            .iter()
            .map(|c| {
                let s = self
                    .estimate(shape_class, c)
                    .map(|e| e.samples)
                    .unwrap_or(0);
                (c.clone(), s)
            })
            .min_by_key(|(_, s)| *s)?;
        if samples >= self.min_samples {
            return None; // all candidates warm — exploit, don't explore
        }
        let tick = self.explore_tick.fetch_add(1, Ordering::Relaxed);
        if tick.is_multiple_of(self.exploration_interval) {
            Some(cand)
        } else {
            None
        }
    }
}

impl RouteConsult for RouteCostModel {
    fn consult(&self, shape_class: &str) -> Option<ClassRecommendation> {
        // Single ArcSwap load + HashMap get — the only consult-path cost. The
        // returned ClassRecommendation is cloned out (cheap: a small Vec) so the
        // Guard is released immediately; the learn-path mutex is never touched.
        self.table.load().by_class.get(shape_class).cloned()
    }
    fn override_active(&self) -> bool {
        RouteCostModel::override_active(self)
    }
    fn override_active_for(&self, shape_class: &str) -> bool {
        RouteCostModel::override_active_for(self, shape_class)
    }
    fn next_exploration_tick(&self) -> u64 {
        self.explore_tick.fetch_add(1, Ordering::Relaxed)
    }
    fn min_samples(&self) -> u64 {
        self.min_samples
    }
    fn min_advantage(&self) -> f64 {
        self.min_advantage
    }
    fn exploration_interval(&self) -> u64 {
        self.exploration_interval
    }
    fn rtt_budget(&self) -> Option<f64> {
        self.rtt_budget
    }
}

impl RouteCostModel {
    /// Force a frozen-table recompute now (tests / a future EXPLAIN nudge).
    /// Production relies on the debounced recompute inside [`Self::observe_by_label`].
    pub fn recompute(&self) {
        self.recompute_locked();
    }

    /// Rebuild the frozen [`RouteRecommendationTable`] from the current `cells`
    /// and publish it via [`ArcSwap::store`]. The SOLE writer of the frozen
    /// table — the consult path only reads it. O(cells) (tiny: ≤ ~16 classes ×
    /// 2 backends). Locks `cells` once for the read.
    fn recompute_locked(&self) {
        let cells = self.cells.lock().unwrap_or_else(|p| p.into_inner());
        // Group observed cells by shape-class (the first tuple element).
        let mut by_class: HashMap<String, Vec<(String, Cell)>> = HashMap::new();
        for ((class, label), cell) in cells.iter() {
            by_class
                .entry(class.clone())
                .or_default()
                .push((label.clone(), *cell));
        }
        let mut table = RouteRecommendationTable {
            by_class: HashMap::new(),
        };
        for (class, entries) in by_class {
            let safe = freshness_safe_backends_from_class(&class);
            // ranked: freshness-safe candidates with ANY history, cheapest-first.
            let mut ranked: Vec<BackendChoice> = entries
                .iter()
                .filter_map(|(label, cell)| {
                    let backend = parse_backend_label(label)?;
                    let is_safe = safe.iter().any(|s| backend_label(s) == *label);
                    if !is_safe || cell.samples == 0 {
                        return None;
                    }
                    Some(BackendChoice {
                        backend,
                        score: self.score(cell.quantities()),
                        samples: cell.samples,
                        get_ops: cell.get_ops,
                    })
                })
                .collect();
            ranked.sort_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            // explore_target: least-sampled freshness-safe candidate still under
            // min_samples. Only meaningful when ≥2 safe candidates exist; a
            // never-observed safe candidate (0 samples) is a valid target.
            let explore_target = if safe.len() < 2 {
                None
            } else {
                safe.iter()
                    .map(|b| {
                        let s = entries
                            .iter()
                            .find(|(l, _)| backend_label(b) == *l)
                            .map(|(_, c)| c.samples)
                            .unwrap_or(0);
                        (b.clone(), s)
                    })
                    .filter(|(_, s)| *s < self.min_samples)
                    .min_by_key(|(_, s)| *s)
                    .map(|(b, _)| b)
            };
            table.by_class.insert(
                class,
                ClassRecommendation {
                    ranked,
                    explore_target,
                },
            );
        }
        self.table.store(Arc::new(table));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_cost_cells_use_route_time_cache_policy_not_completed_outcome() {
        use crate::observability::io_trace::{
            VectorAccessPath, VectorCachePolicy, VectorSearchIntent, VectorStorageScope,
        };

        let enabled = VectorAccessTrace {
            engine: "sst".to_string(),
            dimensions: 384,
            top_k: 10,
            has_filter: false,
            requested_mode: VectorSearchIntent::Exact,
            actual_path: VectorAccessPath::Exact,
            storage_scope: VectorStorageScope::Remote,
            cache_policy: VectorCachePolicy::Full,
        };
        let cold = IoTraceSnapshot {
            get_ops: 1,
            footer_misses: 2,
            l2_misses: 2,
            ..Default::default()
        };
        let warm = IoTraceSnapshot {
            footer_hits: 2,
            l2_hits: 2,
            ..Default::default()
        };

        assert_eq!(
            vector_access_shape(&enabled),
            "vector/engine=sst/dim=384/k=10/filter=no/storage=remote/cache_policy=full"
        );
        assert_eq!(
            VectorCacheOutcome::from_snapshot(&cold),
            VectorCacheOutcome::Cold
        );
        assert_eq!(
            VectorCacheOutcome::from_snapshot(&warm),
            VectorCacheOutcome::Warm
        );

        let ann = VectorAccessTrace {
            requested_mode: VectorSearchIntent::Approximate,
            actual_path: VectorAccessPath::Ann,
            ..enabled.clone()
        };
        assert_eq!(
            vector_access_shape(&enabled),
            vector_access_shape(&ann),
            "caller intent and completed access path must not split the paired cell"
        );
        let model = RouteCostModel::new();
        assert!(observe_vector_access_snapshot(
            &model,
            &IoTraceSnapshot {
                vector_accesses: vec![enabled.clone()],
                ..cold.clone()
            }
        ));
        assert!(observe_vector_access_snapshot(
            &model,
            &IoTraceSnapshot {
                vector_accesses: vec![ann],
                ..warm.clone()
            }
        ));
        let shape = vector_access_shape(&enabled);
        assert!(model.estimate_by_label(&shape, "exact").is_some());
        assert!(model.estimate_by_label(&shape, "ann").is_some());

        let disabled = VectorAccessTrace {
            cache_policy: VectorCachePolicy::Disabled,
            ..enabled.clone()
        };
        assert_ne!(
            vector_access_shape(&disabled),
            vector_access_shape(&enabled),
            "route-time cache policies must not train one pooled cell"
        );
    }

    #[test]
    fn vector_cache_outcome_uses_physical_io_as_authority() {
        let warm = IoTraceSnapshot {
            survivor_l1_hits: 2,
            survivor_l1_misses: 1,
            l2_misses: 1,
            ..Default::default()
        };
        assert_eq!(
            VectorCacheOutcome::from_snapshot(&warm),
            VectorCacheOutcome::Warm
        );

        let mixed = IoTraceSnapshot {
            get_ops: 1,
            survivor_l1_hits: 1,
            survivor_l1_misses: 1,
            ..Default::default()
        };
        assert_eq!(
            VectorCacheOutcome::from_snapshot(&mixed),
            VectorCacheOutcome::Mixed
        );

        let cold = IoTraceSnapshot {
            get_ops: 1,
            survivor_l1_misses: 1,
            l2_misses: 1,
            ..Default::default()
        };
        assert_eq!(
            VectorCacheOutcome::from_snapshot(&cold),
            VectorCacheOutcome::Cold
        );

        let no_activity = IoTraceSnapshot::default();
        assert_eq!(
            VectorCacheOutcome::from_snapshot(&no_activity),
            VectorCacheOutcome::Unknown
        );

        let uncached_read = IoTraceSnapshot {
            get_ops: 1,
            ..Default::default()
        };
        assert_eq!(
            VectorCacheOutcome::from_snapshot(&uncached_read),
            if cfg!(feature = "io-trace") {
                VectorCacheOutcome::Bypass
            } else {
                VectorCacheOutcome::Unknown
            }
        );
    }

    /// TD-PAXRG-1 closeout: a `Disabled` route-time cache policy is stamped
    /// always-on from the real engine cache handles — physical GETs under it
    /// are a KNOWN uncached regime, so the outcome is `Bypass`
    /// feature-independently (previously feature-off builds reported
    /// `unknown`, which made every cache-free paired-cost cohort inadmissible).
    #[test]
    fn disabled_policy_with_gets_is_bypass() {
        use crate::observability::io_trace::{
            VectorAccessPath, VectorCachePolicy, VectorSearchIntent, VectorStorageScope,
        };

        let access = |policy: VectorCachePolicy| VectorAccessTrace {
            engine: "sst".to_string(),
            dimensions: 128,
            top_k: 10,
            has_filter: false,
            requested_mode: VectorSearchIntent::Approximate,
            actual_path: VectorAccessPath::Ann,
            storage_scope: VectorStorageScope::Local,
            cache_policy: policy,
        };

        // Disabled policy + physical GETs ⇒ Bypass, with and without io-trace.
        let disabled = IoTraceSnapshot {
            vector_accesses: vec![access(VectorCachePolicy::Disabled)],
            get_ops: 1,
            ..Default::default()
        };
        assert_eq!(
            VectorCacheOutcome::from_snapshot(&disabled),
            VectorCacheOutcome::Bypass
        );

        // Multi-access: Disabled everywhere ⇒ still a known uncached regime.
        let disabled_multi = IoTraceSnapshot {
            vector_accesses: vec![
                access(VectorCachePolicy::Disabled),
                access(VectorCachePolicy::Disabled),
            ],
            get_ops: 3,
            ..Default::default()
        };
        assert_eq!(
            VectorCacheOutcome::from_snapshot(&disabled_multi),
            VectorCacheOutcome::Bypass
        );

        // One Unknown-policy access ⇒ policy evidence incomplete ⇒ the
        // feature-off build stays Unknown (fail-closed preserved).
        let mixed_policy = IoTraceSnapshot {
            vector_accesses: vec![
                access(VectorCachePolicy::Disabled),
                access(VectorCachePolicy::Unknown),
            ],
            get_ops: 2,
            ..Default::default()
        };
        assert_eq!(
            VectorCacheOutcome::from_snapshot(&mixed_policy),
            if cfg!(feature = "io-trace") {
                VectorCacheOutcome::Bypass
            } else {
                VectorCacheOutcome::Unknown
            }
        );
    }

    #[test]
    fn unknown_vector_route_dimensions_or_outcome_do_not_train_cost_model() {
        use crate::observability::io_trace::{
            VectorAccessPath, VectorCachePolicy, VectorSearchIntent,
        };

        let model = RouteCostModel::new();
        let unknown_policy = IoTraceSnapshot {
            vector_accesses: vec![VectorAccessTrace {
                engine: "sst".to_string(),
                dimensions: 384,
                top_k: 10,
                has_filter: false,
                requested_mode: VectorSearchIntent::Exact,
                actual_path: VectorAccessPath::Exact,
                storage_scope: crate::observability::io_trace::VectorStorageScope::Local,
                cache_policy: VectorCachePolicy::Unknown,
            }],
            get_ops: 1,
            ..Default::default()
        };

        assert!(!observe_vector_access_snapshot(&model, &unknown_policy));
        let unknown_outcome = IoTraceSnapshot {
            vector_accesses: vec![VectorAccessTrace {
                cache_policy: VectorCachePolicy::Disabled,
                ..unknown_policy.vector_accesses[0].clone()
            }],
            ..Default::default()
        };
        assert!(!observe_vector_access_snapshot(&model, &unknown_outcome));
        assert!(model.learned_cell_keys().is_empty());
    }

    #[test]
    fn persist_load_round_trip_warms_a_fresh_model() {
        let m = RouteCostModel::new();
        // Learn two cells from measured snapshots (typed observe so the backend
        // labels match what `estimate` derives).
        m.observe(
            "olap/parquet/large",
            &ComputeBackend::DataFusionLocal,
            &snap(12, 4_000_000, 8),
        );
        m.observe(
            "olap/parquet/large",
            &ComputeBackend::Native,
            &snap(40, 4_000_000, 30),
        );

        let dir = std::env::temp_dir().join(format!("pdb_cost_persist_{}", std::process::id()));
        let path = cost_model_persist_path(&dir);
        assert_eq!(m.persist_to(&path).expect("persist"), 2);

        // A fresh model has no history until it loads the persisted cells.
        let fresh = RouteCostModel::new();
        assert!(
            fresh
                .estimate("olap/parquet/large", &ComputeBackend::DataFusionLocal)
                .is_none()
        );
        assert_eq!(fresh.load_from(&path), 2);
        let est = fresh
            .estimate("olap/parquet/large", &ComputeBackend::DataFusionLocal)
            .expect("warmed estimate");
        assert_eq!(est.get_ops, 12.0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_missing_file_is_a_noop() {
        let m = RouteCostModel::new();
        assert_eq!(
            m.load_from(std::path::Path::new(
                "/nonexistent/pdb/route_cost_cells.json"
            )),
            0
        );
    }

    fn snap(get_ops: u64, bytes_read: u64, compute_ms: u64) -> IoTraceSnapshot {
        let mut s = IoTraceSnapshot {
            get_ops,
            bytes_read,
            ..Default::default()
        };
        if compute_ms > 0 {
            s.compute_ms.insert("engine".to_string(), compute_ms);
        }
        s
    }

    #[test]
    fn cost_quantities_carry_split_and_wait_observability() {
        // ADR-056 AQE-S2: splits_pruned + the runtime-filter wait outcome flow
        // from IoTraceSnapshot → CostQuantities (diagnostic observability for
        // the route model; NOT in the score fn).
        let mut snap = IoTraceSnapshot::default();
        snap.splits_total = 100;
        snap.splits_pruned = 37;
        snap.runtime_filter_arrived = 4;
        snap.runtime_filter_timed_out = 1;
        snap.runtime_filter_wait_ms = 4_200;
        let q = CostQuantities::from_snapshot(&snap);
        assert_eq!(q.splits_total, 100.0);
        assert_eq!(q.splits_pruned, 37.0);
        assert_eq!(q.runtime_filter_arrived, 4.0);
        assert_eq!(q.runtime_filter_timed_out, 1.0);
        assert_eq!(q.runtime_filter_wait_ms, 4_200.0);
    }

    #[test]
    fn whole_object_get_is_a_cost_bearing_round_trip() {
        let model = RouteCostModel::new().with_min_samples(1);
        let snapshot = IoTraceSnapshot {
            get_ops: 1,
            bytes_read: 1 << 20,
            ..Default::default()
        };
        model.observe("vector/sst/exact", &ComputeBackend::Native, &snapshot);

        let estimate = model
            .estimate("vector/sst/exact", &ComputeBackend::Native)
            .expect("whole-object read creates a learned cell");
        let weights = CostWeights::default();
        assert_eq!(estimate.get_ops, 1.0);
        assert!((estimate.score - (weights.per_get + weights.per_mib_read)).abs() < 1e-6);
    }

    #[test]
    fn shape_class_mirrors_scheduler_signals() {
        // Unknown finer signals (the default) keep the coarse 2-part class.
        assert_eq!(
            shape_class(&QueryShape {
                engages_relational: true,
                parquet_backed: true,
                ..Default::default()
            }),
            "olap/parquet"
        );
        assert_eq!(
            shape_class(&QueryShape {
                engages_relational: false,
                parquet_backed: false,
                ..Default::default()
            }),
            "oltp/native"
        );
    }

    #[test]
    fn shape_class_refines_with_known_cardinality_and_partition_signals() {
        use crate::query::compute_scheduler::{CardinalityClass, PartitionFanout};
        // Known finer signals append stable suffixes (cardinality then partition).
        let class = shape_class(&QueryShape {
            engages_relational: true,
            parquet_backed: true,
            cardinality: CardinalityClass::Large,
            partition_fanout: PartitionFanout::Many,
            pax_backed: false,
            native_registerable: false,
            operation_class: Default::default(),
            geometry: Default::default(),
            join_bearing: false,
        });
        assert_eq!(class, "olap/parquet/card=l/part=m");
        // A partially-known shape only appends the known suffix.
        let partial = shape_class(&QueryShape {
            engages_relational: false,
            parquet_backed: false,
            cardinality: CardinalityClass::Small,
            ..Default::default()
        });
        assert_eq!(partial, "oltp/native/card=s");
    }

    #[test]
    fn shape_class_appends_geometry_tier_when_known() {
        use crate::query::compute_scheduler::GeometryClass;
        // TD-EXEC-2 Slice 3: a known geometry estimate appends the geom tier
        // after every other suffix, so cells warm per shape⊕geometry class.
        let class = shape_class(&QueryShape {
            engages_relational: true,
            parquet_backed: true,
            geometry: GeometryClass::from_estimate(8, 4),
            ..Default::default()
        });
        assert_eq!(class, "olap/parquet/geom=mxhi");
        // Unknown geometry contributes nothing — warmed cells stay valid.
        let coarse = shape_class(&QueryShape {
            engages_relational: true,
            parquet_backed: true,
            ..Default::default()
        });
        assert_eq!(coarse, "olap/parquet");
        // The tier composes with the other refinements in stable order.
        let full = shape_class(&QueryShape {
            engages_relational: true,
            parquet_backed: true,
            cardinality: crate::query::compute_scheduler::CardinalityClass::Large,
            operation_class: crate::query::compute_scheduler::OperationClass::Grouped,
            geometry: GeometryClass::from_estimate(3, 1),
            ..Default::default()
        });
        assert_eq!(full, "olap/parquet/card=l/op=grp/geom=sxlo");
    }

    #[test]
    fn geometry_bands_bucket_at_the_documented_cutoffs() {
        use crate::query::compute_scheduler::{BlockingBand, DepthBand, GeometryClass};
        // Depth: ≤6 shallow, 7–12 mid, >12 deep. Blocking: 0 / 1–2 / ≥3.
        assert_eq!(
            GeometryClass::from_estimate(6, 0),
            GeometryClass::Known {
                depth: DepthBand::Shallow,
                blocking: BlockingBand::Zero,
            }
        );
        assert_eq!(
            GeometryClass::from_estimate(7, 2),
            GeometryClass::Known {
                depth: DepthBand::Mid,
                blocking: BlockingBand::Low,
            }
        );
        assert_eq!(
            GeometryClass::from_estimate(13, 3),
            GeometryClass::Known {
                depth: DepthBand::Deep,
                blocking: BlockingBand::High,
            }
        );
    }

    #[test]
    fn consult_falls_back_to_the_warmed_ancestor_class() {
        // TD-EXEC-2 Slice 3 / ADR-058 D3: refining the key must not orphan
        // warmed learning. Warm only the coarse base class; a query keyed with
        // the geometry tier must still be served — by the ancestor's cells —
        // and report which level served it.
        let m = RouteCostModel::new()
            .with_min_samples(1)
            .with_recompute_every(1);
        m.observe(
            "olap/parquet",
            &ComputeBackend::DataFusionLocal,
            &snap(4, 4096, 2),
        );
        // Refined key has no cell of its own → ancestor serves.
        let (rec, level) = m
            .consult_with_fallback("olap/parquet/card=l/geom=mxhi")
            .expect("ancestor cells must serve the refined class");
        assert_eq!(level, "olap/parquet");
        assert!(!rec.ranked().is_empty());
        // Direct (non-fallback) consult still misses the refined key.
        assert!(m.consult("olap/parquet/card=l/geom=mxhi").is_none());
        // Once the refined class warms, it is preferred over the ancestor.
        m.observe(
            "olap/parquet/card=l/geom=mxhi",
            &ComputeBackend::Native,
            &snap(2, 1024, 1),
        );
        let (_, level) = m
            .consult_with_fallback("olap/parquet/card=l/geom=mxhi")
            .expect("refined cells must now serve");
        assert_eq!(level, "olap/parquet/card=l/geom=mxhi");
        // A class with no warmed ancestor at any level misses entirely.
        assert!(m.consult_with_fallback("oltp/native/geom=sx0").is_none());
    }

    #[test]
    fn observe_emits_the_geometry_calibration_sample() {
        // The observe→ingest seam compares the `geom=` band the class was keyed
        // on against the measured plan geometry in the trace. mxhi estimated;
        // depth 3 / blocking 1 measured → sxlo (a divergence sample).
        let before = crate::metrics::route_metrics::ROUTE_GEOMETRY_ESTIMATE_TOTAL
            .with_label_values(&["mxhi", "sxlo"])
            .get();
        let mut s = snap(2, 1024, 1);
        s.plan_depth = 3;
        s.plan_blocking = 1;
        let m = RouteCostModel::new();
        m.observe_by_label("olap/parquet/geom=mxhi", "Native", &s);
        let after = crate::metrics::route_metrics::ROUTE_GEOMETRY_ESTIMATE_TOTAL
            .with_label_values(&["mxhi", "sxlo"])
            .get();
        assert_eq!(after - before, 1.0);
        // No measured geometry (plan_depth 0) → no sample.
        let before = crate::metrics::route_metrics::ROUTE_GEOMETRY_ESTIMATE_TOTAL
            .with_label_values(&["mxhi", "sxlo"])
            .get();
        m.observe_by_label("olap/parquet/geom=mxhi", "Native", &snap(2, 1024, 1));
        let after = crate::metrics::route_metrics::ROUTE_GEOMETRY_ESTIMATE_TOTAL
            .with_label_values(&["mxhi", "sxlo"])
            .get();
        assert_eq!(after - before, 0.0);
    }

    #[test]
    fn cells_warm_independently_per_geometry_class() {
        use crate::query::compute_scheduler::GeometryClass;
        // Two shapes identical except for the geometry band must fold into
        // DIFFERENT cells — the whole point of the tier (TD-EXEC-2 Slice 3).
        let scan_heavy = shape_class(&QueryShape {
            engages_relational: true,
            parquet_backed: true,
            geometry: GeometryClass::from_estimate(3, 1),
            ..Default::default()
        });
        let join_heavy = shape_class(&QueryShape {
            engages_relational: true,
            parquet_backed: true,
            geometry: GeometryClass::from_estimate(9, 4),
            ..Default::default()
        });
        assert_ne!(scan_heavy, join_heavy);
        let m = RouteCostModel::new();
        m.observe(&scan_heavy, &ComputeBackend::Native, &snap(10, 1024, 5));
        assert!(
            m.estimate(&scan_heavy, &ComputeBackend::Native).is_some(),
            "observed class must have a cell"
        );
        assert!(
            m.estimate(&join_heavy, &ComputeBackend::Native).is_none(),
            "sibling geometry class must stay cold — no cross-band bleed"
        );
    }

    #[test]
    fn no_history_yields_no_estimate_or_recommendation() {
        let m = RouteCostModel::new();
        assert!(
            m.estimate("olap/parquet", &ComputeBackend::Native)
                .is_none()
        );
        assert!(
            m.recommend(
                "olap/parquet",
                &[ComputeBackend::Native, ComputeBackend::DataFusionLocal]
            )
            .is_none()
        );
    }

    #[test]
    fn warmup_threshold_gates_recommendation() {
        let m = RouteCostModel::new().with_min_samples(3);
        // Two observations — below the 3-sample warmup → not yet trusted.
        m.observe(
            "olap/parquet",
            &ComputeBackend::Native,
            &snap(100, 1 << 20, 5),
        );
        m.observe(
            "olap/parquet",
            &ComputeBackend::Native,
            &snap(100, 1 << 20, 5),
        );
        assert!(
            m.recommend("olap/parquet", &[ComputeBackend::Native])
                .is_none()
        );
        m.observe(
            "olap/parquet",
            &ComputeBackend::Native,
            &snap(100, 1 << 20, 5),
        );
        assert!(
            m.recommend("olap/parquet", &[ComputeBackend::Native])
                .is_some()
        );
    }

    #[test]
    fn recommends_the_cheaper_backend_from_measured_traces() {
        let m = RouteCostModel::new().with_min_samples(1);
        // DataFusion route pays FEW big coalesced GETs; Native pays MANY small
        // GETs for the same shape — the trace says DataFusion is cheaper here.
        for _ in 0..5 {
            m.observe(
                "olap/parquet",
                &ComputeBackend::DataFusionLocal,
                &snap(4, 16 << 20, 50),
            );
            m.observe(
                "olap/parquet",
                &ComputeBackend::Native,
                &snap(400, 16 << 20, 20),
            );
        }
        let rec = m
            .recommend(
                "olap/parquet",
                &[ComputeBackend::Native, ComputeBackend::DataFusionLocal],
            )
            .expect("history exists");
        // GET-dominated score → DataFusion (4 GETs) beats Native (400 GETs).
        assert_eq!(rec.backend, ComputeBackend::DataFusionLocal);
        assert_eq!(rec.ranked.len(), 2);
        assert_eq!(rec.ranked[0].0, "DataFusionLocal");
        // Cheapest-first ordering.
        assert!(rec.ranked[0].1 < rec.ranked[1].1);
    }

    #[test]
    fn egress_term_is_inert_on_the_free_same_region_path() {
        // A read-only route with NO cross-region bytes must score exactly the read +
        // compute terms — i.e. adding the egress dimension changes nothing on the
        // free path (default-OFF behavior; KEU is zero until bytes move cross-region).
        let m = RouteCostModel::new().with_min_samples(1);
        m.observe(
            "olap/parquet",
            &ComputeBackend::Native,
            &snap(10, 4 << 20, 7),
        );
        let est = m
            .estimate("olap/parquet", &ComputeBackend::Native)
            .expect("history");
        let w = CostWeights::default();
        let expected = w.per_get * 10.0 + w.per_mib_read * 4.0 + w.per_compute_ms * 7.0;
        assert_eq!(est.egress_bytes, 0.0);
        assert!((est.score - expected).abs() < 1e-6);
    }

    #[test]
    fn cross_region_egress_raises_the_route_cost() {
        // Two routes identical in GETs/bytes-read/compute; one also moves bytes
        // cross-region. The egress (KEU) term must make that route cost more —
        // the whole point of metering Dimension 2 into Cost(q).
        let m = RouteCostModel::new().with_min_samples(1);
        // Identical in every term except egress, so the score delta isolates it.
        let same_region = snap(4, 8 << 20, 0);
        let cross_region = IoTraceSnapshot {
            get_ops: 4,
            bytes_read: 8 << 20,
            egress_bytes: 8 << 20,
            ..Default::default()
        };
        m.observe(
            "olap/parquet",
            &ComputeBackend::DataFusionLocal,
            &same_region,
        );
        m.observe("olap/parquet", &ComputeBackend::Native, &cross_region);
        let cheap = m
            .estimate("olap/parquet", &ComputeBackend::DataFusionLocal)
            .expect("history");
        let dear = m
            .estimate("olap/parquet", &ComputeBackend::Native)
            .expect("history");
        assert!(dear.egress_bytes > 0.0 && cheap.egress_bytes == 0.0);
        // The cross-region route is dearer by exactly the egress weight × 8 MiB.
        let delta = dear.score - cheap.score;
        assert!((delta - CostWeights::default().per_mib_egress * 8.0).abs() < 1e-6);
        // And the trace-driven recommendation prefers the same-region route.
        let rec = m
            .recommend(
                "olap/parquet",
                &[ComputeBackend::Native, ComputeBackend::DataFusionLocal],
            )
            .expect("history");
        assert_eq!(rec.backend, ComputeBackend::DataFusionLocal);
    }

    #[test]
    fn ingest_write_term_costs_storage_writes() {
        // The KIU storage-write term folds bytes_written into Cost(q) (inert for
        // read-only routes, active once a route induces writes).
        let m = RouteCostModel::new().with_min_samples(1);
        let written = IoTraceSnapshot {
            get_ops: 0,
            bytes_written: 2 << 20,
            ..Default::default()
        };
        m.observe("oltp/native", &ComputeBackend::Native, &written);
        let est = m
            .estimate("oltp/native", &ComputeBackend::Native)
            .expect("history");
        assert!((est.bytes_written - (2 << 20) as f64).abs() < 1.0);
        assert!((est.score - CostWeights::default().per_mib_written * 2.0).abs() < 1e-6);
    }

    #[test]
    fn estimate_reflects_ewma_of_observations() {
        let m = RouteCostModel::new();
        for _ in 0..10 {
            m.observe("oltp/native", &ComputeBackend::Native, &snap(2, 8192, 1));
        }
        let est = m
            .estimate("oltp/native", &ComputeBackend::Native)
            .expect("history");
        assert_eq!(est.samples, 10);
        // EWMA of a constant series converges to that constant.
        assert!((est.get_ops - 2.0).abs() < 1e-6);
        assert!(est.score > 0.0);
    }

    #[test]
    fn observe_by_label_matches_typed_observe() {
        // The label-keyed ingestion form (used by the io_trace observer) lands in
        // the same cell as the typed observe.
        let m = RouteCostModel::new();
        m.observe_by_label("olap/parquet", "DataFusionLocal", &snap(4, 16 << 20, 50));
        let by_label = m
            .estimate("olap/parquet", &ComputeBackend::DataFusionLocal)
            .expect("label-keyed observation is visible to typed estimate");
        assert_eq!(by_label.samples, 1);
        assert!((by_label.get_ops - 4.0).abs() < 1e-6);
    }

    #[test]
    fn vector_access_cost_cells_admit_only_unambiguous_single_operator_queries() {
        use crate::observability::io_trace::{
            VectorAccessPath, VectorAccessTrace, VectorCachePolicy, VectorSearchIntent,
        };

        let access = VectorAccessTrace {
            engine: "sst".to_string(),
            dimensions: 384,
            top_k: 10,
            has_filter: false,
            requested_mode: VectorSearchIntent::Adaptive,
            actual_path: VectorAccessPath::Exact,
            storage_scope: crate::observability::io_trace::VectorStorageScope::Local,
            cache_policy: VectorCachePolicy::Disabled,
        };
        let model = RouteCostModel::new();
        let single = IoTraceSnapshot {
            get_ops: 2,
            bytes_read: 8 << 20,
            footer_misses: 1,
            vector_accesses: vec![access.clone()],
            ..Default::default()
        };
        assert!(observe_vector_access_snapshot(&model, &single));
        let shape = vector_access_shape(&access);
        let estimate = model
            .estimate_by_label(&shape, "exact")
            .expect("single access is attributable");
        assert_eq!(estimate.samples, 1);
        assert_eq!(estimate.get_ops, 2.0);

        let ambiguous = IoTraceSnapshot {
            get_ops: 99,
            vector_accesses: vec![access.clone(), access],
            ..Default::default()
        };
        assert!(!observe_vector_access_snapshot(&model, &ambiguous));
        assert_eq!(
            model
                .estimate_by_label(&shape, "exact")
                .expect("original cell remains")
                .samples,
            1,
            "aggregate query cost must not be assigned to either of two operators"
        );
    }

    #[test]
    fn global_model_is_usable() {
        // Feed the process-global model directly under a unique key (avoids the
        // observer global, so no cross-test race) and read it back.
        let key = "test/global-usable-unique";
        for _ in 0..3 {
            GLOBAL_ROUTE_COST_MODEL.observe_by_label(key, "Native(Volcano)", &snap(2, 4096, 1));
        }
        assert!(
            GLOBAL_ROUTE_COST_MODEL
                .estimate(key, &ComputeBackend::Native)
                .is_some()
        );
    }

    /// Warm a model so DataFusion is much cheaper than Native for "olap/parquet".
    fn warm_df_cheaper(m: &RouteCostModel) {
        for _ in 0..5 {
            m.observe_by_label("olap/parquet", "DataFusionLocal", &snap(4, 16 << 20, 50));
            m.observe_by_label("olap/parquet", "Native(Volcano)", &snap(400, 16 << 20, 20));
        }
    }

    #[test]
    fn override_respects_the_enable_flag() {
        let m = RouteCostModel::new().with_min_samples(1);
        warm_df_cheaper(&m);
        let cands = [ComputeBackend::Native, ComputeBackend::DataFusionLocal];
        // Default OFF → no override even with a huge advantage.
        assert!(
            m.recommend_override("olap/parquet", &ComputeBackend::Native, &cands)
                .is_none()
        );
        m.set_override_enabled(true);
        let rec = m
            .recommend_override("olap/parquet", &ComputeBackend::Native, &cands)
            .expect("confident, enabled → override");
        assert_eq!(rec.backend, ComputeBackend::DataFusionLocal);
    }

    #[test]
    fn override_scope_parses_global_class_list_and_garbage_forms() {
        // Global bool forms (backward-compatible).
        assert_eq!(OverrideScope::parse("1"), OverrideScope::All);
        assert_eq!(OverrideScope::parse("true"), OverrideScope::All);
        assert_eq!(OverrideScope::parse("TRUE"), OverrideScope::All);
        assert_eq!(OverrideScope::parse(""), OverrideScope::Off);
        assert_eq!(OverrideScope::parse("0"), OverrideScope::Off);
        assert_eq!(OverrideScope::parse("false"), OverrideScope::Off);
        // Per-class staged form: comma-separated class prefixes.
        assert_eq!(
            OverrideScope::parse("olap/parquet/op=agg, olap/parquet/op=cnt"),
            OverrideScope::Classes(vec![
                "olap/parquet/op=agg".to_string(),
                "olap/parquet/op=cnt".to_string(),
            ])
        );
        // Malformed tokens (no `/`) are dropped; all-malformed fails safe OFF —
        // a typo must never silently become a global enable.
        assert_eq!(OverrideScope::parse("yes"), OverrideScope::Off);
        assert_eq!(OverrideScope::parse("olap,oltp"), OverrideScope::Off);
        assert_eq!(
            OverrideScope::parse("garbage, olap/parquet"),
            OverrideScope::Classes(vec!["olap/parquet".to_string()])
        );
    }

    #[test]
    fn override_scope_enables_on_slash_boundaries_only() {
        let scope = OverrideScope::Classes(vec!["olap/parquet/op=agg".to_string()]);
        // Exact match and refinements are enabled…
        assert!(scope.enables("olap/parquet/op=agg"));
        assert!(scope.enables("olap/parquet/op=agg/geom=sx0"));
        // …siblings, ancestors, and lexical near-misses are not.
        assert!(!scope.enables("olap/parquet/op=grp"));
        assert!(!scope.enables("olap/parquet"));
        assert!(!scope.enables("olap/parquet/op=agg2"));
        assert!(OverrideScope::All.enables("anything/at/all"));
        assert!(!OverrideScope::Off.enables("olap/parquet"));
    }

    #[test]
    fn class_scoped_override_flips_only_matching_classes() {
        // TD-EXEC-2 per-class staged go-live: warm identical evidence under two
        // classes; scope the override to one. Only that class may flip — the
        // other stays observe-only, and exploration is scoped identically.
        let m = RouteCostModel::new()
            .with_min_samples(1)
            .with_exploration_interval(1);
        let enabled_class = "olap/parquet/op=agg/geom=sx0";
        let disabled_class = "olap/parquet/op=grp/geom=mxhi";
        for class in [enabled_class, disabled_class] {
            warm_class_df_cheaper(&m, class);
        }
        m.set_override_scope(OverrideScope::Classes(vec![
            "olap/parquet/op=agg".to_string(),
        ]));
        assert!(m.override_active(), "a class scope counts as active");
        let cands = [ComputeBackend::Native, ComputeBackend::DataFusionLocal];
        let rec = m
            .recommend_override(enabled_class, &ComputeBackend::Native, &cands)
            .expect("scoped-in class must flip");
        assert_eq!(rec.backend, ComputeBackend::DataFusionLocal);
        assert!(
            m.recommend_override(disabled_class, &ComputeBackend::Native, &cands)
                .is_none(),
            "scoped-out class must stay observe-only"
        );
        // Exploration honors the same scope: cold candidate under the disabled
        // class is never probed.
        let cold_out = "olap/parquet/op=grp/geom=dxhi";
        m.observe_by_label(cold_out, "Native(Volcano)", &snap(2, 4096, 1));
        assert!(m.exploration_choice(cold_out, &cands).is_none());
        let cold_in = "olap/parquet/op=agg/geom=dxhi";
        m.observe_by_label(cold_in, "Native(Volcano)", &snap(2, 4096, 1));
        assert_eq!(
            m.exploration_choice(cold_in, &cands),
            Some(ComputeBackend::DataFusionLocal)
        );
    }

    /// Warm `class` so DataFusion is much cheaper than Native.
    fn warm_class_df_cheaper(m: &RouteCostModel, class: &str) {
        for _ in 0..5 {
            m.observe_by_label(class, "DataFusionLocal", &snap(4, 16 << 20, 50));
            m.observe_by_label(class, "Native(Volcano)", &snap(400, 16 << 20, 20));
        }
    }

    #[test]
    fn override_requires_min_advantage() {
        // DataFusion only ~5% cheaper than Native → below the 15% gate → no flip.
        let m = RouteCostModel::new()
            .with_min_samples(1)
            .with_min_advantage(0.15);
        m.set_override_enabled(true);
        for _ in 0..5 {
            m.observe_by_label("olap/parquet", "DataFusionLocal", &snap(95, 1 << 20, 10));
            m.observe_by_label("olap/parquet", "Native(Volcano)", &snap(100, 1 << 20, 10));
        }
        let cands = [ComputeBackend::Native, ComputeBackend::DataFusionLocal];
        assert!(
            m.recommend_override("olap/parquet", &ComputeBackend::Native, &cands)
                .is_none(),
            "marginal advantage must not flip the route"
        );
    }

    #[test]
    fn td170_rtt_budget_routes_away_from_byte_cheapest_over_budget_backend() {
        // DataFusion is byte-CHEAPEST (no bytes/compute) but blows the round-trip
        // budget (10 GETs); Native is WITHIN budget (2 GETs) but costs more total
        // bytes — so the soft byte-cost gate would never flip away from DataFusion.
        let warm = |m: &RouteCostModel| {
            for _ in 0..8 {
                m.observe_by_label("olap/parquet", "DataFusionLocal", &snap(10, 0, 0));
                m.observe_by_label("olap/parquet", "Native(Volcano)", &snap(2, 50 << 20, 0));
            }
        };
        let cands = [ComputeBackend::Native, ComputeBackend::DataFusionLocal];

        // No budget: DataFusion is byte-cheapest → recommend = static DataFusion →
        // the over-round-trip route is KEPT (the soft gate can't see the depth cost).
        let m = RouteCostModel::new().with_min_samples(1);
        m.set_override_enabled(true);
        warm(&m);
        assert_eq!(
            m.recommend("olap/parquet", &cands).unwrap().backend,
            ComputeBackend::DataFusionLocal,
            "DataFusion is the byte-cheapest backend"
        );
        assert!(
            m.recommend_override("olap/parquet", &ComputeBackend::DataFusionLocal, &cands)
                .is_none(),
            "no budget → keeps the byte-cheapest static route despite its round-trips"
        );

        // Budget = 3: DataFusion (10 GETs) blows it → HARD override to the
        // within-budget Native, even though Native costs MORE total bytes.
        let mb = RouteCostModel::new()
            .with_min_samples(1)
            .with_rtt_budget(Some(3.0));
        mb.set_override_enabled(true);
        warm(&mb);
        let over = mb
            .recommend_override("olap/parquet", &ComputeBackend::DataFusionLocal, &cands)
            .expect("RTT budget must reshape the route off the over-budget backend");
        assert_eq!(
            over.backend,
            ComputeBackend::Native,
            "RTT budget routes to the within-budget backend over the byte-cheapest one"
        );
    }

    #[test]
    fn override_none_when_static_is_already_cheapest() {
        let m = RouteCostModel::new().with_min_samples(1);
        m.set_override_enabled(true);
        warm_df_cheaper(&m); // DataFusion cheapest
        let cands = [ComputeBackend::Native, ComputeBackend::DataFusionLocal];
        // Static already DataFusion → nothing cheaper to flip to.
        assert!(
            m.recommend_override("olap/parquet", &ComputeBackend::DataFusionLocal, &cands)
                .is_none()
        );
    }

    #[test]
    fn override_needs_warmed_history_for_the_static_route() {
        // Challenger has history but the static route does not → can't prove the
        // override beats it, so no flip.
        let m = RouteCostModel::new().with_min_samples(3);
        m.set_override_enabled(true);
        for _ in 0..5 {
            m.observe_by_label("olap/parquet", "DataFusionLocal", &snap(4, 1 << 20, 5));
        }
        let cands = [ComputeBackend::Native, ComputeBackend::DataFusionLocal];
        assert!(
            m.recommend_override("olap/parquet", &ComputeBackend::Native, &cands)
                .is_none()
        );
    }

    #[test]
    fn exploration_is_off_unless_override_enabled() {
        let m = RouteCostModel::new().with_exploration_interval(1);
        // Native warm, DataFusion unexplored — but override flag is OFF.
        for _ in 0..5 {
            m.observe_by_label("olap/parquet", "Native(Volcano)", &snap(2, 4096, 1));
        }
        let cands = [ComputeBackend::Native, ComputeBackend::DataFusionLocal];
        assert!(m.exploration_choice("olap/parquet", &cands).is_none());
    }

    #[test]
    fn exploration_targets_the_least_sampled_candidate() {
        let m = RouteCostModel::new().with_exploration_interval(1);
        m.set_override_enabled(true);
        // Native warm; DataFusion has no history → it is the under-explored one.
        for _ in 0..5 {
            m.observe_by_label("olap/parquet", "Native(Volcano)", &snap(2, 4096, 1));
        }
        let cands = [ComputeBackend::Native, ComputeBackend::DataFusionLocal];
        assert_eq!(
            m.exploration_choice("olap/parquet", &cands),
            Some(ComputeBackend::DataFusionLocal)
        );
    }

    #[test]
    fn exploration_stops_once_every_candidate_is_warm() {
        let m = RouteCostModel::new().with_exploration_interval(1);
        m.set_override_enabled(true);
        for _ in 0..5 {
            m.observe_by_label("olap/parquet", "Native(Volcano)", &snap(2, 4096, 1));
            m.observe_by_label("olap/parquet", "DataFusionLocal", &snap(4, 8192, 2));
        }
        let cands = [ComputeBackend::Native, ComputeBackend::DataFusionLocal];
        assert!(m.exploration_choice("olap/parquet", &cands).is_none());
    }

    #[test]
    fn exploration_is_rate_limited_by_interval() {
        let m = RouteCostModel::new().with_exploration_interval(3);
        m.set_override_enabled(true);
        for _ in 0..5 {
            m.observe_by_label("olap/parquet", "Native(Volcano)", &snap(2, 4096, 1));
        }
        let cands = [ComputeBackend::Native, ComputeBackend::DataFusionLocal];
        // ticks 0,3 explore; 1,2,4,5 do not (1-in-3).
        let fired: Vec<bool> = (0..6)
            .map(|_| m.exploration_choice("olap/parquet", &cands).is_some())
            .collect();
        assert_eq!(fired, vec![true, false, false, true, false, false]);
    }

    #[test]
    fn tier_multiplier_defaults_to_inert_one() {
        // No resolver installed → multiplier 1.0, and None tenant is always 1.0.
        assert_eq!(tier_entitlement_multiplier(Some("anyone")), 1.0);
        assert_eq!(tier_entitlement_multiplier(None), 1.0);
        // final_cost is then the unscaled base score.
        assert_eq!(final_cost(42.0, Some("anyone")), 42.0);
    }

    #[test]
    fn tier_multiplier_resolver_scales_reported_cost_but_is_routing_neutral() {
        // Use a unique resolver, then clear it, to avoid racing sibling tests on
        // the process-global (mirrors the io_trace observer test discipline).
        set_tier_multiplier_resolver(Some(Box::new(|tenant: &str| match tenant {
            "premium" => 0.5,   // entitled to a discounted reported cost
            "throttled" => 4.0, // surcharged
            "bad" => -1.0,      // invalid → must be rejected fail-safe to 1.0
            _ => 1.0,
        })));

        assert_eq!(tier_entitlement_multiplier(Some("premium")), 0.5);
        assert_eq!(tier_entitlement_multiplier(Some("throttled")), 4.0);
        // Non-positive multiplier is rejected (could invert cost ordering).
        assert_eq!(tier_entitlement_multiplier(Some("bad")), 1.0);
        // Unknown tenant falls through the resolver's own default.
        assert_eq!(tier_entitlement_multiplier(Some("someone-else")), 1.0);

        // final_cost applies the multiplier to the reported cost...
        assert_eq!(final_cost(100.0, Some("premium")), 50.0);
        assert_eq!(final_cost(100.0, Some("throttled")), 400.0);

        // ...but it is routing-neutral: scaling two candidates by the SAME
        // per-tenant multiplier preserves which one is cheaper.
        let (a, b) = (120.0_f64, 200.0_f64);
        for tenant in ["premium", "throttled", "someone-else"] {
            let fa = final_cost(a, Some(tenant));
            let fb = final_cost(b, Some(tenant));
            assert_eq!(a < b, fa < fb, "ordering preserved under tier {tenant}");
        }

        set_tier_multiplier_resolver(None); // reset global for other tests
        assert_eq!(tier_entitlement_multiplier(Some("premium")), 1.0);
    }

    #[test]
    fn recompute_bakes_ranked_cheapest_first() {
        let m = RouteCostModel::new().with_min_samples(1);
        // Native cheap (3 GETs), DataFusion dear (300 GETs) for olap/parquet.
        for _ in 0..3 {
            m.observe_by_label("olap/parquet", "Native(Volcano)", &snap(3, 1 << 20, 1));
            m.observe_by_label("olap/parquet", "DataFusionLocal", &snap(300, 1 << 20, 1));
        }
        m.recompute();
        let consult = m.consult("olap/parquet").expect("baked");
        let ranked = consult.ranked();
        assert_eq!(ranked.len(), 2);
        assert_eq!(
            backend_label(&ranked[0].backend),
            "Native(Volcano)",
            "cheapest first"
        );
        assert!(ranked[0].score < ranked[1].score);
    }

    #[test]
    fn recompute_explore_target_is_least_sampled_cold_candidate() {
        let m = RouteCostModel::new().with_min_samples(3);
        // DataFusion warm; Native never observed (0 samples) → explore Native.
        for _ in 0..5 {
            m.observe_by_label("olap/parquet", "DataFusionLocal", &snap(4, 1 << 20, 1));
        }
        m.recompute();
        let consult = m.consult("olap/parquet").expect("baked");
        assert_eq!(
            consult.explore_target().map(backend_label).as_deref(),
            Some("Native(Volcano)")
        );
    }

    #[test]
    fn recompute_explore_target_none_once_all_safe_candidates_warm() {
        let m = RouteCostModel::new().with_min_samples(1);
        for _ in 0..3 {
            m.observe_by_label("olap/parquet", "Native(Volcano)", &snap(2, 1 << 20, 1));
            m.observe_by_label("olap/parquet", "DataFusionLocal", &snap(4, 1 << 20, 1));
        }
        m.recompute();
        let consult = m.consult("olap/parquet").expect("baked");
        assert_eq!(
            consult.explore_target(),
            None,
            "both warm → nothing to explore"
        );
    }

    #[test]
    fn recompute_only_olap_parquet_exposes_two_safe_candidates() {
        // The freshness-safety gate from the class prefix: only olap/parquet
        // exposes both engines; every other class keeps a single static backend.
        let m = RouteCostModel::new().with_min_samples(1);
        for (class, label) in [
            ("olap/parquet", "Native(Volcano)"),
            ("olap/parquet", "DataFusionLocal"),
            ("olap/native", "Native(Volcano)"),
            ("olap/native", "DataFusionLocal"), // NOT freshness-safe here
            ("oltp/parquet", "DataFusionLocal"), // NOT freshness-safe here
            ("oltp/native", "Native(Volcano)"),
        ] {
            m.observe_by_label(class, label, &snap(2, 1 << 20, 1));
        }
        m.recompute();
        let olap_parquet = m.consult("olap/parquet").expect("baked");
        assert_eq!(
            olap_parquet.ranked().len(),
            2,
            "olap/parquet has two safe engines"
        );
        let olap_native = m.consult("olap/native").expect("baked");
        assert_eq!(
            olap_native.ranked().len(),
            1,
            "olap/native drops non-safe DataFusion"
        );
        assert_eq!(
            backend_label(&olap_native.ranked()[0].backend),
            "Native(Volcano)"
        );
        let oltp_parquet = m.consult("oltp/parquet").expect("baked");
        assert!(
            oltp_parquet.ranked().is_empty(),
            "oltp/parquet keeps only Native (unobserved) — DF is not freshness-safe"
        );
        assert_eq!(
            oltp_parquet.explore_target(),
            None,
            "single safe candidate → no explore"
        );
    }

    /// DuckDB traces remain evidence-only until its I/O quantities are
    /// comparable to the other engines. Even an explicitly enabled route may
    /// collect cells, but the live model must not rank or explore them.
    #[cfg(feature = "duckdb")]
    #[test]
    fn duckdb_cells_are_evidence_only_when_route_enabled() {
        unsafe { std::env::set_var("PROXIMADB_DUCKDB_ROUTE", "1") };
        let m = RouteCostModel::new().with_min_samples(1);
        m.observe_by_label("olap/parquet", "DuckDbCompat", &snap(2, 1 << 20, 1));
        m.observe_by_label("olap/parquet", "DataFusionLocal", &snap(200, 1 << 20, 9));
        m.recompute();
        let consult = m.consult("olap/parquet").expect("baked");
        assert!(
            !consult
                .ranked()
                .iter()
                .any(|c| backend_label(&c.backend) == "DuckDbCompat"),
            "compute-only DuckDB cells must not feed live overrides"
        );
        // OLTP classes never gain DuckDB regardless of the gate.
        m.observe_by_label("oltp/native", "DuckDbCompat", &snap(2, 1 << 20, 1));
        m.recompute();
        let oltp = m.consult("oltp/native").expect("baked");
        assert!(
            !oltp
                .ranked()
                .iter()
                .any(|c| backend_label(&c.backend) == "DuckDbCompat")
        );
    }

    /// Structural safety for builds WITHOUT the duckdb feature: persisted or
    /// stray DuckDbCompat cells (e.g. a cost-model file written by a
    /// duckdb-enabled deployment) are dropped from the ranked set, so no flip
    /// can ever target an engine this build cannot execute.
    #[cfg(not(feature = "duckdb"))]
    #[test]
    fn duckdb_cells_are_dropped_when_feature_absent() {
        let m = RouteCostModel::new().with_min_samples(1);
        m.observe_by_label("olap/parquet", "DuckDbCompat", &snap(2, 1 << 20, 1));
        m.observe_by_label("olap/parquet", "DataFusionLocal", &snap(200, 1 << 20, 9));
        m.recompute();
        let consult = m.consult("olap/parquet").expect("baked");
        assert!(
            !consult
                .ranked()
                .iter()
                .any(|c| backend_label(&c.backend) == "DuckDbCompat"),
            "a build without the duckdb feature must never rank DuckDbCompat"
        );
    }

    #[test]
    fn recompute_skips_unknown_backend_labels() {
        let m = RouteCostModel::new().with_min_samples(1);
        // A synthetic/unknown label must be dropped, never panic.
        m.observe_by_label("olap/parquet", "SomeUnknownEngine", &snap(2, 1 << 20, 1));
        m.observe_by_label("olap/parquet", "Native(Volcano)", &snap(2, 1 << 20, 1));
        m.recompute();
        let consult = m.consult("olap/parquet").expect("baked");
        assert_eq!(consult.ranked().len(), 1);
        assert_eq!(
            backend_label(&consult.ranked()[0].backend),
            "Native(Volcano)"
        );
    }

    #[test]
    fn exploration_never_fires_with_a_single_candidate() {
        // OLTP gets a single freshness-safe candidate → nothing to explore.
        let m = RouteCostModel::new().with_exploration_interval(1);
        m.set_override_enabled(true);
        assert!(
            m.exploration_choice("oltp/native", &[ComputeBackend::Native])
                .is_none()
        );
    }

    // ── C4 Phase-2b: route-time shape-stat cache (T2.1 wiring / T3.3 ratchet) ─

    #[test]
    fn classify_table_shapes_empty_and_missing_is_unknown() {
        use crate::query::compute_scheduler::{CardinalityClass, PartitionFanout};
        // Empty slice → Unknown.
        let (pf, card) = classify_table_shapes(&[]);
        assert_eq!(
            (pf, card),
            (PartitionFanout::Unknown, CardinalityClass::Unknown)
        );
        // A location with no warmed stat → Unknown (conservative: never
        // half-classify — an unknown table could dominate fan-out/cardinality).
        let (pf, card) = classify_table_shapes(&["test://route-shape/missing".to_string()]);
        assert_eq!(
            (pf, card),
            (PartitionFanout::Unknown, CardinalityClass::Unknown)
        );
    }

    #[test]
    fn classify_table_shapes_buckets_and_sums_warmed_stats() {
        use crate::query::compute_scheduler::{CardinalityClass, PartitionFanout};
        // Small / single-row-group table.
        record_table_shape_stat("test://route-shape/small", 1, Some(500));
        let (pf, card) = classify_table_shapes(&["test://route-shape/small".to_string()]);
        assert_eq!(pf, PartitionFanout::Single);
        assert_eq!(card, CardinalityClass::Small);

        // Large / many-row-group table.
        record_table_shape_stat("test://route-shape/large", 5_000, Some(50_000_000));
        let (pf, card) = classify_table_shapes(&["test://route-shape/large".to_string()]);
        assert_eq!(pf, PartitionFanout::Many);
        assert_eq!(card, CardinalityClass::Large);

        // Two warmed tables: row-groups and rows SUM (total GET fan-out / scan
        // cardinality). 12 row-groups → Many; 1.3M rows → Large.
        record_table_shape_stat("test://route-shape/sum-a", 6, Some(600_000));
        record_table_shape_stat("test://route-shape/sum-b", 6, Some(700_000));
        let (pf, card) = classify_table_shapes(&[
            "test://route-shape/sum-a".to_string(),
            "test://route-shape/sum-b".to_string(),
        ]);
        assert_eq!(pf, PartitionFanout::Many);
        assert_eq!(card, CardinalityClass::Large);

        // Conservative: a cold location among warmed ones → Unknown.
        let (pf, card) = classify_table_shapes(&[
            "test://route-shape/sum-a".to_string(),
            "test://route-shape/cold".to_string(),
        ]);
        assert_eq!(
            (pf, card),
            (PartitionFanout::Unknown, CardinalityClass::Unknown)
        );
    }

    #[test]
    fn shape_class_discriminates_fine_parquet_signals() {
        use crate::query::compute_scheduler::{CardinalityClass, PartitionFanout, QueryShape};
        // Coarse (no fine signals) — the backward-compatible 2-part key.
        let coarse = shape_class(&QueryShape {
            engages_relational: true,
            parquet_backed: true,
            ..Default::default()
        });
        assert_eq!(coarse, "olap/parquet");
        // Refined — distinct keys per fine signal, so the model accrues
        // per-(cardinality, fan-out) cells instead of one noisy bucket.
        let big = shape_class(&QueryShape {
            engages_relational: true,
            parquet_backed: true,
            cardinality: CardinalityClass::Large,
            partition_fanout: PartitionFanout::Many,
            pax_backed: false,
            native_registerable: false,
            operation_class: Default::default(),
            geometry: Default::default(),
            join_bearing: false,
        });
        let small = shape_class(&QueryShape {
            engages_relational: true,
            parquet_backed: true,
            cardinality: CardinalityClass::Small,
            partition_fanout: PartitionFanout::Single,
            pax_backed: false,
            native_registerable: false,
            operation_class: Default::default(),
            geometry: Default::default(),
            join_bearing: false,
        });
        assert_ne!(big, coarse);
        assert_ne!(small, coarse);
        assert_ne!(big, small);
    }

    #[test]
    fn warmed_fine_class_lets_cost_override_flip_route() {
        // T2.1 keystone (T3.3 ratchet): with real fine signals the model accrues
        // trustworthy per-(cardinality, fan-out) cost; with override enabled it
        // flips a freshness-safe OLAP-Parquet route to the confidently-cheaper
        // backend. This is the co-design loop actually closing — observe-mode
        // promoted to act-mode on reliable per-class evidence.
        use crate::query::compute_scheduler::{
            CardinalityClass, ComputeScheduler, PartitionFanout, QueryShape,
        };
        let m = RouteCostModel::new()
            .with_min_samples(1)
            .with_min_advantage(0.1)
            .with_recompute_every(1);
        m.set_override_enabled(true);
        // TD-ROUTE-3: a Native flip is only eligible for a native-servable shape
        // (ungrouped scalar aggregate); an Unknown-op shape can never be served by
        // native, so the override must not flip its DECISION onto Native.
        let big = QueryShape {
            engages_relational: true,
            parquet_backed: true,
            cardinality: CardinalityClass::Large,
            partition_fanout: PartitionFanout::Many,
            pax_backed: false,
            native_registerable: false,
            operation_class: crate::query::compute_scheduler::OperationClass::ScalarAggregate,
            geometry: Default::default(),
            join_bearing: false,
        };
        let class = shape_class(&big);
        // Observe Native as confidently cheaper than DataFusion for THIS fine
        // class (far fewer object-store round-trips). Both backends get ≥
        // min_samples → exploration is satisfied (warm), so the exploit override
        // runs deterministically rather than probing.
        for _ in 0..3 {
            m.observe(&class, &ComputeBackend::Native, &snap(2, 8192, 1));
            m.observe(
                &class,
                &ComputeBackend::DataFusionLocal,
                &snap(40, 1 << 20, 8),
            );
        }
        let decision = ComputeScheduler::new().route_select_advised(
            big,
            Some(&m),
            crate::query::compute_scheduler::RouteFlags::default(),
        );
        // Static rule picks DataFusion for OLAP-Parquet; the warmed model
        // overrides to freshness-safe, cheaper Native.
        assert_eq!(decision.backend, ComputeBackend::Native);
        assert!(
            decision.reason.contains("OVERRIDE"),
            "expected override reason, got: {}",
            decision.reason
        );
    }

    #[test]
    fn ancestor_cells_serve_a_geometry_refined_route() {
        // TD-EXEC-2 Slice 3 continuity: a query whose class carries the NEW
        // geometry tier is still served by cells warmed under the pre-tier key
        // (hierarchical consult), the override flips on that ancestor evidence,
        // and the decision discloses which level served it. Without this, every
        // key refinement would reset all learning (ADR-058 D3 violation).
        use crate::query::compute_scheduler::{ComputeScheduler, GeometryClass, QueryShape};
        let m = RouteCostModel::new()
            .with_min_samples(1)
            .with_min_advantage(0.1)
            .with_recompute_every(1);
        m.set_override_enabled(true);
        // Warm ONLY the coarse pre-tier class.
        for _ in 0..3 {
            m.observe("olap/parquet", &ComputeBackend::Native, &snap(2, 8192, 1));
            m.observe(
                "olap/parquet",
                &ComputeBackend::DataFusionLocal,
                &snap(40, 1 << 20, 8),
            );
        }
        // TD-ROUTE-3: native-eligible op so the override→Native flip is a valid
        // (servable) route; the geometry refinement still exercises the ancestor
        // consult down to the warmed coarse `olap/parquet` cells.
        let refined = QueryShape {
            engages_relational: true,
            parquet_backed: true,
            operation_class: crate::query::compute_scheduler::OperationClass::ScalarAggregate,
            geometry: GeometryClass::from_estimate(9, 4),
            ..Default::default()
        };
        assert_eq!(shape_class(&refined), "olap/parquet/op=agg/geom=mxhi");
        let decision = ComputeScheduler::new().route_select_advised(
            refined,
            Some(&m),
            crate::query::compute_scheduler::RouteFlags::default(),
        );
        assert_eq!(decision.backend, ComputeBackend::Native);
        assert!(
            decision.reason.contains("OVERRIDE")
                && decision.reason.contains("[cells: olap/parquet]"),
            "expected ancestor-served override, got: {}",
            decision.reason
        );
    }
}
