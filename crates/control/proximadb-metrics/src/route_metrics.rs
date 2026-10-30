#![allow(clippy::unwrap_used, clippy::expect_used)]
// dedicated metric-registration module: static registration is infallible / fail-fast at startup;
// lazy_static can't carry per-site allows.
// Copyright 2026 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Read-route decision observability (co-design C4 operator surface).
//!
//! The `ComputeScheduler` route decision was previously observable only via a
//! `tracing::debug!` line — invisible in production dashboards. These counters
//! and the consult-latency histogram make the route distribution, the
//! static/override/explore split, and the cost of the (now lock-free) consult
//! itself operationally visible. Labels are deliberately low-cardinality:
//! `backend` is the canonical engine label, `shape_class` the coarse cost-model
//! class key, and `source` one of `static` / `override_exploit` / `override_explore`.
//!
//! `ROUTE_OLAP_ON_NATIVE_TOTAL` is an operator nudge: an analytic (`olap/*`)
//! query served by the Native/Volcano row engine is paying N row-gets instead of
//! a columnar scan — the table is a materialization candidate.

use std::time::Duration;

use crate::{register_counter_vec_safe, register_histogram_vec_safe};
use lazy_static::lazy_static;
use prometheus::{CounterVec, HistogramVec};

lazy_static! {
    /// Route decisions by backend, shape-class, and decision source.
    pub static ref ROUTE_DECISIONS_TOTAL: CounterVec = register_counter_vec_safe(
        "proximadb_route_decisions_total",
        "Read-route decisions by backend, shape-class, and source (static | override_exploit | override_explore)",
        &["backend", "shape_class", "source"],
    );

    /// Consult (route-decision) latency. Proves the consult stays sub-microsecond
    /// now that it is a lock-free `ArcSwap` load + `HashMap` get.
    pub static ref ROUTE_CONSULT_DURATION_SECONDS: HistogramVec = register_histogram_vec_safe(
        "proximadb_route_consult_duration_seconds",
        "Read-route decision latency, by whether a cost model was consulted",
        &["had_model"],
        // Sub-microsecond to ~1ms — the consult is a hot-path, lock-free read.
        vec![1e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5, 5e-5, 1e-4, 1e-3],
    );

    /// Analytic (`olap/*`) queries served by the Native/Volcano row engine — a
    /// materialization nudge (the query would be cheaper over a Parquet base).
    pub static ref ROUTE_OLAP_ON_NATIVE_TOTAL: CounterVec = register_counter_vec_safe(
        "proximadb_route_olap_on_native_total",
        "Analytic queries routed to the Native row engine (unmaterialized) — consider ALTER TABLE MATERIALIZE",
        &["shape_class"],
    );

    /// Consults that found no baked entry for the query's shape-class — the
    /// ADR-058 D3 frozen-table *density* invariant made falsifiable: a hot query
    /// class with a nonzero miss rate after warm-up means the class key (e.g. a
    /// new refinement tier) fragmented the cells faster than they warm.
    pub static ref ROUTE_CONSULT_MISS_TOTAL: CounterVec = register_counter_vec_safe(
        "proximadb_route_consult_miss_total",
        "Cost-model consults that resolved no recommendation for the shape-class (ADR-058 D3 density check)",
        &["shape_class"],
    );

    /// TD-EXEC-2 Slice 3 calibration accuracy: the geometry band the route was
    /// keyed on (AST estimate) vs the measured plan-geometry band, per fold.
    /// Divergence rate = fraction of samples where the labels differ; a rising
    /// off-diagonal means the AST estimator needs a band refit (from the
    /// ledger's measured `plan_depth`/`plan_blocking` distribution).
    pub static ref ROUTE_GEOMETRY_ESTIMATE_TOTAL: CounterVec = register_counter_vec_safe(
        "proximadb_route_geometry_estimate_total",
        "Route-time geometry band estimate vs measured plan-geometry band (TD-EXEC-2 Slice 3 calibration)",
        &["estimated", "measured"],
    );
}

/// Record one route-decision outcome. `source` is the RouteSource (crate::query::compute_scheduler, root — doc-link dropped)
/// label string (`static` / `override_exploit` / `override_explore`).
pub fn record_decision(backend_label: &str, shape_class: &str, source: &str) {
    ROUTE_DECISIONS_TOTAL
        .with_label_values(&[backend_label, shape_class, source])
        .inc();
}

/// Record an analytic query served by the Native row engine (operator nudge).
pub fn record_olap_on_native(shape_class: &str) {
    ROUTE_OLAP_ON_NATIVE_TOTAL
        .with_label_values(&[shape_class])
        .inc();
}

/// Record a consult that resolved no recommendation for `shape_class`.
pub fn record_consult_miss(shape_class: &str) {
    ROUTE_CONSULT_MISS_TOTAL
        .with_label_values(&[shape_class])
        .inc();
}

/// Record one estimated-vs-measured geometry band sample (bare band labels,
/// e.g. `sxlo`; the `geom=` prefix is stripped by the caller).
pub fn record_geometry_estimate(estimated: &str, measured: &str) {
    ROUTE_GEOMETRY_ESTIMATE_TOTAL
        .with_label_values(&[estimated, measured])
        .inc();
}

/// Observe one consult's wall-clock duration. `had_model` distinguishes the
/// pure-static path (no consult) from the advised path.
pub fn observe_consult_duration(had_model: bool, duration: Duration) {
    ROUTE_CONSULT_DURATION_SECONDS
        .with_label_values(&[if had_model { "true" } else { "false" }])
        .observe(duration.as_secs_f64());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_decision_increments_the_right_label_set() {
        // nextest process-isolation: this process's registry starts empty, so an
        // absolute count is meaningful here.
        let before = ROUTE_DECISIONS_TOTAL
            .with_label_values(&["DataFusionLocal", "olap/parquet", "static"])
            .get();
        record_decision("DataFusionLocal", "olap/parquet", "static");
        let after = ROUTE_DECISIONS_TOTAL
            .with_label_values(&["DataFusionLocal", "olap/parquet", "static"])
            .get();
        assert_eq!(after - before, 1.0);
    }

    #[test]
    fn record_olap_on_native_increments() {
        let before = ROUTE_OLAP_ON_NATIVE_TOTAL
            .with_label_values(&["olap/native"])
            .get();
        record_olap_on_native("olap/native");
        record_olap_on_native("olap/native");
        let after = ROUTE_OLAP_ON_NATIVE_TOTAL
            .with_label_values(&["olap/native"])
            .get();
        assert_eq!(after - before, 2.0);
    }

    #[test]
    fn observe_consult_duration_lands_a_sample() {
        // A fresh histogram with one observed sample has count 1.
        let hist = ROUTE_CONSULT_DURATION_SECONDS.with_label_values(&["true"]);
        let before = hist.get_sample_count();
        observe_consult_duration(true, Duration::from_nanos(500));
        let after = hist.get_sample_count();
        assert_eq!(after - before, 1u64);
    }
}
