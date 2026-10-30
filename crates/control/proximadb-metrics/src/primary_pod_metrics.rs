/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! Primary-pod gateway write-router metrics
//!
//! Operator observability for the Slice 4 write-routing gate. Every
//! incoming write decision lands here so operators can answer:
//!
//! * **Are tenants being misrouted?** Spike in
//!   `proximadb_primary_pod_writes_misrouted_total` means clients are
//!   hitting the wrong pod — usually a stale routing table or a
//!   recent reassignment that hasn't propagated.
//! * **Is the gate ever firing?** Steady stream on
//!   `proximadb_primary_pod_writes_allowed_total{outcome="bound"}`
//!   means bindings are present and active. Zero traffic on that
//!   label means no tenants are bound (legacy / unbounded mode).
//! * **Per-tenant binding rate?** Use the `tenant_id` label.
//!
//! Labels are intentionally low-cardinality: 3 outcomes × tenant_id.

use crate::register_counter_vec_safe;
use lazy_static::lazy_static;
use prometheus::CounterVec;

lazy_static! {
    /// Writes that passed the routing gate (registry said "Allow").
    /// The `outcome` label distinguishes:
    ///
    /// * `"bound"` — a binding exists AND points at this pod. The
    ///   correctness payoff case: client got it right.
    /// * `"unbounded"` — no binding for this `(tenant, collection)`.
    ///   Legacy / opt-out tenants; the write proceeds without
    ///   constraint.
    pub static ref WRITES_ALLOWED_TOTAL: CounterVec = register_counter_vec_safe(
        "proximadb_primary_pod_writes_allowed_total",
        "Writes that the primary-pod gate allowed (bound to self or no binding)",
        &["outcome", "tenant_id"],
    );

    /// Writes the gate rejected (registry said "Misrouted"). Each
    /// one corresponds to a 421 Misdirected Request response. Spikes
    /// indicate stale client routing tables or a recent reassignment
    /// the clients haven't observed yet — operators should compare
    /// against `proximadb_collection_pin_operations_total` to see if
    /// a recent pin churn correlates.
    pub static ref WRITES_MISROUTED_TOTAL: CounterVec = register_counter_vec_safe(
        "proximadb_primary_pod_writes_misrouted_total",
        "Writes the primary-pod gate rejected as misrouted (421 Misdirected Request)",
        &["tenant_id"],
    );

    /// Catalog-mirror failures from the Slice 5b.2 REST write-through.
    /// The `reason` label matches the `MirrorFailure::label()` enum
    /// discriminator in `src/network/rest/canonical/primary_pod.rs`:
    ///
    /// * `"no_default_catalog"` — bootstrap; rare in steady state.
    /// * `"catalog_error"` — backend rejected the call (e.g. table
    ///   not yet in catalog, or non-Native backend with the default
    ///   "not supported" trait impl).
    ///
    /// Non-zero values are observable but **not** error-grade until
    /// slice 5d removes the JSON sidecar — until then the registry is
    /// authoritative and the catalog is a forward-prep mirror.
    pub static ref CATALOG_MIRROR_FAILURES_TOTAL: CounterVec = register_counter_vec_safe(
        "proximadb_primary_pod_catalog_mirror_failures_total",
        "Primary-pod catalog mirror failures (REST write-through; registry write still succeeded)",
        &["reason"],
    );
}

/// Record that a write was allowed because a binding pointed at this
/// pod. Operators alert on the absence of this metric (across all
/// tenants) — it means the registry is empty in production.
pub fn record_allowed_bound(tenant_id: &str) {
    WRITES_ALLOWED_TOTAL
        .with_label_values(&["bound", tenant_id])
        .inc();
}

/// Record that a write was allowed because no binding existed.
/// Operators use the ratio of `bound` vs `unbounded` to see how much
/// of their tenant fleet has opted in.
pub fn record_allowed_unbounded(tenant_id: &str) {
    WRITES_ALLOWED_TOTAL
        .with_label_values(&["unbounded", tenant_id])
        .inc();
}

/// Record that a write was rejected as misrouted. Always corresponds
/// to a 421 response — never a silent retry — so the metric is
/// authoritative for "how often clients are hitting the wrong pod."
pub fn record_misrouted(tenant_id: &str) {
    WRITES_MISROUTED_TOTAL.with_label_values(&[tenant_id]).inc();
}

/// Record a catalog-mirror failure. `reason` must match a
/// `MirrorFailure::label()` value so dashboards key on a stable set.
/// Intentionally no `tenant_id` label — keeps cardinality bounded and
/// matches the operator question this metric answers ("is the mirror
/// healthy?", not "which tenant?").
pub fn record_catalog_mirror_failure(reason: &str) {
    CATALOG_MIRROR_FAILURES_TOTAL
        .with_label_values(&[reason])
        .inc();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_allowed_bound_increments_outcome_bound() {
        let before = WRITES_ALLOWED_TOTAL
            .with_label_values(&["bound", "test-tenant-a"])
            .get();
        record_allowed_bound("test-tenant-a");
        let after = WRITES_ALLOWED_TOTAL
            .with_label_values(&["bound", "test-tenant-a"])
            .get();
        assert!((after - before - 1.0).abs() < 1e-9);
    }

    #[test]
    fn record_allowed_unbounded_increments_outcome_unbounded() {
        let before = WRITES_ALLOWED_TOTAL
            .with_label_values(&["unbounded", "test-tenant-b"])
            .get();
        record_allowed_unbounded("test-tenant-b");
        let after = WRITES_ALLOWED_TOTAL
            .with_label_values(&["unbounded", "test-tenant-b"])
            .get();
        assert!((after - before - 1.0).abs() < 1e-9);
    }

    #[test]
    fn record_catalog_mirror_failure_increments_reason_label() {
        let before = CATALOG_MIRROR_FAILURES_TOTAL
            .with_label_values(&["test-reason-a"])
            .get();
        record_catalog_mirror_failure("test-reason-a");
        let after = CATALOG_MIRROR_FAILURES_TOTAL
            .with_label_values(&["test-reason-a"])
            .get();
        assert!((after - before - 1.0).abs() < 1e-9);
    }

    #[test]
    fn record_catalog_mirror_failure_does_not_bump_write_counters() {
        // The mirror failure counter is in a different metric family;
        // a mirror failure must NOT also bump the routing counters.
        // Locking this in catches any future refactor that funnels
        // them through the same helper.
        let before_allowed = WRITES_ALLOWED_TOTAL
            .with_label_values(&["bound", "test-tenant-mirror"])
            .get();
        let before_misrouted = WRITES_MISROUTED_TOTAL
            .with_label_values(&["test-tenant-mirror"])
            .get();
        record_catalog_mirror_failure("test-reason-b");
        let after_allowed = WRITES_ALLOWED_TOTAL
            .with_label_values(&["bound", "test-tenant-mirror"])
            .get();
        let after_misrouted = WRITES_MISROUTED_TOTAL
            .with_label_values(&["test-tenant-mirror"])
            .get();
        assert!((after_allowed - before_allowed).abs() < 1e-9);
        assert!((after_misrouted - before_misrouted).abs() < 1e-9);
    }

    #[test]
    fn record_misrouted_increments_only_the_misrouted_counter() {
        let before_misrouted = WRITES_MISROUTED_TOTAL
            .with_label_values(&["test-tenant-c"])
            .get();
        let before_allowed_bound = WRITES_ALLOWED_TOTAL
            .with_label_values(&["bound", "test-tenant-c"])
            .get();
        record_misrouted("test-tenant-c");
        let after_misrouted = WRITES_MISROUTED_TOTAL
            .with_label_values(&["test-tenant-c"])
            .get();
        let after_allowed_bound = WRITES_ALLOWED_TOTAL
            .with_label_values(&["bound", "test-tenant-c"])
            .get();
        assert!((after_misrouted - before_misrouted - 1.0).abs() < 1e-9);
        assert!(
            (after_allowed_bound - before_allowed_bound).abs() < 1e-9,
            "misrouted writes must not bump the allowed counter — they are mutually exclusive"
        );
    }
}
