/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! DML lock-manager observability (A9 foundation gate).
//!
//! Operator answers:
//!
//! * **Are DML locks contended?** A rising
//!   `proximadb_dml_lock_acquisitions_total{outcome="conflict"|"held"}`
//!   rate means writes are blocking each other (or other pods) — a sign of
//!   hot tables or leases too short for the workload.
//! * **How long does acquisition take?**
//!   `proximadb_dml_lock_acquisition_duration_seconds` carries the wall-clock
//!   cost of an acquire (in-memory scan + one object-store CAS). A shift right
//!   means the object-store metadata path is on the critical write path.
//! * **How many locks are held right now?** `proximadb_dml_locks_held_current`
//!   per resource type — sizing the in-memory registry and bounding renewal
//!   load.
//! * **Are renewals failing?** `proximadb_dml_lock_renewal_failures_total`
//!   spikes precede silent lease loss (a pod keeps serving writes its lease
//!   no longer authorizes).
//!
//! Labels are intentionally low-cardinality: a fixed `outcome` enum, a fixed
//! `resource_type` enum, and `tenant_id` only on the acquisition counter
//! (matches `primary_pod_metrics` precedent; the histogram/gauge/renewal
//! counter omit it to bound cardinality).

use std::time::Duration;

use crate::{register_counter_vec_safe, register_histogram_vec_safe, register_int_gauge_vec_safe};
use lazy_static::lazy_static;
use prometheus::{CounterVec, HistogramVec, IntGaugeVec};

/// Bucket layout for DML lock acquisition latency. Locks should be cheap
/// (an in-memory scan plus one object-store CAS), so the buckets span
/// sub-millisecond up to a few seconds — anything past ~1s means the
/// object-store metadata path is dominating the write.
fn acquisition_duration_buckets() -> Vec<f64> {
    vec![
        0.000_5, // 0.5 ms — in-memory fast path only
        0.001,   // 1 ms
        0.002_5, // 2.5 ms
        0.005,   // 5 ms
        0.01,    // 10 ms — one local CAS
        0.025,   // 25 ms
        0.05,    // 50 ms
        0.1,     // 100 ms — one cloud CAS
        0.25,    // 250 ms
        0.5,     // 500 ms
        1.0,     // 1 s — metadata-bound; investigate
        5.0,     // 5 s — pathological / object-store stall
    ]
}

lazy_static! {
    /// DML lock acquisition attempts by outcome. The `outcome` label is one
    /// of `acquired` / `conflict` / `held` / `fenced`, matching
    /// See `crate::cluster::partition_lease::LockOutcome` (root, doc-link dropped).
    pub static ref DML_LOCK_ACQUISITIONS_TOTAL: CounterVec = register_counter_vec_safe(
        "proximadb_dml_lock_acquisitions_total",
        "DML lock acquisition attempts by outcome",
        &["outcome", "resource_type", "tenant_id"],
    );

    /// Wall-clock duration of a DML lock acquisition (in-memory scan + the
    /// object-store CAS that establishes the durable lease).
    pub static ref DML_LOCK_ACQUISITION_DURATION_SECONDS: HistogramVec =
        register_histogram_vec_safe(
            "proximadb_dml_lock_acquisition_duration_seconds",
            "Wall-clock duration of DML lock acquisition attempts",
            &["outcome", "resource_type"],
            acquisition_duration_buckets(),
        );

    /// DML locks currently held by this pod, by resource type. Inc on
    /// acquire, dec on release.
    pub static ref DML_LOCKS_HELD_CURRENT: IntGaugeVec = register_int_gauge_vec_safe(
        "proximadb_dml_locks_held_current",
        "DML locks currently held by this pod",
        &["resource_type"],
    );

    /// Failures renewing a held DML/resource lease. Sustained rate precedes
    /// silent lease loss (a pod serving writes its lease no longer covers).
    pub static ref DML_LOCK_RENEWAL_FAILURES_TOTAL: CounterVec = register_counter_vec_safe(
        "proximadb_dml_lock_renewal_failures_total",
        "DML/resource lease renewal failures",
        &["resource_type"],
    );
}

/// Record a DML lock acquisition outcome + its wall-clock duration. Bumps the
/// per-outcome counter (with tenant) and observes the latency histogram.
pub fn record_acquisition(outcome: &str, resource_type: &str, tenant_id: &str, elapsed: Duration) {
    DML_LOCK_ACQUISITIONS_TOTAL
        .with_label_values(&[outcome, resource_type, tenant_id])
        .inc();
    DML_LOCK_ACQUISITION_DURATION_SECONDS
        .with_label_values(&[outcome, resource_type])
        .observe(elapsed.as_secs_f64());
}

/// Mark one more DML lock held for `resource_type`.
pub fn inc_held(resource_type: &str) {
    DML_LOCKS_HELD_CURRENT
        .with_label_values(&[resource_type])
        .inc();
}

/// Mark one fewer DML lock held for `resource_type`.
pub fn dec_held(resource_type: &str) {
    DML_LOCKS_HELD_CURRENT
        .with_label_values(&[resource_type])
        .dec();
}

/// Record a lease-renewal failure for `resource_type`.
pub fn record_renewal_failure(resource_type: &str) {
    DML_LOCK_RENEWAL_FAILURES_TOTAL
        .with_label_values(&[resource_type])
        .inc();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_acquisition_increments_counter() {
        let before = DML_LOCK_ACQUISITIONS_TOTAL
            .with_label_values(&["acquired", "table", "test-tenant-a"])
            .get();
        record_acquisition(
            "acquired",
            "table",
            "test-tenant-a",
            Duration::from_micros(800),
        );
        let after = DML_LOCK_ACQUISITIONS_TOTAL
            .with_label_values(&["acquired", "table", "test-tenant-a"])
            .get();
        assert!((after - before - 1.0).abs() < 1e-9);
    }

    #[test]
    fn record_acquisition_observes_histogram_without_panic() {
        // Histograms have no trivial .get(); assert observe is well-formed.
        record_acquisition(
            "conflict",
            "schema",
            "test-tenant-b",
            Duration::from_millis(3),
        );
        record_acquisition(
            "held",
            "table",
            "test-tenant-c",
            Duration::from_secs_f64(0.075),
        );
    }

    #[test]
    fn inc_and_dec_held_balances_gauge() {
        let before = DML_LOCKS_HELD_CURRENT.with_label_values(&["table"]).get();
        inc_held("table");
        inc_held("table");
        assert_eq!(
            DML_LOCKS_HELD_CURRENT.with_label_values(&["table"]).get(),
            before + 2
        );
        dec_held("table");
        dec_held("table");
        assert_eq!(
            DML_LOCKS_HELD_CURRENT.with_label_values(&["table"]).get(),
            before
        );
    }

    #[test]
    fn record_renewal_failure_increments_counter() {
        let before = DML_LOCK_RENEWAL_FAILURES_TOTAL
            .with_label_values(&["collection"])
            .get();
        record_renewal_failure("collection");
        let after = DML_LOCK_RENEWAL_FAILURES_TOTAL
            .with_label_values(&["collection"])
            .get();
        assert!((after - before - 1.0).abs() < 1e-9);
    }
}
