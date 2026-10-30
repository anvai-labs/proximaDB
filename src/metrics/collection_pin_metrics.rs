/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! Collection-pin registry metrics
//!
//! Operator observability for the per-collection pin registry (Phase 6
//! data plane). Pins override the access-pattern tier policy, so
//! operators need a live signal of what's pinned and where:
//!
//! * **Currently pinned** — a `GaugeVec` indexed by target tier
//!   ("memory" / "nvme_ssd" / "cloud"). Drives the "pinned collections
//!   per target" dashboard panel.
//! * **Pin operations** — a `CounterVec` of pin/unpin actions over
//!   time, labelled by target. Lets operators audit churn and
//!   correlate pin activity with the just-landed tier-migration
//!   counters.
//!
//! Both metrics are intentionally low-cardinality: 3 target values × 2
//! operations = 6 label combinations max. Safe at default Prometheus
//! scrape cadence.

use lazy_static::lazy_static;
use prometheus::{CounterVec, IntGaugeVec};

use crate::storage::collection_pinning::CollectionPinTarget;
use proximadb_metrics::{register_counter_vec_safe, register_int_gauge_vec_safe};

lazy_static! {
    /// Number of collections currently pinned to each target tier.
    /// Recomputed on every pin/unpin so the gauge always reflects the
    /// authoritative in-memory registry state. Operators dashboard the
    /// breakdown ("how much memory tier is committed to pins?") and
    /// alert when a tier approaches a capacity ceiling.
    pub static ref COLLECTION_PINS_CURRENT: IntGaugeVec = register_int_gauge_vec_safe(
        "proximadb_collection_pins_current",
        "Number of collections currently pinned to each target tier",
        &["target"],
    );

    /// Total pin/unpin operations over time, labelled by op + target.
    /// Operators alert on excess churn (`rate > N pins/sec`) and
    /// correlate pin actions with the resulting tier-migration
    /// counters from `proximadb_tier_migrations_total`.
    pub static ref COLLECTION_PIN_OPERATIONS_TOTAL: CounterVec = register_counter_vec_safe(
        "proximadb_collection_pin_operations_total",
        "Total pin/unpin operations on the collection pin registry",
        &["operation", "target"],
    );
}

/// Stable lowercase label for a pin target. Mirrors
/// `CollectionPinTarget::label` so dashboards work uniformly across
/// the operator REST API and the Prometheus surface.
fn target_label(target: CollectionPinTarget) -> &'static str {
    target.label()
}

/// Record a `pin` operation on the counter. Does NOT touch the
/// current-pins gauge — the caller is responsible for the gauge
/// because re-pin semantics (replace a previous pin) require both
/// a decrement on the old target and an increment on the new one,
/// which a single helper can't cleanly express.
pub fn record_pin(target: CollectionPinTarget) {
    COLLECTION_PIN_OPERATIONS_TOTAL
        .with_label_values(&["pin", target_label(target)])
        .inc();
}

/// Record an `unpin` operation on the counter. Mirrors
/// [`record_pin`]; gauge adjustment is the caller's responsibility.
pub fn record_unpin(previous_target: CollectionPinTarget) {
    COLLECTION_PIN_OPERATIONS_TOTAL
        .with_label_values(&["unpin", target_label(previous_target)])
        .inc();
}

/// Increment the current-pins gauge by `+1` for `target`. Call this
/// when a fresh pin lands on a collection that wasn't previously
/// pinned, or when a re-pin moves to a new target after the previous
/// target was decremented.
pub fn inc_current_pin(target: CollectionPinTarget) {
    COLLECTION_PINS_CURRENT
        .with_label_values(&[target_label(target)])
        .inc();
}

/// Decrement the current-pins gauge by `1` for `target`. Call this
/// on unpin (the previous target was removed) or on the previous
/// target of a re-pin (before incrementing the new target).
pub fn dec_current_pin(target: CollectionPinTarget) {
    COLLECTION_PINS_CURRENT
        .with_label_values(&[target_label(target)])
        .dec();
}

/// Reset the current-pins gauge to a fresh count derived from the
/// authoritative registry state. Called after registry loads from
/// disk (where the in-memory state appears all-at-once and the
/// per-pin `record_pin` hook never fired) so the gauge agrees with
/// the registry from the first scrape after startup.
pub fn reset_current_pins(memory_count: i64, nvme_count: i64, cloud_count: i64) {
    COLLECTION_PINS_CURRENT
        .with_label_values(&[target_label(CollectionPinTarget::Memory)])
        .set(memory_count);
    COLLECTION_PINS_CURRENT
        .with_label_values(&[target_label(CollectionPinTarget::NvmeSsd)])
        .set(nvme_count);
    COLLECTION_PINS_CURRENT
        .with_label_values(&[target_label(CollectionPinTarget::Cloud)])
        .set(cloud_count);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_pin_increments_counter_only() {
        let before_counter = COLLECTION_PIN_OPERATIONS_TOTAL
            .with_label_values(&["pin", "nvme_ssd"])
            .get();
        let before_gauge = COLLECTION_PINS_CURRENT
            .with_label_values(&["nvme_ssd"])
            .get();

        record_pin(CollectionPinTarget::NvmeSsd);

        let after_counter = COLLECTION_PIN_OPERATIONS_TOTAL
            .with_label_values(&["pin", "nvme_ssd"])
            .get();
        let after_gauge = COLLECTION_PINS_CURRENT
            .with_label_values(&["nvme_ssd"])
            .get();

        assert!((after_counter - before_counter - 1.0).abs() < 1e-9);
        assert_eq!(
            after_gauge, before_gauge,
            "record_pin must not adjust the gauge (caller's responsibility)"
        );
    }

    #[test]
    fn record_unpin_increments_counter_only() {
        let before_counter = COLLECTION_PIN_OPERATIONS_TOTAL
            .with_label_values(&["unpin", "cloud"])
            .get();
        let before_gauge = COLLECTION_PINS_CURRENT.with_label_values(&["cloud"]).get();

        record_unpin(CollectionPinTarget::Cloud);

        let after_counter = COLLECTION_PIN_OPERATIONS_TOTAL
            .with_label_values(&["unpin", "cloud"])
            .get();
        let after_gauge = COLLECTION_PINS_CURRENT.with_label_values(&["cloud"]).get();

        assert!((after_counter - before_counter - 1.0).abs() < 1e-9);
        assert_eq!(
            after_gauge, before_gauge,
            "record_unpin must not adjust the gauge (caller's responsibility)"
        );
    }

    #[test]
    fn inc_dec_current_pin_round_trip_leaves_gauge_unchanged() {
        // Re-pin semantics: a transition memory→nvme_ssd should
        // decrement memory by 1 and increment nvme_ssd by 1.
        let before_memory = COLLECTION_PINS_CURRENT.with_label_values(&["memory"]).get();
        let before_nvme = COLLECTION_PINS_CURRENT
            .with_label_values(&["nvme_ssd"])
            .get();

        inc_current_pin(CollectionPinTarget::Memory);
        dec_current_pin(CollectionPinTarget::Memory);
        inc_current_pin(CollectionPinTarget::NvmeSsd);

        assert_eq!(
            COLLECTION_PINS_CURRENT.with_label_values(&["memory"]).get(),
            before_memory,
            "memory gauge must net out to zero after balanced inc/dec"
        );
        assert_eq!(
            COLLECTION_PINS_CURRENT
                .with_label_values(&["nvme_ssd"])
                .get(),
            before_nvme + 1,
            "nvme_ssd gauge must reflect the new target after re-pin"
        );

        // Cleanup so this test doesn't poison gauges for sibling tests.
        dec_current_pin(CollectionPinTarget::NvmeSsd);
    }

    #[test]
    fn reset_current_pins_overwrites_gauge_to_authoritative_values() {
        // Dirty the gauges with some prior state then reset.
        inc_current_pin(CollectionPinTarget::Memory);
        inc_current_pin(CollectionPinTarget::Memory);
        inc_current_pin(CollectionPinTarget::NvmeSsd);

        reset_current_pins(7, 3, 1);

        assert_eq!(
            COLLECTION_PINS_CURRENT.with_label_values(&["memory"]).get(),
            7
        );
        assert_eq!(
            COLLECTION_PINS_CURRENT
                .with_label_values(&["nvme_ssd"])
                .get(),
            3
        );
        assert_eq!(
            COLLECTION_PINS_CURRENT.with_label_values(&["cloud"]).get(),
            1
        );
    }

    #[test]
    fn target_label_matches_pin_target_label() {
        // Operators wire dashboards against these strings via two
        // surfaces (REST + metrics); a mismatch silently breaks
        // alerts. Lock the agreement in here.
        assert_eq!(
            target_label(CollectionPinTarget::Memory),
            CollectionPinTarget::Memory.label()
        );
        assert_eq!(
            target_label(CollectionPinTarget::NvmeSsd),
            CollectionPinTarget::NvmeSsd.label()
        );
        assert_eq!(
            target_label(CollectionPinTarget::Cloud),
            CollectionPinTarget::Cloud.label()
        );
    }
}
