// Copyright 2025 Vijaykumar Singh.
// Licensed under the Apache License, Version 2.0.

//! Prometheus metrics for AXIS HNSW recall-target drift observability.
//!
//! These metrics let operators chart recall drift over time and
//! configure alerts without polling
//! `GET /api/v2/_diagnostics/collections/:id/route-health` per
//! collection. They're populated by the route-health handler (one
//! per GET) and the `/recall-tune` handler (one per POST + one
//! when a hot-swap lands).
//!
//! # Metrics
//!
//! * `axis_recall_drift_status{collection,kind}` — one-hot **gauge**
//!   per (collection, kind). Exactly one `kind` label per collection
//!   is `1`, the others are `0`. The four kinds are the stable
//!   string literals also surfaced in the route-health
//!   `recall_drift.kind` field: `none`, `ef_search_only`,
//!   `rebuild_required`, `unwired`. The one-hot encoding makes
//!   alerts trivial — e.g.
//!   `axis_recall_drift_status{kind="rebuild_required"} == 1`
//!   fires when any collection needs reclustering.
//!
//! * `axis_recall_drift_observations_total{collection}` — count of
//!   times the route-health / recall-tune handlers ran
//!   `detect_recall_drift` for a given collection. Cheap proxy for
//!   "is anyone actually polling drift state for this collection?".
//!
//! * `axis_recall_drift_hot_swap_applied_total{collection}` — count
//!   of successful in-place ef_search hot-swaps via the
//!   `/recall-tune` endpoint. Lets operators see how often drift
//!   self-heals without rebuild.
//!
//! # Cardinality
//!
//! The `collection` label is high-cardinality (one series per
//! collection). The `kind` label is bounded to 4 values. Total
//! gauge series ≈ 4 × num_collections. For deployments with
//! over 10K collections this is the kind of label that should be
//! collapsed via Prometheus relabeling before persistence.

use crate::{register_counter_vec_safe, register_gauge_vec_safe};
use lazy_static::lazy_static;
use prometheus::{CounterVec, GaugeVec};

/// The four stable kind strings, ordered to match the route-health
/// `recall_drift.kind` enum. Exported so the route-health unit test
/// can assert exact-equal.
pub const DRIFT_KIND_LABELS: &[&str] = &["unwired", "none", "ef_search_only", "rebuild_required"];

lazy_static! {
    /// One-hot gauge: exactly one `kind` label per `collection` is
    /// 1, the others are 0. Mirrors the route-health
    /// `recall_drift.kind` field. See module docs for cardinality
    /// notes.
    pub static ref AXIS_RECALL_DRIFT_STATUS: GaugeVec = register_gauge_vec_safe(
        "axis_recall_drift_status",
        "AXIS HNSW recall-target drift state for a collection. One-hot \
         per (collection, kind). kind ∈ {unwired, none, ef_search_only, \
         rebuild_required}. Use `kind=\"rebuild_required\" == 1` for alerts.",
        &["collection", "kind"],
    );

    /// Cheap proxy for "is anyone actually polling drift state for
    /// this collection?". Bumped every time the route-health or
    /// recall-tune handlers run `detect_recall_drift`.
    pub static ref AXIS_RECALL_DRIFT_OBSERVATIONS_TOTAL: CounterVec = register_counter_vec_safe(
        "axis_recall_drift_observations_total",
        "Count of times detect_recall_drift ran for a collection \
         (via route-health GET or recall-tune POST).",
        &["collection"],
    );

    /// Successful in-place ef_search hot-swaps. Distinguished by
    /// the `trigger` label: `"operator"` (POST /recall-tune) or
    /// `"sweeper"` (RecallDriftSweeper auto-apply). Operators chart
    /// this to see how often drift self-heals without rebuild and
    /// who's driving the heal.
    pub static ref AXIS_RECALL_DRIFT_HOT_SWAP_APPLIED_TOTAL: CounterVec =
        register_counter_vec_safe(
            "axis_recall_drift_hot_swap_applied_total",
            "Count of successful hot-swap apply operations. The \
             `trigger` label distinguishes operator-driven \
             (via POST /recall-tune) from sweeper-driven \
             (automatic RecallDriftSweeper apply). Each operation \
             may have multiple per-spec changes — this counter is \
             per apply call, not per spec.",
            &["collection", "trigger"],
        );
}

/// Stable label values for the `trigger` dimension on
/// [`AXIS_RECALL_DRIFT_HOT_SWAP_APPLIED_TOTAL`]. Exported so
/// callers + dashboards reference the same literals.
pub const HOT_SWAP_TRIGGER_OPERATOR: &str = "operator";
pub const HOT_SWAP_TRIGGER_SWEEPER: &str = "sweeper";

/// Record an observation of drift state for a collection. Updates
/// the one-hot gauge so the chosen `kind` is 1 and the other three
/// kinds are 0, then bumps the observations counter.
///
/// `kind` must be one of [`DRIFT_KIND_LABELS`] — typically obtained
/// from the route-health `recall_drift.kind` field. Unknown values
/// are recorded against "unwired" (defensive; should not happen if
/// callers route through the typed `DriftKind` enum).
pub fn record_recall_drift_observation(collection: &str, kind: &str) {
    let resolved = if DRIFT_KIND_LABELS.contains(&kind) {
        kind
    } else {
        "unwired"
    };
    for &label in DRIFT_KIND_LABELS {
        let value = if label == resolved { 1.0 } else { 0.0 };
        AXIS_RECALL_DRIFT_STATUS
            .with_label_values(&[collection, label])
            .set(value);
    }
    AXIS_RECALL_DRIFT_OBSERVATIONS_TOTAL
        .with_label_values(&[collection])
        .inc();
}

/// Record a successful hot-swap apply for a collection. `trigger`
/// must be one of [`HOT_SWAP_TRIGGER_OPERATOR`] /
/// [`HOT_SWAP_TRIGGER_SWEEPER`] — keeps dashboards able to
/// distinguish manual from automatic resolution. One increment per
/// apply call, not per spec touched.
pub fn record_recall_drift_hot_swap_applied(collection: &str, trigger: &str) {
    AXIS_RECALL_DRIFT_HOT_SWAP_APPLIED_TOTAL
        .with_label_values(&[collection, trigger])
        .inc();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drift_kind_labels_match_route_health_enum() {
        // The route-health response uses these strings verbatim. Pin
        // both ends so a typo can't drift.
        assert_eq!(DRIFT_KIND_LABELS.len(), 4);
        assert!(DRIFT_KIND_LABELS.contains(&"unwired"));
        assert!(DRIFT_KIND_LABELS.contains(&"none"));
        assert!(DRIFT_KIND_LABELS.contains(&"ef_search_only"));
        assert!(DRIFT_KIND_LABELS.contains(&"rebuild_required"));
    }

    #[test]
    fn record_observation_one_hot_invariant() {
        let collection = "test_one_hot_collection_unique_name";
        record_recall_drift_observation(collection, "rebuild_required");

        // Exactly one kind == 1, the rest == 0.
        let mut active_count = 0;
        for &label in DRIFT_KIND_LABELS {
            let v = AXIS_RECALL_DRIFT_STATUS
                .with_label_values(&[collection, label])
                .get();
            if v == 1.0 {
                active_count += 1;
            } else {
                assert_eq!(v, 0.0, "kind {} should be 0", label);
            }
        }
        assert_eq!(
            active_count, 1,
            "exactly one kind label must be active (got {})",
            active_count
        );
    }

    #[test]
    fn record_observation_flips_correctly() {
        let collection = "test_flip_collection_unique_name";

        // Start at "ef_search_only".
        record_recall_drift_observation(collection, "ef_search_only");
        assert_eq!(
            AXIS_RECALL_DRIFT_STATUS
                .with_label_values(&[collection, "ef_search_only"])
                .get(),
            1.0
        );

        // Move to "none" — the previous "ef_search_only" must flip
        // back to 0.
        record_recall_drift_observation(collection, "none");
        assert_eq!(
            AXIS_RECALL_DRIFT_STATUS
                .with_label_values(&[collection, "none"])
                .get(),
            1.0
        );
        assert_eq!(
            AXIS_RECALL_DRIFT_STATUS
                .with_label_values(&[collection, "ef_search_only"])
                .get(),
            0.0,
            "previous kind must flip to 0"
        );
    }

    #[test]
    fn unknown_kind_clamps_to_unwired() {
        let collection = "test_unknown_collection_unique_name";
        record_recall_drift_observation(collection, "made_up_garbage");
        assert_eq!(
            AXIS_RECALL_DRIFT_STATUS
                .with_label_values(&[collection, "unwired"])
                .get(),
            1.0
        );
    }

    #[test]
    fn observations_counter_increments() {
        let collection = "test_obs_counter_unique_collection_name";
        let before = AXIS_RECALL_DRIFT_OBSERVATIONS_TOTAL
            .with_label_values(&[collection])
            .get();
        record_recall_drift_observation(collection, "none");
        record_recall_drift_observation(collection, "ef_search_only");
        let after = AXIS_RECALL_DRIFT_OBSERVATIONS_TOTAL
            .with_label_values(&[collection])
            .get();
        assert_eq!(after - before, 2.0);
    }

    #[test]
    fn hot_swap_counter_increments_per_trigger() {
        let collection = "test_hot_swap_counter_unique_collection_name";
        let op_before = AXIS_RECALL_DRIFT_HOT_SWAP_APPLIED_TOTAL
            .with_label_values(&[collection, HOT_SWAP_TRIGGER_OPERATOR])
            .get();
        let sw_before = AXIS_RECALL_DRIFT_HOT_SWAP_APPLIED_TOTAL
            .with_label_values(&[collection, HOT_SWAP_TRIGGER_SWEEPER])
            .get();
        record_recall_drift_hot_swap_applied(collection, HOT_SWAP_TRIGGER_OPERATOR);
        record_recall_drift_hot_swap_applied(collection, HOT_SWAP_TRIGGER_SWEEPER);
        record_recall_drift_hot_swap_applied(collection, HOT_SWAP_TRIGGER_SWEEPER);
        let op_after = AXIS_RECALL_DRIFT_HOT_SWAP_APPLIED_TOTAL
            .with_label_values(&[collection, HOT_SWAP_TRIGGER_OPERATOR])
            .get();
        let sw_after = AXIS_RECALL_DRIFT_HOT_SWAP_APPLIED_TOTAL
            .with_label_values(&[collection, HOT_SWAP_TRIGGER_SWEEPER])
            .get();
        assert_eq!(op_after - op_before, 1.0);
        assert_eq!(sw_after - sw_before, 2.0);
    }
}
