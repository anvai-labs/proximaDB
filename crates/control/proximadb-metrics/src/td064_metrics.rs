/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! TD-064 predicate-aware vector search metrics
//!
//! Operator observability for the predicate-aware search path. Counters here
//! are populated by `AxisManager` when post-filter trimming drops the
//! survivor count below the requested `top_k` — the same condition that
//! surfaces as `predicate_shortfall` on `SearchPlanTrace`.
//!
//! Labels are kept low-cardinality (collection_id, ann_filtering_mode) so
//! these are safe to scrape from Prometheus at default cadence.

use crate::register_counter_vec_safe;
use lazy_static::lazy_static;
use prometheus::CounterVec;

lazy_static! {
    /// Number of TD-064 predicate-aware searches that returned fewer matches
    /// than the requested `top_k` after post-filter trimming.
    ///
    /// A non-zero value indicates either:
    /// * the post-filter shrank the candidate pool below `top_k` (legitimate
    ///   shortfall — caller should consider PreFilter mode), or
    /// * cached metadata excluded records under fail-closed policy (records
    ///   inserted before TD-064 wiring landed; rebuild the index).
    pub static ref AXIS_PREDICATE_SHORTFALL_TOTAL: CounterVec = register_counter_vec_safe(
        "axis_predicate_shortfall_total",
        "Count of predicate-aware searches that returned fewer than top_k results (TD-064)",
        &["collection_id", "ann_filtering_mode"],
    );

    /// Sum of missing results across all shortfall events
    /// (`requested_k - returned_k`). Combined with `..._total`, lets
    /// operators compute average shortfall magnitude.
    pub static ref AXIS_PREDICATE_SHORTFALL_MISSING_SUM: CounterVec = register_counter_vec_safe(
        "axis_predicate_shortfall_missing_sum",
        "Sum of missing-result counts across predicate-shortfall events (TD-064)",
        &["collection_id", "ann_filtering_mode"],
    );
}

/// Record a predicate-shortfall event. Both counters are incremented atomically.
pub fn record_shortfall(
    collection_id: &str,
    ann_filtering_mode: &str,
    requested_k: u32,
    returned_k: u32,
) {
    let missing = requested_k.saturating_sub(returned_k);
    if missing == 0 {
        return;
    }
    AXIS_PREDICATE_SHORTFALL_TOTAL
        .with_label_values(&[collection_id, ann_filtering_mode])
        .inc();
    AXIS_PREDICATE_SHORTFALL_MISSING_SUM
        .with_label_values(&[collection_id, ann_filtering_mode])
        .inc_by(missing as f64);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_shortfall_increments_when_missing_nonzero() {
        let before_total = AXIS_PREDICATE_SHORTFALL_TOTAL
            .with_label_values(&["coll_a", "inline"])
            .get();
        let before_sum = AXIS_PREDICATE_SHORTFALL_MISSING_SUM
            .with_label_values(&["coll_a", "inline"])
            .get();

        record_shortfall("coll_a", "inline", 10, 3);

        let after_total = AXIS_PREDICATE_SHORTFALL_TOTAL
            .with_label_values(&["coll_a", "inline"])
            .get();
        let after_sum = AXIS_PREDICATE_SHORTFALL_MISSING_SUM
            .with_label_values(&["coll_a", "inline"])
            .get();

        assert!((after_total - before_total - 1.0).abs() < 1e-9);
        assert!((after_sum - before_sum - 7.0).abs() < 1e-9);
    }

    #[test]
    fn record_shortfall_is_no_op_when_returned_meets_requested() {
        let before_total = AXIS_PREDICATE_SHORTFALL_TOTAL
            .with_label_values(&["coll_b", "inline"])
            .get();

        record_shortfall("coll_b", "inline", 10, 10);
        record_shortfall("coll_b", "inline", 10, 12);

        let after_total = AXIS_PREDICATE_SHORTFALL_TOTAL
            .with_label_values(&["coll_b", "inline"])
            .get();

        assert_eq!(before_total, after_total);
    }
}
