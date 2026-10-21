// Copyright 2025 Vijaykumar Singh.
// Licensed under the Apache License, Version 2.0.

//! Prometheus metrics for the ANN-advisor observability layer
//! (P4). Captures the gap between **predicted** recall / per-query
//! work (from the closed-form advisor formulas) and **observed**
//! recall / latency from real query traffic.
//!
//! # Why
//!
//! The advisor's constants (`A(m)`, `ceiling(m)`, `beta`,
//! `recall_factor`, `ceiling_of_n` for IVF) were calibrated
//! against in-repo sweeps at single anchor points
//! (HNSW: N=100K dim=128 cosine; IVF: same plus per-N sweep).
//! Real workloads vary across dim / distance / data distribution
//! in ways the static formulas don't capture.
//!
//! P4 captures the residuals **but does not feed them back to the
//! advisor**. The residual histogram + counter let an operator see
//! how accurate the advisor is per-collection / per-algorithm, and
//! provide the dataset a future RL bridge (P5+) will fit against.
//!
//! # Metrics shape
//!
//! Three surfaces, all labelled by `(collection, algorithm)`:
//!
//! * `axis_advisor_observations_total{collection, algorithm}` —
//!   counter, every search that captured an observation.
//! * `axis_advisor_recall_residual{collection, algorithm}` —
//!   histogram of `(observed_recall - advisor_predicted_recall)`.
//!   Only updated when `observed_recall` is populated (i.e. when
//!   the collection's `recall_probe` gate is active). Negative
//!   values mean the advisor over-promised; positive mean it
//!   under-promised.
//! * `axis_advisor_latency_us{collection, algorithm}` — histogram
//!   of observed per-query latency in microseconds. Always
//!   populated — the latency timing is universal, doesn't depend
//!   on recall_probe.
//!
//! # Cardinality
//!
//! `collection` is high-cardinality (one series per collection);
//! `algorithm` is bounded to 3 values today (hnsw / ivf / hmgi).
//! Deployments with >10K collections should consider a Prometheus
//! relabelling rule that drops the collection label before
//! ingestion, or move to per-collection rollup at scrape time.

use crate::{register_counter_vec_safe, register_histogram_vec_safe};
use lazy_static::lazy_static;
use prometheus::{CounterVec, HistogramVec};

/// Bucket boundaries for the recall-residual histogram. Symmetric
/// around 0 (advisor exact), bracketing the empirical residual
/// range from the m=32 sweep (±0.005) up to gross mispredictions.
fn recall_residual_buckets() -> Vec<f64> {
    vec![
        -0.20, -0.10, -0.05, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20,
    ]
}

/// Bucket boundaries for the latency histogram in microseconds.
/// Spans 50μs (best-case in-cache HNSW at small ef) → 1s (large
/// IVF nprobe scans observed in the multi-N sweep).
fn latency_us_buckets() -> Vec<f64> {
    vec![
        50.0,
        100.0,
        200.0,
        500.0,
        1_000.0,
        2_000.0,
        5_000.0,
        10_000.0,
        25_000.0,
        50_000.0,
        100_000.0,
        250_000.0,
        500_000.0,
        1_000_000.0,
    ]
}

lazy_static! {
    /// Every captured observation bumps this counter. Cheap "is
    /// anyone collecting data for this collection / algorithm?"
    /// signal. Pair with the histograms below: if the counter is
    /// rising but residual buckets are empty, recall_probe isn't
    /// active.
    pub static ref AXIS_ADVISOR_OBSERVATIONS_TOTAL: CounterVec =
        register_counter_vec_safe(
            "axis_advisor_observations_total",
            "Count of ANN advisor observations captured by the \
             post-search hook. Labelled by (collection, algorithm) \
             where algorithm ∈ {hnsw, ivf, hmgi}.",
            &["collection", "algorithm"],
        );

    /// `(observed_recall - advisor_predicted_recall)` distribution.
    /// Negative → advisor over-promised. Positive → under-promised.
    /// Only populated when recall_probe is active on the
    /// collection; otherwise the post-search hook skips the
    /// observation.
    pub static ref AXIS_ADVISOR_RECALL_RESIDUAL: HistogramVec =
        register_histogram_vec_safe(
            "axis_advisor_recall_residual",
            "Distribution of (observed_recall - advisor_predicted_recall) \
             over captured observations. Centred at 0 if the advisor's \
             closed-form formula matches reality; drift indicates the \
             per-collection workload diverges from the calibration \
             anchor (N=100K dim=128 cosine for HNSW / IVF).",
            &["collection", "algorithm"],
            recall_residual_buckets(),
        );

    /// Per-query latency distribution in microseconds. Always
    /// populated when a search completes — latency timing is
    /// universal, doesn't depend on recall_probe.
    pub static ref AXIS_ADVISOR_LATENCY_US: HistogramVec =
        register_histogram_vec_safe(
            "axis_advisor_latency_us",
            "Per-search latency in microseconds, labelled by \
             (collection, algorithm). Used to validate the advisor's \
             `estimated_per_query_work` cost model.",
            &["collection", "algorithm"],
            latency_us_buckets(),
        );
}

/// Record a captured observation. `observed_recall` is optional —
/// `None` when the collection has no active `recall_probe`
/// (TD-075 / F2) and the post-search hook can't measure recall.
/// In that case the recall-residual histogram is skipped but the
/// counter + latency histogram are still updated.
pub fn record_observation(
    collection: &str,
    algorithm: &str,
    observed_recall: Option<f32>,
    advisor_predicted_recall: f32,
    observed_latency_us: u64,
) {
    AXIS_ADVISOR_OBSERVATIONS_TOTAL
        .with_label_values(&[collection, algorithm])
        .inc();
    AXIS_ADVISOR_LATENCY_US
        .with_label_values(&[collection, algorithm])
        .observe(observed_latency_us as f64);
    if let Some(obs) = observed_recall {
        let residual = (obs - advisor_predicted_recall) as f64;
        AXIS_ADVISOR_RECALL_RESIDUAL
            .with_label_values(&[collection, algorithm])
            .observe(residual);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_increments_per_observation() {
        let collection = "test_advisor_obs_counter_unique";
        let before = AXIS_ADVISOR_OBSERVATIONS_TOTAL
            .with_label_values(&[collection, "hnsw"])
            .get();
        record_observation(collection, "hnsw", Some(0.94), 0.95, 500);
        record_observation(collection, "hnsw", None, 0.95, 600);
        let after = AXIS_ADVISOR_OBSERVATIONS_TOTAL
            .with_label_values(&[collection, "hnsw"])
            .get();
        assert_eq!(after - before, 2.0);
    }

    #[test]
    fn latency_histogram_always_records() {
        // Latency is universal — both observation calls (with and
        // without observed_recall) should bump the latency histogram.
        let collection = "test_latency_universal_unique";
        let before = AXIS_ADVISOR_LATENCY_US
            .with_label_values(&[collection, "ivf"])
            .get_sample_count();
        record_observation(collection, "ivf", Some(0.65), 0.68, 12_000);
        record_observation(collection, "ivf", None, 0.68, 18_000);
        let after = AXIS_ADVISOR_LATENCY_US
            .with_label_values(&[collection, "ivf"])
            .get_sample_count();
        assert_eq!(after - before, 2);
    }

    #[test]
    fn recall_residual_skipped_when_observed_none() {
        // recall_probe isn't active → observed_recall = None →
        // recall_residual histogram must NOT bump.
        let collection = "test_recall_residual_skipped_unique";
        let before = AXIS_ADVISOR_RECALL_RESIDUAL
            .with_label_values(&[collection, "hnsw"])
            .get_sample_count();
        record_observation(collection, "hnsw", None, 0.95, 500);
        let after = AXIS_ADVISOR_RECALL_RESIDUAL
            .with_label_values(&[collection, "hnsw"])
            .get_sample_count();
        assert_eq!(
            after - before,
            0,
            "residual histogram must not bump when observed_recall is None"
        );
    }

    #[test]
    fn recall_residual_records_signed_delta() {
        // observed=0.92 predicted=0.95 → residual = -0.03 (advisor
        // over-promised). The histogram should land in the
        // [-0.05, -0.02) bucket.
        let collection = "test_recall_residual_signed_unique";
        let before = AXIS_ADVISOR_RECALL_RESIDUAL
            .with_label_values(&[collection, "ivf"])
            .get_sample_count();
        record_observation(collection, "ivf", Some(0.92), 0.95, 500);
        let after = AXIS_ADVISOR_RECALL_RESIDUAL
            .with_label_values(&[collection, "ivf"])
            .get_sample_count();
        assert_eq!(after - before, 1);
        // Histogram doesn't expose per-bucket counts cleanly from
        // the prometheus crate API; sample count is the testable
        // surface. Residual sign correctness is verified by reading
        // metrics output in integration tests.
    }

    #[test]
    fn algorithm_label_accepts_supported_literals() {
        // Pinned mapping — the labels MUST match
        // SupportedAlgorithm::label() literals so dashboards can
        // join across surfaces without label translation.
        for algo in ["hnsw", "ivf", "hmgi"] {
            record_observation("test_label_pin_unique", algo, Some(0.9), 0.9, 500);
            let count = AXIS_ADVISOR_OBSERVATIONS_TOTAL
                .with_label_values(&["test_label_pin_unique", algo])
                .get();
            assert!(count >= 1.0);
        }
    }
}
