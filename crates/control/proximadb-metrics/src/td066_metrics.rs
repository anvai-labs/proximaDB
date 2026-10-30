/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! TD-066 canonical-WAL-authority observability metrics (Option E of the
//! TD-066 (c) Part 2 LSN-correlation design,
//! `docs/12-design/TD_066_PART2_LSN_CORRELATION_DESIGN_2026_05_28.adoc`).
//!
//! Today's commits (`b36a24b17` emission + `4ece74250` production wiring +
//! `18c846ae8` recovery read-side observability) make the canonical WAL
//! durably carry `CanonicalOperation::Checkpoint(SnapshotManifest)`
//! entries and let ORION recovery scan them. Recovery BEHAVIOR is
//! unchanged in Part 1; these metrics expose **whether the wiring is
//! healthy** so operators can detect three failure modes without
//! touching the durability path:
//!
//! 1. `orion_recovery_canonical_checkpoint_age_seconds` — wall-clock
//!    age (seconds since `timestamp_ms` on the manifest) of the latest
//!    Checkpoint for the graph being recovered. High values indicate
//!    either flush_wal isn't being called often enough OR the canonical
//!    WAL isn't being persisted (production wiring bug). Reported on
//!    every `OrionGraphEngine::recover` call.
//!
//! 2. `orion_recovery_canonical_checkpoint_present_total` — count of
//!    recoveries where a canonical checkpoint was found for the graph.
//!    Pair with `..._absent_total` below to compute a hit ratio across
//!    deployment rollouts.
//!
//! 3. `orion_recovery_canonical_checkpoint_absent_total` — count of
//!    recoveries where NO canonical checkpoint was found. Expected on
//!    fresh deployments and pre-wired (legacy) configurations; should
//!    drop to zero once production wiring is fully rolled out.
//!
//! Labels are kept low-cardinality (`graph_id`) — same cardinality
//! profile as the other graph metrics.

use crate::{register_counter_vec_safe, register_gauge_vec_safe};
use lazy_static::lazy_static;
use prometheus::{CounterVec, GaugeVec};

lazy_static! {
    /// Age (in seconds) of the latest canonical Checkpoint for the graph
    /// being recovered. Computed as `now_wall - manifest.timestamp_ms / 1000`.
    /// Reset on each `OrionGraphEngine::recover` call.
    ///
    /// **Operator interpretation**:
    /// * Stable low value across restarts → canonical emission + production
    ///   wiring are healthy.
    /// * Steadily rising across restarts → `flush_wal` not being called
    ///   often enough; consider operator-side periodic flush.
    /// * Unset / NaN → no canonical checkpoint found (see
    ///   `..._absent_total`).
    pub static ref ORION_RECOVERY_CANONICAL_CHECKPOINT_AGE_SECONDS: GaugeVec =
        register_gauge_vec_safe(
            "orion_recovery_canonical_checkpoint_age_seconds",
            "Wall-clock age of the latest canonical Checkpoint for this graph at \
             ORION recovery time (TD-066). High values indicate stale or missing \
             canonical-WAL persistence; sustained zero indicates emission + wiring \
             are healthy.",
            &["graph_id"],
        );

    /// Recoveries that found a canonical Checkpoint for the graph.
    /// Pair with `..._absent_total` to compute a presence ratio.
    pub static ref ORION_RECOVERY_CANONICAL_CHECKPOINT_PRESENT_TOTAL: CounterVec =
        register_counter_vec_safe(
            "orion_recovery_canonical_checkpoint_present_total",
            "Count of ORION recoveries that found a canonical Checkpoint for this \
             graph (TD-066 canonical WAL authority observability)",
            &["graph_id"],
        );

    /// Recoveries that found NO canonical Checkpoint for the graph.
    /// Expected on fresh deployments and pre-wired (legacy) configurations.
    pub static ref ORION_RECOVERY_CANONICAL_CHECKPOINT_ABSENT_TOTAL: CounterVec =
        register_counter_vec_safe(
            "orion_recovery_canonical_checkpoint_absent_total",
            "Count of ORION recoveries that found NO canonical Checkpoint for this \
             graph. Expected on fresh deployments and pre-wired (legacy) configs; \
             should drop toward zero as production wiring rolls out (TD-066).",
            &["graph_id"],
        );
}

/// Record the outcome of an `OrionPersistence::canonical_checkpoint_lsn`
/// call during `OrionGraphEngine::recover`.
///
/// * `checkpoint_lsn` — `Some(lsn)` when a Checkpoint was found for the
///   graph; `None` when the canonical WAL is missing, the file doesn't
///   exist yet, or no Checkpoint references this graph.
/// * `checkpoint_timestamp_ms` — the `SnapshotManifest.timestamp_ms`
///   from the same matching entry. Used to compute the age gauge.
///   Only meaningful when `checkpoint_lsn.is_some()`.
pub fn record_recovery_checkpoint_observation(
    graph_id: &str,
    checkpoint_lsn: Option<u64>,
    checkpoint_timestamp_ms: Option<u64>,
) {
    if checkpoint_lsn.is_some() {
        ORION_RECOVERY_CANONICAL_CHECKPOINT_PRESENT_TOTAL
            .with_label_values(&[graph_id])
            .inc();
        if let Some(ts_ms) = checkpoint_timestamp_ms {
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(ts_ms);
            let age_seconds = now_ms.saturating_sub(ts_ms) as f64 / 1000.0;
            ORION_RECOVERY_CANONICAL_CHECKPOINT_AGE_SECONDS
                .with_label_values(&[graph_id])
                .set(age_seconds);
        }
    } else {
        ORION_RECOVERY_CANONICAL_CHECKPOINT_ABSENT_TOTAL
            .with_label_values(&[graph_id])
            .inc();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_recovery_with_checkpoint_bumps_present_and_sets_age() {
        let graph_id = "test-graph-td066-metric-present";
        // Use a timestamp from 10 seconds ago.
        let ten_s_ago = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
            - 10_000;
        let before_present = ORION_RECOVERY_CANONICAL_CHECKPOINT_PRESENT_TOTAL
            .with_label_values(&[graph_id])
            .get();
        record_recovery_checkpoint_observation(graph_id, Some(42), Some(ten_s_ago));
        let after_present = ORION_RECOVERY_CANONICAL_CHECKPOINT_PRESENT_TOTAL
            .with_label_values(&[graph_id])
            .get();
        assert_eq!(after_present - before_present, 1.0);

        let age = ORION_RECOVERY_CANONICAL_CHECKPOINT_AGE_SECONDS
            .with_label_values(&[graph_id])
            .get();
        // Within 5 s slack to account for clock skew between the test
        // setup and the metric record.
        assert!(
            (9.0..=15.0).contains(&age),
            "age should reflect ~10s; got {}",
            age
        );
    }

    #[test]
    fn record_recovery_without_checkpoint_bumps_absent_only() {
        let graph_id = "test-graph-td066-metric-absent";
        let before_absent = ORION_RECOVERY_CANONICAL_CHECKPOINT_ABSENT_TOTAL
            .with_label_values(&[graph_id])
            .get();
        let before_present = ORION_RECOVERY_CANONICAL_CHECKPOINT_PRESENT_TOTAL
            .with_label_values(&[graph_id])
            .get();
        record_recovery_checkpoint_observation(graph_id, None, None);
        let after_absent = ORION_RECOVERY_CANONICAL_CHECKPOINT_ABSENT_TOTAL
            .with_label_values(&[graph_id])
            .get();
        let after_present = ORION_RECOVERY_CANONICAL_CHECKPOINT_PRESENT_TOTAL
            .with_label_values(&[graph_id])
            .get();
        assert_eq!(after_absent - before_absent, 1.0);
        assert_eq!(
            after_present, before_present,
            "absent must not bump present"
        );
    }
}
