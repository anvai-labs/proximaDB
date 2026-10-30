/*
 * Copyright 2025 ProximaDB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! Tier-migration pipeline metrics
//!
//! Operator observability for the tier-migration pipeline. The
//! [`TierMigrationExecutor`](crate::storage::tiering::TierMigrationExecutor)
//! increments these counters/gauges/histograms on every `execute` call,
//! providing live throughput, success rate, and saturation signals.
//!
//! ## Label cardinality
//!
//! Labels are intentionally low-cardinality:
//! * `result` — `"success"` or `"failed"` (2 values)
//! * `source` / `target` — `PerformanceTier` (4 values each: Hot/Warm/Cold/Archive)
//! * `collection_id` — bounded by tenant count
//!
//! Migration durations span seconds (local rename) to minutes (cloud
//! multipart copy of large segments), so the histogram buckets cover
//! ~10ms → ~1hr in roughly half-decade steps.

use lazy_static::lazy_static;
use prometheus::{CounterVec, HistogramVec, IntGauge};

use crate::storage::tiering::MigrationResult;
use crate::storage::tiering::policy::PerformanceTier;
use proximadb_metrics::{
    register_counter_vec_safe, register_histogram_vec_safe, register_int_gauge_safe,
};

/// Bucket layout for the migration duration histogram. Spans 10ms →
/// 1hr in roughly half-decade steps so both local renames (~ms) and
/// cloud cross-region copies (~minutes) land in distinct buckets.
fn migration_duration_buckets() -> Vec<f64> {
    vec![
        0.01,   // 10 ms — local rename
        0.05,   // 50 ms — local copy of small file
        0.25,   // 250 ms — local copy of medium file
        1.0,    // 1 s — small cloud upload
        5.0,    // 5 s — medium cloud transfer
        30.0,   // 30 s — large cloud transfer
        120.0,  // 2 min — multipart cloud copy
        600.0,  // 10 min — very large transfer
        1800.0, // 30 min — cross-region large transfer
        3600.0, // 1 hr — pathological / hung migration
    ]
}

lazy_static! {
    /// Total migrations attempted, labelled by outcome + tier transition
    /// + collection. Operators alert on `result="failed"` rate spikes
    /// and dashboard the success-vs-failure split per tier transition.
    pub static ref TIER_MIGRATIONS_TOTAL: CounterVec = register_counter_vec_safe(
        "proximadb_tier_migrations_total",
        "Total tier-migration attempts (TierMigrationExecutor.execute calls)",
        &["result", "source", "target", "collection_id"],
    );

    /// Bytes moved during migrations, labelled by outcome + tier
    /// transition. For failed migrations this is 0 (we use bytes_migrated
    /// from MigrationResult which is set to the actual transferred size
    /// on success). Combine with `..._total` to get average migration
    /// size: `rate(bytes_total) / rate(total)`.
    pub static ref TIER_MIGRATION_BYTES_TOTAL: CounterVec = register_counter_vec_safe(
        "proximadb_tier_migration_bytes_total",
        "Total bytes moved by the tier-migration executor",
        &["result", "source", "target"],
    );

    /// Per-migration duration histogram. Drives latency SLOs and the
    /// "is the executor hung?" alert (look for `_bucket{le="+Inf"}` that
    /// haven't incremented in N minutes). Bucket layout covers local
    /// rename (~10ms) through cloud cross-region transfer (~1hr) in
    /// roughly half-decade steps.
    pub static ref TIER_MIGRATION_DURATION_SECONDS: HistogramVec = register_histogram_vec_safe(
        "proximadb_tier_migration_duration_seconds",
        "Wall-clock duration of individual tier-migration attempts",
        &["result", "source", "target"],
        migration_duration_buckets(),
    );

    /// In-flight migration count. Incremented at the start of
    /// `execute`, decremented at the end (success or failure). A
    /// rising-and-not-falling gauge indicates a stuck migration.
    pub static ref TIER_MIGRATION_IN_FLIGHT: IntGauge = register_int_gauge_safe(
        "proximadb_tier_migration_in_flight",
        "Number of migrations currently being executed (in flight)",
    );
}

/// Format a `PerformanceTier` as a stable, low-cardinality label.
/// Operators wire dashboards against these exact strings; keep the
/// match exhaustive so a new tier variant triggers a compile error.
pub fn tier_label(tier: PerformanceTier) -> &'static str {
    match tier {
        PerformanceTier::Hot => "hot",
        PerformanceTier::Warm => "warm",
        PerformanceTier::Cold => "cold",
        PerformanceTier::Archive => "archive",
    }
}

/// RAII guard for the in-flight gauge. `inc` on construction, `dec`
/// on drop — guarantees the gauge is decremented even if `execute`
/// panics or returns early.
pub struct InFlightGuard;

impl InFlightGuard {
    pub fn enter() -> Self {
        TIER_MIGRATION_IN_FLIGHT.inc();
        Self
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        TIER_MIGRATION_IN_FLIGHT.dec();
    }
}

/// Record a completed migration attempt. Increments the total
/// counter, the bytes counter (only for successes, since failed
/// migrations have 0 bytes moved), and observes the duration histogram.
///
/// Designed to be called immediately after `MigrationResult` is
/// produced — typically the last line of `TierMigrationExecutor.execute`
/// before returning.
pub fn record_migration_result(result: &MigrationResult) {
    let result_label = if result.success { "success" } else { "failed" };
    let source = tier_label(result.source_tier);
    let target = tier_label(result.target_tier);

    TIER_MIGRATIONS_TOTAL
        .with_label_values(&[result_label, source, target, &result.collection])
        .inc();

    // Only count bytes on success — a failed migration that copied
    // 0 bytes should not be visible as bandwidth usage. (The total
    // counter still increments, so the failure rate is observable.)
    if result.success {
        TIER_MIGRATION_BYTES_TOTAL
            .with_label_values(&[result_label, source, target])
            .inc_by(result.bytes_migrated as f64);
    }

    TIER_MIGRATION_DURATION_SECONDS
        .with_label_values(&[result_label, source, target])
        .observe(result.duration.as_secs_f64());
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn sample_result(success: bool, bytes: u64, duration_ms: u64) -> MigrationResult {
        MigrationResult {
            task_id: "test-1".to_string(),
            collection: "test_collection".to_string(),
            item_id: "seg-1.sst".to_string(),
            source_tier: PerformanceTier::Hot,
            target_tier: PerformanceTier::Cold,
            success,
            bytes_migrated: bytes,
            duration: Duration::from_millis(duration_ms),
            error: if success {
                None
            } else {
                Some("test failure".to_string())
            },
        }
    }

    #[test]
    fn record_success_increments_total_and_bytes_counters() {
        let before_total = TIER_MIGRATIONS_TOTAL
            .with_label_values(&["success", "hot", "cold", "test_collection"])
            .get();
        let before_bytes = TIER_MIGRATION_BYTES_TOTAL
            .with_label_values(&["success", "hot", "cold"])
            .get();

        record_migration_result(&sample_result(true, 4096, 100));

        let after_total = TIER_MIGRATIONS_TOTAL
            .with_label_values(&["success", "hot", "cold", "test_collection"])
            .get();
        let after_bytes = TIER_MIGRATION_BYTES_TOTAL
            .with_label_values(&["success", "hot", "cold"])
            .get();

        assert!((after_total - before_total - 1.0).abs() < 1e-9);
        assert!((after_bytes - before_bytes - 4096.0).abs() < 1e-9);
    }

    #[test]
    fn record_failure_increments_total_but_not_bytes() {
        // Use a distinct collection to avoid cross-test interference
        // on the labelled total counter.
        let mut r = sample_result(false, 0, 50);
        r.collection = "test_failure_collection".to_string();

        let before_total = TIER_MIGRATIONS_TOTAL
            .with_label_values(&["failed", "hot", "cold", "test_failure_collection"])
            .get();
        let before_bytes = TIER_MIGRATION_BYTES_TOTAL
            .with_label_values(&["failed", "hot", "cold"])
            .get();

        record_migration_result(&r);

        let after_total = TIER_MIGRATIONS_TOTAL
            .with_label_values(&["failed", "hot", "cold", "test_failure_collection"])
            .get();
        let after_bytes = TIER_MIGRATION_BYTES_TOTAL
            .with_label_values(&["failed", "hot", "cold"])
            .get();

        assert!(
            (after_total - before_total - 1.0).abs() < 1e-9,
            "failed migrations must still increment the total counter"
        );
        assert!(
            (after_bytes - before_bytes).abs() < 1e-9,
            "failed migrations must not advance the bytes counter"
        );
    }

    #[test]
    fn in_flight_guard_increments_and_decrements() {
        let before = TIER_MIGRATION_IN_FLIGHT.get();
        {
            let _guard = InFlightGuard::enter();
            assert_eq!(
                TIER_MIGRATION_IN_FLIGHT.get(),
                before + 1,
                "guard must increment in-flight gauge on enter"
            );
        }
        assert_eq!(
            TIER_MIGRATION_IN_FLIGHT.get(),
            before,
            "guard must decrement in-flight gauge on drop"
        );
    }

    #[test]
    fn in_flight_guard_decrements_even_on_unwind() {
        // Verify Drop runs even when the surrounding scope panics
        // (compile-time check via std::panic::catch_unwind).
        let before = TIER_MIGRATION_IN_FLIGHT.get();
        let result = std::panic::catch_unwind(|| {
            let _guard = InFlightGuard::enter();
            assert_eq!(TIER_MIGRATION_IN_FLIGHT.get(), before + 1);
            panic!("force unwind");
        });
        assert!(result.is_err(), "test setup must have panicked");
        assert_eq!(
            TIER_MIGRATION_IN_FLIGHT.get(),
            before,
            "in-flight gauge must decrement on panic unwind too"
        );
    }

    #[test]
    fn tier_label_is_stable_for_dashboards() {
        // Operators wire dashboards against these strings — change
        // them and you break alerts. This test is a change-detector.
        assert_eq!(tier_label(PerformanceTier::Hot), "hot");
        assert_eq!(tier_label(PerformanceTier::Warm), "warm");
        assert_eq!(tier_label(PerformanceTier::Cold), "cold");
        assert_eq!(tier_label(PerformanceTier::Archive), "archive");
    }
}
