// Copyright 2025 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! TurboQuant scoring-kernel metrics (P8.A — ADR-021).
//!
//! Per `TURBOQUANT_LLD_2026_05_30.adoc` §"xCatalog & EXPLAIN Wiring"
//! Q12: the kernel-level allowlist mask path exposes a single counter,
//! `proximadb_turboquant_blocks_skipped_by_mask_total`, so operators can
//! quantify the selective-filter win delivered by P5's `CandidateMaskSet`
//! + the kernel's 32-vector block early-exit (`turboquant::mask::
//! block_has_allowed`).
//!
//! Labels follow the LLD Q12 convention: `collection_id` + `bit_width`.
//! Cardinality stays bounded because TurboQuant is the only producer of
//! the value (one row per `(collection, 2|4)` pair).
//!
//! Gated by `experimental-turboquant` so the default build never carries
//! the additional `prometheus::CounterVec` registration cost.

use crate::register_counter_vec_safe;
use lazy_static::lazy_static;
use prometheus::CounterVec;

lazy_static! {
    /// Cumulative count of 32-vector blocks short-circuited by the
    /// TurboQuant kernel's mask early-exit (`turboquant::mask::
    /// block_has_allowed`). Increments when a `CandidateMaskSet` is
    /// forwarded into a search call and the relevant bitmap window is
    /// zero, so the kernel skips the block before any LUT lookup or
    /// scoring work.
    ///
    /// Operator interpretation: a non-zero, growing value confirms
    /// selective filters are reaching the kernel rather than degrading
    /// to oversample-then-post-filter (LLD §"In-Kernel Allowlist").
    pub static ref TURBOQUANT_BLOCKS_SKIPPED_BY_MASK_TOTAL: CounterVec =
        register_counter_vec_safe(
            "proximadb_turboquant_blocks_skipped_by_mask_total",
            "Cumulative 32-vector blocks short-circuited by CandidateMaskSet early-exit \
             in the TurboQuant scoring kernel (ADR-021 / TURBOQUANT_LLD §11)",
            &["collection_id", "bit_width"],
        );
}

/// Record `n` block-skips for the given `(collection_id, bit_width)` pair.
///
/// Callers should pass the bit_width as a stable string token (`"2"` or
/// `"4"`) so it lines up with the label cardinality expected by Prometheus
/// scrapers. No-op when `n == 0`.
pub fn record_blocks_skipped(collection_id: &str, bit_width: &str, n: u64) {
    if n == 0 {
        return;
    }
    TURBOQUANT_BLOCKS_SKIPPED_BY_MASK_TOTAL
        .with_label_values(&[collection_id, bit_width])
        .inc_by(n as f64);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_blocks_skipped_increments_when_n_nonzero() {
        // Use a unique collection_id per test so this doesn't race with
        // any cumulative state from other tests in the same process.
        let coll = "tq_test_inc";
        let bw = "2";
        let before = TURBOQUANT_BLOCKS_SKIPPED_BY_MASK_TOTAL
            .with_label_values(&[coll, bw])
            .get();

        record_blocks_skipped(coll, bw, 7);

        let after = TURBOQUANT_BLOCKS_SKIPPED_BY_MASK_TOTAL
            .with_label_values(&[coll, bw])
            .get();
        assert!(
            (after - before - 7.0).abs() < 1e-6,
            "expected delta of 7, got {} → {}",
            before,
            after,
        );
    }

    #[test]
    fn record_blocks_skipped_zero_is_noop() {
        let coll = "tq_test_zero";
        let bw = "4";
        let before = TURBOQUANT_BLOCKS_SKIPPED_BY_MASK_TOTAL
            .with_label_values(&[coll, bw])
            .get();

        record_blocks_skipped(coll, bw, 0);

        let after = TURBOQUANT_BLOCKS_SKIPPED_BY_MASK_TOTAL
            .with_label_values(&[coll, bw])
            .get();
        assert!(
            (after - before).abs() < 1e-6,
            "n=0 must not advance the counter; saw {} → {}",
            before,
            after,
        );
    }

    #[test]
    fn record_blocks_skipped_distinct_labels_independent() {
        let bw = "2";
        let c1 = "tq_test_labelA";
        let c2 = "tq_test_labelB";
        let c1_before = TURBOQUANT_BLOCKS_SKIPPED_BY_MASK_TOTAL
            .with_label_values(&[c1, bw])
            .get();
        let c2_before = TURBOQUANT_BLOCKS_SKIPPED_BY_MASK_TOTAL
            .with_label_values(&[c2, bw])
            .get();

        record_blocks_skipped(c1, bw, 3);

        let c1_after = TURBOQUANT_BLOCKS_SKIPPED_BY_MASK_TOTAL
            .with_label_values(&[c1, bw])
            .get();
        let c2_after = TURBOQUANT_BLOCKS_SKIPPED_BY_MASK_TOTAL
            .with_label_values(&[c2, bw])
            .get();
        assert!((c1_after - c1_before - 3.0).abs() < 1e-6);
        // Incrementing c1 must NOT advance c2's counter.
        assert!((c2_after - c2_before).abs() < 1e-6);
    }

    #[test]
    fn record_blocks_skipped_bit_widths_separate_series() {
        let coll = "tq_test_bw";
        let bw2 = "2";
        let bw4 = "4";
        let bw2_before = TURBOQUANT_BLOCKS_SKIPPED_BY_MASK_TOTAL
            .with_label_values(&[coll, bw2])
            .get();
        let bw4_before = TURBOQUANT_BLOCKS_SKIPPED_BY_MASK_TOTAL
            .with_label_values(&[coll, bw4])
            .get();

        record_blocks_skipped(coll, bw2, 5);

        let bw2_after = TURBOQUANT_BLOCKS_SKIPPED_BY_MASK_TOTAL
            .with_label_values(&[coll, bw2])
            .get();
        let bw4_after = TURBOQUANT_BLOCKS_SKIPPED_BY_MASK_TOTAL
            .with_label_values(&[coll, bw4])
            .get();
        assert!((bw2_after - bw2_before - 5.0).abs() < 1e-6);
        assert!((bw4_after - bw4_before).abs() < 1e-6);
    }
}
