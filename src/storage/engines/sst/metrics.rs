// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0
//! TD-RDSTRAT-8 two-level IVF coarse-probe operator metrics (SST engine).
//!
//! Aggregate operator view of the coarse probe — whether it is engaging, how
//! much of Region-A it avoids reading, and how often it falls back to the whole
//! region scan. These are the signals the default-on promotion gate watches
//! (`proximadb_ivf_whole_region_fallback_total` climbing ⇒ nprobe too small).
//!
//! Global counters (no collection label): the per-query, per-tenant *durable*
//! record lives in the io_trace warehouse `VectorAnnPayload` (ADR-066); this
//! Prometheus surface is the real-time aggregate operator dashboard. A
//! collection label can be layered on once the warehouse data shows
//! per-collection demand — keeping cardinality intentional, not eager.

use lazy_static::lazy_static;
use prometheus::{IntCounter, IntCounterVec, register_int_counter, register_int_counter_vec};

fn counter(name: &str, help: &str) -> IntCounter {
    register_int_counter!(name, help).unwrap_or_else(|_| {
        IntCounter::new(format!("{name}_fallback"), help)
            .unwrap_or_else(|_| unreachable!("valid counter metric descriptor"))
    })
}

// NOTE: unlabeled AGGREGATE operator surface. The per-tenant billing versions
// in metrics/consumption_metrics.rs own the canonical
// `proximadb_ivf_*_total{tenant_id}` names (sum over tenant = this aggregate);
// these carry `_agg_` to avoid a duplicate-registration crash when both init
// on a probe-armed search (default-ON since #1210). In-process readers use the
// Rust statics below, unaffected by the prometheus name.
lazy_static! {
    pub static ref IVF_CELLS_TOTAL: IntCounter = counter(
        "proximadb_ivf_cells_agg_total",
        "Persisted coarse centroids seen by the TD-RDSTRAT-8 two-level IVF coarse probe",
    );
    pub static ref IVF_CELLS_PROBED: IntCounter = counter(
        "proximadb_ivf_cells_probed_agg_total",
        "Coarse centroids ranked in RAM by the nprobe-scoped TD-RDSTRAT-8 probe",
    );
    pub static ref IVF_PROBED_ROWS: IntCounter = counter(
        "proximadb_ivf_probed_rows_agg_total",
        "Rows ranged-read from the probed cells' Region-A extents (TD-RDSTRAT-8)",
    );
    pub static ref IVF_FETCH_ROUNDS: IntCounter = counter(
        "proximadb_ivf_fetch_rounds_agg_total",
        "Coalesced Region-A ranged-read runs issued by the TD-RDSTRAT-8 probe (the GET round-trip budget it pays)",
    );
    pub static ref IVF_WHOLE_REGION_FALLBACK: IntCounter = counter(
        "proximadb_ivf_whole_region_fallback_total",
        "Segments where the armed TD-RDSTRAT-8 coarse probe missed and the read fell back to the whole Region-A scan",
    );
    pub static ref PAX_TIER_GET_OPS: IntCounterVec = register_int_counter_vec!(
        "proximadb_pax_tier_get_ops_agg_total",
        "Physical ranged GETs per PAX cascade tier (TD-RDSTRAT-13 C1; tier labels = CacheTier::label)",
        &["tier"]
    )
    .unwrap_or_else(|_| IntCounterVec::new(
        prometheus::Opts::new("proximadb_pax_tier_get_ops_agg_total_fallback", "fallback"),
        &["tier"]
    )
    .unwrap_or_else(|_| unreachable!("valid counter-vec metric descriptor")));
    pub static ref PAX_TIER_GET_BYTES: IntCounterVec = register_int_counter_vec!(
        "proximadb_pax_tier_get_bytes_agg_total",
        "Physical ranged-GET bytes per PAX cascade tier (TD-RDSTRAT-13 C1)",
        &["tier"]
    )
    .unwrap_or_else(|_| IntCounterVec::new(
        prometheus::Opts::new("proximadb_pax_tier_get_bytes_agg_total_fallback", "fallback"),
        &["tier"]
    )
    .unwrap_or_else(|_| unreachable!("valid counter-vec metric descriptor")));
    pub static ref PAX_METADATA_OPS: IntCounter = counter(
        "proximadb_pax_metadata_ops_agg_total",
        "Object-storage HEAD/metadata ops on the SST search path (TD-RDSTRAT-13 C3 removes the per-segment HEAD)",
    );
    pub static ref IVF_PROBE_RANK_US: IntCounter = counter(
        "proximadb_ivf_probe_rank_us_agg_total",
        "Cumulative coarse-probe rank wall microseconds incl. morsel join (TD-RDSTRAT-13 C2 compute term)",
    );
    pub static ref IVF_PROBE_RANK_CALLS: IntCounter = counter(
        "proximadb_ivf_probe_rank_calls_agg_total",
        "Coarse-probe rank invocations (divide IVF_PROBE_RANK_US by this for the mean compute term)",
    );
    pub static ref REGION_B_RERANK_US: IntCounter = counter(
        "proximadb_region_b_rerank_us_agg_total",
        "Cumulative Region-B survivor fetch+rerank wall microseconds (TD-RDSTRAT-13 PR-B tail attribution)",
    );
    pub static ref REGION_D_FETCH_DECODE_US: IntCounter = counter(
        "proximadb_region_d_fetch_decode_us_agg_total",
        "Cumulative Region-D top-k OID fetch+decode wall microseconds (TD-RDSTRAT-13 PR-B tail attribution)",
    );
    pub static ref REHYDRATE_US: IntCounter = counter(
        "proximadb_rehydrate_us_agg_total",
        "Cumulative top-k rehydration wall microseconds (TD-RDSTRAT-13 PR-B)",
    );
    pub static ref CASCADE_TOTAL_US: IntCounter = counter(
        "proximadb_cascade_total_us_agg_total",
        "Cumulative whole-cascade wall microseconds (TD-RDSTRAT-13 PR-B umbrella)",
    );
}

/// Tier label for the per-tier aggregate counters. Must stay in sync with the
/// SST `CacheTier` declaration order (the io_trace `tier_get_ops` index
/// contract): IdxA, Ctl, Meta, ProbeA, Surv, OID.
pub const TIER_LABELS: [&str; 6] = ["IdxA", "Ctl", "Meta", "ProbeA", "Surv", "OID"];

/// Record a coarse-probe outcome to the aggregate operator surface. Mirrors the
/// durable per-query record in `io_trace::record_ivf_coarse_probe` (which lands
/// in the warehouse `VectorAnnPayload`); this is the real-time operator aggregate.
pub fn record_ivf_coarse_probe(
    cells_total: u64,
    cells_probed: u64,
    probed_rows: u64,
    fetch_rounds: u64,
    whole_region_fallback: bool,
) {
    IVF_CELLS_TOTAL.inc_by(cells_total);
    IVF_CELLS_PROBED.inc_by(cells_probed);
    IVF_PROBED_ROWS.inc_by(probed_rows);
    IVF_FETCH_ROUNDS.inc_by(fetch_rounds);
    if whole_region_fallback {
        IVF_WHOLE_REGION_FALLBACK.inc();
    }
}

/// Record one physical ranged GET to its PAX cascade tier (TD-RDSTRAT-13 C1).
/// `tier_idx` indexes [`TIER_LABELS`]; out-of-range is dropped (drift must
/// fail tests, not fold tiers).
pub fn record_tier_get(tier_idx: usize, bytes: u64) {
    if tier_idx < TIER_LABELS.len() {
        PAX_TIER_GET_OPS
            .with_label_values(&[TIER_LABELS[tier_idx]])
            .inc();
        PAX_TIER_GET_BYTES
            .with_label_values(&[TIER_LABELS[tier_idx]])
            .inc_by(bytes);
    }
}

/// Record one object-storage metadata op (HEAD) on the search path (C3).
pub fn record_metadata_op() {
    PAX_METADATA_OPS.inc();
}

/// Record one coarse-probe rank wall duration in microseconds (C2).
pub fn record_ivf_probe_rank_us(us: u64) {
    IVF_PROBE_RANK_US.inc_by(us);
    IVF_PROBE_RANK_CALLS.inc();
}

/// Record the Region-B survivor fetch+rerank wall in microseconds (PR-B).
pub fn record_region_b_rerank_us(us: u64) {
    REGION_B_RERANK_US.inc_by(us);
}

/// Record the Region-D top-k OID fetch+decode wall in microseconds (PR-B).
pub fn record_region_d_fetch_decode_us(us: u64) {
    REGION_D_FETCH_DECODE_US.inc_by(us);
}

/// Record the top-k rehydration wall in microseconds (PR-B).
pub fn record_rehydrate_us(us: u64) {
    REHYDRATE_US.inc_by(us);
}

/// Record the whole-cascade umbrella wall in microseconds (PR-B).
pub fn record_cascade_total_us(us: u64) {
    CASCADE_TOTAL_US.inc_by(us);
}
