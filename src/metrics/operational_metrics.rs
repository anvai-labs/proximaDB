// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! TD-METRICS-1: real operational counters, registered in the process-default
//! prometheus registry (scraped by `/metrics/prometheus` via
//! `prometheus::gather()`).
//!
//! These replace the never-incremented struct-exporter stanzas
//! (`QueryMetrics`/`StorageMetrics`) that used to render
//! `proximadb_queries_total 0` — the names are reclaimed here as live
//! counters (the stale hand-written stanzas were removed from
//! `export_system_metrics` in the same change, so there is no name
//! collision on the scrape output).
//!
//! Covered signals:
//! - **query rate/outcome**: `proximadb_queries_total` /
//!   `proximadb_queries_failed_total` + `proximadb_search_latency_seconds`,
//!   incremented at the query-facade search entry.
//! - **cache effectiveness** (the co-design hot-path signal — DRAM vs object
//!   store): survivor-range cache hits/misses/bytes (mirrored from
//!   `TenantCache`'s internal atomics) and segment-invariants cache
//!   hits/misses/bytes (counted at the lookup boundary).

use lazy_static::lazy_static;
use prometheus::{
    Histogram, IntCounter, IntCounterVec, IntGauge, register_histogram, register_int_counter,
    register_int_counter_vec, register_int_gauge,
};

fn counter(name: &str, help: &str) -> IntCounter {
    register_int_counter!(name, help).unwrap_or_else(|_| {
        IntCounter::new(format!("{name}_fallback"), help)
            .unwrap_or_else(|_| unreachable!("valid counter metric descriptor"))
    })
}

lazy_static::lazy_static! {
    /// TD-COMPACT-13: compaction input deletes that failed and are pending
    /// reconciliation. >0 means superseded segments are still serving (cost
    /// defect, results correct) — observable rot, never silent.
    pub static ref PAX_PENDING_RETIREMENTS: IntGauge = gauge(
        "proximadb_pax_pending_retirements",
        "Compaction input deletes pending reconciliation (TD-COMPACT-13)",
    );
    pub static ref PAX_RETIREMENT_RETRIES_TOTAL: IntCounter = counter(
        "proximadb_pax_retirement_retries_total",
        "Retried compaction input deletes (TD-COMPACT-13)",
    );
    pub static ref PAX_RETIREMENT_RECORD_TOTAL: IntCounter = counter(
        "proximadb_pax_retirement_record_total",
        "Retirement obligations recorded to the pending ledger (TD-COMPACT-13)",
    );
}

fn gauge(name: &str, help: &str) -> IntGauge {
    register_int_gauge!(name, help).unwrap_or_else(|_| {
        IntGauge::new(format!("{name}_fallback"), help)
            .unwrap_or_else(|_| unreachable!("valid gauge metric descriptor"))
    })
}

fn gauge_vec(name: &str, help: &str, labels: &[&str]) -> prometheus::IntGaugeVec {
    prometheus::register_int_gauge_vec!(name, help, labels).unwrap_or_else(|_| {
        prometheus::IntGaugeVec::new(
            prometheus::Opts::new(format!("{name}_fallback"), help),
            labels,
        )
        .unwrap_or_else(|_| unreachable!("valid gauge-vec metric descriptor"))
    })
}

fn counter_vec(name: &str, help: &str, labels: &[&str]) -> Option<IntCounterVec> {
    register_int_counter_vec!(name, help, labels).ok()
}

fn histogram(name: &str, help: &str, buckets: Vec<f64>) -> Histogram {
    register_histogram!(name, help, buckets.clone()).unwrap_or_else(|_| {
        Histogram::with_opts(
            prometheus::HistogramOpts::new(format!("{name}_fallback"), help).buckets(buckets),
        )
        .unwrap_or_else(|_| unreachable!("valid histogram metric descriptor"))
    })
}

lazy_static! {
    /// Vector/record searches served through the query facade (all surfaces).
    pub static ref QUERIES_TOTAL: IntCounter = counter(
        "proximadb_queries_total",
        "Total search queries processed through the query facade",
    );
    pub static ref QUERIES_FAILED_TOTAL: IntCounter = counter(
        "proximadb_queries_failed_total",
        "Search queries that returned an error from the query facade",
    );
    /// TD-PAXDIR-1 D7 (P5b design §4.1): sequential object-store request
    /// waves along each vector query's critical path. Integer buckets, so a
    /// p50 read from the histogram is exact up to 10 waves (the P5b gate is
    /// ≤ 3 cold, ≤ 1 warm).
    pub static ref PAX_QUERY_WAVES: Histogram = histogram(
        "proximadb_pax_query_waves",
        "Sequential object-store request waves per vector query (critical path)",
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 16.0, 20.0, 30.0, 50.0],
    );
    /// Buckets span DRAM-hot (sub-ms) through cold object-store scans (10s).
    pub static ref SEARCH_LATENCY_SECONDS: Histogram = histogram(
        "proximadb_search_latency_seconds",
        "End-to-end search latency at the query facade",
        vec![0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    );

    /// Survivor-range cache (SST engine RAM cache over survivor-block ranges).
    /// Absolute values mirrored from `TenantCache`'s internal atomics after
    /// each lookup — gauges, not counters, because the source is a running
    /// total we sample rather than an event we observe.
    pub static ref SURVIVOR_CACHE_HITS: IntGauge = gauge(
        "proximadb_survivor_cache_hits",
        "Cumulative survivor-range cache hits (0 GETs served from DRAM)",
    );
    pub static ref SURVIVOR_CACHE_MISSES: IntGauge = gauge(
        "proximadb_survivor_cache_misses",
        "Cumulative survivor-range cache misses (paid object-store GETs)",
    );
    pub static ref SURVIVOR_CACHE_BYTES: IntGauge = gauge(
        "proximadb_survivor_cache_bytes",
        "Resident bytes in the survivor-range cache",
    );

    /// TD-CACHE-3 S1: per-tenant survivor-cache series (bounded — only the
    /// top tenants by resident bytes are emitted per sync; see
    /// `SURVIVOR_TENANT_SERIES_CAP`). The noisy-neighbor / hot-tier
    /// entitlement signal.
    pub static ref SURVIVOR_CACHE_TENANT_BYTES: prometheus::IntGaugeVec = gauge_vec(
        "proximadb_survivor_cache_tenant_bytes",
        "Resident survivor-cache bytes per tenant (top tenants only)",
        &["tenant_id"],
    );
    pub static ref SURVIVOR_CACHE_TENANT_HITS: prometheus::IntGaugeVec = gauge_vec(
        "proximadb_survivor_cache_tenant_hits",
        "Cumulative survivor-cache hits per tenant (top tenants only)",
        &["tenant_id"],
    );
    /// TD-CACHE-3 S3: bytes held in the tenant's PINNED floor segment (true-pin).
    pub static ref SURVIVOR_CACHE_TENANT_PINNED_BYTES: prometheus::IntGaugeVec = gauge_vec(
        "proximadb_survivor_cache_tenant_pinned_bytes",
        "Survivor-cache bytes pinned in the tenant's floor segment (never evictable by other tenants)",
        &["tenant_id"]
    );
    /// TD-CACHE-3 S3: the tenant's ENTITLED floor bytes (tier contract). Billing
    /// true-up = pinned vs entitled; a sustained gap under pressure means
    /// residency cannot be honored (capacity signal).
    pub static ref SURVIVOR_CACHE_TENANT_ENTITLED_BYTES: prometheus::IntGaugeVec = gauge_vec(
        "proximadb_survivor_cache_tenant_entitled_bytes",
        "Survivor-cache floor bytes the tenant's tier entitles it to (0 = no floor)",
        &["tenant_id"]
    );
    pub static ref SURVIVOR_CACHE_TENANT_MISSES: prometheus::IntGaugeVec = gauge_vec(
        "proximadb_survivor_cache_tenant_misses",
        "Cumulative survivor-cache misses per tenant (top tenants only)",
        &["tenant_id"],
    );

    /// Segment-invariants cache (footer/region-A metadata; a hit skips 3
    /// ranged GETs per segment per query).
    pub static ref SEGMENT_INVARIANTS_CACHE_HITS_TOTAL: IntCounter = counter(
        "proximadb_segment_invariants_cache_hits_total",
        "Segment-invariants cache hits (footer + A0 metadata served from DRAM)",
    );
    pub static ref SEGMENT_INVARIANTS_CACHE_MISSES_TOTAL: IntCounter = counter(
        "proximadb_segment_invariants_cache_misses_total",
        "Segment-invariants cache misses (footer + A0 metadata fetched from storage)",
    );
    pub static ref SEGMENT_INVARIANTS_CACHE_BYTES: IntGauge = gauge(
        "proximadb_segment_invariants_cache_bytes",
        "Resident bytes in the segment-invariants cache",
    );
    pub static ref CACHE_LOCAL_DISK_HITS_TOTAL: Option<IntCounterVec> = counter_vec(
        "proximadb_cache_local_disk_hits_total",
        "Persistent local-cache hits that avoided an object-store GET",
        &["tier"],
    );
    pub static ref CACHE_LOCAL_DISK_MISSES_TOTAL: Option<IntCounterVec> = counter_vec(
        "proximadb_cache_local_disk_misses_total",
        "Persistent local-cache namespace lookup misses; another local parent lookup may still avoid object storage",
        &["tier"],
    );
    pub static ref CACHE_LOCAL_DISK_BYTES: prometheus::IntGaugeVec = gauge_vec(
        "proximadb_cache_local_disk_bytes",
        "Resident persistent local-cache bytes by PAX cache tier",
        &["tier"],
    );
    /// TD-COMPACT-1 S2: compaction I/O + wall-clock. Counters (event-sourced
    /// at compaction completion), scraped by /metrics/prometheus. A stuck-at-0
    /// bytes counter alongside advancing runs now points at the size probes
    /// (which WARN on failure) instead of silently reading 0.0 MB/s.
    pub static ref COMPACTIONS_TOTAL: IntCounter = counter(
        "proximadb_compactions_total",
        "LSM compactions completed",
    );
    pub static ref COMPACTION_BYTES_READ_TOTAL: IntCounter = counter(
        "proximadb_compaction_bytes_read_total",
        "Bytes read from input segments across completed compactions",
    );
    pub static ref COMPACTION_BYTES_WRITTEN_TOTAL: IntCounter = counter(
        "proximadb_compaction_bytes_written_total",
        "Bytes written to merged output segments across completed compactions",
    );
    /// Buckets span sub-second flush-sized merges through multi-minute
    /// re-encode compactions (the TD-COMPACT-1 regression regime).
    pub static ref COMPACTION_SECONDS: Histogram = histogram(
        "proximadb_compaction_seconds",
        "Wall-clock duration of completed LSM compactions",
        vec![0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0],
    );
    pub static ref COMPACTION_MEMORY_RESERVED_BYTES: IntGauge = gauge(
        "proximadb_compaction_memory_reserved_bytes",
        "Projected peak resident bytes reserved by all in-flight compactions",
    );
    pub static ref COMPACTION_SCRATCH_RESERVED_BYTES: IntGauge = gauge(
        "proximadb_compaction_scratch_reserved_bytes",
        "Projected local scratch bytes reserved by all in-flight compactions",
    );
    pub static ref COMPACTION_DECLINED_TOTAL: Option<IntCounterVec> = counter_vec(
        "proximadb_compaction_declined_total",
        "Compactions deliberately declined with every input left authoritative, by reason",
        &["reason"],
    );
    pub static ref COMPACTION_SPILL_TOTAL: IntCounter = counter(
        "proximadb_compaction_spill_total",
        "Compactions admitted with the deterministic local-spill execution plan",
    );
    pub static ref COMPACTION_MEMORY_ADMISSION_WAITS_TOTAL: IntCounter = counter(
        "proximadb_compaction_memory_admission_waits_total",
        "Compaction tasks that waited for process-wide weighted memory admission",
    );
    pub static ref COMPACTION_QUEUE_DEPTH: IntGauge = gauge(
        "proximadb_compaction_queue_depth",
        "Compaction tasks queued across all collections",
    );
    pub static ref COMPACTION_RUNNING: IntGauge = gauge(
        "proximadb_compaction_running",
        "Compaction tasks currently executing across all collections",
    );
}

/// Mirror monotonic cache-internal local-disk counters into Prometheus counters and
/// publish the current resident-byte gauge.
pub fn sync_local_disk_stats(tier: &str, stats: proximadb_cache::L2CacheStats) {
    if let Some(family) = CACHE_LOCAL_DISK_HITS_TOTAL.as_ref() {
        let hits = family.with_label_values(&[tier]);
        let recorded_hits = hits.get();
        if stats.hits > recorded_hits {
            hits.inc_by(stats.hits - recorded_hits);
        }
    }
    if let Some(family) = CACHE_LOCAL_DISK_MISSES_TOTAL.as_ref() {
        let misses = family.with_label_values(&[tier]);
        let recorded_misses = misses.get();
        if stats.misses > recorded_misses {
            misses.inc_by(stats.misses - recorded_misses);
        }
    }
    CACHE_LOCAL_DISK_BYTES
        .with_label_values(&[tier])
        .set(stats.resident_bytes.min(i64::MAX as u64) as i64);
}

/// Force-initialize every lazy metric family at boot so they are REGISTERED
/// (and scrape as 0) before the first event. Without this, a family is
/// invisible to /metrics/prometheus until its first increment — absent-vs-zero
/// ambiguity that cost a diagnostic cycle during the TD-COMPACT-1 ratchet run.
pub fn touch() {
    QUERIES_TOTAL.get();
    QUERIES_FAILED_TOTAL.get();
    let _ = SEARCH_LATENCY_SECONDS.get_sample_count();
    SURVIVOR_CACHE_HITS.get();
    SURVIVOR_CACHE_MISSES.get();
    SURVIVOR_CACHE_BYTES.get();
    SEGMENT_INVARIANTS_CACHE_HITS_TOTAL.get();
    SEGMENT_INVARIANTS_CACHE_MISSES_TOTAL.get();
    SEGMENT_INVARIANTS_CACHE_BYTES.get();
    if let Some(family) = CACHE_LOCAL_DISK_HITS_TOTAL.as_ref() {
        family.with_label_values(&["survivor"]).get();
        family.with_label_values(&["invariants"]).get();
    }
    if let Some(family) = CACHE_LOCAL_DISK_MISSES_TOTAL.as_ref() {
        family.with_label_values(&["survivor"]).get();
        family.with_label_values(&["invariants"]).get();
    }
    CACHE_LOCAL_DISK_BYTES
        .with_label_values(&["survivor"])
        .get();
    CACHE_LOCAL_DISK_BYTES
        .with_label_values(&["invariants"])
        .get();
    COMPACTIONS_TOTAL.get();
    COMPACTION_BYTES_READ_TOTAL.get();
    COMPACTION_BYTES_WRITTEN_TOTAL.get();
    COMPACTION_MEMORY_RESERVED_BYTES.get();
    COMPACTION_SCRATCH_RESERVED_BYTES.get();
    COMPACTION_SPILL_TOTAL.get();
    if let Some(family) = COMPACTION_DECLINED_TOTAL.as_ref() {
        family
            .with_label_values(&["spill_base_quant_unsupported"])
            .get();
        family.with_label_values(&["spill_inputs_not_exact"]).get();
    }
    COMPACTION_MEMORY_ADMISSION_WAITS_TOTAL.get();
    COMPACTION_QUEUE_DEPTH.get();
    COMPACTION_RUNNING.get();
    let _ = COMPACTION_SECONDS.get_sample_count();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operational_metrics_register_and_move() {
        let q0 = QUERIES_TOTAL.get();
        QUERIES_TOTAL.inc();
        assert_eq!(QUERIES_TOTAL.get(), q0 + 1);
        SEARCH_LATENCY_SECONDS.observe(0.02);
        SURVIVOR_CACHE_HITS.set(7);
        assert_eq!(SURVIVOR_CACHE_HITS.get(), 7);
        SEGMENT_INVARIANTS_CACHE_HITS_TOTAL.inc();
        assert!(SEGMENT_INVARIANTS_CACHE_HITS_TOTAL.get() >= 1);
        assert!(
            CACHE_LOCAL_DISK_HITS_TOTAL.is_some() && CACHE_LOCAL_DISK_MISSES_TOTAL.is_some(),
            "the exact persistent-cache metric families must register"
        );
        CACHE_LOCAL_DISK_BYTES
            .with_label_values(&["survivor"])
            .set(1024);
        assert_eq!(
            CACHE_LOCAL_DISK_BYTES
                .with_label_values(&["survivor"])
                .get(),
            1024
        );
    }
}
