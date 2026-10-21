use crate::metrics::cache::TierMetrics;
use crate::storage::cache::backend::CacheTier;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tracing::debug;
use tracing::error;

// --- Prometheus exposure (TD-cache-metrics / Phase-1 observability gap) -----
// StorageCacheMetrics keeps in-process atomics for the health monitor / sizing
// optimizer. These statics mirror the `consumption_metrics` pattern so the same
// counters/gauges ALSO land in the default Prometheus registry and are scraped
// at `/metrics/prometheus` (and shipped via OTLP). Registration is infallible:
// the fallback returns a fresh (unregistered) metric so a duplicate-registration
// error never panics at startup — matches the single sanctioned fallback site
// in `crate::metrics::consumption_metrics`.
use lazy_static::lazy_static;
use prometheus::{
    IntCounter, IntCounterVec, IntGauge, Opts, register_int_counter, register_int_counter_vec,
    register_int_gauge,
};

// Static metric registration is infallible / fail-fast at startup (mirrors
// `crate::metrics::consumption_metrics`); the fallback `.unwrap()` constructs a
// fresh metric from always-valid Opts, so it cannot panic.
#[allow(clippy::unwrap_used)]
fn registered_int_counter_vec(
    name: &'static str,
    help: &'static str,
    labels: &[&str],
) -> IntCounterVec {
    register_int_counter_vec!(Opts::new(name, help), labels).unwrap_or_else(|err| {
        error!("failed to register {}: {}", name, err);
        IntCounterVec::new(Opts::new(name, ""), labels).unwrap()
    })
}

#[allow(clippy::unwrap_used)]
fn registered_int_counter(name: &'static str, help: &'static str) -> IntCounter {
    register_int_counter!(Opts::new(name, help)).unwrap_or_else(|err| {
        error!("failed to register {}: {}", name, err);
        IntCounter::with_opts(Opts::new(name, "")).unwrap()
    })
}

#[allow(clippy::unwrap_used)]
fn registered_int_gauge(name: &'static str, help: &'static str) -> IntGauge {
    register_int_gauge!(Opts::new(name, help)).unwrap_or_else(|err| {
        error!("failed to register {}: {}", name, err);
        IntGauge::with_opts(Opts::new(name, "")).unwrap()
    })
}

lazy_static! {
    /// Cache hits by tier (L1 = hot in-mem, L2 = warm, L3 = cold/disk). The
    /// `hits / (hits + misses)` ratio is the operator's cache-effectiveness
    /// signal — the thing the SIFT1M hot-vs-cold re-run (TD-cache Phase 2)
    /// needs to be measurable instead of inferred from latency alone.
    pub static ref CACHE_HITS_TOTAL: IntCounterVec = registered_int_counter_vec(
        "proximadb_cache_hits_total",
        "Cache hits by tier (l1/l2/l3)",
        &["tier"],
    );
    pub static ref CACHE_MISSES_TOTAL: IntCounter = registered_int_counter(
        "proximadb_cache_misses_total",
        "Cache misses (no tier hit) — drives the hit-ratio denominator",
    );
    pub static ref CACHE_EVICTIONS_TOTAL: IntCounter = registered_int_counter(
        "proximadb_cache_evictions_total",
        "Cache evictions (capacity pressure / churn)",
    );
    pub static ref CACHE_ENTRIES: IntGauge = registered_int_gauge(
        "proximadb_cache_entries",
        "Current cache entry count",
    );
    pub static ref CACHE_BYTES: IntGauge = registered_int_gauge(
        "proximadb_cache_bytes",
        "Current cache size in bytes",
    );
}

fn cache_tier_str(tier: CacheTier) -> &'static str {
    match tier {
        CacheTier::L1 => "l1",
        CacheTier::L2 => "l2",
        CacheTier::L3 => "l3",
    }
}

/// Backwards-compat alias for [`StorageCacheMetrics`].
pub type CacheMetrics = StorageCacheMetrics;

/// Cache metrics collector
#[derive(Debug, Clone)]
pub struct StorageCacheMetrics {
    // Hit/Miss counters per tier
    l1_hits: Arc<AtomicU64>,
    l2_hits: Arc<AtomicU64>,
    l3_hits: Arc<AtomicU64>,
    misses: Arc<AtomicU64>,

    // Operation counters
    gets: Arc<AtomicU64>,
    puts: Arc<AtomicU64>,
    invalidations: Arc<AtomicU64>,
    evictions: Arc<AtomicU64>,
    _promotions: Arc<AtomicU64>,
    invalidation_cascades: Arc<AtomicU64>,

    // Size metrics
    total_entries: Arc<AtomicUsize>,
    total_bytes: Arc<AtomicUsize>,

    // Latency tracking (in microseconds)
    get_latency_sum: Arc<AtomicU64>,
    get_latency_count: Arc<AtomicU64>,
    put_latency_sum: Arc<AtomicU64>,
    put_latency_count: Arc<AtomicU64>,

    // Start time for rate calculations
    start_time: Instant,
}

impl StorageCacheMetrics {
    pub fn new() -> Self {
        Self {
            l1_hits: Arc::new(AtomicU64::new(0)),
            l2_hits: Arc::new(AtomicU64::new(0)),
            l3_hits: Arc::new(AtomicU64::new(0)),
            misses: Arc::new(AtomicU64::new(0)),
            gets: Arc::new(AtomicU64::new(0)),
            puts: Arc::new(AtomicU64::new(0)),
            invalidations: Arc::new(AtomicU64::new(0)),
            evictions: Arc::new(AtomicU64::new(0)),
            _promotions: Arc::new(AtomicU64::new(0)),
            invalidation_cascades: Arc::new(AtomicU64::new(0)),
            total_entries: Arc::new(AtomicUsize::new(0)),
            total_bytes: Arc::new(AtomicUsize::new(0)),
            get_latency_sum: Arc::new(AtomicU64::new(0)),
            get_latency_count: Arc::new(AtomicU64::new(0)),
            put_latency_sum: Arc::new(AtomicU64::new(0)),
            put_latency_count: Arc::new(AtomicU64::new(0)),
            start_time: Instant::now(),
        }
    }

    pub fn record_hit(&self, tier: CacheTier) {
        match tier {
            CacheTier::L1 => self.l1_hits.fetch_add(1, Ordering::Relaxed),
            CacheTier::L2 => self.l2_hits.fetch_add(1, Ordering::Relaxed),
            CacheTier::L3 => self.l3_hits.fetch_add(1, Ordering::Relaxed),
        };
        self.gets.fetch_add(1, Ordering::Relaxed);
        CACHE_HITS_TOTAL
            .with_label_values(&[cache_tier_str(tier)])
            .inc();
    }

    pub fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
        self.gets.fetch_add(1, Ordering::Relaxed);
        CACHE_MISSES_TOTAL.inc();
    }

    pub fn record_put(&self) {
        self.puts.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_invalidation(&self) {
        self.invalidations.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_eviction(&self) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
        CACHE_EVICTIONS_TOTAL.inc();
    }

    pub fn record_get_latency(&self, duration: Duration) {
        let micros = duration.as_micros() as u64;
        self.get_latency_sum.fetch_add(micros, Ordering::Relaxed);
        self.get_latency_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_put_latency(&self, duration: Duration) {
        let micros = duration.as_micros() as u64;
        self.put_latency_sum.fetch_add(micros, Ordering::Relaxed);
        self.put_latency_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn update_size(&self, entries: usize, bytes: usize) {
        self.total_entries.store(entries, Ordering::Relaxed);
        self.total_bytes.store(bytes, Ordering::Relaxed);
        CACHE_ENTRIES.set(entries as i64);
        CACHE_BYTES.set(bytes as i64);
    }

    pub fn size_bytes(&self) -> usize {
        self.total_bytes.load(Ordering::Relaxed)
    }

    pub fn snapshot(&self) -> StorageCacheMetricsSnapshot {
        let total_hits = self.l1_hits.load(Ordering::Relaxed)
            + self.l2_hits.load(Ordering::Relaxed)
            + self.l3_hits.load(Ordering::Relaxed);
        let total_gets = self.gets.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);

        let hit_rate = if total_gets > 0 {
            (total_hits as f64) / (total_gets as f64)
        } else {
            0.0
        };

        let get_latency_count = self.get_latency_count.load(Ordering::Relaxed);
        let avg_get_latency_us = self
            .get_latency_sum
            .load(Ordering::Relaxed)
            .checked_div(get_latency_count)
            .unwrap_or(0);

        let put_latency_count = self.put_latency_count.load(Ordering::Relaxed);
        let avg_put_latency_us = self
            .put_latency_sum
            .load(Ordering::Relaxed)
            .checked_div(put_latency_count)
            .unwrap_or(0);

        let elapsed_secs = self.start_time.elapsed().as_secs_f64();
        let gets_per_sec = if elapsed_secs > 0.0 {
            (total_gets as f64) / elapsed_secs
        } else {
            0.0
        };

        StorageCacheMetricsSnapshot {
            l1_hits: self.l1_hits.load(Ordering::Relaxed),
            l2_hits: self.l2_hits.load(Ordering::Relaxed),
            l3_hits: self.l3_hits.load(Ordering::Relaxed),
            misses,
            total_gets,
            total_puts: self.puts.load(Ordering::Relaxed),
            invalidations: self.invalidations.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            hit_rate,
            total_entries: self.total_entries.load(Ordering::Relaxed),
            total_bytes: self.total_bytes.load(Ordering::Relaxed),
            avg_get_latency_us,
            avg_put_latency_us,
            gets_per_sec,
            uptime_secs: elapsed_secs as u64,
        }
    }

    pub fn reset(&self) {
        self.l1_hits.store(0, Ordering::Relaxed);
        self.l2_hits.store(0, Ordering::Relaxed);
        self.l3_hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.gets.store(0, Ordering::Relaxed);
        self.puts.store(0, Ordering::Relaxed);
        self.invalidations.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
        self.get_latency_sum.store(0, Ordering::Relaxed);
        self.get_latency_count.store(0, Ordering::Relaxed);
        self.put_latency_sum.store(0, Ordering::Relaxed);
        self.put_latency_count.store(0, Ordering::Relaxed);
    }

    // Methods needed by the integrated cache metrics
    pub fn tier_hits(&self, tier: CacheTier) -> u64 {
        match tier {
            CacheTier::L1 => self.l1_hits.load(Ordering::Relaxed),
            CacheTier::L2 => self.l2_hits.load(Ordering::Relaxed),
            CacheTier::L3 => self.l3_hits.load(Ordering::Relaxed),
        }
    }

    pub fn tier_misses(&self, tier: CacheTier) -> u64 {
        // Calculate misses based on tier and total misses
        let total_misses = self.misses.load(Ordering::Relaxed);
        match tier {
            CacheTier::L1 => total_misses / 3, // Rough estimate
            CacheTier::L2 => total_misses / 3,
            CacheTier::L3 => total_misses / 3,
        }
    }

    pub fn tier_hit_rate(&self, tier: CacheTier) -> f64 {
        let hits = self.tier_hits(tier);
        let misses = self.tier_misses(tier);
        let total = hits + misses;
        if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        }
    }

    pub fn hit_rate(&self) -> f64 {
        let total_hits = self.l1_hits.load(Ordering::Relaxed)
            + self.l2_hits.load(Ordering::Relaxed)
            + self.l3_hits.load(Ordering::Relaxed);
        let total_gets = self.gets.load(Ordering::Relaxed);

        if total_gets > 0 {
            (total_hits as f64) / (total_gets as f64)
        } else {
            0.0
        }
    }

    pub fn avg_latency_ms(&self, _tier: CacheTier) -> f64 {
        // Convert microseconds to milliseconds
        let count = self.get_latency_count.load(Ordering::Relaxed);
        if count > 0 {
            let sum_us = self.get_latency_sum.load(Ordering::Relaxed);
            (sum_us as f64 / count as f64) / 1000.0
        } else {
            0.0
        }
    }

    pub fn p99_latency_ms(&self, _tier: CacheTier) -> f64 {
        // For now, return a simple estimate (2x average)
        self.avg_latency_ms(_tier) * 2.0
    }

    pub fn tier_entries(&self, _tier: CacheTier) -> usize {
        // Return total entries divided by 3 as an estimate per tier
        self.total_entries.load(Ordering::Relaxed) / 3
    }

    pub fn tier_size_bytes(&self, _tier: CacheTier) -> usize {
        // Return total bytes divided by 3 as an estimate per tier
        self.total_bytes.load(Ordering::Relaxed) / 3
    }

    /// Build the framework-facing [`TierMetrics`] snapshot for one tier.
    ///
    /// Single source of the per-tier field mapping used by the cache health
    /// monitor and the performance optimizer when assembling
    /// `CacheMetricsSnapshot` (dedup: was 12 hand-rolled literals).
    pub(crate) fn tier_metrics(&self, tier: CacheTier) -> TierMetrics {
        TierMetrics {
            hits: self.tier_hits(tier),
            misses: self.tier_misses(tier),
            hit_rate: self.tier_hit_rate(tier),
            avg_latency_ms: self.avg_latency_ms(tier),
            p99_latency_ms: self.p99_latency_ms(tier),
            entries: self.tier_entries(tier),
            size_bytes: self.tier_size_bytes(tier),
        }
    }

    pub fn total_allocated_bytes(&self) -> usize {
        self.total_bytes.load(Ordering::Relaxed)
    }

    pub fn used_bytes(&self) -> usize {
        self.total_bytes.load(Ordering::Relaxed)
    }

    pub fn fragmentation_ratio(&self) -> f64 {
        // Simple estimate: assume 10% fragmentation
        0.1
    }

    pub fn evictions_by_strategy(&self, _strategy: &str) -> u64 {
        // For now, return total evictions divided by number of strategies
        self.evictions.load(Ordering::Relaxed) / 3
    }

    pub fn ttl_evictions(&self) -> u64 {
        // Estimate as 10% of total evictions
        self.evictions.load(Ordering::Relaxed) / 10
    }

    pub fn cross_cache_hits(&self) -> u64 {
        // Use L2 hits as a proxy for cross-cache hits
        self.l2_hits.load(Ordering::Relaxed)
    }

    pub fn prefetch_success_rate(&self) -> f64 {
        // Estimate based on L1 hit rate
        let l1_hits = self.l1_hits.load(Ordering::Relaxed);
        let total_gets = self.gets.load(Ordering::Relaxed);
        if total_gets > 0 {
            (l1_hits as f64) / (total_gets as f64)
        } else {
            0.0
        }
    }

    pub fn invalidation_cascades(&self) -> u64 {
        self.invalidation_cascades.load(Ordering::Relaxed)
    }

    pub fn memory_rebalances(&self) -> u64 {
        // Not tracked yet
        0
    }

    pub fn allocation_failures(&self) -> u64 {
        // Not tracked yet - would track when we can't allocate memory
        0
    }

    pub fn total_evictions(&self) -> u64 {
        self.evictions.load(Ordering::Relaxed)
    }

    pub fn lru_evictions(&self) -> u64 {
        // Estimate as 60% of total evictions
        (self.evictions.load(Ordering::Relaxed) as f64 * 0.6) as u64
    }

    pub fn lfu_evictions(&self) -> u64 {
        // Estimate as 20% of total evictions
        (self.evictions.load(Ordering::Relaxed) as f64 * 0.2) as u64
    }

    pub fn arc_evictions(&self) -> u64 {
        // Estimate as 10% of total evictions
        (self.evictions.load(Ordering::Relaxed) as f64 * 0.1) as u64
    }

    pub fn memory_pressure_evictions(&self) -> u64 {
        // Estimate as 10% of total evictions
        (self.evictions.load(Ordering::Relaxed) as f64 * 0.1) as u64
    }

    pub fn record_invalidation_cascade(&self) {
        self.invalidation_cascades.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_memory_rebalance(&self) {
        // Not tracked in detail yet, but we can track it as an event
        // For now, this is a no-op
    }

    pub fn total_gets(&self) -> u64 {
        self.gets.load(Ordering::Relaxed)
    }

    pub fn total_puts(&self) -> u64 {
        self.puts.load(Ordering::Relaxed)
    }

    pub fn total_entries(&self) -> usize {
        self.total_entries.load(Ordering::Relaxed)
    }
}

impl Default for StorageCacheMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of cache metrics at a point in time
#[derive(Debug, Clone)]
pub struct StorageCacheMetricsSnapshot {
    pub l1_hits: u64,
    pub l2_hits: u64,
    pub l3_hits: u64,
    pub misses: u64,
    pub total_gets: u64,
    pub total_puts: u64,
    pub invalidations: u64,
    pub evictions: u64,
    pub hit_rate: f64,
    pub total_entries: usize,
    pub total_bytes: usize,
    pub avg_get_latency_us: u64,
    pub avg_put_latency_us: u64,
    pub gets_per_sec: f64,
    pub uptime_secs: u64,
}

impl StorageCacheMetricsSnapshot {
    pub fn print_summary(&self) {
        debug!("=== Cache Metrics Summary ===");
        debug!("Hit Rate: {:.2}%", self.hit_rate * 100.0);
        debug!(
            "L1 Hits: {} ({:.1}%)",
            self.l1_hits,
            (self.l1_hits as f64 / self.total_gets.max(1) as f64) * 100.0
        );
        debug!(
            "L2 Hits: {} ({:.1}%)",
            self.l2_hits,
            (self.l2_hits as f64 / self.total_gets.max(1) as f64) * 100.0
        );
        debug!(
            "L3 Hits: {} ({:.1}%)",
            self.l3_hits,
            (self.l3_hits as f64 / self.total_gets.max(1) as f64) * 100.0
        );
        debug!(
            "Misses: {} ({:.1}%)",
            self.misses,
            (self.misses as f64 / self.total_gets.max(1) as f64) * 100.0
        );
        debug!("Total Gets: {}", self.total_gets);
        debug!("Total Puts: {}", self.total_puts);
        debug!("Evictions: {}", self.evictions);
        debug!("Invalidations: {}", self.invalidations);
        debug!("Entries: {}", self.total_entries);
        debug!("Size: {} MB", self.total_bytes / (1024 * 1024));
        debug!("Avg Get Latency: {} μs", self.avg_get_latency_us);
        debug!("Avg Put Latency: {} μs", self.avg_put_latency_us);
        debug!("Gets/sec: {:.1}", self.gets_per_sec);
        debug!("Uptime: {} seconds", self.uptime_secs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prometheus::Encoder;

    /// The cache counters/gauges must land in the default Prometheus registry
    /// (scraped at `/metrics/prometheus` + shipped via OTLP) — the Phase-1
    /// observability gap. This exercises the record_* path and asserts the
    /// metric families are gathered (no server boot required).
    #[test]
    fn cache_metrics_exposed_in_prometheus_registry() {
        let m = StorageCacheMetrics::new();
        m.record_hit(CacheTier::L1);
        m.record_hit(CacheTier::L2);
        m.record_hit(CacheTier::L3);
        m.record_miss();
        m.record_miss();
        m.record_eviction();
        m.update_size(42, 4096);

        let families = prometheus::gather();
        let mut buf = Vec::new();
        let encoder = prometheus::TextEncoder::new();
        encoder.encode(&families, &mut buf).expect("encode");
        let out = String::from_utf8(buf).expect("utf8");

        // The five cache metrics must be exposed ...
        assert!(
            out.contains("proximadb_cache_hits_total"),
            "hits_total missing"
        );
        assert!(
            out.contains("proximadb_cache_misses_total"),
            "misses_total missing"
        );
        assert!(
            out.contains("proximadb_cache_evictions_total"),
            "evictions_total missing"
        );
        assert!(out.contains("proximadb_cache_entries"), "entries missing");
        assert!(out.contains("proximadb_cache_bytes"), "bytes missing");
        // ... with the per-tier labels on hits.
        assert!(out.contains("tier=\"l1\""), "tier=l1 label missing");
        assert!(out.contains("tier=\"l2\""), "tier=l2 label missing");
        assert!(out.contains("tier=\"l3\""), "tier=l3 label missing");
    }
}
