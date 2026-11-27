/*
 * Copyright 2025 Vijaykumar Singh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

//! Vector Object Economy Benchmark
//!
//! Measures the per-component cost of the Phase 4/5 strong-route building
//! blocks at several scales, then prints a comparison table against
//! turbopuffer's published architecture numbers (3-4 GETs / 400-500ms cold
//! for 1M docs, ~14ms warm p50).
//!
//! What this benchmark measures TODAY (composable, no full e2e harness):
//!
//! * **Merge throughput** — `merge_delta_with_directory_results` at 100 /
//!   1K / 10K / 100K record sets. Tells us how much budget the merge step
//!   consumes in the strong-route critical path.
//! * **Directory serialize/deserialize round-trip** — at 100 / 1K / 10K /
//!   100K block counts. Tells us the cost of the cold-query "load
//!   directory" step.
//! * **Tombstone-suppression overhead** — same workload as merge, but with
//!   50% of delta records as tombstones. Validates the suppression branch
//!   doesn't dominate the merge cost.
//!
//! What this benchmark explicitly DOES NOT measure (full e2e harness
//! deferred):
//!
//! * Object-store GET count per query (needs real S3 / mocked range
//!   tracker)
//! * End-to-end cold/warm query p50/p99 (needs full VectorOperationsService
//!   + temp filesystem + 1M synthetic vectors — multi-minute setup)
//! * Recall @ k (needs ground-truth dataset like SIFT/GIST)
//!
//! Those are the numbers that directly compare to turbopuffer's claims;
//! the building-block costs measured here are what determines whether
//! ProximaDB has the headroom to hit them.
//!
//! Usage:
//! ```bash
//! cargo run --release --bin bench_vector_object_economy
//! ```

use proximadb::core::search::VectorFreshnessMode;
use proximadb::core::search::merge::merge_delta_with_directory_results;
use proximadb::core::search::results::OptimizedSearchRecord;
use proximadb::storage::engines::sst::object_economy_directory::{
    CentroidEncoding, ObjectEconomyBlockEntry, ObjectEconomyFileEntry, VectorObjectEconomyDirectory,
};
use proximadb_catalog::CatalogAuthorityMode;
use std::collections::BTreeMap;
use std::time::Instant;

fn main() {
    println!("================================================================================");
    println!("   ProximaDB — Vector Object Economy Benchmark (Phase 4/5 building blocks)");
    println!("================================================================================");
    println!();

    let merge_results = bench_merge_throughput();
    let directory_results = bench_directory_serde_throughput();
    let tombstone_results = bench_tombstone_suppression_overhead();
    let dispatch_results = bench_decision_dispatch_cost();

    print_comparison_table(
        &merge_results,
        &directory_results,
        &tombstone_results,
        &dispatch_results,
    );

    println!();
    println!("Deferred (full e2e harness required for direct turbopuffer comparison):");
    println!("  • Object-store GET count per query");
    println!("  • End-to-end cold / warm p50/p99 latency at 1M vectors");
    println!("  • Recall @ k against SIFT/GIST ground truth");
    println!();
}

// ────────────────────────────────────────────────────────────────────────
// Building-block benchmarks
// ────────────────────────────────────────────────────────────────────────

struct MergeMeasurement {
    label: &'static str,
    records_per_set: usize,
    total_records_processed: u64,
    elapsed_ns: u128,
}

impl MergeMeasurement {
    fn records_per_sec(&self) -> f64 {
        if self.elapsed_ns == 0 {
            return 0.0;
        }
        self.total_records_processed as f64 * 1_000_000_000.0 / self.elapsed_ns as f64
    }

    fn ns_per_record(&self) -> f64 {
        if self.total_records_processed == 0 {
            return 0.0;
        }
        self.elapsed_ns as f64 / self.total_records_processed as f64
    }
}

fn bench_merge_throughput() -> Vec<MergeMeasurement> {
    println!("📊 Bench: merge_delta_with_directory_results (single-thread)");
    println!("   (delta + directory equal-sized sets; 25% overlap; no tombstones)");
    println!();

    let scales = [
        ("100 records", 100usize, 5_000usize), // 5000 iterations
        ("1K records", 1_000, 500),
        ("10K records", 10_000, 50),
        ("100K records", 100_000, 5),
    ];

    let mut results = Vec::new();
    for (label, n, iters) in scales {
        let delta_template = synthetic_records("delta", n);
        let directory_template = synthetic_records_overlapping("a", n, n / 4);

        let start = Instant::now();
        let mut total_returned = 0u64;
        for _ in 0..iters {
            let delta = delta_template.clone();
            let directory = directory_template.clone();
            let merged = merge_delta_with_directory_results(delta, directory, n);
            total_returned += merged.len() as u64;
        }
        let elapsed_ns = start.elapsed().as_nanos();
        // sanity touch — keep the merged result from being optimised away
        std::hint::black_box(total_returned);

        let total = (n as u64) * 2 * iters as u64; // delta + directory per iter
        let m = MergeMeasurement {
            label,
            records_per_set: n,
            total_records_processed: total,
            elapsed_ns,
        };
        println!(
            "   {:<14} {:>12} ns/record   {:>14} records/sec   (over {} iter × 2×{})",
            label,
            format_f64(m.ns_per_record(), 1),
            format_with_commas(m.records_per_sec() as u64),
            iters,
            n,
        );
        results.push(m);
    }
    println!();
    results
}

fn bench_tombstone_suppression_overhead() -> MergeMeasurement {
    println!("📊 Bench: merge with 50% tombstone delta (suppression overhead)");
    let n = 10_000usize;
    let iters = 50usize;

    let delta_template = synthetic_delta_with_tombstones("a", n, 0.5);
    let directory_template = synthetic_records("a", n);

    let start = Instant::now();
    let mut total_returned = 0u64;
    for _ in 0..iters {
        let delta = delta_template.clone();
        let directory = directory_template.clone();
        let merged = merge_delta_with_directory_results(delta, directory, n);
        total_returned += merged.len() as u64;
    }
    let elapsed_ns = start.elapsed().as_nanos();
    std::hint::black_box(total_returned);

    let total = (n as u64) * 2 * iters as u64;
    let m = MergeMeasurement {
        label: "10K records, 50% tombstones",
        records_per_set: n,
        total_records_processed: total,
        elapsed_ns,
    };
    println!(
        "   {:<28} {:>12} ns/record   {:>14} records/sec   (over {} iter × 2×{})",
        m.label,
        format_f64(m.ns_per_record(), 1),
        format_with_commas(m.records_per_sec() as u64),
        iters,
        n,
    );
    println!();
    m
}

struct DirectorySerdeMeasurement {
    label: &'static str,
    block_count: usize,
    serialized_bytes: usize,
    serialize_ns: u128,
    deserialize_ns: u128,
}

fn bench_directory_serde_throughput() -> Vec<DirectorySerdeMeasurement> {
    println!("📊 Bench: VectorObjectEconomyDirectory serialize/deserialize");
    println!("   (one file per directory; 32-dim FP16 centroids; metadata zone maps)");
    println!();

    let scales = [
        ("100 blocks", 100usize, 500usize),
        ("1K blocks", 1_000, 100),
        ("10K blocks", 10_000, 10),
        // 100K omitted: serde_json round-trip cost dominates and would
        // saturate the bench's wall time; the 10K number scales linearly.
    ];

    let mut results = Vec::new();
    for (label, block_count, iters) in scales {
        let directory = synthetic_directory(block_count);

        // Serialize warm-up + measurement
        let serialized_bytes = directory.serialize().expect("serialize").len();
        let start = Instant::now();
        for _ in 0..iters {
            let bytes = directory.serialize().expect("serialize");
            std::hint::black_box(bytes);
        }
        let serialize_ns = start.elapsed().as_nanos();

        let bytes = directory.serialize().expect("serialize");
        let start = Instant::now();
        for _ in 0..iters {
            let decoded = VectorObjectEconomyDirectory::deserialize(&bytes).expect("deserialize");
            std::hint::black_box(decoded);
        }
        let deserialize_ns = start.elapsed().as_nanos();

        let m = DirectorySerdeMeasurement {
            label,
            block_count,
            serialized_bytes,
            serialize_ns: serialize_ns / iters as u128,
            deserialize_ns: deserialize_ns / iters as u128,
        };
        println!(
            "   {:<12} ser {:>8}µs   deser {:>8}µs   payload {:>10}",
            m.label,
            ns_to_us(m.serialize_ns),
            ns_to_us(m.deserialize_ns),
            format_bytes(m.serialized_bytes as u64),
        );
        results.push(m);
    }
    println!();
    results
}

struct DispatchMeasurement {
    ns_per_call: f64,
}

fn bench_decision_dispatch_cost() -> DispatchMeasurement {
    println!("📊 Bench: VectorFreshnessMode::should_scan_delta dispatch (hot inline path)");

    let modes = [
        VectorFreshnessMode::Strong,
        VectorFreshnessMode::BoundedStale {
            max_staleness_ms: 1_000,
        },
        VectorFreshnessMode::StaleOk,
    ];
    let iters = 10_000_000;

    let start = Instant::now();
    let mut count = 0u64;
    for i in 0..iters {
        let mode = &modes[(i as usize) % modes.len()];
        if mode.should_scan_delta(100, 50) {
            count += 1;
        }
    }
    let elapsed = start.elapsed().as_nanos();
    std::hint::black_box(count);

    let ns_per_call = elapsed as f64 / iters as f64;
    println!(
        "   {:>12} iters     {:>8.2} ns/call   (Strong/BoundedStale/StaleOk round-robin)",
        format_with_commas(iters as u64),
        ns_per_call
    );
    println!();
    DispatchMeasurement { ns_per_call }
}

// ────────────────────────────────────────────────────────────────────────
// Comparison table
// ────────────────────────────────────────────────────────────────────────

fn print_comparison_table(
    merge: &[MergeMeasurement],
    directory: &[DirectorySerdeMeasurement],
    tombstone: &MergeMeasurement,
    dispatch: &DispatchMeasurement,
) {
    println!();
    println!("================================================================================");
    println!("   Headroom analysis vs turbopuffer's published architecture numbers");
    println!("================================================================================");
    println!();
    println!("turbopuffer cold path (1M docs):   ~400-500 ms total, 3-4 round trips");
    println!("turbopuffer warm path (1M docs):   ~14 ms p50");
    println!();
    println!("Building-block budgets (what we just measured):");
    println!();

    if let Some(m100k) = merge.iter().find(|m| m.records_per_set == 100_000) {
        // Per-iteration cost = ns_per_record × records processed per iter
        // (delta + directory = 2 × records_per_set).
        let per_iter_us =
            (m100k.ns_per_record() * (m100k.records_per_set as f64 * 2.0) / 1_000.0) as u128;
        println!(
            "  Merge 100K records:          ~{}µs  ({}M records/sec)",
            per_iter_us,
            format_f64(m100k.records_per_sec() / 1_000_000.0, 2)
        );
    }
    if let Some(d10k) = directory.iter().find(|d| d.block_count == 10_000) {
        let dload_us = ns_to_us(d10k.deserialize_ns);
        println!(
            "  Load 10K-block directory:    ~{}µs   payload {}",
            dload_us,
            format_bytes(d10k.serialized_bytes as u64),
        );
    }
    let tombstone_per_iter_us =
        (tombstone.ns_per_record() * (tombstone.records_per_set as f64 * 2.0) / 1_000.0) as u128;
    println!(
        "  Tombstone-merge 10K:         ~{}µs  ({}M records/sec)",
        tombstone_per_iter_us,
        format_f64(tombstone.records_per_sec() / 1_000_000.0, 2)
    );
    println!(
        "  Freshness-mode dispatch:     ~{:.1} ns/call  ({}M calls/sec)",
        dispatch.ns_per_call,
        format_f64(1_000_000_000.0 / dispatch.ns_per_call / 1_000_000.0, 1)
    );
    println!();
    println!("Implication: directory load + merge cost is small enough that the cold-query");
    println!("budget (~400ms) is dominated by object-storage round-trips, not local CPU. The");
    println!("strong-route delta merge adds <1ms per 10K-record set, leaving the full");
    println!("turbopuffer-style cold budget available for remote I/O.");
    println!();
    println!("Full e2e cold/warm benchmark — required to make a direct turbopuffer claim —");
    println!("is deferred (needs 1M-vector dataset, mocked S3, recall ground truth).");
}

// ────────────────────────────────────────────────────────────────────────
// Synthetic data generators
// ────────────────────────────────────────────────────────────────────────

fn synthetic_records(prefix: &str, n: usize) -> Vec<OptimizedSearchRecord> {
    (0..n)
        .map(|i| OptimizedSearchRecord {
            id: format!("{prefix}-{i:08}"),
            vector_id: Some(format!("{prefix}-{i:08}")),
            score: 1.0 - (i as f32) / (n as f32),
            similarity: Some(1.0 - (i as f32) / (n as f32)),
            ..Default::default()
        })
        .collect()
}

/// Build a directory whose IDs overlap with a "delta" set built via
/// `synthetic_records("delta", n)` for the first `overlap` records, and
/// are distinct ("a"-prefixed) for the rest. Models the realistic case
/// where a fraction of delta IDs are updates to already-flushed records.
fn synthetic_records_overlapping(
    distinct_prefix: &str,
    n: usize,
    overlap: usize,
) -> Vec<OptimizedSearchRecord> {
    let mut records = Vec::with_capacity(n);
    for i in 0..overlap.min(n) {
        records.push(OptimizedSearchRecord {
            id: format!("delta-{i:08}"),
            vector_id: Some(format!("delta-{i:08}")),
            score: 0.5,
            similarity: Some(0.5),
            ..Default::default()
        });
    }
    for i in overlap..n {
        records.push(OptimizedSearchRecord {
            id: format!("{distinct_prefix}-{i:08}"),
            vector_id: Some(format!("{distinct_prefix}-{i:08}")),
            score: 0.5,
            similarity: Some(0.5),
            ..Default::default()
        });
    }
    records
}

fn synthetic_delta_with_tombstones(
    prefix: &str,
    n: usize,
    tombstone_ratio: f64,
) -> Vec<OptimizedSearchRecord> {
    let tombstone_cutoff = (n as f64 * tombstone_ratio) as usize;
    (0..n)
        .map(|i| {
            let id = format!("{prefix}-{i:08}");
            if i < tombstone_cutoff {
                OptimizedSearchRecord {
                    id: id.clone(),
                    vector_id: Some(id),
                    score: 0.0,
                    similarity: Some(0.0),
                    expires_at: Some(0), // tombstone marker
                    ..Default::default()
                }
            } else {
                OptimizedSearchRecord {
                    id: id.clone(),
                    vector_id: Some(id),
                    score: 0.9,
                    similarity: Some(0.9),
                    ..Default::default()
                }
            }
        })
        .collect()
}

fn synthetic_directory(block_count: usize) -> VectorObjectEconomyDirectory {
    let blocks: Vec<ObjectEconomyBlockEntry> = (0..block_count)
        .map(|i| ObjectEconomyBlockEntry {
            centroid_radius: 0.0,
            block_id: i as u32,
            offset: i as u64 * 4096,
            serialized_len: 4096,
            record_count: 64,
            centroid_fp16: Some(vec![0u16; 32]),
            centroid_fp32: None,
            zorder_code: None,
            metadata_min_values: serde_json::Map::new(),
            metadata_max_values: serde_json::Map::new(),
            metadata_null_counts: BTreeMap::new(),
        })
        .collect();

    let file_entry = ObjectEconomyFileEntry {
        file_id: "l0_0001".to_string(),
        object_url: "s3://bucket/coll/data/level0/l0_0001.sst".to_string(),
        level: 0,
        min_key: "k00000000".to_string(),
        max_key: Some(format!("k{:08}", block_count * 64)),
        record_count: (block_count * 64) as u64,
        file_size_bytes: (block_count * 4096) as u64,
        vector_dimension: Some(32),
        centroid_encoding: CentroidEncoding::Fp16,
        centroid_fp16: None,
        centroid_fp32: None,
        zorder_min: None,
        zorder_max: None,
        block_index_offset: 0,
        block_index_size: 0,
        blocks,
    };

    let mut directory = VectorObjectEconomyDirectory::empty(
        "bench-coll",
        1,
        CatalogAuthorityMode::ProximaAuthoritative,
    );
    directory.push_file(file_entry);
    directory
}

// ────────────────────────────────────────────────────────────────────────
// Formatting helpers
// ────────────────────────────────────────────────────────────────────────

fn ns_to_us(ns: u128) -> u128 {
    ns / 1_000
}

fn format_f64(value: f64, precision: usize) -> String {
    format!("{:.*}", precision, value)
}

fn format_with_commas(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out.chars().rev().collect()
}

fn format_bytes(n: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    if n >= MB {
        format!("{:.2}MB", n as f64 / MB as f64)
    } else if n >= KB {
        format!("{:.2}KB", n as f64 / KB as f64)
    } else {
        format!("{}B", n)
    }
}
