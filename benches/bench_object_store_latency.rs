// Copyright 2026 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Object-store round-trip latency floor (co-design T3.2 — cloud-latency evidence).
//!
//! The dominant cost term for a cloud DB is the **I/O round-trip**, not CPU
//! (co-design P5). This bench measures that floor in isolation — the raw
//! object-store operations a ProximaDB scan/manifest read actually issues — so a
//! routing/format/cache decision can be justified against the *measured* RTT,
//! not a guess:
//!
//! * **GET** — a full-object GET (the round-trip a footer read or a small-segment
//!   scan pays).
//! * **ranged GET** — a byte-range GET (the footer-first / IVF-cold-load path
//!   the co-design leans on to avoid whole-object scans).
//! * **LIST** — a prefix enumeration (manifest / directory fan-out cost).
//!
//! ## Running
//!
//! Always runs (defaults to an in-process `memory://` store) so CI proves the
//! harness compiles and is sane. **In-process memory latencies are NOT cloud
//! latency** — they have no network RTT. To produce REAL cloud-latency evidence
//! point the bench at a real backend:
//!
//! ```text
//! BENCH_OBJECT_STORE_URL=s3://my-bench-bucket/path     cargo bench --bench bench_object_store_latency --features aws
//! BENCH_OBJECT_STORE_URL=file:///tmp/os-bench          cargo bench --bench bench_object_store_latency
//! BENCH_OBJECT_STORE_URL=memory://                     cargo bench --bench bench_object_store_latency
//! ```
//!
//! (`s3://` needs the `aws` feature, which compiles the S3 backend into
//! `object_store::parse_url`; `file://` and `memory://` work in a default build.)
//! S3 credentials come from the standard chain (env vars / IRSA / profile).

use std::sync::Arc;
use std::time::Instant;

use bytes::Bytes;
use futures::TryStreamExt;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::path::Path;
use tokio::runtime::Runtime;

// ────────────────────────────────────────────────────────────────────────
// main
// ────────────────────────────────────────────────────────────────────────

fn main() {
    let cfg = BenchConfig::from_env();

    println!("================================================================================");
    println!("   ProximaDB — Object-Store Round-Trip Latency Floor (T3.2)");
    println!("================================================================================");
    println!();
    println!("Configuration:");
    println!("  store_url:  {}", cfg.store_url);
    println!("  iterations: {} per (op × size)", cfg.iterations);
    println!("  sizes:      {} bytes", fmt_sizes(&cfg.sizes));
    if cfg.is_memory() {
        println!();
        println!("  ⚠ IN-PROCESS memory store — NO network RTT. These numbers are a");
        println!("    harness smoke check, NOT cloud latency. Set BENCH_OBJECT_STORE_URL");
        println!("    to s3:// (with --features aws) or file:// for real RTT evidence.");
    }
    println!();

    let rt = Runtime::new().expect("tokio runtime");
    rt.block_on(async move {
        match open_store(&cfg.store_url).await {
            Ok(store) => run_bench(&cfg, store).await,
            Err(e) => {
                eprintln!(
                    "error: could not open object store `{}`: {e}",
                    cfg.store_url
                );
                eprintln!("note: s3:// requires the `aws` feature (cargo bench --features aws);");
                eprintln!("      file:// and memory:// work in a default build.");
                std::process::exit(1);
            }
        }
    });
}

// ────────────────────────────────────────────────────────────────────────
// Config
// ────────────────────────────────────────────────────────────────────────

struct BenchConfig {
    store_url: String,
    iterations: usize,
    sizes: Vec<usize>,
}

impl BenchConfig {
    fn from_env() -> Self {
        Self {
            store_url: std::env::var("BENCH_OBJECT_STORE_URL")
                .unwrap_or_else(|_| "memory://os-bench".to_string()),
            iterations: env_usize("BENCH_OS_ITERATIONS", 200),
            // Footer-like, small-segment, large-segment — the shapes a scan pays.
            sizes: vec![4 * 1024, 64 * 1024, 1024 * 1024],
        }
    }

    fn is_memory(&self) -> bool {
        self.store_url.starts_with("memory://")
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn fmt_sizes(sizes: &[usize]) -> String {
    sizes
        .iter()
        .map(|s| format_bytes(*s as u64))
        .collect::<Vec<_>>()
        .join(", ")
}

// ────────────────────────────────────────────────────────────────────────
// Store construction
// ────────────────────────────────────────────────────────────────────────

async fn open_store(url: &str) -> Result<Arc<dyn ObjectStore>, Box<dyn std::error::Error>> {
    // In-process memory is the always-available default (no feature needed).
    if let Some(rest) = url.strip_prefix("memory://") {
        let _ = rest; // path is irrelevant for the in-memory store
        return Ok(Arc::new(object_store::memory::InMemory::new()));
    }
    let parsed = url::Url::parse(url)?;
    // Route through the production `store_for_url` (ADR-098 M5): it forwards
    // the process environment (lower-cased) into `parse_url_opts`, so the
    // standard S3 emulator contract (AWS_ENDPOINT, AWS_ALLOW_HTTP=true,
    // AWS_VIRTUAL_HOSTED_STYLE_REQUEST=false, keys) reaches MinIO/LocalStack —
    // bare `parse_url` builds a real-AWS endpoint and cannot target an
    // emulator, which is exactly the run this bench's own notes ask for.
    let (store, _) = proximadb_object_store::store_for_url(parsed.as_str())?;
    Ok(store)
}

// ────────────────────────────────────────────────────────────────────────
// Bench
// ────────────────────────────────────────────────────────────────────────

async fn run_bench(cfg: &BenchConfig, store: Arc<dyn ObjectStore>) {
    // Unique prefix so concurrent runs / leftovers don't interfere.
    let prefix_str = format!("os-bench-pid{}", std::process::id());
    let prefix = Path::from(prefix_str.clone());
    let obj_path = |size: usize| Path::from(format!("{prefix_str}/obj-{size}"));

    // Seed one object per size, plus a few siblings for a realistic LIST.
    for &size in &cfg.sizes {
        let body = object_store::PutPayload::from_bytes(Bytes::from(vec![0u8; size]));
        store
            .put(&obj_path(size), body)
            .await
            .expect("put seed object");
    }
    for i in 0..5 {
        let path = Path::from(format!("{prefix_str}/sibling-{i}"));
        store
            .put(
                &path,
                object_store::PutPayload::from_bytes(Bytes::from_static(b"x")),
            )
            .await
            .expect("put sibling");
    }

    println!(
        "Results (per-operation latency, {} iterations each):",
        cfg.iterations
    );
    println!();
    println!(
        "  {:<10} {:>10} {:>12} {:>12} {:>12} {:>12}",
        "op", "size", "p50 (ms)", "p95 (ms)", "p99 (ms)", "mean (ms)"
    );
    println!("  {}", "-".repeat(72));

    for &size in &cfg.sizes {
        let path = obj_path(size);

        // GET (full object) — the dominant round-trip a scan/footer-read pays.
        let get = measure(cfg.iterations, || async {
            let _ = store.get(&path).await?.bytes().await?;
            Ok::<_, object_store::Error>(())
        })
        .await;
        print_row("GET", size, &get);

        // ranged GET (last 4 KiB) — the footer-first / IVF cold-load path.
        let range_start = (size.saturating_sub(4 * 1024)) as u64;
        let range_end = size as u64;
        let ranged = measure(cfg.iterations, || async {
            let _ = store.get_range(&path, range_start..range_end).await?;
            Ok::<_, object_store::Error>(())
        })
        .await;
        print_row("GET range", size, &ranged);
    }

    // LIST (prefix enumeration) — manifest / directory fan-out.
    let list = measure(cfg.iterations, || async {
        let _ = store.list(Some(&prefix)).try_collect::<Vec<_>>().await?;
        Ok::<_, object_store::Error>(())
    })
    .await;
    print_row("LIST", 0, &list);

    println!();

    // Cleanup so repeated runs against a real bucket don't accumulate.
    let _ = cleanup(&store, &prefix).await;
}

/// Measure `op` `iters` times, returning the latency distribution in microseconds.
async fn measure<F, Fut>(iters: usize, op: F) -> LatencyDist
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<(), object_store::Error>>,
{
    // Warmup: prime any connection pool / TLS handshake so we measure steady-state.
    for _ in 0..(iters.min(10)) {
        let _ = op().await;
    }
    let mut samples: Vec<u64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        match op().await {
            Ok(()) => samples.push(start.elapsed().as_micros() as u64),
            // A failing op (e.g. transient) is recorded as the observed latency;
            // the bench reports round-trip cost, not just success cost.
            Err(_) => samples.push(start.elapsed().as_micros() as u64),
        }
    }
    LatencyDist::from_samples(samples)
}

async fn cleanup(store: &Arc<dyn ObjectStore>, prefix: &Path) -> Result<(), object_store::Error> {
    let mut stream = store.list(Some(prefix));
    while let Some(entry) = stream.try_next().await? {
        store.delete(&entry.location).await?;
    }
    Ok(())
}

// ────────────────────────────────────────────────────────────────────────
// Reporting
// ────────────────────────────────────────────────────────────────────────

struct LatencyDist {
    samples: Vec<u64>, // microseconds, sorted
}

impl LatencyDist {
    fn from_samples(mut samples: Vec<u64>) -> Self {
        samples.sort_unstable();
        Self { samples }
    }

    fn percentile(&self, p: f64) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let idx = ((self.samples.len() as f64 - 1.0) * p).round() as usize;
        self.samples[idx.min(self.samples.len() - 1)] as f64
    }

    fn mean(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        (self.samples.iter().sum::<u64>() as f64) / self.samples.len() as f64
    }
}

fn print_row(op: &str, size: usize, dist: &LatencyDist) {
    println!(
        "  {:<10} {:>10} {:>12.3} {:>12.3} {:>12.3} {:>12.3}",
        op,
        format_bytes(size as u64),
        dist.percentile(0.50) / 1000.0,
        dist.percentile(0.95) / 1000.0,
        dist.percentile(0.99) / 1000.0,
        dist.mean() / 1000.0,
    );
}

fn format_bytes(n: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * 1024;
    if n == 0 {
        return "-".to_string();
    }
    if n >= MIB {
        format!("{}MiB", n / MIB)
    } else if n >= KIB {
        format!("{}KiB", n / KIB)
    } else {
        format!("{n}B")
    }
}
