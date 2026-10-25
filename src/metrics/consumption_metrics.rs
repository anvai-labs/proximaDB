#![allow(clippy::unwrap_used, clippy::expect_used)]
// dedicated metric-registration module: static registration is infallible / fail-fast at startup; lazy_static can't carry per-site allows
// Copyright 2026 ProximaDB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Prometheus consumption telemetry: request/byte-level operational counters
//!
//! Generic per-tenant usage counters (object-store ops, storage byte-seconds,
//! task time). Pricing/billing is an operator/control-plane concern — this OSS
//! module only emits neutral telemetry attributed by `tenant_id`.

use std::net::IpAddr;
use std::sync::{LazyLock, Mutex};

use ipnet::IpNet;
use lazy_static::lazy_static;
use prometheus::{CounterVec, GaugeVec, Opts, register_counter_vec, register_gauge_vec};
use tracing::error;

// Register-or-fallback helpers. Static registration is infallible / fail-fast at
// startup (see module-level allow); centralizing the fallback keeps the single
// sanctioned panic site per vec type instead of repeating it at every metric.
fn registered_counter_vec(name: &'static str, help: &'static str, labels: &[&str]) -> CounterVec {
    register_counter_vec!(Opts::new(name, help), labels).unwrap_or_else(|err| {
        // No-panic mandate #4: registration can fail (e.g. a duplicate name).
        // The fallback MUST NOT reuse an empty help — prometheus rejects that
        // too, so the old `""` here panicked the tokio worker on the first
        // search (an unregistered-but-usable vec is the correct degradation).
        error!("failed to register {name} (using unregistered fallback): {err}");
        CounterVec::new(Opts::new(name, help), labels)
            .expect("counter descriptor with non-empty help is always valid")
    })
}

fn registered_gauge_vec(name: &'static str, help: &'static str, labels: &[&str]) -> GaugeVec {
    register_gauge_vec!(Opts::new(name, help), labels).unwrap_or_else(|err| {
        error!("failed to register {name} (using unregistered fallback): {err}");
        GaugeVec::new(Opts::new(name, help), labels)
            .expect("gauge descriptor with non-empty help is always valid")
    })
}

lazy_static! {
    pub static ref OBJECT_STORE_OPS_TOTAL: CounterVec = registered_counter_vec(
        "proximadb_object_store_ops_total",
        "Total number of object store I/O operations (put, get, list, delete)",
        &["tenant_id", "operation"]
    );
    /// TD-IOTRACE-2: per-tenant physical object-store ranged GETs — the accurate
    /// per-query GET count from the io-trace snapshot (post-#1081 single-source
    /// counting, e.g. 51/query @ SIFT1M not 102). The billing input for the
    /// physical part of the two-part KRU rate (TD-IOTRACE-3). Neutral count only;
    /// AnvaiOps (control plane) applies the $/rate-card.
    pub static ref OBJECT_STORE_GETS_TOTAL: CounterVec = registered_counter_vec(
        "proximadb_object_store_gets_total",
        "Per-tenant physical object-store ranged GETs (TD-IOTRACE-2, KRU physical)",
        &["tenant_id"]
    );
    /// TD-IOTRACE-2: per-tenant physical object-store bytes read — the accurate
    /// per-query read bytes from the io-trace snapshot (post-#1081). Companion to
    /// [`OBJECT_STORE_GETS_TOTAL`]; the bytes side of the KRU physical rate input.
    /// Neutral count only; AnvaiOps applies the $/rate-card.
    pub static ref OBJECT_STORE_BYTES_READ_TOTAL: CounterVec = registered_counter_vec(
        "proximadb_object_store_bytes_read_total",
        "Per-tenant physical object-store bytes read (TD-IOTRACE-2, KRU physical)",
        &["tenant_id"]
    );
    /// TD-PAXDIR-1 D7: per-tenant physical object-store **read requests** —
    /// GET + LIST + HEAD, the quantity the P5b gate's `requests_per_query`
    /// is defined on (design §4.2 G3). Neutral count only.
    pub static ref OBJECT_STORE_REQUESTS_TOTAL: CounterVec = registered_counter_vec(
        "proximadb_object_store_requests_total",
        "Per-tenant physical object-store read requests: GET + LIST + HEAD (TD-PAXDIR-1 D7)",
        &["tenant_id"]
    );
    pub static ref IVF_CELLS_TOTAL: CounterVec = registered_counter_vec(
        "proximadb_ivf_cells_total",
        "Persisted-IVF cells considered by vector queries (TD-RDSTRAT-8)",
        &["tenant_id"]
    );
    pub static ref IVF_CELLS_PROBED_TOTAL: CounterVec = registered_counter_vec(
        "proximadb_ivf_cells_probed_total",
        "Persisted-IVF cells probed by vector queries (TD-RDSTRAT-8)",
        &["tenant_id"]
    );
    pub static ref IVF_PROBED_ROWS_TOTAL: CounterVec = registered_counter_vec(
        "proximadb_ivf_probed_rows_total",
        "Rows covered by persisted-IVF probes (TD-RDSTRAT-8)",
        &["tenant_id"]
    );
    pub static ref IVF_REGION_A_BYTES_TOTAL: CounterVec = registered_counter_vec(
        "proximadb_ivf_region_a_bytes_read_total",
        "Physical PAX Region-A (RaBitQ) bytes read by vector queries (TD-RDSTRAT-8)",
        &["tenant_id"]
    );
    pub static ref IVF_REGION_B_BYTES_TOTAL: CounterVec = registered_counter_vec(
        "proximadb_ivf_region_b_bytes_read_total",
        "Physical PAX Region-B (SQ8) bytes read by vector queries (TD-RDSTRAT-8)",
        &["tenant_id"]
    );
    pub static ref IVF_FETCH_ROUNDS_TOTAL: CounterVec = registered_counter_vec(
        "proximadb_ivf_fetch_rounds_total",
        "Coalesced Region-A probe ranged-read runs (TD-RDSTRAT-8)",
        &["tenant_id"]
    );
    pub static ref IVF_WHOLE_REGION_FALLBACK_TOTAL: CounterVec = registered_counter_vec(
        "proximadb_ivf_whole_region_fallback_total",
        "Armed v3 probes that fell back to a whole Region-A scan (TD-RDSTRAT-8)",
        &["tenant_id"]
    );
    /// TD-RDSTRAT-12: per-tenant parallel read-rounds counter — the count of
    /// concurrent rounds issued by `FileSystem::read_ranges` when bounded parallelism
    /// is enabled (`PROXIMADB_FS_READ_RANGES_PARALLEL > 1`). Surface for the
    /// "waves-not-sums" latency model: `latency ≈ rounds × RTT + ...`.
    pub static ref READ_RANGES_FETCH_ROUNDS_TOTAL: CounterVec = registered_counter_vec(
        "proximadb_read_ranges_fetch_rounds_total",
        "Parallel read-rounds issued by bounded-concurrent FileSystem::read_ranges (TD-RDSTRAT-12)",
        &["tenant_id"]
    );
    /// TD-RDSTRAT-12: per-tenant max-inflight gauge — peak concurrent reads in
    /// a single `read_ranges` call. Shows the actual parallelism achieved vs the
    /// `PROXIMADB_FS_READ_RANGES_PARALLEL` cap.
    pub static ref READ_RANGES_MAX_INFLIGHT: GaugeVec = registered_gauge_vec(
        "proximadb_read_ranges_max_inflight",
        "Peak concurrent reads in a FileSystem::read_ranges call (TD-RDSTRAT-12)",
        &["tenant_id"]
    );
    pub static ref STORAGE_BYTES_SECONDS: GaugeVec = registered_gauge_vec(
        "proximadb_storage_bytes_seconds",
        "GB-seconds or raw bytes stored per tenant",
        &["tenant_id", "storage_type"]
    );
    pub static ref TASK_EXECUTION_TIME_MS: CounterVec = registered_counter_vec(
        "proximadb_task_execution_time_ms",
        "Total analytical task execution time in milliseconds (DataFusion/Polars compute)",
        &["tenant_id", "engine"]
    );
    /// Per-tenant cache footprint + hit/miss snapshot (the multitenant cache
    /// observability for fairness/noisy-neighbor + chargeback). Gauges carry the
    /// current cumulative totals per tenant from `TenantCache::tenant_stats`.
    pub static ref CACHE_TENANT_STATS: GaugeVec = registered_gauge_vec(
        "proximadb_cache_tenant",
        "Per-tenant cache stats snapshot (metric label selects bytes/hits/misses/hit_ratio/evictions)",
        &["tenant_id", "cache", "metric"]
    );
    /// Per-tenant bytes moved OUT of the deployment — the **KOU** (Outgress Unit)
    /// meter (co-design Dimension 2 / §2.2). `direction` separates `read`
    /// (object store → compute) from `result` (compute → client); `kou_locality`
    /// is the cost discriminator (cross-region ~$0.02/GB, internet ~$0.09-0.12/GB
    /// by cloud; same-region/AZ free). Networking is frequently the dominant TCO
    /// term. Neutral telemetry only: the cloud topology and the $ weights are
    /// control-plane (anvaiops) policy. (KEU = embedding units is a separate meter.)
    pub static ref KOU_BYTES_TOTAL: CounterVec = registered_counter_vec(
        "proximadb_kou_bytes_total",
        "Outgress bytes per tenant by locality + direction (KOU, Dimension 2)",
        &["tenant_id", "kou_locality", "direction"]
    );
    /// Per-tenant embedding units — the **KEU** (Kilo-Embedding-Unit) meter
    /// (Dimension 5). Distinct from general read KRU. `provider` identifies the
    /// embedding service (e.g. "openai", "victor", "huggingface"); `model` names
    /// the model (e.g. "text-embedding-3-small", "bge-base-en-v1.5"); `operation`
    /// is the embedding operation type (e.g. "embed", "embed_batch").
    /// Neutral telemetry only: the per-(provider, model) $ weight is control-plane
    /// (anvaiops) policy.
    pub static ref KEU_UNITS_TOTAL: CounterVec = registered_counter_vec(
        "proximadb_keu_units_total",
        "Embedding units (Kilo-Embedding-Units, KEU, Dimension 5) per tenant by provider, model, and operation",
        &["tenant_id", "provider", "model", "operation"]
    );
    /// Bytes written to object storage per tenant **by access tier** — write-time
    /// visibility into the cold-tier cost lever (ADR-036/TD-173; TD-168 graph cold
    /// payloads). `tier` is the canonical name (`hot`/`cool`/`cold`/`archive`),
    /// known at the `put_with_tier` call site. This is a *written-bytes* counter,
    /// distinct from the periodic *resident-bytes* `STORAGE_BYTES_SECONDS` gauge
    /// (whose Cool-vs-Hot split needs collection-level tier metadata — a separate
    /// metering change). Neutral telemetry; the per-tier $ weight is control-plane
    /// (anvaiops) policy.
    pub static ref OBJECT_STORE_WRITE_BYTES_BY_TIER: CounterVec = registered_counter_vec(
        "proximadb_object_store_write_bytes_by_tier_total",
        "Bytes written to object storage per tenant by access tier (write-time; the cold-tier cost lever)",
        &["tenant_id", "tier"]
    );
    /// TD-WLP-8 (ADR-061 D4): resident in-memory **working-set** bytes of a
    /// `Churn` collection — the unflushed WAL/memtable delta it keeps hot for
    /// immediate-freshness reads. A level gauge (current resident bytes),
    /// sampled at the `Churn` read seam; the companion soft ceiling
    /// (`PROXIMADB_CHURN_WORKING_SET_MB`) makes an over-budget collection
    /// observable so it degrades gracefully instead of silently OOMing.
    /// Emitted only for `Churn` collections (`AppendBulk` never samples).
    pub static ref CHURN_WORKING_SET_BYTES: GaugeVec = registered_gauge_vec(
        "proximadb_churn_collection_working_set_bytes",
        "Resident in-memory working-set bytes of a Churn collection (unflushed WAL/memtable delta)",
        &["tenant_id", "collection_id"]
    );
}

/// Cloud provider hosting the object store — the multi-cloud axis for egress
/// pricing (AWS / Azure / GCP each price data movement differently). OSS
/// **mechanism**: it is *inferred from the storage URL scheme* the operator
/// already configures (so it can never be set wrong), never hand-declared. The
/// $ weights per (provider, locality) are control-plane (anvaiops) **policy**.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CloudProvider {
    Aws,
    Azure,
    Gcp,
    /// Local filesystem / dev / unknown — no egress cost.
    #[default]
    Local,
}

impl CloudProvider {
    /// Stable label string.
    pub fn as_str(self) -> &'static str {
        match self {
            CloudProvider::Aws => "aws",
            CloudProvider::Azure => "azure",
            CloudProvider::Gcp => "gcp",
            CloudProvider::Local => "local",
        }
    }

    /// Infer the provider from an object-store URL scheme — the same scheme the
    /// `FilesystemFactory` dispatches on (`s3`, `gs`/`gcs`, `az`/`adls`/`abfs`,
    /// `file`). Unknown / empty → [`CloudProvider::Local`]. This is the
    /// mistake-proof seam: the provider follows the storage URL, not a separate
    /// env an operator could contradict.
    pub fn from_url_scheme(scheme: &str) -> Self {
        match scheme.trim().to_ascii_lowercase().as_str() {
            "s3" | "s3a" => CloudProvider::Aws,
            "gs" | "gcs" => CloudProvider::Gcp,
            "az" | "azure" | "adls" | "abfs" | "abfss" => CloudProvider::Azure,
            _ => CloudProvider::Local,
        }
    }
}

/// Locality of bytes moving out of the deployment — the **KOU** (Outgress Unit)
/// dimension (§2.2), cloud-neutral. Used for two flows: *read-egress* (object
/// store → compute) and *result-egress* (compute → client). Distinct from
/// **KEU = embedding units** (a separate, embedding-compute meter).
///
/// Region-granular by design: object-store reads are billed by *region*, not
/// availability zone — same-region access is **free on every cloud** (S3 via the
/// VPC gateway endpoint, ADLS/Blob is regional, GCS via private access). So
/// `Local`/`CrossAz` are recorded for visibility but **not** chargeable; only
/// `CrossRegion`/`CrossCloud`/`OnPrem`/`Internet` cost. `Unknown` is recorded and
/// surfaced (never silently charged or freed) so an operator fixes the scope map.
/// OSS records the neutral bucket; the per-(provider, bucket) $ weight is
/// anvaiops policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KouLocality {
    /// Compute and the other endpoint co-located (same region/zone) — free.
    /// The safe default so outgress stays inert until placement is known.
    #[default]
    Local,
    /// Same region, different availability zone — **free for object storage** on
    /// all clouds (recorded for visibility + the future compute↔compute meter).
    CrossAz,
    /// Different region, same cloud — chargeable cross-region transfer (~$0.02/GB).
    CrossRegion,
    /// Different cloud provider — chargeable (internet-tier, expensive).
    CrossCloud,
    /// To/from on-premises (over VPN / dedicated link / declared CIDR).
    OnPrem,
    /// Public internet (~$0.09-0.12/GB by cloud) — the dominant result-egress term.
    Internet,
    /// Could not be classified (no scope-map rule matched) — recorded + surfaced,
    /// never silently treated as free or charged.
    Unknown,
}

impl KouLocality {
    /// Stable metric-label string.
    pub fn as_str(self) -> &'static str {
        match self {
            KouLocality::Local => "local",
            KouLocality::CrossAz => "cross_az",
            KouLocality::CrossRegion => "cross_region",
            KouLocality::CrossCloud => "cross_cloud",
            KouLocality::OnPrem => "on_prem",
            KouLocality::Internet => "internet",
            KouLocality::Unknown => "unknown",
        }
    }

    /// Whether bytes through this locality are chargeable outgress. `Local` and
    /// `CrossAz` object-store access is free on every cloud, and `Unknown` is not
    /// charged (it is surfaced for the operator to fix), so only genuinely
    /// cross-boundary movement feeds the cost model's egress term + the trace.
    pub fn is_chargeable_egress(self) -> bool {
        matches!(
            self,
            KouLocality::CrossRegion
                | KouLocality::CrossCloud
                | KouLocality::OnPrem
                | KouLocality::Internet
        )
    }

    /// Parse an explicit operator override / scope-map value, accepting obvious
    /// aliases. `None` for an unrecognized value (caller keeps the derived value
    /// rather than silently mis-charging).
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "local" | "same_az" | "colocated" | "same_zone" => Some(KouLocality::Local),
            "cross_az" | "same_region" | "region" => Some(KouLocality::CrossAz),
            "cross_region" | "crossregion" => Some(KouLocality::CrossRegion),
            "cross_cloud" | "crosscloud" => Some(KouLocality::CrossCloud),
            "on_prem" | "onprem" | "on_premises" => Some(KouLocality::OnPrem),
            "internet" | "egress" | "public" => Some(KouLocality::Internet),
            "unknown" => Some(KouLocality::Unknown),
            _ => None,
        }
    }

    /// Derive the **read-egress** locality (object store → compute) from a
    /// [`PlacementContext`] — the per-cloud truth table, the mistake-proof core.
    /// A different store cloud → `CrossCloud`; region equality (not zone) decides
    /// the rest; GCS multi-region scopes (`US`/`EU`/`ASIA`) containing the compute
    /// region count as same-region (`CrossAz`, free).
    pub fn derive_read(ctx: &PlacementContext) -> Self {
        // No cloud, or we don't know where compute runs → assume free (never
        // invent a charge); the caller logs the assumption.
        if ctx.store_provider == CloudProvider::Local || ctx.compute_region.trim().is_empty() {
            return KouLocality::Local;
        }
        // Compute in a different cloud than the store → cross-cloud read.
        if ctx.compute_provider != CloudProvider::Local
            && ctx.compute_provider != ctx.store_provider
        {
            return KouLocality::CrossCloud;
        }
        let compute = ctx.compute_region.trim().to_ascii_lowercase();
        let store = ctx.store_region.trim().to_ascii_lowercase();
        if store.is_empty() {
            return KouLocality::Local; // store region unknown → don't over-charge
        }
        if compute == store || gcs_multi_region_contains(&store, &compute) {
            // Same region (object store is regional) → free. Recorded as CrossAz
            // only when zones are known to differ; otherwise the free Local.
            KouLocality::Local
        } else {
            KouLocality::CrossRegion
        }
    }
}

/// Whether a GCS multi-region / dual-region bucket scope (`us`, `eu`, `asia`)
/// contains a specific compute region (e.g. `us-central1` ∈ `us`). Region-scoped
/// store values fall through to plain equality in the caller.
fn gcs_multi_region_contains(store: &str, compute: &str) -> bool {
    matches!(store, "us" | "eu" | "asia") && compute.starts_with(store)
}

/// Inputs to [`KouLocality::derive_read`]: the store cloud + region (from the
/// storage URL scheme + backend config) and where compute runs
/// (`PROXIMADB_COMPUTE_REGION`, control-plane stamped; `compute_provider`
/// defaults to the store provider — same cloud — unless declared otherwise).
#[derive(Debug, Clone, Default)]
pub struct PlacementContext {
    pub store_provider: CloudProvider,
    pub compute_provider: CloudProvider,
    pub compute_region: String,
    pub store_region: String,
}

/// The process-wide derived object-store egress locality, computed once at
/// startup by [`install_placement`]. Defaults to `Local` (free, inert) until the
/// deployment installs a real [`PlacementContext`] — so OSS-standalone / dev runs
/// never bill egress.
static OBJECT_STORE_LOCALITY: LazyLock<Mutex<KouLocality>> =
    LazyLock::new(|| Mutex::new(KouLocality::Local));

/// Install the deployment's placement (called once at startup). Derives the
/// read-egress locality from `ctx`, applies the `PROXIMADB_OBJECT_STORE_LOCALITY`
/// override when present+valid, stores it, and returns the effective value so the
/// caller can log it. A cloud provider with an unknown compute region logs a WARN
/// (visible assumption, not a silent free pass).
pub fn install_placement(ctx: &PlacementContext) -> KouLocality {
    let derived = KouLocality::derive_read(ctx);
    let effective = match std::env::var("PROXIMADB_OBJECT_STORE_LOCALITY") {
        Ok(v) if !v.trim().is_empty() => match KouLocality::parse(&v) {
            Some(override_loc) => override_loc,
            None => {
                tracing::warn!(
                    value = %v,
                    "ignoring invalid PROXIMADB_OBJECT_STORE_LOCALITY override; \
                     using derived locality"
                );
                derived
            }
        },
        _ => derived,
    };
    if ctx.store_provider != CloudProvider::Local && ctx.compute_region.trim().is_empty() {
        tracing::warn!(
            store_provider = ctx.store_provider.as_str(),
            "PROXIMADB_COMPUTE_REGION unset on a cloud deployment; assuming \
             same-region (free) egress — set it so cross-region egress is metered"
        );
    }
    tracing::info!(
        store_provider = ctx.store_provider.as_str(),
        compute_region = %ctx.compute_region,
        store_region = %ctx.store_region,
        locality = effective.as_str(),
        "co-design Dimension 2 (KOU): derived read-egress locality"
    );
    *OBJECT_STORE_LOCALITY
        .lock()
        .unwrap_or_else(|p| p.into_inner()) = effective;
    effective
}

/// The effective object-store read-egress locality for this deployment (derived
/// at startup; `Local` until [`install_placement`] runs).
pub fn object_store_egress_locality() -> KouLocality {
    *OBJECT_STORE_LOCALITY
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

/// Build a [`PlacementContext`] from the storage URL + control-plane-stamped env,
/// for the startup [`install_placement`] call. Provider is *inferred* from the
/// storage URL scheme (mistake-proof); `compute_region` / `store_region` are
/// stamped by the control plane (`PROXIMADB_COMPUTE_REGION` / `PROXIMADB_STORE_REGION`,
/// the latter falling back to the AWS region envs for the S3 backend). All default
/// empty → derives `Local` (free, inert). `compute_provider` defaults to the store
/// provider (same cloud); cross-cloud reads are a declared exception (not inferred).
pub fn placement_from_env(storage_url: &str) -> PlacementContext {
    let scheme = storage_url.split("://").next().unwrap_or("");
    let provider = CloudProvider::from_url_scheme(scheme);
    let env = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
    let store_region = env("PROXIMADB_STORE_REGION")
        .or_else(|| {
            if provider == CloudProvider::Aws {
                env("PROXIMADB_S3_REGION")
                    .or_else(|| env("AWS_REGION"))
                    .or_else(|| env("AWS_DEFAULT_REGION"))
            } else {
                None
            }
        })
        .unwrap_or_default();
    PlacementContext {
        store_provider: provider,
        compute_provider: provider,
        compute_region: env("PROXIMADB_COMPUTE_REGION").unwrap_or_default(),
        store_region,
    }
}

// ── Client-locality classification (KOU result-egress, co-design D2) ─────────
//
// Classify a client's source IP against a control-plane-authored CIDR→locality
// scope map (terraform emits it from the VPC/subnet CIDRs it provisions, so it
// can never be hand-mis-written). Longest-prefix match; no match → `Internet`
// for a public IP, `Unknown` for an unrecognized private IP (surfaced, never
// silently free/charged). No fragile cloud IP-range databases are consulted.

/// One scope-map rule: a CIDR and the [`KouLocality`] of a client whose source
/// IP falls in it (the terraform-baked AZ/region decision is encoded in the bucket).
#[derive(Debug, Clone)]
pub struct ScopeRule {
    pub net: IpNet,
    pub locality: KouLocality,
}

/// Ordered CIDR→locality rules; classification is **longest-prefix wins** (a /32
/// host rule overrides a /16 VPC rule). Default-empty → every client is `Unknown`.
#[derive(Debug, Clone, Default)]
pub struct NetworkScopeMap {
    rules: Vec<ScopeRule>,
}

impl NetworkScopeMap {
    pub fn from_rules(rules: Vec<ScopeRule>) -> Self {
        Self { rules }
    }

    /// Parse a JSON array `[{"cidr":"10.0.0.0/16","locality":"cross_az"}, …]`.
    /// Unparseable CIDRs / localities are skipped with a WARN (never panics) —
    /// a malformed rule degrades to `Unknown` for those IPs, never a wrong charge.
    pub fn from_json(json: &str) -> Self {
        let mut rules = Vec::new();
        let parsed: serde_json::Value = match serde_json::from_str(json) {
            Ok(v) => v,
            Err(e) => {
                error!("network scope map: invalid JSON ({e}); using empty map");
                return Self::default();
            }
        };
        if let Some(arr) = parsed.as_array() {
            for entry in arr {
                let cidr = entry.get("cidr").and_then(|v| v.as_str());
                let loc = entry
                    .get("locality")
                    .and_then(|v| v.as_str())
                    .and_then(KouLocality::parse);
                match (cidr.and_then(|c| c.parse::<IpNet>().ok()), loc) {
                    (Some(net), Some(locality)) => rules.push(ScopeRule { net, locality }),
                    _ => error!("network scope map: skipping invalid rule {entry}"),
                }
            }
        }
        Self { rules }
    }

    /// Classify a client source IP. Longest-prefix match → the rule's locality;
    /// no match → `Internet` if the IP is public, else `Unknown`.
    pub fn classify(&self, ip: IpAddr) -> KouLocality {
        match self
            .rules
            .iter()
            .filter(|r| r.net.contains(&ip))
            .max_by_key(|r| r.net.prefix_len())
        {
            Some(r) => r.locality,
            None if is_public_ip(ip) => KouLocality::Internet,
            None => KouLocality::Unknown,
        }
    }
}

/// Whether an IP is public (routable on the internet) — i.e. not loopback,
/// private (RFC1918 / unique-local), or link-local. A non-public unmatched IP is
/// `Unknown` (surfaced); a public unmatched IP is `Internet`.
fn is_public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            !(v4.is_private() || v4.is_loopback() || v4.is_link_local() || v4.is_unspecified())
        }
        IpAddr::V6(v6) => {
            // not loopback / unspecified / unique-local (fc00::/7) / link-local (fe80::/10)
            let seg = v6.segments();
            !(v6.is_loopback()
                || v6.is_unspecified()
                || (seg[0] & 0xfe00) == 0xfc00
                || (seg[0] & 0xffc0) == 0xfe80)
        }
    }
}

/// Process-wide client-locality scope map, installed at startup by
/// [`install_network_scope_map`]. Default-empty → every client `Unknown` (safe).
static NETWORK_SCOPE_MAP: LazyLock<Mutex<NetworkScopeMap>> =
    LazyLock::new(|| Mutex::new(NetworkScopeMap::default()));

/// Install the deployment's client scope map (called once at startup). Reads the
/// JSON file at `PROXIMADB_NETWORK_SCOPE_MAP` when `map` is `None`; default-empty
/// when the env is unset/unreadable (safe: clients classify as `Unknown`).
pub fn install_network_scope_map(map: Option<NetworkScopeMap>) {
    let map = map.unwrap_or_else(|| match std::env::var("PROXIMADB_NETWORK_SCOPE_MAP") {
        Ok(path) if !path.trim().is_empty() => match std::fs::read_to_string(&path) {
            Ok(json) => NetworkScopeMap::from_json(&json),
            Err(e) => {
                error!("network scope map: cannot read {path} ({e}); using empty map");
                NetworkScopeMap::default()
            }
        },
        _ => NetworkScopeMap::default(),
    });
    *NETWORK_SCOPE_MAP.lock().unwrap_or_else(|p| p.into_inner()) = map;
}

/// Classify a client source IP against the installed scope map (KOU result-egress
/// locality). `Unknown` until [`install_network_scope_map`] loads rules.
pub fn classify_client(ip: IpAddr) -> KouLocality {
    NETWORK_SCOPE_MAP
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .classify(ip)
}

/// Per-request edge classification — the client IP and its derived KOU locality,
/// computed ONCE at a surface boundary (REST / pgwire / Arrow Flight) and carried
/// for the request. This is the unifying seam of the edge consolidation (HLD
/// E0→E1): every surface classifies the client identically and meters
/// result-egress through one entry point, instead of each re-deriving locality.
/// OSS carries only the neutral bucket; the per-(provider, bucket) $ weight is
/// anvaiops policy.
#[derive(Debug, Clone, Copy)]
pub struct EdgePolicyContext {
    /// The client's source IP (the rate-limit + locality subject).
    pub client_ip: IpAddr,
    /// The client's KOU locality (free vs chargeable result-egress).
    pub kou_locality: KouLocality,
}

impl EdgePolicyContext {
    /// Classify a client source IP against the installed scope map. An
    /// unspecified (`0.0.0.0`/`::`) or loopback address is treated as `Local`
    /// (free, inert): these are embedded / same-host callers, never a chargeable
    /// remote, so result-egress is never mis-charged for an in-process or
    /// loopback client.
    pub fn classify(client_ip: IpAddr) -> Self {
        let kou_locality = if client_ip.is_unspecified() || client_ip.is_loopback() {
            KouLocality::Local
        } else {
            classify_client(client_ip)
        };
        Self {
            client_ip,
            kou_locality,
        }
    }

    /// Record result-egress (compute → this client) bytes for the request's
    /// tenant through the convergent KOU meter (`direction = "result"`). No-ops on
    /// zero bytes and on the free path; folds chargeable bytes into the active
    /// query's egress trace, exactly like the read-egress boundary.
    pub fn record_result_egress(&self, tenant_id: Option<&str>, bytes: u64) {
        record_kou_bytes(tenant_id, self.kou_locality, "result", bytes);
    }

    /// The egress-aware result-shaping decision for this request (see
    /// [`ResultShapePolicy`]).
    pub fn shape_policy(&self) -> ResultShapePolicy {
        ResultShapePolicy::for_locality(self.kou_locality)
    }
}

/// Egress-aware result-shaping decision, derived from a request's KOU locality.
/// This is the one place the edge decides *how* to serialize a response based on
/// where the client is — the EdgePolicyPlane "shaping stage" consulted by every
/// surface (Arrow Flight compression, REST advisory) so the policy is uniform and
/// not re-derived per surface.
///
/// Active for chargeable/far clients but **lossless only by default**: it enables
/// byte-minimization (compression / columnar density) that the client decodes
/// transparently — it never drops or truncates results. Lossy levers (result
/// caps, reduced rerank) are intentionally NOT decided here; they require an
/// explicit, signaled opt-in. The $ weight of "far" stays anvaiops policy; OSS
/// keys purely on the neutral locality bucket.
#[derive(Debug, Clone, Copy)]
pub struct ResultShapePolicy {
    /// The locality this decision was derived from.
    pub kou_locality: KouLocality,
    /// Prefer compressing the result body (lossless). Set for chargeable
    /// (cross-region/-cloud/on-prem/internet) clients where egress dominates.
    pub compress: bool,
}

impl ResultShapePolicy {
    /// Derive the shaping decision from a KOU locality. Compression is on for
    /// chargeable localities, off on the free path (saves CPU for near clients).
    pub fn for_locality(kou_locality: KouLocality) -> Self {
        Self {
            kou_locality,
            compress: kou_locality.is_chargeable_egress(),
        }
    }
}

/// Helper to record an object store operation.
pub fn record_object_store_op(tenant_id: Option<&str>, operation: &str) {
    let t_id = tenant_id.unwrap_or("default");
    OBJECT_STORE_OPS_TOTAL
        .with_label_values(&[t_id, operation])
        .inc();
}

/// Helper to update storage capacity usage gauge.
pub fn record_storage_bytes(tenant_id: Option<&str>, storage_type: &str, bytes: f64) {
    let t_id = tenant_id.unwrap_or("default");
    STORAGE_BYTES_SECONDS
        .with_label_values(&[t_id, storage_type])
        .set(bytes);
}

/// TD-WLP-8: set the `Churn` collection working-set level gauge. Called from
/// the `Churn` read seam with the collection's current unflushed byte count;
/// pure gauge `.set()` (the soft-ceiling decision lives at the call site,
/// which owns the env). An absent/empty tenant is attributed to `"default"`.
pub fn record_churn_working_set_bytes(tenant_id: Option<&str>, collection_id: &str, bytes: u64) {
    let t_id = tenant_id.unwrap_or("default");
    CHURN_WORKING_SET_BYTES
        .with_label_values(&[t_id, collection_id])
        .set(bytes as f64);
}

/// ADR-030 **KSU**: snapshot per-tenant *resident* storage bytes from the live
/// collection set and `.set()` the `STORAGE_BYTES_SECONDS` **level** gauge
/// (Prometheus integrates the level to byte-seconds downstream — KSU is an
/// accrual, not a per-flush byte count, so this is a periodic snapshot rather
/// than a write-path event). Groups collections by owner (tenant) and sums
/// `stats.data_size_bytes`; an absent/empty owner is attributed to `"default"`
/// (fail-closed, never dropped). Pure aggregation + the gauge set — the caller
/// drives it on an interval (the storage-snapshot daemon in `database.rs`).
pub fn record_storage_snapshot(
    collections: &[crate::proto::proximadb_v1::Collection],
) -> Vec<TenantStorageUsage> {
    let usage = aggregate_tenant_storage(collections);
    for u in &usage {
        record_storage_bytes(Some(&u.tenant_id), "sst", u.resident_bytes as f64);
    }
    usage
}

/// One tenant's aggregated **resident** storage at snapshot time — the unit the
/// Pre-extracted to `proximadb-tenant` (foundation) for the observability crate
/// extraction. Re-exported here so all existing callers are unchanged.
pub use proximadb_tenant::TenantStorageUsage;

/// Pure aggregation backing [`record_storage_snapshot`]: sum
/// `stats.data_size_bytes` per owning tenant across the live collection set.
/// An absent/empty `config.owner` is attributed to `"default"` (fail-closed,
/// never dropped — KSU must account for every resident byte). No side effects:
/// callers drive the Prometheus level gauge ([`record_storage_snapshot`]) and/or
/// the durable metering sink (TD-161) from the returned rows. Ordered by
/// `tenant_id` (`BTreeMap`) so the persisted record + tests are deterministic.
pub fn aggregate_tenant_storage(
    collections: &[crate::proto::proximadb_v1::Collection],
) -> Vec<TenantStorageUsage> {
    use std::collections::BTreeMap;
    let mut by_tenant: BTreeMap<&str, u64> = BTreeMap::new();
    for c in collections {
        let tenant = c
            .config
            .as_ref()
            .and_then(|cfg| cfg.owner.as_deref())
            .filter(|o| !o.is_empty())
            .unwrap_or("default");
        let bytes = c.stats.as_ref().map_or(0i64, |s| s.data_size_bytes).max(0) as u64;
        *by_tenant.entry(tenant).or_insert(0) += bytes;
    }
    by_tenant
        .into_iter()
        .map(|(tenant_id, resident_bytes)| TenantStorageUsage {
            tenant_id: tenant_id.to_string(),
            resident_bytes,
        })
        .collect()
}

/// Helper to record compute execution time.
pub fn record_task_execution_time(tenant_id: Option<&str>, engine: &str, duration_ms: f64) {
    let t_id = tenant_id.unwrap_or("default");
    TASK_EXECUTION_TIME_MS
        .with_label_values(&[t_id, engine])
        .inc_by(duration_ms);
}

/// ADR-030: install the always-on **billing** observer on the per-query I/O trace.
///
/// At each query's trace flush this reads the *same* `IoTraceSnapshot` the perf
/// observers read and emits the per-tenant **KRU** read-compute meter from the
/// snapshot's per-engine `compute_ms` map — closing the verified-unwired
/// `record_task_execution_time` gap. Always-on (never behind the perf `io-trace`
/// feature gate; ADR-027 non-entanglement). Called once at startup, after the
/// route-cost observer. Idempotent / replaceable in tests.
pub fn install_billing_observer() {
    crate::observability::io_trace::set_billing_observer(Some(Box::new(|snap, tenant_id| {
        let t_id = tenant_id.unwrap_or("default");
        // KRU (read-compute): per-(tenant, engine), straight from the engine-keyed
        // compute_ms map — no extra attribution needed.
        for (engine, ms) in &snap.compute_ms {
            if *ms > 0 {
                record_task_execution_time(tenant_id, engine, *ms as f64);
            }
        }
        // TD-IOTRACE-2: the physical object-store meters — the accurate per-query
        // ranged-GET count + bytes (post-#1081 single-source io-trace). These are
        // the billing input for the physical part of the two-part KRU rate
        // (TD-IOTRACE-3). Neutral counts; AnvaiOps applies the $/rate-card.
        if snap.range_gets > 0 {
            OBJECT_STORE_GETS_TOTAL
                .with_label_values(&[t_id])
                .inc_by(snap.range_gets as f64);
        }
        if snap.bytes_read > 0 {
            OBJECT_STORE_BYTES_READ_TOTAL
                .with_label_values(&[t_id])
                .inc_by(snap.bytes_read as f64);
        }
        // TD-PAXDIR-1 D7: request count (GET + LIST + HEAD) and, for vector
        // queries, the query's sequential wave count.
        let requests = snap.read_requests();
        if requests > 0 {
            OBJECT_STORE_REQUESTS_TOTAL
                .with_label_values(&[t_id])
                .inc_by(requests as f64);
        }
        if !snap.vector_accesses.is_empty()
            && let Some(waves) = snap.waves
        {
            crate::metrics::operational_metrics::PAX_QUERY_WAVES.observe(waves as f64);
        }
        // TD-RDSTRAT-8 PR-C1: persisted-IVF coarse-probe physical meters (neutral
        // counts; AnvaiOps applies the $/rate-card). Region-A/B bytes are the
        // tier-split physical read cost the probe paid.
        for (counter, value) in [
            (&*IVF_CELLS_TOTAL, snap.ivf_cells_total),
            (&*IVF_CELLS_PROBED_TOTAL, snap.ivf_cells_probed),
            (&*IVF_PROBED_ROWS_TOTAL, snap.ivf_probed_rows),
            (&*IVF_REGION_A_BYTES_TOTAL, snap.ivf_region_a_bytes),
            (&*IVF_REGION_B_BYTES_TOTAL, snap.ivf_region_b_bytes),
            (&*IVF_FETCH_ROUNDS_TOTAL, snap.ivf_fetch_rounds),
            (
                &*IVF_WHOLE_REGION_FALLBACK_TOTAL,
                snap.ivf_whole_region_fallback,
            ),
            // TD-RDSTRAT-12 §2: bounded-concurrent wave meters. Wave sites bridge
            // the FS single-slot metrics into this scope additively (see
            // `drain_and_forward_read_ranges_metrics` callers); the closure drain
            // below is the catch-all for waves no site bridged.
            (
                &*READ_RANGES_FETCH_ROUNDS_TOTAL,
                snap.read_ranges_fetch_rounds,
            ),
        ] {
            if value > 0 {
                counter.with_label_values(&[t_id]).inc_by(value as f64);
            }
        }
        // GaugeVec cannot join the CounterVec tuple above: set it separately and
        // never clobber a previously-observed peak with 0.
        if snap.read_ranges_max_inflight > 0 {
            READ_RANGES_MAX_INFLIGHT
                .with_label_values(&[t_id])
                .set(snap.read_ranges_max_inflight as f64);
        }
        // Production caller for the FS→Prometheus drain (TD-RDSTRAT-12 §2): the
        // FS single-slot metrics are `take()`-drained exactly once — either a
        // wave site already forwarded them into this scope via
        // `drain_and_forward_read_ranges_metrics` (which `take()`s first), or
        // this catch-all emits them here. Mutually exclusive per metric
        // instance by `take()` semantics.
        drain_and_emit_read_ranges_metrics(t_id);
    })));
}

/// Record outgress bytes for one tenant — the single convergent **KOU** meter
/// (Dimension 2). `direction` is `"read"` (object store → compute) or `"result"`
/// (compute → client). It does two things at once so every boundary records
/// identically:
///
/// 1. Increments the per-tenant [`KOU_BYTES_TOTAL`] counter (the chargeback
///    source of truth, attributed by `(tenant, kou_locality, direction)`).
/// 2. When the locality is *chargeable* (cross-region/-cloud/on-prem/internet,
///    never the free local / cross-AZ / same-region path), folds the bytes into
///    the active per-query I/O trace's egress term — the quantity the route cost
///    model's egress weight minimizes. Outside a query scope this silently no-ops.
///
/// Free-path bytes are metered (for visibility) but deliberately do NOT enter the
/// cost model, so the egress cost term stays inert on the free path.
/// Record `bytes` written to object storage for one tenant at access `tier` — the
/// write-time half of the cold-tier cost lever (TD-168/TD-173). `tier` is the
/// canonical name from `ObjectAccessTier::as_str` (`hot`/`cool`/`cold`/`archive`).
/// Empty `tenant_id` is attributed to `default` (single-tenant). No-ops on zero.
pub fn record_object_store_write_bytes_by_tier(tenant_id: &str, tier: &str, bytes: u64) {
    if bytes == 0 {
        return;
    }
    let t_id = if tenant_id.is_empty() {
        "default"
    } else {
        tenant_id
    };
    OBJECT_STORE_WRITE_BYTES_BY_TIER
        .with_label_values(&[t_id, tier])
        .inc_by(bytes as f64);
}

pub fn record_kou_bytes(
    tenant_id: Option<&str>,
    locality: KouLocality,
    direction: &str,
    bytes: u64,
) {
    if bytes == 0 {
        return;
    }
    let t_id = tenant_id.unwrap_or("default");
    KOU_BYTES_TOTAL
        .with_label_values(&[t_id, locality.as_str(), direction])
        .inc_by(bytes as f64);
    if locality.is_chargeable_egress() {
        crate::observability::io_trace::record_egress_bytes(bytes);
    }
}

/// Record embedding units for one tenant — the **KEU** (Kilo-Embedding-Unit) meter
/// (Dimension 5). `provider` identifies the embedding service (e.g. "openai",
/// "victor", "huggingface"); `model` names the model; `operation` is the operation
/// type (e.g. "embed", "embed_batch"). It does two things at once:
///
/// 1. Increments the per-tenant [`KEU_UNITS_TOTAL`] counter in **kilo-units**
///    (tokens / 1000.0) — the chargeback source of truth, attributed by
///    `(tenant, provider, model, operation)`.
/// 2. Folds the raw token counts into the active per-query I/O trace's embedding
///    term — the quantity the cost model consumes. Outside a query scope this
///    silently no-ops.
///
/// KEU is a TAM surface: metered per-tenant, priced downstream in AnvaiOps.
/// This is a neutral quantity at this layer (no in-engine pricing).
pub fn record_keu_units(
    tenant_id: Option<&str>,
    provider: &str,
    model: &str,
    operation: &str,
    input_tokens: u64,
    output_tokens: u64,
) {
    let total_tokens = input_tokens.saturating_add(output_tokens);
    if total_tokens == 0 {
        return;
    }

    let t_id = tenant_id.unwrap_or("default");
    // Record in kilo-units (KEU = Kilo-Embedding-Units)
    KEU_UNITS_TOTAL
        .with_label_values(&[t_id, provider, model, operation])
        .inc_by(total_tokens as f64 / 1000.0);

    // Also fold into the per-query trace (inside a scope, silently no-ops outside)
    crate::observability::io_trace::record_embedding(input_tokens, output_tokens);
}

/// Publish a per-tenant cache stats snapshot (e.g. from the footer cache's
/// `tenant_stats()`) onto the `proximadb_cache_tenant` gauge. `cache` names the
/// cache (e.g. "footer").
pub fn record_cache_tenant_stats(cache: &str, stats: &[proximadb_cache::TenantCacheStat]) {
    for s in stats {
        let g = |metric: &str, v: f64| {
            CACHE_TENANT_STATS
                .with_label_values(&[s.tenant.as_str(), cache, metric])
                .set(v);
        };
        g("bytes", s.bytes as f64);
        g("hits", s.hits as f64);
        g("misses", s.misses as f64);
        g("evictions", s.evictions as f64);
        g("hit_ratio", s.hit_ratio);
    }
}

/// TD-RDSTRAT-12: Record bounded-concurrent `FileSystem::read_ranges` metrics.
///
/// Called by the SST engine (or other callers with tenant context) after draining
/// the thread-local `read_ranges` metrics via `drain_read_ranges_metrics()`.
/// Emits to the per-tenant Prometheus counters.
pub fn record_read_ranges_metrics(tenant_id: &str, fetch_rounds: u64, max_inflight: u64) {
    READ_RANGES_FETCH_ROUNDS_TOTAL
        .with_label_values(&[tenant_id])
        .inc_by(fetch_rounds as f64);
    READ_RANGES_MAX_INFLIGHT
        .with_label_values(&[tenant_id])
        .set(max_inflight as f64);
}

/// TD-RDSTRAT-12: Drain and emit bounded-concurrent `FileSystem::read_ranges` metrics.
///
/// Convenience function that combines `drain_read_ranges_metrics()` (from
/// `proximadb_storage_filesystem_types`) and `record_read_ranges_metrics()`.
/// Call this after operations that use `FileSystem::read_ranges` with parallelism
/// enabled (`PROXIMADB_FS_READ_RANGES_PARALLEL > 1`). No-op if no metrics were
/// recorded (e.g., sequential reads or no `read_ranges` calls since last drain).
pub fn drain_and_emit_read_ranges_metrics(tenant_id: &str) {
    use proximadb_storage_filesystem_types::drain_read_ranges_metrics;
    if let Some(metrics) = drain_read_ranges_metrics() {
        record_read_ranges_metrics(tenant_id, metrics.fetch_rounds, metrics.max_inflight);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observability::io_trace;

    /// ADR-030 KSU: the storage snapshot groups collections by owner (tenant) and
    /// sets the per-tenant resident-bytes level gauge; absent owner → "default".
    #[test]
    fn storage_snapshot_groups_by_tenant_and_sets_gauge() {
        use crate::proto::proximadb_v1::{Collection, CollectionConfig, CollectionStats};
        let mk = |owner: Option<&str>, bytes: i64| Collection {
            config: Some(CollectionConfig {
                owner: owner.map(str::to_string),
                ..Default::default()
            }),
            stats: Some(CollectionStats {
                data_size_bytes: bytes,
                ..Default::default()
            }),
            ..Default::default()
        };
        // Unique tenant labels so the global gauge assertion is isolated.
        let cols = vec![
            mk(Some("ksu_acme"), 100),
            mk(Some("ksu_acme"), 200),
            mk(Some("ksu_globex"), 50),
            mk(None, 7), // unowned → "ksu_default" bucket is "default"
        ];
        record_storage_snapshot(&cols);
        assert_eq!(
            STORAGE_BYTES_SECONDS
                .with_label_values(&["ksu_acme", "sst"])
                .get(),
            300.0,
            "acme's two collections must sum"
        );
        assert_eq!(
            STORAGE_BYTES_SECONDS
                .with_label_values(&["ksu_globex", "sst"])
                .get(),
            50.0
        );
        assert_eq!(
            STORAGE_BYTES_SECONDS
                .with_label_values(&["default", "sst"])
                .get(),
            7.0,
            "unowned bytes attributed to default (fail-closed)"
        );
    }

    /// TD-WLP-8: the Churn working-set gauge round-trips per (tenant, collection)
    /// and attributes an absent tenant to "default" (fail-closed).
    #[test]
    fn churn_working_set_gauge_round_trips() {
        record_churn_working_set_bytes(Some("ksu_wlp8"), "col_hot", 4096);
        assert_eq!(
            CHURN_WORKING_SET_BYTES
                .with_label_values(&["ksu_wlp8", "col_hot"])
                .get(),
            4096.0
        );
        // A later sample overwrites the level (gauge, not counter).
        record_churn_working_set_bytes(Some("ksu_wlp8"), "col_hot", 2048);
        assert_eq!(
            CHURN_WORKING_SET_BYTES
                .with_label_values(&["ksu_wlp8", "col_hot"])
                .get(),
            2048.0
        );
        // Absent tenant → "default".
        record_churn_working_set_bytes(None, "col_unowned", 512);
        assert_eq!(
            CHURN_WORKING_SET_BYTES
                .with_label_values(&["default", "col_unowned"])
                .get(),
            512.0
        );
    }

    #[test]
    fn provider_inferred_from_url_scheme() {
        assert_eq!(CloudProvider::from_url_scheme("s3"), CloudProvider::Aws);
        assert_eq!(CloudProvider::from_url_scheme("gs"), CloudProvider::Gcp);
        assert_eq!(CloudProvider::from_url_scheme("gcs"), CloudProvider::Gcp);
        assert_eq!(CloudProvider::from_url_scheme("abfs"), CloudProvider::Azure);
        assert_eq!(CloudProvider::from_url_scheme("adls"), CloudProvider::Azure);
        assert_eq!(CloudProvider::from_url_scheme("file"), CloudProvider::Local);
        assert_eq!(CloudProvider::from_url_scheme(""), CloudProvider::Local);
    }

    #[test]
    fn kou_locality_strings_and_chargeability() {
        assert_eq!(KouLocality::Local.as_str(), "local");
        assert_eq!(KouLocality::CrossAz.as_str(), "cross_az");
        assert_eq!(KouLocality::CrossRegion.as_str(), "cross_region");
        assert_eq!(KouLocality::CrossCloud.as_str(), "cross_cloud");
        assert_eq!(KouLocality::OnPrem.as_str(), "on_prem");
        assert_eq!(KouLocality::Internet.as_str(), "internet");
        assert_eq!(KouLocality::Unknown.as_str(), "unknown");
        // Local / cross-AZ object-store access is FREE on every cloud; Unknown is
        // surfaced, not charged. Only genuine cross-boundary movement is chargeable.
        assert!(!KouLocality::Local.is_chargeable_egress());
        assert!(!KouLocality::CrossAz.is_chargeable_egress());
        assert!(!KouLocality::Unknown.is_chargeable_egress());
        assert!(KouLocality::CrossRegion.is_chargeable_egress());
        assert!(KouLocality::CrossCloud.is_chargeable_egress());
        assert!(KouLocality::OnPrem.is_chargeable_egress());
        assert!(KouLocality::Internet.is_chargeable_egress());
        assert_eq!(KouLocality::default(), KouLocality::Local);
    }

    #[test]
    fn derive_read_is_region_granular_per_cloud() {
        let ctx = |p, c: &str, s: &str| PlacementContext {
            store_provider: p,
            compute_provider: p,
            compute_region: c.into(),
            store_region: s.into(),
        };
        // Same region → free (even if zones differ — object store is regional).
        assert_eq!(
            KouLocality::derive_read(&ctx(CloudProvider::Aws, "us-east-1", "us-east-1")),
            KouLocality::Local
        );
        // Different region → chargeable cross-region.
        assert_eq!(
            KouLocality::derive_read(&ctx(CloudProvider::Azure, "eastus", "westus")),
            KouLocality::CrossRegion
        );
        // GCS multi-region bucket "us" containing the compute region → free.
        assert_eq!(
            KouLocality::derive_read(&ctx(CloudProvider::Gcp, "us-central1", "us")),
            KouLocality::Local
        );
        assert_eq!(
            KouLocality::derive_read(&ctx(CloudProvider::Gcp, "europe-west1", "us")),
            KouLocality::CrossRegion
        );
        // Compute in a different cloud than the store → cross-cloud read.
        assert_eq!(
            KouLocality::derive_read(&PlacementContext {
                store_provider: CloudProvider::Gcp,
                compute_provider: CloudProvider::Aws,
                compute_region: "us-east-1".into(),
                store_region: "us-central1".into(),
            }),
            KouLocality::CrossCloud
        );
        // Unknown compute region or local provider → free default (never charge).
        assert_eq!(
            KouLocality::derive_read(&ctx(CloudProvider::Aws, "", "us-east-1")),
            KouLocality::Local
        );
        assert_eq!(
            KouLocality::derive_read(&ctx(CloudProvider::Local, "us-east-1", "us-east-1")),
            KouLocality::Local
        );
    }

    #[test]
    fn override_parse_rejects_unknown() {
        assert_eq!(
            KouLocality::parse("cross_region"),
            Some(KouLocality::CrossRegion)
        );
        assert_eq!(
            KouLocality::parse("CROSS-REGION"),
            Some(KouLocality::CrossRegion)
        );
        assert_eq!(
            KouLocality::parse("cross_cloud"),
            Some(KouLocality::CrossCloud)
        );
        assert_eq!(KouLocality::parse("on_prem"), Some(KouLocality::OnPrem));
        assert_eq!(KouLocality::parse("internet"), Some(KouLocality::Internet));
        // Unknown → None so the caller keeps the derived value (no silent mischarge).
        assert_eq!(KouLocality::parse("nonsense"), None);
        assert_eq!(KouLocality::parse(""), None);
    }

    #[tokio::test]
    async fn chargeable_egress_feeds_the_query_trace_but_free_path_does_not() {
        // Inside a query scope, only chargeable localities contribute to the
        // per-query egress term the cost model consumes.
        let snap = io_trace::scope(async {
            record_kou_bytes(Some("t"), KouLocality::CrossRegion, "read", 1_000);
            record_kou_bytes(Some("t"), KouLocality::Internet, "result", 2_000);
            record_kou_bytes(Some("t"), KouLocality::CrossAz, "read", 9_000); // free → not traced
            record_kou_bytes(Some("t"), KouLocality::Local, "read", 5_000); // free → not traced
            record_kou_bytes(Some("t"), KouLocality::Unknown, "result", 7_000); // surfaced, not charged
            record_kou_bytes(Some("t"), KouLocality::CrossRegion, "read", 0); // zero → ignored
            io_trace::snapshot()
        })
        .await
        .expect("snapshot inside scope");
        assert_eq!(snap.egress_bytes, 3_000);
    }

    #[test]
    fn scope_map_classifies_by_longest_prefix_with_safe_defaults() {
        // Terraform-style rules: compute subnet/AZ → local; rest of VPC → cross_az;
        // a declared on-prem range → on_prem. Longest-prefix wins.
        let map = NetworkScopeMap::from_json(
            r#"[
              {"cidr":"10.0.1.0/24","locality":"local"},
              {"cidr":"10.0.0.0/16","locality":"cross_az"},
              {"cidr":"203.0.113.0/24","locality":"on_prem"}
            ]"#,
        );
        assert_eq!(
            map.classify("10.0.1.5".parse().unwrap()),
            KouLocality::Local
        ); // /24 beats /16
        assert_eq!(
            map.classify("10.0.2.5".parse().unwrap()),
            KouLocality::CrossAz
        ); // /16
        assert_eq!(
            map.classify("203.0.113.7".parse().unwrap()),
            KouLocality::OnPrem
        );
        // Unmatched public IP → Internet; unmatched private IP → Unknown (surfaced).
        assert_eq!(
            map.classify("8.8.8.8".parse().unwrap()),
            KouLocality::Internet
        );
        assert_eq!(
            map.classify("192.168.9.9".parse().unwrap()),
            KouLocality::Unknown
        );
    }

    #[tokio::test]
    async fn billing_observer_emits_physical_object_store_meters() {
        // TD-IOTRACE-2: the always-on billing observer must emit the accurate
        // per-query range_gets + bytes_read as per-tenant neutral counters (the
        // physical part of the two-part KRU rate). A fresh tenant id keeps the
        // CounterVec children at their 0 baseline so the assertion is exact.
        install_billing_observer();
        let tenant = "test-iotrace2-physical-meters";
        io_trace::instrument(Some(tenant.to_string()), "test.iotrace2", async {
            io_trace::record_range_gets(7);
            io_trace::record_bytes_read(4096);
            io_trace::record_ivf_coarse_probe(64, 8, 4096, 3, false);
            io_trace::record_ivf_coarse_probe(0, 0, 0, 0, true); // armed-but-missed fallback
            io_trace::record_pax_region_bytes(200_000, 800_000);
        })
        .await;
        assert_eq!(
            OBJECT_STORE_GETS_TOTAL.with_label_values(&[tenant]).get(),
            7.0
        );
        assert_eq!(
            OBJECT_STORE_BYTES_READ_TOTAL
                .with_label_values(&[tenant])
                .get(),
            4096.0
        );
        // TD-RDSTRAT-8 PR-C1: IVF coarse-probe physical meters.
        assert_eq!(IVF_CELLS_TOTAL.with_label_values(&[tenant]).get(), 64.0);
        assert_eq!(
            IVF_CELLS_PROBED_TOTAL.with_label_values(&[tenant]).get(),
            8.0
        );
        assert_eq!(
            IVF_PROBED_ROWS_TOTAL.with_label_values(&[tenant]).get(),
            4096.0
        );
        assert_eq!(
            IVF_REGION_A_BYTES_TOTAL.with_label_values(&[tenant]).get(),
            200_000.0
        );
        assert_eq!(
            IVF_REGION_B_BYTES_TOTAL.with_label_values(&[tenant]).get(),
            800_000.0
        );
        assert_eq!(
            IVF_FETCH_ROUNDS_TOTAL.with_label_values(&[tenant]).get(),
            3.0
        );
        assert_eq!(
            IVF_WHOLE_REGION_FALLBACK_TOTAL
                .with_label_values(&[tenant])
                .get(),
            1.0
        );
        // Restore global observer state for the rest of the suite.
        io_trace::set_billing_observer(None);
    }

    #[test]
    fn empty_scope_map_is_safe_unknown_or_internet() {
        let empty = NetworkScopeMap::default();
        assert_eq!(
            empty.classify("10.0.0.1".parse().unwrap()),
            KouLocality::Unknown
        ); // private → surfaced
        assert_eq!(
            empty.classify("1.1.1.1".parse().unwrap()),
            KouLocality::Internet
        ); // public → notice
        // Malformed JSON / rules degrade to empty, never panic.
        assert_eq!(
            NetworkScopeMap::from_json("not json").classify("10.0.0.1".parse().unwrap()),
            KouLocality::Unknown
        );
        let partial = NetworkScopeMap::from_json(
            r#"[{"cidr":"bogus","locality":"local"},{"cidr":"10.0.0.0/8","locality":"cross_region"}]"#,
        );
        assert_eq!(
            partial.classify("10.1.2.3".parse().unwrap()),
            KouLocality::CrossRegion
        );
    }

    #[test]
    fn placement_from_env_infers_provider_and_defaults_safe() {
        // Provider inferred from the storage URL scheme; with no region env the
        // context derives the free Local locality (safe default, no env mutation).
        let p = placement_from_env("s3://my-bucket/data");
        assert_eq!(p.store_provider, CloudProvider::Aws);
        assert_eq!(p.compute_provider, CloudProvider::Aws);
        assert_eq!(
            placement_from_env("gs://b/x").store_provider,
            CloudProvider::Gcp
        );
        assert_eq!(
            placement_from_env("abfs://b/x").store_provider,
            CloudProvider::Azure
        );
        let local = placement_from_env("file://./data");
        assert_eq!(local.store_provider, CloudProvider::Local);
        // file:// + no compute region → derive Local (free), regardless of env.
        assert_eq!(KouLocality::derive_read(&local), KouLocality::Local);
    }

    #[tokio::test]
    async fn record_kou_outside_scope_is_a_noop() {
        // No active trace: the counter still records, but nothing panics and the
        // trace stays empty (helper silently no-ops the egress term).
        record_kou_bytes(Some("t"), KouLocality::CrossRegion, "read", 1_234);
        assert!(io_trace::snapshot().is_none());
    }

    #[test]
    fn edge_context_treats_in_process_and_loopback_callers_as_free() {
        // Unspecified (embedded / unset peer) and loopback callers are never a
        // chargeable remote — they classify Local so result-egress is never
        // mis-charged for an in-process or same-host client, regardless of the
        // scope map. This is the guard that keeps the meter inert by default.
        let unspec = EdgePolicyContext::classify("0.0.0.0".parse().unwrap());
        assert_eq!(unspec.kou_locality, KouLocality::Local);
        assert!(!unspec.kou_locality.is_chargeable_egress());
        assert_eq!(
            EdgePolicyContext::classify("::".parse().unwrap()).kou_locality,
            KouLocality::Local
        );
        assert_eq!(
            EdgePolicyContext::classify("127.0.0.1".parse().unwrap()).kou_locality,
            KouLocality::Local
        );
        assert_eq!(
            EdgePolicyContext::classify("::1".parse().unwrap()).kou_locality,
            KouLocality::Local
        );

        // Recording result-egress on the free path mutates no trace.
        unspec.record_result_egress(Some("t"), 4096);
        assert!(io_trace::snapshot().is_none());
    }

    #[test]
    fn shape_policy_compresses_only_for_chargeable_clients() {
        // Lossless byte-minimization is on for far/chargeable clients (egress
        // dominates), off on the free path (save CPU) and for Unknown (never act
        // on an unclassified client).
        for loc in [
            KouLocality::CrossRegion,
            KouLocality::CrossCloud,
            KouLocality::OnPrem,
            KouLocality::Internet,
        ] {
            assert!(ResultShapePolicy::for_locality(loc).compress, "{loc:?}");
        }
        for loc in [
            KouLocality::Local,
            KouLocality::CrossAz,
            KouLocality::Unknown,
        ] {
            assert!(!ResultShapePolicy::for_locality(loc).compress, "{loc:?}");
        }
        // EdgePolicyContext exposes the same decision for a loopback (free) caller.
        assert!(
            !EdgePolicyContext::classify("127.0.0.1".parse().unwrap())
                .shape_policy()
                .compress
        );
    }

    #[tokio::test]
    async fn record_keu_feeds_query_trace_and_prometheus() {
        // Inside a query scope, KEU recording contributes to the embedding trace
        // and increments the Prometheus counter in kilo-units.
        use crate::observability::io_trace;

        let snap = io_trace::scope(async {
            record_keu_units(
                Some("tenant-a"),
                "openai",
                "text-embedding-3-small",
                "embed",
                1000,
                2000,
            );
            record_keu_units(
                Some("tenant-a"),
                "victor",
                "bge-base-en-v1.5",
                "embed_batch",
                500,
                1500,
            );
            record_keu_units(
                Some("tenant-a"),
                "openai",
                "text-embedding-3-small",
                "embed",
                0,
                0,
            ); // zero → ignored
            io_trace::snapshot()
        })
        .await
        .expect("snapshot inside scope");

        // Verify trace captured the embedding operations
        assert_eq!(snap.embedding_calls, 2); // 2 non-zero calls
        assert_eq!(snap.embedding_input_tokens, 1500); // 1000 + 500
        assert_eq!(snap.embedding_output_tokens, 3500); // 2000 + 1500
        assert_eq!(snap.total_embedding_tokens(), 5000);
    }

    #[tokio::test]
    async fn record_keu_outside_scope_is_a_noop_for_trace() {
        // No active trace: the counter still records, but nothing panics and the
        // trace stays empty (helper silently no-ops the trace term).
        record_keu_units(
            Some("t"),
            "openai",
            "text-embedding-3-small",
            "embed",
            1000,
            2000,
        );
        assert!(io_trace::snapshot().is_none());
    }

    #[test]
    fn record_keu_with_default_tenant() {
        // Verify default tenant handling works
        record_keu_units(None, "victor", "bge-base-en-v1.5", "embed", 500, 1000);
        // Counter should have been incremented (we can't easily read it back, but this should not panic)
    }

    #[tokio::test]
    async fn keu_tracks_independently_from_kou() {
        // KEU and KOU are separate meters — they should not interfere
        use crate::observability::io_trace;

        let snap = io_trace::scope(async {
            record_keu_units(Some("t"), "openai", "model", "embed", 100, 200);
            record_kou_bytes(Some("t"), KouLocality::CrossRegion, "read", 1000);
            io_trace::snapshot()
        })
        .await
        .expect("snapshot inside scope");

        assert_eq!(snap.embedding_calls, 1);
        assert_eq!(snap.embedding_input_tokens, 100);
        assert_eq!(snap.embedding_output_tokens, 200);
        assert_eq!(snap.egress_bytes, 1000);
    }

    /// TD-RDSTRAT-12 §2: the billing observer must surface the wave metrics —
    /// `read_ranges_fetch_rounds` as a counter and `read_ranges_max_inflight`
    /// as a guarded gauge — at scope close. Teeth: breaks if the tuple/gauge
    /// wiring regresses or `is_empty()` drops wave-only traces again.
    #[tokio::test]
    async fn billing_observer_emits_read_ranges_wave_metrics() {
        use crate::observability::io_trace;

        install_billing_observer();
        let tenant = "rdstrat12_wave_obs";

        io_trace::instrument(Some(tenant.to_string()), "test.read_ranges_wave", async {
            io_trace::record_read_ranges(3, 8);
        })
        .await;

        assert_eq!(
            READ_RANGES_FETCH_ROUNDS_TOTAL
                .with_label_values(&[tenant])
                .get(),
            3.0,
            "wave fetch-rounds must reach the per-tenant counter at scope close"
        );
        assert_eq!(
            READ_RANGES_MAX_INFLIGHT.with_label_values(&[tenant]).get(),
            8.0,
            "wave peak-inflight must reach the per-tenant gauge at scope close"
        );
    }
}
