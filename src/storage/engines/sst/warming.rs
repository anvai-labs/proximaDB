// Copyright (C) 2026 ProximaDB
// SPDX-License-Identifier: Apache-2.0

//! TD-CACHE-1 S2 — restart cache warming: shutdown warm-set manifests +
//! boot replay, defusing the post-eviction cold herd at its source.
//!
//! Co-design posture (2026-07-24 review): warming is **demand-proven or
//! contract-driven, never speculative**. A tenant gets warmed only when a
//! `warm_cache.json` manifest exists for them (i.e. they were measurably hot
//! before shutdown); idle/free collections stay lazy first-touch, so Spot
//! churn never generates unnecessary object-store traffic. Replay runs inside
//! the tenant's `io_trace::instrument` scope so (a) replayed entries key under
//! the REAL tenant (matching later lookups) and (b) replay GETs are metered
//! to the tenant they benefit.
//!
//! The manifest is advisory and mixed-safe: absent/corrupt/stale ⇒ cold start;
//! entries referencing compacted-away segments fail their ranged read and are
//! skipped.

use std::sync::Arc;

use crate::storage::engines::sst::survivor_range_cache::WarmKey;
use crate::storage::persistence::filesystem::FilesystemFactory;
use crate::storage::trait_components::path_resolver::DrPathBuilder;

/// Manifest schema version (bump on incompatible change; readers skip
/// unknown versions — advisory artifact, never load-bearing).
const WARM_MANIFEST_VERSION: u32 = 1;
/// Entries older than this are ignored (the working set has likely moved).
const WARM_MANIFEST_MAX_AGE_SECS: i64 = 24 * 3600;
/// Top-K hot ranges persisted per tenant (~100 B/entry ⇒ ≤ ~200 KB).
const WARM_MANIFEST_TOP_K: usize = 2000;

#[derive(serde::Serialize, serde::Deserialize)]
struct WarmManifest {
    version: u32,
    created_unix: i64,
    entries: Vec<WarmKey>,
}

/// Per-tenant manifest object URL under the DrPath manifests subprefix —
/// fail-closed on an invalid tenant id (same recipe as
/// `metrics::metering_writer::ksu_snapshot_object_url`).
fn manifest_url(base_url: &str, tenant_id: &str) -> anyhow::Result<String> {
    // TD-CACHE-5: build_tenant_system returns a DrTenantSystemPath, whose only
    // subprefixes are the tenant-system ones (_metering/_trace/_cache). The warm
    // manifest is a tenant-system object, so it lives under cache_subprefix()
    // (<tenant_root>_cache/) — NOT the collection-scoped manifests_subprefix(),
    // which previously rendered the malformed `data/default///manifests/…`.
    let resolved = DrPathBuilder::build_tenant_system(None, tenant_id)
        .map_err(|e| anyhow::anyhow!("invalid tenant id for warm manifest: {e}"))?;
    let prefix = resolved.cache_subprefix();
    let base = base_url.trim_end_matches('/');
    Ok(format!("{base}/{prefix}warm_cache.json"))
}

/// Shutdown-side: persist each hot tenant's warm set. Best-effort — errors are
/// logged and skipped (shutdown must not block on the object store). Returns
/// the number of manifests written.
pub async fn emit_warm_manifests(factory: &Arc<FilesystemFactory>, base_url: &str) -> usize {
    let Some((_invariants, survivor)) = crate::storage::engines::sst::core::get_warm_tier_caches()
    else {
        return 0;
    };
    let now = chrono::Utc::now().timestamp();
    let mut written = 0usize;
    for (tenant, entries) in survivor.warm_set(WARM_MANIFEST_TOP_K) {
        if entries.is_empty() {
            continue;
        }
        let manifest = WarmManifest {
            version: WARM_MANIFEST_VERSION,
            created_unix: now,
            entries,
        };
        let (Ok(url), Ok(bytes)) = (
            manifest_url(base_url, &tenant),
            serde_json::to_vec(&manifest),
        ) else {
            continue;
        };
        let options = proximadb_storage_filesystem_types::FileOptions {
            overwrite: true,
            create_dirs: true,
            ..Default::default()
        };
        match factory.write(&url, &bytes, Some(options)).await {
            Ok(()) => {
                written += 1;
                tracing::info!(
                    tenant,
                    entries = manifest.entries.len(),
                    "🔥 warm manifest persisted"
                );
            }
            Err(e) => tracing::warn!(tenant, "warm manifest write failed (skipped): {e}"),
        }
    }
    written
}

/// Boot-side: replay one tenant's manifest, if present and fresh. Runs the
/// replays inside the tenant's `io_trace` scope (tenant-correct cache keys +
/// metering). Also prefetches the referenced segments' CONTROL invariants
/// (header/footer/A0) so first queries skip their 3 control GETs. Returns
/// (ranges_loaded, segments_prefilled).
pub async fn replay_tenant(
    factory: &Arc<FilesystemFactory>,
    base_url: &str,
    tenant_id: &str,
) -> (usize, usize) {
    let Some((invariants, survivor)) = crate::storage::engines::sst::core::get_warm_tier_caches()
    else {
        return (0, 0);
    };
    let Ok(url) = manifest_url(base_url, tenant_id) else {
        return (0, 0);
    };
    let Ok(bytes) = factory.read(&url).await else {
        return (0, 0); // no manifest = tenant wasn't hot = nothing to warm
    };
    let Ok(manifest) = serde_json::from_slice::<WarmManifest>(&bytes) else {
        tracing::warn!(tenant_id, "warm manifest unreadable — cold start");
        return (0, 0);
    };
    if manifest.version != WARM_MANIFEST_VERSION
        || chrono::Utc::now().timestamp() - manifest.created_unix > WARM_MANIFEST_MAX_AGE_SECS
    {
        return (0, 0); // stale/unknown — the working set has moved on
    }

    let tenant_owned = tenant_id.to_string();
    crate::observability::io_trace::instrument(Some(tenant_owned), "warm.replay", async {
        let mut loaded = 0usize;
        let mut prefilled = 0usize;
        let mut seen_segments = std::collections::HashSet::new();
        for entry in &manifest.entries {
            let Ok(fs) = factory.get_filesystem(&entry.p) else {
                continue;
            };
            // Control-invariants prefill once per referenced segment.
            // `Ok(false)` = already warm (recovery/compaction scans can win
            // the race) or a legacy segment — both benign.
            if seen_segments.insert(entry.p.clone()) {
                match crate::storage::engines::sst::segment_format::prefetch_segment_invariants(
                    fs.as_ref(),
                    &entry.p,
                    &invariants,
                )
                .await
                {
                    Ok(true) => prefilled += 1,
                    Ok(false) => {
                        tracing::debug!(segment = %entry.p, "invariants prefill skipped (warm/legacy)")
                    }
                    Err(e) => {
                        tracing::warn!(segment = %entry.p, "invariants prefill failed: {e}")
                    }
                }
            }
            let path = entry.p.clone();
            let (off, len) = (entry.o, entry.l);
            let fs_for_load = fs.clone();
            if survivor
                .replay_entry(entry, || async move {
                    fs_for_load.read_range(&path, off, len).await
                })
                .await
            {
                loaded += 1;
            }
        }
        (loaded, prefilled)
    })
    .await
}

/// TD-CACHE-1 S3: order tenants for replay by cache-tier entitlement —
/// highest `TierSpec.weight` first (ties: larger `floor_frac`, then tenant id
/// for determinism). A contract-pinned (paid hot-tier) tenant's warmup is an
/// entitled cost, so it must not queue behind free-tier replays on a shared
/// boot. Unknown tenants resolve to `default_tier`; no policy ⇒ plain
/// deterministic name order.
pub fn tier_ordered(
    mut tenants: Vec<String>,
    policy: Option<&proximadb_cache::TierPolicy>,
    tier_of: &dyn Fn(&str) -> Option<String>,
) -> Vec<String> {
    let rank = |tenant: &str| -> (u32, f64) {
        let Some(policy) = policy else {
            return (0, 0.0);
        };
        let tier = tier_of(tenant).unwrap_or_else(|| policy.default_tier.clone());
        policy
            .tiers
            .get(&tier)
            .or_else(|| policy.tiers.get(&policy.default_tier))
            .map(|spec| (spec.weight, spec.floor_frac))
            .unwrap_or((0, 0.0))
    };
    tenants.sort_by(|a, b| {
        let (wa, fa) = rank(a);
        let (wb, fb) = rank(b);
        wb.cmp(&wa).then(fb.total_cmp(&fa)).then_with(|| a.cmp(b))
    });
    tenants
}

/// Boot warm task body: replay every tenant that has a manifest, hot tiers
/// first (S3). Tenants are derived from the catalog's collections; tenants
/// without manifests cost one cheap negative read each. Gated by
/// `PROXIMADB_CACHE_PREFILL` (unset/1 = on).
pub async fn boot_warm(factory: Arc<FilesystemFactory>, base_url: String, tenants: Vec<String>) {
    if std::env::var("PROXIMADB_CACHE_PREFILL").ok().as_deref() == Some("0") {
        tracing::info!("cache warming disabled (PROXIMADB_CACHE_PREFILL=0)");
        return;
    }
    // Same operator policy the survivor cache's admission floors use.
    let policy = std::env::var("PROXIMADB_CACHE_TIERS_PATH")
        .ok()
        .and_then(|path| std::fs::read_to_string(&path).ok())
        .and_then(|json| proximadb_cache::TierPolicy::from_json(&json).ok());
    // TD-TENANT-3 S3: canonical-then-raw probe, so boot-warm ordering agrees
    // with the admission floors the survivor cache resolves for the same tenant.
    let tenants = tier_ordered(tenants, policy.as_ref(), &|t| {
        let Some(policy) = policy.as_ref() else {
            return crate::services::record_store::tenant_tier(t);
        };
        crate::services::record_store::tenant_tier_policy_key(t, |key| policy.has_tier(key))
    });
    let mut total_ranges = 0usize;
    let mut total_segments = 0usize;
    for tenant in &tenants {
        let (r, s) = replay_tenant(&factory, &base_url, tenant).await;
        total_ranges += r;
        total_segments += s;
    }
    if total_ranges > 0 || total_segments > 0 {
        tracing::info!(
            tenants = tenants.len(),
            ranges = total_ranges,
            segments = total_segments,
            "🔥 restart cache warming complete"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TD-CACHE-7: the warm-manifest URL for a CLOUD base must be well-formed —
    /// container preserved, no `//` run, no legacy `…/manifests/…` malformation.
    /// This is the always-on guard against the recurring container-strip /
    /// double-slash class (TD-FLUSH-6, TD-CACHE-5) that repeatedly silently
    /// disabled restart warming on object storage.
    #[test]
    fn manifest_url_is_well_formed_on_cloud_bases() {
        for base in [
            "azure://benchbucket/proximadb/collections",
            "azure://benchbucket/proximadb/collections/", // trailing slash
            "s3://mybucket/data",
            "adls://container/root",
        ] {
            let url = manifest_url(base, "tenant-b").expect("manifest url");
            let scheme_end = url.find("://").unwrap() + 3;
            let after_scheme = &url[scheme_end..];
            assert!(
                !after_scheme.contains("//"),
                "no `//` run after the scheme in {url}"
            );
            assert!(
                !url.contains("/manifests/"),
                "must NOT use the legacy collection-scoped manifests/ prefix: {url}"
            );
            // The container (first path segment after scheme) must survive.
            let container = after_scheme.split('/').next().unwrap();
            assert!(!container.is_empty(), "container present in {url}");
            assert!(
                url.ends_with("warm_cache.json") && url.contains("tenant-b"),
                "tenant-scoped warm_cache.json: {url}"
            );
        }
    }

    /// TD-CACHE-7: full warm-manifest ROUND-TRIP through the FilesystemFactory
    /// against a real object-store backend — the boundary that kept regressing
    /// with ZERO coverage. Emit → write (create_dirs) → read → deserialize.
    /// Gated on a cloud endpoint; run via the cloud-emulator tier or locally:
    ///   PROXIMADB_AZURE_EMULATOR=1 AZURITE_BLOB_STORAGE_URL=http://127.0.0.1:11000 \
    ///   AZURE_STORAGE_ACCOUNT=devstoreaccount1 \
    ///   PROXIMADB_WARM_TEST_BASE=azure://benchbucket/warmtest \
    ///   cargo test -p proximadb --features azure warm_manifest_round_trips -- --ignored --nocapture
    #[tokio::test]
    #[ignore = "requires a cloud object-store endpoint (PROXIMADB_WARM_TEST_BASE + Azurite/MinIO/creds)"]
    async fn warm_manifest_round_trips_on_object_store() {
        // Accept the dedicated var, or slot into the shared cloud-emulator
        // env (the runner exports PROXIMADB_OBJECT_STORE_URL for the whole tier).
        let base = std::env::var("PROXIMADB_WARM_TEST_BASE")
            .or_else(|_| {
                std::env::var("PROXIMADB_OBJECT_STORE_URL").map(|b| format!("{b}/warm-manifest-rt"))
            })
            .expect("set PROXIMADB_WARM_TEST_BASE or PROXIMADB_OBJECT_STORE_URL");
        let factory =
            std::sync::Arc::new(FilesystemFactory::create_default().await.expect("factory"));
        let url = manifest_url(&base, "tenant-a").expect("url");
        let manifest = WarmManifest {
            version: WARM_MANIFEST_VERSION,
            created_unix: 1_700_000_000,
            entries: vec![WarmKey {
                k: 0,
                p: "seg1.pax".to_string(),
                o: 128,
                l: 4096,
            }],
        };
        let bytes = serde_json::to_vec(&manifest).unwrap();
        let options = proximadb_storage_filesystem_types::FileOptions {
            overwrite: true,
            create_dirs: true,
            ..Default::default()
        };
        factory
            .write(&url, &bytes, Some(options))
            .await
            .unwrap_or_else(|e| panic!("write {url} failed (container-strip regression?): {e}"));
        let read = factory
            .read(&url)
            .await
            .unwrap_or_else(|e| panic!("read {url} failed: {e}"));
        let got: WarmManifest = serde_json::from_slice(&read).expect("round-trip deserialize");
        assert_eq!(got.entries.len(), 1);
        assert_eq!(got.entries[0].p, "seg1.pax");
        assert_eq!(got.entries[0].l, 4096);
    }

    /// TD-CACHE-1 S4: notice parsers — Azure fires only on lifecycle-ending
    /// event types; GCP on the TRUE flag; malformed input never fires.
    #[test]
    fn eviction_notice_parsers() {
        assert!(azure_eviction_imminent(
            r#"{"DocumentIncarnation":1,"Events":[{"EventId":"x","EventType":"Preempt","ResourceType":"VirtualMachine","NotBefore":"Mon, 19 Sep 2016 18:29:47 GMT"}]}"#
        ));
        assert!(azure_eviction_imminent(
            r#"{"Events":[{"EventType":"Freeze"},{"EventType":"Terminate"}]}"#
        ));
        assert!(!azure_eviction_imminent(
            r#"{"Events":[{"EventType":"Freeze"}]}"#
        ));
        assert!(!azure_eviction_imminent(r#"{"Events":[]}"#));
        assert!(!azure_eviction_imminent("not json"));
        assert!(gcp_eviction_imminent("TRUE\n"));
        assert!(!gcp_eviction_imminent("FALSE"));
        assert!(!gcp_eviction_imminent(""));
    }

    /// TD-CACHE-1 S3: replay order is tier weight desc, then floor desc, then
    /// tenant id; unknown tenants rank at default tier; no policy = name order.
    #[test]
    fn tier_ordered_puts_hot_tiers_first() {
        let policy = proximadb_cache::TierPolicy::from_json(
            r#"{
                "default_tier": "free",
                "tiers": {
                    "enterprise": {"weight": 8, "floor_frac": 0.5, "ceiling_frac": 1.0},
                    "pro":        {"weight": 4, "floor_frac": 0.2, "ceiling_frac": 0.8},
                    "free":       {"weight": 1, "floor_frac": 0.0, "ceiling_frac": 0.5}
                }
            }"#,
        )
        .unwrap();
        let tier_of = |t: &str| -> Option<String> {
            match t {
                "acme" => Some("enterprise".to_string()),
                "beta_corp" => Some("pro".to_string()),
                _ => None, // unknown → default_tier (free)
            }
        };
        let tenants = vec![
            "zeta".to_string(),
            "beta_corp".to_string(),
            "alpha".to_string(),
            "acme".to_string(),
        ];
        assert_eq!(
            tier_ordered(tenants.clone(), Some(&policy), &tier_of),
            vec!["acme", "beta_corp", "alpha", "zeta"],
            "weight desc, unknown tenants at default tier in name order"
        );
        assert_eq!(
            tier_ordered(tenants, None, &tier_of),
            vec!["acme", "alpha", "beta_corp", "zeta"],
            "no policy = deterministic name order"
        );
    }
}

// ---------------------------------------------------------------------------
// TD-CACHE-1 S4 — eviction-notice fast-drain.
//
// Spot preemption gives an advance notice (Azure Scheduled Events ~30 s
// before Preempt; AWS spot/instance-action 2 min; GCP preempted flag) that
// arrives BEFORE any SIGTERM — and a hard kill can follow without one. The
// watcher polls the cloud's instance-metadata endpoint (opt-in,
// `PROXIMADB_EVICTION_WATCH=azure|aws|gcp`) and on the first notice runs the
// fast-drain: persist the warm-set manifests (S2, idempotent — the SIGTERM
// path re-emits) and flush the memtable, so the replacement node warms even
// when the instance dies ungracefully. Pre-signaling the replacement node is
// control-plane orchestration (commercial, ADR-060 boundary) — not here.
// ---------------------------------------------------------------------------

/// Azure Scheduled Events: eviction-imminent when any event is a
/// Preempt/Terminate/Redeploy (all end this instance's serving life).
pub fn azure_eviction_imminent(scheduled_events_json: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(scheduled_events_json)
        .ok()
        .and_then(|v| v.get("Events").and_then(|e| e.as_array()).cloned())
        .map(|events| {
            events.iter().any(|e| {
                matches!(
                    e.get("EventType").and_then(|t| t.as_str()),
                    Some("Preempt") | Some("Terminate") | Some("Redeploy")
                )
            })
        })
        .unwrap_or(false)
}

/// GCP: the `instance/preempted` metadata value flips to `TRUE`.
pub fn gcp_eviction_imminent(preempted_body: &str) -> bool {
    preempted_body.trim().eq_ignore_ascii_case("true")
}

/// One IMDS probe for the given provider. AWS needs no body parse: the
/// `spot/instance-action` document only EXISTS (HTTP 200) once an
/// interruption is scheduled.
async fn probe_eviction(client: &reqwest::Client, provider: &str) -> Result<bool, String> {
    match provider {
        "azure" => {
            let resp = client
                .get("http://169.254.169.254/metadata/scheduledevents?api-version=2020-07-01")
                .header("Metadata", "true")
                .send()
                .await
                .map_err(|e| e.to_string())?;
            let body = resp.text().await.map_err(|e| e.to_string())?;
            Ok(azure_eviction_imminent(&body))
        }
        "aws" => {
            // Shared IMDSv2 helper — the same session-token flow environment
            // detection uses (TD-SECSCAN-1 §4).
            let resp = crate::core::utils::aws_imds::aws_imdsv2_get(
                client,
                crate::core::utils::aws_imds::AWS_IMDS_BASE,
                "/latest/meta-data/spot/instance-action",
            )
            .await?;
            Ok(resp.status().is_success())
        }
        "gcp" => {
            let resp = client
                .get("http://metadata.google.internal/computeMetadata/v1/instance/preempted")
                .header("Metadata-Flavor", "Google")
                .send()
                .await
                .map_err(|e| e.to_string())?;
            let body = resp.text().await.map_err(|e| e.to_string())?;
            Ok(gcp_eviction_imminent(&body))
        }
        other => Err(format!("unknown eviction-watch provider '{other}'")),
    }
}

/// S4 watch loop. Polls every 5 s; consecutive probe failures back off to
/// 60 s (a non-cloud box with the flag set must not spam). On the first
/// notice: emit warm manifests, then run the injected memtable flush, then
/// return (the normal SIGTERM path re-emits both — this is the safety copy
/// for the hard-kill case).
pub async fn eviction_watch(
    provider: String,
    factory: Arc<FilesystemFactory>,
    base_url: String,
    flush: Arc<dyn Fn() -> futures::future::BoxFuture<'static, ()> + Send + Sync>,
) {
    let Ok(client) = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
    else {
        tracing::warn!("eviction watch disabled: http client init failed");
        return;
    };
    tracing::info!(provider, "👁 eviction-notice watch armed (S4 fast-drain)");
    let mut consecutive_failures = 0u32;
    loop {
        match probe_eviction(&client, &provider).await {
            Ok(true) => break,
            Ok(false) => consecutive_failures = 0,
            Err(e) => {
                consecutive_failures = consecutive_failures.saturating_add(1);
                if consecutive_failures == 5 {
                    tracing::warn!(
                        provider,
                        "eviction watch: 5 consecutive probe failures (not on this \
                         cloud, or IMDS blocked?) — backing off to 60s: {e}"
                    );
                }
            }
        }
        let secs = if consecutive_failures >= 5 { 60 } else { 5 };
        tokio::time::sleep(std::time::Duration::from_secs(secs)).await;
    }
    tracing::warn!(
        provider,
        "⚡ EVICTION NOTICE — fast-drain: manifests + flush"
    );
    let n = emit_warm_manifests(&factory, &base_url).await;
    flush().await;
    tracing::warn!(
        manifests = n,
        "⚡ fast-drain complete — awaiting SIGTERM/kill"
    );
}
