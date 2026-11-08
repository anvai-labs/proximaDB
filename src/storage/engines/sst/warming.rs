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

/// TD-CACHE-11: one replay candidate. The manifest LIVES at the write side's
/// key — the recording scope's `CacheScope::label()` (the numeric stable id,
/// or `shared`) — while the replay must RUN as the tenant it benefits (io_trace
/// tenant name + stable id), so replayed entries key under the same request
/// scope the recorded lookups used and the replay I/O is metered to that
/// tenant. A catalog tenant with no minted stable id has no manifest to find
/// (the write side never keyed manifests by tenant NAME) and is simply not a
/// candidate.
#[derive(Debug, Clone)]
pub struct ReplayTarget {
    /// io_trace tenant NAME for metering/attribution; `None` = shared scope.
    pub io_trace_tenant: Option<String>,
    /// Stable id the replay scope re-binds (must match the recorded scope).
    /// `None` = the shared scope.
    pub stable_id: Option<u64>,
}

impl ReplayTarget {
    /// A named tenant replay: the manifest label IS the stable id's rendering
    /// (`CacheScope::label()` derives from the same id), so the invalid state
    /// of a label disagreeing with the scope is unrepresentable.
    pub fn tenant(name: impl Into<String>, stable_id: u64) -> Self {
        Self {
            io_trace_tenant: Some(name.into()),
            stable_id: Some(stable_id),
        }
    }

    /// The shared scope replay (scope-less io_trace).
    pub fn shared() -> Self {
        Self {
            io_trace_tenant: None,
            stable_id: None,
        }
    }

    /// The directory label the manifest was written under
    /// (`CacheScope::label()` of the recording scope).
    pub fn manifest_label(&self) -> String {
        self.stable_id
            .map(|id| id.to_string())
            .unwrap_or_else(|| "shared".to_string())
    }
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
    target: &ReplayTarget,
) -> (usize, usize) {
    let Some((invariants, survivor)) = crate::storage::engines::sst::core::get_warm_tier_caches()
    else {
        return (0, 0);
    };
    replay_target_with(factory, base_url, target, &invariants, &survivor).await
}

/// Testable core: same as [`replay_tenant`] with the caches passed in (the
/// boot path fetches the engine-armed globals; tests install their own).
pub(crate) async fn replay_target_with(
    factory: &Arc<FilesystemFactory>,
    base_url: &str,
    target: &ReplayTarget,
    invariants: &Arc<crate::storage::engines::sst::segment_format::SegmentInvariantsCache>,
    survivor: &Arc<crate::storage::engines::sst::survivor_range_cache::SurvivorRangeCache>,
) -> (usize, usize) {
    let Ok(url) = manifest_url(base_url, &target.manifest_label()) else {
        return (0, 0);
    };

    // TD-CACHE-11: re-bind the RECORDED request scope so replayed entries land
    // where the tenant-scoped lookups will find them. Shared manifests replay
    // scope-less (Shared is the no-stable-id fallback), tenant manifests under
    // instrument_with_stable_tenant — NEVER by parsing a tenant name as an id
    // (that mis-billing was tried and reverted; see the TD).
    let replay_scope = async {
        // The manifest GET is part of the replay's cost — keep it INSIDE the
        // io_trace scope so the one read that decides whether warming happens
        // at all is metered/attributed like the entry loads (review round-3).
        let Ok(bytes) = factory.read(&url).await else {
            return (0usize, 0usize); // no manifest = tenant wasn't hot = nothing to warm
        };
        let manifest = match serde_json::from_slice::<WarmManifest>(&bytes) {
            Ok(m) => m,
            Err(_) => {
                tracing::warn!(manifest_label = %target.manifest_label(), "warm manifest unreadable — cold start");
                return (0usize, 0usize);
            }
        };
        if manifest.version != WARM_MANIFEST_VERSION
            || chrono::Utc::now().timestamp() - manifest.created_unix > WARM_MANIFEST_MAX_AGE_SECS
        {
            return (0usize, 0usize); // stale/unknown — the working set has moved on
        }
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
                    invariants,
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
    };
    let (loaded, prefilled) = if target.stable_id.is_some() {
        crate::observability::io_trace::instrument_with_stable_tenant(
            target.io_trace_tenant.clone(),
            target.stable_id,
            "warm.replay",
            replay_scope,
        )
        .await
    } else {
        crate::observability::io_trace::instrument(None, "warm.replay", replay_scope).await
    };
    (loaded, prefilled)
}

/// Boot warm task body: replay every tenant that has a manifest, hot tiers
/// first (S3). Tenants are derived from the catalog's collections; tenants
/// without manifests cost one cheap negative read each. Gated by
/// `PROXIMADB_CACHE_PREFILL` (unset/1 = on).
pub async fn boot_warm(
    factory: Arc<FilesystemFactory>,
    base_url: String,
    targets: Vec<ReplayTarget>,
) {
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
    // with the admission floors the survivor cache resolves for the same
    // tenant. Rank per target ONCE (no per-comparison allocation), hottest
    // tier first (weight desc, floor desc), then ascending name — the
    // DEL-prefixed shared sentinel therefore sorts LAST (entitled tenants
    // never queue behind the shared warmup)
    // documented ascending order.
    let mut targets = targets;
    {
        let rank = |t: &ReplayTarget| -> (u32, f64) {
            let Some(policy) = policy.as_ref() else {
                return (0, 0.0);
            };
            let Some(name) = t.io_trace_tenant.as_deref() else {
                return (0, 0.0);
            };
            let tier = tier_key_for(name, policy);
            policy
                .tiers
                .get(&tier)
                .or_else(|| policy.tiers.get(&policy.default_tier))
                .map(|spec| (spec.weight, spec.floor_frac))
                .unwrap_or((0, 0.0))
        };
        let mut keyed: Vec<_> = targets
            .iter()
            .map(|t| {
                let (w, f) = rank(t);
                let name = t
                    .io_trace_tenant
                    .clone()
                    .unwrap_or_else(|| "\u{7f}shared".to_string());
                (w, f, name, t.clone())
            })
            .collect();
        keyed.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| b.1.total_cmp(&a.1))
                .then_with(|| a.2.cmp(&b.2))
        });
        targets = keyed.into_iter().map(|(_, _, _, t)| t).collect();
    }
    let mut total_ranges = 0usize;
    let mut total_segments = 0usize;
    for target in &targets {
        let (r, s) = replay_tenant(&factory, &base_url, target).await;
        total_ranges += r;
        total_segments += s;
    }
    if total_ranges > 0 || total_segments > 0 {
        tracing::info!(
            targets = targets.len(),
            ranges = total_ranges,
            segments = total_segments,
            "\u{1f525} restart cache warming complete"
        );
    }
}

/// Tier key for a replay target's tenant name: the canonical-then-raw
/// stamped-tier probe (`tenant_tier_policy_key`) the admission floors use —
/// a raw commercial alias resolves to its canonical tier, a name that IS a
/// policy tier binds directly.
fn tier_key_for(name: &str, policy: &proximadb_cache::TierPolicy) -> String {
    crate::services::record_store::tenant_tier_policy_key(name, |key| policy.has_tier(key))
        .unwrap_or_else(|| policy.default_tier.clone())
}

#[cfg(test)]
mod tests {
    /// TD-CACHE-11 gate: a manifest written under the stable-id scope label is
    /// FOUND by a replay target built from (name, stable id), replayed under
    /// the SAME scope, and the replayed range is a DRAM hit for a tenant-scoped
    /// lookup afterwards. This is the restart-warm loop end-to-end minus the
    /// engine: emit-side keying + replay-side discovery + scope re-binding.
    #[tokio::test]
    async fn replay_finds_manifest_by_stable_id_label_and_serves_a_dram_hit() {
        use crate::storage::engines::sst::survivor_range_cache::SurvivorRangeCache;
        use crate::storage::persistence::filesystem::FilesystemFactory;

        let tmp = tempfile::tempdir().expect("tempdir");
        let tmp = tmp.path().to_str().expect("utf8 tmpdir").to_string();
        let factory = std::sync::Arc::new(FilesystemFactory::create_default().await.unwrap());
        let base = format!("file://{tmp}");

        let survivor = std::sync::Arc::new(SurvivorRangeCache::new(4 * 1024 * 1024));
        let invariants = std::sync::Arc::new(
            crate::storage::engines::sst::segment_format::SegmentInvariantsCache::new(
                4 * 1024 * 1024,
            ),
        );
        // Explicit caches via replay_target_with, NOT set_warm_tier_caches:
        // the global is a first-writer-wins OnceLock (no reset), so under an
        // in-process runner an engine-arming test running first would silently
        // win and this gate would exercise foreign caches. Deterministic here
        // means passing the state in — the seam the fix added.

        // A real referenced range: the "segment" bytes replay will load.
        let seg_url = format!("{base}/data/42/collections/c/segments/seg0.pax");
        let range = vec![7_u8; 4096];
        factory
            .write(
                &seg_url,
                &range,
                Some(proximadb_storage_filesystem_types::FileOptions {
                    create_dirs: true,
                    ..Default::default()
                }),
            )
            .await
            .expect("segment write");

        // Emit-side shape: manifest under the STABLE-ID scope label (what
        // `warm_set` + `manifest_url` produce), never under the tenant name.
        let manifest = WarmManifest {
            version: WARM_MANIFEST_VERSION,
            created_unix: chrono::Utc::now().timestamp(),
            entries: vec![WarmKey {
                k: 1,
                p: seg_url.clone(),
                o: 0,
                l: range.len() as u64,
            }],
        };
        let url = manifest_url(&base, "42").expect("manifest url");
        factory
            .write(
                &url,
                &serde_json::to_vec(&manifest).unwrap(),
                Some(proximadb_storage_filesystem_types::FileOptions {
                    create_dirs: true,
                    ..Default::default()
                }),
            )
            .await
            .expect("manifest write");

        // Replay-side: the boot target built from (name, stable id) via the
        // catalog resolver finds it under the label and replays as acme(42).
        let target = ReplayTarget::tenant("acme", 42);
        let (loaded, _prefilled) =
            replay_target_with(&factory, &base, &target, &invariants, &survivor).await;
        assert_eq!(loaded, 1, "the manifest's range must replay");

        // Shared-path coverage (review round-3): the shared scope's manifest
        // URL must validate and replay too — the whole single-tenant story
        // rests on data/shared/_cache/warm_cache.json being readable.
        let shared_manifest = WarmManifest {
            version: WARM_MANIFEST_VERSION,
            created_unix: chrono::Utc::now().timestamp(),
            entries: manifest.entries.clone(),
        };
        let shared_url = manifest_url(&base, "shared").expect("shared manifest url");
        factory
            .write(
                &shared_url,
                &serde_json::to_vec(&shared_manifest).unwrap(),
                Some(proximadb_storage_filesystem_types::FileOptions {
                    create_dirs: true,
                    ..Default::default()
                }),
            )
            .await
            .expect("shared manifest write");
        let (shared_loaded, _) = replay_target_with(
            &factory,
            &base,
            &ReplayTarget::shared(),
            &invariants,
            &survivor,
        )
        .await;
        assert_eq!(shared_loaded, 1, "the shared-scope manifest must replay");

        // THE GATE: a tenant-scoped lookup afterwards is a DRAM hit — the
        // loader would fail if it were invoked, so an Ok proves the bytes came
        // from the replayed cache, under the right scope.
        crate::observability::io_trace::instrument_with_stable_tenant(
            Some("acme".to_string()),
            Some(42),
            "test",
            async {
                let got = survivor
                    .get_or_fetch(
                        proximadb_cache::CacheKind::Other,
                        &seg_url,
                        0,
                        range.len() as u64,
                        || async {
                            Err::<Vec<u8>, _>(
                                proximadb_storage_filesystem_types::FilesystemError::NotFound(
                                    "loader must not run: the replayed range must be a DRAM hit"
                                        .to_string(),
                                ),
                            )
                        },
                    )
                    .await
                    .expect("lookup ok");
                assert_eq!(&got[..], &range[..], "DRAM hit serves the replayed bytes");
            },
        )
        .await;
    }

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
