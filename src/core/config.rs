use crate::ai::LLMConfig;
use crate::network::NetworkConfig;
use crate::query::unified::RerankConfig;
use crate::security::SecurityConfig;
use proximadb_storage_filesystem_types::ObjectAccessTier;
use serde::{Deserialize, Serialize};
use tracing::info;

/// Top-level application configuration loaded from `config.toml`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Server identity and bind settings
    pub server: ServerConfig,
    /// Storage engine and data layout settings
    pub storage: StorageConfig,
    /// Raft consensus cluster settings
    pub consensus: ConsensusConfig,
    /// REST/gRPC API settings
    pub api: ApiConfig,
    /// Metrics and logging settings
    pub monitoring: MonitoringConfig,
    /// Network transport configuration (optional)
    pub network: Option<NetworkConfig>,
    /// TLS certificate and key paths (optional)
    pub tls: Option<TlsConfig>,
    /// Hardware detection overrides (optional)
    #[allow(dead_code)]
    pub hardware: Option<HardwareConfig>,
    /// Semantic Knowledge Store configuration (optional)
    pub sks: Option<SksConfig>,
    /// Unified security configuration (optional)
    pub security: Option<SecurityConfig>,
    /// Global cache runtime configuration (optional)
    pub cache: Option<CacheRuntimeConfig>,
    /// Graph runtime configuration (optional)
    pub graph: Option<GraphRuntimeConfig>,
    /// Hybrid query runtime configuration (optional)
    pub hybrid: Option<HybridRuntimeConfig>,
    /// Query processing configuration (including RL planner)
    #[serde(default)]
    pub query: Option<QueryConfig>,
    /// LLM integration configuration (optional)
    pub llm: Option<LLMConfig>,
    /// Async-ingest queue runtime configuration (optional).
    ///
    /// When present, ProximaDB opens the queue subsystem at startup
    /// and async ingest (`/v2/documents?mode=async`) routes through
    /// `producer.send` → background drainer → bulk-load. When absent,
    /// async ingest degrades to inline embed.
    ///
    /// Override precedence (highest wins) — see
    /// [`QueueRuntimeConfig::resolve`] for the implementation:
    ///   1. CLI flag (none today; reserved)
    ///   2. Environment variable (`PROXIMADB_QUEUE_ROOT`,
    ///      `PROXIMADB_QUEUE_OBJECT_ARCHIVE`, `PROXIMADB_QUEUE_SYNC_MODE`,
    ///      `PROXIMADB_EMBED_DRAINER_PARTITIONS`)
    ///   3. TOML `[queue]` section (this field) — the canonical artifact
    ///   4. Built-in defaults
    ///
    /// The TOML is intentionally the authoritative declarative source
    /// (downloadable from object store, version-controlled, identical
    /// across replicas). Env vars exist for per-pod emergency tuning
    /// without rebuilding the artifact.
    pub queue: Option<QueueRuntimeConfig>,
    /// Observability configuration (optional) — currently the durable io_trace ETL
    /// sink (TD-TRACE-2 / ADR-066). Absent ⇒ the sink is off (default).
    #[serde(default)]
    pub observability: Option<ObservabilityConfig>,
    /// Postgres-compatible OLTP (`[oltp]`, ADR-095). Absent ⇒ every gate off.
    #[serde(default)]
    pub oltp: Option<OltpConfig>,
}

/// `[oltp]` (ADR-095). Default OFF (mandate #8).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OltpConfig {
    /// pgwire COMMIT is one all-or-nothing WAL group frame (TD-PGOLTP-1).
    /// Env override: `PROXIMADB_OLTP_ATOMIC_COMMIT`.
    #[serde(default)]
    pub atomic_commit: bool,
    /// The reattachable durable volume holding all non-object-store durable
    /// state, fenced by a writer epoch (TD-PGOLTP-2). Absent ⇒ no fencing.
    #[serde(default)]
    pub durable_volume: Option<DurableVolumeConfig>,
}

/// `[oltp.durable_volume]` (ADR-095 D1/D2). Env overrides:
/// `PROXIMADB_OLTP_DURABLE_VOLUME` (root), `PROXIMADB_OLTP_DEPLOYMENT_ID`,
/// `PROXIMADB_OLTP_CONTROL_URL`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurableVolumeConfig {
    /// Absolute local path of the mounted volume (EBS / Azure Disk / PD / EFS).
    pub root: String,
    /// Names the epoch log; every host that may own this volume uses the same
    /// id. Required: a shared default would let two deployments that share a
    /// `control_url` fence each other.
    #[serde(default)]
    pub deployment_id: Option<String>,
    /// Object-store URL of the writer-epoch authority. Must be reachable by
    /// every host that may own the volume and must NOT live on the volume.
    #[serde(default)]
    pub control_url: Option<String>,
    /// How often the watchdog re-checks ownership.
    #[serde(default = "default_fence_check_interval_ms")]
    pub fence_check_interval_ms: u64,
}

fn default_fence_check_interval_ms() -> u64 {
    1000
}

impl OltpConfig {
    /// `[oltp.durable_volume]` with env overrides applied; `None` when neither
    /// the TOML section nor `PROXIMADB_OLTP_DURABLE_VOLUME` is set.
    pub fn resolve_durable_volume(config: Option<&Self>) -> Option<DurableVolumeConfig> {
        let env = |key: &str| std::env::var(key).ok().filter(|v| !v.trim().is_empty());
        let mut volume = config.and_then(|c| c.durable_volume.clone());
        if let Some(root) = env("PROXIMADB_OLTP_DURABLE_VOLUME") {
            volume = Some(match volume {
                Some(existing) => DurableVolumeConfig { root, ..existing },
                None => DurableVolumeConfig {
                    root,
                    deployment_id: None,
                    control_url: None,
                    fence_check_interval_ms: default_fence_check_interval_ms(),
                },
            });
        }
        let mut volume = volume?;
        if let Some(id) = env("PROXIMADB_OLTP_DEPLOYMENT_ID") {
            volume.deployment_id = Some(id);
        }
        if let Some(url) = env("PROXIMADB_OLTP_CONTROL_URL") {
            volume.control_url = Some(url);
        }
        Some(volume)
    }
}

pub use proximadb_config::{HardwareConfig, SksConfig, TlsConfig};

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            storage: StorageConfig::default(),
            consensus: ConsensusConfig::default(),
            api: ApiConfig::default(),
            monitoring: MonitoringConfig::default(),
            network: None,
            tls: None,
            hardware: Some(HardwareConfig::default()),
            sks: None, // SKS disabled by default
            security: None,
            cache: None,
            graph: Some(GraphRuntimeConfig::default()),
            hybrid: Some(HybridRuntimeConfig::default()),
            query: None, // Uses default RL planner settings when None
            llm: None,
            queue: None,
            observability: None,
            oltp: None,
        }
    }
}

/// Observability configuration (`[observability]` TOML table). Today it carries the
/// durable io_trace ETL sink; more observability knobs can nest here later.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ObservabilityConfig {
    /// Durable per-query trace sink (`[observability.io_trace_sink]`, TD-TRACE-2).
    #[serde(default)]
    pub io_trace_sink: Option<IoTraceSinkConfig>,
}

/// Durable io_trace ETL sink configuration (`[observability.io_trace_sink]`,
/// TD-TRACE-2 / ADR-066). The TOML is the canonical declarative source; env vars
/// (`PROXIMADB_IO_TRACE_SINK_*`) are per-pod overrides; defaults fill the gaps —
/// see [`IoTraceSinkConfig::resolve`]. **Default-OFF** (`enabled = false`).
///
/// ```toml
/// [observability.io_trace_sink]
/// enabled = false
/// local_dir = "./log/io_trace"
/// segment_bytes = 4194304
/// flush_interval_s = 60
/// compression = "zstd"
/// format = "jsonl"
/// spool_max_bytes = 268435456
/// pending_max_bytes = 1073741824
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IoTraceSinkConfig {
    /// Master switch — the sink observer is installed only when this is true.
    #[serde(default)]
    pub enabled: bool,
    /// Local directory where JSONL+zstd segments are written (S1). Default
    /// `./log/io_trace`.
    pub local_dir: Option<String>,
    /// Seal a segment when its (uncompressed) size reaches this many bytes.
    /// Default 4 MiB.
    pub segment_bytes: Option<u64>,
    /// Also seal + drain on this interval (seconds), whichever comes first.
    /// Default 60.
    pub flush_interval_s: Option<u64>,
    /// Segment compression codec. Default (and only value in S1) `zstd`.
    pub compression: Option<String>,
    /// Segment record format. Default (and only value in S1) `jsonl`.
    pub format: Option<String>,
    /// Bounded in-memory spool cap (bytes-equivalent record budget); on overflow the
    /// oldest queued record is dropped (best-effort). Default 256 MiB.
    pub spool_max_bytes: Option<u64>,
    /// Hard durable pending-file cap. A segment that would cross it is returned
    /// to ingress; sealed files are never evicted. Default 1 GiB.
    pub pending_max_bytes: Option<u64>,
    /// Object-store destination URI (consumed in S2 — parsed now, unused in S1).
    pub object_store_uri: Option<String>,
    /// Per-object access tier for the object-store dispatch (S2). e.g. `cool`.
    pub access_tier: Option<String>,
    /// Key partitioning scheme for the object-store dispatch (S2). `date`/`tenant`/`host`.
    pub partition_by: Option<String>,
    /// Retention days for object-store lifecycle GC (S2).
    pub retention_days: Option<u64>,
    /// TD-TRACE-2 S4b: interval (seconds) for the background warehouse compactor that
    /// projects the durable envelopes into Iceberg-managed Parquet. `None`/`0` ⇒ off;
    /// requires `object_store_uri` (the compactor reads the object-store segments).
    pub warehouse_compaction_interval_s: Option<u64>,
}

impl IoTraceSinkConfig {
    /// Layer env (`PROXIMADB_IO_TRACE_SINK_*`) over TOML over defaults. Returns
    /// `None` when the sink is disabled (no observer installed, no worker spawned).
    /// Mirrors [`QueueRuntimeConfig::resolve`].
    pub fn resolve(toml_section: Option<&IoTraceSinkConfig>) -> Option<ResolvedIoTraceSinkConfig> {
        Self::try_resolve(toml_section).unwrap_or(None)
    }

    /// Strict runtime resolution. Unlike [`Self::resolve`], this reports invalid
    /// active settings so database startup cannot silently run a different
    /// delivery contract than the operator requested.
    pub fn try_resolve(
        toml_section: Option<&IoTraceSinkConfig>,
    ) -> Result<Option<ResolvedIoTraceSinkConfig>, String> {
        let from_toml = toml_section.cloned().unwrap_or_default();

        let enabled = std::env::var("PROXIMADB_IO_TRACE_SINK")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(from_toml.enabled);
        if !enabled {
            return Ok(None);
        }

        let env_u64 = |key: &str| std::env::var(key).ok().and_then(|v| v.parse::<u64>().ok());

        let local_dir = std::env::var("PROXIMADB_IO_TRACE_SINK_LOCAL_DIR")
            .ok()
            .or(from_toml.local_dir.clone())
            .unwrap_or_else(|| "./log/io_trace".to_string());
        let segment_bytes = env_u64("PROXIMADB_IO_TRACE_SINK_SEGMENT_BYTES")
            .or(from_toml.segment_bytes)
            .unwrap_or(4 * 1024 * 1024);
        let flush_interval_s = env_u64("PROXIMADB_IO_TRACE_SINK_FLUSH_INTERVAL_S")
            .or(from_toml.flush_interval_s)
            .unwrap_or(60)
            .max(1);
        let spool_max_bytes = env_u64("PROXIMADB_IO_TRACE_SINK_SPOOL_MAX_BYTES")
            .or(from_toml.spool_max_bytes)
            .unwrap_or(256 * 1024 * 1024);
        let pending_max_bytes = env_u64("PROXIMADB_IO_TRACE_SINK_PENDING_MAX_BYTES")
            .or(from_toml.pending_max_bytes)
            .unwrap_or(1024 * 1024 * 1024);
        let compression = std::env::var("PROXIMADB_IO_TRACE_SINK_COMPRESSION")
            .ok()
            .or(from_toml.compression.clone())
            .unwrap_or_else(|| "zstd".to_string())
            .to_ascii_lowercase();
        let format = std::env::var("PROXIMADB_IO_TRACE_SINK_FORMAT")
            .ok()
            .or(from_toml.format.clone())
            .unwrap_or_else(|| "jsonl".to_string())
            .to_ascii_lowercase();

        if compression != "zstd" {
            return Err(format!(
                "unsupported io_trace sink compression `{compression}`; only `zstd` is implemented"
            ));
        }
        if format != "jsonl" {
            return Err(format!(
                "unsupported io_trace sink format `{format}`; only `jsonl` is implemented"
            ));
        }
        if segment_bytes == 0 || spool_max_bytes == 0 || pending_max_bytes == 0 {
            return Err(
                "io_trace sink segment_bytes, spool_max_bytes, and pending_max_bytes must be non-zero"
                    .to_string(),
            );
        }

        // S2 (ADR-066 D4): optional object-store dispatch. When set, sealed segments
        // are PUT to this store; when unset, the sink stays local-only (S1). The
        // access tier is the at-rest GB-month cost lever — default Cold (rare
        // access); an unrecognised value falls back to Cold rather than failing.
        let object_store_uri = std::env::var("PROXIMADB_IO_TRACE_SINK_OBJECT_STORE_URI")
            .ok()
            .or(from_toml.object_store_uri.clone());
        let access_tier_text = std::env::var("PROXIMADB_IO_TRACE_SINK_ACCESS_TIER")
            .ok()
            .or(from_toml.access_tier.clone());
        let access_tier = match access_tier_text {
            Some(value) => ObjectAccessTier::parse(&value)
                .ok_or_else(|| format!("unsupported io_trace sink access_tier `{value}`"))?,
            None => ObjectAccessTier::Cold,
        };

        // S4b: background warehouse compaction interval. Env over TOML; `0` normalizes
        // to `None` (off). Only meaningful with an `object_store_uri`.
        let warehouse_compaction_interval_s =
            env_u64("PROXIMADB_IO_TRACE_SINK_WAREHOUSE_COMPACTION_INTERVAL_S")
                .or(from_toml.warehouse_compaction_interval_s)
                .filter(|s| *s > 0);

        Ok(Some(ResolvedIoTraceSinkConfig {
            local_dir,
            segment_bytes,
            flush_interval_s,
            spool_max_bytes,
            pending_max_bytes,
            compression,
            format,
            object_store_uri,
            access_tier,
            warehouse_compaction_interval_s,
        }))
    }
}

/// Resolved io_trace sink settings after the precedence layers fold. All S1 spool
/// fields are populated; the S2 object-store fields carry the dispatch target
/// (`object_store_uri` `None` ⇒ local-only) and the at-rest tier.
#[derive(Debug, Clone)]
pub struct ResolvedIoTraceSinkConfig {
    pub local_dir: String,
    pub segment_bytes: u64,
    pub flush_interval_s: u64,
    pub spool_max_bytes: u64,
    pub pending_max_bytes: u64,
    pub compression: String,
    pub format: String,
    /// Object-store dispatch target (S2). `None` ⇒ segments stay local-only.
    pub object_store_uri: Option<String>,
    /// Per-object access tier for the object-store PUT (S2). Default `Cold`.
    pub access_tier: ObjectAccessTier,
    /// S4b background warehouse-compaction interval (seconds). `None` ⇒ off (also off
    /// unless `object_store_uri` is set).
    pub warehouse_compaction_interval_s: Option<u64>,
}

/// Queue subsystem runtime configuration. Lives at the `[queue]` TOML
/// section. See `Config.queue` for the precedence rules — short version:
/// the TOML is the canonical declarative source; env vars exist for
/// per-pod emergency overrides; defaults fill in the gaps.
///
/// Wire-format example:
///
/// ```toml
/// [queue]
/// root = "file:///var/lib/proximadb/queue"
/// object_archive = "adls://example-operator/queue-cold"   # optional
/// sync_mode = "strict"                             # "strict" or "lazy"
/// drainer_partitions = "0..16"                    # this replica's range
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QueueRuntimeConfig {
    /// Filesystem URL where queue segments live (e.g.
    /// `file:///var/lib/proximadb/queue`, `adls://...`, `s3://...`).
    /// Required to enable async ingest.
    pub root: Option<String>,
    /// Optional second-tier archive URL. Sealed segments are mirrored
    /// here so the queue survives ECS pod loss / k8s reschedule onto a
    /// fresh-disk node.
    pub object_archive: Option<String>,
    /// `"strict"` (default, fsync-before-ack) or `"lazy"`.
    pub sync_mode: Option<String>,
    /// Partition range this replica drains, e.g. `"0..16"` or
    /// `"0,3..6,9"`. Single-replica deployments leave it unset; multi-
    /// replica deployments pass disjoint ranges per pod (the cross-
    /// process leases enforce ownership).
    pub drainer_partitions: Option<String>,
}

impl QueueRuntimeConfig {
    /// Build the runtime queue settings by layering env over TOML over
    /// defaults. Returns `None` when none of the layers supplied a
    /// `root` URL — that's the explicit signal that async ingest is
    /// disabled.
    ///
    /// Precedence (highest wins):
    /// 1. **Environment variable** (`PROXIMADB_QUEUE_*`) — per-pod
    ///    emergency override. Set in the k8s/ECS task spec.
    /// 2. **TOML `[queue]` section** — canonical declarative artifact.
    ///    Downloadable from object store; identical across replicas.
    /// 3. **Compiled-in defaults** — `sync_mode = "strict"`,
    ///    `drainer_partitions = "0..16"`.
    ///
    /// A future CLI-flag layer (e.g. `--queue-root <url>`) would slot
    /// above env without changing the return type.
    pub fn resolve(toml_section: Option<&QueueRuntimeConfig>) -> Option<ResolvedQueueConfig> {
        let from_toml = toml_section.cloned().unwrap_or_default();
        let root = std::env::var("PROXIMADB_QUEUE_ROOT")
            .ok()
            .or(from_toml.root.clone());
        // No root anywhere → async ingest stays off entirely.
        let root = root?;

        let object_archive = std::env::var("PROXIMADB_QUEUE_OBJECT_ARCHIVE")
            .ok()
            .or(from_toml.object_archive.clone());
        let sync_mode = std::env::var("PROXIMADB_QUEUE_SYNC_MODE")
            .ok()
            .or(from_toml.sync_mode.clone())
            .unwrap_or_else(|| "strict".to_string());
        let drainer_partitions = std::env::var("PROXIMADB_EMBED_DRAINER_PARTITIONS")
            .ok()
            .or(from_toml.drainer_partitions.clone())
            .unwrap_or_else(|| "0..16".to_string());

        Some(ResolvedQueueConfig {
            root,
            object_archive,
            sync_mode,
            drainer_partitions,
        })
    }
}

/// Resolved queue configuration after the precedence layers are
/// folded. All fields are populated (no Options) so callers can
/// consume directly without per-field defaulting.
#[derive(Debug, Clone)]
pub struct ResolvedQueueConfig {
    pub root: String,
    pub object_archive: Option<String>,
    pub sync_mode: String,
    pub drainer_partitions: String,
}

#[cfg(test)]
mod queue_config_tests {
    use super::*;

    /// Restoring env vars after a test mutates them is fragile (other
    /// tests run in the same process and see the changes). Each test
    /// scopes its env reads via this guard, which restores the value
    /// on drop.
    struct EnvGuard {
        key: String,
        original: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &str, value: Option<&str>) -> Self {
            let original = std::env::var(key).ok();
            match value {
                Some(v) => unsafe { std::env::set_var(key, v) },
                None => unsafe { std::env::remove_var(key) },
            }
            EnvGuard {
                key: key.to_string(),
                original,
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match self.original.take() {
                Some(v) => unsafe { std::env::set_var(&self.key, v) },
                None => unsafe { std::env::remove_var(&self.key) },
            }
        }
    }

    /// Without any TOML section AND without env vars, async ingest is
    /// disabled — resolve returns None. This is the "no behavior
    /// change for unaware deployments" guarantee.
    #[test]
    fn resolve_returns_none_when_no_toml_and_no_env() {
        let _g_root = EnvGuard::set("PROXIMADB_QUEUE_ROOT", None);
        let _g_arc = EnvGuard::set("PROXIMADB_QUEUE_OBJECT_ARCHIVE", None);
        let _g_sync = EnvGuard::set("PROXIMADB_QUEUE_SYNC_MODE", None);
        let _g_part = EnvGuard::set("PROXIMADB_EMBED_DRAINER_PARTITIONS", None);
        assert!(QueueRuntimeConfig::resolve(None).is_none());
        assert!(QueueRuntimeConfig::resolve(Some(&QueueRuntimeConfig::default())).is_none());
    }

    /// TD-TRACE-2: the sink is default-OFF — absent config or `enabled=false`
    /// resolves to `None` (no observer, no worker).
    #[test]
    fn io_trace_sink_disabled_resolves_to_none() {
        let _g = EnvGuard::set("PROXIMADB_IO_TRACE_SINK", None);
        assert!(IoTraceSinkConfig::resolve(None).is_none());
        assert!(IoTraceSinkConfig::resolve(Some(&IoTraceSinkConfig::default())).is_none());
    }

    /// TD-TRACE-2: an enabled TOML section resolves; unset fields fall to defaults.
    #[test]
    fn io_trace_sink_enabled_toml_fills_defaults() {
        let _g0 = EnvGuard::set("PROXIMADB_IO_TRACE_SINK", None);
        let _g1 = EnvGuard::set("PROXIMADB_IO_TRACE_SINK_LOCAL_DIR", None);
        let _g2 = EnvGuard::set("PROXIMADB_IO_TRACE_SINK_SEGMENT_BYTES", None);
        let _g3 = EnvGuard::set("PROXIMADB_IO_TRACE_SINK_FLUSH_INTERVAL_S", None);
        let _g4 = EnvGuard::set("PROXIMADB_IO_TRACE_SINK_OBJECT_STORE_URI", None);
        let _g5 = EnvGuard::set("PROXIMADB_IO_TRACE_SINK_ACCESS_TIER", None);
        let _g6 = EnvGuard::set("PROXIMADB_IO_TRACE_SINK_PENDING_MAX_BYTES", None);
        let toml = IoTraceSinkConfig {
            enabled: true,
            local_dir: Some("/tmp/tr".to_string()),
            ..Default::default()
        };
        let r = IoTraceSinkConfig::resolve(Some(&toml)).expect("enabled → Some");
        assert_eq!(r.local_dir, "/tmp/tr");
        assert_eq!(r.segment_bytes, 4 * 1024 * 1024);
        assert_eq!(r.flush_interval_s, 60);
        assert_eq!(r.spool_max_bytes, 256 * 1024 * 1024);
        assert_eq!(r.pending_max_bytes, 1024 * 1024 * 1024);
        assert_eq!(r.compression, "zstd");
        assert_eq!(r.format, "jsonl");
        // S2 defaults: no object store (local-only), Cold tier.
        assert_eq!(r.object_store_uri, None);
        assert_eq!(r.access_tier, ObjectAccessTier::Cold);
        assert_eq!(r.warehouse_compaction_interval_s, None);
    }

    #[test]
    fn io_trace_sink_strict_resolution_rejects_unsupported_active_settings() {
        let _g0 = EnvGuard::set("PROXIMADB_IO_TRACE_SINK", None);
        let _g1 = EnvGuard::set("PROXIMADB_IO_TRACE_SINK_COMPRESSION", None);
        let _g2 = EnvGuard::set("PROXIMADB_IO_TRACE_SINK_FORMAT", None);
        let _g3 = EnvGuard::set("PROXIMADB_IO_TRACE_SINK_ACCESS_TIER", None);

        for config in [
            IoTraceSinkConfig {
                enabled: true,
                compression: Some("gzip".to_string()),
                ..Default::default()
            },
            IoTraceSinkConfig {
                enabled: true,
                format: Some("parquet".to_string()),
                ..Default::default()
            },
            IoTraceSinkConfig {
                enabled: true,
                access_tier: Some("cheapish".to_string()),
                ..Default::default()
            },
        ] {
            assert!(IoTraceSinkConfig::try_resolve(Some(&config)).is_err());
        }
    }

    /// TOML-only flow: a TOML `[queue]` section with `root` set
    /// activates the queue. Defaults fill in everything else.
    #[test]
    fn resolve_uses_toml_when_env_unset() {
        let _g_root = EnvGuard::set("PROXIMADB_QUEUE_ROOT", None);
        let _g_arc = EnvGuard::set("PROXIMADB_QUEUE_OBJECT_ARCHIVE", None);
        let _g_sync = EnvGuard::set("PROXIMADB_QUEUE_SYNC_MODE", None);
        let _g_part = EnvGuard::set("PROXIMADB_EMBED_DRAINER_PARTITIONS", None);
        let toml = QueueRuntimeConfig {
            root: Some("file:///srv/queue".to_string()),
            object_archive: Some("adls://example-operator/archive".to_string()),
            sync_mode: Some("lazy".to_string()),
            drainer_partitions: Some("0..4".to_string()),
        };
        let resolved = QueueRuntimeConfig::resolve(Some(&toml)).expect("resolved");
        assert_eq!(resolved.root, "file:///srv/queue");
        assert_eq!(
            resolved.object_archive.as_deref(),
            Some("adls://example-operator/archive")
        );
        assert_eq!(resolved.sync_mode, "lazy");
        assert_eq!(resolved.drainer_partitions, "0..4");
    }

    /// Env beats TOML — the documented precedence in `Config.queue`.
    /// All four fields swap when env is set even when TOML provided a
    /// different value.
    #[test]
    fn resolve_env_overrides_toml() {
        let _g_root = EnvGuard::set("PROXIMADB_QUEUE_ROOT", Some("file:///emergency"));
        let _g_arc = EnvGuard::set("PROXIMADB_QUEUE_OBJECT_ARCHIVE", Some("s3://hotfix"));
        let _g_sync = EnvGuard::set("PROXIMADB_QUEUE_SYNC_MODE", Some("strict"));
        let _g_part = EnvGuard::set("PROXIMADB_EMBED_DRAINER_PARTITIONS", Some("8..16"));
        let toml = QueueRuntimeConfig {
            root: Some("file:///canonical".to_string()),
            object_archive: Some("adls://canonical".to_string()),
            sync_mode: Some("lazy".to_string()),
            drainer_partitions: Some("0..16".to_string()),
        };
        let resolved = QueueRuntimeConfig::resolve(Some(&toml)).expect("resolved");
        assert_eq!(resolved.root, "file:///emergency");
        assert_eq!(resolved.object_archive.as_deref(), Some("s3://hotfix"));
        assert_eq!(resolved.sync_mode, "strict");
        assert_eq!(resolved.drainer_partitions, "8..16");
    }

    /// Env-only flow: when there's no TOML section at all, env vars
    /// can still activate the queue (useful for ad-hoc local runs).
    #[test]
    fn resolve_env_alone_activates_queue() {
        let _g_root = EnvGuard::set("PROXIMADB_QUEUE_ROOT", Some("file:///env-only"));
        let _g_arc = EnvGuard::set("PROXIMADB_QUEUE_OBJECT_ARCHIVE", None);
        let _g_sync = EnvGuard::set("PROXIMADB_QUEUE_SYNC_MODE", None);
        let _g_part = EnvGuard::set("PROXIMADB_EMBED_DRAINER_PARTITIONS", None);
        let resolved = QueueRuntimeConfig::resolve(None).expect("resolved");
        assert_eq!(resolved.root, "file:///env-only");
        assert!(resolved.object_archive.is_none());
        assert_eq!(resolved.sync_mode, "strict", "default fills in");
        assert_eq!(resolved.drainer_partitions, "0..16", "default fills in");
    }

    /// Per-field precedence: env can override SOME fields while TOML
    /// supplies others. This is the "partial emergency tuning" case —
    /// e.g., a hot pod gets a different partition range without
    /// touching the canonical TOML.
    #[test]
    fn resolve_per_field_override() {
        let _g_root = EnvGuard::set("PROXIMADB_QUEUE_ROOT", None);
        let _g_arc = EnvGuard::set("PROXIMADB_QUEUE_OBJECT_ARCHIVE", None);
        let _g_sync = EnvGuard::set("PROXIMADB_QUEUE_SYNC_MODE", None);
        let _g_part = EnvGuard::set("PROXIMADB_EMBED_DRAINER_PARTITIONS", Some("12..16"));
        let toml = QueueRuntimeConfig {
            root: Some("file:///canonical".to_string()),
            object_archive: Some("adls://canonical".to_string()),
            sync_mode: Some("strict".to_string()),
            drainer_partitions: Some("0..16".to_string()),
        };
        let resolved = QueueRuntimeConfig::resolve(Some(&toml)).expect("resolved");
        // TOML wins for the unmasked fields...
        assert_eq!(resolved.root, "file:///canonical");
        assert_eq!(resolved.object_archive.as_deref(), Some("adls://canonical"));
        assert_eq!(resolved.sync_mode, "strict");
        // ...env wins for the one field that's set.
        assert_eq!(resolved.drainer_partitions, "12..16");
    }

    /// Round-trip the [queue] section through serde + TOML to lock the
    /// wire format. Future renames will fail this test loudly.
    #[test]
    fn queue_section_round_trips_through_toml() {
        let original = QueueRuntimeConfig {
            root: Some("file:///canonical".to_string()),
            object_archive: Some("adls://example-operator/archive".to_string()),
            sync_mode: Some("strict".to_string()),
            drainer_partitions: Some("0..16".to_string()),
        };
        let serialized = toml::to_string(&original).expect("ser");
        // Lock the literal field names — these are the public TOML
        // surface and changing them is a breaking config-file change.
        assert!(serialized.contains("root ="));
        assert!(serialized.contains("object_archive ="));
        assert!(serialized.contains("sync_mode ="));
        assert!(serialized.contains("drainer_partitions ="));
        let restored: QueueRuntimeConfig = toml::from_str(&serialized).expect("de");
        assert_eq!(restored.root, original.root);
        assert_eq!(restored.object_archive, original.object_archive);
        assert_eq!(restored.sync_mode, original.sync_mode);
        assert_eq!(restored.drainer_partitions, original.drainer_partitions);
    }
}

/// Runtime cache configuration for the unified Cross-Cache Orchestrator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheRuntimeConfig {
    /// Total memory budget for orchestrator-managed caches (in MB)
    pub total_memory_mb: u64,

    /// Enable cache warming service
    pub enable_warming: bool,

    /// Memory rebalancing configuration
    pub rebalancing: CacheRebalancingConfig,

    /// Eviction configuration
    pub eviction: CacheEvictionConfig,

    /// Per-cache-type configurations
    pub types: CacheTypesConfig,

    /// Cache warming configuration
    pub warming: CacheWarmingConfig,
}

fn default_orchestrator_budget_mb() -> u64 {
    512
}

impl Default for CacheRuntimeConfig {
    fn default() -> Self {
        Self {
            total_memory_mb: default_orchestrator_budget_mb(),
            enable_warming: false, // Disabled by default
            rebalancing: CacheRebalancingConfig::default(),
            eviction: CacheEvictionConfig::default(),
            types: CacheTypesConfig::default(),
            warming: CacheWarmingConfig::default(),
        }
    }
}

/// Configuration for periodic cache rebalancing across cache tiers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheRebalancingConfig {
    /// Whether automatic rebalancing is enabled
    pub enabled: bool,
    /// Seconds between rebalancing passes
    pub interval_seconds: u64,
    /// Minimum hit-rate below which a cache tier is considered cold
    pub min_hit_rate_threshold: f64,
}

impl Default for CacheRebalancingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_seconds: 300,       // 5 minutes
            min_hit_rate_threshold: 0.1, // 10% minimum hit rate
        }
    }
}

/// Configuration for cache eviction policies and thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEvictionConfig {
    /// Whether automatic eviction is enabled
    pub enabled: bool,
    /// Seconds between eviction checks
    pub check_interval_seconds: u64,
    /// Number of entries to evict per batch
    pub batch_size: usize,
    /// Memory usage percentage that triggers eviction
    pub memory_threshold_percent: u8,
    /// Ordered list of eviction policies to apply
    pub policies: Vec<EvictionPolicyConfig>,
}

impl Default for CacheEvictionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            check_interval_seconds: 60,
            batch_size: 100,
            memory_threshold_percent: 90,
            policies: vec![
                EvictionPolicyConfig {
                    policy_type: "lru".to_string(),
                    max_items: Some(10000),
                    batch_size: Some(100),
                    max_age_seconds: None,
                    cleanup_interval_seconds: None,
                },
                EvictionPolicyConfig {
                    policy_type: "ttl".to_string(),
                    max_items: None,
                    batch_size: None,
                    max_age_seconds: Some(3600),
                    cleanup_interval_seconds: Some(300),
                },
            ],
        }
    }
}

/// Configuration for a single eviction policy (LRU, TTL, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvictionPolicyConfig {
    /// Policy type identifier (e.g., "lru", "ttl")
    #[serde(rename = "type")]
    pub policy_type: String,
    /// Maximum number of cached items before eviction starts
    pub max_items: Option<usize>,
    /// Number of entries to evict at once
    pub batch_size: Option<usize>,
    /// Maximum age in seconds before an entry expires (TTL policy)
    pub max_age_seconds: Option<u64>,
    /// Seconds between TTL cleanup sweeps
    pub cleanup_interval_seconds: Option<u64>,
}

/// Per-cache-type memory allocation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheTypesConfig {
    /// Allocation for the vector data cache
    pub vector: CacheTypeConfig,
    /// Allocation for the query result cache
    pub query: CacheTypeConfig,
    /// Allocation for the metadata cache
    pub metadata: CacheTypeConfig,
    /// Allocation for the index node cache
    pub index: CacheTypeConfig,
    /// Allocation for the bitmap filter cache
    pub filter: CacheTypeConfig,
}

impl Default for CacheTypesConfig {
    fn default() -> Self {
        Self {
            vector: CacheTypeConfig {
                initial_allocation_mb: 200,
                min_allocation_mb: 50,
                max_allocation_mb: 400,
            },
            query: CacheTypeConfig {
                initial_allocation_mb: 150,
                min_allocation_mb: 30,
                max_allocation_mb: 300,
            },
            metadata: CacheTypeConfig {
                initial_allocation_mb: 50,
                min_allocation_mb: 10,
                max_allocation_mb: 100,
            },
            index: CacheTypeConfig {
                initial_allocation_mb: 50,
                min_allocation_mb: 10,
                max_allocation_mb: 100,
            },
            filter: CacheTypeConfig {
                initial_allocation_mb: 50,
                min_allocation_mb: 10,
                max_allocation_mb: 100,
            },
        }
    }
}

/// Memory allocation bounds for a single cache type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheTypeConfig {
    /// Initial memory allocation in megabytes
    pub initial_allocation_mb: u64,
    /// Minimum memory allocation in megabytes
    pub min_allocation_mb: u64,
    /// Maximum memory allocation in megabytes
    pub max_allocation_mb: u64,
}

/// Configuration for proactive cache warming strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheWarmingConfig {
    /// Warming strategy names (e.g., "popularity", "time_based")
    pub strategies: Vec<String>,
    /// Whether to warm the cache immediately on server startup
    pub warm_on_startup: bool,
    /// Number of entries to warm per batch
    pub warm_batch_size: usize,
    /// Minimum access count for an entry to qualify for popularity warming
    pub popularity_threshold: u32,
    /// Lookback window in hours for time-based warming
    pub time_window_hours: u64,
}

impl Default for CacheWarmingConfig {
    fn default() -> Self {
        Self {
            strategies: vec!["popularity".to_string(), "time_based".to_string()],
            warm_on_startup: false,
            warm_batch_size: 100,
            popularity_threshold: 10,
            time_window_hours: 24,
        }
    }
}

pub use proximadb_config::{GraphRuntimeConfig, HybridRuntimeConfig};

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            storage_locations: vec![StorageLocation::default()],
            metadata_url: "file://./metadata".to_string(),
            assignment_config: AssignmentConfig::default(),
            wal_config: WriteBufferUserConfig::default(),
            prune_mode: None,
            mmap_enabled: true,
            sst_config: Some(SstConfig::default()),
            viper_config: Some(ViperConfig::default()),
            cache_size_mb: 512,
            bloom_filter_config: Some(BloomFilterConfig::default()),
            compaction_config: CompactionConfig::default(),
            filesystem_config: FilesystemOptimizationConfig::default(),
            optimization: OptimizationConfig::default(),
            reservation_pools: Vec::new(),
        }
    }
}

pub use proximadb_config::ServerConfig;

/// Backwards-compat alias for [`CoreStorageConfig`].
pub type StorageConfig = CoreStorageConfig;

/// Storage engine layout, WAL, compaction, and filesystem optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreStorageConfig {
    /// Storage locations - each can host WriteBuffer, data, and indexes
    pub storage_locations: Vec<StorageLocation>,

    /// Single metadata URL for consistency (e.g., "file:///fast-ssd/proximadb/metadata_info")
    pub metadata_url: String,

    /// Assignment configuration
    pub assignment_config: AssignmentConfig,

    /// Write buffer configuration (global memtable settings)
    pub wal_config: WriteBufferUserConfig,

    /// Search pruning configuration
    #[serde(default)]
    pub prune_mode: Option<PruneModeConfig>,

    /// Enable memory-mapped I/O for storage engines
    pub mmap_enabled: bool,
    /// SST (Sorted String Table) engine configuration
    pub sst_config: Option<SstConfig>,
    /// VIPER columnar engine configuration
    pub viper_config: Option<ViperConfig>,
    /// Global read cache size in megabytes
    pub cache_size_mb: u64,
    /// Bloom filter settings for metadata filtering
    pub bloom_filter_config: Option<BloomFilterConfig>,

    /// Common compaction configuration (can be overridden per engine)
    pub compaction_config: CompactionConfig,

    /// Filesystem optimization settings
    pub filesystem_config: FilesystemOptimizationConfig,

    /// Performance optimization settings
    #[serde(default)]
    pub optimization: OptimizationConfig,

    /// TD-TIER-1 SC3: operator-declared local reservation pools. Empty (the
    /// default) leaves the reservation feature entirely inert — no authority
    /// is constructed, admission is unchanged. The whole list is validated at
    /// config load (`validate_pool_configs`: per-pool shape, duplicate
    /// id/path rejection) and each pool again when an authority opens it, so
    /// a bad declaration fails at startup rather than degrading silently.
    /// Each pool is EXCLUSIVELY OWNED per process (and per host via the
    /// owner.lock): a second in-process `CollectionService` construction or a
    /// second process on the same pool directory fails startup by design.
    #[serde(default)]
    pub reservation_pools: Vec<crate::services::collection::reservation::ReservationPoolConfig>,
}

pub use proximadb_config::{
    AdvancedPruneConfig, AssignmentConfig, CompactionConfig, ConsensusConfig, DiskClass,
    FilesystemOptimizationConfig, IoBudgetConfig, MonitoringConfig, OptimizationConfig,
    PruneModeConfig, StorageLocation, TempStrategy, TransactionalOperationsConfig,
};
use proximadb_storage_common::iops_budget::{
    IopsBudget, clear_location_budgets, register_location_budget,
};

impl StorageConfig {
    /// Resolve the SST configuration that every runtime SST instance must use.
    ///
    /// `storage.compaction_config` is the common operator-facing policy.
    /// `storage.sst_config.compaction_config`, when present, is the explicit
    /// engine override. Centralizing the merge here prevents one SST instance
    /// from silently falling back to code defaults while another honors TOML.
    pub fn effective_sst_config(&self) -> SstConfig {
        let mut config = self.sst_config.clone().unwrap_or_default();
        if config.compaction_config.is_none() {
            config.compaction_config = Some(self.compaction_config.clone());
        }
        config
    }

    /// Get storage URLs from locations
    pub fn storage_urls(&self) -> Vec<String> {
        self.storage_locations
            .iter()
            .map(|loc| loc.url.clone())
            .collect()
    }

    /// Get WAL URLs derived from storage URLs
    pub fn write_buffer_urls(&self) -> Vec<String> {
        self.storage_locations
            .iter()
            .map(|loc| format!("{}/wal", loc.url.trim_end_matches('/')))
            .collect()
    }

    /// Get data URLs derived from storage URLs
    pub fn data_urls(&self) -> Vec<String> {
        self.storage_locations
            .iter()
            .map(|loc| format!("{}/data", loc.url.trim_end_matches('/')))
            .collect()
    }

    /// Get index URLs derived from storage URLs
    pub fn index_urls(&self) -> Vec<String> {
        self.storage_locations
            .iter()
            .map(|loc| format!("{}/index", loc.url.trim_end_matches('/')))
            .collect()
    }

    /// Resolve one location's `io_budget` into the concrete [`IopsBudget`] the
    /// storage leaf consumes (TD-IOBUDGET-1).
    ///
    /// Base profile: `disk_class` when present (`ssd` → LOCAL, `hdd` → CLOUD
    /// per ADR-073, `cloud` → the path scheme's cloud default), otherwise the
    /// path's scheme+env default. Each explicit `*_bytes` field then overrides
    /// that profile's value. Validation is fail-closed: a 64 KiB floor on
    /// `min` and `min ≤ target ≤ max` — an operator error must die at startup,
    /// not silently size GETs.
    pub fn resolve_location_io_budget(
        url: &str,
        cfg: &IoBudgetConfig,
    ) -> Result<IopsBudget, String> {
        const MIN_FLOOR_BYTES: u64 = 64 * 1024;

        let mut budget = match cfg.disk_class {
            Some(DiskClass::Ssd) => IopsBudget::LOCAL,
            Some(DiskClass::Hdd) => IopsBudget::CLOUD,
            Some(DiskClass::Cloud) => IopsBudget::scheme_cloud_budget(url),
            None => IopsBudget::for_scheme_and_env(url),
        };
        if let Some(v) = cfg.min_bytes {
            budget.min = v;
        }
        if let Some(v) = cfg.target_bytes {
            budget.target = v;
        }
        if let Some(v) = cfg.max_bytes {
            budget.max = v;
        }
        if budget.min < MIN_FLOOR_BYTES {
            return Err(format!(
                "min_bytes {} is below the {MIN_FLOOR_BYTES}-byte floor; a smaller ranged GET cannot amortize I/O",
                budget.min
            ));
        }
        if budget.min > budget.target || budget.target > budget.max {
            return Err(format!(
                "bounds must satisfy min ≤ target ≤ max, got min={} target={} max={}",
                budget.min, budget.target, budget.max
            ));
        }
        Ok(budget)
    }

    /// Fail closed when the same location URL carries io_budget sections that
    /// resolve to DIFFERENT budgets (TD-IOBUDGET-1 review finding). The leaf
    /// registry is keyed by URL with insert-overwrite, so two entries for one
    /// URL would silently let the last one win for every path under that
    /// prefix. Duplicate URLs with identical (or absent) budgets stay allowed
    /// — deliberate weight duplication pre-exists and is not a budget
    /// conflict.
    pub fn validate_io_budget_conflicts(&self) -> Result<(), String> {
        let mut resolved_by_url: std::collections::HashMap<&str, IopsBudget> =
            std::collections::HashMap::new();
        for location in &self.storage_locations {
            let Some(cfg) = &location.io_budget else {
                continue;
            };
            let budget = Self::resolve_location_io_budget(&location.url, cfg).map_err(|err| {
                format!(
                    "storage.storage_locations[{}] io_budget: {err}",
                    location.url
                )
            })?;
            match resolved_by_url.get(location.url.as_str()) {
                Some(prev) if *prev != budget => {
                    return Err(format!(
                        "storage_locations contains duplicate URL {} with conflicting \
                         io_budgets ({prev:?} vs {budget:?}); a location URL must resolve \
                         to exactly one budget",
                        location.url
                    ));
                }
                Some(_) => {}
                None => {
                    resolved_by_url.insert(location.url.as_str(), budget);
                }
            }
        }
        Ok(())
    }

    /// Validate + register every location's `io_budget` into the storage leaf
    /// and emit the resolved table (the boot audit artifact, TD-IOBUDGET-1).
    /// Called authoritatively from `ProximaDB::start` beside the coarse-probe
    /// seed, so the TOML surface is effective regardless of engine construction
    /// order. Locations without `io_budget` keep their scheme defaults and the
    /// `PROXIMADB_DISK_CLASS` env hint.
    pub fn register_io_budgets(&self) -> anyhow::Result<()> {
        // Authoritative-REPLACE seeding: drain before re-registering so a
        // location dropped from a later config cannot keep serving its stale
        // budget to an in-process re-boot. Then refuse conflicting duplicate
        // URLs BEFORE registering anything (nothing wins a conflict).
        clear_location_budgets();
        self.validate_io_budget_conflicts()
            .map_err(|err| anyhow::anyhow!("{err}"))?;
        for location in &self.storage_locations {
            match &location.io_budget {
                Some(cfg) => {
                    let budget =
                        Self::resolve_location_io_budget(&location.url, cfg).map_err(|err| {
                            anyhow::anyhow!(
                                "storage.storage_locations[{}] io_budget: {err}",
                                location.url
                            )
                        })?;
                    register_location_budget(&location.url, budget);
                    info!(
                        url = %location.url,
                        min_bytes = budget.min,
                        target_bytes = budget.target,
                        max_bytes = budget.max,
                        "io_budget: registered per-location ranged-GET budget"
                    );
                }
                None => {
                    let scheme_default = IopsBudget::for_scheme_and_env(&location.url);
                    info!(
                        url = %location.url,
                        min_bytes = scheme_default.min,
                        target_bytes = scheme_default.target,
                        max_bytes = scheme_default.max,
                        "io_budget: no per-location budget configured; using scheme default"
                    );
                }
            }
        }
        Ok(())
    }

    /// Fail closed when memory-mapped I/O is enabled against a CLOUD object
    /// store (S3 / GCS / Azure ADLS/Blob). `mmap` requires local files; on a
    /// remote object store it is meaningless and unsafe — the docs mark
    /// `enable_mmap = false` as CRITICAL for cloud. Operators must opt out
    /// explicitly via `[storage.optimization] enable_mmap = false`. Local
    /// (`file://` / bare-path) backends are unaffected.
    pub fn validate_cloud_mmap(&self) -> anyhow::Result<()> {
        if !self.optimization.enable_mmap {
            return Ok(());
        }
        // Object-store schemes across S3 (s3/s3a), GCS (gs/gcs), and Azure
        // (az/azure/adls/abfs/abfss). Matches the schemes the FileSystem /
        // warehouse object-store backends accept.
        const CLOUD_SCHEMES: &[&str] = &[
            "s3", "s3a", "gs", "gcs", "az", "azure", "adls", "abfs", "abfss",
        ];
        for url in self.storage_urls() {
            let scheme = url
                .split_once("://")
                .map(|(s, _)| s)
                .unwrap_or("")
                .to_ascii_lowercase();
            if CLOUD_SCHEMES.contains(&scheme.as_str()) {
                anyhow::bail!(
                    "enable_mmap=true is invalid for cloud object storage '{url}' (scheme '{scheme}'): \
                     memory-mapped I/O requires local files. Set `[storage.optimization] enable_mmap = false` \
                     for cloud deployments."
                );
            }
        }
        Ok(())
    }
}

// Default implementation moved to line 115

#[cfg(test)]
mod cloud_mmap_guard_tests {
    use super::*;

    fn storage_with(url: &str, enable_mmap: bool) -> StorageConfig {
        let mut s = StorageConfig::default();
        s.storage_locations = vec![StorageLocation {
            url: url.to_string(),
            ..Default::default()
        }];
        s.optimization.enable_mmap = enable_mmap;
        s
    }

    #[test]
    fn cloud_url_with_mmap_is_rejected() {
        for url in [
            "s3://bucket/proximadb",
            "gs://bucket/proximadb",
            "adls://account/container",
            "abfs://container@account/p",
            "az://container/p",
        ] {
            assert!(
                storage_with(url, true).validate_cloud_mmap().is_err(),
                "{url} with enable_mmap=true must be rejected"
            );
        }
    }

    #[test]
    fn cloud_url_without_mmap_is_allowed() {
        assert!(
            storage_with("s3://bucket/proximadb", false)
                .validate_cloud_mmap()
                .is_ok()
        );
    }

    #[test]
    fn local_url_with_mmap_is_allowed() {
        assert!(
            storage_with("file:///nvme/proximadb", true)
                .validate_cloud_mmap()
                .is_ok()
        );
        assert!(
            storage_with("/nvme/proximadb", true)
                .validate_cloud_mmap()
                .is_ok()
        );
    }
}

/// User-facing write buffer configuration (from TOML files)
/// This is the simple configuration that users specify in their config files.
/// It gets converted to the internal WALConfig for the engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriteBufferUserConfig {
    /// Total write buffer size across all collections in MB
    pub write_buffer_size_mb: u64,
    /// Threshold in bytes to trigger flush when a collection's total unflushed data exceeds this
    pub memory_flush_size_bytes: usize,
    /// Memtable implementation type (BTree, SkipList)
    pub memtable_type: String,
    /// Sync mode for durability (PerBatch, Periodic, None)
    pub sync_mode: String,
    /// Directory for write-ahead log files
    pub write_buffer_directory: String,
    /// Enable write-ahead logging
    pub enable_wal: bool,
    /// Global manifest location (optional)
    pub global_manifest_url: Option<String>,

    /// ADR-069 S1 — opt-in local WAL root, e.g. `file:///waldisk`. When set, the
    /// per-collection WAL is written under `{dir}/{collection_id}/wal/` on a local
    /// reattachable disk, decoupled from the collection's object-store data
    /// assignment (WAL = high-frequency small appends; each object-store PUT is a
    /// paid write-txn). Resolves through the FilesystemFactory scheme, so a future
    /// value could be `s3://…` (S3 Express) etc. Overridden by env
    /// `PROXIMADB_WAL_LOCAL_DIR`. `None`/unset ⇒ legacy WAL-co-located-with-data.
    #[serde(default)]
    pub wal_local_dir: Option<String>,

    /// ADR-069 D2 — time-based WAL flush interval (seconds); 0 = disabled. The RPO
    /// floor: bounds worst-case loss on permanent local-volume loss to this interval for
    /// low-traffic collections that never reach the inline size trigger
    /// (`memory_flush_size_bytes`). With `sync_mode = "PerBatch"` (default) acknowledged
    /// writes are fsync'd, so process-kill RPO is 0 and this only bounds volume-loss RPO.
    /// Defaults to 300s so the inline size trigger (throughput) dominates and this stays
    /// a coarse safety net (fewer/larger segments → fewer object-store PUTs).
    #[serde(default = "default_flush_interval_secs")]
    pub flush_interval_secs: u64,
    /// TD-FLUSH-3 S1 — minimum PREDICTED segment size (MB) before the inline
    /// size trigger may flush; 0 = floor disabled (legacy behavior). Predicted
    /// bytes = Σ_records dim × k with k = 1/8 (RaBitQ) + 1 (SQ8) [+ 4 when the
    /// f32 tier is enabled] — i.e. the bytes the segment will actually occupy,
    /// NOT the serialized-record bytes `memory_flush_size_bytes` counts
    /// (~1.6 KB/record, which made "16 MB" ≈ 5 MB raw and produced the 104
    /// tiny Azurite segments → 1,812 GETs/query). The RPO timer
    /// (`flush_interval_secs`) and the capacity watermarks OVERRIDE the floor
    /// (a trickle collection still drains). Default 128 (≈0.9M vectors/segment
    /// at 128d) tuned for object-store reads; set 32 for local-disk-only
    /// deployments (per-scheme auto-detection is a follow-up).
    #[serde(default = "default_flush_floor_predicted_mb")]
    pub flush_floor_predicted_mb: u64,
    /// ADR-069 D6 — per-collection WAL size budget (bytes) for the capacity flush +
    /// backpressure watermarks; 0 = disabled.
    #[serde(default)]
    pub wal_max_bytes: usize,
    /// ADR-069 D3 — fraction of `wal_max_bytes` at which to force a flush.
    #[serde(default = "default_high_watermark_pct")]
    pub high_watermark_pct: f64,
    /// ADR-069 D3 — fraction of `wal_max_bytes` at which to apply write backpressure.
    #[serde(default = "default_critical_watermark_pct")]
    pub critical_watermark_pct: f64,
}

/// Serde default for [`WriteBufferUserConfig::high_watermark_pct`] (ADR-069 D3).
fn default_high_watermark_pct() -> f64 {
    0.80
}

/// Serde default for [`WriteBufferUserConfig::critical_watermark_pct`] (ADR-069 D3).
fn default_critical_watermark_pct() -> f64 {
    0.95
}

/// Serde default for [`WriteBufferUserConfig::flush_interval_secs`] (ADR-069 D2). The
/// RPO floor — coarse on purpose so the inline size trigger dominates throughput and this
/// only bounds volume-loss RPO for trickle workloads. See the field doc for rationale.
/// TD-FLUSH-3 demotion: this timer is explicitly NOT a durability mechanism (the WAL
/// replays on restart) and NOT a segment-sizing mechanism (`flush_floor_predicted_mb`
/// governs that) — it exists solely to bound restart replay time / volume-loss RPO for
/// collections that never reach the size floor.
fn default_flush_interval_secs() -> u64 {
    300
}

/// Serde default for [`WriteBufferUserConfig::flush_floor_predicted_mb`] (TD-FLUSH-3 S1).
/// 128 MB predicted ≈ 0.9M vectors/segment at 128d RaBitQ+SQ8 — the Parquet-convention
/// band that keeps cold GETs/query (∝ file count × ~26) bounded on object storage.
fn default_flush_floor_predicted_mb() -> u64 {
    128
}

impl Default for WriteBufferUserConfig {
    fn default() -> Self {
        Self {
            write_buffer_size_mb: 8192, // 8GB total across all collections
            memory_flush_size_bytes: 16 * 1024 * 1024, // 16MB per collection (aggregate)
            memtable_type: "BTree".to_string(),
            sync_mode: "PerBatch".to_string(),
            write_buffer_directory: "./data/write_buffer".to_string(),
            enable_wal: true,
            global_manifest_url: None,
            wal_local_dir: None,
            // ADR-069 D2: time floor defaults to 300s (RPO safety net; the inline
            // size trigger handles throughput). wal_max_bytes stays 0 → S4 backpressure OFF.
            flush_interval_secs: default_flush_interval_secs(),
            flush_floor_predicted_mb: default_flush_floor_predicted_mb(),
            wal_max_bytes: 0,
            high_watermark_pct: default_high_watermark_pct(),
            critical_watermark_pct: default_critical_watermark_pct(),
        }
    }
}

impl WriteBufferUserConfig {
    /// Convert user configuration to internal engine configuration.
    ///
    /// This is the SINGLE canonical `WriteBufferUserConfig -> WALConfig` conversion,
    /// shared by the embedded/library path (called directly here) and the server
    /// path (`SharedServices::convert_toml_to_wal_config` delegates to this then
    /// applies server-only bits). Both paths honor the TOML WAL config identically
    /// (TD-CONFIG-CONSOLIDATE-1 step 2 — previously the embedded path silently
    /// ignored sync_mode / memtable_type / flush thresholds / multi_disk).
    ///
    /// NOTE: `multi_disk.data_directories` is OVERWRITTEN from `storage_locations`
    /// by both callers (`database.rs` / `engine.rs`) after this returns, so the
    /// value set here is a fallback only. `enable_optimized_writer` is `false`
    /// (embedded default); the server path overrides it to `enable_wal`.
    pub fn to_engine_config(&self) -> crate::storage::persistence::write_ahead_log::WALConfig {
        use crate::storage::persistence::write_ahead_log::{
            WALConfig, WriteBufferStrategyType,
            config::{
                CompressionConfig, DiskDistributionStrategy, MemTableConfig, MemTableType,
                MultiDiskConfig, PerformanceConfig, SyncMode,
            },
        };
        let mib = 1024 * 1024;
        WALConfig {
            strategy_type: WriteBufferStrategyType::default(),
            memtable: MemTableConfig {
                global_memory_limit: self.write_buffer_size_mb as usize * mib,
                memtable_type: match self.memtable_type.to_lowercase().as_str() {
                    "btree" => MemTableType::BTree,
                    "skiplist" => MemTableType::SkipList,
                    _ => MemTableType::BTree,
                },
                ..MemTableConfig::default()
            },
            multi_disk: MultiDiskConfig {
                data_directories: vec![self.write_buffer_directory.clone()],
                distribution_strategy: DiskDistributionStrategy::RoundRobin,
                collection_affinity: true,
            },
            compression: CompressionConfig::default(),
            encryption: Default::default(), // TD-016: Encryption disabled by default
            performance: PerformanceConfig {
                memory_flush_size_bytes: self.memory_flush_size_bytes,
                global_flush_threshold: self.write_buffer_size_mb as usize * mib,
                sync_mode: match self.sync_mode.to_lowercase().as_str() {
                    "perbatch" | "always" => SyncMode::PerBatch,
                    "periodic" => SyncMode::Periodic,
                    "none" => SyncMode::Never,
                    _ => SyncMode::PerBatch,
                },
                // ADR-069/TD-WAL-1: the tiered time/capacity flush knobs.
                flush_interval_secs: self.flush_interval_secs,
                flush_floor_predicted_mb: self.flush_floor_predicted_mb,
                wal_max_bytes: self.wal_max_bytes,
                high_watermark_pct: self.high_watermark_pct,
                critical_watermark_pct: self.critical_watermark_pct,
                ..PerformanceConfig::default()
            },
            enable_mvcc: true,
            enable_ttl: true,
            enable_background_compaction: true,
            collection_overrides: std::collections::HashMap::new(),
            global_manifest_url: self.global_manifest_url.clone(),
            wal_local_dir: self.wal_local_dir.clone(),
            enable_optimized_writer: false, // embedded default; server overrides to enable_wal
            optimized_writer_batch_size: None,
            optimized_writer_batch_timeout_ms: None,
            optimized_writer_threads: None,
            optimized_writer_enable_combining: None,
        }
    }
}

/// SST (Sorted String Table) engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SstConfig {
    /// Number of levels in the SST tree
    pub level_count: u8,
    /// Minimum files before compaction triggers (DEPRECATED - use compaction_config)
    pub compaction_threshold: u32,

    /// Compaction configuration (overrides common config if specified)
    pub compaction_config: Option<CompactionConfig>,
    /// SSTable block size in KB. Configurable from TOML, defaults to 1MB.
    ///
    /// **Performance Guidelines:**
    /// - **256-512KB**: Good for memory-constrained environments
    /// - **1MB**: Optimal for EC2 GP2/GP3 and modern SSDs (default)
    /// - **2-4MB**: Best for high-throughput workloads with ample memory
    ///
    /// **Cloud-Optimized Block Size (MB):**
    /// - 3MB: Universal optimization for AWS EBS gp3/st1, Azure Premium SSD, GCS Standard
    /// - 2MB: Memory-constrained environments
    /// - 4MB: Very large sparse vector deployments
    ///
    /// Block size in KB (256-16384 KB range, default 2048 KB = 2MB)
    /// **Backward Compatibility:**
    /// Changing this value during restarts is safe. Each SSTable block stores its own
    /// length prefix [block_len:4][block_data], so existing files continue to work.
    /// Mixed block sizes within the same system are fully supported.
    pub block_size_kb: u32,
    /// Compaction strategy (leveled, tiered, unified)
    pub compaction_strategy: String,
    /// Compression algorithm (snappy, lz4, zstd, none)
    pub compression: String,
    /// Compression level (1-22 for ZSTD, ignored for other algorithms)
    pub compression_level: i32,
    /// Bloom filter configuration for SST files
    pub bloom_filter_config: Option<BloomFilterConfig>,
    /// Cache size for SST blocks in MB
    pub cache_size_mb: u64,
    /// TD-CACHE-4: budget (MB) for the per-segment invariants cache — the
    /// ADR-065 warm tier holding Region A (RaBitQ codes, `d/8+8` B/vector →
    /// ~24 MB per 1M vectors at 128d) + control prefix + footer; a hit elides
    /// the 3 control GETs per segment per query. Priority-then-recency
    /// eviction (TD-CACHE-2). 256 MB pins ~10× 1M-vector or ~100× 100k-vector
    /// collections. 0 disables. Env override:
    /// `PROXIMADB_SEGMENT_INVARIANTS_CACHE_MB`. Distinct from `cache_size_mb`
    /// (legacy SST block cache) and `decompression_cache_config` (decompressed
    /// block LRU).
    pub segment_invariants_cache_mb: u64,
    /// TD-CACHE-4 + ADR-065 update: budget (MB) for the survivor/OID range
    /// cache (Region B SQ8 rerank ranges, `d` B/vector, + Region D OID
    /// ranges). DEFAULT-ON at 1024 MB (decision 2026-07-24: ADR-065's
    /// default-OFF predated sound eviction (TD-CACHE-2) and was never armed in
    /// production anyway (TD-CACHE-4) — with it off, every novel query re-pays
    /// the full ranged-GET chain). 0 disables. Env override:
    /// `PROXIMADB_SURVIVOR_CACHE_BUDGET_MB`.
    pub survivor_cache_mb: u64,
    /// TD-DELVEC-1 WI-3c: budget (MB) for the per-segment OID→position resolver
    /// cache. 0 disables (resolver lazy-loaded on each delete — no in-memory
    /// cache). Env override: `PROXIMADB_OID_RESOLVER_CACHE_MB`.
    pub oid_resolver_cache_mb: u64,
    /// Optional shared persistent cache root on a local filesystem.
    /// `None` keeps the historical DRAM-only path. Fast SSD/NVMe media is
    /// recommended for latency, but the cache is correct on any local disk.
    /// Env override: `PROXIMADB_CACHE_LOCAL_DISK_PATH`.
    #[serde(default)]
    pub cache_local_disk_path: Option<String>,
    /// Shared persistent cache budget in GiB. Env override:
    /// `PROXIMADB_CACHE_LOCAL_DISK_MAX_GB`.
    #[serde(default = "default_cache_local_disk_max_gb")]
    pub cache_local_disk_max_gb: u64,
    /// Post-publication write-time cache population policy. Env override:
    /// `PROXIMADB_CACHE_ON_WRITE`.
    #[serde(default)]
    pub cache_on_write: CacheOnWritePolicy,
    /// Maximum files per level
    pub max_files_per_level: u32,
    /// Size multiplier between levels
    pub level_size_multiplier: f64,
    /// Maximum number of levels
    pub max_levels: u8,
    /// Number of background compaction threads
    pub background_thread_count: u32,
    /// TD-COMPACT-7 (ADR-076 D2): L0 admission **slowdown** watermark. When the
    /// live L0 file count for a collection reaches this, the flush path inserts
    /// a brief (~1ms) delay before writing the next L0, giving the async
    /// background compaction (TD-COMPACT-6) headroom to drain L0. This is the
    /// soft backpressure half of the producer/consumer rate-control loop.
    /// Default 6 — below `max_files_per_level` (10) so the `l0_stop_trigger`,
    /// not this, is the hard ceiling. 0 disables the slowdown. Env override:
    /// `PROXIMADB_L0_SLOWDOWN_TRIGGER`.
    #[serde(default = "default_l0_slowdown_trigger")]
    pub l0_slowdown_trigger: u32,
    /// TD-COMPACT-7 (ADR-076 D2): L0 admission **stop** watermark — the hard
    /// ceiling. When the live L0 file count reaches this, the flush refuses to
    /// write a new L0 and returns backpressure (the memtable retains the data,
    /// durable in the WAL; the write path slows until compaction drains L0).
    /// This is the hard backpressure half that bounds unbounded L0 growth during
    /// sustained ingest that outpaces compaction. Default = `max_files_per_level`
    /// (10); must be ≤ `max_files_per_level`. 0 disables the stop (flush always
    /// proceeds). Env override: `PROXIMADB_L0_STOP_TRIGGER`.
    #[serde(default = "default_l0_stop_trigger")]
    pub l0_stop_trigger: u32,
    /// Data directory for SST files
    pub data_directory: String,
    /// Enable memory-mapped I/O for SST files
    pub mmap_enabled: bool,
    /// Enable prefetching for sequential reads
    pub prefetch_enabled: bool,
    /// Prefetch size in KB
    pub prefetch_size_kb: u32,
    /// Decompression cache configuration
    pub decompression_cache_config:
        Option<crate::storage::engines::sst::decompression_cache::CacheConfig>,

    /// Vector encoding strategy: Controls how vectors are encoded in blocks
    ///
    /// # Available Strategies (Data-Driven from 12-Pattern Benchmarks):
    ///
    /// * `"FullVector"` (DEFAULT) - Row-wise encoding, fastest decode
    ///   - Compression: 18-20%
    ///   - Performance: Fastest decode in 8/12 configs, wins 5/12 (42%)
    ///   - Best for: Vector databases, WORM workloads, RAG, semantic search
    ///   - Use when: Query speed is critical (embeddings queried thousands of times)
    ///
    /// * `"GroupedBlock"` - Block-grouped encoding, best balanced
    ///   - Compression: 18-21%
    ///   - Performance: Wins 6/12 configs (50% - HIGHEST), fastest encode
    ///   - Best for: Mixed workloads, ETL, data pipelines
    ///   - Use when: Encode and decode are equally important
    ///
    /// * `"GroupedField"` - Field-grouped encoding, maximum compression
    ///   - Compression: 19-22% (BEST)
    ///   - Performance: Wins 1/12 (8%), slower decode
    ///   - Best for: Storage-critical, cost optimization, cold storage
    ///   - Use when: Storage costs dominate, data read infrequently
    ///
    /// * `"TransposeBlock"` - Transpose block-grouped encoding
    ///   - Compression: 19-20%
    ///   - Performance: Good for large batches (>4096 vectors)
    ///   - Best for: Batch analytics, columnar processing
    ///   - Use when: Processing large batches with dimensional correlation
    ///
    /// * `"TransposeField"` - ⚠️ NOT RECOMMENDED (very slow)
    ///   - Compression: 17-19%
    ///   - Performance: Slowest encode/decode (14-121ms encode, 4-170ms decode)
    ///   - Only use if: You have very specific dimensional correlation requirements
    ///   - Warning: 10-40x slower than other strategies, use with caution
    ///
    /// * `"Auto"` - Automatic selection based on workload
    ///   - Currently resolves to FullVector for vector database workloads
    ///   - Safe default choice for production
    ///
    /// # Configuration Examples:
    ///
    /// ```toml
    /// # config.toml
    ///
    /// # Vector database (default - fastest decode)
    /// [storage.sst_config]
    /// vector_encoding_strategy = "FullVector"
    ///
    /// # Balanced workload
    /// [storage.sst_config]
    /// vector_encoding_strategy = "GroupedBlock"
    ///
    /// # Storage optimization
    /// [storage.sst_config]
    /// vector_encoding_strategy = "GroupedField"
    /// ```
    ///
    /// # Performance Data:
    ///
    /// Based on comprehensive 12-pattern benchmark (sparse, gaussian, quantized,
    /// sinusoidal, random, clustered, time-series, dense, high-freq, power-law,
    /// binary, exponential) representing production ML embeddings:
    ///
    /// - OpenAI 1536D: FullVector 19.6% comp, 0.75ms decode (FASTEST)
    /// - BERT 1024D:   FullVector 19.1% comp, 4.71ms decode (FASTEST)
    /// - BERT 768D:    FullVector 18.9% comp, 4.06ms decode (FASTEST)
    /// - MiniLM 384D:  FullVector 18.5% comp, 0.26ms decode (FASTEST)
    ///
    /// Default: FullVector (optimal for vector databases with WORM characteristics)
    ///
    /// See: docs/performance/encoding_strategies.adoc for detailed guide
    #[serde(default = "default_vector_encoding_strategy")]
    pub vector_encoding_strategy: String,

    /// Block storage format: Controls how blocks are serialized to disk
    ///
    /// # Available Formats:
    ///
    /// * `"ProximaBlocks"` (DEFAULT) - ProximaDB's native block format
    ///   - Optimized for vector workloads with cache-line alignment
    ///   - B+ tree index for O(log n) ID lookups
    ///   - Supports quantization and compression
    ///   - Best for: Production vector databases
    ///
    /// * `"ArrowBlock"` - Arrow IPC based storage format
    ///   - Standard Arrow IPC files (compatible with PyArrow, DuckDB, Polars)
    ///   - Zero-copy reads via memory mapping
    ///   - Sidecar B+ tree index file (.idx)
    ///   - Best for: Interoperability with Arrow ecosystem
    ///
    /// # Configuration Example:
    ///
    /// ```toml
    /// [storage.sst_config]
    /// # Use Arrow IPC format for interoperability
    /// block_format = "ArrowBlock"
    ///
    /// # Use ProximaBlocks for production (default)
    /// block_format = "ProximaBlocks"
    /// ```
    ///
    /// Default: ProximaBlocks
    #[serde(default = "default_block_format")]
    pub block_format: String,

    /// Tier-migration configuration. When `None` or when the inner
    /// config has `enabled = false`, the engine's flush / search /
    /// compaction tiering hooks are no-ops.
    ///
    /// When `enabled = true`, the engine attaches an
    /// `SstTieringIntegration` at bootstrap (see
    /// `src/network/shared_services.rs`) and the policy engine begins
    /// observing per-collection access patterns, flush events, and
    /// compaction-time evaluations. Physical byte-movement between
    /// tier paths is still deferred — the policy engine generates
    /// migration tasks but no executor moves the bytes yet.
    ///
    /// Configure under `[storage.sst_config.tiering]` in TOML.
    /// Default: `None` (disabled).
    #[serde(default)]
    pub tiering: Option<crate::storage::engines::sst::tiering_integration::SstTieringConfig>,

    /// TD-SEARCH-2: inter-file search parallelism degree.
    /// `0` = 50% of CPU cores (wise default — leaves cores for flush/compaction).
    /// `1` = sequential (debugging).
    /// `n > 1` = exactly n concurrent segment-file scans.
    /// Hot-path override: `PROXIMADB_SEARCH_PARALLEL_FILES` env var.
    #[serde(default)]
    pub search_parallel_files: u16,

    /// TD-RDSTRAT-8 / COGS: the PAX coarse-probe (IVF) system — compaction
    /// trains + emits the A0 coarse directory (`enable_write_train`) and search
    /// probes the nearest cells (`enable_read_probe`, geometric `nprobe`).
    /// Configure under `[storage.sst_config.coarse_probe]` in TOML. Env overrides
    /// still win (`PROXIMADB_PAX_READ_COARSE_PROBE`, `..._NPROBE`,
    /// `PROXIMADB_PAX_WRITE_A0_TRAIN`). Default: `None` ⇒ `CoarseProbeConfig::default()`.
    #[serde(default)]
    pub coarse_probe: Option<CoarseProbeConfig>,
}

/// PAX coarse-probe (IVF) configuration — the master cloud-cost lever (COGS
/// measurement arc: IVF cuts GETs/query ~4× and per-tenant COGS ~4× vs full-scan,
/// recall ~0.98). Compaction trains + emits the A0 coarse directory when
/// `enable_write_train` is on; search probes the nearest `nprobe` cells when
/// `enable_read_probe` is on. `nprobe` defaults to the geometric
/// `ceil(sqrt(ncells) × multiplier)` (sub-linear in corpus: V^0.25; ~56× fewer
/// ranks than full-scan at 1M), clamped to `[nprobe_min, nprobe_max]`. Env vars
/// override per-field at call time (see `segment_format.rs`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoarseProbeConfig {
    /// READ kill-switch: search probes the A0 coarse directory. Default ON.
    pub enable_read_probe: bool,
    /// WRITE gate: compaction trains + emits the v3/A0 coarse directory. Default
    /// ON (the unblock — was OFF; mixed-read-safe: v1/v3 coexist, the read path
    /// probes only when `a0_len > 0`). Env: `PROXIMADB_PAX_WRITE_A0_TRAIN`.
    pub enable_write_train: bool,
    /// Geometric nprobe multiplier: `nprobe = ceil(sqrt(ncells) × multiplier)`.
    ///
    /// Default 2.0. The settled one-segment SIFT1M geometry has 30 cells:
    /// multiplier 1.0 probes 6 and measured **recall@10 = 0.9795**, below the
    /// 0.98 ratchet, while multiplier 2.0 probes 11 and measured 0.9822 on an
    /// independent 1,000-query slice (TD-SEARCH-3 §S5).
    ///
    /// Two caveats that the sweep records and this default should not obscure:
    ///
    /// * **The GET comparison is not planner-matched.** The 6-cell reading was
    ///   taken under the *local* planner (26.69 GET/query); the 11-cell reading
    ///   under the *Azure* planner (15.90). The multiplier change and the
    ///   bounded-coalescing change were measured together, so "2.0 costs
    ///   nothing in requests" does not follow from these two rows. Bytes did
    ///   rise, 24.66 → 48.82 MB/query.
    /// * Those are **diagnostic sweeps over the immutable acceptance segment**,
    ///   not a post-change evidence run; the full three-phase harness remains
    ///   the release gate.
    ///
    /// This is process-global: `coarse_probe_nprobe` takes only `k_c`, with no
    /// collection or query identity, so there is no per-collection retune
    /// without a signature change (TD-RDSTRAT-9 tracks the missing controller).
    /// Operators may lower this only with a representative recall gate.
    #[serde(default = "default_nprobe_multiplier")]
    pub nprobe_multiplier: f32,
    /// Minimum nprobe (floor). Default 3.
    #[serde(default = "default_nprobe_min")]
    pub nprobe_min: usize,
    /// Maximum nprobe (0 = unlimited; capped at ncells anyway). Default 0.
    #[serde(default)]
    pub nprobe_max: usize,
    /// WRITE gate (TD-IVF-3): floor on the coarse-PCA projection width used to
    /// train the A0 directory. **0 = disabled** (the legacy
    /// `max(1.5·log2 dim, 1.5·log2 k)` formula, which yields 10–14 components).
    ///
    /// Measurement puts the GET/query optimum at 32 (128-d) and 64 (384-d and
    /// 768-d), and wider widths are markedly more robust to poor cell geometry
    /// (degrading `k_c` 90→30 costs +21% at width 32 but only +2.6% at 64) —
    /// which matters because production `k_c` is derived from corpus size and
    /// varies over orders of magnitude. Ships at 0 pending the seed-sensitivity
    /// and PCA-conditioning bake. Env: `PROXIMADB_IVF_NCOMP_FLOOR`.
    #[serde(default)]
    pub ncomp_floor: usize,
}

/// Cache population performed only after a segment is atomically published.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum CacheOnWritePolicy {
    /// Preserve lazy read-through behavior.
    None,
    /// Seed control metadata and Region A.
    #[default]
    Invariant,
    /// Seed invariant data plus the complete SQ8 Region B parent range.
    All,
}

impl CacheOnWritePolicy {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "none" => Some(Self::None),
            "invariant" => Some(Self::Invariant),
            "all" => Some(Self::All),
            _ => None,
        }
    }

    pub fn includes_invariants(self) -> bool {
        matches!(self, Self::Invariant | Self::All)
    }

    pub fn includes_survivors(self) -> bool {
        matches!(self, Self::All)
    }
}

fn default_nprobe_multiplier() -> f32 {
    2.0
}
fn default_nprobe_min() -> usize {
    3
}

impl Default for CoarseProbeConfig {
    fn default() -> Self {
        Self {
            enable_read_probe: true,
            enable_write_train: true, // FLIP: emit A0 at compaction by default (COGS lever)
            nprobe_multiplier: default_nprobe_multiplier(),
            nprobe_min: default_nprobe_min(),
            nprobe_max: 0,
            ncomp_floor: 0, // TD-IVF-3: disabled until the bake clears it
        }
    }
}

fn default_block_format() -> String {
    "ProximaBlocks".to_string()
}

/// VIPER (columnar storage) engine configuration
///
/// **Note**: VIPER uses Parquet's native columnar format and does NOT use ProximaDataBlocks.
/// Therefore, it does not have `vector_encoding_strategy` (that's only for ProximaDataBlocks
/// which use columnar encoding within blocks, as used by SST engine).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViperConfig {
    /// Parquet file configuration
    pub row_group_size: usize,
    /// Compression for Parquet files (snappy, gzip, lz4, zstd)
    pub compression: String,
    /// Compression level (1-22 for ZSTD, ignored for other algorithms)
    pub compression_level: i32,
    /// Enable statistics in Parquet files
    pub enable_statistics: bool,
    /// Data directory for VIPER files
    pub data_directory: String,
    /// Cache size for columnar data in MB
    pub cache_size_mb: u64,

    /// Compaction configuration (overrides common config if specified)
    pub compaction_config: Option<CompactionConfig>,
}

impl Default for ViperConfig {
    fn default() -> Self {
        Self {
            row_group_size: 65536,           // ~32MB row groups for 128D vectors
            compression: "zstd".to_string(), // ZSTD for better compression
            compression_level: 3,            // Balanced speed/compression
            enable_statistics: true,
            data_directory: "./data/viper_data".to_string(),
            cache_size_mb: 512,
            compaction_config: None, // Use common config by default
        }
    }
}
#[allow(dead_code)]
fn default_compression_level() -> i32 {
    3 // Balanced compression level
}

fn default_vector_encoding_strategy() -> String {
    "FullVector".to_string() // Default to FullVector (best for vector databases - WORM workloads)
}

/// Default background compaction thread count: 50% of available cores,
/// clamped to [1, 4].
///
/// `available_parallelism()` respects cgroup CPU limits, so a container
/// capped at 4 vCPUs (e.g. an Azure B4ms node with a 3.5-CPU pod limit)
/// gets 2 compaction threads — a fixed 4 was measured to saturate small
/// nodes and starve concurrent search/gRPC/REST. The cap keeps the
/// pre-existing ceiling of 4 on large hosts, where compaction parallelism
/// beyond that showed no benefit and steals cores from queries.
fn default_l0_slowdown_trigger() -> u32 {
    6
}

fn default_l0_stop_trigger() -> u32 {
    10
}

fn default_cache_local_disk_max_gb() -> u64 {
    100
}

fn default_background_thread_count() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32 / 2)
        .unwrap_or(2)
        .clamp(1, 4)
}

// BloomFilterConfig moved to core::bloom module for polymorphic design
// Re-export for backward compatibility
pub use crate::core::bloom::BloomFilterConfig;

impl Default for SstConfig {
    fn default() -> Self {
        Self {
            level_count: 7,
            compaction_threshold: 5,
            compaction_config: None, // Use common config by default
            block_size_kb: 256, // 256KB default - optimized for low-latency random access on NVMe
            compaction_strategy: "leveled".to_string(),
            compression: "lz4".to_string(), // LZ4 default - 7% faster than no compression (measured)
            compression_level: 3,           // LZ4 compression level
            bloom_filter_config: Some(BloomFilterConfig::default()),
            cache_size_mb: 128,
            segment_invariants_cache_mb: 256,
            survivor_cache_mb: 1024,
            oid_resolver_cache_mb: 64,
            cache_local_disk_path: None,
            cache_local_disk_max_gb: default_cache_local_disk_max_gb(),
            cache_on_write: CacheOnWritePolicy::default(),
            max_files_per_level: 10,
            level_size_multiplier: 10.0,
            max_levels: 7,
            background_thread_count: default_background_thread_count(),
            // TD-COMPACT-7 D2: L0 admission watermarks (soft + hard backpressure).
            l0_slowdown_trigger: default_l0_slowdown_trigger(),
            l0_stop_trigger: default_l0_stop_trigger(),
            data_directory: "./sst_data".to_string(),
            mmap_enabled: true,
            prefetch_enabled: true,
            prefetch_size_kb: 64,
            decompression_cache_config: Some(
                crate::storage::engines::sst::decompression_cache::CacheConfig::default(),
            ),
            vector_encoding_strategy: default_vector_encoding_strategy(),
            block_format: default_block_format(),
            tiering: None,            // Default: tier-migration disabled
            search_parallel_files: 0, // TD-SEARCH-2: 0 = 50% of CPU cores
            coarse_probe: None,       // CoarseProbeConfig::default() applied at SstEngine init
        }
    }
}

impl SstConfig {
    /// Validate SST configuration parameters for optimal performance and correctness
    ///
    /// Note: block_size_kb changes are backward compatible since each SSTable block
    /// stores its own length prefix [block_len:4][block_data], allowing mixed block sizes.
    pub fn validate(&self) -> Result<(), String> {
        if self.level_count == 0 {
            return Err("level_count must be greater than 0".to_string());
        }
        if self.compaction_threshold == 0 {
            return Err("compaction_threshold must be greater than 0".to_string());
        }
        if let Some(compaction) = &self.compaction_config {
            if compaction.higher_level_file_threshold < 2 {
                return Err("compaction higher_level_file_threshold must be at least 2".to_string());
            }
            if !compaction.level_multiplier.is_finite() || compaction.level_multiplier <= 0.0 {
                return Err("compaction level_multiplier must be finite and positive".to_string());
            }
            if !(1.0..=1.5).contains(&compaction.level_multiplier) {
                tracing::warn!(
                    level_multiplier = compaction.level_multiplier,
                    "SST compaction level_multiplier is outside the vector-search \
                     recommendation [1.0, 1.5]; segment accumulation or rewrite churn may result"
                );
            }
            if !compaction.memory_amplification_factor.is_finite()
                || compaction.memory_amplification_factor < 1.0
            {
                return Err(
                    "compaction memory_amplification_factor must be finite and at least 1.0"
                        .to_string(),
                );
            }
            if !compaction.memory_budget_fraction.is_finite()
                || !(0.0..=1.0).contains(&compaction.memory_budget_fraction)
                || compaction.memory_budget_fraction == 0.0
            {
                return Err(
                    "compaction memory_budget_fraction must be finite and in (0.0, 1.0]"
                        .to_string(),
                );
            }
            if !compaction.available_memory_fraction.is_finite()
                || !(0.0..=1.0).contains(&compaction.available_memory_fraction)
                || compaction.available_memory_fraction == 0.0
            {
                return Err(
                    "compaction available_memory_fraction must be finite and in (0.0, 1.0]"
                        .to_string(),
                );
            }
            if !compaction.spill_scratch_amplification_factor.is_finite()
                || compaction.spill_scratch_amplification_factor < 1.0
            {
                return Err(
                    "compaction spill_scratch_amplification_factor must be finite and at least 1.0"
                        .to_string(),
                );
            }
            if !compaction.spill_available_disk_fraction.is_finite()
                || !(0.0..=1.0).contains(&compaction.spill_available_disk_fraction)
                || compaction.spill_available_disk_fraction == 0.0
            {
                return Err(
                    "compaction spill_available_disk_fraction must be finite and in (0.0, 1.0]"
                        .to_string(),
                );
            }
            if compaction.spill_enabled {
                if compaction.spill_working_memory_mb == 0 {
                    return Err(
                        "compaction spill_working_memory_mb must be greater than zero when spill is enabled"
                            .to_string(),
                    );
                }
                let spill_directory = compaction
                    .spill_directory
                    .as_deref()
                    .map(str::trim)
                    .filter(|path| !path.is_empty())
                    .ok_or_else(|| {
                        "compaction spill_directory is required when spill is enabled".to_string()
                    })?;
                if spill_directory.contains("://") {
                    return Err(
                        "compaction spill_directory must be a local filesystem path, not a URL"
                            .to_string(),
                    );
                }
            }
        }
        if let Some(path) = &self.cache_local_disk_path {
            if path.trim().is_empty() {
                return Err("cache_local_disk_path must not be empty".to_string());
            }
            if path.contains("://") {
                return Err(
                    "cache_local_disk_path must be a local filesystem path, not a URL".to_string(),
                );
            }
        }

        // Validate block size for optimal performance and storage compatibility
        if self.block_size_kb < 256 {
            return Err(
                "block_size_kb must be at least 256KB for reasonable I/O performance".to_string(),
            );
        }
        if self.block_size_kb > 4096 {
            return Err("block_size_kb should not exceed 4096KB (4MB) to avoid excessive memory usage per block".to_string());
        }

        // Performance recommendations for common deployment scenarios
        match self.block_size_kb {
            256..=512 => {
                // 256-512KB - Optimized for random access and point queries
                info!(
                    "block_size_kb={}KB - Optimized for random access and NVMe alignment",
                    self.block_size_kb
                );
            }
            513..=1023 => {
                // 512KB-1MB range
                info!(
                    "block_size_kb={}KB - Balanced for mixed random/sequential access",
                    self.block_size_kb
                );
            }
            1024 => {
                // 1MB - New default for balanced workloads
                info!("block_size_kb=1024KB (1MB) - Default balanced configuration");
            }
            1025..=2047 => {
                // 1-2MB range
                info!(
                    "block_size_kb={}KB - Good for cloud storage patterns",
                    self.block_size_kb
                );
            }
            2048..=4096 => {
                // 2-4MB - Maximum allowed, optimized for sequential scans
                info!(
                    "block_size_kb={}KB ({}MB) - Optimized for sequential scans and large transfers",
                    self.block_size_kb,
                    self.block_size_kb / 1024
                );
            }
            _ => {
                // Should not reach here due to validation
                info!(
                    "block_size_kb={}KB - Out of recommended range",
                    self.block_size_kb
                );
            }
        }

        Ok(())
    }

    /// Get block size in bytes for internal use
    pub fn block_size_bytes(&self) -> usize {
        (self.block_size_kb as usize) * 1024
    }

    /// Create test-specific SST configuration with smaller block sizes for quantization testing
    /// This helps demonstrate quantization clustering with smaller blocks while keeping
    /// server default at 2048KB for production performance
    pub fn test_config_256kb() -> Self {
        Self {
            block_size_kb: 256,              // Small blocks for quantization clustering tests
            compression: "zstd".to_string(), // Zstd compression for tests
            compression_level: 3,            // Balanced compression level
            ..Default::default()
        }
    }

    /// Create test-specific SST configuration with 512KB blocks
    pub fn test_config_512kb() -> Self {
        Self {
            block_size_kb: 512,
            compression: "zstd".to_string(), // Zstd compression for tests
            compression_level: 3,            // Balanced compression level
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod sst_config_validation_tests {
    use super::{CompactionConfig, SstConfig, StorageConfig};

    #[test]
    fn compaction_multiplier_guard_is_soft_but_invalid_values_fail() {
        let mut config = SstConfig {
            compaction_config: Some(CompactionConfig {
                level_multiplier: 2.0,
                ..CompactionConfig::default()
            }),
            ..SstConfig::default()
        };
        assert!(
            config.validate().is_ok(),
            "2.0 warns but remains overridable"
        );

        if let Some(compaction) = &mut config.compaction_config {
            compaction.level_multiplier = 0.0;
        }
        assert!(config.validate().is_err());

        if let Some(compaction) = &mut config.compaction_config {
            compaction.level_multiplier = 1.0;
            compaction.higher_level_file_threshold = 1;
        }
        assert!(config.validate().is_err());
    }

    #[test]
    fn compaction_memory_policy_fails_closed_on_invalid_ratios() {
        let mut config = SstConfig::default();
        let compaction = config
            .compaction_config
            .get_or_insert_with(CompactionConfig::default);
        compaction.memory_amplification_factor = 0.99;
        assert!(config.validate().is_err());

        let compaction = config
            .compaction_config
            .get_or_insert_with(CompactionConfig::default);
        compaction.memory_amplification_factor = 12.0;
        compaction.memory_budget_fraction = 0.0;
        assert!(config.validate().is_err());

        let compaction = config
            .compaction_config
            .get_or_insert_with(CompactionConfig::default);
        compaction.memory_budget_fraction = 0.25;
        compaction.available_memory_fraction = 1.01;
        assert!(config.validate().is_err());
    }

    #[test]
    fn compaction_spill_policy_is_default_off_and_requires_local_scratch() {
        let defaults = CompactionConfig::default();
        assert!(!defaults.spill_enabled);
        assert_eq!(defaults.spill_scratch_amplification_factor, 10.0);

        let mut config = SstConfig::default();
        let compaction = config
            .compaction_config
            .get_or_insert_with(CompactionConfig::default);
        compaction.spill_enabled = true;
        assert!(config.validate().is_err());

        let compaction = config
            .compaction_config
            .get_or_insert_with(CompactionConfig::default);
        compaction.spill_directory = Some("az://container/scratch".to_string());
        assert!(config.validate().is_err());

        let compaction = config
            .compaction_config
            .get_or_insert_with(CompactionConfig::default);
        compaction.spill_directory = Some("/var/lib/proximadb/compaction".to_string());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn persistent_cache_path_must_be_a_local_path() {
        let config = SstConfig {
            cache_local_disk_path: Some("file:///tmp/cache".to_string()),
            ..SstConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn common_compaction_config_reaches_sst_runtime() {
        let mut storage = StorageConfig::default();
        storage.compaction_config.higher_level_file_threshold = 2;
        storage.compaction_config.level_multiplier = 1.25;

        let effective = storage.effective_sst_config();
        let compaction = effective
            .compaction_config
            .expect("common compaction config must be inherited");

        assert_eq!(compaction.higher_level_file_threshold, 2);
        assert_eq!(compaction.level_multiplier, 1.25);
    }

    #[test]
    fn explicit_sst_compaction_config_overrides_common_policy() {
        let mut storage = StorageConfig::default();
        storage.compaction_config.higher_level_file_threshold = 2;
        let mut engine_override = CompactionConfig::default();
        engine_override.higher_level_file_threshold = 4;
        storage
            .sst_config
            .get_or_insert_with(SstConfig::default)
            .compaction_config = Some(engine_override);

        let effective = storage.effective_sst_config();
        assert_eq!(
            effective
                .compaction_config
                .expect("SST override must be retained")
                .higher_level_file_threshold,
            4
        );
    }
}

pub use proximadb_config::ApiConfig;

// RETIRED (TD-CONFIG-CONSOLIDATE-1): WalStorageConfig + WalDistributionStrategy — dead twin of WriteBufferUserConfig; do not re-add.

// Helper functions for serde defaults
#[allow(dead_code)]
fn default_collection_affinity() -> bool {
    true
}
#[allow(dead_code)]
fn default_memory_flush_size() -> usize {
    2 * 1024 * 1024 // 2MB - reduced for faster recovery as per CLAUDE.md
}
#[allow(dead_code)]
fn default_global_flush_threshold() -> usize {
    4 * 1024 * 1024 * 1024 // 4GB - recommended for global memory threshold
}
#[allow(dead_code)]
fn default_strategy_type() -> Option<String> {
    None
}
#[allow(dead_code)]
fn default_memtable_type() -> Option<String> {
    None
}
#[allow(dead_code)]
fn default_sync_mode() -> Option<String> {
    None
}
#[allow(dead_code)]
fn default_write_buffer_size_mb() -> Option<usize> {
    None
}
#[allow(dead_code)]
fn default_concurrent_flushes() -> Option<usize> {
    None
}
#[allow(dead_code)]
fn default_global_shrink_factor() -> Option<f64> {
    Some(0.4) // 40% shrink factor - recommended for global threshold management
}

/// Query processing configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QueryConfig {
    /// RL-based adaptive query planner configuration
    #[serde(default)]
    pub rl_planner: RLPlannerConfig,
    /// Cross-modal reranking policy. Defaults are neutral and disabled.
    #[serde(default)]
    pub reranking: RerankConfig,
}

impl QueryConfig {
    /// Validate query-scoped runtime policy before the server starts.
    pub fn validate(&self) -> anyhow::Result<()> {
        self.reranking.validate()
    }
}

/// RL-based Adaptive Query Planner Configuration
///
/// Controls how the reinforcement learning query planner learns and selects
/// optimal execution paths across storage engines, indexes, and quantization strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RLPlannerConfig {
    /// Enable RL-based planning (false = use static heuristics)
    #[serde(default = "default_rl_enabled")]
    pub enabled: bool,

    /// Use Thompson Sampling (true) or epsilon-greedy (false) for action selection
    #[serde(default = "default_thompson_sampling")]
    pub thompson_sampling: bool,

    /// Exploration rate for epsilon-greedy fallback (0.0 - 1.0)
    #[serde(default = "default_exploration_rate")]
    pub exploration_rate: f32,

    /// Size of experience replay buffer for batch learning
    #[serde(default = "default_experience_buffer_size")]
    pub experience_buffer_size: usize,

    /// Number of experiences before batch policy update
    #[serde(default = "default_batch_update_interval")]
    pub batch_update_interval: usize,

    /// Log all query executions to JSONL for offline analysis
    #[serde(default = "default_log_all_executions")]
    pub log_all_executions: bool,

    /// Path for execution logs (JSONL format) - None for no file logging
    #[serde(default)]
    pub log_path: Option<String>,

    /// Default optimization goal: MinLatency, MaxRecall, MaxThroughput, Balanced
    #[serde(default = "default_optimization_goal")]
    pub default_goal: String,
}

fn default_rl_enabled() -> bool {
    true
}

fn default_thompson_sampling() -> bool {
    true
}

fn default_exploration_rate() -> f32 {
    0.1
}

fn default_experience_buffer_size() -> usize {
    10_000
}

fn default_batch_update_interval() -> usize {
    100
}

fn default_log_all_executions() -> bool {
    true
}

fn default_optimization_goal() -> String {
    "Balanced".to_string()
}

impl Default for RLPlannerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            thompson_sampling: true,
            exploration_rate: 0.1,
            experience_buffer_size: 10_000,
            batch_update_interval: 100,
            log_all_executions: true,
            log_path: None,
            default_goal: "Balanced".to_string(),
        }
    }
}

impl RLPlannerConfig {
    /// Convert to the RL planner module's config type
    pub fn to_rl_planner_config(&self) -> crate::query::rl_planner::RLPlannerConfig {
        use crate::query::rl_planner::OptimizationGoal;

        let goal = match self.default_goal.to_lowercase().as_str() {
            "minlatency" | "min_latency" => OptimizationGoal::MinLatency,
            "maxrecall" | "max_recall" => OptimizationGoal::MaxRecall,
            "maxthroughput" | "max_throughput" => OptimizationGoal::MaxThroughput,
            _ => OptimizationGoal::Balanced,
        };

        crate::query::rl_planner::RLPlannerConfig {
            enabled: self.enabled,
            exploration_rate: self.exploration_rate,
            thompson_sampling: self.thompson_sampling,
            experience_buffer_size: self.experience_buffer_size,
            batch_update_interval: self.batch_update_interval,
            log_all_executions: self.log_all_executions,
            log_path: self.log_path.clone(),
            default_goal: goal,
        }
    }
}
