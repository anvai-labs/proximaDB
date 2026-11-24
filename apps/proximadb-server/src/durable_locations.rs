//! Durable state a server writes OUTSIDE its leased storage roots.
//!
//! Some durable locations are chosen by per-process environment (k8s/ECS task
//! spec, systemd `Environment=`) or by config sections the ownership lease does
//! not cover. A cold backup run from an operator's shell cannot see the
//! server's environment, so the SERVING process records what it actually
//! resolved — absolute paths, from its own env + config + cwd — in
//! `<data_dir>/.proximadb-durable-locations.json` at boot. Backup reads that
//! record (and also checks its own environment) and refuses if any recorded
//! local location lies outside every leased root, instead of silently
//! producing a backup that omits it.

use std::path::{Path, PathBuf};

use proximadb::core::Config;
use proximadb::storage::persistence::write_ahead_log::{
    ResolvedDurabilityMode,
    config::{DurabilityMode, resolve_durability_mode},
};
use serde::{Deserialize, Serialize};

pub(crate) const DURABLE_LOCATIONS_FILE: &str = ".proximadb-durable-locations.json";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct DurableLocation {
    /// Where the location came from (config key or env var) — for the operator.
    pub(crate) source: String,
    /// The location as resolved by the process that recorded it: an absolute
    /// local path, or a remote URL (object storage is durable on its own).
    pub(crate) location: String,
    /// TD-SPINE-5 (ADR-098 D3): the WAL durability mode in force for this
    /// location — `Some` on WAL-serving entries, `None` on unrelated ones.
    /// Additive (`serde(default)`): records written before the field existed
    /// read back with `None`.
    #[serde(default)]
    pub(crate) durability_mode: Option<DurabilityMode>,
}

/// TD-SPINE-5: the boot durability-mode resolution, computed from `config`
/// and the INJECTED env (test seam) using the same pure ladder the engine
/// uses: declared (`PROXIMADB_WAL_DURABILITY_MODE` > TOML
/// `durability_mode`) > a set `wal_local_dir` (`PROXIMADB_WAL_LOCAL_DIR` >
/// TOML) ⇒ `local_volume` > the scheme of `write_buffer_directory`. The
/// `{data_dir}/wal` default root applies in `local_volume` with no explicit
/// root.
fn resolve_wal_durability(
    config: &Config,
    env: impl Fn(&str) -> Option<String>,
) -> ResolvedDurabilityMode {
    let env = |key: &str| env(key).filter(|value| !value.trim().is_empty());
    let declared = env("PROXIMADB_WAL_DURABILITY_MODE")
        .and_then(|value| DurabilityMode::parse(&value))
        .or(config.storage.wal_config.durability_mode);
    let wal_local_dir = env("PROXIMADB_WAL_LOCAL_DIR").or_else(|| {
        config
            .storage
            .wal_config
            .wal_local_dir
            .as_deref()
            .map(|dir| dir.trim().trim_end_matches('/').to_string())
            .filter(|dir| !dir.is_empty())
    });
    resolve_durability_mode(
        declared,
        wal_local_dir.as_deref(),
        &config.storage.wal_config.write_buffer_directory,
        Some(&config.server.data_dir),
    )
}

/// Durable locations the lease does not cover, resolved from `config` and the
/// environment `env` (injected for tests). Relative local paths are made
/// absolute against the CURRENT process's cwd.
pub(crate) fn effective_durable_locations(
    config: &Config,
    env: impl Fn(&str) -> Option<String>,
) -> Vec<DurableLocation> {
    let env = |key: &str| env(key).filter(|value| !value.trim().is_empty());
    let mut found = Vec::new();
    let mut push = |source: &str, location: String, durability_mode: Option<DurabilityMode>| {
        found.push(DurableLocation {
            source: source.to_string(),
            location: absolutize(&location),
            durability_mode,
        });
    };

    if let Some(root) = config.queue.as_ref().and_then(|queue| queue.root.clone()) {
        push("[queue] root", root, None);
    }
    if let Some(root) = env("PROXIMADB_QUEUE_ROOT") {
        push("PROXIMADB_QUEUE_ROOT", root, None);
    }
    // TD-SPINE-5: the WAL durability mode rides every WAL-serving entry, and
    // the resolution itself is an entry (source names the mode + how it was
    // resolved; location is the effective WAL root: an explicit root, the
    // `{data_dir}/wal` default, or the primary WAL location under
    // object_store_serverless).
    let durability = resolve_wal_durability(config, env);
    if let Some(dir) = env("PROXIMADB_WAL_LOCAL_DIR") {
        push("PROXIMADB_WAL_LOCAL_DIR", dir, Some(durability.mode));
    }
    if let Some(dir) = config
        .storage
        .wal_config
        .wal_local_dir
        .as_deref()
        .map(|dir| dir.trim().trim_end_matches('/').to_string())
        .filter(|dir| !dir.is_empty())
    {
        push(
            "[storage.wal_config] wal_local_dir",
            dir,
            Some(durability.mode),
        );
    }
    {
        let effective_root = durability
            .effective_wal_root(
                config
                    .storage
                    .wal_config
                    .wal_local_dir
                    .as_deref()
                    .map(|dir| dir.trim().trim_end_matches('/'))
                    .filter(|dir| !dir.is_empty()),
            )
            .unwrap_or_else(|| {
                config
                    .storage
                    .wal_config
                    .write_buffer_directory
                    .trim_end_matches('/')
                    .to_string()
            });
        push(
            &format!(
                "[storage.wal_config] durability_mode ({} — {})",
                durability.mode,
                durability.source.label(),
            ),
            effective_root,
            Some(durability.mode),
        );
    }
    if let Some(url) = config.storage.wal_config.global_manifest_url.clone() {
        push("[storage.wal_config] global_manifest_url", url, None);
    }
    if let Some(path) = env("PROXIMADB_CORPUS_VERSION_PATH") {
        push("PROXIMADB_CORPUS_VERSION_PATH", path, None);
    }
    // io_trace sink spool — mirrors `ResolvedIoTraceSinkConfig::try_resolve`'s
    // env > TOML > default ladder for the enable switch and the directory.
    let sink = config
        .observability
        .as_ref()
        .and_then(|observability| observability.io_trace_sink.clone())
        .unwrap_or_default();
    let sink_enabled = env("PROXIMADB_IO_TRACE_SINK")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(sink.enabled);
    if sink_enabled {
        let dir = env("PROXIMADB_IO_TRACE_SINK_LOCAL_DIR")
            .or(sink.local_dir)
            .unwrap_or_else(|| "./log/io_trace".to_string());
        push("io_trace sink local_dir", dir, None);
    }
    if let Some(path) = config
        .query
        .as_ref()
        .and_then(|query| query.rl_planner.log_path.clone())
    {
        push("[query.rl_planner] log_path", path, None);
    }
    found
}

/// Local path of a location, if it is local (`file://` or scheme-less).
pub(crate) fn local_path(location: &str) -> Option<PathBuf> {
    if let Some(path) = location.strip_prefix("file://") {
        Some(PathBuf::from(path))
    } else if !location.contains("://") {
        Some(PathBuf::from(location))
    } else {
        None
    }
}

fn absolutize(location: &str) -> String {
    match local_path(location) {
        Some(path) => std::path::absolute(&path)
            .unwrap_or(path)
            .display()
            .to_string(),
        None => location.to_string(),
    }
}

/// Record this serving process's effective durable locations (atomic write).
/// Called at boot, after the ownership lease is held.
pub(crate) fn record(config: &Config) -> anyhow::Result<()> {
    let locations = effective_durable_locations(config, |key| std::env::var(key).ok());
    write_record(&config.server.data_dir, &locations)
}

pub(crate) fn write_record(data_dir: &Path, locations: &[DurableLocation]) -> anyhow::Result<()> {
    let target = data_dir.join(DURABLE_LOCATIONS_FILE);
    let tmp = data_dir.join(format!("{DURABLE_LOCATIONS_FILE}.tmp"));
    let body = serde_json::to_vec_pretty(locations)?;
    std::fs::write(&tmp, body).map_err(|e| anyhow::anyhow!("write {}: {e}", tmp.display()))?;
    std::fs::File::open(&tmp)
        .and_then(|file| file.sync_all())
        .map_err(|e| anyhow::anyhow!("fsync {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, &target)
        .map_err(|e| anyhow::anyhow!("publish {}: {e}", target.display()))?;
    Ok(())
}

/// The record a server left in `data_dir`, if any.
pub(crate) fn read_record(data_dir: &Path) -> anyhow::Result<Option<Vec<DurableLocation>>> {
    let path = data_dir.join(DURABLE_LOCATIONS_FILE);
    match std::fs::read(&path) {
        Ok(bytes) => {
            Ok(Some(serde_json::from_slice(&bytes).map_err(|e| {
                anyhow::anyhow!("parse {}: {e}", path.display())
            })?))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(anyhow::anyhow!("read {}: {e}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_of(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |key| {
            pairs
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, v)| (*v).to_string())
        }
    }

    #[test]
    fn env_overrides_are_recorded_absolute() {
        let config = Config::default();
        let found = effective_durable_locations(
            &config,
            env_of(&[
                ("PROXIMADB_WAL_LOCAL_DIR", "relative/wal"),
                ("PROXIMADB_QUEUE_ROOT", "s3://bucket/queue"),
                ("PROXIMADB_CORPUS_VERSION_PATH", "/abs/corpus"),
            ]),
        );
        let wal = found
            .iter()
            .find(|loc| loc.source == "PROXIMADB_WAL_LOCAL_DIR")
            .expect("wal");
        assert!(Path::new(&wal.location).is_absolute(), "{}", wal.location);
        assert!(found.iter().any(|loc| loc.location == "s3://bucket/queue"));
        assert!(found.iter().any(|loc| loc.location == "/abs/corpus"));
    }

    #[test]
    fn io_trace_spool_recorded_only_when_enabled() {
        let config = Config::default();
        let off = effective_durable_locations(&config, env_of(&[]));
        assert!(off.iter().all(|loc| !loc.source.contains("io_trace")));
        let on = effective_durable_locations(&config, env_of(&[("PROXIMADB_IO_TRACE_SINK", "1")]));
        assert!(
            on.iter().any(
                |loc| loc.source.contains("io_trace") && loc.location.ends_with("log/io_trace")
            )
        );
    }

    #[test]
    fn record_round_trips() {
        let temp = tempfile::tempdir().expect("tempdir");
        let locations = vec![DurableLocation {
            source: "x".into(),
            location: "/y".into(),
            durability_mode: Some(DurabilityMode::LocalVolume),
        }];
        write_record(temp.path(), &locations).expect("write");
        assert_eq!(read_record(temp.path()).expect("read"), Some(locations));
        assert_eq!(
            read_record(&temp.path().join("missing")).expect("read missing"),
            None
        );
    }

    /// TD-SPINE-5 negative control on the additive field: a record written
    /// BEFORE the mode existed (no `durability_mode` key in the JSON) reads
    /// back with `None` — mixed-version records stay readable.
    #[test]
    fn pre_spine5_record_reads_back_without_a_mode() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            temp.path().join(DURABLE_LOCATIONS_FILE),
            br#"[{"source":"x","location":"/y"}]"#,
        )
        .expect("write legacy record");
        let read = read_record(temp.path())
            .expect("read")
            .expect("some record");
        assert_eq!(
            read,
            vec![DurableLocation {
                source: "x".into(),
                location: "/y".into(),
                durability_mode: None,
            }]
        );
    }

    /// TD-SPINE-5 (ADR-098 D3): the record names the resolved durability mode
    /// and the effective WAL root for a default local deployment —
    /// `local_volume` inferred from the (local) `write_buffer_directory`
    /// scheme, root defaulted to `{data_dir}/wal`.
    #[test]
    fn wal_durability_mode_recorded_local_volume_default_root() {
        let temp = tempfile::tempdir().expect("tempdir");
        let mut config = Config::default();
        config.server.data_dir = temp.path().to_path_buf();
        let found = effective_durable_locations(&config, env_of(&[]));
        let mode_entry = found
            .iter()
            .find(|loc| loc.source.contains("durability_mode"))
            .expect("durability mode entry");
        assert_eq!(
            mode_entry.durability_mode,
            Some(DurabilityMode::LocalVolume)
        );
        assert!(mode_entry.source.contains("local_volume"));
        assert!(
            mode_entry.location.ends_with("/wal"),
            "defaulted root {{data_dir}}/wal: {}",
            mode_entry.location
        );
        assert!(
            mode_entry
                .location
                .starts_with(temp.path().display().to_string().as_str()),
            "root lives under data_dir: {}",
            mode_entry.location
        );
    }

    /// TD-SPINE-5 (ADR-098 D3): an object-store primary WAL location infers
    /// `object_store_serverless` (the ADR-063 posture) and keeps the
    /// catalog-derived base location — no local default root is recorded.
    #[test]
    fn wal_durability_mode_recorded_object_store_serverless() {
        let temp = tempfile::tempdir().expect("tempdir");
        let mut config = Config::default();
        config.server.data_dir = temp.path().to_path_buf();
        config.storage.wal_config.write_buffer_directory = "s3://bucket/wal".to_string();
        let found = effective_durable_locations(&config, env_of(&[]));
        let mode_entry = found
            .iter()
            .find(|loc| loc.source.contains("durability_mode"))
            .expect("durability mode entry");
        assert_eq!(
            mode_entry.durability_mode,
            Some(DurabilityMode::ObjectStoreServerless)
        );
        assert_eq!(mode_entry.location, "s3://bucket/wal");
    }

    /// TD-SPINE-5: a declared mode (TOML) wins over inference and is recorded
    /// with its declared source.
    #[test]
    fn declared_durability_mode_wins_in_the_record() {
        let temp = tempfile::tempdir().expect("tempdir");
        let mut config = Config::default();
        config.server.data_dir = temp.path().to_path_buf();
        config.storage.wal_config.durability_mode = Some(DurabilityMode::ObjectStoreServerless);
        let found = effective_durable_locations(&config, env_of(&[]));
        let mode_entry = found
            .iter()
            .find(|loc| loc.source.contains("durability_mode"))
            .expect("durability mode entry");
        assert_eq!(
            mode_entry.durability_mode,
            Some(DurabilityMode::ObjectStoreServerless)
        );
        assert!(mode_entry.source.contains("declared"));
    }
}
